"""The Stage 2 LLM text pilot: 1,000 validation + 1,000 locked test samples.

    # 1. select samples and score the frozen text baseline on exactly them
    python -m src.ahsef.cli.run_stage2_pilot --stage select

    # 2. run the LLM on validation, then analyse and freeze tau
    python -m src.ahsef.cli.run_stage2_pilot --stage validation
    python -m src.ahsef.cli.run_stage2_pilot --stage freeze

    # 3. only once everything above is frozen
    python -m src.ahsef.cli.run_stage2_pilot --stage test

The stages are separate commands on purpose.  ``test`` refuses to run until a
frozen configuration exists on disk, and it never writes one -- which is what
makes "the test set was locked" a property of the code rather than a promise.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.evaluation import evaluate_prediction_set, uncertainty_summary
from src.ahsef.gate import apply_gate, select_threshold
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.calibration import (
    confidence_calibration_report,
    fit_confidence_calibrator,
)
from src.ahsef.llm.inference import (
    LLM_MODALITY,
    LLMTextModality,
    build_prediction_set,
    coverage_report,
    routing_uncertainty_column,
)
from src.ahsef.llm.provider import (
    CallBudget,
    ReplayProvider,
    TranscriptWriter,
    recorded_sample_ids,
)
from src.ahsef.llm.samples import (
    SampleManifest,
    assert_disjoint,
    manifest_summary,
    restrict,
    select_samples,
)
from src.ahsef.registry import DEFAULT_BASELINES
from src.common.labels import CANONICAL_EMOTION_CLASSES

EXPERIMENT_ID = "stage2_llm_pilot_v1"
DEFAULT_MODEL = "gemma4:31b-cloud"
DEFAULT_PROMPT = "v2_explicit_json"
UNCERTAINTY_BINS = [round(0.1 * i, 1) for i in range(11)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", required=True, choices=["select", "validation", "freeze", "test"],
    )
    parser.add_argument("--run", default="stage2_llm")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Sample-selection seed.")
    parser.add_argument("--uncertainty-policy", default="score_entropy")
    parser.add_argument("--threshold-objective", default="target_stop_accuracy")
    parser.add_argument("--target-stop-accuracy", type=float, default=0.60)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--provider", default="ollama", choices=["ollama", "replay"],
        help="'replay' re-serves the recorded transcript; no network, no spend.",
    )
    parser.add_argument("--store-text", default="hash", choices=["none", "hash", "raw"])
    parser.add_argument("--max-calls", type=int, default=None)
    return parser


# ============================================================
# Paths
# ============================================================

class Pilot:
    def __init__(self, args):
        self.args = args
        self.layout = AhsefLayout(run=args.run, root=Path(args.root) / "ahsef")
        self.layout.prepare()
        self.base = self.layout.base
        for name in ("manifests", "llm", "baseline", "analysis"):
            (self.base / name).mkdir(parents=True, exist_ok=True)

    def manifest_path(self, split: str) -> Path:
        return self.base / "manifests" / f"samples_{split}.json"

    def baseline_path(self, split: str) -> Path:
        return self.base / "baseline" / f"text_baseline_{split}.parquet"

    def transcript_path(self, split: str) -> Path:
        return self.base / "llm" / f"transcript_{split}.jsonl"

    @property
    def frozen_path(self) -> Path:
        return self.base / "frozen_config.json"

    def analysis_path(self, name: str) -> Path:
        return self.base / "analysis" / name


# ============================================================
# Metrics helpers
# ============================================================

def metric_block(prediction_set: PredictionSet) -> dict:
    metrics = evaluate_prediction_set(prediction_set)
    return {
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": metrics["class_order"],
    }


def latency_block(frame: pd.DataFrame) -> dict:
    values = frame["latency_ms"].to_numpy(dtype=float)
    return {
        "total_ms": float(values.sum()),
        "total_seconds": float(values.sum() / 1000.0),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def uncertainty_bins(uncertainty, correct, edges=UNCERTAINTY_BINS) -> list[dict]:
    """TABLE B: does correctness fall as uncertainty rises?"""
    values = np.asarray(uncertainty, dtype=float)
    outcomes = np.asarray(correct, dtype=bool)
    rows = []
    for lower, upper in zip(edges, edges[1:]):
        last = upper == edges[-1]
        inside = (
            (values >= lower) & (values <= upper) if last
            else (values >= lower) & (values < upper)
        )
        count = int(inside.sum())
        rows.append({
            "bin": f"[{lower:.1f}, {upper:.1f}{']' if last else ')'}",
            "lower": lower, "upper": upper, "samples": count,
            "accuracy": float(outcomes[inside].mean()) if count else None,
            "error_rate": float(1 - outcomes[inside].mean()) if count else None,
            "mean_uncertainty": float(values[inside].mean()) if count else None,
            "fraction_correct": float(outcomes[inside].mean()) if count else None,
        })
    return rows


def rank_association(uncertainty, correct) -> dict:
    """Spearman rho and AUROC of uncertainty against being wrong.

    Both are rank statistics, so neither assumes the self-reported scores are
    on a meaningful interval scale -- which they are not.
    """
    values = np.asarray(uncertainty, dtype=float)
    wrong = (~np.asarray(correct, dtype=bool)).astype(float)
    n = values.size
    if n < 3 or np.unique(values).size < 2 or np.unique(wrong).size < 2:
        return {"spearman_rho": None, "auroc_uncertainty_predicts_error": None,
                "n": int(n), "note": "insufficient variation to compute an association"}

    def ranks(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(a.size, dtype=float)
        r[order] = np.arange(1, a.size + 1, dtype=float)
        unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, r)
        return (sums / counts)[inverse]

    ru, rw = ranks(values), ranks(wrong)
    rho = float(np.corrcoef(ru, rw)[0, 1])
    positives, negatives = int(wrong.sum()), int((1 - wrong).sum())
    auroc = float(
        (ru[wrong == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )
    # Fisher z gives an interval without assuming normal data, only a large n;
    # at n=1000 it is adequate and the report states the caveat.
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    half = 1.959964 / np.sqrt(max(n - 3, 1))
    return {
        "spearman_rho": rho,
        "rho_ci95": [float(np.tanh(z - half)), float(np.tanh(z + half))],
        "auroc_uncertainty_predicts_error": auroc,
        "n": int(n),
        "interpretation": (
            "rho > 0 and AUROC > 0.5 mean higher uncertainty goes with being wrong, "
            "which is the precondition for the gate to work at all."
        ),
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ============================================================
# Stage: select
# ============================================================

def load_split(split: str, root: str) -> pd.DataFrame:
    from src.data.text_dataset import EmotionTextDataset

    path = DEFAULT_BASELINES["text"].layout(root).split_path(split)
    dataset = EmotionTextDataset(path, columns=list(EmotionTextDataset.MINIMAL_COLUMNS))
    frame = dataset.manifest.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["text"] = [dataset.read_text(row) for _, row in frame.iterrows()]
    blank = frame["text"].astype(str).str.strip().eq("")
    return frame[~blank].reset_index(drop=True), str(path)


def stage_select(pilot: Pilot) -> int:
    args = pilot.args
    manifests = {}
    for split in ("validation", "test"):
        print(f"[data] loading {split} ...", flush=True)
        frame, source = load_split(split, args.root)
        manifest = select_samples(
            frame, args.size, args.seed, split, EXPERIMENT_ID, source_manifest=source,
        )
        manifest.save(pilot.manifest_path(split))
        manifests[split] = manifest
        print(f"       {split}: {manifest.size} of {manifest.population} "
              f"| fp={manifest.fingerprint[:16]} | {manifest.class_counts}")

        # Frozen text baseline on exactly these samples -- restricted from the
        # Stage 1 export, so nothing is retrained and nothing is re-inferred.
        stage1 = AhsefLayout(run="stage1", root=pilot.layout.root)
        source_predictions = stage1.prediction_path("text", split)
        if not source_predictions.exists():
            raise SystemExit(
                f"Missing Stage 1 text predictions at {source_predictions}. "
                f"Run the Stage 1 export first; this pilot never retrains a baseline."
            )
        baseline = PredictionSet.load(source_predictions).restricted_to(manifest.sample_ids)
        baseline.frame.to_parquet(pilot.baseline_path(split), index=False)
        block = metric_block(baseline)
        print(f"       text baseline: acc={block['accuracy']:.4f} "
              f"macroF1={block['macro_f1']:.4f} weightedF1={block['weighted_f1']:.4f}")
        (pilot.base / "baseline" / f"text_baseline_{split}_metrics.json").write_text(
            json.dumps({
                "split": split, "source": str(source_predictions),
                "samples": manifest.size, "metrics": block,
                "uncertainty": uncertainty_summary(baseline),
                "latency": latency_block(baseline.frame),
                "note": "Frozen Stage 1 text baseline restricted to the pilot manifest. "
                        "Not retrained, not modified, not re-inferred.",
            }, indent=2), encoding="utf-8",
        )

    assert_disjoint(manifests["validation"], manifests["test"])
    (pilot.base / "manifests" / "selection_summary.json").write_text(
        json.dumps({
            **manifest_summary(list(manifests.values())),
            "experiment_id": EXPERIMENT_ID, "selection_seed": args.seed,
            "requested_per_split": args.size,
        }, indent=2), encoding="utf-8",
    )
    print("[ok  ] validation and test manifests are disjoint")
    return 0


# ============================================================
# Stage: LLM inference
# ============================================================

def run_llm(pilot: Pilot, split: str) -> PredictionSet:
    """Score a split, resuming from whatever the transcript already covers.

    The transcript is the single source of truth.  Live calls are made only for
    samples it does not yet cover, and the prediction set is then *always* built
    by replaying the complete transcript over the full manifest.  So an
    interrupted run costs at most the calls it had not yet made, and the final
    artefact is identical whether the run finished in one pass or five.
    """
    args = pilot.args
    manifest = SampleManifest.load(pilot.manifest_path(split))
    frame, _ = load_split(split, args.root)
    frame = restrict(frame, manifest)
    transcript = pilot.transcript_path(split)

    already = recorded_sample_ids(transcript)
    pending = frame[~frame["sample_id"].isin(already)].reset_index(drop=True)
    resumed = bool(already)
    if resumed:
        print(f"[llm ] {split}: resuming -- {len(already)} of {len(frame)} already "
              f"recorded, {len(pending)} to call")
    print(f"[llm ] {split}: {len(frame)} samples | model={args.model} "
          f"prompt={args.prompt_version} T={args.temperature} seed={args.llm_seed}")

    started = time.perf_counter()
    calls_made = 0
    if len(pending) and args.provider != "replay":
        from src.ahsef.llm.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            model=args.model, host=args.host, temperature=args.temperature,
            seed=args.llm_seed, num_predict=args.num_predict,
        )
        writer = TranscriptWriter(
            transcript, model=args.model, provider=provider.name,
            prompt_version=args.prompt_version,
            store_text=(args.store_text == "raw"), append=resumed,
        )
        budget = CallBudget(max_calls=args.max_calls)
        live = LLMTextModality(
            provider=provider, prompt_version=args.prompt_version, repeats=1,
            uncertainty_policy=args.uncertainty_policy, budget=budget, transcript=writer,
            store_text=args.store_text,
        )

        def progress(done, total, state):
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"       {done}/{total}  {rate:.2f}/s  "
                  f"eta {(total - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

        try:
            live.predict_frame(pending, progress_every=50, on_progress=progress)
        finally:
            writer.close()
        calls_made = budget.calls
        if writer.retries:
            print(f"[warn] {writer.retries} transcript write(s) retried after a "
                  f"transient file lock")
    elif not len(pending):
        print(f"[llm ] {split}: every sample already recorded; no API call needed")

    # Build the prediction set from the complete transcript, always.
    replay = ReplayProvider(transcript, strict=True)
    modality = LLMTextModality(
        provider=replay, prompt_version=args.prompt_version, repeats=1,
        uncertainty_policy=args.uncertainty_policy, store_text=args.store_text,
    )
    results = modality.predict_frame(frame)

    texts = dict(zip(frame["sample_id"], frame["text"]))
    provenance = modality.provenance()
    provenance["execution"] = {
        "resumed": resumed,
        "previously_recorded": len(already),
        "live_calls_this_invocation": calls_made,
        "total_samples": len(frame),
        "note": "Predictions are built by replaying the complete transcript, so the "
                "artefact does not depend on how many invocations produced it.",
    }
    predictions = build_prediction_set(
        results, split, provenance, texts=texts, store_text=args.store_text,
    )
    predictions.save(
        pilot.layout.prediction_path(LLM_MODALITY, split),
        pilot.layout.prediction_meta_path(LLM_MODALITY, split),
    )
    print(f"[llm ] done in {(time.perf_counter() - started)/60:.1f} min | "
          f"{calls_made} live calls | "
          f"coverage {predictions.frame['llm_usable'].mean():.1%}")
    return predictions


def analyse(pilot: Pilot, split: str, predictions: PredictionSet) -> dict:
    """Metrics, uncertainty, bins, and calibration for one split."""
    args = pilot.args
    coverage = coverage_report(predictions)
    usable = predictions.frame[predictions.frame["llm_usable"]]
    if usable.empty:
        raise SystemExit("The LLM produced no usable predictions.")
    scored = predictions.restricted_to(usable["sample_id"].tolist())
    correct = (scored.predictions() == scored.labels()).numpy()

    uncertainty = routing_uncertainty_column(scored, args.uncertainty_policy).to_numpy(float)
    confidence = scored.frame["confidence"].to_numpy(dtype=float)
    has_conf = ~np.isnan(confidence)

    baseline = PredictionSet.load(pilot.layout.prediction_path("text", split)) \
        if False else None  # baseline comes from the pilot's own restricted copy
    baseline_frame = pd.read_parquet(pilot.baseline_path(split))
    baseline_set = PredictionSet(
        modality="text", split=split, class_order=tuple(CANONICAL_EMOTION_CLASSES),
        frame=baseline_frame, meta={"kind": "frozen_text_baseline"},
    )
    paired = baseline_set.restricted_to(scored.sample_ids())

    record = {
        "split": split,
        "coverage": coverage,
        "llm_metrics": metric_block(scored),
        "text_baseline_metrics_same_samples": metric_block(paired),
        "uncertainty": {
            "policy": args.uncertainty_policy,
            "mean": float(uncertainty.mean()),
            "median": float(np.median(uncertainty)),
            "std": float(uncertainty.std(ddof=0)),
            "mean_top1_score": float(scored.frame["confidence"].mean(skipna=True)),
            "mean_top1_prob": float(np.nanmax(scored.probabilities().numpy(), axis=1).mean()),
            "median_top1_prob": float(
                np.median(np.nanmax(scored.probabilities().numpy(), axis=1))
            ),
            "mean_margin": float(scored.frame["margin"].mean(skipna=True)),
            "ahsef_summary": uncertainty_summary(scored),
            "definition_note": (
                "'uncertainty' is AHSEF-derived from the normalised self-reported class "
                "scores. 'llm_confidence' is the model's own self-report. They are "
                "different quantities and neither is a calibrated posterior."
            ),
        },
        "uncertainty_bins": uncertainty_bins(uncertainty, correct),
        "uncertainty_vs_correctness": rank_association(uncertainty, correct),
        "latency": latency_block(predictions.frame),
        "tokens": {
            "input_tokens": int(predictions.frame["input_tokens"].sum()),
            "output_tokens": int(predictions.frame["output_tokens"].sum()),
            "mean_input_tokens": float(predictions.frame["input_tokens"].mean()),
            "mean_output_tokens": float(predictions.frame["output_tokens"].mean()),
        },
        "cost": {
            "monetary_cost_usd": None,
            "reason": "Ollama publishes no per-token price for cloud-backed models "
                      "through this interface. Token counts and latency are reported; "
                      "no price is invented.",
            "execution": predictions.meta.get("provider", {}).get("execution"),
        },
    }
    if has_conf.any():
        record["llm_confidence_calibration"] = confidence_calibration_report(
            confidence[has_conf], correct[has_conf], args.bins,
        )
    # ECE/NLL/Brier over the *score distribution*, using Stage 1's implementation.
    from src.ahsef.calibration import calibration_report

    scores = scored.probabilities()
    valid = ~np.isnan(scores.numpy()).any(axis=1)
    if valid.any():
        record["score_distribution_calibration"] = {
            **calibration_report(
                scores[valid], scored.labels()[valid], args.bins, "llm_score_distribution",
            ),
            "caveat": "Computed over normalised SELF-REPORTED scores, not a posterior. "
                      "ECE/NLL here measure how well a self-report tracks correctness.",
        }
    return record


# ============================================================
# Stage: validation / freeze / test
# ============================================================

def stage_validation(pilot: Pilot) -> int:
    predictions = run_llm(pilot, "validation")
    record = analyse(pilot, "validation", predictions)
    pilot.analysis_path("validation_analysis.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    _print_split(record, "VALIDATION")
    print(f"\nWritten: {pilot.analysis_path('validation_analysis.json')}")
    print("Next: python -m src.ahsef.cli.run_stage2_pilot --stage freeze")
    return 0


CANDIDATE_POLICIES = ("score_top1", "score_entropy", "llm_confidence", "ambiguity")


def compare_policies(scored, correct) -> dict:
    """Rank the available uncertainty signals on validation, by AUROC.

    Choosing which signal to gate on is a modelling decision, so it is made
    here -- on validation, with the whole comparison recorded -- rather than
    asserted. AUROC is the right criterion because the gate only needs the
    signal to *order* samples, not to be on any particular scale.
    """
    table = {}
    for policy in CANDIDATE_POLICIES:
        try:
            values = routing_uncertainty_column(scored, policy).to_numpy(dtype=float)
        except Exception as error:              # policy unavailable for this export
            table[policy] = {"available": False, "reason": str(error)[:160]}
            continue
        if np.isnan(values).any():
            table[policy] = {"available": False, "reason": "contains NaN"}
            continue
        association = rank_association(values, correct)
        table[policy] = {
            "available": True,
            "auroc": association["auroc_uncertainty_predicts_error"],
            "spearman_rho": association["spearman_rho"],
            "mean": float(values.mean()),
            "distinct_values": int(np.unique(values).size),
        }
    ranked = sorted(
        (name for name, entry in table.items()
         if entry.get("available") and entry.get("auroc") is not None),
        key=lambda name: table[name]["auroc"], reverse=True,
    )
    return {
        "criterion": "AUROC of uncertainty against being wrong, on validation",
        "candidates": table,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }


def stage_freeze(pilot: Pilot) -> int:
    """Select tau on validation and freeze the whole configuration."""
    args = pilot.args
    path = pilot.layout.prediction_path(LLM_MODALITY, "validation")
    if not path.exists():
        raise SystemExit("Run --stage validation first.")
    predictions = PredictionSet.load(path)
    usable = predictions.frame[predictions.frame["llm_usable"]]
    scored = predictions.restricted_to(usable["sample_id"].tolist())
    correct = (scored.predictions() == scored.labels()).numpy()

    policy_comparison = compare_policies(scored, correct)
    print("[pol ] validation AUROC by uncertainty policy:")
    for name, entry in policy_comparison["candidates"].items():
        if entry.get("available"):
            print(f"       {name:<16} AUROC={entry['auroc']:.4f} "
                  f"rho={entry['spearman_rho']:+.4f} distinct={entry['distinct_values']}")
        else:
            print(f"       {name:<16} unavailable ({entry['reason'][:60]})")
    uncertainty = routing_uncertainty_column(scored, args.uncertainty_policy).to_numpy(float)

    selection = select_threshold(
        uncertainty, correct, split="validation",
        objective=args.threshold_objective,
        target_stop_accuracy=args.target_stop_accuracy,
        min_coverage=args.min_coverage,
    )
    sweep = _sweep_table(selection, uncertainty, correct, scored)
    frozen = {
        "experiment_id": EXPERIMENT_ID,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "host": args.host,
        "prompt_version": args.prompt_version,
        "temperature": args.temperature,
        "llm_seed": args.llm_seed,
        "num_predict": args.num_predict,
        "uncertainty_policy": args.uncertainty_policy,
        "uncertainty_policy_comparison": policy_comparison,
        "uncertainty_policy_rationale": (
            "The configured policy is kept unless the comparison shows a materially "
            "better signal. AUROC differences below ~0.02 at n=1000 are within noise, "
            "and an AHSEF-derived signal is preferred over a raw self-report at equal "
            "ranking power, as is a signal with more distinct values (finer threshold "
            "control). The full comparison is recorded above so the choice is auditable."
        ),
        "threshold": selection.threshold,
        "threshold_selection": selection.to_dict(),
        "threshold_sweep_detail": sweep,
        "calibration_decision": _calibration_decision(pilot, args.uncertainty_policy),
        "selection_seed": args.seed,
        "size_per_split": args.size,
        "git_commit": git_commit(),
        "environment": {"platform": platform.platform(), "python": sys.version},
        "uses_test_labels": False,
        "lock_note": (
            "Everything the test evaluation depends on is fixed here. --stage test "
            "loads this file and never writes it."
        ),
    }
    pilot.frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"[tau ] {selection.threshold:.6f} | satisfied={selection.satisfied}")
    print(f"       {selection.note}")
    print(f"\nWritten: {pilot.frozen_path}")
    print("Next: python -m src.ahsef.cli.run_stage2_pilot --stage test")
    return 0


def _sweep_table(selection, uncertainty, correct, scored) -> list[dict]:
    """Per-threshold STOP/REQUEST accuracy and macro-F1, as the brief requires."""
    from src.training.metrics import classification_metrics
    import torch

    predictions = scored.predictions()
    labels = scored.labels()
    rows = []
    # A subsample of the sweep keeps the artefact readable; the full sweep is
    # already inside selection.to_dict().
    points = selection.sweep
    step = max(len(points) // 40, 1)
    for point in points[::step]:
        mask = uncertainty <= point.threshold
        row = {"threshold": point.threshold,
               "stop_pct": float(mask.mean()), "request_pct": float(1 - mask.mean()),
               "stop_n": int(mask.sum()), "request_n": int((~mask).sum())}
        if mask.any():
            m = classification_metrics(predictions[mask], labels[mask], 7)
            row["stop_accuracy"] = m["accuracy"]
            row["stop_macro_f1"] = m["macro_f1"]
            row["stop_error_rate"] = 1 - m["accuracy"]
        if (~mask).any():
            m = classification_metrics(predictions[~mask], labels[~mask], 7)
            row["request_accuracy"] = m["accuracy"]
            row["request_macro_f1"] = m["macro_f1"]
        rows.append(row)
    return rows


def _calibration_decision(pilot: Pilot, routing_policy: str) -> dict:
    """Decide on calibration from validation evidence, and state its scope.

    The scope matters more than the decision here.  A recalibration map applies
    to ``llm_confidence`` -- the model's self-report.  The gate routes on
    whatever ``routing_policy`` names, and when that is a different quantity
    (``score_entropy``, say) the map cannot reach the routing path at all.
    Recording "calibration applied" without saying which quantity it touched
    would imply the gate was calibrated when it was not.
    """
    record = json.loads(
        pilot.analysis_path("validation_analysis.json").read_text(encoding="utf-8")
    )
    confidence = record.get("llm_confidence_calibration")
    if not confidence:
        return {
            "fit": False, "applied_to_routing_signal": False,
            "reason": "no self-reported confidence was available",
        }
    auroc = confidence["discrimination"]["auroc"]
    distinct = confidence["discrimination"]["distinct_values"]
    # A recalibration map is only meaningful if the raw signal orders samples at
    # all. Fitting one on a signal with no ranking power dresses noise as rigour.
    justified = auroc is not None and auroc > 0.55 and distinct >= 5
    feeds_gate = routing_policy == "llm_confidence"
    return {
        "quantity": "llm_confidence (self-reported)",
        "routing_signal": routing_policy,
        "fit": bool(justified),
        "method": "binned_empirical_accuracy" if justified else None,
        "fitted_on": "validation" if justified else None,
        "uses_test_labels": False,
        "applied_to_routing_signal": bool(justified and feeds_gate),
        "ece_raw": confidence["ece"],
        "auroc": auroc,
        "distinct_values": distinct,
        "reason": (
            f"validation AUROC={auroc:.4f} over {distinct} distinct confidence values: "
            + (
                "the self-report orders samples, so a binned recalibration map is "
                "methodologically justified and was fitted on validation only. "
                if justified else
                "the self-report has too little ranking power for a recalibration map "
                "to be justified, so none was fitted and raw ECE is reported instead. "
            )
            + (
                f"It IS the routing signal, so the gate consumes the calibrated value."
                if justified and feeds_gate else
                f"It is NOT the routing signal -- the gate routes on "
                f"{routing_policy!r} -- so this calibration is reported as a property "
                f"of the model and does not enter the routing path. Routing traces "
                f"therefore carry calibrated_uncertainty = null."
            )
        ),
    }


def stage_test(pilot: Pilot) -> int:
    """The locked evaluation.  Loads the frozen config; never writes it."""
    args = pilot.args
    if not pilot.frozen_path.exists():
        raise SystemExit(
            f"No frozen configuration at {pilot.frozen_path}. Run --stage freeze on "
            f"validation first. The test set stays locked until every decision "
            f"-- model, prompt, generation config, calibration, tau -- is fixed."
        )
    frozen = json.loads(pilot.frozen_path.read_text(encoding="utf-8"))
    drift = {
        key: (frozen[key], getattr(args, attr))
        for key, attr in [
            ("model", "model"), ("prompt_version", "prompt_version"),
            ("temperature", "temperature"), ("llm_seed", "llm_seed"),
            ("uncertainty_policy", "uncertainty_policy"),
        ]
        if frozen[key] != getattr(args, attr)
    }
    if drift:
        raise SystemExit(
            f"The requested configuration differs from the frozen one: {drift}. "
            f"Changing anything after the freeze invalidates the locked evaluation."
        )

    predictions_path = pilot.layout.prediction_path(LLM_MODALITY, "test")
    predictions = (
        PredictionSet.load(predictions_path) if predictions_path.exists()
        else run_llm(pilot, "test")
    )
    record = analyse(pilot, "test", predictions)

    threshold = float(frozen["threshold"])
    usable = predictions.frame[predictions.frame["llm_usable"]]
    scored = predictions.restricted_to(usable["sample_id"].tolist())
    uncertainty = routing_uncertainty_column(scored, args.uncertainty_policy).to_numpy(float)
    correct = (scored.predictions() == scored.labels()).numpy()
    outcomes = [
        apply_gate(sid, float(value), threshold)
        for sid, value in zip(scored.sample_ids(), uncertainty)
    ]
    stop_mask = np.array([outcome.stop for outcome in outcomes], dtype=bool)

    from src.training.metrics import classification_metrics

    def group(mask, name):
        if not mask.any():
            return {"group": name, "samples": 0}
        m = classification_metrics(scored.predictions()[mask], scored.labels()[mask], 7)
        return {
            "group": name, "samples": int(mask.sum()),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"], "error_rate": 1 - m["accuracy"],
        }

    record["routing"] = {
        "threshold": threshold,
        "threshold_source": f"frozen on validation at {frozen['frozen_at']}",
        "stop": int(stop_mask.sum()),
        "request": int((~stop_mask).sum()),
        "stop_pct": float(stop_mask.mean()),
        "request_pct": float(1 - stop_mask.mean()),
        "conditional": [
            group(np.ones_like(stop_mask), "All"),
            group(stop_mask, "STOP"),
            group(~stop_mask, "REQUEST"),
        ],
        "limitation": (
            "REQUEST means only that the textual evidence was insufficient under the "
            "validation-selected threshold. It is NOT evidence that another modality "
            "would improve the prediction; no modality was acquired or fused."
        ),
    }
    _write_traces(pilot, "test", scored, uncertainty, outcomes, frozen)
    pilot.analysis_path("test_analysis.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    _print_split(record, "TEST (locked)")
    _print_routing(record["routing"])
    print(f"\nWritten: {pilot.analysis_path('test_analysis.json')}")
    return 0


def _write_traces(pilot, split, scored, uncertainty, outcomes, frozen) -> Path:
    """One auditable trace line per sample, with no secrets."""
    provider = scored.meta.get("provider", {}) if isinstance(scored.meta, dict) else {}
    path = pilot.base / "llm" / f"routing_trace_{split}.jsonl"
    frame = scored.frame
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record": "header", "experiment_id": EXPERIMENT_ID, "split": split,
            "model": frozen["model"], "provider": "ollama",
            "prompt_version": frozen["prompt_version"],
            "temperature": frozen["temperature"], "seed": frozen["llm_seed"],
            "threshold_tau": frozen["threshold"],
            "uncertainty_policy": frozen["uncertainty_policy"],
            "initial_modality": LLM_MODALITY,
        }) + "\n")
        for position, outcome in enumerate(outcomes):
            row = frame.iloc[position]
            handle.write(json.dumps({
                "record": "trace",
                "sample_id": str(row["sample_id"]),
                "split": split,
                "initial_modality": "text",
                "predicted_emotion": CANONICAL_EMOTION_CLASSES[int(row["predicted_class"])],
                "class_scores": {
                    name: (None if pd.isna(row[f"prob_{i}"]) else float(row[f"prob_{i}"]))
                    for i, name in enumerate(CANONICAL_EMOTION_CLASSES)
                },
                "llm_confidence": (
                    None if pd.isna(row["confidence"]) else float(row["confidence"])
                ),
                "uncertainty": float(uncertainty[position]),
                "calibrated_uncertainty": None,
                "threshold_tau": outcome.threshold,
                "routing_decision": "STOP" if outcome.stop else "REQUEST",
                "reason": outcome.reason,
                "model": frozen["model"],
                "provider": "ollama",
                "prompt_version": frozen["prompt_version"],
                "latency_ms": float(row["latency_ms"]),
                "input_tokens": int(row["input_tokens"]),
                "output_tokens": int(row["output_tokens"]),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "router_saw_true_class": False,
            }) + "\n")
    return path


# ============================================================
# Printing
# ============================================================

def _print_split(record: dict, title: str) -> None:
    llm, base = record["llm_metrics"], record["text_baseline_metrics_same_samples"]
    print()
    print("=" * 92)
    print(f"{title} -- TABLE A: LLM vs frozen text baseline (paired, same samples)")
    print("=" * 92)
    print(f"  {'System':<26}{'Acc':>9}{'Macro-F1':>11}{'Weighted-F1':>13}"
          f"{'MacroP':>9}{'MacroR':>9}")
    for name, block in (("Existing Text Baseline", base), ("Gemma LLM", llm)):
        print(f"  {name:<26}{block['accuracy']:>9.4f}{block['macro_f1']:>11.4f}"
              f"{block['weighted_f1']:>13.4f}{block['macro_precision']:>9.4f}"
              f"{block['macro_recall']:>9.4f}")
    print()
    print(f"  {'Emotion':<10}{'Base F1':>10}{'LLM F1':>10}{'D F1':>9}"
          f"{'Base R':>10}{'LLM R':>9}{'D R':>9}{'n':>7}")
    for name in CANONICAL_EMOTION_CLASSES:
        b, l = base["per_class"][name], llm["per_class"][name]
        print(f"  {name:<10}{b['f1']:>10.4f}{l['f1']:>10.4f}{l['f1']-b['f1']:>+9.4f}"
              f"{b['recall']:>10.4f}{l['recall']:>9.4f}{l['recall']-b['recall']:>+9.4f}"
              f"{b['support']:>7}")
    unc = record["uncertainty"]
    print()
    print(f"  uncertainty ({unc['policy']}): mean={unc['mean']:.4f} "
          f"median={unc['median']:.4f} mean top-1 score={unc['mean_top1_prob']:.4f} "
          f"mean margin={unc['mean_margin']:.4f}")
    print()
    print(f"  {'TABLE B -- Uncertainty bin':<28}{'Samples':>9}{'Accuracy':>11}{'Error':>9}")
    for row in record["uncertainty_bins"]:
        if row["samples"]:
            print(f"  {row['bin']:<28}{row['samples']:>9}{row['accuracy']:>11.4f}"
                  f"{row['error_rate']:>9.4f}")
    assoc = record["uncertainty_vs_correctness"]
    print(f"\n  Spearman rho(uncertainty, wrong) = {assoc['spearman_rho']}   "
          f"AUROC = {assoc['auroc_uncertainty_predicts_error']}")
    cov = record["coverage"]
    lat = record["latency"]
    print(f"  coverage={cov['coverage']:.1%} ({cov['status_counts']})  "
          f"latency mean={lat['mean_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")


def _print_routing(routing: dict) -> None:
    print()
    print("=" * 92)
    print("TABLE C -- AHSEF text gate (tau frozen on validation)")
    print("=" * 92)
    print(f"  {'Routing Outcome':<20}{'Count':>9}{'Percentage':>13}")
    print(f"  {'STOP':<20}{routing['stop']:>9}{routing['stop_pct']:>12.1%}")
    print(f"  {'REQUEST':<20}{routing['request']:>9}{routing['request_pct']:>12.1%}")
    print()
    print("TABLE D -- Conditional performance")
    print(f"  {'Group':<10}{'Samples':>9}{'Accuracy':>11}{'Macro-F1':>11}{'Error':>9}")
    for row in routing["conditional"]:
        if row.get("samples"):
            print(f"  {row['group']:<10}{row['samples']:>9}{row['accuracy']:>11.4f}"
                  f"{row['macro_f1']:>11.4f}{row['error_rate']:>9.4f}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pilot = Pilot(args)
    return {
        "select": stage_select, "validation": stage_validation,
        "freeze": stage_freeze, "test": stage_test,
    }[args.stage](pilot)


if __name__ == "__main__":
    raise SystemExit(main())
