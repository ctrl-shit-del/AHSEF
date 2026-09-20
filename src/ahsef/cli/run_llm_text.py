"""Evaluate the LLM text modality on its own, before it routes anything.

    # record a run against the real API (costs money -- budget flags are enforced)
    python -m src.ahsef.cli.run_llm_text --run stage2_llm --split validation \
        --provider anthropic --max-samples 1000 --max-cost-usd 5.00

    # re-analyse that exact run with no network and no spend
    python -m src.ahsef.cli.run_llm_text --run stage2_llm --split validation \
        --provider replay

Scores a stratified sample of the frozen text experiment's split, writes an
AHSEF prediction set, and reports the LLM against the frozen text baseline on
the *same* samples.

Three things this command does not do: it does not choose a threshold, it does
not fuse anything, and it does not touch a Stage 1 artefact.  Everything lands
under ``experiments/ahsef/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.evaluation import evaluate_prediction_set, uncertainty_summary
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.calibration import (
    calibration_provenance,
    confidence_calibration_report,
    fit_confidence_calibrator,
)
from src.ahsef.llm.inference import (
    LLM_MODALITY,
    LLMTextModality,
    build_prediction_set,
    coverage_report,
)
from src.ahsef.llm.provider import (
    BudgetExceeded,
    CallBudget,
    ReplayProvider,
    TranscriptWriter,
)
from src.ahsef.llm.uncertainty import DEFAULT_UNCERTAINTY_POLICY, UNCERTAINTY_POLICIES
from src.ahsef.registry import DEFAULT_BASELINES
from src.common.labels import CANONICAL_EMOTION_CLASSES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage2_llm", help="AHSEF run directory name.")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument(
        "--split", default="validation", choices=["validation", "test"],
        help="Test is locked: run it only after every validation decision is frozen.",
    )
    parser.add_argument(
        "--provider", default="replay", choices=["anthropic", "replay"],
        help="'replay' re-serves a recorded transcript; no network, no spend.",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Calls per sample. >1 measures empirical uncertainty from the model's own "
             "nondeterminism, and multiplies cost by the same factor.",
    )
    parser.add_argument(
        "--uncertainty-policy", default=DEFAULT_UNCERTAINTY_POLICY,
        choices=list(UNCERTAINTY_POLICIES),
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Stratified subsample size. Omit to score the whole split.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subsampling seed.")
    parser.add_argument(
        "--max-cost-usd", type=float, default=None,
        help="Hard spend ceiling, enforced before every call.",
    )
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument(
        "--store-text", default="hash", choices=["none", "hash", "raw"],
        help="What of the input text reaches the artefacts. Default: a hash only.",
    )
    parser.add_argument("--transcript", default=None, help="Override transcript path.")
    parser.add_argument("--bins", type=int, default=10, help="Calibration bins.")
    return parser


# ============================================================
# Sampling
# ============================================================

def load_text_split(split: str, root: str) -> pd.DataFrame:
    """The frozen text baseline's own view of a split, with transcripts resolved.

    Built through :class:`EmotionTextDataset` rather than by re-reading the
    parquet, so the eligibility filter, the column set, and the
    metadata-vs-file text resolution are exactly the ones the baseline used.
    Scoring a different row set would make the comparison meaningless.
    """
    from src.data.text_dataset import EmotionTextDataset

    reference = DEFAULT_BASELINES["text"]
    path = reference.layout(root).split_path(split)
    if not path.exists():
        raise SystemExit(f"Missing text manifest: {path}")

    dataset = EmotionTextDataset(path, columns=list(EmotionTextDataset.MINIMAL_COLUMNS))
    frame = dataset.manifest.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["text"] = [dataset.read_text(row) for _, row in frame.iterrows()]

    blank = frame["text"].astype(str).str.strip().eq("")
    if bool(blank.any()):
        print(f"[data] dropping {int(blank.sum())} records whose transcript resolved empty")
    return frame[~blank].reset_index(drop=True)


def stratified_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Class-proportional subsample with every present class represented.

    Largest-remainder allocation over ``canonical_emotion_id``, mirroring the
    sampler the baselines used, so the subsample is not a fresh design decision.
    """
    if size >= len(frame):
        return frame.reset_index(drop=True)
    groups = {int(key): part for key, part in frame.groupby("canonical_emotion_id")}
    total = len(frame)
    exact = {key: len(part) * size / total for key, part in groups.items()}
    counts = {key: max(1, int(value)) for key, value in exact.items()}
    # Largest-remainder top-up / trim so the total lands exactly on `size`.
    while sum(counts.values()) < size:
        key = max(exact, key=lambda k: exact[k] - counts[k])
        counts[key] += 1
        exact[key] -= 1e-9
    while sum(counts.values()) > size:
        key = max(counts, key=lambda k: counts[k] - exact[k])
        if counts[key] > 1:
            counts[key] -= 1
        else:
            exact[key] += 1e9
    parts = [
        part.sample(n=min(counts[key], len(part)), random_state=seed)
        for key, part in groups.items()
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ============================================================
# Baseline comparison
# ============================================================

def baseline_on_same_samples(
    layout: AhsefLayout, split: str, sample_ids, stage1_run: str = "stage1"
) -> dict | None:
    """The frozen text baseline restricted to exactly the LLM's samples."""
    path = AhsefLayout(run=stage1_run, root=layout.root).prediction_path("text", split)
    if not path.exists():
        return None
    baseline = PredictionSet.load(path)
    known = set(baseline.sample_ids())
    shared = [identifier for identifier in sample_ids if identifier in known]
    if not shared:
        return None
    restricted = baseline.restricted_to(shared)
    metrics = evaluate_prediction_set(restricted)
    return {
        "source": str(path),
        "samples": len(shared),
        "metrics": metrics,
        "uncertainty": uncertainty_summary(restricted),
        "note": "Frozen text baseline scored on exactly the samples the LLM saw.",
    }


# ============================================================
# Main
# ============================================================

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()
    llm_dir = layout.base / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = Path(args.transcript) if args.transcript else (
        llm_dir / f"transcript__{args.split}.jsonl"
    )

    # ---------------------------------------------------------------- data
    frame = load_text_split(args.split, args.root)
    if args.max_samples:
        frame = stratified_sample(frame, args.max_samples, args.seed)
    print(f"[data] {args.split}: {len(frame)} samples "
          f"({frame['dataset'].value_counts().to_dict()})")

    # ------------------------------------------------------------ provider
    budget = CallBudget(max_calls=args.max_calls, max_cost_usd=args.max_cost_usd)
    if args.provider == "replay":
        provider = ReplayProvider(transcript_path, strict=True)
        writer = None
        print(f"[llm ] replaying {transcript_path} "
              f"({len(provider._records)} recorded prompts)")
    else:
        from src.ahsef.llm.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(
            model=args.model, max_tokens=args.max_tokens, effort=args.effort,
        )
        writer = TranscriptWriter(
            transcript_path, model=args.model, provider=provider.name,
            prompt_version=args.prompt_version, store_text=(args.store_text == "raw"),
        )
        estimated = len(frame) * args.repeats
        print(f"[llm ] {provider.name}/{args.model} effort={args.effort} "
              f"repeats={args.repeats} -> up to {estimated} calls")
        if args.max_cost_usd is None and args.max_calls is None:
            print("[warn] no --max-cost-usd or --max-calls set; the run is uncapped",
                  file=sys.stderr)

    modality = LLMTextModality(
        provider=provider, prompt_version=args.prompt_version, repeats=args.repeats,
        uncertainty_policy=args.uncertainty_policy, budget=budget, transcript=writer,
        store_text=args.store_text,
    )

    def progress(done: int, total: int, budget_state: CallBudget) -> None:
        print(f"       {done}/{total}  calls={budget_state.calls} "
              f"est=${budget_state.cost_usd:.4f}", flush=True)

    try:
        results = modality.predict_frame(frame, on_progress=progress)
    except BudgetExceeded as error:
        print(f"[stop] {error}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.close()

    texts = dict(zip(frame["sample_id"], frame["text"]))
    predictions = build_prediction_set(
        results, args.split, modality.provenance(), texts=texts, store_text=args.store_text,
    )
    predictions.save(
        layout.prediction_path(LLM_MODALITY, args.split),
        layout.prediction_meta_path(LLM_MODALITY, args.split),
    )

    # ---------------------------------------------------------- evaluation
    coverage = coverage_report(predictions)
    usable = predictions.frame[predictions.frame["llm_usable"]]
    if usable.empty:
        raise SystemExit("The LLM produced no usable predictions; nothing to evaluate.")
    scored = predictions.restricted_to(usable["sample_id"].tolist())
    metrics = evaluate_prediction_set(scored)
    correct = (scored.predictions() == scored.labels()).numpy()

    confidence = scored.frame["confidence"].to_numpy(dtype=float)
    has_confidence = ~np.isnan(confidence)
    calibration = None
    calibrator = None
    if has_confidence.any():
        calibration = confidence_calibration_report(
            confidence[has_confidence], correct[has_confidence], args.bins,
        )
        if args.split == "validation":
            calibrator = fit_confidence_calibrator(
                confidence[has_confidence], correct[has_confidence],
                split="validation", num_bins=args.bins,
            )

    baseline = baseline_on_same_samples(
        layout, args.split, predictions.frame["sample_id"].tolist()
    )

    # ------------------------------------------------------------- report
    report = {
        "split": args.split,
        "modality": LLM_MODALITY,
        "provenance": modality.provenance(),
        "coverage": coverage,
        "metrics": metrics,
        "uncertainty": uncertainty_summary(scored),
        "confidence_calibration": calibration,
        "confidence_calibrator": calibrator.to_dict() if calibrator else None,
        "calibration_policy": calibration_provenance(),
        "cost": {
            **budget.to_dict(),
            "mean_latency_ms": float(predictions.frame["latency_ms"].mean()),
            "total_latency_ms": float(predictions.frame["latency_ms"].sum()),
        },
        "text_baseline_same_samples": baseline,
        "artefacts": {
            "predictions": str(layout.prediction_path(LLM_MODALITY, args.split)),
            "transcript": str(transcript_path),
        },
        "scientific_note": (
            "class_scores are self-reported, not calibrated posteriors. Metrics are "
            "computed on the samples the LLM could answer; 'coverage' states the rest."
        ),
    }
    report_path = layout.report_path(f"llm_text_{args.split}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if calibrator is not None:
        (llm_dir / "confidence_calibrator.json").write_text(
            json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8"
        )

    _print_report(args, report, metrics, coverage, calibration, baseline)
    print(f"\nWritten: {layout.prediction_path(LLM_MODALITY, args.split)}")
    print(f"Written: {report_path}")
    return 0


def _print_report(args, report, metrics, coverage, calibration, baseline) -> None:
    print()
    print("=" * 84)
    print(f"LLM TEXT MODALITY  ({args.split})")
    print("=" * 84)
    print(f"  samples             : {coverage['samples']}")
    print(f"  usable answers      : {coverage['usable']}  "
          f"(coverage {coverage['coverage']:.1%})")
    print(f"  with class_scores   : {coverage['with_class_scores']}  "
          f"({coverage['score_coverage']:.1%})")
    print(f"  status              : {coverage['status_counts']}")
    if coverage["unmappable_examples"]:
        print(f"  unmappable labels   : {coverage['unmappable_examples']}")
    print()
    print(f"  accuracy            : {metrics['accuracy']:.4f}")
    print(f"  macro-F1            : {metrics['macro_f1']:.4f}")
    print(f"  weighted-F1         : {metrics['weighted_f1']:.4f}")
    print(f"  macro precision     : {metrics['macro_precision']:.4f}")
    print(f"  macro recall        : {metrics['macro_recall']:.4f}")
    print()
    print(f"  {'emotion':<10}{'precision':>11}{'recall':>9}{'F1':>9}{'support':>9}")
    for name in CANONICAL_EMOTION_CLASSES:
        entry = metrics["per_class"][name]
        print(f"  {name:<10}{entry['precision']:>11.4f}{entry['recall']:>9.4f}"
              f"{entry['f1']:>9.4f}{entry['support']:>9}")
    if calibration:
        print()
        print(f"  mean confidence     : {calibration['mean_confidence']:.4f}")
        print(f"  ECE                 : {calibration['ece']:.4f}")
        print(f"  AUROC(conf,correct) : {calibration['discrimination']['auroc']}")
        print(f"  separation          : "
              f"{calibration['confidence_vs_correctness']['separation']}")
    if baseline:
        print()
        print("  frozen text baseline on the same samples:")
        print(f"    accuracy={baseline['metrics']['accuracy']:.4f}  "
              f"macro-F1={baseline['metrics']['macro_f1']:.4f}  "
              f"weighted-F1={baseline['metrics']['weighted_f1']:.4f}")
    print()
    print(f"  calls={report['cost']['calls']}  "
          f"est. cost=${report['cost']['estimated_cost_usd']:.4f}  "
          f"mean latency={report['cost']['mean_latency_ms']:.1f} ms")


if __name__ == "__main__":
    raise SystemExit(main())
