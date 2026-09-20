"""PHASE B-1 and PHASE C -- baseline audio vs strong audio, then on the pool.

    # B-1: is the strong expert materially better on the full validation split?
    python -m src.ahsef.cli.run_audio_expert_comparison --stage compare

    # C: does that change what AHSEF can exploit on the 509 aligned samples?
    python -m src.ahsef.cli.run_audio_expert_comparison --stage aligned

Validation only.  Neither stage reads a test label, and ``--split`` is not
offered: the locked 482-sample test pool stays shut until the routing
configuration is frozen.

Artefacts land in ``experiments/ahsef/analysis/audio_expert/``.  Nothing is
written into ``experiments/audio/``, ``experiments/ahsef/stage1``,
``stage2_llm`` or ``stage3_text_audio``.
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

from src.ahsef.analysis.expert_comparison import (
    MATERIAL_IMPROVEMENT,
    calibration_block,
    encoder_cost_from_provenance,
    headroom_comparison,
    improvement_verdict,
    load_phase_a_headroom,
    latency_profile,
    metric_block,
    pair_headroom,
)
from src.ahsef.evaluation import uncertainty_summary
from src.ahsef.fusion import FusionSpec, fuse_prediction_sets, select_weights
from src.ahsef.identity import AlignmentIndex
from src.ahsef.inference import BaselinePredictor, PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.registry import DEFAULT_BASELINES
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.audio_strong_experiment import DEFAULT_CACHE_ROOT
from src.training.audio_strong_manifest import STRONG_EXPERIMENT

OUTPUT = Path("experiments") / "ahsef" / "analysis" / "audio_expert"
STRONG_MODALITY = "audio_strong"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=["compare", "aligned"])
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--stage1-run", default="stage1")
    parser.add_argument("--stage3-run", default="stage3_text_audio")
    parser.add_argument("--experiment", default=STRONG_EXPERIMENT)
    parser.add_argument("--iteration", default=None)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--fusion-method", default="weighted_probability",
                        choices=["weighted_probability", "log_opinion_pool"])
    return parser


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# ============================================================
# Loading
# ============================================================

def strong_predictions(args, split: str = "validation") -> PredictionSet:
    """Score the strong expert, caching the export so re-runs are instant."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parquet = output / f"{STRONG_MODALITY}__{split}.parquet"
    if parquet.exists():
        return PredictionSet.load(parquet, parquet.with_suffix(".json"))

    from src.ahsef.registry import BaselineRef

    reference = BaselineRef(
        STRONG_MODALITY, args.experiment, "canonical_emotion_id", args.iteration
    )
    iteration = reference.resolve_iteration(args.root)
    print(f"[pred] scoring {args.experiment}/iteration {iteration} on {split} ...")
    predictor = BaselinePredictor(
        modality=STRONG_MODALITY, experiment=args.experiment,
        iteration=iteration, root=args.root, device="cpu",
    )
    predictions = predictor.predict(split)
    predictions.save(parquet, parquet.with_suffix(".json"))
    return predictions


def baseline_predictions(args, split: str = "validation") -> PredictionSet:
    layout = AhsefLayout(run=args.stage1_run, root=Path(args.root) / "ahsef")
    path = layout.prediction_path("audio", split)
    if not path.exists():
        raise SystemExit(
            f"Missing the frozen audio baseline export at {path}; Phase B-1 compares "
            f"against it and will not re-infer it."
        )
    return PredictionSet.load(path, layout.prediction_meta_path("audio", split))


def llm_predictions(args, split: str = "validation") -> PredictionSet:
    layout = AhsefLayout(run=args.stage3_run, root=Path(args.root) / "ahsef")
    path = layout.prediction_path(LLM_MODALITY, split)
    if not path.exists():
        raise SystemExit(
            f"Missing the Stage 3 Gemma export at {path}. Phase C compares audio "
            f"experts against the LLM on the aligned pool."
        )
    return PredictionSet.load(path, layout.prediction_meta_path(LLM_MODALITY, split))


# ============================================================
# Stage: compare  (PHASE B-1)
# ============================================================

def stage_compare(args) -> int:
    baseline = baseline_predictions(args)
    strong = strong_predictions(args)

    shared = sorted(set(baseline.sample_ids()) & set(strong.sample_ids()))
    if len(shared) != len(baseline.frame):
        raise SystemExit(
            f"The two experts cover different validation samples "
            f"({len(baseline.frame)} vs {len(strong.frame)}, {len(shared)} shared). "
            f"The comparison is only meaningful on identical samples."
        )
    baseline = baseline.restricted_to(shared)
    strong = strong.restricted_to(shared)
    print(f"[cmp ] {len(shared)} identical validation samples")

    baseline_metrics = metric_block(baseline, "audio_baseline")
    strong_metrics = metric_block(strong, "audio_strong")
    verdict = improvement_verdict(baseline_metrics, strong_metrics)

    encoder_ms, encoder_name = encoder_cost_from_provenance(
        Path(DEFAULT_CACHE_ROOT) / "validation" / "extraction_provenance.json"
    )
    record = {
        "phase": "B-1 -- baseline audio vs strong audio",
        "split": "validation",
        "samples": len(shared),
        "identical_samples": True,
        "systems": {
            "audio_baseline": baseline_metrics,
            "audio_strong": strong_metrics,
        },
        "calibration": {
            "audio_baseline": calibration_block(baseline, "audio_baseline", args.bins),
            "audio_strong": calibration_block(strong, "audio_strong", args.bins),
        },
        "uncertainty": {
            "audio_baseline": uncertainty_summary(baseline),
            "audio_strong": uncertainty_summary(strong),
        },
        "latency": {
            "audio_baseline": latency_profile(baseline),
            "audio_strong": latency_profile(strong, encoder_ms, encoder_name),
        },
        "resource_cost": _resource_cost(baseline, strong),
        "verdict": verdict,
        "test_data_accessed": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "environment": {"platform": platform.platform(), "python": sys.version},
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase_b1_comparison.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    _print_compare(record)
    print(f"\nWritten: {output / 'phase_b1_comparison.json'}")
    return 0


def _resource_cost(baseline: PredictionSet, strong: PredictionSet) -> dict:
    """Trainable parameters on each side, and the frozen encoder charged apart."""
    def parameters(item: PredictionSet) -> int | None:
        model = item.meta.get("model") or {}
        return model.get("parameters")

    return {
        "audio_baseline": {
            "trainable_parameters": parameters(baseline),
            "pretrained_parameters": 0,
            "pretrained_weights_used": False,
        },
        "audio_strong": {
            "trainable_parameters": parameters(strong),
            "pretrained_parameters": 94_400_000,
            "pretrained_weights_used": True,
            "note": (
                "The 94.4M encoder parameters are pretrained and frozen -- not "
                "trained by this project and not counted as trainable. They are "
                "listed because they are paid for at inference time."
            ),
        },
    }


def _print_compare(record: dict) -> None:
    left = record["systems"]["audio_baseline"]
    right = record["systems"]["audio_strong"]
    print()
    print("=" * 96)
    print(f"PHASE B-1 -- audio baseline vs audio strong "
          f"(validation, n={record['samples']}, identical samples)")
    print("=" * 96)
    print(f"  {'System':<18}{'Acc':>9}{'Macro-F1':>11}{'Weighted-F1':>13}"
          f"{'MacroP':>9}{'MacroR':>9}{'BalAcc':>9}")
    for name, block in (("audio_baseline", left), ("audio_strong", right)):
        print(f"  {name:<18}{block['accuracy']:>9.4f}{block['macro_f1']:>11.4f}"
              f"{block['weighted_f1']:>13.4f}{block['macro_precision']:>9.4f}"
              f"{block['macro_recall']:>9.4f}{block['balanced_accuracy']:>9.4f}")
    print()
    print(f"  {'Class':<10}{'n':>7}{'base F1':>10}{'strong F1':>11}{'dF1':>9}"
          f"{'base R':>9}{'strong R':>10}{'dR':>9}")
    for klass, entry in record["verdict"]["per_class"].items():
        print(f"  {klass:<10}{entry['support']:>7}{entry['baseline_f1']:>10.4f}"
              f"{entry['strong_f1']:>11.4f}{entry['f1_delta']:>+9.4f}"
              f"{entry['baseline_recall']:>9.4f}{entry['strong_recall']:>10.4f}"
              f"{entry['recall_delta']:>+9.4f}")
    print()
    for name in ("audio_baseline", "audio_strong"):
        cal = record["calibration"][name]
        unc = record["uncertainty"][name]
        lat = record["latency"][name]
        if cal.get("available"):
            print(f"  {name:<18} ECE={cal['ece']:.4f} MCE={cal['mce']:.4f} "
                  f"Brier={cal['brier']:.4f} NLL={cal['nll']:.4f} "
                  f"meanConf={cal['mean_confidence']:.4f}")
        print(f"  {'':<18} mean uncertainty={unc['mean_normalized_entropy']:.4f} "
              f"separation={unc.get('uncertainty_separation', float('nan')):+.4f} "
              f"| latency {lat['deployment_ms_per_sample']:.2f} ms/sample")
    verdict = record["verdict"]
    print()
    print("=" * 96)
    print("PHASE B-1 DECISION CHECKPOINT")
    print("=" * 96)
    print(f"  rule      : {verdict['rule']['rule']}")
    print(f"  measured  : {verdict['statement']}")
    print(f"  improved  : {verdict['classes_improved'] or 'none'}")
    print(f"  regressed : {verdict['classes_regressed'] or 'none'}")
    print(f"  DECISION  : {verdict['decision']}")


# ============================================================
# Stage: aligned  (PHASE C)
# ============================================================

def stage_aligned(args) -> int:
    index = AlignmentIndex.from_experiments(
        [
            ("audio", DEFAULT_BASELINES["audio"].experiment, "canonical_emotion_id",
             "emotion_7class"),
            ("text", DEFAULT_BASELINES["text"].experiment, "canonical_emotion_id",
             "emotion_7class"),
        ],
        root=args.root,
    )
    pool = index.fusion_pool(["audio", "text"], "validation", minimum=1)
    print(f"[pool] {len(pool)} aligned Text+Audio validation samples")

    baseline = baseline_predictions(args)
    strong = strong_predictions(args)
    llm = llm_predictions(args)

    # The strong expert must cover the same pool. It uses the baseline's
    # validation manifest verbatim, so a gap here would mean the cache is
    # incomplete rather than that the pool is wrong.
    missing = [item for item in pool if item not in set(strong.sample_ids())]
    if missing:
        raise SystemExit(
            f"{len(missing)} pooled samples have no strong-audio prediction "
            f"(e.g. {missing[:5]}). Extract their features and re-score before "
            f"comparing; a missing prediction is never substituted."
        )
    usable = set(llm.frame.loc[llm.frame["llm_usable"].astype(bool), "sample_id"].astype(str))
    pool = [item for item in pool if item in usable]
    print(f"[pool] {len(pool)} of those have usable Gemma evidence")

    systems = {
        "text_llm": llm.restricted_to(pool),
        "audio_baseline": baseline.restricted_to(pool),
        "audio_strong": strong.restricted_to(pool),
    }
    blocks = {name: metric_block(item, name) for name, item in systems.items()}

    # Fusion weights chosen on validation, per expert, by macro-F1 -- the same
    # rule Stage 3 froze, so the two fusions are comparable to each other and to it.
    fusions, specs = {}, {}
    for name, audio_set in (("baseline", systems["audio_baseline"]),
                            ("strong", systems["audio_strong"])):
        spec = select_weights(
            {"text_llm": systems["text_llm"], "audio": audio_set}, pool,
            "validation", method=args.fusion_method, objective="macro_f1",
        )
        fused = fuse_prediction_sets(
            {"text_llm": systems["text_llm"], "audio": audio_set}, spec, pool
        )
        specs[name] = spec.to_dict()
        fusions[name] = fused
        blocks[f"text_llm+audio_{name}"] = metric_block(fused, f"text_llm+audio_{name}")
        print(f"[fuse] text_llm+audio_{name}: weights={spec.weights}")

    headroom = {
        "baseline": pair_headroom(systems["text_llm"], systems["audio_baseline"], pool),
        "strong": pair_headroom(systems["text_llm"], systems["audio_strong"], pool),
    }
    comparison = headroom_comparison(
        headroom["baseline"], headroom["strong"], load_phase_a_headroom()
    )

    record = {
        "phase": "C -- strong audio on the aligned Text+Audio pool",
        "split": "validation",
        "pool_samples": len(pool),
        "systems": blocks,
        "fusion_specs": specs,
        "headroom": comparison,
        "fusion_gain": {
            name: {
                "over_text_llm": (
                    blocks[f"text_llm+audio_{name}"]["macro_f1"]
                    - blocks["text_llm"]["macro_f1"]
                ),
                "accuracy_over_text_llm": (
                    blocks[f"text_llm+audio_{name}"]["accuracy"]
                    - blocks["text_llm"]["accuracy"]
                ),
            }
            for name in ("baseline", "strong")
        },
        "test_data_accessed": False,
        "locked_test_pool_opened": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "phase_c_aligned.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    _print_aligned(record)
    print(f"\nWritten: {output / 'phase_c_aligned.json'}")
    return 0


def _print_aligned(record: dict) -> None:
    print()
    print("=" * 96)
    print(f"PHASE C -- aligned Text+Audio validation pool (n={record['pool_samples']})")
    print("=" * 96)
    print(f"  {'System':<24}{'Acc':>9}{'Macro-F1':>11}{'Weighted-F1':>13}{'BalAcc':>9}")
    for name in ("text_llm", "audio_baseline", "audio_strong",
                 "text_llm+audio_baseline", "text_llm+audio_strong"):
        block = record["systems"].get(name)
        if not block:
            continue
        print(f"  {name:<24}{block['accuracy']:>9.4f}{block['macro_f1']:>11.4f}"
              f"{block['weighted_f1']:>13.4f}{block['balanced_accuracy']:>9.4f}")
    print()
    print(f"  {'Per-class F1':<24}" + "".join(f"{c[:7]:>9}" for c in CANONICAL_EMOTION_CLASSES))
    for name in ("text_llm", "audio_baseline", "audio_strong",
                 "text_llm+audio_baseline", "text_llm+audio_strong"):
        block = record["systems"].get(name)
        if not block:
            continue
        print(f"  {name:<24}" + "".join(
            f"{block['per_class'][c]['f1']:>9.3f}" for c in CANONICAL_EMOTION_CLASSES
        ))
    print()
    print("  Routing headroom against Gemma:")
    print(f"  {'Audio expert':<18}{'best single':>13}{'oracle':>9}{'headroom':>11}"
          f"{'disagree':>11}{'audio fixes LLM':>17}")
    for name in ("baseline_audio", "strong_audio"):
        entry = record["headroom"][name]
        print(f"  {name:<18}{entry['best_single_accuracy']:>13.4f}"
              f"{entry['oracle_either_accuracy']:>9.4f}"
              f"{entry['headroom_over_best_single']:>+11.4f}"
              f"{entry['disagreement_rate']:>11.3f}"
              f"{entry['right_fixes_left']:>17}")
    print()
    print(f"  {record['headroom']['statement']}")
    phase_a = record["headroom"]["phase_a"]
    if phase_a.get("available"):
        recorded = phase_a["recorded"]
        print()
        print("  Against Phase A (%s, n=%d):" % (phase_a["pair"], recorded["samples"]))
        print(f"  {'':<18}{'oracle':>13}{'headroom':>11}{'disagree':>11}")
        print(f"  {'phase A recorded':<18}{recorded['oracle_either_accuracy']:>13.4f}"
              f"{recorded['headroom_over_best_single']:>11.4f}"
              f"{recorded['disagreement_rate']:>11.3f}")
        print(f"  {'phase C strong':<18}"
              f"{record['headroom']['strong_audio']['oracle_either_accuracy']:>13.4f}"
              f"{record['headroom']['strong_audio']['headroom_over_best_single']:>11.4f}"
              f"{record['headroom']['strong_audio']['disagreement_rate']:>11.3f}")
        print(f"  baseline reproduces Phase A: "
              f"{phase_a['baseline_reproduces_phase_a']}")
    print()
    print(f"  {record['headroom']['phase_a_statement']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"compare": stage_compare, "aligned": stage_aligned}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
