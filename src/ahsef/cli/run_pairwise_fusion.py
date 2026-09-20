"""Anchor + candidate pairwise fusion, per-sample dU, and the class-wise table.

    python -m src.ahsef.cli.run_pairwise_fusion --run stage1 --anchor audio
    python -m src.ahsef.cli.run_pairwise_fusion --run stage1 --anchor text --lambda-cost 0.2

For each candidate modality this command:

1. asks :mod:`src.ahsef.identity` whether the pair may be fused at all, and
   records the refusal when it may not;
2. selects the fusion weight on **validation** by an exhaustive grid scan;
3. applies that fixed weight to **test**, which is opened once, for evaluation;
4. writes per-sample ``dU(m | A)`` records for both splits;
5. reports accuracy, macro/weighted F1, per-class precision/recall/F1,
   uncertainty, latency, and compute cost for every fused pair.

It also emits the modality x emotion per-class F1 table over the unimodal
baselines, which is the evidence for whether candidates are complementary at
all.

No parameter here is chosen using test labels.  The scan refuses to run on the
test split, and the weight applied to test is the one validation produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ahsef.costs import CostModel, cost_from_predictions
from src.ahsef.evaluation import (
    agreement,
    full_report,
    modality_emotion_table,
    render_modality_emotion_table,
)
from src.ahsef.fusion import FusionSpec, fuse_prediction_sets, select_weights
from src.ahsef.identity import AlignmentIndex, SampleAlignmentError, SplitContaminationError
from src.ahsef.information_gain import (
    candidate_gain_table,
    delta_uncertainty_records,
    gain_by_uncertainty_band,
    summarise_delta_uncertainty,
)
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.provenance import AhsefPolicy, write_provenance
from src.ahsef.registry import DEFAULT_BASELINES, ordered_modalities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage1")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument("--anchor", default="audio", choices=sorted(DEFAULT_BASELINES))
    parser.add_argument(
        "--candidates", nargs="+", default=None, choices=sorted(DEFAULT_BASELINES),
        help="Default: every modality other than the anchor.",
    )
    parser.add_argument(
        "--method", default="weighted_probability",
        choices=["weighted_probability", "log_opinion_pool"],
    )
    parser.add_argument(
        "--weight-objective", default="macro_f1",
        choices=["macro_f1", "accuracy", "weighted_f1"],
    )
    parser.add_argument(
        "--weight-grid-step", type=float, default=0.05,
        help="Granularity of the validation weight scan.",
    )
    parser.add_argument(
        "--minimum-pool", type=int, default=30,
        help="Smallest co-split pool accepted for a fused evaluation.",
    )
    parser.add_argument(
        "--apply-calibration", action="store_true",
        help="Apply each modality's validation-fitted temperature before fusing.",
    )
    parser.add_argument("--cost-normalization", default="max", choices=["max", "minmax", "sum"])
    parser.add_argument(
        "--lambda-cost", type=float, default=0.10,
        help="UGAPR compute-cost penalty; recorded now, consumed in stage 2.",
    )
    parser.add_argument(
        "--mu-latency", type=float, default=0.10,
        help="UGAPR latency penalty; recorded now, consumed in stage 2.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load(layout: AhsefLayout, modality: str, split: str) -> PredictionSet | None:
    path = layout.prediction_path(modality, split)
    return PredictionSet.load(path) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()

    # A non-default fusion rule is a different experiment, so it gets its own
    # artefact names rather than overwriting the default run.
    variant = None if args.method == "weighted_probability" else args.method
    suffix = f"__{variant}" if variant else ""

    candidates = ordered_modalities(
        args.candidates or [m for m in DEFAULT_BASELINES if m != args.anchor]
    )
    candidates = [name for name in candidates if name != args.anchor]
    participants = [args.anchor, *candidates]

    # ---------------------------------------------------------- predictions
    sets: dict[str, dict[str, PredictionSet]] = {}
    for modality in participants:
        loaded = {
            split: item for split in ("validation", "test")
            if (item := _load(layout, modality, split)) is not None
        }
        if loaded:
            sets[modality] = loaded
    if args.anchor not in sets:
        raise SystemExit(
            f"No exported predictions for the anchor {args.anchor!r}. "
            f"Run: python -m src.ahsef.cli.export_predictions --run {args.run}"
        )

    # ---------------------------------------------------------- calibration
    temperatures: dict[str, float] = {}
    if args.apply_calibration:
        if not layout.temperature_path.exists():
            raise SystemExit(
                f"--apply-calibration needs {layout.temperature_path}; "
                f"run src.ahsef.cli.run_calibration first."
            )
        record = json.loads(layout.temperature_path.read_text(encoding="utf-8"))
        for modality, entry in record["modalities"].items():
            if entry.get("recommended") == "temperature_scaled":
                temperatures[modality] = float(entry["temperature"])
        for modality, splits in sets.items():
            if modality in temperatures:
                sets[modality] = {
                    split: item.with_temperature(temperatures[modality])
                    for split, item in splits.items()
                }

    # ------------------------------------------------------------ alignment
    index = AlignmentIndex.from_experiments(
        [
            (name, DEFAULT_BASELINES[name].experiment,
             DEFAULT_BASELINES[name].label_column, DEFAULT_BASELINES[name].task(args.root))
            for name in participants
        ],
        root=args.root,
    )

    # ---------------------------------------------------------- cost model
    cost_sets = {
        modality: splits.get("test") or splits["validation"]
        for modality, splits in sets.items()
    }
    cost_model = CostModel(
        {
            # Geometry comes from the frozen run summary, not from the export,
            # so the compute proxy does not depend on when a modality was scored.
            name: cost_from_predictions(
                item, DEFAULT_BASELINES[name].model_record(args.root)
            )
            for name, item in cost_sets.items()
        },
        normalization=args.cost_normalization,
    )
    cost_table = cost_model.table(participants)

    # -------------------------------------------------------- unimodal view
    unimodal_reports = {
        modality: full_report(splits["test"], name=modality)
        for modality, splits in sets.items() if "test" in splits
    }
    emotion_table = modality_emotion_table(unimodal_reports)

    print("=" * 90)
    print("MODALITY x EMOTION  (per-class F1, each modality on its own test partition)")
    print("=" * 90)
    print(render_modality_emotion_table(emotion_table))
    if emotion_table["excluded_label_spaces"]:
        for order, names in emotion_table["excluded_label_spaces"].items():
            print(f"\n  excluded (different label space): {names} -> [{order}]")

    # ---------------------------------------------------------- pairwise
    grid = tuple(
        round(args.weight_grid_step * step, 6)
        for step in range(int(round(1.0 / args.weight_grid_step)) + 1)
    )
    pair_results: dict[str, dict] = {}
    refusals: dict[str, dict] = {}
    delta_frames_validation: dict[str, object] = {}
    delta_frames_test: dict[str, object] = {}

    print()
    print("=" * 90)
    print(f"PAIRWISE FUSION FROM ANCHOR '{args.anchor}'")
    print("=" * 90)

    for candidate in candidates:
        pair = (args.anchor, candidate)
        label = layout.pair_label(pair)
        if candidate not in sets:
            refusals[label] = {
                "reason": "no exported predictions for the candidate",
                "candidate": candidate,
            }
            print(f"\n  {label}: SKIPPED - no exported predictions")
            continue

        # Every reason a pair is unfusable is collected, not just the first one:
        # audio+physiology has both an empty intersection and an incompatible
        # label space, and reporting only the first would understate it.
        blockers: list[str] = []
        pools: dict[str, list[str]] = {}
        for split in ("validation", "test"):
            try:
                pools[split] = index.fusion_pool(pair, split, minimum=args.minimum_pool)
            except (SampleAlignmentError, SplitContaminationError) as error:
                blockers.append(str(error))
        if index[args.anchor].task != index[candidate].task:
            blockers.append(
                f"label spaces differ: {args.anchor} solves "
                f"{index[args.anchor].task!r}, {candidate} solves {index[candidate].task!r}; "
                f"their posteriors are over different class sets and cannot be pooled."
            )
        if blockers:
            matrix = index.pair_matrix(*pair)
            refusals[label] = {
                "reason": blockers[0],
                "blockers": blockers,
                "candidate": candidate,
                "split_overlap_matrix": matrix,
                "co_split": {split: matrix[split][split] for split in matrix},
            }
            print(f"\n  {label}: NOT FUSABLE")
            for blocker in blockers:
                print(f"      - {blocker}")
            continue

        # Weights are chosen on validation. This is the only place a weight is
        # decided, and it cannot see the test split.
        validation_sets = {name: sets[name]["validation"] for name in pair}
        spec = select_weights(
            validation_sets, pools["validation"], split="validation",
            method=args.method, grid=grid, objective=args.weight_objective,
        )

        pair_dir = layout.fusion_pair_dir(pair, variant)
        pair_dir.mkdir(parents=True, exist_ok=True)
        record: dict = {
            "pair": label,
            "anchor": args.anchor,
            "candidate": candidate,
            "fusion": spec.to_dict(),
            "calibration_applied": {
                name: temperatures.get(name, 1.0) for name in pair
            },
            "pool_sizes": {split: len(ids) for split, ids in pools.items()},
            "splits": {},
        }

        for split, ids in pools.items():
            anchor_only = sets[args.anchor][split].restricted_to(ids)
            candidate_only = sets[candidate][split].restricted_to(ids)
            fused = fuse_prediction_sets(
                {name: sets[name][split] for name in pair}, spec, ids
            )
            fused.save(
                layout.fusion_predictions_path(pair, split, variant),
                layout.fusion_predictions_path(pair, split, variant).with_suffix(".json"),
            )

            frame = delta_uncertainty_records(
                anchor_only, fused, candidate, ids,
                cost_model=cost_model,
                # Normalise over the candidates that were actually measured;
                # a candidate with no export has no cost to normalise against.
                candidate_set=[name for name in candidates if name in cost_model],
            )
            frame.to_parquet(layout.delta_uncertainty_path(pair, split, variant), index=False)
            (delta_frames_validation if split == "validation" else delta_frames_test)[
                candidate
            ] = frame

            record["splits"][split] = {
                "samples": len(ids),
                "anchor_only": full_report(anchor_only, name=f"{args.anchor} (pool)"),
                "candidate_only": full_report(candidate_only, name=f"{candidate} (pool)"),
                "fused": full_report(fused, name=label),
                "agreement": agreement(anchor_only, candidate_only, ids),
                "delta_uncertainty": summarise_delta_uncertainty(frame),
                "delta_uncertainty_by_band": gain_by_uncertainty_band(frame),
                "artefacts": {
                    "fused_predictions": str(layout.fusion_predictions_path(pair, split, variant)),
                    "delta_uncertainty": str(layout.delta_uncertainty_path(pair, split, variant)),
                },
            }

        for split, entry in record["splits"].items():
            layout.fusion_metrics_path(pair, split, variant).write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
        layout.fusion_summary_path(pair, variant).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        pair_results[label] = record

        weight = spec.weights[args.anchor]
        print(f"\n  {label}   weight({args.anchor})={weight:.2f} "
              f"selected on validation by {args.weight_objective}")
        for split in ("validation", "test"):
            if split not in record["splits"]:
                continue
            entry = record["splits"][split]
            print(
                f"      {split:<11} n={entry['samples']:<6} "
                f"anchor macroF1={entry['anchor_only']['metrics']['macro_f1']:.4f}  "
                f"cand macroF1={entry['candidate_only']['metrics']['macro_f1']:.4f}  "
                f"fused macroF1={entry['fused']['metrics']['macro_f1']:.4f}  "
                f"acc={entry['fused']['metrics']['accuracy']:.4f}  "
                f"mean dU={entry['delta_uncertainty']['mean_delta_uncertainty']:+.4f}"
            )

    # ------------------------------------------------------------- summary
    policy = AhsefPolicy(
        seed=args.seed,
        apply_calibration=args.apply_calibration,
        fusion_method=args.method,
        fusion_weight_objective=args.weight_objective,
        cost_normalization=args.cost_normalization,
        lambda_cost=args.lambda_cost,
        mu_latency=args.mu_latency,
        notes={
            "stage": "stage 1 -- HSIG, UGAPR, and the dynamic router are not implemented",
            "lambda_mu": "recorded for reproducibility; no routing decision uses them yet",
        },
    )
    summary = {
        "anchor": args.anchor,
        "candidates": candidates,
        "policy": policy.to_dict(),
        "cost_model": cost_table,
        "modality_emotion_table": emotion_table,
        "unimodal_reports": unimodal_reports,
        "pairs": pair_results,
        "refused_pairs": refusals,
        "empirical_information_gain": {
            "validation": candidate_gain_table(delta_frames_validation)
            if delta_frames_validation else {},
            "test": candidate_gain_table(delta_frames_test) if delta_frames_test else {},
        },
        "safety": {
            "weights_selected_on": "validation",
            "weights_selected_on_test": False,
            "calibration_fitted_on": "validation",
            "test_opened_for": "evaluation only, after every parameter was fixed",
            "fusion_pool_rule": "co-split, contamination-checked, label-agreement-checked",
        },
    }
    path = layout.report_path(f"pairwise_fusion_{args.anchor}{suffix}.json")
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_provenance(
        layout.provenance_path, run=args.run, stage="pairwise_fusion", policy=policy,
        baselines={
            name: {
                "experiment": splits[next(iter(splits))].meta.get("experiment"),
                "iteration": splits[next(iter(splits))].meta.get("iteration"),
                "checkpoint_sha256": splits[next(iter(splits))].meta.get("checkpoint_sha256"),
            }
            for name, splits in sets.items()
        },
        extra={
            "fused_pairs": sorted(pair_results),
            "refused_pairs": {name: entry["reason"] for name, entry in refusals.items()},
        },
    )

    print()
    print("=" * 90)
    print("COST MODEL (measured)")
    print("=" * 90)
    for name, entry in cost_table["modalities"].items():
        print(
            f"  {name:<12} latency={entry['latency_ms']:9.3f} ms  "
            f"(norm {entry['normalized_latency']:.3f})   "
            f"compute={entry['compute_units']:.3e}  (norm {entry['normalized_cost']:.3f})"
        )
    print()
    print(f"Written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
