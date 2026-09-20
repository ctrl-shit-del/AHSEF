"""CLI entry point that generates a representative fraction experiment.

Example
-------
    python -m src.preprocessing.sampling.run_sampler \
        --modality image --fraction 0.25 --seed 42

Writes ``experiments/image/25pct/metadata/{train,validation,test}.parquet``
plus ``sampling_summary.json``.  Nothing under ``metadata/experiments`` or the
legacy ``results``/``checkpoints`` trees is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.experiment_layout import ExperimentLayout
from src.preprocessing.sampling.rules import MODALITY_RULES, TASK_RULES
from src.preprocessing.sampling.sampler import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SOURCE,
    HOLDOUT_OVERFLOW,
    SPLIT_POLICIES,
    SPLITS,
    ExperimentSampler,
    SamplerConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a representative, stratified fraction of a modality pool.",
    )
    parser.add_argument("--modality", default="image", choices=sorted(MODALITY_RULES))
    parser.add_argument("--task", default="emotion_7class", choices=sorted(TASK_RULES))
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Contributing datasets; defaults to the modality's canonical contributors.",
    )
    parser.add_argument(
        "--stratify-by", nargs="+", default=["dataset", "canonical_emotion_id"],
        help="Stratification columns.",
    )
    parser.add_argument(
        "--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    parser.add_argument("--split-policy", default="official_aware", choices=list(SPLIT_POLICIES))
    parser.add_argument(
        "--holdout-overflow", default="drop", choices=list(HOLDOUT_OVERFLOW),
        help="What to do with selected records whose only permitted split is full "
             "(a dataset whose official holdout exceeds the experiment quota). "
             "'drop' keeps the split ratios exact; 'absorb' uses the whole selected "
             "budget and lets the holdout split grow. Neither ever leaks into train.",
    )
    parser.add_argument("--minimum-per-stratum", type=int, default=1)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the derived experiments/<modality>/<fraction>/metadata directory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing experiment metadata directory.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    layout = ExperimentLayout(args.modality, args.fraction, Path(args.experiment_root))
    output_dir = Path(args.output_dir) if args.output_dir else layout.metadata_dir

    existing = [output_dir / f"{split}.parquet" for split in SPLITS]
    if any(path.exists() for path in existing) and not args.force:
        print(
            f"Refusing to overwrite existing experiment metadata in {output_dir}.\n"
            f"Pass --force to regenerate it.",
            file=sys.stderr,
        )
        return 2

    config = SamplerConfig(
        modality=args.modality,
        task=args.task,
        fraction=args.fraction,
        seed=args.seed,
        stratify_by=tuple(args.stratify_by),
        split_ratios=tuple(args.split_ratios),
        datasets=tuple(args.datasets) if args.datasets else None,
        split_policy=args.split_policy,
        holdout_overflow=args.holdout_overflow,
        minimum_per_stratum=args.minimum_per_stratum,
        batch_size=args.batch_size,
        source=args.source,
    )

    summary = ExperimentSampler(config).run(output_dir)

    if not args.quiet:
        _report(summary, layout, output_dir)
    return 0 if summary["integrity"]["leakage_checks_passed"] else 1


def _report(summary: dict, layout: ExperimentLayout, output_dir: Path) -> None:
    selection = summary["selection"]
    splits = summary["splits"]
    print("=" * 70)
    print(f"EXPERIMENT SAMPLE: {layout.name}")
    print("=" * 70)
    print(f"Source                : {summary['source']}")
    print(f"Output                : {output_dir}")
    print(f"Contributing datasets : {', '.join(summary['contributing_datasets'])}")
    print(f"Stratified by         : {', '.join(summary['stratification_fields'])}")
    print(f"Seed                  : {summary['seed']}")
    print()
    print(f"Eligible pool         : {summary['pool']['eligible_records']:,}")
    print(f"Requested fraction    : {selection['requested_fraction']}")
    print(f"Selected records      : {selection['selected_records']:,}")
    print(f"Actual fraction       : {selection['actual_fraction']:.6f}")
    print()
    for split in SPLITS:
        print(
            f"{split:<11} {splits['counts'][split]:>9,}  "
            f"({splits['actual_ratios'][split]:.4f})  "
            f"datasets={splits['dataset_counts'][split]}"
        )
    print()
    print("Dataset counts (selected):", json.dumps(selection["dataset_counts"], sort_keys=True))
    print("Class counts   (selected):", json.dumps(selection["class_counts"], sort_keys=True))
    print(
        f"Max dataset share drift  : "
        f"{summary['representativeness']['max_dataset_share_drift']:.6f}"
    )
    print(
        f"Max class share drift    : "
        f"{summary['representativeness']['max_class_share_drift']:.6f}"
    )
    print()
    deviations = summary["deviations"]
    print(f"Minimum-per-stratum adjustments : {len(deviations['minimum_per_stratum_adjustments'])}")
    print(f"Quota exceptions                : {len(deviations['quota_exceptions'])}")
    print(f"Unplaced eligible records       : {deviations['unplaced_eligible_records']}")
    if deviations["unplaced_eligible_records"]:
        print(f"    {deviations['unplaced_reason']}")
    integrity = summary["integrity"]
    print(f"Duplicate sample IDs            : {integrity['duplicate_sample_ids']}")
    print(f"Cross-split sample IDs          : {integrity['cross_split_sample_ids']}")
    print(f"Official holdout in train       : {integrity['official_holdout_in_train']}")
    print(f"Leakage checks passed           : {integrity['leakage_checks_passed']}")
    print()
    print(f"Summary written to {output_dir / 'sampling_summary.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
