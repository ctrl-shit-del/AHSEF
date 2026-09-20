"""CLI entry point that builds the WESAD physiological-window experiment.

Example
-------
    python -m src.preprocessing.sampling.run_physiology_sampler \
        --window-seconds 60 --stride-seconds 10 --seed 42

Writes ``experiments/physiology/full/metadata/{train,validation,test}.parquet``
plus ``sampling_summary.json``.  The full eligible window pool is used -- there
is no fraction to sample -- and the split is made over whole subjects so no
recording spans two partitions.

Nothing under ``metadata/`` or the legacy ``results``/``checkpoints`` trees is
touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import LABEL_SPACES
from src.common.paths import DATASETS_DIR
from src.preprocessing.sampling.physiology_windows import (
    DEFAULT_SOURCE,
    SPLITS,
    PhysiologyWindowBuilder,
    PhysiologyWindowConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the subject-disjoint WESAD physiological-window experiment.",
    )
    parser.add_argument("--dataset", default="WESAD")
    parser.add_argument(
        "--task", default="wesad_state_3class",
        choices=sorted(name for name in LABEL_SPACES if name.startswith("wesad")),
        help="Declared label space. WESAD has no canonical emotion target.",
    )
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--sample-rate", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
        metavar=("TRAIN", "VALIDATION", "TEST"),
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--datasets-root", default=str(DATASETS_DIR))
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override the derived experiments/physiology/full/metadata directory.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing experiment metadata directory.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    layout = ExperimentLayout("physiology", 1.0, Path(args.experiment_root))
    output_dir = Path(args.output_dir) if args.output_dir else layout.metadata_dir

    existing = [output_dir / f"{split}.parquet" for split in SPLITS]
    if any(path.exists() for path in existing) and not args.force:
        print(
            f"Refusing to overwrite existing experiment metadata in {output_dir}.\n"
            f"Pass --force to regenerate it.",
            file=sys.stderr,
        )
        return 2

    config = PhysiologyWindowConfig(
        dataset=args.dataset,
        task=args.task,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        sample_rate=args.sample_rate,
        seed=args.seed,
        split_ratios=tuple(args.split_ratios),
        source=args.source,
        datasets_root=args.datasets_root,
    )
    summary = PhysiologyWindowBuilder(config).run(output_dir)

    if not args.quiet:
        _report(summary, layout, output_dir)
    return 0 if summary["integrity"]["leakage_checks_passed"] else 1


def _report(summary: dict, layout: ExperimentLayout, output_dir: Path) -> None:
    splits = summary["splits"]
    print("=" * 70)
    print(f"EXPERIMENT WINDOWS: {layout.name}")
    print("=" * 70)
    print(f"Source                : {summary['source']}")
    print(f"Output                : {output_dir}")
    print(f"Label space           : {summary['label_space']['name']} "
          f"({', '.join(summary['label_space']['classes'])})")
    print(f"Seed                  : {summary['seed']}")
    print(f"Feature dimension     : {summary['features']['dimension']}")
    print()
    print(f"Subjects scanned      : {summary['eligibility']['subjects_scanned']}")
    print(f"Windows extracted     : {summary['pool']['eligible_records']:,}")
    print()
    for split in SPLITS:
        print(
            f"{split:<11} {splits['counts'][split]:>8,}  "
            f"({splits['actual_ratios'][split]:.4f})  "
            f"subjects={splits['subjects'][split]}"
        )
    print()
    print("Class counts (windows):", json.dumps(summary["selection"]["class_counts"], sort_keys=True))
    print("Windows per subject   :",
          json.dumps(summary["deviations"]["windows_per_subject"], sort_keys=True))
    integrity = summary["integrity"]
    print()
    print(f"Duplicate sample IDs            : {integrity['duplicate_sample_ids']}")
    print(f"Subjects in multiple splits     : {integrity['subjects_in_multiple_splits']}")
    print(f"Invalid label records           : {integrity['invalid_label_records']}")
    print(f"Leakage checks passed           : {integrity['leakage_checks_passed']}")
    print()
    print(f"Summary written to {output_dir / 'sampling_summary.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
