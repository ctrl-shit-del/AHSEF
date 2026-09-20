"""CLI entry point that audits a generated experiment subset.

Examples
--------
    python -m src.preprocessing.sampling.run_validation \
        --experiment image_25pct --verify-determinism

    python -m src.preprocessing.sampling.run_validation --experiment text_full
    python -m src.preprocessing.sampling.run_validation --experiment physiology_full

The physiology experiment is built by window expansion rather than record-level
sampling, so it is audited by its own validator -- selected automatically from
the modality recorded in ``sampling_summary.json``.

Exits non-zero when any check fails, so it is safe to gate training on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.experiment_layout import ExperimentLayout
from src.common.paths import DATASETS_DIR
from src.preprocessing.sampling.validate import DEFAULT_FILE_CHECKS, validate_experiment
from src.preprocessing.sampling.validate_physiology import validate_physiology_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a sampled experiment subset.")
    parser.add_argument(
        "--experiment", default="image_25pct",
        help="Experiment identity, for example image_25pct, text_full, physiology_full.",
    )
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument(
        "--metadata-dir", default=None,
        help="Override the derived experiments/<modality>/<fraction>/metadata directory.",
    )
    parser.add_argument(
        "--source", default=None,
        help="Override the standardized source recorded in sampling_summary.json.",
    )
    parser.add_argument("--datasets-root", default=str(DATASETS_DIR))
    parser.add_argument(
        "--check-files", type=int, default=DEFAULT_FILE_CHECKS,
        help="Number of media files to stat; 0 disables, -1 checks every selected record.",
    )
    parser.add_argument(
        "--verify-determinism", action="store_true",
        help="Replan from the source and compare the recorded plan digest (one extra scan).",
    )
    parser.add_argument("--report", default=None, help="Override the report output path.")
    return parser


def _modality_of(metadata_dir: Path, fallback: str) -> str:
    summary = metadata_dir / "sampling_summary.json"
    if not summary.exists():
        return fallback
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    return (payload.get("experiment") or {}).get("modality", fallback)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    layout = ExperimentLayout.from_name(args.experiment, Path(args.experiment_root))
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else layout.metadata_dir
    report_path = Path(args.report) if args.report else metadata_dir / "validation_report.json"

    modality = _modality_of(metadata_dir, layout.modality)
    if modality == "physiology":
        report = validate_physiology_experiment(
            metadata_dir=metadata_dir,
            check_files=args.check_files,
            datasets_root=Path(args.datasets_root),
        )
        if args.verify_determinism:
            report.notes.append(
                "Determinism replan is not applicable to the physiology window builder; "
                "the recorded sample_id digests already pin the selection."
            )
    else:
        report = validate_experiment(
            metadata_dir=metadata_dir,
            source=args.source,
            check_files=args.check_files,
            verify_determinism=args.verify_determinism,
            datasets_root=Path(args.datasets_root),
        )

    payload = report.to_dict()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 70)
    print(f"EXPERIMENT VALIDATION: {layout.name}")
    print("=" * 70)
    print(f"Metadata : {metadata_dir}")
    print(f"Modality : {modality}")
    print(f"Counts   : {report.counts}")
    print()
    for check in report.checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"[{status}] {check['check']}")
        if not check["passed"]:
            print(f"        {json.dumps(check['detail'], sort_keys=True, default=str)}")
    for note in report.notes:
        print(f"[NOTE] {note}")
    print()
    print(f"Overall: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Report written to {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
