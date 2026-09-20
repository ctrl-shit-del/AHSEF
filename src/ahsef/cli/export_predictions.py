"""Export per-sample predictions from every frozen unimodal baseline.

    python -m src.ahsef.cli.export_predictions --run stage1
    python -m src.ahsef.cli.export_predictions --run stage1 --modalities audio text
    python -m src.ahsef.cli.export_predictions --run stage1 --splits validation

Reads the baselines strictly read-only: the architecture is rebuilt from each
run summary, the best checkpoint is hashed before and after the pass, and every
artefact is written under ``experiments/ahsef/<run>/``.

Existing exports are skipped unless ``--overwrite`` is given, because a video
pass over 343 clips is minutes of CPU and there is no reason to repeat it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ahsef.costs import cost_from_predictions
from src.ahsef.inference import BaselinePredictor
from src.ahsef.layout import SPLITS, AhsefLayout
from src.ahsef.provenance import AhsefPolicy, write_provenance
from src.ahsef.registry import DEFAULT_BASELINES, ordered_modalities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage1", help="AHSEF run directory name.")
    parser.add_argument(
        "--root", default="experiments", help="Root holding the baseline experiments."
    )
    parser.add_argument(
        "--ahsef-root", default=None,
        help="Root for AHSEF artefacts (default: <root>/ahsef).",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=None,
        choices=sorted(DEFAULT_BASELINES),
        help="Subset of modalities to export (default: all five).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["validation", "test"], choices=list(SPLITS),
        help="Splits to score. Train is available but not exported by default.",
    )
    parser.add_argument(
        "--iteration", default=None,
        help="Force one iteration label for every modality (default: each "
             "experiment summary's validation-best iteration).",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded in provenance.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run modalities whose predictions already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()

    modalities = ordered_modalities(args.modalities)
    exported: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for modality in modalities:
        pending = [
            split for split in args.splits
            if args.overwrite or not layout.prediction_path(modality, split).exists()
        ]
        if not pending:
            print(f"[skip]   {modality}: all requested splits already exported")
            reference = DEFAULT_BASELINES[modality]
            exported[modality] = {
                "experiment": reference.experiment,
                "iteration": args.iteration or reference.resolve_iteration(args.root),
                "splits": {},
                "reused_existing_export": True,
            }
            continue

        print(f"[export] {modality}: {', '.join(pending)}", flush=True)
        try:
            predictor = BaselinePredictor.for_modality(
                modality, root=args.root, iteration=args.iteration, device=args.device
            )
        except (FileNotFoundError, ValueError) as error:
            failures[modality] = str(error)
            print(f"[FAIL]   {modality}: {error}", file=sys.stderr)
            continue

        record = {
            "experiment": predictor.experiment,
            "iteration": predictor.iteration,
            "task": predictor.run_summary.get("task"),
            "class_order": list(predictor.class_order),
            "checkpoint": str(predictor.checkpoint_path),
            "checkpoint_sha256": predictor.checkpoint_digest,
            "splits": {},
        }
        for split in pending:
            prediction_set = predictor.predict(split)
            prediction_set.save(
                layout.prediction_path(modality, split),
                layout.prediction_meta_path(modality, split),
            )
            cost = cost_from_predictions(prediction_set)
            record["splits"][split] = {
                "samples": int(len(prediction_set.frame)),
                "path": str(layout.prediction_path(modality, split)),
                "mean_latency_ms": cost.latency_ms,
                "compute_units": cost.compute_units,
            }
            print(
                f"         {split:<11} {len(prediction_set.frame):>6} samples  "
                f"{cost.latency_ms:8.3f} ms/sample",
                flush=True,
            )
        exported[modality] = record

    policy = AhsefPolicy(seed=args.seed)
    write_provenance(
        layout.provenance_path, run=args.run, stage="export_predictions",
        policy=policy, baselines=exported,
        extra={
            "splits_exported": list(args.splits),
            "device": args.device,
            "failures": failures,
            "command": " ".join(["python", "-m", "src.ahsef.cli.export_predictions", *(argv or sys.argv[1:])]),
        },
    )
    print(f"\nProvenance: {layout.provenance_path}")
    print(json.dumps({"exported": sorted(exported), "failed": sorted(failures)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
