"""Measure baseline calibration and fit a validation-only temperature.

    python -m src.ahsef.cli.run_calibration --run stage1
    python -m src.ahsef.cli.run_calibration --run stage1 --bins 20

For each exported modality this reports ECE, MCE, Brier, NLL, reliability bins,
and the confidence-vs-correctness split, on validation *and* on test, and fits
one temperature on validation.

The temperature is fitted on validation logits and never on test logits; the
test report exists so that a reader can see whether a validation-fitted
correction transferred, which is a *result*, not an input to any decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ahsef.calibration import (
    calibration_report,
    compare_calibration,
    fit_temperature,
    recommend_calibration,
)
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.provenance import AhsefPolicy, write_provenance
from src.ahsef.uncertainty import probabilities_from_logits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage1")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--ahsef-root", default=None)
    parser.add_argument("--bins", type=int, default=15, help="Equal-width ECE bins.")
    parser.add_argument(
        "--ece-threshold", type=float, default=0.05,
        help="ECE at or below which a baseline is called well calibrated.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=200, help="Temperature optimisation steps."
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ahsef_root = Path(args.ahsef_root) if args.ahsef_root else Path(args.root) / "ahsef"
    layout = AhsefLayout(run=args.run, root=ahsef_root)
    layout.prepare()

    available = layout.existing_predictions()
    modalities = sorted({modality for modality, _ in available})
    if not modalities:
        raise SystemExit(
            f"No predictions in {layout.predictions_dir}. "
            f"Run: python -m src.ahsef.cli.export_predictions --run {args.run}"
        )

    temperatures: dict[str, dict] = {}
    summary_rows = []

    print("=" * 100)
    print(f"{'modality':<12}{'split':<12}{'N':>7}{'acc':>9}{'conf':>9}"
          f"{'ECE':>9}{'MCE':>9}{'NLL':>9}{'T':>8}{'ECE(T)':>9}")
    print("=" * 100)

    for modality in modalities:
        validation_path = layout.prediction_path(modality, "validation")
        if not validation_path.exists():
            print(f"[skip] {modality}: no validation predictions, cannot fit a temperature")
            continue
        validation = PredictionSet.load(validation_path)

        scaler = fit_temperature(
            validation.logits(), validation.labels(), split="validation",
            max_iter=args.max_iter, learning_rate=args.learning_rate,
        )
        validation_reports = compare_calibration(
            validation.logits(), validation.labels(), scaler, num_bins=args.bins
        )
        recommendation = recommend_calibration(validation_reports, args.ece_threshold)

        record = {
            "modality": modality,
            "class_order": list(validation.class_order),
            "experiment": validation.meta.get("experiment"),
            "iteration": validation.meta.get("iteration"),
            "checkpoint_sha256": validation.meta.get("checkpoint_sha256"),
            "validation": validation_reports,
            "recommendation": recommendation,
        }
        temperatures[modality] = {
            **scaler.to_dict(),
            "recommended": recommendation["recommended"],
        }

        print(
            f"{modality:<12}{'validation':<12}{validation_reports['raw']['samples']:>7}"
            f"{validation_reports['raw']['accuracy']:>9.4f}"
            f"{validation_reports['raw']['mean_confidence']:>9.4f}"
            f"{validation_reports['raw']['ece']:>9.4f}"
            f"{validation_reports['raw']['mce']:>9.4f}"
            f"{validation_reports['raw']['nll']:>9.4f}"
            f"{scaler.temperature:>8.3f}"
            f"{validation_reports['calibrated']['ece']:>9.4f}"
        )
        summary_rows.append({
            "modality": modality, "split": "validation",
            "accuracy": validation_reports["raw"]["accuracy"],
            "mean_confidence": validation_reports["raw"]["mean_confidence"],
            "ece": validation_reports["raw"]["ece"],
            "temperature": scaler.temperature,
            "ece_calibrated": validation_reports["calibrated"]["ece"],
            "recommended": recommendation["recommended"],
        })

        test_path = layout.prediction_path(modality, "test")
        if test_path.exists():
            test = PredictionSet.load(test_path)
            raw = calibration_report(
                probabilities_from_logits(test.logits()), test.labels(), args.bins, "raw"
            )
            calibrated = calibration_report(
                scaler.apply(test.logits()), test.labels(), args.bins, "temperature_scaled"
            )
            record["test"] = {
                "raw": raw, "calibrated": calibrated,
                "temperature_source": "fitted on validation; applied unchanged to test",
                "temperature_fitted_on_test": False,
            }
            print(
                f"{'':<12}{'test':<12}{raw['samples']:>7}{raw['accuracy']:>9.4f}"
                f"{raw['mean_confidence']:>9.4f}{raw['ece']:>9.4f}{raw['mce']:>9.4f}"
                f"{raw['nll']:>9.4f}{scaler.temperature:>8.3f}{calibrated['ece']:>9.4f}"
            )
            summary_rows.append({
                "modality": modality, "split": "test",
                "accuracy": raw["accuracy"], "mean_confidence": raw["mean_confidence"],
                "ece": raw["ece"], "temperature": scaler.temperature,
                "ece_calibrated": calibrated["ece"],
                "recommended": recommendation["recommended"],
            })

        layout.calibration_path(modality).write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    layout.temperature_path.write_text(
        json.dumps({
            "fitted_on_split": "validation",
            "uses_test_labels": False,
            "ece_bins": args.bins,
            "ece_threshold": args.ece_threshold,
            "modalities": temperatures,
        }, indent=2),
        encoding="utf-8",
    )

    policy = AhsefPolicy(calibration_ece_bins=args.bins)
    write_provenance(
        layout.provenance_path, run=args.run, stage="calibration", policy=policy,
        baselines={
            modality: {"temperature": record["temperature"],
                       "recommended": record["recommended"]}
            for modality, record in temperatures.items()
        },
        extra={"summary": summary_rows},
    )

    print()
    print(f"Written: {layout.temperature_path}")
    for modality in temperatures:
        print(f"Written: {layout.calibration_path(modality)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
