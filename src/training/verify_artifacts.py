"""Post-run verification of an iteration's checkpoints and result artefacts.

Answers, without retraining: are all artefacts present, does ``best.pt`` reload
into the architecture the run actually recorded, does it really correspond to
the highest validation macro-F1 in the training history, were the class weights
derived from the training split alone, is the confusion matrix square in the
declared class space, and did the test partition stay closed until training had
finished?

The architecture is rebuilt from ``run_summary['model']`` through the modality
registry, so the same command verifies any modality:

    python -m src.training.verify_artifacts --experiment image_25pct --iteration 1
    python -m src.training.verify_artifacts --experiment audio_25pct
    python -m src.training.verify_artifacts --experiment physiology_full

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.experiment_layout import ExperimentLayout
from src.training.checkpoint import CheckpointManager
from src.training.modalities import FORWARD_PROBES, build_model_from_record


REQUIRED_RESULTS = (
    "training_history.json",
    "validation_metrics.json",
    "test_metrics.json",
    "confusion_matrix.json",
    "run_summary.json",
    "class_weights.json",
)
TOLERANCE = 1e-6


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _forward_probe(model, run_summary: dict) -> torch.Tensor:
    """Run one bounded forward pass shaped for the recorded architecture."""
    record = run_summary.get("model") or {}
    config = run_summary.get("config") or {}
    probe = FORWARD_PROBES.get(record.get("class"))
    if probe is None:
        raise ValueError(f"No forward probe registered for {record.get('class')!r}")
    input_shape, lengths_shape, dtype = probe(record, config)
    inputs = torch.ones(*input_shape, dtype=dtype)
    if lengths_shape is None:
        return model(inputs)
    lengths = torch.ones(*lengths_shape, dtype=torch.long) * input_shape[1]
    return model(inputs, lengths)


def verify_iteration(layout: ExperimentLayout, iteration: str) -> dict:
    """Return a check report for one iteration's artefacts."""
    checks: list[dict] = []

    def record(name: str, passed: bool, detail=None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    results_dir = layout.results_dir(iteration)
    for name in REQUIRED_RESULTS:
        record(f"results/{name}", (results_dir / name).exists(), str(results_dir / name))
    record("config.json", layout.config_path(iteration).exists(), str(layout.config_path(iteration)))
    record("logs/training.log", layout.log_path(iteration).exists(), str(layout.log_path(iteration)))

    for label, path in (("best.pt", layout.best_checkpoint(iteration)),
                        ("last.pt", layout.last_checkpoint(iteration))):
        record(f"checkpoints/{label}", path.exists(), str(path))

    run_summary = _read_json(results_dir / "run_summary.json")
    history = _read_json(results_dir / "training_history.json")
    weights_record = _read_json(results_dir / "class_weights.json")
    confusion = _read_json(results_dir / "confusion_matrix.json")

    if run_summary is None or history is None:
        record("artefacts_readable", False, "run_summary.json or training_history.json missing")
        return {"iteration": iteration, "passed": False, "checks": checks}
    record("artefacts_readable", True)

    config = run_summary["config"]
    num_classes = int(config["num_classes"])

    try:
        model = build_model_from_record(run_summary)
        checkpoint = CheckpointManager.load(
            layout.best_checkpoint(iteration), model=model, optimizer=None, device="cpu",
        )
        record("best_checkpoint_loads_into_recorded_architecture", True, {
            "class": (run_summary.get("model") or {}).get("class"),
            "parameters": sum(p.numel() for p in model.parameters()),
            "epoch": checkpoint["epoch"],
        })
    except Exception as error:  # pragma: no cover - surfaced to the operator
        record("best_checkpoint_loads_into_recorded_architecture", False, str(error))
        return {"iteration": iteration, "passed": False, "checks": checks}

    record("best_checkpoint_monitors_validation_macro_f1",
           checkpoint.get("monitor") == "macro_f1", checkpoint.get("monitor"))

    best_row = max(history, key=lambda row: row["val_macro_f1"])
    record(
        "best_checkpoint_is_the_highest_validation_macro_f1",
        checkpoint["epoch"] == best_row["epoch"]
        and abs(float(checkpoint["monitor_value"]) - best_row["val_macro_f1"]) <= TOLERANCE,
        {"checkpoint_epoch": checkpoint["epoch"], "history_best_epoch": best_row["epoch"],
         "checkpoint_value": checkpoint["monitor_value"], "history_value": best_row["val_macro_f1"]},
    )
    record(
        "run_summary_agrees_with_checkpoint",
        run_summary["best_epoch"] == checkpoint["epoch"]
        and abs(run_summary["best_val_macro_f1"] - float(checkpoint["monitor_value"])) <= TOLERANCE,
        {"run_summary": run_summary["best_epoch"], "checkpoint": checkpoint["epoch"]},
    )

    extra = checkpoint.get("extra") or {}
    record(
        "checkpoint_carries_experiment_identity",
        extra.get("experiment") == layout.name and str(extra.get("iteration")) == str(iteration)
        and len(extra.get("class_weights") or []) == num_classes,
        {"experiment": extra.get("experiment"), "iteration": extra.get("iteration")},
    )

    try:
        logits = _forward_probe(model, run_summary)
        forward_ok = tuple(logits.shape) == (2, num_classes)
        forward_detail = {"expected": [2, num_classes], "observed": list(logits.shape)}
    except Exception as error:  # pragma: no cover - surfaced to the operator
        forward_ok, forward_detail = False, str(error)
    record("reloaded_model_emits_num_classes_logits", forward_ok, forward_detail)

    if weights_record is not None:
        record(
            "class_weights_derived_from_train_only",
            weights_record.get("computed_from_split") == "train"
            and weights_record.get("uses_validation_labels") is False
            and weights_record.get("uses_test_labels") is False,
            {"computed_from": weights_record.get("computed_from")},
        )
        record(
            "class_weight_count_matches_class_space",
            len(weights_record.get("weights", [])) == num_classes,
            len(weights_record.get("weights", [])),
        )
        record(
            "class_weight_order_matches_declared_class_order",
            list(weights_record.get("class_order") or []) == list(run_summary.get("class_order")
                                                                  or weights_record.get("class_order") or []),
            {"weights": weights_record.get("class_order"), "run": run_summary.get("class_order")},
        )

    if confusion is not None:
        matrices = {key: confusion.get(key) for key in ("validation", "test")}
        square = all(
            isinstance(matrix, list)
            and len(matrix) == num_classes
            and all(isinstance(row, list) and len(row) == num_classes for row in matrix)
            for matrix in matrices.values()
        )
        record(
            "confusion_matrices_are_square_in_the_declared_class_space",
            square,
            {"num_classes": num_classes,
             "class_order": confusion.get("class_order"),
             "shapes": {key: (len(value) if isinstance(value, list) else None)
                        for key, value in matrices.items()}},
        )

    for name, metrics in (("validation", _read_json(results_dir / "validation_metrics.json")),
                          ("test", _read_json(results_dir / "test_metrics.json"))):
        if metrics is None:
            continue
        required = (
            "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "per_class_precision", "per_class_recall", "per_class_f1", "support",
            "confusion_matrix", "samples",
        )
        missing = [key for key in required if key not in metrics]
        record(f"{name}_metrics_complete", not missing, {"missing": missing})

    record(
        "test_partition_opened_only_after_training",
        all(event["split"] != "test" or event["phase"] == "evaluation"
            for event in run_summary.get("loader_events", [])),
        run_summary.get("loader_events"),
    )

    record(
        "iteration_artefacts_are_isolated",
        str(layout.iteration_dir(iteration)) in str(run_summary.get("log_file", ""))
        and str(layout.iteration_dir(iteration)) in str(
            (run_summary.get("checkpoints") or {}).get("best", "")
        ),
        {"log_file": run_summary.get("log_file"),
         "checkpoints": run_summary.get("checkpoints")},
    )

    return {
        "iteration": iteration,
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def verify_experiment_summary(layout: ExperimentLayout) -> dict:
    """Check that the experiment-level summary exists and is internally sound."""
    checks: list[dict] = []

    def record(name: str, passed: bool, detail=None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    summary = _read_json(layout.experiment_summary_path)
    record("experiment_summary.json", summary is not None, str(layout.experiment_summary_path))
    if summary is None:
        return {"iteration": "experiment", "passed": False, "checks": checks}

    comparison = summary.get("comparison") or {}
    expected = ("best_val_macro_f1", "test_accuracy", "test_macro_f1",
                "test_weighted_f1", "test_loss")
    missing = [name for name in expected if name not in comparison]
    record("experiment_summary_compares_every_headline_metric", not missing, {"missing": missing})

    consistent = all(
        block.get("n") == len(block.get("values", []))
        and (block.get("n") or 0) >= 1
        for block in comparison.values()
    )
    record("experiment_summary_statistics_are_consistent", consistent, comparison)
    record(
        "experiment_summary_lists_iterations",
        bool(summary.get("iterations")),
        {"iteration_count": summary.get("iteration_count")},
    )
    return {"iteration": "experiment", "passed": all(c["passed"] for c in checks), "checks": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify iteration checkpoints and artefacts.")
    parser.add_argument("--experiment", default="image_25pct")
    parser.add_argument("--experiment-root", default="experiments")
    parser.add_argument(
        "--iteration", action="append", default=None,
        help="Iteration label; repeatable. Defaults to every iteration found on disk.",
    )
    parser.add_argument(
        "--skip-experiment-summary", action="store_true",
        help="Verify iterations only, without checking experiment_summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = ExperimentLayout.from_name(args.experiment, Path(args.experiment_root))
    iterations = args.iteration or layout.existing_iterations()

    if not iterations:
        print(f"No iterations found under {layout.base}")
        return 1

    reports = [verify_iteration(layout, iteration) for iteration in iterations]
    if not args.skip_experiment_summary:
        reports.append(verify_experiment_summary(layout))

    ok = True
    for report in reports:
        ok = ok and report["passed"]
        print("=" * 70)
        print(f"{layout.name} | {report['iteration']}: "
              f"{'PASSED' if report['passed'] else 'FAILED'}")
        print("=" * 70)
        for check in report["checks"]:
            print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}")
            if not check["passed"]:
                print(f"        {json.dumps(check['detail'], default=str)}")
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
