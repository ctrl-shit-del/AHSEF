"""Aggregate independent training iterations into one experiment summary.

Iteration results are never merged or averaged into a single prediction set;
each iteration keeps its own artefacts and this module only reports them side
by side, with descriptive statistics over the numeric iterations.  Debug runs
are listed but excluded from the statistics.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import LABEL_SPACES


COMPARED_METRICS = (
    "best_val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_loss",
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _iteration_record(layout: ExperimentLayout, label: str) -> dict | None:
    run_summary = _read_json(layout.results_dir(label) / "run_summary.json")
    if run_summary is None:
        return None
    # A deferred run records test_metrics as null rather than as an empty dict.
    test_metrics = run_summary.get("test_metrics") or {}
    validation_metrics = run_summary.get("validation_metrics", {})
    config = run_summary.get("config", {})
    return {
        "iteration": label,
        "debug": bool(run_summary.get("debug")),
        "modality": run_summary.get("modality") or config.get("modality"),
        "task": run_summary.get("task") or config.get("task"),
        "model": run_summary.get("model"),
        "seed": config.get("seed"),
        "run_seed": config.get("run_seed") if config.get("run_seed") is not None else config.get("seed"),
        "epochs_configured": run_summary.get("epochs_configured"),
        "epochs_completed": run_summary.get("epochs_completed"),
        "best_epoch": run_summary.get("best_epoch"),
        "best_val_macro_f1": run_summary.get("best_val_macro_f1"),
        "validation_macro_f1": validation_metrics.get("macro_f1"),
        "validation_accuracy": validation_metrics.get("accuracy"),
        "validation_weighted_f1": validation_metrics.get("weighted_f1"),
        "test_loss": test_metrics.get("loss"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_macro_f1": test_metrics.get("macro_f1"),
        "test_weighted_f1": test_metrics.get("weighted_f1"),
        "test_per_class_f1": test_metrics.get("per_class_f1"),
        "samples": run_summary.get("samples"),
        "early_stopping": run_summary.get("early_stopping"),
        "stop_reason": run_summary.get("stop_reason"),
        "training_seconds": run_summary.get("training_seconds"),
        "total_seconds": run_summary.get("total_seconds"),
        "checkpoints": run_summary.get("checkpoints"),
        "config_path": str(layout.config_path(label)),
        "results_dir": str(layout.results_dir(label)),
        "log_file": run_summary.get("log_file"),
    }


def _statistics(records: list[dict]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for metric in COMPARED_METRICS:
        values = [record[metric] for record in records if isinstance(record.get(metric), (int, float))]
        if not values:
            continue
        stats[metric] = {
            "values": values,
            "n": len(values),
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return stats


def _run_summaries(layout: ExperimentLayout, records: list[dict]) -> list[dict]:
    summaries = []
    for record in records:
        payload = _read_json(Path(record["results_dir"]) / "run_summary.json")
        if payload is not None:
            summaries.append(payload)
    return summaries


def _class_names(layout: ExperimentLayout, records: list[dict]) -> list[str] | None:
    """The declared class order, taken from the iterations that reported one.

    Iterations written before ``class_order`` was promoted to a top-level field
    still recorded it inside ``class_weights``, so read that as a fallback
    rather than reporting an unknown order for a completed experiment.
    """
    for summary in _run_summaries(layout, records):
        order = summary.get("class_order") or (summary.get("class_weights") or {}).get("class_order")
        if order:
            return list(order)
    return None


def _task_name(records: list[dict], class_names: list[str] | None) -> str | None:
    """The task an iteration declared, or the label space its class order names."""
    for record in records:
        if record.get("task"):
            return record["task"]
    if class_names:
        for space in LABEL_SPACES.values():
            if list(space.classes) == list(class_names):
                return space.name
    return None


def _task_limitation(layout: ExperimentLayout, records: list[dict]) -> str | None:
    for summary in _run_summaries(layout, records):
        limitation = summary.get("task_limitation")
        if limitation:
            return limitation
    return None


def build_experiment_summary(layout: ExperimentLayout) -> dict:
    """Collect every iteration on disk into one comparison document."""
    sampling = _read_json(layout.sampling_summary_path)
    validation_report = _read_json(layout.validation_report_path)

    iterations = [
        record for record in
        (_iteration_record(layout, label) for label in layout.existing_iterations())
        if record is not None
    ]
    comparable = [record for record in iterations if not record["debug"]]
    stats = _statistics(comparable)

    ranked = sorted(
        (record for record in comparable if isinstance(record.get("best_val_macro_f1"), (int, float))),
        key=lambda record: record["best_val_macro_f1"],
        reverse=True,
    )
    best = ranked[0] if ranked else None

    reference = next((record for record in comparable), iterations[0] if iterations else None)
    class_names = _class_names(layout, comparable or iterations)

    return {
        "experiment": layout.name,
        "modality": layout.modality,
        "fraction": layout.fraction,
        "fraction_label": layout.label,
        "dataset_fraction": (
            sampling["selection"]["actual_fraction"] if sampling else layout.fraction
        ),
        "task": _task_name(comparable or iterations, class_names),
        "class_names": class_names,
        "task_limitation": _task_limitation(layout, comparable or iterations),
        "datasets_used": (
            sorted(sampling["selection"]["dataset_counts"])
            if sampling and sampling.get("selection", {}).get("dataset_counts")
            else (sampling.get("contributing_datasets") if sampling else None)
        ),
        "dataset_counts": (
            sampling["selection"]["dataset_counts"] if sampling else None
        ),
        "split_counts": (
            sampling["splits"]["counts"] if sampling
            else (reference.get("samples") if reference else None)
        ),
        "iteration_seeds": {
            record["iteration"]: record.get("run_seed") for record in iterations
        },
        "best_epochs": {
            record["iteration"]: record.get("best_epoch") for record in iterations
        },
        "durations_seconds": {
            record["iteration"]: record.get("total_seconds") for record in iterations
        },
        "base_dir": str(layout.base),
        "metadata_dir": str(layout.metadata_dir),
        "selection_metric": "validation macro_f1 (higher is better)",
        "note": (
            "Iterations are independent repeated runs. Test predictions are never "
            "averaged across iterations; each iteration's artefacts stand alone."
        ),
        "sampling_summary": (
            {
                "path": str(layout.sampling_summary_path),
                "seed": sampling["seed"],
                "sampling_method": sampling["sampling_method"],
                "contributing_datasets": sampling["contributing_datasets"],
                "stratification_fields": sampling["stratification_fields"],
                "eligible_records": sampling["pool"]["eligible_records"],
                "selected_records": sampling["selection"]["selected_records"],
                "requested_fraction": sampling["selection"]["requested_fraction"],
                "actual_fraction": sampling["selection"]["actual_fraction"],
                "split_counts": sampling["splits"]["counts"],
                "actual_split_ratios": sampling["splits"]["actual_ratios"],
                "integrity": sampling["integrity"],
            }
            if sampling else None
        ),
        "validation_report": (
            {
                "path": str(layout.validation_report_path),
                "passed": validation_report.get("passed"),
                "failures": validation_report.get("failures"),
            }
            if validation_report else None
        ),
        "iterations": iterations,
        "iteration_count": len(iterations),
        "comparable_iterations": [record["iteration"] for record in comparable],
        "comparison": stats,
        "best_iteration": (
            {
                "iteration": best["iteration"],
                "best_val_macro_f1": best["best_val_macro_f1"],
                "best_epoch": best["best_epoch"],
                "test_accuracy": best["test_accuracy"],
                "test_macro_f1": best["test_macro_f1"],
                "test_weighted_f1": best["test_weighted_f1"],
                "checkpoints": best["checkpoints"],
            }
            if best else None
        ),
    }


def write_experiment_summary(layout: ExperimentLayout) -> Path:
    """Write ``experiment_summary.json`` and return its path."""
    summary = build_experiment_summary(layout)
    path = layout.experiment_summary_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
