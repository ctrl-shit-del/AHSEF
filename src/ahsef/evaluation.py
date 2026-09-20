"""Evaluation over AHSEF prediction sets.

Reuses :func:`src.training.metrics.classification_metrics` verbatim so a fused
number and a baseline number are produced by the same code, and adds the
things AHSEF specifically has to report: uncertainty summaries, cost and
latency summaries, and the modality x emotion table that motivates the whole
routing argument.

Every function here takes ground truth. Nothing here is called during routing;
labels enter only after a routing decision has already been made.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.ahsef.inference import PredictionSet
from src.training.metrics import classification_metrics


def evaluate_prediction_set(prediction_set: PredictionSet) -> dict:
    """Accuracy, macro/weighted F1, and per-class precision/recall/F1."""
    metrics = classification_metrics(
        prediction_set.predictions(), prediction_set.labels(), prediction_set.num_classes
    )
    metrics["samples"] = int(len(prediction_set.frame))
    metrics["class_order"] = list(prediction_set.class_order)
    metrics["per_class"] = {
        name: {
            "precision": metrics["per_class_precision"][index],
            "recall": metrics["per_class_recall"][index],
            "f1": metrics["per_class_f1"][index],
            "support": metrics["support"][index],
        }
        for index, name in enumerate(prediction_set.class_order)
    }
    return metrics


def uncertainty_summary(prediction_set: PredictionSet) -> dict:
    """Distributional summary of the confidence / uncertainty columns."""
    frame = prediction_set.frame
    correct = (prediction_set.predictions() == prediction_set.labels()).numpy()
    summary = {
        "samples": int(len(frame)),
        "mean_normalized_entropy": float(frame["normalized_entropy"].mean()),
        "median_normalized_entropy": float(frame["normalized_entropy"].median()),
        "std_normalized_entropy": float(frame["normalized_entropy"].std(ddof=0)),
        "mean_predictive_entropy": float(frame["predictive_entropy"].mean()),
        "mean_confidence": float(frame["confidence"].mean()),
        "mean_margin": float(frame["margin"].mean()),
    }
    if correct.any():
        summary["mean_uncertainty_when_correct"] = float(
            frame.loc[correct, "normalized_entropy"].mean()
        )
    if (~correct).any():
        summary["mean_uncertainty_when_wrong"] = float(
            frame.loc[~correct, "normalized_entropy"].mean()
        )
    if correct.any() and (~correct).any():
        summary["uncertainty_separation"] = (
            summary["mean_uncertainty_when_wrong"] - summary["mean_uncertainty_when_correct"]
        )
        summary["separation_note"] = (
            "Positive means the model is more uncertain when it is wrong, which is the "
            "precondition for uncertainty-triggered routing to help at all."
        )
    return summary


def cost_summary(prediction_set: PredictionSet) -> dict:
    """Measured latency of producing this prediction set, per sample."""
    frame = prediction_set.frame
    return {
        "mean_latency_ms": float(frame["latency_ms"].mean()),
        "median_latency_ms": float(frame["latency_ms"].median()),
        "total_latency_ms": float(frame["latency_ms"].sum()),
        "mean_inference_ms": float(frame["inference_ms"].mean()),
        "mean_feature_ms": float(frame["feature_ms"].mean()),
    }


def full_report(prediction_set: PredictionSet, name: str | None = None) -> dict:
    """Metrics, uncertainty, and cost for one prediction set in one record."""
    return {
        "name": name or prediction_set.modality,
        "modality": prediction_set.modality,
        "split": prediction_set.split,
        "class_order": list(prediction_set.class_order),
        "metrics": evaluate_prediction_set(prediction_set),
        "uncertainty": uncertainty_summary(prediction_set),
        "cost": cost_summary(prediction_set),
        "provenance": {
            key: prediction_set.meta.get(key)
            for key in (
                "experiment", "iteration", "checkpoint_sha256", "temperature",
                "kind", "component_modalities", "fusion", "pool_size", "manifest",
            )
            if key in prediction_set.meta
        },
    }


# ============================================================
# Modality x emotion
# ============================================================

def modality_emotion_table(
    reports: Mapping[str, dict], metric: str = "f1"
) -> dict:
    """Per-class ``metric`` for every modality, plus best / worst per emotion.

    Modalities whose class order differs from the majority are reported in a
    separate section rather than being forced into the same row space: a WESAD
    ``stress`` F1 does not belong in an ``angry`` column.
    """
    orders: dict[tuple[str, ...], list[str]] = {}
    for name, report in reports.items():
        orders.setdefault(tuple(report["class_order"]), []).append(name)
    if not orders:
        raise ValueError("No reports to tabulate")

    primary_order = max(orders, key=lambda key: len(orders[key]))
    primary = orders[primary_order]
    other = {
        "|".join(order): sorted(names)
        for order, names in orders.items() if order != primary_order
    }

    rows = {}
    for name in sorted(primary):
        per_class = reports[name]["metrics"]["per_class"]
        rows[name] = {
            emotion: float(per_class[emotion][metric]) for emotion in primary_order
        }

    ranking = {}
    for emotion in primary_order:
        ordered = sorted(rows, key=lambda name: rows[name][emotion], reverse=True)
        ranking[emotion] = {
            "best": ordered[0],
            "best_value": rows[ordered[0]][emotion],
            "second_best": ordered[1] if len(ordered) > 1 else None,
            "second_best_value": rows[ordered[1]][emotion] if len(ordered) > 1 else None,
            "worst": ordered[-1],
            "worst_value": rows[ordered[-1]][emotion],
            "spread": rows[ordered[0]][emotion] - rows[ordered[-1]][emotion],
        }

    return {
        "metric": metric,
        "class_order": list(primary_order),
        "rows": rows,
        "ranking_per_emotion": ranking,
        "excluded_label_spaces": other,
        "note": (
            "Cells are per-class scores on each modality's own test partition. "
            "Those partitions are different sample sets drawn from different corpora, "
            "so a column comparison indicates which modality is stronger for an "
            "emotion in this project's data, not a per-sample complementarity."
        ),
    }


def render_modality_emotion_table(table: Mapping) -> str:
    """Fixed-width text rendering of :func:`modality_emotion_table`."""
    classes = list(table["class_order"])
    width = max(12, *(len(name) for name in table["rows"])) if table["rows"] else 12
    header = f"{'modality':<{width}}" + "".join(f"{name:>10}" for name in classes)
    lines = [header, "-" * len(header)]
    for name, values in table["rows"].items():
        lines.append(
            f"{name:<{width}}" + "".join(f"{values[c]:>10.4f}" for c in classes)
        )
    lines.append("-" * len(header))
    lines.append(f"{'best':<{width}}" + "".join(
        f"{table['ranking_per_emotion'][c]['best'][:9]:>10}" for c in classes
    ))
    lines.append(f"{'worst':<{width}}" + "".join(
        f"{table['ranking_per_emotion'][c]['worst'][:9]:>10}" for c in classes
    ))
    return "\n".join(lines)


def compare_reports(reports: Sequence[dict]) -> list[dict]:
    """Flatten a set of reports into one comparable table of headline numbers."""
    return [
        {
            "name": report["name"],
            "split": report["split"],
            "samples": report["metrics"]["samples"],
            "accuracy": report["metrics"]["accuracy"],
            "macro_f1": report["metrics"]["macro_f1"],
            "weighted_f1": report["metrics"]["weighted_f1"],
            "mean_normalized_entropy": report["uncertainty"]["mean_normalized_entropy"],
            "mean_confidence": report["uncertainty"]["mean_confidence"],
            "mean_latency_ms": report["cost"]["mean_latency_ms"],
        }
        for report in reports
    ]


def agreement(left: PredictionSet, right: PredictionSet, sample_ids: Sequence[str]) -> dict:
    """How often two modalities agree, and what happens when they do not.

    Complementarity lives in the disagreement cells: if the candidate is never
    right where the anchor is wrong, acquiring it cannot help this sample set
    no matter how the fusion is weighted.
    """
    a = left.restricted_to(sample_ids)
    b = right.restricted_to(sample_ids)
    a_correct = a.predictions() == a.labels()
    b_correct = b.predictions() == b.labels()
    same = a.predictions() == b.predictions()
    total = int(len(sample_ids))
    return {
        "samples": total,
        "left": left.modality,
        "right": right.modality,
        "prediction_agreement_rate": float(same.float().mean()),
        "both_correct": int((a_correct & b_correct).sum()),
        "only_left_correct": int((a_correct & ~b_correct).sum()),
        "only_right_correct": int((~a_correct & b_correct).sum()),
        "both_wrong": int((~a_correct & ~b_correct).sum()),
        "complementarity_headroom": float(int((~a_correct & b_correct).sum()) / total)
        if total else 0.0,
        "headroom_note": (
            "complementarity_headroom is the fraction of samples the anchor gets wrong "
            "and the candidate gets right. It is the ceiling on what any fusion of "
            "these two frozen models could recover on this pool."
        ),
    }


def per_class_f1_vector(prediction_set: PredictionSet) -> dict[str, float]:
    metrics = evaluate_prediction_set(prediction_set)
    return {
        name: float(metrics["per_class"][name]["f1"])
        for name in prediction_set.class_order
    }


def macro_f1(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    return float(classification_metrics(predictions, labels, num_classes)["macro_f1"])
