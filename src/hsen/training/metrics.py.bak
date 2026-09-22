"""Every metric this phase reports, computed in the declared class order.

Section 11 is explicit that aggregate accuracy must not be allowed to hide a
weak class, so every categorical result carries per-class precision, recall,
F1 and support alongside the aggregates, and the confusion matrix is indexed by
the label space's declared order rather than by whatever order the labels
happened to appear in.

Three metric families, because there are three tasks:

single-label
    Accuracy, weighted F1 (IEMOCAP's primary), macro F1, per-class F1,
    confusion matrix.

multi-label
    Micro-F1 and macro-F1, plus the exact-set-match accuracy LDDU and CARAT
    report as "accuracy" -- which is a much harsher number than per-sample
    top-1 and must never be printed under the same heading.

regression
    CCC, MAE, RMSE and Pearson r.  CCC is first because it is the one that
    catches a model predicting the mean: such a model has respectable MAE and a
    CCC near zero.

CMU-MOSEI's sentiment protocol additionally reports Acc7, Acc2 and a binary F1,
and the two Acc2 conventions in the literature differ on whether neutral clips
are excluded.  Both are computed and both are named, because quoting one as the
other is the most common way MOSEI numbers stop being comparable.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.hsen.labels.base import MISSING_CLASS_ID


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ======================================================================
# Categorical
# ======================================================================

def classification_metrics(
    logits, targets, class_names: tuple[str, ...], prefix: str = "",
) -> dict:
    """Accuracy, weighted/macro F1, per-class detail and the confusion matrix."""
    logits, targets = _to_numpy(logits), _to_numpy(targets).astype(np.int64)
    valid = targets != MISSING_CLASS_ID
    labels = list(range(len(class_names)))

    if not valid.any():
        empty = {f"{prefix}{name}": float("nan")
                 for name in ("accuracy", "weighted_f1", "macro_f1")}
        return empty | {f"{prefix}per_class": {}, f"{prefix}confusion_matrix": [],
                        f"{prefix}support": 0}

    predictions = logits[valid].argmax(axis=-1)
    truth = targets[valid]

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    return {
        f"{prefix}accuracy": float(accuracy_score(truth, predictions)),
        f"{prefix}weighted_f1": float(
            f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        f"{prefix}macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        ),
        f"{prefix}per_class": per_class,
        f"{prefix}confusion_matrix": confusion_matrix(
            truth, predictions, labels=labels
        ).tolist(),
        f"{prefix}support": int(valid.sum()),
    }


def multilabel_metrics(
    logits, targets, class_names: tuple[str, ...], threshold: float = 0.5, prefix: str = "",
) -> dict:
    """Micro/macro F1, per-label F1, and exact-set-match accuracy."""
    logits, targets = _to_numpy(logits), _to_numpy(targets)
    valid = np.isfinite(targets).all(axis=-1)
    if not valid.any():
        return {f"{prefix}{name}": float("nan")
                for name in ("accuracy", "micro_f1", "macro_f1")} | {f"{prefix}support": 0}

    probabilities = 1.0 / (1.0 + np.exp(-logits[valid]))
    predictions = (probabilities >= threshold).astype(np.int64)
    truth = targets[valid].astype(np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, labels=list(range(len(class_names))), zero_division=0,
    )
    return {
        # The LDDU/CARAT convention: a sample counts as correct only when the
        # whole predicted label set matches. Reporting per-label accuracy under
        # this name would inflate it by tens of points.
        f"{prefix}accuracy": float((predictions == truth).all(axis=1).mean()),
        f"{prefix}micro_f1": float(
            f1_score(truth, predictions, average="micro", zero_division=0)
        ),
        f"{prefix}macro_f1": float(
            f1_score(truth, predictions, average="macro", zero_division=0)
        ),
        f"{prefix}per_class": {
            name: {
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]), "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        f"{prefix}support": int(valid.sum()),
    }


# ======================================================================
# Regression
# ======================================================================

def concordance_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if prediction.size < 2:
        return float("nan")
    prediction_mean, target_mean = prediction.mean(), target.mean()
    prediction_var, target_var = prediction.var(), target.var()
    covariance = ((prediction - prediction_mean) * (target - target_mean)).mean()
    denominator = prediction_var + target_var + (prediction_mean - target_mean) ** 2
    return float(2 * covariance / denominator) if denominator > 1e-12 else float("nan")


def regression_metrics(prediction, target, prefix: str = "") -> dict:
    """CCC, MAE, RMSE and Pearson r over the finitely-supervised rows."""
    prediction, target = _to_numpy(prediction).ravel(), _to_numpy(target).ravel()
    valid = np.isfinite(prediction) & np.isfinite(target)
    if valid.sum() < 2:
        return {f"{prefix}{name}": float("nan")
                for name in ("ccc", "mae", "rmse", "pearson")} | {f"{prefix}support": int(valid.sum())}

    prediction, target = prediction[valid], target[valid]
    correlation = (
        float(np.corrcoef(prediction, target)[0, 1])
        if prediction.std() > 1e-12 and target.std() > 1e-12 else float("nan")
    )
    return {
        f"{prefix}ccc": concordance_correlation(prediction, target),
        f"{prefix}mae": float(np.abs(prediction - target).mean()),
        f"{prefix}rmse": float(np.sqrt(((prediction - target) ** 2).mean())),
        f"{prefix}pearson": correlation,
        f"{prefix}support": int(valid.size),
    }


# ======================================================================
# CMU-MOSEI sentiment protocol
# ======================================================================

def sentiment_metrics(logits, class_ids, valence_prediction=None, valence_target=None) -> dict:
    """Acc7, both Acc2 conventions, and the binary F1 that accompanies them.

    The class ids are the seven ordinal buckets, so a continuous score is
    recovered as ``bucket - 3`` for the binary split.  ``acc2_non_neutral``
    excludes neutral clips (the older Zadeh convention); ``acc2_non_negative``
    keeps them on the positive side (the convention most recent papers use).
    Both appear because the difference between them is worth several points.
    """
    logits, class_ids = _to_numpy(logits), _to_numpy(class_ids).astype(np.int64)
    valid = class_ids != MISSING_CLASS_ID
    if not valid.any():
        return {}

    predicted_bucket = logits[valid].argmax(axis=-1)
    true_bucket = class_ids[valid]
    predicted_score = predicted_bucket.astype(np.float64) - 3.0
    true_score = true_bucket.astype(np.float64) - 3.0

    metrics = {"acc7": float(accuracy_score(true_bucket, predicted_bucket))}

    non_neutral = true_score != 0
    if non_neutral.any():
        truth = (true_score[non_neutral] > 0).astype(np.int64)
        predictions = (predicted_score[non_neutral] > 0).astype(np.int64)
        metrics["acc2_non_neutral"] = float(accuracy_score(truth, predictions))
        metrics["f1_non_neutral"] = float(f1_score(truth, predictions, zero_division=0))

    truth = (true_score >= 0).astype(np.int64)
    predictions = (predicted_score >= 0).astype(np.int64)
    metrics["acc2_non_negative"] = float(accuracy_score(truth, predictions))
    metrics["f1_non_negative"] = float(f1_score(truth, predictions, zero_division=0))

    # The regression head predicts the same quantity on a [-1, 1] scale, so its
    # MAE and correlation are directly comparable with the published MOSEI
    # numbers once rescaled back to [-3, 3].
    if valence_prediction is not None and valence_target is not None:
        scaled = regression_metrics(
            _to_numpy(valence_prediction) * 3.0, _to_numpy(valence_target) * 3.0,
            prefix="score_",
        )
        metrics.update(scaled)
    return metrics


# ======================================================================
# Driver
# ======================================================================

def evaluate_predictions(
    outputs: dict, targets: dict, class_names: tuple[str, ...], task: str = "single_label",
    dataset: str | None = None,
) -> dict:
    """Every metric appropriate to one task, from stacked epoch predictions."""
    if task == "multi_label":
        metrics = multilabel_metrics(outputs["logits"], targets["multilabel"], class_names)
    else:
        metrics = classification_metrics(outputs["logits"], targets["class_id"], class_names)

    for name in ("valence", "arousal"):
        if name in outputs and name in targets:
            metrics.update(regression_metrics(outputs[name], targets[name], prefix=f"{name}_"))

    if dataset == "CMU-MOSEI" and task == "single_label":
        metrics["sentiment"] = sentiment_metrics(
            outputs["logits"], targets["class_id"],
            outputs.get("valence"), targets.get("valence"),
        )
    return metrics


def primary_metric_name(dataset: str, task: str) -> str:
    """Which validation number selects the best checkpoint.

    Section 16 fixes IEMOCAP's as weighted F1.  CMU-MOSEI's multi-label protocol
    is ranked by micro-F1, matching LDDU's headline; its sentiment protocol is
    ranked by weighted F1 for consistency with the categorical head being
    trained, with Acc7 and both Acc2 conventions reported alongside.
    """
    if task == "multi_label":
        return "micro_f1"
    return "weighted_f1"
