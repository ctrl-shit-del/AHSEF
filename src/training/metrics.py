import math

import torch


def mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Mean Squared Error.
    """

    prediction = prediction.detach().float()
    target = target.detach().float()

    value = torch.mean(
        (prediction - target) ** 2
    )

    return float(value.item())


def mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Mean Absolute Error.
    """

    prediction = prediction.detach().float()
    target = target.detach().float()

    value = torch.mean(
        torch.abs(prediction - target)
    )

    return float(value.item())


def rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Root Mean Squared Error.
    """

    return math.sqrt(
        mse(
            prediction,
            target,
        )
    )


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Calculate all basic regression metrics.
    """

    return {
        "mse": mse(
            prediction,
            target,
        ),
        "mae": mae(
            prediction,
            target,
        ),
        "rmse": rmse(
            prediction,
            target,
        ),
    }


def classification_metrics(prediction: torch.Tensor, target: torch.Tensor, num_classes: int = 7) -> dict:
    """Finite, zero-division-safe categorical metrics and confusion matrix."""
    prediction, target = prediction.detach().long().cpu(), target.detach().long().cpu()
    matrix = torch.bincount(target * num_classes + prediction, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = matrix.diag().float()
    support = matrix.sum(dim=1).float()
    predicted = matrix.sum(dim=0).float()
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(tp))
    total = support.sum()
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * support).sum() / total) if total else 0.0,
        "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(), "support": support.long().tolist(), "confusion_matrix": matrix.tolist(),
    }


def class_weights(labels: torch.Tensor, num_classes: int = 7) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    return torch.where(counts > 0, counts.sum() / (num_classes * counts), torch.zeros_like(counts))
