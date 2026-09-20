"""Predictive uncertainty and confidence for categorical predictions.

Two quantities are deliberately kept distinct throughout AHSEF because they
answer different questions and are calibrated differently:

``confidence``
    ``max_c p_c`` -- how strongly the model backs its *top* class.

``uncertainty``
    normalised predictive entropy ``H(p) / log(K)`` -- how spread the whole
    distribution is, in ``[0, 1]`` regardless of the number of classes.

Neither is a calibrated probability of correctness until it has been checked
against outcomes; see :mod:`src.ahsef.calibration`.  Nothing in this module
claims otherwise, and no function here looks at a label.

All functions accept and return :class:`torch.Tensor` and operate on the last
axis, so they work on a single distribution ``[K]`` or a batch ``[N, K]``.
"""

from __future__ import annotations

import math

import torch


#: Distributions are clamped away from exactly zero before the log so that
#: ``0 * log 0`` contributes 0 rather than NaN.  The value is far below any
#: probability a 7-class softmax produces in float32.
EPSILON = 1e-12


def _as_float(values: torch.Tensor | list | tuple) -> torch.Tensor:
    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    return tensor.detach().to(torch.float64) if tensor.dtype != torch.float64 else tensor.detach()


def probabilities_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over the last axis, optionally temperature-scaled.

    ``temperature > 1`` softens the distribution, ``< 1`` sharpens it.  A
    temperature is a *calibration* parameter and must have been fitted on
    validation data; this function does not fit anything.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return torch.softmax(_as_float(logits) / float(temperature), dim=-1)


def normalize_probabilities(scores: torch.Tensor) -> torch.Tensor:
    """Rescale non-negative scores so the last axis sums to one.

    Used by probability-level fusion, where a weighted sum of two normalised
    distributions is already normalised only when the weights sum to one.  A
    row that sums to zero is an error, not something to paper over.
    """
    scores = _as_float(scores)
    if torch.any(scores < 0):
        raise ValueError("Cannot normalise a score vector containing negative entries")
    total = scores.sum(dim=-1, keepdim=True)
    if torch.any(total <= 0):
        raise ValueError("Cannot normalise a score vector that sums to zero")
    return scores / total


def _validated(probabilities: torch.Tensor, tolerance: float = 1e-5) -> torch.Tensor:
    probabilities = _as_float(probabilities)
    if probabilities.shape[-1] < 2:
        raise ValueError("A predictive distribution needs at least two classes")
    if torch.any(probabilities < -tolerance):
        raise ValueError("Probabilities must be non-negative")
    total = probabilities.sum(dim=-1)
    if torch.any((total - 1.0).abs() > tolerance):
        raise ValueError(
            f"Probabilities must sum to 1 along the last axis "
            f"(worst deviation {float((total - 1.0).abs().max()):.6g})"
        )
    return probabilities.clamp(min=0.0)


def predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """``U(p) = -sum_c p_c log p_c`` in nats, over the last axis."""
    probabilities = _validated(probabilities)
    entropy = -(probabilities * torch.log(probabilities.clamp(min=EPSILON))).sum(dim=-1)
    # A one-hot distribution yields -0.0 from the sum; clamping keeps the sign
    # out of the JSON artefacts without changing any value.
    return entropy.clamp(min=0.0) + 0.0  # '+ 0.0' turns -0.0 into 0.0


def max_entropy(num_classes: int) -> float:
    """``log K`` -- the entropy of the uniform distribution over ``K`` classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    return math.log(num_classes)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """``U_norm(p) = U(p) / log K``, bounded in ``[0, 1]``.

    Normalising by ``log K`` is what makes an uncertainty from the 3-class
    physiology model numerically comparable with one from a 7-class emotion
    model.  It does **not** make the two models' outputs semantically
    comparable -- see :mod:`src.ahsef.fusion`.
    """
    probabilities = _validated(probabilities)
    entropy = predictive_entropy(probabilities)
    return (entropy / max_entropy(probabilities.shape[-1])).clamp(0.0, 1.0)


def confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """``max_c p_c`` -- the probability mass on the predicted class."""
    return _validated(probabilities).max(dim=-1).values


def predicted_class(probabilities: torch.Tensor) -> torch.Tensor:
    """``argmax_c p_c``.  Ties resolve to the lowest class id, as in torch."""
    return _validated(probabilities).argmax(dim=-1)


def probability_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """``p_(1) - p_(2)`` -- the gap between the top two classes."""
    probabilities = _validated(probabilities)
    top2 = probabilities.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def uncertainty_record(probabilities: torch.Tensor) -> dict[str, float]:
    """Every scalar AHSEF stores for one predictive distribution."""
    probabilities = _validated(probabilities)
    if probabilities.ndim != 1:
        raise ValueError("uncertainty_record expects a single distribution [K]")
    return {
        "num_classes": int(probabilities.shape[-1]),
        "confidence": float(confidence(probabilities)),
        "predictive_entropy": float(predictive_entropy(probabilities)),
        "normalized_entropy": float(normalized_entropy(probabilities)),
        "margin": float(probability_margin(probabilities)),
        "predicted_class": int(predicted_class(probabilities)),
    }


def uncertainty_columns(probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    """Batched form of :func:`uncertainty_record`, for a ``[N, K]`` matrix."""
    probabilities = _validated(probabilities)
    if probabilities.ndim != 2:
        raise ValueError("uncertainty_columns expects a batch of distributions [N, K]")
    return {
        "confidence": confidence(probabilities),
        "predictive_entropy": predictive_entropy(probabilities),
        "normalized_entropy": normalized_entropy(probabilities),
        "margin": probability_margin(probabilities),
        "predicted_class": predicted_class(probabilities),
    }


#: Declared once so that every artefact records which definition produced its
#: numbers rather than leaving it implicit.
UNCERTAINTY_DEFINITION = {
    "confidence": "max_c p_c",
    "uncertainty": "normalized predictive entropy: -sum_c p_c ln(p_c) / ln(K)",
    "entropy_base": "natural logarithm (nats)",
    "margin": "p_(1) - p_(2), the top-1 minus top-2 probability",
    "range": "normalized_entropy in [0, 1]; confidence in [1/K, 1]",
    "calibrated": False,
    "note": (
        "Entropy is a property of the predictive distribution, not a calibrated "
        "probability of correctness. Calibration is measured separately in "
        "src.ahsef.calibration and reported alongside, never assumed."
    ),
}
