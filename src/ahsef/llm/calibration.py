"""Calibrating a self-report against outcomes.

Stage 1 calibrated posteriors: it had logits, so a temperature was the natural
correction.  The LLM gives a *number the model wrote about itself*, and whether
that number tracks correctness is an open empirical question -- exactly the
question this module answers, and the only thing that can turn
``llm_confidence`` into something a threshold may lean on.

Two corrections are offered, and neither is applied automatically:

**Binned recalibration** maps a self-reported confidence onto the empirical
accuracy observed at that confidence on validation.  It needs no distribution
and works when the model returns nothing but a scalar.  It is monotone-agnostic
-- if the model's confidence is anti-correlated with being right, the map will
show that rather than hiding it.

**Temperature scaling of the self-reported scores** reuses Stage 1's fitter on
``log(normalised class_scores)``.  Those are not logits, and the artefacts say
so; the transform is legitimate arithmetic on a score vector, and its value is
decided by measurement like everything else.

Both fit on validation and refuse any other split, through the same guard
Stage 1 uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from src.ahsef.calibration import (
    CALIBRATION_FIT_SPLIT,
    CalibrationLeakageError,
    ReliabilityBin,
    TemperatureScaler,
    fit_temperature,
)


DEFAULT_BINS = 10


def _validated(confidence: Sequence[float], correct: Sequence[bool]):
    values = np.asarray(confidence, dtype=float)
    outcomes = np.asarray(correct, dtype=bool)
    if values.shape != outcomes.shape:
        raise ValueError(
            f"confidence has {values.shape} entries but correctness has {outcomes.shape}"
        )
    if values.size == 0:
        raise ValueError("Cannot analyse calibration over an empty set")
    if np.isnan(values).any():
        raise ValueError(
            "confidence contains NaN; samples without a self-reported confidence must be "
            "excluded explicitly rather than imputed"
        )
    if ((values < 0) | (values > 1)).any():
        raise ValueError("confidence values must lie in [0, 1]")
    return values, outcomes


def confidence_reliability_bins(
    confidence: Sequence[float], correct: Sequence[bool], num_bins: int = DEFAULT_BINS
) -> list[ReliabilityBin]:
    """Reliability bins for a scalar self-report, in Stage 1's bin structure."""
    values, outcomes = _validated(confidence, correct)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ReliabilityBin] = []
    for index in range(num_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        inside = (
            (values >= lower) & (values <= upper) if index == 0
            else (values > lower) & (values <= upper)
        )
        count = int(inside.sum())
        bins.append(ReliabilityBin(
            lower=lower, upper=upper, count=count,
            mean_confidence=float(values[inside].mean()) if count else 0.0,
            accuracy=float(outcomes[inside].mean()) if count else 0.0,
        ))
    return bins


def confidence_calibration_report(
    confidence: Sequence[float],
    correct: Sequence[bool],
    num_bins: int = DEFAULT_BINS,
    label: str = "llm_confidence",
) -> dict:
    """ECE and friends for a self-reported confidence.

    ``discrimination`` is reported alongside calibration because they are
    different virtues and an LLM can fail either independently: a model whose
    confidence is a constant 0.9 can be perfectly calibrated on a set it gets
    right 90% of the time while carrying no per-sample information at all.
    """
    values, outcomes = _validated(confidence, correct)
    bins = confidence_reliability_bins(values, outcomes, num_bins)
    total = sum(item.count for item in bins)
    ece = sum(item.count / total * abs(item.gap) for item in bins)
    right, wrong = values[outcomes], values[~outcomes]
    spread = float(values.std(ddof=0))
    return {
        "label": label,
        "samples": int(values.size),
        "num_bins": num_bins,
        "accuracy": float(outcomes.mean()),
        "mean_confidence": float(values.mean()),
        "overconfidence": float(values.mean() - outcomes.mean()),
        "ece": float(ece),
        "mce": float(max((abs(item.gap) for item in bins if item.count), default=0.0)),
        "brier": float(np.mean((values - outcomes.astype(float)) ** 2)),
        "confidence_vs_correctness": {
            "mean_confidence_when_correct": float(right.mean()) if right.size else None,
            "mean_confidence_when_wrong": float(wrong.mean()) if wrong.size else None,
            "separation": (
                float(right.mean() - wrong.mean()) if right.size and wrong.size else None
            ),
        },
        "discrimination": {
            "confidence_std": spread,
            "distinct_values": int(np.unique(values).size),
            "auroc": _auroc(values, outcomes),
            "note": (
                "A self-report can be well calibrated in aggregate and still useless per "
                "sample. AUROC near 0.5, a near-zero separation, or very few distinct "
                "values all mean the number cannot support a per-sample gate."
            ),
        },
        "reliability_bins": [item.to_dict() for item in bins],
        "is_a_posterior": False,
    }


def _auroc(scores: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Rank-based AUROC of confidence against correctness; ties handled properly."""
    positives, negatives = int(outcomes.sum()), int((~outcomes).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1, dtype=float)
    # Average ranks within tied score groups so a constant confidence gives 0.5.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[outcomes].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


# ============================================================
# Binned recalibration
# ============================================================

@dataclass
class BinnedConfidenceCalibrator:
    """Map self-reported confidence to observed accuracy, fitted on validation.

    Bins with too few samples inherit the global accuracy rather than a noisy
    local estimate, and the record says which bins were backed off.
    """

    edges: list[float] = field(default_factory=list)
    mapped: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    fitted_on_split: str | None = None
    fitted_on_samples: int = 0
    min_bin_count: int = 20
    fallback: float = 0.0
    backed_off_bins: list[int] = field(default_factory=list)

    def apply(self, confidence: Sequence[float]) -> np.ndarray:
        values = np.asarray(confidence, dtype=float)
        if not self.edges:
            raise RuntimeError("Calibrator has not been fitted")
        indices = np.clip(
            np.digitize(values, np.asarray(self.edges[1:-1]), right=True), 0,
            len(self.mapped) - 1,
        )
        return np.asarray(self.mapped, dtype=float)[indices]

    def to_dict(self) -> dict:
        return {
            "method": "binned_empirical_accuracy",
            "edges": self.edges,
            "mapped_accuracy": self.mapped,
            "bin_counts": self.counts,
            "min_bin_count": self.min_bin_count,
            "backed_off_bins": self.backed_off_bins,
            "global_fallback": self.fallback,
            "fitted_on_split": self.fitted_on_split,
            "fitted_on_samples": self.fitted_on_samples,
            "uses_test_labels": False,
            "note": (
                "Maps a self-reported confidence onto the accuracy observed at that "
                "confidence on validation. Bins below min_bin_count fall back to the "
                "global validation accuracy instead of a noisy local estimate."
            ),
        }


def fit_confidence_calibrator(
    confidence: Sequence[float],
    correct: Sequence[bool],
    split: str,
    num_bins: int = DEFAULT_BINS,
    min_bin_count: int = 20,
) -> BinnedConfidenceCalibrator:
    """Fit the binned map.  Validation only, enforced."""
    if split != CALIBRATION_FIT_SPLIT:
        raise CalibrationLeakageError(
            f"A confidence calibrator may only be fitted on the "
            f"{CALIBRATION_FIT_SPLIT!r} split; refusing to fit on {split!r}."
        )
    values, outcomes = _validated(confidence, correct)
    edges = list(np.linspace(0.0, 1.0, num_bins + 1))
    global_accuracy = float(outcomes.mean())
    mapped: list[float] = []
    counts: list[int] = []
    backed_off: list[int] = []
    for index in range(num_bins):
        lower, upper = edges[index], edges[index + 1]
        inside = (
            (values >= lower) & (values <= upper) if index == 0
            else (values > lower) & (values <= upper)
        )
        count = int(inside.sum())
        counts.append(count)
        if count >= min_bin_count:
            mapped.append(float(outcomes[inside].mean()))
        else:
            mapped.append(global_accuracy)
            backed_off.append(index)
    return BinnedConfidenceCalibrator(
        edges=edges, mapped=mapped, counts=counts, fitted_on_split=split,
        fitted_on_samples=int(values.size), min_bin_count=min_bin_count,
        fallback=global_accuracy, backed_off_bins=backed_off,
    )


def fit_score_temperature(
    log_scores: torch.Tensor, labels: torch.Tensor, split: str
) -> TemperatureScaler:
    """Stage 1's temperature fitter applied to log self-reported scores.

    Reuses the validated implementation and its validation-only guard.  The
    inputs are ``log(normalised class_scores)`` -- a transform of a self-report,
    not a model's pre-softmax output -- and the returned record is labelled that
    way by the caller.
    """
    return fit_temperature(log_scores, labels, split=split)


def calibration_provenance() -> dict:
    return {
        "confidence_calibration": "binned empirical accuracy, fitted on validation",
        "score_calibration": (
            "Stage 1 temperature scaling applied to log(normalised class_scores); the "
            "inputs are self-reported scores, not logits, and are never reported as such"
        ),
        "fit_split": CALIBRATION_FIT_SPLIT,
        "uses_test_labels": False,
        "applied_automatically": False,
        "note": (
            "Calibration is measured and reported. Whether to apply it is an explicit "
            "flag, so the uncalibrated numbers stay directly comparable to the frozen "
            "baselines."
        ),
    }
