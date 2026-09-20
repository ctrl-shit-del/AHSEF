"""Calibration measurement and validation-only temperature scaling.

AHSEF routes on confidence and uncertainty, so how well those numbers track
correctness is a *precondition* of the routing argument, not a footnote.  This
module measures it (ECE, MCE, reliability bins, Brier, NLL) and, when a model
turns out to be miscalibrated, fits the lightest possible correction -- a
single temperature on the logits.

The safety property this module enforces is stated once and checked in code:

    **A temperature is fitted on the validation split and on nothing else.**

:func:`fit_temperature` refuses any split other than ``validation``, and the
record it returns names the split it saw, so a downstream artefact can be
audited without re-running anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.ahsef.uncertainty import confidence as _confidence
from src.ahsef.uncertainty import probabilities_from_logits


DEFAULT_BINS = 15

#: The only split a calibration parameter may be fitted on.
CALIBRATION_FIT_SPLIT = "validation"


class CalibrationLeakageError(RuntimeError):
    """Raised when a calibration parameter is about to be fitted on non-validation data."""


# ============================================================
# Measurement
# ============================================================

@dataclass(frozen=True)
class ReliabilityBin:
    """One equal-width confidence bin of a reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        """Signed ``confidence - accuracy``; positive means over-confident."""
        return self.mean_confidence - self.accuracy

    def to_dict(self) -> dict:
        return {
            "lower": self.lower, "upper": self.upper, "count": self.count,
            "mean_confidence": self.mean_confidence, "accuracy": self.accuracy,
            "gap": self.gap,
        }


def _confidence_and_correctness(
    probabilities: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.as_tensor(probabilities).detach().to(torch.float64)
    labels = torch.as_tensor(labels).detach().long().reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a [N, K] matrix")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probabilities has {probabilities.shape[0]} rows but labels has {labels.shape[0]}"
        )
    predicted = probabilities.argmax(dim=-1)
    return _confidence(probabilities), (predicted == labels).to(torch.float64)


def reliability_bins(
    probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = DEFAULT_BINS
) -> list[ReliabilityBin]:
    """Equal-width confidence bins over ``[0, 1]``; empty bins are reported too.

    Empty bins are kept rather than dropped: a model whose confidence never
    leaves ``[0.2, 0.4]`` is telling you something, and silently omitting the
    bins it never enters hides it.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1")
    conf, correct = _confidence_and_correctness(probabilities, labels)
    edges = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
    bins: list[ReliabilityBin] = []
    for index in range(num_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        # Left-open, right-closed, with the first bin closed on the left so
        # that every confidence value lands in exactly one bin.
        inside = (conf > lower) & (conf <= upper) if index else (conf >= lower) & (conf <= upper)
        count = int(inside.sum())
        bins.append(ReliabilityBin(
            lower=lower, upper=upper, count=count,
            mean_confidence=float(conf[inside].mean()) if count else 0.0,
            accuracy=float(correct[inside].mean()) if count else 0.0,
        ))
    return bins


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = DEFAULT_BINS
) -> float:
    """``ECE = sum_b (n_b / N) * |acc_b - conf_b|`` over equal-width bins."""
    bins = reliability_bins(probabilities, labels, num_bins)
    total = sum(item.count for item in bins)
    if total == 0:
        raise ValueError("Cannot compute ECE over an empty prediction set")
    return sum(item.count / total * abs(item.gap) for item in bins)


def maximum_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = DEFAULT_BINS
) -> float:
    """The worst absolute bin gap among non-empty bins."""
    bins = [item for item in reliability_bins(probabilities, labels, num_bins) if item.count]
    if not bins:
        raise ValueError("Cannot compute MCE over an empty prediction set")
    return max(abs(item.gap) for item in bins)


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Multi-class Brier score: mean squared error against the one-hot target."""
    probabilities = torch.as_tensor(probabilities).detach().to(torch.float64)
    labels = torch.as_tensor(labels).detach().long().reshape(-1)
    onehot = torch.zeros_like(probabilities)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    return float(((probabilities - onehot) ** 2).sum(dim=1).mean())


def negative_log_likelihood(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean ``-log p_true`` in nats."""
    probabilities = torch.as_tensor(probabilities).detach().to(torch.float64)
    labels = torch.as_tensor(labels).detach().long().reshape(-1)
    picked = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    return float(-torch.log(picked.clamp(min=1e-12)).mean())


def calibration_report(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = DEFAULT_BINS,
    label: str = "raw",
) -> dict:
    """Every calibration number AHSEF records for one prediction set."""
    conf, correct = _confidence_and_correctness(probabilities, labels)
    bins = reliability_bins(probabilities, labels, num_bins)
    total = sum(item.count for item in bins)
    return {
        "label": label,
        "samples": int(conf.shape[0]),
        "num_bins": num_bins,
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfidence": float(conf.mean() - correct.mean()),
        "ece": sum(item.count / total * abs(item.gap) for item in bins),
        "mce": max((abs(item.gap) for item in bins if item.count), default=0.0),
        "brier": brier_score(probabilities, labels),
        "nll": negative_log_likelihood(probabilities, labels),
        "confidence_vs_correctness": {
            "mean_confidence_when_correct":
                float(conf[correct > 0].mean()) if float(correct.sum()) else None,
            "mean_confidence_when_wrong":
                float(conf[correct == 0].mean()) if float((1 - correct).sum()) else None,
        },
        "reliability_bins": [item.to_dict() for item in bins],
    }


# ============================================================
# Temperature scaling
# ============================================================

@dataclass
class TemperatureScaler:
    """A single positive scalar dividing the logits before the softmax.

    ``temperature == 1`` is the identity, so an uncalibrated model and a
    calibrated one share one code path and one artefact schema.
    """

    temperature: float = 1.0
    fitted_on_split: str | None = None
    fitted_on_samples: int | None = None
    optimisation: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled probabilities for ``logits``."""
        return probabilities_from_logits(logits, temperature=self.temperature)

    def to_dict(self) -> dict:
        return {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "fitted_on_split": self.fitted_on_split,
            "fitted_on_samples": self.fitted_on_samples,
            "uses_test_labels": False,
            "identity": self.temperature == 1.0,
            "optimisation": dict(self.optimisation),
        }

    @classmethod
    def from_dict(cls, record: dict) -> "TemperatureScaler":
        return cls(
            temperature=float(record["temperature"]),
            fitted_on_split=record.get("fitted_on_split"),
            fitted_on_samples=record.get("fitted_on_samples"),
            optimisation=dict(record.get("optimisation") or {}),
        )

    @classmethod
    def identity(cls) -> "TemperatureScaler":
        return cls(
            temperature=1.0, fitted_on_split=None, fitted_on_samples=None,
            optimisation={"method": "none", "note": "raw logits, no calibration applied"},
        )


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    split: str,
    max_iter: int = 200,
    learning_rate: float = 0.05,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> TemperatureScaler:
    """Fit one temperature by minimising validation NLL.

    Raises :class:`CalibrationLeakageError` unless ``split`` is
    ``'validation'``.  That guard is the whole reason this function takes a
    split name it does not otherwise need.
    """
    if split != CALIBRATION_FIT_SPLIT:
        raise CalibrationLeakageError(
            f"Temperature scaling may only be fitted on the {CALIBRATION_FIT_SPLIT!r} "
            f"split; refusing to fit on {split!r}. Fitting on test data would make "
            f"every downstream uncertainty threshold invalid."
        )
    logits = torch.as_tensor(logits).detach().to(torch.float64)
    labels = torch.as_tensor(labels).detach().long().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits must be [N, K] and align with [N] labels")
    if logits.shape[0] == 0:
        raise ValueError("Cannot fit a temperature on an empty split")

    low, high = bounds
    # Optimising log-temperature keeps the parameter positive without a
    # projection step, and a plain Adam loop is reproducible on CPU where an
    # LBFGS line search is not.
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([log_t], lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[float] = []
    for _ in range(max_iter):
        optimiser.zero_grad()
        loss = criterion(logits / log_t.exp().clamp(low, high), labels)
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach()))

    temperature = float(log_t.detach().exp().clamp(low, high))
    with torch.no_grad():
        before = float(criterion(logits, labels))
        after = float(criterion(logits / temperature, labels))
    return TemperatureScaler(
        temperature=temperature,
        fitted_on_split=split,
        fitted_on_samples=int(labels.shape[0]),
        optimisation={
            "method": "adam_on_log_temperature",
            "objective": "cross_entropy_nll",
            "iterations": max_iter,
            "learning_rate": learning_rate,
            "bounds": list(bounds),
            "nll_before": before,
            "nll_after": after,
            "nll_improved": after < before,
            "loss_history_tail": history[-5:],
        },
    )


def compare_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    scaler: TemperatureScaler,
    num_bins: int = DEFAULT_BINS,
) -> dict:
    """Raw and temperature-scaled calibration reports side by side."""
    raw = probabilities_from_logits(logits, temperature=1.0)
    calibrated = scaler.apply(logits)
    return {
        "raw": calibration_report(raw, labels, num_bins, label="raw"),
        "calibrated": calibration_report(calibrated, labels, num_bins, label="temperature_scaled"),
        "scaler": scaler.to_dict(),
    }


def recommend_calibration(reports: dict, ece_threshold: float = 0.05) -> dict:
    """State plainly whether the fitted temperature should be used.

    The rule is deliberately conservative and explicit rather than automatic:
    a temperature is recommended only when it actually lowers ECE on the split
    it was fitted on.
    """
    raw_ece = reports["raw"]["ece"]
    calibrated_ece = reports["calibrated"]["ece"]
    return {
        "ece_threshold": ece_threshold,
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "raw_is_well_calibrated": raw_ece <= ece_threshold,
        "temperature_reduces_ece": calibrated_ece < raw_ece,
        "recommended": "temperature_scaled" if calibrated_ece < raw_ece else "raw",
    }
