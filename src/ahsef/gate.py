"""The AHSEF sufficiency gate: is the current evidence enough?

One decision, made per sample, from one number::

    U <= tau   ->  STOP
    U >  tau   ->  REQUEST an additional modality

The gate is deliberately modality-agnostic -- it takes an uncertainty series
and knows nothing about where it came from -- so the same code serves the LLM
text gate now and any later anchor.

The whole scientific weight of this component sits on where ``tau`` comes from.
:func:`select_threshold` sweeps every candidate on a named split, records the
entire curve, and **refuses to run on test**.  A threshold chosen where the
answer is already known is not a threshold, it is a result quoted backwards.

What the gate does *not* do is decide which modality to acquire.  It emits
``REQUEST``; :mod:`src.ahsef.hsig` and :mod:`src.ahsef.ugapr` will answer
"which", and until they exist the router says so rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


#: Splits on which a routing threshold may be chosen.
THRESHOLD_SELECTION_SPLITS = ("train", "validation")

#: How ``tau`` is picked from the sweep.  All are validation-only.
THRESHOLD_OBJECTIVES = (
    "target_stop_accuracy",   # widest coverage whose stopped set hits an accuracy floor
    "max_separation",         # largest accuracy gap between stopped and routed sets
    "quantile",               # a fixed coverage target, ignoring outcomes entirely
)

DEFAULT_OBJECTIVE = "target_stop_accuracy"


class ThresholdLeakageError(RuntimeError):
    """Raised when a routing threshold would be chosen using evaluation data."""


class GateDecision:
    """The two outcomes, named once."""

    STOP = "stop"
    REQUEST = "request_additional_modality"


@dataclass(frozen=True)
class SweepPoint:
    """One candidate threshold, scored on the selection split."""

    threshold: float
    stopped: int
    routed: int
    coverage: float
    stop_accuracy: float | None
    route_accuracy: float | None
    overall_accuracy: float
    separation: float | None

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold, "stopped": self.stopped, "routed": self.routed,
            "coverage": self.coverage, "stop_accuracy": self.stop_accuracy,
            "route_accuracy": self.route_accuracy, "overall_accuracy": self.overall_accuracy,
            "separation": self.separation,
        }


@dataclass
class ThresholdSelection:
    """A chosen ``tau`` together with the evidence that chose it."""

    threshold: float
    objective: str
    selected_on_split: str
    samples: int
    target: float | None
    sweep: list[SweepPoint] = field(default_factory=list)
    selected_point: SweepPoint | None = None
    satisfied: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "selected_on_split": self.selected_on_split,
            "uses_test_labels": False,
            "samples": self.samples,
            "target": self.target,
            "objective_satisfied": self.satisfied,
            "selected_point": self.selected_point.to_dict() if self.selected_point else None,
            "sweep": [point.to_dict() for point in self.sweep],
            "note": self.note,
            "rule": "STOP when U <= tau; otherwise REQUEST an additional modality.",
        }


def _validated(uncertainty, correct) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(uncertainty, dtype=float)
    outcomes = np.asarray(correct, dtype=bool)
    if values.ndim != 1:
        raise ValueError("uncertainty must be one-dimensional")
    if values.shape != outcomes.shape:
        raise ValueError(
            f"uncertainty has {values.shape[0]} entries but correctness has "
            f"{outcomes.shape[0]}"
        )
    if values.size == 0:
        raise ValueError("Cannot select a threshold from an empty split")
    if np.isnan(values).any():
        raise ValueError(
            "uncertainty contains NaN; a sample with no uncertainty signal cannot "
            "be routed and must be excluded explicitly before selection."
        )
    return values, outcomes


def sweep_thresholds(
    uncertainty: Sequence[float],
    correct: Sequence[bool],
    grid: Sequence[float] | None = None,
) -> list[SweepPoint]:
    """Score every candidate threshold, keeping the whole curve.

    The default grid is the sorted unique uncertainty values, so the sweep is
    exhaustive over decisions the data can actually distinguish rather than an
    arbitrary set of round numbers.
    """
    values, outcomes = _validated(uncertainty, correct)
    candidates = (
        np.asarray(sorted(set(float(value) for value in values)), dtype=float)
        if grid is None else np.asarray(sorted(set(float(g) for g in grid)), dtype=float)
    )
    overall = float(outcomes.mean())
    points: list[SweepPoint] = []
    for threshold in candidates:
        stop_mask = values <= threshold
        stopped, routed = int(stop_mask.sum()), int((~stop_mask).sum())
        stop_accuracy = float(outcomes[stop_mask].mean()) if stopped else None
        route_accuracy = float(outcomes[~stop_mask].mean()) if routed else None
        separation = (
            stop_accuracy - route_accuracy
            if stop_accuracy is not None and route_accuracy is not None else None
        )
        points.append(SweepPoint(
            threshold=float(threshold), stopped=stopped, routed=routed,
            coverage=stopped / len(values), stop_accuracy=stop_accuracy,
            route_accuracy=route_accuracy, overall_accuracy=overall, separation=separation,
        ))
    return points


def select_threshold(
    uncertainty: Sequence[float],
    correct: Sequence[bool],
    split: str,
    objective: str = DEFAULT_OBJECTIVE,
    target_stop_accuracy: float = 0.75,
    target_coverage: float = 0.5,
    min_coverage: float = 0.05,
    grid: Sequence[float] | None = None,
) -> ThresholdSelection:
    """Choose ``tau`` on ``split``.  Refuses any split but train/validation.

    ``target_stop_accuracy``
        Widest coverage whose stopped set still reaches the accuracy floor.
        This is the operating question the gate exists to answer: *how much of
        the data can text alone handle at an acceptable quality?*

    ``max_separation``
        Largest accuracy gap between stopped and routed samples, subject to
        ``min_coverage``. Answers a different question: where does uncertainty
        best discriminate?

    ``quantile``
        A pure coverage target that never looks at correctness -- useful as a
        control, since it shows what the threshold buys over ignoring outcomes.
    """
    if split not in THRESHOLD_SELECTION_SPLITS:
        raise ThresholdLeakageError(
            f"A routing threshold may only be selected on {THRESHOLD_SELECTION_SPLITS}; "
            f"refusing to select on {split!r}. Choosing tau on the evaluation split "
            f"would make every routing number that follows invalid."
        )
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(f"objective must be one of {THRESHOLD_OBJECTIVES}, got {objective!r}")

    values, outcomes = _validated(uncertainty, correct)
    points = sweep_thresholds(values, outcomes, grid)

    if objective == "quantile":
        threshold = float(np.quantile(values, min(max(target_coverage, 0.0), 1.0)))
        chosen = min(points, key=lambda point: abs(point.threshold - threshold))
        return ThresholdSelection(
            threshold=chosen.threshold, objective=objective, selected_on_split=split,
            samples=len(values), target=target_coverage, sweep=points, selected_point=chosen,
            satisfied=True,
            note=f"tau set at the {target_coverage:.0%} quantile of validation uncertainty; "
                 f"correctness was not consulted.",
        )

    if objective == "max_separation":
        eligible = [
            point for point in points
            if point.separation is not None and point.coverage >= min_coverage
            and point.routed > 0
        ]
        if not eligible:
            return _degenerate(points, objective, split, len(values), min_coverage)
        chosen = max(eligible, key=lambda point: point.separation)
        return ThresholdSelection(
            threshold=chosen.threshold, objective=objective, selected_on_split=split,
            samples=len(values), target=min_coverage, sweep=points, selected_point=chosen,
            satisfied=True,
            note=f"tau maximises the stopped-minus-routed accuracy gap "
                 f"({chosen.separation:+.4f}) at coverage {chosen.coverage:.1%}.",
        )

    # target_stop_accuracy: widest coverage that still clears the floor.
    eligible = [
        point for point in points
        if point.stop_accuracy is not None and point.stop_accuracy >= target_stop_accuracy
        and point.coverage >= min_coverage
    ]
    if not eligible:
        best = max(
            (point for point in points if point.stop_accuracy is not None),
            key=lambda point: point.stop_accuracy, default=None,
        )
        reachable = best.stop_accuracy if best else None
        return ThresholdSelection(
            threshold=float(min(values)) if best is None else best.threshold,
            objective=objective, selected_on_split=split, samples=len(values),
            target=target_stop_accuracy, sweep=points, selected_point=best, satisfied=False,
            note=(
                f"No threshold reaches stop-accuracy {target_stop_accuracy:.2f} at "
                f"coverage >= {min_coverage:.0%}; the best achievable on this split is "
                f"{reachable if reachable is None else round(reachable, 4)}. tau is set to "
                f"that best point and the objective is recorded as UNSATISFIED -- the gate "
                f"cannot promise an accuracy this evidence does not support."
            ),
        )
    chosen = max(eligible, key=lambda point: (point.coverage, point.threshold))
    return ThresholdSelection(
        threshold=chosen.threshold, objective=objective, selected_on_split=split,
        samples=len(values), target=target_stop_accuracy, sweep=points,
        selected_point=chosen, satisfied=True,
        note=f"tau is the widest-coverage threshold whose stopped set still reaches "
             f"accuracy {target_stop_accuracy:.2f} (achieved {chosen.stop_accuracy:.4f} "
             f"at coverage {chosen.coverage:.1%}).",
    )


def _degenerate(points, objective, split, samples, min_coverage) -> ThresholdSelection:
    fallback = points[len(points) // 2] if points else None
    return ThresholdSelection(
        threshold=fallback.threshold if fallback else 1.0, objective=objective,
        selected_on_split=split, samples=samples, target=min_coverage, sweep=points,
        selected_point=fallback, satisfied=False,
        note="No candidate threshold splits the data into a non-empty stopped and routed "
             "set at the required coverage; tau is a placeholder and the objective is "
             "recorded as UNSATISFIED.",
    )


# ============================================================
# Applying the gate
# ============================================================

@dataclass(frozen=True)
class GateOutcome:
    """The gate's verdict for one sample."""

    sample_id: str
    uncertainty: float
    threshold: float
    stop: bool
    reason: str

    @property
    def decision(self) -> str:
        return GateDecision.STOP if self.stop else GateDecision.REQUEST

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id, "uncertainty": self.uncertainty,
            "threshold": self.threshold, "stop": self.stop,
            "decision": self.decision, "reason": self.reason,
        }


def apply_gate(sample_id: str, uncertainty: float, threshold: float) -> GateOutcome:
    """Decide for one sample.  Never sees a label."""
    if uncertainty != uncertainty:  # NaN
        raise ValueError(
            f"Sample {sample_id!r} has no uncertainty value; it cannot be gated and must "
            f"be reported as ungatable rather than defaulted in either direction."
        )
    stop = bool(uncertainty <= threshold)
    return GateOutcome(
        sample_id=str(sample_id), uncertainty=float(uncertainty), threshold=float(threshold),
        stop=stop,
        reason=(
            f"uncertainty {uncertainty:.4f} <= tau {threshold:.4f}: text evidence is "
            f"sufficient"
            if stop else
            f"uncertainty {uncertainty:.4f} > tau {threshold:.4f}: text evidence is "
            f"insufficient, an additional modality is required"
        ),
    )


def gate_summary(outcomes: Sequence[GateOutcome], correct: Sequence[bool] | None = None) -> dict:
    """Counts and, when outcomes are known, the accuracy either side of ``tau``.

    ``correct`` is optional and is consumed strictly after the decisions were
    made -- it never enters :func:`apply_gate`.
    """
    if not outcomes:
        raise ValueError("No gate outcomes to summarise")
    total = len(outcomes)
    stopped = sum(1 for outcome in outcomes if outcome.stop)
    record = {
        "samples": total,
        "threshold": outcomes[0].threshold,
        "stopped": stopped,
        "requested_additional_modality": total - stopped,
        "stop_rate": stopped / total,
        "request_rate": (total - stopped) / total,
        "mean_uncertainty": float(np.mean([o.uncertainty for o in outcomes])),
    }
    if correct is not None:
        outcomes_array = np.asarray(correct, dtype=bool)
        if outcomes_array.shape[0] != total:
            raise ValueError("correct must align with outcomes")
        stop_mask = np.array([outcome.stop for outcome in outcomes], dtype=bool)
        record["outcome"] = {
            "overall_accuracy": float(outcomes_array.mean()),
            "accuracy_when_stopped": (
                float(outcomes_array[stop_mask].mean()) if stop_mask.any() else None
            ),
            "accuracy_when_routed": (
                float(outcomes_array[~stop_mask].mean()) if (~stop_mask).any() else None
            ),
            "note": "Read after routing. The gate never sees a label.",
        }
    return record
