"""HSIG -- Hierarchical Sensor/Modality Information Gain: the interface only.

HSIG's job is to answer, without paying for the candidate: *how much would
acquiring modality ``m`` improve the decision for this sample?*

**No estimator is implemented here, and none is faked.**  Stage 1 measured that
the originally specified target -- uncertainty reduction
``dU(m|A) = U(A) - U(A u {m})`` -- is not usable: its sign flips with the
choice of anchor and with the fusion rule, and its per-sample correlation with
actual decision improvement peaked at 0.143 across every fusable pair.  An HSIG
regressing ``dU`` would learn to refuse acquisitions that demonstrably help.

So this module fixes the *contract* that a real estimator must satisfy, in the
shape Stage 1's evidence recommends -- expected **decision improvement**, not
entropy reduction -- and ships exactly one implementation:
:class:`UnavailableHSIG`, which declines to answer and says why.  A router
holding it reports "no gain estimate available" rather than inventing a number.

The target a stage-3 estimator should regress, per
``docs/ahsef_stage1.md`` section 8.2::

    gain(m | A) = P(correct | A u {m}) - P(correct | A)

estimated on validation only, from the paired records the pairwise-fusion
command already writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


#: The quantity a real HSIG must estimate.  Recorded in artefacts so a future
#: implementation cannot quietly revert to the rejected target.
HSIG_TARGET = "expected_decision_improvement"

HSIG_TARGET_DEFINITION = {
    "target": HSIG_TARGET,
    "formula": "gain(m | A) = P(correct | A u {m}) - P(correct | A)",
    "estimated_on": "validation only",
    "rejected_target": "delta_uncertainty = U(A) - U(A u {m})",
    "why_rejected": (
        "Stage 1 measured it directly: the sign depends on which modality is the anchor "
        "and on the fusion rule, in five of six configurations it was negative while "
        "accuracy rose 8-10 points, and its largest per-sample correlation with actual "
        "decision improvement was 0.143. See docs/ahsef_stage1.md section 8.2 and "
        "experiments/ahsef/stage1/reports/gain_diagnostics_validation.json."
    ),
}


class HSIGUnavailable(RuntimeError):
    """Raised when a gain estimate is requested but no estimator is implemented."""


@dataclass(frozen=True)
class RoutingState:
    """What the router knows about one sample when it asks HSIG a question.

    Deliberately label-free: everything here is derivable at inference time.
    A field carrying ground truth would make every estimate untrustworthy.
    """

    sample_id: str
    active_modalities: tuple[str, ...]
    uncertainty: float
    confidence: float | None = None
    predicted_class: int | None = None
    #: Modality-specific signals an estimator may condition on.
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.active_modalities:
            raise ValueError("A routing state must have at least one active modality")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                f"uncertainty must be normalised into [0, 1], got {self.uncertainty}"
            )

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "active_modalities": list(self.active_modalities),
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
            "features": dict(self.features),
        }


@dataclass(frozen=True)
class GainEstimate:
    """HSIG's answer for one candidate, or its refusal to give one."""

    modality: str
    #: Expected decision improvement; ``None`` when no estimator can answer.
    expected_improvement: float | None
    available: bool
    #: How the estimate was produced, or why it could not be.
    basis: str
    #: Optional interval, when an estimator can express its own uncertainty.
    interval: tuple[float, float] | None = None

    @property
    def estimated(self) -> bool:
        return self.expected_improvement is not None

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "expected_improvement": self.expected_improvement,
            "available": self.available,
            "estimated": self.estimated,
            "basis": self.basis,
            "interval": list(self.interval) if self.interval else None,
            "target": HSIG_TARGET,
        }


@runtime_checkable
class HSIG(Protocol):
    """The contract a real estimator must satisfy."""

    name: str

    def estimate_gain(self, state: RoutingState, candidate_modality: str) -> GainEstimate:
        """Expected decision improvement from acquiring ``candidate_modality``."""
        ...

    def estimate_all(
        self, state: RoutingState, candidates: Sequence[str]
    ) -> dict[str, GainEstimate]:
        ...

    def provenance(self) -> dict:
        ...


class UnavailableHSIG:
    """The only HSIG that exists at stage 2: one that declines to guess.

    Every candidate comes back with ``expected_improvement=None`` and a reason.
    This is not a placeholder to be quietly replaced by a constant -- it is the
    honest state of the system, and a router holding it must log "no gain
    estimate available" rather than proceeding as if it had one.
    """

    name = "unavailable"

    def __init__(self, reason: str | None = None):
        self.reason = reason or (
            "HSIG is not implemented. Stage 1 rejected the delta-uncertainty target and "
            "a decision-improvement estimator has not yet been fitted on validation."
        )

    def estimate_gain(self, state: RoutingState, candidate_modality: str) -> GainEstimate:
        return GainEstimate(
            modality=candidate_modality, expected_improvement=None,
            available=False, basis=self.reason,
        )

    def estimate_all(
        self, state: RoutingState, candidates: Sequence[str]
    ) -> dict[str, GainEstimate]:
        return {name: self.estimate_gain(state, name) for name in candidates}

    def require(self, state: RoutingState, candidate_modality: str) -> float:
        """For callers that cannot proceed without a number: raise, never invent one."""
        raise HSIGUnavailable(
            f"No gain estimate for {candidate_modality!r} on sample {state.sample_id!r}: "
            f"{self.reason}"
        )

    def provenance(self) -> dict:
        return {
            "hsig": self.name,
            "implemented": False,
            "reason": self.reason,
            **HSIG_TARGET_DEFINITION,
        }
