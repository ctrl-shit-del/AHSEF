"""UGAPR -- Utility-Guided Adaptive Progressive Routing: the interface only.

UGAPR ranks the inactive candidates by::

    J(m) = gain_hat(m) - lambda * C(m) - mu * L(m)

where ``gain_hat`` comes from HSIG, ``C`` and ``L`` are the normalised compute
and latency costs Stage 1 already measures, and ``lambda`` / ``mu`` are the
configured penalties.

**No selection is faked.**  ``J`` is undefined without ``gain_hat``, and stage 2
has no HSIG (see :mod:`src.ahsef.hsig`).  So :meth:`UGAPR.rank` returns scored
candidates whose utility is ``None``, marks the selection as unavailable, and
names the missing input.  It does *not* fall back to "pick the cheapest" or
"pick the first" -- either would be a hard-coded modality order wearing a
utility function's clothes, which is precisely what this project forbids.

What is real here is the cost side: ``lambda`` and ``mu`` are configurable and
recorded, the cost model is Stage 1's measured one, and availability is decided
by Stage 1's alignment index rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from src.ahsef.costs import CostModel
from src.ahsef.hsig import GainEstimate, RoutingState


@dataclass(frozen=True)
class UtilityWeights:
    """The configurable penalties in ``J``.  Never hidden in code."""

    lambda_cost: float = 0.10
    mu_latency: float = 0.10
    min_gain: float = 0.0
    min_utility: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_cost < 0 or self.mu_latency < 0:
            raise ValueError("lambda and mu must be non-negative")

    def to_dict(self) -> dict:
        return {
            "lambda_cost": self.lambda_cost, "mu_latency": self.mu_latency,
            "min_gain": self.min_gain, "min_utility": self.min_utility,
            "formula": "J(m) = gain_hat(m) - lambda * C(m) - mu * L(m)",
        }


@dataclass(frozen=True)
class CandidateUtility:
    """One candidate's full scoring record, including why it may be unusable."""

    modality: str
    available: bool
    gain: GainEstimate
    normalized_cost: float | None
    normalized_latency: float | None
    utility: float | None
    unavailable_reason: str | None = None

    @property
    def scorable(self) -> bool:
        return self.utility is not None

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "available": self.available,
            "estimated_improvement": self.gain.expected_improvement,
            "gain_basis": self.gain.basis,
            "normalized_cost": self.normalized_cost,
            "normalized_latency": self.normalized_latency,
            "utility": self.utility,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass
class RankingResult:
    """UGAPR's output: every candidate scored, and a selection or a refusal."""

    candidates: list[CandidateUtility] = field(default_factory=list)
    selected_modality: str | None = None
    selection_available: bool = False
    reason: str = ""
    weights: UtilityWeights = field(default_factory=UtilityWeights)

    def to_dict(self) -> dict:
        return {
            "candidate_scores": {
                item.modality: item.to_dict() for item in self.candidates
            },
            "selected_modality": self.selected_modality,
            "selection_available": self.selection_available,
            "reason": self.reason,
            "weights": self.weights.to_dict(),
        }

    def as_candidate_scores(self) -> list:
        """Adapt to the routing-log ``CandidateScore`` shape."""
        from src.ahsef.routing_log import CandidateScore

        return [
            CandidateScore(
                modality=item.modality,
                available=item.available,
                estimated_delta_uncertainty=item.gain.expected_improvement,
                normalized_cost=item.normalized_cost,
                normalized_latency=item.normalized_latency,
                utility=item.utility,
                unavailable_reason=item.unavailable_reason,
            )
            for item in self.candidates
        ]


class UGAPR:
    """Rank inactive candidates by utility -- when a gain estimate exists."""

    def __init__(
        self,
        weights: UtilityWeights | None = None,
        cost_model: CostModel | None = None,
    ):
        self.weights = weights or UtilityWeights()
        self.cost_model = cost_model

    def rank(
        self,
        state: RoutingState,
        gains: Mapping[str, GainEstimate],
        availability: Mapping[str, str | None] | None = None,
        candidate_set: Sequence[str] | None = None,
    ) -> RankingResult:
        """Score every candidate; select only if utility is genuinely defined.

        ``availability`` maps a candidate to ``None`` (available) or to the
        reason it is not -- Stage 1's alignment index supplies it, so an
        unfusable modality is refused on evidence rather than on assumption.
        """
        names = list(gains)
        pool = [name for name in (candidate_set or names) if name in self.cost_model] \
            if self.cost_model else list(candidate_set or names)

        scored: list[CandidateUtility] = []
        for name in names:
            reason = (availability or {}).get(name)
            available = reason is None
            gain = gains[name]
            cost = latency = None
            if self.cost_model is not None and name in self.cost_model:
                cost = self.cost_model.normalized_cost(name, pool or None)
                latency = self.cost_model.normalized_latency(name, pool or None)
            # J is undefined without a gain estimate, an availability, and both
            # cost terms. Any missing piece leaves utility None -- not zero.
            utility = None
            if available and gain.estimated and cost is not None and latency is not None:
                utility = (
                    float(gain.expected_improvement)
                    - self.weights.lambda_cost * cost
                    - self.weights.mu_latency * latency
                )
            scored.append(CandidateUtility(
                modality=name, available=available, gain=gain,
                normalized_cost=cost, normalized_latency=latency,
                utility=utility, unavailable_reason=reason,
            ))

        return self._select(scored)

    def _select(self, scored: list[CandidateUtility]) -> RankingResult:
        usable = [item for item in scored if item.scorable]
        if not usable:
            missing_gain = [item.modality for item in scored if not item.gain.estimated]
            unavailable = [item.modality for item in scored if not item.available]
            if missing_gain:
                reason = (
                    f"No utility could be computed: HSIG returned no gain estimate for "
                    f"{sorted(missing_gain)}. J(m) is undefined without it, and UGAPR does "
                    f"not substitute a default -- selecting on cost alone would be a "
                    f"hard-coded modality order."
                )
            elif unavailable:
                reason = (
                    f"No candidate is available for this sample "
                    f"(unavailable: {sorted(unavailable)})."
                )
            else:
                reason = "No candidate could be scored."
            return RankingResult(
                candidates=scored, selected_modality=None, selection_available=False,
                reason=reason, weights=self.weights,
            )

        eligible = [
            item for item in usable
            if (item.gain.expected_improvement or 0.0) >= self.weights.min_gain
            and (item.utility or 0.0) >= self.weights.min_utility
        ]
        if not eligible:
            return RankingResult(
                candidates=scored, selected_modality=None, selection_available=False,
                reason=(
                    f"Every scorable candidate fell below the configured floors "
                    f"(min_gain={self.weights.min_gain}, "
                    f"min_utility={self.weights.min_utility}); acquiring is not worth it."
                ),
                weights=self.weights,
            )
        best = max(eligible, key=lambda item: item.utility)
        return RankingResult(
            candidates=scored, selected_modality=best.modality, selection_available=True,
            reason=f"highest utility J={best.utility:.6f}", weights=self.weights,
        )

    def provenance(self) -> dict:
        return {
            "ugapr": "interface",
            "implemented": False,
            "weights": self.weights.to_dict(),
            "cost_model": self.cost_model.table() if self.cost_model else None,
            "note": (
                "Utility is computed only when HSIG supplies a gain estimate. With no "
                "estimator implemented, UGAPR reports selection_available=False and "
                "selects nothing. It never falls back to a cost-only or fixed order."
            ),
        }
