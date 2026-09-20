"""PHASE G -- the Text -> HSIG -> UGAPR -> Audio -> Fusion router.

One sample, one loop::

    text/LLM evidence
        -> HSIG estimates gain(audio | current state)
        -> UGAPR computes J = gain - lambda*cost - mu*latency
        -> J >= tau ?  REQUEST_AUDIO : STOP
        -> if REQUEST_AUDIO and audio is available:
               audio inference, fusion, new prediction and uncertainty
           if REQUEST_AUDIO and audio is NOT available:
               REQUESTED_BUT_UNAVAILABLE -- keep the text prediction, fabricate nothing

Four properties are structural, not conventional:

**The router never receives a label.**  :meth:`TextAudioRouter.route` takes a
:class:`~src.ahsef.stage3.features.Stage3State` and nothing else; there is no
parameter through which ground truth could arrive, and the decision record
stamps ``router_saw_true_class: false``.

**The decision is not a hard-coded uncertainty rule.**  Uncertainty is one
feature among several inside HSIG; the acquisition decision is made on the UGAPR
utility, which is a function of the estimated gain and the measured cost and
latency.  The ablation suite includes the uncertainty-only policy precisely so
the difference can be measured rather than asserted.

**An unavailable modality is reported, never imputed.**  No zeros, no uniform
distribution, no neutral, no substituted modality.  The sample keeps its
text-only prediction and the record says the acquisition was requested and
failed.

**Cost is real even with one candidate.**  With ``{audio}`` alone UGAPR cannot
demonstrate *ranking*, and the code says so in
:data:`SINGLE_CANDIDATE_LIMITATION` rather than implying otherwise; what it can
and does do is decide whether the single candidate clears its own price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ahsef.costs import CostModel
from src.ahsef.hsig import GainEstimate, RoutingState
from src.ahsef.inference import PredictionSet
from src.ahsef.routing_log import CandidateScore, RoutingStep, RoutingTrace
from src.ahsef.stage3.features import Stage3State
from src.ahsef.ugapr import UGAPR
from src.common.labels import CANONICAL_EMOTION_CLASSES

#: The Stage 3 candidate registry.  A list, not a hard-coded name: adding a
#: second candidate is a configuration change, not a rewrite of the router.
DEFAULT_CANDIDATES: tuple[str, ...] = ("audio",)

SINGLE_CANDIDATE_LIMITATION = (
    "The Stage 3 candidate set has exactly one member, because no sample in this "
    "project has two simultaneously available candidate modalities. UGAPR therefore "
    "demonstrates the accept/reject half of its job -- does the candidate clear its "
    "own cost and latency price? -- and NOT the ranking half. Audio is not selected "
    "because it is the only option: a sample whose utility falls below the frozen "
    "threshold is stopped with audio unacquired, and the ablations report how often "
    "that happens."
)


class DecisionKind:
    STOP = "STOP"
    REQUEST = "REQUEST_AUDIO"


class Availability:
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "REQUESTED_BUT_UNAVAILABLE"
    NOT_REQUESTED = "NOT_REQUESTED"


class RouterLabelError(RuntimeError):
    """Raised if ground truth is passed into the routing path."""


# ============================================================
# Availability
# ============================================================

class AudioAvailability:
    """Whether audio can actually be obtained for a sample, decided on evidence.

    Availability is read from the audio prediction set the frozen baseline
    produced -- an id with no row has no obtainable audio evidence -- optionally
    narrowed further by an explicit deny set.  It is never assumed from the
    corpus, and never assumed from the fact that the pool was built.
    """

    def __init__(
        self,
        audio: PredictionSet | None = None,
        available_ids: Sequence[str] | None = None,
        unavailable_reason: str = "no audio prediction exists for this sample id",
    ):
        if audio is None and available_ids is None:
            raise ValueError(
                "AudioAvailability needs either an audio prediction set or an explicit "
                "id list; defaulting to 'everything is available' would make the "
                "unavailable branch untestable."
            )
        self._ids = set(
            available_ids if available_ids is not None else audio.sample_ids()
        )
        self.reason = unavailable_reason

    def reason_for(self, sample_id: str) -> str | None:
        """``None`` when available, otherwise why it is not."""
        return None if str(sample_id) in self._ids else self.reason

    def __contains__(self, sample_id: str) -> bool:
        return str(sample_id) in self._ids

    def provenance(self) -> dict:
        return {
            "source": "frozen audio prediction set sample ids",
            "available_samples": len(self._ids),
            "unavailable_reason": self.reason,
            "policy": (
                "A requested but unavailable modality yields "
                "REQUESTED_BUT_UNAVAILABLE. The text-only prediction is kept and no "
                "fused prediction is produced. Nothing is zero-filled, uniform-filled, "
                "set to neutral, or substituted with another modality."
            ),
        }


# ============================================================
# The decision record
# ============================================================

@dataclass
class RoutingDecision:
    """Everything Phase G requires recorded for one sample."""

    sample_id: str
    split: str
    initial_modality: str
    initial_prediction: int
    initial_uncertainty: float
    initial_confidence: float | None
    hsig_predicted_gain: float | None
    hsig_basis: str
    ugapr_utility: float | None
    normalized_cost: float | None
    normalized_latency: float | None
    threshold: float
    decision: str
    requested_modality: str | None
    availability: str
    audio_prediction: int | None
    fused_prediction: int | None
    final_prediction: int
    final_uncertainty: float
    final_confidence: float | None
    routing_step: int
    modalities_activated: int
    latency_ms: float
    stop_reason: str
    reason: str
    timestamp: str
    provenance: dict = field(default_factory=dict)
    #: Attached after routing, for evaluation only. Never read by the router.
    true_class: int | None = None

    @property
    def final_correctness(self) -> bool | None:
        if self.true_class is None:
            return None
        return int(self.final_prediction) == int(self.true_class)

    def to_dict(self) -> dict:
        return {
            "record": "decision",
            "sample_id": self.sample_id,
            "split": self.split,
            "initial_modality": self.initial_modality,
            "initial_prediction": self.initial_prediction,
            "initial_prediction_label": _label(self.initial_prediction),
            "initial_uncertainty": self.initial_uncertainty,
            "initial_confidence": self.initial_confidence,
            "hsig_predicted_gain": self.hsig_predicted_gain,
            "hsig_basis": self.hsig_basis,
            "ugapr_utility": self.ugapr_utility,
            "normalized_cost": self.normalized_cost,
            "normalized_latency": self.normalized_latency,
            "acquisition_threshold": self.threshold,
            "decision": self.decision,
            "requested_modality": self.requested_modality,
            "availability": self.availability,
            "audio_prediction": self.audio_prediction,
            "audio_prediction_label": _label(self.audio_prediction),
            "fused_prediction": self.fused_prediction,
            "fused_prediction_label": _label(self.fused_prediction),
            "final_prediction": self.final_prediction,
            "final_prediction_label": _label(self.final_prediction),
            "final_uncertainty": self.final_uncertainty,
            "final_confidence": self.final_confidence,
            "routing_step": self.routing_step,
            "modalities_activated": self.modalities_activated,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "router_saw_true_class": False,
            "true_class": self.true_class,
            "true_label": _label(self.true_class),
            "final_correctness": self.final_correctness,
            "provenance": dict(self.provenance),
        }


def _label(class_id: int | None) -> str | None:
    if class_id is None or class_id < 0 or class_id >= len(CANONICAL_EMOTION_CLASSES):
        return None
    return CANONICAL_EMOTION_CLASSES[int(class_id)]


# ============================================================
# The router
# ============================================================

class TextAudioRouter:
    """The Stage 3 dynamic router.  Decides, acquires, fuses, and records."""

    def __init__(
        self,
        hsig,
        ugapr: UGAPR,
        threshold: float,
        availability: AudioAvailability,
        audio: PredictionSet,
        fused: PredictionSet,
        candidates: Sequence[str] = DEFAULT_CANDIDATES,
        cost_model: CostModel | None = None,
        text_latency_ms: float = 0.0,
        audio_latency_ms: float = 0.0,
        split: str = "validation",
        policy: Mapping | None = None,
    ):
        self.hsig = hsig
        self.ugapr = ugapr
        self.threshold = float(threshold)
        self.availability = availability
        self.candidates = tuple(candidates)
        self.cost_model = cost_model
        self.text_latency_ms = float(text_latency_ms)
        self.audio_latency_ms = float(audio_latency_ms)
        self.split = split
        self.policy = dict(policy or {})

        # UGAPR normalises a candidate's cost over the set it is handed. Handing
        # it the one-member candidate set would make audio's normalised cost
        # exactly 1.0 by construction -- an artefact of there being nothing to
        # compare against, not a measurement. It gets the full measured registry
        # instead, which is also what the freeze computed its penalty from.
        self._normalisation_pool = (
            sorted(cost_model.costs) if cost_model is not None else list(self.candidates)
        )

        # Acquisition results are looked up by id, never by position: the router
        # asks "what does audio say about THIS sample", and a positional lookup
        # would answer with a different sample's evidence.
        self._audio = audio.frame.set_index(audio.frame["sample_id"].astype(str))
        self._fused = fused.frame.set_index(fused.frame["sample_id"].astype(str))

    # ---------------------------------------------------------------- decide

    def decide(self, state: RoutingState) -> tuple[dict, dict, GainEstimate]:
        """HSIG then UGAPR, in the order the real decision happens.

        Two questions are asked, and conflating them is what makes an
        unavailable modality look like an unwanted one:

        ``pricing``
            *Is this candidate worth its price?*  Scored with every candidate
            marked available, because whether audio is obtainable for this
            particular sample has no bearing on whether audio is worth
            obtaining.  This is the utility the acquisition threshold applies
            to.

        ``selection``
            *Given what is actually obtainable, what would UGAPR take?*  Scored
            with the real availability, so the record shows the refusal and its
            reason.

        A router that only asked the second question would report "did not
        acquire" for a sample it very much wanted to acquire and could not --
        turning a data-availability failure into a policy decision.
        """
        gains = self.hsig.estimate_all(state, self.candidates)
        pricing = self.ugapr.rank(
            state, gains,
            availability={name: None for name in self.candidates},
            candidate_set=self._normalisation_pool,
        )
        availability = {
            name: (self.availability.reason_for(state.sample_id) if name == "audio" else
                   f"{name} is not registered as an acquirable candidate at Stage 3")
            for name in self.candidates
        }
        selection = self.ugapr.rank(
            state, gains, availability=availability,
            candidate_set=self._normalisation_pool,
        )
        return pricing.to_dict(), selection.to_dict(), gains.get("audio")

    # ----------------------------------------------------------------- route

    def route(self, state: Stage3State) -> tuple[RoutingDecision, RoutingTrace]:
        """Route one sample.  Takes no label and has no parameter for one."""
        if hasattr(state, "true_class"):
            raise RouterLabelError(
                "The routing state carries a true_class attribute. The router must "
                "not be able to see ground truth."
            )
        routing_state = state.to_routing_state()
        pricing, ranked, estimate = self.decide(routing_state)
        priced = pricing["candidate_scores"].get("audio", {})
        scores = ranked["candidate_scores"].get("audio", {})
        # The threshold applies to the price-adjusted utility, which exists
        # whether or not this sample's audio can be obtained.
        utility = priced.get("utility")
        gain = priced.get("estimated_improvement")
        availability_reason = scores.get("unavailable_reason")

        # The acquisition rule. Not "if uncertain": a threshold on the utility,
        # which is the estimated gain net of the measured cost and latency price.
        wants_audio = utility is not None and utility >= self.threshold

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        common = dict(
            sample_id=state.sample_id, split=self.split, initial_modality="text_llm",
            initial_prediction=int(state.predicted_class),
            initial_uncertainty=float(state.uncertainty),
            initial_confidence=state.confidence,
            hsig_predicted_gain=gain, hsig_basis=estimate.basis if estimate else "",
            ugapr_utility=utility,
            normalized_cost=priced.get("normalized_cost"),
            normalized_latency=priced.get("normalized_latency"),
            threshold=self.threshold, timestamp=timestamp,
            provenance={
                "hsig": getattr(self.hsig, "name", type(self.hsig).__name__),
                "ugapr_weights": ranked["weights"],
                "candidates": list(self.candidates),
                "ugapr_pricing_reason": pricing["reason"],
                "ugapr_selection_reason": ranked["reason"],
                "ugapr_selection_available": ranked["selection_available"],
                "single_candidate_limitation": SINGLE_CANDIDATE_LIMITATION,
                **self.policy,
            },
        )

        if not wants_audio:
            reason = (
                f"J(audio) = {utility:.6f} < tau {self.threshold:.6f}: the estimated "
                f"gain does not cover audio's cost and latency price"
                if utility is not None else
                f"no utility could be computed for audio ({ranked['reason']}); the "
                f"router does not acquire on an undefined utility"
            )
            decision = RoutingDecision(
                **common,
                decision=DecisionKind.STOP, requested_modality=None,
                availability=Availability.NOT_REQUESTED,
                audio_prediction=None, fused_prediction=None,
                final_prediction=int(state.predicted_class),
                final_uncertainty=float(state.uncertainty),
                final_confidence=state.confidence,
                routing_step=1, modalities_activated=1,
                latency_ms=self.text_latency_ms,
                stop_reason=(
                    "no_candidate_above_min_utility" if utility is not None
                    else "no_candidate_available"
                ),
                reason=reason,
            )
            return decision, self._trace(decision, ranked, pricing, acquired=False)

        if availability_reason is not None or state.sample_id not in self.availability:
            # Requested and could not be obtained. The text prediction stands;
            # no fused prediction is produced and no filler is invented.
            decision = RoutingDecision(
                **common,
                decision=DecisionKind.REQUEST, requested_modality="audio",
                availability=Availability.UNAVAILABLE,
                audio_prediction=None, fused_prediction=None,
                final_prediction=int(state.predicted_class),
                final_uncertainty=float(state.uncertainty),
                final_confidence=state.confidence,
                routing_step=1, modalities_activated=1,
                latency_ms=self.text_latency_ms,
                stop_reason="requested_modality_unavailable",
                reason=(
                    f"REQUESTED_BUT_UNAVAILABLE: audio was worth acquiring "
                    f"(J={utility:.6f} >= tau {self.threshold:.6f}) but could not be "
                    f"obtained ({availability_reason or 'no audio evidence for this id'}). "
                    f"The text-only prediction stands; no fused prediction was produced."
                ),
            )
            return decision, self._trace(decision, ranked, pricing, acquired=False)

        audio_row = self._audio.loc[state.sample_id]
        fused_row = self._fused.loc[state.sample_id]
        decision = RoutingDecision(
            **common,
            decision=DecisionKind.REQUEST, requested_modality="audio",
            availability=Availability.AVAILABLE,
            audio_prediction=int(audio_row["predicted_class"]),
            fused_prediction=int(fused_row["predicted_class"]),
            final_prediction=int(fused_row["predicted_class"]),
            final_uncertainty=float(fused_row["normalized_entropy"]),
            final_confidence=float(fused_row["confidence"]),
            routing_step=2, modalities_activated=2,
            latency_ms=self.text_latency_ms + self.audio_latency_ms,
            stop_reason="candidate_pool_exhausted",
            reason=(
                f"REQUEST_AUDIO: J(audio) = {utility:.6f} >= tau {self.threshold:.6f}. "
                f"Audio acquired and fused; uncertainty moved "
                f"{state.uncertainty:.4f} -> {float(fused_row['normalized_entropy']):.4f}."
            ),
        )
        return decision, self._trace(decision, ranked, pricing, acquired=True)

    def route_all(self, states: Sequence[Stage3State]):
        decisions, traces = [], []
        for state in states:
            decision, trace = self.route(state)
            decisions.append(decision)
            traces.append(trace)
        return decisions, traces

    # ----------------------------------------------------------------- trace

    def _trace(
        self, decision: RoutingDecision, ranked: dict, pricing: dict, acquired: bool
    ) -> RoutingTrace:
        """A Stage 1 :class:`RoutingTrace`, so Stage 3 stays auditable by Stage 1 tools.

        Availability comes from the availability-aware ranking and the utility
        from the pricing one, so a trace shows both what the candidate was worth
        and whether it could be had.
        """
        priced = pricing["candidate_scores"]
        candidates = [
            CandidateScore(
                modality=name,
                available=bool(record["available"]),
                estimated_delta_uncertainty=priced[name]["estimated_improvement"],
                normalized_cost=priced[name]["normalized_cost"],
                normalized_latency=priced[name]["normalized_latency"],
                utility=priced[name]["utility"],
                unavailable_reason=record["unavailable_reason"],
            )
            for name, record in ranked["candidate_scores"].items()
        ]
        hsig_block = {
            name: float(record["estimated_improvement"])
            for name, record in priced.items()
            if record["estimated_improvement"] is not None
        }
        first = RoutingStep(
            step=1, active_modalities=["text_llm"],
            predicted_class=decision.initial_prediction,
            predicted_label=_label(decision.initial_prediction) or "unmapped",
            confidence=float(decision.initial_confidence or 0.0),
            uncertainty=float(np.clip(decision.initial_uncertainty, 0.0, 1.0)),
            stop=not acquired,
            stop_reason=None if acquired else decision.stop_reason,
            hsig=hsig_block,
            ugapr={
                name: {**priced[name], "available": record["available"],
                       "unavailable_reason": record["unavailable_reason"]}
                for name, record in ranked["candidate_scores"].items()
            },
            candidates=candidates,
            selected_modality="audio" if acquired else None,
            cumulative_latency_ms=self.text_latency_ms,
        )
        steps = [first]
        if acquired:
            steps.append(RoutingStep(
                step=2, active_modalities=["text_llm", "audio"],
                predicted_class=int(decision.final_prediction),
                predicted_label=_label(decision.final_prediction) or "unmapped",
                confidence=float(decision.final_confidence or 0.0),
                uncertainty=float(np.clip(decision.final_uncertainty, 0.0, 1.0)),
                stop=True, stop_reason="candidate_pool_exhausted",
                cumulative_latency_ms=decision.latency_ms,
            ))
        return RoutingTrace(
            sample_id=decision.sample_id, dataset="MSP-Podcast", split=self.split,
            class_order=list(CANONICAL_EMOTION_CLASSES), steps=steps,
            true_class=decision.true_class,
            policy={
                "acquisition_threshold": self.threshold,
                "candidates": list(self.candidates),
                **self.policy,
            },
        )

    def provenance(self) -> dict:
        return {
            "router": "stage3_text_audio",
            "candidates": list(self.candidates),
            "acquisition_threshold": self.threshold,
            "rule": "REQUEST_AUDIO when J(audio) >= tau, else STOP",
            "hsig": self.hsig.provenance(),
            "ugapr": self.ugapr.provenance(),
            "availability": self.availability.provenance(),
            "latency_ms": {
                "text_llm": self.text_latency_ms, "audio": self.audio_latency_ms,
            },
            "router_saw_true_class": False,
            "single_candidate_limitation": SINGLE_CANDIDATE_LIMITATION,
        }


# ============================================================
# Summaries
# ============================================================

def decisions_frame(decisions: Sequence[RoutingDecision]) -> pd.DataFrame:
    return pd.DataFrame([decision.to_dict() for decision in decisions])


def attach_truth(
    decisions: Sequence[RoutingDecision], labels: Mapping[str, int]
) -> list[RoutingDecision]:
    """Attach ground truth *after* every decision is made, for evaluation only."""
    for decision in decisions:
        decision.true_class = int(labels[decision.sample_id])
    return list(decisions)


def decision_summary(decisions: Sequence[RoutingDecision]) -> dict:
    frame = decisions_frame(decisions)
    total = len(frame)
    requested = frame["decision"] == "REQUEST_AUDIO"
    unavailable = frame["availability"] == Availability.UNAVAILABLE
    acquired = requested & ~unavailable
    return {
        "samples": int(total),
        "stop": int((~requested).sum()),
        "request_audio": int(requested.sum()),
        "acquisition_rate": float(requested.mean()) if total else None,
        "audio_acquired": int(acquired.sum()),
        "requested_but_unavailable": int(unavailable.sum()),
        "average_modalities_activated": float(frame["modalities_activated"].mean())
        if total else None,
        "mean_latency_ms": float(frame["latency_ms"].mean()) if total else None,
        "stop_reasons": {
            str(key): int(value) for key, value in frame["stop_reason"].value_counts().items()
        },
        "uncertainty_before_acquisition": {
            "all": float(frame["initial_uncertainty"].mean()) if total else None,
            "stopped": float(frame.loc[~requested, "initial_uncertainty"].mean())
            if (~requested).any() else None,
            "requested": float(frame.loc[requested, "initial_uncertainty"].mean())
            if requested.any() else None,
        },
        "uncertainty_after_acquisition": {
            "requested_and_acquired": float(frame.loc[acquired, "final_uncertainty"].mean())
            if acquired.any() else None,
            "mean_reduction": float(
                (frame.loc[acquired, "initial_uncertainty"]
                 - frame.loc[acquired, "final_uncertainty"]).mean()
            ) if acquired.any() else None,
        },
        "hsig_gain": {
            "mean_when_stopped": float(frame.loc[~requested, "hsig_predicted_gain"].mean())
            if (~requested).any() else None,
            "mean_when_requested": float(frame.loc[requested, "hsig_predicted_gain"].mean())
            if requested.any() else None,
        },
    }
