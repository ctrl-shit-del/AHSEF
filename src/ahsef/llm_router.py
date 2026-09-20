"""The stage-2 router: LLM text in, STOP or a recorded acquisition request out.

The loop this implements is exactly the one the stage-2 objective describes,
and no more::

    LLM(text) -> uncertainty -> U <= tau ?
        yes -> STOP
        no  -> which modality?  ask availability, then HSIG, then UGAPR
                 -> nothing available   : stop, and say so
                 -> no gain estimate    : stop, and say so
                 -> a selection         : (stage 3 acquires and re-evaluates)

Three properties are load-bearing:

**Availability is decided on evidence.**  Stage 1 established that these corpora
are not globally aligned, so "request video" is meaningless for a sample that
has no aligned video record.  The router consults
:class:`~src.ahsef.identity.AlignmentIndex` and reports *"additional modality
requested, but unavailable for this sample"* rather than pretending a fusion
could happen.

**No hard-coded order.**  The router never names a modality itself.  Candidates
come from the registry, availability from the alignment index, ranking from
UGAPR.  With no HSIG implemented the honest outcome is "cannot choose", and
that is what gets logged -- not a fallback to the first or cheapest candidate.

**The router never sees a label.**  ``true_class`` rides along on the trace for
post-hoc evaluation and is stamped ``router_saw_true_class: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from src.ahsef.gate import GateOutcome, apply_gate
from src.ahsef.hsig import GainEstimate, RoutingState, UnavailableHSIG
from src.ahsef.identity import AlignmentIndex
from src.ahsef.inference import PredictionSet
from src.ahsef.llm.inference import LLM_MODALITY, routing_uncertainty_column
from src.ahsef.routing_log import RoutingStep, RoutingTrace
from src.ahsef.ugapr import UGAPR, RankingResult
from src.common.labels import CANONICAL_EMOTION_CLASSES


#: Why a sample stopped.  All map onto Stage 1's declared STOP_REASONS.
REASON_SUFFICIENT = "uncertainty_below_threshold"
REASON_NO_CANDIDATE = "no_candidate_available"
REASON_CANNOT_RANK = "no_candidate_above_min_gain"


@dataclass
class SampleRouting:
    """One sample's complete stage-2 routing outcome."""

    sample_id: str
    dataset: str
    split: str
    predicted_class: int
    predicted_label: str
    confidence: float | None
    uncertainty: float
    threshold: float
    gate: GateOutcome
    ranking: RankingResult | None
    trace: RoutingTrace
    true_class: int | None = None
    availability: Mapping[str, str | None] = field(default_factory=dict)

    @property
    def stopped(self) -> bool:
        return self.gate.stop

    @property
    def requested(self) -> bool:
        return not self.gate.stop

    @property
    def acquired_anything(self) -> bool:
        return bool(self.ranking and self.ranking.selection_available)

    def flat_record(self) -> dict:
        """The compact per-sample routing record named in the stage-2 brief."""
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "split": self.split,
            "active_modalities": [LLM_MODALITY],
            "prediction": self.predicted_label,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "threshold": self.threshold,
            "stop": self.gate.stop,
            "reason": self.trace.final.stop_reason,
            "explanation": self.gate.reason,
            "requested_additional_modality": self.requested,
            "candidate_availability": dict(self.availability),
            "selected_modality": (
                self.ranking.selected_modality if self.ranking else None
            ),
            "selection_available": bool(self.ranking and self.ranking.selection_available),
            "router_saw_true_class": False,
        }


class LLMTextRouter:
    """Route one split of LLM predictions through the sufficiency gate."""

    def __init__(
        self,
        threshold: float,
        candidates: Sequence[str],
        alignment: AlignmentIndex | None = None,
        hsig: UnavailableHSIG | None = None,
        ugapr: UGAPR | None = None,
        uncertainty_policy: str = "score_entropy",
        policy_record: Mapping | None = None,
    ):
        self.threshold = float(threshold)
        self.candidates = list(candidates)
        self.alignment = alignment
        self.hsig = hsig or UnavailableHSIG()
        self.ugapr = ugapr or UGAPR()
        self.uncertainty_policy = uncertainty_policy
        self.policy_record = dict(policy_record or {})

    # --------------------------------------------------------- availability

    def availability_for(self, sample_id: str, split: str) -> dict[str, str | None]:
        """Which candidates have an aligned record for this exact sample.

        Without an alignment index nothing can be claimed, so every candidate is
        marked unavailable with that as the stated reason -- the safe direction,
        since the alternative is asserting a fusion that may not exist.
        """
        if self.alignment is None:
            return {
                name: "no alignment index was supplied; availability cannot be verified"
                for name in self.candidates
            }
        result: dict[str, str | None] = {}
        for name in self.candidates:
            if name not in self.alignment.modalities:
                result[name] = f"{name!r} is not in the alignment index"
                continue
            if sample_id in self.alignment[name].split_ids(split):
                result[name] = None
                continue
            elsewhere = [
                other for other in ("train", "validation", "test")
                if sample_id in self.alignment[name].split_ids(other)
            ]
            result[name] = (
                f"no aligned {name} record for this sample in the {split} split"
                + (
                    f"; it appears in that pool's {elsewhere[0]} split, which cannot be "
                    f"used here without leaking that model's training data"
                    if elsewhere else ""
                )
            )
        return result

    # -------------------------------------------------------------- routing

    def route_sample(
        self,
        sample_id: str,
        dataset: str,
        split: str,
        predicted_class: int,
        uncertainty: float,
        confidence: float | None,
        true_class: int | None = None,
    ) -> SampleRouting:
        gate = apply_gate(sample_id, uncertainty, self.threshold)
        label = (
            CANONICAL_EMOTION_CLASSES[predicted_class]
            if 0 <= predicted_class < len(CANONICAL_EMOTION_CLASSES) else "unresolved"
        )

        if gate.stop:
            step = RoutingStep(
                step=1, active_modalities=[LLM_MODALITY],
                predicted_class=max(predicted_class, 0), predicted_label=label,
                confidence=float(confidence if confidence is not None else 0.0),
                uncertainty=float(uncertainty), stop=True, stop_reason=REASON_SUFFICIENT,
            )
            return SampleRouting(
                sample_id=sample_id, dataset=dataset, split=split,
                predicted_class=predicted_class, predicted_label=label,
                confidence=confidence, uncertainty=float(uncertainty),
                threshold=self.threshold, gate=gate, ranking=None,
                trace=self._trace(sample_id, dataset, split, [step], true_class),
                true_class=true_class, availability={},
            )

        # Insufficient evidence: find out what could actually be acquired.
        availability = self.availability_for(sample_id, split)
        state = RoutingState(
            sample_id=sample_id, active_modalities=(LLM_MODALITY,),
            uncertainty=float(uncertainty), confidence=confidence,
            predicted_class=predicted_class if predicted_class >= 0 else None,
            features={"llm_uncertainty": float(uncertainty)},
        )
        gains: dict[str, GainEstimate] = self.hsig.estimate_all(state, self.candidates)
        ranking = self.ugapr.rank(state, gains, availability=availability)

        reason = (
            REASON_NO_CANDIDATE
            if all(value is not None for value in availability.values())
            else REASON_CANNOT_RANK
        )
        # Stage 2 stops either way -- there is no acquisition step yet. The trace
        # records *why* it stopped, which is the whole point of the exercise.
        step = RoutingStep(
            step=1, active_modalities=[LLM_MODALITY],
            predicted_class=max(predicted_class, 0), predicted_label=label,
            confidence=float(confidence if confidence is not None else 0.0),
            uncertainty=float(uncertainty), stop=True, stop_reason=reason,
            hsig={
                name: estimate.expected_improvement
                for name, estimate in gains.items()
                if estimate.expected_improvement is not None
            },
            ugapr={
                item.modality: item.to_dict() for item in ranking.candidates
            },
            candidates=ranking.as_candidate_scores(),
        )
        return SampleRouting(
            sample_id=sample_id, dataset=dataset, split=split,
            predicted_class=predicted_class, predicted_label=label,
            confidence=confidence, uncertainty=float(uncertainty),
            threshold=self.threshold, gate=gate, ranking=ranking,
            trace=self._trace(sample_id, dataset, split, [step], true_class),
            true_class=true_class, availability=availability,
        )

    def route_prediction_set(
        self, prediction_set: PredictionSet, skip_unusable: bool = True
    ) -> list[SampleRouting]:
        """Route a whole split.  Unusable LLM answers are reported, not routed.

        A sample the LLM could not answer has no uncertainty to gate on.
        Forcing it to one side of ``tau`` would invent a decision, so it is
        excluded here and counted in the coverage report instead.
        """
        frame = prediction_set.frame
        uncertainty = routing_uncertainty_column(prediction_set, self.uncertainty_policy)
        routings: list[SampleRouting] = []
        for position, (_, row) in enumerate(frame.iterrows()):
            value = uncertainty.iloc[position]
            if skip_unusable and (not bool(row.get("llm_usable", False)) or pd.isna(value)):
                continue
            routings.append(self.route_sample(
                sample_id=str(row["sample_id"]),
                dataset=str(row.get("dataset", "")),
                split=str(row["split"]),
                predicted_class=int(row["predicted_class"]),
                uncertainty=float(value),
                confidence=(
                    None if pd.isna(row.get("confidence")) else float(row["confidence"])
                ),
                true_class=int(row["true_class"]) if "true_class" in frame.columns else None,
            ))
        return routings

    # ------------------------------------------------------------- helpers

    def _trace(self, sample_id, dataset, split, steps, true_class) -> RoutingTrace:
        return RoutingTrace(
            sample_id=sample_id, dataset=dataset, split=split,
            class_order=list(CANONICAL_EMOTION_CLASSES), steps=steps,
            true_class=true_class, policy=self.policy(),
        )

    def policy(self) -> dict:
        return {
            "stage": "stage2_llm_text_gate",
            "anchor_modality": LLM_MODALITY,
            "uncertainty_threshold": self.threshold,
            "uncertainty_policy": self.uncertainty_policy,
            "candidates": list(self.candidates),
            "hsig": self.hsig.provenance(),
            "ugapr": self.ugapr.provenance(),
            "acquisition_implemented": False,
            "acquisition_note": (
                "Stage 2 decides whether more evidence is needed and records what could "
                "have been acquired. It does not acquire or fuse anything: no second "
                "modality is activated, so no claim about improved recognition is made."
            ),
            **self.policy_record,
        }


def routing_summary(routings: Sequence[SampleRouting]) -> dict:
    """Counts, rates, and -- read strictly afterwards -- accuracy either side."""
    if not routings:
        raise ValueError("No routings to summarise")
    total = len(routings)
    stopped = [item for item in routings if item.stopped]
    requested = [item for item in routings if item.requested]

    unavailable_only = [
        item for item in requested
        if item.availability and all(value is not None for value in item.availability.values())
    ]
    reasons: dict[str, int] = {}
    for item in routings:
        key = item.trace.final.stop_reason or "unspecified"
        reasons[key] = reasons.get(key, 0) + 1

    record = {
        "samples": total,
        "threshold": routings[0].threshold,
        "text_sufficient_stop": len(stopped),
        "text_insufficient_request": len(requested),
        "stop_rate": len(stopped) / total,
        "request_rate": len(requested) / total,
        "requests_with_no_available_modality": len(unavailable_only),
        "requests_fulfilled": sum(1 for item in routings if item.acquired_anything),
        "stop_reasons": reasons,
        "average_modalities_activated": 1.0,
        "average_modalities_note": (
            "Exactly 1.0 by construction: stage 2 implements the decision, not the "
            "acquisition. This is not a result about modality economy."
        ),
    }
    labelled = [item for item in routings if item.true_class is not None]
    if labelled:
        def accuracy(items):
            return (
                sum(1 for i in items if i.predicted_class == i.true_class) / len(items)
                if items else None
            )
        record["outcome"] = {
            "overall_accuracy": accuracy(labelled),
            "accuracy_when_stopped": accuracy([i for i in labelled if i.stopped]),
            "accuracy_when_requested": accuracy([i for i in labelled if i.requested]),
            "note": "Computed after every routing decision was made; the router is "
                    "label-blind.",
        }
    return record
