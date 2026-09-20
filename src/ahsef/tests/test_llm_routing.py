"""Stage-2 routing: availability, the HSIG/UGAPR boundary, and the trace."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ahsef.hsig import (
    HSIG_TARGET_DEFINITION,
    GainEstimate,
    HSIGUnavailable,
    RoutingState,
    UnavailableHSIG,
)
from src.ahsef.identity import AlignmentIndex, ModalitySplits
from src.ahsef.llm.inference import LLM_MODALITY, build_prediction_set
from src.ahsef.llm.provider import ScriptedProvider
from src.ahsef.llm.inference import LLMTextModality
from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm_router import LLMTextRouter, routing_summary
from src.ahsef.ugapr import UGAPR, UtilityWeights


def splits(modality: str, validation, task="emotion_7class") -> ModalitySplits:
    ids = {"train": frozenset(), "validation": frozenset(validation), "test": frozenset()}
    return ModalitySplits(
        modality=modality, experiment=f"{modality}_x", metadata_dir="x", ids=ids,
        labels={key: {i: 0 for i in value} for key, value in ids.items()}, task=task,
    )


@pytest.fixture
def alignment() -> AlignmentIndex:
    """s0 can acquire audio; s1 can acquire video; s2 can acquire nothing."""
    return AlignmentIndex({
        "audio": splits("audio", ["s0"]),
        "video": splits("video", ["s1"]),
        "image": splits("image", []),
    })


@pytest.fixture
def router(alignment) -> LLMTextRouter:
    return LLMTextRouter(
        threshold=0.5, candidates=["audio", "video", "image"], alignment=alignment,
    )


def route(router, sample_id="s0", uncertainty=0.9, predicted=1, true=1):
    return router.route_sample(
        sample_id=sample_id, dataset="MELD", split="validation",
        predicted_class=predicted, uncertainty=uncertainty, confidence=0.4, true_class=true,
    )


# ------------------------------------------------------------ availability

def test_availability_comes_from_the_alignment_index(router):
    available = router.availability_for("s0", "validation")
    assert available["audio"] is None
    assert available["video"] is not None
    assert available["image"] is not None


def test_a_sample_with_nothing_aligned_reports_every_candidate_unavailable(router):
    available = router.availability_for("s2", "validation")
    assert all(reason is not None for reason in available.values())


def test_without_an_alignment_index_nothing_is_claimed_available():
    """The safe direction: never assert a fusion that cannot be verified."""
    bare = LLMTextRouter(threshold=0.5, candidates=["audio"], alignment=None)
    available = bare.availability_for("s0", "validation")
    assert available["audio"] is not None
    assert "cannot be verified" in available["audio"]


def test_a_cross_split_match_is_reported_as_a_leakage_hazard():
    index = AlignmentIndex({
        "audio": ModalitySplits(
            modality="audio", experiment="a", metadata_dir="x",
            ids={"train": frozenset(["s9"]), "validation": frozenset(), "test": frozenset()},
            labels={"train": {"s9": 0}, "validation": {}, "test": {}},
            task="emotion_7class",
        )
    })
    router = LLMTextRouter(threshold=0.5, candidates=["audio"], alignment=index)
    reason = router.availability_for("s9", "validation")["audio"]
    assert "train" in reason and "leaking" in reason


# ------------------------------------------------------------------ gate

def test_low_uncertainty_stops_with_the_sufficiency_reason(router):
    routing = route(router, uncertainty=0.2)
    assert routing.stopped
    assert routing.trace.final.stop_reason == "uncertainty_below_threshold"
    assert routing.ranking is None


def test_high_uncertainty_requests_an_additional_modality(router):
    routing = route(router, uncertainty=0.9)
    assert routing.requested
    assert routing.ranking is not None


def test_a_request_with_nothing_aligned_says_so(router):
    routing = route(router, sample_id="s2", uncertainty=0.9)
    assert routing.requested
    assert routing.trace.final.stop_reason == "no_candidate_available"
    assert not routing.acquired_anything


def test_a_request_with_an_aligned_candidate_still_acquires_nothing(router):
    """Availability is not enough: without HSIG there is no gain estimate."""
    routing = route(router, sample_id="s0", uncertainty=0.9)
    assert routing.requested
    assert any(reason is None for reason in routing.availability.values())
    assert not routing.acquired_anything
    assert routing.ranking.selected_modality is None


# ------------------------------------------------------- HSIG / UGAPR

def test_hsig_declines_rather_than_returning_a_number():
    hsig = UnavailableHSIG()
    state = RoutingState("s0", ("text_llm",), 0.9)
    estimate = hsig.estimate_gain(state, "audio")
    assert estimate.expected_improvement is None
    assert estimate.estimated is False
    assert "not implemented" in estimate.basis


def test_hsig_require_raises_instead_of_inventing():
    with pytest.raises(HSIGUnavailable):
        UnavailableHSIG().require(RoutingState("s0", ("text_llm",), 0.9), "audio")


def test_hsig_provenance_records_the_rejected_stage1_target():
    record = UnavailableHSIG().provenance()
    assert record["implemented"] is False
    assert record["target"] == "expected_decision_improvement"
    assert "delta_uncertainty" in record["rejected_target"]
    assert "0.143" in HSIG_TARGET_DEFINITION["why_rejected"]


def test_routing_state_rejects_unnormalised_uncertainty():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RoutingState("s0", ("text_llm",), 1.4)


def test_routing_state_needs_an_active_modality():
    with pytest.raises(ValueError, match="at least one active modality"):
        RoutingState("s0", (), 0.5)


def test_ugapr_selects_nothing_without_a_gain_estimate():
    ugapr = UGAPR()
    state = RoutingState("s0", ("text_llm",), 0.9)
    gains = UnavailableHSIG().estimate_all(state, ["audio", "video"])
    result = ugapr.rank(state, gains, availability={"audio": None, "video": None})
    assert result.selection_available is False
    assert result.selected_modality is None
    assert "does not substitute a default" in result.reason


def test_ugapr_does_not_fall_back_to_the_cheapest_candidate():
    """A cost-only fallback would be a hard-coded order wearing a utility function."""
    from src.ahsef.costs import CostModel, ModalityCost

    costs = CostModel({
        "audio": ModalityCost("audio", 30.0, 9e8, 35335, 25664, "test", 10),
        "video": ModalityCost("video", 350.0, 9e10, 1772039, 55296, "test", 10),
    })
    ugapr = UGAPR(cost_model=costs)
    state = RoutingState("s0", ("text_llm",), 0.9)
    gains = UnavailableHSIG().estimate_all(state, ["audio", "video"])
    result = ugapr.rank(state, gains, availability={"audio": None, "video": None})
    assert result.selected_modality is None
    # Costs were still computed and reported -- only the selection is withheld.
    scores = result.to_dict()["candidate_scores"]
    assert scores["audio"]["normalized_cost"] < scores["video"]["normalized_cost"]


def test_ugapr_computes_utility_once_a_gain_estimate_exists():
    from src.ahsef.costs import CostModel, ModalityCost

    costs = CostModel({"audio": ModalityCost("audio", 10.0, 100.0, 1, 100, "test", 1)})
    ugapr = UGAPR(UtilityWeights(lambda_cost=0.1, mu_latency=0.1), cost_model=costs)
    state = RoutingState("s0", ("text_llm",), 0.9)
    gains = {"audio": GainEstimate("audio", 0.5, True, "stub estimator")}
    result = ugapr.rank(state, gains, availability={"audio": None})
    assert result.selection_available is True
    assert result.selected_modality == "audio"
    assert result.candidates[0].utility == pytest.approx(0.5 - 0.1 - 0.1)


def test_ugapr_respects_the_configured_floors():
    from src.ahsef.costs import CostModel, ModalityCost

    costs = CostModel({"audio": ModalityCost("audio", 10.0, 100.0, 1, 100, "test", 1)})
    ugapr = UGAPR(UtilityWeights(min_gain=0.9), cost_model=costs)
    state = RoutingState("s0", ("text_llm",), 0.9)
    gains = {"audio": GainEstimate("audio", 0.1, True, "stub")}
    result = ugapr.rank(state, gains, availability={"audio": None})
    assert result.selection_available is False
    assert "below the configured floors" in result.reason


def test_an_unavailable_candidate_gets_no_utility_even_with_a_gain():
    from src.ahsef.costs import CostModel, ModalityCost

    costs = CostModel({"audio": ModalityCost("audio", 10.0, 100.0, 1, 100, "test", 1)})
    ugapr = UGAPR(cost_model=costs)
    state = RoutingState("s0", ("text_llm",), 0.9)
    gains = {"audio": GainEstimate("audio", 0.9, True, "stub")}
    result = ugapr.rank(state, gains, availability={"audio": "no aligned record"})
    assert result.selected_modality is None
    assert result.candidates[0].utility is None


def test_ugapr_weights_are_configurable_and_recorded():
    record = UGAPR(UtilityWeights(lambda_cost=0.3, mu_latency=0.7)).provenance()
    assert record["weights"]["lambda_cost"] == 0.3
    assert record["weights"]["mu_latency"] == 0.7
    assert "J(m)" in record["weights"]["formula"]


# ------------------------------------------------------------------ trace

def test_the_trace_is_valid_and_terminates(router):
    for uncertainty in (0.2, 0.9):
        trace = route(router, uncertainty=uncertainty).trace
        assert trace.final.stop is True
        assert trace.modalities_activated == 1
        assert trace.to_dict()["router_saw_true_class"] is False


def test_a_request_trace_records_every_candidate_with_its_reason(router):
    trace = route(router, sample_id="s0", uncertainty=0.9).trace
    scored = {item.modality: item for item in trace.final.candidates}
    assert set(scored) == {"audio", "video", "image"}
    assert scored["audio"].available is True
    assert scored["video"].available is False
    assert scored["video"].unavailable_reason
    assert all(item.utility is None for item in scored.values())


def test_the_rendered_trace_shows_the_decision(router):
    rendered = route(router, uncertainty=0.2).trace.render()
    assert "Active      = [text_llm]" in rendered
    assert "Stopping decision = TRUE" in rendered


def test_the_flat_record_matches_the_specified_shape(router):
    record = route(router, uncertainty=0.9).flat_record()
    for key in ("sample_id", "active_modalities", "prediction", "uncertainty",
                "threshold", "stop", "reason"):
        assert key in record
    assert record["active_modalities"] == [LLM_MODALITY]
    assert record["stop"] is False
    assert record["router_saw_true_class"] is False


def test_the_policy_denies_that_anything_was_acquired(router):
    policy = router.policy()
    assert policy["acquisition_implemented"] is False
    assert "does not acquire or fuse" in policy["acquisition_note"]


# ------------------------------------------------------- prediction sets

def make_predictions(confidences):
    scores = [{name: 0.1 for name in CANONICAL_EMOTIONS} for _ in confidences]
    provider = ScriptedProvider([
        {"emotion": "happy", "class_scores": score, "confidence": value,
         "evidence_strength": 0.5, "ambiguity": 1.0 - value, "evidence": "c",
         "abstain": False}
        for score, value in zip(scores, confidences)
    ])
    modality = LLMTextModality(provider, uncertainty_policy="llm_confidence")
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(len(confidences))],
        "dataset": ["MELD"] * len(confidences),
        "text": [f"t{i}" for i in range(len(confidences))],
        "canonical_emotion_id": [1] * len(confidences),
    })
    return build_prediction_set(
        modality.predict_frame(frame), "validation", modality.provenance()
    )


def test_routing_a_prediction_set_splits_on_the_threshold(alignment):
    predictions = make_predictions([0.9, 0.9, 0.1, 0.1])
    router = LLMTextRouter(
        threshold=0.5, candidates=["audio"], alignment=alignment,
        uncertainty_policy="llm_confidence",
    )
    routings = router.route_prediction_set(predictions)
    summary = routing_summary(routings)
    assert summary["samples"] == 4
    assert summary["text_sufficient_stop"] == 2   # uncertainty 0.1 <= 0.5
    assert summary["text_insufficient_request"] == 2


def test_unusable_answers_are_excluded_from_routing_not_forced(alignment):
    provider = ScriptedProvider([
        {"emotion": "happy", "class_scores": {n: 0.1 for n in CANONICAL_EMOTIONS},
         "confidence": 0.9, "evidence_strength": 0.5, "ambiguity": 0.1,
         "evidence": "c", "abstain": False},
        "{malformed",
    ])
    modality = LLMTextModality(provider, uncertainty_policy="llm_confidence")
    frame = pd.DataFrame({
        "sample_id": ["s0", "s1"], "dataset": ["MELD"] * 2,
        "text": ["a", "b"], "canonical_emotion_id": [1, 1],
    })
    predictions = build_prediction_set(
        modality.predict_frame(frame), "validation", modality.provenance()
    )
    router = LLMTextRouter(
        threshold=0.5, candidates=["audio"], alignment=alignment,
        uncertainty_policy="llm_confidence",
    )
    routings = router.route_prediction_set(predictions)
    assert len(routings) == 1
    assert routings[0].sample_id == "s0"


def test_summary_states_that_one_modality_is_activated_by_construction(alignment):
    predictions = make_predictions([0.9, 0.1])
    router = LLMTextRouter(
        threshold=0.5, candidates=["audio"], alignment=alignment,
        uncertainty_policy="llm_confidence",
    )
    summary = routing_summary(router.route_prediction_set(predictions))
    assert summary["average_modalities_activated"] == 1.0
    assert "not a result about modality economy" in summary["average_modalities_note"]


def test_summary_reports_accuracy_after_routing(alignment):
    predictions = make_predictions([0.9, 0.1])
    router = LLMTextRouter(
        threshold=0.5, candidates=["audio"], alignment=alignment,
        uncertainty_policy="llm_confidence",
    )
    summary = routing_summary(router.route_prediction_set(predictions))
    assert summary["outcome"]["overall_accuracy"] == pytest.approx(1.0)
    assert "label-blind" in summary["outcome"]["note"]
