"""The Text -> Audio router: decisions, fusion, unavailability, and auditability."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.costs import CostModel, ModalityCost
from src.ahsef.fusion import FusionSpec
from src.ahsef.hsig import GainEstimate, RoutingState
from src.ahsef.routing_log import STOP_REASONS, RoutingTrace
from src.ahsef.stage3.features import build_features, states_from_features
from src.ahsef.stage3.hsig_model import Stage3HSIG, targets_from_oracle
from src.ahsef.stage3.oracle import build_oracle_table, fuse
from src.ahsef.stage3.router import (
    Availability,
    AudioAvailability,
    DecisionKind,
    TextAudioRouter,
    attach_truth,
    decision_summary,
    decisions_frame,
)
from src.ahsef.tests.stage3_fixtures import scenario
from src.ahsef.ugapr import UGAPR, UtilityWeights


class ConstantHSIG:
    """A stand-in that returns a fixed gain, so threshold behaviour is isolated."""

    name = "constant"

    def __init__(self, value):
        self.value = value

    def estimate_gain(self, state, candidate_modality):
        if self.value is None:
            return GainEstimate(candidate_modality, None, False, "no estimate")
        return GainEstimate(candidate_modality, float(self.value), True, "constant")

    def estimate_all(self, state, candidates):
        return {name: self.estimate_gain(state, name) for name in candidates}

    def provenance(self):
        return {"hsig": self.name, "value": self.value}


def cost_model() -> CostModel:
    return CostModel({
        "audio": ModalityCost("audio", 12.0, 1e8, 100_000, 1000, "synthetic", 10),
        "text_llm": ModalityCost("text_llm", 4000.0, 2e13, 32_700_000_000, 700,
                                 "synthetic", 10),
    })


@pytest.fixture
def material():
    data = scenario(n=60, seed=5)
    spec = FusionSpec(
        method="weighted_probability", weights={"text_llm": 0.5, "audio": 0.5},
        selected_on_split="validation",
    )
    fused = fuse(data["text"], data["audio"], spec, data["ids"])
    oracle = build_oracle_table(data["text"], data["audio"], fused, data["ids"])
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(data["text"].frame, uncertainty)
    return {
        "data": data, "fused": fused, "oracle": oracle, "features": features,
        "states": states_from_features(features),
        "targets": targets_from_oracle(oracle, "validation"),
    }


def build_router(material, hsig, threshold, availability=None):
    return TextAudioRouter(
        hsig=hsig,
        ugapr=UGAPR(
            UtilityWeights(lambda_cost=0.1, mu_latency=0.1, min_gain=-1.0,
                           min_utility=-2.0),
            cost_model=cost_model(),
        ),
        threshold=threshold,
        availability=availability or AudioAvailability(material["data"]["audio"]),
        audio=material["data"]["audio"], fused=material["fused"],
        cost_model=cost_model(), text_latency_ms=4000.0, audio_latency_ms=12.0,
        split="validation",
    )


# ------------------------------------------------------------ stop / request

def test_high_gain_requests_audio_and_fuses(material):
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    decision, trace = router.route(material["states"][0])
    assert decision.decision == DecisionKind.REQUEST
    assert decision.availability == Availability.AVAILABLE
    assert decision.requested_modality == "audio"
    assert decision.audio_prediction is not None
    assert decision.fused_prediction is not None
    assert decision.final_prediction == decision.fused_prediction
    assert decision.modalities_activated == 2
    assert decision.routing_step == 2
    assert len(trace.steps) == 2


def test_low_gain_stops_without_acquiring(material):
    router = build_router(material, ConstantHSIG(-0.5), threshold=0.0)
    decision, trace = router.route(material["states"][0])
    assert decision.decision == DecisionKind.STOP
    assert decision.availability == Availability.NOT_REQUESTED
    assert decision.audio_prediction is None
    assert decision.fused_prediction is None
    assert decision.final_prediction == decision.initial_prediction
    assert decision.modalities_activated == 1
    assert decision.stop_reason == "no_candidate_above_min_utility"
    assert len(trace.steps) == 1


def test_the_decision_is_a_threshold_on_utility_not_on_uncertainty(material):
    """Two samples with the same gain decide identically however uncertain they are."""
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    states = material["states"]
    uncertainties = [state.uncertainty for state in states]
    low = states[int(np.argmin(uncertainties))]
    high = states[int(np.argmax(uncertainties))]
    assert router.route(low)[0].decision == router.route(high)[0].decision


def test_audio_cost_is_normalised_over_the_registry_not_the_candidate_set(material):
    """Normalising over the one-member candidate set would force 1.0 by construction."""
    router = build_router(material, ConstantHSIG(0.5), threshold=0.0)
    decision, _ = router.route(material["states"][0])
    assert decision.normalized_cost < 1.0
    assert decision.normalized_latency < 1.0
    # Audio really is far cheaper than the LLM on both axes, so the penalty at
    # the configured lambda/mu is small. That is a measurement, not a bug, and
    # the report has to say so rather than implying the cost term is decisive.
    assert decision.ugapr_utility == pytest.approx(
        0.5 - 0.1 * decision.normalized_cost - 0.1 * decision.normalized_latency
    )


def test_cost_penalty_can_flip_a_positive_gain_to_stop(material):
    """A gain that does not cover its price must not be acquired."""
    penalty_free = TextAudioRouter(
        hsig=ConstantHSIG(0.05),
        ugapr=UGAPR(
            UtilityWeights(lambda_cost=0.0, mu_latency=0.0, min_gain=-1.0,
                           min_utility=-2.0),
            cost_model=cost_model(),
        ),
        threshold=0.03, availability=AudioAvailability(material["data"]["audio"]),
        audio=material["data"]["audio"], fused=material["fused"],
        cost_model=cost_model(), split="validation",
    )
    expensive = TextAudioRouter(
        hsig=ConstantHSIG(0.05),
        ugapr=UGAPR(
            UtilityWeights(lambda_cost=20.0, mu_latency=20.0, min_gain=-1.0,
                           min_utility=-2.0),
            cost_model=cost_model(),
        ),
        threshold=0.03, availability=AudioAvailability(material["data"]["audio"]),
        audio=material["data"]["audio"], fused=material["fused"],
        cost_model=cost_model(), split="validation",
    )
    state = material["states"][0]
    assert penalty_free.route(state)[0].decision == DecisionKind.REQUEST
    assert expensive.route(state)[0].decision == DecisionKind.STOP


def test_no_gain_estimate_means_stop_not_a_guess(material):
    router = build_router(material, ConstantHSIG(None), threshold=0.0)
    decision, _ = router.route(material["states"][0])
    assert decision.decision == DecisionKind.STOP
    assert decision.hsig_predicted_gain is None
    assert decision.ugapr_utility is None
    assert decision.stop_reason == "no_candidate_available"
    assert "does not acquire on an undefined utility" in decision.reason


# --------------------------------------------------------------- unavailable

def test_requested_but_unavailable_fabricates_nothing(material):
    """The core integrity rule: a failed acquisition invents no prediction."""
    router = build_router(
        material, ConstantHSIG(0.9), threshold=0.0,
        availability=AudioAvailability(
            available_ids=[], unavailable_reason="no audio for this corpus"
        ),
    )
    decision, trace = router.route(material["states"][0])
    assert decision.decision == DecisionKind.REQUEST
    assert decision.availability == Availability.UNAVAILABLE
    assert decision.audio_prediction is None
    assert decision.fused_prediction is None
    assert decision.final_prediction == decision.initial_prediction
    assert decision.final_uncertainty == decision.initial_uncertainty
    assert decision.stop_reason == "requested_modality_unavailable"
    assert "REQUESTED_BUT_UNAVAILABLE" in decision.reason
    assert len(trace.steps) == 1


def test_unavailable_is_distinct_from_never_requested(material):
    unavailable = build_router(
        material, ConstantHSIG(0.9), threshold=0.0,
        availability=AudioAvailability(available_ids=[]),
    ).route(material["states"][0])[0]
    stopped = build_router(material, ConstantHSIG(-0.9), threshold=0.0).route(
        material["states"][0]
    )[0]
    assert unavailable.availability != stopped.availability
    assert unavailable.stop_reason != stopped.stop_reason


def test_unavailable_prediction_is_not_zero_uniform_or_neutral(material):
    """Guard the specific fabrications the brief forbids."""
    data = material["data"]
    non_neutral = next(
        index for index, value in enumerate(data["text"].frame["predicted_class"])
        if int(value) != 0
    )
    router = build_router(
        material, ConstantHSIG(0.9), threshold=0.0,
        availability=AudioAvailability(available_ids=[]),
    )
    decision, _ = router.route(material["states"][non_neutral])
    record = decision.to_dict()
    assert record["final_prediction"] == record["initial_prediction"] != 0
    assert record["fused_prediction"] is None
    assert record["audio_prediction"] is None


def test_stop_reason_is_a_declared_value(material):
    assert "requested_modality_unavailable" in STOP_REASONS
    for hsig, threshold in ((ConstantHSIG(0.9), 0.0), (ConstantHSIG(-0.9), 0.0)):
        decision, _ = build_router(material, hsig, threshold).route(material["states"][0])
        assert decision.stop_reason in STOP_REASONS


# ------------------------------------------------------- fusion + uncertainty

def test_fused_prediction_comes_from_the_fused_set_by_id(material):
    """Rows must be looked up by sample id, never by position."""
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    shuffled = list(reversed(material["states"]))
    for state in shuffled[:5]:
        decision, _ = router.route(state)
        row = material["fused"].frame.set_index(
            material["fused"].frame["sample_id"].astype(str)
        ).loc[state.sample_id]
        assert decision.fused_prediction == int(row["predicted_class"])
        assert decision.final_uncertainty == pytest.approx(float(row["normalized_entropy"]))


def test_uncertainty_is_updated_after_acquisition(material):
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    changed = 0
    for state in material["states"][:20]:
        decision, _ = router.route(state)
        if decision.final_uncertainty != decision.initial_uncertainty:
            changed += 1
    assert changed > 0, "fusion must actually move the uncertainty"


def test_latency_accumulates_only_when_audio_is_acquired(material):
    acquire = build_router(material, ConstantHSIG(0.9), 0.0).route(material["states"][0])[0]
    stop = build_router(material, ConstantHSIG(-0.9), 0.0).route(material["states"][0])[0]
    assert acquire.latency_ms == pytest.approx(4012.0)
    assert stop.latency_ms == pytest.approx(4000.0)


# ------------------------------------------------------------- auditability

def test_the_router_never_receives_a_label(material):
    """Structural: there is no parameter through which a label could arrive."""
    import inspect

    signature = inspect.signature(TextAudioRouter.route)
    assert list(signature.parameters) == ["self", "state"]
    for state in material["states"][:3]:
        assert not hasattr(state, "true_class")
        assert "true_class" not in state.features


def test_decision_record_carries_every_required_field(material):
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    record = router.route(material["states"][0])[0].to_dict()
    for field in (
        "sample_id", "initial_prediction", "initial_uncertainty",
        "hsig_predicted_gain", "ugapr_utility", "decision", "requested_modality",
        "availability", "audio_prediction", "fused_prediction", "final_uncertainty",
        "final_correctness", "routing_step", "timestamp", "provenance",
    ):
        assert field in record, field
    assert record["router_saw_true_class"] is False


def test_truth_is_attached_only_after_every_decision(material):
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    decisions, _ = router.route_all(material["states"])
    assert all(decision.true_class is None for decision in decisions)
    assert all(decision.final_correctness is None for decision in decisions)

    labels = dict(zip(
        material["oracle"]["sample_id"].astype(str),
        material["oracle"]["true_class"].astype(int),
    ))
    attach_truth(decisions, labels)
    assert all(decision.true_class is not None for decision in decisions)
    assert all(decision.final_correctness is not None for decision in decisions)


def test_trace_is_a_valid_stage1_routing_trace(material):
    router = build_router(material, ConstantHSIG(0.9), threshold=0.0)
    _, trace = router.route(material["states"][0])
    assert isinstance(trace, RoutingTrace)
    record = trace.to_dict()
    assert record["router_saw_true_class"] is False
    assert record["modalities_activated"] == 2
    assert record["acquisitions"] == ["audio"]
    assert "audio" in trace.render()


def test_multi_step_trace_records_hsig_and_ugapr(material):
    router = build_router(material, ConstantHSIG(0.42), threshold=0.0)
    _, trace = router.route(material["states"][0])
    first = trace.steps[0]
    assert first.hsig["audio"] == pytest.approx(0.42)
    assert first.ugapr["audio"]["utility"] is not None
    assert first.selected_modality == "audio"
    assert trace.steps[1].active_modalities == ["text_llm", "audio"]


def test_deterministic_replay(material):
    """Same inputs, same decisions -- twice, from two router instances."""
    first = build_router(material, ConstantHSIG(0.2), threshold=0.05)
    second = build_router(material, ConstantHSIG(0.2), threshold=0.05)
    left, _ = first.route_all(material["states"])
    right, _ = second.route_all(material["states"])
    for a, b in zip(left, right):
        assert a.decision == b.decision
        assert a.final_prediction == b.final_prediction
        assert a.ugapr_utility == pytest.approx(b.ugapr_utility)


def test_router_with_a_fitted_hsig_produces_a_mixed_policy(material):
    """End to end: a real estimator must not collapse to always or never."""
    model = Stage3HSIG.fit(material["features"], material["targets"])
    router = build_router(material, model, threshold=0.0)
    decisions, _ = router.route_all(material["states"])
    frame = decisions_frame(decisions)
    rate = float((frame["decision"] == "REQUEST_AUDIO").mean())
    assert 0.0 < rate < 1.0, f"policy collapsed to a constant (rate={rate})"


def test_summary_separates_requested_from_acquired(material):
    router = build_router(
        material, ConstantHSIG(0.9), threshold=0.0,
        availability=AudioAvailability(available_ids=[]),
    )
    decisions, _ = router.route_all(material["states"])
    summary = decision_summary(decisions)
    assert summary["request_audio"] == len(decisions)
    assert summary["audio_acquired"] == 0
    assert summary["requested_but_unavailable"] == len(decisions)
    assert summary["average_modalities_activated"] == 1.0


def test_provenance_declares_the_single_candidate_limitation(material):
    record = build_router(material, ConstantHSIG(0.1), 0.0).provenance()
    assert record["candidates"] == ["audio"]
    assert record["router_saw_true_class"] is False
    assert "not selected because it is the only option" in \
        record["single_candidate_limitation"]


def test_availability_requires_explicit_evidence():
    with pytest.raises(ValueError, match="untestable"):
        AudioAvailability()
