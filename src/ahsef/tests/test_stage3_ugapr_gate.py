"""UGAPR utility, cost/latency penalties, and the redesigned gate objective."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.costs import CostModel, ModalityCost
from src.ahsef.hsig import GainEstimate, RoutingState
from src.ahsef.stage3.objective import (
    GATE_OBJECTIVES,
    OBJECTIVE_RATIONALE,
    SELECTED_OBJECTIVE,
    ThresholdLeakageError,
    compare_objectives,
    routed_predictions,
    score_policy,
    select_gate_threshold,
    sensitivity_to_penalties,
    sweep_utility_threshold,
)
from src.ahsef.ugapr import UGAPR, UtilityWeights
from src.common.labels import CANONICAL_EMOTION_CLASSES


def cost_model() -> CostModel:
    """A registry where audio is cheap and the LLM is not."""
    return CostModel({
        "audio": ModalityCost("audio", 12.0, 1e8, 100_000, 1000, "synthetic", 10),
        "text_llm": ModalityCost("text_llm", 4000.0, 2e13, 32_700_000_000, 700,
                                 "synthetic", 10),
    })


def state(sample_id="s0", uncertainty=0.5) -> RoutingState:
    return RoutingState(
        sample_id=sample_id, active_modalities=("text_llm",), uncertainty=uncertainty,
        confidence=1 - uncertainty, predicted_class=0, features={},
    )


# ------------------------------------------------------------------- UGAPR

def test_utility_subtracts_cost_and_latency():
    model = cost_model()
    ugapr = UGAPR(UtilityWeights(lambda_cost=0.2, mu_latency=0.3), cost_model=model)
    gains = {"audio": GainEstimate("audio", 0.5, True, "test")}
    result = ugapr.rank(state(), gains, availability={"audio": None})
    entry = result.candidates[0]
    expected = 0.5 - 0.2 * entry.normalized_cost - 0.3 * entry.normalized_latency
    assert entry.utility == pytest.approx(expected)
    assert entry.utility < 0.5, "the penalty must actually reduce the utility"


def test_audio_is_cheaper_than_the_llm_on_both_axes():
    model = cost_model()
    assert model.normalized_cost("audio") < model.normalized_cost("text_llm")
    assert model.normalized_latency("audio") < model.normalized_latency("text_llm")


def test_no_gain_estimate_means_no_utility_and_no_selection():
    ugapr = UGAPR(UtilityWeights(), cost_model=cost_model())
    gains = {"audio": GainEstimate("audio", None, False, "no estimator")}
    result = ugapr.rank(state(), gains, availability={"audio": None})
    assert result.candidates[0].utility is None
    assert result.selection_available is False
    assert "does not substitute a default" in result.reason


def test_unavailable_candidate_is_not_scored():
    ugapr = UGAPR(UtilityWeights(), cost_model=cost_model())
    gains = {"audio": GainEstimate("audio", 0.9, True, "test")}
    result = ugapr.rank(state(), gains, availability={"audio": "no audio for this id"})
    assert result.candidates[0].utility is None
    assert result.selection_available is False


def test_a_large_penalty_can_make_acquisition_not_worth_it():
    ugapr = UGAPR(
        UtilityWeights(lambda_cost=10.0, mu_latency=10.0, min_utility=0.0),
        cost_model=cost_model(),
    )
    gains = {"audio": GainEstimate("audio", 0.05, True, "test")}
    result = ugapr.rank(state(), gains, availability={"audio": None})
    assert result.selection_available is False
    assert "below the configured floors" in result.reason


# ------------------------------------------------- routed system scoring

def test_routed_predictions_pick_the_right_source():
    text = np.array([0, 1, 2])
    fused = np.array([3, 4, 5])
    acquire = np.array([False, True, False])
    np.testing.assert_array_equal(routed_predictions(text, fused, acquire), [0, 4, 2])


def test_score_policy_counts_modalities_and_latency():
    text = np.array([0, 0, 0, 0])
    fused = np.array([1, 1, 1, 1])
    truth = np.array([1, 1, 0, 0])
    outcome = score_policy(
        text, fused, np.array([True, True, False, False]), truth, 0.0,
        CANONICAL_EMOTION_CLASSES, text_latency_ms=100.0, audio_latency_ms=100.0,
    )
    assert outcome.acquisition_rate == 0.5
    assert outcome.mean_modalities == 1.5
    assert outcome.accuracy == 1.0
    assert outcome.normalized_latency == pytest.approx(0.75)


def test_never_and_always_acquire_are_both_in_the_sweep():
    text = np.array([0, 0, 1, 1])
    fused = np.array([1, 1, 0, 0])
    truth = np.array([0, 0, 1, 1])
    sweep = sweep_utility_threshold(
        np.array([0.1, 0.2, 0.3, 0.4]), text, fused, truth,
        CANONICAL_EMOTION_CLASSES, 100.0, 100.0,
    )
    rates = {point.acquisition_rate for point in sweep}
    assert 0.0 in rates and 1.0 in rates


def test_sweep_refuses_nan_utility():
    with pytest.raises(ValueError, match="NaN"):
        sweep_utility_threshold(
            np.array([0.1, np.nan]), np.array([0, 0]), np.array([1, 1]),
            np.array([0, 1]), CANONICAL_EMOTION_CLASSES, 1.0, 1.0,
        )


# ------------------------------------------------------ objective selection

def test_threshold_selection_refuses_test():
    with pytest.raises(ThresholdLeakageError, match="only be selected on"):
        select_gate_threshold(
            np.array([0.1, 0.2]), np.array([0, 0]), np.array([1, 1]), np.array([0, 1]),
            split="test", class_names=CANONICAL_EMOTION_CLASSES,
            text_latency_ms=1.0, audio_latency_ms=1.0,
        )


def test_selected_objective_is_declared_and_per_class_aware():
    assert SELECTED_OBJECTIVE == "macro_f1_acquisition_latency"
    assert SELECTED_OBJECTIVE in GATE_OBJECTIVES
    assert OBJECTIVE_RATIONALE["selected"] == SELECTED_OBJECTIVE
    assert "macro-F1" in OBJECTIVE_RATIONALE["why"]
    assert "declared" in OBJECTIVE_RATIONALE


def test_stage2_objective_is_not_reused():
    assert "target_stop_accuracy" not in GATE_OBJECTIVES


def test_acquisition_penalty_reduces_the_acquisition_rate():
    """A higher alpha must never make the policy acquire more."""
    rng = np.random.default_rng(0)
    n = 200
    truth = rng.integers(0, 3, n)
    text_wrong = rng.random(n) > 0.5
    fused_wrong = rng.random(n) > 0.8
    text = np.where(text_wrong, (truth + 1) % 7, truth)
    fused = np.where(fused_wrong, (truth + 2) % 7, truth)
    utility = rng.random(n)

    rates = []
    for alpha in (0.0, 0.5, 5.0):
        selection = select_gate_threshold(
            utility, text, fused, truth, split="validation",
            class_names=CANONICAL_EMOTION_CLASSES, text_latency_ms=100.0,
            audio_latency_ms=100.0, alpha=alpha, beta=0.0,
        )
        rates.append(selection.selected.acquisition_rate)
    assert rates[0] >= rates[-1]
    assert rates[-1] == pytest.approx(0.0), "a huge penalty must stop acquiring"


def test_comparison_reports_every_candidate_objective():
    text = np.array([0, 0, 1, 1, 2, 2])
    fused = np.array([1, 1, 1, 1, 2, 2])
    truth = np.array([1, 1, 1, 1, 2, 2])
    sweep = sweep_utility_threshold(
        np.linspace(0, 1, 6), text, fused, truth, CANONICAL_EMOTION_CLASSES, 10.0, 10.0
    )
    comparison = compare_objectives(sweep, alpha=0.05, beta=0.02)
    assert set(comparison["candidates"]) == set(GATE_OBJECTIVES)
    assert comparison["selected"] == SELECTED_OBJECTIVE
    assert comparison["candidates"][SELECTED_OBJECTIVE]["selected_for_stage3"] is True
    assert comparison["selection_was_declared_in_advance"] is True


def test_selection_record_states_it_used_no_test_labels():
    text = np.array([0, 0, 1, 1])
    fused = np.array([1, 1, 1, 1])
    truth = np.array([1, 1, 1, 1])
    record = select_gate_threshold(
        np.array([0.1, 0.4, 0.6, 0.9]), text, fused, truth, split="validation",
        class_names=CANONICAL_EMOTION_CLASSES, text_latency_ms=10.0,
        audio_latency_ms=10.0,
    ).to_dict()
    assert record["uses_test_labels"] is False
    assert record["selected_on_split"] == "validation"
    assert "macro_f1" in record["objective_formula"]


def test_sensitivity_table_covers_the_penalty_grid():
    text = np.array([0, 0, 1, 1])
    fused = np.array([1, 1, 0, 0])
    truth = np.array([1, 1, 1, 1])
    sweep = sweep_utility_threshold(
        np.array([0.1, 0.4, 0.6, 0.9]), text, fused, truth,
        CANONICAL_EMOTION_CLASSES, 10.0, 10.0,
    )
    rows = sensitivity_to_penalties(sweep)
    assert len(rows) == 15
    assert {row["alpha"] for row in rows} == {0.0, 0.02, 0.05, 0.10, 0.20}
