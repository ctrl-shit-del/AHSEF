"""The sufficiency gate: threshold selection, leakage guards, and decisions."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.gate import (
    THRESHOLD_OBJECTIVES,
    GateDecision,
    ThresholdLeakageError,
    apply_gate,
    gate_summary,
    select_threshold,
    sweep_thresholds,
)


def separable(n: int = 400):
    """Low uncertainty is usually right, high uncertainty usually wrong."""
    uncertainty = np.linspace(0.0, 1.0, n)
    correct = uncertainty < 0.5
    return uncertainty, correct


# ------------------------------------------------------------------- sweep

def test_sweep_covers_every_distinguishable_threshold():
    uncertainty, correct = separable(50)
    points = sweep_thresholds(uncertainty, correct)
    assert len(points) == len(set(uncertainty))
    assert points[0].threshold <= points[-1].threshold


def test_coverage_rises_monotonically_with_the_threshold():
    uncertainty, correct = separable(100)
    coverages = [point.coverage for point in sweep_thresholds(uncertainty, correct)]
    assert coverages == sorted(coverages)


def test_stopped_and_routed_counts_always_partition_the_split():
    uncertainty, correct = separable(60)
    for point in sweep_thresholds(uncertainty, correct):
        assert point.stopped + point.routed == 60


def test_nan_uncertainty_is_refused_rather_than_ordered():
    with pytest.raises(ValueError, match="NaN"):
        sweep_thresholds([0.1, float("nan")], [True, False])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="entries"):
        sweep_thresholds([0.1, 0.2], [True])


def test_an_empty_split_is_refused():
    with pytest.raises(ValueError, match="empty split"):
        sweep_thresholds([], [])


# --------------------------------------------------------- leakage guards

def test_a_threshold_cannot_be_selected_on_test():
    uncertainty, correct = separable()
    with pytest.raises(ThresholdLeakageError, match="refusing to select on 'test'"):
        select_threshold(uncertainty, correct, split="test")


def test_the_refusal_explains_why_it_matters():
    uncertainty, correct = separable()
    with pytest.raises(ThresholdLeakageError, match="invalid"):
        select_threshold(uncertainty, correct, split="test")


def test_validation_and_train_are_permitted():
    uncertainty, correct = separable()
    for split in ("validation", "train"):
        assert select_threshold(uncertainty, correct, split=split).selected_on_split == split


def test_the_selection_record_declares_no_test_labels_were_used():
    uncertainty, correct = separable()
    record = select_threshold(uncertainty, correct, "validation").to_dict()
    assert record["uses_test_labels"] is False
    assert record["selected_on_split"] == "validation"


# ---------------------------------------------------------- objectives

def test_target_stop_accuracy_takes_the_widest_coverage_that_clears_the_floor():
    uncertainty, correct = separable(400)
    selection = select_threshold(
        uncertainty, correct, "validation",
        objective="target_stop_accuracy", target_stop_accuracy=0.95,
    )
    assert selection.satisfied
    assert selection.selected_point.stop_accuracy >= 0.95
    # Anything wider would drop below the floor.
    wider = [
        point for point in selection.sweep
        if point.coverage > selection.selected_point.coverage
    ]
    assert all(point.stop_accuracy < 0.95 for point in wider if point.stop_accuracy)


def test_an_unreachable_target_is_reported_unsatisfied_not_silently_met():
    """The gate must not promise an accuracy the evidence cannot support."""
    uncertainty = np.linspace(0, 1, 100)
    correct = np.zeros(100, dtype=bool)  # always wrong
    selection = select_threshold(
        uncertainty, correct, "validation", target_stop_accuracy=0.9,
    )
    assert selection.satisfied is False
    assert "UNSATISFIED" in selection.note


def test_max_separation_maximises_the_accuracy_gap():
    uncertainty, correct = separable(400)
    selection = select_threshold(
        uncertainty, correct, "validation", objective="max_separation",
    )
    assert selection.satisfied
    best = selection.selected_point.separation
    eligible = [
        point.separation for point in selection.sweep
        if point.separation is not None and point.coverage >= 0.05 and point.routed > 0
    ]
    assert best == pytest.approx(max(eligible))


def test_quantile_objective_never_consults_correctness():
    uncertainty = np.linspace(0, 1, 100)
    right = select_threshold(uncertainty, np.ones(100, bool), "validation",
                             objective="quantile", target_coverage=0.5)
    wrong = select_threshold(uncertainty, np.zeros(100, bool), "validation",
                             objective="quantile", target_coverage=0.5)
    assert right.threshold == wrong.threshold
    assert "correctness was not consulted" in right.note


def test_min_coverage_is_respected():
    uncertainty, correct = separable(400)
    selection = select_threshold(
        uncertainty, correct, "validation", target_stop_accuracy=1.0, min_coverage=0.3,
    )
    if selection.satisfied:
        assert selection.selected_point.coverage >= 0.3


def test_an_unknown_objective_is_refused():
    uncertainty, correct = separable()
    with pytest.raises(ValueError, match="objective must be one of"):
        select_threshold(uncertainty, correct, "validation", objective="vibes")


def test_every_declared_objective_runs():
    uncertainty, correct = separable(200)
    for objective in THRESHOLD_OBJECTIVES:
        selection = select_threshold(uncertainty, correct, "validation", objective=objective)
        assert 0.0 <= selection.threshold <= 1.0


# ------------------------------------------------------------- decisions

def test_the_rule_is_stop_when_at_or_below_tau():
    assert apply_gate("s", 0.4, 0.5).stop is True
    assert apply_gate("s", 0.5, 0.5).stop is True   # boundary is inclusive
    assert apply_gate("s", 0.6, 0.5).stop is False


def test_decision_names_are_stable():
    assert apply_gate("s", 0.1, 0.5).decision == GateDecision.STOP
    assert apply_gate("s", 0.9, 0.5).decision == GateDecision.REQUEST


def test_the_reason_states_both_numbers():
    outcome = apply_gate("s", 0.9, 0.5)
    assert "0.9000" in outcome.reason and "0.5000" in outcome.reason
    assert "insufficient" in outcome.reason


def test_a_sample_with_no_uncertainty_cannot_be_gated():
    """Defaulting it either way would invent a decision."""
    with pytest.raises(ValueError, match="cannot be gated"):
        apply_gate("s", float("nan"), 0.5)


def test_gate_never_receives_a_label():
    import inspect

    signature = inspect.signature(apply_gate)
    assert set(signature.parameters) == {"sample_id", "uncertainty", "threshold"}


# --------------------------------------------------------------- summary

def test_summary_counts_both_sides():
    outcomes = [apply_gate(f"s{i}", value, 0.5) for i, value in enumerate([0.1, 0.2, 0.8, 0.9])]
    record = gate_summary(outcomes)
    assert record["stopped"] == 2
    assert record["requested_additional_modality"] == 2
    assert record["stop_rate"] == pytest.approx(0.5)


def test_summary_reports_accuracy_either_side_when_outcomes_are_supplied():
    outcomes = [apply_gate(f"s{i}", value, 0.5) for i, value in enumerate([0.1, 0.2, 0.8, 0.9])]
    record = gate_summary(outcomes, correct=[True, True, False, False])
    assert record["outcome"]["accuracy_when_stopped"] == pytest.approx(1.0)
    assert record["outcome"]["accuracy_when_routed"] == pytest.approx(0.0)
    assert "never sees a label" in record["outcome"]["note"]


def test_summary_without_outcomes_reports_no_accuracy():
    outcomes = [apply_gate("s", 0.1, 0.5)]
    assert "outcome" not in gate_summary(outcomes)


def test_summary_refuses_misaligned_outcomes():
    outcomes = [apply_gate("s", 0.1, 0.5)]
    with pytest.raises(ValueError, match="must align"):
        gate_summary(outcomes, correct=[True, False])
