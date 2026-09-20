"""Whether dU deserves to be HSIG's target -- measured, not assumed."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ahsef.gain_diagnostics import diagnose, rank_agreement, render, verdict


def frame(delta, before, after, truth):
    """A dU record table with explicit before/after predictions."""
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(len(delta))],
        "delta_uncertainty": delta,
        "predicted_before": before,
        "predicted_after": after,
        "true_class": truth,
    })


def test_a_faithful_signal_is_reported_as_usable():
    """dU positive exactly where the acquisition fixed the prediction."""
    record = diagnose(
        frame(delta=[0.4, 0.4, -0.4, -0.4], before=[0, 0, 1, 1],
              after=[1, 1, 0, 0], truth=[1, 1, 1, 1]),
        "faithful",
    )
    assert record["correlation_delta_vs_improvement"] == pytest.approx(1.0)
    assert record["sign_agrees_with_accuracy_gain"] is True
    assert verdict([record])["delta_uncertainty_is_a_usable_hsig_target"] is True


def test_a_signal_whose_sign_contradicts_the_gain_is_reported_as_unusable():
    """Accuracy improves while dU is uniformly negative -- the text+video case."""
    record = diagnose(
        frame(delta=[-0.2, -0.2, -0.2, -0.2], before=[0, 0, 0, 1],
              after=[1, 1, 1, 1], truth=[1, 1, 1, 1]),
        "contradictory",
    )
    assert record["accuracy_gain"] > 0
    assert record["mean_delta_uncertainty"] < 0
    assert record["sign_agrees_with_accuracy_gain"] is False
    summary = verdict([record])
    assert summary["delta_uncertainty_is_a_usable_hsig_target"] is False
    assert summary["pairs_where_delta_sign_contradicts_accuracy_gain"] == ["contradictory"]
    assert "Do not train HSIG to predict dU" in summary["recommendation"]


def test_an_uninformative_signal_is_reported_as_unusable():
    record = diagnose(
        frame(delta=[0.1, -0.1, 0.1, -0.1], before=[0, 0, 1, 1],
              after=[1, 1, 1, 1], truth=[1, 1, 1, 1]),
        "noise",
    )
    assert abs(record["correlation_delta_vs_improvement"]) < 0.3
    assert verdict([record])["delta_uncertainty_is_a_usable_hsig_target"] is False


def test_a_constant_delta_reports_no_correlation_rather_than_zero():
    """A pair where dU never varies is informative, not a zero correlation."""
    record = diagnose(
        frame(delta=[-0.2] * 4, before=[0, 0, 1, 1], after=[1, 1, 1, 1],
              truth=[1, 1, 1, 1]),
        "constant",
    )
    assert record["correlation_delta_vs_improvement"] is None


def test_fixed_and_broken_counts_are_separated():
    record = diagnose(
        frame(delta=[0.1, 0.1, 0.1], before=[0, 1, 1], after=[1, 0, 1],
              truth=[1, 1, 1]),
        "mixed",
    )
    assert record["fixed"] == 1
    assert record["broken"] == 1
    assert record["accuracy_before"] == pytest.approx(2 / 3)
    assert record["accuracy_after"] == pytest.approx(2 / 3)


def test_ranking_disagreement_is_surfaced():
    """dU prefers one candidate, realised accuracy prefers the other."""
    rows = [
        {"name": "audio+text", "mean_delta_uncertainty": 0.05, "accuracy_gain": 0.01},
        {"name": "audio+video", "mean_delta_uncertainty": -0.20, "accuracy_gain": 0.10},
    ]
    record = rank_agreement(rows)
    assert record["ranked_by_mean_delta_uncertainty"] == ["audio+text", "audio+video"]
    assert record["ranked_by_realised_accuracy_gain"] == ["audio+video", "audio+text"]
    assert record["rankings_agree"] is False
    assert record["top_choice_agrees"] is False


def test_ranking_needs_two_candidates():
    record = rank_agreement([{"name": "a", "mean_delta_uncertainty": 0.0, "accuracy_gain": 0.0}])
    assert record["comparable"] is False


def test_render_produces_one_row_per_pair():
    rows = [
        diagnose(frame([0.1, -0.1], [0, 1], [1, 1], [1, 1]), "a"),
        diagnose(frame([0.2, -0.2], [0, 1], [1, 1], [1, 1]), "b"),
    ]
    lines = render(rows).splitlines()
    assert len(lines) == 4  # header, rule, two rows
    assert lines[2].startswith("a")
    assert lines[3].startswith("b")
