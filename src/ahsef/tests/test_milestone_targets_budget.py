"""PHASE D / E: routing targets defined in closed form, and the budget curve."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.milestone.budget import (
    DEFAULT_BUDGETS,
    BudgetError,
    PoolOutcomes,
    acquire_top_k,
    budget_curve,
    compare_policies,
    efficiency_summary,
    oracle_scores,
    random_curve,
    score_at_budget,
)
from src.ahsef.milestone.targets import (
    REJECTED_BY_STAGE1,
    TARGET_DEFINITIONS,
    TARGET_NAMES,
    RoutingOutcomes,
    TargetError,
    build_all_targets,
    build_target,
    class_weights_from_labels,
    describe_targets,
)
from src.common.labels import CANONICAL_EMOTION_CLASSES


def outcomes(text, fused, truth, text_u=None, fused_u=None) -> RoutingOutcomes:
    n = len(truth)
    return RoutingOutcomes(
        sample_ids=[f"s{i}" for i in range(n)],
        true_class=np.asarray(truth, dtype=int),
        text_prediction=np.asarray(text, dtype=int),
        fused_prediction=np.asarray(fused, dtype=int),
        text_uncertainty=np.asarray(
            text_u if text_u is not None else [0.5] * n, dtype=float
        ),
        fused_uncertainty=np.asarray(
            fused_u if fused_u is not None else [0.4] * n, dtype=float
        ),
    )


# ============================================================
# Targets
# ============================================================

def test_every_declared_target_is_defined_and_implemented():
    item = outcomes([0, 1], [1, 1], [1, 1])
    assert set(TARGET_DEFINITIONS) == set(TARGET_NAMES)
    built = build_all_targets(item)
    assert set(built) == set(TARGET_NAMES)
    for name, values in built.items():
        assert values.shape == (2,), name


def test_an_unknown_target_is_refused():
    with pytest.raises(TargetError, match="Unknown routing target"):
        build_target("whatever_wins", outcomes([0], [0], [0]))


def test_binary_correction_is_only_the_fix_event():
    #        fix   already right   harm   both wrong
    text =  [0,    1,              1,     0]
    fused = [1,    1,              0,     2]
    truth = [1,    1,              1,     3]
    values = build_target("binary_correction", outcomes(text, fused, truth))
    assert list(values) == [1.0, 0.0, 0.0, 0.0]


def test_signed_gain_expresses_harm_as_negative():
    text =  [0, 1, 1, 0]
    fused = [1, 1, 0, 2]
    truth = [1, 1, 1, 3]
    values = build_target("signed_gain", outcomes(text, fused, truth))
    assert list(values) == [1.0, 0.0, -1.0, 0.0]


def test_class_balanced_gain_rewards_a_rare_class_more():
    """The whole point: a fix on a rare class must outweigh one on a common class."""
    # neutral is common (8 samples), surprise is rare (1 sample).
    truth = [0] * 8 + [6]
    text = [1] * 8 + [1]          # every text prediction wrong
    fused = [0] * 8 + [6]         # every fused prediction right
    values = build_target("class_balanced_gain", outcomes(text, fused, truth))
    assert values[-1] > values[0], "the rare-class correction must be worth more"
    assert values[0] > 0


def test_class_weights_are_inverse_frequency_and_mean_one():
    weights = class_weights_from_labels(np.array([0] * 9 + [1]), num_classes=7)
    assert weights[1] > weights[0]
    present = weights[weights > 0]
    assert float(present.mean()) == pytest.approx(1.0)
    # Absent classes get zero, not infinity.
    assert weights[2] == 0.0


def test_macro_f1_marginal_is_zero_where_acquisition_changes_nothing():
    text = [0, 1, 2, 3]
    fused = [0, 1, 2, 3]           # identical
    values = build_target("macro_f1_marginal", outcomes(text, fused, text))
    assert np.allclose(values, 0.0)


def test_macro_f1_marginal_prefers_the_rare_class_correction():
    """The target that matches the reported metric must rank by macro-F1 impact."""
    truth = [0] * 20 + [6]
    text = [1] * 20 + [1]
    fused = [0] * 20 + [6]
    values = build_target("macro_f1_marginal", outcomes(text, fused, truth))
    assert values[-1] > values[0], (
        "fixing the only 'surprise' sample must move macro-F1 more than fixing one "
        "of twenty 'neutral' samples"
    )


def test_macro_f1_marginal_is_negative_for_a_harmful_acquisition():
    truth = [0, 0, 1, 1]
    text = [0, 0, 1, 1]            # all correct
    fused = [0, 0, 1, 0]           # acquisition breaks the last one
    values = build_target("macro_f1_marginal", outcomes(text, fused, truth))
    assert values[3] < 0
    assert np.allclose(values[:3], 0.0)


def test_uncertainty_reduction_is_carried_as_a_rejected_control():
    assert REJECTED_BY_STAGE1 == "uncertainty_reduction"
    assert "REJECTED" in TARGET_DEFINITIONS["uncertainty_reduction"]
    values = build_target(
        "uncertainty_reduction",
        outcomes([0], [0], [0], text_u=[0.9], fused_u=[0.4]),
    )
    assert values[0] == pytest.approx(0.5)


def test_no_target_is_computable_without_labels():
    """Every target reads the truth: they are supervision, never features."""
    item = outcomes([0, 1], [1, 1], [1, 1])
    for name in TARGET_NAMES:
        if name == "uncertainty_reduction":
            continue      # the one exception, and it is the rejected one
        shifted = outcomes([0, 1], [1, 1], [0, 0])
        assert not np.array_equal(
            build_target(name, item), build_target(name, shifted)
        ), name


def test_describe_reports_where_the_corrections_land():
    truth = [0] * 10 + [6] * 2
    text = [1] * 12
    fused = [0] * 10 + [6] * 2
    record = describe_targets(outcomes(text, fused, truth))
    assert record["where_the_corrections_are"]["neutral"]["corrections"] == 10
    assert record["where_the_corrections_are"]["surprise"]["corrections"] == 2
    assert record["correction_concentration"]["top_classes"][0] == "neutral"
    assert record["outcomes"]["corrections"] == 12


def test_empty_pool_is_refused():
    with pytest.raises(TargetError, match="empty pool"):
        RoutingOutcomes(
            sample_ids=[], true_class=np.array([]), text_prediction=np.array([]),
            fused_prediction=np.array([]), text_uncertainty=np.array([]),
            fused_uncertainty=np.array([]),
        )


# ============================================================
# Budget curve
# ============================================================

def pool(text, fused, truth, audio=None) -> PoolOutcomes:
    return PoolOutcomes(
        sample_ids=[f"s{i}" for i in range(len(truth))],
        true_class=np.asarray(truth, dtype=int),
        text_prediction=np.asarray(text, dtype=int),
        fused_prediction=np.asarray(fused, dtype=int),
        audio_prediction=np.asarray(audio if audio is not None else fused, dtype=int),
        text_latency_ms=3450.0, audio_latency_ms=28.0,
    )


def test_top_k_acquires_exactly_the_budget():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    assert acquire_top_k(scores, 0.0).sum() == 0
    assert acquire_top_k(scores, 0.5).sum() == 2
    assert acquire_top_k(scores, 1.0).sum() == 4
    assert list(acquire_top_k(scores, 0.5)) == [False, True, True, False]


def test_top_k_is_deterministic_under_ties():
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    first = acquire_top_k(scores, 0.5)
    second = acquire_top_k(scores, 0.5)
    assert list(first) == list(second)
    assert first.sum() == 2


def test_top_k_refuses_nan_scores():
    with pytest.raises(BudgetError, match="NaN"):
        acquire_top_k(np.array([0.1, np.nan]), 0.5)


def test_zero_budget_is_text_only_and_full_budget_is_always_fusion():
    item = pool([0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0])
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    none = score_at_budget(item, acquire_top_k(scores, 0.0), 0.0, "p")
    everything = score_at_budget(item, acquire_top_k(scores, 1.0), 1.0, "p")
    assert none["accuracy"] == pytest.approx(0.5)          # text right on 2 of 4
    assert everything["accuracy"] == pytest.approx(0.5)    # fused right on 2 of 4
    assert none["average_modalities_per_sample"] == 1.0
    assert everything["average_modalities_per_sample"] == 2.0


def test_corrections_harms_and_waste_are_counted_separately():
    #        correct   harm     waste (no change)   both wrong
    text =  [0,        1,       2,                  0]
    fused = [1,        0,       2,                  3]
    truth = [1,        1,       2,                  4]
    item = pool(text, fused, truth)
    row = score_at_budget(item, np.ones(4, dtype=bool), 1.0, "p")
    assert row["corrected_predictions"] == 1
    assert row["harmed_predictions"] == 1
    assert row["unnecessary_acquisitions"] == 1
    assert row["net_corrections"] == 0
    assert row["useful_acquisition_rate"] == pytest.approx(0.25)


def test_latency_and_compute_scale_with_the_acquisition_rate():
    item = pool([0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1])
    half = score_at_budget(item, acquire_top_k(np.arange(4.0), 0.5), 0.5, "p")
    assert half["mean_latency_ms"] == pytest.approx(3450.0 + 28.0 * 0.5)
    assert half["average_modalities_per_sample"] == pytest.approx(1.5)


def test_the_curve_covers_every_requested_budget():
    item = pool([0] * 20, [1] * 20, [1] * 20)
    curve = budget_curve(item, np.random.default_rng(0).random(20), "p")
    assert [row["requested_budget"] for row in curve] == list(DEFAULT_BUDGETS)


def test_oracle_ranking_is_the_ceiling_at_every_budget():
    rng = np.random.default_rng(3)
    n = 60
    truth = rng.integers(0, 4, n)
    text = np.where(rng.random(n) > 0.5, (truth + 1) % 7, truth)
    fused = np.where(rng.random(n) > 0.3, truth, (truth + 2) % 7)
    item = pool(text, fused, truth)

    oracle = budget_curve(item, oracle_scores(item), "oracle")
    other = budget_curve(item, rng.random(n), "random-ish")
    for left, right in zip(oracle, other):
        if left["requested_budget"] in (0.0, 1.0):
            continue
        assert left["macro_f1"] >= right["macro_f1"] - 1e-12, left["requested_budget"]


def test_random_control_reports_an_interval_at_interior_budgets():
    item = pool([0] * 40, [1] * 40, ([1] * 20) + ([0] * 20))
    rows = random_curve(item, budgets=(0.0, 0.5, 1.0), repeats=20)
    interior = rows[1]
    assert interior["repeats"] == 20
    assert interior["macro_f1"]["p2.5"] <= interior["macro_f1"]["mean"]
    assert interior["macro_f1"]["mean"] <= interior["macro_f1"]["p97.5"]
    # The endpoints are deterministic, so they need only one draw.
    assert rows[0]["repeats"] == 1 and rows[2]["repeats"] == 1


def test_endpoints_are_marked_non_comparable_against_random():
    """Every policy is identical at 0% and 100%; those are not wins."""
    rng = np.random.default_rng(5)
    n = 50
    truth = rng.integers(0, 3, n)
    text = np.where(rng.random(n) > 0.5, (truth + 1) % 7, truth)
    fused = np.where(rng.random(n) > 0.4, truth, (truth + 2) % 7)
    item = pool(text, fused, truth)
    comparison = compare_policies(
        item, {"p": rng.random(n)}, budgets=(0.0, 0.5, 1.0), repeats=20
    )
    verdicts = comparison["beats_random"]["p"]["per_budget"]
    assert verdicts["0.00"]["comparable"] is False
    assert verdicts["1.00"]["comparable"] is False
    assert verdicts["0.50"]["comparable"] is True
    assert "rather than counted as wins" in comparison["beats_random"]["p"]["note"]


def test_comparison_always_adds_the_oracle_and_random_lines():
    item = pool([0] * 30, [1] * 30, ([1] * 15) + ([0] * 15))
    comparison = compare_policies(
        item, {"mine": np.arange(30.0)}, budgets=(0.0, 0.25, 1.0), repeats=10
    )
    assert "oracle" in comparison["curves"]
    assert comparison["random_control"]
    assert comparison["primary_metric"] == "macro_f1"
    assert "majority-class" in comparison["why_macro_f1"]


def test_efficiency_summary_reports_retention_and_activation():
    item = pool([0] * 30, [1] * 30, ([1] * 20) + ([0] * 10))
    comparison = compare_policies(
        item, {"mine": np.arange(30.0)}, budgets=(0.0, 0.5, 1.0), repeats=5
    )
    summary = efficiency_summary(comparison)
    assert "mine" in summary["per_policy"]
    row = summary["per_policy"]["mine"][0]
    for key in ("macro_f1_retained", "modalities_per_sample", "mean_latency_ms",
                "unnecessary_acquisitions"):
        assert key in row
    assert "above 100%" in summary["reading"]
