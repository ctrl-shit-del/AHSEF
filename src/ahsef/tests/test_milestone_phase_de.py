"""PHASE D / E: the frozen cost model, the selection rule, and the test lock.

The tests that matter most here are the ones that fail *quietly* in production
if they are absent: a cost model that silently prices audio at the cached-head
latency, a selection rule whose tie-break depends on dict ordering, a random
control that is not reproducible, and a driver that could read a test artefact
without anything noticing.  Each of those produces a plausible-looking report.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ahsef.milestone.budget import (
    DEFAULT_BUDGETS,
    PoolOutcomes,
    acquire_top_k,
    annotate_curve,
    budget_curve,
    compare_policies,
    oracle_scores,
    random_curve,
    score_at_budget,
)
from src.ahsef.milestone.costs import (
    COST_POLICY,
    CostConfigurationError,
    FrozenCosts,
    assert_encoder_is_charged,
    load_frozen_costs,
)
from src.ahsef.milestone.quality import (
    class_concentration,
    feature_set_ablation,
    target_distribution,
    target_quality,
)
from src.ahsef.milestone.reproducibility import (
    LOCKED_ARTEFACT_ROOTS,
    LockedArtefactError,
    PROTOCOL_VERSION,
    EvaluationLockError,
    assert_locked_artefacts_unchanged,
    assert_validation_only,
    locked_artefact_digest,
    sequence_fingerprint,
    verify_locked_artefacts,
)
from src.ahsef.milestone.selection import (
    MATERIAL_ROUTING_SUCCESS,
    MAX_ACQUISITION_SHARE_OF_ALWAYS_FUSION,
    MIN_RETAINED_FUSION_GAIN,
    PARSIMONY_MARGIN,
    PARSIMONY_REFERENCE,
    SELECTION_BUDGETS,
    SELECTION_RULE,
    SelectionError,
    majority_classes_of,
    minority_f1_mass,
    rank_candidates,
    routing_decision,
    selection_score,
)
from src.ahsef.milestone.targets import RoutingOutcomes, build_all_targets


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


def pool(text, fused, truth, audio_ms=289.0, text_ms=3449.0, **kwargs) -> PoolOutcomes:
    n = len(truth)
    return PoolOutcomes(
        sample_ids=[f"s{i}" for i in range(n)],
        true_class=np.asarray(truth, dtype=int),
        text_prediction=np.asarray(text, dtype=int),
        fused_prediction=np.asarray(fused, dtype=int),
        audio_prediction=np.asarray(fused, dtype=int),
        text_latency_ms=text_ms,
        audio_latency_ms=audio_ms,
        text_compute_units=1.0,
        audio_compute_units=10.0,
        **kwargs,
    )


def costs(audio_ms=289.0644, head_ms=0.1813) -> FrozenCosts:
    return FrozenCosts(
        text_llm_latency_ms=3449.7,
        audio_strong_latency_ms=audio_ms,
        text_llm_compute_units=2.24e13,
        audio_strong_compute_units=6.04e12,
        audio_strong_encoder_ms=audio_ms - head_ms,
        audio_strong_head_ms=head_ms,
        audio_strong_encoder_parameters=94_370_944,
        audio_strong_head_parameters=266_003,
        sources={"audio_strong_latency": "phase_b1"},
    )


# ============================================================
# Frozen cost usage
# ============================================================

def test_the_frozen_cost_model_charges_the_encoder_not_just_the_head():
    model = costs()
    assert_encoder_is_charged(model)
    record = model.to_dict()["audio_strong"]
    assert record["includes_encoder"] and record["includes_head"]
    # The head alone is three orders of magnitude cheaper; the deployment figure
    # must be dominated by the encoder rather than by the cached forward pass.
    assert record["latency_ms_per_sample"] > 100 * record["head_ms_per_sample"]


def test_pricing_audio_at_the_cached_head_latency_is_refused():
    with pytest.raises(CostConfigurationError, match="cached-head"):
        assert_encoder_is_charged(costs(audio_ms=0.1813))


def test_a_deployment_cost_that_does_not_exceed_the_head_is_refused():
    with pytest.raises(CostConfigurationError, match="encoder has not been charged"):
        assert_encoder_is_charged(costs(audio_ms=150.0, head_ms=150.0))


def test_the_cost_configuration_declares_it_was_not_re_measured():
    record = costs().to_dict()
    assert record["re_measured_on_evaluation_pool"] is False
    assert record["frozen"] is True
    assert "prohibited" in COST_POLICY["re_measurement"]


def test_a_missing_cost_artefact_raises_rather_than_being_invented(tmp_path):
    with pytest.raises(CostConfigurationError, match="will not re-measure"):
        load_frozen_costs(
            tmp_path / "absent_b1.json",
            tmp_path / "absent_stage3.json",
            tmp_path / "absent_extraction.json",
        )


def test_the_cost_fingerprint_moves_with_the_price():
    assert costs().fingerprint() != costs(audio_ms=300.0).fingerprint()


def test_the_repository_cost_configuration_matches_the_declared_figure():
    """The real artefacts must price strong audio at roughly 289 ms/sample."""
    model = load_frozen_costs()
    assert_encoder_is_charged(model)
    assert 280.0 < model.audio_strong_latency_ms < 300.0
    assert model.audio_strong_encoder_parameters > 90_000_000


# ============================================================
# Budget matching
# ============================================================

def test_every_policy_acquires_the_same_count_at_a_matched_budget():
    item = pool([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1] * 10, [1] * 10)
    left = acquire_top_k(np.arange(10, dtype=float), 0.30)
    right = acquire_top_k(np.arange(10, dtype=float)[::-1], 0.30)
    assert left.sum() == right.sum() == 3
    # Same count, different samples -- which is the only thing a policy
    # comparison at a matched rate is allowed to differ in.
    assert not np.array_equal(left, right)


def test_random_control_acquires_the_same_count_as_a_learned_policy():
    item = pool([0] * 20, [1] * 20, [1] * 20)
    rows = random_curve(item, (0.25,), repeats=5, seed=7)
    learned = budget_curve(item, np.arange(20, dtype=float), "learned", (0.25,))
    assert rows[0]["acquired"] == learned[0]["acquired"] == 5


def test_random_control_is_reproducible_for_a_seed():
    item = pool([0] * 30, [1] * 15 + [0] * 15, [1] * 30)
    first = random_curve(item, (0.20, 0.50), repeats=25, seed=11)
    again = random_curve(item, (0.20, 0.50), repeats=25, seed=11)
    different = random_curve(item, (0.20, 0.50), repeats=25, seed=12)
    assert [row["macro_f1"]["mean"] for row in first] == \
           [row["macro_f1"]["mean"] for row in again]
    assert [row["macro_f1"]["mean"] for row in first] != \
           [row["macro_f1"]["mean"] for row in different]


def test_the_oracle_never_falls_below_a_learned_policy_at_a_matched_budget():
    truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    text = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    fused = [0, 1, 2, 1, 1, 2, 2, 1, 2, 0]
    item = pool(text, fused, truth)
    ceiling = budget_curve(item, oracle_scores(item), "oracle", DEFAULT_BUDGETS)
    rng = np.random.default_rng(3)
    learned = budget_curve(item, rng.random(10), "learned", DEFAULT_BUDGETS)
    for high, low in zip(ceiling, learned):
        assert high["macro_f1"] >= low["macro_f1"] - 1e-9


def test_endpoints_are_identical_for_every_policy():
    truth = [0, 1, 2, 0, 1, 2]
    item = pool([0] * 6, [0, 1, 2, 1, 1, 2], truth)
    a = budget_curve(item, np.arange(6, dtype=float), "a", (0.0, 1.0))
    b = budget_curve(item, np.arange(6, dtype=float)[::-1], "b", (0.0, 1.0))
    assert a[0]["macro_f1"] == b[0]["macro_f1"]
    assert a[0]["acquired"] == 0
    assert a[1]["macro_f1"] == b[1]["macro_f1"]
    assert a[1]["acquired"] == 6


# ============================================================
# Uncertainty and cost reporting
# ============================================================

def test_uncertainty_after_acquisition_only_changes_on_acquired_samples():
    item = pool(
        [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1],
        text_uncertainty=np.array([0.8, 0.6, 0.4, 0.2]),
        fused_uncertainty=np.array([0.1, 0.1, 0.1, 0.1]),
    )
    row = score_at_budget(item, np.array([True, False, False, False]), 0.25, "p")
    assert row["mean_uncertainty_before_acquisition"] == pytest.approx(0.5)
    # Only the first sample switched to the fused value: (0.1+0.6+0.4+0.2)/4.
    assert row["mean_uncertainty_after_acquisition"] == pytest.approx(0.325)


def test_uncertainty_columns_are_absent_when_no_uncertainty_was_supplied():
    row = score_at_budget(
        pool([0, 0], [1, 1], [1, 1]), np.array([True, False]), 0.5, "p"
    )
    assert "mean_uncertainty_before_acquisition" not in row


def test_acquisition_cost_is_zero_at_zero_budget_and_full_at_full_budget():
    item = pool([0] * 10, [1] * 10, [1] * 10, audio_ms=289.0)
    curve = budget_curve(item, np.arange(10, dtype=float), "p", (0.0, 0.5, 1.0))
    assert curve[0]["acquisition_latency_ms"] == 0.0
    assert curve[1]["acquisition_latency_ms"] == pytest.approx(144.5)
    assert curve[2]["acquisition_latency_ms"] == pytest.approx(289.0)


def test_incremental_cost_is_undefined_rather_than_huge_when_nothing_was_gained():
    truth = [0, 1, 0, 1]
    item = pool([0, 1, 0, 1], [0, 1, 0, 1], truth)   # fusion changes nothing
    curve = budget_curve(item, np.arange(4, dtype=float), "p", (0.0, 0.5))
    text_only, always = curve[0], curve[-1]
    annotated = annotate_curve(
        curve, random_curve(item, (0.0, 0.5), repeats=3, seed=1), curve,
        text_only, always,
    )
    assert annotated[1]["incremental_cost_per_macro_f1"] is None


def test_annotation_compares_against_the_same_budget_on_every_reference():
    truth = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    item = pool([0] * 10, [0, 1, 2, 1, 1, 2, 2, 1, 2, 0], truth)
    scores = np.arange(10, dtype=float)
    curve = budget_curve(item, scores, "p", DEFAULT_BUDGETS)
    ceiling = budget_curve(item, oracle_scores(item), "oracle", DEFAULT_BUDGETS)
    control = random_curve(item, DEFAULT_BUDGETS, repeats=20, seed=5)
    annotated = annotate_curve(curve, control, ceiling, curve[0], curve[-1])
    for row, reference in zip(annotated, ceiling):
        assert row["requested_budget"] == reference["requested_budget"]
        assert row["gap_to_oracle"]["macro_f1"] == pytest.approx(
            reference["macro_f1"] - row["macro_f1"]
        )


# ============================================================
# Selection rule
# ============================================================

def curve_at(values) -> list[dict]:
    """A minimal curve carrying only what the selection rule reads."""
    rows = []
    for budget, macro in values.items():
        rows.append({
            "requested_budget": budget,
            "macro_f1": macro,
            "acquired": int(round(budget * 100)),
            "acquisition_rate": budget,
            "accuracy": macro,
            "weighted_f1": macro,
            "per_class_f1": {"neutral": macro, "happy": macro, "sad": macro},
        })
    return rows


FLAT = {b: 0.30 for b in DEFAULT_BUDGETS}


def test_selection_averages_exactly_the_declared_budgets():
    values = dict(FLAT)
    values.update({0.05: 0.10, 0.10: 0.20, 0.15: 0.30, 0.20: 0.40, 0.25: 0.50})
    values[0.75] = 0.99          # outside the selection region; must not count
    score = selection_score(curve_at(values))
    assert score["mean_macro_f1_over_selection_budgets"] == pytest.approx(0.30)
    assert score["budgets"] == [float(b) for b in SELECTION_BUDGETS]


def test_a_curve_on_a_different_grid_is_refused_rather_than_partially_scored():
    with pytest.raises(SelectionError, match="not applicable"):
        selection_score(curve_at({0.0: 0.3, 0.5: 0.4, 1.0: 0.5}))


def beats(*budgets) -> dict:
    return {
        "any_budget_above_random": bool(budgets),
        "budgets_above_random": [f"{b:.2f}" for b in budgets],
    }


def test_an_inadmissible_field_reports_routing_signal_insufficient():
    curves = {"a": curve_at(FLAT), "b": curve_at(FLAT)}
    verdict = rank_candidates(
        curves, {"a": beats(), "b": beats()}, 0.40, 0.30, PARSIMONY_REFERENCE
    )
    assert verdict["selected"] is None
    assert verdict["outcome"] == "ROUTING SIGNAL INSUFFICIENT"


def test_the_richer_target_wins_only_when_it_clears_the_parsimony_margin():
    reference = dict(FLAT)
    rich = {**FLAT, **{b: 0.30 + 2 * PARSIMONY_MARGIN for b in SELECTION_BUDGETS}}
    curves = {PARSIMONY_REFERENCE: curve_at(reference), "rich": curve_at(rich)}
    verdict = rank_candidates(
        curves, {PARSIMONY_REFERENCE: beats(0.10), "rich": beats(0.10)}, 0.40, 0.30,
    )
    assert verdict["selected"] == "rich"
    assert verdict["parsimony_applied"] is False


def test_a_near_tie_is_resolved_in_favour_of_the_simpler_signal():
    reference = dict(FLAT)
    barely = {**FLAT, **{b: 0.30 + PARSIMONY_MARGIN / 2 for b in SELECTION_BUDGETS}}
    curves = {PARSIMONY_REFERENCE: curve_at(reference), "rich": curve_at(barely)}
    verdict = rank_candidates(
        curves, {PARSIMONY_REFERENCE: beats(0.10), "rich": beats(0.10)}, 0.40, 0.30,
    )
    assert verdict["selected"] == PARSIMONY_REFERENCE
    assert verdict["parsimony_applied"] is True
    assert "simpler signal is selected" in verdict["parsimony_note"]


def test_the_ranking_is_deterministic_under_an_exact_tie():
    curves = {name: curve_at(FLAT) for name in ("zulu", "alpha", "mike")}
    verdicts = {name: beats(0.10) for name in curves}
    first = rank_candidates(curves, verdicts, 0.40, 0.30, PARSIMONY_REFERENCE)
    # Reversed insertion order must not change the winner.
    reversed_curves = {name: curves[name] for name in reversed(list(curves))}
    second = rank_candidates(reversed_curves, verdicts, 0.40, 0.30, PARSIMONY_REFERENCE)
    assert first["selected"] == second["selected"] == "alpha"


def test_the_oracle_is_never_a_selectable_candidate():
    curves = {"oracle": curve_at({**FLAT, **{b: 0.99 for b in SELECTION_BUDGETS}}),
              "real": curve_at(FLAT)}
    verdict = rank_candidates(
        curves, {"oracle": beats(0.10), "real": beats(0.10)}, 0.40, 0.30,
    )
    assert verdict["selected"] == "real"
    assert "oracle" not in verdict["per_candidate"]


def test_the_selection_rule_is_declared_in_advance_and_is_macro_f1_led():
    assert SELECTION_RULE["metric"] == "macro_f1"
    assert SELECTION_RULE["uses_test_labels"] is False
    assert "before any Phase D" in SELECTION_RULE["declared"]


# ============================================================
# The decision rule
# ============================================================

def row(macro, acquired, per_class, budget=0.10, accuracy=0.5) -> dict:
    return {
        "requested_budget": budget, "acquired": acquired,
        "acquisition_rate": budget, "macro_f1": macro, "accuracy": accuracy,
        "weighted_f1": macro, "per_class_f1": per_class,
    }


TEXT_CLASSES = {"neutral": 0.60, "happy": 0.40, "sad": 0.10, "angry": 0.20}
MAJORITY = ["neutral", "happy"]


def control(mean, upper) -> dict:
    return {"macro_f1": {"mean": mean, "p97.5": upper}}


def test_material_success_requires_all_four_conditions():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    good = row(0.36, 10, {**TEXT_CLASSES, "sad": 0.30, "angry": 0.35})
    verdict = routing_decision(good, control(0.32, 0.34), text, always, MAJORITY)
    assert verdict["material_routing_success"] is True
    assert verdict["verdict"] == "MATERIAL ROUTING SUCCESS"


def test_a_policy_inside_the_random_interval_fails_however_good_it_looks():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    good = row(0.36, 10, {**TEXT_CLASSES, "sad": 0.30, "angry": 0.35})
    verdict = routing_decision(good, control(0.35, 0.38), text, always, MAJORITY)
    assert verdict["condition_1_beats_matched_random"]["met"] is False
    assert verdict["material_routing_success"] is False


def test_a_gain_bought_entirely_in_the_majority_classes_fails_condition_two():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    # Macro-F1 rises, but every point of it came from neutral and happy.
    neutral_only = row(0.36, 10, {**TEXT_CLASSES, "neutral": 0.95, "happy": 0.75})
    verdict = routing_decision(neutral_only, control(0.32, 0.34), text, always, MAJORITY)
    assert verdict["condition_1_beats_matched_random"]["met"] is True
    assert verdict["condition_2_not_only_majority_corrections"]["met"] is False
    assert verdict["material_routing_success"] is False


def test_a_policy_that_retains_too_little_fusion_gain_fails_condition_three():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    thin = row(0.32, 10, {**TEXT_CLASSES, "sad": 0.30})
    verdict = routing_decision(thin, control(0.30, 0.31), text, always, MAJORITY)
    assert verdict["condition_3_retains_fusion_gain"]["met"] is False
    assert verdict["condition_3_retains_fusion_gain"]["retained_fraction"] == \
        pytest.approx(0.2)


def test_a_policy_that_acquires_almost_everywhere_fails_condition_four():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    greedy = row(0.38, 80, {**TEXT_CLASSES, "sad": 0.35, "angry": 0.40}, budget=0.80)
    verdict = routing_decision(greedy, control(0.34, 0.36), text, always, MAJORITY)
    assert verdict["condition_4_substantially_fewer_acquisitions"]["met"] is False
    assert verdict["material_routing_success"] is False


def test_beating_always_fusion_is_not_required():
    assert "not required" in MATERIAL_ROUTING_SUCCESS["not_required"].lower() or \
        "always-fusion" in MATERIAL_ROUTING_SUCCESS["not_required"]
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    below = row(0.36, 10, {**TEXT_CLASSES, "sad": 0.30, "angry": 0.35})
    assert below["macro_f1"] < always["macro_f1"]
    assert routing_decision(
        below, control(0.32, 0.34), text, always, MAJORITY
    )["material_routing_success"] is True


def test_minority_mass_sums_only_the_classes_outside_the_two_largest():
    assert minority_f1_mass(row(0.3, 1, TEXT_CLASSES), MAJORITY) == pytest.approx(0.30)


def test_majority_classes_are_read_from_the_pool_not_assumed():
    labels = np.array([2] * 50 + [3] * 40 + [0] * 5 + [1] * 3)
    assert majority_classes_of(labels) == ["sad", "angry"]


def test_the_declared_thresholds_are_the_ones_the_code_applies():
    text = row(0.30, 0, TEXT_CLASSES, budget=0.0)
    always = row(0.40, 100, TEXT_CLASSES, budget=1.0)
    edge = row(
        0.30 + MIN_RETAINED_FUSION_GAIN * 0.10,
        int(MAX_ACQUISITION_SHARE_OF_ALWAYS_FUSION * 100),
        {**TEXT_CLASSES, "sad": 0.30},
    )
    verdict = routing_decision(edge, control(0.30, 0.31), text, always, MAJORITY)
    assert verdict["condition_3_retains_fusion_gain"]["met"] is True
    assert verdict["condition_4_substantially_fewer_acquisitions"]["met"] is True


# ============================================================
# The test lock
# ============================================================

@pytest.mark.parametrize("path", [
    "experiments/ahsef/stage1/predictions/text__test.parquet",
    "experiments/ahsef/stage3_text_audio/hsig/oracle_test.parquet",
    "experiments/ahsef/stage3_text_audio/routing/decisions_test.jsonl",
    "experiments/ahsef/stage1/fusion/text+audio/test_predictions.parquet",
])
def test_every_test_artefact_shape_is_refused(path):
    with pytest.raises(EvaluationLockError, match="validation-only"):
        assert_validation_only([path])


def test_validation_artefacts_pass_the_lock():
    kept = assert_validation_only([
        "experiments/ahsef/stage3_text_audio/predictions/text_llm__validation.parquet",
        "experiments/ahsef/stage3_text_audio/hsig/features_validation.parquet",
    ])
    assert len(kept) == 2


def test_a_test_path_hidden_among_validation_paths_is_still_caught():
    with pytest.raises(EvaluationLockError):
        assert_validation_only([
            "experiments/ahsef/stage3_text_audio/hsig/features_validation.parquet",
            "experiments/ahsef/stage1/predictions/audio__test.parquet",
        ])


def test_no_target_or_feature_can_be_built_from_a_test_split():
    """The targets read labels, so the only defence is that they never see test.

    This asserts the shape of that defence rather than re-testing the targets:
    every path the driver opens goes through the lock, and the lock rejects the
    test partition by name.
    """
    from src.ahsef.milestone.estimator import EstimatorLeakageError, assert_fit_split

    with pytest.raises(EstimatorLeakageError):
        assert_fit_split("test")
    assert assert_fit_split("validation") == "validation"


# ============================================================
# Locked artefacts
# ============================================================

def test_a_baseline_is_established_on_the_first_run_and_says_so(tmp_path):
    root = tmp_path / "locked"
    root.mkdir()
    (root / "frozen.json").write_text('{"a": 1}', encoding="utf-8")
    first = verify_locked_artefacts(tmp_path / "baseline.json", (root,))
    assert first["verified"] is None
    assert first["baseline_established"] is True


def test_an_unchanged_locked_directory_verifies(tmp_path):
    root = tmp_path / "locked"
    root.mkdir()
    (root / "frozen.json").write_text('{"a": 1}', encoding="utf-8")
    verify_locked_artefacts(tmp_path / "baseline.json", (root,))
    second = verify_locked_artefacts(tmp_path / "baseline.json", (root,))
    assert second["verified"] is True
    assert_locked_artefacts_unchanged(second)


@pytest.mark.parametrize("mutate", [
    lambda root: (root / "frozen.json").write_text('{"a": 2}', encoding="utf-8"),
    lambda root: (root / "frozen.json").unlink(),
    lambda root: (root / "frozen.json").rename(root / "renamed.json"),
    lambda root: (root / "extra.json").write_text("{}", encoding="utf-8"),
])
def test_any_change_to_a_locked_artefact_is_detected(tmp_path, mutate):
    root = tmp_path / "locked"
    root.mkdir()
    (root / "frozen.json").write_text('{"a": 1}', encoding="utf-8")
    verify_locked_artefacts(tmp_path / "baseline.json", (root,))
    mutate(root)
    after = verify_locked_artefacts(tmp_path / "baseline.json", (root,))
    assert after["verified"] is False
    with pytest.raises(LockedArtefactError):
        assert_locked_artefacts_unchanged(after)


def test_the_locked_roots_cover_every_frozen_stage():
    names = {root.name for root in LOCKED_ARTEFACT_ROOTS}
    assert {"stage1", "stage2_llm", "stage3_text_audio", "audio_strong"} <= names


def test_a_pool_fingerprint_is_order_independent_but_content_sensitive():
    assert sequence_fingerprint(["b", "a"]) == sequence_fingerprint(["a", "b"])
    assert sequence_fingerprint(["a", "b"]) != sequence_fingerprint(["a", "c"])


# ============================================================
# Target quality
# ============================================================

def test_a_perfect_score_separates_useful_acquisitions_completely():
    item = outcomes([0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1])
    perfect = (~item.text_correct).astype(float)
    quality = target_quality(
        "t", perfect, perfect, item, np.zeros(4)
    )
    assert quality["identifying_useful_acquisitions"]["auroc"] == pytest.approx(1.0)


# Four neutral, three happy, and one each of sad/angry/fear -- the two largest
# classes are 7 of 10, so a signal has room to prefer them or to avoid them.
CONCENTRATION_TRUTH = [0, 0, 0, 0, 1, 1, 1, 2, 3, 4]


def test_class_concentration_flags_a_signal_that_favours_the_majority():
    item = outcomes([6] * 10, [5] * 10, CONCENTRATION_TRUTH)
    majority_seeking = np.array([9.0, 8, 7, 6, 5, 4, 3, 0, 0, 0])
    record = class_concentration(majority_seeking, item, fraction=0.4)
    assert record["majority_classes"] == ["neutral", "happy"]
    assert record["majority_pool_share"] == pytest.approx(0.7)
    assert record["majority_selected_share"] == pytest.approx(1.0)
    assert record["favours_majority_classes"] is True
    assert record["majority_over_representation"] > 1.0


def test_class_concentration_does_not_flag_a_minority_seeking_signal():
    item = outcomes([6] * 10, [5] * 10, CONCENTRATION_TRUTH)
    minority_seeking = np.array([0.0, 0, 0, 0, 0, 0, 0, 9, 8, 7])
    record = class_concentration(minority_seeking, item, fraction=0.3)
    assert record["majority_selected_share"] == pytest.approx(0.0)
    assert record["favours_majority_classes"] is False
    assert record["per_class"]["sad"]["over_representation"] > 1.0


def test_target_distribution_reports_whether_a_target_can_express_harm():
    item = outcomes([0, 1, 0], [1, 0, 0], [1, 1, 0])
    built = build_all_targets(item)
    assert target_distribution(built["binary_correction"], item)["expresses_harm"] is False
    assert target_distribution(built["signed_gain"], item)["expresses_harm"] is True


def test_the_feature_ablation_reports_no_help_when_richer_sets_do_not_win():
    records = [
        {"feature_set": "uncertainty_only", "features": ["u"],
         "quality": {"identifying_useful_acquisitions": {"auroc": 0.75},
                     "rank_agreement": {"spearman_vs_signed_gain": 0.2}}},
        {"feature_set": "evidence_only", "features": list("abcdefghij"),
         "quality": {"identifying_useful_acquisitions": {"auroc": 0.752},
                     "rank_agreement": {"spearman_vs_signed_gain": 0.2}}},
    ]
    summary = feature_set_ablation(records)
    assert summary["richer_features_help"] is False
    assert summary["simplest"] == "uncertainty_only"
    assert summary["reproduced"].startswith("yes")


# ============================================================
# The recorded run
# ============================================================

RESULTS = "experiments/ahsef/milestone/phase_de_results.json"


def recorded():
    from pathlib import Path

    path = Path(RESULTS)
    if not path.exists():
        pytest.skip(f"{RESULTS} has not been produced yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_recorded_run_never_opened_the_test_partition():
    record = recorded()
    assert record["test_lock"]["test_partition_opened"] is False
    assert record["test_lock"]["test_labels_used"] is False
    assert record["test_lock"]["thresholds_tuned_on_test"] is False
    assert record["phase_d"]["test_partition_opened"] is False


def test_the_recorded_run_left_every_locked_artefact_untouched():
    record = recorded()
    after = record["test_lock"]["locked_artefacts_after"]
    assert after["verified"] is not False
    assert not after.get("changed_roots")


def test_the_recorded_run_carries_every_required_fingerprint():
    provenance = recorded()["reproducibility"]
    for key in (
        "dataset_fingerprint", "alignment_fingerprint", "model_fingerprints",
        "estimator_configuration", "target_fingerprints", "feature_configuration",
        "seed", "budget_grid", "cost_configuration", "protocol_version",
    ):
        assert provenance.get(key) not in (None, {}, []), key
    assert provenance["protocol_version"] == PROTOCOL_VERSION


def test_the_recorded_run_priced_audio_with_the_frozen_encoder_cost():
    configuration = recorded()["reproducibility"]["cost_configuration"]
    assert configuration["re_measured_on_evaluation_pool"] is False
    assert 280.0 < configuration["audio_strong"]["latency_ms_per_sample"] < 300.0
    assert configuration["audio_strong"]["includes_encoder"] is True


def test_the_recorded_run_used_one_estimator_family_for_every_target():
    estimator = recorded()["phase_d"]["estimator"]
    assert estimator["identical_across_targets"] is True
    assert estimator["family"].startswith("ridge")


def test_the_recorded_budget_grid_is_the_one_phase_e_asked_for():
    assert recorded()["reproducibility"]["budget_grid"] == list(DEFAULT_BUDGETS)


def test_the_recorded_estimators_reload_and_predict_from_stored_coefficients():
    from pathlib import Path

    from src.ahsef.milestone.estimator import MilestoneHSIG

    directory = Path("experiments/ahsef/milestone/estimators")
    if not directory.exists():
        pytest.skip("no estimator artefacts yet")
    for path in sorted(directory.glob("*.json")):
        model = MilestoneHSIG.load(path)
        assert model.coefficients
        assert MilestoneHSIG.load(path).fingerprint() == model.fingerprint()
