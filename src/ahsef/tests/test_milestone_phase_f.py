"""PHASE F: the lock conditions, the frozen decision rule, and the paired statistics.

The failure this file exists to prevent is the quiet one: a locked evaluation
that reports a plausible number after a condition silently failed, a policy that
drifted between being frozen and being applied, or an unpaired interval quoted
where a paired one was available. Each of those produces a readable report and a
wrong conclusion.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ahsef.milestone.phase_f import (
    GENERALIZATION_RULE,
    LOCK_CONDITIONS,
    PHASE_F_PROTOCOL,
    LockCheckError,
    acquisition_quality,
    assert_locks_passed,
    comparison,
    generalization_verdict,
    paired_bootstrap,
    ranking_fingerprint,
    retained_fusion_gain,
    routed_predictions,
    routing_metrics,
    routing_trace,
    system_metrics,
    uncertainty_transfer,
)
from src.ahsef.milestone.budget import acquire_top_k
from src.common.labels import CANONICAL_EMOTION_CLASSES


# ============================================================
# The frozen decision rule
# ============================================================

def test_routed_prediction_takes_fusion_only_where_audio_was_acquired():
    text = np.array([0, 1, 2, 3])
    fused = np.array([4, 5, 6, 0])
    acquire = np.array([True, False, True, False])
    assert list(routed_predictions(text, fused, acquire)) == [4, 1, 6, 3]


def test_no_acquisition_is_exactly_text_only():
    text = np.array([0, 1, 2, 3])
    fused = np.array([4, 5, 6, 0])
    assert list(routed_predictions(text, fused, np.zeros(4, bool))) == list(text)


def test_full_acquisition_is_exactly_always_fusion():
    text = np.array([0, 1, 2, 3])
    fused = np.array([4, 5, 6, 0])
    assert list(routed_predictions(text, fused, np.ones(4, bool))) == list(fused)


def test_the_ranking_acquires_the_highest_uncertainty_samples():
    uncertainty = np.array([0.1, 0.9, 0.5, 0.7, 0.2, 0.8, 0.3, 0.4, 0.6, 0.05])
    acquire = acquire_top_k(uncertainty, 0.30)
    assert int(acquire.sum()) == 3
    assert list(np.flatnonzero(acquire)) == [1, 3, 5]


def test_the_acquisition_set_is_deterministic_under_ties():
    uncertainty = np.full(10, 0.5)
    first = acquire_top_k(uncertainty, 0.20)
    second = acquire_top_k(uncertainty, 0.20)
    assert np.array_equal(first, second)
    assert list(np.flatnonzero(first)) == [0, 1]


def test_the_ranking_fingerprint_identifies_the_acquired_set():
    ids = ["a", "b", "c", "d"]
    left = np.array([True, False, True, False])
    right = np.array([True, True, False, False])
    assert ranking_fingerprint(ids, left) == ranking_fingerprint(ids, left.copy())
    assert ranking_fingerprint(ids, left) != ranking_fingerprint(ids, right)


# ============================================================
# Budget enforcement
# ============================================================

def test_routing_metrics_confirm_the_frozen_budget_was_respected():
    acquire = acquire_top_k(np.arange(100, dtype=float), 0.10)
    record = routing_metrics(acquire, np.arange(100, dtype=float), "ahsef", 0.10)
    assert record["acquisition_count"] == 10
    assert record["acquisition_rate"] == pytest.approx(0.10)
    assert record["budget_respected"] is True
    assert record["average_modalities_per_sample"] == pytest.approx(1.10)
    assert record["text_only_samples"] == 90
    assert record["text_plus_audio_samples"] == 10


def test_a_budget_violation_is_reported_rather_than_hidden():
    acquire = np.zeros(100, dtype=bool)
    acquire[:25] = True
    assert routing_metrics(acquire, np.zeros(100), "p", 0.10)["budget_respected"] is False


def test_acquired_samples_carry_higher_uncertainty_than_the_rest():
    uncertainty = np.linspace(0.0, 1.0, 50)
    acquire = acquire_top_k(uncertainty, 0.20)
    record = routing_metrics(acquire, uncertainty, "ahsef", 0.20)
    assert record["uncertainty"]["acquired"]["mean"] > \
        record["uncertainty"]["not_acquired"]["mean"]
    assert record["uncertainty"]["separation"] > 0


# ============================================================
# Acquisition quality -- Phase D/E definitions, unchanged
# ============================================================

def test_acquisition_quality_partitions_every_acquisition_exactly_once():
    text_correct = np.array([False, True, True, False, False])
    fused_correct = np.array([True, False, True, False, True])
    acquire = np.array([True, True, True, True, False])
    text_pred = np.array([1, 2, 3, 4, 5])
    fused_pred = np.array([0, 0, 3, 0, 0])
    record = acquisition_quality(
        acquire, text_correct, fused_correct, text_pred, fused_pred, "p"
    )
    assert record["acquisitions"] == 4
    total = (
        record["text_wrong_audio_fixes"] + record["text_correct_audio_breaks"]
        + record["both_correct"] + record["both_wrong"]
    )
    assert total == 4
    assert record["net_corrections"] == 1 - 1


def test_correction_precision_equals_the_phase_de_useful_acquisition_rate():
    text_correct = np.array([False, False, True, True])
    fused_correct = np.array([True, True, True, True])
    acquire = np.ones(4, dtype=bool)
    record = acquisition_quality(
        acquire, text_correct, fused_correct, np.array([1, 1, 2, 2]),
        np.array([0, 0, 2, 2]), "p",
    )
    assert record["correction_precision"] == pytest.approx(0.5)
    assert record["correction_precision"] == record["useful_acquisition_rate"]
    assert record["harm_rate"] == pytest.approx(0.0)


def test_an_acquisition_that_changed_nothing_is_counted_as_waste():
    record = acquisition_quality(
        np.ones(3, dtype=bool), np.array([True, True, True]),
        np.array([True, True, True]), np.array([1, 2, 3]), np.array([1, 2, 3]), "p",
    )
    assert record["unnecessary_acquisitions"] == 3
    assert record["net_corrections"] == 0


def test_no_acquisitions_gives_undefined_rates_rather_than_zero():
    record = acquisition_quality(
        np.zeros(3, dtype=bool), np.array([True] * 3), np.array([True] * 3),
        np.zeros(3, int), np.zeros(3, int), "p",
    )
    assert record["correction_precision"] is None
    assert record["harm_rate"] is None


# ============================================================
# Metrics
# ============================================================

def test_system_metrics_report_every_class_even_when_absent():
    truth = np.array([0, 0, 1, 1])
    block = system_metrics(np.array([0, 0, 1, 1]), truth, "perfect")
    assert block["accuracy"] == pytest.approx(1.0)
    assert set(block["per_class"]) == set(CANONICAL_EMOTION_CLASSES)
    assert block["per_class"]["sad"]["support"] == 0
    assert len(block["confusion_matrix"]) == len(CANONICAL_EMOTION_CLASSES)


def test_comparison_reports_all_three_headline_deltas():
    left = {"macro_f1": 0.40, "accuracy": 0.60, "weighted_f1": 0.50}
    right = {"macro_f1": 0.30, "accuracy": 0.55, "weighted_f1": 0.45}
    block = comparison(left, right, "x")
    assert block["delta_macro_f1"] == pytest.approx(0.10)
    assert block["delta_accuracy"] == pytest.approx(0.05)
    assert block["delta_weighted_f1"] == pytest.approx(0.05)
    assert block["relative_macro_f1_gain"] == pytest.approx(1 / 3)


# ============================================================
# Retained fusion gain
# ============================================================

def test_retained_gain_uses_the_declared_formula():
    record = retained_fusion_gain(
        {"macro_f1": 0.35}, {"macro_f1": 0.30}, {"macro_f1": 0.40}
    )
    assert record["fusion_gain"] == pytest.approx(0.10)
    assert record["ahsef_gain"] == pytest.approx(0.05)
    assert record["retained_gain"] == pytest.approx(0.50)


def test_retained_gain_is_flagged_meaningless_when_fusion_did_not_help():
    record = retained_fusion_gain(
        {"macro_f1": 0.32}, {"macro_f1": 0.30}, {"macro_f1": 0.28}
    )
    assert record["fusion_gain"] < 0
    assert record["note"] is not None
    assert "meaningless" in record["note"]


# ============================================================
# Paired statistics
# ============================================================

def test_the_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 3, 200)
    left = truth.copy()
    left[:40] = (left[:40] + 1) % 3
    right = truth.copy()
    right[:80] = (right[:80] + 1) % 3
    first = paired_bootstrap(left, right, truth, "macro_f1", seed=7, resamples=200)
    again = paired_bootstrap(left, right, truth, "macro_f1", seed=7, resamples=200)
    assert first["ci95"] == again["ci95"]
    assert first["paired"] is True
    assert first["point_estimate"] > 0


def test_identical_systems_give_a_zero_point_estimate():
    truth = np.array([0, 1, 2] * 20)
    same = truth.copy()
    record = paired_bootstrap(same, same, truth, "macro_f1", seed=1, resamples=100)
    assert record["point_estimate"] == pytest.approx(0.0)
    assert record["excludes_zero"] is False


def test_a_clear_difference_produces_an_interval_excluding_zero():
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 3, 300)
    good = truth.copy()
    bad = (truth + 1) % 3
    record = paired_bootstrap(good, bad, truth, "macro_f1", seed=5, resamples=300)
    assert record["excludes_zero"] is True
    assert record["ci95"][0] > 0


# ============================================================
# Uncertainty transfer
# ============================================================

def test_uncertainty_that_ranks_errors_perfectly_gives_auroc_one():
    text_correct = np.array([True] * 10 + [False] * 10)
    uncertainty = np.concatenate([np.linspace(0.0, 0.4, 10), np.linspace(0.6, 1.0, 10)])
    record = uncertainty_transfer(uncertainty, text_correct, bins=4)
    assert record["auroc_identifying_incorrect_text"] == pytest.approx(1.0)
    assert record["monotone_decreasing_accuracy"] is True


def test_uncertainty_transfer_never_fits_or_recalibrates():
    record = uncertainty_transfer(
        np.linspace(0, 1, 40), np.array([True, False] * 20), bins=4
    )
    assert record["recalibrated_on_test"] is False
    assert record["threshold_selected_on_test"] is False
    assert record["anything_fitted_on_test"] is False


def test_accuracy_bins_cover_every_sample_exactly_once():
    uncertainty = np.linspace(0, 1, 100)
    record = uncertainty_transfer(uncertainty, np.array([True] * 50 + [False] * 50), 5)
    assert sum(row["samples"] for row in record["accuracy_by_uncertainty_bin"]) == 100


# ============================================================
# The verdict
# ============================================================

CLASSES = {"neutral": 0.60, "happy": 0.40, "sad": 0.10, "angry": 0.20,
           "fear": 0.0, "disgust": 0.0, "surprise": 0.0}
MAJORITY = ["neutral", "happy"]


def block(macro, per_class=None, accuracy=0.5):
    return {
        "macro_f1": macro, "accuracy": accuracy, "weighted_f1": macro,
        "per_class_f1": dict(per_class or CLASSES),
    }


def route(count):
    return {"acquisition_count": count, "average_modalities_per_sample": 1.1}


def verdict_for(ahsef_macro, random_macro, per_class, ci=(0.01, 0.09), acquisitions=50):
    ahsef = block(ahsef_macro, per_class)
    text = block(0.30, CLASSES)
    rnd = block(random_macro, CLASSES)
    always = block(0.40, CLASSES)
    return generalization_verdict(
        ahsef, text, rnd, always,
        retained_fusion_gain(ahsef, text, always),
        {"ci95": list(ci), "excludes_zero": ci[0] > 0},
        MAJORITY, route(acquisitions), route(500),
    )


def test_a_clear_win_is_classified_as_generalizes():
    better = {**CLASSES, "sad": 0.30, "angry": 0.35}
    record = verdict_for(0.36, 0.32, better)
    assert record["verdict"] == "A. GENERALIZES"
    assert record["criterion_2_retains_meaningful_fusion_gain"]["met"] is True


def test_losing_to_random_cannot_be_classified_as_generalizes():
    better = {**CLASSES, "sad": 0.30, "angry": 0.35}
    record = verdict_for(0.34, 0.36, better)
    assert record["criterion_1_beats_matched_random"]["met"] is False
    assert record["verdict"] != "A. GENERALIZES"


def test_a_gain_confined_to_neutral_cannot_be_classified_as_generalizes():
    neutral_only = {**CLASSES, "neutral": 0.95}
    record = verdict_for(0.36, 0.32, neutral_only)
    assert record["criterion_4_not_only_neutral"]["met"] is False
    assert record["verdict"] == "B. PARTIAL GENERALIZATION"


def test_no_advantage_over_text_only_is_classified_as_does_not_generalize():
    record = verdict_for(0.29, 0.31, CLASSES)
    assert record["criterion_5_beats_text_only"]["met"] is False
    assert record["verdict"] == "C. DOES NOT GENERALIZE"


def test_losing_most_of_the_fusion_gain_is_at_best_partial():
    better = {**CLASSES, "sad": 0.30}
    record = verdict_for(0.31, 0.305, better)
    assert record["criterion_2_retains_meaningful_fusion_gain"]["met"] is False
    assert record["verdict"] == "B. PARTIAL GENERALIZATION"


def test_the_verdict_is_macro_f1_led_not_accuracy_led():
    assert GENERALIZATION_RULE["primary_metric"] == "macro_f1"
    assert "instead of it" in GENERALIZATION_RULE["accuracy_is_not_the_criterion"]


def test_the_verdict_records_that_the_policy_was_not_changed():
    record = verdict_for(0.36, 0.32, {**CLASSES, "sad": 0.30, "angry": 0.35})
    assert record["policy_changed_after_seeing_test"] is False


# ============================================================
# Lock enforcement
# ============================================================

def test_a_failed_lock_condition_stops_the_evaluation():
    record = {
        "all_passed": False, "failed": ["6_test_manifest_matches_lock"],
    }
    with pytest.raises(LockCheckError, match="6_test_manifest_matches_lock"):
        assert_locks_passed(record)


def test_a_partially_run_lock_check_does_not_count_as_passed():
    with pytest.raises(LockCheckError):
        assert_locks_passed({"all_passed": False, "failed": []})


def test_every_declared_condition_has_a_stable_name():
    assert len(LOCK_CONDITIONS) == 10
    assert len(set(LOCK_CONDITIONS)) == 10
    assert LOCK_CONDITIONS[0].startswith("1_")
    assert LOCK_CONDITIONS[-1].startswith("10_")


# ============================================================
# Routing trace
# ============================================================

def test_the_trace_has_one_auditable_line_per_sample():
    rows = routing_trace(
        ["a", "b", "c"], np.array([0.9, 0.5, 0.1]), np.array([1, 2, 3]),
        np.array([True, False, False]), np.array([0, 1, 2]), np.array([1, 1, 1]),
        np.array([1, 1, 2]), np.array([1, 1, 2]),
    )
    assert len(rows) == 3
    assert rows[0]["outcome"] == "corrected"
    assert rows[0]["modalities_used"] == ["text_llm", "audio_strong"]
    assert rows[1]["modalities_used"] == ["text_llm"]
    assert rows[1]["outcome"] == "text_only_correct"
    assert all(row["decided_on"].startswith("normalised score entropy") for row in rows)


def test_the_trace_marks_a_harmful_acquisition_as_harmed():
    rows = routing_trace(
        ["a"], np.array([0.9]), np.array([1]), np.array([True]), np.array([1]),
        np.array([0]), np.array([0]), np.array([1]),
    )
    assert rows[0]["outcome"] == "harmed"
    assert rows[0]["final_correct"] is False


# ============================================================
# The recorded run
# ============================================================

LOCKS = "experiments/ahsef/milestone/phase_f_lock_checks.json"
RESULTS = "experiments/ahsef/milestone/phase_f_results.json"


def recorded(path):
    from pathlib import Path

    handle = Path(path)
    if not handle.exists():
        pytest.skip(f"{path} has not been produced yet")
    return json.loads(handle.read_text(encoding="utf-8"))


def test_all_ten_lock_conditions_passed_in_the_recorded_run():
    record = recorded(LOCKS)
    assert record["all_passed"] is True
    assert record["failed"] == []
    assert len(record["checks"]) == 10
    assert all(row["passed"] for row in record["checks"])


def test_the_lock_stage_read_no_test_payload():
    record = recorded(LOCKS)
    assert record["test_payload_read"] is False
    assert record["test_labels_read"] is False


def test_the_recorded_evaluation_applied_the_frozen_policy_unmodified():
    record = recorded(RESULTS)
    applied = record["frozen_policy_applied"]
    assert applied["modified_by_phase_f"] is False
    assert applied["selected_target"] == "uncertainty_only"
    assert applied["budget"] == 0.10
    assert applied["fusion"]["weights"] == {"audio": 0.6, "text_llm": 0.4}
    assert applied["sha256"] == record["lock_checks"]["frozen_policy_sha256"]


def test_the_recorded_evaluation_respected_the_frozen_budget():
    routing = recorded(RESULTS)["routing"]["E_ahsef_at_budget"]
    assert routing["budget_respected"] is True
    assert routing["acquisition_rate"] == pytest.approx(0.10, abs=0.005)


def test_the_recorded_evaluation_leaked_no_test_label_into_any_decision():
    audit = recorded(RESULTS)["leakage_audit"]
    for key in (
        "test_labels_used_for_policy_selection",
        "test_labels_used_for_threshold_selection",
        "test_labels_used_for_budget_selection",
        "test_labels_used_for_model_selection",
        "test_labels_used_for_prompt_selection",
        "test_labels_used_for_fusion_selection",
        "test_labels_used_for_uncertainty_selection",
        "policy_modified_after_seeing_test_results",
    ):
        assert audit[key] is False, key
    assert audit["only_oracle_uses_test_labels"] is True


def test_the_oracle_is_marked_as_a_label_aware_analysis_only_bound():
    oracle = recorded(RESULTS)["systems"]["F_oracle_at_budget"]
    assert oracle["label_aware"] is True
    assert "ANALYSIS ONLY" in oracle["marking"]


def test_the_recorded_evaluation_did_not_resample_the_test_set():
    dataset = recorded(RESULTS)["test_dataset"]
    assert dataset["resampled"] is False
    assert dataset["rebalanced"] is False
    assert dataset["manifest_changed"] is False


def test_the_recorded_evaluation_priced_audio_from_the_frozen_configuration():
    cost = recorded(RESULTS)["cost"]
    assert cost["re_estimated_from_test_behaviour"] is False
    assert cost["configuration"]["re_measured_on_evaluation_pool"] is False
    assert 280.0 < cost["configuration"]["audio_strong"]["latency_ms_per_sample"] < 300.0


def test_the_recorded_run_carries_every_required_fingerprint():
    provenance = recorded(RESULTS)["reproducibility"]
    for key in (
        "frozen_policy_fingerprint", "test_manifest_fingerprint",
        "alignment_fingerprint", "text_model_fingerprint", "audio_model_fingerprint",
        "wav2vec2_checkpoint_fingerprint", "prompt_fingerprint",
        "generation_configuration", "fusion_configuration",
        "uncertainty_configuration", "budget", "seed", "cost_configuration",
        "software", "timestamp", "sample_counts",
    ):
        assert provenance.get(key) not in (None, {}, []), key
    assert provenance["protocol_version"] == PHASE_F_PROTOCOL


def test_the_routing_trace_covers_every_evaluated_sample():
    from pathlib import Path

    record = recorded(RESULTS)
    path = Path("experiments/ahsef/milestone/phase_f_routing_trace.jsonl")
    if not path.exists():
        pytest.skip("no routing trace yet")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == record["test_dataset"]["evaluated_samples"]
    acquired = sum(1 for row in rows if row["acquired_audio"])
    assert acquired == record["routing"]["E_ahsef_at_budget"]["acquisition_count"]
