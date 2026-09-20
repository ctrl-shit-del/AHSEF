"""PHASE B-1 / C: the improvement bar, the latency honesty, the headroom test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ahsef.analysis.expert_comparison import (
    MATERIAL_IMPROVEMENT,
    calibration_block,
    encoder_cost_from_provenance,
    headroom_comparison,
    improvement_verdict,
    latency_profile,
    metric_block,
    pair_headroom,
)
from src.ahsef.tests.conftest import make_prediction_set
from src.common.labels import CANONICAL_EMOTION_CLASSES

ANALYSIS = Path("experiments/ahsef/analysis/audio_expert")


def one_hot(predictions, n_classes=7, strength=6.0):
    logits = torch.zeros((len(predictions), n_classes), dtype=torch.float64)
    for row, klass in enumerate(predictions):
        logits[row, int(klass)] = strength
    return logits


def make(predictions, labels, modality="audio", latency_ms=1.0):
    ids = [f"s{i}" for i in range(len(labels))]
    return make_prediction_set(
        modality, ids, one_hot(predictions), labels,
        split="validation", latency_ms=latency_ms,
    )


# ============================================================
# The bar
# ============================================================

def test_the_bar_is_declared_and_macro_f1_led():
    assert MATERIAL_IMPROVEMENT["primary_metric"] == "macro_f1"
    assert MATERIAL_IMPROVEMENT["min_macro_f1_gain"] == 0.03
    assert "before any strong-expert result" in MATERIAL_IMPROVEMENT["declared"]
    assert "majority-class" in MATERIAL_IMPROVEMENT["rule"]
    # The consequence of failing must also be pre-declared.
    assert "Do not proceed" in MATERIAL_IMPROVEMENT["if_not_met"]


def block(macro_f1, accuracy, weighted=0.4, balanced=0.3, samples=100):
    return {
        "macro_f1": macro_f1, "accuracy": accuracy, "weighted_f1": weighted,
        "balanced_accuracy": balanced, "samples": samples,
        "per_class": {
            name: {"f1": macro_f1, "recall": macro_f1, "support": 10}
            for name in CANONICAL_EMOTION_CLASSES
        },
    }


def test_a_clear_gain_meets_the_bar():
    verdict = improvement_verdict(block(0.26, 0.35), block(0.32, 0.40))
    assert verdict["material_improvement"] is True
    assert verdict["macro_f1_gain"] == pytest.approx(0.06)
    assert verdict["decision"].startswith("PROCEED")


def test_a_gain_below_the_bar_does_not_meet_it():
    verdict = improvement_verdict(block(0.26, 0.35), block(0.28, 0.40))
    assert verdict["meets_macro_f1_bar"] is False
    assert verdict["material_improvement"] is False
    assert "DO NOT PROCEED" in verdict["decision"]


def test_a_macro_gain_bought_with_an_accuracy_regression_fails():
    """Guards the failure mode where a model trades the majority class away."""
    verdict = improvement_verdict(block(0.26, 0.40), block(0.32, 0.35))
    assert verdict["meets_macro_f1_bar"] is True
    assert verdict["meets_accuracy_bar"] is False
    assert verdict["material_improvement"] is False


def test_the_bar_is_exactly_inclusive():
    assert improvement_verdict(block(0.26, 0.35), block(0.29, 0.35))[
        "material_improvement"] is True
    assert improvement_verdict(block(0.26, 0.35), block(0.2899, 0.35))[
        "material_improvement"] is False


def test_the_verdict_names_improved_and_regressed_classes():
    baseline = block(0.3, 0.4)
    strong = block(0.3, 0.4)
    strong["per_class"]["happy"] = {"f1": 0.6, "recall": 0.6, "support": 10}
    strong["per_class"]["sad"] = {"f1": 0.1, "recall": 0.1, "support": 10}
    verdict = improvement_verdict(baseline, strong)
    assert "happy" in verdict["classes_improved"]
    assert "sad" in verdict["classes_regressed"]


def test_the_statement_reports_both_directions_numerically():
    statement = improvement_verdict(block(0.26, 0.35), block(0.32, 0.40))["statement"]
    assert "0.2600 -> 0.3200" in statement
    assert "MEETS" in statement


# ============================================================
# Metrics
# ============================================================

def test_metric_block_covers_every_phase_b1_quantity():
    record = metric_block(make([0, 1, 2, 3], [0, 1, 2, 3]), "x")
    for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision",
                "macro_recall", "balanced_accuracy", "confusion_matrix"):
        assert key in record
    assert set(record["per_class"]) == set(CANONICAL_EMOTION_CLASSES)
    for entry in record["per_class"].values():
        assert set(entry) == {"support", "precision", "recall", "f1"}


def test_metric_block_excludes_unusable_rows():
    item = make([0, 1, 2, 3], [0, 1, 2, 3])
    item.frame.loc[1, "predicted_class"] = -1
    record = metric_block(item, "x")
    assert record["samples"] == 3
    assert record["excluded_unusable"] == 1


def test_calibration_block_is_produced():
    record = calibration_block(make([0, 1, 2, 3], [0, 1, 2, 3]), "x")
    assert record["available"] is True
    for key in ("ece", "mce", "brier", "nll", "mean_confidence"):
        assert key in record


# ============================================================
# Latency honesty
# ============================================================

def test_a_from_scratch_expert_is_charged_only_its_measured_pass():
    record = latency_profile(make([0, 1], [0, 1], latency_ms=5.0))
    assert record["deployment_ms_per_sample"] == pytest.approx(5.0)
    assert record["encoder_ms_per_sample"] is None
    assert "forward pass" in record["measured_covers"]


def test_a_frozen_encoder_expert_is_charged_the_encoder_too():
    """The head's measured pass understates this expert by orders of magnitude."""
    record = latency_profile(
        make([0, 1], [0, 1], latency_ms=0.3), encoder_ms_per_clip=275.0,
        encoder_name="WAV2VEC2_BASE",
    )
    assert record["measured_ms_per_sample"] == pytest.approx(0.3)
    assert record["encoder_ms_per_sample"] == pytest.approx(275.0)
    assert record["deployment_ms_per_sample"] == pytest.approx(275.3)
    assert "cached features only" in record["measured_covers"]
    assert "understate" in record["note"]


def test_encoder_cost_is_read_from_measured_provenance(tmp_path):
    path = tmp_path / "extraction_provenance.json"
    path.write_text(json.dumps({
        "extractor": {"bundle": "WAV2VEC2_BASE"},
        "this_invocation": {"rate_per_second": 4.0},
    }), encoding="utf-8")
    milliseconds, name = encoder_cost_from_provenance(path)
    assert milliseconds == pytest.approx(250.0)
    assert name == "WAV2VEC2_BASE"


def test_a_missing_provenance_yields_no_invented_cost(tmp_path):
    assert encoder_cost_from_provenance(tmp_path / "absent.json") == (None, None)


# ============================================================
# Phase C headroom
# ============================================================

def test_pair_headroom_matches_hand_computation():
    #          both ok  left ok  right ok  both wrong
    labels =  [0,       1,       2,        3]
    left =    [0,       1,       5,        6]
    right =   [0,       4,       2,        5]
    record = pair_headroom(make(left, labels, "text_llm"), make(right, labels), [
        f"s{i}" for i in range(4)
    ])
    assert record["left_accuracy"] == pytest.approx(0.5)
    assert record["right_accuracy"] == pytest.approx(0.5)
    assert record["oracle_either_accuracy"] == pytest.approx(0.75)
    assert record["headroom_over_best_single"] == pytest.approx(0.25)
    assert record["right_fixes_left"] == 1
    assert record["left_fixes_right"] == 1
    assert record["both_wrong"] == 1


def test_pair_headroom_refuses_disagreeing_labels():
    ids = ["s0", "s1"]
    left = make([0, 1], [0, 1], "text_llm")
    right = make([0, 1], [0, 2])
    with pytest.raises(ValueError, match="disagree about the true class"):
        pair_headroom(left, right, ids)


def test_a_stronger_expert_that_only_agrees_more_gives_the_router_less():
    """The central Phase C subtlety, made a test rather than a footnote."""
    baseline = {"oracle_either_accuracy": 0.70, "headroom_over_best_single": 0.23}
    # More accurate in isolation, but its new correct answers duplicate the LLM's,
    # so the oracle ceiling does not move.
    strong = {"oracle_either_accuracy": 0.70, "headroom_over_best_single": 0.18}
    record = headroom_comparison(baseline, strong)
    assert record["more_room_for_routing"] is False
    assert "NOT more for the router" in record["statement"]
    assert "agrees with Gemma more often" in record["why_this_matters"]


def test_a_genuinely_complementary_expert_widens_the_ceiling():
    baseline = {"oracle_either_accuracy": 0.70, "headroom_over_best_single": 0.23}
    strong = {"oracle_either_accuracy": 0.76, "headroom_over_best_single": 0.26}
    record = headroom_comparison(baseline, strong)
    assert record["more_room_for_routing"] is True
    assert record["oracle_accuracy_delta"] == pytest.approx(0.06)
    assert record["headroom_delta"] == pytest.approx(0.03)
    assert "more for the router to exploit" in record["statement"]


# ============================================================
# The produced artefacts
# ============================================================

@pytest.mark.skipif(
    not (ANALYSIS / "phase_b1_comparison.json").exists(),
    reason="Phase B-1 has not been run",
)
def test_written_b1_used_identical_samples_and_no_test_data():
    record = json.loads(
        (ANALYSIS / "phase_b1_comparison.json").read_text(encoding="utf-8")
    )
    assert record["split"] == "validation"
    assert record["identical_samples"] is True
    assert record["test_data_accessed"] is False
    assert record["systems"]["audio_baseline"]["samples"] == \
        record["systems"]["audio_strong"]["samples"]
    # The frozen encoder must be charged, not hidden.
    assert record["latency"]["audio_strong"]["encoder_ms_per_sample"] is not None


@pytest.mark.skipif(
    not (ANALYSIS / "phase_c_aligned.json").exists(),
    reason="Phase C has not been run",
)
def test_written_phase_c_did_not_open_the_locked_pool():
    record = json.loads((ANALYSIS / "phase_c_aligned.json").read_text(encoding="utf-8"))
    assert record["split"] == "validation"
    assert record["locked_test_pool_opened"] is False
    assert record["test_data_accessed"] is False
    for spec in record["fusion_specs"].values():
        assert spec["selected_on_split"] == "validation"
        assert spec["uses_test_labels"] is False
