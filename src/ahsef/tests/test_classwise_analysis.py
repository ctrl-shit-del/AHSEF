"""PHASE A: class-wise modality analysis, and the comparisons it refuses to make."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ahsef.analysis.classwise import (
    ALIGNED_PAIRS,
    EMOTION_MODALITIES,
    AnalysisError,
    aligned_comparison,
    complementarity_summary,
    modality_profile,
    modality_profiles,
    specialization_verdicts,
)
from src.ahsef.tests.conftest import make_prediction_set
from src.common.labels import CANONICAL_EMOTION_CLASSES

ANALYSIS_DIR = Path("experiments/ahsef/analysis/classwise")


def one_hot(predictions, n_classes=7, strength=6.0):
    logits = torch.zeros((len(predictions), n_classes), dtype=torch.float64)
    for row, klass in enumerate(predictions):
        logits[row, int(klass)] = strength
    return logits


def pair_sets(labels, left_pred, right_pred, split="validation"):
    ids = [f"s{i}" for i in range(len(labels))]
    left = make_prediction_set("audio", ids, one_hot(left_pred), labels, split=split)
    right = make_prediction_set("text", ids, one_hot(right_pred), labels, split=split)
    return ids, left, right


# ------------------------------------------------------------------ profiles

def test_profile_reports_every_class():
    labels = [0, 1, 2, 3, 4, 5, 6]
    item = make_prediction_set("audio", [f"s{i}" for i in range(7)],
                               one_hot(labels), labels, split="validation")
    record = modality_profile(item)
    assert set(record["per_class"]) == set(CANONICAL_EMOTION_CLASSES)
    for entry in record["per_class"].values():
        for key in ("support", "precision", "recall", "f1"):
            assert key in entry
    assert len(record["confusion_matrix"]) == 7
    assert record["accuracy"] == pytest.approx(1.0)


def test_profile_declares_itself_non_comparable():
    """The single most important property of this table."""
    labels = [0, 1, 2, 3]
    item = make_prediction_set("audio", [f"s{i}" for i in range(4)],
                               one_hot(labels), labels, split="validation")
    record = modality_profile(item)
    assert record["comparable_across_modalities"] is False
    assert "not comparable" in record["note"]
    assert "different corpora" in record["note"]


def test_profile_excludes_unusable_predictions_and_counts_them():
    labels = [0, 1, 2, 3]
    item = make_prediction_set("text", [f"s{i}" for i in range(4)],
                               one_hot(labels), labels, split="validation")
    item.frame.loc[2, "predicted_class"] = -1
    record = modality_profile(item)
    assert record["samples"] == 3
    assert record["excluded_unusable"] == 1


def test_profiles_group_by_task_and_keep_label_spaces_apart():
    labels = [0, 1, 2, 3]
    emotion = make_prediction_set("audio", [f"a{i}" for i in range(4)],
                                  one_hot(labels), labels, split="validation")
    wesad = make_prediction_set(
        "physiology", [f"p{i}" for i in range(3)],
        one_hot([0, 1, 2], n_classes=3), [0, 1, 2], split="validation",
        class_order=("baseline", "stress", "amusement"),
    )
    record = modality_profiles({"audio": emotion, "physiology": wesad})
    assert record["profiles"]["audio"]["task"] == "emotion_7class"
    assert record["profiles"]["physiology"]["task"] == "wesad_state_3class"
    assert set(record["tasks"]) == {"emotion_7class", "wesad_state_3class"}
    assert record["comparable_across_modalities"] is False


# ------------------------------------------------------------ aligned pairs

def test_comparison_refuses_different_class_spaces():
    labels = [0, 1, 2]
    emotion = make_prediction_set("audio", ["a", "b", "c"], one_hot(labels), labels,
                                  split="validation")
    wesad = make_prediction_set(
        "physiology", ["a", "b", "c"], one_hot(labels, n_classes=3), labels,
        split="validation", class_order=("baseline", "stress", "amusement"),
    )
    with pytest.raises(AnalysisError, match="class spaces differ"):
        aligned_comparison(emotion, wesad, ["a", "b", "c"])


def test_comparison_refuses_disagreeing_labels():
    """Two sets that disagree about truth are not describing the same samples."""
    ids = ["a", "b", "c"]
    left = make_prediction_set("audio", ids, one_hot([0, 1, 2]), [0, 1, 2],
                               split="validation")
    right = make_prediction_set("text", ids, one_hot([0, 1, 2]), [0, 1, 3],
                                split="validation")
    with pytest.raises(AnalysisError, match="not describing the same recordings"):
        aligned_comparison(left, right, ids)


def test_comparison_refuses_an_empty_pool():
    ids, left, right = pair_sets([0, 1], [0, 1], [0, 1])
    with pytest.raises(AnalysisError, match="empty aligned pool"):
        aligned_comparison(left, right, [])


def test_outcome_split_is_exhaustive_and_correct():
    #        both ok   left ok   right ok   both wrong
    labels = [0,        1,        2,         3]
    left =   [0,        1,        5,         6]
    right =  [0,        4,        2,         5]
    ids, a, b = pair_sets(labels, left, right)
    record = aligned_comparison(a, b, ids, "audio", "text")
    outcome = record["overall"]["outcome"]
    assert outcome["both_correct"] == 1
    assert outcome["left_correct_right_wrong"] == 1
    assert outcome["right_correct_left_wrong"] == 1
    assert outcome["both_wrong"] == 1
    assert outcome["samples"] == 4
    total = (outcome["both_correct"] + outcome["left_correct_right_wrong"]
             + outcome["right_correct_left_wrong"] + outcome["both_wrong"])
    assert total == 4


def test_correction_opportunity_is_the_other_modality_being_right():
    labels = [0, 1, 2, 3]
    left = [0, 1, 5, 6]     # right on 0,1
    right = [0, 4, 2, 5]    # right on 0,2
    ids, a, b = pair_sets(labels, left, right)
    record = aligned_comparison(a, b, ids, "audio", "text")
    # audio is wrong on samples 2 and 3; text rescues only sample 2.
    assert record["overall"]["correction_opportunity_for_left"] == 1
    # text is wrong on 1 and 3; audio rescues only sample 1.
    assert record["overall"]["correction_opportunity_for_right"] == 1


def test_oracle_ceiling_is_the_union_of_correct():
    labels = [0, 1, 2, 3]
    left = [0, 1, 5, 6]
    right = [0, 4, 2, 5]
    ids, a, b = pair_sets(labels, left, right)
    record = aligned_comparison(a, b, ids, "audio", "text")
    # samples 0,1,2 recoverable by one of the two; sample 3 by neither.
    assert record["overall"]["ceiling_if_either_were_trusted"] == pytest.approx(0.75)


def test_comparison_pairs_by_id_not_position():
    labels = [0, 1, 2, 3]
    ids, a, b = pair_sets(labels, [0, 1, 2, 3], [0, 1, 2, 3])
    forward = aligned_comparison(a, b, ids, "audio", "text")
    reverse = aligned_comparison(a, b, list(reversed(ids)), "audio", "text")
    assert forward["overall"]["outcome"] == reverse["overall"]["outcome"]


def test_perfect_agreement_leaves_no_correction_opportunity():
    labels = [0, 1, 2, 3, 0, 1]
    ids, a, b = pair_sets(labels, labels, labels)
    record = aligned_comparison(a, b, ids, "audio", "text")
    assert record["overall"]["disagreement_rate"] == 0.0
    assert record["overall"]["correction_opportunity_for_left"] == 0
    assert record["overall"]["correction_opportunity_for_right"] == 0
    assert record["overall"]["ceiling_if_either_were_trusted"] == pytest.approx(1.0)


# ------------------------------------------------------------ specialization

def test_thin_support_declines_to_name_a_winner():
    """A class with a handful of samples must not produce a verdict."""
    labels = [4] * 5 + [0] * 30
    left = [4] * 5 + [0] * 30          # perfect
    right = [0] * 5 + [0] * 30         # wrong on every fear sample
    ids, a, b = pair_sets(labels, left, right)
    record = aligned_comparison(a, b, ids, "audio", "text")
    fear = record["per_class"]["fear"]
    assert fear["support"] == 5
    assert fear["stronger"]["winner"] is None
    assert "support is 5" in fear["stronger"]["reason"]


def test_a_narrow_f1_gap_is_called_a_tie():
    rng = np.random.default_rng(0)
    labels = list(rng.integers(0, 2, 200))
    ids, a, b = pair_sets(labels, labels, labels)
    record = aligned_comparison(a, b, ids, "audio", "text")
    assert record["per_class"]["neutral"]["stronger"]["winner"] is None
    assert "tie" in record["per_class"]["neutral"]["stronger"]["reason"]


def test_a_consistent_chain_is_not_a_disagreement():
    """'text > video' and 'text_llm > text' rank consistently; earlier code did not."""
    comparisons = {
        "text+video": {
            "left": "text", "right": "video", "split": "validation",
            "per_class": {
                name: {
                    "support": 100, "left_f1": 0.5, "right_f1": 0.2,
                    "stronger": {"winner": "text", "reason": "x"},
                }
                for name in CANONICAL_EMOTION_CLASSES
            },
        },
        "text+text_llm": {
            "left": "text", "right": "text_llm", "split": "validation",
            "per_class": {
                name: {
                    "support": 100, "left_f1": 0.2, "right_f1": 0.6,
                    "stronger": {"winner": "text_llm", "reason": "y"},
                }
                for name in CANONICAL_EMOTION_CLASSES
            },
        },
    }
    record = specialization_verdicts(comparisons)
    for name in CANONICAL_EMOTION_CLASSES:
        entry = record["per_class"][name]
        assert entry["verdict"] == "text_llm", name
        assert entry["undefeated"] == ["text_llm"]


def test_two_undefeated_modalities_yield_no_verdict():
    comparisons = {
        "audio+text": {
            "left": "audio", "right": "text", "split": "validation",
            "per_class": {
                "happy": {"support": 100, "left_f1": 0.6, "right_f1": 0.2,
                          "stronger": {"winner": "audio", "reason": "x"}},
            },
        },
        "image+video": {
            "left": "image", "right": "video", "split": "validation",
            "per_class": {
                "happy": {"support": 100, "left_f1": 0.6, "right_f1": 0.2,
                          "stronger": {"winner": "image", "reason": "y"}},
            },
        },
    }
    entry = specialization_verdicts(comparisons)["per_class"]["happy"]
    assert entry["verdict"] is None
    assert entry["undefeated"] == ["audio", "image"]
    assert "do not rank them" in entry["reason"]


def test_a_cycle_yields_no_verdict():
    def block(left, right, winner):
        return {
            "left": left, "right": right, "split": "validation",
            "per_class": {"happy": {"support": 100, "left_f1": 0.6, "right_f1": 0.2,
                                    "stronger": {"winner": winner, "reason": "z"}}},
        }
    comparisons = {
        "a+b": block("audio", "text", "audio"),
        "b+c": block("text", "video", "text"),
        "c+a": block("video", "audio", "video"),
    }
    entry = specialization_verdicts(comparisons)["per_class"]["happy"]
    assert entry["verdict"] is None
    assert "cycle" in entry["reason"]


def test_no_evidence_yields_an_explicit_undecided():
    record = specialization_verdicts({})
    assert record["classes_with_a_verdict"] == []
    assert set(record["classes_without_a_verdict"]) == set(CANONICAL_EMOTION_CLASSES)
    for entry in record["per_class"].values():
        assert entry["verdict"] is None
        assert "no aligned comparison" in entry["reason"]


def test_coverage_caveat_names_the_uncompared_modalities():
    caveat = specialization_verdicts({})["coverage_caveat"]
    assert "image" in caveat and "physiology" in caveat
    assert "audio+text" in caveat and "text+video" in caveat


# ---------------------------------------------------------- complementarity

def test_headroom_is_oracle_minus_best_single():
    labels = [0, 1, 2, 3]
    ids, a, b = pair_sets(labels, [0, 1, 5, 6], [0, 4, 2, 5])
    comparisons = {"audio+text": aligned_comparison(a, b, ids, "audio", "text")}
    entry = complementarity_summary(comparisons)["pairs"]["audio+text"]
    assert entry["headroom_over_best_single"] == pytest.approx(
        entry["oracle_either_accuracy"] - entry["best_single_accuracy"]
    )
    assert entry["best_single_accuracy"] == pytest.approx(0.5)


def test_declared_pairs_match_stage1_findings():
    assert ALIGNED_PAIRS == (("audio", "text"), ("text", "video"))
    assert "physiology" not in EMOTION_MODALITIES


# --------------------------------------------------- the produced artefacts

@pytest.mark.skipif(
    not (ANALYSIS_DIR / "classwise_validation.json").exists(),
    reason="run the classwise analysis first",
)
def test_written_analysis_is_read_only_and_complete():
    record = json.loads(
        (ANALYSIS_DIR / "classwise_validation.json").read_text(encoding="utf-8")
    )
    assert record["integrity"]["reads_only"] is True
    assert record["integrity"]["models_loaded"] == 0
    assert record["integrity"]["feeds_model_selection"] is False
    assert record["integrity"]["feeds_threshold_selection"] is False
    assert record["integrity"]["feeds_hsig_training"] is False
    assert record["split"] == "validation"
    assert set(record["modality_profiles"]["profiles"]) >= {"audio", "text", "image"}
    assert "audio+text" in record["aligned_comparisons"]


@pytest.mark.skipif(
    not (ANALYSIS_DIR / "classwise_validation.json").exists(),
    reason="run the classwise analysis first",
)
def test_written_analysis_only_compares_aligned_pairs():
    """No head-to-head may exist for a pair with no shared samples."""
    record = json.loads(
        (ANALYSIS_DIR / "classwise_validation.json").read_text(encoding="utf-8")
    )
    permitted = {"audio+text", "text+video", "audio+text_llm", "text+text_llm"}
    assert set(record["aligned_comparisons"]) <= permitted
    for block in record["aligned_comparisons"].values():
        assert block["aligned"] is True
        assert block["samples"] > 0
