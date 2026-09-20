"""Evaluation over prediction sets, and the modality x emotion table."""

from __future__ import annotations

import pytest
import torch

from src.ahsef.evaluation import (
    agreement,
    compare_reports,
    evaluate_prediction_set,
    full_report,
    modality_emotion_table,
    per_class_f1_vector,
    render_modality_emotion_table,
    uncertainty_summary,
)
from src.ahsef.tests.conftest import EMOTION_CLASSES, make_prediction_set


POOL = [f"s{i}" for i in range(6)]


def test_a_perfect_model_scores_one(anchor_set):
    logits = torch.zeros(6, 7, dtype=torch.float64)
    labels = anchor_set.frame["true_class"].tolist()
    for row, label in enumerate(labels):
        logits[row, label] = 10.0
    perfect = make_prediction_set("audio", POOL, logits, labels)
    metrics = evaluate_prediction_set(perfect)
    assert metrics["accuracy"] == pytest.approx(1.0)
    # Macro-F1 averages over all seven declared classes, including the four
    # with no support here, so it is deliberately below 1.
    assert metrics["per_class"]["neutral"]["f1"] == pytest.approx(1.0)


def test_metrics_use_the_declared_class_order(anchor_set):
    metrics = evaluate_prediction_set(anchor_set)
    assert metrics["class_order"] == list(EMOTION_CLASSES)
    assert list(metrics["per_class"]) == list(EMOTION_CLASSES)


def test_uncertainty_summary_separates_correct_from_wrong(anchor_set):
    summary = uncertainty_summary(anchor_set)
    assert "mean_uncertainty_when_correct" in summary
    assert "mean_uncertainty_when_wrong" in summary
    assert summary["uncertainty_separation"] == pytest.approx(
        summary["mean_uncertainty_when_wrong"] - summary["mean_uncertainty_when_correct"]
    )


def test_full_report_carries_metrics_uncertainty_and_cost(anchor_set):
    report = full_report(anchor_set, name="audio")
    assert set(report) >= {"name", "metrics", "uncertainty", "cost", "provenance"}
    assert report["cost"]["mean_latency_ms"] == pytest.approx(1.0)


# ------------------------------------------------------------- complementarity

def test_agreement_counts_the_four_outcome_cells(anchor_set, candidate_set):
    record = agreement(anchor_set, candidate_set, POOL)
    total = (
        record["both_correct"] + record["only_left_correct"]
        + record["only_right_correct"] + record["both_wrong"]
    )
    assert total == 6
    # The candidate rescues s1 and s3, which is the headroom any fusion has.
    assert record["only_right_correct"] >= 2
    assert record["complementarity_headroom"] == pytest.approx(
        record["only_right_correct"] / 6
    )


def test_a_model_agreeing_with_itself_has_no_headroom(anchor_set):
    twin = make_prediction_set(
        "text", POOL, anchor_set.logits(), anchor_set.frame["true_class"].tolist()
    )
    record = agreement(anchor_set, twin, POOL)
    assert record["prediction_agreement_rate"] == pytest.approx(1.0)
    assert record["only_right_correct"] == 0
    assert record["complementarity_headroom"] == pytest.approx(0.0)


# ---------------------------------------------------- modality x emotion

def test_table_has_one_row_per_modality_and_one_column_per_class(
    anchor_set, candidate_set
):
    reports = {
        "audio": full_report(anchor_set, "audio"),
        "text": full_report(candidate_set, "text"),
    }
    table = modality_emotion_table(reports)
    assert sorted(table["rows"]) == ["audio", "text"]
    assert list(table["class_order"]) == list(EMOTION_CLASSES)
    assert set(table["rows"]["audio"]) == set(EMOTION_CLASSES)


def test_table_names_best_second_and_worst_per_emotion(anchor_set, candidate_set):
    reports = {
        "audio": full_report(anchor_set, "audio"),
        "text": full_report(candidate_set, "text"),
    }
    ranking = modality_emotion_table(reports)["ranking_per_emotion"]
    for emotion in EMOTION_CLASSES:
        entry = ranking[emotion]
        assert entry["best_value"] >= entry["worst_value"]
        assert entry["spread"] == pytest.approx(entry["best_value"] - entry["worst_value"])


def test_a_different_label_space_is_excluded_not_forced_into_the_table(
    anchor_set, candidate_set, wesad_set
):
    reports = {
        "audio": full_report(anchor_set, "audio"),
        "text": full_report(candidate_set, "text"),
        "physiology": full_report(wesad_set, "physiology"),
    }
    table = modality_emotion_table(reports)
    assert "physiology" not in table["rows"]
    assert any("physiology" in names for names in table["excluded_label_spaces"].values())


def test_table_renders_as_fixed_width_text(anchor_set, candidate_set):
    table = modality_emotion_table({
        "audio": full_report(anchor_set, "audio"),
        "text": full_report(candidate_set, "text"),
    })
    rendered = render_modality_emotion_table(table)
    assert "neutral" in rendered and "audio" in rendered and "best" in rendered


def test_compare_reports_flattens_to_one_row_each(anchor_set, candidate_set):
    rows = compare_reports([
        full_report(anchor_set, "audio"), full_report(candidate_set, "text"),
    ])
    assert [row["name"] for row in rows] == ["audio", "text"]
    assert all("macro_f1" in row and "mean_latency_ms" in row for row in rows)


def test_per_class_vector_matches_the_full_report(anchor_set):
    vector = per_class_f1_vector(anchor_set)
    report = evaluate_prediction_set(anchor_set)
    for name, value in vector.items():
        assert value == pytest.approx(report["per_class"][name]["f1"])
