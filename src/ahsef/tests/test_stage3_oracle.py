"""PHASE C: the oracle table, fusion, and the gain target it produces."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.fusion import FusionCompatibilityError, FusionSpec
from src.ahsef.hsig import HSIG_TARGET
from src.ahsef.stage3.oracle import (
    ORACLE_COLUMNS,
    OracleError,
    build_oracle_table,
    choose_fusion_spec,
    coverage_note,
    fuse,
    oracle_report,
    usable_text_ids,
)
from src.ahsef.tests.stage3_fixtures import make_audio_set, make_llm_set, scenario


@pytest.fixture
def material():
    data = scenario(n=80, seed=13)
    spec = FusionSpec(
        method="weighted_probability", weights={"text_llm": 0.5, "audio": 0.5},
        selected_on_split="validation",
    )
    fused = fuse(data["text"], data["audio"], spec, data["ids"])
    table = build_oracle_table(data["text"], data["audio"], fused, data["ids"])
    return {"data": data, "spec": spec, "fused": fused, "table": table}


# -------------------------------------------------------------------- fusion

def test_fusion_weights_may_not_be_chosen_on_test(material):
    data = material["data"]
    with pytest.raises(OracleError, match="quoted backwards"):
        choose_fusion_spec(data["text"], data["audio"], data["ids"], "test")


def test_fusion_weight_selection_is_recorded(material):
    data = material["data"]
    spec = choose_fusion_spec(data["text"], data["audio"], data["ids"], "validation")
    assert spec.selected_on_split == "validation"
    assert spec.selection["method"] == "exhaustive_grid_scan"
    assert spec.selection["scanned"]
    assert spec.to_dict()["uses_test_labels"] is False


def test_fusion_refuses_a_different_class_space(material):
    from src.ahsef.tests.conftest import make_prediction_set
    import torch

    data = material["data"]
    wesad = make_prediction_set(
        "physiology", data["ids"][:3], torch.zeros((3, 3), dtype=torch.float64),
        [0, 1, 2], split="validation", class_order=("baseline", "stress", "amusement"),
    )
    with pytest.raises(FusionCompatibilityError, match="different class spaces"):
        fuse(data["text"], wesad, material["spec"], data["ids"][:3])


def test_fusion_pairs_rows_by_id_not_position(material):
    """A shuffled candidate must fuse to the same result."""
    data = material["data"]
    shuffled_ids = list(reversed(data["ids"]))
    forward = fuse(data["text"], data["audio"], material["spec"], data["ids"])
    reverse = fuse(data["text"], data["audio"], material["spec"], shuffled_ids)
    forward_map = dict(zip(forward.sample_ids(), forward.predictions().tolist()))
    reverse_map = dict(zip(reverse.sample_ids(), reverse.predictions().tolist()))
    assert forward_map == reverse_map


# ------------------------------------------------------------- oracle table

def test_table_has_every_declared_column(material):
    for column in ORACLE_COLUMNS:
        assert column in material["table"].columns, column


def test_signed_gain_is_the_target_realisation(material):
    table = material["table"]
    expected = table["fused_correct"].astype(int) - table["text_correct"].astype(int)
    assert (table["signed_gain"] == expected).all()
    assert set(table["signed_gain"].unique()) <= {-1, 0, 1}


def test_y_gain_and_y_harm_are_mutually_exclusive(material):
    table = material["table"]
    assert not ((table["y_gain"] == 1) & (table["y_harm"] == 1)).any()


def test_delta_uncertainty_is_carried_but_marked_diagnostic(material):
    table = material["table"]
    assert "delta_uncertainty_diagnostic_only" in table.columns
    report = oracle_report(
        table, material["data"]["text"], material["data"]["audio"], material["fused"],
        material["data"]["ids"], material["spec"],
    )
    block = report["delta_uncertainty_diagnostic"]
    assert block["status"].startswith("REJECTED")
    assert "0.143" in block["why_rejected"]
    assert report["target"]["estimator_target"] == HSIG_TARGET


def test_table_refuses_misaligned_sets(material):
    """Different labels for the same id means they are not the same samples."""
    data = material["data"]
    broken = make_audio_set(
        data["ids"], data["audio"].frame["predicted_class"].tolist(),
        [(int(v) + 1) % 7 for v in data["labels"]],
    )
    with pytest.raises(OracleError, match="not the same samples"):
        build_oracle_table(data["text"], broken, material["fused"], data["ids"])


def test_audio_available_column_is_present(material):
    assert material["table"]["audio_available"].all()


# ---------------------------------------------------------------- reporting

def test_report_covers_every_required_phase_c_number(material):
    report = oracle_report(
        material["table"], material["data"]["text"], material["data"]["audio"],
        material["fused"], material["data"]["ids"], material["spec"],
    )
    for system in ("text_llm", "audio", "text_llm+audio"):
        block = report["systems"][system]
        for metric in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            assert metric in block
        assert len(block["per_class_f1"]) == 7
    outcome = report["acquisition_outcome"]
    for key in (
        "fixed_by_audio", "fixed_fraction", "broken_by_audio", "broken_fraction",
        "net_improvement", "mean_signed_gain",
    ):
        assert key in outcome
    assert len(report["per_class_improvement"]) == 7


def test_report_includes_small_count_intervals(material):
    report = oracle_report(
        material["table"], material["data"]["text"], material["data"]["audio"],
        material["fused"], material["data"]["ids"], material["spec"],
    )
    interval = report["acquisition_outcome"]["fixed_fraction_ci95"]
    assert interval[0] <= report["acquisition_outcome"]["fixed_fraction"] <= interval[1]


def test_net_improvement_is_fixes_minus_breaks(material):
    report = oracle_report(
        material["table"], material["data"]["text"], material["data"]["audio"],
        material["fused"], material["data"]["ids"], material["spec"],
    )
    outcome = report["acquisition_outcome"]
    assert outcome["net_improvement"] == (
        outcome["fixed_by_audio"] - outcome["broken_by_audio"]
    )


# ------------------------------------------------------------------ coverage

def test_unusable_llm_rows_are_excluded_and_counted():
    """A sample the LLM refused must be counted, never given a filler prediction."""
    ids = ["a", "b", "c", "d"]
    text = make_llm_set(ids, [0, 1, -1, 2], [0, 1, 2, 2], usable=[True, True, False, True])
    retained = usable_text_ids(text)
    assert retained == ["a", "b", "d"]
    note = coverage_note(text, retained)
    assert note["pooled_samples"] == 4
    assert note["usable_text_evidence"] == 3
    assert note["excluded_no_text_evidence"] == 1
    assert note["status_counts"]["abstain"] == 1
    assert "never assigned a filler prediction" in note["rule"]
