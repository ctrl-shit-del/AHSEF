"""The pilot orchestrator: analysis helpers and the test-lock enforcement."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ahsef.cli.run_stage2_pilot import (
    Pilot,
    build_parser,
    rank_association,
    uncertainty_bins,
)


def args(**overrides):
    values = {
        "stage": "select", "run": "t", "root": "experiments", "model": "gemma4:31b-cloud",
        "host": "http://localhost:11434", "prompt_version": "v2_explicit_json",
        "temperature": 0.0, "llm_seed": 42, "num_predict": 700, "size": 10, "seed": 42,
        "uncertainty_policy": "score_entropy",
        "threshold_objective": "target_stop_accuracy", "target_stop_accuracy": 0.6,
        "min_coverage": 0.05, "bins": 10, "provider": "replay", "store_text": "hash",
        "max_calls": None,
    }
    values.update(overrides)
    return type("Args", (), values)()


# ------------------------------------------------------------------- bins

def test_bins_partition_every_sample():
    rng = np.random.default_rng(0)
    uncertainty = rng.random(500)
    correct = rng.random(500) < (1 - uncertainty)
    rows = uncertainty_bins(uncertainty, correct)
    assert sum(row["samples"] for row in rows) == 500
    assert len(rows) == 10


def test_the_top_bin_is_closed_so_uncertainty_one_is_counted():
    rows = uncertainty_bins(np.array([1.0]), np.array([True]))
    assert rows[-1]["samples"] == 1
    assert rows[-1]["bin"].endswith("]")


def test_an_empty_bin_reports_none_not_zero_accuracy():
    """Zero accuracy in an empty bin would read as a real failure."""
    rows = uncertainty_bins(np.array([0.05, 0.06]), np.array([True, True]))
    assert rows[0]["samples"] == 2
    assert rows[5]["samples"] == 0
    assert rows[5]["accuracy"] is None
    assert rows[5]["error_rate"] is None


def test_accuracy_and_error_rate_are_complementary():
    rng = np.random.default_rng(1)
    uncertainty = rng.random(300)
    correct = rng.random(300) < 0.5
    for row in uncertainty_bins(uncertainty, correct):
        if row["samples"]:
            assert row["accuracy"] + row["error_rate"] == pytest.approx(1.0)


# ------------------------------------------------------------ association

def test_a_perfectly_predictive_uncertainty_scores_at_the_ceiling():
    uncertainty = np.linspace(0, 1, 200)
    correct = uncertainty < 0.5
    record = rank_association(uncertainty, correct)
    assert record["spearman_rho"] > 0.8
    assert record["auroc_uncertainty_predicts_error"] > 0.99


def test_an_uninformative_uncertainty_scores_near_chance():
    rng = np.random.default_rng(2)
    uncertainty = rng.random(2000)
    correct = rng.random(2000) < 0.5
    record = rank_association(uncertainty, correct)
    assert abs(record["spearman_rho"]) < 0.1
    assert 0.45 < record["auroc_uncertainty_predicts_error"] < 0.55


def test_an_inverted_uncertainty_is_reported_negative_not_hidden():
    uncertainty = np.linspace(0, 1, 200)
    correct = uncertainty > 0.5      # confident exactly when wrong
    record = rank_association(uncertainty, correct)
    assert record["spearman_rho"] < -0.8
    assert record["auroc_uncertainty_predicts_error"] < 0.01


def test_a_constant_uncertainty_yields_no_association():
    record = rank_association(np.full(50, 0.5), np.array([True, False] * 25))
    assert record["spearman_rho"] is None
    assert "insufficient variation" in record["note"]


def test_the_confidence_interval_brackets_the_estimate():
    uncertainty = np.linspace(0, 1, 400)
    correct = uncertainty < 0.5
    record = rank_association(uncertainty, correct)
    low, high = record["rho_ci95"]
    assert low <= record["spearman_rho"] <= high


# ------------------------------------------------------------- test lock

def test_the_test_stage_refuses_without_a_frozen_configuration(tmp_path):
    from src.ahsef.cli.run_stage2_pilot import stage_test

    pilot = Pilot(args(stage="test", root=str(tmp_path)))
    with pytest.raises(SystemExit, match="No frozen configuration"):
        stage_test(pilot)


def test_the_test_stage_refuses_a_configuration_that_drifted(tmp_path):
    from src.ahsef.cli.run_stage2_pilot import stage_test

    pilot = Pilot(args(stage="test", root=str(tmp_path)))
    pilot.frozen_path.write_text(json.dumps({
        "model": "gemma4:31b-cloud", "prompt_version": "v2_explicit_json",
        "temperature": 0.0, "llm_seed": 42, "uncertainty_policy": "score_entropy",
        "threshold": 0.5,
    }), encoding="utf-8")

    drifted = Pilot(args(stage="test", root=str(tmp_path), temperature=0.7))
    with pytest.raises(SystemExit, match="differs from the frozen one"):
        stage_test(drifted)

    swapped = Pilot(args(stage="test", root=str(tmp_path), model="llama3:8b"))
    with pytest.raises(SystemExit, match="differs from the frozen one"):
        stage_test(swapped)

    del pilot


def test_a_frozen_configuration_declares_no_test_labels_were_used(tmp_path):
    """Structural check on the artefact the lock depends on."""
    pilot = Pilot(args(root=str(tmp_path)))
    record = {
        "model": "gemma4:31b-cloud", "prompt_version": "v2_explicit_json",
        "temperature": 0.0, "llm_seed": 42, "uncertainty_policy": "score_entropy",
        "threshold": 0.5, "uses_test_labels": False,
    }
    pilot.frozen_path.write_text(json.dumps(record), encoding="utf-8")
    loaded = json.loads(pilot.frozen_path.read_text(encoding="utf-8"))
    assert loaded["uses_test_labels"] is False


# ---------------------------------------------------------------- layout

def test_the_pilot_writes_only_under_its_own_run_directory(tmp_path):
    pilot = Pilot(args(root=str(tmp_path)))
    for path in (pilot.manifest_path("validation"), pilot.baseline_path("test"),
                 pilot.transcript_path("validation"), pilot.frozen_path,
                 pilot.analysis_path("x.json")):
        assert str(pilot.base) in str(path)
    assert pilot.base.name == "t"


def test_the_default_model_and_prompt_are_the_pilot_ones():
    parsed = build_parser().parse_args(["--stage", "select"])
    assert parsed.model == "gemma4:31b-cloud"
    assert parsed.prompt_version == "v2_explicit_json"
    assert parsed.temperature == 0.0
    assert parsed.size == 1000


def test_an_unknown_stage_is_refused():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--stage", "publish"])
