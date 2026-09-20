"""PHASE J/K: the freeze, the drift checks, and the Stage 1/2 regression guards.

These are the tests that make "the test set was locked" a property of the code
rather than a claim in a report.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from src.ahsef.cli.stage3_stages import STAGE2_FROZEN_TAU, _assert_no_drift
from src.ahsef.fusion import FusionSpec
from src.ahsef.stage3 import STAGE3_CLAIM, STAGE3_CLAIM_LIMITS
from src.ahsef.stage3.features import build_features
from src.ahsef.stage3.hsig_model import (
    HSIGLeakageError,
    Stage3HSIG,
    out_of_fold_gain,
    targets_from_oracle,
)
from src.ahsef.stage3.objective import (
    SELECTED_OBJECTIVE,
    ThresholdLeakageError,
    select_gate_threshold,
)
from src.ahsef.stage3.oracle import OracleError, build_oracle_table, choose_fusion_spec, fuse
from src.ahsef.tests.stage3_fixtures import scenario
from src.common.labels import CANONICAL_EMOTION_CLASSES

STAGE1 = Path("experiments/ahsef/stage1")
STAGE2 = Path("experiments/ahsef/stage2_llm")
STAGE3 = Path("experiments/ahsef/stage3_text_audio")


@pytest.fixture
def fitted():
    data = scenario(n=140, seed=21)
    spec = FusionSpec(
        method="weighted_probability", weights={"text_llm": 0.5, "audio": 0.5},
        selected_on_split="validation",
    )
    fused = fuse(data["text"], data["audio"], spec, data["ids"])
    oracle = build_oracle_table(data["text"], data["audio"], fused, data["ids"])
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(data["text"].frame, uncertainty)
    targets = targets_from_oracle(oracle, "validation")
    return {
        "model": Stage3HSIG.fit(features, targets), "oracle": oracle,
        "features": features, "targets": targets, "data": data,
    }


def frozen_config(model: Stage3HSIG) -> dict:
    return {
        "frozen_at": "2026-08-30T00:00:00",
        "text_llm": {
            "model": "gemma4:31b-cloud", "prompt_version": "v2_explicit_json",
            "llm_seed": 42, "temperature": 0.0,
        },
        "uncertainty_policy": "score_entropy",
        "hsig": {
            "estimator_type": model.estimator_type, "feature_set": model.feature_set,
            "seed": 42, "folds": 5, "fingerprint_sha256": model.fingerprint(),
        },
        "fusion": {"method": "weighted_probability"},
        "gate": {
            "objective": SELECTED_OBJECTIVE, "threshold": 0.1,
            "alpha_acquisition": 0.05, "beta_latency": 0.02,
        },
        "ugapr": {"lambda_cost": 0.10, "mu_latency": 0.10},
    }


def args(**overrides) -> Namespace:
    base = dict(
        uncertainty_policy="score_entropy", model="gemma4:31b-cloud",
        prompt_version="v2_explicit_json", llm_seed=42, temperature=0.0,
        lambda_cost=0.10, mu_latency=0.10, alpha_acquisition=0.05, beta_latency=0.02,
        gate_objective=SELECTED_OBJECTIVE, hsig_seed=42, hsig_folds=5,
        fusion_method="weighted_probability",
    )
    base.update(overrides)
    return Namespace(**base)


# ------------------------------------------------------------- drift refusal

def test_a_matching_configuration_passes(fitted):
    _assert_no_drift(frozen_config(fitted["model"]), args(), fitted["model"])


@pytest.mark.parametrize("field,value", [
    ("lambda_cost", 0.5),
    ("mu_latency", 0.5),
    ("alpha_acquisition", 0.2),
    ("beta_latency", 0.1),
    ("gate_objective", "accuracy"),
    ("uncertainty_policy", "llm_confidence"),
    ("model", "some-other-model"),
    ("prompt_version", "v1"),
    ("llm_seed", 7),
    ("temperature", 0.7),
    ("hsig_seed", 1),
    ("hsig_folds", 10),
    ("fusion_method", "log_opinion_pool"),
])
def test_every_frozen_value_is_checked(fitted, field, value):
    with pytest.raises(SystemExit, match="CONFIGURATION DRIFT"):
        _assert_no_drift(frozen_config(fitted["model"]), args(**{field: value}), fitted["model"])


def test_a_refit_on_different_data_is_refused(fitted):
    """Retraining HSIG on new data after the freeze must abort the evaluation.

    Refitting on the *same* data is bit-identical (lbfgs is deterministic), and
    the fingerprint correctly does not move for that. What must be caught is a
    refit that saw anything else -- which is what retraining at test time is.
    """
    targets = fitted["targets"]
    half = len(targets.sample_ids) // 2
    subset = type(targets)(
        sample_ids=targets.sample_ids[:half],
        text_correct=targets.text_correct[:half],
        fused_correct=targets.fused_correct[:half],
        split="validation",
    )
    refitted = Stage3HSIG.fit(fitted["features"], subset)
    assert refitted.fingerprint() != fitted["model"].fingerprint()
    with pytest.raises(SystemExit) as error:
        _assert_no_drift(frozen_config(fitted["model"]), args(), refitted)
    assert "hsig_fingerprint" in str(error.value)
    assert "refitted since the freeze" in str(error.value)


def test_an_edited_coefficient_is_refused(fitted):
    """Hand-tuning the frozen estimator must be caught by the fingerprint."""
    tampered = Stage3HSIG.load(
        fitted["model"].save(Path(__import__("tempfile").mkdtemp()) / "hsig.json")
    )
    tampered.components["fused_correct"]["coefficients"][0] += 0.25
    with pytest.raises(SystemExit, match="hsig_fingerprint"):
        _assert_no_drift(frozen_config(fitted["model"]), args(), tampered)


def test_a_changed_estimator_type_is_refused(fitted):
    swapped = Stage3HSIG.fit(
        fitted["features"], fitted["targets"], estimator_type="fix_logistic"
    )
    with pytest.raises(SystemExit) as error:
        _assert_no_drift(frozen_config(fitted["model"]), args(), swapped)
    assert "hsig_estimator_type" in str(error.value) or "hsig_fingerprint" in str(error.value)


def test_a_changed_feature_set_is_refused(fitted):
    data = fitted["data"]
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    wide = Stage3HSIG.fit(
        build_features(data["text"].frame, uncertainty, "evidence_plus_class"),
        fitted["targets"], feature_set="evidence_plus_class",
    )
    with pytest.raises(SystemExit) as error:
        _assert_no_drift(frozen_config(fitted["model"]), args(), wide)
    assert "feature_set" in str(error.value) or "fingerprint" in str(error.value)


# ------------------------------------------------------ selection-time refusal

def test_threshold_cannot_be_reselected_on_test(fitted):
    oracle = fitted["oracle"]
    with pytest.raises(ThresholdLeakageError):
        select_gate_threshold(
            np.zeros(len(oracle)),
            oracle["text_prediction"].to_numpy(dtype=int),
            oracle["fused_prediction"].to_numpy(dtype=int),
            oracle["true_class"].to_numpy(dtype=int),
            split="test", class_names=CANONICAL_EMOTION_CLASSES,
            text_latency_ms=1.0, audio_latency_ms=1.0,
        )


def test_hsig_cannot_be_refitted_on_test(fitted):
    targets = fitted["targets"]
    leaky = type(targets)(
        sample_ids=targets.sample_ids, text_correct=targets.text_correct,
        fused_correct=targets.fused_correct, split="test",
    )
    with pytest.raises(HSIGLeakageError):
        Stage3HSIG.fit(fitted["features"], leaky)
    with pytest.raises(HSIGLeakageError):
        out_of_fold_gain(fitted["features"], leaky)


def test_fusion_weights_cannot_be_tuned_on_test(fitted):
    data = fitted["data"]
    with pytest.raises(OracleError):
        choose_fusion_spec(data["text"], data["audio"], data["ids"], "test")


# ----------------------------------------------------- Stage 1/2 regressions

@pytest.mark.skipif(not STAGE1.exists(), reason="no Stage 1 artefacts")
def test_stage1_artefacts_are_intact():
    for name in (
        "alignment/alignment_report.json", "alignment/alignment_matrix.json",
        "predictions/audio__validation.parquet", "predictions/text__validation.parquet",
        "calibration/audio.json", "provenance.json",
    ):
        assert (STAGE1 / name).exists(), name


@pytest.mark.skipif(not STAGE1.exists(), reason="no Stage 1 artefacts")
def test_stage1_alignment_conclusion_is_unchanged():
    """Stage 3 must not have altered the finding it is built on."""
    report = json.loads(
        (STAGE1 / "alignment" / "alignment_report.json").read_text(encoding="utf-8")
    )
    for split in ("validation", "test"):
        candidates = report["fusability_from_anchor"][split]["candidates"]
        assert candidates["text"]["fusable"] is True
        for other in ("image", "video", "physiology"):
            assert candidates[other]["fusable"] is False
        assert "label spaces differ" in " ".join(candidates["physiology"]["blockers"])


@pytest.mark.skipif(not STAGE2.exists(), reason="no Stage 2 artefacts")
def test_stage2_freeze_is_untouched():
    frozen = json.loads((STAGE2 / "frozen_config.json").read_text(encoding="utf-8"))
    assert frozen["threshold"] == pytest.approx(STAGE2_FROZEN_TAU)
    assert frozen["uncertainty_policy"] == "score_entropy"
    assert frozen["model"] == "gemma4:31b-cloud"
    assert frozen["prompt_version"] == "v2_explicit_json"
    assert frozen["uses_test_labels"] is False


@pytest.mark.skipif(not STAGE2.exists(), reason="no Stage 2 artefacts")
def test_stage2_transcripts_are_preserved():
    for name in ("transcript_validation.jsonl", "transcript_test.jsonl"):
        path = STAGE2 / "llm" / name
        assert path.exists()
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["record"] == "header"
        assert header["model"] == "gemma4:31b-cloud"


def test_stage3_inherits_rather_than_reselects_the_stage2_policy():
    from src.ahsef.stage3.llm_pass import FROZEN_LLM_CONFIG

    assert FROZEN_LLM_CONFIG["uncertainty_policy"] == "score_entropy"
    assert FROZEN_LLM_CONFIG["model"] == "gemma4:31b-cloud"
    assert FROZEN_LLM_CONFIG["prompt_version"] == "v2_explicit_json"
    assert FROZEN_LLM_CONFIG["temperature"] == 0.0
    assert FROZEN_LLM_CONFIG["llm_seed"] == 42
    assert "never how it is asked" in FROZEN_LLM_CONFIG["note"]


def test_stage3_does_not_reuse_the_stage2_gate_objective():
    from src.ahsef.gate import THRESHOLD_OBJECTIVES
    from src.ahsef.stage3.objective import GATE_OBJECTIVES

    assert "target_stop_accuracy" in THRESHOLD_OBJECTIVES
    assert "target_stop_accuracy" not in GATE_OBJECTIVES


# ------------------------------------------------------------------- claim

def test_the_declared_claim_is_the_defensible_one():
    assert "whether additional audio evidence is worth acquiring" in STAGE3_CLAIM
    assert "best modality" not in STAGE3_CLAIM
    assert "proof of concept" in STAGE3_CLAIM_LIMITS
    assert "Physiology" in STAGE3_CLAIM_LIMITS


# ------------------------------------------- the produced frozen config

@pytest.mark.skipif(
    not (STAGE3 / "frozen_config.json").exists(), reason="run --stage freeze first"
)
def test_written_frozen_config_records_every_required_field():
    frozen = json.loads(
        (STAGE3 / "frozen_config.json").read_text(encoding="utf-8")
    )
    assert frozen["uses_test_labels"] is False
    assert frozen["text_llm"]["model"]
    assert frozen["text_llm"]["prompt_version"]
    assert frozen["audio_baseline"]["checkpoint_sha256"]
    assert frozen["audio_baseline"]["retrained_by_stage3"] is False
    assert frozen["uncertainty_policy"]
    assert frozen["hsig"]["estimator_type"]
    assert frozen["hsig"]["features"]
    assert frozen["hsig"]["target_definition"]["formula"]
    assert frozen["hsig"]["fingerprint_sha256"]
    assert frozen["fusion"]["method"] and frozen["fusion"]["weights"]
    assert frozen["gate"]["objective"] and frozen["gate"]["threshold"] is not None
    assert frozen["ugapr"]["lambda_cost"] is not None
    assert frozen["ugapr"]["mu_latency"] is not None
    assert frozen["costs"]["definition"]["compute_units"]
    assert frozen["latency"]["text_llm_ms"] and frozen["latency"]["audio_ms"]
    assert frozen["alignment"]["validation_pool_fingerprint_sha256"]
    assert frozen["alignment"]["test_pool_fingerprint_sha256"]
    assert frozen["seeds"]["hsig_seed"] is not None
    assert frozen["git_commit"]
    assert frozen["environment"]["python"]


@pytest.mark.skipif(
    not (STAGE3 / "frozen_config.json").exists(), reason="run --stage freeze first"
)
def test_frozen_hsig_fingerprint_matches_the_model_on_disk():
    frozen = json.loads((STAGE3 / "frozen_config.json").read_text(encoding="utf-8"))
    model = Stage3HSIG.load(STAGE3 / "hsig" / "hsig_model.json")
    assert model.fingerprint() == frozen["hsig"]["fingerprint_sha256"]
    assert model.fitted_on_split == "validation"


# --------------------------------------- the acquisition price is frozen too

def test_cost_model_rebuilds_exactly_from_the_frozen_record():
    """Re-measuring the price on the evaluation split would move the threshold."""
    from src.ahsef.costs import CostModel, ModalityCost
    from src.ahsef.stage3.costs import cost_model_from_frozen

    original = CostModel({
        "audio": ModalityCost("audio", 27.9, 1.2e8, 120_000, 1000, "validation", 509),
        "text_llm": ModalityCost("text_llm", 3449.7, 2.3e13, 32_700_000_000, 700,
                                 "validation", 509),
    })
    frozen = {"costs": original.table()}
    rebuilt = cost_model_from_frozen(frozen)
    for modality in ("audio", "text_llm"):
        assert rebuilt.costs[modality].latency_ms == original.costs[modality].latency_ms
        assert rebuilt.costs[modality].compute_units == \
            original.costs[modality].compute_units
        assert rebuilt.normalized_cost(modality) == \
            pytest.approx(original.normalized_cost(modality))
        assert rebuilt.normalized_latency(modality) == \
            pytest.approx(original.normalized_latency(modality))


@pytest.mark.skipif(
    not (STAGE3 / "reports" / "locked_test.json").exists(), reason="run --stage test first"
)
def test_the_locked_test_priced_audio_from_the_frozen_config():
    record = json.loads(
        (STAGE3 / "reports" / "locked_test.json").read_text(encoding="utf-8")
    )
    frozen = json.loads((STAGE3 / "frozen_config.json").read_text(encoding="utf-8"))
    assert record["frozen_config"]["reselected_on_test"] is False
    assert record["frozen_config"]["hsig_refitted_on_test"] is False
    assert record["frozen_config"]["fusion_weights_tuned_on_test"] is False
    assert record["frozen_config"]["threshold"] == frozen["gate"]["threshold"]
    assert record["frozen_config"]["hsig_fingerprint_sha256"] == \
        frozen["hsig"]["fingerprint_sha256"]
