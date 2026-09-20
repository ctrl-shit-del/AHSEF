"""The milestone estimator: held constant so Phase D isolates the target."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ahsef.milestone.estimator import (
    ALPHA_GRID,
    DEFAULT_FOLDS,
    MILESTONE_HSIG_SCHEMA,
    EstimatorLeakageError,
    MilestoneHSIG,
    assert_fit_split,
    out_of_fold_scores,
    select_alpha,
    spearman,
)
from src.ahsef.stage3.features import DEFAULT_FEATURE_SET, FEATURE_SETS


def features_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the declared feature sets name."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"sample_id": [f"s{i:04d}" for i in range(n)]})
    for name in FEATURE_SETS["evidence_plus_class"]:
        frame[name] = rng.random(n)
    return frame


def linear_target(frame: pd.DataFrame) -> np.ndarray:
    """A target the estimator can actually recover, so failures are real."""
    return (
        2.0 * frame["uncertainty"].to_numpy()
        - 1.0 * frame["score_top1"].to_numpy()
        + 0.05 * np.random.default_rng(1).normal(size=len(frame))
    )


# ============================================================
# Leakage
# ============================================================

def test_fitting_on_test_is_refused():
    with pytest.raises(EstimatorLeakageError, match="may only be fitted on"):
        assert_fit_split("test")
    assert assert_fit_split("validation") == "validation"
    assert assert_fit_split("train") == "train"


def test_out_of_fold_refuses_the_test_split():
    frame = features_frame(60)
    with pytest.raises(EstimatorLeakageError):
        out_of_fold_scores(
            frame, frame["sample_id"].tolist(), linear_target(frame), "test"
        )


def test_fit_refuses_the_test_split():
    frame = features_frame(60)
    with pytest.raises(EstimatorLeakageError):
        MilestoneHSIG.fit(
            frame, frame["sample_id"].tolist(), linear_target(frame), "t", "test"
        )


def test_alpha_selection_refuses_the_test_split():
    frame = features_frame(60)
    with pytest.raises(EstimatorLeakageError):
        select_alpha(frame, frame["sample_id"].tolist(), linear_target(frame), "test")


def test_a_label_bearing_feature_set_is_refused():
    frame = features_frame(30)
    frame["true_class"] = 0
    from src.ahsef.stage3.features import FeatureLeakageError, assert_label_free

    with pytest.raises(FeatureLeakageError):
        assert_label_free(frame, ["uncertainty", "true_class"])


# ============================================================
# Out-of-fold behaviour
# ============================================================

def test_out_of_fold_scores_every_sample_exactly_once():
    frame = features_frame(120)
    scores = out_of_fold_scores(
        frame, frame["sample_id"].tolist(), linear_target(frame), "validation"
    )
    assert scores.shape == (120,)
    assert np.isfinite(scores).all()


def test_out_of_fold_recovers_a_learnable_signal():
    frame = features_frame(300)
    target = linear_target(frame)
    scores = out_of_fold_scores(
        frame, frame["sample_id"].tolist(), target, "validation"
    )
    assert spearman(scores, target) > 0.8


def test_out_of_fold_differs_from_in_sample():
    """If they matched, calling the scores 'out of fold' would be empty."""
    frame = features_frame(150)
    target = linear_target(frame)
    ids = frame["sample_id"].tolist()
    model = MilestoneHSIG.fit(frame, ids, target, "t", "validation")
    assert not np.allclose(
        model.predict(frame, ids),
        out_of_fold_scores(frame, ids, target, "validation"),
    )


def test_out_of_fold_is_deterministic_for_a_seed():
    frame = features_frame(100)
    target = linear_target(frame)
    ids = frame["sample_id"].tolist()
    first = out_of_fold_scores(frame, ids, target, "validation", seed=7)
    second = out_of_fold_scores(frame, ids, target, "validation", seed=7)
    third = out_of_fold_scores(frame, ids, target, "validation", seed=8)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, third)


def test_a_constant_target_does_not_crash_the_folds():
    """A fold whose training half is constant has no direction to fit."""
    frame = features_frame(60)
    scores = out_of_fold_scores(
        frame, frame["sample_id"].tolist(), np.zeros(60), "validation"
    )
    assert np.isfinite(scores).all()
    assert np.allclose(scores, 0.0)


def test_a_non_finite_feature_is_refused_not_imputed():
    frame = features_frame(40)
    frame.loc[3, "uncertainty"] = np.nan
    with pytest.raises(ValueError, match="Nothing is imputed"):
        out_of_fold_scores(
            frame, frame["sample_id"].tolist(), linear_target(frame), "validation"
        )


def test_a_missing_sample_is_refused():
    frame = features_frame(20)
    with pytest.raises(KeyError, match="No features for"):
        out_of_fold_scores(
            frame, frame["sample_id"].tolist() + ["ghost"],
            np.zeros(21), "validation",
        )


# ============================================================
# Alpha selection
# ============================================================

def test_alpha_is_selected_on_validation_and_recorded():
    frame = features_frame(200)
    record = select_alpha(
        frame, frame["sample_id"].tolist(), linear_target(frame), "validation"
    )
    assert record["selected_alpha"] in ALPHA_GRID
    assert record["selected_on_split"] == "validation"
    assert record["uses_test_labels"] is False
    assert set(record["grid"]) == {str(float(a)) for a in ALPHA_GRID}


# ============================================================
# Serialisation
# ============================================================

def test_the_model_round_trips_and_predicts_identically(tmp_path):
    frame = features_frame(120)
    ids = frame["sample_id"].tolist()
    model = MilestoneHSIG.fit(frame, ids, linear_target(frame), "signed_gain", "validation")
    reloaded = MilestoneHSIG.load(model.save(tmp_path / "hsig.json"))
    assert reloaded.fingerprint() == model.fingerprint()
    np.testing.assert_allclose(reloaded.predict(frame, ids), model.predict(frame, ids))


def test_the_artefact_is_plain_readable_json(tmp_path):
    frame = features_frame(80)
    model = MilestoneHSIG.fit(
        frame, frame["sample_id"].tolist(), linear_target(frame), "t", "validation"
    )
    record = json.loads(model.save(tmp_path / "m.json").read_text(encoding="utf-8"))
    assert record["schema"] == MILESTONE_HSIG_SCHEMA
    assert record["estimator"] == "ridge"
    assert record["uses_test_labels"] is False
    assert len(record["coefficients"]) == len(FEATURE_SETS[DEFAULT_FEATURE_SET])


def test_prediction_uses_only_the_stored_coefficients():
    """A frozen policy must replay from the artefact, not from scikit-learn."""
    frame = features_frame(100)
    ids = frame["sample_id"].tolist()
    model = MilestoneHSIG.fit(frame, ids, linear_target(frame), "t", "validation")

    matrix = frame.set_index(frame["sample_id"].astype(str)).loc[
        ids, list(FEATURE_SETS[DEFAULT_FEATURE_SET])
    ].to_numpy(dtype=float)
    mean = np.asarray(model.scaler_mean)
    scale = np.asarray(model.scaler_scale)
    expected = ((matrix - mean) / scale) @ np.asarray(model.coefficients) + model.intercept
    np.testing.assert_allclose(model.predict(frame, ids), expected, rtol=1e-9)


def test_the_fingerprint_moves_when_a_coefficient_does():
    frame = features_frame(60)
    model = MilestoneHSIG.fit(
        frame, frame["sample_id"].tolist(), linear_target(frame), "t", "validation"
    )
    before = model.fingerprint()
    model.coefficients[0] += 0.5
    assert model.fingerprint() != before


def test_refitting_the_same_data_is_bit_identical():
    frame = features_frame(90)
    ids = frame["sample_id"].tolist()
    target = linear_target(frame)
    first = MilestoneHSIG.fit(frame, ids, target, "t", "validation")
    second = MilestoneHSIG.fit(frame, ids, target, "t", "validation")
    assert first.fingerprint() == second.fingerprint()


def test_importance_is_reported_on_a_comparable_scale():
    frame = features_frame(120)
    model = MilestoneHSIG.fit(
        frame, frame["sample_id"].tolist(), linear_target(frame), "t", "validation"
    )
    record = model.importance()
    assert set(record["coefficients"]) == set(FEATURE_SETS[DEFAULT_FEATURE_SET])
    assert "standardised" in record["interpretation"]
    # The planted signal is strongest on uncertainty, so it should rank first.
    assert record["ranked_by_absolute_weight"][0] == "uncertainty"


def test_the_estimator_family_is_the_same_for_every_target():
    """Phase D varies the target; holding the family fixed is what isolates it."""
    frame = features_frame(150)
    ids = frame["sample_id"].tolist()
    binary = (linear_target(frame) > np.median(linear_target(frame))).astype(float)
    continuous = linear_target(frame)
    for target, name in ((binary, "binary_correction"), (continuous, "signed_gain")):
        model = MilestoneHSIG.fit(frame, ids, target, name, "validation")
        assert model.to_dict()["estimator"] == "ridge"
        assert model.feature_set == DEFAULT_FEATURE_SET
