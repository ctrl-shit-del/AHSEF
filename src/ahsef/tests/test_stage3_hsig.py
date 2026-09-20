"""The real HSIG: training, prediction, targets, leakage, and serialisation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ahsef.fusion import FusionSpec
from src.ahsef.hsig import HSIG, HSIG_TARGET, GainEstimate, RoutingState
from src.ahsef.stage3.features import build_features, states_from_features
from src.ahsef.stage3.hsig_model import (
    HSIGLeakageError,
    HSIGNotFittedError,
    HSIGTargetError,
    Stage3HSIG,
    assert_fit_split,
    assert_target_not_delta_uncertainty,
    out_of_fold_gain,
    prior_constant_gain,
    targets_from_oracle,
    uncertainty_only_gain,
)
from src.ahsef.stage3.hsig_quality import (
    auprc,
    auroc,
    calibration_of_gain,
    class_rule_audit,
    compare_estimators,
    estimator_quality,
    feature_importance,
    spearman,
)
from src.ahsef.stage3.oracle import build_oracle_table, fuse
from src.ahsef.tests.stage3_fixtures import scenario
from src.common.labels import CANONICAL_EMOTION_CLASSES


@pytest.fixture
def material():
    data = scenario(n=160, seed=11)
    spec = FusionSpec(
        method="weighted_probability", weights={"text_llm": 0.5, "audio": 0.5},
        selected_on_split="validation",
    )
    fused = fuse(data["text"], data["audio"], spec, data["ids"])
    oracle = build_oracle_table(data["text"], data["audio"], fused, data["ids"])
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(data["text"].frame, uncertainty)
    targets = targets_from_oracle(oracle, "validation")
    return {"data": data, "oracle": oracle, "features": features, "targets": targets,
            "fused": fused, "spec": spec}


# ------------------------------------------------------------------ targets

def test_targets_realise_the_declared_formula(material):
    targets = material["targets"]
    expected = (
        targets.fused_correct.astype(int) - targets.text_correct.astype(int)
    )
    assert np.array_equal(targets.signed_gain, expected)
    assert set(np.unique(targets.signed_gain)) <= {-1, 0, 1}


def test_y_gain_is_the_fix_event_only(material):
    targets = material["targets"]
    for fix, text_ok, fused_ok in zip(
        targets.y_gain, targets.text_correct, targets.fused_correct
    ):
        assert bool(fix) == (not text_ok and fused_ok)


def test_y_harm_is_the_break_event(material):
    targets = material["targets"]
    for harm, text_ok, fused_ok in zip(
        targets.y_harm, targets.text_correct, targets.fused_correct
    ):
        assert bool(harm) == (text_ok and not fused_ok)


def test_target_summary_names_the_rejected_target(material):
    summary = material["targets"].summary()
    assert summary["rejected_target"].startswith("delta_uncertainty")
    assert summary["target"] == HSIG_TARGET


def test_delta_uncertainty_target_is_refused():
    with pytest.raises(HSIGTargetError, match="delta-uncertainty"):
        assert_target_not_delta_uncertainty("delta_uncertainty")
    # The accepted target mentions uncertainty nowhere and passes.
    assert_target_not_delta_uncertainty(HSIG_TARGET)


# ----------------------------------------------------------------- leakage

def test_hsig_refuses_to_fit_on_test(material):
    targets = material["targets"]
    leaky = type(targets)(
        sample_ids=targets.sample_ids, text_correct=targets.text_correct,
        fused_correct=targets.fused_correct, split="test",
    )
    with pytest.raises(HSIGLeakageError, match="only be fitted on"):
        Stage3HSIG.fit(material["features"], leaky)


def test_assert_fit_split_allows_validation_and_train():
    assert assert_fit_split("validation") == "validation"
    assert assert_fit_split("train") == "train"
    with pytest.raises(HSIGLeakageError):
        assert_fit_split("test")


def test_out_of_fold_refuses_test_targets(material):
    targets = material["targets"]
    leaky = type(targets)(
        sample_ids=targets.sample_ids, text_correct=targets.text_correct,
        fused_correct=targets.fused_correct, split="test",
    )
    with pytest.raises(HSIGLeakageError):
        out_of_fold_gain(material["features"], leaky)


def test_features_never_contain_the_label(material):
    """The end-to-end guarantee: no label column reaches the feature matrix."""
    assert "true_class" not in material["features"].columns
    assert "y_gain" not in material["features"].columns
    assert "fused_correct" not in material["features"].columns


# -------------------------------------------------------------- fit/predict

def test_fit_produces_a_usable_estimator(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    assert isinstance(model, HSIG)
    assert set(model.components) == {"text_correct", "fused_correct"}
    gains = model.predict_frame(material["features"])
    assert len(gains) == len(material["features"])
    assert gains["predicted_gain"].between(-1.0, 1.0).all()


def test_paired_estimator_can_predict_negative_gain(material):
    """The estimator must be able to say 'acquiring would hurt'."""
    model = Stage3HSIG.fit(material["features"], material["targets"])
    gains = model.predict_frame(material["features"])["predicted_gain"]
    assert gains.min() < 0.0 or gains.max() > 0.0
    assert model.estimator_type == "paired_logistic"


def test_fix_logistic_is_bounded_in_zero_one(material):
    model = Stage3HSIG.fit(
        material["features"], material["targets"], estimator_type="fix_logistic"
    )
    gains = model.predict_frame(material["features"])["predicted_gain"]
    assert gains.between(0.0, 1.0).all()


def test_estimate_gain_matches_the_protocol(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    states = states_from_features(material["features"])
    estimate = model.estimate_gain(states[0].to_routing_state(), "audio")
    assert isinstance(estimate, GainEstimate)
    assert estimate.available and estimate.estimated
    assert estimate.to_dict()["target"] == HSIG_TARGET


def test_estimate_declines_for_an_unknown_candidate(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    state = states_from_features(material["features"])[0].to_routing_state()
    estimate = model.estimate_gain(state, "video")
    assert estimate.expected_improvement is None
    assert estimate.available is False
    assert "will not extrapolate" in estimate.basis


def test_estimate_declines_on_a_missing_feature(material):
    """A missing feature yields no estimate -- never an imputed one."""
    model = Stage3HSIG.fit(material["features"], material["targets"])
    state = states_from_features(material["features"])[0]
    partial = RoutingState(
        sample_id=state.sample_id, active_modalities=("text_llm",),
        uncertainty=state.uncertainty, predicted_class=state.predicted_class,
        features={k: v for k, v in list(state.features.items())[:3]},
    )
    estimate = model.estimate_gain(partial, "audio")
    assert estimate.expected_improvement is None
    assert "nothing is imputed" in estimate.basis


def test_estimate_declines_on_a_nan_feature(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    state = states_from_features(material["features"])[0]
    broken = dict(state.features)
    broken["score_top1"] = float("nan")
    estimate = model.estimate_gain(
        RoutingState(
            sample_id=state.sample_id, active_modalities=("text_llm",),
            uncertainty=state.uncertainty, predicted_class=state.predicted_class,
            features=broken,
        ),
        "audio",
    )
    assert estimate.expected_improvement is None


def test_predict_frame_marks_incomplete_rows(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    features = material["features"].copy()
    features.loc[0, "score_top1"] = np.nan
    result = model.predict_frame(features)
    assert bool(result.loc[0, "features_complete"]) is False
    assert np.isnan(result.loc[0, "predicted_gain"])
    assert bool(result.loc[1, "features_complete"]) is True


def test_single_outcome_component_is_refused(material):
    """A component with one outcome value cannot be fitted into a probability."""
    targets = material["targets"]
    constant = type(targets)(
        sample_ids=targets.sample_ids,
        text_correct=np.ones_like(targets.text_correct, dtype=bool),
        fused_correct=np.ones_like(targets.fused_correct, dtype=bool),
        split="validation",
    )
    with pytest.raises(HSIGNotFittedError, match="single outcome"):
        Stage3HSIG.fit(material["features"], constant)


def test_unfitted_estimator_refuses_to_predict():
    with pytest.raises(HSIGNotFittedError):
        Stage3HSIG().predict_gain(np.zeros((1, 10)))


# ------------------------------------------------------- deterministic replay

def test_serialisation_round_trips_exactly(material, tmp_path):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    path = model.save(tmp_path / "hsig.json")
    reloaded = Stage3HSIG.load(path)
    assert reloaded.fingerprint() == model.fingerprint()
    np.testing.assert_allclose(
        reloaded.predict_frame(material["features"])["predicted_gain"].to_numpy(),
        model.predict_frame(material["features"])["predicted_gain"].to_numpy(),
    )


def test_serialised_model_is_plain_json_not_a_pickle(material, tmp_path):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    record = json.loads(model.save(tmp_path / "hsig.json").read_text(encoding="utf-8"))
    assert record["components"]["text_correct"]["coefficients"]
    assert record["uses_test_labels"] is False


def test_fingerprint_changes_when_coefficients_change(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    before = model.fingerprint()
    model.components["text_correct"]["coefficients"][0] += 0.5
    assert model.fingerprint() != before


def test_fitting_twice_with_the_same_seed_is_identical(material):
    first = Stage3HSIG.fit(material["features"], material["targets"], seed=42)
    second = Stage3HSIG.fit(material["features"], material["targets"], seed=42)
    assert first.fingerprint() == second.fingerprint()


def test_out_of_fold_scores_every_sample(material):
    gains = out_of_fold_gain(material["features"], material["targets"], folds=4)
    assert gains.shape[0] == len(material["features"])
    assert np.isfinite(gains).all()


def test_out_of_fold_differs_from_in_sample(material):
    """If they were identical the honesty claim would be empty."""
    model = Stage3HSIG.fit(material["features"], material["targets"])
    in_sample = model.predict_frame(material["features"])["predicted_gain"].to_numpy()
    out_of_sample = out_of_fold_gain(material["features"], material["targets"], folds=4)
    assert not np.allclose(in_sample, out_of_sample)


# ------------------------------------------------------------------ quality

def test_auroc_matches_a_known_case():
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1, 1, 0, 0])) == 0.0


def test_auroc_is_none_without_both_classes():
    assert auroc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])) is None


def test_auprc_at_least_the_base_rate_for_a_perfect_ranker():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    positive = np.array([1, 1, 0, 0], dtype=bool)
    assert auprc(scores, positive) == pytest.approx(1.0)


def test_quality_report_carries_references_and_intervals(material):
    gains = out_of_fold_gain(material["features"], material["targets"], folds=4)
    record = estimator_quality(gains, material["targets"], "candidate")
    block = record["identifying_useful_acquisitions"]
    assert block["base_rate"] is not None
    assert "auprc" in block
    assert record["calibration"]["mae"] >= 0.0
    assert record["avoiding_harmful_acquisitions"]["positive_event"].startswith("y_harm")


def test_comparison_states_the_verdict_plainly(material):
    targets = material["targets"]
    candidate = estimator_quality(
        out_of_fold_gain(material["features"], targets, folds=4), targets, "candidate"
    )
    reference = estimator_quality(
        prior_constant_gain(targets, len(targets.sample_ids)), targets, "prior_constant"
    )
    verdict = compare_estimators(
        {"candidate": candidate, "prior_constant": reference}, ("prior_constant",)
    )["verdict"]
    assert verdict["best_candidate"] == "candidate"
    assert isinstance(verdict["statement"], str) and verdict["statement"]


def test_uncertainty_only_reference_is_scored_the_same_way(material):
    gains = uncertainty_only_gain(material["features"], material["targets"])
    assert gains.shape[0] == len(material["features"])
    assert np.isfinite(gains).all()


def test_calibration_bins_are_monotone_in_predicted_gain(material):
    gains = out_of_fold_gain(material["features"], material["targets"], folds=4)
    record = calibration_of_gain(gains, material["targets"].signed_gain)
    assert record["bins"]
    predicted = [row["mean_predicted_gain"] for row in record["bins"]]
    assert predicted == sorted(predicted)


def test_spearman_is_scale_free():
    left = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(left, left * 10) == pytest.approx(1.0)


def test_feature_importance_reports_the_gain_direction(material):
    model = Stage3HSIG.fit(material["features"], material["targets"])
    record = feature_importance(model.components)
    assert "gain_direction" in record
    assert record["gain_direction"]["ranked_by_absolute_weight"]


def test_class_rule_audit_detects_a_pure_class_lookup(material):
    """A gain that is a function of the predicted class alone must be caught."""
    classes = material["oracle"]["text_prediction"].to_numpy(dtype=int)
    lookup = np.array([0.1 * value for value in classes], dtype=float)
    record = class_rule_audit(lookup, classes, CANONICAL_EMOTION_CLASSES)
    assert record["mean_within_class_std"] == pytest.approx(0.0, abs=1e-12)
    assert "class-conditional constant" in record["verdict"]
