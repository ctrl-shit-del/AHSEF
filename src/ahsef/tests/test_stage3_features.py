"""HSIG feature construction: label-free, never imputed, auditable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ahsef.stage3.features import (
    CLASS_FEATURES,
    DEFAULT_FEATURE_SET,
    EVIDENCE_FEATURES,
    FEATURE_SETS,
    LABEL_BEARING_COLUMNS,
    FeatureLeakageError,
    MissingFeatureError,
    assert_label_free,
    build_features,
    complete_mask,
    feature_matrix,
    feature_provenance,
    missing_feature_report,
    states_from_features,
)
from src.ahsef.tests.stage3_fixtures import make_llm_set, scenario


@pytest.fixture
def llm_set():
    return scenario(n=40, seed=3)["text"]


def test_features_are_built_for_every_row(llm_set):
    uncertainty = llm_set.frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(llm_set.frame, uncertainty)
    assert len(features) == len(llm_set.frame)
    assert list(features["sample_id"]) == llm_set.sample_ids()


def test_every_declared_feature_is_present(llm_set):
    uncertainty = llm_set.frame["llm_normalized_score_entropy"].to_numpy()
    for feature_set in FEATURE_SETS:
        features = build_features(llm_set.frame, uncertainty, feature_set)
        matrix, names = feature_matrix(features, feature_set)
        assert names == list(FEATURE_SETS[feature_set])
        assert matrix.shape == (len(features), len(names))


def test_no_feature_is_label_bearing():
    for feature_set, names in FEATURE_SETS.items():
        assert not set(names) & LABEL_BEARING_COLUMNS, feature_set


def test_assert_label_free_rejects_a_label_column():
    frame = pd.DataFrame({"sample_id": ["a"], "true_class": [1]})
    with pytest.raises(FeatureLeakageError, match="label-bearing"):
        assert_label_free(frame, ["true_class"])


def test_class_features_live_in_their_own_set():
    assert set(CLASS_FEATURES) & set(FEATURE_SETS["evidence_plus_class"])
    assert not set(CLASS_FEATURES) & set(FEATURE_SETS["evidence_only"])
    assert FEATURE_SETS["evidence_only"] == EVIDENCE_FEATURES


def test_top1_and_top2_are_ordered(llm_set):
    uncertainty = llm_set.frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(llm_set.frame, uncertainty)
    assert (features["score_top1"] >= features["score_top2"] - 1e-12).all()
    assert (features["score_margin"] >= -1e-12).all()


def test_missing_scores_produce_missing_features_not_zeros():
    """A sample the LLM gave no scores for must not be silently filled."""
    llm = make_llm_set(["a", "b"], [0, 1], [0, 1])
    for index in range(7):
        llm.frame.loc[1, f"prob_{index}"] = np.nan

    uncertainty = llm.frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(llm.frame, uncertainty)
    mask = complete_mask(features)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    # and specifically: not zero-filled
    assert np.isnan(features.loc[1, "score_top1"])


def test_missing_feature_report_counts_rather_than_hides():
    llm = make_llm_set(["a", "b", "c"], [0, 1, 2], [0, 1, 2])
    llm.frame.loc[2, "prob_0"] = np.nan
    uncertainty = llm.frame["llm_normalized_score_entropy"].to_numpy()
    report = missing_feature_report(build_features(llm.frame, uncertainty))
    assert report["rows"] == 3
    assert report["incomplete_rows"] == 1
    assert report["complete_rows"] == 2
    assert "Nothing is imputed" in report["policy"]


def test_feature_matrix_refuses_an_absent_feature(llm_set):
    uncertainty = llm_set.frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(llm_set.frame, uncertainty)
    with pytest.raises(MissingFeatureError):
        feature_matrix(features.drop(columns=["score_top1"]), DEFAULT_FEATURE_SET)


def test_uncertainty_length_must_match(llm_set):
    with pytest.raises(ValueError, match="entries"):
        build_features(llm_set.frame, np.zeros(3))


def test_states_carry_no_label(llm_set):
    uncertainty = llm_set.frame["llm_normalized_score_entropy"].to_numpy()
    states = states_from_features(build_features(llm_set.frame, uncertainty))
    assert len(states) == len(llm_set.frame)
    for state in states:
        assert not hasattr(state, "true_class")
        routing = state.to_routing_state()
        assert routing.active_modalities == ("text_llm",)
        assert 0.0 <= routing.uncertainty <= 1.0
        assert "true_class" not in routing.features


def test_provenance_declares_the_definition():
    record = feature_provenance()
    assert record["label_free"] is True
    assert record["imputation"].startswith("none")
    assert record["features"] == list(FEATURE_SETS[DEFAULT_FEATURE_SET])
