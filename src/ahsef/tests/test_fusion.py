"""Probability fusion: the arithmetic, and the two guards around it."""

from __future__ import annotations

import pytest
import torch

from src.ahsef.fusion import (
    FusionCompatibilityError,
    FusionSpec,
    WeightSelectionError,
    assert_fusable,
    fuse_prediction_sets,
    fuse_probabilities,
    select_weights,
)
from src.ahsef.tests.conftest import make_prediction_set


POOL = [f"s{i}" for i in range(6)]


# ---------------------------------------------------------------- weights

def test_spec_normalises_weights_to_sum_to_one():
    spec = FusionSpec(weights={"audio": 3.0, "text": 1.0})
    assert spec.weights["audio"] == pytest.approx(0.75)
    assert sum(spec.weights.values()) == pytest.approx(1.0)


def test_spec_rejects_negative_and_all_zero_weights():
    with pytest.raises(ValueError, match="non-negative"):
        FusionSpec(weights={"audio": -1.0, "text": 2.0})
    with pytest.raises(ValueError, match="not all be zero"):
        FusionSpec(weights={"audio": 0.0, "text": 0.0})


def test_uniform_spec_records_that_no_data_chose_it():
    spec = FusionSpec.uniform(["audio", "text"])
    assert spec.weights == {"audio": 0.5, "text": 0.5}
    assert spec.selected_on_split is None


# ------------------------------------------------------------- arithmetic

def test_weighted_fusion_of_identical_posteriors_is_the_identity():
    posterior = torch.softmax(torch.randn(5, 7, dtype=torch.float64), dim=-1)
    fused = fuse_probabilities({"a": posterior, "b": posterior}, {"a": 0.5, "b": 0.5})
    assert torch.allclose(fused, posterior)


def test_fused_posterior_is_a_valid_distribution():
    torch.manual_seed(0)
    left = torch.softmax(torch.randn(20, 7, dtype=torch.float64), dim=-1)
    right = torch.softmax(torch.randn(20, 7, dtype=torch.float64), dim=-1)
    for method in ("weighted_probability", "log_opinion_pool"):
        fused = fuse_probabilities({"a": left, "b": right}, {"a": 0.3, "b": 0.7}, method)
        assert torch.allclose(fused.sum(dim=-1), torch.ones(20, dtype=torch.float64))
        assert float(fused.min()) >= 0.0


def test_full_weight_on_one_modality_recovers_that_modality():
    torch.manual_seed(1)
    left = torch.softmax(torch.randn(4, 7, dtype=torch.float64), dim=-1)
    right = torch.softmax(torch.randn(4, 7, dtype=torch.float64), dim=-1)
    fused = fuse_probabilities({"a": left, "b": right}, {"a": 1.0, "b": 0.0})
    assert torch.allclose(fused, left)


def test_geometric_pool_is_sharper_than_the_linear_pool_on_agreement():
    """Two models agreeing should be more decisive under the geometric pool."""
    from src.ahsef.uncertainty import normalized_entropy

    posterior = torch.tensor([[0.6, 0.2, 0.2]], dtype=torch.float64)
    linear = fuse_probabilities({"a": posterior, "b": posterior}, {"a": 0.5, "b": 0.5})
    geometric = fuse_probabilities(
        {"a": posterior, "b": posterior}, {"a": 0.5, "b": 0.5}, "log_opinion_pool"
    )
    assert float(normalized_entropy(geometric)) <= float(normalized_entropy(linear))


def test_shape_mismatch_is_refused():
    with pytest.raises(FusionCompatibilityError, match="shapes disagree"):
        fuse_probabilities(
            {"a": torch.ones(3, 7) / 7, "b": torch.ones(3, 3) / 3}, {"a": 1.0, "b": 1.0}
        )


def test_unknown_method_is_refused():
    posterior = torch.ones(2, 7, dtype=torch.float64) / 7
    with pytest.raises(ValueError, match="method must be"):
        fuse_probabilities({"a": posterior}, {"a": 1.0}, method="magic")


# ------------------------------------------------------------ class-space

def test_different_class_spaces_cannot_be_fused(anchor_set, wesad_set):
    with pytest.raises(FusionCompatibilityError, match="different class spaces"):
        assert_fusable({"audio": anchor_set, "physiology": wesad_set})


def test_different_splits_cannot_be_fused(anchor_set):
    other = make_prediction_set(
        "text", POOL, torch.zeros(6, 7, dtype=torch.float64), [0] * 6, split="validation"
    )
    with pytest.raises(FusionCompatibilityError, match="different splits"):
        assert_fusable({"audio": anchor_set, "text": other})


# --------------------------------------------------------- sample identity

def test_fusion_reindexes_to_the_requested_pool_order(anchor_set, candidate_set):
    """Rows are paired by sample id, never by manifest position."""
    shuffled = candidate_set.restricted_to(list(reversed(POOL)))
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    straight = fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, spec, POOL)
    reordered = fuse_prediction_sets({"audio": anchor_set, "text": shuffled}, spec, POOL)
    assert straight.sample_ids() == reordered.sample_ids() == POOL
    assert torch.allclose(straight.probabilities(), reordered.probabilities())


def test_fusion_refuses_a_sample_the_candidate_has_never_seen(anchor_set, candidate_set):
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    with pytest.raises(KeyError, match="no prediction"):
        fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, spec, ["s0", "sX"])


def test_fusion_refuses_when_the_two_sets_disagree_about_the_label(anchor_set):
    """A shared id carrying different targets is not the same sample."""
    wrong_labels = make_prediction_set(
        "text", POOL, torch.zeros(6, 7, dtype=torch.float64), [6, 6, 6, 6, 6, 6]
    )
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    with pytest.raises(FusionCompatibilityError, match="disagree about the true class"):
        fuse_prediction_sets({"audio": anchor_set, "text": wrong_labels}, spec, POOL)


def test_fusion_refuses_an_empty_pool(anchor_set, candidate_set):
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    with pytest.raises(ValueError, match="empty sample pool"):
        fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, spec, [])


def test_fused_latency_is_the_sum_of_component_latencies(anchor_set, candidate_set):
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    fused = fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, spec, POOL)
    expected = (
        anchor_set.frame["latency_ms"].iloc[0] + candidate_set.frame["latency_ms"].iloc[0]
    )
    assert fused.frame["latency_ms"].iloc[0] == pytest.approx(expected)


def test_fused_set_records_its_components(anchor_set, candidate_set):
    spec = FusionSpec(weights={"audio": 0.5, "text": 0.5})
    fused = fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, spec, POOL)
    assert fused.meta["component_modalities"] == ["audio", "text"]
    assert fused.meta["fusion"]["uses_test_labels"] is False
    assert fused.modality == "audio+text"


# ----------------------------------------------------- weight selection

def test_weights_cannot_be_selected_on_test(anchor_set, candidate_set):
    with pytest.raises(WeightSelectionError, match="refusing to select on 'test'"):
        select_weights({"audio": anchor_set, "text": candidate_set}, POOL, split="test")


def test_weight_selection_requires_the_sets_to_be_from_that_split(anchor_set, candidate_set):
    with pytest.raises(WeightSelectionError, match="must come from"):
        select_weights(
            {"audio": anchor_set, "text": candidate_set}, POOL, split="validation"
        )


def test_weight_selection_reports_the_whole_grid_it_scanned():
    left = make_prediction_set(
        "audio", POOL, torch.eye(7, dtype=torch.float64)[[0, 1, 1, 2, 2, 3]] * 4,
        [0, 1, 1, 2, 2, 3], split="validation",
    )
    right = make_prediction_set(
        "text", POOL, torch.eye(7, dtype=torch.float64)[[0, 1, 1, 2, 2, 3]] * 4,
        [0, 1, 1, 2, 2, 3], split="validation",
    )
    spec = select_weights({"audio": left, "text": right}, POOL, split="validation")
    assert spec.selected_on_split == "validation"
    assert spec.selection["method"] == "exhaustive_grid_scan"
    assert len(spec.selection["scanned"]) > 1
    assert spec.to_dict()["uses_test_labels"] is False
