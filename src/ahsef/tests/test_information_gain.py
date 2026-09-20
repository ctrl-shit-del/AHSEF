"""Per-sample dU(m | A), its summary, and the cost model it is priced against."""

from __future__ import annotations

import pytest
import torch

from src.ahsef.costs import CostModel, ModalityCost, cost_from_predictions
from src.ahsef.fusion import FusionSpec, fuse_prediction_sets
from src.ahsef.information_gain import (
    candidate_gain_table,
    delta_uncertainty_records,
    gain_by_uncertainty_band,
    summarise_delta_uncertainty,
)
from src.ahsef.tests.conftest import make_prediction_set


POOL = [f"s{i}" for i in range(6)]
SPEC = FusionSpec(weights={"audio": 0.5, "text": 0.5})


def records(anchor_set, candidate_set, cost_model=None):
    fused = fuse_prediction_sets({"audio": anchor_set, "text": candidate_set}, SPEC, POOL)
    return delta_uncertainty_records(
        anchor_set, fused, "text", POOL, cost_model=cost_model, candidate_set=["text"]
    )


# ------------------------------------------------------------------- delta

def test_delta_is_uncertainty_before_minus_after(anchor_set, candidate_set):
    frame = records(anchor_set, candidate_set)
    assert len(frame) == 6
    expected = frame["uncertainty_before"] - frame["uncertainty_after"]
    assert (frame["delta_uncertainty"] - expected).abs().max() == pytest.approx(0.0)


def test_fusing_a_modality_with_itself_yields_exactly_zero_gain(anchor_set):
    """Adding no new evidence must produce dU = 0, not noise."""
    twin = make_prediction_set(
        "text", POOL, anchor_set.logits(), anchor_set.frame["true_class"].tolist()
    )
    frame = records(anchor_set, twin)
    assert frame["delta_uncertainty"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_a_confident_candidate_reduces_uncertainty_on_an_uncertain_anchor(
    anchor_set, candidate_set
):
    frame = records(anchor_set, candidate_set).set_index("sample_id")
    # s1 is near-uniform under the anchor and confident under the candidate.
    assert frame.loc["s1", "uncertainty_before"] > 0.99
    assert frame.loc["s1", "delta_uncertainty"] > 0.0


def test_a_flat_candidate_can_raise_uncertainty_and_that_is_recorded_as_negative(
    anchor_set,
):
    flat = make_prediction_set(
        "text", POOL, torch.zeros(6, 7, dtype=torch.float64),
        anchor_set.frame["true_class"].tolist(),
    )
    frame = records(anchor_set, flat)
    assert (frame["delta_uncertainty"] < 0).any()


def test_records_are_per_sample_not_aggregated(anchor_set, candidate_set):
    frame = records(anchor_set, candidate_set)
    assert frame["sample_id"].tolist() == POOL
    assert frame["delta_uncertainty"].nunique() > 1


def test_prediction_change_flag_matches_the_class_columns(anchor_set, candidate_set):
    frame = records(anchor_set, candidate_set)
    assert (
        frame["prediction_changed"] == (frame["predicted_before"] != frame["predicted_after"])
    ).all()


def test_active_modalities_are_carried_on_every_row(anchor_set, candidate_set):
    frame = records(anchor_set, candidate_set)
    assert set(frame["active_modalities"]) == {"audio"}
    assert set(frame["candidate_modality"]) == {"text"}


# ----------------------------------------------------------------- summary

def test_summary_reports_spread_not_only_the_mean(anchor_set, candidate_set):
    summary = summarise_delta_uncertainty(records(anchor_set, candidate_set))
    for key in ("mean_delta_uncertainty", "median_delta_uncertainty",
                "std_delta_uncertainty", "min_delta_uncertainty",
                "max_delta_uncertainty", "positive_delta_rate"):
        assert key in summary
    assert summary["samples"] == 6


def test_summary_counts_corrections_in_both_directions(anchor_set, candidate_set):
    summary = summarise_delta_uncertainty(records(anchor_set, candidate_set))
    outcome = summary["outcome"]
    assert outcome["fixed_by_candidate"] >= 1  # the candidate rescues s1 and s3
    assert outcome["net_corrections"] == (
        outcome["fixed_by_candidate"] - outcome["broken_by_candidate"]
    )


def test_band_table_partitions_the_samples(anchor_set, candidate_set):
    bands = gain_by_uncertainty_band(records(anchor_set, candidate_set))
    assert sum(band["samples"] for band in bands) == 6


def test_candidate_gain_table_ranks_candidates(anchor_set, candidate_set):
    flat = make_prediction_set(
        "video", POOL, torch.zeros(6, 7, dtype=torch.float64),
        anchor_set.frame["true_class"].tolist(),
    )
    fused_flat = fuse_prediction_sets(
        {"audio": anchor_set, "text": flat}, SPEC, POOL
    )
    table = candidate_gain_table({
        "text": records(anchor_set, candidate_set),
        "video": delta_uncertainty_records(anchor_set, fused_flat, "video", POOL),
    })
    assert table["ranked_by_mean_delta_uncertainty"][0] == "text"


# -------------------------------------------------------------- cost model

def test_cost_is_derived_from_measurement_not_invented(anchor_set):
    cost = cost_from_predictions(anchor_set)
    assert cost.latency_ms == pytest.approx(anchor_set.frame["latency_ms"].mean())
    assert cost.parameters == 1000
    assert cost.samples == 6


def test_an_explicit_architecture_record_supplies_the_input_geometry(anchor_set):
    """The compute proxy prices the frozen model, not whatever the export happened
    to record, so a run summary passed in overrides the sidecar."""
    plain = cost_from_predictions(anchor_set)
    assert plain.input_elements == 1  # 'Synthetic' has no known geometry

    audio_like = {
        "class": "AudioEmotionBaseline", "parameters": 35335,
        "max_frames": 401, "n_mels": 64,
    }
    priced = cost_from_predictions(anchor_set, audio_like)
    assert priced.input_elements == 401 * 64
    assert priced.compute_units == pytest.approx(35335 * 401 * 64)
    # Latency stays measured either way.
    assert priced.latency_ms == pytest.approx(plain.latency_ms)


def test_max_normalisation_puts_the_costliest_candidate_at_one():
    model = CostModel({
        "text": ModalityCost("text", 1.0, 10.0, 1, 10, "test", 6),
        "video": ModalityCost("video", 4.0, 40.0, 4, 10, "test", 6),
    })
    assert model.normalized_latency("video") == pytest.approx(1.0)
    assert model.normalized_latency("text") == pytest.approx(0.25)


def test_normalisation_is_computed_over_the_candidate_set_being_ranked():
    """Restricting the candidate set changes the normaliser, and that is recorded."""
    model = CostModel({
        "text": ModalityCost("text", 1.0, 10.0, 1, 10, "test", 6),
        "video": ModalityCost("video", 4.0, 40.0, 4, 10, "test", 6),
        "image": ModalityCost("image", 2.0, 20.0, 2, 10, "test", 6),
    })
    assert model.normalized_latency("text", ["text", "video"]) == pytest.approx(0.25)
    assert model.normalized_latency("text", ["text", "image"]) == pytest.approx(0.5)
    assert model.table(["text", "image"])["normalised_over"] == ["image", "text"]


def test_minmax_normalisation_of_a_single_candidate_is_maximal_not_free():
    model = CostModel(
        {"text": ModalityCost("text", 1.0, 10.0, 1, 10, "test", 6)},
        normalization="minmax",
    )
    assert model.normalized_latency("text") == pytest.approx(1.0)


def test_unknown_modality_is_refused():
    model = CostModel({"text": ModalityCost("text", 1.0, 10.0, 1, 10, "test", 6)})
    with pytest.raises(KeyError):
        model.normalized_cost("video")


def test_unknown_normalisation_is_refused():
    with pytest.raises(ValueError, match="normalization must be"):
        CostModel(
            {"text": ModalityCost("text", 1.0, 10.0, 1, 10, "test", 6)},
            normalization="vibes",
        )
