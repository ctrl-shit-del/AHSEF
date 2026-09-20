"""ECE, reliability bins, and the validation-only temperature guard."""

from __future__ import annotations

import pytest
import torch

from src.ahsef.calibration import (
    CalibrationLeakageError,
    TemperatureScaler,
    brier_score,
    calibration_report,
    compare_calibration,
    expected_calibration_error,
    fit_temperature,
    maximum_calibration_error,
    negative_log_likelihood,
    recommend_calibration,
    reliability_bins,
)


def perfectly_calibrated(num_classes: int = 2, samples: int = 1000):
    """Confidence 0.8 on every sample, correct on exactly 80% of them."""
    probabilities = torch.full((samples, num_classes), 0.2 / (num_classes - 1),
                               dtype=torch.float64)
    probabilities[:, 0] = 0.8
    labels = torch.zeros(samples, dtype=torch.long)
    labels[int(samples * 0.8):] = 1
    return probabilities, labels


# --------------------------------------------------------------------- ECE

def test_perfectly_calibrated_predictions_have_near_zero_ece():
    probabilities, labels = perfectly_calibrated()
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0, abs=1e-9)


def test_fully_overconfident_predictions_have_maximal_ece():
    samples = 100
    probabilities = torch.zeros(samples, 2, dtype=torch.float64)
    probabilities[:, 0] = 1.0
    labels = torch.ones(samples, dtype=torch.long)  # always wrong, always certain
    assert expected_calibration_error(probabilities, labels) == pytest.approx(1.0)
    assert maximum_calibration_error(probabilities, labels) == pytest.approx(1.0)


def test_ece_is_bounded_and_bins_partition_the_samples():
    torch.manual_seed(0)
    probabilities = torch.softmax(torch.randn(400, 7, dtype=torch.float64), dim=-1)
    labels = torch.randint(0, 7, (400,))
    bins = reliability_bins(probabilities, labels, num_bins=15)
    assert sum(item.count for item in bins) == 400
    assert 0.0 <= expected_calibration_error(probabilities, labels) <= 1.0


def test_empty_bins_are_reported_not_dropped():
    probabilities, labels = perfectly_calibrated()
    bins = reliability_bins(probabilities, labels, num_bins=10)
    assert len(bins) == 10
    assert any(item.count == 0 for item in bins)


def test_gap_sign_identifies_overconfidence():
    samples = 100
    probabilities = torch.zeros(samples, 2, dtype=torch.float64)
    probabilities[:, 0] = 0.9
    probabilities[:, 1] = 0.1
    labels = torch.zeros(samples, dtype=torch.long)
    labels[50:] = 1  # 50% accurate at 90% confidence
    report = calibration_report(probabilities, labels)
    assert report["overconfidence"] == pytest.approx(0.4)


# ------------------------------------------------------------ other scores

def test_brier_and_nll_are_zero_for_a_perfect_confident_model():
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    assert brier_score(probabilities, labels) == pytest.approx(0.0)
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(0.0, abs=1e-9)


def test_report_separates_confidence_when_right_from_when_wrong():
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4]], dtype=torch.float64
    )
    labels = torch.tensor([0, 1, 0])
    report = calibration_report(probabilities, labels)
    split = report["confidence_vs_correctness"]
    assert split["mean_confidence_when_correct"] == pytest.approx(0.75)
    assert split["mean_confidence_when_wrong"] == pytest.approx(0.9)


# -------------------------------------------------------- leakage guard

def test_temperature_cannot_be_fitted_on_test():
    logits = torch.randn(50, 7, dtype=torch.float64)
    labels = torch.randint(0, 7, (50,))
    with pytest.raises(CalibrationLeakageError, match="validation"):
        fit_temperature(logits, labels, split="test")


def test_temperature_cannot_be_fitted_on_train_either():
    logits = torch.randn(50, 7, dtype=torch.float64)
    labels = torch.randint(0, 7, (50,))
    with pytest.raises(CalibrationLeakageError):
        fit_temperature(logits, labels, split="train")


def test_fitted_scaler_records_the_split_it_saw():
    torch.manual_seed(3)
    logits = torch.randn(200, 7, dtype=torch.float64)
    labels = torch.randint(0, 7, (200,))
    scaler = fit_temperature(logits, labels, split="validation", max_iter=50)
    record = scaler.to_dict()
    assert record["fitted_on_split"] == "validation"
    assert record["uses_test_labels"] is False
    assert record["fitted_on_samples"] == 200


# ------------------------------------------------------------ temperature

def test_temperature_fitting_reduces_nll_on_an_overconfident_model():
    torch.manual_seed(4)
    # Logits far larger than the signal they carry: classic overconfidence.
    truth = torch.randint(0, 7, (2000,))
    logits = torch.randn(2000, 7, dtype=torch.float64)
    logits[torch.arange(2000), truth] += 1.0
    logits = logits * 8.0
    scaler = fit_temperature(logits, truth, split="validation")
    assert scaler.temperature > 1.0
    assert scaler.optimisation["nll_after"] < scaler.optimisation["nll_before"]


def test_identity_scaler_leaves_probabilities_untouched():
    logits = torch.randn(10, 7, dtype=torch.float64)
    scaler = TemperatureScaler.identity()
    assert scaler.temperature == 1.0
    assert torch.allclose(scaler.apply(logits), torch.softmax(logits, dim=-1))


def test_scaler_round_trips_through_a_dict():
    torch.manual_seed(5)
    logits = torch.randn(100, 7, dtype=torch.float64)
    labels = torch.randint(0, 7, (100,))
    scaler = fit_temperature(logits, labels, split="validation", max_iter=20)
    restored = TemperatureScaler.from_dict(scaler.to_dict())
    assert restored.temperature == pytest.approx(scaler.temperature)
    assert restored.fitted_on_split == "validation"


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        TemperatureScaler(temperature=0.0)


def test_recommendation_prefers_raw_when_the_temperature_does_not_help():
    probabilities, labels = perfectly_calibrated()
    logits = probabilities.log()
    reports = compare_calibration(logits, labels, TemperatureScaler(temperature=5.0))
    recommendation = recommend_calibration(reports)
    assert recommendation["recommended"] == "raw"
    assert recommendation["temperature_reduces_ece"] is False
