"""Calibrating an LLM self-report: measurement, and the validation-only guard."""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.calibration import CalibrationLeakageError
from src.ahsef.llm.calibration import (
    BinnedConfidenceCalibrator,
    calibration_provenance,
    confidence_calibration_report,
    confidence_reliability_bins,
    fit_confidence_calibrator,
)


def calibrated(n: int = 1000):
    """Confidence 0.8 everywhere, correct on exactly 80% of samples."""
    confidence = np.full(n, 0.8)
    correct = np.zeros(n, dtype=bool)
    correct[: int(n * 0.8)] = True
    return confidence, correct


# ------------------------------------------------------------ measurement

def test_a_perfectly_calibrated_self_report_has_near_zero_ece():
    confidence, correct = calibrated()
    report = confidence_calibration_report(confidence, correct)
    assert report["ece"] == pytest.approx(0.0, abs=1e-9)
    assert report["accuracy"] == pytest.approx(0.8)


def test_systematic_overconfidence_is_measured_as_such():
    confidence = np.full(100, 0.95)
    correct = np.zeros(100, dtype=bool)
    correct[:50] = True
    report = confidence_calibration_report(confidence, correct)
    assert report["overconfidence"] == pytest.approx(0.45)
    assert report["ece"] == pytest.approx(0.45)


def test_bins_partition_every_sample():
    rng = np.random.default_rng(0)
    confidence = rng.random(500)
    correct = rng.random(500) < confidence
    bins = confidence_reliability_bins(confidence, correct, num_bins=10)
    assert sum(item.count for item in bins) == 500
    assert len(bins) == 10


def test_a_constant_confidence_is_reported_as_non_discriminating():
    """Calibration and usefulness are different virtues; both are reported."""
    confidence, correct = calibrated()
    report = confidence_calibration_report(confidence, correct)
    assert report["ece"] == pytest.approx(0.0, abs=1e-9)      # well calibrated
    assert report["discrimination"]["auroc"] == pytest.approx(0.5)  # and useless
    assert report["discrimination"]["confidence_std"] == pytest.approx(0.0)
    assert report["discrimination"]["distinct_values"] == 1


def test_a_discriminating_self_report_scores_above_chance():
    rng = np.random.default_rng(1)
    confidence = rng.random(2000)
    correct = rng.random(2000) < confidence
    report = confidence_calibration_report(confidence, correct)
    assert report["discrimination"]["auroc"] > 0.7
    assert report["confidence_vs_correctness"]["separation"] > 0


def test_an_anti_correlated_self_report_is_not_hidden():
    rng = np.random.default_rng(2)
    confidence = rng.random(2000)
    correct = rng.random(2000) > confidence      # confident exactly when wrong
    report = confidence_calibration_report(confidence, correct)
    assert report["discrimination"]["auroc"] < 0.3
    assert report["confidence_vs_correctness"]["separation"] < 0


def test_auroc_is_none_when_every_sample_shares_an_outcome():
    report = confidence_calibration_report(np.full(10, 0.5), np.ones(10, dtype=bool))
    assert report["discrimination"]["auroc"] is None


def test_the_report_denies_being_a_posterior():
    confidence, correct = calibrated()
    assert confidence_calibration_report(confidence, correct)["is_a_posterior"] is False


def test_out_of_range_and_nan_confidences_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        confidence_calibration_report(np.array([1.5]), np.array([True]))
    with pytest.raises(ValueError, match="NaN"):
        confidence_calibration_report(np.array([float("nan")]), np.array([True]))


# --------------------------------------------------------- leakage guard

def test_a_calibrator_cannot_be_fitted_on_test():
    confidence, correct = calibrated(100)
    with pytest.raises(CalibrationLeakageError, match="validation"):
        fit_confidence_calibrator(confidence, correct, split="test")


def test_a_fitted_calibrator_records_the_split_it_saw():
    confidence, correct = calibrated(200)
    calibrator = fit_confidence_calibrator(confidence, correct, split="validation")
    record = calibrator.to_dict()
    assert record["fitted_on_split"] == "validation"
    assert record["uses_test_labels"] is False
    assert record["fitted_on_samples"] == 200


# --------------------------------------------------------- recalibration

def test_the_map_sends_confidence_to_observed_accuracy():
    rng = np.random.default_rng(3)
    confidence = rng.random(4000)
    correct = rng.random(4000) < confidence
    calibrator = fit_confidence_calibrator(confidence, correct, "validation", num_bins=10)
    mapped = calibrator.apply(np.array([0.05, 0.95]))
    assert mapped[0] < mapped[1]
    assert 0.0 <= mapped.min() <= mapped.max() <= 1.0


def test_a_sparse_bin_backs_off_to_the_global_accuracy():
    """A three-sample bin should not become a calibrated estimate."""
    confidence = np.concatenate([np.full(200, 0.5), np.full(3, 0.95)])
    correct = np.concatenate([
        np.array([True] * 100 + [False] * 100), np.array([True] * 3)
    ])
    calibrator = fit_confidence_calibrator(
        confidence, correct, "validation", num_bins=10, min_bin_count=20,
    )
    assert calibrator.backed_off_bins
    assert calibrator.apply(np.array([0.95]))[0] == pytest.approx(calibrator.fallback)


def test_an_unfitted_calibrator_refuses_to_apply():
    with pytest.raises(RuntimeError, match="not been fitted"):
        BinnedConfidenceCalibrator().apply(np.array([0.5]))


def test_calibration_is_measured_but_not_applied_automatically():
    record = calibration_provenance()
    assert record["applied_automatically"] is False
    assert record["uses_test_labels"] is False
    assert record["fit_split"] == "validation"
