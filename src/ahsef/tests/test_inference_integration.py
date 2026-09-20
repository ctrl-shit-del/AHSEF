"""Integration checks against the real frozen baselines.

Marked ``integration`` because they open real checkpoints and real media.  Run
them with::

    python -m pytest src/ahsef/tests/test_inference_integration.py -m integration

Physiology is the subject: 198 test windows read from a parquet, no media
decoding, so the check costs a second rather than minutes -- while still
exercising the whole path (run summary -> architecture -> checkpoint -> the
baseline's own dataloader -> per-sample records).

The assertion that matters is the last one: AHSEF's inference layer must
reproduce the accuracy the frozen baseline recorded for itself.  If it does
not, the layer is scoring something other than the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ahsef.inference import (
    BaselinePredictor,
    PredictionSet,
    file_digest,
    resolve_class_order,
)
from src.ahsef.registry import baseline


pytestmark = pytest.mark.integration

MODALITY = "physiology"
ROOT = Path("experiments")


def _available() -> bool:
    try:
        reference = baseline(MODALITY)
        iteration = reference.resolve_iteration(ROOT)
        return reference.layout(ROOT).best_checkpoint(iteration).exists()
    except (FileNotFoundError, ValueError):
        return False


requires_baseline = pytest.mark.skipif(
    not _available(), reason=f"the frozen {MODALITY} baseline is not present"
)


@pytest.fixture(scope="module")
def predictor() -> BaselinePredictor:
    return BaselinePredictor.for_modality(MODALITY, root=ROOT)


@pytest.fixture(scope="module")
def predictions(predictor) -> PredictionSet:
    return predictor.predict("test")


@requires_baseline
def test_predictor_reads_the_validation_selected_iteration(predictor):
    summary = json.loads(
        predictor.layout.experiment_summary_path.read_text(encoding="utf-8")
    )
    assert predictor.iteration == str(summary["best_iteration"]["iteration"])
    assert summary["selection_metric"].startswith("validation")


@requires_baseline
def test_class_order_comes_from_the_run_summary_not_a_default(predictor):
    assert predictor.class_order == ("baseline", "stress", "amusement")
    assert predictor.class_order_source == "run_summary.class_order"


@requires_baseline
def test_every_test_sample_gets_exactly_one_record(predictor, predictions):
    expected = predictor.run_summary["samples"]["test"]
    assert len(predictions.frame) == expected
    assert predictions.frame["sample_id"].is_unique


@requires_baseline
def test_probabilities_are_a_valid_distribution(predictions):
    row_sums = predictions.probabilities().sum(dim=-1)
    assert float((row_sums - 1.0).abs().max()) < 1e-9
    assert float(predictions.frame["normalized_entropy"].min()) >= 0.0
    assert float(predictions.frame["normalized_entropy"].max()) <= 1.0


@requires_baseline
def test_predicted_class_is_the_argmax_of_the_probabilities(predictions):
    assert (predictions.probabilities().argmax(dim=-1) == predictions.predictions()).all()


@requires_baseline
def test_the_frozen_checkpoint_is_unchanged_by_a_pass(predictor, predictions):
    assert file_digest(predictor.checkpoint_path) == predictor.checkpoint_digest
    assert predictions.meta["checkpoint_unchanged"] is True


@requires_baseline
def test_latency_is_measured_not_declared(predictions):
    assert predictions.meta["latency"]["measured"] is True
    assert float(predictions.frame["latency_ms"].min()) > 0.0


@requires_baseline
def test_reproduces_the_accuracy_the_baseline_recorded(predictor, predictions):
    """The whole point: AHSEF must be scoring the same model on the same data."""
    recorded = predictor.run_summary["test_metrics"]["accuracy"]
    measured = float((predictions.predictions() == predictions.labels()).float().mean())
    assert measured == pytest.approx(recorded, abs=1e-4)


@requires_baseline
def test_prediction_set_round_trips_through_disk(predictions, tmp_path):
    path = tmp_path / "physiology__test.parquet"
    predictions.save(path)
    restored = PredictionSet.load(path)
    assert restored.class_order == predictions.class_order
    assert restored.sample_ids() == predictions.sample_ids()
    assert restored.meta["checkpoint_sha256"] == predictions.meta["checkpoint_sha256"]


@requires_baseline
def test_a_prediction_frame_without_its_sidecar_is_refused(predictions, tmp_path):
    path = tmp_path / "orphan.parquet"
    predictions.frame.to_parquet(path, index=False)
    with pytest.raises(FileNotFoundError, match="sidecar"):
        PredictionSet.load(path)
