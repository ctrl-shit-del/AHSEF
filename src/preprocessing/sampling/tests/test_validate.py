"""Tests for the independent experiment-subset auditor."""

import pandas as pd
import pytest

from src.preprocessing.sampling.sampler import ExperimentSampler, SamplerConfig
from src.preprocessing.sampling.tests.test_sampler import _standardized_frame
from src.preprocessing.sampling.validate import validate_experiment


@pytest.fixture
def generated(tmp_path):
    source = tmp_path / "standardized.parquet"
    _standardized_frame().to_parquet(source, index=False)
    metadata_dir = tmp_path / "metadata"
    config = SamplerConfig(source=str(source), batch_size=256)
    ExperimentSampler(config).run(metadata_dir)
    return source, metadata_dir


def _validate(metadata_dir, **overrides):
    options = dict(check_files=0, verify_determinism=True)
    options.update(overrides)
    return validate_experiment(metadata_dir, **options)


def test_generated_experiment_passes_every_check(generated):
    _, metadata_dir = generated
    report = _validate(metadata_dir)

    assert report.passed, report.failures
    names = {check["check"] for check in report.checks}
    for expected in (
        "no_duplicate_sample_ids_within_split",
        "no_sample_id_across_splits",
        "canonical_labels_valid",
        "image_modality_available_for_every_record",
        "dataset_identity_preserved",
        "official_holdout_never_in_train",
        "split_ratios_within_tolerance",
        "dataset_representation_preserved",
        "class_representation_preserved",
        "source_paths_unchanged",
        "sampling_deterministic_for_seed",
        "counts_match_sampling_summary",
        "sample_id_digests_match_sampling_summary",
    ):
        assert expected in names
    assert report.counts["total"] == sum(
        report.counts[split] for split in ("train", "validation", "test")
    )


def test_cross_split_leakage_is_detected(generated):
    _, metadata_dir = generated
    train = pd.read_parquet(metadata_dir / "train.parquet")
    test = pd.read_parquet(metadata_dir / "test.parquet")
    leaked = train.head(1).copy()
    leaked["experiment_split"] = "test"
    pd.concat([test, leaked], ignore_index=True).to_parquet(
        metadata_dir / "test.parquet", index=False
    )

    report = _validate(metadata_dir, verify_determinism=False)
    failed = {check["check"] for check in report.failures}
    assert not report.passed
    assert "no_sample_id_across_splits" in failed


def test_duplicate_within_a_split_is_detected(generated):
    _, metadata_dir = generated
    train = pd.read_parquet(metadata_dir / "train.parquet")
    pd.concat([train, train.head(2)], ignore_index=True).to_parquet(
        metadata_dir / "train.parquet", index=False
    )

    report = _validate(metadata_dir, verify_determinism=False)
    failed = {check["check"] for check in report.failures}
    assert "no_duplicate_sample_ids_within_split" in failed


def test_rewritten_source_path_is_detected(generated):
    _, metadata_dir = generated
    train = pd.read_parquet(metadata_dir / "train.parquet")
    train.loc[0, "image_path"] = "somewhere/else.png"
    train.to_parquet(metadata_dir / "train.parquet", index=False)

    report = _validate(metadata_dir, verify_determinism=False)
    failed = {check["check"] for check in report.failures}
    assert "source_paths_unchanged" in failed


def test_official_holdout_promoted_into_train_is_detected(generated):
    _, metadata_dir = generated
    train = pd.read_parquet(metadata_dir / "train.parquet")
    train.loc[0, "training_split"] = "test"
    train.to_parquet(metadata_dir / "train.parquet", index=False)

    report = _validate(metadata_dir, verify_determinism=False)
    failed = {check["check"] for check in report.failures}
    assert "official_holdout_never_in_train" in failed


def test_missing_modality_record_is_detected(generated):
    _, metadata_dir = generated
    train = pd.read_parquet(metadata_dir / "train.parquet")
    train.loc[0, "has_image"] = False
    train.to_parquet(metadata_dir / "train.parquet", index=False)

    report = _validate(metadata_dir, verify_determinism=False)
    failed = {check["check"] for check in report.failures}
    assert "image_modality_available_for_every_record" in failed


def test_missing_summary_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="sampling_summary.json"):
        validate_experiment(tmp_path / "nothing-here")
