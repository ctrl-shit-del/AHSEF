"""Tests for the WESAD physiological-window experiment builder.

The recordings here are synthetic pickles shaped exactly like WESAD's -- a few
seconds of chest signal at a low rate -- so nothing touches the real corpus.
The property that matters most is subject disjointness: windows from one
recording are highly autocorrelated, so a subject appearing in two partitions
would make the reported score meaningless.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.common.labels import WESAD_STATE_3CLASS
from src.preprocessing.sampling.physiology_windows import (
    CHEST_CHANNELS,
    FEATURE_NAMES,
    FEATURE_STATISTICS,
    SPLITS,
    PhysiologyWindowBuilder,
    PhysiologyWindowConfig,
    iter_subject_windows,
    window_features,
)
from src.preprocessing.sampling.validate_physiology import validate_physiology_experiment


RATE = 10
SUBJECTS = ("S2", "S3", "S4", "S5", "S6")
#: 1 baseline, 2 stress, 3 amusement, 4 meditation, 0 transition.
CONDITION_SECONDS = ((0, 2), (1, 12), (0, 2), (2, 10), (0, 2), (3, 8), (0, 2), (4, 6))


def _labels() -> np.ndarray:
    blocks = [np.full(seconds * RATE, code, dtype=np.int64) for code, seconds in CONDITION_SECONDS]
    return np.concatenate(blocks)


def _recording(subject: str, seed: int) -> dict:
    labels = _labels()
    generator = np.random.default_rng(seed)
    samples = labels.shape[0]
    # A per-condition offset makes the windows separable, so a run that learns
    # nothing is a bug in the pipeline rather than in the data.
    offset = labels.astype(np.float32)[:, None]
    return {
        "subject": subject,
        "label": labels,
        "signal": {
            "chest": {
                "ACC": generator.normal(size=(samples, 3)).astype(np.float32) + offset,
                "ECG": generator.normal(size=(samples, 1)).astype(np.float32) + offset,
                "EMG": generator.normal(size=(samples, 1)).astype(np.float32) + offset,
                "EDA": generator.normal(size=(samples, 1)).astype(np.float32) + offset,
                "Temp": generator.normal(size=(samples, 1)).astype(np.float32) + offset,
                "Resp": generator.normal(size=(samples, 1)).astype(np.float32) + offset,
            }
        },
    }


def _standardized_row(subject: str) -> dict:
    return {
        "sample_id": f"WESAD|{subject}",
        "dataset": "WESAD",
        "speaker": subject,
        "training_split": None,
        "target_type": "physiological_state",
        "emotion_target_valid": False,
        "canonical_emotion": None,
        "canonical_emotion_id": None,
        "has_physiology": True,
        "physiology_source": "file",
        "physiology_path": f"WESAD/{subject}/{subject}.pkl",
        "has_audio": False, "has_video": False, "has_image": False, "has_text": False,
        "audio_source": None, "video_source": None, "image_source": None, "text_source": None,
        "audio_path": None, "video_path": None, "image_path": None,
        "text": None, "text_path": None, "extras": "{}",
    }


@pytest.fixture
def wesad(tmp_path, subjects=SUBJECTS):
    datasets_root = tmp_path / "datasets"
    for index, subject in enumerate(subjects):
        path = datasets_root / "WESAD" / subject / f"{subject}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(_recording(subject, seed=index), handle)

    source = tmp_path / "standardized.parquet"
    pd.DataFrame([_standardized_row(subject) for subject in subjects]).to_parquet(
        source, index=False
    )
    return {"source": source, "datasets_root": datasets_root, "subjects": list(subjects)}


def _config(wesad, **overrides):
    defaults = dict(
        window_seconds=2.0,
        stride_seconds=1.0,
        sample_rate=RATE,
        seed=42,
        source=str(wesad["source"]),
        datasets_root=str(wesad["datasets_root"]),
    )
    defaults.update(overrides)
    return PhysiologyWindowConfig(**defaults)


def _run(wesad, output_dir, **overrides):
    return PhysiologyWindowBuilder(_config(wesad, **overrides)).run(output_dir)


def _frames(output_dir):
    return {split: pd.read_parquet(output_dir / f"{split}.parquet") for split in SPLITS}


# ============================================================
# Feature extraction
# ============================================================

def test_feature_vector_layout_is_channel_major_and_fixed_width():
    window = np.arange(40, dtype=np.float32).reshape(5, 8)
    features = window_features(window)
    assert features.shape == (len(CHEST_CHANNELS) * len(FEATURE_STATISTICS),)
    assert features.shape[0] == len(FEATURE_NAMES)
    # First channel, first statistic is that channel's mean.
    assert features[0] == pytest.approx(window[:, 0].mean())
    assert FEATURE_NAMES[0] == f"{CHEST_CHANNELS[0]}_mean"


def test_feature_extraction_rejects_a_non_window():
    with pytest.raises(ValueError):
        window_features(np.zeros(10, dtype=np.float32))


# ============================================================
# Window extraction
# ============================================================

def test_only_label_homogeneous_windows_survive(wesad, tmp_path):
    config = _config(wesad)
    path = wesad["datasets_root"] / "WESAD" / "S2" / "S2.pkl"
    windows = list(iter_subject_windows(path, config))

    assert windows
    assert {window["state"] for window in windows} <= set(WESAD_STATE_3CLASS.classes)
    with open(path, "rb") as handle:
        labels = pickle.load(handle)["label"]
    for window in windows:
        segment = labels[window["start_sample"]:window["end_sample"]]
        assert len(set(segment.tolist())) == 1
        assert window["features"].shape == (len(FEATURE_NAMES),)


def test_conditions_outside_the_label_space_are_never_mapped_onto_it(wesad):
    windows = list(iter_subject_windows(
        wesad["datasets_root"] / "WESAD" / "S2" / "S2.pkl", _config(wesad)
    ))
    # The recording contains a meditation block; the 3-class space excludes it.
    assert "meditation" not in {window["state"] for window in windows}
    windows_4 = list(iter_subject_windows(
        wesad["datasets_root"] / "WESAD" / "S2" / "S2.pkl",
        _config(wesad, task="wesad_state_4class"),
    ))
    assert "meditation" in {window["state"] for window in windows_4}


def test_a_missing_chest_channel_is_an_error_not_a_silent_zero(tmp_path, wesad):
    path = wesad["datasets_root"] / "WESAD" / "S2" / "S2.pkl"
    payload = pickle.loads(path.read_bytes())
    del payload["signal"]["chest"]["EDA"]
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="EDA"):
        list(iter_subject_windows(path, _config(wesad)))


# ============================================================
# Subject-aware splitting
# ============================================================

def test_no_subject_appears_in_more_than_one_partition(wesad, tmp_path):
    summary = _run(wesad, tmp_path / "out")
    frames = _frames(tmp_path / "out")

    owners = {}
    for split, frame in frames.items():
        for subject in frame["subject"]:
            assert owners.setdefault(subject, split) == split
    assert summary["integrity"]["subjects_in_multiple_splits"] == 0
    assert summary["integrity"]["leakage_checks_passed"]


def test_every_partition_receives_at_least_one_subject(wesad, tmp_path):
    summary = _run(wesad, tmp_path / "out")
    for split in SPLITS:
        assert summary["splits"]["subject_counts"][split] >= 1
        assert summary["splits"]["counts"][split] > 0
    assert sum(summary["splits"]["subject_counts"].values()) == len(wesad["subjects"])


def test_splitting_is_deterministic_and_seed_sensitive(wesad, tmp_path):
    first = _run(wesad, tmp_path / "a", seed=42)
    again = _run(wesad, tmp_path / "b", seed=42)
    other = _run(wesad, tmp_path / "c", seed=9)

    assert first["splits"]["subjects"] == again["splits"]["subjects"]
    assert first["splits"]["sample_id_digests"] == again["splits"]["sample_id_digests"]
    assert first["splits"]["subjects"] != other["splits"]["subjects"]


def test_too_few_subjects_is_refused_rather_than_producing_an_empty_split(tmp_path):
    source = tmp_path / "standardized.parquet"
    pd.DataFrame([_standardized_row("S2"), _standardized_row("S3")]).to_parquet(source, index=False)
    builder = PhysiologyWindowBuilder(
        PhysiologyWindowConfig(source=str(source), datasets_root=str(tmp_path), sample_rate=RATE)
    )
    with pytest.raises(ValueError, match="at least 3 subjects"):
        builder.split_subjects(["S2", "S3"])


# ============================================================
# Artefacts and provenance
# ============================================================

def test_the_full_window_pool_is_used_and_recorded(wesad, tmp_path):
    summary = _run(wesad, tmp_path / "out")
    frames = _frames(tmp_path / "out")
    total = sum(len(frame) for frame in frames.values())

    assert summary["selection"]["actual_fraction"] == 1.0
    assert summary["selection"]["selected_records"] == total
    assert summary["pool"]["eligible_records"] == total
    assert summary["label_space"]["classes"] == list(WESAD_STATE_3CLASS.classes)
    assert summary["features"]["dimension"] == len(FEATURE_NAMES)


def test_every_declared_class_is_present_and_correctly_named(wesad, tmp_path):
    _run(wesad, tmp_path / "out")
    frame = pd.concat(_frames(tmp_path / "out").values())
    assert set(frame["state_id"]) == set(WESAD_STATE_3CLASS.valid_ids)
    for _, row in frame.iterrows():
        assert row["state"] == WESAD_STATE_3CLASS.name_of(int(row["state_id"]))


def test_recording_paths_are_preserved_verbatim(wesad, tmp_path):
    _run(wesad, tmp_path / "out")
    frame = pd.concat(_frames(tmp_path / "out").values())
    for _, row in frame.iterrows():
        assert row["physiology_path"] == f"WESAD/{row['subject']}/{row['subject']}.pkl"
        assert row["physiology_source"] == "file"


def test_window_identifiers_are_unique(wesad, tmp_path):
    _run(wesad, tmp_path / "out")
    frame = pd.concat(_frames(tmp_path / "out").values())
    assert frame["sample_id"].nunique() == len(frame)


def test_configuration_guards(wesad):
    with pytest.raises(ValueError):
        _config(wesad, window_seconds=0)
    with pytest.raises(ValueError):
        _config(wesad, stride_seconds=5.0, window_seconds=2.0)
    with pytest.raises(ValueError):
        _config(wesad, split_ratios=(0.5, 0.1, 0.1))
    with pytest.raises(ValueError, match="chest"):
        _config(wesad, signal_group="wrist")


# ============================================================
# Independent validation
# ============================================================

def test_the_independent_validator_passes_a_freshly_built_experiment(wesad, tmp_path):
    _run(wesad, tmp_path / "out")
    report = validate_physiology_experiment(
        tmp_path / "out", check_files=-1, datasets_root=wesad["datasets_root"]
    )
    assert report.passed, report.failures
    names = {check["check"] for check in report.checks}
    assert "subjects_are_disjoint_across_splits" in names
    assert "state_labels_valid" in names
    assert "full_pool_is_used" in names


def test_the_validator_detects_a_subject_spanning_two_partitions(wesad, tmp_path):
    output = tmp_path / "out"
    _run(wesad, output)
    train = pd.read_parquet(output / "train.parquet")
    test = pd.read_parquet(output / "test.parquet")
    # Move one training subject's first window into test, keeping ids unique.
    leaked = train.iloc[[0]].copy()
    leaked["experiment_split"] = "test"
    leaked["sample_id"] = leaked["sample_id"] + "|leaked"
    pd.concat([test, leaked], ignore_index=True).to_parquet(output / "test.parquet", index=False)

    report = validate_physiology_experiment(
        output, check_files=0, datasets_root=wesad["datasets_root"]
    )
    assert not report.passed
    failed = {check["check"] for check in report.failures}
    assert "subjects_are_disjoint_across_splits" in failed
