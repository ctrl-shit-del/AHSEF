"""Sampler tests for the audio, text, and video modality pools.

``test_sampler.py`` already covers the image pool in depth.  This module covers
what generalising to the other modalities introduced: per-source payload rules
(MELD text in a metadata column, MSP-Podcast text in a file), the full-data
mode used by text, and the guarantee that a modality only ever selects records
that really carry that modality.

Every fixture is a tiny synthetic standardized table in ``tmp_path``.  No real
dataset, audio file, or video file is touched.
"""

import pandas as pd
import pytest

from src.common.labels import EMOTION_7CLASS
from src.preprocessing.sampling.sampler import SPLITS, ExperimentSampler, SamplerConfig


CLASS_NAMES = EMOTION_7CLASS.classes

#: Audio pool: three corpora, unbalanced the way the real ones are.
AUDIO_POOL = {
    "RAVDESS": {"train": 24},
    "CREMA-D": {"train": 40},
    "MSP-Podcast": {"train": 60, "test": 12},
}
#: Text pool: MELD is metadata-backed, MSP-Podcast is file-backed.
TEXT_POOL = {"MELD": {"train": 30, "test": 8}, "MSP-Podcast": {"train": 50}}
VIDEO_POOL = {"MELD": {"train": 32, "test": 8}}


def _base(dataset, official, class_id, index, kind="x", **overrides):
    row = {
        "sample_id": f"{kind}|{dataset}|{official}|{class_id}|{index}",
        "dataset": dataset,
        "split": official,
        "training_split": official,
        "evaluation_group": None,
        "target_type": "categorical_emotion",
        "emotion_target_valid": True,
        "canonical_emotion": CLASS_NAMES[class_id],
        "canonical_emotion_id": class_id,
        "canonical_emotion_valid": True,
        "speaker": f"spk{index}",
        "has_audio": False, "has_video": False, "has_image": False,
        "has_text": False, "has_physiology": False,
        "audio_source": None, "video_source": None, "image_source": None,
        "text_source": None, "physiology_source": None,
        "audio_path": None, "video_path": None, "image_path": None,
        "physiology_path": None, "text": None, "text_path": None,
        "extras": "{}",
    }
    row.update(overrides)
    return row


def _audio(dataset, official, class_id, index, **overrides):
    payload = {
        "has_audio": True,
        "audio_source": "file",
        "audio_path": f"{dataset}/{official}/{class_id}/{index}.wav",
    }
    payload.update(overrides)
    return _base(dataset, official, class_id, index, kind="audio", **payload)


def _text(dataset, official, class_id, index, **overrides):
    if dataset == "MSP-Podcast":
        payload = {
            "has_text": True, "text_source": "file",
            "text_path": f"{dataset}/Transcripts/{class_id}-{index}.txt",
        }
    else:
        payload = {
            "has_text": True, "text_source": "metadata",
            "text": f"an utterance about {CLASS_NAMES[class_id]} number {index}",
        }
    payload.update(overrides)
    return _base(dataset, official, class_id, index, kind="text", **payload)


def _video(dataset, official, class_id, index, **overrides):
    payload = {
        "has_video": True,
        "video_source": "file",
        "video_path": f"{dataset}/{official}/dia{class_id}_utt{index}.mp4",
    }
    payload.update(overrides)
    return _base(dataset, official, class_id, index, kind="video", **payload)


def _frame() -> pd.DataFrame:
    rows = []
    for pool, builder in ((AUDIO_POOL, _audio), (TEXT_POOL, _text), (VIDEO_POOL, _video)):
        for dataset, splits in pool.items():
            for official, count in splits.items():
                for class_id in range(len(CLASS_NAMES)):
                    for index in range(count):
                        rows.append(builder(dataset, official, class_id, index))

    # Records that must never be selected by any modality.
    for index in range(6):
        # CMU-MOSEI: real modalities, but sentiment target and feature-container
        # provenance -- excluded by the task rule and the modality rule alike.
        rows.append(_base(
            "CMU-MOSEI", "train", 1, f"sentiment-{index}",
            target_type="sentiment", emotion_target_valid=False,
            canonical_emotion=None, canonical_emotion_id=None, canonical_emotion_valid=False,
            has_audio=True, audio_source="feature_container",
            has_video=True, video_source="feature_container",
            has_text=True, text_source="metadata", text="a sentiment-annotated clip",
        ))
        # Declared text with neither payload column populated.
        rows.append(_base(
            "MELD", "train", 2, f"empty-text-{index}",
            has_text=True, text_source="metadata", text=None,
        ))
        # Declared audio without a path.
        rows.append(_audio("RAVDESS", "train", 3, f"nopath-{index}", audio_path=None))
        # Valid audio whose emotion target is invalid.
        rows.append(_audio(
            "CREMA-D", "train", 4, f"invalid-{index}",
            emotion_target_valid=False, canonical_emotion=None, canonical_emotion_id=None,
        ))
    return pd.DataFrame(rows)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "standardized.parquet"
    _frame().to_parquet(path, index=False)
    return path


def _run(source, output_dir, modality, **overrides):
    defaults = dict(
        modality=modality, task="emotion_7class", fraction=0.25, seed=42,
        source=str(source), batch_size=128,
    )
    defaults.update(overrides)
    return ExperimentSampler(SamplerConfig(**defaults)).run(output_dir)


def _frames(output_dir):
    return {split: pd.read_parquet(output_dir / f"{split}.parquet") for split in SPLITS}


# ============================================================
# Modality isolation
# ============================================================

@pytest.mark.parametrize(
    "modality,column,expected_datasets",
    [
        ("audio", "audio_path", {"RAVDESS", "CREMA-D", "MSP-Podcast"}),
        ("video", "video_path", {"MELD"}),
    ],
)
def test_only_records_carrying_the_modality_are_selected(
    source, tmp_path, modality, column, expected_datasets
):
    summary = _run(source, tmp_path / modality, modality)
    frames = _frames(tmp_path / modality)
    selected = pd.concat(frames.values())

    assert set(selected["dataset"]) <= expected_datasets
    assert selected[column].notna().all()
    assert selected[f"has_{modality}"].all()
    assert (selected[f"{modality}_source"] == "file").all()
    assert summary["integrity"]["records_with_missing_modality_source"] == 0
    assert summary["integrity"]["leakage_checks_passed"]


def test_text_selects_both_metadata_and_file_backed_records(source, tmp_path):
    summary = _run(source, tmp_path / "text", "text", fraction=1.0)
    selected = pd.concat(_frames(tmp_path / "text").values())

    assert set(selected["dataset"]) == {"MELD", "MSP-Podcast"}
    metadata_backed = selected[selected["text_source"] == "metadata"]
    file_backed = selected[selected["text_source"] == "file"]
    assert len(metadata_backed) and len(file_backed)
    # Each source must carry its own payload, and only its own.
    assert metadata_backed["text"].notna().all()
    assert file_backed["text_path"].notna().all()
    assert file_backed["text"].isna().all()
    assert summary["integrity"]["records_with_missing_modality_source"] == 0


def test_records_declaring_text_without_a_payload_are_excluded(source, tmp_path):
    _run(source, tmp_path / "text", "text", fraction=1.0)
    selected = pd.concat(_frames(tmp_path / "text").values())
    assert not any("empty-text" in sample_id for sample_id in selected["sample_id"])


def test_sentiment_records_never_enter_the_emotion_task(source, tmp_path):
    for modality in ("audio", "video", "text"):
        _run(source, tmp_path / f"x-{modality}", modality, fraction=1.0)
        selected = pd.concat(_frames(tmp_path / f"x-{modality}").values())
        assert "CMU-MOSEI" not in set(selected["dataset"])
        assert selected["canonical_emotion_id"].isin(range(7)).all()


# ============================================================
# Fraction and full-data modes
# ============================================================

def test_quarter_fraction_is_realised_within_rounding(source, tmp_path):
    summary = _run(source, tmp_path / "audio", "audio", fraction=0.25)
    assert summary["selection"]["actual_fraction"] == pytest.approx(0.25, abs=0.02)
    assert summary["selection"]["selected_records"] < summary["pool"]["eligible_records"]


def test_full_data_mode_absorbs_the_entire_eligible_pool(source, tmp_path):
    summary = _run(source, tmp_path / "text", "text", fraction=1.0, holdout_overflow="absorb")
    assert summary["selection"]["actual_fraction"] == 1.0
    assert summary["selection"]["selected_records"] == summary["pool"]["eligible_records"]
    assert summary["deviations"]["unplaced_eligible_records"] == 0
    counts = summary["splits"]["counts"]
    assert sum(counts.values()) == summary["pool"]["eligible_records"]


def test_dropping_overflow_reports_the_shortfall_instead_of_hiding_it(source, tmp_path):
    # MELD's official test partition is proportionally larger than the 10%
    # experiment test quota, so at full scale some records fit nowhere. Under
    # the 'drop' policy the holdout splits keep exactly their quota and the
    # shortfall lands in training, which is why it must be reported.
    dropped = _run(source, tmp_path / "drop", "text", fraction=1.0, holdout_overflow="drop")
    absorbed = _run(source, tmp_path / "absorb", "text", fraction=1.0, holdout_overflow="absorb")

    unplaced = dropped["deviations"]["unplaced_eligible_records"]
    assert unplaced > 0
    assert dropped["deviations"]["unplaced_reason"]
    assert dropped["deviations"]["holdout_overflow"] == "drop"
    assert (
        dropped["selection"]["selected_records"] + unplaced
        == dropped["pool"]["eligible_records"]
    )
    # Absorbing recovers exactly the dropped records, all of them into holdout
    # splits: training is identical under both policies.
    assert (
        absorbed["selection"]["selected_records"]
        == dropped["selection"]["selected_records"] + unplaced
    )
    assert absorbed["splits"]["counts"]["train"] == dropped["splits"]["counts"]["train"]
    assert absorbed["splits"]["counts"]["test"] > dropped["splits"]["counts"]["test"]


def test_absorbing_is_a_no_op_when_every_quota_can_be_filled(source, tmp_path):
    # At a quarter of the pool every stratum has enough unlocked records, so
    # the two policies must produce byte-identical selections. This is what
    # makes the completed image experiment reproducible either way.
    dropped = _run(source, tmp_path / "d25", "video", fraction=0.25, holdout_overflow="drop")
    absorbed = _run(source, tmp_path / "a25", "video", fraction=0.25, holdout_overflow="absorb")
    assert dropped["deviations"]["unplaced_eligible_records"] == 0
    assert dropped["selection"]["plan_digest"] == absorbed["selection"]["plan_digest"]
    assert dropped["splits"]["sample_id_digests"] == absorbed["splits"]["sample_id_digests"]


def test_neither_overflow_mode_leaks_a_holdout_into_training(source, tmp_path):
    for mode in ("drop", "absorb"):
        summary = _run(
            source, tmp_path / f"o-{mode}", "text", fraction=1.0, holdout_overflow=mode
        )
        train = pd.read_parquet(tmp_path / f"o-{mode}" / "train.parquet")
        assert not set(train["training_split"]) & {"test", "validation"}
        assert summary["integrity"]["official_holdout_in_train"] == 0


def test_split_ratios_hold_for_every_modality(source, tmp_path):
    for modality, fraction in (("audio", 0.25), ("video", 0.25), ("text", 1.0)):
        summary = _run(source, tmp_path / f"r-{modality}", modality, fraction=fraction)
        # Ratios are measured against what was actually written, so the drop
        # policy keeps them exact even where an official holdout overflows.
        actual = summary["splits"]["actual_ratios"]
        assert actual["train"] == pytest.approx(0.8, abs=0.03)
        assert actual["validation"] == pytest.approx(0.1, abs=0.03)
        assert actual["test"] == pytest.approx(0.1, abs=0.03)


# ============================================================
# Representation, determinism, leakage
# ============================================================

def test_every_contributing_dataset_and_class_survives_sampling(source, tmp_path):
    summary = _run(source, tmp_path / "audio", "audio", fraction=0.25)
    assert set(summary["selection"]["dataset_counts"]) == {"RAVDESS", "CREMA-D", "MSP-Podcast"}
    assert set(summary["selection"]["class_counts"]) == {str(index) for index in range(7)}
    assert summary["representativeness"]["max_dataset_share_drift"] < 0.02
    assert summary["representativeness"]["max_class_share_drift"] < 0.02


def test_a_large_dataset_cannot_wipe_out_a_small_one(source, tmp_path):
    summary = _run(source, tmp_path / "audio", "audio", fraction=0.25)
    counts = summary["selection"]["dataset_counts"]
    assert counts["RAVDESS"] > 0
    # Proportional allocation, not winner-takes-all.
    pool = summary["pool"]["dataset_counts"]
    for dataset in counts:
        assert counts[dataset] == pytest.approx(pool[dataset] * 0.25, abs=7)


def test_sampling_is_deterministic_and_seed_sensitive(source, tmp_path):
    first = _run(source, tmp_path / "a", "audio", seed=42)
    again = _run(source, tmp_path / "b", "audio", seed=42)
    other = _run(source, tmp_path / "c", "audio", seed=7)

    assert first["selection"]["plan_digest"] == again["selection"]["plan_digest"]
    assert first["splits"]["sample_id_digests"] == again["splits"]["sample_id_digests"]
    assert first["selection"]["plan_digest"] != other["selection"]["plan_digest"]


def test_no_duplicate_or_cross_split_records(source, tmp_path):
    for modality in ("audio", "video", "text"):
        summary = _run(source, tmp_path / f"d-{modality}", modality, fraction=1.0)
        frames = _frames(tmp_path / f"d-{modality}")
        ids = [sample_id for frame in frames.values() for sample_id in frame["sample_id"]]
        assert len(ids) == len(set(ids))
        assert summary["integrity"]["duplicate_sample_ids"] == 0
        assert summary["integrity"]["cross_split_sample_ids"] == 0


def test_official_holdouts_never_reach_experiment_training(source, tmp_path):
    for modality in ("audio", "video", "text"):
        summary = _run(source, tmp_path / f"h-{modality}", modality, fraction=1.0)
        train = _frames(tmp_path / f"h-{modality}")["train"]
        assert not set(train["training_split"]) & {"test", "validation"}
        assert summary["integrity"]["official_holdout_in_train"] == 0


def test_modality_paths_are_copied_verbatim(source, tmp_path):
    original = pd.read_parquet(source).set_index("sample_id")
    _run(source, tmp_path / "audio", "audio", fraction=1.0)
    selected = pd.concat(_frames(tmp_path / "audio").values()).set_index("sample_id")
    for sample_id, row in selected.iterrows():
        assert row["audio_path"] == original.loc[sample_id, "audio_path"]


def test_experiment_split_column_labels_every_partition(source, tmp_path):
    _run(source, tmp_path / "video", "video", fraction=1.0)
    for split, frame in _frames(tmp_path / "video").items():
        assert (frame["experiment_split"] == split).all()
