"""Focused tests for the representative fraction sampler.

Every test builds a tiny synthetic standardized table in ``tmp_path``.  No
real dataset, image file, or 25% run is ever touched.
"""

import json

import pandas as pd
import pytest

from src.preprocessing.sampling.sampler import (
    SPLITS,
    ExperimentSampler,
    SamplerConfig,
)
from src.preprocessing.standardization.targets import CANONICAL_EMOTIONS


CLASS_NAMES = sorted(CANONICAL_EMOTIONS, key=CANONICAL_EMOTIONS.get)

# Pool shape: AffectNet+ 200/class (180 train + 20 official validation),
# FERPlus 100/class (all train), RAF-DB 40/class (30 train + 10 official test).
POOL = {
    "AffectNet+": {"train": 180, "validation": 20},
    "FERPlus": {"train": 100},
    "RAF-DB": {"train": 30, "test": 10},
}
ELIGIBLE_TOTAL = sum(sum(splits.values()) for splits in POOL.values()) * len(CLASS_NAMES)


def _record(dataset, official, class_id, index, **overrides):
    row = {
        "sample_id": f"{dataset}|{official}|{class_id}|{index}",
        "dataset": dataset,
        "split": official,
        "training_split": official,
        "evaluation_group": None,
        "target_type": "categorical_emotion",
        "emotion_target_valid": True,
        "canonical_emotion": CLASS_NAMES[class_id],
        "canonical_emotion_id": class_id,
        "canonical_emotion_valid": True,
        "has_audio": False,
        "has_video": False,
        "has_image": True,
        "has_text": False,
        "has_physiology": False,
        "audio_source": None,
        "video_source": None,
        "image_source": "file",
        "text_source": None,
        "physiology_source": None,
        "audio_path": None,
        "video_path": None,
        "image_path": f"{dataset}/{official}/{class_id}/{index}.png",
        "physiology_path": None,
        "text": None,
        "extras": "{}",
    }
    row.update(overrides)
    return row


def _standardized_frame() -> pd.DataFrame:
    rows = []
    for dataset, splits in POOL.items():
        for official, count in splits.items():
            for class_id in range(len(CLASS_NAMES)):
                for index in range(count):
                    rows.append(_record(dataset, official, class_id, index))

    # Records that must never be selected.
    for index in range(10):
        rows.append(_record(
            "MELD", "train", 0, f"video-{index}",
            has_image=False, image_source=None, image_path=None,
            has_video=True, video_source="file", video_path=f"MELD/{index}.mp4",
        ))
    for index in range(5):
        rows.append(_record("AffectNet+", "train", 1, f"derived-{index}", image_source="derived"))
        rows.append(_record("AffectNet+", "train", 1, f"nopath-{index}", image_path=None))
        rows.append(_record(
            "AffectNet+", "train", 1, f"invalid-{index}",
            emotion_target_valid=False, canonical_emotion=None, canonical_emotion_id=None,
        ))
        rows.append(_record("AffectNet+", "train", 1, f"vad-{index}", target_type="vad"))
    return pd.DataFrame(rows)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "standardized.parquet"
    _standardized_frame().to_parquet(path, index=False)
    return path


def _config(source, **overrides):
    defaults = dict(
        modality="image",
        task="emotion_7class",
        fraction=0.25,
        seed=42,
        source=str(source),
        batch_size=256,
    )
    defaults.update(overrides)
    return SamplerConfig(**defaults)


def _run(source, output_dir, **overrides):
    return ExperimentSampler(_config(source, **overrides)).run(output_dir)


def _frames(output_dir):
    return {split: pd.read_parquet(output_dir / f"{split}.parquet") for split in SPLITS}


# ============================================================
# Determinism
# ============================================================

def test_sampling_is_deterministic_for_a_fixed_seed(source, tmp_path):
    first = _run(source, tmp_path / "a")
    second = _run(source, tmp_path / "b")

    assert first["selection"]["plan_digest"] == second["selection"]["plan_digest"]
    assert first["splits"]["sample_id_digests"] == second["splits"]["sample_id_digests"]
    assert first["splits"]["counts"] == second["splits"]["counts"]

    left, right = _frames(tmp_path / "a"), _frames(tmp_path / "b")
    for split in SPLITS:
        assert sorted(left[split]["sample_id"]) == sorted(right[split]["sample_id"])


def test_a_different_seed_changes_the_selection(source, tmp_path):
    first = _run(source, tmp_path / "a", seed=42)
    other = _run(source, tmp_path / "c", seed=7)
    assert first["selection"]["plan_digest"] != other["selection"]["plan_digest"]
    # The size contract is unchanged; only membership moves.
    assert first["splits"]["counts"] == other["splits"]["counts"]


# ============================================================
# Representativeness
# ============================================================

def test_selected_fraction_is_close_to_the_request_and_recorded_exactly(source, tmp_path):
    summary = _run(source, tmp_path / "out")
    selection = summary["selection"]

    assert summary["pool"]["eligible_records"] == ELIGIBLE_TOTAL
    assert selection["selected_records"] == pytest.approx(ELIGIBLE_TOTAL * 0.25, abs=25)
    assert selection["actual_fraction"] == pytest.approx(0.25, abs=0.01)
    assert selection["actual_fraction"] == selection["selected_records"] / ELIGIBLE_TOTAL


def test_dataset_and_class_representation_are_preserved(source, tmp_path):
    summary = _run(source, tmp_path / "out")

    assert summary["representativeness"]["max_dataset_share_drift"] <= 0.01
    assert summary["representativeness"]["max_class_share_drift"] <= 0.01

    # AffectNet+ is the largest contributor but must not swallow the subset.
    selected = summary["selection"]["dataset_counts"]
    assert set(selected) == {"AffectNet+", "FERPlus", "RAF-DB"}
    assert all(count > 0 for count in selected.values())
    assert len(summary["selection"]["class_counts"]) == len(CLASS_NAMES)
    assert all(count > 0 for count in summary["selection"]["class_counts"].values())


def test_split_ratios_are_eighty_ten_ten(source, tmp_path):
    summary = _run(source, tmp_path / "out")
    actual = summary["splits"]["actual_ratios"]
    assert actual["train"] == pytest.approx(0.8, abs=0.02)
    assert actual["validation"] == pytest.approx(0.1, abs=0.02)
    assert actual["test"] == pytest.approx(0.1, abs=0.02)
    assert sum(summary["splits"]["counts"].values()) == summary["selection"]["selected_records"]


# ============================================================
# Integrity
# ============================================================

def test_no_duplicate_or_shared_sample_ids(source, tmp_path):
    summary = _run(source, tmp_path / "out")
    frames = _frames(tmp_path / "out")

    ids = {split: set(frames[split]["sample_id"]) for split in SPLITS}
    for split in SPLITS:
        assert len(ids[split]) == len(frames[split])
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])

    assert summary["integrity"]["duplicate_sample_ids"] == 0
    assert summary["integrity"]["cross_split_sample_ids"] == 0
    assert summary["integrity"]["leakage_checks_passed"] is True


def test_only_file_backed_images_from_contributing_datasets_are_selected(source, tmp_path):
    _run(source, tmp_path / "out")
    frames = _frames(tmp_path / "out")
    combined = pd.concat(frames.values(), ignore_index=True)

    assert set(combined["dataset"]) <= {"AffectNet+", "FERPlus", "RAF-DB"}
    assert "MELD" not in set(combined["dataset"])
    assert combined["has_image"].all()
    assert (combined["image_source"] == "file").all()
    assert combined["image_path"].notna().all()
    assert (combined["target_type"] == "categorical_emotion").all()
    assert combined["emotion_target_valid"].all()
    assert set(combined["canonical_emotion_id"].astype(int)) <= set(range(7))
    assert not combined["sample_id"].str.contains("derived-|nopath-|invalid-|vad-").any()


def test_canonical_label_ids_match_the_canonical_mapping(source, tmp_path):
    _run(source, tmp_path / "out")
    combined = pd.concat(_frames(tmp_path / "out").values(), ignore_index=True)
    for _, row in combined.head(50).iterrows():
        assert CANONICAL_EMOTIONS[row["canonical_emotion"]] == int(row["canonical_emotion_id"])


def test_official_holdouts_never_reach_the_training_split(source, tmp_path):
    summary = _run(source, tmp_path / "out")
    frames = _frames(tmp_path / "out")

    assert set(frames["train"]["training_split"]) == {"train"}
    assert summary["integrity"]["official_holdout_in_train"] == 0

    # RAF-DB official test records may only appear in the experiment test split.
    for split in ("train", "validation"):
        official = frames[split]["training_split"]
        assert "test" not in set(official)


def test_source_paths_are_preserved_verbatim(source, tmp_path):
    _run(source, tmp_path / "out")
    frames = _frames(tmp_path / "out")
    expected = _standardized_frame().set_index("sample_id")["image_path"].to_dict()

    for split in SPLITS:
        for sample_id, path in zip(frames[split]["sample_id"], frames[split]["image_path"]):
            assert path == expected[sample_id]


def test_experiment_split_column_is_appended(source, tmp_path):
    _run(source, tmp_path / "out")
    frames = _frames(tmp_path / "out")
    for split in SPLITS:
        assert set(frames[split]["experiment_split"]) == {split}


# ============================================================
# Documented deviations and configuration
# ============================================================

def test_summary_documents_method_stratification_and_deviations(source, tmp_path):
    summary = _run(source, tmp_path / "out")

    assert summary["stratification_fields"] == ["dataset", "canonical_emotion_id"]
    assert "stratified proportional selection" in summary["sampling_method"]
    assert summary["seed"] == 42
    assert summary["config"]["fraction"] == 0.25
    assert summary["pool"]["dataset_class_counts"]
    assert summary["splits"]["dataset_counts"]["train"]
    assert summary["splits"]["class_counts"]["test"]
    assert "policy" in summary["deviations"]
    assert len(summary["strata"]) == len(POOL) * len(CLASS_NAMES)

    on_disk = json.loads((tmp_path / "out" / "sampling_summary.json").read_text())
    assert on_disk["selection"]["plan_digest"] == summary["selection"]["plan_digest"]


def test_tiny_strata_are_raised_to_the_minimum_and_documented(source, tmp_path):
    summary = _run(source, tmp_path / "tiny", fraction=0.001)
    adjustments = summary["deviations"]["minimum_per_stratum_adjustments"]

    assert adjustments, "expected the minimum-per-stratum floor to be documented"
    assert all(entry["enforced_minimum"] >= 1 for entry in adjustments)
    assert summary["selection"]["selected_records"] >= len(summary["strata"])
    assert summary["selection"]["actual_fraction"] > 0.001


def test_dataset_restriction_is_honoured(source, tmp_path):
    summary = _run(source, tmp_path / "rafdb", datasets=("RAF-DB",))
    assert summary["contributing_datasets"] == ["RAF-DB"]
    assert set(summary["selection"]["dataset_counts"]) == {"RAF-DB"}


def test_free_policy_ignores_official_boundaries(source, tmp_path):
    summary = _run(source, tmp_path / "free", split_policy="free")
    frames = _frames(tmp_path / "free")
    official_in_train = set(frames["train"]["training_split"])
    assert summary["config"]["split_policy"] == "free"
    assert official_in_train - {"train"}, "free policy should mix official partitions"


def test_sampler_rejects_invalid_configuration(source):
    with pytest.raises(ValueError):
        SamplerConfig(fraction=0.0)
    with pytest.raises(ValueError):
        SamplerConfig(split_ratios=(0.8, 0.1, 0.2))
    with pytest.raises(ValueError):
        SamplerConfig(split_policy="whatever")
    with pytest.raises(ValueError):
        ExperimentSampler(SamplerConfig(modality="telepathy", source=str(source)))
    with pytest.raises(FileNotFoundError):
        ExperimentSampler(_config("does-not-exist.parquet")).scan()
