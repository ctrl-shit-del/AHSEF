import json

import pandas as pd

from src.preprocessing.standardization.experiments import (
    build_emotion_7class,
    build_iemocap_session_experiment,
    build_physiology_metadata,
    build_sentiment,
)
from src.preprocessing.standardization.validator import validate_parquet


def _metadata() -> pd.DataFrame:
    rows = []
    for dataset, split, emotion, group in [
        ("AffectNet+", "train", "happy", None),
        ("AffectNet+", "validation", "sad", None),
        ("MELD", "test", "angry", None),
        ("IEMOCAP", "Session1", "neutral", "Session1"),
        ("IEMOCAP", "Session4", "fear", "Session4"),
        ("IEMOCAP", "Session5", "disgust", "Session5"),
    ]:
        rows.append({
            "sample_id": f"{dataset}-{split}", "dataset": dataset, "split": split,
            "training_split": split if split in {"train", "validation", "test"} else None,
            "evaluation_group": group, "target_type": "categorical_emotion",
            "emotion_target_valid": True, "canonical_emotion": emotion,
            "canonical_emotion_id": {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fear": 4, "disgust": 5}[emotion],
            "has_audio": False, "has_video": False, "has_image": True, "has_text": False, "has_physiology": False,
            "audio_source": None, "video_source": None, "image_source": "file", "text_source": None, "physiology_source": None,
            "feature_file": None, "feature_split": None, "feature_id": None,
        })
    for split, feature_split in [("train", "train"), ("validation", "valid"), ("test", "test")]:
        rows.append({
            "sample_id": f"mosei-{split}", "dataset": "CMU-MOSEI", "split": split, "training_split": split,
            "evaluation_group": None, "target_type": "sentiment", "emotion_target_valid": False,
            "canonical_emotion": None, "canonical_emotion_id": None, "sentiment_target_valid": True, "sentiment_score": 1.0,
            "has_audio": True, "has_video": True, "has_image": False, "has_text": True, "has_physiology": False,
            "audio_source": "feature_container", "video_source": "feature_container", "image_source": None, "text_source": "metadata", "physiology_source": None,
            "feature_file": "CMU-MOSEI/Processed/aligned_50.pkl", "feature_split": feature_split, "feature_id": f"id-{split}",
        })
    rows.append({
        "sample_id": "wesad-1", "dataset": "WESAD", "split": None, "training_split": None, "evaluation_group": None,
        "target_type": "physiological_state", "emotion_target_valid": False, "canonical_emotion": None, "canonical_emotion_id": None,
        "has_audio": False, "has_video": False, "has_image": False, "has_text": False, "has_physiology": True,
        "audio_source": None, "video_source": None, "image_source": None, "text_source": None, "physiology_source": "file",
        "physiology_path": "WESAD/S1/S1.pkl", "feature_file": None, "feature_split": None, "feature_id": None,
    })
    return pd.DataFrame(rows)


def test_builders_preserve_explicit_policies(tmp_path):
    source = tmp_path / "standardized.parquet"
    _metadata().to_parquet(source, index=False)
    emotion = build_emotion_7class(source, tmp_path / "emotion")
    assert emotion["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert not (tmp_path / "emotion" / "train.parquet").read_bytes() == b""

    iemocap = build_iemocap_session_experiment("Session5", "Session4", source, tmp_path / "iemocap")
    assert iemocap["counts"] == {"train": 1, "validation": 1, "test": 1}

    sentiment = build_sentiment(source, tmp_path / "sentiment")
    assert sentiment["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert json.loads((tmp_path / "sentiment" / "summary.json").read_text())["target_definition"] == "sentiment_score"

    physiology = build_physiology_metadata(source, tmp_path / "physiology")
    assert physiology["records"] == 1


def test_streaming_validator_rejects_missing_modality_source(tmp_path):
    source = tmp_path / "standardized.parquet"
    metadata = _metadata()
    metadata["emotion"] = metadata["canonical_emotion"]
    metadata["canonical_emotion_valid"] = metadata["canonical_emotion"].notna()
    metadata["vad_target_valid"] = False
    metadata.to_parquet(source, index=False)
    report = validate_parquet(source, batch_size=2)
    assert report["rows"] == len(metadata)

    metadata.loc[0, "image_source"] = None
    metadata.to_parquet(source, index=False)
    try:
        validate_parquet(source, batch_size=2)
    except ValueError as error:
        assert "image is available" in str(error)
    else:
        raise AssertionError("Validator accepted an available modality without a source")
