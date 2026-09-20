"""Tiny synthetic experiment fixtures for the modality baseline tests.

Every helper here writes a handful of records into ``tmp_path``: a few hundred
audio samples, a one-line transcript, a ten-frame clip.  Nothing reads the real
corpora, and nothing produced here is large enough to be mistaken for a real
experiment.

The manifests deliberately give each split a *different* class distribution, so
a class weight computed from the wrong partition is detectable rather than
merely suspicious.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.labels import EMOTION_7CLASS, WESAD_STATE_3CLASS


SPLIT_CLASSES = {
    "train": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 1, 1, 0, 2, 3, 4, 5, 6, 1, 0],
    "validation": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
    "test": [6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2],
}

SAMPLE_RATE = 8_000


def _base_row(split: str, index: int, dataset: str, label: int) -> dict:
    return {
        "sample_id": f"{split}-{index}",
        "dataset": dataset,
        "canonical_emotion_id": label,
        "training_split": "train",
        "experiment_split": split,
        "has_audio": False, "has_video": False, "has_image": False,
        "has_text": False, "has_physiology": False,
        "audio_source": None, "video_source": None, "image_source": None,
        "text_source": None, "physiology_source": None,
        "audio_path": None, "video_path": None, "image_path": None,
        "text": None, "text_path": None, "physiology_path": None,
    }


def _write(metadata_dir: Path, split: str, rows: list[dict]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(metadata_dir / f"{split}.parquet", index=False)


# ============================================================
# Audio
# ============================================================

def write_audio_experiment(
    datasets_root: Path, metadata_dir: Path, datasets=("RAVDESS", "CREMA-D", "MSP-Podcast")
) -> None:
    """Write short class-dependent tones plus the three split manifests."""
    import soundfile as sf

    for split, labels in SPLIT_CLASSES.items():
        rows = []
        for index, label in enumerate(labels):
            dataset = datasets[index % len(datasets)]
            relative = f"{dataset}/{split}/{index}.wav"
            path = datasets_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            duration = 0.25
            time = np.arange(int(duration * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
            tone = 0.4 * np.sin(2 * np.pi * (110.0 * (label + 1)) * time)
            sf.write(path, tone.astype(np.float32), SAMPLE_RATE)

            row = _base_row(split, index, dataset, label)
            row.update(has_audio=True, audio_source="file", audio_path=relative)
            rows.append(row)
        _write(metadata_dir, split, rows)


# ============================================================
# Text
# ============================================================

WORDS = {
    0: "the meeting is scheduled for tuesday",
    1: "this is wonderful news thank you",
    2: "i miss them more than i can say",
    3: "that is completely unacceptable",
    4: "something is moving behind the door",
    5: "i cannot even look at that",
    6: "i did not expect any of this",
}


def write_text_experiment(datasets_root: Path, metadata_dir: Path) -> None:
    """Write MELD-style metadata text and MSP-Podcast-style transcript files."""
    for split, labels in SPLIT_CLASSES.items():
        rows = []
        for index, label in enumerate(labels):
            sentence = f"{WORDS[label]} number {index}"
            if index % 2:
                dataset, relative = "MSP-Podcast", f"MSP-Podcast/Transcripts/{split}-{index}.txt"
                path = datasets_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(sentence, encoding="utf-8")
                payload = {"has_text": True, "text_source": "file", "text_path": relative}
            else:
                dataset = "MELD"
                payload = {"has_text": True, "text_source": "metadata", "text": sentence}
            row = _base_row(split, index, dataset, label)
            row.update(payload)
            rows.append(row)
        _write(metadata_dir, split, rows)


# ============================================================
# Video
# ============================================================

def write_video_experiment(datasets_root: Path, metadata_dir: Path, frames: int = 6) -> bool:
    """Write tiny MELD-style clips; returns False if no encoder is available."""
    import cv2

    for split, labels in SPLIT_CLASSES.items():
        rows = []
        for index, label in enumerate(labels):
            relative = f"MELD/{split}/dia{index}_utt0.mp4"
            path = datasets_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (16, 16)
            )
            if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
                writer.release()
                return False
            for step in range(frames):
                value = (label * 30 + step * 3) % 256
                writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
            writer.release()

            row = _base_row(split, index, "MELD", label)
            row.update(has_video=True, video_source="file", video_path=relative)
            rows.append(row)
        _write(metadata_dir, split, rows)
    return True


# ============================================================
# Physiology
# ============================================================

#: Subject-disjoint by construction: each split owns its own subjects.
PHYSIOLOGY_SUBJECTS = {
    "train": ("S2", "S3", "S4"),
    "validation": ("S5",),
    "test": ("S6",),
}
FEATURE_DIM = 12


def write_physiology_experiment(
    metadata_dir: Path, windows_per_subject: int = 12, feature_dim: int = FEATURE_DIM
) -> None:
    """Write subject-disjoint window manifests with separable state features."""
    generator = np.random.default_rng(0)
    for split, subjects in PHYSIOLOGY_SUBJECTS.items():
        rows = []
        for subject in subjects:
            for index in range(windows_per_subject):
                state_id = index % WESAD_STATE_3CLASS.num_classes
                features = generator.normal(loc=float(state_id), scale=0.2, size=feature_dim)
                rows.append({
                    "sample_id": f"WESAD|{subject}|{index:06d}",
                    "dataset": "WESAD",
                    "subject": subject,
                    "state": WESAD_STATE_3CLASS.name_of(state_id),
                    "state_id": state_id,
                    "window_index": index,
                    "start_second": float(index),
                    "end_second": float(index + 1),
                    "features": features.astype(np.float32).tolist(),
                    "physiology_path": f"WESAD/{subject}/{subject}.pkl",
                    "has_physiology": True,
                    "physiology_source": "file",
                    "training_split": None,
                    "experiment_split": split,
                })
        _write(metadata_dir, split, rows)


def write_leaky_physiology_experiment(metadata_dir: Path) -> None:
    """Same, but with one subject deliberately present in train and validation."""
    write_physiology_experiment(metadata_dir)
    train = pd.read_parquet(metadata_dir / "train.parquet")
    validation = pd.read_parquet(metadata_dir / "validation.parquet")
    leaked = train.iloc[:3].copy()
    leaked["experiment_split"] = "validation"
    leaked["sample_id"] = leaked["sample_id"] + "|leaked"
    pd.concat([validation, leaked], ignore_index=True).to_parquet(
        metadata_dir / "validation.parquet", index=False
    )


CLASS_NAMES = EMOTION_7CLASS.classes
