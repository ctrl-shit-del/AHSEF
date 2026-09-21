"""Lazy, modality-filtered audio dataset for categorical emotion experiments.

Contributing datasets are RAVDESS, CREMA-D, IEMOCAP, and MSP-Podcast -- every
corpus whose standardized metadata declares a file-backed ``audio_path`` and a
valid canonical emotion.  CMU-MOSEI audio is deliberately absent: it is
feature-container-backed and carries a sentiment target, not a categorical
emotion one.

Each item decodes at most ``max_seconds`` of one file, converts it to a fixed
-width log-mel tensor, and reports how many frames are real, so batches are
uniform without a custom collate and pooling never averages over padding.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.common.paths import DATASETS_DIR
from src.data.features.logmel import LogMelConfig, LogMelExtractor
from src.data.loaders.audio import AudioLoader
from src.data.modality_dataset import ModalityManifestDataset


class EmotionAudioDataset(ModalityManifestDataset):
    """File-backed audio only; a missing modality is never filled in."""

    MODALITY = "audio"
    LABEL_COLUMN = "canonical_emotion_id"

    #: Smallest column set the dataset can operate on.
    MINIMAL_COLUMNS = (
        "sample_id", "dataset", "canonical_emotion_id", "has_audio", "audio_source", "audio_path",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        max_seconds: float = 4.0,
        feature_config: LogMelConfig | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
        datasets_root: Path | None = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            max_samples=max_samples,
            seed=seed,
            columns=columns,
        )
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.max_seconds = float(max_seconds)
        self.feature_config = feature_config or LogMelConfig()
        self.extractor = LogMelExtractor(self.feature_config)
        self.max_frames = self.feature_config.frames_for(self.max_seconds)
        self.loader = AudioLoader(Path(datasets_root) if datasets_root else DATASETS_DIR)

    @property
    def feature_dim(self) -> int:
        return self.feature_config.n_mels

    def __getitem__(self, index: int) -> dict:
        row = self.row(index)
        waveform = self.loader.load_waveform(
            row["audio_path"],
            target_rate=self.feature_config.sample_rate,
            max_seconds=self.max_seconds,
        )
        features, valid = self.extractor.fixed_length(waveform, self.max_frames)
        return {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "audio": features,
            "audio_lengths": torch.tensor(valid, dtype=torch.long),
            "label": torch.tensor(int(row[self.LABEL_COLUMN]), dtype=torch.long),
        }
