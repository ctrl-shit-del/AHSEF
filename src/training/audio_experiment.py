"""Audio-only 7-class categorical emotion baseline over a sampled subset.

Contributing datasets are RAVDESS, CREMA-D, IEMOCAP, and MSP-Podcast: every
corpus whose standardized metadata declares ``has_audio`` with
``audio_source == 'file'``, a real ``audio_path``, and a valid canonical
emotion.  CMU-MOSEI is excluded even though it declares audio -- its audio is
feature-container-backed (``aligned_50.pkl``) and its target is sentiment, not
categorical emotion.

The front end is a bounded log-mel spectrogram computed with core PyTorch (see
``src.data.features.logmel``); the model pools it over time and classifies the
pooled statistics.  Nothing is downloaded and no pretrained weights are used.

Canonical, immutable class order (see ``src.common.labels``)::

    0 neutral, 1 happy, 2 sad, 3 angry, 4 fear, 5 disgust, 6 surprise

The module never runs itself; see ``src.training.run_audio_experiment``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from src.data.audio_dataset import EmotionAudioDataset
from src.data.dataloader import create_emotion_audio_dataloader
from src.data.features.logmel import LogMelConfig
from src.models.baselines import AudioEmotionBaseline
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    BaseExperimentConfig,
    BaseExperimentRunner,
)


DESCRIPTION = "Audio-only modality-filtered 7-class categorical emotion baseline."


@dataclass
class AudioExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one audio training iteration."""

    experiment_name: str = "audio_25pct"
    description: str = DESCRIPTION
    modality: str = "audio"
    batch_size: int = 32

    # front end
    max_seconds: float = 4.0
    sample_rate: int = 16_000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64

    # model
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.n_mels < 1:
            raise ValueError("n_mels must be at least 1")

    @property
    def feature_config(self) -> LogMelConfig:
        return LogMelConfig(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )


def debug_config(config: AudioExperimentConfig) -> AudioExperimentConfig:
    """Return a tiny, CPU-fast variant that exercises the whole pipeline."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


class AudioExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the audio baseline."""

    INPUT_KEY = "audio"
    LENGTHS_KEY = "audio_lengths"
    LABEL_COLUMN = "canonical_emotion_id"

    config: AudioExperimentConfig

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        return create_emotion_audio_dataloader(
            self.manifest_path(split),
            batch_size=self.config.batch_size,
            max_seconds=self.config.max_seconds,
            feature_config=self.config.feature_config,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            shuffle=shuffle,
            columns=list(EmotionAudioDataset.MINIMAL_COLUMNS),
            num_workers=self.config.num_workers,
        )

    def build_model(self) -> nn.Module:
        return AudioEmotionBaseline(
            n_mels=self.config.n_mels,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        )

    def model_record(self, model: nn.Module) -> dict:
        return {
            "n_mels": self.config.n_mels,
            "hidden_dim": self.config.hidden_dim,
            "pooling": "masked mean and standard deviation over time",
            "front_end": self.config.feature_config.to_dict(),
            "max_seconds": self.config.max_seconds,
            "max_frames": self.config.feature_config.frames_for(self.config.max_seconds),
        }

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record.update({
            "max_seconds": self.config.max_seconds,
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "n_mels": self.config.n_mels,
            "hidden_dim": self.config.hidden_dim,
        })
        return record


def run_iteration(config: AudioExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return AudioExperimentRunner(config).run()
