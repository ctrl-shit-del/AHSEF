"""Video-only 7-class categorical emotion baseline over a sampled subset.

The modality rule admits MELD and CMU-MOSEI.  In practice only MELD
contributes: CMU-MOSEI's video is ``feature_container`` provenance behind
``aligned_50.pkl`` rather than a file-backed ``video_path``, and its target is
sentiment rather than categorical emotion, so it satisfies neither the modality
rule nor the task rule.  ``sampling_summary.json`` records which datasets
actually contributed.

Each clip contributes a handful of uniformly spaced frames rather than its full
frame sequence, so the baseline stays CPU-feasible and memory stays
proportional to ``num_frames`` rather than to clip duration.

The module never runs itself; see ``src.training.run_video_experiment``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from src.data.dataloader import create_emotion_video_dataloader
from src.data.video_dataset import EmotionVideoDataset
from src.models.baselines import TEMPORAL_MODES, VideoEmotionBaseline
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    BaseExperimentConfig,
    BaseExperimentRunner,
)


DESCRIPTION = "Video-only modality-filtered 7-class categorical emotion baseline."


@dataclass
class VideoExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one video training iteration."""

    experiment_name: str = "video_25pct"
    description: str = DESCRIPTION
    modality: str = "video"
    batch_size: int = 8

    # front end
    num_frames: int = 8
    frame_size: int = 48

    # model
    hidden_dim: int = 256
    temporal: str = "mean"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if self.frame_size < 1:
            raise ValueError("frame_size must be at least 1")
        if self.temporal not in TEMPORAL_MODES:
            raise ValueError(f"temporal must be one of {TEMPORAL_MODES}, got {self.temporal!r}")


def debug_config(config: VideoExperimentConfig) -> VideoExperimentConfig:
    """Return a tiny, CPU-fast variant that exercises the whole pipeline."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


class VideoExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the video baseline."""

    INPUT_KEY = "video"
    LENGTHS_KEY = "video_lengths"
    LABEL_COLUMN = "canonical_emotion_id"

    config: VideoExperimentConfig

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        return create_emotion_video_dataloader(
            self.manifest_path(split),
            batch_size=self.config.batch_size,
            num_frames=self.config.num_frames,
            frame_size=self.config.frame_size,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            shuffle=shuffle,
            columns=list(EmotionVideoDataset.MINIMAL_COLUMNS),
            num_workers=self.config.num_workers,
        )

    def build_model(self) -> nn.Module:
        return VideoEmotionBaseline(
            frame_size=self.config.frame_size,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            temporal=self.config.temporal,
        )

    def model_record(self, model: nn.Module) -> dict:
        return {
            "num_frames": self.config.num_frames,
            "frame_size": self.config.frame_size,
            "hidden_dim": self.config.hidden_dim,
            "temporal": self.config.temporal,
            "pooling": (
                "masked mean over frames"
                if self.config.temporal == "mean"
                else "single-layer GRU followed by masked mean over frames"
            ),
        }

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record.update({
            "num_frames": self.config.num_frames,
            "frame_size": self.config.frame_size,
            "hidden_dim": self.config.hidden_dim,
            "temporal": self.config.temporal,
        })
        return record


def run_iteration(config: VideoExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return VideoExperimentRunner(config).run()
