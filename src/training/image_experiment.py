"""Image-only 7-class categorical emotion baseline over a sampled subset.

This is an explicitly modality-filtered experiment.  Only records whose
standardized metadata declares ``has_image`` with ``image_source == 'file'``
and a real ``image_path`` take part; the contributing datasets are AffectNet+,
FERPlus, and RAF-DB.  MELD is excluded even though it carries other
modalities -- no tensor is ever fabricated for an absent modality.

Canonical, immutable class order (see ``src.common.labels``)::

    0 neutral, 1 happy, 2 sad, 3 angry, 4 fear, 5 disgust, 6 surprise

The experiment protocol -- train-only class weights, validation-driven
checkpoint selection, a test partition opened exactly once after training, and
fully isolated iterations -- lives in
:class:`~src.training.base_experiment.BaseExperimentRunner` and is shared with
the audio, text, video, and physiology baselines.  Only the dataloader and the
architecture are image-specific.

The module never runs itself; see ``src.training.run_image_experiment``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from src.common.labels import EMOTION_7CLASS
from src.data.dataloader import create_emotion_image_dataloader
from src.data.image_dataset import EmotionImageDataset
from src.models.multimodal import ImageEmotionBaseline
from src.preprocessing.standardization.targets import CANONICAL_EMOTIONS
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    MONITOR,
    BaseExperimentConfig,
    BaseExperimentRunner,
    resolve_device,
    save_json,
    set_seeds,
)


CLASS_NAMES = EMOTION_7CLASS.classes
DESCRIPTION = "Image-only modality-filtered 7-class categorical emotion baseline."

__all__ = [
    "CANONICAL_EMOTIONS",
    "CLASS_NAMES",
    "DEBUG_OVERRIDES",
    "DESCRIPTION",
    "MONITOR",
    "ImageExperimentConfig",
    "ImageExperimentRunner",
    "debug_config",
    "resolve_device",
    "run_iteration",
    "save_json",
    "set_seeds",
]


# ============================================================
# Configuration
# ============================================================

@dataclass
class ImageExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one image training iteration."""

    experiment_name: str = "image_25pct"
    description: str = DESCRIPTION
    modality: str = "image"

    # model geometry
    image_size: int = 48
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_size < 1:
            raise ValueError("image_size must be at least 1")


def debug_config(config: ImageExperimentConfig) -> ImageExperimentConfig:
    """Return a tiny, CPU-fast variant that exercises the whole pipeline."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


# ============================================================
# Runner
# ============================================================

class ImageExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the image baseline."""

    INPUT_KEY = "image"
    LABEL_COLUMN = "canonical_emotion_id"

    config: ImageExperimentConfig

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        return create_emotion_image_dataloader(
            self.manifest_path(split),
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            shuffle=shuffle,
            columns=list(EmotionImageDataset.MINIMAL_COLUMNS),
            num_workers=self.config.num_workers,
        )

    def build_model(self) -> nn.Module:
        return ImageEmotionBaseline(
            image_size=self.config.image_size,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        )

    def model_record(self, model: nn.Module) -> dict:
        return {
            "image_size": self.config.image_size,
            "hidden_dim": self.config.hidden_dim,
        }

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record["image_size"] = self.config.image_size
        record["hidden_dim"] = self.config.hidden_dim
        return record


def run_iteration(config: ImageExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return ImageExperimentRunner(config).run()
