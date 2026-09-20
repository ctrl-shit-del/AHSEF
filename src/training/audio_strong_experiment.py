"""``audio_strong`` -- a frozen-wav2vec2 audio expert for the 7-class task.

Same protocol as every other experiment in this project: class weights from the
training partition only, checkpoint selected on validation macro-F1, test opened
once and only after training.  What differs is the front end.  The Layer-3 audio
baseline learns from log-mel spectrograms with no pretrained weights; this
experiment reads frozen wav2vec2 representations and trains only a small
layer-weighted probe on top.

The comparison is deliberately controlled:

* **Evaluation samples are identical.**  ``audio_strong``'s validation and test
  manifests are byte-for-byte the baseline's, checked at manifest-build time.
  Every reported difference is therefore a difference between experts, not
  between evaluation sets.
* **Training samples are not.**  The training partition is class-capped at 4,000
  to fit a CPU-only feature-extraction budget, so class weights are recomputed
  from the capped subset. That difference is real and is reported rather than
  absorbed.

Nothing here writes into ``experiments/audio/``; the artefacts live under
``experiments/audio_strong/`` and the baseline stays exactly as it was.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch.nn as nn

from src.data.cached_feature_dataset import (
    CachedFeatureDataset,
    create_cached_feature_dataloader,
)
from src.data.features.wav2vec2_features import (
    DEFAULT_BUNDLE,
    FeatureStore,
    Wav2Vec2Config,
)
from src.models.strong_experts import LayerWeightedProbe
from src.training.audio_strong_manifest import STRONG_EXPERIMENT
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    BaseExperimentConfig,
    BaseExperimentRunner,
)

DESCRIPTION = (
    "Audio 7-class emotion expert over frozen wav2vec2 representations "
    "(layer-weighted probe head)."
)

DEFAULT_CACHE_ROOT = Path("experiments") / "audio_strong" / "features"


@dataclass
class AudioStrongExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one ``audio_strong`` iteration."""

    experiment_name: str = STRONG_EXPERIMENT
    description: str = DESCRIPTION
    modality: str = "audio_strong"
    task: str = "emotion_7class"
    num_classes: int = 7

    # The head is tiny and the features are cached, so a larger batch and a
    # longer schedule cost seconds rather than hours.
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 6
    min_epochs: int = 5

    #: This experiment's test split stays shut until the downstream routing
    #: configuration is frozen, so it defaults to deferred rather than relying on
    #: whoever runs it remembering a flag. Phase F opens it by constructing this
    #: config with ``defer_test=False`` explicitly -- a deliberate code path,
    #: not something that can happen out of habit.
    defer_test: bool = True

    # frozen front end
    bundle: str = DEFAULT_BUNDLE
    max_seconds: float = 4.0
    cache_root: str = str(DEFAULT_CACHE_ROOT)

    # head
    hidden_dim: int = 256
    dropout: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")

    @property
    def extractor_config(self) -> Wav2Vec2Config:
        return Wav2Vec2Config(bundle=self.bundle, max_seconds=self.max_seconds)

    def cache_dir(self, split: str) -> Path:
        return Path(self.cache_root) / split


def debug_config(config: AudioStrongExperimentConfig) -> AudioStrongExperimentConfig:
    """A tiny variant that exercises the whole pipeline in seconds."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


class AudioStrongExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the strong audio expert."""

    INPUT_KEY = "features"
    LENGTHS_KEY = None
    LABEL_COLUMN = "canonical_emotion_id"

    config: AudioStrongExperimentConfig

    def _store(self, split: str) -> FeatureStore:
        directory = self.config.cache_dir(split)
        if not (directory / "index.json").exists():
            raise FileNotFoundError(
                f"No feature cache at {directory}. Extract it first:\n"
                f"  python -m src.training.extract_audio_features --split {split}\n"
                f"The encoder is frozen, so this only has to happen once."
            )
        return FeatureStore.open(directory, self.config.extractor_config.fingerprint)

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        return create_cached_feature_dataloader(
            self.manifest_path(split),
            store=self._store(split),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            num_workers=self.config.num_workers,
        )

    def build_model(self) -> nn.Module:
        store = self._store("train")
        return LayerWeightedProbe(
            num_layers=int(store.num_layers),
            feature_dim=int(store.feature_dim),
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

    def model_record(self, model: nn.Module) -> dict:
        store = self._store("train")
        record = model.describe() if hasattr(model, "describe") else {}
        record.update({
            "front_end": {
                **self.config.extractor_config.to_dict(),
                "feature_cache": str(self.config.cache_root),
                "cache_fingerprint": self.config.extractor_config.fingerprint,
                "cached_train_samples": len(store),
            },
            "num_layers": int(store.num_layers),
            "feature_dim": int(store.feature_dim),
            "hidden_dim": self.config.hidden_dim,
            "num_classes": self.config.num_classes,
            "dropout": self.config.dropout,
        })
        return record

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record.update({
            "bundle": self.config.bundle,
            "max_seconds": self.config.max_seconds,
            "hidden_dim": self.config.hidden_dim,
            "dropout": self.config.dropout,
            "encoder_frozen": True,
            "encoder_trained_by_this_project": False,
            "cache_root": self.config.cache_root,
        })
        return record


def run_iteration(config: AudioStrongExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return AudioStrongExperimentRunner(config).run()
