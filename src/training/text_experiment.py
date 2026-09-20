"""Text-only 7-class categorical emotion baseline over the full eligible pool.

Contributing datasets, as declared by the modality rule, are MELD, CMU-MOSEI,
and MSP-Podcast.  Which of them actually contribute records depends on the
standardized metadata and is reported in ``sampling_summary.json``:

* MELD transcripts are metadata-backed (``text_source == 'metadata'``) and
  carry canonical categorical emotion, so they contribute.
* MSP-Podcast transcripts are file-backed (``text_source == 'file'``,
  ``text_path``) and carry consensus categorical emotion, so they contribute
  once the standardized metadata has been regenerated with the MSP-Podcast
  emotion mapping and the ``text_path`` column.
* CMU-MOSEI text carries a *sentiment* target (``target_type == 'sentiment'``,
  ``canonical_emotion_valid == False``).  It is filtered out rather than having
  its positive/negative annotation reinterpreted as happy/sad -- that would
  fabricate emotion labels.

The representation is a hashing tokenizer plus a learned embedding table: no
model download, no internet access, and no fitted vocabulary that could carry
information from validation or test into training.

The module never runs itself; see ``src.training.run_text_experiment``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.nn as nn

from src.data.dataloader import create_emotion_text_dataloader
from src.data.features.text_tokenizer import TokenizerConfig
from src.data.text_dataset import EmotionTextDataset
from src.models.baselines import TextEmotionBaseline
from src.training.base_experiment import (
    DEBUG_OVERRIDES,
    BaseExperimentConfig,
    BaseExperimentRunner,
)


DESCRIPTION = "Text-only modality-filtered 7-class categorical emotion baseline."


@dataclass
class TextExperimentConfig(BaseExperimentConfig):
    """Everything needed to reproduce one text training iteration."""

    experiment_name: str = "text_full"
    description: str = DESCRIPTION
    modality: str = "text"
    batch_size: int = 64

    # front end
    vocab_size: int = 32_768
    max_tokens: int = 64
    cache_text: bool = True

    # model
    embedding_dim: int = 128
    hidden_dim: int = 256

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1")

    @property
    def tokenizer_config(self) -> TokenizerConfig:
        return TokenizerConfig(vocab_size=self.vocab_size, max_tokens=self.max_tokens)


def debug_config(config: TextExperimentConfig) -> TextExperimentConfig:
    """Return a tiny, CPU-fast variant that exercises the whole pipeline."""
    return replace(config, debug=True, **DEBUG_OVERRIDES)


class TextExperimentRunner(BaseExperimentRunner):
    """Execute one reproducible iteration of the text baseline."""

    INPUT_KEY = "text"
    LENGTHS_KEY = "text_lengths"
    LABEL_COLUMN = "canonical_emotion_id"

    config: TextExperimentConfig

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        return create_emotion_text_dataloader(
            self.manifest_path(split),
            batch_size=self.config.batch_size,
            tokenizer_config=self.config.tokenizer_config,
            max_samples=max_samples,
            seed=self.config.effective_run_seed,
            shuffle=shuffle,
            columns=list(EmotionTextDataset.MINIMAL_COLUMNS),
            num_workers=self.config.num_workers,
            cache_text=self.config.cache_text,
        )

    def build_model(self) -> nn.Module:
        return TextEmotionBaseline(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_classes=self.config.num_classes,
        )

    def model_record(self, model: nn.Module) -> dict:
        return {
            "vocab_size": self.config.vocab_size,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "pooling": "masked mean over tokens",
            "tokenizer": self.config.tokenizer_config.to_dict(),
            "pretrained_weights": None,
        }

    def _hyperparameter_record(self) -> dict:
        record = super()._hyperparameter_record()
        record.update({
            "vocab_size": self.config.vocab_size,
            "max_tokens": self.config.max_tokens,
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
        })
        return record


def run_iteration(config: TextExperimentConfig) -> dict:
    """Execute a single iteration and return its run summary."""
    return TextExperimentRunner(config).run()
