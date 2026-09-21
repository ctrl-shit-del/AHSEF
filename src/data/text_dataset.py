"""Lazy, modality-filtered text dataset for categorical emotion experiments.

Text arrives two different ways in this project and neither substitutes for the
other: MELD transcripts live in the metadata ``text`` column, MSP-Podcast
transcripts live in files addressed by ``text_path``.  The dataset honours the
``text_source`` recorded for each record rather than guessing.

CMU-MOSEI text is present in the metadata but carries a *sentiment* target
(``target_type == 'sentiment'``, ``canonical_emotion_valid == False``), so no
record of it survives the 7-class target filter.  Converting its
positive/negative annotation into happy/sad would fabricate emotion labels;
see ``src.preprocessing.standardization.targets``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.common.paths import DATASETS_DIR
from src.data.features.text_tokenizer import HashingTokenizer, TokenizerConfig
from src.data.modality_dataset import ModalityManifestDataset


class EmotionTextDataset(ModalityManifestDataset):
    """Metadata- and file-backed transcripts, encoded with a hashing tokenizer."""

    MODALITY = "text"
    LABEL_COLUMN = "canonical_emotion_id"

    #: Smallest column set the dataset can operate on.
    MINIMAL_COLUMNS = (
        "sample_id", "dataset", "canonical_emotion_id",
        "has_text", "text_source", "text", "text_path",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer_config: TokenizerConfig | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
        datasets_root: Path | None = None,
        cache_text: bool = True,
    ):
        super().__init__(
            manifest_path=manifest_path,
            max_samples=max_samples,
            seed=seed,
            columns=columns,
        )
        self.tokenizer_config = tokenizer_config or TokenizerConfig()
        self.tokenizer = HashingTokenizer(self.tokenizer_config)
        self.datasets_root = Path(datasets_root) if datasets_root else DATASETS_DIR
        # Transcripts are a few dozen bytes each; re-reading a quarter of a
        # million tiny files once per epoch costs far more than holding them.
        self.cache_text = bool(cache_text)
        self._cache: dict[str, str] = {}

    @property
    def vocab_size(self) -> int:
        return self.tokenizer_config.vocab_size

    @property
    def max_tokens(self) -> int:
        return self.tokenizer_config.max_tokens

    def read_text(self, row) -> str:
        source = row["text_source"]
        if source == "file":
            relative = row["text_path"]
            if not relative:
                raise ValueError(
                    f"Record {row['sample_id']} declares file-backed text without a text_path"
                )
            key = str(relative)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            path = self.datasets_root / key
            if not path.exists():
                raise FileNotFoundError(f"Transcript not found: {path}")
            value = path.read_text(encoding="utf-8", errors="replace").strip()
            if self.cache_text:
                self._cache[key] = value
            return value
        value = row["text"]
        return "" if value is None else str(value)

    def __getitem__(self, index: int) -> dict:
        row = self.row(index)
        tokens, valid = self.tokenizer.encode(self.read_text(row))
        return {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "text": tokens,
            "text_lengths": torch.tensor(valid, dtype=torch.long),
            "label": torch.tensor(int(row[self.LABEL_COLUMN]), dtype=torch.long),
        }
