"""Dataset over pre-extracted, cached encoder features.

The expensive half of a frozen-encoder experiment is extraction; the cheap half
is training the head.  Separating them means the head can be retrained,
re-seeded and re-tuned in seconds without touching an audio file, and it means
the *representation* is fixed across every head experiment -- so a difference
between two heads is a difference between heads.

Two properties are enforced rather than assumed:

**A missing feature is an error, never a substitute.**  If the cache does not
cover a manifest row, the dataset says so and names the ids.  Filling the gap
with zeros would silently train the head on a fabricated recording.

**Order comes from the manifest, not the cache.**  Rows are gathered by
``sample_id``, so shard layout, extraction order and resume boundaries cannot
change what a batch contains.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.features.wav2vec2_features import FeatureStore


class CachedFeatureDataset(Dataset):
    """Manifest rows paired with their cached ``[num_layers, feature_dim]`` features."""

    LABEL_COLUMN = "canonical_emotion_id"
    MINIMAL_COLUMNS = ("sample_id", "dataset", "canonical_emotion_id")

    def __init__(
        self,
        manifest_path: str | Path,
        store: FeatureStore,
        max_samples: int | None = None,
        seed: int = 42,
        label_column: str | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.store = store
        self.label_column = label_column or self.LABEL_COLUMN

        frame = pd.read_parquet(self.manifest_path)
        frame["sample_id"] = frame["sample_id"].astype(str)
        if max_samples is not None and max_samples < len(frame):
            # Seeded, so a bounded debug run is reproducible rather than
            # whatever the first N rows happen to be.
            frame = frame.sample(n=max_samples, random_state=seed)
        self.manifest = frame.sort_values("sample_id", kind="mergesort").reset_index(
            drop=True
        )

        missing = self.store.missing(self.manifest["sample_id"])
        if missing:
            raise KeyError(
                f"{len(missing)} of {len(self.manifest)} samples in "
                f"{self.manifest_path} have no cached features (e.g. {missing[:5]}). "
                f"Run the extractor for this split; a missing feature is never "
                f"replaced with zeros."
            )
        # One gather up front: the store reads whole shards, so per-item access
        # would re-open and decompress the same shard thousands of times.
        self.features = torch.from_numpy(
            self.store.load(self.manifest["sample_id"].tolist())
        ).to(torch.float32)
        self.labels = torch.tensor(
            self.manifest[self.label_column].astype(int).to_numpy(), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.manifest)

    @property
    def num_layers(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

    def labels_tensor(self) -> torch.Tensor:
        return self.labels

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        return {
            "sample_id": str(row["sample_id"]),
            "dataset": str(row.get("dataset", "")),
            "features": self.features[index],
            "label": self.labels[index],
        }


def create_cached_feature_dataloader(
    manifest_path: str | Path,
    store: FeatureStore,
    batch_size: int = 64,
    shuffle: bool = False,
    max_samples: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """A dataloader over cached features, seeded so shuffling is reproducible."""
    dataset = CachedFeatureDataset(
        manifest_path, store, max_samples=max_samples, seed=seed
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        drop_last=False,
    )
