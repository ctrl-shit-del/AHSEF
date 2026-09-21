"""Lazy, modality-filtered image dataset for categorical emotion experiments."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.paths import DATASETS_DIR
from src.data.loaders.image import ImageLoader


class EmotionImageDataset(Dataset):
    """Use only explicit file-backed images; missing modalities are never filled."""

    #: Smallest column set the dataset can operate on; passing it keeps large
    #: metadata columns such as ``extras`` out of memory for big manifests.
    MINIMAL_COLUMNS = (
        "sample_id", "dataset", "canonical_emotion_id", "has_image", "image_source", "image_path",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 32,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        manifest = pd.read_parquet(self.manifest_path, columns=list(columns) if columns else None)
        required = {"sample_id", "dataset", "canonical_emotion_id", "has_image", "image_source", "image_path"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Image manifest missing columns: {sorted(missing)}")
        self.manifest = manifest[
            manifest["has_image"].fillna(False)
            & manifest["image_source"].eq("file")
            & manifest["image_path"].notna()
            & manifest["canonical_emotion_id"].isin(range(7))
        ].copy()
        if self.manifest.empty:
            raise ValueError("No valid file-backed image records in manifest")
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            self.manifest = self.manifest.sample(n=min(max_samples, len(self.manifest)), random_state=seed).sort_index()
        self.manifest.reset_index(drop=True, inplace=True)
        self.loader = ImageLoader(DATASETS_DIR)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        return {
            "sample_id": row["sample_id"], "dataset": row["dataset"],
            "image": self.loader.load_tensor(row["image_path"], self.image_size),
            "label": torch.tensor(int(row["canonical_emotion_id"]), dtype=torch.long),
        }
