from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.loaders.mosei_loader import (
    MOSEIFeatureStore,
)
from src.data.datasets.base import BaseManifestDataset
from src.data.resolver import SampleResolver


class ExperimentDataset(BaseManifestDataset):
    """Common lazy dataset for experiment-specific Parquet metadata.

    The experiment manifest, rather than a training script, defines both the
    target and the requested split.  Missing modalities are returned as
    explicit ``None`` values; this class never substitutes another modality.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        target_column: str,
        load_modalities: bool = True,
        resolver: SampleResolver | None = None,
    ):
        self.target_column = target_column
        self.load_modalities = load_modalities
        manifest_path = Path(manifest_path)
        # The manifest is a self-contained standardized-metadata subset, so it
        # is both the task index and a compact resolver index.
        resolver = resolver or SampleResolver(metadata_path=manifest_path)
        super().__init__(manifest_path=manifest_path, resolver=resolver)

    def _validate_manifest(self) -> None:
        super()._validate_manifest()
        if self.target_column not in self.manifest.columns:
            raise ValueError(f"Manifest missing target column: {self.target_column}")
        if self.manifest[self.target_column].isna().any():
            raise ValueError(f"Manifest has missing targets in: {self.target_column}")

    def get_target(self, record: dict):
        value = record[self.target_column]
        if self.target_column == "canonical_emotion_id":
            return torch.tensor(int(value), dtype=torch.long)
        return torch.tensor(float(value), dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        record = self.get_record(index)
        resolved = self.resolver.resolve(
            record["sample_id"], load_modalities=self.load_modalities
        )
        return {
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record.get("experiment_split", record.get("training_split")),
            "label": self.get_target(record),
            "audio": resolved.audio,
            "video": resolved.video,
            "image": resolved.image,
            "text": resolved.text,
            "physiology": resolved.physiology,
            "metadata": record,
        }


class MOSEIDataset(Dataset):
    """
    PyTorch Dataset for CMU-MOSEI.

    Reads records from a task manifest and retrieves
    corresponding features from aligned_50.pkl.

    Target contract:
        regression      -> float32
        classification  -> long
        sentiment_score -> float32
    """

    def __init__(
        self,
        manifest_path: str | Path,
        feature_path: str | Path,
    ):

        self.manifest_path = Path(
            manifest_path
        )

        self.feature_path = Path(
            feature_path
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found:\n"
                f"{self.manifest_path}"
            )

        self.metadata = pd.read_parquet(
            self.manifest_path
        )

        if len(self.metadata) == 0:
            raise ValueError(
                f"Manifest contains no records:\n"
                f"{self.manifest_path}"
            )

        self.feature_store = MOSEIFeatureStore(
            self.feature_path
        )

        print(
            f"MOSEI Dataset:"
            f"\n  records: {len(self.metadata):,}"
            f"\n  manifest: {self.manifest_path}"
        )

    def __len__(self) -> int:

        return len(self.metadata)

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        row = self.metadata.iloc[index]

        feature_split = str(
            row["feature_split"]
        )

        feature_id = str(
            row["feature_id"]
        )

        features = self.feature_store.get(
            split=feature_split,
            feature_id=feature_id,
        )

        # -------------------------------------------------
        # Features
        # -------------------------------------------------

        audio = torch.as_tensor(
            features["audio"],
            dtype=torch.float32,
        )

        vision = torch.as_tensor(
            features["vision"],
            dtype=torch.float32,
        )

        text = torch.as_tensor(
            features["text"],
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Targets
        # -------------------------------------------------

        regression = torch.as_tensor(
            features["regression"],
            dtype=torch.float32,
        )

        classification = torch.as_tensor(
            features["classification"],
            dtype=torch.long,
        )

        sentiment_score = torch.tensor(
            float(row["sentiment_score"]),
            dtype=torch.float32,
        )

        # -------------------------------------------------
        # Sample
        # -------------------------------------------------

        sample = {

            "sample_id": row["sample_id"],

            "dataset": row["dataset"],

            "split": row["training_split"],

            "audio": audio,

            "vision": vision,

            "text": text,

            "regression": regression,

            "classification": classification,

            "sentiment_score": sentiment_score,

        }

        return sample
