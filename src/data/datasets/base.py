from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from src.data.resolver import SampleResolver


class BaseManifestDataset(Dataset, ABC):
    """
    Base PyTorch Dataset for task-specific manifests.

    Design:
        Manifest
            ↓
        BaseManifestDataset
            ↓
        SampleResolver
            ↓
        Modality loaders

    The dataset is lazy:
    metadata is loaded once, while actual modality data is
    resolved only when __getitem__() is called.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolver: SampleResolver | None = None,
    ):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found:\n{self.manifest_path}"
            )

        if self.manifest_path.suffix != ".parquet":
            raise ValueError(
                "Only Parquet manifests are currently supported."
            )

        print(
            f"Loading manifest: {self.manifest_path}"
        )

        self.manifest = pd.read_parquet(
            self.manifest_path
        )

        if "sample_id" not in self.manifest.columns:
            raise ValueError(
                "Manifest must contain 'sample_id'."
            )

        if self.manifest["sample_id"].duplicated().any():
            raise ValueError(
                "Manifest contains duplicate sample IDs."
            )

        self.resolver = (
            resolver
            if resolver is not None
            else SampleResolver()
        )

        self._validate_manifest()

        print(
            f"Loaded {len(self.manifest):,} samples."
        )

    def _validate_manifest(self) -> None:
        """
        Validate the minimum schema required by the
        PyTorch dataset layer.
        """

        required_columns = {
            "sample_id",
            "dataset",
        }

        missing = (
            required_columns
            - set(self.manifest.columns)
        )

        if missing:
            raise ValueError(
                "Manifest missing required columns: "
                + ", ".join(sorted(missing))
            )

    def __len__(self) -> int:
        return len(self.manifest)

    def get_record(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Return the metadata row corresponding to an index.
        """

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index out of range: {index}"
            )

        row = self.manifest.iloc[index]

        return row.to_dict()

    def resolve_sample(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Resolve the actual sample using SampleResolver.
        """

        record = self.get_record(index)

        sample_id = record["sample_id"]

        resolved = self.resolver.resolve(
            sample_id
        )

        if resolved is None:
            raise RuntimeError(
                f"Resolver returned None for "
                f"sample: {sample_id}"
            )

        return resolved

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Resolve one sample and attach task-specific target.
        """

        record = self.get_record(index)

        resolved = self.resolve_sample(index)

        sample = {
            "sample_id": record["sample_id"],
            "dataset": record["dataset"],
            "split": record.get("training_split"),
            "evaluation_group": record.get(
                "evaluation_group"
            ),
            "resolved": resolved,
        }

        sample["target"] = self.get_target(
            record
        )

        return sample

    @abstractmethod
    def get_target(
        self,
        record: dict[str, Any],
    ) -> Any:
        """
        Extract the task-specific target from a manifest row.
        """
        raise NotImplementedError