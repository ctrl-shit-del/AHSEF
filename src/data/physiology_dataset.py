"""Window-level physiological dataset for the WESAD baseline.

WESAD's standardized metadata holds one record per subject -- fifteen rows,
each pointing at a multi-hour recording -- which is a pointer index, not a
training set.  The window builder
(``src.preprocessing.sampling.physiology_windows``) expands those pointers once
into fixed-length, label-homogeneous windows with compact per-channel
statistics, and writes them as experiment split Parquet.

That means training reads only small numeric vectors: no ``.pkl`` is reopened
during training, and the whole experiment fits comfortably in RAM.  Feature
standardisation is supplied by the runner and is fitted on the training
partition alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.common.labels import WESAD_STATE_3CLASS, LabelSpace


FEATURE_COLUMN = "features"
LABEL_COLUMN = "state_id"


class PhysiologyWindowDataset(Dataset):
    """Precomputed physiological window features with train-fitted scaling."""

    LABEL_COLUMN = LABEL_COLUMN
    MINIMAL_COLUMNS = (
        "sample_id", "dataset", "subject", "state", LABEL_COLUMN, FEATURE_COLUMN,
    )

    def __init__(
        self,
        manifest_path: str | Path,
        label_space: LabelSpace = WESAD_STATE_3CLASS,
        mean: np.ndarray | list[float] | None = None,
        std: np.ndarray | list[float] | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.label_space = label_space
        wanted = list(columns) if columns else list(self.MINIMAL_COLUMNS)
        manifest = pd.read_parquet(self.manifest_path, columns=wanted)
        missing = set(self.MINIMAL_COLUMNS) - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Physiology window manifest {self.manifest_path} is missing columns: "
                f"{sorted(missing)}"
            )

        labels = pd.to_numeric(manifest[LABEL_COLUMN], errors="coerce")
        manifest = manifest[
            labels.isin(list(label_space.valid_ids)) & manifest[FEATURE_COLUMN].notna()
        ].copy()
        if manifest.empty:
            raise ValueError(
                f"No physiology windows with a valid {label_space.name} target in "
                f"{self.manifest_path}"
            )
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            manifest = manifest.sample(
                n=min(max_samples, len(manifest)), random_state=seed
            ).sort_index()
        manifest.reset_index(drop=True, inplace=True)
        self.manifest = manifest

        features = np.stack([np.asarray(value, dtype=np.float32)
                             for value in manifest[FEATURE_COLUMN]])
        widths = {row.shape[0] for row in features}
        if len(widths) != 1:
            raise ValueError(f"Physiology windows have inconsistent feature widths: {sorted(widths)}")
        self.features = torch.from_numpy(np.ascontiguousarray(features))
        self.labels = torch.tensor(manifest[LABEL_COLUMN].astype(int).tolist(), dtype=torch.long)

        self.mean, self.std = self._resolve_scaling(mean, std)
        # Standardise once: the whole split is a few thousand short vectors,
        # so doing it per item would only repeat identical arithmetic.
        self.features = (self.features - self.mean) / self.std

    # ------------------------------------------------------------- scaling

    def _resolve_scaling(self, mean, std) -> tuple[torch.Tensor, torch.Tensor]:
        width = self.features.shape[1]
        if mean is None or std is None:
            # Identity scaling; the runner supplies train-fitted statistics.
            return torch.zeros(width), torch.ones(width)
        mean_tensor = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        std_tensor = torch.as_tensor(np.asarray(std, dtype=np.float32))
        if mean_tensor.shape != (width,) or std_tensor.shape != (width,):
            raise ValueError(
                f"Scaling statistics must have shape ({width},); got "
                f"{tuple(mean_tensor.shape)} and {tuple(std_tensor.shape)}"
            )
        # A constant channel has zero variance; leave it untouched rather than
        # dividing by zero and producing NaNs the loss would silently absorb.
        std_tensor = torch.where(std_tensor > 1e-8, std_tensor, torch.ones_like(std_tensor))
        return mean_tensor, std_tensor

    @property
    def raw_features(self) -> torch.Tensor:
        """Windows with the current standardisation undone."""
        return self.features * self.std + self.mean

    def set_scaling(self, mean, std) -> None:
        """Replace the standardisation, re-deriving from the raw windows."""
        raw = self.raw_features
        self.mean, self.std = torch.zeros_like(self.mean), torch.ones_like(self.std)
        self.features = raw
        self.mean, self.std = self._resolve_scaling(mean, std)
        self.features = (raw - self.mean) / self.std

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def subjects(self) -> list[str]:
        return sorted({str(value) for value in self.manifest["subject"]})

    # ------------------------------------------------------------------ data

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        return {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "subject": row["subject"],
            "physiology": self.features[index],
            "label": self.labels[index],
        }


def fit_feature_scaling(dataset: PhysiologyWindowDataset) -> tuple[list[float], list[float]]:
    """Return per-feature mean and standard deviation of a dataset's windows.

    Call this on the *training* dataset only; fitting it on validation or test
    windows would leak their distribution into the model's inputs.
    """
    raw = dataset.raw_features
    return raw.mean(dim=0).tolist(), raw.std(dim=0, unbiased=False).tolist()
