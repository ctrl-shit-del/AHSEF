from pathlib import Path
from typing import Any

import pickle

import numpy as np
import torch


class MOSEIFeatureStore:
    """
    Access CMU-MOSEI aligned_50.pkl features.

    The PKL contains:

        train
        valid
        test

    Each split contains:

        raw_text
        audio
        vision
        text
        text_bert
        id
        annotations
        classification_labels
        regression_labels

    Feature dimensions:

        audio  -> (N, 50, 74)
        vision -> (N, 50, 35)
        text   -> (N, 50, 768)
    """

    def __init__(self, feature_path: str | Path):

        self.feature_path = Path(feature_path)

        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"CMU-MOSEI feature file not found:\n"
                f"{self.feature_path}"
            )

        self._data: dict[str, Any] | None = None

        self._indices: dict[str, dict[str, int]] = {}

    def load(self) -> None:

        if self._data is not None:
            return

        print(
            f"Loading CMU-MOSEI features:\n"
            f"  {self.feature_path}"
        )

        with open(
            self.feature_path,
            "rb",
        ) as f:

            self._data = pickle.load(f)

        print("CMU-MOSEI features loaded.")

        self._build_indices()

    def _build_indices(self) -> None:

        if self._data is None:
            raise RuntimeError(
                "Feature data has not been loaded."
            )

        self._indices.clear()

        for split, split_data in self._data.items():

            ids = split_data["id"]

            index = {}

            for i, feature_id in enumerate(ids):

                index[str(feature_id)] = i

            self._indices[split] = index

            print(
                f"  {split:5} : "
                f"{len(index):,} feature IDs"
            )

    def _resolve(
        self,
        split: str,
        feature_id: str,
    ) -> tuple[dict[str, Any], int]:

        self.load()

        if self._data is None:
            raise RuntimeError(
                "Feature data unavailable."
            )

        if split not in self._data:
            raise KeyError(
                f"Unknown CMU-MOSEI split: {split}"
            )

        if split not in self._indices:
            raise KeyError(
                f"No index available for split: {split}"
            )

        index = self._indices[split].get(
            str(feature_id)
        )

        if index is None:
            raise KeyError(
                f"Feature ID not found: "
                f"{feature_id} "
                f"in split {split}"
            )

        return self._data[split], index

    def get(
        self,
        split: str,
        feature_id: str,
    ) -> dict[str, Any]:

        split_data, index = self._resolve(
            split,
            feature_id,
        )

        audio = split_data["audio"][index]
        vision = split_data["vision"][index]
        text = split_data["text"][index]

        regression = (
            split_data["regression_labels"][index]
        )

        classification = (
            split_data["classification_labels"][index]
        )

        return {
            "audio": torch.as_tensor(
                np.asarray(audio),
                dtype=torch.float32,
            ),

            "vision": torch.as_tensor(
                np.asarray(vision),
                dtype=torch.float32,
            ),

            "text": torch.as_tensor(
                np.asarray(text),
                dtype=torch.float32,
            ),

            "regression": torch.as_tensor(
                np.asarray(regression),
                dtype=torch.float32,
            ),

            "classification": torch.as_tensor(
                np.asarray(classification),
            ),
        }

    def __len__(self) -> int:

        self.load()

        return sum(
            len(index)
            for index in self._indices.values()
        )

    def close(self) -> None:

        self._data = None
        self._indices.clear()

    def __del__(self):

        self.close()