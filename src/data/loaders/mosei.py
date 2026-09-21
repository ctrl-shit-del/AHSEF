import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np


class MOSEILoader:
    """
    Loader for CMU-MOSEI aligned_50.pkl.

    The PKL contains:

        train
        valid
        test

    Each split contains:

        raw_text
        audio
        vision
        id
        text
        text_bert
        annotations
        classification_labels
        regression_labels

    A feature ID has the form:

        video_id$_$clip_id
    """

    def __init__(
        self,
        datasets_root: Path,
        feature_file: str,
    ):
        self.datasets_root = datasets_root

        self.feature_path = (
            datasets_root / feature_file
        )

        self._data: Optional[dict] = None
        self._indexes: dict[str, dict[str, int]] = {}

    def _load(self):

        if self._data is not None:
            return

        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"CMU-MOSEI feature file not found: "
                f"{self.feature_path}"
            )

        print(
            "Loading CMU-MOSEI feature container..."
        )

        with open(
            self.feature_path,
            "rb",
        ) as f:

            self._data = pickle.load(f)

        print(
            "CMU-MOSEI feature container loaded."
        )

        self._build_indexes()

    def _build_indexes(self):

        if self._data is None:
            raise RuntimeError(
                "Feature container is not loaded."
            )

        for split, split_data in self._data.items():

            ids = split_data["id"]

            self._indexes[split] = {
                str(feature_id): index
                for index, feature_id
                in enumerate(ids)
            }

    @staticmethod
    def normalize_split(
        split: str,
    ) -> str:

        if split == "validation":
            return "valid"

        return split

    def get(
        self,
        feature_split: str,
        feature_id: str,
    ) -> dict[str, Any]:

        self._load()

        split = self.normalize_split(
            feature_split
        )

        if split not in self._data:
            raise KeyError(
                f"Unknown CMU-MOSEI split: {split}"
            )

        index = self._indexes[split].get(
            str(feature_id)
        )

        if index is None:
            raise KeyError(
                f"CMU-MOSEI feature ID not found: "
                f"{feature_id}"
            )

        data = self._data[split]

        return {
            "audio": data["audio"][index],
            "video": data["vision"][index],
            "text_features": data["text"][index],
            "text_bert": data["text_bert"][index],
            "raw_text": data["raw_text"][index],
            "annotation": data["annotations"][index],
            "classification_label": data[
                "classification_labels"
            ][index],
            "regression_label": data[
                "regression_labels"
            ][index],
        }

    def verify(
        self,
        feature_split: str,
        feature_id: str,
    ) -> bool:

        self._load()

        split = self.normalize_split(
            feature_split
        )

        return (
            split in self._indexes
            and feature_id in self._indexes[split]
        )