"""
RAF-DB Dataset Adapter
"""

from typing import List

import pandas as pd

from src.common.models import EmotionRecord

from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import RAFDB_EMOTIONS


class RAFDBAdapter(BaseAdapter):

    DATASET_NAME = "RAF-DB"

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        root = self.dataset_dir

        for split in ("train", "test"):

            csv_path = root / f"{split}_labels.csv"

            if not csv_path.exists():
                continue

            dataframe = pd.read_csv(csv_path)

            split_root = root / "DATASET" / split

            for _, row in dataframe.iterrows():

                filename = row["image"]

                label = str(row["label"])

                image_path = split_root / label / filename

                if not image_path.exists():
                    continue

                self.add(

                    self.create_record(

                        sample_id=self.make_sample_id(
                            image_path.stem,
                        ),

                        split=split,

                        modalities=["image"],

                        raw_emotion=label,

                        emotion=RAFDB_EMOTIONS.get(
                            label,
                            "unknown",
                        ),

                        image_path=self.relative_path(
                            image_path
                        ),

                        extras={
                            "label": int(label),
                        },
                    )

                )

        return self.samples