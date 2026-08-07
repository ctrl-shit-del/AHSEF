from typing import List

import pandas as pd

from src.common.models import EmotionRecord
from src.preprocessing.core.serializer import MetadataSerializer


class MasterMetadataBuilder:

    def __init__(self):

        self.frames = []

    def add(self, samples: List[EmotionRecord]):

        self.frames.append(
            MetadataSerializer.to_dataframe(samples)
        )

    def build(self):

        if not self.frames:
            return pd.DataFrame()

        master = pd.concat(
            self.frames,
            ignore_index=True,
        )

        expected_rows = sum(
            len(frame)
            for frame in self.frames
        )

        assert len(master) == expected_rows

        return master

    def save_csv(self, output_path):

        df = self.build()

        df.to_csv(
            output_path,
            index=False,
        )

    def save_parquet(self, output_path):

        df = self.build()

        df.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

    def summary(self):

        df = self.build()

        datasets = {}

        for dataset, group in df.groupby("dataset"):

            datasets[dataset] = {

                "samples": int(len(group)),

                "modalities": (
                    group["modalities"]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),

                "emotions": (
                    group["emotion"]
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),

                "splits": (
                    group["split"]
                    .fillna("unspecified")
                    .value_counts()
                    .sort_index()
                    .to_dict()
                ),
            }

        return {

            "num_datasets": int(
                df["dataset"].nunique()
            ),

            "total_samples": int(
                len(df)
            ),

            "dataset_distribution": (
                df["dataset"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),

            "modality_distribution": (
                df["modalities"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),

            "emotion_distribution": (
                df["emotion"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),

            "datasets": datasets,
        }