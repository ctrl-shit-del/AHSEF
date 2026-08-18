from typing import List

import pandas as pd

from src.common.models import EmotionRecord
from src.preprocessing.core.serializer import MetadataSerializer
from src.utils.logger import get_logger

logger = get_logger("MasterMetadata")


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

        # ---------- Normalize dataframe types ----------

        object_columns = master.select_dtypes(
            include="object"
        ).columns

        for column in object_columns:

            master[column] = master[column].astype("string")

        # keep numeric columns numeric
        for column in (
            "valence",
            "arousal",
            "dominance",
            "sentiment_score",
            "duration",
            "segment_start",
            "segment_end",
        ):

            if column in master.columns:

                master[column] = pd.to_numeric(
                    master[column],
                    errors="coerce",
                )

        return master

    def save_csv(self, output_path):

        df = self.build()

        df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Saved {len(df)} samples -> {output_path}"
        )

    def save_parquet(self, output_path):

        df = self.build()

        df.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

        logger.info(
            f"Saved {len(df)} samples -> {output_path}"
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