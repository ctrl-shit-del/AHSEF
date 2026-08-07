"""
FERPlus Dataset Adapter

Reads the FERPlus parquet dataset, extracts images on the
first run, and converts every sample into an EmotionRecord.
"""

from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import FERPLUS_EMOTIONS


class FERPlusAdapter(BaseAdapter):

    DATASET_NAME = "FERPlus"

    PARQUET_FILE = "data/train-00000-of-00001.parquet"

    def extract_images(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
    ) -> None:
        """
        Extract PNG images stored inside the parquet file.

        Extraction is performed only once.
        """

        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for idx, row in dataframe.iterrows():

            output_file = image_dir / f"{idx:06d}.png"

            if output_file.exists():
                continue

            image_bytes = row["image"]["bytes"]

            image = Image.open(
                BytesIO(image_bytes)
            )

            image.save(output_file)

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        parquet_path = (
            self.dataset_dir /
            self.PARQUET_FILE
        )

        if not parquet_path.exists():

            raise FileNotFoundError(
                parquet_path
            )

        dataframe = pd.read_parquet(
            parquet_path
        )

        image_dir = (
            self.dataset_dir /
            "images"
        )

        if not image_dir.exists():

            self.logger.info(
                "Extracting FERPlus images..."
            )

            self.extract_images(
                dataframe,
                image_dir,
            )

            self.logger.info(
                "Image extraction completed."
            )

        self.logger.info(
            "Building metadata..."
        )

        for idx, row in dataframe.iterrows():

            image_file = (
                image_dir /
                f"{idx:06d}.png"
            )

            self.add(

                self.create_record(

                    sample_id=str(idx),

                    split="train",

                    modalities=["image"],

                    raw_emotion=row["label"],

                    emotion=FERPLUS_EMOTIONS.get(
                        row["label"],
                        "unknown",
                    ),

                    image_path=self.relative_path(
                        image_file
                    ),

                )

            )

        return self.samples