import json
from pathlib import Path
from typing import List

import pandas as pd

from src.common.models import EmotionRecord
from src.utils.io import ensure_dir
from src.utils.logger import get_logger


logger = get_logger("MetadataSerializer")


class MetadataSerializer:
    """
    Serialize EmotionRecord objects into different metadata formats.
    """

    @staticmethod
    def to_dataframe(samples: List[EmotionRecord]) -> pd.DataFrame:
        """
        Convert EmotionRecord objects into a pandas DataFrame.
        """

        rows = []

        for sample in samples:

            rows.append({

                "sample_id": sample.sample_id,

                "dataset": sample.dataset,

                "speaker": sample.speaker,

                "gender": sample.gender,

                "raw_emotion": sample.raw_emotion,

                "emotion": sample.emotion,

                "split": sample.split,

                "modalities": ",".join(sample.modalities),

                "audio_path": sample.audio_path,

                "video_path": sample.video_path,

                "image_path": sample.image_path,

                "text": sample.text,

                "physiology_path": sample.physiology_path,

                "valence": sample.valence,

                "arousal": sample.arousal,

                "dominance": sample.dominance,

                "duration": sample.duration,

                "extras": json.dumps(
                    sample.extras,
                    ensure_ascii=False,
                ),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def save_csv(
        samples: List[EmotionRecord],
        output_path: Path,
    ) -> None:
        """
        Save metadata as CSV.
        """

        ensure_dir(output_path.parent)

        df = MetadataSerializer.to_dataframe(samples)

        df.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            f"Saved {len(df)} samples -> {output_path}"
        )

    @staticmethod
    def save_parquet(
        samples: List[EmotionRecord],
        output_path: Path,
    ) -> None:
        """
        Save metadata as Parquet.
        """

        ensure_dir(output_path.parent)

        df = MetadataSerializer.to_dataframe(samples)

        df.to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

        logger.info(
            f"Saved {len(df)} samples -> {output_path}"
        )

    @staticmethod
    def save_json(
        data: dict,
        output_path: Path,
    ) -> None:
        """
        Save statistics or metadata as JSON.
        """

        ensure_dir(output_path.parent)

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            f"Saved JSON -> {output_path}"
        )