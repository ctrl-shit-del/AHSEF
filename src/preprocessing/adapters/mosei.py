import csv
from typing import List

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import UNIFIED_EMOTIONS


class MOSEIAdapter(BaseAdapter):

    DATASET_NAME = "CMU-MOSEI"

    LABEL_FILE = "label.csv"

    FEATURE_FILE = "Processed/aligned_50.pkl"

    SPLIT_MAPPING = {
        "train": "train",
        "valid": "validation",
        "test": "test",
    }

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():

            raise FileNotFoundError(
                f"CMU-MOSEI dataset directory not found:\n"
                f"{self.dataset_dir}"
            )

        label_file = (
            self.dataset_dir /
            self.LABEL_FILE
        )

        feature_file = (
            self.dataset_dir /
            self.FEATURE_FILE
        )

        if not label_file.exists():

            raise FileNotFoundError(
                f"CMU-MOSEI label file not found:\n"
                f"{label_file}"
            )

        if not feature_file.exists():

            raise FileNotFoundError(
                f"CMU-MOSEI feature file not found:\n"
                f"{feature_file}"
            )

        self.logger.info(
            "Processing label.csv"
        )

        with open(
            label_file,
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                # -----------------------------------------
                # Basic identifiers
                # -----------------------------------------

                video_id = (
                    row["video_id"]
                    .strip()
                )

                clip_id = (
                    row["clip_id"]
                    .strip()
                )

                mode = (
                    row["mode"]
                    .strip()
                    .lower()
                )

                split = self.SPLIT_MAPPING.get(
                    mode,
                    mode,
                )

                # -----------------------------------------
                # Text
                # -----------------------------------------

                text = (
                    row.get("text", "")
                    .strip()
                )

                # -----------------------------------------
                # Annotation
                # -----------------------------------------

                annotation = (
                    row.get("annotation", "")
                    .strip()
                )

                # -----------------------------------------
                # Sentiment label
                # -----------------------------------------

                label_value = row.get(
                    "label",
                    ""
                ).strip()

                label = None

                if label_value:
                    try:
                        label = float(label_value)
                    except ValueError:
                        label = None

                # -----------------------------------------
                # Dimension labels
                #
                # CMU-MOSEI label.csv currently contains
                # label_T, label_A and label_V columns,
                # but inspection showed all three are empty.
                # Preserve them as None.
                # -----------------------------------------

                label_T = None
                label_A = None
                label_V = None

                # -----------------------------------------
                # Emotion
                # -----------------------------------------

                emotion = UNIFIED_EMOTIONS.get(
                    annotation.lower(),
                    annotation.lower(),
                )

                # -----------------------------------------
                # PKL feature identifier
                # -----------------------------------------

                feature_id = (
                    f"{video_id}$_${clip_id}"
                )

                # -----------------------------------------
                # Create record
                # -----------------------------------------

                record = self.create_record(

                    sample_id=self.make_sample_id(
                        video_id,
                        clip_id,
                    ),

                    split=split,

                    modalities=[
                        "audio",
                        "video",
                        "text",
                    ],

                    raw_emotion=annotation,

                    emotion=emotion,

                    text=text,

                    extras={

                        "video_id": video_id,

                        "clip_id": clip_id,

                        "annotation": annotation,

                        "mode": mode,

                        "label": label,

                        "label_T": label_T,

                        "label_A": label_A,

                        "label_V": label_V,

                        "feature_file": (
                            "CMU-MOSEI/Processed/aligned_50.pkl"
                        ),

                        "feature_split": mode,

                        "feature_id": feature_id,
                    },
                )

                # -----------------------------------------
                # Sentiment score
                # -----------------------------------------

                record.sentiment_score = label

                self.add(record)

        return self.samples