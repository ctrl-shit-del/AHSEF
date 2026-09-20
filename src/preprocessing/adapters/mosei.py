import csv
from typing import List

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import MOSEI_EMOTIONS


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
                # Identifiers
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
                # Original annotation
                # -----------------------------------------

                annotation = (
                    row.get("annotation", "")
                    .strip()
                )

                # -----------------------------------------
                # Semantic mapping
                # -----------------------------------------

                emotion = MOSEI_EMOTIONS.get(
                    annotation
                )

                annotation_state = None

                # -----------------------------------------
                # Sentiment score
                # -----------------------------------------

                label = None

                label_value = (
                    row.get("label", "")
                    .strip()
                )

                if label_value:

                    try:

                        label = float(
                            label_value
                        )

                    except ValueError:

                        label = None

                # -----------------------------------------
                # MOSEI V/A/D fields
                #
                # These columns are currently empty in the
                # inspected label.csv.
                # -----------------------------------------

                label_T = None
                label_A = None
                label_V = None

                # -----------------------------------------
                # Feature identifier
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

                    annotation_state=annotation_state,

                    text=text,

                    sentiment_score=label,

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

                self.add(record)

        return self.samples