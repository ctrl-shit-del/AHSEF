import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.common.paths import DATASETS_DIR
from src.data.records import LoadedSample
from src.data.loaders.audio import AudioLoader
from src.data.loaders.image import ImageLoader
from src.data.loaders.mosei import MOSEILoader
from src.data.loaders.physiology import PhysiologyLoader
from src.data.loaders.text import TextLoader
from src.data.loaders.video import VideoLoader


STANDARDIZED_PATH = Path(
    "metadata/standardized/standardized.parquet"
)


class SampleResolver:
    """
    Resolve standardized metadata records into
    model-ready samples.
    """

    def __init__(
        self,
        metadata_path: Path = STANDARDIZED_PATH,
    ):

        self.metadata_path = metadata_path

        self.df = pd.read_parquet(
            metadata_path
        )

        self.audio_loader = AudioLoader(
            DATASETS_DIR
        )

        self.image_loader = ImageLoader(
            DATASETS_DIR
        )

        self.video_loader = VideoLoader(
            DATASETS_DIR
        )

        self.text_loader = TextLoader(DATASETS_DIR)

        self.physiology_loader = (
            PhysiologyLoader(DATASETS_DIR)
        )

        self._mosei_loader: Optional[
            MOSEILoader
        ] = None

    def _get_mosei_loader(
        self,
        feature_file: str,
    ) -> MOSEILoader:

        if self._mosei_loader is None:

            self._mosei_loader = MOSEILoader(
                datasets_root=DATASETS_DIR,
                feature_file=feature_file,
            )

        return self._mosei_loader

    def get_row(
        self,
        sample_id: str,
    ) -> pd.Series:

        matches = self.df[
            self.df["sample_id"]
            == sample_id
        ]

        if len(matches) == 0:
            raise KeyError(
                f"Sample not found: {sample_id}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Duplicate sample ID: {sample_id}"
            )

        return matches.iloc[0]

    def resolve(
        self,
        sample_id: str,
        load_modalities: bool = True,
    ) -> LoadedSample:

        row = self.get_row(
            sample_id
        )

        extras = {}

        if pd.notna(row["extras"]):

            try:
                extras = json.loads(
                    row["extras"]
                )
            except (
                json.JSONDecodeError,
                TypeError,
            ):
                extras = {}

        sample = LoadedSample(

            sample_id=row["sample_id"],

            dataset=row["dataset"],

            split=row["training_split"]
            if pd.notna(row["training_split"])
            else None,

            emotion=row["canonical_emotion"]
            if pd.notna(
                row["canonical_emotion"]
            )
            else None,

            emotion_id=int(
                row["canonical_emotion_id"]
            )
            if pd.notna(
                row["canonical_emotion_id"]
            )
            else None,

            valence=row["valence"]
            if pd.notna(row["valence"])
            else None,

            arousal=row["arousal"]
            if pd.notna(row["arousal"])
            else None,

            dominance=row["dominance"]
            if pd.notna(row["dominance"])
            else None,

            sentiment=row["sentiment_score"]
            if pd.notna(
                row["sentiment_score"]
            )
            else None,

            extras=extras,
        )

        if not load_modalities:
            return sample

        # -----------------------------------------------------
        # CMU-MOSEI
        # -----------------------------------------------------

        if row["dataset"] == "CMU-MOSEI":

            feature_file = row[
                "feature_file"
            ]

            feature_split = row[
                "feature_split"
            ]

            feature_id = row[
                "feature_id"
            ]

            if (
                pd.notna(feature_file)
                and pd.notna(feature_split)
                and pd.notna(feature_id)
            ):

                loader = self._get_mosei_loader(
                    feature_file
                )

                features = loader.get(
                    feature_split,
                    feature_id,
                )

                sample.audio = features[
                    "audio"
                ]

                sample.video = features[
                    "video"
                ]

                sample.text = (
                    str(features["raw_text"])
                )

                sample.extras[
                    "mosei_features"
                ] = {
                    "text_features": features[
                        "text_features"
                    ],
                    "text_bert": features[
                        "text_bert"
                    ],
                    "classification_label":
                        features[
                            "classification_label"
                        ],
                    "regression_label":
                        features[
                            "regression_label"
                        ],
                }

            return sample

        # -----------------------------------------------------
        # Normal text
        # -----------------------------------------------------

        if row["has_text"]:
            if row["text_source"] == "metadata" and pd.notna(row["text"]):
                sample.text = self.text_loader.load(text=row["text"])
            elif row["text_source"] == "file":
                transcript_path = extras.get("transcript_path")
                if not transcript_path:
                    raise ValueError(f"File-backed text has no transcript path: {sample_id}")
                sample.text = self.text_loader.load(relative_path=transcript_path)

        # -----------------------------------------------------
        # Audio
        # -----------------------------------------------------

        if (
            row["has_audio"]
            and pd.notna(row["audio_path"])
        ):

            sample.audio = (
                self.audio_loader.load(
                    row["audio_path"]
                )
            )

        # -----------------------------------------------------
        # Video
        # -----------------------------------------------------

        if (
            row["has_video"]
            and pd.notna(row["video_path"])
        ):

            sample.video = (
                self.video_loader.load(
                    row["video_path"]
                )
            )

        # -----------------------------------------------------
        # Image
        # -----------------------------------------------------

        if (
            row["has_image"]
            and pd.notna(row["image_path"])
        ):

            sample.image = (
                self.image_loader.load(
                    row["image_path"]
                )
            )

        # -----------------------------------------------------
        # Physiology
        # -----------------------------------------------------

        if (
            row["has_physiology"]
            and pd.notna(
                row["physiology_path"]
            )
        ):

            sample.physiology = (
                self.physiology_loader.load(
                    row["physiology_path"]
                )
            )

        return sample
