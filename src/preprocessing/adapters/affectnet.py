"""
AffectNet+ Dataset Adapter
"""

import json
from typing import List

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import AFFECTNET_EMOTIONS


class AffectNetAdapter(BaseAdapter):

    DATASET_NAME = "AffectNet+"

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        human_root = self.dataset_dir / "human_annotated"

        subsets = {

            "train": human_root / "train_set",

            "validation": human_root / "validation_set",

        }

        for split, root in subsets.items():

            image_dir = root / "images"

            annotation_dir = root / "annotations"

            if not image_dir.exists():
                continue

            for image_path in sorted(image_dir.glob("*.jpg")):

                json_path = annotation_dir / f"{image_path.stem}.json"

                if not json_path.exists():
                    continue

                with open(json_path, "r") as f:

                    annotation = json.load(f)

                emotion_id = annotation.get(
                    "human-label",
                    8,
                )

                meta = annotation.get(
                    "meta-data",
                    {},
                )

                gender = None

                if "gender" in meta:

                    gender = max(
                        meta["gender"],
                        key=meta["gender"].get,
                    )

                self.add(

                    self.create_record(

                        sample_id=self.make_sample_id(
                            split,
                            image_path.stem,
                        ),

                        split=split,

                        modalities=["image"],

                        raw_emotion=emotion_id,

                        emotion=AFFECTNET_EMOTIONS.get(
                            emotion_id,
                            "unknown",
                        ),

                        image_path=self.relative_path(
                            image_path
                        ),

                        valence=meta.get(
                            "valence"
                        ),

                        arousal=meta.get(
                            "arousal"
                        ),

                        gender=gender,

                        extras={

                            "age": meta.get("age"),

                            "race": meta.get("race"),

                            "pose": meta.get("pose"),

                            "soft_label": annotation.get(
                                "soft-label"
                            ),

                            "subset": annotation.get(
                                "subset"
                            ),

                            "landmark_29": meta.get(
                                "landmark-29"
                            ),

                            "landmark_68": meta.get(
                                "landmark-68"
                            ),
                        },
                    )

                )

        return self.samples