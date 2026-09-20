"""
WESAD Dataset Adapter

One EmotionRecord is created for each subject.
The protocol timeline is stored in metadata and later used by the
physiological feature extraction pipeline.
"""

import pickle
from typing import List

import numpy as np

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import (
    WESAD_EMOTIONS,
    WESAD_ANNOTATION_STATES,
)


WESAD_SAMPLING_RATES = {

    "chest": {

        "ACC": 700,
        "ECG": 700,
        "EMG": 700,
        "EDA": 700,
        "Temp": 700,
        "Resp": 700,

    },

    "wrist": {

        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,

    },

}


class WESADAdapter(BaseAdapter):

    DATASET_NAME = "WESAD"

    def _extract_protocol(self, labels: np.ndarray):

        protocol = []

        changes = np.where(
            labels[:-1] != labels[1:]
        )[0] + 1

        start = 0

        for end in changes:

            label = int(labels[start])

            protocol.append({

                "raw_emotion": label,

                "emotion": WESAD_EMOTIONS.get(label),

                "annotation_state": WESAD_ANNOTATION_STATES.get(
                    label
                ),

                "segment_start": int(start),

                "segment_end": int(len(labels)),

                "segment_length": int(
                    len(labels) - start
                ),

            })

            start = end

        label = int(labels[start])

        protocol.append({

            "raw_emotion": label,

            "emotion": WESAD_EMOTIONS.get(
                label,
                "unknown",
            ),

            "segment_start": int(start),

            "segment_end": int(len(labels)),

            "segment_length": int(
                len(labels) - start
            ),

        })

        return protocol

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        subjects = sorted(

            subject

            for subject in self.dataset_dir.iterdir()

            if subject.is_dir()
            and subject.name.startswith("S")

        )

        for subject_dir in subjects:

            subject = subject_dir.name

            pkl_path = subject_dir / f"{subject}.pkl"

            if not pkl_path.exists():
                continue

            with open(
                pkl_path,
                "rb",
            ) as f:

                data = pickle.load(
                    f,
                    encoding="latin1",
                )

            protocol = self._extract_protocol(
                data["label"]
            )

            chest_signals = list(
                data["signal"]["chest"].keys()
            )

            wrist_signals = list(
                data["signal"]["wrist"].keys()
            )

            unique, counts = np.unique(
                data["label"],
                return_counts=True,
            )

            self.add(

                self.create_record(

                    sample_id=subject,

                    modalities=["physiology"],

                    physiology_path=self.relative_path(
                        pkl_path
                    ),

                    speaker=subject,

                    extras={

                        "subject": subject,

                        "sampling_rates":
                            WESAD_SAMPLING_RATES,

                        "signals": {

                            "chest": chest_signals,

                            "wrist": wrist_signals,

                        },

                        "signal_shapes": {

                            "chest": {

                                signal: list(
                                    data["signal"]["chest"][signal].shape
                                )

                                for signal in chest_signals

                            },

                            "wrist": {

                                signal: list(
                                    data["signal"]["wrist"][signal].shape
                                )

                                for signal in wrist_signals

                            },

                        },

                        "protocol": protocol,

                        "num_samples": int(
                            len(data["label"])
                        ),

                        "label_counts": {

                            int(label): int(count)

                            for label, count in zip(
                                unique,
                                counts,
                            )

                        },

                        "devices": [

                            "RespiBAN",

                            "Empatica E4",

                        ],

                    },

                )

            )

        return self.samples