"""
MELD Dataset Adapter

Phase 1

- Parses MELD utterance-level emotion annotations
- Links utterance-level MP4 video
- Preserves transcript text
- Preserves speaker information
- Preserves sentiment
- Preserves dialogue/episode/season metadata
- Computes utterance duration from StartTime/EndTime

Dataset structure:

datasets/MELD/
├── train_sent_emo.csv
├── dev_sent_emo.csv
├── test_sent_emo.csv
├── train_splits/
│   ├── dia0_utt0.mp4
│   ├── dia0_utt1.mp4
│   └── ...
├── dev_splits_complete/
│   ├── dia0_utt0.mp4
│   └── ...
└── output_repeated_splits_test/
    ├── dia0_utt0.mp4
    └── ...
"""

import csv
from pathlib import Path
from typing import List, Optional

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import MELD_EMOTIONS


class MELDAdapter(BaseAdapter):

    DATASET_NAME = "MELD"

    SPLIT_FILES = {
        "train": "train_sent_emo.csv",
        "validation": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    VIDEO_DIRECTORIES = {
        "train": "train_splits",
        "validation": "dev_splits_complete",
        "test": "output_repeated_splits_test",
    }

    def _parse_time(self, value: str) -> Optional[float]:
        """
        Convert MELD timestamp to seconds.

        MELD uses timestamps such as:

            00:20:57,256

        Returns:
            Seconds as float.
        """

        if not value:
            return None

        value = value.strip()

        try:
            hours, minutes, seconds = value.split(":")

            seconds = seconds.replace(",", ".")

            return (
                int(hours) * 3600
                + int(minutes) * 60
                + float(seconds)
            )

        except (ValueError, TypeError):

            return None

    def _find_video(
        self,
        split: str,
        dialogue_id: str,
        utterance_id: str,
    ) -> Optional[Path]:
        """
        Locate the MELD MP4 corresponding to one utterance.

        Expected filename:

            dia{Dialogue_ID}_utt{Utterance_ID}.mp4
        """

        video_directory_name = self.VIDEO_DIRECTORIES[split]

        video_directory = (
            self.dataset_dir
            / video_directory_name
        )

        filename = (
            f"dia{dialogue_id}_utt{utterance_id}.mp4"
        )

        video_file = video_directory / filename

        if video_file.exists():
            return video_file

        return None

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():

            raise FileNotFoundError(
                f"MELD dataset directory not found:\n"
                f"{self.dataset_dir}"
            )

        for split, csv_filename in self.SPLIT_FILES.items():

            annotation_file = (
                self.dataset_dir
                / csv_filename
            )

            if not annotation_file.exists():

                self.logger.warning(
                    f"Annotation file missing: "
                    f"{annotation_file}"
                )

                continue

            self.logger.info(
                f"Processing {split}: "
                f"{annotation_file.name}"
            )

            with open(
                annotation_file,
                "r",
                encoding="utf-8",
                errors="ignore",
                newline="",
            ) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    # -----------------------------------------
                    # Basic fields
                    # -----------------------------------------

                    dialogue_id = (
                        row.get("Dialogue_ID", "")
                        .strip()
                    )

                    utterance_id = (
                        row.get("Utterance_ID", "")
                        .strip()
                    )

                    utterance = (
                        row.get("Utterance", "")
                        .strip()
                    )

                    speaker = (
                        row.get("Speaker", "")
                        .strip()
                    )

                    raw_emotion = (
                        row.get("Emotion", "")
                        .strip()
                        .lower()
                    )

                    sentiment = (
                        row.get("Sentiment", "")
                        .strip()
                        .lower()
                    )

                    # -----------------------------------------
                    # Emotion mapping
                    # -----------------------------------------

                    emotion = MELD_EMOTIONS.get(
                        raw_emotion,
                        raw_emotion,
                    )

                    # -----------------------------------------
                    # Timing
                    # -----------------------------------------

                    start_time = self._parse_time(
                        row.get("StartTime", "")
                    )

                    end_time = self._parse_time(
                        row.get("EndTime", "")
                    )

                    duration = None

                    if (
                        start_time is not None
                        and end_time is not None
                    ):

                        duration = end_time - start_time

                        if duration < 0:
                            duration = None

                    # -----------------------------------------
                    # Video
                    # -----------------------------------------

                    video_file = self._find_video(
                        split=split,
                        dialogue_id=dialogue_id,
                        utterance_id=utterance_id,
                    )

                    video_path = None

                    if video_file is not None:

                        video_path = self.relative_path(
                            video_file
                        )

                    # -----------------------------------------
                    # Sample ID
                    # -----------------------------------------

                    sample_id = f"{split}_dia{dialogue_id}_utt{utterance_id}"

                    # -----------------------------------------
                    # Modalities
                    # -----------------------------------------

                    modalities = []

                    if video_path is not None:
                        modalities.append("video")

                    if utterance:
                        modalities.append("text")

                    # -----------------------------------------
                    # Create record
                    # -----------------------------------------

                    record = self.create_record(

                        sample_id=sample_id,

                        split=split,

                        modalities=modalities,

                        raw_emotion=raw_emotion,

                        emotion=emotion,

                        valence=None,

                        arousal=None,

                        dominance=None,

                        video_path=video_path,

                        text=utterance,

                        speaker=speaker,

                        duration=duration,

                        segment_start=start_time,

                        segment_end=end_time,

                        extras={

                            "dialogue_id": dialogue_id,

                            "utterance_id": utterance_id,

                            "sentiment": sentiment,

                            "season": row.get(
                                "Season",
                                "",
                            ).strip(),

                            "episode": row.get(
                                "Episode",
                                "",
                            ).strip(),

                            "sr_no": row.get(
                                "Sr No.",
                                "",
                            ).strip(),

                        },
                    )

                    self.add(record)

        return self.samples