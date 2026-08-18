"""
IEMOCAP Dataset Adapter

Phase 2

- Parses utterance-level emotion annotations
- Links utterance audio
- Transcript/video/MOCAP will be added later
"""

import re
from typing import List

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import IEMOCAP_EMOTIONS


EMOTION_PATTERN = re.compile(
    r"\[(.*?) - (.*?)\]\s+(\S+)\s+(\S+)\s+\[(.*?),(.*?),(.*?)\]"
)


class IEMOCAPAdapter(BaseAdapter):

    DATASET_NAME = "IEMOCAP"

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_dir}"
            )

        sessions = sorted(
            self.dataset_dir.glob("Session*")
        )

        for session in sessions:

            evaluation_dir = (
                session
                / "dialog"
                / "EmoEvaluation"
            )

            if not evaluation_dir.exists():
                continue

            for evaluation_file in sorted(
                evaluation_dir.glob("*.txt")
            ):

                dialog = evaluation_file.stem

                with open(
                    evaluation_file,
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as f:

                    lines = f.readlines()

                for line in lines:

                    match = EMOTION_PATTERN.match(line)

                    if match is None:
                        continue

                    (
                        start,
                        end,
                        utterance,
                        raw_emotion,
                        valence,
                        arousal,
                        dominance,
                    ) = match.groups()

                    start = float(start)
                    end = float(end)

                    valence = float(valence)
                    arousal = float(arousal)
                    dominance = float(dominance)

                    emotion = IEMOCAP_EMOTIONS.get(
                        raw_emotion,
                        "unknown",
                    )

                    # IEMOCAP utterance IDs look like:
                    # Ses01F_impro01_F000
                    #
                    # The speaker is the final F/M marker,
                    # not the first part of the utterance ID.
                    speaker_code = utterance.split("_")[-1][0]

                    gender = (
                        "female"
                        if speaker_code == "F"
                        else "male"
                        if speaker_code == "M"
                        else None
                    )

                    audio_file = (
                        session
                        / "sentences"
                        / "wav"
                        / dialog
                        / f"{utterance}.wav"
                    )

                    audio_path = None

                    if audio_file.exists():
                        audio_path = self.relative_path(
                            audio_file
                        )

                    modalities = []

                    if audio_path is not None:
                        modalities.append("audio")

                    record = self.create_record(

                        sample_id=utterance,

                        split=session.name,

                        modalities=modalities,

                        raw_emotion=raw_emotion,

                        emotion=emotion,

                        valence=valence,

                        arousal=arousal,

                        dominance=dominance,

                        audio_path=audio_path,

                        speaker=utterance,

                        gender=gender,

                        duration=end - start,

                        segment_start=start,

                        segment_end=end,

                        extras={
                            "session": session.name,
                            "dialog": dialog,
                            "utterance": utterance,
                        },

                    )

                    self.add(record)

        return self.samples