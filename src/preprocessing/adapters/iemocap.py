"""
IEMOCAP Dataset Adapter

Phase 2

- Parses utterance-level emotion annotations
- Links utterance audio
- Attaches the gold transcript for each utterance
- Video/MOCAP will be added later

Transcripts come from ``Session*/dialog/transcriptions/<dialog>.txt``, one line
per utterance, keyed by exactly the utterance id this adapter already uses:

    Ses01F_impro01_F000 [006.2901-008.2357]: Excuse me.

They are *gold* transcripts, not ASR output, which is the whole reason to read
them here rather than transcribing at training time.  Attaching them at scan
time means the text modality is recorded the same way as every other one -- a
populated payload column plus a declared modality -- and the standardizer
derives ``has_text`` and ``text_source='metadata'`` from that without any
IEMOCAP-specific rule.

A handful of transcript lines carry a bare speaker marker instead of an
utterance id (``M: Yeah.``); they are continuations with no annotation of their
own and are skipped.  Every one of the 10,039 annotated utterances is covered.
"""

import re
from typing import List

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import (
    IEMOCAP_EMOTIONS,
    IEMOCAP_ANNOTATION_STATES,
)


EMOTION_PATTERN = re.compile(
    r"\[(.*?) - (.*?)\]\s+(\S+)\s+(\S+)\s+\[(.*?),(.*?),(.*?)\]"
)

#: ``<utterance_id> [start-end]: transcript``
TRANSCRIPT_PATTERN = re.compile(
    r"^(\S+)\s*\[[\d.]+-[\d.]+\]:\s*(.*)$"
)


def parse_transcript_file(path) -> dict:
    """Map utterance id to transcript text for one dialog.

    Returns an empty mapping when the file is absent, so a partially extracted
    IEMOCAP copy degrades to "no text for that dialog" rather than failing the
    whole scan.
    """
    if not path.exists():
        return {}

    transcripts = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = TRANSCRIPT_PATTERN.match(line.strip())
            if match is None:
                continue
            utterance, text = match.group(1), match.group(2).strip()
            if text:
                transcripts[utterance] = text

    return transcripts


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

                transcripts = parse_transcript_file(
                    session
                    / "dialog"
                    / "transcriptions"
                    / f"{dialog}.txt"
                )

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

                    raw_emotion = raw_emotion.strip().lower()

                    emotion = IEMOCAP_EMOTIONS.get(
                        raw_emotion
                    )

                    annotation_state = IEMOCAP_ANNOTATION_STATES.get(
                        raw_emotion
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

                    text = transcripts.get(utterance)

                    modalities = []

                    if audio_path is not None:
                        modalities.append("audio")

                    if text:
                        modalities.append("text")

                    record = self.create_record(

                        sample_id=utterance,

                        split=session.name,

                        modalities=modalities,

                        raw_emotion=raw_emotion,

                        emotion=emotion,

                        annotation_state=annotation_state,

                        valence=valence,

                        arousal=arousal,

                        dominance=dominance,

                        audio_path=audio_path,

                        text=text,

                        speaker=speaker_code,

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