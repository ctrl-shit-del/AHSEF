from typing import Dict, List

from pathlib import Path

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import (
    MSP_PODCAST_EMOTIONS,
    MSP_PODCAST_ANNOTATION_STATES,
)

class MSPPodcastAdapter(BaseAdapter):

    DATASET_NAME = "MSP-Podcast"

    LABEL_FILE = (
        "Labels/Labels/labels_consensus.csv"
    )

    AUDIO_DIR = "Audios"
    TRANSCRIPT_DIR = "Transcripts"
    FORCE_ALIGNED_DIR = "ForceAligned"

    def __init__(self):

        super().__init__()

        self.audio_index = {}
        self.transcript_index = {}
        self.force_aligned_index = {}

    # ------------------------------------------------------------------
    # FILE INDEX
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index(
        directory: Path,
        extension: str,
    ) -> Dict[str, Path]:

        index = {}

        if not directory.exists():
            return index

        for path in directory.rglob(f"*{extension}"):

            # Ignore macOS metadata files.
            if path.name.startswith("._"):
                continue

            index[path.stem] = path

        return index

    def _build_indexes(self):

        self.logger.info(
            "Building MSP-Podcast file indexes..."
        )

        self.audio_index = self._build_index(
            self.dataset_dir / self.AUDIO_DIR,
            ".wav",
        )

        self.transcript_index = self._build_index(
            self.dataset_dir / self.TRANSCRIPT_DIR,
            ".txt",
        )

        self.force_aligned_index = self._build_index(
            self.dataset_dir / self.FORCE_ALIGNED_DIR,
            ".TextGrid",
        )

        self.logger.info(
            f"Audio files indexed: "
            f"{len(self.audio_index)}"
        )

        self.logger.info(
            f"Transcripts indexed: "
            f"{len(self.transcript_index)}"
        )

        self.logger.info(
            f"TextGrids indexed: "
            f"{len(self.force_aligned_index)}"
        )

    # ------------------------------------------------------------------
    # EMOTION
    # ------------------------------------------------------------------

    def normalize_emotion(self, code):

        code = str(code).strip().upper()

        if code not in MSP_PODCAST_EMOTIONS:

            raise ValueError(
                f"Unknown MSP-Podcast emotion code: {code}"
            )

        return MSP_PODCAST_EMOTIONS[code]

    # ------------------------------------------------------------------
    # SCAN
    # ------------------------------------------------------------------

    def scan(self) -> List[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():

            raise FileNotFoundError(
                f"MSP-Podcast dataset directory not found:\n"
                f"{self.dataset_dir}"
            )

        label_file = (
            self.dataset_dir / self.LABEL_FILE
        )

        if not label_file.exists():

            raise FileNotFoundError(
                f"MSP-Podcast label file not found:\n"
                f"{label_file}"
            )

        # --------------------------------------------------------------
        # Build indexes once.
        # --------------------------------------------------------------

        self._build_indexes()

        # --------------------------------------------------------------
        # Load consensus labels.
        # --------------------------------------------------------------

        self.logger.info(
            "Loading MSP-Podcast consensus labels..."
        )

        import csv

        with open(
            label_file,
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                filename = (
                    row["FileName"]
                    .strip()
                )

                sample_id = Path(
                    filename
                ).stem

                raw_code = str(row["EmoClass"]).strip().upper()

                emotion = self.normalize_emotion(raw_code)

                annotation_state = MSP_PODCAST_ANNOTATION_STATES.get(
                    raw_code
                )

                # ------------------------------------------------------
                # Audio
                # ------------------------------------------------------

                audio_file = self.audio_index.get(
                    sample_id
                )

                audio_path = None

                if audio_file is not None:

                    audio_path = self.relative_path(
                        audio_file
                    )

                # ------------------------------------------------------
                # Transcript
                # ------------------------------------------------------

                transcript_file = (
                    self.transcript_index.get(
                        sample_id
                    )
                )

                transcript_path = None

                if transcript_file is not None:

                    transcript_path = self.relative_path(
                        transcript_file
                    )

                # ------------------------------------------------------
                # Forced alignment
                # ------------------------------------------------------

                force_aligned_file = (
                    self.force_aligned_index.get(
                        sample_id
                    )
                )

                force_aligned_path = None

                if force_aligned_file is not None:

                    force_aligned_path = self.relative_path(
                        force_aligned_file
                    )

                # ------------------------------------------------------
                # Modalities
                # ------------------------------------------------------

                modalities = []

                if audio_path is not None:
                    modalities.append("audio")

                if transcript_path is not None:
                    modalities.append("text")

                # ------------------------------------------------------
                # Numeric labels
                # ------------------------------------------------------

                arousal = None
                valence = None
                dominance = None

                try:
                    arousal = float(
                        row["EmoAct"]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    pass

                try:
                    valence = float(
                        row["EmoVal"]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    pass

                try:
                    dominance = float(
                        row["EmoDom"]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    pass

                # ------------------------------------------------------
                # Speaker / gender
                # ------------------------------------------------------

                speaker_id = (
                    row["SpkrID"]
                    .strip()
                    if row.get("SpkrID")
                    else None
                )

                gender = (
                    row["Gender"]
                    .strip()
                    if row.get("Gender")
                    else None
                )

                # ------------------------------------------------------
                # Record
                # ------------------------------------------------------

                record = self.create_record(

                    sample_id=self.make_sample_id(
                        Path(filename).stem
                    ),

                    split=str(row["Split_Set"]),

                    modalities=modalities,

                    raw_emotion=raw_code,

                    emotion=emotion,

                    annotation_state=annotation_state,

                    valence=valence,

                    arousal=arousal,

                    dominance=dominance,

                    audio_path=audio_path,

                    text=None,

                    speaker=speaker_id,

                    gender=gender,

                    extras={
                        "file_name": filename,
                        "emotion_code": raw_code,
                        "speaker_id": speaker_id,
                        "gender": gender,
                        "transcript_path": transcript_path,
                        "force_alignment_path": force_aligned_path,
                        "label_file": "Labels/Labels/labels_consensus.csv",
                        "dataset_split": str(row["Split_Set"]),
                    },
                )

                self.add(record)

        return self.samples