"""
RAVDESS Dataset Adapter

Scans the RAVDESS dataset and converts every sample into an
EmotionRecord object.
"""
from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import (
    RAVDESS_EMOTIONS,
    INTENSITY,
    STATEMENT,
    REPETITION,
)


class RAVDESSAdapter(BaseAdapter):
    """
    Adapter for the RAVDESS dataset.

    Dataset Structure:

    datasets/
        RAVDESS/
            Speech/
                Actor_01/
                ...
            Song/
                Actor_01/
                ...
    """

    DATASET_NAME = "RAVDESS"

    def scan(self) -> list[EmotionRecord]:
        """
        Scan the RAVDESS dataset.

        Returns
        -------
        List[EmotionRecord]
            List containing metadata for every sample.
        """
        self.clear()
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_dir}"
            )

        for subset in ["Speech", "Song"]:

            subset_dir = self.dataset_dir / subset

            if not subset_dir.exists():
                continue

            for actor_dir in sorted(subset_dir.iterdir()):

                if not actor_dir.is_dir():
                    continue

                for wav_file in sorted(actor_dir.glob("*.wav")):

                    parts = wav_file.stem.split("-")

                    # Expected filename:
                    # 03-01-05-02-01-02-12.wav

                    if len(parts) != 7:
                        continue

                    (
                        modality,
                        vocal_channel,
                        emotion_code,
                        intensity_code,
                        statement_code,
                        repetition_code,
                        actor_id,
                    ) = parts

                    emotion = RAVDESS_EMOTIONS.get(
                        emotion_code,
                        "unknown",
                    )

                    intensity = INTENSITY.get(
                        intensity_code,
                        "unknown",
                    )

                    statement = STATEMENT.get(
                        statement_code,
                        "unknown",
                    )

                    repetition = REPETITION.get(
                        repetition_code,
                        None,
                    )

                    gender = (
                        "male"
                        if int(actor_id) % 2 == 1
                        else "female"
                    )

                    sample = EmotionRecord(

                        sample_id=self.make_sample_id(
                            subset.lower(),
                            wav_file.stem,
                        ),

                        dataset="RAVDESS",

                        split=None,

                        modalities=["audio"],

                        raw_emotion=emotion_code,

                        emotion=RAVDESS_EMOTIONS.get(
                            emotion_code,
                            "unknown",
                        ),

                        valence=None,
                        arousal=None,
                        dominance=None,

                        audio_path=self.relative_path(wav_file),

                        video_path=None,
                        
                        image_path=None,
                        
                        text=None,
                        
                        physiology_path=None,

                        speaker=actor_id,

                        gender=gender,

                        duration=None,

                        extras={
                            "subset": subset.lower(),
                            "actor": actor_id,
                            "modality_code": modality,
                            "channel": vocal_channel,
                            "intensity": intensity,
                            "statement": statement,
                            "repetition": repetition,
                        },
                    )

                    self.add(sample)

        return self.samples