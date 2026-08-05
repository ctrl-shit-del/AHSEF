from pathlib import Path

from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import CREMAD_EMOTIONS


class CREMADAdapter(BaseAdapter):

    DATASET_NAME = "CREMA-D"

    def scan(self) -> list[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():

            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_dir}"
            )

        audio_dir = self.dataset_dir / "AudioWAV"

        if not audio_dir.exists():

            raise FileNotFoundError(
                f"Missing AudioWAV directory:\n{audio_dir}"
            )

        for wav_file in sorted(audio_dir.glob("*.wav")):

            parts = wav_file.stem.split("_")

            if len(parts) != 4:
                continue

            actor_id, sentence, emotion_code, level = parts

            emotion = CREMAD_EMOTIONS.get(
                emotion_code,
                "unknown",
            )

            sample = EmotionRecord(

                sample_id=self.make_sample_id(
                    wav_file.stem,
                ),

                dataset="CREMA-D",

                split=None,

                modalities=["audio"],

                raw_emotion=emotion_code,

                emotion=CREMAD_EMOTIONS.get(
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

                gender=None,

                duration=None,

                extras={

                    "sentence": sentence,

                    "emotion_code": emotion_code,

                    "level": level,

                },
            )

            self.add(sample)

        return self.samples