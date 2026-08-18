from src.common.models import EmotionRecord
from src.preprocessing.core.base_adapter import BaseAdapter
from src.preprocessing.emotion_mapping import (
    RAVDESS_EMOTIONS,
    INTENSITY,
    STATEMENT,
    REPETITION,
)


class RAVDESSAdapter(BaseAdapter):

    DATASET_NAME = "RAVDESS"

    def scan(self) -> list[EmotionRecord]:

        self.clear()

        if not self.dataset_exists():

            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.dataset_dir}"
            )

        for subset in ("Speech", "Song"):

            subset_dir = self.dataset_dir / subset

            if not subset_dir.exists():
                continue

            actor_dirs = sorted(

                path

                for path in subset_dir.iterdir()

                if path.is_dir()

            )

            for actor_dir in actor_dirs:

                actor = actor_dir.name.replace(
                    "Actor_",
                    "",
                )

                for wav_file in sorted(
                    actor_dir.glob("*.wav")
                ):

                    parts = wav_file.stem.split("-")

                    if len(parts) != 7:
                        continue

                    (
                        modality,
                        vocal_channel,
                        emotion_code,
                        intensity,
                        statement,
                        repetition,
                        actor_id,
                    ) = parts

                    self.add(

                        self.create_record(

                            sample_id=self.make_sample_id(
                                subset.lower(),
                                wav_file.stem,
                            ),

                            split=subset.lower(),

                            modalities=["audio"],

                            raw_emotion=emotion_code,

                            emotion=RAVDESS_EMOTIONS.get(
                                emotion_code,
                                "unknown",
                            ),

                            audio_path=self.relative_path(
                                wav_file,
                            ),

                            speaker=actor,

                            gender=(
                                "male"
                                if int(actor) % 2
                                else "female"
                            ),

                            extras={

                                "subset": subset,

                                "modality": modality,

                                "vocal_channel": vocal_channel,

                                "intensity": INTENSITY.get(
                                    intensity,
                                ),

                                "statement": STATEMENT.get(
                                    statement,
                                ),

                                "repetition": REPETITION.get(
                                    repetition,
                                ),

                            },

                        )

                    )

        return self.samples