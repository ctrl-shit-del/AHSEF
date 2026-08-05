from typing import List
from pathlib import Path

from src.common.constants import MODALITIES
from src.common.models import EmotionRecord
from src.common.paths import DATASETS_DIR


class MetadataValidator:
    """
    Validates EmotionRecord objects before serialization.
    """

    @staticmethod
    def validate(samples: List[EmotionRecord]) -> None:

        errors = []

        errors.extend(
            MetadataValidator._check_empty(samples)
        )

        errors.extend(
            MetadataValidator._check_required_fields(samples)
        )

        errors.extend(
            MetadataValidator._check_modalities(samples)
        )

        errors.extend(
            MetadataValidator._check_paths(samples)
        )

        errors.extend(
            MetadataValidator._check_duplicates(samples)
        )

        if errors:

            raise ValueError(
                "\n\n".join(errors)
            )

    @staticmethod
    def _check_empty(samples):

        errors = []

        if len(samples) == 0:

            errors.append(
                "No samples found."
            )

        return errors

    @staticmethod
    def _check_required_fields(samples):

        errors = []

        for sample in samples:

            if not sample.sample_id:

                errors.append(
                    "Sample missing sample_id."
                )

            if not sample.dataset:

                errors.append(
                    f"{sample.sample_id}: dataset missing"
                )

            if not sample.emotion:

                errors.append(
                    f"{sample.sample_id}: emotion missing"
                )

        return errors

    @staticmethod
    def _check_modalities(samples):

        errors = []

        for sample in samples:

            for modality in sample.modalities:

                if modality not in MODALITIES:

                    errors.append(

                        f"{sample.sample_id}: "

                        f"Unknown modality '{modality}'"

                    )

        return errors

    @staticmethod
    def _check_paths(samples):

        errors = []

        for sample in samples:

            paths = [

                sample.audio_path,

                sample.video_path,

                sample.image_path,

                sample.physiology_path,

            ]

            for relative_path in paths:

                if relative_path is None:
                    continue

                full_path = DATASETS_DIR / relative_path

                if not full_path.exists():

                    errors.append(

                        f"{sample.sample_id}: "

                        f"Missing file -> {relative_path}"

                    )

        return errors

    @staticmethod
    def _check_duplicates(samples):

        errors = []

        seen = set()

        for sample in samples:

            if sample.sample_id in seen:

                errors.append(

                    f"Duplicate sample_id: {sample.sample_id}"

                )

            seen.add(sample.sample_id)

        return errors
    @staticmethod
    def absolute_path(self, relative_path: str) -> Path:

        return DATASETS_DIR / relative_path