from typing import List

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

        errors.extend(
            MetadataValidator._check_dimensions(samples)
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

            if not sample.modalities:
                errors.append(
                    f"{sample.sample_id}: modalities missing"
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

        modality_paths = {
            "audio": "audio_path",
            "video": "video_path",
            "image": "image_path",
            "physiology": "physiology_path",
        }

        FEATURE_BACKED_DATASETS = {
            "CMU-MOSEI",
        }

        for sample in samples:

            for modality in sample.modalities:

                if sample.dataset in FEATURE_BACKED_DATASETS:
                    continue

                attribute = modality_paths.get(modality)

                if attribute is None:
                    continue

                relative_path = getattr(
                    sample,
                    attribute,
                )

                if relative_path is None:

                    errors.append(
                        f"{sample.sample_id}: "
                        f"{attribute} missing"
                    )

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

            key = (
                sample.dataset,
                sample.sample_id,
            )

            if key in seen:

                errors.append(
                    f"Duplicate sample: {key}"
                )

            seen.add(key)

        return errors

    @staticmethod
    def _check_dimensions(samples):

        errors = []

        for sample in samples:

            # -------------------------------------------------
            # IEMOCAP
            # -------------------------------------------------

            if sample.dataset == "IEMOCAP":

                if sample.valence is not None:

                    if not (
                        1.0
                        <= sample.valence
                        <= 5.5
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid valence"
                        )

                if sample.arousal is not None:

                    if not (
                        1.0
                        <= sample.arousal
                        <= 5.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid arousal"
                        )

                if sample.dominance is not None:

                    if not (
                        0.5
                        <= sample.dominance
                        <= 5.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid dominance"
                        )

            # -------------------------------------------------
            # MSP-Podcast
            # -------------------------------------------------

            elif sample.dataset == "MSP-Podcast":

                # MSP-Podcast uses a 1-7 scale
                # for Valence, Arousal and Dominance.

                if sample.valence is not None:

                    if not (
                        1.0
                        <= sample.valence
                        <= 7.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid valence"
                        )

                if sample.arousal is not None:

                    if not (
                        1.0
                        <= sample.arousal
                        <= 7.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid arousal"
                        )

                if sample.dominance is not None:

                    if not (
                        1.0
                        <= sample.dominance
                        <= 7.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid dominance"
                        )

            # -------------------------------------------------
            # Other datasets
            # -------------------------------------------------

            else:

                if sample.valence is not None:

                    if not (
                        -1.0
                        <= sample.valence
                        <= 1.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid valence"
                        )

                if sample.arousal is not None:

                    if not (
                        -1.0
                        <= sample.arousal
                        <= 1.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid arousal"
                        )

                if sample.dominance is not None:

                    if not (
                        -1.0
                        <= sample.dominance
                        <= 1.0
                    ):

                        errors.append(
                            f"{sample.sample_id}: "
                            f"invalid dominance"
                        )

        return errors

    @staticmethod
    def _check_extras(samples):

        errors = []

        for sample in samples:

            if sample.extras is None:

                errors.append(
                    f"{sample.sample_id}: extras is None"
                )

        return errors