"""Modality-availability and task-validity rules for experiment sampling.

The standardized metadata records modality availability explicitly
(``has_<modality>``) together with the provenance of that modality
(``<modality>_source``).  Sampling never fabricates a modality: a record is
eligible only when availability, source, and -- where a file backs the
modality -- the path are all present.

Values arriving from Arrow may be ``None`` or a float ``NaN`` depending on how
a nullable column was materialised, so every predicate goes through the
NaN-safe coercions below rather than trusting Python truthiness (``bool(nan)``
is ``True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.common.labels import EMOTION_7CLASS


# ============================================================
# NaN-safe coercions
# ============================================================

def is_missing(value: Any) -> bool:
    """True for ``None``, float ``NaN``, and empty strings."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def as_bool(value: Any) -> bool:
    """Coerce a nullable flag to ``bool`` without treating ``NaN`` as true."""
    return False if is_missing(value) else bool(value)


def as_label(value: Any) -> int | None:
    """Coerce a nullable integral label to ``int``; return ``None`` otherwise."""
    if is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


# ============================================================
# Rules
# ============================================================

@dataclass(frozen=True)
class ModalityRule:
    """Eligibility contract for one modality.

    A modality can be backed differently by different datasets.  MELD text
    lives in the metadata ``text`` column while MSP-Podcast text lives in a
    transcript file addressed by ``text_path``; both are real text, and
    ``source_requirements`` states which column has to be present for which
    ``<modality>_source`` value.  When it is not given, ``path_column`` and
    ``value_column`` apply uniformly to every accepted source, which is the
    behaviour every single-source modality relies on.
    """

    modality: str
    availability_column: str
    source_column: str
    accepted_sources: tuple[str, ...]
    path_column: str | None = None
    value_column: str | None = None
    default_datasets: tuple[str, ...] | None = None
    #: ``{source: (column, ...)}`` -- at least one listed column must be
    #: present for a record carried by that source.
    source_requirements: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self.source_requirements is not None:
            unknown = set(self.source_requirements) - set(self.accepted_sources)
            if unknown:
                raise ValueError(
                    f"source_requirements for modality {self.modality!r} names "
                    f"unaccepted sources: {sorted(unknown)}"
                )
            object.__setattr__(
                self,
                "source_requirements",
                {source: tuple(columns) for source, columns in self.source_requirements.items()},
            )

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = [self.availability_column, self.source_column]
        if self.path_column:
            columns.append(self.path_column)
        if self.value_column:
            columns.append(self.value_column)
        for required in (self.source_requirements or {}).values():
            columns.extend(required)
        # ``dict.fromkeys`` keeps declaration order while de-duplicating.
        return tuple(dict.fromkeys(columns))

    def payload_columns(self, source: str | None = None) -> tuple[str, ...]:
        """Columns that may carry the modality payload for ``source``."""
        if source is not None and self.source_requirements:
            required = self.source_requirements.get(str(source))
            if required is not None:
                return required
        return tuple(column for column in (self.path_column, self.value_column) if column)

    def is_eligible(self, row: Mapping[str, Any]) -> bool:
        if not as_bool(row.get(self.availability_column)):
            return False
        source = row.get(self.source_column)
        if is_missing(source) or str(source) not in self.accepted_sources:
            return False
        required = (self.source_requirements or {}).get(str(source))
        if required is not None:
            return any(not is_missing(row.get(column)) for column in required)
        if self.path_column and is_missing(row.get(self.path_column)):
            return False
        if self.value_column and is_missing(row.get(self.value_column)):
            return False
        return True


@dataclass(frozen=True)
class TaskRule:
    """Target-validity contract for one experiment task."""

    task: str
    target_type: str
    validity_column: str
    label_column: str | None = None
    valid_label_ids: tuple[int, ...] | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns = ["target_type", self.validity_column]
        if self.label_column:
            columns.append(self.label_column)
        return tuple(columns)

    def label_of(self, row: Mapping[str, Any]) -> int | None:
        """Return the validated canonical label, or ``None`` when unusable."""
        if self.label_column is None:
            return None
        label = as_label(row.get(self.label_column))
        if label is None:
            return None
        if self.valid_label_ids is not None and label not in self.valid_label_ids:
            return None
        return label

    def is_eligible(self, row: Mapping[str, Any]) -> bool:
        if row.get("target_type") != self.target_type:
            return False
        if not as_bool(row.get(self.validity_column)):
            return False
        if self.label_column is not None and self.label_of(row) is None:
            return False
        return True


#: Datasets that may contribute to each modality's common-task pool.  These
#: are membership filters, not promises: a listed dataset only contributes the
#: records that also satisfy the modality rule and the task rule.
IMAGE_DATASETS = ("AffectNet+", "FERPlus", "RAF-DB")
AUDIO_DATASETS = ("RAVDESS", "CREMA-D", "IEMOCAP", "MSP-Podcast")
TEXT_DATASETS = ("MELD", "CMU-MOSEI", "MSP-Podcast")
VIDEO_DATASETS = ("MELD", "CMU-MOSEI")
PHYSIOLOGY_DATASETS = ("WESAD",)

MODALITY_RULES: dict[str, ModalityRule] = {
    "image": ModalityRule(
        modality="image",
        availability_column="has_image",
        source_column="image_source",
        accepted_sources=("file",),
        path_column="image_path",
        default_datasets=IMAGE_DATASETS,
    ),
    "audio": ModalityRule(
        modality="audio",
        availability_column="has_audio",
        source_column="audio_source",
        accepted_sources=("file",),
        path_column="audio_path",
        default_datasets=AUDIO_DATASETS,
    ),
    "video": ModalityRule(
        modality="video",
        availability_column="has_video",
        source_column="video_source",
        accepted_sources=("file",),
        path_column="video_path",
        default_datasets=VIDEO_DATASETS,
    ),
    "text": ModalityRule(
        modality="text",
        availability_column="has_text",
        source_column="text_source",
        accepted_sources=("metadata", "file"),
        value_column="text",
        path_column="text_path",
        default_datasets=TEXT_DATASETS,
        # Metadata-backed transcripts live in ``text``; file-backed ones
        # (MSP-Podcast) live behind ``text_path``.  Neither substitutes for
        # the other, so each source declares its own requirement.
        source_requirements={"metadata": ("text",), "file": ("text_path",)},
    ),
    "physiology": ModalityRule(
        modality="physiology",
        availability_column="has_physiology",
        source_column="physiology_source",
        accepted_sources=("file",),
        path_column="physiology_path",
        default_datasets=PHYSIOLOGY_DATASETS,
    ),
}

TASK_RULES: dict[str, TaskRule] = {
    "emotion_7class": TaskRule(
        task="emotion_7class",
        target_type="categorical_emotion",
        validity_column="emotion_target_valid",
        label_column="canonical_emotion_id",
        valid_label_ids=EMOTION_7CLASS.valid_ids,
    ),
    "sentiment": TaskRule(
        task="sentiment",
        target_type="sentiment",
        validity_column="sentiment_target_valid",
    ),
}


def get_modality_rule(modality: str) -> ModalityRule:
    try:
        return MODALITY_RULES[modality]
    except KeyError as error:
        raise ValueError(
            f"Unknown modality {modality!r}; expected one of {sorted(MODALITY_RULES)}"
        ) from error


def get_task_rule(task: str) -> TaskRule:
    try:
        return TASK_RULES[task]
    except KeyError as error:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_RULES)}") from error
