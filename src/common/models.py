from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(slots=True)
class EmotionRecord:
    """
    Unified metadata representation for a single sample
    across all supported emotion datasets.
    """

    sample_id: str
    dataset: str

    split: str | None

    modalities: list[str]

    raw_emotion: Optional[str | int]

    emotion: Optional[str]

    valence: float | None
    arousal: float | None
    dominance: float | None

    audio_path: str | None
    video_path: str | None
    image_path: str | None
    text: str | None
    physiology_path: str | None

    speaker: str | None
    gender: str | None

    duration: float | None

    extras: dict[str, Any] = field(default_factory=dict)