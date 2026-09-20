from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EmotionRecord:

    sample_id: str
    dataset: str

    split: Optional[str] = None
    modalities: list[str] = field(default_factory=list)

    raw_emotion: Optional[Any] = None
    emotion: Optional[str] = None
    annotation_state: Optional[str] = None

    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None
    sentiment_score: Optional[float] = None

    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None
    physiology_path: Optional[str] = None

    speaker: Optional[str] = None
    gender: Optional[str] = None
    duration: Optional[float] = None

    segment_start: Optional[float] = None
    segment_end: Optional[float] = None

    extras: dict[str, Any] = field(default_factory=dict)