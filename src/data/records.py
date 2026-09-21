from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoadedSample:
    """
    Model-ready representation of one metadata sample.

    The object contains the original sample identity and
    whichever modalities/features are available.
    """

    sample_id: str
    dataset: str
    split: Optional[str] = None

    emotion: Optional[str] = None
    emotion_id: Optional[int] = None

    valence: Optional[float] = None
    arousal: Optional[float] = None
    dominance: Optional[float] = None

    sentiment: Optional[float] = None

    text: Optional[str] = None

    audio: Any = None
    video: Any = None
    image: Any = None
    physiology: Any = None

    extras: dict[str, Any] = field(
        default_factory=dict
    )