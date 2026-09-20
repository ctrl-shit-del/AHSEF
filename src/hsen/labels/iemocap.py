"""IEMOCAP label adapter: categorical protocol plus dimensional valence/arousal.

Two protocols, because the literature reports under two and they are not
interchangeable:

``erc6``
    neutral / happy / sad / angry / excited / frustrated.  The conversational
    protocol ESED, DQ-Former and AdaIGN report against, and the one this phase
    treats as primary -- the survey's IEMOCAP reference point (72.66 weighted
    F1) is an ``erc6`` number.

``ser4``
    neutral / happy / sad / angry with ``excited`` merged into ``happy``.  The
    speech-emotion protocol behind the widely quoted 5,531-utterance subset and
    emotion2vec's 71.79 weighted accuracy.

Neither is the project's canonical seven-class taxonomy, and that is the point.
``excited`` and ``frustrated`` are 2,890 of IEMOCAP's 10,039 utterances and have
no canonical slot, so the canonical space keeps 4,639 utterances and produces a
test partition with ten ``fear`` samples and no ``disgust`` at all.  Reporting
against that would be reporting against a protocol nobody else runs.

Dimensional labels.  IEMOCAP annotates valence, arousal and dominance on a
1-to-5 scale for every utterance, all 10,039 of them.  They are rescaled to
``[-1, 1]`` by a fixed affine map, ``(x - 3) / 2`` -- a constant of the
annotation scale, not a statistic of any split, so no normalisation information
crosses from training data into validation or test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.labels import LabelSpace, get_label_space
from src.hsen.labels.base import MISSING_CLASS_ID, LabelSet

#: The annotation scale IEMOCAP's valence/arousal/dominance ratings live on.
VAD_SCALE_MIN = 1.0
VAD_SCALE_MAX = 5.0

#: Source emotion string to protocol class, per protocol.  ``ser4`` folds
#: ``excited`` into ``happy``; that merge is the protocol's definition, so it
#: belongs here and not in the model.
PROTOCOL_MAPPINGS: dict[str, dict[str, str]] = {
    "erc6": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "excited": "excited",
        "frustrated": "frustrated",
    },
    "ser4": {
        "neutral": "neutral",
        "happy": "happy",
        "excited": "happy",
        "sad": "sad",
        "angry": "angry",
    },
}

#: Where the source emotion string is looked for, in order.  ``emotion`` is the
#: standardized metadata's column; ``source_emotion`` is the manifest's.
EMOTION_COLUMNS: tuple[str, ...] = ("emotion", "source_emotion")

PROTOCOL_SPACES: dict[str, str] = {
    "erc6": "iemocap_erc6",
    "ser4": "iemocap_ser4",
}


def normalise_vad(values: pd.Series) -> np.ndarray:
    """Rescale a 1-5 IEMOCAP dimensional rating to ``[-1, 1]``.

    A fixed affine transform of the known annotation range, deliberately not a
    z-score: fitting a mean and standard deviation would make the target scale
    depend on which rows happened to be in the training split, which is a
    leakage path even though it looks like harmless preprocessing.  Ratings
    outside the documented range are treated as unlabelled rather than clipped,
    because a value off the scale means the annotation was not what we think it
    is.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    in_range = np.isfinite(numeric) & (numeric >= VAD_SCALE_MIN) & (numeric <= VAD_SCALE_MAX)
    midpoint = (VAD_SCALE_MAX + VAD_SCALE_MIN) / 2.0
    half_range = (VAD_SCALE_MAX - VAD_SCALE_MIN) / 2.0
    scaled = np.full(numeric.shape, np.nan, dtype=np.float32)
    scaled[in_range] = ((numeric[in_range] - midpoint) / half_range).astype(np.float32)
    return scaled


@dataclass
class IEMOCAPLabelAdapter:
    """Build IEMOCAP targets under one of the two standard protocols."""

    protocol: str = "erc6"
    #: Column holding the source emotion string, or ``None`` to resolve it.
    #: The adapter runs at two stages over two tables -- the standardized
    #: metadata, where the column is ``emotion``, and the built manifest, where
    #: it is ``source_emotion`` -- and resolving rather than hardcoding is what
    #: lets one adapter define the protocol for both instead of two copies of
    #: the class mapping drifting apart.
    emotion_column: str | None = None

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOL_MAPPINGS:
            raise ValueError(
                f"Unknown IEMOCAP protocol {self.protocol!r}; "
                f"expected one of {sorted(PROTOCOL_MAPPINGS)}"
            )
        self.name = f"iemocap_{self.protocol}"

    @property
    def mapping(self) -> dict[str, str]:
        return PROTOCOL_MAPPINGS[self.protocol]

    def resolve_emotion_column(self, frame: pd.DataFrame) -> str:
        candidates = [self.emotion_column] if self.emotion_column else EMOTION_COLUMNS
        for column in candidates:
            if column in frame.columns:
                return column
        raise ValueError(
            f"No IEMOCAP emotion column found; looked for {candidates} in "
            f"{sorted(frame.columns)[:12]}..."
        )

    def label_space(self) -> LabelSpace:
        return get_label_space(PROTOCOL_SPACES[self.protocol])

    def selects(self, frame: pd.DataFrame) -> pd.Series:
        """Rows whose annotated emotion is one this protocol recognises.

        Everything else -- ``xxx`` (no majority agreement), ``other``, and the
        low-count ``fear`` / ``surprise`` / ``disgust`` annotations IEMOCAP
        carries but no IEMOCAP protocol reports -- is excluded here, once,
        before any split is drawn.
        """
        emotions = frame[self.resolve_emotion_column(frame)].astype("string")
        return emotions.isin(list(self.mapping)).fillna(False)

    def build(self, frame: pd.DataFrame) -> LabelSet:
        space = self.label_space()
        column = self.resolve_emotion_column(frame)
        emotions = frame[column].astype("string")

        class_id = np.full(len(frame), MISSING_CLASS_ID, dtype=np.int64)
        for source, target in self.mapping.items():
            class_id[(emotions == source).to_numpy(dtype=bool)] = space.id_of(target)

        valence = normalise_vad(frame["valence"]) if "valence" in frame else None
        arousal = normalise_vad(frame["arousal"]) if "arousal" in frame else None

        return LabelSet(
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            label_space=space,
            class_id=class_id,
            valence=valence,
            arousal=arousal,
            provenance={
                "adapter": "IEMOCAPLabelAdapter",
                "protocol": self.protocol,
                "emotion_column": column,
                "class_mapping": dict(self.mapping),
                "excited_merged_into_happy": self.protocol == "ser4",
                "vad_scale": [VAD_SCALE_MIN, VAD_SCALE_MAX],
                "vad_transform": "(x - 3) / 2 -> [-1, 1]; fixed, not fitted",
            },
        )
