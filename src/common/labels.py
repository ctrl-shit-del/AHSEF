"""Centralised, explicitly ordered label spaces for every baseline experiment.

Class order is a scientific contract, not a presentation detail: the confusion
matrix, the per-class arrays, the class-weight vector, the checkpoint metadata,
and the logs all index the same positions.  Sorting a class list
alphabetically anywhere in the codebase silently rotates every one of those
artefacts, so the order is declared exactly once here and imported everywhere
else.

The canonical categorical emotion order is, and must remain::

    0 neutral, 1 happy, 2 sad, 3 angry, 4 fear, 5 disgust, 6 surprise

WESAD is the one modality pool whose standardized metadata carries no
canonical emotion at all (``target_type == 'physiological_state'``).  Rather
than forcing invalid emotion labels onto it, it gets its own declared label
space; see :data:`WESAD_STATE_3CLASS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


# ============================================================
# Canonical orders
# ============================================================

#: The immutable 7-class categorical emotion order.  Never sort this.
CANONICAL_EMOTION_CLASSES: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
)

#: The IEMOCAP six-way conversational protocol (ESED, DQ-Former, AdaIGN and the
#: wider ERC literature).  ``excited`` and ``frustrated`` are the two classes the
#: canonical seven-class taxonomy has no slot for, and they are also the two that
#: make the protocol comparable: dropping them leaves 4,639 of 10,039 utterances
#: and a test partition with single-digit minority classes.
IEMOCAP_ERC6_CLASSES: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "excited",
    "frustrated",
)

#: The IEMOCAP four-way speech-emotion protocol, in which ``excited`` is merged
#: into ``happy``.  This is the split behind the widely quoted 5,531-utterance
#: figure and the protocol emotion2vec reports against.
IEMOCAP_SER4_CLASSES: tuple[str, ...] = ("neutral", "happy", "sad", "angry")

#: CMU-MOSEI's seven sentiment buckets, ordered from most negative to most
#: positive.  The order is the scale, so this one is not merely a convention:
#: rounding a sentiment score in [-3, 3] and adding 3 gives the index directly,
#: and any reordering would break that correspondence silently.
MOSEI_SENTIMENT7_CLASSES: tuple[str, ...] = (
    "highly_negative",
    "negative",
    "weakly_negative",
    "neutral",
    "weakly_positive",
    "positive",
    "highly_positive",
)

#: The six non-exclusive CMU-MOSEI emotions, in the order LDDU, CARAT and
#: TAILOR report per-label results in.  Declared now so the multi-label head
#: has a stable class order the day the annotations are available; nothing in
#: this repository can populate it yet (see :mod:`src.hsen.labels.mosei`).
MOSEI_EMOTION6_CLASSES: tuple[str, ...] = (
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
)

#: WESAD study conditions in the order used by the standard 3-class protocol.
WESAD_STATE_CLASSES: tuple[str, ...] = ("baseline", "stress", "amusement")

#: The 3-class protocol plus the meditation condition, kept as an explicit
#: opt-in so the default baseline stays comparable with published WESAD work.
WESAD_STATE_CLASSES_EXTENDED: tuple[str, ...] = (*WESAD_STATE_CLASSES, "meditation")


# ============================================================
# Label space
# ============================================================

@dataclass(frozen=True)
class LabelSpace:
    """An ordered, immutable set of class names for one classification task."""

    name: str
    classes: tuple[str, ...]
    description: str = ""
    #: Set when the space is deliberately not the common 7-class emotion task.
    limitation: str | None = None
    _index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        classes = tuple(self.classes)
        if not classes:
            raise ValueError(f"Label space {self.name!r} must declare at least one class")
        if len(set(classes)) != len(classes):
            raise ValueError(f"Label space {self.name!r} contains duplicate class names")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(
            self, "_index", MappingProxyType({name: index for index, name in enumerate(classes)})
        )

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def mapping(self) -> Mapping[str, int]:
        """Class name to identifier, in declaration order."""
        return self._index

    def id_of(self, class_name: str) -> int:
        try:
            return self._index[class_name]
        except KeyError as error:
            raise ValueError(
                f"{class_name!r} is not a class of label space {self.name!r}; "
                f"expected one of {list(self.classes)}"
            ) from error

    def name_of(self, class_id: int) -> str:
        if not 0 <= int(class_id) < len(self.classes):
            raise ValueError(
                f"class id {class_id} is outside label space {self.name!r} "
                f"(0..{len(self.classes) - 1})"
            )
        return self.classes[int(class_id)]

    @property
    def valid_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self.classes)))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "classes": list(self.classes),
            "num_classes": self.num_classes,
            "mapping": dict(self._index),
            "description": self.description,
            "limitation": self.limitation,
        }


EMOTION_7CLASS = LabelSpace(
    name="emotion_7class",
    classes=CANONICAL_EMOTION_CLASSES,
    description="Common 7-class categorical emotion taxonomy shared by every dataset "
                "whose standardized metadata declares a valid canonical emotion.",
)

IEMOCAP_ERC6 = LabelSpace(
    name="iemocap_erc6",
    classes=IEMOCAP_ERC6_CLASSES,
    description="IEMOCAP six-way conversational emotion protocol "
                "(neutral / happy / sad / angry / excited / frustrated).",
    limitation=(
        "Two of these classes -- excited and frustrated -- are outside the canonical "
        "seven-class taxonomy, so an iemocap_erc6 prediction is not directly comparable "
        "with a seven-class emotion_7class prediction and the two must never share a "
        "confusion matrix or a class-weight vector."
    ),
)

IEMOCAP_SER4 = LabelSpace(
    name="iemocap_ser4",
    classes=IEMOCAP_SER4_CLASSES,
    description="IEMOCAP four-way speech-emotion protocol with excited merged into "
                "happy (the standard 5,531-utterance subset).",
    limitation=(
        "The happy class is the union of IEMOCAP's happy and excited annotations. "
        "That merge is the standard protocol, but it means per-class happy scores are "
        "not comparable with a space that keeps the two apart, such as iemocap_erc6."
    ),
)

MOSEI_SENTIMENT7 = LabelSpace(
    name="mosei_sentiment7",
    classes=MOSEI_SENTIMENT7_CLASSES,
    description="CMU-MOSEI seven-point sentiment buckets, the Acc7 protocol: "
                "class id = round(sentiment_score) + 3 for a score in [-3, 3].",
    limitation=(
        "These are ordinal sentiment buckets, not categorical emotions. Macro-F1 "
        "over them treats adjacent buckets as unrelated classes, which understates "
        "a model that is consistently one bucket out; report Acc7, Acc2 and MAE "
        "alongside it rather than in place of it."
    ),
)

MOSEI_EMOTION6 = LabelSpace(
    name="mosei_emotion6",
    classes=MOSEI_EMOTION6_CLASSES,
    description="CMU-MOSEI six-way multi-label emotion annotations "
                "(the LDDU / CARAT / TAILOR protocol).",
    limitation=(
        "Multi-label and non-exclusive: a sample may carry several of these at once, "
        "or none. Accuracy over this space is the exact-set-match accuracy those "
        "papers report, not per-sample top-1 accuracy, and the two differ sharply."
    ),
)

WESAD_STATE_3CLASS = LabelSpace(
    name="wesad_state_3class",
    classes=WESAD_STATE_CLASSES,
    description="WESAD study conditions (baseline / stress / amusement).",
    limitation=(
        "WESAD carries physiological-state annotations, not categorical emotion: every "
        "WESAD record in the standardized metadata has target_type='physiological_state' "
        "and canonical_emotion_valid=False. Forcing the 7-class emotion taxonomy onto it "
        "would fabricate labels, so the physiology baseline reports this task instead."
    ),
)

WESAD_STATE_4CLASS = LabelSpace(
    name="wesad_state_4class",
    classes=WESAD_STATE_CLASSES_EXTENDED,
    description="WESAD study conditions including the meditation condition.",
    limitation=WESAD_STATE_3CLASS.limitation,
)


LABEL_SPACES: Mapping[str, LabelSpace] = MappingProxyType({
    space.name: space
    for space in (
        EMOTION_7CLASS,
        IEMOCAP_ERC6,
        IEMOCAP_SER4,
        MOSEI_SENTIMENT7,
        MOSEI_EMOTION6,
        WESAD_STATE_3CLASS,
        WESAD_STATE_4CLASS,
    )
})


def get_label_space(name: str) -> LabelSpace:
    try:
        return LABEL_SPACES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown label space {name!r}; expected one of {sorted(LABEL_SPACES)}"
        ) from error


# ============================================================
# Convenience aliases used across the training stack
# ============================================================

#: Ordered class names of the default baseline task.
CLASS_NAMES: tuple[str, ...] = EMOTION_7CLASS.classes

#: Class name to canonical identifier for the default baseline task.
CANONICAL_EMOTION_IDS: Mapping[str, int] = EMOTION_7CLASS.mapping

NUM_CANONICAL_EMOTIONS: int = EMOTION_7CLASS.num_classes


def class_names(num_classes: int | None = None) -> tuple[str, ...]:
    """Return the canonical emotion order, optionally truncated.

    Truncation exists only for bounded unit tests that exercise a smaller head;
    experiments always use the full space.
    """
    if num_classes is None:
        return CANONICAL_EMOTION_CLASSES
    if not 1 <= num_classes <= len(CANONICAL_EMOTION_CLASSES):
        raise ValueError(
            f"num_classes must be in 1..{len(CANONICAL_EMOTION_CLASSES)}, got {num_classes}"
        )
    return CANONICAL_EMOTION_CLASSES[:num_classes]
