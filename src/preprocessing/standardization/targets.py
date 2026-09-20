from typing import Optional

from src.common.labels import EMOTION_7CLASS


# The canonical class order lives in ``src.common.labels`` so that the
# metadata layer, the training stack, and every artefact index the same
# positions.  This dict is the standardization-layer view of that one
# definition; it is never re-sorted.
CANONICAL_EMOTIONS = dict(EMOTION_7CLASS.mapping)


# Dataset-specific mappings into the canonical 7-class
# categorical emotion space.
#
# None means that the original annotation should not be
# treated as one of the canonical 7 emotion classes.

DATASET_EMOTION_MAPPING = {

    "RAVDESS": {
        "neutral": "neutral",
        "calm": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
    },

    "CREMA-D": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
    },

    "IEMOCAP": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",

        # Preserve these as dataset-specific emotions.
        "excited": None,
        "frustrated": None,
        "other": None,
    },

    "MELD": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
    },

    "FERPlus": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
    },

    "RAF-DB": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",
    },

    "AffectNet+": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",

        # Dataset-specific / invalid for the 7-class target.
        "contempt": None,
        "none": None,
        "uncertain": None,
        "non_face": None,
    },

    # MSP-Podcast ships consensus categorical emotion labels
    # (labels_consensus.csv, EmoClass) that already live in the
    # canonical taxonomy.  Codes outside it stay dataset-specific.
    "MSP-Podcast": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "surprise": "surprise",

        # Dataset-specific / invalid for the 7-class target.
        "contempt": None,
        "other": None,
    },

    # CMU-MOSEI is primarily a sentiment dataset.
    # Do NOT convert positive/negative into happy/sad.
    "CMU-MOSEI": {
        "positive": None,
        "neutral": None,
        "negative": None,
    },

    # WESAD uses physiological-state labels rather than
    # the canonical categorical emotion taxonomy.
    "WESAD": {},
}


def get_canonical_emotion(
    dataset: str,
    emotion: Optional[str],
) -> Optional[str]:

    if emotion is None:
        return None

    emotion = str(emotion).strip().lower()

    mapping = DATASET_EMOTION_MAPPING.get(
        dataset,
        {},
    )

    return mapping.get(emotion)


def get_canonical_emotion_id(
    canonical_emotion: Optional[str],
) -> Optional[int]:

    if canonical_emotion is None:
        return None

    return CANONICAL_EMOTIONS.get(
        canonical_emotion
    )


def is_canonical_emotion_valid(
    canonical_emotion: Optional[str],
) -> bool:

    return canonical_emotion in CANONICAL_EMOTIONS