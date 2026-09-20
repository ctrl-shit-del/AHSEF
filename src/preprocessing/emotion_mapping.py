UNIFIED_EMOTIONS = {
    "neutral": "neutral",

    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",

    "sad": "sad",
    "sadness": "sad",

    "angry": "angry",
    "anger": "angry",

    "fear": "fear",
    "fearful": "fear",

    "disgust": "disgust",

    "surprise": "surprise",

    "contempt": "contempt",

    "excited": "excited",

    "frustrated": "frustrated",

    "stress": "stress",

    "calm": "calm",

    "other": "other",
}

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

INTENSITY = {
    "01": "normal",
    "02": "strong",
}

STATEMENT = {
    "01": "kids_are_talking",
    "02": "dogs_are_sitting",
}

REPETITION = {
    "01": 1,
    "02": 2,
}

CREMAD_EMOTIONS = {

    "ANG": "angry",

    "DIS": "disgust",

    "FEA": "fear",

    "HAP": "happy",

    "NEU": "neutral",

    "SAD": "sad",

}

RAFDB_EMOTIONS = {

    "1": "surprise",

    "2": "fear",

    "3": "disgust",

    "4": "happy",

    "5": "sad",

    "6": "angry",

    "7": "neutral",
}

AFFECTNET_EMOTIONS = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "angry",
    7: "contempt",
    8: None,
    9: None,
    10: None,
}

AFFECTNET_ANNOTATION_STATES = {
    8: "none",
    9: "uncertain",
    10: "non_face",
}

FERPLUS_EMOTIONS = {

    0: "neutral",

    1: "happy",

    2: "surprise",

    3: "sad",

    4: "angry",

    5: "disgust",

    6: "fear",
}

WESAD_EMOTIONS = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
    6: None,
    7: None,
}

WESAD_ANNOTATION_STATES = {
    0: "transition",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore",
    6: "ignore",
    7: "ignore",
}

IEMOCAP_EMOTIONS = {
    "ang": "angry",
    "hap": "happy",
    "exc": "excited",
    "sad": "sad",
    "neu": "neutral",
    "fru": "frustrated",
    "fea": "fear",
    "sur": "surprise",
    "dis": "disgust",
    "oth": "other",
    "xxx": None,
}

IEMOCAP_ANNOTATION_STATES = {
    "xxx": "unknown",
}

MELD_EMOTIONS = {

    "anger": "angry",

    "disgust": "disgust",

    "fear": "fear",

    "joy": "happy",

    "neutral": "neutral",

    "sadness": "sad",

    "surprise": "surprise",

}

MOSEI_EMOTIONS = {
    "Positive": "happy",
    "Neutral": "neutral",
    "Negative": "sad",
}

MOSEI_ANNOTATION_STATES = {
    "Positive": "positive_sentiment",
    "Neutral": "neutral_sentiment",
    "Negative": "negative_sentiment",
}

MSP_PODCAST_EMOTIONS = {
    "N": "neutral",
    "H": "happy",
    "A": "angry",
    "S": "sad",
    "U": "surprise",
    "F": "fear",
    "D": "disgust",
    "C": "contempt",
    "O": "other",
    "X": None,
}

MSP_PODCAST_ANNOTATION_STATES = {
    "X": "no_agreement",
}

DATASET_EMOTION_MAPPINGS = {
    "RAVDESS": RAVDESS_EMOTIONS,
    "CREMA-D": CREMAD_EMOTIONS,
    "RAF-DB": RAFDB_EMOTIONS,
    "FER-Plus": FERPLUS_EMOTIONS,
    "AffectNet+": AFFECTNET_EMOTIONS,
    "WESAD": WESAD_EMOTIONS,
    "IEMOCAP": IEMOCAP_EMOTIONS,
    "MELD": MELD_EMOTIONS,
    "CMU-MOSEI": MOSEI_EMOTIONS,
    "MSP-Podcast": MSP_PODCAST_EMOTIONS,
}