"""Canonical emotion mapping: what is mapped, what is refused, and why."""

from __future__ import annotations

import pytest

from src.ahsef.llm.mapping import (
    AMBIGUOUS,
    CANONICAL_EMOTIONS,
    LOOKUP,
    EmotionMappingError,
    classify_emotion_word,
    emotion_id,
    map_emotion,
    map_score_keys,
    mapping_provenance,
)


# ------------------------------------------------------------------ mapping

@pytest.mark.parametrize("raw,expected", [
    ("neutral", "neutral"), ("happy", "happy"), ("sad", "sad"), ("angry", "angry"),
    ("fear", "fear"), ("disgust", "disgust"), ("surprise", "surprise"),
])
def test_every_canonical_name_maps_to_itself(raw, expected):
    assert map_emotion(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("joy", "happy"), ("joyful", "happy"), ("delighted", "happy"),
    ("sadness", "sad"), ("sorrow", "sad"),
    ("anger", "angry"), ("furious", "angry"),
    ("fearful", "fear"), ("afraid", "fear"), ("terrified", "fear"),
    ("disgusted", "disgust"), ("revolted", "disgust"),
    ("surprised", "surprise"), ("astonished", "surprise"),
    ("calm", "neutral"), ("indifferent", "neutral"),
])
def test_documented_synonyms_map_as_specified(raw, expected):
    assert map_emotion(raw) == expected


@pytest.mark.parametrize("raw", ["HAPPY", "  Happy  ", "Joy", "JOYFUL", "no_emotion",
                                 "no-emotion", "no emotion", "sadness."])
def test_case_whitespace_punctuation_and_separators_are_normalised(raw):
    assert classify_emotion_word(raw).ok


def test_accents_are_stripped():
    assert map_emotion("colère".replace("colère", "angry")) == "angry"
    assert classify_emotion_word("jóy").canonical == "happy"


def test_emotion_id_uses_the_declared_class_order():
    assert emotion_id("neutral") == 0
    assert emotion_id("joy") == 1
    assert emotion_id("surprised") == 6
    assert [emotion_id(name) for name in CANONICAL_EMOTIONS] == list(range(7))


# --------------------------------------------------------------- refusals

@pytest.mark.parametrize("raw", ["excited", "anxious", "shocked", "contempt",
                                 "frustrated", "confused", "bored", "mixed"])
def test_contested_words_are_refused_by_name_not_guessed(raw):
    result = classify_emotion_word(raw)
    assert result.kind == "ambiguous"
    assert result.canonical is None
    assert result.reason and len(result.reason) > 10


def test_an_unknown_word_is_unknown_not_neutral():
    """The dangerous failure would be folding junk into a real class."""
    result = classify_emotion_word("banana")
    assert result.kind == "unknown"
    assert result.canonical is None


def test_a_multi_label_answer_is_refused_with_that_reason():
    result = classify_emotion_word("happy, sad")
    assert result.kind == "unknown"
    assert "more than one emotion" in result.reason


def test_empty_and_missing_answers_are_refused():
    assert classify_emotion_word("").kind == "unknown"
    assert classify_emotion_word(None).kind == "unknown"
    assert classify_emotion_word("   ").kind == "unknown"


def test_explicit_abstention_is_its_own_outcome():
    result = classify_emotion_word("abstain")
    assert result.kind == "abstain"
    assert result.canonical is None


def test_map_emotion_raises_with_the_kind_attached():
    with pytest.raises(EmotionMappingError) as info:
        map_emotion("excited")
    assert info.value.kind == "ambiguous"
    with pytest.raises(EmotionMappingError) as info:
        map_emotion("banana")
    assert info.value.kind == "unknown"


# ----------------------------------------------------------------- tables

def test_no_word_is_both_mappable_and_refused():
    assert not set(LOOKUP) & set(AMBIGUOUS)


def test_every_canonical_class_has_synonyms():
    assert set(LOOKUP.values()) == set(CANONICAL_EMOTIONS)


def test_provenance_carries_the_whole_table():
    record = mapping_provenance()
    assert record["canonical_classes"] == list(CANONICAL_EMOTIONS)
    assert set(record["synonyms"]) == set(CANONICAL_EMOTIONS)
    assert "excited" in record["ambiguous_refused"]


# ----------------------------------------------------------- score mapping

def test_score_keys_are_mapped_and_ordered_canonically():
    scores = {"calm": 0.1, "joy": 0.5, "sadness": 0.1, "anger": 0.1,
              "fear": 0.1, "disgust": 0.05, "surprise": 0.05}
    mapped = map_score_keys(scores)
    assert list(mapped) == list(CANONICAL_EMOTIONS)
    assert mapped["happy"] == pytest.approx(0.5)
    assert mapped["neutral"] == pytest.approx(0.1)


def test_a_missing_class_in_the_scores_is_an_error_not_a_zero_fill():
    """A zero-filled class would read as a confident denial the model never made."""
    with pytest.raises(EmotionMappingError, match="no score for"):
        map_score_keys({"happy": 1.0})


def test_two_keys_mapping_to_one_class_is_refused():
    scores = {name: 0.1 for name in CANONICAL_EMOTIONS}
    scores["joy"] = 0.2  # also maps to happy
    with pytest.raises(EmotionMappingError, match="both map to"):
        map_score_keys(scores)


def test_a_negative_score_is_refused():
    scores = {name: 0.1 for name in CANONICAL_EMOTIONS}
    scores["happy"] = -1.0
    with pytest.raises(EmotionMappingError, match="negative"):
        map_score_keys(scores)


def test_an_unmappable_score_key_is_refused():
    scores = {name: 0.1 for name in CANONICAL_EMOTIONS}
    scores["excited"] = 0.3
    with pytest.raises(EmotionMappingError):
        map_score_keys(scores)
