"""Tests for the centralised label spaces.

The canonical emotion order is a scientific contract shared by the confusion
matrix, the per-class arrays, the class-weight vector, and the logs.  These
tests exist because the order was previously re-derived in several places and
an alphabetical sort silently rotated every one of those artefacts.
"""

import pytest

from src.common.labels import (
    CANONICAL_EMOTION_CLASSES,
    CLASS_NAMES,
    EMOTION_7CLASS,
    LABEL_SPACES,
    WESAD_STATE_3CLASS,
    WESAD_STATE_4CLASS,
    LabelSpace,
    class_names,
    get_label_space,
)
from src.preprocessing.standardization.targets import CANONICAL_EMOTIONS
from src.training.classification_evaluator import CLASS_NAMES as EVALUATOR_CLASS_NAMES


EXPECTED = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")


# ============================================================
# Canonical ordering
# ============================================================

def test_canonical_emotion_order_is_exactly_the_declared_one():
    assert CANONICAL_EMOTION_CLASSES == EXPECTED
    assert CLASS_NAMES == EXPECTED
    assert EMOTION_7CLASS.classes == EXPECTED


def test_canonical_order_is_not_alphabetical():
    # The previously observed failure: ['angry', 'disgust', 'fear', ...].
    assert list(CANONICAL_EMOTION_CLASSES) != sorted(CANONICAL_EMOTION_CLASSES)
    assert CANONICAL_EMOTION_CLASSES[0] == "neutral"
    assert CANONICAL_EMOTION_CLASSES[-1] == "surprise"


def test_ids_follow_declaration_order():
    for index, name in enumerate(EXPECTED):
        assert EMOTION_7CLASS.id_of(name) == index
        assert EMOTION_7CLASS.name_of(index) == name
    assert EMOTION_7CLASS.valid_ids == (0, 1, 2, 3, 4, 5, 6)


def test_standardization_targets_share_the_single_definition():
    assert CANONICAL_EMOTIONS == dict(EMOTION_7CLASS.mapping)
    assert list(CANONICAL_EMOTIONS) == list(EXPECTED)


def test_evaluator_class_names_follow_the_canonical_order():
    assert [EVALUATOR_CLASS_NAMES[index] for index in range(7)] == list(EXPECTED)


def test_class_names_helper_truncates_without_reordering():
    assert class_names() == EXPECTED
    assert class_names(3) == ("neutral", "happy", "sad")
    with pytest.raises(ValueError):
        class_names(0)
    with pytest.raises(ValueError):
        class_names(8)


# ============================================================
# Label space behaviour
# ============================================================

def test_unknown_class_and_id_are_rejected_with_the_expected_order():
    with pytest.raises(ValueError, match="neutral"):
        EMOTION_7CLASS.id_of("contempt")
    with pytest.raises(ValueError):
        EMOTION_7CLASS.name_of(7)


def test_duplicate_or_empty_class_lists_are_rejected():
    with pytest.raises(ValueError):
        LabelSpace("broken", ())
    with pytest.raises(ValueError):
        LabelSpace("broken", ("a", "a"))


def test_wesad_spaces_declare_the_limitation_instead_of_forcing_emotions():
    assert WESAD_STATE_3CLASS.classes == ("baseline", "stress", "amusement")
    assert WESAD_STATE_3CLASS.num_classes == 3
    assert WESAD_STATE_4CLASS.classes[:3] == WESAD_STATE_3CLASS.classes
    assert "physiological_state" in WESAD_STATE_3CLASS.limitation
    # No WESAD class may collide with the emotion taxonomy.
    assert not set(WESAD_STATE_4CLASS.classes) & set(CANONICAL_EMOTION_CLASSES)


def test_registry_lookup():
    assert get_label_space("emotion_7class") is EMOTION_7CLASS
    # Exact, not a subset check: an accidentally registered space is a real
    # mistake worth catching, so adding one is meant to require editing this
    # line. The four protocol spaces below were added for the HSEN phase, where
    # the canonical seven-class taxonomy is not the task being reported.
    assert set(LABEL_SPACES) == {
        "emotion_7class",
        "iemocap_erc6",
        "iemocap_ser4",
        "mosei_sentiment7",
        "mosei_emotion6",
        "wesad_state_3class",
        "wesad_state_4class",
    }
    with pytest.raises(ValueError, match="Unknown label space"):
        get_label_space("nope")


def test_to_dict_preserves_order():
    payload = EMOTION_7CLASS.to_dict()
    assert payload["classes"] == list(EXPECTED)
    assert payload["num_classes"] == 7
    assert list(payload["mapping"]) == list(EXPECTED)
