"""The LLM output contract: parsing, malformed responses, and missing probabilities."""

from __future__ import annotations

import json

import pytest

from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm.prompts import PROMPT_VERSIONS, get_prompt, prompt_provenance
from src.ahsef.llm.schema import (
    LLMEmotionResponse,
    parse_response,
    response_json_schema,
    schema_provenance,
)


def body(**overrides) -> dict:
    record = {
        "emotion": "happy",
        "class_scores": {"neutral": 0.05, "happy": 0.70, "sad": 0.05, "angry": 0.05,
                         "fear": 0.05, "disgust": 0.05, "surprise": 0.05},
        "confidence": 0.82,
        "evidence_strength": 0.7,
        "ambiguity": 0.2,
        "evidence": "'so glad to hear it'",
        "abstain": False,
    }
    record.update(overrides)
    return record


# ----------------------------------------------------------------- schema

def test_schema_pins_the_closed_label_space():
    schema = response_json_schema()
    assert schema["properties"]["emotion"]["enum"] == list(CANONICAL_EMOTIONS)
    assert schema["additionalProperties"] is False


def test_schema_requires_every_canonical_class_in_the_scores():
    scores = response_json_schema()["properties"]["class_scores"]
    assert scores["required"] == list(CANONICAL_EMOTIONS)
    assert scores["additionalProperties"] is False


def test_schema_provenance_states_scores_are_not_probabilities():
    """The disclaimer travels with the artefact, not only with the source code."""
    record = schema_provenance()
    assert "not calibrated probabilities" in record["score_semantics"].lower()
    assert "never repaired" in record["rejection_policy"].lower()


# ---------------------------------------------------------------- parsing

def test_a_well_formed_response_parses_and_maps():
    response = parse_response(body())
    assert response.usable
    assert response.emotion == "happy"
    assert response.class_id == 1
    assert response.llm_confidence == pytest.approx(0.82)
    assert response.has_scores


def test_json_text_and_decoded_objects_parse_identically():
    assert parse_response(json.dumps(body())).emotion == parse_response(body()).emotion


def test_a_synonym_is_mapped_during_parsing():
    response = parse_response(body(emotion="joy"))
    assert response.emotion == "happy"
    assert response.mapping.raw == "joy"


# ------------------------------------------------------- malformed input

def test_malformed_json_is_recorded_not_raised():
    response = parse_response("{not json at all")
    assert not response.usable
    assert response.error_kind == "malformed_json"
    assert response.class_id is None


def test_a_json_array_is_rejected():
    assert parse_response("[1, 2, 3]").error_kind == "malformed_json"


def test_a_missing_required_field_is_rejected():
    response = parse_response({"emotion": "happy"})
    assert response.error_kind == "missing_field"
    assert not response.usable


def test_an_out_of_range_confidence_is_rejected():
    assert parse_response(body(confidence=1.7)).error_kind == "out_of_range"
    assert parse_response(body(confidence=-0.1)).error_kind == "out_of_range"


def test_an_unmappable_emotion_is_rejected_and_keeps_the_raw_word():
    response = parse_response(body(emotion="excited"))
    assert not response.usable
    assert response.error_kind == "ambiguous"
    assert response.mapping.raw == "excited"


def test_a_hallucinated_class_never_becomes_a_prediction():
    response = parse_response(body(emotion="ecstatic_rage"))
    assert response.class_id is None
    assert response.error_kind == "unknown"


def test_parse_response_rejects_a_wrong_python_type():
    with pytest.raises(TypeError):
        parse_response(12345)


# --------------------------------------------------- missing probabilities

def test_absent_scores_still_yield_a_usable_label():
    """The label is the primary output; scores are a bonus, not a precondition."""
    response = parse_response(body(class_scores=None))
    assert response.usable
    assert response.emotion == "happy"
    assert not response.has_scores
    assert response.class_scores is None


def test_all_zero_scores_are_dropped_rather_than_normalised():
    zeros = {name: 0.0 for name in CANONICAL_EMOTIONS}
    response = parse_response(body(class_scores=zeros))
    assert response.usable
    assert not response.has_scores


def test_scores_with_a_bad_key_are_dropped_but_the_label_survives():
    broken = {name: 0.1 for name in CANONICAL_EMOTIONS}
    broken["excited"] = 0.5
    response = parse_response(body(class_scores=broken))
    assert response.usable
    assert not response.has_scores


def test_scores_of_the_wrong_type_are_dropped():
    assert not parse_response(body(class_scores=[0.1] * 7)).has_scores


def test_incomplete_scores_are_dropped_not_zero_filled():
    response = parse_response(body(class_scores={"happy": 1.0}))
    assert response.usable
    assert response.class_scores is None


# -------------------------------------------------------------- abstention

def test_abstention_is_recorded_as_such_and_is_not_a_prediction():
    response = parse_response(body(abstain=True))
    assert response.abstain
    assert not response.usable
    assert response.class_id is None
    assert response.error is None  # an abstention is not an error


def test_the_word_abstain_in_the_emotion_field_also_abstains():
    response = parse_response(body(emotion="abstain"))
    assert response.abstain
    assert not response.usable


# ------------------------------------------------------------ serialisation

def test_serialised_response_declares_scores_are_not_probabilities():
    record = parse_response(body()).to_dict()
    assert record["scores_are_calibrated_probabilities"] is False
    assert record["schema"].startswith("ahsef.llm.response")


def test_rejected_constructor_produces_an_unusable_response():
    response = LLMEmotionResponse.rejected("boom", "call_failed")
    assert not response.usable
    assert response.error_kind == "call_failed"


# ---------------------------------------------------------------- prompts

def test_prompt_is_versioned_and_hashed():
    prompt = get_prompt("v1")
    assert prompt.version == "v1"
    assert len(prompt.sha256) == 64


def test_prompt_renders_the_text_into_the_template():
    rendered = get_prompt("v1").render_user("I am delighted")
    assert "I am delighted" in rendered
    assert "{text}" not in rendered


def test_prompt_forbids_the_out_of_scope_inferences():
    system = get_prompt("v1").system.lower()
    for phrase in ["protected", "mental-health", "diagnose"]:
        assert phrase in system


def test_prompt_names_every_canonical_class():
    system = get_prompt("v1").system
    for name in CANONICAL_EMOTIONS:
        assert name in system


def test_unknown_prompt_version_is_refused():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_prompt("v99")


def test_prompt_provenance_can_omit_the_text_but_keeps_the_hash():
    record = prompt_provenance("v1", include_text=False)
    assert "system" not in record
    assert record["sha256"] == get_prompt("v1").sha256
    assert record["available_versions"] == sorted(PROMPT_VERSIONS)
