"""LLM uncertainty: normalisation, entropy, votes, and the policy boundary."""

from __future__ import annotations

import math

import pytest
import torch

from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm.schema import parse_response
from src.ahsef.llm.uncertainty import (
    LLM_UNCERTAINTY_DEFINITION,
    UNCERTAINTY_POLICIES,
    LLMUncertainty,
    UncertaintyPolicyError,
    majority_vote,
    score_distribution,
    uncertainty_from_response,
    vote_statistics,
)


def body(**overrides) -> dict:
    record = {
        "emotion": "happy",
        "class_scores": {name: 0.1 for name in CANONICAL_EMOTIONS},
        "confidence": 0.8, "evidence_strength": 0.6, "ambiguity": 0.25,
        "evidence": "cue", "abstain": False,
    }
    record.update(overrides)
    return record


def peaked(name: str = "happy", mass: float = 0.7) -> dict:
    rest = (1.0 - mass) / 6
    return {item: (mass if item == name else rest) for item in CANONICAL_EMOTIONS}


# ------------------------------------------------------- score distribution

def test_scores_are_normalised_into_the_canonical_order():
    distribution = score_distribution({name: 2.0 for name in CANONICAL_EMOTIONS})
    assert torch.allclose(distribution.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert distribution.shape == (7,)
    assert distribution[0] == pytest.approx(1 / 7)


def test_unnormalised_scores_are_accepted_and_rescaled():
    """The model is told 'larger is better', not 'must sum to one'."""
    scores = {name: 0.0 for name in CANONICAL_EMOTIONS}
    scores["happy"], scores["sad"] = 8.0, 2.0
    distribution = score_distribution(scores)
    assert distribution[1] == pytest.approx(0.8)
    assert distribution[2] == pytest.approx(0.2)


def test_a_missing_class_is_refused():
    with pytest.raises(ValueError, match="missing canonical classes"):
        score_distribution({"happy": 1.0})


# ------------------------------------------------------------------ entropy

def test_uniform_scores_give_maximal_normalised_entropy():
    record = uncertainty_from_response(parse_response(body()))
    assert record.llm_normalized_score_entropy == pytest.approx(1.0)
    assert record.llm_score_entropy == pytest.approx(math.log(7))


def test_a_peaked_score_vector_gives_low_entropy():
    record = uncertainty_from_response(parse_response(body(class_scores=peaked(mass=0.94))))
    assert record.llm_normalized_score_entropy < 0.3
    assert record.llm_score_margin > 0.8


def test_top1_and_top2_are_reported():
    scores = {name: 0.01 for name in CANONICAL_EMOTIONS}
    scores["happy"], scores["surprise"] = 0.6, 0.3
    record = uncertainty_from_response(parse_response(body(class_scores=scores)))
    assert record.llm_score_top1 == "happy"
    assert record.llm_score_top2 == "surprise"


def test_score_top1_is_one_minus_the_max_normalised_score():
    """The specification's primary measure, over self-reported scores."""
    record = uncertainty_from_response(parse_response(body(class_scores=peaked(mass=0.7))))
    assert record.llm_score_top1_prob == pytest.approx(0.7)
    assert record.llm_score_top1_uncertainty == pytest.approx(0.3)
    assert record.under_policy("score_top1") == pytest.approx(0.3)


def test_score_top1_is_distinct_from_entropy_and_from_self_reported_confidence():
    record = uncertainty_from_response(
        parse_response(body(confidence=0.8, class_scores=peaked(mass=0.7)))
    )
    assert record.under_policy("score_top1") == pytest.approx(0.3)
    assert record.under_policy("llm_confidence") == pytest.approx(0.2)
    assert record.under_policy("score_entropy") != pytest.approx(0.3)


def test_score_top1_on_a_uniform_score_vector_is_the_maximum():
    record = uncertainty_from_response(parse_response(body()))
    assert record.llm_score_top1_uncertainty == pytest.approx(1 - 1 / 7)


def test_score_top1_is_unavailable_without_scores():
    record = uncertainty_from_response(parse_response(body(class_scores=None)))
    assert record.llm_score_top1_uncertainty is None
    with pytest.raises(UncertaintyPolicyError):
        record.under_policy("score_top1")


def test_the_top1_class_name_and_its_value_are_separate_fields():
    record = uncertainty_from_response(parse_response(body(class_scores=peaked("sad", 0.6))))
    assert record.llm_score_top1 == "sad"
    assert isinstance(record.llm_score_top1_prob, float)


def test_llm_uncertainty_is_one_minus_confidence_not_entropy():
    """The two are different quantities and must not coincide by construction."""
    record = uncertainty_from_response(parse_response(body(confidence=0.8)))
    assert record.llm_uncertainty == pytest.approx(0.2)
    assert record.llm_normalized_score_entropy == pytest.approx(1.0)
    assert record.llm_uncertainty != record.llm_normalized_score_entropy


def test_absent_scores_leave_entropy_none_rather_than_defaulted():
    record = uncertainty_from_response(parse_response(body(class_scores=None)))
    assert record.llm_normalized_score_entropy is None
    assert record.llm_score_entropy is None
    assert record.llm_uncertainty == pytest.approx(0.2)


# -------------------------------------------------------------------- votes

def test_unanimous_votes_have_zero_disagreement():
    disagreement, entropy, agreement = vote_statistics(["happy"] * 5)
    assert disagreement == pytest.approx(0.0)
    assert entropy == pytest.approx(0.0)
    assert agreement == pytest.approx(1.0)


def test_split_votes_raise_disagreement():
    disagreement, entropy, agreement = vote_statistics(["happy", "sad", "happy", "angry"])
    assert disagreement == pytest.approx(0.5)
    assert agreement == pytest.approx(0.5)
    assert entropy > 0.0


def test_vote_entropy_is_normalised_over_the_canonical_space_not_the_votes_seen():
    """Otherwise two-way and seven-way splits would both read as 1.0."""
    two_way = vote_statistics(["happy", "sad"])[1]
    seven_way = vote_statistics(list(CANONICAL_EMOTIONS))[1]
    assert two_way < seven_way
    assert seven_way == pytest.approx(1.0)


def test_majority_vote_breaks_ties_by_declared_class_order_not_randomly():
    assert majority_vote(["sad", "happy"]) == "happy"      # happy precedes sad
    assert majority_vote(["surprise", "neutral"]) == "neutral"
    assert majority_vote(["sad", "sad", "happy"]) == "sad"


def test_repeats_travel_with_the_record():
    record = uncertainty_from_response(
        parse_response(body()), votes=["happy", "happy", "sad"]
    )
    assert record.repeats == 3
    assert record.llm_vote_entropy is not None
    assert record.llm_vote_agreement == pytest.approx(2 / 3)


def test_a_single_sample_reports_no_empirical_uncertainty():
    """Zero disagreement from one draw would be a claim the measurement cannot make."""
    record = uncertainty_from_response(parse_response(body()))
    assert record.llm_vote_entropy is None
    assert record.repeats == 1


# ----------------------------------------------------------------- policies

def test_each_policy_selects_its_own_signal():
    record = uncertainty_from_response(
        parse_response(body(confidence=0.75, ambiguity=0.4, class_scores=peaked())),
        votes=["happy", "happy", "sad"],
    )
    assert record.under_policy("llm_confidence") == pytest.approx(0.25)
    assert record.under_policy("ambiguity") == pytest.approx(0.4)
    assert record.under_policy("score_entropy") == pytest.approx(
        record.llm_normalized_score_entropy
    )
    assert record.under_policy("empirical_votes") == pytest.approx(record.llm_vote_entropy)


def test_max_of_available_is_the_most_pessimistic_signal():
    record = uncertainty_from_response(
        parse_response(body(confidence=0.9, ambiguity=0.6, class_scores=peaked(mass=0.95)))
    )
    assert record.under_policy("max_of_available") == pytest.approx(0.6)


def test_a_policy_with_no_signal_raises_rather_than_substituting():
    record = uncertainty_from_response(parse_response(body(class_scores=None)))
    with pytest.raises(UncertaintyPolicyError, match="Refusing to substitute"):
        record.under_policy("score_entropy")


def test_max_of_available_with_nothing_available_raises():
    empty = LLMUncertainty(None, None, None, None, None, None, None, None, None)
    with pytest.raises(UncertaintyPolicyError):
        empty.under_policy("max_of_available")


def test_an_unknown_policy_is_refused():
    record = uncertainty_from_response(parse_response(body()))
    with pytest.raises(ValueError, match="policy must be one of"):
        record.under_policy("vibes")


def test_available_policies_reflect_what_the_response_carries():
    with_scores = uncertainty_from_response(parse_response(body()))
    assert "score_entropy" in with_scores.available_policies()
    without = uncertainty_from_response(parse_response(body(class_scores=None)))
    assert "score_entropy" not in without.available_policies()
    assert "llm_confidence" in without.available_policies()


# --------------------------------------------------------------- semantics

def test_the_definition_record_denies_being_a_posterior():
    assert LLM_UNCERTAINTY_DEFINITION["calibrated"] is False
    disclaimer = LLM_UNCERTAINTY_DEFINITION["not_a_posterior"].lower()
    assert "no quantity in this module is a posterior" in disclaimer
    assert "NOT predictive entropy" in LLM_UNCERTAINTY_DEFINITION["llm_uncertainty"]
    assert set(LLM_UNCERTAINTY_DEFINITION["policies"]) == set(UNCERTAINTY_POLICIES)


def test_non_canonical_votes_are_refused():
    with pytest.raises(ValueError, match="non-canonical labels"):
        vote_statistics(["happy", "excited"])


def test_every_llm_field_name_is_prefixed():
    record = uncertainty_from_response(parse_response(body())).to_dict()
    for key in record:
        assert key.startswith("llm_") or key == "repeats"
