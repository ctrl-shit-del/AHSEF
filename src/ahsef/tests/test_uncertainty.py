"""Entropy, normalised entropy, confidence, margin, and probability normalisation."""

from __future__ import annotations

import math

import pytest
import torch

from src.ahsef.uncertainty import (
    confidence,
    max_entropy,
    normalize_probabilities,
    normalized_entropy,
    predicted_class,
    predictive_entropy,
    probabilities_from_logits,
    probability_margin,
    uncertainty_columns,
    uncertainty_record,
)


def uniform(num_classes: int = 7) -> torch.Tensor:
    return torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float64)


def one_hot(index: int, num_classes: int = 7) -> torch.Tensor:
    values = torch.zeros(num_classes, dtype=torch.float64)
    values[index] = 1.0
    return values


# ---------------------------------------------------------------- entropy

def test_uniform_distribution_has_maximum_entropy():
    assert predictive_entropy(uniform(7)) == pytest.approx(math.log(7))
    assert normalized_entropy(uniform(7)) == pytest.approx(1.0)


def test_one_hot_distribution_has_zero_entropy():
    assert predictive_entropy(one_hot(3)) == pytest.approx(0.0, abs=1e-12)
    assert normalized_entropy(one_hot(3)) == pytest.approx(0.0, abs=1e-12)
    # A one-hot vector must not report -0.0 in a JSON artefact.
    assert math.copysign(1.0, float(normalized_entropy(one_hot(3)))) > 0


def test_normalized_entropy_is_bounded_for_random_distributions():
    torch.manual_seed(0)
    probabilities = torch.softmax(torch.randn(500, 7, dtype=torch.float64), dim=-1)
    values = normalized_entropy(probabilities)
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_normalisation_makes_class_counts_comparable():
    """A 3-class and a 7-class uniform posterior are both maximally uncertain."""
    assert normalized_entropy(uniform(3)) == pytest.approx(1.0)
    assert normalized_entropy(uniform(7)) == pytest.approx(1.0)
    # ...while the raw entropies are not comparable at all.
    assert predictive_entropy(uniform(3)) < predictive_entropy(uniform(7))


def test_entropy_is_monotone_in_sharpness():
    sharp = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64)
    flat = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    assert predictive_entropy(sharp) < predictive_entropy(flat)


def test_max_entropy_matches_log_k():
    assert max_entropy(7) == pytest.approx(math.log(7))
    with pytest.raises(ValueError):
        max_entropy(1)


# ------------------------------------------------- confidence and margin

def test_confidence_is_the_top_probability_not_the_entropy():
    probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    assert confidence(probabilities) == pytest.approx(0.5)
    assert predicted_class(probabilities) == 0
    # The two quantities are genuinely different numbers.
    assert float(confidence(probabilities)) != pytest.approx(
        float(normalized_entropy(probabilities))
    )


def test_margin_is_top1_minus_top2():
    probabilities = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    assert probability_margin(probabilities) == pytest.approx(0.2)
    assert probability_margin(uniform(7)) == pytest.approx(0.0)
    assert probability_margin(one_hot(0)) == pytest.approx(1.0)


def test_batched_columns_match_the_single_sample_record():
    torch.manual_seed(1)
    probabilities = torch.softmax(torch.randn(8, 7, dtype=torch.float64), dim=-1)
    columns = uncertainty_columns(probabilities)
    for index in range(8):
        record = uncertainty_record(probabilities[index])
        assert float(columns["confidence"][index]) == pytest.approx(record["confidence"])
        assert float(columns["normalized_entropy"][index]) == pytest.approx(
            record["normalized_entropy"]
        )
        assert int(columns["predicted_class"][index]) == record["predicted_class"]


# ------------------------------------------------------------ validation

def test_unnormalised_input_is_rejected_rather_than_silently_renormalised():
    with pytest.raises(ValueError, match="sum to 1"):
        predictive_entropy(torch.tensor([0.5, 0.3], dtype=torch.float64))


def test_negative_probabilities_are_rejected():
    with pytest.raises(ValueError):
        predictive_entropy(torch.tensor([1.5, -0.5], dtype=torch.float64))


def test_single_class_distribution_is_rejected():
    with pytest.raises(ValueError, match="at least two classes"):
        predictive_entropy(torch.tensor([1.0], dtype=torch.float64))


# --------------------------------------------------------- normalisation

def test_normalize_probabilities_sums_to_one():
    scores = torch.tensor([[2.0, 1.0, 1.0], [0.5, 0.5, 0.0]], dtype=torch.float64)
    normalised = normalize_probabilities(scores)
    assert torch.allclose(normalised.sum(dim=-1), torch.ones(2, dtype=torch.float64))
    assert normalised[0, 0] == pytest.approx(0.5)


def test_normalize_probabilities_rejects_a_zero_row():
    with pytest.raises(ValueError, match="sums to zero"):
        normalize_probabilities(torch.zeros(1, 3, dtype=torch.float64))


def test_normalize_probabilities_rejects_negative_scores():
    with pytest.raises(ValueError, match="negative"):
        normalize_probabilities(torch.tensor([[1.0, -1.0]], dtype=torch.float64))


def test_normalising_an_already_normalised_vector_is_a_no_op():
    probabilities = torch.softmax(torch.randn(4, 7, dtype=torch.float64), dim=-1)
    assert torch.allclose(normalize_probabilities(probabilities), probabilities)


# ----------------------------------------------------------- temperature

def test_higher_temperature_raises_uncertainty():
    logits = torch.tensor([[3.0, 1.0, 0.0]], dtype=torch.float64)
    cold = normalized_entropy(probabilities_from_logits(logits, 0.5))
    warm = normalized_entropy(probabilities_from_logits(logits, 2.0))
    assert float(cold) < float(warm)


def test_temperature_does_not_change_the_argmax():
    torch.manual_seed(2)
    logits = torch.randn(32, 7, dtype=torch.float64)
    base = probabilities_from_logits(logits, 1.0).argmax(dim=-1)
    for temperature in (0.5, 2.0, 5.0):
        scaled = probabilities_from_logits(logits, temperature).argmax(dim=-1)
        assert torch.equal(base, scaled)


def test_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        probabilities_from_logits(torch.zeros(1, 3), temperature=0.0)
