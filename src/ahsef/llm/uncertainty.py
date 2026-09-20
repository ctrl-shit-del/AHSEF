"""Uncertainty for an LLM modality, kept separate from posterior uncertainty.

Stage 1 established one distinction (confidence vs. normalised entropy).  The
LLM forces a second one, and it matters more:

**A statistical baseline's softmax is a posterior over the label space.  An
LLM's self-report is a claim about itself.**  Both land in ``[0, 1]``, both can
be plotted on the same axis, and treating them as the same quantity would be
the single easiest way to produce a wrong result in this project.

So every LLM-derived quantity carries an ``llm_`` prefix and is defined here,
next to a statement of what it is not:

``llm_confidence``
    The model's self-reported backing for its chosen label.  Not a posterior,
    not calibrated until measured against outcomes.

``llm_uncertainty``
    ``1 - llm_confidence``.  A rescaled self-report, **not** predictive entropy.

``llm_score_entropy`` / ``llm_normalized_score_entropy``
    Shannon entropy of the *normalised self-reported scores*.  Arithmetically
    identical to Stage 1's entropy, semantically not the same object: it
    measures the spread of a judgement the model wrote down, not of a
    posterior it computed.  Reported under its own name for that reason.

``llm_sample_disagreement`` / ``llm_vote_entropy``
    Empirical uncertainty from repeated sampling of the same input -- the one
    quantity here that is a measurement rather than a self-report.

:func:`combined_uncertainty` picks one of these as *the* routing signal
according to an explicitly named policy, and every artefact records which
policy produced the number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm.schema import LLMEmotionResponse
from src.ahsef.uncertainty import (
    normalize_probabilities,
    normalized_entropy,
    predictive_entropy,
    probability_margin,
)


#: How the single routing uncertainty is derived.  Configurable, and recorded.
UNCERTAINTY_POLICIES = (
    "llm_confidence",       # 1 - self-reported confidence
    "score_top1",           # 1 - max_c score_c, over the normalised scores
    "score_entropy",        # normalised entropy of self-reported class scores
    "ambiguity",            # the model's self-reported ambiguity field
    "empirical_votes",      # normalised entropy of repeated-sample votes
    "max_of_available",     # the most pessimistic available signal
)

DEFAULT_UNCERTAINTY_POLICY = "score_entropy"


class UncertaintyPolicyError(ValueError):
    """Raised when a policy is asked for a signal the response does not carry."""


def score_distribution(scores: Mapping[str, float]) -> torch.Tensor:
    """Normalise self-reported scores into a distribution over the canonical order.

    The result is a *score distribution*, not a posterior.  Callers that write
    it to disk must name it accordingly.
    """
    missing = [name for name in CANONICAL_EMOTIONS if name not in scores]
    if missing:
        raise ValueError(f"Score dictionary is missing canonical classes {missing}")
    vector = torch.tensor(
        [float(scores[name]) for name in CANONICAL_EMOTIONS], dtype=torch.float64
    )
    return normalize_probabilities(vector)


@dataclass(frozen=True)
class LLMUncertainty:
    """Every uncertainty-like quantity available for one LLM answer."""

    llm_confidence: float | None
    llm_uncertainty: float | None
    llm_ambiguity: float | None
    llm_evidence_strength: float | None
    llm_score_entropy: float | None
    llm_normalized_score_entropy: float | None
    llm_score_margin: float | None
    #: The *name* of the best- and second-best-scoring class.
    llm_score_top1: str | None
    llm_score_top2: str | None
    #: The top-1 normalised score, and the specification's primary measure
    #: ``U(x) = 1 - max_c score_c`` derived from it.
    llm_score_top1_prob: float | None = None
    llm_score_top1_uncertainty: float | None = None
    llm_sample_disagreement: float | None = None
    llm_vote_entropy: float | None = None
    llm_vote_agreement: float | None = None
    repeats: int = 1

    def to_dict(self) -> dict:
        return {
            "llm_confidence": self.llm_confidence,
            "llm_uncertainty": self.llm_uncertainty,
            "llm_ambiguity": self.llm_ambiguity,
            "llm_evidence_strength": self.llm_evidence_strength,
            "llm_score_entropy": self.llm_score_entropy,
            "llm_normalized_score_entropy": self.llm_normalized_score_entropy,
            "llm_score_margin": self.llm_score_margin,
            "llm_score_top1": self.llm_score_top1,
            "llm_score_top2": self.llm_score_top2,
            "llm_score_top1_prob": self.llm_score_top1_prob,
            "llm_score_top1_uncertainty": self.llm_score_top1_uncertainty,
            "llm_sample_disagreement": self.llm_sample_disagreement,
            "llm_vote_entropy": self.llm_vote_entropy,
            "llm_vote_agreement": self.llm_vote_agreement,
            "repeats": self.repeats,
        }

    def available_policies(self) -> list[str]:
        available = []
        if self.llm_uncertainty is not None:
            available.append("llm_confidence")
        if self.llm_normalized_score_entropy is not None:
            available.append("score_entropy")
        if self.llm_score_top1_uncertainty is not None:
            available.append("score_top1")
        if self.llm_ambiguity is not None:
            available.append("ambiguity")
        if self.llm_vote_entropy is not None:
            available.append("empirical_votes")
        if available:
            available.append("max_of_available")
        return available

    def under_policy(self, policy: str) -> float:
        """The single routing uncertainty this policy selects, in ``[0, 1]``."""
        if policy not in UNCERTAINTY_POLICIES:
            raise ValueError(
                f"policy must be one of {UNCERTAINTY_POLICIES}, got {policy!r}"
            )
        values = {
            "llm_confidence": self.llm_uncertainty,
            "score_entropy": self.llm_normalized_score_entropy,
            "score_top1": self.llm_score_top1_uncertainty,
            "ambiguity": self.llm_ambiguity,
            "empirical_votes": self.llm_vote_entropy,
        }
        if policy == "max_of_available":
            present = [value for value in values.values() if value is not None]
            if not present:
                raise UncertaintyPolicyError(
                    "No uncertainty signal is available for this response; the router "
                    "cannot be given a fabricated one."
                )
            return float(max(present))
        value = values[policy]
        if value is None:
            raise UncertaintyPolicyError(
                f"Policy {policy!r} needs a signal this response does not carry "
                f"(available: {self.available_policies()}). Refusing to substitute one."
            )
        return float(value)


def uncertainty_from_response(
    response: LLMEmotionResponse,
    votes: Sequence[str] | None = None,
) -> LLMUncertainty:
    """Derive every available uncertainty signal from one validated response.

    ``votes`` are the canonical labels produced by repeated sampling of the
    same input, including this response's own label.
    """
    confidence = response.llm_confidence
    uncertainty = None if confidence is None else 1.0 - float(confidence)

    entropy = normalised = margin = None
    top1 = top2 = None
    top1_value = top1_uncertainty = None
    if response.class_scores:
        distribution = score_distribution(response.class_scores)
        entropy = float(predictive_entropy(distribution))
        normalised = float(normalized_entropy(distribution))
        margin = float(probability_margin(distribution))
        order = distribution.argsort(descending=True)
        top1 = CANONICAL_EMOTIONS[int(order[0])]
        top2 = CANONICAL_EMOTIONS[int(order[1])]
        top1_value = float(distribution.max())
        # The specification's primary measure: U(x) = 1 - max_c P(c|x),
        # computed over the NORMALISED SELF-REPORTED scores, not a posterior.
        top1_uncertainty = 1.0 - top1_value

    disagreement = vote_entropy = agreement = None
    repeats = 1
    if votes:
        repeats = len(votes)
        disagreement, vote_entropy, agreement = vote_statistics(votes)

    return LLMUncertainty(
        llm_confidence=confidence,
        llm_uncertainty=uncertainty,
        llm_ambiguity=response.ambiguity,
        llm_evidence_strength=response.evidence_strength,
        llm_score_entropy=entropy,
        llm_normalized_score_entropy=normalised,
        llm_score_margin=margin,
        llm_score_top1=top1,
        llm_score_top2=top2,
        llm_score_top1_prob=top1_value,
        llm_score_top1_uncertainty=top1_uncertainty,
        llm_sample_disagreement=disagreement,
        llm_vote_entropy=vote_entropy,
        llm_vote_agreement=agreement,
        repeats=repeats,
    )


def vote_statistics(votes: Sequence[str]) -> tuple[float, float, float]:
    """``(disagreement, normalised vote entropy, modal agreement)`` over repeats.

    This is the only genuinely *empirical* uncertainty available for an LLM
    whose sampling parameters are not exposed: it measures how stable the
    answer is under the model's own nondeterminism.  With a single sample it is
    zero by construction, which is a statement about the measurement, not about
    the model's certainty -- hence ``repeats`` travels with it everywhere.
    """
    if not votes:
        raise ValueError("vote_statistics needs at least one vote")
    counts = Counter(str(vote) for vote in votes)
    unknown = sorted(set(counts) - set(CANONICAL_EMOTIONS))
    if unknown:
        raise ValueError(
            f"vote_statistics received non-canonical labels {unknown}; votes must "
            f"already have been mapped into the seven-class space."
        )
    total = sum(counts.values())
    modal = max(counts.values()) / total
    # The distribution spans the whole canonical space, with zeros for classes
    # nobody voted for. Building it over only the observed classes would make a
    # unanimous vote a one-element vector, and would normalise a two-way split
    # the same as a seven-way one.
    distribution = torch.tensor(
        [counts.get(name, 0) / total for name in CANONICAL_EMOTIONS], dtype=torch.float64
    )
    entropy = float(predictive_entropy(distribution))
    ceiling = float(torch.log(torch.tensor(float(len(CANONICAL_EMOTIONS)))))
    return 1.0 - modal, min(entropy / ceiling, 1.0), modal


def majority_vote(votes: Sequence[str]) -> str:
    """The modal label; ties break toward the canonical class order, not at random."""
    if not votes:
        raise ValueError("majority_vote needs at least one vote")
    counts = Counter(str(vote) for vote in votes)
    best = max(counts.values())
    tied = [name for name, count in counts.items() if count == best]
    ordered = [name for name in CANONICAL_EMOTIONS if name in tied]
    return ordered[0] if ordered else sorted(tied)[0]


#: Recorded in every artefact so a reader knows exactly what the numbers mean.
LLM_UNCERTAINTY_DEFINITION = {
    "llm_confidence": "model's self-reported backing for its chosen label, in [0, 1]",
    "llm_uncertainty": "1 - llm_confidence; a rescaled self-report, NOT predictive entropy",
    "llm_ambiguity": "model's self-reported fit of a competing label, in [0, 1]",
    "llm_evidence_strength": "model's self-reported amount of affective evidence in the text",
    "llm_score_top1_uncertainty":
        "U(x) = 1 - max_c score_c over the normalised self-reported scores. This is "
        "the specification's primary uncertainty measure. It is arithmetically the "
        "usual 1-minus-top-1 form, but the input is a self-report, not a posterior.",
    "llm_normalized_score_entropy":
        "Shannon entropy of the normalised self-reported class scores, divided by ln(7); "
        "arithmetically the Stage 1 formula, semantically the spread of a written-down "
        "judgement rather than of a computed posterior",
    "llm_vote_entropy":
        "normalised entropy of labels from repeated sampling of the same input; the only "
        "empirical (measured, not self-reported) uncertainty available here",
    "policies": list(UNCERTAINTY_POLICIES),
    "calibrated": False,
    "not_a_posterior": (
        "No quantity in this module is a posterior over the label space. The LLM does not "
        "expose token logprobs through this interface, so nothing here is comparable to a "
        "softmax from the frozen baselines except by measurement against outcomes."
    ),
}
