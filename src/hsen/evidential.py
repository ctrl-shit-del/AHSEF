"""Evidential (Dirichlet) classification: evidence, the uncertainty
decomposition, and the objective that produces them.

    head output  ->  evidence e = softplus(f)  ->  alpha = e + 1  ->  Dir(alpha)

A softmax returns a point on the simplex and nothing else.  Two inputs that
produce ``[0.5, 0.5]`` -- one because the model has seen a great deal of
conflicting evidence, one because it has seen nothing at all -- are
indistinguishable in its output, and no post-hoc transform of that output can
separate them.  Section 8's composite uncertainty needs them separated, because
UGAPR's answer differs: conflicting evidence is not improved by acquiring more
of the same modality, and an absence of evidence is.

The Dirichlet separates them by carrying total evidence mass ``S = sum(alpha)``
alongside the mean.  Both cases have the same mean; only the second has small
``S``.

What does the separating is the vacuity/dissonance SPLIT, not the composite
scalar.  Measured on the pair that matters -- ``alpha = [51]*6`` against
``alpha = [1]*6``, identical uniform means -- normalised entropy is 1.000 for
both and composite is 0.980 against 1.000, while vacuity is 0.020 against
1.000.  So the composite number is for ranking which predictions are unsafe;
vacuity is the one an acquisition policy reads, and UGAPR should read vacuity
rather than the composite.  Every component is reported separately for that
reason, and H4 is evaluated against each of them rather than against the
composite alone.

DEFINITIONS, all in [0, 1]

vacuity
    ``K / S``.  The mass of the Dirichlet that is not evidence.  One when the
    model has no evidence at all (alpha = 1, the uniform prior), approaching
    zero as evidence accumulates.  This is what "I have not seen enough" means
    and it is the quantity an acquisition policy should act on.

dissonance
    Jousang's measure of evidence that is present but mutually contradictory:
    two classes with large and comparable belief mass.  Zero when belief
    concentrates on one class, regardless of how much of it there is.  More
    data of the same kind does not reduce it.

composite
    ``vacuity + (1 - vacuity) * dissonance``.  Read as: all mass that is not
    evidence counts as uncertain, and of the mass that IS evidence, the
    conflicting share counts too.  Stays in [0, 1] by construction, reduces to
    vacuity when belief is unanimous and to dissonance when evidence is
    abundant.  Deliberately not a sum or a mean of the two, either of which can
    leave [0, 1] or double-count the vacuous part.

Entropy of the Dirichlet mean is also computed, because H4's claim is that the
composite measure beats entropy and a claim like that is only worth making
against the same model's own entropy rather than a different model's.

THE OBJECTIVE

Sensoy et al. (2018), the Bayes-risk form:

    L_i = sum_k (y_k - p_k)^2 + p_k (1 - p_k) / (S + 1)   +   lambda_t * KL

The first term is the expected sum of squares under the Dirichlet; the second
is its variance, which is what pushes ``S`` up only where the evidence
justifies it.  The KL term is computed against ``alpha_tilde = y + (1 - y) *
alpha`` -- the true class's evidence removed -- so it penalises evidence for
wrong classes and never punishes the model for being confident where it is
right.

``lambda_t`` anneals from 0 to 1 over the first ``kl_anneal_epochs``.  Applied
at full weight from the start it dominates before any evidence exists and
collapses the model onto the uniform prior; that is the standard failure of
this loss and the annealing is not optional.

The MSE form is used rather than the digamma/log form because it is the one
with bounded gradients at small ``S``, which is every sample for the first
epoch or two.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from src.hsen.labels.base import MISSING_CLASS_ID

#: Guards log and division where alpha can legitimately approach its lower
#: bound of 1 and evidence its lower bound of 0.
EPSILON = 1e-10


# ======================================================================
# Evidence
# ======================================================================

def evidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Non-negative evidence from an unconstrained head output.

    ``softplus`` rather than ``relu`` or ``exp``: relu kills the gradient for
    every class the model currently assigns no evidence, which is most of them
    early on, and exp overflows in float16 under autocast long before the
    evidence is meaningful.
    """
    return functional.softplus(logits)


def alpha_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Dirichlet parameters, ``alpha = evidence + 1``.

    The ``+1`` is the uniform Dirichlet prior, so ``alpha = 1`` everywhere is
    "no evidence" rather than a degenerate distribution.

    Monotone in the logits, which matters more than it looks: ``argmax(alpha)``
    equals ``argmax(logits)``, so every existing accuracy and F1 number is
    computed on the same predicted class whether or not the head is read
    evidentially.  Nothing in metrics.py had to change to keep reporting them.
    """
    return evidence_from_logits(logits) + 1.0


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    """``alpha / S`` -- the expected categorical under the Dirichlet."""
    return alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=EPSILON)


def belief(alpha: torch.Tensor) -> torch.Tensor:
    """Per-class belief mass ``b_k = e_k / S``.  Sums to ``1 - vacuity``."""
    return (alpha - 1.0) / alpha.sum(dim=-1, keepdim=True).clamp(min=EPSILON)


# ======================================================================
# The decomposition
# ======================================================================

def vacuity(alpha: torch.Tensor) -> torch.Tensor:
    """``K / S``: the share of the distribution that is prior rather than evidence."""
    num_classes = alpha.shape[-1]
    return num_classes / alpha.sum(dim=-1).clamp(min=EPSILON)


def dissonance(alpha: torch.Tensor) -> torch.Tensor:
    """Jousang's dissonance: belief mass that is present and contradictory.

        diss = sum_i [ b_i * sum_{j != i} b_j Bal(b_j, b_i) / sum_{j != i} b_j ]
        Bal(b_j, b_i) = 1 - |b_j - b_i| / (b_j + b_i)

    ``Bal`` is 1 for two equal beliefs and 0 when one dwarfs the other, so the
    measure is large exactly when two or more classes hold comparable, large
    belief -- genuine conflict -- and small both when one class dominates and
    when there is no belief at all.  The second case is vacuity's job, not this
    one's, which is why the two are reported separately before being combined.
    """
    b = belief(alpha).clamp(min=0.0)
    # [B, K, K] pairwise, diagonal excluded.
    bi = b.unsqueeze(-1)                      # [B, K, 1]
    bj = b.unsqueeze(-2)                      # [B, 1, K]
    balance = 1.0 - (bj - bi).abs() / (bj + bi).clamp(min=EPSILON)
    off_diagonal = 1.0 - torch.eye(b.shape[-1], device=b.device, dtype=b.dtype)
    balance = balance * off_diagonal

    numerator = (bj * balance).sum(dim=-1)                       # [B, K]
    denominator = (bj * off_diagonal).sum(dim=-1).clamp(min=EPSILON)
    return (b * numerator / denominator).sum(dim=-1).clamp(0.0, 1.0)


def composite_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
    """``vacuity + (1 - vacuity) * dissonance``, in [0, 1] by construction."""
    u = vacuity(alpha)
    return (u + (1.0 - u) * dissonance(alpha)).clamp(0.0, 1.0)


def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the Dirichlet MEAN, normalised to [0, 1].

    The comparator H4 has to beat.  Normalised by ``log K`` so it is on the
    same scale as the composite measure and the two can be compared without a
    rescaling step that could be accused of doing the work.
    """
    probabilities = dirichlet_mean(alpha)
    entropy = -(probabilities * (probabilities + EPSILON).log()).sum(dim=-1)
    import math
    return entropy / math.log(alpha.shape[-1])


def uncertainty_decomposition(alpha: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every scalar this module defines, for one batch of Dirichlets."""
    return {
        "vacuity": vacuity(alpha),
        "dissonance": dissonance(alpha),
        "composite": composite_uncertainty(alpha),
        "dirichlet_entropy": dirichlet_entropy(alpha),
        "evidence_total": alpha.sum(dim=-1) - alpha.shape[-1],
    }


# ======================================================================
# The objective
# ======================================================================

def kl_to_uniform_dirichlet(alpha: torch.Tensor) -> torch.Tensor:
    """``KL(Dir(alpha) || Dir(1))``, per sample.

    Zero exactly at ``alpha = 1``, so a model holding no evidence pays nothing
    and the term only ever penalises evidence that has been placed somewhere.
    """
    num_classes = alpha.shape[-1]
    ones = torch.ones_like(alpha)
    total = alpha.sum(dim=-1, keepdim=True)

    divergence = (
        torch.lgamma(total).squeeze(-1)
        - torch.lgamma(torch.tensor(float(num_classes), device=alpha.device))
        - torch.lgamma(alpha).sum(dim=-1)
        + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(total))).sum(dim=-1)
    )
    # A KL divergence cannot be negative. At alpha = 1 the lgamma and digamma
    # terms cancel to roughly -6e-8 in float64 and worse under autocast, and an
    # unclamped negative here would make the regulariser reward placing
    # evidence -- small, but with the wrong sign, which is the one kind of
    # numerical error that does not average out.
    return divergence.clamp(min=0.0)


def evidential_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    kl_weight: float = 0.0,
    alpha_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Bayes-risk evidential loss with an annealed KL regulariser.

    Masked on ``MISSING_CLASS_ID`` on exactly the same rule as ``focal_loss``,
    so the two are interchangeable in the objective and an unlabelled row
    contributes nothing to either.

    ``alpha_weights`` is the same per-class vector focal loss takes, applied
    per sample by true class.  Passing it keeps the two heads comparable: if
    one were class-weighted and the other not, any difference in calibration
    between them would be partly a difference in what they were asked to
    optimise.

    Returns the components separately -- the KL term's size relative to the
    fit term is the diagnostic for whether annealing is working.
    """
    valid = targets != MISSING_CLASS_ID
    zero = logits.sum() * 0.0
    if not bool(valid.any()):
        return {"evidential": zero, "fit": zero, "variance": zero, "kl": zero}

    logits, targets = logits[valid], targets[valid]
    if int(targets.min()) < 0 or int(targets.max()) >= num_classes:
        # one_hot's own message for a negative id is "Class values must be
        # non-negative", which says nothing about which sentinel leaked
        # through. A label space whose MISSING_CLASS_ID is not the one this
        # module masked on would land exactly here.
        raise ValueError(
            f"Class ids outside [0, {num_classes}) reached the evidential loss: "
            f"min {int(targets.min())}, max {int(targets.max())}. Unlabelled rows "
            f"must carry MISSING_CLASS_ID ({MISSING_CLASS_ID})."
        )
    alpha = alpha_from_logits(logits)
    total = alpha.sum(dim=-1, keepdim=True)
    probabilities = alpha / total.clamp(min=EPSILON)
    onehot = functional.one_hot(targets, num_classes).to(probabilities.dtype)

    fit = ((onehot - probabilities) ** 2).sum(dim=-1)
    variance = (probabilities * (1.0 - probabilities) / (total + 1.0)).sum(dim=-1)

    # Evidence for the TRUE class is removed before the penalty, so the term
    # only ever asks "why do you believe the wrong things", never "why are you
    # sure". Without this the regulariser fights the fit term directly.
    alpha_tilde = onehot + (1.0 - onehot) * alpha
    kl = kl_to_uniform_dirichlet(alpha_tilde)

    per_sample = fit + variance + kl_weight * kl
    if alpha_weights is not None:
        weights = alpha_weights.to(per_sample.device)[targets]
        per_sample = per_sample * weights

    return {
        "evidential": per_sample.mean(),
        "fit": fit.mean(),
        "variance": variance.mean(),
        "kl": kl.mean(),
    }


def kl_anneal_weight(epoch: int, anneal_epochs: int, maximum: float = 1.0) -> float:
    """``min(1, epoch / anneal_epochs) * maximum``, with epoch counted from 1.

    Zero weight in epoch 1 is deliberate: there is no evidence yet to
    regularise, and a penalty applied to its absence is a penalty on nothing
    that still has a gradient.
    """
    if anneal_epochs <= 0:
        return float(maximum)
    return float(maximum) * min(1.0, max(0, epoch - 1) / float(anneal_epochs))