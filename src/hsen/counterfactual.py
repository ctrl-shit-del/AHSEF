"""Counterfactual modality masking -- section 17.4.1.

Two jobs, one mechanism.

TRAINING.  CMU-MOSEI carries audio, video and text on every one of its 16,326
training samples, so a model trained on it never once sees a missing modality.
Masking one at inference time is then an input the model has no experience of,
and whatever it predicts is not evidence about that modality's value -- it is
evidence about how the model behaves off its training distribution.  HSIG
regresses exactly those masked predictions, so the distortion would land
directly in the routing policy.

:func:`drop_modalities` fixes that by masking modalities at random during
training, per sample and per modality, so every subset the router can ask about
is a subset the model has been trained to handle.

INFERENCE.  :func:`subset_availability` builds the availability flags for one
named subset, which is what the export pass uses to produce
``P(correct | subset)`` for every subset without retraining anything.

WHAT THIS IS NOT.  It is not the same as CMU-MOSEI's own missing-modality
literature (MMIN, MPLMM), where an absent modality is damage to be repaired.
Here an absent modality is a choice not to pay, and the model's job is to be
honest about what it can and cannot conclude without it -- which is why the
availability flags are fed to the fusion as explicit inputs rather than
inferred from a run of zeros.

NEVER ALL.  A sample with nothing observed has nothing to predict from, and the
Husformer fusion raises on it by design.  Every function here guarantees at
least one modality survives.
"""

from __future__ import annotations

import torch

#: Recorded in the run summary so a policy fitted on masked predictions can
#: never be mistaken for one fitted on a model that never saw masking.
COUNTERFACTUAL_PROVENANCE = "per-sample, per-modality Bernoulli drop; >=1 kept"


def subset_availability(
    reference: dict[str, torch.Tensor], subset: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Availability flags for exactly ``subset``, intersected with what exists.

    ``reference`` is the batch's own availability, so a sample that genuinely
    lacks a modality is never claimed to have it.  That matters on IEMOCAP,
    where video is absent everywhere: asking for the ``audio+video+text``
    subset there must yield ``audio+text`` rather than a fabricated video
    stream.
    """
    if not subset:
        raise ValueError("The empty subset is not a routing state; ask for at least one.")
    unknown = [m for m in subset if m not in reference]
    if unknown:
        raise ValueError(f"Subset names a modality the batch does not carry: {unknown}")

    return {
        modality: (flags & True if modality in subset else torch.zeros_like(flags))
        for modality, flags in reference.items()
    }


def drop_modalities(
    available: dict[str, torch.Tensor],
    probability: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Randomly mask modalities per sample, keeping at least one.

    ``probability`` is the per-modality chance that a present modality is
    hidden for that sample in that batch.  Sampling per batch rather than once
    per sample means a sample is seen under several different subsets across
    epochs, which is the point: the model learns every routing state rather
    than one fixed partition of them.

    The keep-one guarantee is applied after sampling, by restoring one
    uniformly chosen modality among those the sample actually has. Restoring a
    *fixed* modality instead -- the first, say -- would teach the model that one
    stream is always present, which is the opposite of what this is for.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"modality_dropout must be in [0, 1], got {probability}")
    if probability == 0.0:
        return available

    names = list(available)
    flags = torch.stack([available[m] for m in names], dim=1)          # [B, M]
    device = flags.device

    keep = torch.rand(flags.shape, device=device, generator=generator) >= probability
    dropped = flags & keep

    # Rows that lost everything they had get one of their real modalities back.
    empty = ~dropped.any(dim=1) & flags.any(dim=1)
    if bool(empty.any()):
        # Random among the modalities the row actually has: draw uniform noise,
        # mask it to the available ones, and take the argmax. A row with one
        # available modality can only pick that one, which is correct.
        noise = torch.rand(flags.shape, device=device, generator=generator)
        noise = noise.masked_fill(~flags, -1.0)
        rescue = torch.zeros_like(dropped)
        rescue[torch.arange(flags.shape[0], device=device), noise.argmax(dim=1)] = True
        dropped = dropped | (rescue & empty.unsqueeze(1))

    return {name: dropped[:, index] for index, name in enumerate(names)}


def dropout_summary(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor],
) -> dict[str, float]:
    """What a masking step actually did, for the run record."""
    names = list(before)
    kept = torch.stack([after[m] for m in names], dim=1)
    had = torch.stack([before[m] for m in names], dim=1)
    return {
        "modalities_before": float(had.float().sum(dim=1).mean()),
        "modalities_after": float(kept.float().sum(dim=1).mean()),
        "rows_with_none": int((~kept.any(dim=1)).sum()),
        **{f"kept_{m}": float(after[m].float().mean()) for m in names},
    }