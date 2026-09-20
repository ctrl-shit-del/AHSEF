"""PHASES H and I -- the validation experiment and the ablation suite.

Every system compared here is scored by the *same* function on the *same*
samples, so a difference between two rows is a difference between two policies
and nothing else.  A system is described by two arrays: which samples it
acquired audio for, and what it therefore predicted.  Everything else -- the
metrics, the latency accounting, the calibration, the improvement analysis --
follows from those.

The ablation suite exists to answer one question the headline table cannot:
is the benefit coming from *intelligent routing*, or simply from adding audio?
So it includes the two systems that need no router at all (audio always, fused
always), the policy Stage 2 would have used (a threshold on uncertainty), the
router with its cost model removed, and the router with a perfect gain estimate
substituted for HSIG.  If the dynamic policy does not sit meaningfully above the
uncertainty threshold and meaningfully below the oracle, that is the result and
it is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from src.ahsef.calibration import calibration_report
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics


@dataclass(frozen=True)
class SystemInputs:
    """The fixed material every candidate policy is scored against."""

    sample_ids: list[str]
    true_class: np.ndarray
    text_prediction: np.ndarray
    audio_prediction: np.ndarray
    fused_prediction: np.ndarray
    text_uncertainty: np.ndarray
    fused_uncertainty: np.ndarray
    text_probabilities: np.ndarray
    audio_probabilities: np.ndarray
    fused_probabilities: np.ndarray
    text_latency_ms: float
    audio_latency_ms: float

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        for name in (
            "true_class", "text_prediction", "audio_prediction", "fused_prediction",
            "text_uncertainty", "fused_uncertainty",
        ):
            if getattr(self, name).shape[0] != n:
                raise ValueError(f"{name} has {getattr(self, name).shape[0]} rows, expected {n}")

    @property
    def size(self) -> int:
        return len(self.sample_ids)


def system_report(
    inputs: SystemInputs,
    acquire: np.ndarray,
    name: str,
    fixed_prediction: np.ndarray | None = None,
    fixed_probabilities: np.ndarray | None = None,
    bins: int = 10,
) -> dict:
    """Score one policy end to end.

    ``acquire`` says which samples the policy paid for audio on.  The realised
    prediction is the fused one where it acquired and the text one where it did
    not -- unless ``fixed_prediction`` overrides that, which is how the
    audio-only ablation (a system that never consults text) is expressed.
    """
    acquire = np.asarray(acquire, dtype=bool)
    if acquire.shape[0] != inputs.size:
        raise ValueError("The acquisition mask must cover every sample")

    if fixed_prediction is not None:
        predicted = np.asarray(fixed_prediction, dtype=int)
        probabilities = fixed_probabilities
    else:
        predicted = np.where(acquire, inputs.fused_prediction, inputs.text_prediction)
        probabilities = np.where(
            acquire[:, None], inputs.fused_probabilities, inputs.text_probabilities
        )

    truth = torch.tensor(inputs.true_class, dtype=torch.long)
    metrics = classification_metrics(
        torch.tensor(predicted, dtype=torch.long), truth, len(CANONICAL_EMOTION_CLASSES)
    )
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    correct = predicted == inputs.true_class
    mean_latency = (
        inputs.text_latency_ms + inputs.audio_latency_ms * float(acquire.mean())
        if fixed_prediction is None else None
    )

    record = {
        "system": name,
        "samples": inputs.size,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class_f1": {
            class_name: metrics["per_class_f1"][index]
            for index, class_name in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "support": {
            class_name: metrics["support"][index]
            for index, class_name in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "correct": int(correct.sum()),
        "acquisition": {
            "audio_acquisition_rate": float(acquire.mean()),
            "audio_acquired": int(acquire.sum()),
            "average_modalities_activated": (
                1.0 + float(acquire.mean()) if fixed_prediction is None else 1.0
            ),
        },
        "latency": _latency_block(inputs, acquire, correct, mean_latency, fixed_prediction),
        "uncertainty": _uncertainty_block(inputs, acquire),
        "improvement_among_requested": _improvement_block(inputs, acquire),
    }
    if probabilities is not None:
        usable = np.isfinite(probabilities).all(axis=1)
        if usable.any():
            record["calibration"] = {
                **calibration_report(
                    torch.tensor(probabilities[usable], dtype=torch.float64),
                    truth[torch.tensor(usable)], bins, name,
                ),
                "caveat": (
                    "Text-side probabilities are NORMALISED SELF-REPORTED LLM scores, "
                    "not posteriors (Stage 2 measured their NLL as worse than uniform). "
                    "These figures measure how well that self-report tracks correctness "
                    "and must not be read as posterior calibration."
                ),
                "rows_scored": int(usable.sum()),
                "rows_without_scores": int((~usable).sum()),
            }
    return record


def _latency_block(inputs, acquire, correct, mean_latency, fixed_prediction) -> dict:
    if fixed_prediction is not None:
        # The audio-only system pays audio latency and nothing else.
        mean_latency = inputs.audio_latency_ms
    total = mean_latency * inputs.size
    return {
        "text_latency_ms": inputs.text_latency_ms,
        "audio_latency_ms": inputs.audio_latency_ms,
        "mean_latency_ms_per_sample": float(mean_latency),
        "total_latency_ms": float(total),
        "latency_per_correct_prediction_ms": (
            float(total / int(correct.sum())) if int(correct.sum()) else None
        ),
        "normalized_latency": float(
            mean_latency / (inputs.text_latency_ms + inputs.audio_latency_ms)
        ) if (inputs.text_latency_ms + inputs.audio_latency_ms) > 0 else None,
        "note": (
            "Latency is the modelled per-sample cost of the policy: the text pass "
            "always, plus the audio pass on the samples the policy acquired. The "
            "text figure is measured LLM wall clock, not generation time."
        ),
    }


def _uncertainty_block(inputs, acquire) -> dict:
    return {
        "mean_before_acquisition": float(inputs.text_uncertainty.mean()),
        "mean_before_among_stopped": float(inputs.text_uncertainty[~acquire].mean())
        if (~acquire).any() else None,
        "mean_before_among_requested": float(inputs.text_uncertainty[acquire].mean())
        if acquire.any() else None,
        "mean_after_among_requested": float(inputs.fused_uncertainty[acquire].mean())
        if acquire.any() else None,
        "mean_reduction_among_requested": float(
            (inputs.text_uncertainty[acquire] - inputs.fused_uncertainty[acquire]).mean()
        ) if acquire.any() else None,
        "note": (
            "Uncertainty reduction is a diagnostic. Stage 1 established it is not a "
            "valid acquisition target and it is not used as one here."
        ),
    }


def _improvement_block(inputs, acquire) -> dict:
    """Did acquisition help on the samples this policy actually paid for?"""
    if not acquire.any():
        return {"requested": 0, "note": "this policy acquired audio for no sample"}
    text_correct = inputs.text_prediction[acquire] == inputs.true_class[acquire]
    fused_correct = inputs.fused_prediction[acquire] == inputs.true_class[acquire]
    fixed = int((~text_correct & fused_correct).sum())
    broken = int((text_correct & ~fused_correct).sum())
    requested = int(acquire.sum())
    return {
        "requested": requested,
        "fixed_by_audio": fixed,
        "broken_by_audio": broken,
        "net_improvement": fixed - broken,
        "actual_improvement_rate": fixed / requested,
        "actual_harm_rate": broken / requested,
        "net_improvement_rate": (fixed - broken) / requested,
        "precision_of_acquisition": fixed / requested,
        "note": (
            "'Improvement rate' is the fraction of PAID acquisitions that turned a "
            "wrong text answer into a right fused one. It is the number that says "
            "whether the router spent well, and it is read only after routing."
        ),
    }


# ============================================================
# Policies
# ============================================================

def never_acquire(size: int) -> np.ndarray:
    return np.zeros(size, dtype=bool)


def always_acquire(size: int) -> np.ndarray:
    return np.ones(size, dtype=bool)


def threshold_policy(score: np.ndarray, threshold: float, greater_is_acquire: bool = True):
    score = np.asarray(score, dtype=float)
    return score >= threshold if greater_is_acquire else score <= threshold


def compare_systems(
    inputs: SystemInputs,
    policies: Mapping[str, np.ndarray],
    audio_only_probabilities: np.ndarray | None = None,
    bins: int = 10,
) -> dict:
    """Score a named set of policies and add the derived comparison columns."""
    reports = {}
    for name, acquire in policies.items():
        if name == "audio_only":
            reports[name] = system_report(
                inputs, always_acquire(inputs.size), name,
                fixed_prediction=inputs.audio_prediction,
                fixed_probabilities=(
                    audio_only_probabilities if audio_only_probabilities is not None
                    else inputs.audio_probabilities
                ),
                bins=bins,
            )
        else:
            reports[name] = system_report(inputs, np.asarray(acquire, dtype=bool), name, bins=bins)

    full = reports.get("always_fusion")
    if full:
        for name, record in reports.items():
            record["relative_to_always_fusion"] = {
                "accuracy_retained": (
                    record["accuracy"] / full["accuracy"] if full["accuracy"] else None
                ),
                "macro_f1_retained": (
                    record["macro_f1"] / full["macro_f1"] if full["macro_f1"] else None
                ),
                "accuracy_delta": record["accuracy"] - full["accuracy"],
                "macro_f1_delta": record["macro_f1"] - full["macro_f1"],
                "acquisition_rate_delta": (
                    record["acquisition"]["audio_acquisition_rate"]
                    - full["acquisition"]["audio_acquisition_rate"]
                ),
                "latency_saved_fraction": (
                    1.0 - record["latency"]["mean_latency_ms_per_sample"]
                    / full["latency"]["mean_latency_ms_per_sample"]
                    if full["latency"]["mean_latency_ms_per_sample"] else None
                ),
            }
    return {
        "samples": inputs.size,
        "class_order": list(CANONICAL_EMOTION_CLASSES),
        "systems": reports,
        "central_result": _central_result(reports),
    }


def _central_result(reports: Mapping[str, dict]) -> dict:
    """Performance against modalities activated, and against latency."""
    frontier = [
        {
            "system": name,
            "average_modalities_activated":
                record["acquisition"]["average_modalities_activated"],
            "audio_acquisition_rate": record["acquisition"]["audio_acquisition_rate"],
            "mean_latency_ms": record["latency"]["mean_latency_ms_per_sample"],
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "weighted_f1": record["weighted_f1"],
        }
        for name, record in reports.items()
    ]
    frontier.sort(key=lambda row: row["average_modalities_activated"])
    return {
        "performance_vs_modalities_activated": frontier,
        "performance_vs_latency": sorted(frontier, key=lambda row: row["mean_latency_ms"]),
        "reading": (
            "A dynamic policy earns its complexity only if it sits ABOVE the line "
            "joining text-only to always-fusion on these axes. Sitting on or below "
            "that line means the same performance was available by acquiring at "
            "random, or by not routing at all."
        ),
    }


def random_policy_reference(
    inputs: SystemInputs, rate: float, seed: int = 42, repeats: int = 200
) -> dict:
    """What acquiring at random at the same rate would have achieved.

    This is the control the central claim needs.  A router that acquires for 40%
    of samples and beats text-only has proved nothing until it also beats
    acquiring for a random 40%.
    """
    rng = np.random.default_rng(seed)
    count = int(round(rate * inputs.size))
    accuracies, macros = [], []
    for _ in range(repeats):
        acquire = np.zeros(inputs.size, dtype=bool)
        if count:
            acquire[rng.choice(inputs.size, count, replace=False)] = True
        predicted = np.where(acquire, inputs.fused_prediction, inputs.text_prediction)
        metrics = classification_metrics(
            torch.tensor(predicted, dtype=torch.long),
            torch.tensor(inputs.true_class, dtype=torch.long),
            len(CANONICAL_EMOTION_CLASSES),
        )
        accuracies.append(metrics["accuracy"])
        macros.append(metrics["macro_f1"])
    return {
        "acquisition_rate": rate,
        "repeats": repeats,
        "seed": seed,
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "p2.5": float(np.percentile(accuracies, 2.5)),
            "p97.5": float(np.percentile(accuracies, 97.5)),
        },
        "macro_f1": {
            "mean": float(np.mean(macros)),
            "p2.5": float(np.percentile(macros, 2.5)),
            "p97.5": float(np.percentile(macros, 97.5)),
        },
        "purpose": (
            "The control for the central claim: a router must beat acquiring at "
            "random at the SAME rate, not merely beat text-only."
        ),
    }


def build_inputs(
    sample_ids: Sequence[str],
    text_set,
    audio_set,
    fused_set,
    text_uncertainty: np.ndarray,
    text_latency_ms: float,
    audio_latency_ms: float,
) -> SystemInputs:
    """Assemble :class:`SystemInputs` from three aligned prediction sets."""
    ids = [str(item) for item in sample_ids]
    text = text_set.restricted_to(ids)
    audio = audio_set.restricted_to(ids)
    fused = fused_set.restricted_to(ids)
    if not (text.sample_ids() == audio.sample_ids() == fused.sample_ids()):
        raise ValueError("Prediction sets did not reindex identically to the pool order")
    return SystemInputs(
        sample_ids=ids,
        true_class=text.labels().numpy(),
        text_prediction=text.predictions().numpy(),
        audio_prediction=audio.predictions().numpy(),
        fused_prediction=fused.predictions().numpy(),
        text_uncertainty=np.asarray(text_uncertainty, dtype=float),
        fused_uncertainty=fused.frame["normalized_entropy"].to_numpy(dtype=float),
        text_probabilities=text.probabilities().numpy(),
        audio_probabilities=audio.probabilities().numpy(),
        fused_probabilities=fused.probabilities().numpy(),
        text_latency_ms=float(text_latency_ms),
        audio_latency_ms=float(audio_latency_ms),
    )
