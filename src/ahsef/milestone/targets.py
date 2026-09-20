"""PHASE D -- candidate routing targets, defined mathematically before fitting.

Stage 3 trained HSIG to predict ``gain(m|A) = P(correct|A∪{m}) − P(correct|A)``
and then reported macro-F1.  Those are not the same objective, and on this
corpus they disagree: 62% of the corrections audio supplies land in ``neutral``
and ``angry``, the two largest classes, where a macro average barely rewards
them.  A router optimised for corrections will therefore spend its budget where
the reported metric gains least.  That is the most likely explanation for the
Stage 3 locked-test result, and Phase D tests it by putting the alternatives
side by side.

Every candidate below is a **per-sample supervision signal** computed from the
text prediction, the fused prediction and the true label.  Each is defined here
in closed form *before* any of them is fitted, so the comparison cannot be a
search for the target that happened to win.

============================  ============================================
target                        value for sample *i*
============================  ============================================
``binary_correction``         ``1`` if text wrong and fused right, else ``0``
``signed_gain``               ``fused_correct − text_correct`` ∈ {−1, 0, +1}
``class_balanced_gain``       ``signed_gain × w(true class)``, ``w ∝ 1/count``
``macro_f1_marginal``         ``macroF1(fuse only i) − macroF1(fuse nothing)``
``uncertainty_reduction``     ``U(text) − U(fused)``  (Stage 1 rejected this)
============================  ============================================

``macro_f1_marginal`` is the one that matches the reported metric exactly: it
asks what acquiring on *this* sample does to the number the policy is graded on.
``class_balanced_gain`` is a cheaper approximation of the same intuition.
``uncertainty_reduction`` is included because Phase D asks for it and because
re-measuring a rejected target on new data is worth more than citing the old
rejection; it is expected to fail again, and failing again is a result.

A target is never a feature.  These quantities all read the true label and are
used only to *fit* the estimator on validation; the router at inference sees
none of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

#: Declared before any target was fitted.
TARGET_NAMES: tuple[str, ...] = (
    "binary_correction",
    "signed_gain",
    "class_balanced_gain",
    "macro_f1_marginal",
    "uncertainty_reduction",
)

#: Stage 1 measured this one and rejected it; it is carried only as a control.
REJECTED_BY_STAGE1 = "uncertainty_reduction"

TARGET_DEFINITIONS: Mapping[str, str] = {
    "binary_correction":
        "1 if the text prediction is wrong and the fused prediction is correct, "
        "else 0. Stage 3's 'fix_logistic' target. Cannot express harm.",
    "signed_gain":
        "fused_correct - text_correct, in {-1, 0, +1}. The per-sample realisation "
        "of gain(m|A) = P(correct|A u {m}) - P(correct|A). Stage 3's frozen target.",
    "class_balanced_gain":
        "signed_gain weighted by the inverse frequency of the sample's TRUE class, "
        "normalised to mean 1. A correction on a rare class is worth more, which "
        "is what a macro average rewards and what an unweighted gain ignores.",
    "macro_f1_marginal":
        "macro_f1(text everywhere except this sample, fused here) minus "
        "macro_f1(text everywhere). The exact marginal contribution of acquiring "
        "on this sample to the metric the policy is graded on.",
    "uncertainty_reduction":
        "normalised predictive entropy of the text posterior minus that of the "
        "fused posterior. REJECTED as an HSIG target by Stage 1; carried as a "
        "control so the rejection is re-measured rather than merely cited.",
}


class TargetError(RuntimeError):
    """Raised when a routing target cannot be computed from the given inputs."""


@dataclass(frozen=True)
class RoutingOutcomes:
    """The post-acquisition facts every candidate target is derived from.

    Built once, after paying for audio on every validation sample. Labels enter
    here and stay here: the router never receives this object.
    """

    sample_ids: list[str]
    true_class: np.ndarray
    text_prediction: np.ndarray
    fused_prediction: np.ndarray
    text_uncertainty: np.ndarray
    fused_uncertainty: np.ndarray
    num_classes: int = 7

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        for name in ("true_class", "text_prediction", "fused_prediction",
                     "text_uncertainty", "fused_uncertainty"):
            if getattr(self, name).shape[0] != n:
                raise TargetError(
                    f"{name} has {getattr(self, name).shape[0]} rows, expected {n}"
                )
        if n == 0:
            raise TargetError("Cannot build routing targets from an empty pool")

    @property
    def text_correct(self) -> np.ndarray:
        return self.text_prediction == self.true_class

    @property
    def fused_correct(self) -> np.ndarray:
        return self.fused_prediction == self.true_class

    def summary(self) -> dict:
        return {
            "samples": len(self.sample_ids),
            "text_accuracy": float(self.text_correct.mean()),
            "fused_accuracy": float(self.fused_correct.mean()),
            "corrections": int((~self.text_correct & self.fused_correct).sum()),
            "harms": int((self.text_correct & ~self.fused_correct).sum()),
        }


# ============================================================
# The targets
# ============================================================

def binary_correction(outcomes: RoutingOutcomes) -> np.ndarray:
    return (~outcomes.text_correct & outcomes.fused_correct).astype(float)


def signed_gain(outcomes: RoutingOutcomes) -> np.ndarray:
    return outcomes.fused_correct.astype(float) - outcomes.text_correct.astype(float)


def class_weights_from_labels(labels: np.ndarray, num_classes: int = 7) -> np.ndarray:
    """Inverse-frequency weights over present classes, normalised to mean 1.

    Absent classes get weight 0 rather than infinity; normalising to mean 1
    keeps the weighted target on the same scale as the unweighted one, so the
    two are comparable without rescaling the estimator.
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes).astype(float)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=float)
    weights[present] = 1.0 / counts[present]
    if weights[present].size:
        weights[present] /= weights[present].mean()
    return weights


def class_balanced_gain(outcomes: RoutingOutcomes) -> np.ndarray:
    weights = class_weights_from_labels(outcomes.true_class, outcomes.num_classes)
    return signed_gain(outcomes) * weights[outcomes.true_class.astype(int)]


def macro_f1_marginal(outcomes: RoutingOutcomes) -> np.ndarray:
    """What acquiring on each sample alone does to macro-F1.

    Computed exactly, one sample at a time, against the all-text baseline. The
    result is tiny in magnitude -- one sample out of a few hundred moves macro-F1
    by ~1e-3 -- but only its *ordering* is used to fit an estimator, and the
    ordering is what distinguishes a correction that helps the metric from one
    that does not.
    """
    truth = torch.tensor(outcomes.true_class, dtype=torch.long)
    text = np.asarray(outcomes.text_prediction, dtype=int)
    fused = np.asarray(outcomes.fused_prediction, dtype=int)

    base = classification_metrics(
        torch.tensor(text, dtype=torch.long), truth, outcomes.num_classes
    )["macro_f1"]

    values = np.zeros(len(text), dtype=float)
    for index in range(len(text)):
        if text[index] == fused[index]:
            continue                      # acquiring changes nothing for this sample
        swapped = text.copy()
        swapped[index] = fused[index]
        values[index] = classification_metrics(
            torch.tensor(swapped, dtype=torch.long), truth, outcomes.num_classes
        )["macro_f1"] - base
    return values


def uncertainty_reduction(outcomes: RoutingOutcomes) -> np.ndarray:
    return outcomes.text_uncertainty - outcomes.fused_uncertainty


TARGET_FUNCTIONS: Mapping[str, Callable[[RoutingOutcomes], np.ndarray]] = {
    "binary_correction": binary_correction,
    "signed_gain": signed_gain,
    "class_balanced_gain": class_balanced_gain,
    "macro_f1_marginal": macro_f1_marginal,
    "uncertainty_reduction": uncertainty_reduction,
}


def build_target(name: str, outcomes: RoutingOutcomes) -> np.ndarray:
    try:
        function = TARGET_FUNCTIONS[name]
    except KeyError as error:
        raise TargetError(
            f"Unknown routing target {name!r}; declared targets are {TARGET_NAMES}"
        ) from error
    return function(outcomes)


def build_all_targets(outcomes: RoutingOutcomes) -> dict[str, np.ndarray]:
    return {name: build_target(name, outcomes) for name in TARGET_NAMES}


# ============================================================
# Description
# ============================================================

def describe_targets(outcomes: RoutingOutcomes) -> dict:
    """What each candidate target looks like on this pool, before any fitting."""
    targets = build_all_targets(outcomes)
    per_class_corrections = {}
    corrections = ~outcomes.text_correct & outcomes.fused_correct
    for index, name in enumerate(CANONICAL_EMOTION_CLASSES):
        mask = outcomes.true_class == index
        per_class_corrections[name] = {
            "support": int(mask.sum()),
            "corrections": int((corrections & mask).sum()),
            "share_of_all_corrections": (
                float((corrections & mask).sum() / max(corrections.sum(), 1))
            ),
        }

    record = {
        "outcomes": outcomes.summary(),
        "definitions": dict(TARGET_DEFINITIONS),
        "rejected_by_stage1": REJECTED_BY_STAGE1,
        "targets": {
            name: {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "nonzero": int((values != 0).sum()),
                "positive": int((values > 0).sum()),
                "negative": int((values < 0).sum()),
                "distinct_values": int(np.unique(values).size),
            }
            for name, values in targets.items()
        },
        "correlations": {
            left: {
                right: _safe_corr(targets[left], targets[right])
                for right in TARGET_NAMES if right != left
            }
            for left in TARGET_NAMES
        },
        "where_the_corrections_are": per_class_corrections,
        "why_this_matters": (
            "If most corrections fall in the largest classes, a target that counts "
            "corrections equally will send the router to samples a macro average "
            "barely rewards. The per-class breakdown above is the evidence for or "
            "against that being the Stage 3 failure mode."
        ),
    }
    concentration = sorted(
        per_class_corrections.items(),
        key=lambda item: -item[1]["share_of_all_corrections"],
    )
    top_two = concentration[:2]
    record["correction_concentration"] = {
        "top_classes": [name for name, _ in top_two],
        "share_in_top_two": float(
            sum(entry["share_of_all_corrections"] for _, entry in top_two)
        ),
        "reading": (
            f"{top_two[0][1]['share_of_all_corrections']:.0%} of all corrections land "
            f"in '{top_two[0][0]}' and {top_two[1][1]['share_of_all_corrections']:.0%} "
            f"in '{top_two[1][0]}'."
        ) if len(top_two) == 2 else None,
    }
    return record


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return None if np.isnan(value) else value
