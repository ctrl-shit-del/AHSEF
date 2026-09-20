"""PHASE F -- the routing-policy objective, redesigned after Stage 2's failure.

Stage 2 selected its threshold with ``target_stop_accuracy``: the widest
coverage whose stopped set still reached an accuracy floor.  On a corpus that is
38.5% neutral, and with an LLM that answers ``neutral`` 61.8% of the time, that
objective did exactly what it was asked to and exactly the wrong thing -- it
found the neutral-heavy region, reported a high stopped-set accuracy, and let
macro-F1 collapse.  An objective that cannot see the minority classes will
always prefer a policy that ignores them.

So Stage 3 scores a whole *routed system*, not a stopped subset, and it scores it
with a per-class-aware metric::

    utility(tau) = macro_f1(tau) - alpha * acquisition_rate(tau)
                                 - beta  * normalized_latency(tau)

where the prediction for each sample is the text prediction when the router
stops and the fused prediction when it acquires -- which is what the deployed
system would actually output.

Four candidate objectives are swept and reported.  The selected one is declared
*in advance* here, with its reason, rather than picked afterwards from the
numbers; the comparison exists so the choice is auditable, exactly as Stage 2's
uncertainty-policy comparison was.  Nothing in this module may run on test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from src.training.metrics import classification_metrics

#: Splits on which a routing threshold may be selected.
THRESHOLD_SELECTION_SPLITS = ("train", "validation")

GATE_OBJECTIVES = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_f1_acquisition",
    "macro_f1_acquisition_latency",
)

#: Declared before any Stage 3 number was looked at.  See the module docstring.
SELECTED_OBJECTIVE = "macro_f1_acquisition_latency"

OBJECTIVE_RATIONALE = {
    "selected": SELECTED_OBJECTIVE,
    "declared": "in advance of inspecting any Stage 3 routing result",
    "why": (
        "Stage 2 showed that an accuracy-shaped objective on this corpus selects a "
        "neutral-heavy operating point with substantially higher accuracy and "
        "catastrophically lower macro-F1. Macro-F1 weights every emotion equally, so "
        "a policy cannot buy utility by ignoring the minority classes. The "
        "acquisition and latency penalties are what make the objective a ROUTING "
        "objective rather than a pure accuracy objective: without them the optimum "
        "is trivially 'always acquire'."
    ),
    "why_not_accuracy": (
        "It reproduces the Stage 2 failure mode directly on this corpus."
    ),
    "why_not_balanced_accuracy": (
        "It equalises recall but ignores precision, so it rewards a policy that "
        "predicts rare classes indiscriminately -- the opposite failure to accuracy's."
    ),
    "why_not_macro_f1_alone": (
        "With no penalty the optimum is always-acquire, which is the ablation this "
        "stage exists to compare against, not a routing policy."
    ),
    "alpha_meaning": (
        "alpha is the macro-F1 the policy must gain to justify acquiring audio for "
        "EVERY sample. alpha=0.05 means an across-the-board acquisition has to be "
        "worth 0.05 macro-F1; acquiring for 20% of samples has to be worth 0.01."
    ),
    "beta_meaning": (
        "beta prices the added wall-clock latency of acquisition, normalised to the "
        "always-acquire policy, on the same macro-F1 scale as alpha."
    ),
}


class ThresholdLeakageError(RuntimeError):
    """Raised when a routing threshold would be chosen using evaluation data."""


@dataclass(frozen=True)
class RoutedOutcome:
    """The realised system output at one threshold, and what it cost."""

    threshold: float
    acquired: int
    stopped: int
    acquisition_rate: float
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    normalized_latency: float
    mean_modalities: float
    per_class_f1: dict = field(default_factory=dict)

    def objective_value(self, objective: str, alpha: float, beta: float) -> float:
        if objective == "accuracy":
            return self.accuracy
        if objective == "macro_f1":
            return self.macro_f1
        if objective == "balanced_accuracy":
            return self.balanced_accuracy
        if objective == "macro_f1_acquisition":
            return self.macro_f1 - alpha * self.acquisition_rate
        if objective == "macro_f1_acquisition_latency":
            return (
                self.macro_f1
                - alpha * self.acquisition_rate
                - beta * self.normalized_latency
            )
        raise ValueError(f"objective must be one of {GATE_OBJECTIVES}, got {objective!r}")

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "acquired": self.acquired,
            "stopped": self.stopped,
            "acquisition_rate": self.acquisition_rate,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "balanced_accuracy": self.balanced_accuracy,
            "normalized_latency": self.normalized_latency,
            "mean_modalities": self.mean_modalities,
            "per_class_f1": dict(self.per_class_f1),
        }


def routed_predictions(
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    acquire: np.ndarray,
) -> np.ndarray:
    """What the deployed system would output: text when it stops, fused when it acquires."""
    text_prediction = np.asarray(text_prediction, dtype=int)
    fused_prediction = np.asarray(fused_prediction, dtype=int)
    acquire = np.asarray(acquire, dtype=bool)
    if not (text_prediction.shape == fused_prediction.shape == acquire.shape):
        raise ValueError("routed_predictions requires three arrays of equal length")
    return np.where(acquire, fused_prediction, text_prediction)


def score_policy(
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    acquire: np.ndarray,
    true_class: np.ndarray,
    threshold: float,
    class_names: Sequence[str],
    text_latency_ms: float,
    audio_latency_ms: float,
) -> RoutedOutcome:
    """Score one acquisition pattern as a complete routed system.

    ``normalized_latency`` is the policy's mean per-sample latency expressed as a
    fraction of the always-acquire policy's, so it lives on ``[text_only_share,
    1]`` and is comparable across thresholds without depending on the absolute
    millisecond figures.
    """
    acquire = np.asarray(acquire, dtype=bool)
    truth = torch.tensor(np.asarray(true_class, dtype=int), dtype=torch.long)
    predictions = torch.tensor(
        routed_predictions(text_prediction, fused_prediction, acquire), dtype=torch.long
    )
    metrics = classification_metrics(predictions, truth, len(class_names))
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    always = text_latency_ms + audio_latency_ms
    mean_latency = text_latency_ms + audio_latency_ms * float(acquire.mean())
    return RoutedOutcome(
        threshold=float(threshold),
        acquired=int(acquire.sum()),
        stopped=int((~acquire).sum()),
        acquisition_rate=float(acquire.mean()),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        balanced_accuracy=float(np.mean(recalls)) if recalls else 0.0,
        normalized_latency=float(mean_latency / always) if always > 0 else 0.0,
        mean_modalities=1.0 + float(acquire.mean()),
        per_class_f1={
            name: metrics["per_class_f1"][index] for index, name in enumerate(class_names)
        },
    )


def sweep_utility_threshold(
    utility: np.ndarray,
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    true_class: np.ndarray,
    class_names: Sequence[str],
    text_latency_ms: float,
    audio_latency_ms: float,
    grid: Sequence[float] | None = None,
) -> list[RoutedOutcome]:
    """Score every candidate acquisition threshold on the UGAPR utility.

    The rule is ``acquire when J(audio) >= threshold``.  The default grid is the
    sorted unique utilities plus one point above the maximum, so both extremes
    -- never acquire and always acquire -- are genuinely represented.
    """
    utility = np.asarray(utility, dtype=float)
    if np.isnan(utility).any():
        raise ValueError(
            "The utility vector contains NaN. A sample with no computable utility "
            "cannot be swept and must be excluded explicitly."
        )
    if grid is None:
        unique = np.unique(utility)
        candidates = np.concatenate([
            [float(unique.min()) - 1e-9], unique, [float(unique.max()) + 1e-9]
        ])
    else:
        candidates = np.asarray(sorted(set(float(value) for value in grid)), dtype=float)
    return [
        score_policy(
            text_prediction, fused_prediction, utility >= threshold, true_class,
            threshold, class_names, text_latency_ms, audio_latency_ms,
        )
        for threshold in candidates
    ]


@dataclass
class GateSelection:
    """A chosen acquisition threshold together with the evidence that chose it."""

    threshold: float
    objective: str
    alpha: float
    beta: float
    selected_on_split: str
    samples: int
    selected: RoutedOutcome
    sweep: list[RoutedOutcome] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "objective": self.objective,
            "objective_formula": (
                "utility = macro_f1 - alpha * acquisition_rate - beta * normalized_latency"
            ),
            "alpha_acquisition": self.alpha,
            "beta_latency": self.beta,
            "selected_on_split": self.selected_on_split,
            "uses_test_labels": False,
            "samples": self.samples,
            "selected_point": self.selected.to_dict(),
            "objective_value_at_selected": self.selected.objective_value(
                self.objective, self.alpha, self.beta
            ),
            "sweep": [point.to_dict() for point in self.sweep],
            "objective_comparison": dict(self.comparison),
            "rationale": dict(OBJECTIVE_RATIONALE),
            "rule": "REQUEST_AUDIO when the UGAPR utility J(audio) >= threshold; "
                    "otherwise STOP.",
            "note": self.note,
        }


def compare_objectives(
    sweep: Sequence[RoutedOutcome], alpha: float, beta: float
) -> dict:
    """What each candidate objective would have chosen, for audit.

    Reported so a reader can see the operating point the rejected objectives
    prefer -- in particular, whether the accuracy-shaped ones reproduce the
    Stage 2 neutral-heavy failure on this pool too.
    """
    table = {}
    for objective in GATE_OBJECTIVES:
        best = max(sweep, key=lambda point: point.objective_value(objective, alpha, beta))
        table[objective] = {
            "selected_threshold": best.threshold,
            "objective_value": best.objective_value(objective, alpha, beta),
            "acquisition_rate": best.acquisition_rate,
            "accuracy": best.accuracy,
            "macro_f1": best.macro_f1,
            "balanced_accuracy": best.balanced_accuracy,
            "per_class_f1": dict(best.per_class_f1),
            "selected_for_stage3": objective == SELECTED_OBJECTIVE,
        }
    return {
        "candidates": table,
        "selected": SELECTED_OBJECTIVE,
        "selection_was_declared_in_advance": True,
        "rationale": dict(OBJECTIVE_RATIONALE),
    }


def select_gate_threshold(
    utility: np.ndarray,
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    true_class: np.ndarray,
    split: str,
    class_names: Sequence[str],
    text_latency_ms: float,
    audio_latency_ms: float,
    objective: str = SELECTED_OBJECTIVE,
    alpha: float = 0.05,
    beta: float = 0.02,
) -> GateSelection:
    """Choose the acquisition threshold on ``split``.  Refuses to run on test."""
    if split not in THRESHOLD_SELECTION_SPLITS:
        raise ThresholdLeakageError(
            f"An acquisition threshold may only be selected on "
            f"{THRESHOLD_SELECTION_SPLITS}; refusing to select on {split!r}. Choosing "
            f"it where the answer is known is not a selection, it is a result quoted "
            f"backwards."
        )
    if objective not in GATE_OBJECTIVES:
        raise ValueError(f"objective must be one of {GATE_OBJECTIVES}, got {objective!r}")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    sweep = sweep_utility_threshold(
        utility, text_prediction, fused_prediction, true_class, class_names,
        text_latency_ms, audio_latency_ms,
    )
    best = max(sweep, key=lambda point: point.objective_value(objective, alpha, beta))
    return GateSelection(
        threshold=best.threshold, objective=objective, alpha=alpha, beta=beta,
        selected_on_split=split, samples=int(np.asarray(true_class).size),
        selected=best, sweep=sweep,
        comparison=compare_objectives(sweep, alpha, beta),
        note=(
            f"tau maximises {objective} on {split}. At the selected point the policy "
            f"acquires audio for {best.acquisition_rate:.1%} of samples and reaches "
            f"macro-F1 {best.macro_f1:.4f} (accuracy {best.accuracy:.4f})."
        ),
    )


def sensitivity_to_penalties(
    sweep: Sequence[RoutedOutcome],
    alphas: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20),
    betas: Sequence[float] = (0.0, 0.02, 0.05),
    objective: str = SELECTED_OBJECTIVE,
) -> list[dict]:
    """Where the selected threshold moves as the penalties change.

    Frozen alpha and beta are a policy choice, not a measurement, so the
    artefact records what the policy would have been at other prices instead of
    presenting one operating point as inevitable.
    """
    rows = []
    for alpha in alphas:
        for beta in betas:
            best = max(sweep, key=lambda point: point.objective_value(objective, alpha, beta))
            rows.append({
                "alpha": float(alpha), "beta": float(beta),
                "threshold": best.threshold,
                "acquisition_rate": best.acquisition_rate,
                "macro_f1": best.macro_f1,
                "accuracy": best.accuracy,
            })
    return rows
