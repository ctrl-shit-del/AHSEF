"""PHASE E -- the acquisition-budget curve.

Stage 3 reported its router at a single operating point.  That is not enough to
support or refute the central AHSEF claim, because a policy evaluated at one
budget cannot be distinguished from a lucky threshold.  The claim is about a
*curve*: performance as a function of how many modalities the system activates.

So every policy here is expressed as a **ranking** of samples by how much the
policy wants to acquire on each, and is then evaluated at a fixed grid of
budgets.  Ranking-then-budgeting has three properties that matter:

* Policies are compared at **matched acquisition rates**, so a difference is a
  difference in *which* samples were chosen, never in how many.
* The random control is exact rather than approximate: at each budget it is the
  mean over repeated seeded draws of the same size.
* The oracle upper bound is well defined: acquire on the samples with the
  largest true gain first.

The primary metric is macro-F1.  Accuracy is reported beside it and never
instead of it: the pool is 34% neutral, and Stage 2 established that an
accuracy-led reading of this corpus rewards majority-class behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

#: The budgets Phase E requires, as fractions of the pool.
DEFAULT_BUDGETS: tuple[float, ...] = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00
)

#: Repeats for the random-acquisition control at each budget.
RANDOM_REPEATS = 200


class BudgetError(RuntimeError):
    """Raised when a budget curve cannot be computed as specified."""


@dataclass(frozen=True)
class PoolOutcomes:
    """Everything a budget curve is computed from, for one aligned pool."""

    sample_ids: list[str]
    true_class: np.ndarray
    text_prediction: np.ndarray
    fused_prediction: np.ndarray
    audio_prediction: np.ndarray
    text_latency_ms: float
    audio_latency_ms: float
    text_compute_units: float = 0.0
    audio_compute_units: float = 0.0
    num_classes: int = 7
    #: Normalised predictive entropy before and after acquiring audio. Optional
    #: because a curve is well defined without it; when present, Phase E reports
    #: what acquisition did to the system's own confidence as well as to its
    #: accuracy, which are not the same question and can disagree.
    text_uncertainty: np.ndarray | None = None
    fused_uncertainty: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    @property
    def text_correct(self) -> np.ndarray:
        return self.text_prediction == self.true_class

    @property
    def fused_correct(self) -> np.ndarray:
        return self.fused_prediction == self.true_class


def acquire_top_k(scores: np.ndarray, budget: float) -> np.ndarray:
    """Acquire on the ``budget`` fraction of samples the policy ranks highest.

    Ties are broken by index rather than at random, so the mask is deterministic
    and two policies with identical scores produce identical masks.
    """
    scores = np.asarray(scores, dtype=float)
    if np.isnan(scores).any():
        raise BudgetError(
            "The policy score vector contains NaN; a sample with no score cannot be "
            "ranked and must be excluded explicitly."
        )
    count = int(round(float(budget) * scores.size))
    count = max(0, min(count, scores.size))
    mask = np.zeros(scores.size, dtype=bool)
    if count:
        # -scores then stable sort: highest first, ties in index order.
        order = np.argsort(-scores, kind="stable")
        mask[order[:count]] = True
    return mask


def score_at_budget(
    outcomes: PoolOutcomes, acquire: np.ndarray, budget: float, policy: str
) -> dict:
    """Everything Phase E asks for, at one budget for one policy."""
    acquire = np.asarray(acquire, dtype=bool)
    predicted = np.where(acquire, outcomes.fused_prediction, outcomes.text_prediction)
    metrics = classification_metrics(
        torch.tensor(predicted, dtype=torch.long),
        torch.tensor(outcomes.true_class, dtype=torch.long),
        outcomes.num_classes,
    )
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    text_ok = outcomes.text_correct
    fused_ok = outcomes.fused_correct
    corrected = int((acquire & ~text_ok & fused_ok).sum())
    harmed = int((acquire & text_ok & ~fused_ok).sum())
    # An acquisition that changed nothing was paid for and bought nothing. It is
    # not a mistake in the way a harm is, but it is waste and is counted apart.
    wasted = int((acquire & (outcomes.text_prediction == outcomes.fused_prediction)).sum())
    rate = float(acquire.mean()) if acquire.size else 0.0

    uncertainty = {}
    if outcomes.text_uncertainty is not None and outcomes.fused_uncertainty is not None:
        before = np.asarray(outcomes.text_uncertainty, dtype=float)
        after = np.where(acquire, np.asarray(outcomes.fused_uncertainty, dtype=float), before)
        uncertainty = {
            "mean_uncertainty_before_acquisition": float(before.mean()),
            "mean_uncertainty_after_acquisition": float(after.mean()),
            "mean_uncertainty_reduction": float((before - after).mean()),
            "mean_uncertainty_on_acquired_before": (
                float(before[acquire].mean()) if acquire.any() else None
            ),
            "mean_uncertainty_on_acquired_after": (
                float(after[acquire].mean()) if acquire.any() else None
            ),
        }

    return {
        "policy": policy,
        "requested_budget": float(budget),
        "acquisition_rate": rate,
        "acquired": int(acquire.sum()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class_f1": {
            name: metrics["per_class_f1"][index]
            for index, name in enumerate(CANONICAL_EMOTION_CLASSES[:outcomes.num_classes])
        },
        "average_modalities_per_sample": 1.0 + rate,
        "corrected_predictions": corrected,
        "harmed_predictions": harmed,
        "net_corrections": corrected - harmed,
        "unnecessary_acquisitions": wasted,
        "useful_acquisition_rate": (
            corrected / int(acquire.sum()) if int(acquire.sum()) else None
        ),
        "mean_latency_ms": outcomes.text_latency_ms + outcomes.audio_latency_ms * rate,
        "mean_compute_units": (
            outcomes.text_compute_units + outcomes.audio_compute_units * rate
        ),
        "acquisition_latency_ms": outcomes.audio_latency_ms * rate,
        "acquisition_compute_units": outcomes.audio_compute_units * rate,
        **uncertainty,
    }


def budget_curve(
    outcomes: PoolOutcomes,
    scores: np.ndarray,
    policy: str,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
) -> list[dict]:
    """One policy, scored across the whole budget grid."""
    return [
        score_at_budget(outcomes, acquire_top_k(scores, budget), budget, policy)
        for budget in budgets
    ]


def random_curve(
    outcomes: PoolOutcomes,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    repeats: int = RANDOM_REPEATS,
    seed: int = 42,
) -> list[dict]:
    """The control: acquiring uniformly at random at each budget.

    This is the line every learned policy has to beat. Beating text-only is not
    evidence of routing; beating a random budget of the same size is.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for budget in budgets:
        count = int(round(float(budget) * outcomes.size))
        draws = []
        for _ in range(repeats if 0 < count < outcomes.size else 1):
            mask = np.zeros(outcomes.size, dtype=bool)
            if count:
                mask[rng.choice(outcomes.size, count, replace=False)] = True
            draws.append(score_at_budget(outcomes, mask, budget, "random"))
        rows.append({
            "policy": "random",
            "requested_budget": float(budget),
            "acquisition_rate": float(count / outcomes.size) if outcomes.size else 0.0,
            "acquired": count,
            "repeats": len(draws),
            **{
                metric: {
                    "mean": float(np.mean([d[metric] for d in draws])),
                    "p2.5": float(np.percentile([d[metric] for d in draws], 2.5)),
                    "p97.5": float(np.percentile([d[metric] for d in draws], 97.5)),
                }
                for metric in ("accuracy", "macro_f1", "weighted_f1")
            },
            "mean_latency_ms": draws[0]["mean_latency_ms"],
            "average_modalities_per_sample": draws[0]["average_modalities_per_sample"],
        })
    return rows


def oracle_scores(outcomes: PoolOutcomes) -> np.ndarray:
    """The best possible ranking: acquire where the true signed gain is largest.

    Unachievable by construction -- it reads the labels -- and reported as the
    ceiling any real policy is measured against.
    """
    return outcomes.fused_correct.astype(float) - outcomes.text_correct.astype(float)


def compare_policies(
    outcomes: PoolOutcomes,
    policies: Mapping[str, np.ndarray],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    seed: int = 42,
    repeats: int = RANDOM_REPEATS,
) -> dict:
    """Every policy across the whole budget grid, plus the random and oracle lines."""
    curves = {
        name: budget_curve(outcomes, scores, name, budgets)
        for name, scores in policies.items()
    }
    curves["oracle"] = budget_curve(
        outcomes, oracle_scores(outcomes), "oracle", budgets
    )
    random_rows = random_curve(outcomes, budgets, repeats, seed)

    return {
        "pool_samples": outcomes.size,
        "budgets": [float(value) for value in budgets],
        "curves": curves,
        "random_control": random_rows,
        "endpoints": {
            "text_only": curves["oracle"][0],
            "always_fusion": curves["oracle"][-1],
        },
        "beats_random": {
            name: _beats_random(curve, random_rows) for name, curve in curves.items()
        },
        "primary_metric": "macro_f1",
        "why_macro_f1": (
            "The pool is roughly one-third neutral. Stage 2 established that an "
            "accuracy-led reading of this corpus rewards majority-class behaviour, "
            "so accuracy is reported beside macro-F1 and never instead of it."
        ),
        "reading": (
            "A policy earns its complexity only where its macro-F1 sits above the "
            "random control's 97.5th percentile at the SAME budget. Sitting inside "
            "the control interval means the same performance was available by "
            "acquiring at random, whatever the absolute numbers look like."
        ),
    }


def annotate_curve(
    curve: Sequence[Mapping],
    random_rows: Sequence[Mapping],
    oracle_curve: Sequence[Mapping],
    text_only: Mapping,
    always_fusion: Mapping,
) -> list[dict]:
    """Add every comparative column Phase E asks for, at matched budgets.

    The comparisons are made row by row against the *same* requested budget on
    each reference curve rather than against a single headline number. A gap to
    always-fusion quoted at one budget and a gap to random quoted at another
    would not be a comparison at all.

    ``incremental_cost_per_macro_f1`` is the acquisition latency spent divided by
    the macro-F1 bought over text-only. It is undefined when nothing was bought,
    and reported as ``None`` rather than as a large number, because a ratio with
    a near-zero denominator is not a cost-effectiveness estimate.
    """
    random_by_budget = {
        round(float(row["requested_budget"]), 6): row for row in random_rows
    }
    oracle_by_budget = {
        round(float(row["requested_budget"]), 6): row for row in oracle_curve
    }
    annotated = []
    for row in curve:
        budget = round(float(row["requested_budget"]), 6)
        control = random_by_budget.get(budget) or {}
        ceiling = oracle_by_budget.get(budget) or {}
        random_macro = (control.get("macro_f1") or {}).get("mean")
        gain = row["macro_f1"] - text_only["macro_f1"]
        spent = row.get("acquisition_latency_ms") or 0.0
        annotated.append({
            **row,
            "improvement_over_text_only": {
                "macro_f1": gain,
                "accuracy": row["accuracy"] - text_only["accuracy"],
                "weighted_f1": row["weighted_f1"] - text_only["weighted_f1"],
            },
            "improvement_over_random": {
                "macro_f1": (
                    row["macro_f1"] - random_macro if random_macro is not None else None
                ),
                "accuracy": (
                    row["accuracy"] - (control.get("accuracy") or {}).get("mean")
                    if (control.get("accuracy") or {}).get("mean") is not None else None
                ),
                "random_p97.5_macro_f1": (control.get("macro_f1") or {}).get("p97.5"),
                "above_random_upper_bound": bool(
                    0.0 < row["acquisition_rate"] < 1.0
                    and (control.get("macro_f1") or {}).get("p97.5") is not None
                    and row["macro_f1"] > (control["macro_f1"])["p97.5"]
                ),
            },
            "gap_to_always_fusion": {
                "macro_f1": always_fusion["macro_f1"] - row["macro_f1"],
                "retained_fraction_of_fusion_gain": (
                    gain / (always_fusion["macro_f1"] - text_only["macro_f1"])
                    if always_fusion["macro_f1"] != text_only["macro_f1"] else None
                ),
                "acquisitions_saved": always_fusion["acquired"] - row["acquired"],
            },
            "gap_to_oracle": {
                "macro_f1": (
                    ceiling["macro_f1"] - row["macro_f1"] if ceiling else None
                ),
                "fraction_of_oracle_gain_captured": (
                    gain / (ceiling["macro_f1"] - text_only["macro_f1"])
                    if ceiling and ceiling["macro_f1"] != text_only["macro_f1"] else None
                ),
            },
            "incremental_cost_per_macro_f1": (
                spent / gain if gain > 1e-9 else None
            ),
            "incremental_cost_note": (
                "acquisition latency (ms/sample, averaged over the pool) divided by "
                "the macro-F1 gained over text-only. None where no macro-F1 was "
                "gained: a ratio over a vanishing denominator is not an estimate."
            ),
        })
    return annotated


def _beats_random(curve: Sequence[Mapping], random_rows: Sequence[Mapping]) -> dict:
    """At which budgets does this policy clear the random control's upper bound?"""
    by_budget = {row["requested_budget"]: row for row in random_rows}
    verdicts = {}
    for row in curve:
        control = by_budget.get(row["requested_budget"])
        if control is None:
            continue
        interior = 0.0 < row["acquisition_rate"] < 1.0
        verdicts[f"{row['requested_budget']:.2f}"] = {
            "macro_f1": row["macro_f1"],
            "random_mean": control["macro_f1"]["mean"],
            "random_p97.5": control["macro_f1"]["p97.5"],
            "above_random_upper_bound": bool(
                interior and row["macro_f1"] > control["macro_f1"]["p97.5"]
            ),
            "comparable": interior,
        }
    decisive = [b for b, v in verdicts.items() if v["above_random_upper_bound"]]
    return {
        "per_budget": verdicts,
        "budgets_above_random": decisive,
        "any_budget_above_random": bool(decisive),
        "note": (
            "At 0% and 100% every policy is identical by construction, so those "
            "budgets are marked non-comparable rather than counted as wins."
        ),
    }


def efficiency_summary(comparison: Mapping, reference: str = "always_fusion") -> dict:
    """How much of full fusion each policy retains, and at what activation cost."""
    always = comparison["curves"]["oracle"][-1]
    rows = {}
    for name, curve in comparison["curves"].items():
        for row in curve:
            if row["requested_budget"] in (0.0, 1.0):
                continue
            rows.setdefault(name, []).append({
                "budget": row["requested_budget"],
                "macro_f1": row["macro_f1"],
                "macro_f1_retained": (
                    row["macro_f1"] / always["macro_f1"] if always["macro_f1"] else None
                ),
                "modalities_per_sample": row["average_modalities_per_sample"],
                "mean_latency_ms": row["mean_latency_ms"],
                "useful_acquisition_rate": row["useful_acquisition_rate"],
                "unnecessary_acquisitions": row["unnecessary_acquisitions"],
            })
    return {
        "reference": reference,
        "reference_macro_f1": always["macro_f1"],
        "per_policy": rows,
        "reading": (
            "'Macro-F1 retained' above 100% means the policy beat acquiring "
            "everywhere -- possible, because some acquisitions actively harm the "
            "prediction and a selective policy can decline them."
        ),
    }
