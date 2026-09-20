"""PHASE D (evaluation) -- how good is each candidate routing target?

A target is not judged by how well it can be predicted.  ``uncertainty_reduction``
is nearly deterministic in the LLM's own posterior and would win a fit-quality
contest outright while being useless for routing.  What matters is whether the
*fitted score* orders samples by how much acquiring audio would help, and whether
that ordering is one a macro average rewards.

So every target is scored on five axes, all out of fold, all on validation:

``identifies useful acquisitions``
    AUROC and AUPRC against the event "audio fixed a wrong text prediction".
    AUPRC is always printed beside the base rate, because the event is rare and
    an AUROC of 0.75 on a 6.7% positive rate is compatible with a precision that
    makes the router useless at its operating point.

``avoids harmful acquisitions``
    the same question for "audio broke a correct text prediction", read so that
    a *useful* estimator gives harmful samples a LOW score.

``rank agreement``
    Spearman against the realised signed gain -- the quantity any router is
    ultimately trying to order by, whatever it was fitted on.

``majority-class bias``
    the share of the top-decile scored samples whose true class is one of the
    two largest.  This is the Stage 3 failure mode written as a number: a target
    that sends the router to neutral and angry will show it here before any
    budget curve is drawn.

``macro-F1 alignment``
    Spearman between the fitted score and ``macro_f1_marginal``.  A target whose
    ranking disagrees with the metric it will be graded on is not going to be
    rescued by a budget grid.

Calibration is reported for completeness and is deliberately *not* part of the
selection rule.  The policy acquires top-k, so only the ordering is used; a
target that ranks perfectly and is offset by a constant would route perfectly.
Saying so explicitly is better than reporting a calibration number and letting a
reader assume it mattered.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.ahsef.milestone.targets import RoutingOutcomes, TARGET_DEFINITIONS
from src.ahsef.stage3.hsig_quality import auprc, auroc, calibration_of_gain, spearman
from src.common.labels import CANONICAL_EMOTION_CLASSES

#: Fraction of the pool treated as "where the router would actually spend" when
#: measuring class concentration.  10% is on the Phase E budget grid, so the
#: number describes a real operating point rather than an abstract decile.
TOP_FRACTION = 0.10


def _top_mask(scores: np.ndarray, fraction: float = TOP_FRACTION) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    count = max(1, int(round(fraction * scores.size)))
    mask = np.zeros(scores.size, dtype=bool)
    mask[np.argsort(-scores, kind="stable")[:count]] = True
    return mask


def class_concentration(
    values: np.ndarray, outcomes: RoutingOutcomes, fraction: float = TOP_FRACTION
) -> dict:
    """Where in class space does this signal point the router?

    ``majority`` here means the two largest classes *of this pool*, computed from
    the support rather than assumed, so the measure survives a change of pool.
    """
    counts = np.bincount(
        outcomes.true_class.astype(int), minlength=outcomes.num_classes
    )
    ranked = np.argsort(-counts, kind="stable")
    majority = set(int(index) for index in ranked[:2] if counts[index] > 0)
    mask = _top_mask(values, fraction)
    selected = outcomes.true_class[mask].astype(int)

    per_class = {}
    for index, name in enumerate(CANONICAL_EMOTION_CLASSES[:outcomes.num_classes]):
        pool_share = float(counts[index] / max(counts.sum(), 1))
        selected_share = (
            float((selected == index).sum() / max(selected.size, 1))
        )
        per_class[name] = {
            "pool_support": int(counts[index]),
            "pool_share": pool_share,
            "selected": int((selected == index).sum()),
            "selected_share": selected_share,
            "over_representation": (
                selected_share / pool_share if pool_share > 0 else None
            ),
        }

    majority_pool_share = float(
        sum(counts[index] for index in majority) / max(counts.sum(), 1)
    )
    majority_selected_share = float(
        sum(1 for value in selected if value in majority) / max(selected.size, 1)
    )
    return {
        "top_fraction": float(fraction),
        "selected_samples": int(mask.sum()),
        "majority_classes": [
            CANONICAL_EMOTION_CLASSES[index] for index in sorted(majority)
        ],
        "majority_pool_share": majority_pool_share,
        "majority_selected_share": majority_selected_share,
        "majority_over_representation": (
            majority_selected_share / majority_pool_share
            if majority_pool_share > 0 else None
        ),
        "favours_majority_classes": bool(
            majority_selected_share > majority_pool_share
        ),
        "per_class": per_class,
        "reading": (
            "over_representation above 1.0 means the signal sends the router to "
            "that class more often than the pool contains it. A macro average "
            "rewards spending on the rare classes, so a majority "
            "over-representation above 1.0 is the Stage 3 failure mode appearing "
            "before any budget curve is drawn."
        ),
    }


def target_distribution(values: np.ndarray, outcomes: RoutingOutcomes) -> dict:
    """The shape of the supervision signal itself, before anything is fitted."""
    values = np.asarray(values, dtype=float)
    counts = np.bincount(outcomes.true_class.astype(int), minlength=outcomes.num_classes)
    per_class = {}
    for index, name in enumerate(CANONICAL_EMOTION_CLASSES[:outcomes.num_classes]):
        mask = outcomes.true_class.astype(int) == index
        if not mask.any():
            per_class[name] = {"support": 0}
            continue
        per_class[name] = {
            "support": int(counts[index]),
            "mean": float(values[mask].mean()),
            "positive": int((values[mask] > 0).sum()),
            "negative": int((values[mask] < 0).sum()),
            "share_of_total_positive_mass": (
                float(values[mask][values[mask] > 0].sum() / values[values > 0].sum())
                if (values > 0).any() else None
            ),
        }
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "nonzero": int((values != 0).sum()),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "distinct_values": int(np.unique(values).size),
        "expresses_harm": bool((values < 0).any()),
        "per_class": per_class,
    }


def target_quality(
    name: str,
    scores: np.ndarray,
    target_values: np.ndarray,
    outcomes: RoutingOutcomes,
    macro_f1_marginal_values: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> dict:
    """Everything Phase D asks for about one fitted target, out of fold."""
    scores = np.asarray(scores, dtype=float)
    fixed = (~outcomes.text_correct & outcomes.fused_correct)
    harmed = (outcomes.text_correct & ~outcomes.fused_correct)
    realised = outcomes.fused_correct.astype(float) - outcomes.text_correct.astype(float)

    harm_auroc = auroc(scores, harmed)
    top = _top_mask(scores, fraction)

    return {
        "target": name,
        "definition": TARGET_DEFINITIONS.get(name),
        "identifying_useful_acquisitions": {
            "positive_event": "audio fixed a wrong text prediction",
            "positives": int(fixed.sum()),
            "base_rate": float(fixed.mean()),
            "auroc": auroc(scores, fixed),
            "auprc": auprc(scores, fixed),
            "auprc_baseline_is_base_rate": float(fixed.mean()),
            "precision_at_top": float(fixed[top].mean()) if top.any() else None,
            "recall_at_top": (
                float(fixed[top].sum() / fixed.sum()) if fixed.sum() else None
            ),
        },
        "avoiding_harmful_acquisitions": {
            "positive_event": "audio broke a correct text prediction",
            "positives": int(harmed.sum()),
            "base_rate": float(harmed.mean()),
            "auroc_low_score_predicts_harm": (
                1.0 - harm_auroc if harm_auroc is not None else None
            ),
            "harms_in_top": int(harmed[top].sum()),
            "note": (
                "Reported as 1 - AUROC because a useful score should give harmful "
                "acquisitions a LOW value. Above 0.5 means it does."
            ),
        },
        "rank_agreement": {
            "spearman_vs_signed_gain": spearman(scores, realised),
            "spearman_vs_own_target": spearman(scores, target_values),
            "spearman_vs_macro_f1_marginal": spearman(scores, macro_f1_marginal_values),
        },
        "calibration": calibration_of_gain(scores, realised),
        "calibration_note": (
            "Reported for completeness and deliberately NOT part of the selection "
            "rule: the policy acquires top-k, so only the ordering of the score is "
            "used and a constant offset would not change a single decision."
        ),
        "class_concentration": class_concentration(scores, outcomes, fraction),
        "net_value_at_top": {
            "acquisitions": int(top.sum()),
            "corrections": int(fixed[top].sum()),
            "harms": int(harmed[top].sum()),
            "net": int(fixed[top].sum() - harmed[top].sum()),
            "wasted": int(
                (top & (outcomes.text_prediction == outcomes.fused_prediction)).sum()
            ),
        },
    }


def compare_against_uncertainty(
    per_target: Mapping[str, Mapping], reference: str = "uncertainty_only"
) -> dict:
    """The comparison Phase D exists to make: is any target better than the gate?

    Stage 3 selected ``uncertainty_only`` as its feature set after a richer
    evidence model failed to beat it out of fold. If that repeats here, the
    honest conclusion is that the routing signal on this corpus is uncertainty
    and nothing more -- which is a finding, not a failure to find one.
    """
    baseline = per_target.get(reference)
    if baseline is None:
        return {"available": False, "reason": f"no {reference} row to compare against"}

    def value(record: Mapping, *path: str):
        node = record
        for key in path:
            node = (node or {}).get(key)
        return node

    rows = {}
    for name, record in per_target.items():
        if name == reference:
            continue
        rows[name] = {
            "auroc_delta": _delta(
                value(record, "identifying_useful_acquisitions", "auroc"),
                value(baseline, "identifying_useful_acquisitions", "auroc"),
            ),
            "auprc_delta": _delta(
                value(record, "identifying_useful_acquisitions", "auprc"),
                value(baseline, "identifying_useful_acquisitions", "auprc"),
            ),
            "spearman_vs_signed_gain_delta": _delta(
                value(record, "rank_agreement", "spearman_vs_signed_gain"),
                value(baseline, "rank_agreement", "spearman_vs_signed_gain"),
            ),
            "spearman_vs_macro_f1_marginal_delta": _delta(
                value(record, "rank_agreement", "spearman_vs_macro_f1_marginal"),
                value(baseline, "rank_agreement", "spearman_vs_macro_f1_marginal"),
            ),
            "majority_over_representation": value(
                record, "class_concentration", "majority_over_representation"
            ),
            "reference_majority_over_representation": value(
                baseline, "class_concentration", "majority_over_representation"
            ),
        }
    return {
        "available": True,
        "reference": reference,
        "reference_auroc": value(baseline, "identifying_useful_acquisitions", "auroc"),
        "per_target": rows,
        "why": (
            "Stage 3 found that a ten-feature evidence model reduced to the "
            "one-feature uncertainty model out of fold. Every richer candidate here "
            "is therefore reported as a delta against uncertainty alone, so a "
            "target that merely reproduces the gate cannot be presented as an "
            "improvement on it."
        ),
    }


def _delta(left, right):
    if left is None or right is None:
        return None
    return float(left) - float(right)


def feature_set_ablation(records: Sequence[Mapping]) -> dict:
    """Does a richer feature set help, holding the target fixed?

    This is the Stage 3 reproduction. It is a separate question from which
    target to regress, and conflating the two is how a routing experiment ends
    up unable to say which choice bought what.
    """
    rows = {
        record["feature_set"]: {
            "auroc": (record.get("quality") or {})
                .get("identifying_useful_acquisitions", {}).get("auroc"),
            "spearman_vs_signed_gain": (record.get("quality") or {})
                .get("rank_agreement", {}).get("spearman_vs_signed_gain"),
            "features": record.get("features"),
            "feature_count": len(record.get("features") or []),
        }
        for record in records
    }
    usable = {
        name: row for name, row in rows.items() if row["auroc"] is not None
    }
    best = max(usable, key=lambda name: usable[name]["auroc"]) if usable else None
    simplest = min(rows, key=lambda name: rows[name]["feature_count"]) if rows else None
    richer_helps = bool(
        best and simplest and best != simplest
        and usable.get(best, {}).get("auroc", 0)
        > (usable.get(simplest, {}).get("auroc") or 0) + 0.01
    )
    return {
        "per_feature_set": rows,
        "best_by_auroc": best,
        "simplest": simplest,
        "richer_features_help": richer_helps,
        "stage3_finding": (
            "Stage 3 fitted evidence_only and evidence_plus_class alongside "
            "uncertainty_only and selected uncertainty_only: the extra nine "
            "features did not improve out-of-fold quality."
        ),
        "reproduced": (
            None if best is None else
            "yes -- the richer feature sets again fail to clear uncertainty alone "
            "by 0.01 AUROC" if not richer_helps else
            f"no -- {best} clears the simplest feature set by more than 0.01 AUROC "
            f"on this pool, unlike in Stage 3"
        ),
    }
