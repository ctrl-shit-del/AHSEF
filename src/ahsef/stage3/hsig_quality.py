"""PHASE D (evaluation) -- is the gain estimate any good, and better than what?

The goal here is explicitly *not* classifier accuracy.  HSIG's job is to order
samples by how much acquiring audio would help, so the metrics are the ones that
measure ordering and magnitude:

``AUROC``
    can the estimate separate acquisitions that helped from acquisitions that
    did not?

``AUPRC`` (with the base rate stated beside it)
    the same question at the operating point that matters when the positive
    event is rare.  On an imbalanced target AUROC alone flatters a model that
    would be useless in practice, so both are always reported together.

``Calibration of the predicted gain``
    mean absolute and squared error of the predicted gain against the realised
    signed gain, plus a decile table.  A model that ranks well but is offset by
    0.3 will still route badly once a cost penalty is subtracted from it.

``Spearman correlation``
    predicted against observed, as a scale-free check on the ranking.

Every number is computed from **out-of-fold** predictions, and every number is
computed identically for two deliberately weak references -- a constant at the
base rate, and uncertainty alone -- so the report always answers "better than
what?".  Bootstrap intervals are included because at n=509 with roughly one
positive in six, a point estimate on its own would invite a conclusion the data
cannot carry.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.ahsef.stage3.hsig_model import GainTargets

BOOTSTRAP_RESAMPLES = 2000


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    raw = np.empty(values.size, dtype=float)
    raw[order] = np.arange(1, values.size + 1, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size)
    np.add.at(sums, inverse, raw)
    return (sums / counts)[inverse]


def auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank-based AUROC, tie-corrected.  ``None`` when one class is absent."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive).astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0 or scores.size < 3:
        return None
    ranked = _ranks(scores)
    return float(
        (ranked[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def auprc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    """Average precision, computed directly so no extra dependency is needed."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive).astype(bool)
    if positive.sum() == 0 or scores.size == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    labels = positive[order]
    cumulative_tp = np.cumsum(labels)
    precision = cumulative_tp / np.arange(1, labels.size + 1)
    return float((precision * labels).sum() / labels.sum())


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(np.corrcoef(_ranks(left), _ranks(right))[0, 1])
    return None if np.isnan(value) else value


def _bootstrap(
    statistic, scores: np.ndarray, outcome: np.ndarray, seed: int = 42,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    n = scores.size
    values = []
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        value = statistic(scores[index], outcome[index])
        if value is not None:
            values.append(value)
    if len(values) < resamples // 10:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def calibration_of_gain(
    predicted: np.ndarray, observed: np.ndarray, bins: int = 10
) -> dict:
    """How far the predicted gain sits from the realised signed gain."""
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    error = predicted - observed
    edges = np.quantile(predicted, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for lower, upper in zip(edges, edges[1:]):
        last = upper == edges[-1]
        inside = (
            (predicted >= lower) & (predicted <= upper) if last
            else (predicted >= lower) & (predicted < upper)
        )
        count = int(inside.sum())
        if not count:
            continue
        rows.append({
            "lower": float(lower), "upper": float(upper), "samples": count,
            "mean_predicted_gain": float(predicted[inside].mean()),
            "mean_observed_gain": float(observed[inside].mean()),
            "residual": float(predicted[inside].mean() - observed[inside].mean()),
        })
    return {
        "mean_predicted_gain": float(predicted.mean()),
        "mean_observed_gain": float(observed.mean()),
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "bins": rows,
        "note": (
            "Bins are quantiles of the predicted gain, so each holds a comparable "
            "number of samples. A monotone mean_observed_gain across bins is the "
            "property routing depends on; the residual column says whether the "
            "magnitude is also usable once a cost penalty is subtracted."
        ),
    }


def estimator_quality(
    predicted: np.ndarray,
    targets: GainTargets,
    name: str,
    seed: int = 42,
    bins: int = 10,
) -> dict:
    """The full quality record for one out-of-fold gain estimate."""
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(predicted)
    if not finite.all():
        predicted = predicted[finite]
    y_gain = targets.y_gain[finite]
    y_harm = targets.y_harm[finite]
    signed = targets.signed_gain[finite]

    base_rate = float(y_gain.mean()) if y_gain.size else None
    return {
        "estimator": name,
        "samples_scored": int(finite.sum()),
        "samples_unscored": int((~finite).sum()),
        "identifying_useful_acquisitions": {
            "positive_event": "y_gain -- audio fixed a wrong text prediction",
            "positives": int(y_gain.sum()),
            "base_rate": base_rate,
            "auroc": auroc(predicted, y_gain),
            "auroc_ci95": _bootstrap(auroc, predicted, y_gain, seed),
            "auprc": auprc(predicted, y_gain),
            "auprc_ci95": _bootstrap(auprc, predicted, y_gain, seed),
            "auprc_baseline_is_base_rate": base_rate,
            "imbalance_note": (
                "AUPRC is compared against the base rate, not against 0.5. An AUPRC "
                "at or below the base rate means the estimator adds nothing at the "
                "operating point regardless of its AUROC."
            ),
        },
        "avoiding_harmful_acquisitions": {
            "positive_event": "y_harm -- audio broke a correct text prediction",
            "positives": int(y_harm.sum()),
            "base_rate": float(y_harm.mean()) if y_harm.size else None,
            "auroc_low_gain_predicts_harm": (
                None if auroc(predicted, y_harm) is None else 1.0 - auroc(predicted, y_harm)
            ),
            "note": (
                "Reported as 1 - AUROC because a *useful* estimator should give "
                "harmful acquisitions a LOW predicted gain. Values above 0.5 mean it "
                "does."
            ),
        },
        "predicted_vs_observed": {
            "spearman_rho": spearman(predicted, signed),
            "pearson_r": (
                None if np.unique(predicted).size < 2 or np.unique(signed).size < 2
                else float(np.corrcoef(predicted, signed)[0, 1])
            ),
            "target": "signed_gain = fused_correct - text_correct",
        },
        "calibration": calibration_of_gain(predicted, signed, bins),
        "distribution": {
            "mean": float(predicted.mean()) if predicted.size else None,
            "std": float(predicted.std(ddof=0)) if predicted.size else None,
            "min": float(predicted.min()) if predicted.size else None,
            "max": float(predicted.max()) if predicted.size else None,
            "fraction_positive": float((predicted > 0).mean()) if predicted.size else None,
        },
    }


def compare_estimators(records: Mapping[str, dict], reference_names: Sequence[str]) -> dict:
    """Rank estimators and state plainly whether HSIG beat the weak references."""
    def score(entry: dict) -> float:
        value = entry["identifying_useful_acquisitions"]["auroc"]
        return -np.inf if value is None else value

    ranked = sorted(records, key=lambda name: score(records[name]), reverse=True)
    references = [name for name in reference_names if name in records]
    best_reference = max(references, key=lambda name: score(records[name])) \
        if references else None
    candidates = [name for name in records if name not in reference_names]
    best_candidate = max(candidates, key=lambda name: score(records[name])) \
        if candidates else None

    verdict = None
    if best_candidate and best_reference:
        candidate_auroc = score(records[best_candidate])
        reference_auroc = score(records[best_reference])
        interval = records[best_candidate]["identifying_useful_acquisitions"]["auroc_ci95"]
        beats = candidate_auroc > reference_auroc
        separated = bool(interval and interval[0] > reference_auroc)
        verdict = {
            "best_candidate": best_candidate,
            "best_candidate_auroc": None if candidate_auroc == -np.inf else candidate_auroc,
            "best_reference": best_reference,
            "best_reference_auroc": None if reference_auroc == -np.inf else reference_auroc,
            "candidate_beats_reference_point_estimate": bool(beats),
            "candidate_ci95_excludes_reference": separated,
            "statement": (
                f"{best_candidate} reaches AUROC "
                f"{candidate_auroc:.4f} against {best_reference}'s {reference_auroc:.4f}; "
                + (
                    "the 95% bootstrap interval of the candidate excludes the "
                    "reference, so the improvement is not attributable to resampling "
                    "noise alone."
                    if separated else
                    "the 95% bootstrap interval of the candidate CONTAINS the "
                    "reference value, so this pool cannot establish that the learned "
                    "estimator beats the simple reference. The routing result must be "
                    "read with that in mind."
                )
            ) if candidate_auroc != -np.inf and reference_auroc != -np.inf else
            "AUROC was undefined for at least one estimator on this pool.",
        }
    return {
        "ranked_by_auroc": ranked,
        "references": references,
        "verdict": verdict,
        "criterion": "AUROC of the out-of-fold predicted gain against y_gain",
    }


def feature_importance(components: Mapping[str, Mapping]) -> dict:
    """Standardised coefficients, which are the importances for a linear model."""
    report = {}
    for role, record in components.items():
        pairs = dict(zip(record["feature_names"], record["coefficients"]))
        report[role] = {
            "coefficients": pairs,
            "ranked_by_absolute_weight": sorted(
                pairs, key=lambda name: abs(pairs[name]), reverse=True
            ),
            "intercept": record["intercept"],
        }
    if {"text_correct", "fused_correct"} <= set(components):
        text = dict(zip(
            components["text_correct"]["feature_names"],
            components["text_correct"]["coefficients"],
        ))
        fused = dict(zip(
            components["fused_correct"]["feature_names"],
            components["fused_correct"]["coefficients"],
        ))
        difference = {name: fused[name] - text[name] for name in text}
        report["gain_direction"] = {
            "coefficients": difference,
            "ranked_by_absolute_weight": sorted(
                difference, key=lambda name: abs(difference[name]), reverse=True
            ),
            "note": (
                "The difference of the two components' weights. A positive entry "
                "means that raising the feature raises the predicted gain -- this is "
                "what actually drives the acquisition decision."
            ),
        }
    report["interpretation"] = (
        "Inputs are standardised before fitting, so a coefficient is the change in "
        "log-odds per standard deviation of that feature and the magnitudes are "
        "directly comparable."
    )
    return report


def class_rule_audit(
    predicted: np.ndarray,
    predicted_classes: np.ndarray,
    class_names: Sequence[str],
    uses_class_features: bool = True,
) -> dict:
    """Check the estimator has not degenerated into "always acquire for class X".

    Conditioning on class identity is permitted, but a model whose gain is
    essentially a lookup on the predicted class is a hard-coded routing rule
    with a fitted-model label on it.  The spread of predicted gain *within* each
    class is what distinguishes the two.
    """
    predicted = np.asarray(predicted, dtype=float)
    classes = np.asarray(predicted_classes, dtype=int)
    finite = np.isfinite(predicted)
    rows = {}
    for index, name in enumerate(class_names):
        mask = finite & (classes == index)
        count = int(mask.sum())
        rows[name] = {
            "samples": count,
            "mean_predicted_gain": float(predicted[mask].mean()) if count else None,
            "std_predicted_gain": float(predicted[mask].std(ddof=0)) if count else None,
            "min": float(predicted[mask].min()) if count else None,
            "max": float(predicted[mask].max()) if count else None,
        }
    within = [row["std_predicted_gain"] for row in rows.values()
              if row["std_predicted_gain"] is not None]
    between = [row["mean_predicted_gain"] for row in rows.values()
               if row["mean_predicted_gain"] is not None]
    mean_within = float(np.mean(within)) if within else None
    spread_between = float(np.std(between, ddof=0)) if len(between) > 1 else None
    if not uses_class_features:
        verdict = (
            "The estimator's feature set contains no class indicator, so it cannot be "
            "a class lookup as a matter of construction: it never sees the predicted "
            "class. Any between-class variation below is an image of how uncertainty "
            "itself differs across the classes the LLM predicts, not a rule keyed on "
            "the class."
        )
    elif mean_within and spread_between and mean_within > spread_between:
        verdict = (
            "The estimator varies substantially within each predicted class, so it is "
            "not a class lookup."
        )
    else:
        verdict = (
            "Predicted gain varies more BETWEEN predicted classes than within them. "
            "The estimator is close to a class-conditional constant and should be "
            "read as one; the feature-set comparison in this report says whether "
            "dropping class identity changes the routing outcome."
        )
    return {
        "uses_class_features": bool(uses_class_features),
        "per_predicted_class": rows,
        "mean_within_class_std": mean_within,
        "between_class_std": spread_between,
        "within_to_between_ratio": (
            None if not mean_within or not spread_between else mean_within / spread_between
        ),
        "verdict": verdict,
        "why_this_matters": (
            "Stage 3 forbids a hard-coded 'if class X then acquire' rule. A learned "
            "model that reduces to that rule is the same thing, so it is measured "
            "rather than assumed away."
        ),
    }
