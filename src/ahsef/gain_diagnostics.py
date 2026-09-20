"""Is dU actually a usable information-gain signal for routing?

AHSEF's specification defines information gain as uncertainty reduction::

    dU(m | A) = U(A) - U(A u {m})

and HSIG's job in stage 2 is to predict it.  Before building a predictor for a
quantity, it is worth checking that the quantity means what the design assumes
it means -- namely, that a large dU indicates acquiring ``m`` was worth doing.

This module measures that directly, per sample, on the pairs that are actually
fusable.  It answers three questions:

1. **Sign stability.**  Does dU have a consistent sign for a pair that helps?
2. **Per-sample agreement.**  Does dU correlate with the acquisition actually
   improving the decision?
3. **Ranking.**  If several candidates are available, does ordering them by
   mean dU order them the same way as ordering by realised accuracy gain?

If dU fails these, HSIG must be given a different target -- and that is a
finding about the design, not a bug in the code.  Nothing here is consulted by
a routing decision; it is post-hoc analysis and it reads labels.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def improvement_column(frame: pd.DataFrame) -> pd.Series:
    """``+1`` where acquisition fixed the prediction, ``-1`` where it broke it."""
    before = frame["predicted_before"] == frame["true_class"]
    after = frame["predicted_after"] == frame["true_class"]
    return after.astype(int) - before.astype(int)


def diagnose(frame: pd.DataFrame, name: str) -> dict:
    """Every number needed to judge dU as a routing signal, for one pair."""
    improvement = improvement_column(frame)
    before = frame["predicted_before"] == frame["true_class"]
    after = frame["predicted_after"] == frame["true_class"]
    delta = frame["delta_uncertainty"]

    # Pearson correlation is undefined when either side is constant; that
    # itself is informative (a pair where dU never changes sign), so it is
    # reported as None rather than silently zero.
    correlation = (
        float(delta.corr(improvement.astype(float)))
        if delta.nunique() > 1 and improvement.nunique() > 1
        else None
    )
    helped = delta[improvement > 0]
    hurt = delta[improvement < 0]
    return {
        "name": name,
        "samples": int(len(frame)),
        "mean_delta_uncertainty": float(delta.mean()),
        "positive_delta_rate": float((delta > 0).mean()),
        "accuracy_before": float(before.mean()),
        "accuracy_after": float(after.mean()),
        "accuracy_gain": float(after.mean() - before.mean()),
        "fixed": int((~before & after).sum()),
        "broken": int((before & ~after).sum()),
        "correlation_delta_vs_improvement": correlation,
        "mean_delta_when_fixed": float(helped.mean()) if len(helped) else None,
        "mean_delta_when_broken": float(hurt.mean()) if len(hurt) else None,
        "sign_agrees_with_accuracy_gain": bool(
            (delta.mean() > 0) == (after.mean() - before.mean() > 0)
        ),
    }


def rank_agreement(rows: Sequence[Mapping]) -> dict:
    """Do candidates rank the same by mean dU and by realised accuracy gain?"""
    if len(rows) < 2:
        return {
            "comparable": False,
            "reason": "ranking needs at least two candidates over the same anchor",
        }
    by_delta = [row["name"] for row in
                sorted(rows, key=lambda row: row["mean_delta_uncertainty"], reverse=True)]
    by_gain = [row["name"] for row in
               sorted(rows, key=lambda row: row["accuracy_gain"], reverse=True)]
    return {
        "comparable": True,
        "ranked_by_mean_delta_uncertainty": by_delta,
        "ranked_by_realised_accuracy_gain": by_gain,
        "rankings_agree": by_delta == by_gain,
        "top_choice_agrees": by_delta[0] == by_gain[0],
    }


def verdict(rows: Sequence[Mapping]) -> dict:
    """A plain statement of whether dU can serve as HSIG's target."""
    correlations = [
        abs(row["correlation_delta_vs_improvement"])
        for row in rows if row["correlation_delta_vs_improvement"] is not None
    ]
    sign_failures = [
        row["name"] for row in rows if not row["sign_agrees_with_accuracy_gain"]
    ]
    strongest = max(correlations) if correlations else None
    usable = bool(correlations) and strongest is not None and strongest >= 0.3 and not sign_failures
    return {
        "max_abs_correlation_with_improvement": strongest,
        "pairs_where_delta_sign_contradicts_accuracy_gain": sign_failures,
        "delta_uncertainty_is_a_usable_hsig_target": usable,
        "recommendation": (
            "dU may be used as HSIG's regression target."
            if usable else
            "Do not train HSIG to predict dU. Its sign depends on which model is the "
            "anchor and on the fusion rule, and it carries little per-sample "
            "information about whether the acquisition improved the decision. Use an "
            "improvement-linked target estimated on validation instead -- for example "
            "the change in the probability assigned to the eventually predicted class, "
            "or a validation-fitted estimate of P(correct after) - P(correct before)."
        ),
    }


def render(rows: Sequence[Mapping]) -> str:
    header = (
        f"{'pair / rule':<28}{'n':>6}{'mean dU':>10}{'dU>0':>8}"
        f"{'acc before':>12}{'acc after':>11}{'gain':>9}{'fixed':>7}{'broke':>7}{'corr':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        correlation = row["correlation_delta_vs_improvement"]
        lines.append(
            f"{row['name']:<28}{row['samples']:>6}{row['mean_delta_uncertainty']:>+10.4f}"
            f"{row['positive_delta_rate']:>8.3f}{row['accuracy_before']:>12.4f}"
            f"{row['accuracy_after']:>11.4f}{row['accuracy_gain']:>+9.4f}"
            f"{row['fixed']:>7}{row['broken']:>7}"
            + (f"{correlation:>9.4f}" if correlation is not None else f"{'n/a':>9}")
        )
    return "\n".join(lines)
