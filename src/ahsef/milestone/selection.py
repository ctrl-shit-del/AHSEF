"""PHASE D/E -- the selection and decision rules, written down before the run.

Both rules in this module are the reason the Phase D/E result can be believed at
all.  A target chosen after seeing which one produced the nicest budget curve is
a hyperparameter fitted to the evaluation, and reporting it as a discovery would
be misreporting.  So the rule is code, it is committed before the driver is run,
and the driver applies it mechanically to whatever numbers come back.

The selection rule
------------------
Candidates are ranked on **validation macro-F1 averaged over the low-budget
operating region** -- the 5%, 10%, 15%, 20% and 25% points on the Phase E grid.
Three reasons for that choice, all fixed in advance:

* Macro-F1, not accuracy.  Stage 2 established that an accuracy-shaped objective
  on this corpus selects a neutral-heavy operating point, and Phase D exists
  partly to test whether Stage 3 inherited that problem through its target.
* The **low** budgets, because the AHSEF claim is about getting most of the
  fusion benefit while activating audio rarely.  At 75% every policy has
  acquired nearly everything and the curves converge by construction; averaging
  those points in would dilute the only region where the policies differ.
* An **average** over five budgets rather than a single point, because a single
  budget cannot distinguish a real ordering from a lucky threshold -- which is
  precisely the criticism Phase E was created to answer for Stage 3.

Admissibility comes first: a candidate that never clears the matched-budget
random control's 97.5th percentile at any interior budget is not ranked at all.
A policy that sits inside the random interval everywhere has not been shown to
route, whatever its absolute macro-F1 looks like.

Parsimony breaks near-ties, in favour of the *simpler* signal.  If a fitted
target does not clear the plain uncertainty ranking by ``PARSIMONY_MARGIN``, the
uncertainty ranking wins.  This encodes the Stage 3 lesson directly: a richer
evidence model that merely reproduces the gate should be reported as reproducing
the gate, not shipped as an improvement over it.

The decision rule
-----------------
``MATERIAL_ROUTING_SUCCESS`` has four conditions and all four must hold.  The
second one -- that the improvement is not explained solely by majority-class
corrections -- is the one that makes this different from Stage 3's checkpoint,
and it is checked by asking whether minority-class F1 also rose, not by asking
whether macro-F1 happened to go up.

Beating always-fusion is explicitly **not** required.  Always-fusion pays for
audio on every sample; the claim under test is that most of its benefit survives
paying on few.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from src.common.labels import CANONICAL_EMOTION_CLASSES

#: The budgets averaged by the selection criterion.  Interior and low, because
#: that is the region where selective acquisition either works or does not.
SELECTION_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)

#: Two candidates within this macro-F1 distance are treated as tied, and the
#: simpler signal wins.  0.005 is roughly a tenth of the always-fusion macro-F1
#: gain on this pool, so it is small enough to admit a real difference and large
#: enough to reject one that is arithmetic noise on 509 samples.
PARSIMONY_MARGIN = 0.005

#: The signal a fitted target has to beat to justify itself.
PARSIMONY_REFERENCE = "uncertainty_only"

SELECTION_RULE = {
    "primary_criterion": (
        "mean validation macro-F1 over the low-budget operating region "
        f"{[f'{b:.0%}' for b in SELECTION_BUDGETS]}, using out-of-fold scores"
    ),
    "admissibility_gate": (
        "the policy must exceed the matched-budget random control's 97.5th "
        "percentile macro-F1 at at least one interior budget"
    ),
    "tie_break_1": (
        f"within {PARSIMONY_MARGIN} macro-F1, the simpler signal wins; a fitted "
        f"target must clear {PARSIMONY_REFERENCE} by more than that margin to be "
        f"selected over it"
    ),
    "tie_break_2": "higher retained fraction of the always-fusion macro-F1 gain at 10%",
    "tie_break_3": "alphabetical target name, so the rule is total",
    "metric": "macro_f1",
    "why_not_accuracy": (
        "Stage 2 measured that accuracy-shaped gating on this corpus buys neutral "
        "and loses macro-F1. Selecting on accuracy would reproduce that."
    ),
    "declared": "before any Phase D estimator was fitted or scored",
    "uses_test_labels": False,
    "uses_test_split": False,
}

MATERIAL_ROUTING_SUCCESS = {
    "condition_1": (
        "the selected policy's validation macro-F1 exceeds the matched-budget "
        "random control at the selected budget, above the control's 97.5th "
        "percentile"
    ),
    "condition_2": (
        "the improvement is not explained solely by majority-class corrections: "
        "summed F1 over the classes outside the two largest must also rise "
        "against text-only"
    ),
    "condition_3": (
        "the policy retains at least "
        f"{0.5:.0%} of the always-fusion macro-F1 gain over text-only"
    ),
    "condition_4": (
        "the policy acquires audio on at most half as many samples as "
        "always-fusion"
    ),
    "not_required": (
        "beating always-fusion. Always-fusion pays for every sample; the claim "
        "under test is that most of its benefit survives paying for few."
    ),
    "if_not_met": "report ROUTING SIGNAL INSUFFICIENT and do not open the locked test split",
    "declared": "before any Phase D estimator was fitted or scored",
}

#: Thresholds named in ``MATERIAL_ROUTING_SUCCESS``, as numbers the code applies.
MIN_RETAINED_FUSION_GAIN = 0.50
MAX_ACQUISITION_SHARE_OF_ALWAYS_FUSION = 0.50

#: Subtracting two floats that differ by exactly the threshold can land a few
#: ulps below it -- 0.35 - 0.30 is 0.04999999999999999 -- and a pre-declared bar
#: that rejects the value it names is a bug, not a strict rule.
THRESHOLD_TOLERANCE = 1e-9


class SelectionError(RuntimeError):
    """Raised when the selection rule cannot be applied as declared."""


def _rows_by_budget(curve: Sequence[Mapping]) -> dict[float, Mapping]:
    return {round(float(row["requested_budget"]), 6): row for row in curve}


def selection_score(curve: Sequence[Mapping]) -> dict:
    """Mean macro-F1 over :data:`SELECTION_BUDGETS`, and the points behind it."""
    rows = _rows_by_budget(curve)
    missing = [b for b in SELECTION_BUDGETS if round(b, 6) not in rows]
    if missing:
        raise SelectionError(
            f"The selection rule averages macro-F1 over {SELECTION_BUDGETS} but the "
            f"curve has no row at {missing}. The rule is not applicable to a curve "
            f"computed on a different grid."
        )
    points = {
        f"{budget:.2f}": float(rows[round(budget, 6)]["macro_f1"])
        for budget in SELECTION_BUDGETS
    }
    return {
        "mean_macro_f1_over_selection_budgets": float(np.mean(list(points.values()))),
        "points": points,
        "budgets": [float(b) for b in SELECTION_BUDGETS],
    }


def rank_candidates(
    curves: Mapping[str, Sequence[Mapping]],
    beats_random: Mapping[str, Mapping],
    always_fusion_macro_f1: float,
    text_only_macro_f1: float,
    reference: str = PARSIMONY_REFERENCE,
) -> dict:
    """Apply :data:`SELECTION_RULE` to the Phase E curves and name a winner."""
    considered = {
        name: curve for name, curve in curves.items() if name not in ("oracle",)
    }
    if not considered:
        raise SelectionError("No candidate policies were supplied to rank")

    fusion_gain = always_fusion_macro_f1 - text_only_macro_f1
    scored = {}
    for name, curve in considered.items():
        score = selection_score(curve)
        rows = _rows_by_budget(curve)
        at_ten = rows.get(0.10)
        verdict = beats_random.get(name) or {}
        scored[name] = {
            **score,
            "admissible": bool(verdict.get("any_budget_above_random")),
            "budgets_above_random": list(verdict.get("budgets_above_random") or []),
            "macro_f1_at_10pct": float(at_ten["macro_f1"]) if at_ten else None,
            "retained_fusion_gain_at_10pct": (
                float((at_ten["macro_f1"] - text_only_macro_f1) / fusion_gain)
                if at_ten and fusion_gain else None
            ),
        }

    admissible = {name: row for name, row in scored.items() if row["admissible"]}
    if not admissible:
        return {
            "rule": dict(SELECTION_RULE),
            "per_candidate": scored,
            "admissible": [],
            "selected": None,
            "outcome": "ROUTING SIGNAL INSUFFICIENT",
            "reason": (
                "No candidate exceeded the matched-budget random control's 97.5th "
                "percentile macro-F1 at any interior budget. Every apparent gain on "
                "this pool was available by acquiring at random at the same rate."
            ),
        }

    best = max(
        admissible,
        key=lambda name: (
            admissible[name]["mean_macro_f1_over_selection_budgets"],
            admissible[name]["retained_fusion_gain_at_10pct"] or 0.0,
            # Negated lexicographic order so the max picks the alphabetically
            # first name when everything above ties exactly.
            tuple(-ord(character) for character in name),
        ),
    )

    applied_parsimony = False
    parsimony_note = None
    reference_row = admissible.get(reference)
    if reference_row is not None and best != reference:
        margin = (
            admissible[best]["mean_macro_f1_over_selection_budgets"]
            - reference_row["mean_macro_f1_over_selection_budgets"]
        )
        if margin <= PARSIMONY_MARGIN:
            parsimony_note = (
                f"{best} leads {reference} by {margin:+.4f} macro-F1, inside the "
                f"declared {PARSIMONY_MARGIN} tie margin, so the simpler signal is "
                f"selected. A fitted target that merely reproduces the uncertainty "
                f"gate is reported as reproducing it."
            )
            best = reference
            applied_parsimony = True

    return {
        "rule": dict(SELECTION_RULE),
        "per_candidate": scored,
        "admissible": sorted(admissible),
        "ranking": sorted(
            admissible,
            key=lambda name: -admissible[name]["mean_macro_f1_over_selection_budgets"],
        ),
        "selected": best,
        "parsimony_applied": applied_parsimony,
        "parsimony_note": parsimony_note,
        "outcome": "TARGET SELECTED",
    }


# ============================================================
# The Phase D/E decision
# ============================================================

def minority_f1_mass(row: Mapping, majority_classes: Sequence[str]) -> float:
    """Summed F1 over every class outside the two largest.

    Summed rather than averaged so a class that is absent from the pool cannot
    move the number by changing the denominator.
    """
    per_class = row.get("per_class_f1") or {}
    return float(sum(
        value for name, value in per_class.items() if name not in set(majority_classes)
    ))


def majority_classes_of(true_class: np.ndarray, num_classes: int = 7) -> list[str]:
    counts = np.bincount(np.asarray(true_class, dtype=int), minlength=num_classes)
    ranked = np.argsort(-counts, kind="stable")
    return [
        CANONICAL_EMOTION_CLASSES[index] for index in ranked[:2] if counts[index] > 0
    ]


def routing_decision(
    policy_row: Mapping,
    random_row: Mapping,
    text_only_row: Mapping,
    always_fusion_row: Mapping,
    majority_classes: Sequence[str],
) -> dict:
    """Apply :data:`MATERIAL_ROUTING_SUCCESS` at one operating point."""
    fusion_gain = always_fusion_row["macro_f1"] - text_only_row["macro_f1"]
    policy_gain = policy_row["macro_f1"] - text_only_row["macro_f1"]
    retained = policy_gain / fusion_gain if fusion_gain else None

    control = random_row.get("macro_f1") or {}
    random_mean = control.get("mean")
    random_upper = control.get("p97.5")

    minority_policy = minority_f1_mass(policy_row, majority_classes)
    minority_text = minority_f1_mass(text_only_row, majority_classes)

    beats_random = bool(
        random_upper is not None and policy_row["macro_f1"] > random_upper
    )
    minority_improves = bool(minority_policy > minority_text)
    retains_gain = bool(
        retained is not None
        and retained >= MIN_RETAINED_FUSION_GAIN - THRESHOLD_TOLERANCE
    )
    cheaper = bool(
        always_fusion_row["acquired"]
        and policy_row["acquired"]
        <= MAX_ACQUISITION_SHARE_OF_ALWAYS_FUSION * always_fusion_row["acquired"]
        + THRESHOLD_TOLERANCE
    )
    material = bool(beats_random and minority_improves and retains_gain and cheaper)

    return {
        "rule": dict(MATERIAL_ROUTING_SUCCESS),
        "operating_point": {
            "budget": policy_row["requested_budget"],
            "acquired": policy_row["acquired"],
            "acquisition_rate": policy_row["acquisition_rate"],
            "macro_f1": policy_row["macro_f1"],
            "accuracy": policy_row["accuracy"],
            "weighted_f1": policy_row["weighted_f1"],
        },
        "condition_1_beats_matched_random": {
            "met": beats_random,
            "policy_macro_f1": policy_row["macro_f1"],
            "random_mean": random_mean,
            "random_p97.5": random_upper,
            "margin_over_random_mean": (
                policy_row["macro_f1"] - random_mean if random_mean is not None else None
            ),
        },
        "condition_2_not_only_majority_corrections": {
            "met": minority_improves,
            "majority_classes": list(majority_classes),
            "minority_f1_mass_text_only": minority_text,
            "minority_f1_mass_policy": minority_policy,
            "delta": minority_policy - minority_text,
            "why": (
                "Stage 2 showed that a gate can raise a headline metric purely by "
                "correcting neutral. Requiring the classes outside the two largest "
                "to improve is what separates routing from majority-class polish."
            ),
        },
        "condition_3_retains_fusion_gain": {
            "met": retains_gain,
            "always_fusion_gain_over_text": fusion_gain,
            "policy_gain_over_text": policy_gain,
            "retained_fraction": retained,
            "required": MIN_RETAINED_FUSION_GAIN,
        },
        "condition_4_substantially_fewer_acquisitions": {
            "met": cheaper,
            "policy_acquisitions": policy_row["acquired"],
            "always_fusion_acquisitions": always_fusion_row["acquired"],
            "share": (
                policy_row["acquired"] / always_fusion_row["acquired"]
                if always_fusion_row["acquired"] else None
            ),
            "required_at_most": MAX_ACQUISITION_SHARE_OF_ALWAYS_FUSION,
        },
        "material_routing_success": material,
        "verdict": (
            "MATERIAL ROUTING SUCCESS" if material else "ROUTING SIGNAL INSUFFICIENT"
        ),
        "decision": (
            "PROCEED -- freeze this policy and open the locked test split in Phase F."
            if material else
            "DO NOT PROCEED to locked evaluation. The validation evidence does not "
            "support the routing claim, and opening the test split would spend the "
            "one clean measurement this project has on a policy already known not "
            "to clear its own pre-declared bar."
        ),
    }
