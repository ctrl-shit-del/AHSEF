"""The declared selection rules: which HSIG variant is frozen, and why.

A selection rule that can be restated after seeing the numbers is not a rule.
These tests pin the two Stage 3 rules to properties of the estimators rather
than to any particular result.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ahsef.cli.stage3_stages import HSIG_SELECTION_RULE, _select_hsig_variant
from src.ahsef.stage3.features import FEATURE_SETS, UNCERTAINTY_ONLY
from src.ahsef.stage3.hsig_model import Stage3HSIG, targets_from_oracle
from src.ahsef.stage3.objective import SELECTED_OBJECTIVE


def variant(auroc: float, interval):
    return {
        "identifying_useful_acquisitions": {"auroc": auroc, "auroc_ci95": list(interval)},
    }


def test_uncertainty_only_is_a_candidate_not_merely_a_reference():
    """If one feature predicts the gain best, one feature is what gets frozen."""
    assert "uncertainty_only" in FEATURE_SETS
    assert FEATURE_SETS["uncertainty_only"] == UNCERTAINTY_ONLY == ("uncertainty",)


def test_highest_auroc_wins_when_nothing_is_tied():
    usable = {
        "fix_logistic__a": variant(0.90, (0.88, 0.92)),
        "paired_logistic__b": variant(0.60, (0.55, 0.65)),
    }
    chosen, rule = _select_hsig_variant(usable)
    assert chosen == "fix_logistic__a"
    assert rule["best_by_auroc"] == "fix_logistic__a"
    assert rule["auroc_conceded_to_the_tie_break"] == pytest.approx(0.0)


def test_paired_logistic_wins_a_statistical_tie():
    """The tie-break is a units argument, not a preference for a bigger number."""
    usable = {
        "fix_logistic__a": variant(0.7564, (0.662, 0.839)),
        "paired_logistic__a": variant(0.7555, (0.659, 0.839)),
    }
    chosen, rule = _select_hsig_variant(usable)
    assert chosen == "paired_logistic__a"
    assert rule["best_by_auroc"] == "fix_logistic__a"
    assert rule["auroc_conceded_to_the_tie_break"] == pytest.approx(0.0009, abs=1e-6)
    assert "category error" in rule["why_paired_wins_ties"]


def test_the_tie_break_cannot_rescue_a_clearly_worse_variant():
    usable = {
        "fix_logistic__a": variant(0.90, (0.88, 0.92)),
        "paired_logistic__b": variant(0.50, (0.45, 0.55)),
    }
    chosen, _ = _select_hsig_variant(usable)
    assert chosen == "fix_logistic__a"


def test_the_rule_is_stated_without_reference_to_results():
    assert "highest out-of-fold AUROC" in HSIG_SELECTION_RULE["primary"]
    assert "paired_logistic" in HSIG_SELECTION_RULE["tie_break"]
    # The justification must be about the estimator's output, not its score.
    why = HSIG_SELECTION_RULE["why_paired_wins_ties"]
    assert "DIFFERENCE of probabilities" in why
    assert "cannot express harm" in why


def test_paired_and_fix_really_do_differ_in_range():
    """The units argument has to be true of the estimators, not just asserted."""
    from src.ahsef.fusion import FusionSpec
    from src.ahsef.stage3.features import build_features
    from src.ahsef.stage3.oracle import build_oracle_table, fuse
    from src.ahsef.tests.stage3_fixtures import scenario

    data = scenario(n=140, seed=17)
    spec = FusionSpec(
        method="weighted_probability", weights={"text_llm": 0.5, "audio": 0.5},
        selected_on_split="validation",
    )
    fused = fuse(data["text"], data["audio"], spec, data["ids"])
    oracle = build_oracle_table(data["text"], data["audio"], fused, data["ids"])
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(data["text"].frame, uncertainty, "uncertainty_only")
    targets = targets_from_oracle(oracle, "validation")

    paired = Stage3HSIG.fit(
        features, targets, feature_set="uncertainty_only",
        estimator_type="paired_logistic",
    ).predict_frame(features)["predicted_gain"]
    fix = Stage3HSIG.fit(
        features, targets, feature_set="uncertainty_only", estimator_type="fix_logistic",
    ).predict_frame(features)["predicted_gain"]

    assert fix.min() >= 0.0, "fix_logistic cannot express harm"
    assert paired.min() < 0.0, "paired_logistic must be able to express harm"


def test_the_gate_objective_is_declared_not_derived():
    from src.ahsef.stage3.objective import OBJECTIVE_RATIONALE

    assert OBJECTIVE_RATIONALE["declared"] == "in advance of inspecting any Stage 3 " \
        "routing result"
    assert OBJECTIVE_RATIONALE["selected"] == SELECTED_OBJECTIVE
    for key in ("why_not_accuracy", "why_not_balanced_accuracy", "why_not_macro_f1_alone"):
        assert OBJECTIVE_RATIONALE[key]
