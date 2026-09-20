"""PHASE D and PHASE E -- choose the routing target, then draw the budget curve.

    # D: build every candidate target, fit one estimator family, score them
    # E: run the budget grid, compare against random / oracle / always-fusion
    python -m src.ahsef.cli.run_milestone_phase_de --stage all

Validation only.  The 482-sample locked test pool is never opened, never read
and never named as an input; :func:`assert_validation_only` enforces that at the
point files are loaded rather than asserting it in prose afterwards.

Everything this driver consumes is a stored artefact:

    experiments/ahsef/stage3_text_audio/predictions/text_llm__validation.parquet
    experiments/ahsef/stage3_text_audio/hsig/features_validation.parquet
    experiments/ahsef/analysis/audio_expert/audio_strong__validation.parquet
    experiments/ahsef/analysis/audio_expert/phase_c_aligned.json     (fusion spec)
    experiments/ahsef/analysis/audio_expert/phase_b1_comparison.json (frozen cost)

No model is re-scored, no LLM call is made, and nothing under stage1, stage2_llm,
stage3_text_audio or audio_strong is written.  Output goes to
``experiments/ahsef/milestone/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.fusion import FusionSpec, fuse_prediction_sets
from src.ahsef.identity import AlignmentIndex
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.milestone import MILESTONE_EXPERIMENT_ID
from src.ahsef.milestone.budget import (
    DEFAULT_BUDGETS,
    PoolOutcomes,
    RANDOM_REPEATS,
    annotate_curve,
    compare_policies,
    efficiency_summary,
)
from src.ahsef.milestone.costs import assert_encoder_is_charged, load_frozen_costs
from src.ahsef.milestone.estimator import (
    ALPHA_GRID,
    DEFAULT_FOLDS,
    MilestoneHSIG,
    out_of_fold_scores,
    select_alpha,
)
from src.ahsef.milestone.quality import (
    compare_against_uncertainty,
    feature_set_ablation,
    target_distribution,
    target_quality,
)
from src.ahsef.milestone.reproducibility import (
    PROTOCOL_VERSION,
    array_fingerprint,
    assert_locked_artefacts_unchanged,
    assert_validation_only,
    reproducibility_record,
    sequence_fingerprint,
    verify_locked_artefacts,
)
from src.ahsef.milestone.selection import (
    MATERIAL_ROUTING_SUCCESS,
    SELECTION_RULE,
    majority_classes_of,
    rank_candidates,
    routing_decision,
)
from src.ahsef.milestone.targets import (
    RoutingOutcomes,
    TARGET_NAMES,
    build_all_targets,
    describe_targets,
)
from src.ahsef.registry import DEFAULT_BASELINES
from src.ahsef.stage3.features import DEFAULT_FEATURE_SET, FEATURE_DEFINITION, FEATURE_SETS
from src.common.labels import CANONICAL_EMOTION_CLASSES

SPLIT = "validation"
OUTPUT = Path("experiments") / "ahsef" / "milestone"
ANALYSIS = Path("experiments") / "ahsef" / "analysis" / "audio_expert"
STRONG_MODALITY = "audio_strong"

#: The ridge strength is chosen once, on the target Stage 3 froze, and shared by
#: every candidate.  Selecting it per target would give each candidate a
#: regularisation tuned for it and turn a comparison of targets into a
#: comparison of five separately tuned models.
ALPHA_SELECTION_TARGET = "signed_gain"

#: The plain uncertainty ranking, carried as the reference every fitted target
#: must beat.  Not an estimator: the score IS the routing uncertainty.
UNCERTAINTY_POLICY = "uncertainty_only"

#: Ablation labels, declared here so the report and the artefact agree.
ABLATIONS = (UNCERTAINTY_POLICY,) + TARGET_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="all", choices=["all", "targets", "budget"])
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--stage3-run", default="stage3_text_audio")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--repeats", type=int, default=RANDOM_REPEATS)
    parser.add_argument("--feature-set", default=DEFAULT_FEATURE_SET,
                        choices=sorted(FEATURE_SETS))
    return parser


# ============================================================
# Inputs -- all stored, all validation
# ============================================================

def load_pool(args) -> tuple[list[str], dict, dict]:
    """The 509 aligned Text+Audio validation samples, rebuilt exactly as Phase C did."""
    index = AlignmentIndex.from_experiments(
        [
            ("audio", DEFAULT_BASELINES["audio"].experiment, "canonical_emotion_id",
             "emotion_7class"),
            ("text", DEFAULT_BASELINES["text"].experiment, "canonical_emotion_id",
             "emotion_7class"),
        ],
        root=args.root,
    )
    pool = index.fusion_pool(["audio", "text"], SPLIT, minimum=1)

    layout = AhsefLayout(run=args.stage3_run, root=Path(args.root) / "ahsef")
    llm_path = layout.prediction_path(LLM_MODALITY, SPLIT)
    strong_path = ANALYSIS / f"{STRONG_MODALITY}__{SPLIT}.parquet"
    features_path = layout.base / "hsig" / "features_validation.parquet"
    phase_c_path = ANALYSIS / "phase_c_aligned.json"
    assert_validation_only([llm_path, strong_path, features_path, phase_c_path])

    for path in (llm_path, strong_path, features_path, phase_c_path):
        if not Path(path).exists():
            raise SystemExit(
                f"Phase D/E replays from stored artefacts and {path} is missing. "
                f"Run the phase that produces it; nothing is re-scored here."
            )

    llm = PredictionSet.load(llm_path, layout.prediction_meta_path(LLM_MODALITY, SPLIT))
    strong = PredictionSet.load(strong_path, strong_path.with_suffix(".json"))

    usable = set(
        llm.frame.loc[llm.frame["llm_usable"].astype(bool), "sample_id"].astype(str)
    )
    covered = set(strong.sample_ids())
    pool = [item for item in pool if item in usable and item in covered]
    if not pool:
        raise SystemExit("The aligned validation pool is empty; nothing to route over.")

    return pool, {
        "text_llm": llm.restricted_to(pool),
        "audio_strong": strong.restricted_to(pool),
    }, {
        "text_llm_predictions": str(llm_path),
        "audio_strong_predictions": str(strong_path),
        "features": str(features_path),
        "phase_c": str(phase_c_path),
        "alignment_report": str(layout.alignment_report_path),
    }


def frozen_fusion_spec(phase_c_path: Path) -> FusionSpec:
    """The Phase C fusion weights, read rather than re-selected.

    Re-selecting them here would be a second validation fit on the same pool the
    routing policy is being chosen on, and the two choices would then be
    entangled. Phase C already made this one on validation by macro-F1.
    """
    record = json.loads(Path(phase_c_path).read_text(encoding="utf-8"))
    spec = (record.get("fusion_specs") or {}).get("strong")
    if not spec:
        raise SystemExit(
            f"{phase_c_path} carries no frozen strong-audio fusion spec; Phase D/E "
            f"will not choose one of its own."
        )
    return FusionSpec(
        method=spec["method"],
        weights=dict(spec["weights"]),
        selected_on_split=spec.get("selected_on_split"),
        selection={**(spec.get("selection") or {}), "reused_from": str(phase_c_path)},
    )


def build_outcomes(
    pool: list[str], predictions: dict, fused: PredictionSet
) -> RoutingOutcomes:
    text = predictions["text_llm"]
    return RoutingOutcomes(
        sample_ids=list(pool),
        true_class=text.frame["true_class"].to_numpy(dtype=int),
        text_prediction=text.frame["predicted_class"].to_numpy(dtype=int),
        fused_prediction=fused.frame["predicted_class"].to_numpy(dtype=int),
        text_uncertainty=text.frame["normalized_entropy"].to_numpy(dtype=float),
        fused_uncertainty=fused.frame["normalized_entropy"].to_numpy(dtype=float),
        num_classes=len(CANONICAL_EMOTION_CLASSES),
    )


def routing_uncertainty(features: pd.DataFrame, pool: list[str]) -> np.ndarray:
    """The frozen Stage 2/3 routing uncertainty, in the pool's order."""
    indexed = features.set_index(features["sample_id"].astype(str))
    return indexed.loc[pool, "uncertainty"].to_numpy(dtype=float)


# ============================================================
# PHASE D
# ============================================================

def phase_d(args, pool, predictions, features, outcomes) -> dict:
    targets = build_all_targets(outcomes)
    uncertainty = routing_uncertainty(features, pool)

    alpha_choice = select_alpha(
        features, pool, targets[ALPHA_SELECTION_TARGET], SPLIT,
        args.feature_set, args.folds, args.seed, ALPHA_GRID,
    )
    alpha = float(alpha_choice["selected_alpha"])
    alpha_choice["selected_on_target"] = ALPHA_SELECTION_TARGET
    alpha_choice["shared_by_every_target"] = True
    alpha_choice["why"] = (
        "One strength for every candidate. Tuning alpha per target would give each "
        "candidate a regularisation fitted for it, and the comparison would then be "
        "between five separately tuned models rather than between five targets."
    )

    scores: dict[str, np.ndarray] = {UNCERTAINTY_POLICY: uncertainty}
    models: dict[str, MilestoneHSIG] = {}
    quality: dict[str, dict] = {}
    distributions: dict[str, dict] = {}

    for name in TARGET_NAMES:
        oof = out_of_fold_scores(
            features, pool, targets[name], SPLIT, args.feature_set, alpha,
            args.folds, args.seed,
        )
        scores[name] = oof
        model = MilestoneHSIG.fit(
            features, pool, targets[name], name, SPLIT, args.feature_set, alpha,
            args.seed,
        )
        model.quality = {"note": "quality is reported out of fold, see targets block"}
        models[name] = model
        distributions[name] = target_distribution(targets[name], outcomes)
        quality[name] = target_quality(
            name, oof, targets[name], outcomes, targets["macro_f1_marginal"]
        )

    quality[UNCERTAINTY_POLICY] = target_quality(
        UNCERTAINTY_POLICY, uncertainty, uncertainty, outcomes,
        targets["macro_f1_marginal"],
    )
    quality[UNCERTAINTY_POLICY]["definition"] = (
        "the frozen Stage 2/3 routing uncertainty, used directly as the routing "
        "score. No estimator, no target, no fitting -- the ablation every learned "
        "candidate has to justify itself against."
    )

    ablation = _feature_set_ablation(args, features, pool, targets, outcomes, alpha)

    return {
        "phase": "D -- routing target selection",
        "split": SPLIT,
        "pool_samples": len(pool),
        "targets_declared": list(TARGET_NAMES),
        "target_description": describe_targets(outcomes),
        "target_distributions": distributions,
        "alpha_selection": alpha_choice,
        "estimator": {
            "family": "ridge on standardised features",
            "alpha": alpha,
            "folds": args.folds,
            "seed": args.seed,
            "feature_set": args.feature_set,
            "features": list(FEATURE_SETS[args.feature_set]),
            "identical_across_targets": True,
            "protocol": (
                "seeded K-fold; every reported score is out of fold, so no sample "
                "contributes to the model that scores it"
            ),
        },
        "target_quality": quality,
        "versus_uncertainty_only": compare_against_uncertainty(quality, UNCERTAINTY_POLICY),
        "feature_set_ablation": ablation,
        "coefficients": {name: model.importance() for name, model in models.items()},
        "models": {name: model.to_dict() for name, model in models.items()},
        "target_fingerprints": {
            name: array_fingerprint(values) for name, values in targets.items()
        },
        "scores": {name: array_fingerprint(value) for name, value in scores.items()},
        "test_partition_opened": False,
    }, scores, models, targets, alpha


def _feature_set_ablation(args, features, pool, targets, outcomes, alpha) -> dict:
    """Does a richer feature set help, holding the target fixed?

    Stage 3 asked exactly this and answered no. Reproducing the question on the
    strong-audio pool is the only way to know whether that answer was about the
    features or about the weaker expert they were asked to route for.
    """
    target = targets[ALPHA_SELECTION_TARGET]
    records = []
    for feature_set in ("uncertainty_only", "evidence_only", "evidence_plus_class"):
        oof = out_of_fold_scores(
            features, pool, target, SPLIT, feature_set, alpha, args.folds, args.seed
        )
        records.append({
            "feature_set": feature_set,
            "features": list(FEATURE_SETS[feature_set]),
            "quality": target_quality(
                ALPHA_SELECTION_TARGET, oof, target, outcomes,
                targets["macro_f1_marginal"],
            ),
        })
    summary = feature_set_ablation(records)
    summary["target_held_fixed"] = ALPHA_SELECTION_TARGET
    summary["alpha_held_fixed"] = alpha
    return summary


# ============================================================
# PHASE E
# ============================================================

def phase_e(args, pool, outcomes, scores, costs, stage3_scores=None) -> dict:
    pool_outcomes = PoolOutcomes(
        sample_ids=list(pool),
        true_class=outcomes.true_class,
        text_prediction=outcomes.text_prediction,
        fused_prediction=outcomes.fused_prediction,
        audio_prediction=outcomes.fused_prediction,
        text_latency_ms=costs.text_llm_latency_ms,
        audio_latency_ms=costs.audio_strong_latency_ms,
        text_compute_units=costs.text_llm_compute_units,
        audio_compute_units=costs.audio_strong_compute_units,
        num_classes=outcomes.num_classes,
        text_uncertainty=outcomes.text_uncertainty,
        fused_uncertainty=outcomes.fused_uncertainty,
    )

    policies = dict(scores)
    if stage3_scores is not None:
        policies["stage3_hsig_reference"] = stage3_scores

    comparison = compare_policies(
        pool_outcomes, policies, DEFAULT_BUDGETS, args.seed, args.repeats
    )
    text_only = comparison["curves"]["oracle"][0]
    always_fusion = comparison["curves"]["oracle"][-1]

    comparison["curves"] = {
        name: annotate_curve(
            curve, comparison["random_control"], comparison["curves"]["oracle"],
            text_only, always_fusion,
        )
        for name, curve in comparison["curves"].items()
    }
    comparison["efficiency"] = efficiency_summary(comparison)
    comparison["cost_configuration"] = costs.to_dict()
    comparison["ablations"] = {
        "declared": list(ABLATIONS),
        "stage3_reference_included": stage3_scores is not None,
        "stage3_reference_caveat": (
            "The Stage 3 HSIG was fitted on this same 509-sample validation pool and "
            "is replayed here in sample. Its curve is optimistic by construction and "
            "is carried as a historical reference, not as a competing candidate."
        ),
    }
    return comparison, pool_outcomes, text_only, always_fusion


def choose_operating_point(
    curve, random_rows, text_only, always_fusion, majority_classes
) -> dict:
    """The smallest budget at which every declared success condition holds.

    Smallest, not best: the claim is about activating audio rarely, so among the
    budgets that satisfy the rule the cheapest one is the one that supports the
    claim. If none satisfies it, the best interior budget by macro-F1 is reported
    instead and the verdict is negative -- the point is still named, so the
    reader can see how far short it fell.
    """
    random_by_budget = {
        round(float(row["requested_budget"]), 6): row for row in random_rows
    }
    interior = [
        row for row in curve if 0.0 < float(row["acquisition_rate"]) < 1.0
    ]
    evaluated = []
    for row in sorted(interior, key=lambda item: item["requested_budget"]):
        decision = routing_decision(
            row, random_by_budget.get(round(float(row["requested_budget"]), 6), {}),
            text_only, always_fusion, majority_classes,
        )
        evaluated.append(decision)
        if decision["material_routing_success"]:
            return {
                "selected_budget": row["requested_budget"],
                "rule": "smallest budget meeting every MATERIAL_ROUTING_SUCCESS condition",
                "decision": decision,
                "all_budgets": evaluated,
            }
    fallback = max(interior, key=lambda item: item["macro_f1"]) if interior else None
    return {
        "selected_budget": fallback["requested_budget"] if fallback else None,
        "rule": (
            "no budget met every condition; the best interior budget by macro-F1 is "
            "reported so the shortfall is visible"
        ),
        "decision": next(
            (
                item for item in evaluated
                if fallback and item["operating_point"]["budget"]
                == fallback["requested_budget"]
            ),
            None,
        ),
        "all_budgets": evaluated,
    }


# ============================================================
# Driver
# ============================================================

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    lock_before = verify_locked_artefacts(output / "locked_artefacts_baseline.json")

    costs = load_frozen_costs()
    assert_encoder_is_charged(costs)

    pool, predictions, sources = load_pool(args)
    print(f"[pool] {len(pool)} aligned Text+Audio validation samples with usable Gemma")

    spec = frozen_fusion_spec(Path(sources["phase_c"]))
    fused = fuse_prediction_sets(
        {"text_llm": predictions["text_llm"], "audio": predictions["audio_strong"]},
        spec, pool,
    )
    print(f"[fuse] frozen Phase C weights {dict(spec.weights)}")

    outcomes = build_outcomes(pool, predictions, fused)
    features = pd.read_parquet(sources["features"])

    print("[D   ] building targets and fitting the shared estimator family ...")
    d_record, scores, models, targets, alpha = phase_d(
        args, pool, predictions, features, outcomes
    )

    stage3_scores = _stage3_reference(args, features, pool)

    print("[E   ] running the budget grid ...")
    comparison, pool_outcomes, text_only, always_fusion = phase_e(
        args, pool, outcomes, scores, costs, stage3_scores
    )

    candidates = {
        name: curve for name, curve in comparison["curves"].items()
        if name in set(ABLATIONS)
    }
    ranking = rank_candidates(
        candidates, comparison["beats_random"], always_fusion["macro_f1"],
        text_only["macro_f1"], UNCERTAINTY_POLICY,
    )

    majority = majority_classes_of(outcomes.true_class, outcomes.num_classes)
    selected = ranking["selected"]
    operating = (
        choose_operating_point(
            comparison["curves"][selected], comparison["random_control"],
            text_only, always_fusion, majority,
        )
        if selected else None
    )

    lock_after = verify_locked_artefacts(output / "locked_artefacts_baseline.json")
    assert_locked_artefacts_unchanged(lock_after)

    provenance = reproducibility_record(
        pool=pool,
        predictions={**predictions, "text_llm+audio_strong": fused},
        alignment={
            "pool_sha256": sequence_fingerprint(pool),
            "pool_size": len(pool),
            "modalities": ["text_llm", "audio_strong"],
            "built_by": "AlignmentIndex.fusion_pool(['audio','text'], 'validation')",
            "restricted_to": "samples with usable Gemma evidence and cached audio features",
        },
        feature_configuration={
            "feature_set": args.feature_set,
            "features": list(FEATURE_SETS[args.feature_set]),
            "definition": dict(FEATURE_DEFINITION),
            "label_free": True,
            "uncertainty_policy": "score_entropy (inherited from the Stage 2 freeze)",
        },
        estimator_configuration=d_record["estimator"],
        target_fingerprints=d_record["target_fingerprints"],
        budget_grid=DEFAULT_BUDGETS,
        cost_configuration=costs.to_dict(),
        seed=args.seed,
        sources=sources,
    )

    record = {
        "experiment_id": MILESTONE_EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "phase": "D/E -- routing target selection and budget curves",
        "split": SPLIT,
        "pool_samples": len(pool),
        "fusion": spec.to_dict(),
        "reference_points": {
            "text_only": text_only,
            "always_fusion": always_fusion,
            "always_fusion_macro_f1_gain": (
                always_fusion["macro_f1"] - text_only["macro_f1"]
            ),
        },
        "phase_d": d_record,
        "phase_e": comparison,
        "selection": ranking,
        "operating_point": operating,
        "majority_classes": majority,
        "per_class_at_key_budgets": _per_class_table(comparison, selected),
        "decision": (operating or {}).get("decision"),
        "verdict": (
            (operating or {}).get("decision", {}).get("verdict")
            if operating and operating.get("decision")
            else "ROUTING SIGNAL INSUFFICIENT"
        ),
        "reproducibility": provenance,
        "test_lock": {
            "test_partition_opened": False,
            "test_labels_used": False,
            "test_split_read": False,
            "thresholds_tuned_on_test": False,
            "enforced_by": (
                "assert_validation_only() on every input path, and a driver that "
                "names no test artefact"
            ),
            "locked_artefacts_before": lock_before,
            "locked_artefacts_after": lock_after,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    path = output / "phase_de_results.json"
    path.write_text(json.dumps(record, indent=2, default=_json_default), encoding="utf-8")
    for name, model in models.items():
        model.save(output / "estimators" / f"{name}.json")

    frozen = _frozen_policy(record, models, spec, costs, args)
    frozen_path = output / "frozen_routing_policy.json"
    frozen_path.write_text(
        json.dumps(frozen, indent=2, default=_json_default), encoding="utf-8"
    )

    _print_report(record)
    print(f"\nWritten: {path}")
    print(f"Written: {output / 'estimators'} ({len(models)} estimator artefacts)")
    print(f"Written: {frozen_path}")
    return 0


def _frozen_policy(record, models, spec, costs, args) -> dict:
    """The routing policy, frozen so Phase F can verify rather than re-choose.

    Phase F must be able to check that nothing moved between the validation
    selection and the locked evaluation. That is only possible if the policy is
    stored as values it can compare against -- the target, the score, the budget,
    the fusion weights and the fingerprints -- rather than as a description a
    later run would have to re-derive.
    """
    selected = record["selection"]["selected"]
    operating = record.get("operating_point") or {}
    estimator = models.get(selected)
    return {
        "experiment_id": MILESTONE_EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "selected_target": selected,
        "selection_rule": dict(SELECTION_RULE),
        "selection_outcome": record["selection"]["outcome"],
        "parsimony_applied": record["selection"].get("parsimony_applied"),
        "routing_rule": {
            "score": (
                "the frozen Stage 2/3 routing uncertainty (normalised score "
                "entropy of the Gemma response), used directly"
                if selected == UNCERTAINTY_POLICY else
                f"ridge estimator fitted on the {selected} target over the "
                f"{args.feature_set} feature set"
            ),
            "requires_estimator": selected != UNCERTAINTY_POLICY,
            "estimator_fingerprint": estimator.fingerprint() if estimator else None,
            "estimator_artefact": (
                f"experiments/ahsef/milestone/estimators/{selected}.json"
                if estimator else None
            ),
            "decision": "acquire audio on the top-k samples by score",
            "tie_break": "index order, so the acquired set is deterministic",
            "k": "budget x pool size, rounded to nearest",
        },
        "acquisition_budget": {
            "selected": operating.get("selected_budget"),
            "rule": operating.get("rule"),
            "grid": [float(value) for value in DEFAULT_BUDGETS],
        },
        "fusion": spec.to_dict(),
        "experts": {
            "text": "gemma4:31b-cloud, Stage 2 frozen prompt and decoding",
            "audio": "audio_strong -- frozen WAV2VEC2_BASE + LayerWeightedProbe",
        },
        "costs": costs.to_dict(),
        "seeds": {"estimator_seed": args.seed, "random_control_seed": args.seed},
        "validation_performance": operating.get("decision", {}).get("operating_point"),
        "material_routing_success": (
            operating.get("decision", {}).get("material_routing_success")
        ),
        "fingerprints": {
            "pool": record["reproducibility"]["dataset_fingerprint"][
                "validation_pool_sha256"
            ],
            "cost_configuration": costs.fingerprint(),
            "targets": record["phase_d"]["target_fingerprints"],
        },
        "uses_test_labels": False,
        "test_partition_opened": False,
        "lock_note": (
            "Everything the locked Phase F evaluation depends on is fixed here. "
            "Reselecting the target, refitting the estimator, moving the budget or "
            "retuning a fusion weight after this point invalidates the evaluation, "
            "and Phase F must verify this file before opening the test split."
        ),
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value)} is not JSON serialisable")


def _stage3_reference(args, features, pool):
    """The Stage 3 HSIG, replayed as a historical reference where it exists."""
    path = (
        Path(args.root) / "ahsef" / args.stage3_run / "hsig" / "hsig_model.json"
    )
    if not path.exists():
        return None
    from src.ahsef.stage3.hsig_model import Stage3HSIG

    model = Stage3HSIG.load(path)
    frame = model.predict_frame(features, pool)
    values = frame["predicted_gain"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    print(f"[ref ] Stage 3 HSIG replayed ({model.estimator_type}, {model.feature_set})")
    return values


def _per_class_table(comparison, selected) -> dict:
    """Per-class F1 at the operating points Phase E singles out."""
    key_budgets = (0.05, 0.10, 0.20, 0.50, 1.00)
    table = {}
    for name, curve in comparison["curves"].items():
        rows = {
            f"{row['requested_budget']:.2f}": row["per_class_f1"]
            for row in curve
            if round(float(row["requested_budget"]), 6) in
            {round(b, 6) for b in key_budgets}
        }
        table[name] = rows
    table["text_only_reference"] = comparison["curves"]["oracle"][0]["per_class_f1"]
    table["selected_policy"] = selected
    table["reading"] = (
        "The question is whether selective acquisition raises the minority classes "
        "or only polishes neutral. Compare each row against text_only_reference "
        "class by class, not on the macro average alone."
    )
    return table


def _print_report(record) -> None:
    print()
    print("=" * 100)
    print(f"PHASE D -- routing target selection (validation, n={record['pool_samples']})")
    print("=" * 100)
    quality = record["phase_d"]["target_quality"]
    print(f"  {'Target':<24}{'AUROC':>9}{'AUPRC':>9}{'base':>8}"
          f"{'rho(gain)':>11}{'rho(macroF1)':>14}{'maj x':>8}")
    for name, block in quality.items():
        useful = block["identifying_useful_acquisitions"]
        rank = block["rank_agreement"]
        concentration = block["class_concentration"]
        print(f"  {name:<24}"
              f"{_fmt(useful['auroc']):>9}{_fmt(useful['auprc']):>9}"
              f"{useful['base_rate']:>8.3f}"
              f"{_fmt(rank['spearman_vs_signed_gain']):>11}"
              f"{_fmt(rank['spearman_vs_macro_f1_marginal']):>14}"
              f"{_fmt(concentration['majority_over_representation']):>8}")

    ablation = record["phase_d"]["feature_set_ablation"]
    print()
    print(f"  Feature-set ablation (target held at {ablation['target_held_fixed']}):")
    for name, row in ablation["per_feature_set"].items():
        print(f"    {name:<22}{row['feature_count']:>3} features  "
              f"AUROC={_fmt(row['auroc']):>7}  rho={_fmt(row['spearman_vs_signed_gain']):>7}")
    print(f"    richer features help: {ablation['richer_features_help']}")
    print(f"    Stage 3 finding reproduced: {ablation['reproduced']}")

    print()
    print("=" * 100)
    print("PHASE E -- budget curves (macro-F1 is primary)")
    print("=" * 100)
    text_only = record["reference_points"]["text_only"]
    always = record["reference_points"]["always_fusion"]
    print(f"  text-only      macro-F1={text_only['macro_f1']:.4f}  "
          f"acc={text_only['accuracy']:.4f}")
    print(f"  always-fusion  macro-F1={always['macro_f1']:.4f}  "
          f"acc={always['accuracy']:.4f}  "
          f"(gain {record['reference_points']['always_fusion_macro_f1_gain']:+.4f})")
    print()
    random_rows = {
        f"{row['requested_budget']:.2f}": row for row in record["phase_e"]["random_control"]
    }
    for name, curve in record["phase_e"]["curves"].items():
        print(f"  {name}")
        print(f"    {'budget':>8}{'n':>6}{'macroF1':>10}{'acc':>9}{'wF1':>9}"
              f"{'rand mean':>11}{'rand p97.5':>12}{'>rand':>7}{'ms/samp':>10}")
        for row in curve:
            control = random_rows.get(f"{row['requested_budget']:.2f}", {})
            macro = (control.get("macro_f1") or {})
            print(f"    {row['requested_budget']:>8.2f}{row['acquired']:>6}"
                  f"{row['macro_f1']:>10.4f}{row['accuracy']:>9.4f}"
                  f"{row['weighted_f1']:>9.4f}"
                  f"{_fmt(macro.get('mean')):>11}{_fmt(macro.get('p97.5')):>12}"
                  f"{str(row['improvement_over_random']['above_random_upper_bound'])[:5]:>7}"
                  f"{row['mean_latency_ms']:>10.1f}")
        print()

    print("=" * 100)
    print("SELECTION")
    print("=" * 100)
    selection = record["selection"]
    print(f"  rule     : {selection['rule']['primary_criterion']}")
    print(f"  gate     : {selection['rule']['admissibility_gate']}")
    for name, row in selection["per_candidate"].items():
        print(f"    {name:<24}mean macro-F1 over low budgets="
              f"{row['mean_macro_f1_over_selection_budgets']:.4f}  "
              f"admissible={row['admissible']}  "
              f"above random at {row['budgets_above_random'] or 'no budget'}")
    print(f"  outcome  : {selection['outcome']}")
    print(f"  selected : {selection['selected']}")
    if selection.get("parsimony_note"):
        print(f"  parsimony: {selection['parsimony_note']}")

    decision = record.get("decision")
    print()
    print("=" * 100)
    print("PHASE D/E DECISION")
    print("=" * 100)
    if decision:
        for key in (
            "condition_1_beats_matched_random",
            "condition_2_not_only_majority_corrections",
            "condition_3_retains_fusion_gain",
            "condition_4_substantially_fewer_acquisitions",
        ):
            print(f"  {key:<48}{decision[key]['met']}")
        print(f"  VERDICT  : {decision['verdict']}")
        print(f"  DECISION : {decision['decision']}")
    else:
        print("  VERDICT  : ROUTING SIGNAL INSUFFICIENT (no admissible candidate)")
    lock = record["test_lock"]
    print()
    print(f"  test_partition_opened          : {lock['test_partition_opened']}")
    print(f"  locked artefacts unchanged     : "
          f"{lock['locked_artefacts_after']['verified']}")


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
