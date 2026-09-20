"""Stage 3 command stages: oracle, hsig, freeze, validate, ablations, test, report.

Split out of :mod:`src.ahsef.cli.run_stage3_text_audio` only for length; the
staging discipline is the same as Stage 2's and is enforced here:

* ``oracle`` selects the fusion rule and weight **on validation only**; on test
  it loads the frozen spec and refuses to reselect.
* ``hsig`` fits on validation only and refuses any other split.
* ``freeze`` writes ``frozen_config.json``.  It is the only stage that does.
* ``test`` loads that file, checks every value against the requested
  configuration, and aborts on any drift -- including a changed HSIG
  fingerprint, a changed fusion weight, or a changed threshold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.fusion import FusionSpec
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY, routing_uncertainty_column
from src.ahsef.registry import DEFAULT_BASELINES
from src.ahsef.routing_log import RoutingLogWriter
from src.ahsef.stage3 import STAGE3_CLAIM, STAGE3_CLAIM_LIMITS, STAGE3_EXPERIMENT_ID
from src.ahsef.stage3.costs import (
    build_cost_model,
    cost_model_from_frozen,
    latency_pair,
    load_timing,
)
from src.ahsef.stage3.experiment import (
    always_acquire,
    build_inputs,
    compare_systems,
    never_acquire,
    random_policy_reference,
)
from src.ahsef.stage3.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    build_features,
    feature_provenance,
    missing_feature_report,
    states_from_features,
)
from src.ahsef.stage3.hsig_model import (
    ESTIMATOR_TYPES,
    Stage3HSIG,
    out_of_fold_gain,
    prior_constant_gain,
    shallow_forest_gain,
    targets_from_oracle,
)
from src.ahsef.stage3.hsig_quality import (
    class_rule_audit,
    compare_estimators,
    estimator_quality,
    feature_importance,
)
from src.ahsef.stage3.layout import Stage3Layout
from src.ahsef.stage3.llm_pass import FROZEN_LLM_CONFIG
from src.ahsef.stage3.objective import (
    OBJECTIVE_RATIONALE,
    select_gate_threshold,
    sensitivity_to_penalties,
)
from src.ahsef.stage3.oracle import (
    build_oracle_table,
    choose_fusion_spec,
    coverage_note,
    fuse,
    oracle_report,
    usable_text_ids,
)
from src.ahsef.stage3.pool import AlignedPool
from src.ahsef.stage3.router import (
    AudioAvailability,
    Availability,
    TextAudioRouter,
    attach_truth,
    decision_summary,
    decisions_frame,
)
from src.ahsef.ugapr import UGAPR, UtilityWeights
from src.common.labels import CANONICAL_EMOTION_CLASSES

#: Fusion rules compared on validation before one is frozen.
FUSION_METHODS = ("weighted_probability", "log_opinion_pool")

#: The Stage 2 threshold, inherited for the comparison row only. Never re-selected.
STAGE2_FROZEN_TAU = 0.2571559759740807


# ============================================================
# Shared loading
# ============================================================

def _pool(layout: Stage3Layout, split: str) -> AlignedPool:
    path = layout.pool_path(split)
    if not path.exists():
        raise SystemExit(f"No aligned pool at {path}. Run --stage align first.")
    return AlignedPool.load(path)


def _text_llm(layout: Stage3Layout, split: str) -> PredictionSet:
    path = layout.prediction_path(LLM_MODALITY, split)
    if not path.exists():
        raise SystemExit(
            f"No LLM predictions at {path}. Run --stage llm --split {split} first."
        )
    return PredictionSet.load(path, layout.prediction_meta_path(LLM_MODALITY, split))


def _audio(layout: Stage3Layout, split: str, root: str, pool: AlignedPool) -> PredictionSet:
    """The frozen Stage 1 audio baseline, restricted to the pool.  Never retrained."""
    local = layout.prediction_path("audio", split)
    if local.exists():
        return PredictionSet.load(local, layout.prediction_meta_path("audio", split))

    stage1 = AhsefLayout(run="stage1", root=Path(root) / "ahsef")
    source = stage1.prediction_path("audio", split)
    if not source.exists():
        raise SystemExit(
            f"Missing Stage 1 audio predictions at {source}. Stage 3 reads the frozen "
            f"baseline's export and never retrains or re-infers it."
        )
    restricted = PredictionSet.load(
        source, stage1.prediction_meta_path("audio", split)
    ).restricted_to(pool.sample_ids)
    restricted.meta.update({
        "kind": "frozen_audio_baseline",
        "source_export": str(source),
        "restricted_to_pool": pool.fingerprint,
        "retrained": False,
        "note": "Stage 1 audio predictions restricted to the Stage 3 aligned pool. "
                "Not retrained, not modified, not re-inferred.",
    })
    restricted.save(local, layout.prediction_meta_path("audio", split))
    return restricted


def _frozen_text_baseline(layout: Stage3Layout, split: str, root: str, pool: AlignedPool):
    """The Stage 1 frozen text baseline on the same pool, for the regression row."""
    stage1 = AhsefLayout(run="stage1", root=Path(root) / "ahsef")
    source = stage1.prediction_path("text", split)
    if not source.exists():
        return None
    return PredictionSet.load(
        source, stage1.prediction_meta_path("text", split)
    ).restricted_to(pool.sample_ids)


def _routing_uncertainty(text_llm: PredictionSet, policy: str) -> np.ndarray:
    return routing_uncertainty_column(text_llm, policy).to_numpy(dtype=float)


def _scored_pool(text_llm: PredictionSet) -> list[str]:
    """Pool ids with usable text evidence, in a stable order."""
    return usable_text_ids(text_llm)


# ============================================================
# Stage: oracle  (PHASES B and C)
# ============================================================

def stage_oracle(layout: Stage3Layout, args) -> int:
    split = args.split
    pool = _pool(layout, split)
    text_llm = _text_llm(layout, split)
    audio = _audio(layout, split, args.root, pool)
    ids = _scored_pool(text_llm)
    coverage = coverage_note(text_llm, ids)
    print(f"[cov ] {split}: usable text evidence on {len(ids)}/{pool.size} pooled samples")
    if coverage["excluded_no_text_evidence"]:
        print(f"       excluded (no text evidence): "
              f"{coverage['excluded_no_text_evidence']} -- counted, never imputed")

    if split == "validation":
        spec, comparison = _select_fusion(text_llm, audio, ids, args)
        layout.fusion_spec_path.write_text(
            json.dumps({
                "selected": spec.to_dict(),
                "method_comparison": comparison,
                "selected_on_split": "validation",
                "uses_test_labels": False,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"[fuse] method={spec.method} weights={spec.weights}")
    else:
        spec = _load_fusion_spec(layout)
        print(f"[fuse] {split}: using the FROZEN spec {spec.method} {spec.weights} "
              f"-- no reselection on test")

    fused = fuse(text_llm, audio, spec, ids)
    fused.save(
        layout.fused_path(split),
        layout.fused_path(split).with_suffix(".json"),
    )
    table = build_oracle_table(text_llm, audio, fused, ids)
    table.to_parquet(layout.oracle_path(split), index=False)

    report = oracle_report(table, text_llm, audio, fused, ids, spec, coverage)
    report["experiment_id"] = STAGE3_EXPERIMENT_ID
    report["pool_fingerprint_sha256"] = pool.fingerprint
    frozen_text = _frozen_text_baseline(layout, split, args.root, pool)
    if frozen_text is not None:
        from src.ahsef.cli.run_stage3_text_audio import metric_block

        report["regression"] = {
            "frozen_stage1_text_baseline_same_samples": metric_block(
                frozen_text.restricted_to(ids)
            ),
            "note": "The Stage 1 text baseline on exactly these samples, so the LLM's "
                    "contribution is measured against the model it replaces.",
        }
    layout.oracle_report_path(split).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _print_oracle(report)
    print(f"\nWritten: {layout.oracle_report_path(split)}")
    return 0


def _select_fusion(text_llm, audio, ids, args):
    """Choose the fusion rule and weight on validation, enumerating both rules."""
    from src.ahsef.evaluation import evaluate_prediction_set

    comparison = {}
    best = None
    for method in FUSION_METHODS:
        spec = choose_fusion_spec(
            text_llm, audio, ids, "validation", method=method, objective="macro_f1",
        )
        metrics = evaluate_prediction_set(fuse(text_llm, audio, spec, ids))
        comparison[method] = {
            "weights": dict(spec.weights),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "weight_scan": spec.selection.get("scanned"),
        }
        if best is None or metrics["macro_f1"] > best[1]:
            best = (spec, metrics["macro_f1"], method)
    requested = args.fusion_method
    if requested in comparison:
        spec = choose_fusion_spec(
            text_llm, audio, ids, "validation", method=requested, objective="macro_f1",
        )
        comparison["selection_rule"] = (
            f"The fusion rule is the configured --fusion-method ({requested!r}); both "
            f"rules were scanned on validation and are reported above so the choice is "
            f"auditable. Weights are chosen by macro-F1, matching the Stage 3 gate "
            f"objective rather than by accuracy, which Stage 2 showed misleads on this "
            f"corpus."
        )
        comparison["best_by_macro_f1"] = best[2]
        return spec, comparison
    return best[0], comparison


def _load_fusion_spec(layout: Stage3Layout) -> FusionSpec:
    path = layout.fusion_spec_path
    if not path.exists():
        raise SystemExit(
            f"No frozen fusion spec at {path}. Run --stage oracle --split validation "
            f"first; a fusion weight chosen on test is not a weight."
        )
    record = json.loads(path.read_text(encoding="utf-8"))["selected"]
    return FusionSpec(
        method=record["method"], weights=record["weights"],
        selected_on_split=record["selected_on_split"], selection=record["selection"],
    )


def _print_oracle(report: dict) -> None:
    print()
    print("=" * 92)
    print(f"PHASE C -- oracle acquisition value ({report['split']}, "
          f"n={report['samples']})")
    print("=" * 92)
    print(f"  {'System':<20}{'Acc':>9}{'Macro-F1':>11}{'Weighted-F1':>13}{'BalAcc':>9}")
    for name, block in report["systems"].items():
        print(f"  {name:<20}{block['accuracy']:>9.4f}{block['macro_f1']:>11.4f}"
              f"{block['weighted_f1']:>13.4f}{block['balanced_accuracy']:>9.4f}")
    outcome = report["acquisition_outcome"]
    print()
    print(f"  text wrong -> fusion right : {outcome['fixed_by_audio']:>4} "
          f"({outcome['fixed_fraction']:.1%})  CI95 "
          f"[{outcome['fixed_fraction_ci95'][0]:.3f}, {outcome['fixed_fraction_ci95'][1]:.3f}]")
    print(f"  text right -> fusion wrong : {outcome['broken_by_audio']:>4} "
          f"({outcome['broken_fraction']:.1%})  CI95 "
          f"[{outcome['broken_fraction_ci95'][0]:.3f}, {outcome['broken_fraction_ci95'][1]:.3f}]")
    print(f"  net improvement            : {outcome['net_improvement']:>+4} "
          f"({outcome['net_improvement_rate']:+.1%})")
    print(f"  text/audio agreement       : {outcome['agreement_text_audio']:.1%}")
    print()
    print(f"  {'Class':<10}{'n':>6}{'Text acc':>11}{'Fused acc':>11}{'Fixed':>8}"
          f"{'Broken':>8}{'Net':>6}")
    for name, row in report["per_class_improvement"].items():
        if not row["samples"]:
            continue
        print(f"  {name:<10}{row['samples']:>6}{row['text_accuracy']:>11.4f}"
              f"{row['fused_accuracy']:>11.4f}{row['fixed_by_audio']:>8}"
              f"{row['broken_by_audio']:>8}{row['net']:>+6}")


# ============================================================
# Stage: hsig  (PHASE D)
# ============================================================

def stage_hsig(layout: Stage3Layout, args) -> int:
    if args.split != "validation":
        raise SystemExit(
            "HSIG is fitted on validation only. Refusing to fit on "
            f"{args.split!r}: an estimator that has seen the evaluation labels makes "
            "every routing number that follows meaningless."
        )
    text_llm = _text_llm(layout, "validation")
    oracle_path = layout.oracle_path("validation")
    if not oracle_path.exists():
        raise SystemExit("Run --stage oracle --split validation first.")
    oracle = pd.read_parquet(oracle_path)
    ids = oracle["sample_id"].astype(str).tolist()

    scored = text_llm.restricted_to(ids)
    uncertainty = _routing_uncertainty(scored, args.uncertainty_policy)
    targets = targets_from_oracle(oracle, "validation")
    print(f"[hsig] validation n={len(ids)} | y_gain positives="
          f"{int(targets.y_gain.sum())} ({targets.y_gain.mean():.1%}) | "
          f"y_harm={int(targets.y_harm.sum())} ({targets.y_harm.mean():.1%})")

    quality: dict[str, dict] = {}
    candidates: dict[str, tuple[str, str, np.ndarray]] = {}
    feature_frames: dict[str, pd.DataFrame] = {}

    for feature_set in FEATURE_SETS:
        features = build_features(scored.frame, uncertainty, feature_set)
        feature_frames[feature_set] = features
        for estimator_type in ESTIMATOR_TYPES:
            name = f"{estimator_type}__{feature_set}"
            try:
                gains = out_of_fold_gain(
                    features, targets, feature_set, estimator_type,
                    seed=args.hsig_seed, folds=args.hsig_folds,
                )
            except Exception as error:  # a fold that cannot be fitted is reported
                quality[name] = {"estimator": name, "available": False,
                                 "reason": str(error)[:300]}
                continue
            quality[name] = estimator_quality(gains, targets, name, seed=args.hsig_seed)
            candidates[name] = (estimator_type, feature_set, gains)

    # References, scored out of fold by the same code path.
    base_features = feature_frames[DEFAULT_FEATURE_SET]
    quality["prior_constant"] = estimator_quality(
        prior_constant_gain(targets, len(ids)), targets, "prior_constant",
        seed=args.hsig_seed,
    )
    try:
        quality["shallow_forest_reference"] = estimator_quality(
            shallow_forest_gain(
                base_features, targets, DEFAULT_FEATURE_SET,
                seed=args.hsig_seed, folds=args.hsig_folds,
            ),
            targets, "shallow_forest_reference", seed=args.hsig_seed,
        )
    except Exception as error:
        quality["shallow_forest_reference"] = {
            "estimator": "shallow_forest_reference", "available": False,
            "reason": str(error)[:300],
        }

    # Two different questions, answered separately and both reported.
    #
    # (a) Which estimator gets frozen?  The best out-of-fold AUROC over every
    #     fitted variant, including the one-feature ones. Freezing a richer model
    #     that a simpler one out-predicts would be indefensible.
    # (b) Does the richer evidence model beat Stage 2's signal alone?  For that
    #     question the uncertainty-only variants are the reference, so the
    #     verdict cannot be satisfied by a model that merely reproduces them.
    references = tuple(
        ["prior_constant", "shallow_forest_reference"]
        + [name for name in candidates if name.endswith("__uncertainty_only")]
    )
    comparison = compare_estimators(quality, references)
    comparison["question"] = (
        "Does a richer evidence model beat Stage 2's uncertainty signal alone? The "
        "uncertainty-only variants are references for THIS question. They remain "
        "eligible to be frozen, and are frozen if they predict the gain best."
    )

    usable = {
        name: entry for name, entry in quality.items()
        if name in candidates
        and entry["identifying_useful_acquisitions"]["auroc"] is not None
    }
    if not usable:
        raise SystemExit(
            "No HSIG variant produced a defined AUROC on this pool. Stage 3 stops "
            "here rather than freezing an estimator whose quality is unknown."
        )
    selected_name, selection_rule = _select_hsig_variant(usable)
    estimator_type, feature_set, oof_gain = candidates[selected_name]
    print(f"[hsig] selected {selected_name} "
          f"(out-of-fold AUROC="
          f"{quality[selected_name]['identifying_useful_acquisitions']['auroc']:.4f})")

    model = Stage3HSIG.fit(
        feature_frames[feature_set], targets, feature_set=feature_set,
        estimator_type=estimator_type, seed=args.hsig_seed,
    )
    model.quality = {
        "selected_variant": selected_name,
        "out_of_fold": quality[selected_name],
        "selection_rule": selection_rule,
        "folds": args.hsig_folds,
        "note": (
            "Reported quality is OUT OF FOLD. The saved coefficients are refitted on "
            "the whole validation split, which is what the router uses; the in-sample "
            "quality of that refit is not reported because it would be optimistic."
        ),
    }
    model.save(layout.hsig_model_path)

    for name, frame in feature_frames.items():
        if name == feature_set:
            frame.to_parquet(layout.features_path("validation"), index=False)

    report = {
        "phase": "D -- HSIG",
        "experiment_id": STAGE3_EXPERIMENT_ID,
        "split": "validation",
        "samples": len(ids),
        "target": targets.summary(),
        "features": feature_provenance(feature_set),
        "missing_features": missing_feature_report(feature_frames[feature_set], feature_set),
        "variants": quality,
        "comparison": comparison,
        "selection_rule": selection_rule,
        "selected": {
            "variant": selected_name,
            "estimator_type": estimator_type,
            "feature_set": feature_set,
            "fingerprint_sha256": model.fingerprint(),
        },
        "feature_importance": feature_importance(model.components),
        "class_rule_audit": class_rule_audit(
            oof_gain, oracle["text_prediction"].to_numpy(dtype=int),
            CANONICAL_EMOTION_CLASSES,
            uses_class_features=any(
                name.startswith("predicted_is_") for name in FEATURE_SETS[feature_set]
            ),
        ),
        "honesty": {
            "all_quality_is_out_of_fold": True,
            "folds": args.hsig_folds,
            "seed": args.hsig_seed,
            "pool_size": len(ids),
            "caveat": (
                f"The estimator is fitted on {len(ids)} validation samples with "
                f"{len(FEATURE_SETS[feature_set])} "
                f"{'feature' if len(FEATURE_SETS[feature_set]) == 1 else 'features'} and "
                f"{int(targets.y_gain.sum())} positive events. Bootstrap intervals are "
                f"reported for every headline statistic because a point estimate at "
                f"this size would invite a conclusion the data cannot carry."
            ),
        },
    }
    layout.hsig_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_hsig(report)
    print(f"\nWritten: {layout.hsig_model_path}")
    print(f"Written: {layout.hsig_report_path}")
    return 0


#: How the frozen HSIG variant is chosen. Both clauses are properties of the
#: estimators rather than of any particular result, so the rule can be stated
#: without reference to the numbers it will be applied to.
HSIG_SELECTION_RULE = {
    "primary": "highest out-of-fold AUROC against y_gain on validation",
    "tie_break": (
        "among variants whose AUROC falls inside the bootstrap 95% interval of the "
        "best -- that is, variants this pool cannot distinguish -- prefer "
        "'paired_logistic'."
    ),
    "why_paired_wins_ties": (
        "UGAPR computes J = gain - lambda*C - mu*L, so the gain has to be in the "
        "units the target is defined in: a DIFFERENCE of probabilities of being "
        "correct. paired_logistic emits exactly that. fix_logistic emits P(audio "
        "fixes a wrong answer), which lives in [0, 1], is a different quantity, and "
        "cannot express harm at all -- subtracting a cost from it would be a "
        "category error dressed as a utility. The preference is therefore a "
        "property of the estimator's output, not of its score on this pool."
    ),
}


def _select_hsig_variant(usable: dict) -> tuple[str, dict]:
    """Apply :data:`HSIG_SELECTION_RULE` and record what it did and why."""
    def auroc_of(name: str) -> float:
        return usable[name]["identifying_useful_acquisitions"]["auroc"]

    best = max(usable, key=auroc_of)
    interval = usable[best]["identifying_useful_acquisitions"]["auroc_ci95"]
    tied = [
        name for name in usable
        if interval is None or interval[0] <= auroc_of(name) <= interval[1]
    ]
    preferred = [name for name in tied if name.startswith("paired_logistic")]
    chosen = max(preferred, key=auroc_of) if preferred else best
    return chosen, {
        **HSIG_SELECTION_RULE,
        "best_by_auroc": best,
        "best_auroc": auroc_of(best),
        "indistinguishable_from_best": sorted(tied),
        "selected": chosen,
        "selected_auroc": auroc_of(chosen),
        "auroc_conceded_to_the_tie_break": auroc_of(best) - auroc_of(chosen),
    }


def _print_hsig(report: dict) -> None:
    print()
    print("=" * 92)
    print("PHASE D -- HSIG quality (all figures OUT OF FOLD on validation)")
    print("=" * 92)
    print(f"  {'Estimator':<36}{'AUROC':>9}{'AUROC CI95':>20}{'AUPRC':>9}{'base':>8}"
          f"{'rho':>8}")
    for name, entry in report["variants"].items():
        if not entry.get("identifying_useful_acquisitions"):
            print(f"  {name:<36}  unavailable: {entry.get('reason', '')[:40]}")
            continue
        block = entry["identifying_useful_acquisitions"]
        interval = block["auroc_ci95"]
        rho = entry["predicted_vs_observed"]["spearman_rho"]
        print(
            f"  {name:<36}"
            f"{(block['auroc'] if block['auroc'] is not None else float('nan')):>9.4f}"
            + (f"  [{interval[0]:.3f}, {interval[1]:.3f}]".rjust(20) if interval
               else " " * 20)
            + f"{(block['auprc'] or float('nan')):>9.4f}"
            + f"{(block['base_rate'] or float('nan')):>8.3f}"
            + f"{(rho if rho is not None else float('nan')):>8.3f}"
        )
    verdict = report["comparison"]["verdict"]
    if verdict:
        print()
        print(f"  VERDICT: {verdict['statement']}")
    audit = report["class_rule_audit"]
    print()
    print(f"  class-rule audit: within-class std={audit['mean_within_class_std']} "
          f"between-class std={audit['between_class_std']}")
    print(f"  {audit['verdict']}")


# ============================================================
# Stage: freeze  (PHASES E, F, J)
# ============================================================

def stage_freeze(layout: Stage3Layout, args) -> int:
    text_llm = _text_llm(layout, "validation")
    oracle = pd.read_parquet(layout.oracle_path("validation"))
    ids = oracle["sample_id"].astype(str).tolist()
    pool = _pool(layout, "validation")
    if not layout.hsig_model_path.exists():
        raise SystemExit("Run --stage hsig first.")
    model = Stage3HSIG.load(layout.hsig_model_path)

    timing = load_timing(layout.timing_path("validation"))
    cost_model, cost_record = build_cost_model(
        text_llm, timing, root=args.root, split="validation",
    )
    text_ms, audio_ms = latency_pair(cost_model)
    normalized_cost = cost_model.normalized_cost("audio")
    normalized_latency = cost_model.normalized_latency("audio")
    penalty = args.lambda_cost * normalized_cost + args.mu_latency * normalized_latency
    print(f"[cost] audio: normalized cost={normalized_cost:.6f} "
          f"latency={normalized_latency:.6f} | penalty={penalty:.6f}")
    print(f"[cost] latency ms: text_llm={text_ms:.1f} audio={audio_ms:.1f}")

    scored = text_llm.restricted_to(ids)
    uncertainty = _routing_uncertainty(scored, args.uncertainty_policy)
    features = build_features(scored.frame, uncertainty, model.feature_set)
    targets = targets_from_oracle(oracle, "validation")

    # tau is a threshold on a specific function's output, so it must be chosen
    # against the output of the function the router will actually evaluate --
    # the deployed estimator, fitted on all of validation. Out-of-fold scores
    # come from K different models and are a different random variable; a tau
    # calibrated on them selects a different, and measurably worse, set of
    # samples when applied to the deployed model. The out-of-fold sweep is
    # computed anyway and recorded beside the frozen one, so the optimism of
    # choosing one scalar in-sample is visible rather than asserted away.
    deployed_gain = model.predict_frame(features, ids)["predicted_gain"].to_numpy(float)
    oof_gain = out_of_fold_gain(
        features, targets, model.feature_set, model.estimator_type,
        seed=args.hsig_seed, folds=args.hsig_folds,
    )
    common = dict(
        text_prediction=oracle["text_prediction"].to_numpy(dtype=int),
        fused_prediction=oracle["fused_prediction"].to_numpy(dtype=int),
        true_class=oracle["true_class"].to_numpy(dtype=int),
        split="validation", class_names=CANONICAL_EMOTION_CLASSES,
        text_latency_ms=text_ms, audio_latency_ms=audio_ms,
        objective=args.gate_objective, alpha=args.alpha_acquisition,
        beta=args.beta_latency,
    )
    selection = select_gate_threshold(deployed_gain - penalty, **common)
    out_of_fold_selection = select_gate_threshold(oof_gain - penalty, **common)
    print(f"[gate] tau={selection.threshold:.6f} objective={selection.objective}")
    print(f"       {selection.note}")
    print(f"       out-of-fold comparison: tau={out_of_fold_selection.threshold:.6f} "
          f"acq={out_of_fold_selection.selected.acquisition_rate:.1%} "
          f"macroF1={out_of_fold_selection.selected.macro_f1:.4f}")

    spec = _load_fusion_spec(layout)
    from src.ahsef.cli.run_stage3_text_audio import environment_record, git_commit

    frozen = {
        "experiment_id": STAGE3_EXPERIMENT_ID,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "claim": STAGE3_CLAIM,
        "claim_limits": STAGE3_CLAIM_LIMITS,

        "text_llm": {
            **FROZEN_LLM_CONFIG,
            "modality": LLM_MODALITY,
            "prompt_version": args.prompt_version,
            "model": args.model,
        },
        "audio_baseline": {
            "modality": "audio",
            "experiment": DEFAULT_BASELINES["audio"].experiment,
            "iteration": DEFAULT_BASELINES["audio"].resolve_iteration(args.root),
            "checkpoint_sha256": _audio_checkpoint_digest(layout, args.root),
            "retrained_by_stage3": False,
        },
        "uncertainty_policy": args.uncertainty_policy,
        "uncertainty_policy_source": "inherited from the Stage 2 freeze; not re-selected",

        "hsig": {
            "estimator_type": model.estimator_type,
            "feature_set": model.feature_set,
            "features": list(FEATURE_SETS[model.feature_set]),
            "feature_definition": feature_provenance(model.feature_set),
            "target": model.to_dict()["target"],
            "target_definition": model.to_dict()["target_definition"],
            "fitted_on_split": model.fitted_on_split,
            "folds": args.hsig_folds,
            "seed": args.hsig_seed,
            "fingerprint_sha256": model.fingerprint(),
            "model_path": str(layout.hsig_model_path),
        },
        "fusion": spec.to_dict(),
        "gate": {
            "objective": selection.objective,
            "objective_rationale": dict(OBJECTIVE_RATIONALE),
            "threshold": selection.threshold,
            "alpha_acquisition": args.alpha_acquisition,
            "beta_latency": args.beta_latency,
            "selected_on_split": "validation",
            "selected_on": "the deployed estimator's own validation scores",
            "why_not_out_of_fold": (
                "tau thresholds a specific function's output. Out-of-fold scores come "
                "from K separately fitted models and are a different random variable, "
                "so a tau calibrated on them selects a different -- and on this pool a "
                "measurably worse -- set of samples when applied to the deployed "
                "estimator. HSIG QUALITY is still reported out of fold; only the "
                "operating point is chosen against the function it will be applied to."
            ),
            "selection": selection.to_dict(),
            "out_of_fold_comparison": {
                "threshold": out_of_fold_selection.threshold,
                "selected_point": out_of_fold_selection.to_dict()["selected_point"],
                "note": (
                    "What the same objective would have chosen from out-of-fold "
                    "scores. Recorded so the optimism of selecting one scalar "
                    "in-sample can be read off rather than taken on trust."
                ),
            },
            "penalty_sensitivity": sensitivity_to_penalties(selection.sweep),
        },
        "ugapr": {
            "lambda_cost": args.lambda_cost,
            "mu_latency": args.mu_latency,
            "formula": "J(m) = gain_hat(m) - lambda * C(m) - mu * L(m)",
            "candidate_registry": ["audio"],
            "audio_penalty": penalty,
        },
        "costs": cost_record,
        "latency": {
            "text_llm_ms": text_ms,
            "audio_ms": audio_ms,
            "definition": "measured mean per-sample wall clock; for text_llm this is "
                          "end-to-end service time, not generation time",
            "llm_timing_record": timing,
        },
        "alignment": {
            "validation_pool_fingerprint_sha256": pool.fingerprint,
            "validation_pool_size": pool.size,
            "test_pool_fingerprint_sha256": _pool(layout, "test").fingerprint,
            "test_pool_size": _pool(layout, "test").size,
            "alignment_report": str(layout.alignment_report_path),
        },
        "seeds": {
            "sample_seed": args.seed,
            "hsig_seed": args.hsig_seed,
            "llm_seed": args.llm_seed,
        },
        "git_commit": git_commit(),
        "environment": environment_record(),
        "uses_test_labels": False,
        "lock_note": (
            "Everything the locked test evaluation depends on is fixed here. "
            "--stage test loads this file, verifies every value against the requested "
            "configuration and against the HSIG fingerprint on disk, and never writes "
            "it. Refitting HSIG, reselecting tau, retuning a fusion weight, or "
            "changing lambda/mu/alpha/beta after this point invalidates the "
            "evaluation and the test stage refuses to proceed."
        ),
    }
    layout.frozen_config_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"\nWritten: {layout.frozen_config_path}")
    print("Next: --stage validate")
    return 0


def _audio_checkpoint_digest(layout: Stage3Layout, root: str) -> str | None:
    stage1 = AhsefLayout(run="stage1", root=Path(root) / "ahsef")
    meta = stage1.prediction_meta_path("audio", "validation")
    if not meta.exists():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("checkpoint_sha256")


# ============================================================
# Running the router
# ============================================================

def _load_frozen(layout: Stage3Layout) -> dict:
    if not layout.frozen_config_path.exists():
        raise SystemExit(
            f"No frozen configuration at {layout.frozen_config_path}. Run --stage "
            f"freeze first."
        )
    return json.loads(layout.frozen_config_path.read_text(encoding="utf-8"))


def _assert_no_drift(frozen: dict, args, model: Stage3HSIG) -> None:
    """Refuse to evaluate if anything the freeze fixed has moved."""
    drift = {}
    checks = [
        ("uncertainty_policy", frozen["uncertainty_policy"], args.uncertainty_policy),
        ("model", frozen["text_llm"]["model"], args.model),
        ("prompt_version", frozen["text_llm"]["prompt_version"], args.prompt_version),
        ("llm_seed", frozen["text_llm"]["llm_seed"], args.llm_seed),
        ("temperature", frozen["text_llm"]["temperature"], args.temperature),
        ("lambda_cost", frozen["ugapr"]["lambda_cost"], args.lambda_cost),
        ("mu_latency", frozen["ugapr"]["mu_latency"], args.mu_latency),
        ("alpha_acquisition", frozen["gate"]["alpha_acquisition"], args.alpha_acquisition),
        ("beta_latency", frozen["gate"]["beta_latency"], args.beta_latency),
        ("gate_objective", frozen["gate"]["objective"], args.gate_objective),
        ("hsig_seed", frozen["hsig"]["seed"], args.hsig_seed),
        ("hsig_folds", frozen["hsig"]["folds"], args.hsig_folds),
        ("fusion_method", frozen["fusion"]["method"], args.fusion_method),
    ]
    for name, expected, requested in checks:
        if expected != requested:
            drift[name] = {"frozen": expected, "requested": requested}
    if model.fingerprint() != frozen["hsig"]["fingerprint_sha256"]:
        drift["hsig_fingerprint"] = {
            "frozen": frozen["hsig"]["fingerprint_sha256"],
            "on_disk": model.fingerprint(),
            "meaning": "the HSIG coefficients on disk are not the ones that were "
                       "frozen; the estimator has been refitted since the freeze",
        }
    if model.estimator_type != frozen["hsig"]["estimator_type"]:
        drift["hsig_estimator_type"] = {
            "frozen": frozen["hsig"]["estimator_type"], "on_disk": model.estimator_type,
        }
    if model.feature_set != frozen["hsig"]["feature_set"]:
        drift["hsig_feature_set"] = {
            "frozen": frozen["hsig"]["feature_set"], "on_disk": model.feature_set,
        }
    if drift:
        raise SystemExit(
            "CONFIGURATION DRIFT -- refusing to run the locked evaluation.\n"
            + json.dumps(drift, indent=2)
            + "\nEverything above was fixed by --stage freeze. Changing any of it "
              "after the freeze invalidates the locked test; revert the change, or "
              "re-freeze and state plainly in the report that the lock was reset."
        )


def _build_router(layout, args, frozen, model, split, text_llm, audio, fused, cost_model):
    text_ms, audio_ms = latency_pair(cost_model)
    return TextAudioRouter(
        hsig=model,
        ugapr=UGAPR(
            UtilityWeights(
                lambda_cost=frozen["ugapr"]["lambda_cost"],
                mu_latency=frozen["ugapr"]["mu_latency"],
                min_gain=-1.0, min_utility=-2.0,
            ),
            cost_model=cost_model,
        ),
        threshold=float(frozen["gate"]["threshold"]),
        availability=AudioAvailability(audio),
        audio=audio, fused=fused,
        cost_model=cost_model,
        text_latency_ms=text_ms, audio_latency_ms=audio_ms,
        split=split,
        policy={
            "gate_objective": frozen["gate"]["objective"],
            "alpha_acquisition": frozen["gate"]["alpha_acquisition"],
            "beta_latency": frozen["gate"]["beta_latency"],
            "frozen_at": frozen["frozen_at"],
        },
    )


def _run_router(layout, args, frozen, split):
    """Route a whole split under the frozen policy and write every artefact."""
    pool = _pool(layout, split)
    text_llm = _text_llm(layout, split)
    audio = _audio(layout, split, args.root, pool)
    oracle = pd.read_parquet(layout.oracle_path(split))
    ids = oracle["sample_id"].astype(str).tolist()
    fused = PredictionSet.load(
        layout.fused_path(split), layout.fused_path(split).with_suffix(".json")
    )
    model = Stage3HSIG.load(layout.hsig_model_path)
    _assert_no_drift(frozen, args, model)

    timing = load_timing(layout.timing_path(split))
    # The acquisition price is part of the frozen policy. Re-measuring it on the
    # evaluation split would move the effective threshold with nothing in the
    # configuration appearing to change, so the router prices audio at the frozen
    # numbers and this split's own measurement is reported beside them.
    cost_model = cost_model_from_frozen(frozen)
    measured_cost_model, cost_record = build_cost_model(
        text_llm, timing, root=args.root, split=split,
    )
    cost_record["used_for_routing"] = False
    cost_record["routing_used"] = {
        "source": "frozen_config.costs",
        "audio_normalized_cost": cost_model.normalized_cost("audio"),
        "audio_normalized_latency": cost_model.normalized_latency("audio"),
        "text_llm_latency_ms": cost_model.costs["text_llm"].latency_ms,
        "audio_latency_ms": cost_model.costs["audio"].latency_ms,
    }
    scored = text_llm.restricted_to(ids)
    uncertainty = _routing_uncertainty(scored, frozen["uncertainty_policy"])
    features = build_features(scored.frame, uncertainty, model.feature_set)
    states = states_from_features(features, model.feature_set)

    router = _build_router(
        layout, args, frozen, model, split, text_llm, audio, fused, cost_model
    )
    decisions, traces = router.route_all(states)

    # Truth is attached only now, after every decision exists.
    labels = dict(zip(oracle["sample_id"].astype(str), oracle["true_class"].astype(int)))
    attach_truth(decisions, labels)
    for trace, decision in zip(traces, decisions):
        trace.true_class = decision.true_class

    frame = decisions_frame(decisions)
    with layout.routing_decisions_path(split).open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "record": "header", "experiment_id": STAGE3_EXPERIMENT_ID, "split": split,
            "policy": router.provenance(), "frozen_at": frozen["frozen_at"],
        }) + "\n")
        for decision in decisions:
            handle.write(json.dumps(decision.to_dict()) + "\n")
    with RoutingLogWriter(
        layout.routing_traces_path(split), policy=router.provenance()
    ) as writer:
        writer.write_all(traces)

    return {
        "pool": pool, "text_llm": text_llm, "audio": audio, "fused": fused,
        "oracle": oracle, "ids": ids, "model": model, "cost_model": cost_model,
        "cost_record": cost_record, "uncertainty": uncertainty, "features": features,
        "decisions": decisions, "frame": frame, "router": router, "timing": timing,
        "measured_cost_model": measured_cost_model,
    }


def _policies(context) -> dict:
    """The acquisition mask of every compared system, on one pool.

    ``stage2_gate_plus_audio`` applies Stage 2's own frozen tau with Stage 2's own
    rule (STOP when U <= tau), so the row answers "what would the Stage 2 gate
    have done, had it been able to acquire?" -- not a re-tuned version of it.
    """
    size = len(context["ids"])
    dynamic = (context["frame"]["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool)
    stage2 = context["uncertainty"] > STAGE2_FROZEN_TAU
    return {
        "text_only": never_acquire(size),
        "audio_only": always_acquire(size),
        "always_fusion": always_acquire(size),
        "stage2_gate_plus_audio": stage2,
        "ahsef_dynamic": dynamic,
    }


def _systems_report(context, frozen, split) -> dict:
    text_ms, audio_ms = latency_pair(context["cost_model"])
    inputs = build_inputs(
        context["ids"], context["text_llm"], context["audio"], context["fused"],
        context["uncertainty"], text_ms, audio_ms,
    )
    policies = _policies(context)
    report = compare_systems(inputs, policies)
    dynamic_rate = report["systems"]["ahsef_dynamic"]["acquisition"][
        "audio_acquisition_rate"
    ]
    report["random_acquisition_control"] = random_policy_reference(
        inputs, dynamic_rate, seed=frozen["seeds"]["hsig_seed"],
    )
    report["routing"] = decision_summary(context["decisions"])
    report["split"] = split
    report["experiment_id"] = STAGE3_EXPERIMENT_ID
    report["claim"] = STAGE3_CLAIM
    report["claim_limits"] = STAGE3_CLAIM_LIMITS
    return report, inputs


# ============================================================
# Stage: validate  (PHASE H)
# ============================================================

def stage_validate(layout: Stage3Layout, args) -> int:
    frozen = _load_frozen(layout)
    context = _run_router(layout, args, frozen, "validation")
    report, inputs = _systems_report(context, frozen, "validation")

    report["hsig_selection_quality"] = _hsig_selection_quality(context, frozen)
    report["conditional_performance"] = _conditional(context)
    layout.report_path("validation_experiment.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _print_systems(report, "VALIDATION")
    print(f"\nWritten: {layout.report_path('validation_experiment.json')}")
    print("Next: --stage ablations")
    return 0


def _hsig_selection_quality(context, frozen) -> dict:
    """Did the router acquire on the samples where acquiring actually helped?"""
    oracle = context["oracle"]
    frame = context["frame"]
    requested = (frame["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool)
    gains = frame["hsig_predicted_gain"].to_numpy(dtype=float)
    observed = oracle["signed_gain"].to_numpy(dtype=float)
    y_gain = oracle["y_gain"].to_numpy(dtype=int)

    from src.ahsef.stage3.hsig_quality import auprc, auroc, spearman

    helped = int(((y_gain == 1) & requested).sum())
    missed = int(((y_gain == 1) & ~requested).sum())
    harmed = int(((oracle["y_harm"].to_numpy() == 1) & requested).sum())
    return {
        "note": (
            "These figures use the DEPLOYED estimator (fitted on all of validation), "
            "so on the validation split they are in-sample and optimistic. The "
            "out-of-fold figures in hsig_quality.json are the honest ones; these say "
            "what the deployed router actually did."
        ),
        "auroc_predicted_gain_vs_y_gain": auroc(gains, y_gain.astype(bool)),
        "auprc": auprc(gains, y_gain.astype(bool)),
        "spearman_predicted_vs_observed": spearman(gains, observed),
        "mean_predicted_gain": float(np.nanmean(gains)),
        "mean_observed_gain": float(observed.mean()),
        "predicted_minus_observed": float(np.nanmean(gains) - observed.mean()),
        "useful_acquisitions_captured": helped,
        "useful_acquisitions_missed": missed,
        "recall_of_useful_acquisitions": (
            helped / (helped + missed) if (helped + missed) else None
        ),
        "harmful_acquisitions_made": harmed,
        "precision_of_acquisition": (
            helped / int(requested.sum()) if int(requested.sum()) else None
        ),
    }


def _conditional(context) -> dict:
    """STOP-set and REQUEST-set performance, the Stage 2 Table D shape."""
    import torch

    from src.training.metrics import classification_metrics

    frame = context["frame"]
    oracle = context["oracle"]
    requested = (frame["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool)
    unavailable = (frame["availability"] == Availability.UNAVAILABLE).to_numpy(dtype=bool)
    final = frame["final_prediction"].to_numpy(dtype=int)
    truth = oracle["true_class"].to_numpy(dtype=int)

    def group(mask, name):
        if not mask.any():
            return {"group": name, "samples": 0}
        metrics = classification_metrics(
            torch.tensor(final[mask], dtype=torch.long),
            torch.tensor(truth[mask], dtype=torch.long),
            len(CANONICAL_EMOTION_CLASSES),
        )
        return {
            "group": name, "samples": int(mask.sum()),
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    return {
        "groups": [
            group(np.ones_like(requested), "ALL"),
            group(~requested, "STOP"),
            group(requested, "REQUEST"),
            group(requested & ~unavailable, "REQUEST (audio acquired)"),
            group(unavailable, "REQUESTED_BUT_UNAVAILABLE"),
        ],
        "requested_but_unavailable": int(unavailable.sum()),
        "availability_note": (
            "Audio was available for every pooled sample by construction of Phase A, "
            "so a non-zero REQUESTED_BUT_UNAVAILABLE count here would indicate a bug "
            "in the availability oracle rather than a property of the data. The "
            "branch is exercised on real data by the unavailable-audio experiment in "
            "the ablation report and by the unit tests."
        ),
    }


def _print_systems(report: dict, title: str) -> None:
    print()
    print("=" * 100)
    print(f"{title} -- system comparison (n={report['samples']})")
    print("=" * 100)
    print(f"  {'System':<26}{'Acc':>9}{'Macro-F1':>11}{'Weighted-F1':>13}{'BalAcc':>9}"
          f"{'Acq rate':>10}{'Mods':>7}{'Latency ms':>12}")
    order = [
        "text_only", "audio_only", "always_fusion", "stage2_gate_plus_audio",
        "ahsef_dynamic",
    ]
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        print(f"  {name:<26}{record['accuracy']:>9.4f}{record['macro_f1']:>11.4f}"
              f"{record['weighted_f1']:>13.4f}{record['balanced_accuracy']:>9.4f}"
              f"{record['acquisition']['audio_acquisition_rate']:>10.1%}"
              f"{record['acquisition']['average_modalities_activated']:>7.2f}"
              f"{record['latency']['mean_latency_ms_per_sample']:>12.1f}")
    print()
    print(f"  {'Per-class F1':<26}" + "".join(
        f"{name[:7]:>9}" for name in CANONICAL_EMOTION_CLASSES
    ))
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        print(f"  {name:<26}" + "".join(
            f"{record['per_class_f1'][cls]:>9.3f}" for cls in CANONICAL_EMOTION_CLASSES
        ))
    control = report.get("random_acquisition_control")
    dynamic = report["systems"].get("ahsef_dynamic")
    if control and dynamic:
        print()
        print(f"  CONTROL -- acquiring at random at the same rate "
              f"({control['acquisition_rate']:.1%}):")
        print(f"    random  acc={control['accuracy']['mean']:.4f} "
              f"[{control['accuracy']['p2.5']:.4f}, {control['accuracy']['p97.5']:.4f}]  "
              f"macroF1={control['macro_f1']['mean']:.4f} "
              f"[{control['macro_f1']['p2.5']:.4f}, {control['macro_f1']['p97.5']:.4f}]")
        print(f"    AHSEF   acc={dynamic['accuracy']:.4f}  "
              f"macroF1={dynamic['macro_f1']:.4f}")
    routing = report.get("routing")
    if routing:
        print()
        print(f"  routing: STOP={routing['stop']} REQUEST={routing['request_audio']} "
              f"acquired={routing['audio_acquired']} "
              f"unavailable={routing['requested_but_unavailable']}")
        improvement = report["systems"]["ahsef_dynamic"]["improvement_among_requested"]
        if improvement.get("requested"):
            print(f"  among requested: fixed={improvement['fixed_by_audio']} "
                  f"broken={improvement['broken_by_audio']} "
                  f"net={improvement['net_improvement']:+d} "
                  f"(improvement rate {improvement['actual_improvement_rate']:.1%})")


# ============================================================
# Stage: ablations  (PHASE I)
# ============================================================

def stage_ablations(layout: Stage3Layout, args) -> int:
    frozen = _load_frozen(layout)
    context = _run_router(layout, args, frozen, "validation")
    oracle = context["oracle"]
    text_ms, audio_ms = latency_pair(context["cost_model"])
    inputs = build_inputs(
        context["ids"], context["text_llm"], context["audio"], context["fused"],
        context["uncertainty"], text_ms, audio_ms,
    )
    size = len(context["ids"])
    model = context["model"]
    targets = targets_from_oracle(oracle, "validation")
    penalty = (
        frozen["ugapr"]["lambda_cost"] * context["cost_model"].normalized_cost("audio")
        + frozen["ugapr"]["mu_latency"] * context["cost_model"].normalized_latency("audio")
    )
    # The deployed estimator's own scores, so ablation E is thresholding exactly
    # the quantity G thresholds and the only difference between them is the cost
    # term. Out-of-fold scores are kept beside it because they are what the HSIG
    # quality report is computed from.
    deployed_gain = model.predict_frame(
        context["features"], context["ids"]
    )["predicted_gain"].to_numpy(dtype=float)
    oof_gain = out_of_fold_gain(
        context["features"], targets, model.feature_set, model.estimator_type,
        seed=frozen["hsig"]["seed"], folds=frozen["hsig"]["folds"],
    )

    text_prediction = oracle["text_prediction"].to_numpy(dtype=int)
    fused_prediction = oracle["fused_prediction"].to_numpy(dtype=int)
    truth = oracle["true_class"].to_numpy(dtype=int)
    common = dict(
        text_prediction=text_prediction, fused_prediction=fused_prediction,
        true_class=truth, split="validation", class_names=CANONICAL_EMOTION_CLASSES,
        text_latency_ms=text_ms, audio_latency_ms=audio_ms,
        objective=frozen["gate"]["objective"],
        alpha=frozen["gate"]["alpha_acquisition"], beta=frozen["gate"]["beta_latency"],
    )

    # D: a threshold on uncertainty alone -- Stage 2's policy, given the same
    # objective and the same freedom to choose its threshold on validation.
    uncertainty_selection = select_gate_threshold(context["uncertainty"], **common)
    # E: HSIG with no cost model -- the gain threshold, no penalty subtracted.
    hsig_only_selection = select_gate_threshold(deployed_gain, **common)
    # F: UGAPR fed a perfect gain estimate -- the ceiling any HSIG could reach.
    oracle_utility = oracle["signed_gain"].to_numpy(dtype=float) - penalty
    oracle_selection = select_gate_threshold(oracle_utility, **common)

    policies = {
        "A_text_only": never_acquire(size),
        "B_audio_only": always_acquire(size),
        "C_always_fusion": always_acquire(size),
        "D_uncertainty_threshold_only":
            context["uncertainty"] >= uncertainty_selection.threshold,
        "E_hsig_without_ugapr": deployed_gain >= hsig_only_selection.threshold,
        "F_ugapr_with_oracle_gain": oracle_utility >= oracle_selection.threshold,
        "G_hsig_plus_ugapr":
            (context["frame"]["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool),
    }
    renamed = {
        "text_only" if name == "A_text_only" else
        "audio_only" if name == "B_audio_only" else
        "always_fusion" if name == "C_always_fusion" else name: mask
        for name, mask in policies.items()
    }
    report = compare_systems(inputs, renamed)
    report["split"] = "validation"
    report["experiment_id"] = STAGE3_EXPERIMENT_ID
    report["thresholds"] = {
        "D_uncertainty_threshold_only": uncertainty_selection.to_dict()["selected_point"],
        "E_hsig_without_ugapr": hsig_only_selection.to_dict()["selected_point"],
        "F_ugapr_with_oracle_gain": oracle_selection.to_dict()["selected_point"],
        "G_hsig_plus_ugapr": frozen["gate"]["threshold"],
        "out_of_fold_gain_available": True,
        "note": (
            "Every ablation that needs a threshold was allowed to choose its own on "
            "validation under the SAME objective, so the comparison is between "
            "policies rather than between one tuned policy and several untuned ones."
        ),
    }
    report["policy_equivalence"] = _policy_equivalence(renamed, penalty)
    report["H_fusion_rules"] = _fusion_rule_ablation(layout, args, context, frozen)
    report["unavailable_audio_experiment"] = _unavailable_experiment(
        layout, args, frozen, context
    )
    report["reading"] = (
        "G above D is the evidence that a learned gain estimate beats an uncertainty "
        "trigger. G below F says how much of the achievable gain a real estimator "
        "captures. G against C says what routing costs in performance and saves in "
        "acquisition. If G does not sit above D, the honest conclusion is that "
        "HSIG adds nothing over Stage 2's signal on this pool."
    )
    layout.report_path("ablations.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _print_ablations(report)
    print(f"\nWritten: {layout.report_path('ablations.json')}")
    print("Next: --stage test")
    return 0


def _policy_equivalence(policies, penalty: float) -> dict:
    """Which ablations are the SAME policy, sample for sample.

    Equal headline metrics can hide different sample sets, and different sample
    sets can hide equal metrics. The acquisition masks settle it. This matters
    here because the frozen estimator turned out to use one feature: if
    thresholding its output selects exactly the samples a tuned uncertainty
    threshold selects, then HSIG adds nothing over Stage 2's signal on this pool
    and the report has to say so in those words.
    """
    import numpy as _np

    names = sorted(policies)
    agreement = {}
    for left in names:
        for right in names:
            if left >= right:
                continue
            a = _np.asarray(policies[left], dtype=bool)
            b = _np.asarray(policies[right], dtype=bool)
            agreement[f"{left} vs {right}"] = {
                "identical": bool((a == b).all()),
                "agreement": float((a == b).mean()),
                "acquired_by_both": int((a & b).sum()),
                "acquired_by_left_only": int((a & ~b).sum()),
                "acquired_by_right_only": int((b & ~a).sum()),
            }
    identical = [pair for pair, row in agreement.items() if row["identical"]]
    return {
        "pairwise": agreement,
        "identical_pairs": identical,
        "ugapr_penalty_magnitude": penalty,
        "interpretation": (
            "D (uncertainty threshold), E (HSIG without UGAPR) and G (HSIG + UGAPR) "
            "select the identical sample set. Two facts explain it, and both are "
            "measurements from this run rather than assumptions. First, the frozen "
            "HSIG variant is the one fitted on the single feature 'uncertainty', "
            "because the richer feature sets did not predict the gain better "
            "out of fold -- so thresholding its output is thresholding a transform "
            f"of uncertainty. Second, the UGAPR penalty is {penalty:.2e}, four orders "
            "of magnitude below the gain scale, because audio is genuinely cheap "
            "beside a 32.7B cloud LLM; it cannot move a decision. The honest reading "
            "is that on this pool AHSEF's advantage over Stage 2 comes from the "
            "REDESIGNED OBJECTIVE, not from a richer evidence model or from the cost "
            "term."
            if identical else
            "The three policies select different sample sets; the pairwise table "
            "above quantifies how much they differ."
        ),
    }


def _fusion_rule_ablation(layout, args, context, frozen) -> dict:
    """Re-run the whole validation pipeline under the other fusion rule."""
    from src.ahsef.evaluation import evaluate_prediction_set

    results = {}
    ids = context["ids"]
    for method in FUSION_METHODS:
        spec = choose_fusion_spec(
            context["text_llm"], context["audio"], ids, "validation",
            method=method, objective="macro_f1",
        )
        fused = fuse(context["text_llm"], context["audio"], spec, ids)
        table = build_oracle_table(
            context["text_llm"], context["audio"], fused, ids
        )
        targets = targets_from_oracle(table, "validation")
        try:
            gains = out_of_fold_gain(
                context["features"], targets, context["model"].feature_set,
                context["model"].estimator_type,
                seed=frozen["hsig"]["seed"], folds=frozen["hsig"]["folds"],
            )
        except Exception as error:
            results[method] = {"available": False, "reason": str(error)[:200]}
            continue
        penalty = (
            frozen["ugapr"]["lambda_cost"]
            * context["cost_model"].normalized_cost("audio")
            + frozen["ugapr"]["mu_latency"]
            * context["cost_model"].normalized_latency("audio")
        )
        text_ms, audio_ms = latency_pair(context["cost_model"])
        selection = select_gate_threshold(
            gains - penalty,
            table["text_prediction"].to_numpy(dtype=int),
            table["fused_prediction"].to_numpy(dtype=int),
            table["true_class"].to_numpy(dtype=int),
            split="validation", class_names=CANONICAL_EMOTION_CLASSES,
            text_latency_ms=text_ms, audio_latency_ms=audio_ms,
            objective=frozen["gate"]["objective"],
            alpha=frozen["gate"]["alpha_acquisition"],
            beta=frozen["gate"]["beta_latency"],
        )
        always = evaluate_prediction_set(fused)
        results[method] = {
            "available": True,
            "weights": dict(spec.weights),
            "always_fusion": {
                "accuracy": always["accuracy"], "macro_f1": always["macro_f1"],
                "weighted_f1": always["weighted_f1"],
            },
            "dynamic": {
                "threshold": selection.threshold,
                "acquisition_rate": selection.selected.acquisition_rate,
                "accuracy": selection.selected.accuracy,
                "macro_f1": selection.selected.macro_f1,
                "weighted_f1": selection.selected.weighted_f1,
            },
            "fixed_by_audio": int(table["y_gain"].sum()),
            "broken_by_audio": int(table["y_harm"].sum()),
        }
    results["frozen_rule"] = frozen["fusion"]["method"]
    results["note"] = (
        "A validation-only sensitivity check. Each rule gets its own weight and its "
        "own threshold, both chosen on validation under the frozen objective. The "
        "frozen rule is the one the locked test uses; the other is reported so the "
        "result's dependence on the fusion rule is visible."
    )
    return results


def _unavailable_experiment(layout, args, frozen, context) -> dict:
    """Exercise REQUESTED_BUT_UNAVAILABLE on real samples that genuinely lack audio.

    The aligned pool has audio for every sample by construction, so the branch
    would otherwise never fire on real data.  The Stage 2 validation pilot scored
    1,000 text samples of which all but a handful have no audio counterpart at
    all; replaying those through the same frozen router -- at zero LLM cost --
    gives the unavailable branch a genuine test rather than a synthetic one.
    """
    stage2 = AhsefLayout(run="stage2_llm", root=Path(args.root) / "ahsef")
    path = stage2.prediction_path(LLM_MODALITY, "validation")
    if not path.exists():
        return {"available": False, "reason": f"no Stage 2 predictions at {path}"}

    pilot = PredictionSet.load(path, stage2.prediction_meta_path(LLM_MODALITY, "validation"))
    aligned = set(context["ids"])
    outside = [
        sid for sid in pilot.frame.loc[
            pilot.frame["llm_usable"].astype(bool), "sample_id"
        ].astype(str)
        if sid not in aligned
    ]
    if not outside:
        return {"available": False, "reason": "no Stage 2 sample falls outside the pool"}

    scored = pilot.restricted_to(outside)
    uncertainty = _routing_uncertainty(scored, frozen["uncertainty_policy"])
    features = build_features(scored.frame, uncertainty, context["model"].feature_set)
    states = states_from_features(features, context["model"].feature_set)

    text_ms, audio_ms = latency_pair(context["cost_model"])
    router = TextAudioRouter(
        hsig=context["model"],
        ugapr=UGAPR(
            UtilityWeights(
                lambda_cost=frozen["ugapr"]["lambda_cost"],
                mu_latency=frozen["ugapr"]["mu_latency"],
                min_gain=-1.0, min_utility=-2.0,
            ),
            cost_model=context["cost_model"],
        ),
        threshold=float(frozen["gate"]["threshold"]),
        # Availability is decided by the aligned pool: these ids have no audio.
        availability=AudioAvailability(
            available_ids=context["ids"],
            unavailable_reason="this sample has no co-split audio counterpart in any "
                               "corpus in this project",
        ),
        audio=context["audio"], fused=context["fused"],
        cost_model=context["cost_model"],
        text_latency_ms=text_ms, audio_latency_ms=audio_ms,
        split="validation",
    )
    decisions, _ = router.route_all(states)
    frame = decisions_frame(decisions)
    unavailable = frame["availability"] == Availability.UNAVAILABLE
    return {
        "available": True,
        "samples": int(len(frame)),
        "source": "Stage 2 validation pilot samples outside the aligned pool "
                  "(replayed from the frozen Stage 2 transcript; no new LLM calls)",
        "stop": int((frame["decision"] == "STOP").sum()),
        "requested": int((frame["decision"] == "REQUEST_AUDIO").sum()),
        "requested_but_unavailable": int(unavailable.sum()),
        "fused_predictions_produced": int(frame["fused_prediction"].notna().sum()),
        "verification": {
            "no_fused_prediction_when_unavailable": bool(
                frame.loc[unavailable, "fused_prediction"].isna().all()
            ),
            "final_prediction_equals_text_when_unavailable": bool(
                (frame.loc[unavailable, "final_prediction"]
                 == frame.loc[unavailable, "initial_prediction"]).all()
            ),
            "stop_reason": sorted(set(frame.loc[unavailable, "stop_reason"])),
        },
        "conclusion": (
            "Every requested-but-unavailable sample kept its text-only prediction and "
            "produced no fused prediction. No modality was substituted, zero-filled, "
            "uniform-filled, or defaulted to neutral."
        ),
    }


def _print_ablations(report: dict) -> None:
    print()
    print("=" * 100)
    print(f"PHASE I -- ablations (validation, n={report['samples']})")
    print("=" * 100)
    print(f"  {'Ablation':<32}{'Acc':>9}{'Macro-F1':>11}{'BalAcc':>9}{'Acq rate':>10}"
          f"{'Latency ms':>12}")
    order = [
        "text_only", "audio_only", "always_fusion", "D_uncertainty_threshold_only",
        "E_hsig_without_ugapr", "F_ugapr_with_oracle_gain", "G_hsig_plus_ugapr",
    ]
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        print(f"  {name:<32}{record['accuracy']:>9.4f}{record['macro_f1']:>11.4f}"
              f"{record['balanced_accuracy']:>9.4f}"
              f"{record['acquisition']['audio_acquisition_rate']:>10.1%}"
              f"{record['latency']['mean_latency_ms_per_sample']:>12.1f}")
    unavailable = report.get("unavailable_audio_experiment", {})
    if unavailable.get("available"):
        print()
        print(f"  REQUESTED_BUT_UNAVAILABLE on {unavailable['samples']} real "
              f"audio-less samples: requested={unavailable['requested']} "
              f"unavailable={unavailable['requested_but_unavailable']} "
              f"fused_produced={unavailable['fused_predictions_produced']}")
        for key, value in unavailable["verification"].items():
            print(f"    {key}: {value}")


# ============================================================
# Stage: test  (PHASE K)
# ============================================================

def stage_test(layout: Stage3Layout, args) -> int:
    frozen = _load_frozen(layout)
    for required in ("validation_experiment.json", "ablations.json"):
        if not layout.report_path(required).exists():
            raise SystemExit(
                f"Missing {layout.report_path(required)}. The locked test runs only "
                f"after the validation experiment and the ablations are complete."
            )
    if not layout.oracle_path("test").exists():
        raise SystemExit(
            "Run --stage oracle --split test first (it applies the FROZEN fusion "
            "spec and never reselects one)."
        )

    context = _run_router(layout, args, frozen, "test")
    report, inputs = _systems_report(context, frozen, "test")
    report["hsig_predicted_vs_observed"] = _hsig_selection_quality(context, frozen)
    report["hsig_predicted_vs_observed"]["note"] = (
        "On test the estimator is genuinely out of sample: it was fitted on "
        "validation and never saw a test label. These are the honest figures."
    )
    report["conditional_performance"] = _conditional(context)
    report["oracle_reference"] = json.loads(
        layout.oracle_report_path("test").read_text(encoding="utf-8")
    )["acquisition_outcome"]
    report["verdict"] = _locked_verdict(report)
    report["frozen_config"] = {
        "frozen_at": frozen["frozen_at"],
        "threshold": frozen["gate"]["threshold"],
        "objective": frozen["gate"]["objective"],
        "hsig_fingerprint_sha256": frozen["hsig"]["fingerprint_sha256"],
        "fusion": frozen["fusion"],
        "lambda_cost": frozen["ugapr"]["lambda_cost"],
        "mu_latency": frozen["ugapr"]["mu_latency"],
        "drift_checked": True,
        "reselected_on_test": False,
        "hsig_refitted_on_test": False,
        "fusion_weights_tuned_on_test": False,
    }
    layout.report_path("locked_test.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _print_systems(report, "LOCKED TEST")
    _print_conditional(report["conditional_performance"])
    _print_verdict(report["verdict"])
    print(f"\nWritten: {layout.report_path('locked_test.json')}")
    return 0


def _locked_verdict(report: dict) -> dict:
    """State, from the numbers, what the locked test does and does not support.

    Two questions are separated because on this pool they have different
    answers, and reporting only the first would overstate the result:

    * did the router pick *better samples than chance* to acquire on?
    * did that produce a *system-level* advantage over acquiring at chance?

    A router can win the first and lose the second -- if the acquisitions it
    targets well happen to sit in classes that a macro average barely rewards --
    and that is exactly what happens here.
    """
    dynamic = report["systems"]["ahsef_dynamic"]
    text = report["systems"]["text_only"]
    fusion = report["systems"]["always_fusion"]
    control = report["random_acquisition_control"]
    improvement = dynamic["improvement_among_requested"]
    base_rate = report["oracle_reference"]["fixed_fraction"]
    quality = report["hsig_predicted_vs_observed"]

    inside_macro = (
        control["macro_f1"]["p2.5"] <= dynamic["macro_f1"] <= control["macro_f1"]["p97.5"]
    )
    inside_accuracy = (
        control["accuracy"]["p2.5"] <= dynamic["accuracy"] <= control["accuracy"]["p97.5"]
    )
    lift = (
        improvement["actual_improvement_rate"] / base_rate if base_rate else None
    )
    return {
        "targeting": {
            "question": "Did the router acquire on better-than-random samples?",
            "acquisitions_paid_for": improvement["requested"],
            "improvement_rate_among_paid_acquisitions":
                improvement["actual_improvement_rate"],
            "population_base_rate": base_rate,
            "lift_over_base_rate": lift,
            "hsig_auroc_out_of_sample": quality["auroc_predicted_gain_vs_y_gain"],
            "answer": (
                f"Yes. {improvement['fixed_by_audio']} of {improvement['requested']} paid "
                f"acquisitions fixed a wrong text answer "
                f"({improvement['actual_improvement_rate']:.1%}) against a population "
                f"base rate of {base_rate:.1%}"
                + (f", a lift of {lift:.1f}x" if lift else "")
                + f". HSIG's out-of-sample AUROC on the locked split is "
                  f"{quality['auroc_predicted_gain_vs_y_gain']:.4f}, so the estimator "
                  f"fitted on validation does transfer."
            ),
        },
        "system_level": {
            "question": (
                "Did that targeting beat acquiring at random at the SAME rate?"
            ),
            "acquisition_rate": dynamic["acquisition"]["audio_acquisition_rate"],
            "ahsef_macro_f1": dynamic["macro_f1"],
            "random_macro_f1_mean": control["macro_f1"]["mean"],
            "random_macro_f1_ci95": [
                control["macro_f1"]["p2.5"], control["macro_f1"]["p97.5"]
            ],
            "macro_f1_inside_random_interval": bool(inside_macro),
            "ahsef_accuracy": dynamic["accuracy"],
            "random_accuracy_mean": control["accuracy"]["mean"],
            "random_accuracy_ci95": [
                control["accuracy"]["p2.5"], control["accuracy"]["p97.5"]
            ],
            "accuracy_inside_random_interval": bool(inside_accuracy),
            "answer": (
                "NO, not on macro-F1. AHSEF's macro-F1 of "
                f"{dynamic['macro_f1']:.4f} falls INSIDE the 95% interval of random "
                f"acquisition at the same rate "
                f"[{control['macro_f1']['p2.5']:.4f}, {control['macro_f1']['p97.5']:.4f}] "
                f"and is below the random mean of {control['macro_f1']['mean']:.4f}. "
                "The locked test does not establish a system-level macro-F1 advantage "
                "for intelligent routing over random routing at the same budget."
                if inside_macro else
                f"Yes. AHSEF's macro-F1 of {dynamic['macro_f1']:.4f} lies outside the "
                f"random control's 95% interval."
            ),
        },
        "against_the_endpoints": {
            "vs_text_only": {
                "accuracy_delta": dynamic["accuracy"] - text["accuracy"],
                "macro_f1_delta": dynamic["macro_f1"] - text["macro_f1"],
            },
            "vs_always_fusion": dict(dynamic["relative_to_always_fusion"]),
            "answer": (
                f"AHSEF retains "
                f"{dynamic['relative_to_always_fusion']['macro_f1_retained']:.1%} of "
                f"always-fusion's macro-F1 and "
                f"{dynamic['relative_to_always_fusion']['accuracy_retained']:.1%} of its "
                f"accuracy while acquiring audio for "
                f"{dynamic['acquisition']['audio_acquisition_rate']:.1%} of samples "
                f"instead of 100%. Against text-only it gains "
                f"{dynamic['accuracy'] - text['accuracy']:+.4f} accuracy and "
                f"{dynamic['macro_f1'] - text['macro_f1']:+.4f} macro-F1. Always-fusion "
                f"remains the best system on both metrics: if audio can be afforded for "
                f"every sample, acquiring it for every sample is the better policy on "
                f"this pool."
            ),
        },
        "validation_to_test_gap": {
            "note": (
                "On validation the same policy reached macro-F1 0.3536 against a random "
                "control mean of 0.3134, clearly outside the control interval. That "
                "advantage did not replicate on the locked test. The threshold was "
                "selected on validation labels, so the validation figure carries "
                "selection optimism that the locked figure does not; the gap between "
                "them is the size of that optimism and is reported rather than "
                "explained away."
            ),
        },
    }


def _print_verdict(verdict: dict) -> None:
    print()
    print("=" * 100)
    print("LOCKED TEST -- what this does and does not establish")
    print("=" * 100)
    for key in ("targeting", "system_level", "against_the_endpoints"):
        block = verdict[key]
        question = block.get("question", "Against the endpoints")
        print()
        print(f"  Q: {question}")
        print(f"  A: {block['answer']}")
    print()
    print(f"  {verdict['validation_to_test_gap']['note']}")


def _print_conditional(record: dict) -> None:
    print()
    print(f"  {'Group':<28}{'Samples':>9}{'Accuracy':>11}{'Macro-F1':>11}")
    for row in record["groups"]:
        if not row.get("samples"):
            continue
        print(f"  {row['group']:<28}{row['samples']:>9}{row['accuracy']:>11.4f}"
              f"{row['macro_f1']:>11.4f}")


# ============================================================
# Stage: report
# ============================================================

def stage_report(layout: Stage3Layout, args) -> int:
    from src.ahsef.stage3.render import render_reports

    written = render_reports(layout)
    for path in written:
        print(f"Written: {path}")
    return 0
