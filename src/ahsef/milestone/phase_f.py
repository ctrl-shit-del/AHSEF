"""PHASE F -- the locked test evaluation of the frozen AHSEF routing policy.

This module measures.  It does not choose anything.

Every quantity that could have been chosen -- the routing target, the score, the
budget, the fusion weights, the prompt, the encoder, the probe, the cost model,
the seed -- was fixed in ``frozen_routing_policy.json`` during Phase D/E, on
validation, before the test partition was opened.  Phase F loads that file,
verifies that the world still matches it, applies it once, and reports what
happened.  There is no threshold to sweep here and no variant to prefer, and the
code contains no function that would let one be introduced.

The ten lock conditions
-----------------------
:func:`run_lock_checks` is deliberately run *before* any test payload is read.
It is not a formality: two of its conditions have caught real classes of mistake
in this project's history -- a pool silently re-derived under a changed manifest,
and a cost model quietly re-measured on the evaluation split.  Every condition
reports its own evidence, and :func:`assert_locks_passed` refuses to continue if
any of them fails rather than repairing it.

What the oracle is for
----------------------
:func:`oracle_scores` reads the test labels.  It is the only thing here that
does, it is reported as ``ANALYSIS ONLY -- LABEL-AWARE UPPER BOUND``, and it is
computed *after* the AHSEF acquisition set is already fixed, so there is no code
path by which it could influence a routing decision.  Its role is to say how much
of the available benefit the frozen policy actually captured, which a comparison
against always-fusion alone cannot say -- because on validation the oracle beat
always-fusion, and a ceiling that sits below a reference is not a ceiling.

Statistics
----------
Comparisons between systems are **paired**: the same resampled sample indices are
used to recompute both systems' metrics, because every system here predicts on
the identical pool and an unpaired interval would throw away that structure and
overstate the uncertainty. The resampling convention -- ``default_rng``, 2000
resamples, 2.5/97.5 percentiles -- is the one already used by
``src.ahsef.stage3.hsig_quality``, reused rather than re-invented.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.ahsef.milestone.budget import PoolOutcomes, acquire_top_k, oracle_scores
from src.ahsef.milestone.reproducibility import file_digest, sequence_fingerprint
from src.ahsef.milestone.selection import majority_classes_of, minority_f1_mass
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

PHASE_F_PROTOCOL = "ahsef.milestone.phase_f.v1"

#: Resamples for every paired interval, matching ``hsig_quality.BOOTSTRAP_RESAMPLES``.
BOOTSTRAP_RESAMPLES = 2000

#: The ten conditions the brief requires, named so a failure can be reported as
#: "condition 6 failed" rather than as a stack trace.
LOCK_CONDITIONS: tuple[str, ...] = (
    "1_frozen_policy_exists",
    "2_frozen_policy_fingerprint_recorded",
    "3_validation_configuration_frozen",
    "4_no_prior_test_label_use",
    "5_validation_and_test_disjoint",
    "6_test_manifest_matches_lock",
    "7_alignment_valid",
    "8_no_contaminated_ids",
    "9_model_fingerprints_match",
    "10_cost_configuration_matches",
)


class LockCheckError(RuntimeError):
    """Raised when a Phase F lock condition fails.  Never repaired silently."""


def _ok(name: str, passed: bool, detail: dict, why: str) -> dict:
    return {
        "condition": name,
        "passed": bool(passed),
        "detail": detail,
        "why_it_matters": why,
    }


# ============================================================
# The ten lock checks
# ============================================================

def run_lock_checks(
    policy_path: Path,
    phase_de_path: Path,
    stage3_config_path: Path,
    validation_pool_ids: Sequence[str],
    test_pool_record: Mapping,
    audio_manifest_path: Path,
    text_meta: Mapping,
    audio_meta: Mapping,
    extraction_provenance: Mapping,
    frozen_costs,
) -> dict:
    """Every condition that must hold before the locked partition is opened.

    Each check reports the evidence it used, not merely a boolean, so a reader
    can tell a check that passed from a check that could not run.
    """
    checks = []
    policy_path = Path(policy_path)

    # 1 -- the policy exists at all.
    exists = policy_path.exists()
    checks.append(_ok(
        LOCK_CONDITIONS[0], exists, {"path": str(policy_path), "exists": exists},
        "Without a frozen policy on disk there is nothing to evaluate, and any "
        "policy assembled here would have been chosen after the test split existed.",
    ))
    if not exists:
        return _summarise(checks)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    fingerprint = file_digest(policy_path)

    # 2 -- its fingerprint is recorded, so a later edit is detectable.
    checks.append(_ok(
        LOCK_CONDITIONS[1], bool(fingerprint), {
            "sha256": fingerprint,
            "frozen_at": policy.get("frozen_at"),
            "protocol_version": policy.get("protocol_version"),
        },
        "The fingerprint is what lets a reader confirm that the policy evaluated "
        "here is the policy that was frozen, rather than one edited afterwards.",
    ))

    # 3 -- the validation-side configuration is frozen and self-consistent.
    phase_de = json.loads(Path(phase_de_path).read_text(encoding="utf-8"))
    selected = policy.get("selected_target")
    budget = (policy.get("acquisition_budget") or {}).get("selected")
    consistent = (
        selected == (phase_de.get("selection") or {}).get("selected")
        and budget == ((phase_de.get("operating_point") or {}).get("selected_budget"))
        and policy.get("fusion", {}).get("weights")
        == phase_de.get("fusion", {}).get("weights")
    )
    checks.append(_ok(
        LOCK_CONDITIONS[2], consistent, {
            "selected_target": selected,
            "budget": budget,
            "fusion_weights": policy.get("fusion", {}).get("weights"),
            "phase_de_selected_target": (phase_de.get("selection") or {}).get("selected"),
            "phase_de_budget": (phase_de.get("operating_point") or {}).get(
                "selected_budget"
            ),
            "phase_de_fusion_weights": phase_de.get("fusion", {}).get("weights"),
        },
        "The frozen policy and the Phase D/E record must describe the same policy. "
        "If they disagree, one of them was edited and neither can be trusted.",
    ))

    # 4 -- no test label was used for any earlier decision.
    stage3 = json.loads(Path(stage3_config_path).read_text(encoding="utf-8"))
    declarations = {
        "policy.uses_test_labels": policy.get("uses_test_labels"),
        "policy.test_partition_opened": policy.get("test_partition_opened"),
        "policy.selection_rule.uses_test_labels": (
            policy.get("selection_rule", {}).get("uses_test_labels")
        ),
        "policy.selection_rule.uses_test_split": (
            policy.get("selection_rule", {}).get("uses_test_split")
        ),
        "policy.fusion.uses_test_labels": policy.get("fusion", {}).get("uses_test_labels"),
        "phase_de.test_partition_opened": (
            phase_de.get("test_lock", {}).get("test_partition_opened")
        ),
        "phase_de.test_labels_used": phase_de.get("test_lock", {}).get("test_labels_used"),
        "phase_de.thresholds_tuned_on_test": (
            phase_de.get("test_lock", {}).get("thresholds_tuned_on_test")
        ),
        "phase_d.test_partition_opened": (
            phase_de.get("phase_d", {}).get("test_partition_opened")
        ),
        "stage3.uses_test_labels": stage3.get("uses_test_labels"),
        "text_llm.uses_labels_for_prediction": text_meta.get("uses_labels_for_prediction"),
        "audio_strong.uses_labels_for_prediction": audio_meta.get(
            "uses_labels_for_prediction"
        ),
    }
    clean = all(value is False for value in declarations.values())
    checks.append(_ok(
        LOCK_CONDITIONS[3], clean, {
            "declarations": declarations,
            "decisions_covered": [
                "target selection", "threshold selection", "budget selection",
                "fusion-weight selection", "model selection", "prompt selection",
                "uncertainty selection",
            ],
        },
        "Every one of these was recorded by the phase that made the decision, at "
        "the time it made it. A flag written now would prove nothing.",
    ))

    # 5 -- the two pools share no sample.
    validation_ids = set(str(item) for item in validation_pool_ids)
    test_ids = set(str(item) for item in test_pool_record["sample_ids"])
    overlap = sorted(validation_ids & test_ids)
    checks.append(_ok(
        LOCK_CONDITIONS[4], not overlap, {
            "validation_pool_size": len(validation_ids),
            "test_pool_size": len(test_ids),
            "overlapping_ids": len(overlap),
            "examples": overlap[:5],
            "validation_fingerprint": sequence_fingerprint(sorted(validation_ids)),
            "test_fingerprint": sequence_fingerprint(sorted(test_ids)),
        },
        "A sample in both pools would have contributed to the policy that is now "
        "being evaluated on it.",
    ))

    # 6 -- the test manifest is the one that was locked.
    manifest = json.loads(Path(audio_manifest_path).read_text(encoding="utf-8"))
    locked_test = ((manifest.get("splits") or {}).get("test")) or {}
    recorded_pool_fingerprint = (stage3.get("alignment") or {}).get(
        "test_pool_fingerprint_sha256"
    )
    pool_matches = (
        test_pool_record.get("fingerprint_sha256") == recorded_pool_fingerprint
        and int(test_pool_record.get("size") or 0)
        == int((stage3.get("alignment") or {}).get("test_pool_size") or -1)
    )
    checks.append(_ok(
        LOCK_CONDITIONS[5], pool_matches, {
            "audio_manifest_test_fingerprint": locked_test.get("fingerprint_sha256"),
            "audio_manifest_test_samples": locked_test.get("samples"),
            "aligned_pool_fingerprint": test_pool_record.get("fingerprint_sha256"),
            "stage3_recorded_pool_fingerprint": recorded_pool_fingerprint,
            "aligned_pool_size": test_pool_record.get("size"),
            "stage3_recorded_pool_size": (stage3.get("alignment") or {}).get(
                "test_pool_size"
            ),
        },
        "The pool must be the one whose fingerprint was recorded before the policy "
        "was frozen, not one re-derived now from a manifest that may have moved.",
    ))

    # 7 -- the alignment itself is valid.
    labels = test_pool_record.get("labels") or {}
    alignment_valid = bool(
        test_pool_record.get("labels_used_for_inclusion") is False
        and len(test_pool_record["sample_ids"]) == int(test_pool_record["size"])
        and len(set(test_pool_record["sample_ids"]))
        == len(test_pool_record["sample_ids"])
        and (not labels or len(labels) == int(test_pool_record["size"]))
    )
    checks.append(_ok(
        LOCK_CONDITIONS[6], alignment_valid, {
            "modalities": test_pool_record.get("modalities"),
            "task": test_pool_record.get("task"),
            "size": test_pool_record.get("size"),
            "unique_ids": len(set(test_pool_record["sample_ids"])),
            "labels_used_for_inclusion": test_pool_record.get(
                "labels_used_for_inclusion"
            ),
            "pool_rule": test_pool_record.get("pool_rule"),
        },
        "The pool must contain each id once, must carry both modalities, and must "
        "not have used a label to decide membership.",
    ))

    # 8 -- no id leaked across splits.
    contamination = (test_pool_record.get("checks") or {}).get("cross_split") or {}
    cross_split_ids = contamination.get("shared_across_splits")
    contaminated = overlap or (cross_split_ids or 0)
    checks.append(_ok(
        LOCK_CONDITIONS[7], not contaminated, {
            "validation_test_overlap": len(overlap),
            "recorded_cross_split_check": contamination or "not recorded by the pool",
            "pool_excludes_cross_split_ids": (
                "ids shared across DIFFERENT splits are excluded by the pool rule"
                in (test_pool_record.get("pool_rule") or "")
                or "excluded" in (test_pool_record.get("pool_rule") or "")
            ),
        },
        "A sample that appears in another split's training manifest was seen by a "
        "model that is now being tested on it.",
    ))

    # 9 -- every model is the frozen one.
    text_prompt = (text_meta.get("prompt") or {}).get("sha256")
    text_model = (text_meta.get("provider") or {}).get("model")
    frozen_text = (policy.get("experts") or {}).get("text") or ""
    model_ok = bool(
        audio_meta.get("checkpoint_sha256")
        and audio_meta.get("checkpoint_unchanged") is not False
        and text_prompt
        and text_model
        and text_model.split(":")[0] in frozen_text.split(",")[0]
        and (extraction_provenance.get("extractor") or {}).get("fingerprint_sha256")
        and (extraction_provenance.get("extractor") or {}).get("frozen_encoder") is True
        and (extraction_provenance.get("encoder_trained_by_this_project") is False)
    )
    checks.append(_ok(
        LOCK_CONDITIONS[8], model_ok, {
            "text_model": text_model,
            "text_prompt_sha256": text_prompt,
            "text_prompt_version": (text_meta.get("prompt") or {}).get("version"),
            "text_uncertainty_policy": text_meta.get("uncertainty_policy"),
            "audio_experiment": audio_meta.get("experiment"),
            "audio_iteration": audio_meta.get("iteration"),
            "audio_probe_checkpoint_sha256": audio_meta.get("checkpoint_sha256"),
            "audio_checkpoint_unchanged": audio_meta.get("checkpoint_unchanged"),
            "wav2vec2_config_fingerprint": (
                extraction_provenance.get("extractor") or {}
            ).get("fingerprint_sha256"),
            "wav2vec2_bundle": (extraction_provenance.get("extractor") or {}).get("bundle"),
            "encoder_frozen": (extraction_provenance.get("extractor") or {}).get(
                "frozen_encoder"
            ),
            "frozen_policy_experts": policy.get("experts"),
        },
        "The evaluated experts must be the ones the policy was frozen around. A "
        "different probe checkpoint or a different prompt is a different system.",
    ))

    # 10 -- the cost model is the frozen one.
    recorded = (policy.get("costs") or {})
    cost_matches = bool(
        recorded.get("re_measured_on_evaluation_pool") is False
        and abs(
            float((recorded.get("audio_strong") or {}).get("latency_ms_per_sample", 0))
            - frozen_costs.audio_strong_latency_ms
        ) < 1e-6
        and abs(
            float((recorded.get("text_llm") or {}).get("latency_ms_per_sample", 0))
            - frozen_costs.text_llm_latency_ms
        ) < 1e-6
    )
    checks.append(_ok(
        LOCK_CONDITIONS[9], cost_matches, {
            "policy_audio_ms": (recorded.get("audio_strong") or {}).get(
                "latency_ms_per_sample"
            ),
            "loaded_audio_ms": frozen_costs.audio_strong_latency_ms,
            "policy_text_ms": (recorded.get("text_llm") or {}).get(
                "latency_ms_per_sample"
            ),
            "loaded_text_ms": frozen_costs.text_llm_latency_ms,
            "cost_fingerprint": frozen_costs.fingerprint(),
            "includes_encoder": (recorded.get("audio_strong") or {}).get(
                "includes_encoder"
            ),
            "re_measured_on_evaluation_pool": recorded.get(
                "re_measured_on_evaluation_pool"
            ),
        },
        "Re-pricing audio from test behaviour would make the reported saving a "
        "function of this run's host load rather than of the policy.",
    ))

    return _summarise(checks, policy, fingerprint)


def _summarise(checks, policy=None, fingerprint=None) -> dict:
    failed = [row["condition"] for row in checks if not row["passed"]]
    return {
        "protocol": PHASE_F_PROTOCOL,
        "conditions_declared": list(LOCK_CONDITIONS),
        "conditions_run": [row["condition"] for row in checks],
        "checks": checks,
        "failed": failed,
        "all_passed": not failed and len(checks) == len(LOCK_CONDITIONS),
        "frozen_policy_sha256": fingerprint,
        "frozen_policy": policy,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rule": (
            "If any condition fails, Phase F stops and reports which one. Nothing "
            "is repaired, substituted or worked around."
        ),
    }


def assert_locks_passed(record: Mapping) -> None:
    if not record.get("all_passed"):
        raise LockCheckError(
            f"Phase F lock check failed: {record.get('failed')}. The locked test "
            f"split stays shut. Do not repair the configuration -- report the "
            f"failing condition."
        )


# ============================================================
# Metrics
# ============================================================

def system_metrics(predictions: np.ndarray, truth: np.ndarray, name: str) -> dict:
    """Every aggregate and per-class number Phase F asks for, for one system."""
    num_classes = len(CANONICAL_EMOTION_CLASSES)
    metrics = classification_metrics(
        torch.tensor(np.asarray(predictions, dtype=int), dtype=torch.long),
        torch.tensor(np.asarray(truth, dtype=int), dtype=torch.long),
        num_classes,
    )
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    return {
        "system": name,
        "samples": int(len(truth)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": {
            klass: {
                "support": metrics["support"][index],
                "precision": metrics["per_class_precision"][index],
                "recall": metrics["per_class_recall"][index],
                "f1": metrics["per_class_f1"][index],
            }
            for index, klass in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "per_class_f1": {
            klass: metrics["per_class_f1"][index]
            for index, klass in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": list(CANONICAL_EMOTION_CLASSES),
    }


def routed_predictions(
    text_prediction: np.ndarray, fused_prediction: np.ndarray, acquire: np.ndarray
) -> np.ndarray:
    """The frozen decision rule, written once and used by every routed system."""
    return np.where(
        np.asarray(acquire, dtype=bool),
        np.asarray(fused_prediction, dtype=int),
        np.asarray(text_prediction, dtype=int),
    )


def routing_metrics(
    acquire: np.ndarray, uncertainty: np.ndarray, name: str, budget: float
) -> dict:
    """Acquisition counts, rates and the uncertainty split, for one routed system."""
    acquire = np.asarray(acquire, dtype=bool)
    uncertainty = np.asarray(uncertainty, dtype=float)
    eligible = acquire.size
    return {
        "policy": name,
        "frozen_budget": float(budget),
        "eligible_samples": int(eligible),
        "acquisition_count": int(acquire.sum()),
        "acquisition_rate": float(acquire.mean()) if eligible else 0.0,
        "budget_respected": bool(
            abs(int(acquire.sum()) - round(budget * eligible)) <= 0
        ),
        "text_only_samples": int((~acquire).sum()),
        "text_plus_audio_samples": int(acquire.sum()),
        "average_modalities_per_sample": 1.0 + (
            float(acquire.mean()) if eligible else 0.0
        ),
        "uncertainty": {
            "all": _distribution(uncertainty),
            "acquired": _distribution(uncertainty[acquire]) if acquire.any() else None,
            "not_acquired": (
                _distribution(uncertainty[~acquire]) if (~acquire).any() else None
            ),
            "separation": (
                float(uncertainty[acquire].mean() - uncertainty[~acquire].mean())
                if acquire.any() and (~acquire).any() else None
            ),
        },
    }


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"samples": 0}
    return {
        "samples": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def acquisition_quality(
    acquire: np.ndarray,
    text_correct: np.ndarray,
    fused_correct: np.ndarray,
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    name: str,
) -> dict:
    """The Phase D/E acquisition-quality definitions, applied unchanged.

    ``correction precision`` is Phase E's ``useful_acquisition_rate``: corrections
    divided by acquisitions. The names differ because the Phase F brief names them
    differently; the arithmetic is identical and both are reported so the two
    reports can be read against each other.
    """
    acquire = np.asarray(acquire, dtype=bool)
    fixed = acquire & ~text_correct & fused_correct
    harmed = acquire & text_correct & ~fused_correct
    both_right = acquire & text_correct & fused_correct
    both_wrong = acquire & ~text_correct & ~fused_correct
    unchanged = acquire & (text_prediction == fused_prediction)
    count = int(acquire.sum())
    return {
        "policy": name,
        "acquisitions": count,
        "text_wrong_audio_fixes": int(fixed.sum()),
        "text_correct_audio_breaks": int(harmed.sum()),
        "both_correct": int(both_right.sum()),
        "both_wrong": int(both_wrong.sum()),
        "net_corrections": int(fixed.sum()) - int(harmed.sum()),
        "unnecessary_acquisitions": int(unchanged.sum()),
        "correction_precision": float(fixed.sum() / count) if count else None,
        "harm_rate": float(harmed.sum() / count) if count else None,
        "net_correction_rate": (
            float((int(fixed.sum()) - int(harmed.sum())) / count) if count else None
        ),
        "useful_acquisition_rate": float(fixed.sum() / count) if count else None,
        "definition_source": "src.ahsef.milestone.budget.score_at_budget (Phase D/E)",
    }


# ============================================================
# Uncertainty transfer
# ============================================================

def uncertainty_transfer(
    uncertainty: np.ndarray, text_correct: np.ndarray, bins: int = 5
) -> dict:
    """Does the frozen uncertainty still separate right from wrong on test?

    Nothing is fitted, recalibrated or re-thresholded here. The question is
    whether the ordering the policy relies on survives the split, and the answer
    is a measurement of the frozen quantity as it stands.
    """
    from src.ahsef.stage3.hsig_quality import auroc, spearman

    uncertainty = np.asarray(uncertainty, dtype=float)
    wrong = ~np.asarray(text_correct, dtype=bool)

    edges = np.unique(np.quantile(uncertainty, np.linspace(0, 1, bins + 1)))
    rows = []
    for lower, upper in zip(edges, edges[1:]):
        last = upper == edges[-1]
        inside = (
            (uncertainty >= lower) & (uncertainty <= upper) if last
            else (uncertainty >= lower) & (uncertainty < upper)
        )
        if not inside.any():
            continue
        rows.append({
            "lower": float(lower),
            "upper": float(upper),
            "samples": int(inside.sum()),
            "text_accuracy": float((~wrong)[inside].mean()),
            "mean_uncertainty": float(uncertainty[inside].mean()),
        })

    return {
        "auroc_identifying_incorrect_text": auroc(uncertainty, wrong),
        "spearman_uncertainty_vs_error": spearman(uncertainty, wrong.astype(float)),
        "distribution": _distribution(uncertainty),
        "accuracy_by_uncertainty_bin": rows,
        "monotone_decreasing_accuracy": bool(
            len(rows) > 1
            and all(
                rows[index]["text_accuracy"] >= rows[index + 1]["text_accuracy"] - 1e-9
                for index in range(len(rows) - 1)
            )
        ),
        "recalibrated_on_test": False,
        "threshold_selected_on_test": False,
        "anything_fitted_on_test": False,
        "note": (
            "AUROC above 0.5 means high uncertainty marks samples the text expert "
            "gets wrong -- the property the whole routing policy rests on. This is "
            "a measurement of the frozen quantity, not a re-selection of it."
        ),
    }


# ============================================================
# Paired statistics
# ============================================================

def paired_bootstrap(
    left_predictions: np.ndarray,
    right_predictions: np.ndarray,
    truth: np.ndarray,
    metric: str = "macro_f1",
    seed: int = 42,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Interval on ``left - right``, resampling the SAME indices for both systems.

    Pairing matters here more than usual: every system predicts on the identical
    pool and agrees with every other on the large majority of samples, so an
    unpaired interval would be dominated by variation the comparison shares and
    would be far too wide to say anything.
    """
    num_classes = len(CANONICAL_EMOTION_CLASSES)
    left = np.asarray(left_predictions, dtype=int)
    right = np.asarray(right_predictions, dtype=int)
    truth = np.asarray(truth, dtype=int)

    def score(predictions, labels):
        return classification_metrics(
            torch.tensor(predictions, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            num_classes,
        )[metric]

    point = score(left, truth) - score(right, truth)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        index = rng.integers(0, truth.size, truth.size)
        labels = truth[index]
        if np.unique(labels).size < 2:
            continue
        draws.append(score(left[index], labels) - score(right[index], labels))
    if len(draws) < resamples // 10:
        return {"metric": metric, "point_estimate": float(point), "ci95": None}
    draws = np.asarray(draws, dtype=float)
    return {
        "metric": metric,
        "point_estimate": float(point),
        "ci95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "resamples": int(draws.size),
        "excludes_zero": bool(
            np.percentile(draws, 2.5) > 0 or np.percentile(draws, 97.5) < 0
        ),
        "paired": True,
        "seed": int(seed),
        "method": (
            "paired percentile bootstrap over sample indices; both systems are "
            "rescored on the identical resample. Convention matches "
            "src.ahsef.stage3.hsig_quality."
        ),
    }


def comparison(left: Mapping, right: Mapping, name: str) -> dict:
    """Deltas on all three headline metrics, for one system pair."""
    return {
        "comparison": name,
        "delta_macro_f1": left["macro_f1"] - right["macro_f1"],
        "delta_accuracy": left["accuracy"] - right["accuracy"],
        "delta_weighted_f1": left["weighted_f1"] - right["weighted_f1"],
        "relative_macro_f1_gain": (
            (left["macro_f1"] - right["macro_f1"]) / right["macro_f1"]
            if right["macro_f1"] else None
        ),
    }


def retained_fusion_gain(
    ahsef: Mapping, text_only: Mapping, always_fusion: Mapping
) -> dict:
    """The fraction of the fusion benefit the frozen policy actually captured."""
    fusion_gain = always_fusion["macro_f1"] - text_only["macro_f1"]
    ahsef_gain = ahsef["macro_f1"] - text_only["macro_f1"]
    return {
        "fusion_gain": fusion_gain,
        "ahsef_gain": ahsef_gain,
        "retained_gain": ahsef_gain / fusion_gain if fusion_gain else None,
        "definition": (
            "retained_gain = (MacroF1(AHSEF) - MacroF1(text_only)) / "
            "(MacroF1(always_fusion) - MacroF1(text_only))"
        ),
        "note": (
            "Undefined when always-fusion does not beat text-only on this split. A "
            "negative fusion gain makes the ratio meaningless rather than large, "
            "and it is reported as such rather than quoted."
            if fusion_gain <= 0 else None
        ),
    }


# ============================================================
# The verdict
# ============================================================

GENERALIZATION_RULE = {
    "A_GENERALIZES": (
        "AHSEF beats matched random acquisition on macro-F1 at the frozen budget, "
        "retains a meaningful fraction of the fusion gain, and its improvement is "
        "distributed across classes rather than confined to neutral"
    ),
    "B_PARTIAL": (
        "AHSEF offers a useful cost/performance trade-off but either does not "
        "clearly beat matched random or loses most of the fusion benefit"
    ),
    "C_DOES_NOT_GENERALIZE": (
        "AHSEF does not outperform matched random acquisition, or offers no "
        "meaningful advantage over text-only"
    ),
    "primary_metric": "macro_f1",
    "accuracy_is_not_the_criterion": (
        "Stage 2 measured that accuracy-shaped judgements on this corpus reward "
        "majority-class behaviour. Accuracy is reported beside macro-F1, never "
        "instead of it."
    ),
    "meaningful_fusion_gain_retained": 0.50,
}


def generalization_verdict(
    ahsef: Mapping,
    text_only: Mapping,
    random_policy: Mapping,
    always_fusion: Mapping,
    retained: Mapping,
    ahsef_vs_random: Mapping,
    majority_classes: Sequence[str],
    ahsef_routing: Mapping,
    always_fusion_routing: Mapping,
) -> dict:
    """Apply the declared classification.  Nothing here is tuned after the fact."""
    beats_random = ahsef["macro_f1"] > random_policy["macro_f1"]
    beats_random_confidently = bool(ahsef_vs_random.get("excludes_zero"))
    retained_fraction = retained.get("retained_gain")
    meaningful = bool(
        retained_fraction is not None
        and retained_fraction >= GENERALIZATION_RULE["meaningful_fusion_gain_retained"]
    )
    beats_text = ahsef["macro_f1"] > text_only["macro_f1"]

    minority_ahsef = minority_f1_mass(ahsef, majority_classes)
    minority_text = minority_f1_mass(text_only, majority_classes)
    distributed = bool(minority_ahsef > minority_text)

    cheaper = bool(
        ahsef_routing["acquisition_count"]
        <= 0.5 * always_fusion_routing["acquisition_count"]
    )

    if beats_random and meaningful and distributed and beats_text:
        verdict = "A. GENERALIZES"
    elif beats_text and (beats_random or meaningful):
        verdict = "B. PARTIAL GENERALIZATION"
    else:
        verdict = "C. DOES NOT GENERALIZE"

    return {
        "rule": dict(GENERALIZATION_RULE),
        "criterion_1_beats_matched_random": {
            "met": beats_random,
            "ahsef_macro_f1": ahsef["macro_f1"],
            "random_macro_f1": random_policy["macro_f1"],
            "delta": ahsef["macro_f1"] - random_policy["macro_f1"],
            "paired_ci95": ahsef_vs_random.get("ci95"),
            "ci_excludes_zero": beats_random_confidently,
        },
        "criterion_2_retains_meaningful_fusion_gain": {
            "met": meaningful,
            "retained_gain": retained_fraction,
            "required": GENERALIZATION_RULE["meaningful_fusion_gain_retained"],
            "fusion_gain": retained.get("fusion_gain"),
        },
        "criterion_3_substantially_fewer_modalities": {
            "met": cheaper,
            "ahsef_acquisitions": ahsef_routing["acquisition_count"],
            "always_fusion_acquisitions": always_fusion_routing["acquisition_count"],
            "average_modalities_per_sample": ahsef_routing[
                "average_modalities_per_sample"
            ],
        },
        "criterion_4_not_only_neutral": {
            "met": distributed,
            "majority_classes": list(majority_classes),
            "minority_f1_mass_text_only": minority_text,
            "minority_f1_mass_ahsef": minority_ahsef,
            "delta": minority_ahsef - minority_text,
            "definition_source": "src.ahsef.milestone.selection.minority_f1_mass",
        },
        "criterion_5_beats_text_only": {
            "met": beats_text,
            "delta_macro_f1": ahsef["macro_f1"] - text_only["macro_f1"],
        },
        "verdict": verdict,
        "policy_changed_after_seeing_test": False,
    }


# ============================================================
# Routing trace
# ============================================================

def routing_trace(
    sample_ids: Sequence[str],
    uncertainty: np.ndarray,
    rank: np.ndarray,
    acquire: np.ndarray,
    text_prediction: np.ndarray,
    fused_prediction: np.ndarray,
    final_prediction: np.ndarray,
    true_class: np.ndarray,
) -> list[dict]:
    """One auditable line per sample: what was decided, why, and what followed.

    The true class is included because the trace is an analysis artefact written
    after every decision was made. It is not an input to any of them, and the
    ``decided_on`` field names the only quantity that was.
    """
    rows = []
    for index, identifier in enumerate(sample_ids):
        acquired = bool(acquire[index])
        rows.append({
            "sample_id": str(identifier),
            "uncertainty": float(uncertainty[index]),
            "uncertainty_rank": int(rank[index]),
            "acquired_audio": acquired,
            "decided_on": "normalised score entropy of the Gemma response",
            "modalities_used": ["text_llm", "audio_strong"] if acquired else ["text_llm"],
            "text_prediction": int(text_prediction[index]),
            "fused_prediction": int(fused_prediction[index]),
            "final_prediction": int(final_prediction[index]),
            "true_class": int(true_class[index]),
            "final_correct": bool(final_prediction[index] == true_class[index]),
            "text_correct": bool(text_prediction[index] == true_class[index]),
            "outcome": _outcome(
                acquired,
                bool(text_prediction[index] == true_class[index]),
                bool(fused_prediction[index] == true_class[index]),
            ),
        })
    return rows


def _outcome(acquired: bool, text_correct: bool, fused_correct: bool) -> str:
    if not acquired:
        return "text_only_correct" if text_correct else "text_only_wrong"
    if not text_correct and fused_correct:
        return "corrected"
    if text_correct and not fused_correct:
        return "harmed"
    return "both_correct" if text_correct else "both_wrong"


def ranking_fingerprint(sample_ids: Sequence[str], acquire: np.ndarray) -> str:
    """Hash of the acquired id set, so the exact routing decision is auditable."""
    acquired = sorted(
        str(identifier)
        for identifier, flag in zip(sample_ids, np.asarray(acquire, dtype=bool))
        if flag
    )
    return hashlib.sha256(
        json.dumps(acquired, sort_keys=True).encode("utf-8")
    ).hexdigest()
