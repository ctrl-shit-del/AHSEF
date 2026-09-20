"""PHASE F -- the one locked test evaluation of the frozen AHSEF policy.

    # verify every lock condition WITHOUT touching a test payload
    python -m src.ahsef.cli.run_milestone_phase_f --stage lock

    # open the locked split and evaluate, once
    python -m src.ahsef.cli.run_milestone_phase_f --stage evaluate \\
        --i-am-opening-the-locked-test-split

This driver applies ``experiments/ahsef/milestone/frozen_routing_policy.json``
and reports what happened.  It contains no threshold to sweep, no variant to
prefer and no branch that reads a test metric before making a decision -- the
only decision it makes is the frozen one, and the frozen one reads uncertainty.

``--stage lock`` runs the ten lock conditions and stops.  It reads prediction
*metadata* and the recorded pool manifest, never a test label or a test payload,
so the checks can be inspected before the split is opened.  ``--stage evaluate``
re-runs them and refuses to proceed unless all ten pass.

The Gemma test responses were recorded during Stage 3 and are replayed from the
stored transcript, so this evaluation makes no API call.  Strong-audio test
predictions are scored from the frozen probe checkpoint over cached wav2vec2
features; if that cache is absent the driver says so and stops rather than
extracting features as a side effect of an evaluation run.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.fusion import FusionSpec, fuse_prediction_sets
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.milestone import MILESTONE_EXPERIMENT_ID
from src.ahsef.milestone.budget import acquire_top_k, oracle_scores, PoolOutcomes
from src.ahsef.milestone.costs import assert_encoder_is_charged, load_frozen_costs
from src.ahsef.milestone.phase_f import (
    LOCK_CONDITIONS,
    PHASE_F_PROTOCOL,
    acquisition_quality,
    assert_locks_passed,
    comparison,
    generalization_verdict,
    paired_bootstrap,
    ranking_fingerprint,
    retained_fusion_gain,
    routed_predictions,
    routing_metrics,
    routing_trace,
    run_lock_checks,
    system_metrics,
    uncertainty_transfer,
)
from src.ahsef.milestone.reproducibility import (
    array_fingerprint,
    file_digest,
    git_revision,
    prediction_fingerprint,
    sequence_fingerprint,
    verify_locked_artefacts,
)
from src.ahsef.milestone.selection import majority_classes_of
from src.ahsef.stage3.pool import AlignedPool
from src.common.labels import CANONICAL_EMOTION_CLASSES

SPLIT = "test"
MILESTONE = Path("experiments") / "ahsef" / "milestone"
ANALYSIS = Path("experiments") / "ahsef" / "analysis" / "audio_expert"
STRONG_MODALITY = "audio_strong"

POLICY_PATH = MILESTONE / "frozen_routing_policy.json"
PHASE_DE_PATH = MILESTONE / "phase_de_results.json"
STAGE3_CONFIG = Path("experiments") / "ahsef" / "stage3_text_audio" / "frozen_config.json"
AUDIO_MANIFEST = (
    Path("experiments") / "audio_strong" / "full" / "metadata"
    / "audio_strong_manifest.json"
)
TEST_EXTRACTION_PROVENANCE = (
    Path("experiments") / "audio_strong" / "features" / "test"
    / "extraction_provenance.json"
)
VALIDATION_EXTRACTION_PROVENANCE = (
    Path("experiments") / "audio_strong" / "features" / "validation"
    / "extraction_provenance.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", default="lock", choices=["lock", "evaluate"])
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--stage3-run", default="stage3_text_audio")
    parser.add_argument("--output", default=str(MILESTONE))
    parser.add_argument(
        "--i-am-opening-the-locked-test-split", action="store_true",
        help="Required to evaluate. The locked split is opened once, deliberately.",
    )
    return parser


# ============================================================
# Inputs
# ============================================================

def _layout(args) -> AhsefLayout:
    return AhsefLayout(run=args.stage3_run, root=Path(args.root) / "ahsef")


def load_metadata(args) -> dict:
    """Prediction metadata and pool records -- no test payload, no test label."""
    layout = _layout(args)
    text_meta_path = layout.prediction_meta_path(LLM_MODALITY, SPLIT)
    strong_meta_path = ANALYSIS / f"{STRONG_MODALITY}__{SPLIT}.json"
    if not text_meta_path.exists():
        raise SystemExit(
            f"Missing the Stage 3 Gemma test export metadata at {text_meta_path}. "
            f"Phase F replays recorded responses; it does not call the LLM."
        )
    # The strong expert may not have been scored on test yet. The lock stage must
    # still be able to run, so its metadata falls back to the validation export --
    # the checkpoint and encoder fingerprints are properties of the model, not of
    # the split, and condition 9 is about the model.
    strong_meta_path = (
        strong_meta_path if strong_meta_path.exists()
        else ANALYSIS / f"{STRONG_MODALITY}__validation.json"
    )
    extraction = (
        TEST_EXTRACTION_PROVENANCE if TEST_EXTRACTION_PROVENANCE.exists()
        else VALIDATION_EXTRACTION_PROVENANCE
    )
    return {
        "text_meta": json.loads(text_meta_path.read_text(encoding="utf-8")),
        "audio_meta": json.loads(strong_meta_path.read_text(encoding="utf-8")),
        "extraction": json.loads(extraction.read_text(encoding="utf-8")),
        "audio_meta_source": str(strong_meta_path),
        "extraction_source": str(extraction),
    }


def lock_stage(args) -> dict:
    """The ten conditions, run before any test payload is opened."""
    metadata = load_metadata(args)
    layout = _layout(args)

    validation_pool = AlignedPool.load(layout.alignment_dir / "pool_validation.json")
    test_pool_path = layout.alignment_dir / "pool_test.json"
    if not test_pool_path.exists():
        raise SystemExit(f"Missing the locked test pool record at {test_pool_path}.")
    # AlignedPool.load re-derives and verifies the recorded fingerprint, so an
    # edited pool file fails here rather than silently changing the evaluation.
    test_pool = AlignedPool.load(test_pool_path)

    costs = load_frozen_costs()
    assert_encoder_is_charged(costs)

    record = run_lock_checks(
        policy_path=POLICY_PATH,
        phase_de_path=PHASE_DE_PATH,
        stage3_config_path=STAGE3_CONFIG,
        validation_pool_ids=validation_pool.sample_ids,
        test_pool_record=test_pool.to_dict(),
        audio_manifest_path=AUDIO_MANIFEST,
        text_meta=metadata["text_meta"],
        audio_meta=metadata["audio_meta"],
        extraction_provenance=metadata["extraction"],
        frozen_costs=costs,
    )
    record["metadata_sources"] = {
        "audio_meta": metadata["audio_meta_source"],
        "extraction_provenance": metadata["extraction_source"],
        "test_pool": str(test_pool_path),
    }
    record["test_payload_read"] = False
    record["test_labels_read"] = False
    return record


def load_test_systems(args, test_pool: AlignedPool):
    """The locked test payload.  Only reached after every lock condition passes."""
    layout = _layout(args)
    text_path = layout.prediction_path(LLM_MODALITY, SPLIT)
    strong_path = ANALYSIS / f"{STRONG_MODALITY}__{SPLIT}.parquet"
    if not strong_path.exists():
        raise SystemExit(
            f"Missing the strong-audio test export at {strong_path}.\n"
            f"Score it from the frozen checkpoint over cached wav2vec2 test "
            f"features first:\n"
            f"  python -m src.training.extract_audio_features --split test "
            f"--i-am-running-the-locked-evaluation\n"
            f"  python -m src.ahsef.cli.export_strong_audio_test\n"
            f"Phase F will not extract features as a side effect of evaluating."
        )

    text = PredictionSet.load(text_path, layout.prediction_meta_path(LLM_MODALITY, SPLIT))
    strong = PredictionSet.load(strong_path, strong_path.with_suffix(".json"))

    usable = set(
        text.frame.loc[text.frame["llm_usable"].astype(bool), "sample_id"].astype(str)
    )
    covered = set(strong.sample_ids())
    pool = [item for item in test_pool.sample_ids if item in usable and item in covered]
    dropped = {
        "no_usable_gemma_evidence": [
            item for item in test_pool.sample_ids if item not in usable
        ],
        "no_cached_audio_features": [
            item for item in test_pool.sample_ids
            if item in usable and item not in covered
        ],
    }
    return pool, text.restricted_to(pool), strong.restricted_to(pool), dropped


# ============================================================
# Evaluation
# ============================================================

def evaluate(args) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    locks = lock_stage(args)
    _print_locks(locks)
    assert_locks_passed(locks)
    print("\n[lock] all ten conditions passed -- opening the locked test split\n")

    policy = locks["frozen_policy"]
    budget = float((policy["acquisition_budget"] or {})["selected"])
    seed = int((policy.get("seeds") or {}).get("random_control_seed", 42))
    spec = FusionSpec(
        method=policy["fusion"]["method"],
        weights=dict(policy["fusion"]["weights"]),
        selected_on_split=policy["fusion"].get("selected_on_split"),
        selection={**(policy["fusion"].get("selection") or {}),
                   "applied_by": "phase_f_without_reselection"},
    )

    layout = _layout(args)
    test_pool = AlignedPool.load(layout.alignment_dir / "pool_test.json")
    pool, text, strong, dropped = load_test_systems(args, test_pool)
    print(f"[pool] {len(pool)} of {test_pool.size} locked test samples are eligible")
    for reason, items in dropped.items():
        if items:
            print(f"[pool] dropped {len(items)} for {reason} (e.g. {items[:3]})")

    fused = fuse_prediction_sets(
        {"text_llm": text, "audio": strong}, spec, pool
    )
    print(f"[fuse] frozen weights {dict(spec.weights)} applied without reselection")

    truth = text.frame["true_class"].to_numpy(dtype=int)
    text_prediction = text.frame["predicted_class"].to_numpy(dtype=int)
    audio_prediction = strong.frame["predicted_class"].to_numpy(dtype=int)
    fused_prediction = fused.frame["predicted_class"].to_numpy(dtype=int)
    uncertainty = text.frame["normalized_entropy"].to_numpy(dtype=float)
    text_correct = text_prediction == truth
    fused_correct = fused_prediction == truth

    # ---- the frozen routing decision -------------------------------------
    acquire_ahsef = acquire_top_k(uncertainty, budget)
    order = np.argsort(-uncertainty, kind="stable")
    rank = np.empty(uncertainty.size, dtype=int)
    rank[order] = np.arange(1, uncertainty.size + 1)
    print(f"[rout] AHSEF acquires {int(acquire_ahsef.sum())} of {len(pool)} "
          f"({acquire_ahsef.mean():.2%}) at the frozen {budget:.0%} budget")

    # ---- matched random, label-independent -------------------------------
    rng = np.random.default_rng(seed)
    acquire_random = np.zeros(len(pool), dtype=bool)
    count = int(acquire_ahsef.sum())
    if count:
        acquire_random[rng.choice(len(pool), count, replace=False)] = True

    # ---- oracle: computed AFTER both acquisition sets are already fixed ---
    pool_outcomes = PoolOutcomes(
        sample_ids=list(pool), true_class=truth,
        text_prediction=text_prediction, fused_prediction=fused_prediction,
        audio_prediction=audio_prediction,
        text_latency_ms=0.0, audio_latency_ms=0.0,
    )
    acquire_oracle = acquire_top_k(oracle_scores(pool_outcomes), budget)

    acquire_none = np.zeros(len(pool), dtype=bool)
    acquire_all = np.ones(len(pool), dtype=bool)

    systems = {
        "A_text_only": system_metrics(text_prediction, truth, "A. Gemma text only"),
        "B_audio_strong_only": system_metrics(
            audio_prediction, truth, "B. Audio strong only"
        ),
        "C_always_fusion": system_metrics(
            fused_prediction, truth, "C. Always fusion"
        ),
        "D_random_at_budget": system_metrics(
            routed_predictions(text_prediction, fused_prediction, acquire_random),
            truth, "D. Random acquisition @10%",
        ),
        "E_ahsef_at_budget": system_metrics(
            routed_predictions(text_prediction, fused_prediction, acquire_ahsef),
            truth, "E. AHSEF dynamic @10%",
        ),
        "F_oracle_at_budget": system_metrics(
            routed_predictions(text_prediction, fused_prediction, acquire_oracle),
            truth, "F. Oracle @10% -- ANALYSIS ONLY, LABEL-AWARE UPPER BOUND",
        ),
    }
    systems["F_oracle_at_budget"]["label_aware"] = True
    systems["F_oracle_at_budget"]["marking"] = (
        "ANALYSIS ONLY -- LABEL-AWARE UPPER BOUND. Computed after the AHSEF and "
        "random acquisition sets were already fixed. It influenced no decision."
    )

    ahsef_final = routed_predictions(text_prediction, fused_prediction, acquire_ahsef)
    random_final = routed_predictions(text_prediction, fused_prediction, acquire_random)
    oracle_final = routed_predictions(text_prediction, fused_prediction, acquire_oracle)

    # ---- routing, cost, quality ------------------------------------------
    costs = load_frozen_costs()
    assert_encoder_is_charged(costs)

    routing = {
        "E_ahsef_at_budget": {
            **routing_metrics(acquire_ahsef, uncertainty, "ahsef", budget),
            "top_k_ranking_fingerprint": ranking_fingerprint(pool, acquire_ahsef),
        },
        "D_random_at_budget": {
            **routing_metrics(acquire_random, uncertainty, "random", budget),
            "seed": seed,
            "independent_of_test_labels": True,
            "top_k_ranking_fingerprint": ranking_fingerprint(pool, acquire_random),
        },
        "C_always_fusion": routing_metrics(acquire_all, uncertainty, "always_fusion", 1.0),
        "A_text_only": routing_metrics(acquire_none, uncertainty, "text_only", 0.0),
        "F_oracle_at_budget": {
            **routing_metrics(acquire_oracle, uncertainty, "oracle", budget),
            "label_aware": True,
        },
    }

    quality = {
        "E_ahsef_at_budget": acquisition_quality(
            acquire_ahsef, text_correct, fused_correct, text_prediction,
            fused_prediction, "ahsef",
        ),
        "D_random_at_budget": acquisition_quality(
            acquire_random, text_correct, fused_correct, text_prediction,
            fused_prediction, "random",
        ),
        "C_always_fusion": acquisition_quality(
            acquire_all, text_correct, fused_correct, text_prediction,
            fused_prediction, "always_fusion",
        ),
    }

    # ---- comparisons and statistics --------------------------------------
    comparisons = {
        "ahsef_vs_text_only": comparison(
            systems["E_ahsef_at_budget"], systems["A_text_only"], "AHSEF vs text-only"
        ),
        "ahsef_vs_random": comparison(
            systems["E_ahsef_at_budget"], systems["D_random_at_budget"],
            "AHSEF vs random @10%",
        ),
        "ahsef_vs_always_fusion": comparison(
            systems["E_ahsef_at_budget"], systems["C_always_fusion"],
            "AHSEF vs always-fusion",
        ),
        "ahsef_vs_oracle": comparison(
            systems["E_ahsef_at_budget"], systems["F_oracle_at_budget"],
            "AHSEF vs oracle @10% (label-aware ceiling)",
        ),
    }

    print("[stat] paired bootstrap intervals ...")
    statistics = {
        "ahsef_vs_text_only": {
            metric: paired_bootstrap(ahsef_final, text_prediction, truth, metric, seed)
            for metric in ("macro_f1", "accuracy", "weighted_f1")
        },
        "ahsef_vs_random": {
            metric: paired_bootstrap(ahsef_final, random_final, truth, metric, seed)
            for metric in ("macro_f1", "accuracy", "weighted_f1")
        },
        "ahsef_vs_always_fusion": {
            metric: paired_bootstrap(ahsef_final, fused_prediction, truth, metric, seed)
            for metric in ("macro_f1", "accuracy", "weighted_f1")
        },
        "always_fusion_vs_text_only": {
            "macro_f1": paired_bootstrap(
                fused_prediction, text_prediction, truth, "macro_f1", seed
            ),
        },
        "method": (
            "paired percentile bootstrap over sample indices, 2000 resamples, "
            "seed from the frozen policy. Both systems are rescored on the same "
            "resample because they predict on identical samples."
        ),
    }

    retained = retained_fusion_gain(
        systems["E_ahsef_at_budget"], systems["A_text_only"], systems["C_always_fusion"]
    )
    majority = majority_classes_of(truth, len(CANONICAL_EMOTION_CLASSES))
    verdict = generalization_verdict(
        systems["E_ahsef_at_budget"], systems["A_text_only"],
        systems["D_random_at_budget"], systems["C_always_fusion"], retained,
        statistics["ahsef_vs_random"]["macro_f1"], majority,
        routing["E_ahsef_at_budget"], routing["C_always_fusion"],
    )

    transfer = uncertainty_transfer(uncertainty, text_correct)

    # ---- artefacts --------------------------------------------------------
    trace = routing_trace(
        pool, uncertainty, rank, acquire_ahsef, text_prediction, fused_prediction,
        ahsef_final, truth,
    )
    trace_path = output / "phase_f_routing_trace.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(row, default=_json_default) for row in trace) + "\n",
        encoding="utf-8",
    )

    predictions_path = output / "phase_f_predictions.parquet"
    pd.DataFrame({
        "sample_id": pool,
        "dataset": text.frame["dataset"].tolist(),
        "true_class": truth,
        "uncertainty": uncertainty,
        "uncertainty_rank": rank,
        "text_prediction": text_prediction,
        "audio_prediction": audio_prediction,
        "fused_prediction": fused_prediction,
        "acquired_ahsef": acquire_ahsef,
        "acquired_random": acquire_random,
        "acquired_oracle": acquire_oracle,
        "ahsef_prediction": ahsef_final,
        "random_prediction": random_final,
        "oracle_prediction": oracle_final,
    }).to_parquet(predictions_path, index=False)

    record = {
        "experiment_id": MILESTONE_EXPERIMENT_ID,
        "protocol_version": PHASE_F_PROTOCOL,
        "phase": "F -- locked test evaluation",
        "split": SPLIT,
        "lock_checks": locks,
        "test_dataset": _dataset_block(test_pool, pool, text, dropped),
        "frozen_policy_applied": {
            "sha256": locks["frozen_policy_sha256"],
            "selected_target": policy["selected_target"],
            "routing_rule": policy["routing_rule"],
            "budget": budget,
            "fusion": spec.to_dict(),
            "seed": seed,
            "modified_by_phase_f": False,
        },
        "systems": systems,
        "comparisons": comparisons,
        "retained_fusion_gain": retained,
        "routing": routing,
        "acquisition_quality": quality,
        "uncertainty_transfer": transfer,
        "per_class_routing_analysis": _per_class_block(systems, majority),
        "cost": _cost_block(costs, routing, len(pool)),
        "statistics": statistics,
        "verdict": verdict,
        "leakage_audit": _leakage_audit(),
        "reproducibility": _provenance(
            pool, {"text_llm": text, "audio_strong": strong,
                   "text_llm+audio_strong": fused},
            test_pool, locks, policy, spec, costs, seed, budget,
            {"routing_trace": str(trace_path), "predictions": str(predictions_path)},
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_path = output / "phase_f_results.json"
    results_path.write_text(
        json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
    )
    _print_results(record)
    print(f"\nWritten: {results_path}")
    print(f"Written: {trace_path}")
    print(f"Written: {predictions_path}")
    return 0


# ============================================================
# Report blocks
# ============================================================

def _dataset_block(test_pool, pool, text, dropped) -> dict:
    frame = text.frame
    counts = frame["true_class"].value_counts().to_dict()
    return {
        "locked_pool_size": test_pool.size,
        "locked_pool_fingerprint": test_pool.fingerprint,
        "evaluated_samples": len(pool),
        "evaluated_fingerprint": sequence_fingerprint(pool),
        "excluded": {reason: len(items) for reason, items in dropped.items()},
        "excluded_ids": {
            reason: items[:20] for reason, items in dropped.items() if items
        },
        "datasets": frame["dataset"].value_counts().to_dict(),
        "class_distribution": {
            name: int(counts.get(index, 0))
            for index, name in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "resampled": False,
        "rebalanced": False,
        "manifest_changed": False,
        "ordering": "the recorded pool order, preserved verbatim",
    }


def _per_class_block(systems, majority) -> dict:
    from src.ahsef.milestone.selection import minority_f1_mass

    rows = {}
    for klass in CANONICAL_EMOTION_CLASSES:
        rows[klass] = {
            "support": systems["A_text_only"]["per_class"][klass]["support"],
            "text_only_f1": systems["A_text_only"]["per_class_f1"][klass],
            "audio_only_f1": systems["B_audio_strong_only"]["per_class_f1"][klass],
            "always_fusion_f1": systems["C_always_fusion"]["per_class_f1"][klass],
            "random_f1": systems["D_random_at_budget"]["per_class_f1"][klass],
            "ahsef_f1": systems["E_ahsef_at_budget"]["per_class_f1"][klass],
            "delta_ahsef_vs_text_only": (
                systems["E_ahsef_at_budget"]["per_class_f1"][klass]
                - systems["A_text_only"]["per_class_f1"][klass]
            ),
        }
    minority_ahsef = minority_f1_mass(systems["E_ahsef_at_budget"], majority)
    minority_text = minority_f1_mass(systems["A_text_only"], majority)
    neutral_delta = rows["neutral"]["delta_ahsef_vs_text_only"]
    total_delta = sum(row["delta_ahsef_vs_text_only"] for row in rows.values())
    return {
        "per_class": rows,
        "majority_classes": list(majority),
        "minority_f1_aggregate": {
            "definition": (
                "summed F1 over classes outside the two largest, "
                "src.ahsef.milestone.selection.minority_f1_mass"
            ),
            "text_only": minority_text,
            "ahsef": minority_ahsef,
            "delta": minority_ahsef - minority_text,
        },
        "neutral_share_of_total_f1_delta": (
            neutral_delta / total_delta if abs(total_delta) > 1e-12 else None
        ),
        "improvement_concentrated_in_neutral": bool(
            total_delta > 0 and neutral_delta / total_delta > 0.5
        ),
    }


def _cost_block(costs, routing, samples) -> dict:
    text_ms = costs.text_llm_latency_ms
    audio_ms = costs.audio_strong_latency_ms
    ahsef_rate = routing["E_ahsef_at_budget"]["acquisition_rate"]
    return {
        "configuration": costs.to_dict(),
        "re_estimated_from_test_behaviour": False,
        "per_sample_latency_ms": {
            "text_only": text_ms,
            "audio_strong_only": audio_ms,
            "always_fusion": text_ms + audio_ms,
            "ahsef_expected": text_ms + audio_ms * ahsef_rate,
            "random_expected": text_ms + audio_ms * routing["D_random_at_budget"][
                "acquisition_rate"
            ],
        },
        "average_modalities_per_sample": {
            "text_only": 1.0,
            "always_fusion": 2.0,
            "ahsef": routing["E_ahsef_at_budget"]["average_modalities_per_sample"],
        },
        "saving_versus_always_fusion": {
            "latency_ms_per_sample": audio_ms * (1.0 - ahsef_rate),
            "audio_activations_avoided": (
                routing["C_always_fusion"]["acquisition_count"]
                - routing["E_ahsef_at_budget"]["acquisition_count"]
            ),
            "audio_compute_avoided_fraction": 1.0 - ahsef_rate,
            "total_audio_seconds_saved_over_pool": (
                audio_ms * (1.0 - ahsef_rate) * samples / 1000.0
            ),
        },
        "monetary_cost": (
            "not reported. No price was measured for the Ollama cloud endpoint and "
            "inventing one would put a fabricated number beside measured ones."
        ),
    }


def _leakage_audit() -> dict:
    return {
        "test_labels_used_for_policy_selection": False,
        "test_labels_used_for_threshold_selection": False,
        "test_labels_used_for_budget_selection": False,
        "test_labels_used_for_model_selection": False,
        "test_labels_used_for_prompt_selection": False,
        "test_labels_used_for_fusion_selection": False,
        "test_labels_used_for_uncertainty_selection": False,
        "only_oracle_uses_test_labels": True,
        "oracle_marking": "ANALYSIS ONLY -- LABEL-AWARE UPPER BOUND",
        "oracle_computed_after_acquisition_sets_were_fixed": True,
        "policy_modified_after_seeing_test_results": False,
        "enforced_by": (
            "the frozen policy is loaded from disk and applied; this driver has no "
            "code path that selects a target, a threshold, a budget, a weight or a "
            "model, and lock condition 4 checks the declarations every earlier "
            "phase recorded at the time it made its decisions"
        ),
    }


def _provenance(pool, predictions, test_pool, locks, policy, spec, costs, seed,
                budget, artefacts) -> dict:
    text_meta = predictions["text_llm"].meta
    audio_meta = predictions["audio_strong"].meta
    extraction = (
        json.loads(TEST_EXTRACTION_PROVENANCE.read_text(encoding="utf-8"))
        if TEST_EXTRACTION_PROVENANCE.exists() else {}
    )
    return {
        "protocol_version": PHASE_F_PROTOCOL,
        "frozen_policy_fingerprint": locks["frozen_policy_sha256"],
        "test_manifest_fingerprint": (
            json.loads(AUDIO_MANIFEST.read_text(encoding="utf-8"))
            .get("splits", {}).get("test", {}).get("fingerprint_sha256")
        ),
        "alignment_fingerprint": test_pool.fingerprint,
        "evaluated_pool_fingerprint": sequence_fingerprint(pool),
        "text_model_fingerprint": {
            "model": (text_meta.get("provider") or {}).get("model"),
            "provider": (text_meta.get("provider") or {}).get("provider"),
            "transcript": (text_meta.get("provider") or {}).get("transcript"),
            "deterministic": (text_meta.get("provider") or {}).get("deterministic"),
        },
        "prompt_fingerprint": {
            "version": (text_meta.get("prompt") or {}).get("version"),
            "sha256": (text_meta.get("prompt") or {}).get("sha256"),
        },
        "generation_configuration": {
            "repeats": text_meta.get("repeats"),
            "budget": text_meta.get("budget"),
            "temperature": ((policy.get("experts") or {}).get("text")),
            "uncertainty_policy": text_meta.get("uncertainty_policy"),
        },
        "audio_model_fingerprint": {
            "experiment": audio_meta.get("experiment"),
            "iteration": audio_meta.get("iteration"),
            "probe_checkpoint_sha256": audio_meta.get("checkpoint_sha256"),
            "model": (audio_meta.get("model") or {}).get("class"),
        },
        "wav2vec2_checkpoint_fingerprint": {
            "bundle": (extraction.get("extractor") or {}).get("bundle"),
            "config_fingerprint_sha256": (extraction.get("extractor") or {}).get(
                "fingerprint_sha256"
            ),
            "parameters": (extraction.get("extractor") or {}).get("parameters"),
            "frozen": (extraction.get("extractor") or {}).get("frozen_encoder"),
            "trained_by_this_project": extraction.get(
                "encoder_trained_by_this_project"
            ),
        },
        "fusion_configuration": spec.to_dict(),
        "uncertainty_configuration": {
            "policy": "score_entropy",
            "column": "normalized_entropy",
            "recalibrated_on_test": False,
        },
        "budget": budget,
        "seed": seed,
        "cost_configuration": costs.to_dict(),
        "cost_fingerprint": costs.fingerprint(),
        "model_fingerprints": {
            name: prediction_fingerprint(item) for name, item in predictions.items()
        },
        "sample_counts": {
            "locked_pool": test_pool.size,
            "evaluated": len(pool),
        },
        "stored_artefacts": dict(artefacts),
        "replay": (
            "Every input is a stored artefact. Re-running --stage evaluate "
            "reproduces every number offline: the Gemma responses are replayed "
            "from the Stage 3 transcript and the audio probe reads cached features. "
            "No API call is made."
        ),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git": git_revision(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value)} is not JSON serialisable")


# ============================================================
# Printing
# ============================================================

def _print_locks(record) -> None:
    print("=" * 100)
    print("PHASE F -- TEST LOCK CHECKS (no test payload is read by this stage)")
    print("=" * 100)
    for row in record["checks"]:
        mark = "PASS" if row["passed"] else "FAIL"
        print(f"  [{mark}] {row['condition']}")
        if not row["passed"]:
            print(f"         detail: {json.dumps(row['detail'])[:400]}")
    print(f"\n  conditions run : {len(record['checks'])}/{len(LOCK_CONDITIONS)}")
    print(f"  all passed     : {record['all_passed']}")
    print(f"  policy sha256  : {record['frozen_policy_sha256']}")
    if record["failed"]:
        print(f"  FAILED         : {record['failed']}")


def _print_results(record) -> None:
    print()
    print("=" * 100)
    print(f"PHASE F -- LOCKED TEST RESULTS "
          f"(n={record['test_dataset']['evaluated_samples']})")
    print("=" * 100)
    print(f"  {'System':<52}{'Acc':>9}{'MacroF1':>10}{'wF1':>9}{'MacroP':>9}{'MacroR':>9}")
    for key, block in record["systems"].items():
        print(f"  {block['system']:<52}{block['accuracy']:>9.4f}"
              f"{block['macro_f1']:>10.4f}{block['weighted_f1']:>9.4f}"
              f"{block['macro_precision']:>9.4f}{block['macro_recall']:>9.4f}")

    print()
    print(f"  {'Class':<10}{'n':>6}{'text':>9}{'audio':>9}{'fusion':>9}"
          f"{'random':>9}{'AHSEF':>9}{'dAHSEF':>10}")
    for klass, row in record["per_class_routing_analysis"]["per_class"].items():
        print(f"  {klass:<10}{row['support']:>6}{row['text_only_f1']:>9.4f}"
              f"{row['audio_only_f1']:>9.4f}{row['always_fusion_f1']:>9.4f}"
              f"{row['random_f1']:>9.4f}{row['ahsef_f1']:>9.4f}"
              f"{row['delta_ahsef_vs_text_only']:>+10.4f}")

    routing = record["routing"]["E_ahsef_at_budget"]
    print()
    print(f"  acquisitions        : {routing['acquisition_count']} / "
          f"{routing['eligible_samples']} ({routing['acquisition_rate']:.2%})")
    print(f"  budget respected    : {routing['budget_respected']}")
    print(f"  modalities/sample   : {routing['average_modalities_per_sample']:.3f}")
    print(f"  uncertainty acquired: "
          f"{routing['uncertainty']['acquired']['mean']:.4f} vs not-acquired "
          f"{routing['uncertainty']['not_acquired']['mean']:.4f}")

    quality = record["acquisition_quality"]
    print()
    print(f"  {'Policy':<16}{'acq':>6}{'fixes':>8}{'harms':>8}{'net':>7}"
          f"{'precision':>11}{'harm rate':>11}")
    for key in ("E_ahsef_at_budget", "D_random_at_budget", "C_always_fusion"):
        row = quality[key]
        print(f"  {row['policy']:<16}{row['acquisitions']:>6}"
              f"{row['text_wrong_audio_fixes']:>8}{row['text_correct_audio_breaks']:>8}"
              f"{row['net_corrections']:>7}"
              f"{row['correction_precision']:>11.4f}{row['harm_rate']:>11.4f}")

    print()
    for name, block in record["comparisons"].items():
        print(f"  {name:<26} dMacroF1={block['delta_macro_f1']:+.4f}  "
              f"dAcc={block['delta_accuracy']:+.4f}  "
              f"dwF1={block['delta_weighted_f1']:+.4f}")
    retained = record["retained_fusion_gain"]
    fraction = retained["retained_gain"]
    print(f"\n  fusion gain          : {retained['fusion_gain']:+.4f}")
    print(f"  AHSEF gain           : {retained['ahsef_gain']:+.4f}")
    print(f"  retained gain        : "
          f"{'n/a' if fraction is None else format(fraction, '.1%')}")

    transfer = record["uncertainty_transfer"]
    print(f"\n  uncertainty AUROC vs text error : "
          f"{transfer['auroc_identifying_incorrect_text']}")
    print(f"  spearman uncertainty vs error   : "
          f"{transfer['spearman_uncertainty_vs_error']}")

    stats = record["statistics"]
    print()
    for name in ("ahsef_vs_text_only", "ahsef_vs_random", "ahsef_vs_always_fusion"):
        block = stats[name]["macro_f1"]
        ci = block.get("ci95")
        interval = f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "(no CI)"
        print(f"  {name:<26} macro-F1 {block['point_estimate']:+.4f} {interval}")

    verdict = record["verdict"]
    print()
    print("=" * 100)
    print("PHASE F DECISION")
    print("=" * 100)
    for key in (
        "criterion_1_beats_matched_random",
        "criterion_2_retains_meaningful_fusion_gain",
        "criterion_3_substantially_fewer_modalities",
        "criterion_4_not_only_neutral",
        "criterion_5_beats_text_only",
    ):
        print(f"  {key:<48}{verdict[key]['met']}")
    print(f"\n  VERDICT : {verdict['verdict']}")
    audit = record["leakage_audit"]
    print(f"  policy modified after seeing test : "
          f"{audit['policy_modified_after_seeing_test_results']}")
    print(f"  only oracle uses test labels      : {audit['only_oracle_uses_test_labels']}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "lock":
        record = lock_stage(args)
        _print_locks(record)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        path = output / "phase_f_lock_checks.json"
        path.write_text(
            json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
        )
        print(f"\nWritten: {path}")
        return 0 if record["all_passed"] else 1

    if not args.i_am_opening_the_locked_test_split:
        raise SystemExit(
            "Refusing to evaluate without --i-am-opening-the-locked-test-split. "
            "The locked split is opened once, deliberately, after every "
            "validation-side decision is frozen."
        )
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
