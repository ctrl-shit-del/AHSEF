"""AHSEF Stage 3: dynamic Text -> Audio acquisition and fusion.

Run in order.  Each stage refuses to run before the one it depends on, and
``test`` refuses to run at all until every validation-side decision is frozen::

    python -m src.ahsef.cli.run_stage3_text_audio --stage align
    python -m src.ahsef.cli.run_stage3_text_audio --stage llm      --split validation
    python -m src.ahsef.cli.run_stage3_text_audio --stage llm      --split test
    python -m src.ahsef.cli.run_stage3_text_audio --stage oracle   --split validation
    python -m src.ahsef.cli.run_stage3_text_audio --stage hsig
    python -m src.ahsef.cli.run_stage3_text_audio --stage freeze
    python -m src.ahsef.cli.run_stage3_text_audio --stage validate
    python -m src.ahsef.cli.run_stage3_text_audio --stage ablations
    python -m src.ahsef.cli.run_stage3_text_audio --stage oracle   --split test
    python -m src.ahsef.cli.run_stage3_text_audio --stage test
    python -m src.ahsef.cli.run_stage3_text_audio --stage report

``--stage llm --split test`` records the locked split's LLM transcript but
computes nothing from its labels; the transcript is data, and every decision
that consumes it is already fixed by ``freeze``.  ``--stage oracle --split test``
applies the *frozen* fusion spec and refuses to reselect one.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.ahsef.evaluation import evaluate_prediction_set
from src.ahsef.inference import PredictionSet
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.stage3 import STAGE3_EXPERIMENT_ID
from src.ahsef.stage3.layout import Stage3Layout
from src.ahsef.stage3.llm_pass import FROZEN_LLM_CONFIG, run_llm_pass
from src.ahsef.stage3.pool import (
    MINIMUM_POOL,
    AlignedPool,
    alignment_report,
    assert_pools_disjoint,
    build_aligned_pool,
)

STAGES = (
    "align", "llm", "oracle", "hsig", "freeze", "validate", "ablations", "test", "report",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=list(STAGES))
    parser.add_argument("--run", default="stage3_text_audio")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--minimum-pool", type=int, default=MINIMUM_POOL)

    # --- LLM (inherited from the Stage 2 freeze; overridable only to replay) ---
    parser.add_argument("--model", default=FROZEN_LLM_CONFIG["model"])
    parser.add_argument("--host", default=FROZEN_LLM_CONFIG["host"])
    parser.add_argument("--prompt-version", default=FROZEN_LLM_CONFIG["prompt_version"])
    parser.add_argument("--temperature", type=float, default=FROZEN_LLM_CONFIG["temperature"])
    parser.add_argument("--llm-seed", type=int, default=FROZEN_LLM_CONFIG["llm_seed"])
    parser.add_argument("--num-predict", type=int, default=FROZEN_LLM_CONFIG["num_predict"])
    parser.add_argument(
        "--provider", default="ollama", choices=["ollama", "replay"],
        help="'replay' re-serves the recorded transcript; no network, no spend.",
    )
    parser.add_argument("--store-text", default="hash", choices=["none", "hash", "raw"])
    parser.add_argument("--max-calls", type=int, default=None)

    # --- HSIG / routing policy (validation-only; frozen before test) ---
    parser.add_argument("--uncertainty-policy", default=FROZEN_LLM_CONFIG["uncertainty_policy"])
    parser.add_argument("--fusion-method", default="weighted_probability",
                        choices=["weighted_probability", "log_opinion_pool"])
    parser.add_argument("--hsig-folds", type=int, default=5)
    parser.add_argument("--hsig-seed", type=int, default=42)
    parser.add_argument("--lambda-cost", type=float, default=0.10)
    parser.add_argument("--mu-latency", type=float, default=0.10)
    parser.add_argument("--alpha-acquisition", type=float, default=0.05)
    parser.add_argument("--beta-latency", type=float, default=0.02)
    parser.add_argument("--gate-objective", default="macro_f1_acquisition_latency")
    parser.add_argument("--seed", type=int, default=42)
    return parser


# ============================================================
# Shared helpers
# ============================================================

def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_record() -> dict:
    import sklearn
    import torch

    return {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
    }


def metric_block(prediction_set: PredictionSet) -> dict:
    metrics = evaluate_prediction_set(prediction_set)
    return {
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": metrics["class_order"],
    }


def load_pool(layout: Stage3Layout, split: str) -> AlignedPool:
    path = layout.pool_path(split)
    if not path.exists():
        raise SystemExit(
            f"No aligned pool at {path}. Run --stage align first; Stage 3 will not "
            f"fuse samples whose alignment has not been verified."
        )
    return AlignedPool.load(path)


def load_predictions(layout: Stage3Layout, modality: str, split: str) -> PredictionSet:
    path = layout.prediction_path(modality, split)
    if not path.exists():
        raise SystemExit(
            f"No {modality} predictions for {split} at {path}. Run the stage that "
            f"produces them before this one."
        )
    return PredictionSet.load(path, layout.prediction_meta_path(modality, split))


# ============================================================
# Stage: align  (PHASE A)
# ============================================================

def stage_align(layout: Stage3Layout, args) -> int:
    pools = {}
    for split in ("validation", "test"):
        pool = build_aligned_pool(
            split, root=args.root, dataset_root=args.dataset_root,
            minimum=args.minimum_pool,
        )
        pool.save(layout.pool_path(split))
        pools[split] = pool
        print(f"[pool] {split:<11} n={pool.size:<5} fp={pool.fingerprint[:16]} "
              f"datasets={pool.datasets}")
        print(f"       classes: {pool.class_counts}")
        for name, record in pool.availability.items():
            if record.get("checked") is False:
                continue
            print(f"       {name:<6} available={record['available']}/{record['rows']} "
                  f"missing_assets={record['declared_but_asset_missing']}")

    assert_pools_disjoint(pools["validation"], pools["test"])
    report = alignment_report(pools, minimum=args.minimum_pool)
    report["experiment_id"] = STAGE3_EXPERIMENT_ID
    report["git_commit"] = git_commit()
    report["environment"] = environment_record()
    layout.alignment_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[ok  ] validation and test pools are disjoint and contamination-free")
    print(f"\nWritten: {layout.alignment_report_path}")
    print("Next: --stage llm --split validation")
    return 0


# ============================================================
# Stage: llm  (PHASE B, text side)
# ============================================================

def stage_llm(layout: Stage3Layout, args) -> int:
    pool = load_pool(layout, args.split)
    config = {
        "model": args.model, "host": args.host,
        "prompt_version": args.prompt_version, "temperature": args.temperature,
        "llm_seed": args.llm_seed, "num_predict": args.num_predict,
        "uncertainty_policy": args.uncertainty_policy,
    }
    drift = {
        key: (FROZEN_LLM_CONFIG[key], value)
        for key, value in config.items()
        if key in FROZEN_LLM_CONFIG and FROZEN_LLM_CONFIG[key] != value
    }
    if drift:
        raise SystemExit(
            f"The requested LLM configuration differs from the Stage 2 frozen one: "
            f"{drift}. Stage 3 inherits Stage 2's text evidence source unchanged; "
            f"altering it would make the two stages incomparable."
        )

    predictions, timing = run_llm_pass(
        pool, layout.transcript_path(args.split), layout.timing_path(args.split),
        config=config, provider_name=args.provider, store_text=args.store_text,
        max_calls=args.max_calls, root=args.root,
    )
    predictions.save(
        layout.prediction_path(LLM_MODALITY, args.split),
        layout.prediction_meta_path(LLM_MODALITY, args.split),
    )

    usable = predictions.frame["llm_usable"]
    scored = predictions.restricted_to(
        predictions.frame.loc[usable, "sample_id"].tolist()
    )
    block = metric_block(scored)
    print(f"[llm ] {args.split}: usable {int(usable.sum())}/{len(predictions.frame)} "
          f"| acc={block['accuracy']:.4f} macroF1={block['macro_f1']:.4f} "
          f"weightedF1={block['weighted_f1']:.4f}")
    print(f"       generation mean={timing['generation_latency_ms']['mean']:.0f}ms "
          f"| wall clock per call="
          f"{timing['wall_clock']['mean_ms_per_live_call']}")
    print(f"\nWritten: {layout.prediction_path(LLM_MODALITY, args.split)}")
    return 0


# ============================================================
# Dispatch
# ============================================================

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = Stage3Layout(run=args.run, root=Path(args.root) / "ahsef")
    layout.prepare()

    from src.ahsef.cli import stage3_stages as later

    handlers = {
        "align": stage_align,
        "llm": stage_llm,
        "oracle": later.stage_oracle,
        "hsig": later.stage_hsig,
        "freeze": later.stage_freeze,
        "validate": later.stage_validate,
        "ablations": later.stage_ablations,
        "test": later.stage_test,
        "report": later.stage_report,
    }
    return handlers[args.stage](layout, args)


if __name__ == "__main__":
    raise SystemExit(main())
