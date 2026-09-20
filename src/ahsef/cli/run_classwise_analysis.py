"""PHASE A -- class-wise modality analysis over the frozen Stage 1 exports.

    python -m src.ahsef.cli.run_classwise_analysis --split validation
    python -m src.ahsef.cli.run_classwise_analysis --split test

Reads only.  It opens the prediction exports Stage 1 already wrote, plus the
Stage 3 LLM export where one exists, and writes a JSON record and a Markdown
report under ``experiments/ahsef/analysis/classwise/``.  No model is loaded, no
checkpoint is touched, and nothing under ``experiments/<modality>/`` or any
locked stage directory is written.

``--split test`` reads test *labels*.  That is legitimate here and only here:
this is a descriptive analysis of frozen baselines whose test split was already
opened and reported in Layer 3, and nothing it produces feeds a threshold, a
fitted estimator, or a routing decision.  The record stamps
``feeds_model_selection: false`` so the distinction is auditable rather than
assumed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from src.ahsef.analysis import ANALYSIS_EXPERIMENT_ID
from src.ahsef.analysis.classwise import (
    ALIGNED_PAIRS,
    EMOTION_MODALITIES,
    aligned_comparison,
    complementarity_summary,
    modality_profiles,
    specialization_verdicts,
)
from src.ahsef.identity import AlignmentIndex
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.registry import DEFAULT_BASELINES
from src.common.labels import CANONICAL_EMOTION_CLASSES

DEFAULT_OUTPUT = Path("experiments") / "ahsef" / "analysis" / "classwise"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--stage1-run", default="stage1")
    parser.add_argument("--stage3-run", default="stage3_text_audio")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--include-llm", action="store_true", default=True,
        help="Include the Stage 3 Gemma export in the aligned audio+text comparison.",
    )
    return parser


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_baseline_sets(layout: AhsefLayout, split: str) -> dict[str, PredictionSet]:
    """Every frozen baseline export that exists for this split."""
    sets: dict[str, PredictionSet] = {}
    for modality in DEFAULT_BASELINES:
        path = layout.prediction_path(modality, split)
        if not path.exists():
            continue
        sets[modality] = PredictionSet.load(
            path, layout.prediction_meta_path(modality, split)
        )
    if not sets:
        raise SystemExit(
            f"No Stage 1 prediction exports found for {split!r}. Run "
            f"`python -m src.ahsef.cli.export_predictions --run stage1` first."
        )
    return sets


def load_llm_set(root: str, run: str, split: str) -> PredictionSet | None:
    """The Stage 3 Gemma export, restricted later to whatever pool is in play."""
    layout = AhsefLayout(run=run, root=Path(root) / "ahsef")
    path = layout.prediction_path(LLM_MODALITY, split)
    if not path.exists():
        return None
    return PredictionSet.load(path, layout.prediction_meta_path(LLM_MODALITY, split))


def aligned_pools(root: str, split: str) -> dict[tuple[str, str], list[str]]:
    """The vetted co-split pools for the declared pairs, or nothing for a pair."""
    index = AlignmentIndex.from_experiments(
        [
            (name, reference.experiment, reference.label_column,
             "wesad_state_3class" if name == "physiology" else "emotion_7class")
            for name, reference in DEFAULT_BASELINES.items()
        ],
        root=root,
    )
    pools: dict[tuple[str, str], list[str]] = {}
    for pair in ALIGNED_PAIRS:
        try:
            pools[pair] = index.fusion_pool(list(pair), split, minimum=1)
        except Exception:
            pools[pair] = []
    return pools


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split = args.split
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    stage1 = AhsefLayout(run=args.stage1_run, root=Path(args.root) / "ahsef")
    sets = load_baseline_sets(stage1, split)
    print(f"[load] {split}: {sorted(sets)} baseline exports")

    profiles = modality_profiles(sets)
    for name, record in profiles["profiles"].items():
        print(f"       {name:11s} n={record['samples']:6d} acc={record['accuracy']:.4f} "
              f"macroF1={record['macro_f1']:.4f} task={record['task']}")

    # ------------------------------------------------ aligned head-to-head
    pools = aligned_pools(args.root, split)
    llm = load_llm_set(args.root, args.stage3_run, split) if args.include_llm else None
    comparisons: dict[str, dict] = {}
    for pair, pool in pools.items():
        left, right = pair
        if not pool:
            print(f"[pair] {left}+{right}: no co-split {split} samples -- skipped")
            continue
        if left not in sets or right not in sets:
            print(f"[pair] {left}+{right}: an export is missing -- skipped")
            continue
        comparisons[f"{left}+{right}"] = aligned_comparison(
            sets[left], sets[right], pool, left, right
        )
        print(f"[pair] {left}+{right}: {len(pool)} aligned {split} samples")

    # The Gemma expert only exists over the Stage 3 aligned pool, so it enters
    # exactly one comparison and only over the ids it actually scored.
    if llm is not None and ("audio", "text") in pools and pools[("audio", "text")]:
        scored = llm.frame.loc[llm.frame["llm_usable"].astype(bool), "sample_id"]
        shared = [
            identifier for identifier in pools[("audio", "text")]
            if identifier in set(scored.astype(str))
        ]
        if shared:
            comparisons["audio+text_llm"] = aligned_comparison(
                sets["audio"], llm, shared, "audio", "text_llm"
            )
            comparisons["text+text_llm"] = aligned_comparison(
                sets["text"], llm, shared, "text", "text_llm"
            )
            print(f"[pair] audio+text_llm and text+text_llm: {len(shared)} samples")

    specialization = specialization_verdicts(comparisons)
    complementarity = complementarity_summary(comparisons)

    record = {
        "phase": "A -- class-wise modality analysis",
        "experiment_id": ANALYSIS_EXPERIMENT_ID,
        "split": split,
        "class_order": list(CANONICAL_EMOTION_CLASSES),
        "emotion_modalities": list(EMOTION_MODALITIES),
        "modality_profiles": profiles,
        "aligned_comparisons": comparisons,
        "specialization": specialization,
        "complementarity": complementarity,
        "integrity": {
            "reads_only": True,
            "models_loaded": 0,
            "checkpoints_touched": 0,
            "feeds_model_selection": False,
            "feeds_threshold_selection": False,
            "feeds_hsig_training": False,
            "note": (
                "A descriptive analysis of already-frozen exports. Reading the test "
                "split here does not open it for tuning: nothing this command "
                "produces is consumed by a fitted estimator, a threshold, or a "
                "routing decision."
            ),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "environment": {"platform": platform.platform(), "python": sys.version},
    }
    json_path = output / f"classwise_{split}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    report_path = output / f"classwise_{split}.md"
    report_path.write_text(render_report(record), encoding="utf-8")

    _print_summary(record)
    print(f"\nWritten: {json_path}")
    print(f"Written: {report_path}")
    return 0


# ============================================================
# Rendering
# ============================================================

def _table(header, rows) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _f(value, digits=4):
    return "--" if value is None else f"{float(value):.{digits}f}"


def render_report(record: dict) -> str:
    split = record["split"]
    lines = [
        f"# Phase A — class-wise modality analysis ({split})",
        "",
        f"*Generated by `python -m src.ahsef.cli.run_classwise_analysis --split {split}`"
        f" from the frozen Stage 1 exports. Read-only: no model was loaded and no "
        f"checkpoint was touched.*",
        "",
        "## 1. Per-modality class-wise profiles",
        "",
        "> **These columns are not comparable across rows.** "
        + record["modality_profiles"]["why_not"],
        "",
    ]
    profiles = record["modality_profiles"]["profiles"]
    emotion = [n for n, r in profiles.items() if r["task"] == "emotion_7class"]
    other = [n for n, r in profiles.items() if r["task"] != "emotion_7class"]

    lines += _table(
        ["Modality", "n", "Corpora", "Accuracy", "Macro-F1", "Weighted-F1", "Bal. acc."],
        [
            [name, profiles[name]["samples"],
             ", ".join(profiles[name]["datasets"]) or "--",
             _f(profiles[name]["accuracy"]), _f(profiles[name]["macro_f1"]),
             _f(profiles[name]["weighted_f1"]), _f(profiles[name]["balanced_accuracy"])]
            for name in emotion
        ],
    )
    lines += ["", "Per-class F1 (each on its own split — see the warning above):", ""]
    lines += _table(
        ["Modality"] + list(CANONICAL_EMOTION_CLASSES),
        [
            [name] + [_f(profiles[name]["per_class"][c]["f1"], 3)
                      for c in CANONICAL_EMOTION_CLASSES]
            for name in emotion
        ],
    )
    lines += ["", "Per-class support (how many samples each figure rests on):", ""]
    lines += _table(
        ["Modality"] + list(CANONICAL_EMOTION_CLASSES),
        [
            [name] + [str(profiles[name]["per_class"][c]["support"])
                      for c in CANONICAL_EMOTION_CLASSES]
            for name in emotion
        ],
    )
    if other:
        lines += [
            "", "### Separate label space", "",
            "The following modality solves a different task and shares no class with "
            "the table above. It is reported apart rather than forced into the same "
            "row space.", "",
        ]
        for name in other:
            block = profiles[name]
            lines += [
                f"**{name}** — task `{block['task']}`, classes "
                + ", ".join(f"`{c}`" for c in block["class_order"])
                + f", n = {block['samples']}, accuracy {_f(block['accuracy'])}, "
                  f"macro-F1 {_f(block['macro_f1'])}.",
                "",
            ]

    # ------------------------------------------------------------- aligned
    lines += [
        "## 2. Head-to-head on genuinely aligned samples",
        "",
        "This is the only section in which a class-wise comparison between two "
        "modalities has a referent: the samples, the recordings and the labels are "
        "the same on both sides.",
        "",
    ]
    comparisons = record["aligned_comparisons"]
    if not comparisons:
        lines += ["*No aligned pool exists for this split.*", ""]
    for pair, block in comparisons.items():
        overall = block["overall"]
        lines += [
            f"### {pair} — {block['samples']} aligned {block['split']} samples", "",
        ]
        lines += _table(
            ["System", "Accuracy", "Macro-F1", "Weighted-F1"],
            [
                [block["left"], _f(overall["left"]["accuracy"]),
                 _f(overall["left"]["macro_f1"]), _f(overall["left"]["weighted_f1"])],
                [block["right"], _f(overall["right"]["accuracy"]),
                 _f(overall["right"]["macro_f1"]), _f(overall["right"]["weighted_f1"])],
            ],
        )
        outcome = overall["outcome"]
        lines += ["", "Outcome split:", ""]
        lines += _table(
            ["Outcome", "Count", "Rate"],
            [
                ["both correct", outcome["both_correct"], _f(outcome["both_correct_rate"], 3)],
                [f"{block['left']} correct, {block['right']} wrong",
                 outcome["left_correct_right_wrong"], _f(outcome["left_only_rate"], 3)],
                [f"{block['right']} correct, {block['left']} wrong",
                 outcome["right_correct_left_wrong"], _f(outcome["right_only_rate"], 3)],
                ["both wrong", outcome["both_wrong"], _f(outcome["both_wrong_rate"], 3)],
            ],
        )
        lines += [
            "",
            f"Disagreement rate **{_f(overall['disagreement_rate'], 3)}**. "
            f"An oracle always trusting the correct one of the two would reach "
            f"accuracy **{_f(overall['ceiling_if_either_were_trusted'])}** — the "
            f"ceiling for any router over this pair, not an achievable number.",
            "",
            "Per class:", "",
        ]
        lines += _table(
            ["Class", "Support", f"{block['left']} F1", f"{block['right']} F1", "Δ F1",
             f"fixes for {block['left']}", f"fixes for {block['right']}", "Stronger"],
            [
                [name, entry["support"], _f(entry["left_f1"], 3), _f(entry["right_f1"], 3),
                 f"{entry['f1_difference']:+.3f}",
                 entry["correction_opportunity_for_left"],
                 entry["correction_opportunity_for_right"],
                 entry["stronger"]["winner"] or "—"]
                for name, entry in block["per_class"].items()
            ],
        )
        lines += ["", f"*{block['reading']}*", ""]

    # ------------------------------------------------------ specialization
    special = record["specialization"]
    lines += [
        "## 3. Modality specialization — what the aligned evidence supports", "",
        f"*{special['evidence_rule']}*", "",
    ]
    lines += _table(
        ["Emotion", "Verdict", "Basis"],
        [
            [name, entry["verdict"] or "**undecided**", entry["reason"]]
            for name, entry in special["per_class"].items()
        ],
    )
    lines += [
        "",
        f"Classes with a verdict: "
        + (", ".join(f"`{c}`" for c in special["classes_with_a_verdict"]) or "none")
        + ".",
        "",
        f"Classes without one: "
        + (", ".join(f"`{c}`" for c in special["classes_without_a_verdict"]) or "none")
        + ".",
        "",
        f"> **Coverage caveat.** {special['coverage_caveat']}",
        "",
    ]

    # ----------------------------------------------------- complementarity
    lines += ["## 4. How much room is there for routing?", ""]
    rows = record["complementarity"]["pairs"]
    if rows:
        lines += _table(
            ["Pair", "n", "Best single", "Oracle (either)", "Headroom",
             "Disagreement", "Both wrong"],
            [
                [pair, entry["samples"], _f(entry["best_single_accuracy"]),
                 _f(entry["oracle_either_accuracy"]),
                 f"{entry['headroom_over_best_single']:+.4f}",
                 _f(entry["disagreement_rate"], 3), _f(entry["both_wrong_rate"], 3)]
                for pair, entry in rows.items()
            ],
        )
        lines += ["", f"*{record['complementarity']['reading']}*", ""]
    else:
        lines += ["*No aligned pair available.*", ""]

    lines += [
        "## 5. Integrity", "",
    ]
    lines += _table(
        ["Property", "Value"],
        [[key, value] for key, value in record["integrity"].items() if key != "note"],
    )
    lines += ["", record["integrity"]["note"], ""]
    return "\n".join(lines).rstrip() + "\n"


def _print_summary(record: dict) -> None:
    print()
    print("=" * 96)
    print(f"PHASE A -- class-wise modality analysis ({record['split']})")
    print("=" * 96)
    special = record["specialization"]
    print(f"  {'Emotion':<10}{'Verdict':<14}Basis")
    for name, entry in special["per_class"].items():
        print(f"  {name:<10}{(entry['verdict'] or 'undecided'):<14}{entry['reason'][:60]}")
    print()
    for pair, entry in record["complementarity"]["pairs"].items():
        print(f"  {pair:<18} n={entry['samples']:<5} best_single="
              f"{entry['best_single_accuracy']:.4f} oracle="
              f"{entry['oracle_either_accuracy']:.4f} headroom="
              f"{entry['headroom_over_best_single']:+.4f} "
              f"disagree={entry['disagreement_rate']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
