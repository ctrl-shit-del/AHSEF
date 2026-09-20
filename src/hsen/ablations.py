"""The controlled comparisons of sections 12 and 13.

    # section 12 -- every modality combination
    python -m src.hsen.ablations --study modality --profile cpu25

    # section 13 -- concat vs self-attention vs Husformer
    python -m src.hsen.ablations --study fusion --profile cpu25

    # collect finished runs into one table without training anything
    python -m src.hsen.ablations --study modality --report-only

What makes these ablations rather than a pile of runs: every arm shares the
seed, the manifests, the cached features, the loss, the schedule and the
evaluation code, and differs in exactly one declared axis.  The runner varies
that axis and nothing else, which is why it exists instead of a shell loop
whose arms drift apart over an afternoon.

Section 13 also asks for parameter count, training time and memory alongside the
scores, because Husformer's claim is not "more accurate" but "more accurate per
parameter" -- 78.68 against HusPair's 73.57 at 0.71 M against 3.90 M.  A fusion
table without the cost column cannot test that claim at all.

Both studies run on the CPU 25% profile by default.  Section 14 is explicit that
the fast profile is for choosing candidates and the full CUDA profile is for the
numbers that get reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.hsen.cli import MODALITY_SETS
from src.hsen.experiment import ExperimentError, run_experiment
from src.hsen.models.fusion_variants import FUSION_VARIANTS
from src.hsen.training.hsen_trainer import TrainerConfig
from src.hsen.training.losses import LossConfig

#: Section 12's list, in the order the brief gives it.  Single modalities first,
#: then pairs, then the full model -- so a table read top to bottom shows what
#: each addition bought.
MODALITY_ARMS: tuple[str, ...] = (
    "text", "audio", "video",
    "audio+text", "audio+video", "video+text",
    "audio+video+text",
)

PROFILES = {
    "cpu25": {"device": "cpu", "amp": False, "data_fraction": 0.25,
              "batch_size": 8, "num_workers": 0, "epochs": 15},
    "full_cuda": {"device": "cuda", "amp": True, "data_fraction": 1.0,
                  "batch_size": 32, "num_workers": 4, "epochs": 30},
}


def arm_name(study: str, arm: str, profile: str) -> str:
    return f"ablation_{study}_{arm.replace('+', '_')}_{profile}"


def run_arm(
    experiment: str, study: str, arm: str, profile: str, seed: int,
    output_root: Path, loss_config: LossConfig, evaluate_test: bool,
    epochs: int | None = None, resume: bool = False, force_lock: bool = False,
    modalities: str = "all",
) -> dict:
    """Train one arm and return its row of the table."""
    settings = dict(PROFILES[profile])
    if epochs:
        settings["epochs"] = epochs

    # The modality study varies the modality set; the fusion study holds it
    # fixed and varies the trunk. Which set it holds fixed has to be a choice,
    # not a constant: defaulting to all three means the whole fusion study
    # skips every arm on a corpus whose video is not cached.
    modality_set = MODALITY_SETS[arm] if study == "modality" else MODALITY_SETS[modalities]
    fusion = arm if study == "fusion" else "husformer"

    config = TrainerConfig(
        experiment=experiment,
        profile=arm_name(study, arm, profile),
        output_root=output_root,
        seed=seed,
        evaluate_test=evaluate_test,
        resume=resume,
        force_lock=force_lock,
        **settings,
    )
    started = time.time()
    summary = run_experiment(
        trainer_config=config,
        modalities=modality_set,
        fusion=fusion,
        loss_config=loss_config,
    )
    return row_from_summary(summary, study, arm, time.time() - started)


def row_from_summary(summary: dict, study: str, arm: str, wall_clock: float) -> dict:
    """Flatten one run into a comparable row."""
    primary = summary["primary_metric"]
    validation = summary.get("validation", {})
    test = summary.get("test", {})
    counts = summary["parameter_counts"]
    row = {
        "study": study,
        "arm": arm,
        "modalities": "+".join(summary["model_config"]["modalities"]),
        "fusion": summary["model_config"]["fusion"],
        "best_epoch": summary["best_epoch"],
        f"val_{primary}": summary.get(f"best_val_{primary}"),
        "val_accuracy": validation.get("accuracy"),
        "val_weighted_f1": validation.get("weighted_f1"),
        "val_macro_f1": validation.get("macro_f1"),
        "test_accuracy": test.get("accuracy"),
        "test_weighted_f1": test.get("weighted_f1"),
        "test_macro_f1": test.get("macro_f1"),
        # Section 13's cost columns. Fusion parameters are separated from the
        # total because the projections and heads are identical across fusion
        # arms, so the total understates the difference between the trunks.
        "parameters_total": counts["total"],
        "parameters_fusion": counts["fusion"],
        "training_seconds": summary["training_seconds"],
        "wall_clock_seconds": round(wall_clock, 1),
        "epochs_run": summary["epochs_run"],
        "peak_gpu_mb": summary.get("run_record", {}).get("hardware", {}).get(
            "gpu_total_memory_gb"),
    }
    for affect in ("valence", "arousal"):
        row[f"val_{affect}_ccc"] = validation.get(f"{affect}_ccc")
    return row


def arm_is_complete(output_root: Path, experiment: str, study: str,
                    arm: str, profile: str) -> bool:
    """Whether an arm already has a finished run.

    A study is long -- seven arms at an hour each -- and a machine that
    reclaims the process partway through should not cost the arms that already
    finished.  ``run_summary.json`` is written only at the end of ``run()``, so
    its presence is the honest completion marker; a directory holding only
    checkpoints is a partial run and gets resumed instead.
    """
    return (Path(output_root) / experiment / arm_name(study, arm, profile)
            / "run_summary.json").exists()


def collect_runs(output_root: Path, study: str, profile: str) -> pd.DataFrame:
    """Read finished arms off disk, so a report needs no retraining."""
    rows = []
    for summary_path in sorted(Path(output_root).glob("*/*/run_summary.json")):
        profile_name = summary_path.parent.name
        prefix = f"ablation_{study}_"
        if not profile_name.startswith(prefix) or not profile_name.endswith(profile):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Modality arms are written with '+' flattened to '_' for the directory
        # name, so they need translating back. Fusion arm names contain literal
        # underscores -- 'self_attention' -- and translating those produces
        # 'self+attention', which matches no known arm and silently drops the
        # row. That row is the one that isolates the cross-modal mechanism, so
        # losing it costs the whole point of the study.
        label = profile_name[len(prefix):-(len(profile) + 1)]
        arm = label.replace("_", "+") if study == "modality" else label
        rows.append(row_from_summary(summary, study, arm, summary["training_seconds"]))
    return pd.DataFrame(rows)


def render_table(frame: pd.DataFrame, study: str) -> str:
    """The comparison as readable text."""
    if frame.empty:
        return f"No {study} arms found."

    order = MODALITY_ARMS if study == "modality" else FUSION_VARIANTS
    frame = frame.set_index("arm").reindex([a for a in order if a in set(frame["arm"])])
    frame = frame.reset_index()

    lines = [
        "=" * 96,
        f"ABLATION -- {study}",
        "=" * 96,
        f"{'arm':<20}{'val WF1':>9}{'val mF1':>9}{'test WF1':>10}{'test mF1':>10}"
        f"{'params':>12}{'fusion':>11}{'train s':>10}",
        "-" * 96,
    ]
    for _, row in frame.iterrows():
        def number(value, width, places=4):
            return f"{value:>{width}.{places}f}" if pd.notna(value) else f"{'--':>{width}}"

        lines.append(
            f"{row['arm']:<20}"
            + number(row["val_weighted_f1"], 9)
            + number(row["val_macro_f1"], 9)
            + number(row["test_weighted_f1"], 10)
            + number(row["test_macro_f1"], 10)
            + f"{int(row['parameters_total']):>12,}"
            + f"{int(row['parameters_fusion']):>11,}"
            + f"{row['training_seconds']:>10.0f}"
        )
    lines.append("-" * 96)

    if study == "modality":
        # The widest arm that actually ran, not specifically audio+video+text:
        # a study missing a modality's cache still has a multimodal-vs-unimodal
        # margin worth stating, and hard-coding the tri-modal name would print
        # nothing at all for it.
        scored = frame[frame["val_weighted_f1"].notna()]
        unimodal = scored[scored["arm"].isin(("text", "audio", "video"))]
        multimodal = scored[~scored["arm"].isin(("text", "audio", "video"))]
        if not unimodal.empty and not multimodal.empty:
            best_single = unimodal.loc[unimodal["val_weighted_f1"].idxmax()]
            widest = multimodal.loc[
                multimodal["arm"].str.count(r"\+").idxmax()
                if multimodal["arm"].str.count(r"\+").nunique() > 1
                else multimodal["val_weighted_f1"].idxmax()
            ]
            margin = widest["val_weighted_f1"] - best_single["val_weighted_f1"]
            lines.append(
                f"  {widest['arm']} over the best single modality "
                f"({best_single['arm']}, {best_single['val_weighted_f1']:.4f}): "
                f"{margin:+.4f} weighted F1."
            )
            lines.append(
                "  This margin is the budget the later routing work is allowed to "
                "spend: a modality UGAPR declines to acquire must cost less than it."
            )
        if not unimodal.empty and len(unimodal) > 1:
            ordering = " > ".join(
                f"{row['arm']} {row['val_weighted_f1']:.4f}"
                for _, row in unimodal.sort_values("val_weighted_f1", ascending=False).iterrows()
            )
            lines.append(f"  Unimodal ordering: {ordering}")
    if study == "fusion" and {"husformer", "concat"} <= set(frame["arm"]):
        husformer = frame[frame["arm"] == "husformer"].iloc[0]
        for other in ("self_attention", "concat"):
            if other not in set(frame["arm"]):
                continue
            row = frame[frame["arm"] == other].iloc[0]
            if pd.isna(husformer["val_weighted_f1"]) or pd.isna(row["val_weighted_f1"]):
                continue
            ratio = row["parameters_fusion"] / max(1, husformer["parameters_fusion"])
            lines.append(
                f"  Husformer vs {other}: "
                f"{husformer['val_weighted_f1'] - row['val_weighted_f1']:+.4f} weighted F1 "
                f"at {ratio:.2f}x the fusion parameters."
            )
    lines.append("=" * 96)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, choices=["modality", "fusion"])
    parser.add_argument("--experiment", default="iemocap_erc6")
    parser.add_argument("--profile", default="cpu25", choices=sorted(PROFILES))
    parser.add_argument("--modalities", default="all", choices=sorted(MODALITY_SETS),
                        help="Modality set held fixed by the fusion study. Ignored "
                             "by the modality study, which varies it.")
    parser.add_argument("--arms", default=None,
                        help="Comma-separated subset of arms; defaults to all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-root", default="results/hsen")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Open the test split for every arm. Section 13 wants "
                             "test numbers, but running this before the design is "
                             "settled opens the locked split once per arm.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue partially-trained arms from last.pt and "
                             "skip arms that already have a run_summary.json.")
    parser.add_argument("--force-lock", action="store_true",
                        help="Break a stale run-directory lock on each arm.")
    parser.add_argument("--report-only", action="store_true",
                        help="Collect finished arms into a table without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root) / args.experiment
    default_arms = MODALITY_ARMS if args.study == "modality" else FUSION_VARIANTS
    arms = args.arms.split(",") if args.arms else list(default_arms)

    if not args.report_only:
        loss_config = LossConfig()
        skipped: list[tuple[str, str]] = []
        for index, arm in enumerate(arms, start=1):
            print(f"\n{'#' * 96}\n# {args.study} ablation {index}/{len(arms)}: {arm}\n{'#' * 96}")
            try:
                run_arm(
                    experiment=args.experiment, study=args.study, arm=arm,
                    profile=args.profile, seed=args.seed,
                    output_root=Path(args.output_root), loss_config=loss_config,
                    evaluate_test=args.evaluate_test, epochs=args.epochs,
                    resume=args.resume, force_lock=args.force_lock,
                    modalities=args.modalities,
                )
            except ExperimentError as error:
                # A modality with no cache is a missing extraction step, not a
                # reason to abandon the other six arms. It is recorded as
                # skipped and named again at the end, so an absent row in the
                # table is never mistaken for a row that scored badly.
                skipped.append((arm, str(error).splitlines()[0]))
                print(f"\n[skip] {arm}: {skipped[-1][1]}")

        if skipped:
            print(f"\n{len(skipped)} arm(s) skipped for missing features:")
            for arm, reason in skipped:
                print(f"  {arm:<20} {reason}")

    frame = collect_runs(Path(args.output_root), args.study, args.profile)
    print("\n" + render_table(frame, args.study))
    if not frame.empty:
        destination = output_root / f"ablation_{args.study}_{args.profile}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        print(f"\nTable written to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
