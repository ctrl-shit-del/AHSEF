"""Evaluate a trained HSEN checkpoint and draw the section-19 figures.

    # plots for a finished run
    python -m src.hsen.evaluation --run results/hsen/iemocap_erc6/full_cuda

    # score a checkpoint on a split it has not seen
    python -m src.hsen.evaluation --run results/hsen/iemocap_erc6/full_cuda \\
        --split test --evaluate

Separate from the trainer on purpose.  The trainer's job ends when the best
checkpoint and its history are on disk; re-reading those artefacts to produce
figures and a test score is a different job, and keeping it separate means a
plot can be redrawn or a checkpoint re-scored without re-running training.

``--evaluate --split test`` is the one operation here that opens the locked
partition, and it is never implied: plotting reads the history file and touches
no split at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Agg, because these run headless on a training box and a default backend that
# wants a display turns a finished run into a crash at the plotting step.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_DPI = 150


class EvaluationError(RuntimeError):
    """Raised when a run directory cannot be evaluated."""


def load_run(run_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Read a finished run's summary and per-epoch history."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise EvaluationError(
            f"No run_summary.json in {run_dir}. Train first, or point --run at a "
            f"directory produced by scripts/train_hsen_*.py."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history_path = run_dir / "metrics" / "history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    return summary, history


# ======================================================================
# Figures
# ======================================================================

def plot_losses(history: pd.DataFrame, path: Path, best_epoch: int | None = None) -> Path:
    """Training against validation loss -- the overfitting picture."""
    figure, axes = plt.subplots(figsize=(7, 4.5))
    if "train_total" in history:
        axes.plot(history["epoch"], history["train_total"], marker="o",
                  label="train", linewidth=1.8)
    if "val_loss" in history:
        axes.plot(history["epoch"], history["val_loss"], marker="s",
                  label="validation", linewidth=1.8)
    if best_epoch:
        axes.axvline(best_epoch, color="grey", linestyle="--", linewidth=1,
                     label=f"best epoch ({best_epoch})")
    axes.set_xlabel("epoch")
    axes.set_ylabel("loss")
    axes.set_title("Training vs validation loss")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def plot_validation_scores(history: pd.DataFrame, path: Path,
                           best_epoch: int | None = None) -> Path:
    """Weighted and macro F1 on one axis.

    Deliberately on the same axis: the gap between them *is* the finding. A
    weighted F1 that climbs while macro F1 stays flat is a model getting better
    at the classes that were already easy, which is the thing section 11 asks
    not to let an aggregate hide.
    """
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for column, label, marker in (
        ("val_weighted_f1", "weighted F1", "o"),
        ("val_macro_f1", "macro F1", "s"),
        ("val_micro_f1", "micro F1", "^"),
        ("val_accuracy", "accuracy", "d"),
    ):
        if column in history and history[column].notna().any():
            axes.plot(history["epoch"], history[column], marker=marker,
                      label=label, linewidth=1.8)
    if best_epoch:
        axes.axvline(best_epoch, color="grey", linestyle="--", linewidth=1,
                     label=f"best epoch ({best_epoch})")
    axes.set_xlabel("epoch")
    axes.set_ylabel("score")
    axes.set_ylim(0, 1)
    axes.set_title("Validation categorical metrics")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def plot_regression(history: pd.DataFrame, path: Path) -> Path | None:
    """Valence and arousal CCC and MAE, if either head was supervised."""
    columns = [c for c in ("val_valence_ccc", "val_arousal_ccc",
                           "val_valence_mae", "val_arousal_mae") if c in history]
    if not columns or not history[columns].notna().any().any():
        return None

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for column, label in (("val_valence_ccc", "valence"), ("val_arousal_ccc", "arousal")):
        if column in history:
            left.plot(history["epoch"], history[column], marker="o", label=label)
    left.set_title("Validation CCC")
    left.set_xlabel("epoch")
    left.set_ylabel("CCC")
    left.grid(alpha=0.3)
    left.legend()

    for column, label in (("val_valence_mae", "valence"), ("val_arousal_mae", "arousal")):
        if column in history:
            right.plot(history["epoch"], history[column], marker="s", label=label)
    right.set_title("Validation MAE")
    right.set_xlabel("epoch")
    right.set_ylabel("MAE")
    right.grid(alpha=0.3)
    right.legend()

    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def plot_confusion(matrix, class_names, path: Path, title: str = "Confusion matrix") -> Path:
    """Row-normalised confusion, with raw counts written into the cells.

    Row-normalised because IEMOCAP's classes differ by 3x in size and a raw-count
    heatmap under that imbalance shows only which class is largest.  The counts
    stay in the cells so a bright cell backed by nine samples cannot be mistaken
    for a real result.
    """
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)

    size = max(5.0, 0.9 * len(class_names) + 2.5)
    figure, axes = plt.subplots(figsize=(size, size * 0.85))
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axes.set_yticks(range(len(class_names)), class_names)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(title)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes.text(
                column, row, f"{normalised[row, column]:.2f}\n({int(matrix[row, column])})",
                ha="center", va="center", fontsize=8,
                color="white" if normalised[row, column] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046, label="row-normalised")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def plot_per_class_f1(metrics: dict, path: Path, title: str = "Per-class F1",
                      class_names: tuple[str, ...] | None = None) -> Path | None:
    """Per-class F1 as bars, annotated with support.

    The figure section 11 is really asking for: an aggregate cannot hide a class
    here, because every class has its own bar and its own sample count.
    """
    per_class = metrics.get("per_class")
    if not per_class:
        return None
    # Ordered by the run's declared label space, never by the dict's own key
    # order -- which any JSON round-trip is free to have rearranged.
    names = [name for name in (class_names or ()) if name in per_class] or list(per_class)
    scores = [per_class[name]["f1"] for name in names]
    supports = [per_class[name]["support"] for name in names]

    figure, axes = plt.subplots(figsize=(max(6.0, 1.1 * len(names) + 2), 4.2))
    bars = axes.bar(names, scores, color="#4C72B0")
    for bar, score, support in zip(bars, scores, supports):
        axes.text(bar.get_x() + bar.get_width() / 2, score + 0.02,
                  f"{score:.3f}\nn={support}", ha="center", va="bottom", fontsize=8)
    for name, value in (("weighted F1", metrics.get("weighted_f1")),
                        ("macro F1", metrics.get("macro_f1"))):
        if value is not None and value == value:
            axes.axhline(value, linestyle="--", linewidth=1, alpha=0.7,
                         label=f"{name} {value:.3f}")
    axes.set_ylim(0, 1.15)
    axes.set_ylabel("F1")
    axes.set_title(title)
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
    plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path


def render_figures(run_dir: Path) -> list[Path]:
    """Draw every figure a run's artefacts support."""
    run_dir = Path(run_dir)
    summary, history = load_run(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    from src.common.labels import get_label_space

    # The class order comes from the run's own recorded label space, not from a
    # current default: a figure labelled with today's class order over an older
    # run's confusion matrix would be silently, confidently wrong.
    class_names = get_label_space(summary["model_config"]["label_space"]).classes

    written: list[Path] = []
    best_epoch = summary.get("best_epoch")
    if not history.empty:
        written.append(plot_losses(history, figures_dir / "loss.png", best_epoch))
        written.append(plot_validation_scores(
            history, figures_dir / "validation_scores.png", best_epoch))
        regression = plot_regression(history, figures_dir / "regression.png")
        if regression:
            written.append(regression)

    for split in ("validation", "test"):
        metrics = summary.get(split)
        if not metrics:
            continue
        if metrics.get("confusion_matrix"):
            written.append(plot_confusion(
                metrics["confusion_matrix"], class_names,
                figures_dir / f"confusion_{split}.png",
                title=f"{summary['experiment']} -- {split} confusion",
            ))
        per_class = plot_per_class_f1(
            metrics, figures_dir / f"per_class_f1_{split}.png",
            title=f"{summary['experiment']} -- {split} per-class F1",
            class_names=class_names,
        )
        if per_class:
            written.append(per_class)
    return written


# ======================================================================
# Re-scoring a checkpoint
# ======================================================================

def evaluate_checkpoint(run_dir: Path, split: str = "test",
                        checkpoint: str = "best.pt") -> dict:
    """Rebuild the run from its saved config and score one split.

    The model is rebuilt from the checkpoint's own recorded config rather than
    from current defaults, so re-scoring an old run does not silently evaluate a
    differently-shaped model that happens to load.
    """
    import torch

    from src.hsen.experiment import build_experiment
    from src.hsen.models.hsen import HSENConfig
    from src.hsen.training.hsen_trainer import HSENTrainer, TrainerConfig
    from src.hsen.training.losses import LossConfig

    run_dir = Path(run_dir)
    summary, _ = load_run(run_dir)
    payload = torch.load(run_dir / "checkpoints" / checkpoint,
                         map_location="cpu", weights_only=False)

    stored = dict(summary["model_config"])
    stored["modalities"] = tuple(stored["modalities"])
    model_config = HSENConfig(**stored)

    trainer_config = TrainerConfig(**{
        **summary["trainer_config"],
        "evaluate_test": split == "test",
        "resume": False,
    })
    bundle = build_experiment(
        experiment=trainer_config.experiment,
        modalities=model_config.modalities,
        data_fraction=1.0,           # scoring never subsamples
        seed=trainer_config.seed,
        fusion=model_config.fusion,
        include_test=split == "test",
    )
    trainer = HSENTrainer(
        config=trainer_config,
        model_config=model_config,
        loss_config=LossConfig(**summary["loss_config"]),
        datasets=bundle.datasets,
        manifests=bundle.manifests,
        class_names=bundle.class_names,
        audit=bundle.audit,
    )
    trainer.model.load_state_dict(payload["model_state_dict"])
    metrics = trainer.evaluate(split, prefix="")

    destination = run_dir / "metrics" / f"{split}_metrics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8",
    )
    return metrics


# ======================================================================
# CLI
# ======================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="A run directory.")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score the checkpoint on --split. Without this, only "
                             "figures are drawn and no split is opened.")
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run)

    if args.evaluate:
        if args.split == "test":
            print("Opening the TEST partition. This is the locked split; every "
                  "validation-side decision should already be frozen.\n")
        metrics = evaluate_checkpoint(run_dir, args.split, args.checkpoint)
        for name in ("accuracy", "weighted_f1", "macro_f1", "micro_f1"):
            if name in metrics and metrics[name] == metrics[name]:
                print(f"  {name:<16} {metrics[name]:.4f}")
        for affect in ("valence", "arousal"):
            if f"{affect}_ccc" in metrics:
                print(f"  {affect} CCC/MAE   {metrics[f'{affect}_ccc']:.4f} / "
                      f"{metrics[f'{affect}_mae']:.4f}")

    if args.figures:
        written = render_figures(run_dir)
        print(f"\nFigures written to {run_dir / 'figures'}:")
        for path in written:
            print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
