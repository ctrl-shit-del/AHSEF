"""Image-only 7-class categorical emotion baseline.

This is an explicitly modality-filtered experiment.  Only records with
``has_image == True`` and ``image_source == 'file'`` are included.  The
datasets contributing images are AffectNet+, FERPlus, and RAF-DB.

MELD does not provide an image modality in the standardized experiment and is
excluded by the filter -- no fabricated tensors are substituted.

Class mapping (canonical, immutable):
    0 = neutral, 1 = happy, 2 = sad, 3 = angry,
    4 = fear, 5 = disgust, 6 = surprise

Class weights are computed from the TRAINING SPLIT ONLY.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataloader import create_emotion_image_dataloader
from src.data.image_dataset import EmotionImageDataset
from src.models.multimodal import ImageEmotionBaseline
from src.training.checkpoint import CheckpointManager
from src.training.classification_evaluator import ClassificationEvaluator
from src.training.classification_trainer import ClassificationTrainer
from src.training.metrics import class_weights


# ============================================================
# Configuration
# ============================================================

@dataclass
class EmotionImageConfig:
    """Fully reproducible experiment configuration."""

    experiment_name: str = "emotion_image_baseline"
    experiment_description: str = (
        "Image-only modality-filtered 7-class categorical emotion baseline."
    )

    # data
    manifest_dir: str = "metadata/experiments/emotion_7class"
    modality: str = "image"
    num_classes: int = 7
    class_mapping: dict | None = None
    image_size: int = 48

    # training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    seed: int = 42

    # model
    hidden_dim: int = 256

    # optimizer
    optimizer: str = "AdamW"
    loss: str = "CrossEntropyLoss"
    class_weight_policy: str = "inverse_frequency_from_train"

    # infrastructure
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints/emotion_image_baseline"
    results_dir: str = "results/emotion_image_baseline"

    # bounded mode
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None

    def __post_init__(self):
        if self.class_mapping is None:
            self.class_mapping = {
                "neutral": 0, "happy": 1, "sad": 2, "angry": 3,
                "fear": 4, "disgust": 5, "surprise": 6,
            }


def compute_class_weights(manifest_path: str | Path, num_classes: int = 7) -> torch.Tensor:
    """Compute inverse-frequency class weights from the TRAINING split only."""
    dataset = EmotionImageDataset(manifest_path, image_size=1, max_samples=None)
    labels = torch.tensor(
        dataset.manifest["canonical_emotion_id"].astype(int).tolist(),
        dtype=torch.long,
    )
    weights = class_weights(labels, num_classes)
    return weights


def get_class_distribution(manifest_path: str | Path) -> dict[int, int]:
    """Return per-class sample count from a manifest."""
    dataset = EmotionImageDataset(manifest_path, image_size=1, max_samples=None)
    counts = dataset.manifest["canonical_emotion_id"].astype(int).value_counts().sort_index()
    return {int(k): int(v) for k, v in counts.items()}


def main(config: EmotionImageConfig | None = None):
    config = config or EmotionImageConfig()

    print("=" * 70)
    print("IMAGE-ONLY 7-CLASS EMOTION BASELINE")
    print("=" * 70)
    print()
    print(f"Experiment: {config.experiment_name}")
    print(f"Description: {config.experiment_description}")
    print(f"Device: {config.device}")
    print(f"Image size: {config.image_size}")
    print(f"Hidden dim: {config.hidden_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Epochs: {config.epochs}")
    print(f"Seed: {config.seed}")

    device = torch.device(config.device)
    torch.manual_seed(config.seed)

    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = Path(config.manifest_dir)
    train_manifest = manifest_dir / "train.parquet"
    val_manifest = manifest_dir / "validation.parquet"
    test_manifest = manifest_dir / "test.parquet"

    # --------------------------------------------------------
    # Class weights (training split only)
    # --------------------------------------------------------

    print()
    print("Computing class weights from TRAINING split...")

    weights = compute_class_weights(train_manifest, config.num_classes)
    class_dist = get_class_distribution(train_manifest)

    print(f"Training class distribution: {class_dist}")
    print(f"Class weights: {weights.tolist()}")

    weights_record = {
        "policy": config.class_weight_policy,
        "source": str(train_manifest),
        "distribution": class_dist,
        "weights": weights.tolist(),
    }
    with open(results_dir / "class_weights.json", "w") as f:
        json.dump(weights_record, f, indent=2)

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    print()
    print("Creating data loaders...")

    train_loader = create_emotion_image_dataloader(
        train_manifest, batch_size=config.batch_size,
        image_size=config.image_size, max_samples=config.max_train_samples,
        seed=config.seed, shuffle=True,
    )
    val_loader = create_emotion_image_dataloader(
        val_manifest, batch_size=config.batch_size,
        image_size=config.image_size, max_samples=config.max_val_samples,
        seed=config.seed, shuffle=False,
    )
    test_loader = create_emotion_image_dataloader(
        test_manifest, batch_size=config.batch_size,
        image_size=config.image_size, max_samples=config.max_test_samples,
        seed=config.seed, shuffle=False,
    )

    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = ImageEmotionBaseline(
        image_size=config.image_size,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Total parameters: {total_params:,}")

    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # --------------------------------------------------------
    # Trainer and checkpoint
    # --------------------------------------------------------

    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=config.num_classes,
    )

    checkpoint_manager = CheckpointManager(
        directory=config.checkpoint_dir,
        monitor="macro_f1",
        mode="max",
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = []
    start_time = time.time()

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        print()
        print(f"Epoch {epoch}/{config.epochs}")

        train_result = trainer.train_one_epoch(train_loader)
        val_result = trainer.validate_one_epoch(val_loader)

        epoch_time = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_macro_f1": train_result.macro_f1,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_macro_f1": val_result.macro_f1,
            "epoch_seconds": round(epoch_time, 2),
        })

        print(
            f"  Train      Loss={train_result.loss:.6f} "
            f"Acc={train_result.accuracy:.4f} "
            f"F1={train_result.macro_f1:.4f}"
        )
        print(
            f"  Validation Loss={val_result.loss:.6f} "
            f"Acc={val_result.accuracy:.4f} "
            f"F1={val_result.macro_f1:.4f}"
        )
        print(f"  Time: {epoch_time:.1f}s")

        _, improved = checkpoint_manager.save_epoch(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_result=train_result,
            validation_result=val_result,
        )

        if improved:
            print("  Checkpoint: BEST UPDATED")
        else:
            print("  Checkpoint: best unchanged")

    total_time = time.time() - start_time

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # --------------------------------------------------------
    # Load best checkpoint and evaluate
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    best_checkpoint_path = Path(config.checkpoint_dir) / "best.pt"
    checkpoint = CheckpointManager.load(
        best_checkpoint_path, model=model, optimizer=None, device=device,
    )

    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation macro-F1: {checkpoint['monitor_value']:.4f}")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluator = ClassificationEvaluator(
        model=model,
        criterion=criterion,
        device=device,
        num_classes=config.num_classes,
    )

    print()
    print("=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    val_metrics = evaluator.evaluate(val_loader)
    _print_metrics(val_metrics)

    with open(results_dir / "validation_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    print()
    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    test_metrics = evaluator.evaluate(test_loader)
    _print_metrics(test_metrics)

    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    with open(results_dir / "confusion_matrix.json", "w") as f:
        json.dump({
            "class_names": list(config.class_mapping.keys()),
            "validation": val_metrics["confusion_matrix"],
            "test": test_metrics["confusion_matrix"],
        }, f, indent=2)

    # --------------------------------------------------------
    # Configuration record
    # --------------------------------------------------------

    config_record = {
        "experiment_name": config.experiment_name,
        "experiment_description": config.experiment_description,
        "manifest_dir": config.manifest_dir,
        "modality": config.modality,
        "num_classes": config.num_classes,
        "class_mapping": config.class_mapping,
        "image_size": config.image_size,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "seed": config.seed,
        "hidden_dim": config.hidden_dim,
        "optimizer": config.optimizer,
        "loss": config.loss,
        "class_weight_policy": config.class_weight_policy,
        "device": config.device,
        "checkpoint_dir": config.checkpoint_dir,
        "results_dir": config.results_dir,
        "max_train_samples": config.max_train_samples,
        "max_val_samples": config.max_val_samples,
        "max_test_samples": config.max_test_samples,
    }

    with open(results_dir / "config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    # --------------------------------------------------------
    # Run summary
    # --------------------------------------------------------

    run_summary = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "seed": config.seed,
        "model_parameters": total_params,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "class_distribution": class_dist,
        "class_weights": weights.tolist(),
        "epochs_completed": len(history),
        "best_epoch": checkpoint["epoch"],
        "best_val_macro_f1": checkpoint["monitor_value"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "total_training_seconds": round(total_time, 2),
        "checkpoint_best": str(best_checkpoint_path),
        "checkpoint_last": str(Path(config.checkpoint_dir) / "last.pt"),
    }

    with open(results_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time:.1f}s")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best val macro-F1: {checkpoint['monitor_value']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Results: {results_dir}")
    print(f"Checkpoints: {config.checkpoint_dir}")

    return run_summary


def _print_metrics(metrics: dict):
    """Print structured classification metrics."""
    print(f"  Samples:  {metrics['samples']}")
    print(f"  Loss:     {metrics['loss']:.6f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro P:  {metrics['macro_precision']:.4f}")
    print(f"  Macro R:  {metrics['macro_recall']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Wt. F1:   {metrics['weighted_f1']:.4f}")
    if "per_class" in metrics:
        print()
        print(f"  {'Class':<12} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        print(f"  {'-'*44}")
        for name, vals in metrics["per_class"].items():
            print(
                f"  {name:<12} {vals['precision']:>8.4f} "
                f"{vals['recall']:>8.4f} {vals['f1']:>8.4f} "
                f"{vals['support']:>8d}"
            )


def run_debug(config: EmotionImageConfig | None = None):
    """Run a tiny bounded experiment to verify the full pipeline."""
    config = config or EmotionImageConfig()
    config.experiment_name = "emotion_image_debug"
    config.max_train_samples = 256
    config.max_val_samples = 128
    config.max_test_samples = 128
    config.epochs = 2
    config.batch_size = 32
    config.checkpoint_dir = "checkpoints/emotion_image_debug"
    config.results_dir = "results/emotion_image_debug"
    return main(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image-only emotion baseline")
    parser.add_argument("--debug", action="store_true", help="Run tiny debug experiment")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = EmotionImageConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.max_train is not None:
        cfg.max_train_samples = args.max_train
    if args.max_val is not None:
        cfg.max_val_samples = args.max_val
    if args.max_test is not None:
        cfg.max_test_samples = args.max_test
    if args.seed is not None:
        cfg.seed = args.seed

    if args.debug:
        run_debug(cfg)
    else:
        main(cfg)
