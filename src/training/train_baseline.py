from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataloader import (
    create_mosei_dataloader,
)

from src.models.multimodal import (
    MultimodalSentimentBaseline,
)

from src.training.trainer import Trainer
from src.training.checkpoint import (
    CheckpointManager,
)
from src.training.evaluator import (
    Evaluator,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 5

DEVICE = torch.device(
    "cpu"
)

CHECKPOINT_DIR = Path(
    "checkpoints/cmu_mosei_baseline"
)

RESULTS_DIR = Path(
    "results/cmu_mosei_baseline"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CMU-MOSEI BASELINE TRAINING")
    print("=" * 70)

    print()
    print("DEVICE:", DEVICE)
    print("BATCH SIZE:", BATCH_SIZE)
    print("LEARNING RATE:", LEARNING_RATE)
    print("WEIGHT DECAY:", WEIGHT_DECAY)
    print("EPOCHS:", EPOCHS)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print("Loading datasets...")

    train_loader = create_mosei_dataloader(
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = create_mosei_dataloader(
        split="validation",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = create_mosei_dataloader(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "TRAIN BATCHES:",
        len(train_loader),
    )

    print(
        "VALIDATION BATCHES:",
        len(validation_loader),
    )

    print(
        "TEST BATCHES:",
        len(test_loader),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating baseline model...")

    model = (
        MultimodalSentimentBaseline()
        .to(DEVICE)
    )

    print(model)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Checkpoint manager
    # --------------------------------------------------------

    checkpoint_manager = (
        CheckpointManager(
            directory=CHECKPOINT_DIR,
            monitor="mae",
            mode="min",
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        train_result = (
            trainer.train_one_epoch(
                train_loader
            )
        )

        validation_result = (
            trainer.validate_one_epoch(
                validation_loader
            )
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,

                "train_loss":
                    train_result.loss,

                "train_mae":
                    train_result.mae,

                "train_rmse":
                    train_result.rmse,

                "validation_loss":
                    validation_result.loss,

                "validation_mae":
                    validation_result.mae,

                "validation_rmse":
                    validation_result.rmse,
            }
        )

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"  Train      "
            f"Loss={train_result.loss:.6f} "
            f"MAE={train_result.mae:.6f} "
            f"RMSE={train_result.rmse:.6f}"
        )

        print(
            f"  Validation "
            f"Loss={validation_result.loss:.6f} "
            f"MAE={validation_result.mae:.6f} "
            f"RMSE={validation_result.rmse:.6f}"
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        _, improved = (
            checkpoint_manager.save_epoch(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_result=train_result,
                validation_result=validation_result,
            )
        )

        if improved:
            print(
                "  Checkpoint: BEST UPDATED"
            )
        else:
            print(
                "  Checkpoint: best unchanged"
            )

    # --------------------------------------------------------
    # Global test evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)

    best_checkpoint = (
        CHECKPOINT_DIR /
        "best.pt"
    )

    checkpoint = (
        CheckpointManager.load(
            best_checkpoint,
            model=model,
            optimizer=None,
            device=DEVICE,
        )
    )

    print(
        "Best epoch:",
        checkpoint["epoch"],
    )

    print(
        "Best validation MAE:",
        checkpoint[
            "monitor_value"
        ],
    )

    # --------------------------------------------------------
    # Test evaluator
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        device=DEVICE,
    )

    test_metrics = evaluator.evaluate(
        test_loader
    )

    print()
    print("TEST RESULTS")

    for key, value in test_metrics.items():

        print(
            f"{key:10}: {value}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    import json

    results = {
        "configuration": {
            "batch_size": BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "epochs": EPOCHS,
            "device": str(DEVICE),
        },

        "best_epoch":
            checkpoint["epoch"],

        "best_validation_mae":
            checkpoint[
                "monitor_value"
            ],

        "history": history,

        "test": test_metrics,
    }

    results_path = (
        RESULTS_DIR /
        "results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print()
    print(
        "Results saved:"
    )

    print(
        f"  {results_path}"
    )

    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()