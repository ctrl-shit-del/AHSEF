from pathlib import Path
import tempfile

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.training.checkpoint import (
    CheckpointManager,
)
from src.training.trainer import (
    EpochResult,
)


@pytest.mark.integration
def test_checkpoint_manager():

    print("=" * 70)
    print("CHECKPOINT MANAGER TEST")
    print("=" * 70)

    model = nn.Linear(
        10,
        1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    train_result = EpochResult(
        loss=0.80,
        mae=0.70,
        rmse=0.90,
    )

    validation_result = EpochResult(
        loss=0.60,
        mae=0.55,
        rmse=0.77,
    )

    with tempfile.TemporaryDirectory() as tmp:

        manager = CheckpointManager(
            directory=Path(tmp),
            monitor="mae",
            mode="min",
        )

        print()
        print(
            "Saving epoch 1..."
        )

        last_path, improved = (
            manager.save_epoch(
                model=model,
                optimizer=optimizer,
                epoch=1,
                train_result=train_result,
                validation_result=validation_result,
            )
        )

        print(
            "Last:",
            last_path,
        )

        print(
            "Improved:",
            improved,
        )

        assert last_path.exists()
        assert improved

        best_path = (
            Path(tmp) /
            "best.pt"
        )

        assert best_path.exists()

        print()
        print(
            "Loading best checkpoint..."
        )

        new_model = nn.Linear(
            10,
            1,
        )

        new_optimizer = torch.optim.AdamW(
            new_model.parameters(),
            lr=1e-3,
        )

        checkpoint = (
            CheckpointManager.load(
                best_path,
                model=new_model,
                optimizer=new_optimizer,
                device="cpu",
            )
        )

        print(
            "Epoch:",
            checkpoint["epoch"],
        )

        print(
            "Monitor:",
            checkpoint["monitor"],
        )

        print(
            "Monitor value:",
            checkpoint[
                "monitor_value"
            ],
        )

        assert (
            checkpoint["epoch"]
            == 1
        )

        assert (
            checkpoint["monitor"]
            == "mae"
        )

        assert (
            checkpoint[
                "monitor_value"
            ]
            == 0.55
        )

from src.training.classification_trainer import ClassificationEpochResult

@pytest.mark.integration
def test_checkpoint_manager_classification():
    model = nn.Linear(10, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_result = ClassificationEpochResult(
        loss=0.80,
        accuracy=0.5,
        macro_f1=0.4,
    )
    validation_result = ClassificationEpochResult(
        loss=0.60,
        accuracy=0.6,
        macro_f1=0.5,
    )

    with tempfile.TemporaryDirectory() as tmp:
        manager = CheckpointManager(
            directory=Path(tmp),
            monitor="macro_f1",
            mode="max",
        )

        last_path, improved = manager.save_epoch(
            model=model,
            optimizer=optimizer,
            epoch=1,
            train_result=train_result,
            validation_result=validation_result,
        )

        assert last_path.exists()
        assert improved

        best_path = Path(tmp) / "best.pt"
        assert best_path.exists()

        checkpoint = CheckpointManager.load(
            best_path,
            model=nn.Linear(10, 7),
            optimizer=None,
            device="cpu",
        )

        assert checkpoint["epoch"] == 1
        assert checkpoint["monitor"] == "macro_f1"
        assert checkpoint["monitor_value"] == 0.5
        assert "loss" in checkpoint["train"]
        assert "macro_f1" in checkpoint["train"]
        assert "mae" not in checkpoint["train"]
