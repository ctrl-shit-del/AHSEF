import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.data.dataloader import (
    create_mosei_dataloader,
)

from src.models.multimodal import (
    MultimodalSentimentBaseline,
)

from src.training.trainer import (
    Trainer,
)


@pytest.mark.integration
def test_trainer_one_epoch():

    print("=" * 70)
    print("TRAINER ONE-EPOCH TEST")
    print("=" * 70)

    device = torch.device("cpu")

    print()
    print("DEVICE:", device)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    train_loader = create_mosei_dataloader(
        split="train",
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = create_mosei_dataloader(
        split="validation",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MultimodalSentimentBaseline().to(
        device
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
    )

    # ---------------------------------------------------------
    # One epoch
    # ---------------------------------------------------------

    print()
    print("Running one training epoch...")

    train_result, validation_result = (
        trainer.fit_one_epoch(
            train_loader,
            validation_loader,
        )
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print()
    print("TRAIN")
    print(
        f"  Loss : {train_result.loss:.6f}"
    )
    print(
        f"  MAE  : {train_result.mae:.6f}"
    )
    print(
        f"  RMSE : {train_result.rmse:.6f}"
    )

    print()
    print("VALIDATION")
    print(
        f"  Loss : {validation_result.loss:.6f}"
    )
    print(
        f"  MAE  : {validation_result.mae:.6f}"
    )
    print(
        f"  RMSE : {validation_result.rmse:.6f}"
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert torch.isfinite(
        torch.tensor(train_result.loss)
    )

    assert torch.isfinite(
        torch.tensor(train_result.mae)
    )

    assert torch.isfinite(
        torch.tensor(train_result.rmse)
    )

    assert torch.isfinite(
        torch.tensor(validation_result.loss)
    )

    assert torch.isfinite(
        torch.tensor(validation_result.mae)
    )

    assert torch.isfinite(
        torch.tensor(validation_result.rmse)
    )

    assert train_result.loss >= 0
    assert train_result.mae >= 0
    assert train_result.rmse >= 0

    assert validation_result.loss >= 0
    assert validation_result.mae >= 0
    assert validation_result.rmse >= 0
