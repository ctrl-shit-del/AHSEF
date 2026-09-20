import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.data.dataloader import (
    create_mosei_dataloader,
)

from src.models.multimodal import (
    MultimodalSentimentBaseline,
)


@pytest.mark.smoke
def test_training_smoke():

    print("=" * 70)
    print("CMU-MOSEI TRAINING SMOKE TEST")
    print("=" * 70)

    device = torch.device("cpu")

    print()
    print("DEVICE:", device)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    loader = create_mosei_dataloader(
        split="train",
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MultimodalSentimentBaseline(
        audio_dim=74,
        vision_dim=35,
        text_dim=768,
        hidden_dim=128,
        fusion_dim=256,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.train()

    num_steps = 50

    losses = []

    print()
    print("TRAINING")
    print("-" * 70)

    for step, batch in enumerate(loader):

        if step >= num_steps:
            break

        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        text = batch["text"].to(device)

        target = batch[
            "sentiment_score"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            audio=audio,
            vision=vision,
            text=text,
        )

        loss = criterion(
            prediction,
            target,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{loss}"
            )

        loss.backward()

        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if (
            step == 0
            or (step + 1) % 10 == 0
        ):

            print(
                f"Step {step + 1:3d}/{num_steps} "
                f"Loss: {loss_value:.6f}"
            )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if len(losses) < 2:
        raise RuntimeError(
            "Not enough training steps completed."
        )

    print()
    print("-" * 70)

    print(
        f"Initial loss: {losses[0]:.6f}"
    )

    print(
        f"Final loss:   {losses[-1]:.6f}"
    )

    print(
        f"Minimum loss: {min(losses):.6f}"
    )

    # We don't require monotonic decrease.
    #
    # SGD/Adam losses naturally fluctuate between batches.
    # Instead, compare the average of the first and last
    # few steps.

    window = min(10, len(losses))

    initial_mean = sum(
        losses[:window]
    ) / window

    final_mean = sum(
        losses[-window:]
    ) / window

    print(
        f"First {window} mean: "
        f"{initial_mean:.6f}"
    )

    print(
        f"Last {window} mean:  "
        f"{final_mean:.6f}"
    )

    print()

    if final_mean < initial_mean:
        print(
            "TRAINING SIGNAL: LOSS DECREASED"
        )
    else:
        print(
            "TRAINING SIGNAL: "
            "LOSS DID NOT DECREASE"
        )
