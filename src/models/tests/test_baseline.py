import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.data.dataloader import (
    create_mosei_dataloader,
)

from src.models.multimodal import (
    MultimodalSentimentBaseline,
)


@pytest.mark.integration
def test_baseline_model():

    print("=" * 70)
    print("CMU-MOSEI MULTIMODAL BASELINE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device("cpu")

    print()
    print("DEVICE:", device)

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

    loader = create_mosei_dataloader(
        split="train",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MultimodalSentimentBaseline()

    model = model.to(device)

    print()
    print("MODEL:")
    print(model)

    # ---------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------

    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    text = batch["text"].to(device)

    target = batch[
        "sentiment_score"
    ].to(device)

    print()
    print("INPUTS:")
    print("Audio :", audio.shape)
    print("Vision:", vision.shape)
    print("Text  :", text.shape)
    print("Target:", target.shape)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    model.train()

    prediction = model(
        audio=audio,
        vision=vision,
        text=text,
    )

    print()
    print("PREDICTION:")
    print(
        "Shape:",
        prediction.shape,
    )

    print(
        "Values:",
        prediction.detach(),
    )

    assert prediction.shape == (
        8,
    )

    assert prediction.dtype == (
        torch.float32
    )

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------

    criterion = nn.MSELoss()

    loss = criterion(
        prediction,
        target,
    )

    print()
    print("LOSS:")
    print(loss)

    assert torch.isfinite(loss)

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    optimizer.zero_grad()

    loss.backward()

    # ---------------------------------------------------------
    # Gradient validation
    # ---------------------------------------------------------

    gradient_count = 0

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            gradient_count += 1

            assert torch.isfinite(
                parameter.grad
            ).all()

    print()
    print(
        "PARAMETERS WITH GRADIENTS:",
        gradient_count,
    )

    assert gradient_count > 0

    # ---------------------------------------------------------
    # Optimizer step
    # ---------------------------------------------------------

    optimizer.step()

    print()
    print("OPTIMIZER STEP: PASSED")
