import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from src.data.dataloader import (
    create_mosei_dataloader,
)

from src.models.multimodal import (
    MultimodalSentimentBaseline,
)

from src.training.evaluator import (
    Evaluator,
)


@pytest.mark.integration
def test_evaluator():

    print("=" * 70)
    print("GLOBAL EVALUATOR TEST")
    print("=" * 70)

    device = torch.device(
        "cpu"
    )

    loader = create_mosei_dataloader(
        split="validation",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    model = (
        MultimodalSentimentBaseline()
        .to(device)
    )

    criterion = nn.MSELoss()

    evaluator = Evaluator(
        model=model,
        criterion=criterion,
        device=device,
    )

    metrics = evaluator.evaluate(
        loader
    )

    print()
    print("GLOBAL VALIDATION METRICS")

    for key, value in metrics.items():

        print(
            f"{key:10}: {value}"
        )

    assert metrics["samples"] == 1871

    assert metrics["loss"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0

    assert torch.isfinite(
        torch.tensor(
            metrics["loss"]
        )
    )

    assert torch.isfinite(
        torch.tensor(
            metrics["mae"]
        )
    )

    assert torch.isfinite(
        torch.tensor(
            metrics["rmse"]
        )
    )
