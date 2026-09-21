import pytest

torch = pytest.importorskip("torch")

from src.data.dataloader import (
    create_mosei_dataloader,
)


@pytest.mark.integration
def test_mosei_dataloader_contract():

    print("=" * 70)
    print("CMU-MOSEI DATALOADER CONTRACT TEST")
    print("=" * 70)

    loader = create_mosei_dataloader(
        split="train",
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    print()
    print("BATCH SIZE:", loader.batch_size)
    print("NUMBER OF BATCHES:", len(loader))

    batch = next(iter(loader))

    print()
    print("BATCH CONTENT:")

    for key, value in batch.items():

        if isinstance(value, torch.Tensor):

            print(
                f"{key:20}"
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )

        else:

            print(
                f"{key:20}"
                f"type={type(value).__name__}"
            )

    # ---------------------------------------------------------
    # Expected batch contract
    # ---------------------------------------------------------

    assert batch["audio"].shape == (
        8,
        50,
        74,
    )

    assert batch["vision"].shape == (
        8,
        50,
        35,
    )

    assert batch["text"].shape == (
        8,
        50,
        768,
    )

    assert batch["regression"].shape == (
        8,
    )

    assert batch["classification"].shape == (
        8,
    )

    assert batch["sentiment_score"].shape == (
        8,
    )

    # ---------------------------------------------------------
    # Expected dtypes
    # ---------------------------------------------------------

    assert batch["audio"].dtype == torch.float32
    assert batch["vision"].dtype == torch.float32
    assert batch["text"].dtype == torch.float32

    assert batch["regression"].dtype == torch.float32
    assert batch["classification"].dtype == torch.int64
    assert batch["sentiment_score"].dtype == torch.float32

    # ---------------------------------------------------------
    # Basic target sanity
    # ---------------------------------------------------------

    assert torch.isfinite(
        batch["audio"]
    ).all()

    assert torch.isfinite(
        batch["vision"]
    ).all()

    assert torch.isfinite(
        batch["text"]
    ).all()

    assert torch.isfinite(
        batch["regression"]
    ).all()

    assert torch.isfinite(
        batch["sentiment_score"]
    ).all()

    assert (
        batch["classification"].min() >= 0
    )

    assert (
        batch["classification"].max() <= 2
    )

    print()
    print("SAMPLE IDS:")

    for sample_id in batch["sample_id"]:
        print(" ", sample_id)

    print()
    print("SENTIMENT:")
    print(batch["sentiment_score"])

    print()
    print("CLASSIFICATION:")
    print(batch["classification"])

    print()
    print("DATALOADER CONTRACT TEST PASSED")
