from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.data.dataset import MOSEIDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MANIFEST = (
    PROJECT_ROOT
    / "metadata/experiments/sentiment/train.parquet"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "datasets/CMU-MOSEI/Processed/aligned_50.pkl"
)


@pytest.mark.integration
def test_mosei_dataset():

    print("=" * 70)
    print("CMU-MOSEI PYTORCH DATASET TEST")
    print("=" * 70)

    dataset = MOSEIDataset(
        manifest_path=MANIFEST,
        feature_path=FEATURE_FILE,
    )

    print()
    print("DATASET LENGTH:", len(dataset))

    print()
    print("Loading first sample...")

    sample = dataset[0]

    print()
    print("SAMPLE:")
    print("ID:", sample["sample_id"])
    print("Dataset:", sample["dataset"])
    print("Split:", sample["split"])

    print()
    print("TENSOR SHAPES:")
    print("Audio:", sample["audio"].shape)
    print("Vision:", sample["vision"].shape)
    print("Text:", sample["text"].shape)

    print()
    print("TARGETS:")
    print(
        "Regression:",
        sample["regression"],
    )

    print(
        "Classification:",
        sample["classification"],
    )

    print(
        "Sentiment:",
        sample["sentiment_score"],
    )

    print()
    print("DTYPES:")
    print(
        "Audio:",
        sample["audio"].dtype,
    )

    print(
        "Vision:",
        sample["vision"].dtype,
    )

    print(
        "Text:",
        sample["text"].dtype,
    )

    assert len(dataset) == 16_326
    assert sample["audio"].shape == (50, 74)
    assert sample["vision"].shape == (50, 35)
    assert sample["text"].shape == (50, 768)
    assert sample["audio"].dtype == torch.float32
    assert sample["sentiment_score"].dtype == torch.float32
    assert torch.isfinite(sample["audio"]).all()
