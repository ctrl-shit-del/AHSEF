import pandas as pd
import torch

from src.data.dataloader import experiment_collate


def test_experiment_collate_keeps_missing_modalities_explicit():
    samples = [
        {"sample_id": "a", "dataset": "one", "split": "train", "label": torch.tensor(1), "audio": torch.ones(2, 3), "video": None, "image": None, "text": "one", "physiology": None, "metadata": {}},
        {"sample_id": "b", "dataset": "one", "split": "train", "label": torch.tensor(2), "audio": torch.ones(4, 3), "video": None, "image": None, "text": None, "physiology": None, "metadata": {}},
    ]
    batch = experiment_collate(samples)
    assert batch["audio"].shape == (2, 4, 3)
    assert batch["audio_lengths"].tolist() == [2, 4]
    assert batch["text"] == ["one", None]
    assert batch["video"] == [None, None]
