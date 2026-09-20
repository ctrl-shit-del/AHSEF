"""Synthetic prediction sets, so the fusion tests never open a checkpoint."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from src.ahsef.inference import PredictionSet
from src.ahsef.uncertainty import probabilities_from_logits, uncertainty_columns


EMOTION_CLASSES = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")
WESAD_CLASSES = ("baseline", "stress", "amusement")


def make_prediction_set(
    modality: str,
    sample_ids,
    logits: torch.Tensor,
    labels,
    split: str = "test",
    class_order=EMOTION_CLASSES,
    latency_ms: float = 1.0,
    dataset: str = "SYNTH",
    meta: dict | None = None,
) -> PredictionSet:
    """Build a PredictionSet the way :class:`BaselinePredictor` would."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    probabilities = probabilities_from_logits(logits)
    frame = pd.DataFrame({
        "sample_id": [str(value) for value in sample_ids],
        "dataset": dataset,
        "modality": modality,
        "split": split,
        "true_class": [int(value) for value in labels],
    })
    for index in range(logits.shape[1]):
        frame[f"logit_{index}"] = logits[:, index].numpy()
        frame[f"prob_{index}"] = probabilities[:, index].numpy()
    for name, values in uncertainty_columns(probabilities).items():
        frame[name] = values.numpy()
    frame["latency_ms"] = latency_ms
    frame["inference_ms"] = latency_ms * 0.5
    frame["feature_ms"] = latency_ms * 0.5
    return PredictionSet(
        modality=modality, split=split, class_order=tuple(class_order), frame=frame,
        meta={"experiment": f"{modality}_synthetic", "iteration": "1",
              "model": {"class": "Synthetic", "parameters": 1000}, **(meta or {})},
    )


@pytest.fixture
def anchor_set() -> PredictionSet:
    """Six samples: confident-and-right, uncertain, and confidently wrong."""
    logits = torch.tensor([
        [5.0, 0, 0, 0, 0, 0, 0],   # s0 confident neutral, correct
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # s1 near-uniform, label happy
        [0.0, 4.0, 0, 0, 0, 0, 0],  # s2 confident happy, correct
        [3.0, 0, 0, 0, 0, 0, 0],   # s3 confident neutral, label sad -> wrong
        [0.0, 0.0, 0.5, 0.4, 0, 0, 0],  # s4 mild sad, correct
        [0.0, 0.0, 0.0, 2.0, 0, 0, 0],  # s5 confident angry, correct
    ], dtype=torch.float64)
    return make_prediction_set("audio", [f"s{i}" for i in range(6)], logits,
                               [0, 1, 1, 2, 2, 3])


@pytest.fixture
def candidate_set() -> PredictionSet:
    """Same six samples, a model that is right where the anchor is wrong."""
    logits = torch.tensor([
        [1.0, 0.5, 0, 0, 0, 0, 0],
        [0.0, 3.0, 0, 0, 0, 0, 0],   # s1 confident happy -> fixes the anchor
        [0.0, 1.0, 0, 0, 0, 0, 0],
        [0.0, 0.0, 3.0, 0, 0, 0, 0],  # s3 confident sad -> fixes the anchor
        [0.0, 0.0, 0.6, 0, 0, 0, 0],
        [0.5, 0.0, 0.0, 0.6, 0, 0, 0],
    ], dtype=torch.float64)
    return make_prediction_set("text", [f"s{i}" for i in range(6)], logits,
                               [0, 1, 1, 2, 2, 3], latency_ms=0.2)


@pytest.fixture
def wesad_set() -> PredictionSet:
    """A three-class prediction set that must never be fused with the others."""
    logits = torch.tensor([[2.0, 0.0, 0.0]] * 6, dtype=torch.float64)
    return make_prediction_set(
        "physiology", [f"s{i}" for i in range(6)], logits, [0] * 6,
        class_order=WESAD_CLASSES,
    )
