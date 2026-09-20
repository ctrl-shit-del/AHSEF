"""Synthetic Stage 3 material, so the routing tests never call an LLM.

The frames here mimic exactly the columns
:func:`src.ahsef.llm.inference.build_prediction_set` produces, including the
``prob_*`` columns holding *normalised self-reported scores* and the LLM's own
``llm_confidence`` / ``llm_ambiguity`` self-reports.  Anything a test asserts
about column handling therefore transfers to the real artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ahsef.inference import PredictionSet
from src.ahsef.llm.inference import LLM_MODALITY
from src.ahsef.uncertainty import uncertainty_columns
from src.common.labels import CANONICAL_EMOTION_CLASSES

import torch

CLASSES = tuple(CANONICAL_EMOTION_CLASSES)


def make_llm_set(
    sample_ids,
    predicted,
    labels,
    scores=None,
    split: str = "validation",
    usable=None,
    latency_ms: float = 4000.0,
    seed: int = 0,
) -> PredictionSet:
    """A text_llm prediction set with every column the real exporter writes."""
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    if scores is None:
        scores = np.full((n, 7), 0.05)
        for row, klass in enumerate(predicted):
            if klass >= 0:
                scores[row, klass] = 0.4 + 0.4 * rng.random()
        scores = scores / scores.sum(axis=1, keepdims=True)
    scores = np.asarray(scores, dtype=float)
    usable = [True] * n if usable is None else list(usable)

    ordered = np.sort(scores, axis=1)[:, ::-1]
    top1, top2 = ordered[:, 0], ordered[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(scores > 0, scores * np.log(scores), 0.0), axis=1)

    frame = pd.DataFrame({
        "sample_id": [str(value) for value in sample_ids],
        "dataset": "SYNTH",
        "modality": LLM_MODALITY,
        "split": split,
        "true_class": [int(value) for value in labels],
        "predicted_class": [int(value) for value in predicted],
        "llm_status": ["ok" if flag else "abstain" for flag in usable],
        "llm_usable": usable,
        "llm_abstain": [not flag for flag in usable],
        "llm_raw_emotion": "",
        "llm_error": "",
        "llm_evidence": "",
        "repeats": 1,
        "votes": "",
        "latency_ms": latency_ms,
        "inference_ms": latency_ms,
        "feature_ms": 0.0,
        "cost_usd": 0.0,
        "input_tokens": 700,
        "output_tokens": 150,
        "cache_read_tokens": 0,
        "llm_confidence": np.clip(top1 + 0.1, 0.0, 1.0),
        "llm_uncertainty": 1.0 - np.clip(top1 + 0.1, 0.0, 1.0),
        "llm_ambiguity": 1.0 - top1,
        "llm_evidence_strength": top1,
        "llm_score_entropy": entropy,
        "llm_normalized_score_entropy": entropy / np.log(7),
        "llm_score_margin": top1 - top2,
        "llm_score_top1": top1,
        "llm_score_top2": top2,
    })
    for index in range(7):
        frame[f"prob_{index}"] = scores[:, index]
        frame[f"logit_{index}"] = np.log(np.clip(scores[:, index], 1e-12, None))
    frame["confidence"] = frame["llm_confidence"]
    frame["predictive_entropy"] = entropy
    frame["normalized_entropy"] = entropy / np.log(7)
    frame["margin"] = top1 - top2
    return PredictionSet(
        modality=LLM_MODALITY, split=split, class_order=CLASSES, frame=frame,
        meta={"kind": "llm", "distribution_is_calibrated_posterior": False},
    )


def make_audio_set(
    sample_ids, predicted, labels, split: str = "validation", latency_ms: float = 12.0
) -> PredictionSet:
    """A frozen-baseline-shaped audio prediction set."""
    n = len(sample_ids)
    logits = np.full((n, 7), 0.0)
    for row, klass in enumerate(predicted):
        logits[row, int(klass)] = 2.5
    tensor = torch.tensor(logits, dtype=torch.float64)
    probabilities = torch.softmax(tensor, dim=-1)
    frame = pd.DataFrame({
        "sample_id": [str(value) for value in sample_ids],
        "dataset": "SYNTH",
        "modality": "audio",
        "split": split,
        "true_class": [int(value) for value in labels],
        "predicted_class": [int(value) for value in predicted],
    })
    for index in range(7):
        frame[f"logit_{index}"] = tensor[:, index].numpy()
        frame[f"prob_{index}"] = probabilities[:, index].numpy()
    for name, values in uncertainty_columns(probabilities).items():
        frame[name] = values.numpy()
    frame["latency_ms"] = latency_ms
    frame["inference_ms"] = latency_ms * 0.6
    frame["feature_ms"] = latency_ms * 0.4
    return PredictionSet(
        modality="audio", split=split, class_order=CLASSES, frame=frame,
        meta={"experiment": "audio_25pct", "iteration": "1",
              "model": {"class": "AudioEmotionBaseline", "parameters": 100_000,
                        "max_frames": 300, "n_mels": 64}},
    )


def scenario(n: int = 120, seed: int = 7):
    """A pool where audio helps some samples, harms a few, and is neutral elsewhere.

    Built so the fitted estimator has something real to learn: the samples audio
    fixes are drawn from the high-uncertainty region, which is exactly the
    structure HSIG is supposed to discover without being told.
    """
    rng = np.random.default_rng(seed)
    ids = [f"v{index:04d}" for index in range(n)]
    labels = rng.integers(0, 4, n)
    uncertainty_rank = rng.random(n)

    text_pred = labels.copy()
    wrong = uncertainty_rank > 0.45
    text_pred[wrong] = (labels[wrong] + 1) % 7

    audio_pred = labels.copy()
    audio_wrong = rng.random(n) > 0.7
    audio_pred[audio_wrong] = (labels[audio_wrong] + 2) % 7

    scores = np.full((n, 7), 0.02)
    for row in range(n):
        top = 0.9 - 0.6 * uncertainty_rank[row]
        scores[row, text_pred[row]] = top
        scores[row, (text_pred[row] + 3) % 7] = 0.5 * (1 - top)
    scores = scores / scores.sum(axis=1, keepdims=True)

    return {
        "ids": ids,
        "labels": labels,
        "text": make_llm_set(ids, text_pred, labels, scores=scores, seed=seed),
        "audio": make_audio_set(ids, audio_pred, labels),
    }
