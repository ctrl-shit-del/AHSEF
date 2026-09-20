"""Shape, masking, and gradient tests for the per-modality baselines.

Every classifier must emit exactly ``num_classes`` logits in the declared class
order, and every sequence model must ignore padded steps -- otherwise a short
clip's representation is diluted by zeros and the metric it produces is not the
one being reported.
"""

import pytest
import torch
import torch.nn as nn

from src.common.labels import EMOTION_7CLASS, WESAD_STATE_3CLASS
from src.models.baselines import (
    AudioEmotionBaseline,
    PhysiologyStateBaseline,
    TextEmotionBaseline,
    VideoEmotionBaseline,
    masked_mean,
    masked_mean_std,
    sequence_mask,
)
from src.training.metrics import class_weights, classification_metrics


# ============================================================
# Pooling primitives
# ============================================================

def test_sequence_mask_marks_only_real_steps():
    mask = sequence_mask(torch.tensor([1, 3]), 4)
    assert mask.shape == (2, 4, 1)
    assert mask[0].squeeze(-1).tolist() == [1, 0, 0, 0]
    assert mask[1].squeeze(-1).tolist() == [1, 1, 1, 0]


def test_masked_mean_ignores_padding():
    features = torch.tensor([[[1.0, 1.0], [5.0, 5.0], [0.0, 0.0]]])
    pooled = masked_mean(features, torch.tensor([2]))
    assert torch.allclose(pooled, torch.tensor([[3.0, 3.0]]))
    # Without the mask the zero-padded step drags the mean down.
    assert torch.allclose(masked_mean(features, None), torch.tensor([[2.0, 2.0]]))


def test_masked_mean_std_concatenates_and_stays_finite():
    features = torch.randn(3, 6, 4)
    pooled = masked_mean_std(features, torch.tensor([6, 1, 3]))
    assert pooled.shape == (3, 8)
    assert bool(torch.isfinite(pooled).all())
    # A single valid step has zero variance, which must not become NaN.
    assert float(pooled[1, 4:].abs().max()) < 1e-3


def test_padding_does_not_change_a_pooled_representation():
    model = AudioEmotionBaseline(n_mels=8, hidden_dim=8)
    model.eval()
    real = torch.randn(1, 5, 8)
    padded = torch.cat([real, torch.zeros(1, 7, 8)], dim=1)
    with torch.no_grad():
        short = model(real, torch.tensor([5]))
        long = model(padded, torch.tensor([5]))
    assert torch.allclose(short, long, atol=1e-5)


# ============================================================
# Seven-logit contract
# ============================================================

def _train_step(model, inputs, lengths, labels, num_classes):
    logits = model(inputs, lengths) if lengths is not None else model(inputs)
    assert logits.shape == (labels.shape[0], num_classes)
    weights = class_weights(labels, num_classes)
    loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return logits


def test_audio_baseline_emits_seven_logits():
    labels = torch.tensor([0, 1, 6, 3])
    _train_step(
        AudioEmotionBaseline(n_mels=16, hidden_dim=12),
        torch.randn(4, 10, 16), torch.tensor([10, 4, 7, 1]), labels, 7,
    )


def test_text_baseline_emits_seven_logits():
    labels = torch.tensor([2, 5, 0, 1])
    _train_step(
        TextEmotionBaseline(vocab_size=128, embedding_dim=16, hidden_dim=12),
        torch.randint(1, 128, (4, 9)), torch.tensor([9, 3, 1, 6]), labels, 7,
    )


@pytest.mark.parametrize("temporal", ["mean", "gru"])
def test_video_baseline_emits_seven_logits(temporal):
    labels = torch.tensor([4, 4, 1, 0])
    _train_step(
        VideoEmotionBaseline(frame_size=8, hidden_dim=12, temporal=temporal),
        torch.randn(4, 3, 3, 8, 8), torch.tensor([3, 2, 1, 3]), labels, 7,
    )


def test_physiology_baseline_emits_the_declared_state_logits():
    labels = torch.tensor([0, 1, 2, 1])
    _train_step(
        PhysiologyStateBaseline(feature_dim=20, hidden_dim=8, num_classes=3),
        torch.randn(4, 20), None, labels, WESAD_STATE_3CLASS.num_classes,
    )


def test_every_emotion_baseline_matches_the_canonical_class_count():
    for model in (
        AudioEmotionBaseline(n_mels=8, hidden_dim=8),
        TextEmotionBaseline(vocab_size=32, embedding_dim=8, hidden_dim=8),
        VideoEmotionBaseline(frame_size=4, hidden_dim=8),
    ):
        assert model.classifier.out_features == EMOTION_7CLASS.num_classes


def test_confusion_matrix_is_square_in_the_declared_class_space():
    logits = AudioEmotionBaseline(n_mels=8, hidden_dim=8)(
        torch.randn(6, 5, 8), torch.tensor([5, 5, 5, 2, 1, 3])
    )
    metrics = classification_metrics(logits.argmax(1), torch.tensor([0, 1, 2, 3, 4, 5]), 7)
    assert len(metrics["confusion_matrix"]) == 7
    assert all(len(row) == 7 for row in metrics["confusion_matrix"])
    assert len(metrics["per_class_f1"]) == 7


# ============================================================
# Guard rails
# ============================================================

def test_degenerate_geometry_is_rejected():
    with pytest.raises(ValueError):
        AudioEmotionBaseline(n_mels=0)
    with pytest.raises(ValueError):
        TextEmotionBaseline(vocab_size=1)
    with pytest.raises(ValueError):
        VideoEmotionBaseline(frame_size=0)
    with pytest.raises(ValueError, match="temporal"):
        VideoEmotionBaseline(temporal="attention")
    with pytest.raises(ValueError):
        PhysiologyStateBaseline(feature_dim=0)


def test_text_padding_index_embedding_stays_zero():
    model = TextEmotionBaseline(vocab_size=32, embedding_dim=4, hidden_dim=4)
    assert torch.allclose(model.embedding.weight[0], torch.zeros(4))


def test_models_stay_small_enough_for_a_cpu_baseline():
    budgets = {
        AudioEmotionBaseline(): 200_000,
        TextEmotionBaseline(): 4_500_000,   # dominated by the embedding table
        VideoEmotionBaseline(): 2_000_000,
        PhysiologyStateBaseline(feature_dim=56): 20_000,
    }
    for model, budget in budgets.items():
        assert sum(p.numel() for p in model.parameters()) < budget, type(model).__name__
