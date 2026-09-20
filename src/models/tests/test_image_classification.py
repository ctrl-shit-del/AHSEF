import torch
import torch.nn as nn

from src.models.multimodal import ImageEmotionBaseline
from src.training.metrics import class_weights, classification_metrics


def test_image_classifier_training_step_and_metrics():
    model = ImageEmotionBaseline(image_size=8, hidden_dim=16)
    image = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 1, 6])
    logits = model(image)
    assert logits.shape == (4, 7)
    weights = class_weights(labels)
    assert weights.shape == (7,)
    loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    metrics = classification_metrics(logits.argmax(1), labels)
    assert len(metrics["confusion_matrix"]) == 7
    assert all(len(row) == 7 for row in metrics["confusion_matrix"])
    assert all(torch.isfinite(torch.tensor(value)) for value in (metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]))
