import torch
import torch.nn as nn
from src.training.classification_evaluator import ClassificationEvaluator

def test_classification_evaluator():
    device = torch.device("cpu")
    model = nn.Linear(10, 7).to(device)
    criterion = nn.CrossEntropyLoss()
    
    evaluator = ClassificationEvaluator(
        model=model,
        criterion=criterion,
        device=device,
        num_classes=7
    )
    
    dummy_loader = [
        {"image": torch.randn(4, 10), "label": torch.tensor([0, 1, 2, 6])},
        {"image": torch.randn(4, 10), "label": torch.tensor([3, 4, 5, 0])}
    ]
    
    metrics = evaluator.evaluate(dummy_loader)
    
    assert metrics["samples"] == 8
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 7
    assert "per_class" in metrics
    assert len(metrics["per_class"]) == 7
    assert "neutral" in metrics["per_class"]
