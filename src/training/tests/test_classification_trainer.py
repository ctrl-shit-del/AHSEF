import torch
import torch.nn as nn
from src.training.classification_trainer import ClassificationTrainer, ClassificationEpochResult

def test_classification_trainer_one_epoch():
    device = torch.device("cpu")
    model = nn.Linear(10, 7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    trainer = ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_classes=7
    )
    
    # Create a dummy dataloader (list of dicts)
    dummy_loader = [
        {"image": torch.randn(4, 10), "label": torch.tensor([0, 1, 2, 6])},
        {"image": torch.randn(4, 10), "label": torch.tensor([3, 4, 5, 0])}
    ]
    
    # Train
    train_result = trainer.train_one_epoch(dummy_loader)
    
    assert isinstance(train_result, ClassificationEpochResult)
    assert train_result.loss > 0
    assert 0 <= train_result.accuracy <= 1.0
    assert 0 <= train_result.macro_f1 <= 1.0
    
    # Validate
    val_result = trainer.validate_one_epoch(dummy_loader)
    assert isinstance(val_result, ClassificationEpochResult)
    assert val_result.loss > 0
    assert 0 <= val_result.accuracy <= 1.0
    assert 0 <= val_result.macro_f1 <= 1.0
