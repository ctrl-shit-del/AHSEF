"""Classification trainer for categorical emotion / state experiments.

Parallel to the existing regression ``Trainer`` but produces classification
epoch results with loss, accuracy, and macro-F1.

The trainer is modality-agnostic: it reads one named tensor out of the batch
(``input_key``) and, when the modality produces variable-length sequences, the
matching lengths tensor (``lengths_key``).  Defaults preserve the original
image-only contract exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.training.metrics import classification_metrics


@dataclass
class ClassificationEpochResult:
    """Results returned by one training or validation epoch."""

    loss: float
    accuracy: float
    macro_f1: float


class ClassificationTrainer:
    """Single-modality categorical classification trainer.

    The batch contract is the dict produced by the modality dataloaders:

        <input_key>            : [B, ...]  float32
        <input_key>_lengths    : [B]       long    (sequence modalities only)
        label                  : [B]       long
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int = 7,
        input_key: str = "image",
        lengths_key: str | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.input_key = input_key
        self.lengths_key = lengths_key

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        features = batch[self.input_key].to(self.device)
        label = batch["label"].to(self.device)
        if self.lengths_key and batch.get(self.lengths_key) is not None:
            logits = self.model(features, batch[self.lengths_key].to(self.device))
        else:
            logits = self.model(features)
        return logits, label

    def train_one_epoch(self, loader) -> ClassificationEpochResult:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        batches = 0

        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            logits, label = self._forward(batch)
            loss = self.criterion(logits, label)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(label.detach().cpu())
            batches += 1

        if batches == 0:
            raise RuntimeError("Training loader contains no batches.")

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        metrics = classification_metrics(preds, labels, self.num_classes)

        return ClassificationEpochResult(
            loss=total_loss / batches,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
        )

    @torch.no_grad()
    def validate_one_epoch(self, loader) -> ClassificationEpochResult:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        batches = 0

        for batch in loader:
            logits, label = self._forward(batch)
            loss = self.criterion(logits, label)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")

            total_loss += float(loss.item())
            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(label.detach().cpu())
            batches += 1

        if batches == 0:
            raise RuntimeError("Validation loader contains no batches.")

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        metrics = classification_metrics(preds, labels, self.num_classes)

        return ClassificationEpochResult(
            loss=total_loss / batches,
            accuracy=metrics["accuracy"],
            macro_f1=metrics["macro_f1"],
        )
