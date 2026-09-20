"""Classification evaluator for categorical emotion / state experiments.

Parallel to the existing regression ``Evaluator`` but accumulates predictions
and targets over the full dataset, then computes all classification metrics
including per-class precision/recall/F1 and the confusion matrix.

Per-class arrays and the confusion matrix are indexed by the label space's
declared class order -- never an alphabetical one.  See ``src.common.labels``.
"""

import torch
import torch.nn as nn

from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics


#: Canonical identifier to class name for the default 7-class emotion task.
#: Kept as a module constant for backward compatibility; the ordering comes
#: from ``src.common.labels`` so it can never drift.
CLASS_NAMES = dict(enumerate(CANONICAL_EMOTION_CLASSES))


class ClassificationEvaluator:
    """Global evaluator for classification tasks.

    Predictions and targets are accumulated across the complete dataset
    before metrics are calculated, matching the regression ``Evaluator``
    contract but producing classification-specific results.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int = 7,
        input_key: str = "image",
        lengths_key: str | None = None,
        class_names: tuple[str, ...] | list[str] | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.input_key = input_key
        self.lengths_key = lengths_key
        names = tuple(class_names) if class_names is not None else CANONICAL_EMOTION_CLASSES
        if class_names is not None and len(names) != num_classes:
            raise ValueError(
                f"class_names has {len(names)} entries but num_classes is {num_classes}"
            )
        self.class_names = names

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        features = batch[self.input_key].to(self.device)
        label = batch["label"].to(self.device)
        if self.lengths_key and batch.get(self.lengths_key) is not None:
            logits = self.model(features, batch[self.lengths_key].to(self.device))
        else:
            logits = self.model(features)
        return logits, label

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            logits, label = self._forward(batch)
            loss = self.criterion(logits, label)

            batch_size = label.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            all_preds.append(logits.argmax(dim=1).detach().cpu())
            all_labels.append(label.detach().cpu())

        if total_samples == 0:
            raise RuntimeError("Evaluation loader contains no samples.")

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        metrics = classification_metrics(preds, labels, self.num_classes)

        metrics["loss"] = total_loss / total_samples
        metrics["samples"] = total_samples
        metrics["class_order"] = list(self.class_names[: self.num_classes])

        # Named per-class metrics, in the declared class order.
        per_class = {}
        for class_id, class_name in enumerate(self.class_names):
            if class_id < self.num_classes:
                per_class[class_name] = {
                    "precision": metrics["per_class_precision"][class_id],
                    "recall": metrics["per_class_recall"][class_id],
                    "f1": metrics["per_class_f1"][class_id],
                    "support": metrics["support"][class_id],
                }
        metrics["per_class"] = per_class

        return metrics
