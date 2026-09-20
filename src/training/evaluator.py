import torch
import torch.nn as nn

from src.training.metrics import (
    regression_metrics,
)


class Evaluator:
    """
    Global evaluator for regression tasks.

    Predictions and targets are accumulated across the
    complete dataset before metrics are calculated.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
    ):

        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader,
    ) -> dict[str, float]:

        self.model.eval()

        predictions = []
        targets = []

        total_loss = 0.0
        total_samples = 0

        for batch in loader:

            audio = batch[
                "audio"
            ].to(self.device)

            vision = batch[
                "vision"
            ].to(self.device)

            text = batch[
                "text"
            ].to(self.device)

            target = batch[
                "sentiment_score"
            ].to(self.device)

            prediction = self.model(
                audio=audio,
                vision=vision,
                text=text,
            )

            loss = self.criterion(
                prediction,
                target,
            )

            batch_size = (
                target.shape[0]
            )

            total_loss += (
                float(loss.item())
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions.append(
                prediction.detach()
            )

            targets.append(
                target.detach()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Evaluation loader contains no samples."
            )

        predictions = torch.cat(
            predictions,
            dim=0,
        )

        targets = torch.cat(
            targets,
            dim=0,
        )

        metrics = regression_metrics(
            predictions,
            targets,
        )

        metrics["loss"] = (
            total_loss /
            total_samples
        )

        metrics["samples"] = (
            total_samples
        )

        return metrics