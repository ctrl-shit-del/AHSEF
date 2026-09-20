from dataclasses import dataclass

import torch
import torch.nn as nn

from src.training.metrics import (
    regression_metrics,
)


@dataclass
class EpochResult:
    """
    Results returned by one training or validation epoch.
    """

    loss: float
    mae: float
    rmse: float


class Trainer:
    """
    Generic regression trainer.

    Designed initially for CMU-MOSEI sentiment regression,
    but kept independent of the dataset implementation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def _forward(
        self,
        batch: dict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Move inputs to device and perform model forward pass.
        """

        audio = batch["audio"].to(
            self.device
        )

        vision = batch["vision"].to(
            self.device
        )

        text = batch["text"].to(
            self.device
        )

        target = batch[
            "sentiment_score"
        ].to(self.device)

        prediction = self.model(
            audio=audio,
            vision=vision,
            text=text,
        )

        return prediction, target

    def train_one_epoch(
        self,
        loader,
    ) -> EpochResult:
        """
        Train the model for one complete epoch.
        """

        self.model.train()

        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0

        batches = 0

        for batch in loader:

            self.optimizer.zero_grad(
                set_to_none=True
            )

            prediction, target = (
                self._forward(batch)
            )

            loss = self.criterion(
                prediction,
                target,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            loss.backward()

            self.optimizer.step()

            metrics = regression_metrics(
                prediction,
                target,
            )

            total_loss += float(
                loss.item()
            )

            total_mae += metrics["mae"]
            total_rmse += metrics["rmse"]

            batches += 1

        if batches == 0:
            raise RuntimeError(
                "Training loader contains no batches."
            )

        return EpochResult(
            loss=total_loss / batches,
            mae=total_mae / batches,
            rmse=total_rmse / batches,
        )

    @torch.no_grad()
    def validate_one_epoch(
        self,
        loader,
    ) -> EpochResult:
        """
        Evaluate the model for one complete epoch.
        """

        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0

        batches = 0

        for batch in loader:

            prediction, target = (
                self._forward(batch)
            )

            loss = self.criterion(
                prediction,
                target,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation loss."
                )

            metrics = regression_metrics(
                prediction,
                target,
            )

            total_loss += float(
                loss.item()
            )

            total_mae += metrics["mae"]
            total_rmse += metrics["rmse"]

            batches += 1

        if batches == 0:
            raise RuntimeError(
                "Validation loader contains no batches."
            )

        return EpochResult(
            loss=total_loss / batches,
            mae=total_mae / batches,
            rmse=total_rmse / batches,
        )

    def fit_one_epoch(
        self,
        train_loader,
        validation_loader,
    ) -> tuple[
        EpochResult,
        EpochResult,
    ]:
        """
        Run one training epoch followed by validation.
        """

        train_result = (
            self.train_one_epoch(
                train_loader
            )
        )

        validation_result = (
            self.validate_one_epoch(
                validation_loader
            )
        )

        return (
            train_result,
            validation_result,
        )