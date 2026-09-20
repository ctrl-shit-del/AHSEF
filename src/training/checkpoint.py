from pathlib import Path

import torch
import torch.nn as nn

from src.utils.io import ensure_dir


class CheckpointManager:
    """
    Save and load model training checkpoints.

    Two checkpoint types are maintained:

        best.pt
        last.pt

    `best.pt` is selected using the monitored validation metric.
    """

    def __init__(
        self,
        directory: str | Path,
        monitor: str = "mae",
        mode: str = "min",
    ):

        self.directory = Path(directory)

        ensure_dir(
            self.directory
        )

        if monitor not in {
            "loss",
            "mae",
            "rmse",
            "macro_f1",
            "accuracy",
        }:
            raise ValueError(
                "monitor must be one of: "
                "loss, mae, rmse, macro_f1, accuracy"
            )

        if mode not in {
            "min",
            "max",
        }:
            raise ValueError(
                "mode must be either 'min' or 'max'"
            )

        self.monitor = monitor
        self.mode = mode

        self.best_value = (
            float("inf")
            if mode == "min"
            else float("-inf")
        )

    def is_better(
        self,
        value: float,
    ) -> bool:

        if self.mode == "min":
            return value < self.best_value

        return value > self.best_value

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_result,
        validation_result,
        filename: str,
        extra: dict | None = None,
    ) -> Path:
        """
        Save a complete training checkpoint.
        """

        path = (
            self.directory /
            filename
        )

        state = {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train": {
                k: getattr(train_result, k)
                for k in ["loss", "mae", "rmse", "accuracy", "macro_f1"]
                if hasattr(train_result, k)
            },

            "validation": {
                k: getattr(validation_result, k)
                for k in ["loss", "mae", "rmse", "accuracy", "macro_f1"]
                if hasattr(validation_result, k)
            },

            "monitor": self.monitor,
            "monitor_value":
                getattr(
                    validation_result,
                    self.monitor,
                ),
        }

        if extra is not None:
            state["extra"] = extra

        torch.save(
            state,
            path,
        )

        return path

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_result,
        validation_result,
        extra: dict | None = None,
    ) -> tuple[Path, bool]:
        """
        Save last checkpoint and optionally update best checkpoint.
        """

        last_path = self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_result=train_result,
            validation_result=validation_result,
            filename="last.pt",
            extra=extra,
        )

        value = getattr(
            validation_result,
            self.monitor,
        )

        improved = self.is_better(
            value
        )

        if improved:

            self.best_value = value

            self.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_result=train_result,
                validation_result=validation_result,
                filename="best.pt",
                extra=extra,
            )

        return last_path, improved

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
    ) -> dict:
        """
        Load a checkpoint into model and optionally optimizer.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{path}"
            )

        # Checkpoints are produced by this repository and carry plain-Python
        # experiment metadata alongside tensors, so full unpickling is both
        # safe and required on the PyTorch versions that default
        # `weights_only` to True.
        try:
            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                path,
                map_location=device,
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict"
            in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        return checkpoint