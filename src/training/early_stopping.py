"""Patience-based early stopping for validation-driven model selection.

The policy is deliberately conservative: a single worse epoch never stops
training.  Training stops only once ``patience`` consecutive epochs fail to
improve the monitored metric by more than ``min_delta``, and never before
``min_epochs`` have run.  When those non-improving epochs also form a
monotonic decline the stop is reported as degradation rather than a plateau,
which keeps the run summary honest about *why* it stopped.

Only validation metrics are ever passed here; test metrics are out of reach by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MODES = ("max", "min")


@dataclass(frozen=True)
class EarlyStoppingState:
    """Outcome of one ``update`` call."""

    epoch: int
    value: float
    improved: bool
    should_stop: bool
    best_value: float
    best_epoch: int
    epochs_without_improvement: int
    reason: str | None


class EarlyStopping:
    """Track the monitored metric and decide when further epochs are wasteful."""

    def __init__(
        self,
        patience: int = 2,
        min_delta: float = 1e-4,
        mode: Literal["max", "min"] = "max",
        monitor: str = "val_macro_f1",
        min_epochs: int = 2,
    ):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if min_epochs < 1:
            raise ValueError("min_epochs must be at least 1")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.monitor = monitor
        self.min_epochs = min_epochs

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.stopped_epoch: int | None = None
        self.reason: str | None = None
        self.history: list[float] = []

    # ---------------------------------------------------------------- logic

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def _degrading(self) -> bool:
        """True when the recent non-improving epochs form a monotonic decline."""
        window = self.history[-(self.epochs_without_improvement + 1):]
        if len(window) < 2:
            return False
        if self.mode == "max":
            return all(later < earlier for earlier, later in zip(window, window[1:]))
        return all(later > earlier for earlier, later in zip(window, window[1:]))

    def update(self, value: float, epoch: int) -> EarlyStoppingState:
        """Record one epoch's validation metric and decide whether to stop."""
        value = float(value)
        improved = self._is_improvement(value)
        self.history.append(value)

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        should_stop = (
            self.epochs_without_improvement >= self.patience and epoch >= self.min_epochs
        )
        reason = None
        if should_stop:
            self.stopped_epoch = epoch
            trend = "degraded" if self._degrading() else "did not improve"
            reason = (
                f"early stopping at epoch {epoch}: {self.monitor} {trend} for "
                f"{self.epochs_without_improvement} consecutive epochs "
                f"(patience={self.patience}, min_delta={self.min_delta:g}); "
                f"best {self.monitor}={self.best_value:.6f} at epoch {self.best_epoch}"
            )
            self.reason = reason

        return EarlyStoppingState(
            epoch=epoch,
            value=value,
            improved=improved,
            should_stop=should_stop,
            best_value=self.best_value,
            best_epoch=self.best_epoch,
            epochs_without_improvement=self.epochs_without_improvement,
            reason=reason,
        )

    def completed(self, epochs_run: int, max_epochs: int) -> str:
        """Record and return the terminal reason when the loop finishes."""
        if self.reason is None:
            self.reason = (
                f"completed the configured maximum of {max_epochs} epochs "
                f"({epochs_run} run); best {self.monitor}={self.best_value:.6f} "
                f"at epoch {self.best_epoch}"
            )
        return self.reason

    # -------------------------------------------------------------- summary

    def summary(self) -> dict:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_epochs": self.min_epochs,
            "best_value": self.best_value if self.history else None,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_early": self.stopped_epoch is not None,
            "stopped_epoch": self.stopped_epoch,
            "reason": self.reason,
            "monitored_history": list(self.history),
        }
