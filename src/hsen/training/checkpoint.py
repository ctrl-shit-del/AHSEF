"""Checkpoints that carry everything needed to rebuild the run that made them.

Section 15 lists what every run must record and section 16 lists what must be
saved.  Both are satisfied here rather than by a convention someone has to
remember: a checkpoint holds the model weights, the optimiser and scheduler
state, the full :class:`~src.hsen.models.hsen.HSENConfig`, the loss config, the
class weights, the run's reproducibility record, and the metrics that selected
it.

``best.pt`` is chosen by the **primary validation metric**, never by training
loss.  A checkpoint selected on training loss is selected on how well the model
memorised, which is precisely the thing validation exists to not measure.

The reproducibility record is written next to the checkpoint as plain JSON as
well as inside it, because a ``.pt`` that will not load -- a torch version
change, a refactored class -- should still leave behind a readable account of
what was run.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty() -> bool | None:
    """Whether the working tree had uncommitted changes when the run started.

    Recorded because a revision hash from a dirty tree does not identify the
    code that ran, and a result whose provenance is a hash that never existed is
    worse than one that admits it.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_record(device: torch.device) -> dict:
    record = {
        "device": str(device),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        record |= {
            "gpu_name": properties.name,
            "gpu_total_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
            "gpu_capability": f"{properties.major}.{properties.minor}",
            "gpu_count": torch.cuda.device_count(),
        }
    return record


@dataclass
class RunRecord:
    """The section-15 provenance block, saved beside and inside the checkpoint."""

    experiment: str
    dataset: str
    label_protocol: str
    split_policy: str
    data_fraction: float
    seed: int
    device: str
    profile: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    scheduler: str
    amp: bool
    modalities: list[str]
    fusion: str
    model_config: dict
    loss_config: dict
    parameter_counts: dict
    class_weights: list[float] | None = None
    primary_metric: str = "weighted_f1"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    git_revision: str | None = field(default_factory=git_revision)
    git_dirty: bool | None = field(default_factory=git_dirty)
    hardware: dict = field(default_factory=dict)
    dataset_counts: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HSENCheckpointManager:
    """Write ``best.pt`` and ``last.pt`` plus the run's readable artefacts."""

    def __init__(
        self,
        directory: Path | str,
        monitor: str = "weighted_f1",
        mode: str = "max",
        record: RunRecord | None = None,
    ):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.monitor, self.mode = monitor, mode
        self.record = record
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.directory / "last.pt"

    def is_better(self, value: float) -> bool:
        if value != value:  # NaN never wins; a metric that failed is not an improvement
            return False
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def save_run_record(self) -> Path | None:
        if self.record is None:
            return None
        path = self.directory / "run_config.json"
        path.write_text(
            json.dumps(self.record.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        epoch: int,
        metrics: dict,
        history: list[dict],
        filename: str,
        scaler: Any | None = None,
    ) -> Path:
        path = self.directory / filename
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "metrics": metrics,
            "history": history,
            "monitor": self.monitor,
            "mode": self.mode,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "run_record": self.record.to_dict() if self.record else None,
            # Saved explicitly so a checkpoint can be rebuilt without importing
            # the config dataclass, which is what makes an old checkpoint
            # survive a refactor of it.
            "model_config": self.record.model_config if self.record else None,
        }
        torch.save(payload, path)
        return path

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        epoch: int,
        metrics: dict,
        history: list[dict],
        scaler: Any | None = None,
    ) -> bool:
        """Always write ``last.pt``; write ``best.pt`` only on a real improvement.

        The comparison happens *before* either write, and that ordering matters.
        Writing ``last.pt`` first would stamp it with the previous epoch's
        ``best_value``, so a resumed run would restore a best that is one epoch
        stale and could then overwrite a genuinely better ``best.pt`` with a
        worse epoch -- silently, because "no improvement" is what an ordinary
        epoch reports.
        """
        value = float(metrics.get(self.monitor, float("nan")))
        improved = self.is_better(value)
        if improved:
            self.best_value, self.best_epoch = value, epoch

        self.save(model, optimizer, scheduler, epoch, metrics, history, "last.pt", scaler)
        if improved:
            self.save(model, optimizer, scheduler, epoch, metrics, history, "best.pt", scaler)
        return improved

    def load(self, path: Path | str, map_location: str | torch.device = "cpu") -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint at {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        path: Path | str | None = None,
        scaler: Any | None = None,
    ) -> tuple[int, list[dict]]:
        """Restore a run and return the epoch to continue from and its history."""
        payload = self.load(path or self.last_path)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict"):
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if scaler is not None and payload.get("scaler_state_dict"):
            scaler.load_state_dict(payload["scaler_state_dict"])
        # The best value is restored too. Without it a resumed run starts from
        # -inf and overwrites a better best.pt with the first epoch it sees.
        self.best_value = payload.get("best_value", self.best_value)
        self.best_epoch = payload.get("best_epoch", 0)
        return int(payload["epoch"]) + 1, list(payload.get("history", []))
