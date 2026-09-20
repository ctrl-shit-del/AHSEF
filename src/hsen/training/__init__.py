"""One training loop, one loss, one metric set -- shared by both execution profiles.

Section 5 of the phase brief asks that the training logic not be duplicated
between the CUDA and CPU scripts, and section 23 asks that the two build an
identical model.  Everything that could differ between them lives in
:class:`~src.hsen.training.hsen_trainer.TrainerConfig`; everything that must not
lives here.
"""

from src.hsen.training.checkpoint import HSENCheckpointManager, RunRecord
from src.hsen.training.hsen_trainer import (
    HSENTrainer,
    TrainerConfig,
    assert_same_architecture,
    resolve_device,
    set_seeds,
)
from src.hsen.training.losses import HSENLoss, LossConfig
from src.hsen.training.metrics import evaluate_predictions, primary_metric_name

__all__ = [
    "HSENTrainer", "TrainerConfig", "HSENLoss", "LossConfig",
    "HSENCheckpointManager", "RunRecord",
    "evaluate_predictions", "primary_metric_name",
    "assert_same_architecture", "resolve_device", "set_seeds",
]
