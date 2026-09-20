"""Reusable single-modality baseline experiment runner.

Every modality baseline in this project follows the same protocol, and that
protocol -- not the architecture -- is what makes the five baselines
comparable.  It lives here exactly once:

* Class weights are computed from the **training** partition only.
* The best checkpoint is selected by **validation** macro-F1; early stopping
  monitors the same metric.
* The **test** partition is not opened until training has finished and the
  best checkpoint has been reloaded.  ``loader_events`` in the run summary is
  the audit trail for that ordering.
* Every iteration owns its own directory, log file, config, and checkpoints;
  nothing an earlier iteration wrote is reopened for writing.
* Class order comes from a declared :class:`~src.common.labels.LabelSpace`,
  so the confusion matrix, per-class arrays, class weights, and logs all index
  the same positions.

A modality subclass supplies only what is genuinely modality-specific: how to
build a dataloader, how to build a model, and how to describe that model.  The
image baseline (``src.training.image_experiment``) is the reference
implementation and its artefact layout is the compatibility contract.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import LabelSpace, get_label_space
from src.training.checkpoint import CheckpointManager
from src.training.classification_evaluator import ClassificationEvaluator
from src.training.classification_trainer import ClassificationTrainer
from src.training.early_stopping import EarlyStopping
from src.training.experiment_logging import (
    close_logger,
    create_iteration_logger,
    log_mapping,
    log_section,
)
from src.training.experiment_summary import write_experiment_summary
from src.training.metrics import class_weights


MONITOR = "macro_f1"

#: Bounded overrides that exercise the whole pipeline in seconds on CPU.
DEBUG_OVERRIDES: dict[str, Any] = {
    "epochs": 2,
    "batch_size": 16,
    "max_train": 256,
    "max_val": 128,
    "max_test": 128,
    "patience": 1,
    "min_epochs": 1,
}


# ============================================================
# Configuration
# ============================================================

@dataclass
class BaseExperimentConfig:
    """Protocol-level configuration shared by every modality baseline."""

    experiment_name: str = "image_25pct"
    description: str = "Single-modality categorical baseline."
    iteration: str = "1"
    experiment_root: str = "experiments"
    metadata_dir: str | None = None

    # data / task
    modality: str = "image"
    task: str = "emotion_7class"
    num_classes: int = 7

    # optimisation
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 5
    optimizer: str = "AdamW"
    loss: str = "CrossEntropyLoss"
    class_weight_policy: str = "inverse_frequency_from_train"

    # early stopping
    patience: int = 2
    min_delta: float = 1e-4
    min_epochs: int = 2

    # reproducibility / infrastructure
    seed: int = 42
    run_seed: int | None = None
    device: str = "auto"
    num_workers: int = 0

    # bounded / debug operation
    max_train: int | None = None
    max_val: int | None = None
    max_test: int | None = None
    debug: bool = False

    #: Finish after validation and leave the test partition unopened.
    #:
    #: Default False, so every baseline keeps the original protocol: test is
    #: opened exactly once, after model selection. It exists for experiments
    #: whose evaluation split must stay shut until a downstream configuration is
    #: frozen -- opening it "just to see" is precisely what a locked split
    #: forbids. A deferred run records ``test_deferred: true`` and carries no
    #: test metrics at all, so its artefact cannot be mistaken for a complete one.
    defer_test: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.iteration = ExperimentLayout.iteration_label(self.iteration)
        # Fails fast on an unknown task rather than at checkpoint-writing time.
        space = get_label_space(self.task)
        if self.num_classes != space.num_classes:
            raise ValueError(
                f"num_classes={self.num_classes} contradicts label space "
                f"{space.name!r} which declares {space.num_classes} classes "
                f"{list(space.classes)}"
            )

    @property
    def label_space(self) -> LabelSpace:
        return get_label_space(self.task)

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.label_space.classes

    @property
    def effective_run_seed(self) -> int:
        """Seed governing model init and batch order.

        Defaults to ``seed`` so an iteration is byte-reproducible; pass a
        different ``run_seed`` to obtain an independent repeat over the same
        data split.
        """
        return self.seed if self.run_seed is None else self.run_seed


# ============================================================
# Reproducibility helpers
# ============================================================

def set_seeds(seed: int) -> dict[str, Any]:
    """Seed every RNG used by the run and report what could be made exact."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    notes: list[str] = []
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as error:  # pragma: no cover - depends on the torch build
        notes.append(f"torch.use_deterministic_algorithms unavailable: {error}")
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - CPU-only builds
        notes.append("cuDNN determinism flags unavailable on this build")
    return {
        "python_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "deterministic_algorithms": "warn_only",
        "notes": notes,
    }


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    return torch.device(name)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================
# Runner
# ============================================================

class BaseExperimentRunner:
    """Execute one reproducible iteration of a single-modality baseline."""

    #: Key under which the modality tensor arrives in a batch.
    INPUT_KEY: str = "image"
    #: Key carrying per-sample sequence lengths, for padded sequence modalities.
    LENGTHS_KEY: str | None = None
    #: Manifest column holding the integer target.
    LABEL_COLUMN: str = "canonical_emotion_id"

    def __init__(self, config: BaseExperimentConfig):
        self.config = config
        self.layout = ExperimentLayout.from_name(
            config.experiment_name, Path(config.experiment_root)
        )
        self.iteration = config.iteration
        self.metadata_dir = (
            Path(config.metadata_dir) if config.metadata_dir else self.layout.metadata_dir
        )
        self.label_space = config.label_space
        self.class_names = self.label_space.classes
        # Audit trail proving the test partition is untouched during training.
        self.loader_events: list[dict[str, str]] = []
        self._phase = "init"

    # ------------------------------------------------ modality extension points

    def build_loader(self, split: str, shuffle: bool, max_samples: int | None):
        """Return a dataloader for ``split``.  Implemented per modality."""
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        """Return a freshly initialised model.  Implemented per modality."""
        raise NotImplementedError

    def model_record(self, model: nn.Module) -> dict:
        """Modality-specific architecture fields recorded in the artefacts."""
        return {}

    def prepare(self, train_loader, val_loader, logger) -> dict:
        """Hook run after the training/validation loaders exist.

        Returns a JSON-serialisable record folded into ``run_summary`` and
        ``config.json``.  Anything derived from data here must come from the
        training split alone; the physiology baseline uses it to fit feature
        normalisation.
        """
        return {}

    def extra_summary(self) -> dict:
        """Modality-specific fields folded into ``run_summary.json``."""
        return {}

    # ----------------------------------------------------------- data access

    def manifest_path(self, split: str) -> Path:
        return self.metadata_dir / f"{split}.parquet"

    def _loader(self, split: str, shuffle: bool, max_samples: int | None):
        manifest = self.manifest_path(split)
        if not manifest.exists():
            raise FileNotFoundError(
                f"Experiment metadata missing: {manifest}. "
                f"Generate it first with src.preprocessing.sampling.run_sampler."
            )
        self.loader_events.append({"phase": self._phase, "split": split})
        return self.build_loader(split, shuffle, max_samples)

    def _manifest_of(self, loader):
        manifest = getattr(loader.dataset, "manifest", None)
        if manifest is None:
            raise AttributeError(
                f"{type(loader.dataset).__name__} must expose a 'manifest' DataFrame "
                f"so class weights and dataset counts can be derived from metadata."
            )
        return manifest

    def train_labels(self, loader) -> torch.Tensor:
        manifest = self._manifest_of(loader)
        if self.LABEL_COLUMN not in manifest.columns:
            raise ValueError(
                f"Training manifest is missing the label column {self.LABEL_COLUMN!r}"
            )
        return torch.tensor(manifest[self.LABEL_COLUMN].astype(int).tolist(), dtype=torch.long)

    def dataset_counts(self, loader) -> dict[str, int]:
        manifest = self._manifest_of(loader)
        if "dataset" not in manifest.columns:
            return {}
        counts = manifest["dataset"].value_counts().sort_index().to_dict()
        return {str(key): int(value) for key, value in counts.items()}

    # ------------------------------------------------------------------- run

    def run(self) -> dict:
        config = self.config
        self.layout.prepare_iteration(self.iteration)
        results_dir = self.layout.results_dir(self.iteration)
        logger = create_iteration_logger(
            f"{self.layout.name}.iteration_{self.iteration}",
            self.layout.log_path(self.iteration),
        )
        started_wall = time.time()

        try:
            log_section(logger, f"{self.layout.name} | iteration {self.iteration}")
            logger.info("Description        : %s", config.description)
            logger.info("Modality           : %s", config.modality)
            logger.info("Task / label space : %s", self.label_space.name)
            logger.info("Class order        : %s", ", ".join(self.class_names))
            if self.label_space.limitation:
                logger.warning("Task limitation    : %s", self.label_space.limitation)
            logger.info("Started            : %s", time.strftime("%Y-%m-%dT%H:%M:%S"))
            logger.info("Metadata directory : %s", self.metadata_dir)
            logger.info("Iteration directory: %s", self.layout.iteration_dir(self.iteration))
            logger.info("Debug mode         : %s", config.debug)

            seed_record = set_seeds(config.effective_run_seed)
            device = resolve_device(config.device)
            logger.info("Device             : %s", device)
            log_mapping(logger, "Seeds", seed_record)

            sampling_summary = self._log_sampling_summary(logger)

            # ---------------------------------------------- training inputs
            self._phase = "setup"
            train_loader = self._loader("train", shuffle=True, max_samples=config.max_train)
            val_loader = self._loader("validation", shuffle=False, max_samples=config.max_val)

            preparation = self.prepare(train_loader, val_loader, logger) or {}

            weights, weights_record, distribution = self._class_weights(train_loader)
            save_json(results_dir / "class_weights.json", weights_record)

            dataset_counts = self.dataset_counts(train_loader)
            logger.info("Train samples      : %d", len(train_loader.dataset))
            logger.info("Validation samples : %d", len(val_loader.dataset))
            log_mapping(logger, "Train dataset counts", dataset_counts)
            log_mapping(logger, "Train class counts", weights_record["training_distribution"])
            log_mapping(
                logger, "Class weights (train only)",
                {name: round(weights[index].item(), 6) for index, name in enumerate(self.class_names)},
            )

            # ------------------------------------------------------- model
            model = self.build_model().to(device)
            parameters = sum(p.numel() for p in model.parameters())
            logger.info("Model architecture :\n%s", model)
            logger.info("Parameters         : %d", parameters)

            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )

            config_record = self._config_record(
                device, parameters, seed_record,
                len(train_loader.dataset), len(val_loader.dataset), preparation,
            )
            save_json(self.layout.config_path(self.iteration), config_record)
            log_mapping(logger, "Hyperparameters", self._hyperparameter_record())

            # ------------------------------------------------------ training
            trainer = ClassificationTrainer(
                model=model, optimizer=optimizer, criterion=criterion,
                device=device, num_classes=config.num_classes,
                input_key=self.INPUT_KEY, lengths_key=self.LENGTHS_KEY,
            )
            checkpoints = CheckpointManager(
                directory=self.layout.checkpoints_dir(self.iteration), monitor=MONITOR, mode="max",
            )
            stopper = EarlyStopping(
                patience=config.patience, min_delta=config.min_delta, mode="max",
                monitor="val_macro_f1", min_epochs=config.min_epochs,
            )
            checkpoint_extra = {
                "experiment": self.layout.name,
                "iteration": self.iteration,
                "modality": config.modality,
                "description": config.description,
                "config": asdict(config),
                "task": self.label_space.name,
                "class_order": list(self.class_names),
                "class_mapping": dict(self.label_space.mapping),
                "class_weights": weights.tolist(),
                "class_weight_policy": config.class_weight_policy,
                "metadata_dir": str(self.metadata_dir),
            }
            if preparation:
                checkpoint_extra["preparation"] = preparation

            self._phase = "training"
            log_section(logger, "TRAINING")
            history: list[dict] = []
            checkpoint_updates: list[int] = []
            training_started = time.monotonic()

            for epoch in range(1, config.epochs + 1):
                epoch_started = time.monotonic()
                train_result = trainer.train_one_epoch(train_loader)
                val_result = trainer.validate_one_epoch(val_loader)
                epoch_seconds = time.monotonic() - epoch_started

                _, improved = checkpoints.save_epoch(
                    model=model, optimizer=optimizer, epoch=epoch,
                    train_result=train_result, validation_result=val_result,
                    extra=checkpoint_extra,
                )
                if improved:
                    checkpoint_updates.append(epoch)
                state = stopper.update(val_result.macro_f1, epoch)

                history.append({
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "train_accuracy": train_result.accuracy,
                    "train_macro_f1": train_result.macro_f1,
                    "val_loss": val_result.loss,
                    "val_accuracy": val_result.accuracy,
                    "val_macro_f1": val_result.macro_f1,
                    "best_val_macro_f1": state.best_value,
                    "best_epoch": state.best_epoch,
                    "checkpoint_updated": improved,
                    "epochs_without_improvement": state.epochs_without_improvement,
                    "epoch_seconds": round(epoch_seconds, 3),
                })

                logger.info(
                    "Epoch %d/%d | train loss=%.6f acc=%.4f macroF1=%.4f | "
                    "val loss=%.6f acc=%.4f macroF1=%.4f | %.1fs",
                    epoch, config.epochs,
                    train_result.loss, train_result.accuracy, train_result.macro_f1,
                    val_result.loss, val_result.accuracy, val_result.macro_f1,
                    epoch_seconds,
                )
                logger.info(
                    "  checkpoint=%s | best epoch=%d best val macroF1=%.6f | "
                    "epochs without improvement=%d/%d",
                    "BEST UPDATED" if improved else "best unchanged",
                    state.best_epoch, state.best_value,
                    state.epochs_without_improvement, config.patience,
                )
                if state.should_stop:
                    logger.info("  %s", state.reason)
                    break

            training_seconds = time.monotonic() - training_started
            stop_reason = stopper.completed(len(history), config.epochs)
            logger.info("Training finished  : %s", stop_reason)
            save_json(results_dir / "training_history.json", history)

            # ---------------------------------------- best checkpoint reload
            log_section(logger, "BEST CHECKPOINT")
            best_path = self.layout.best_checkpoint(self.iteration)
            checkpoint = CheckpointManager.load(best_path, model=model, optimizer=None, device=device)
            logger.info("Loaded             : %s", best_path)
            logger.info("Best epoch         : %s", checkpoint["epoch"])
            logger.info("Best val %-9s : %.6f", MONITOR, checkpoint["monitor_value"])
            reload_ok = self._verify_reload(best_path, device, model)
            logger.info("Reload verified    : %s", reload_ok)

            # ------------------------------------------------- final metrics
            evaluator = ClassificationEvaluator(
                model=model, criterion=criterion, device=device,
                num_classes=config.num_classes,
                input_key=self.INPUT_KEY, lengths_key=self.LENGTHS_KEY,
                class_names=self.class_names,
            )
            log_section(logger, "VALIDATION EVALUATION (best checkpoint)")
            validation_metrics = evaluator.evaluate(val_loader)
            self._log_metrics(logger, validation_metrics)

            # The test partition is opened only now, after model selection --
            # unless this run was configured to leave it shut entirely.
            self._phase = "evaluation"
            test_loader = None
            test_metrics = None
            if config.defer_test:
                log_section(logger, "TEST EVALUATION DEFERRED")
                logger.info("Test partition     : NOT OPENED (defer_test=True)")
                logger.info(
                    "Reason             : this experiment's test split stays shut "
                    "until a downstream configuration is frozen"
                )
            else:
                log_section(logger, "TEST EVALUATION (best checkpoint, single pass)")
                test_loader = self._loader("test", shuffle=False, max_samples=config.max_test)
                logger.info("Test samples       : %d", len(test_loader.dataset))
                test_metrics = evaluator.evaluate(test_loader)
                self._log_metrics(logger, test_metrics)

            save_json(results_dir / "validation_metrics.json", validation_metrics)
            if test_metrics is not None:
                save_json(results_dir / "test_metrics.json", test_metrics)
            save_json(results_dir / "confusion_matrix.json", {
                "class_order": list(self.class_names),
                "class_mapping": dict(self.label_space.mapping),
                "validation": validation_metrics["confusion_matrix"],
                "test": test_metrics["confusion_matrix"] if test_metrics else None,
            })

            total_seconds = time.time() - started_wall
            run_summary = {
                "experiment": self.layout.name,
                "iteration": self.iteration,
                "modality": config.modality,
                "task": self.label_space.name,
                "class_order": list(self.class_names),
                "task_limitation": self.label_space.limitation,
                "description": config.description,
                "debug": config.debug,
                "metadata_dir": str(self.metadata_dir),
                "config": asdict(config),
                "seeds": seed_record,
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "python_version": sys.version,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "model": {
                    "class": type(model).__name__,
                    "parameters": parameters,
                    "num_classes": config.num_classes,
                    **self.model_record(model),
                },
                "samples": {
                    "train": len(train_loader.dataset),
                    "validation": len(val_loader.dataset),
                    "test": len(test_loader.dataset) if test_loader else None,
                },
                "train_dataset_counts": dataset_counts,
                "class_weights": weights_record,
                "epochs_configured": config.epochs,
                "epochs_completed": len(history),
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_macro_f1": float(checkpoint["monitor_value"]),
                "checkpoint_updates": checkpoint_updates,
                "early_stopping": stopper.summary(),
                "stop_reason": stop_reason,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "test_deferred": bool(config.defer_test),
                "test_partition_opened": test_metrics is not None,
                "checkpoints": {
                    "best": str(best_path),
                    "last": str(self.layout.last_checkpoint(self.iteration)),
                    "reload_verified": reload_ok,
                },
                "training_seconds": round(training_seconds, 3),
                "total_seconds": round(total_seconds, 3),
                "loader_events": self.loader_events,
                "test_used_for_model_selection": False,
                "sampling_summary_present": sampling_summary is not None,
                "log_file": str(self.layout.log_path(self.iteration)),
            }
            if preparation:
                run_summary["preparation"] = preparation
            run_summary.update(self.extra_summary())
            save_json(results_dir / "run_summary.json", run_summary)

            log_section(logger, "ITERATION COMPLETE")
            logger.info("Best epoch         : %s", run_summary["best_epoch"])
            logger.info("Best val macro-F1  : %.6f", run_summary["best_val_macro_f1"])
            if test_metrics is not None:
                logger.info("Test accuracy      : %.6f", test_metrics["accuracy"])
                logger.info("Test macro-F1      : %.6f", test_metrics["macro_f1"])
                logger.info("Test weighted-F1   : %.6f", test_metrics["weighted_f1"])
            else:
                logger.info("Test               : DEFERRED (partition not opened)")
            logger.info("Total time         : %.1fs", total_seconds)
            logger.info("Results            : %s", results_dir)
            logger.info("Checkpoints        : %s", self.layout.checkpoints_dir(self.iteration))

            summary_path = write_experiment_summary(self.layout)
            logger.info("Experiment summary : %s", summary_path)
            return run_summary
        finally:
            close_logger(logger)

    # ------------------------------------------------------------- helpers

    def _class_weights(self, train_loader) -> tuple[torch.Tensor, dict, list[int]]:
        """Inverse-frequency weights derived from the training split alone."""
        config = self.config
        train_labels = self.train_labels(train_loader)
        weights = class_weights(train_labels, config.num_classes)
        distribution = torch.bincount(train_labels, minlength=config.num_classes).tolist()
        # A class absent from a bounded/debug training subset would receive
        # weight 0, which makes the weighted loss undefined for any batch
        # containing only that class.  Give it neutral weight and say so.
        absent = [index for index, count in enumerate(distribution) if count == 0]
        if absent:
            weights = weights.clone()
            for index in absent:
                weights[index] = 1.0
        record = {
            "policy": config.class_weight_policy,
            "computed_from": str(self.manifest_path("train")),
            "computed_from_split": "train",
            "uses_validation_labels": False,
            "uses_test_labels": False,
            "task": self.label_space.name,
            "label_column": self.LABEL_COLUMN,
            "class_order": list(self.class_names),
            "class_mapping": dict(self.label_space.mapping),
            "training_distribution": {
                name: distribution[index] for index, name in enumerate(self.class_names)
            },
            "classes_absent_from_train": [self.class_names[index] for index in absent],
            "absent_class_weight": 1.0 if absent else None,
            "weights": weights.tolist(),
        }
        return weights, record, distribution

    def _hyperparameter_record(self) -> dict:
        config = self.config
        return {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "optimizer": config.optimizer,
            "loss": config.loss,
            "patience": config.patience,
            "min_delta": config.min_delta,
            "min_epochs": config.min_epochs,
            "seed": config.seed,
            "run_seed": config.effective_run_seed,
        }

    def _log_sampling_summary(self, logger) -> dict | None:
        path = self.metadata_dir / "sampling_summary.json"
        if not path.exists():
            logger.warning(
                "No sampling_summary.json in %s; sampling provenance unavailable",
                self.metadata_dir,
            )
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        try:
            log_section(logger, "SAMPLING PROVENANCE")
            logger.info("Sampling method    : %s", summary["sampling_method"])
            logger.info("Sampling seed      : %s", summary["seed"])
            logger.info("Eligible pool      : %s", summary["pool"]["eligible_records"])
            logger.info("Selected records   : %s", summary["selection"]["selected_records"])
            logger.info("Actual fraction    : %.6f", summary["selection"]["actual_fraction"])
            logger.info("Contributing sets  : %s", ", ".join(summary["contributing_datasets"]))
            log_mapping(logger, "Split counts", summary["splits"]["counts"])
            log_mapping(logger, "Selected dataset counts", summary["selection"]["dataset_counts"])
            log_mapping(logger, "Selected class counts", summary["selection"]["class_counts"])
        except (KeyError, TypeError) as error:
            logger.warning("sampling_summary.json is present but incomplete: %s", error)
        return summary

    def _config_record(
        self, device, parameters: int, seed_record: dict,
        train_size: int, val_size: int, preparation: dict,
    ) -> dict:
        record = {
            "experiment": self.layout.name,
            "iteration": self.iteration,
            "modality": self.config.modality,
            "task": self.label_space.name,
            "description": self.config.description,
            "config": asdict(self.config),
            "class_mapping": dict(self.label_space.mapping),
            "class_order": list(self.class_names),
            "metadata_dir": str(self.metadata_dir),
            "paths": {
                "iteration_dir": str(self.layout.iteration_dir(self.iteration)),
                "checkpoints": str(self.layout.checkpoints_dir(self.iteration)),
                "results": str(self.layout.results_dir(self.iteration)),
                "log": str(self.layout.log_path(self.iteration)),
            },
            "model_parameters": parameters,
            "device": str(device),
            "seeds": seed_record,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "samples": {"train": train_size, "validation": val_size},
            "monitor": {"metric": "validation macro_f1", "mode": "max"},
        }
        if preparation:
            record["preparation"] = preparation
        return record

    def _verify_reload(self, path: Path, device: torch.device, reference: nn.Module) -> bool:
        """Reload the best checkpoint into a fresh model and compare weights."""
        fresh = self.build_model().to(device)
        CheckpointManager.load(path, model=fresh, optimizer=None, device=device)
        left, right = reference.state_dict(), fresh.state_dict()
        if set(left) != set(right):
            return False
        return all(torch.equal(left[key], right[key]) for key in left)

    @staticmethod
    def _log_metrics(logger, metrics: dict) -> None:
        logger.info("Samples            : %d", metrics["samples"])
        logger.info("Loss               : %.6f", metrics["loss"])
        logger.info("Accuracy           : %.6f", metrics["accuracy"])
        logger.info("Macro precision    : %.6f", metrics["macro_precision"])
        logger.info("Macro recall       : %.6f", metrics["macro_recall"])
        logger.info("Macro F1           : %.6f", metrics["macro_f1"])
        logger.info("Weighted F1        : %.6f", metrics["weighted_f1"])
        logger.info("%-12s %9s %9s %9s %9s", "class", "precision", "recall", "f1", "support")
        for name, values in metrics.get("per_class", {}).items():
            logger.info(
                "%-12s %9.4f %9.4f %9.4f %9d",
                name, values["precision"], values["recall"], values["f1"], values["support"],
            )
