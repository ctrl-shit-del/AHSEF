"""The one HSEN training loop.  Both execution profiles call this and nothing else.

Section 5 asks that the training logic not be duplicated and section 23 asks
that the 25%-CPU model be the *same* model as the full-CUDA one.  Those are the
same requirement seen from two sides, and the enforcement here is structural:
``train_hsen_cuda.py`` and ``train_hsen_cpu25.py`` each build a
:class:`TrainerConfig` and call :meth:`HSENTrainer.run`.  Neither script contains
a model, a loss, an optimiser, a metric or an evaluation path of its own, so
there is nowhere for the two to drift apart.

What the profiles are allowed to differ in, and nothing else:

    device            cpu vs cuda
    data_fraction     0.25 vs 1.0
    batch_size        smaller vs larger
    num_workers       fewer vs more
    amp               off vs on
    epochs            optionally shorter

Mixed precision is the one place hardware touches numerics, and it touches only
the *precision* of the arithmetic -- the graph, the parameter shapes and the
initialisation are identical, which :func:`assert_same_architecture` checks
directly rather than trusting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.hsen.data import HSENDataset, build_dataloader
from src.hsen.models.hsen import HSENConfig, build_hsen
from src.hsen.training.checkpoint import HSENCheckpointManager, RunRecord, hardware_record
from src.hsen.training.losses import HSENLoss, LossConfig, balanced_class_weights
from src.hsen.training.metrics import evaluate_predictions, primary_metric_name
from src.hsen.training.runlock import RunLock
from src.training.early_stopping import EarlyStopping

SPLITS = ("train", "validation", "test")


@dataclass
class TrainerConfig:
    """Everything a run needs that is not the architecture itself."""

    experiment: str = "iemocap_erc6"
    dataset: str = "IEMOCAP"
    profile: str = "cpu25"
    output_root: Path = Path("results/hsen")

    # --- compute -------------------------------------------------------
    device: str = "cpu"
    amp: bool = False
    num_workers: int = 0

    # --- data ----------------------------------------------------------
    data_fraction: float = 1.0
    batch_size: int = 16
    eval_batch_size: int | None = None

    # --- schedule ------------------------------------------------------
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    patience: int = 5
    min_epochs: int = 3
    seed: int = 42
    #: Seed for the stratified training-fraction draw. ``None`` means ``seed``,
    #: which is what every run before Phase 15 did. The seed-variance study
    #: pins it so that changing ``seed`` changes initialisation, dropout and
    #: batch order but *not* which 1,062 utterances are trained on -- otherwise
    #: "seed variance" would silently include data-subset variance.
    subset_seed: int | None = None

    # --- bookkeeping ---------------------------------------------------
    log_every: int = 50
    resume: bool = False
    #: Break a stale run-directory lock. Only after confirming the holder is gone.
    force_lock: bool = False
    evaluate_test: bool = False
    max_train_batches: int | None = None   # sanity runs only
    max_eval_batches: int | None = None

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size * 2
        if not 0.0 < self.data_fraction <= 1.0:
            raise ValueError(f"data_fraction must be in (0, 1], got {self.data_fraction}")
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(f"optimizer must be 'adam' or 'adamw', got {self.optimizer!r}")
        if self.scheduler not in ("cosine_warmup", "plateau", "none"):
            raise ValueError(f"Unknown scheduler {self.scheduler!r}")

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.experiment / self.profile

    def to_dict(self) -> dict:
        return asdict(self) | {"output_root": str(self.output_root)}


def resolve_device(name: str) -> torch.device:
    """Resolve a device name, refusing to quietly substitute a different one.

    Section 20: the CUDA profile must fail clearly rather than silently falling
    back to CPU.  A run that reports CUDA in its provenance and ran on CPU is a
    result nobody can reproduce or interpret.
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device {name!r} was requested but torch reports no CUDA device.\n"
            f"  torch.__version__      = {torch.__version__}\n"
            f"  torch.version.cuda     = {torch.version.cuda}\n"
            f"A '+cpu' build never sees a GPU no matter what hardware is present; "
            f"install a CUDA build:\n"
            f"    pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            f"To train on CPU instead, use scripts/train_hsen_cpu25.py."
        )
    return torch.device(name)


def set_seeds(seed: int) -> dict:
    """Seed every generator that affects a run, and report what was seeded."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs some throughput and buys reproducible numbers,
    # which is the trade this project has already made everywhere else.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"seed": seed, "cudnn_deterministic": True, "cudnn_benchmark": False}


def assert_same_architecture(left: nn.Module, right: nn.Module) -> None:
    """Fail unless two models are structurally identical.

    Section 22 asks for a check that the CPU and CUDA profiles instantiate the
    same architecture.  Comparing ``state_dict`` keys and shapes is the direct
    form of that question: same parameter names, same tensor shapes, same total.
    """
    left_state, right_state = left.state_dict(), right.state_dict()
    if set(left_state) != set(right_state):
        only_left = sorted(set(left_state) - set(right_state))[:5]
        only_right = sorted(set(right_state) - set(left_state))[:5]
        raise AssertionError(
            f"Architectures differ in parameter names: "
            f"only in first {only_left}, only in second {only_right}"
        )
    mismatched = [
        name for name in left_state
        if tuple(left_state[name].shape) != tuple(right_state[name].shape)
    ]
    if mismatched:
        raise AssertionError(f"Architectures differ in parameter shapes: {mismatched[:5]}")


def build_optimizer(model: nn.Module, config: TrainerConfig) -> torch.optim.Optimizer:
    # No weight decay on norms and biases. Decaying a LayerNorm gain pulls it
    # toward zero, which is not regularisation, it is damage.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if config.optimizer == "adam":
        return torch.optim.Adam(groups, lr=config.learning_rate)
    return torch.optim.AdamW(groups, lr=config.learning_rate)


def build_scheduler(optimizer, config: TrainerConfig, steps_per_epoch: int):
    """Linear warm-up into cosine decay, or plateau, or nothing.

    Warm-up matters more here than usual: the trunk is trained from scratch on
    top of frozen features, so the first steps see large, badly-scaled gradients
    through an untrained attention stack.  ESED uses a 10% linear warm-up and
    that is the default here.
    """
    if config.scheduler == "none":
        return None
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2,
        )
    total = max(1, steps_per_epoch * config.epochs)
    warmup = max(1, int(total * config.warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class HSENTrainer:
    """Train and evaluate one HSEN configuration."""

    def __init__(
        self,
        config: TrainerConfig,
        model_config: HSENConfig,
        loss_config: LossConfig,
        datasets: dict[str, HSENDataset],
        manifests: dict[str, pd.DataFrame],
        class_names: tuple[str, ...],
        audit: dict | None = None,
    ):
        self.config = config
        self.model_config = model_config
        self.loss_config = loss_config
        self.datasets = datasets
        self.manifests = manifests
        self.class_names = tuple(class_names)
        self.audit = audit or {}

        self.device = resolve_device(config.device)
        self.seeding = set_seeds(config.seed)
        # AMP only makes sense on CUDA. Asking for it on CPU is a config error
        # worth reporting rather than silently ignoring, because a run that
        # believes it used mixed precision and did not is mislabelled.
        self.amp = bool(config.amp and self.device.type == "cuda")
        if config.amp and not self.amp:
            print(f"[warn] AMP requested but device is {self.device}; running in FP32.")

        self.model = build_hsen(model_config).to(self.device)
        self.loss = HSENLoss(
            loss_config,
            num_classes=model_config.num_classes,
            task=model_config.task,
            class_weights=self._class_weights(),
        ).to(self.device)

        self.optimizer = build_optimizer(self.model, config)
        steps = max(1, len(datasets["train"]) // max(1, config.batch_size))
        self.scheduler = build_scheduler(self.optimizer, config, steps)
        # torch.amp.GradScaler("cuda") arrived in torch 2.3; the torch 2.0.x
        # CUDA build on the GPU machine only has torch.cuda.amp.GradScaler, whose
        # first positional argument is init_scale, not the device. Same class,
        # same defaults, so the two spellings are interchangeable here.
        if not self.amp:
            self.scaler = None
        elif hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.cuda.amp.GradScaler()

        self.primary = primary_metric_name(config.dataset, model_config.task)
        # The checkpoint manager reads the metrics dict the validation pass
        # produces, and those keys are prefixed. Monitoring the bare name would
        # look up a key that is never present, score NaN every epoch, and leave
        # best.pt unwritten -- with no error, because "no improvement" is a
        # perfectly ordinary thing for an epoch to report.
        self.monitor_key = f"val_{self.primary}"
        self.checkpoints = HSENCheckpointManager(
            config.run_dir / "checkpoints", monitor=self.monitor_key, mode="max",
            record=self._run_record(),
        )
        self.stopper = EarlyStopping(
            patience=config.patience, mode="max", monitor=self.primary,
            min_epochs=min(config.min_epochs, config.epochs),
        )
        self.history: list[dict] = []

    # ------------------------------------------------------------- setup

    def _class_weights(self) -> torch.Tensor | None:
        """Alpha derived from the **training** split only."""
        setting = self.loss_config.class_weights
        if setting == "none":
            return None
        if isinstance(setting, (list, tuple)):
            return torch.tensor(list(setting), dtype=torch.float32)
        labels = self.datasets["train"].labels
        if labels.class_id is None:
            return None
        return balanced_class_weights(
            torch.from_numpy(labels.class_id.astype(np.int64)), self.model_config.num_classes,
        )

    def _run_record(self) -> RunRecord:
        weights = self.loss.alpha
        return RunRecord(
            experiment=self.config.experiment,
            dataset=self.config.dataset,
            label_protocol=self.model_config.label_space,
            split_policy=self.audit.get("split_policy", "see manifest_summary.json"),
            data_fraction=self.config.data_fraction,
            seed=self.config.seed,
            device=str(self.device),
            profile=self.config.profile,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            optimizer=self.config.optimizer,
            scheduler=self.config.scheduler,
            amp=self.amp,
            modalities=list(self.model_config.modalities),
            fusion=self.model_config.fusion,
            model_config=self.model_config.to_dict(),
            loss_config=self.loss_config.to_dict(),
            parameter_counts=self.model.parameter_counts(),
            class_weights=weights.tolist() if weights.numel() else None,
            primary_metric=self.primary,
            hardware=hardware_record(self.device),
            dataset_counts={
                split: {
                    "samples": len(dataset),
                    "modalities": dataset.availability_summary(),
                }
                for split, dataset in self.datasets.items()
            },
            audit=self.audit,
        )

    def loader(self, split: str, shuffle: bool) -> Any:
        return build_dataloader(
            self.datasets[split],
            batch_size=self.config.batch_size if split == "train" else self.config.eval_batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            drop_last=False,
        )

    def _to_device(self, batch: dict) -> dict:
        move = lambda mapping: {k: v.to(self.device, non_blocking=True) for k, v in mapping.items()}
        return {
            "sample_ids": batch["sample_ids"],
            "features": move(batch["features"]),
            "masks": move(batch["masks"]),
            "available": move(batch["available"]),
            "targets": move(batch["targets"]),
        }

    # ---------------------------------------------------------- one epoch

    def train_one_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        totals: dict[str, float] = {}
        seen = 0
        for index, raw in enumerate(loader):
            if self.config.max_train_batches and index >= self.config.max_train_batches:
                break
            batch = self._to_device(raw)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", enabled=self.amp):
                outputs = self.model(batch["features"], batch["masks"], batch["available"])
                components = self.loss(outputs, batch["targets"])
            total = components["total"]

            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, batch {index}: "
                    f"{ {k: float(v) for k, v in components.items()} }.\n"
                    f"Training is stopped rather than continued through a NaN, "
                    f"which would silently destroy every weight in the model."
                )

            if self.scaler is not None:
                self.scaler.scale(total).backward()
                # Unscale before clipping: clipping scaled gradients would clip
                # to a threshold that changes with the loss scale.
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()

            if self.scheduler is not None and self.config.scheduler == "cosine_warmup":
                self.scheduler.step()

            size = len(batch["sample_ids"])
            seen += size
            for name, value in components.items():
                # detach() before the scalar read: these are live graph nodes
                # during training, and torch warns that converting one to a
                # Python float can hold the graph alive longer than intended.
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size

            if self.config.log_every and index and index % self.config.log_every == 0:
                print(
                    f"    batch {index:>5}  loss {float(total.detach()):.4f}  "
                    f"lr {self.optimizer.param_groups[0]['lr']:.2e}"
                )

        return {f"train_{name}": value / max(1, seen) for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, split: str, prefix: str | None = None) -> dict:
        """Run one split and compute every metric for it."""
        prefix = prefix if prefix is not None else f"{split}_"
        self.model.eval()
        loader = self.loader(split, shuffle=False)

        collected: dict[str, list] = {"logits": [], "valence": [], "arousal": []}
        truths: dict[str, list] = {"class_id": [], "valence": [], "arousal": [], "multilabel": []}
        sample_ids: list[str] = []
        loss_totals: dict[str, float] = {}
        seen = 0

        for index, raw in enumerate(loader):
            if self.config.max_eval_batches and index >= self.config.max_eval_batches:
                break
            batch = self._to_device(raw)
            with torch.autocast("cuda", enabled=self.amp):
                outputs = self.model(batch["features"], batch["masks"], batch["available"])
                components = self.loss(outputs, batch["targets"])

            size = len(batch["sample_ids"])
            seen += size
            sample_ids.extend(batch["sample_ids"])
            for name, value in components.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach()) * size

            # float() before stacking: autocast leaves half-precision tensors,
            # and a metric computed in fp16 is not the metric being reported.
            collected["logits"].append(outputs["logits"].float().cpu())
            for name in ("valence", "arousal"):
                if name in outputs:
                    collected[name].append(outputs[name].float().cpu())
            for name in ("class_id", "valence", "arousal", "multilabel"):
                if name in batch["targets"]:
                    truths[name].append(batch["targets"][name].float().cpu()
                                        if name != "class_id"
                                        else batch["targets"][name].cpu())

        stacked_outputs = {k: torch.cat(v) for k, v in collected.items() if v}
        stacked_truths = {k: torch.cat(v) for k, v in truths.items() if v}

        metrics = evaluate_predictions(
            stacked_outputs, stacked_truths, self.class_names,
            task=self.model_config.task, dataset=self.config.dataset,
        )
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        metrics |= {
            f"{prefix}loss_{name}": value / max(1, seen)
            for name, value in loss_totals.items()
        }
        metrics[f"{prefix}loss"] = metrics.pop(f"{prefix}loss_total", float("nan"))
        metrics[f"{prefix}samples"] = seen
        self._last_predictions = {
            "sample_ids": sample_ids,
            "outputs": stacked_outputs,
            "targets": stacked_truths,
        }
        return metrics

    # ---------------------------------------------------------------- run

    def run(self) -> dict:
        config = self.config
        config.run_dir.mkdir(parents=True, exist_ok=True)
        # Claimed for the life of the run. Two trainers on one directory is not
        # a hypothetical: it happened during this phase and produced artefacts
        # that were individually valid and jointly incoherent.
        lock = RunLock(config.run_dir).acquire(force=config.force_lock)
        try:
            return self._run(lock)
        finally:
            lock.release()

    def _run(self, lock: RunLock) -> dict:
        config = self.config
        self.checkpoints.save_run_record()

        print(self._header())
        start_epoch = 1
        if config.resume and self.checkpoints.last_path.exists():
            start_epoch, self.history = self.checkpoints.resume(
                self.model, self.optimizer, self.scheduler, scaler=self.scaler,
            )
            print(f"Resumed from epoch {start_epoch - 1}; "
                  f"best {self.primary} so far {self.checkpoints.best_value:.4f}\n")

        train_loader = self.loader("train", shuffle=True)
        started = time.time()
        stop_reason = "completed"

        for epoch in range(start_epoch, config.epochs + 1):
            epoch_started = time.time()
            train_metrics = self.train_one_epoch(train_loader, epoch)
            validation_metrics = self.evaluate("validation", prefix="val_")

            if self.scheduler is not None and config.scheduler == "plateau":
                self.scheduler.step(validation_metrics.get(f"val_{self.primary}", 0.0))

            row = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - epoch_started, 2),
                **train_metrics,
                **{k: v for k, v in validation_metrics.items()
                   if not isinstance(v, (dict, list))},
            }
            self.history.append(row)
            print(self._epoch_report(epoch, row, validation_metrics))

            improved = self.checkpoints.update(
                self.model, self.optimizer, self.scheduler, epoch,
                validation_metrics, self.history, scaler=self.scaler,
            )
            if improved:
                print(f"    -> new best {self.primary} "
                      f"{self.checkpoints.best_value:.4f}, checkpoint saved")

            state = self.stopper.update(
                validation_metrics.get(f"val_{self.primary}", float("-inf")), epoch,
            )
            if state.should_stop:
                stop_reason = state.reason or "early stopping"
                print(f"\nStopping early at epoch {epoch}: {stop_reason}")
                break

        elapsed = time.time() - started
        summary = self._finalise(elapsed, stop_reason)
        print(self._final_report(summary))
        return summary

    def _finalise(self, elapsed: float, stop_reason: str) -> dict:
        """Load the best checkpoint, optionally open the test split, save everything."""
        best_epoch = self.checkpoints.best_epoch
        if self.checkpoints.best_path.exists():
            payload = self.checkpoints.load(self.checkpoints.best_path, map_location=self.device)
            self.model.load_state_dict(payload["model_state_dict"])

        summary = {
            "experiment": self.config.experiment,
            "profile": self.config.profile,
            "trainer_config": self.config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "loss_config": self.loss_config.to_dict(),
            "run_record": self.checkpoints.record.to_dict() if self.checkpoints.record else {},
            "primary_metric": self.primary,
            "best_epoch": best_epoch,
            f"best_val_{self.primary}": self.checkpoints.best_value,
            "epochs_run": len(self.history),
            "training_seconds": round(elapsed, 2),
            "stop_reason": stop_reason,
            "early_stopping": self.stopper.summary(),
            "parameter_counts": self.model.parameter_counts(),
            "history": self.history,
        }

        # Validation is re-run against the *restored best* weights so the
        # reported validation numbers belong to the checkpoint that is shipped,
        # not to whichever epoch happened to run last.
        summary["validation"] = self.evaluate("validation", prefix="")

        if self.config.evaluate_test:
            if "test" not in self.datasets:
                raise RuntimeError("Test evaluation requested but no test dataset was built")
            summary["test"] = self.evaluate("test", prefix="")
            self._save_predictions(self.config.run_dir / "test_predictions.parquet")

        self._save(summary)
        return summary

    def _save_predictions(self, path: Path) -> Path:
        record = getattr(self, "_last_predictions", None)
        if not record:
            raise RuntimeError("No predictions to save; run evaluate first")
        outputs, targets = record["outputs"], record["targets"]
        frame = pd.DataFrame({"sample_id": record["sample_ids"]})
        if self.model_config.task == "single_label":
            frame["predicted_id"] = outputs["logits"].argmax(dim=-1).numpy()
            frame["predicted_class"] = [self.class_names[i] for i in frame["predicted_id"]]
            probabilities = torch.softmax(outputs["logits"], dim=-1).numpy()
            for index, name in enumerate(self.class_names):
                frame[f"prob_{name}"] = probabilities[:, index]
        if "class_id" in targets:
            frame["true_id"] = targets["class_id"].numpy()
        for name in ("valence", "arousal"):
            if name in outputs:
                frame[f"predicted_{name}"] = outputs[name].numpy()
            if name in targets:
                frame[f"true_{name}"] = targets[name].numpy()
        frame.to_parquet(path, index=False)
        return path

    def _save(self, summary: dict) -> None:
        """Write the run's artefacts.

        Note the absence of ``sort_keys``. It is tempting -- a sorted JSON diffs
        cleanly -- and it silently destroys the one thing this project treats as
        a scientific contract: ``per_class`` is keyed by class name in *declared*
        order, and sorting it alphabetically rotates every per-class array and
        every figure drawn from one. See :mod:`src.common.labels`.
        """
        run_dir = self.config.run_dir
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        if self.history:
            pd.DataFrame(self.history).to_csv(run_dir / "metrics" / "history.csv", index=False)
        for split in ("validation", "test"):
            if split in summary:
                (run_dir / "metrics" / f"{split}_metrics.json").write_text(
                    json.dumps(summary[split], indent=2, default=str),
                    encoding="utf-8",
                )

    # ------------------------------------------------------------ reports

    def _header(self) -> str:
        counts = self.model.parameter_counts()
        hardware = hardware_record(self.device)
        lines = [
            "=" * 74,
            f"HSEN  --  {self.config.experiment}  ({self.config.profile} profile)",
            "=" * 74,
            f"  device          {self.device}"
            + (f"  [{hardware.get('gpu_name')}, {hardware.get('gpu_total_memory_gb')} GB]"
               if self.device.type == "cuda" else ""),
            f"  mixed precision {'on' if self.amp else 'off'}",
            f"  seed            {self.config.seed}",
            f"  data fraction   {self.config.data_fraction:.0%}",
            f"  modalities      {', '.join(self.model_config.modalities)}",
            f"  fusion          {self.model_config.fusion}  "
            f"(d={self.model_config.model_dim}, {self.model_config.num_layers} layers, "
            f"{self.model_config.num_heads} heads)",
            f"  parameters      {counts['total']:,} total, {counts['trainable']:,} trainable",
            f"  optimizer       {self.config.optimizer} @ lr {self.config.learning_rate:g}, "
            f"{self.config.scheduler} schedule",
            f"  loss            focal(gamma={self.loss_config.focal_gamma}) "
            f"x{self.loss_config.lambda_cls} + val x{self.loss_config.lambda_valence} "
            f"+ aro x{self.loss_config.lambda_arousal}",
            f"  primary metric  val_{self.primary}",
            "",
            "  split sizes     " + "  ".join(
                f"{split}={len(dataset):,}" for split, dataset in self.datasets.items()
            ),
        ]
        for split, dataset in self.datasets.items():
            lines.append(f"  {split:<14}  modality availability "
                         f"{dataset.availability_summary()}")
        lines.append("=" * 74)
        return "\n".join(lines)

    def _epoch_report(self, epoch: int, row: dict, validation: dict) -> str:
        lines = [
            f"Epoch {epoch:02d}/{self.config.epochs}",
            f"  Train Loss:        {row.get('train_total', float('nan')):.4f}"
            f"   (cls {row.get('train_cls', float('nan')):.4f}"
            f"  val {row.get('train_valence', float('nan')):.4f}"
            f"  aro {row.get('train_arousal', float('nan')):.4f})",
            f"  Val Loss:          {row.get('val_loss', float('nan')):.4f}",
        ]
        if self.model_config.task == "multi_label":
            lines += [
                f"  Val Accuracy:      {validation.get('val_accuracy', float('nan')):.4f}"
                f"   (exact set match)",
                f"  Val Micro-F1:      {validation.get('val_micro_f1', float('nan')):.4f}",
                f"  Val Macro-F1:      {validation.get('val_macro_f1', float('nan')):.4f}",
            ]
        else:
            lines += [
                f"  Val Accuracy:      {validation.get('val_accuracy', float('nan')):.4f}",
                f"  Val Weighted-F1:   {validation.get('val_weighted_f1', float('nan')):.4f}",
                f"  Val Macro-F1:      {validation.get('val_macro_f1', float('nan')):.4f}",
            ]
        for name in ("valence", "arousal"):
            key = f"val_{name}_ccc"
            if key in validation and validation[key] == validation[key]:
                lines.append(
                    f"  Val {name.capitalize()[:3]} CCC:       {validation[key]:.4f}"
                    f"   (MAE {validation.get(f'val_{name}_mae', float('nan')):.4f})"
                )
        lines.append(f"  LR:                {row['learning_rate']:.2e}"
                     f"   ({row['seconds']:.1f}s)")
        return "\n".join(lines)

    def _final_report(self, summary: dict) -> str:
        lines = [
            "",
            "=" * 74,
            f"RUN COMPLETE  --  {self.config.experiment} ({self.config.profile})",
            "=" * 74,
            f"  Best epoch                 {summary['best_epoch']}",
            f"  Best validation {self.primary:<10} "
            f"{summary[f'best_val_{self.primary}']:.4f}",
            f"  Epochs run                 {summary['epochs_run']}",
            f"  Training time              {summary['training_seconds']:.1f}s",
            f"  Parameters                 {summary['parameter_counts']['total']:,}",
            f"  Stop reason                {summary['stop_reason']}",
        ]
        for split in ("validation", "test"):
            metrics = summary.get(split)
            if not metrics:
                continue
            lines += ["", f"  {split.upper()}"]
            for name in ("accuracy", "weighted_f1", "macro_f1", "micro_f1"):
                if name in metrics and metrics[name] == metrics[name]:
                    lines.append(f"    {name:<24} {metrics[name]:.4f}")
            for affect in ("valence", "arousal"):
                if f"{affect}_ccc" in metrics and metrics[f"{affect}_ccc"] == metrics[f"{affect}_ccc"]:
                    lines.append(
                        f"    {affect} CCC/MAE/RMSE     "
                        f"{metrics[f'{affect}_ccc']:.4f} / "
                        f"{metrics[f'{affect}_mae']:.4f} / "
                        f"{metrics[f'{affect}_rmse']:.4f}"
                    )
            if metrics.get("per_class"):
                lines.append("    per-class F1:")
                for name, values in metrics["per_class"].items():
                    lines.append(
                        f"      {name:<14} {values['f1']:.4f}  (n={values['support']})"
                    )
            if metrics.get("sentiment"):
                lines.append("    sentiment protocol:")
                for name, value in metrics["sentiment"].items():
                    if isinstance(value, float) and value == value:
                        lines.append(f"      {name:<22} {value:.4f}")
        lines += ["", f"  Artefacts: {self.config.run_dir}", "=" * 74]
        return "\n".join(lines)
