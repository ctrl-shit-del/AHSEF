"""One prediction interface for all five frozen unimodal baselines.

Every baseline already knows how to build its own dataloader and how to
describe its own architecture; what none of them expose is a *per-sample*
record carrying logits, probabilities, the identity of the sample, and the
measured cost of producing it.  AHSEF needs exactly that, identically shaped
across modalities, before it can reason about which modality to acquire next.

This module adds that layer without touching the baselines:

* the model is rebuilt from ``run_summary['model']`` through
  :func:`src.training.modalities.build_model_from_record`, exactly as
  ``verify_artifacts`` does, so the architecture comes from the recorded run
  rather than from whatever the code defines today;
* the dataloader is built by the modality's own runner, so the front end,
  filtering, and sample ordering are the ones the baseline was trained with;
* the checkpoint file is hashed before and after the pass and the run aborts
  if the digest changes -- a frozen baseline stays frozen;
* latency is measured, not assumed.

The record schema is identical for a 7-class emotion model and the 3-class
WESAD model.  ``class_order`` in the sidecar metadata is what tells them apart,
and :mod:`src.ahsef.fusion` refuses to mix two different class orders.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from src.ahsef.registry import BaselineRef, baseline
from src.ahsef.uncertainty import (
    UNCERTAINTY_DEFINITION,
    probabilities_from_logits,
    uncertainty_columns,
)
from src.common.experiment_layout import SPLITS, ExperimentLayout
from src.common.labels import get_label_space
from src.training.checkpoint import CheckpointManager
from src.training.modalities import build_model_from_record, get_runner_class, get_spec


PREDICTION_SCHEMA_VERSION = "ahsef.predictions.v1"


class FrozenCheckpointError(RuntimeError):
    """Raised if a baseline checkpoint changed on disk during an AHSEF pass."""


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, used to prove a frozen checkpoint stayed frozen."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


#: Where a run summary may declare its class order, most authoritative first.
#: The image baseline predates ``run_summary['class_order']`` -- it was trained
#: before the label-space refactor -- but it does record the same ordered list
#: under ``class_weights``.  Resolving through a declared chain, and recording
#: which link answered, is honest; silently defaulting to the 7-class order
#: would be the failure mode that quietly rotates a confusion matrix.
CLASS_ORDER_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run_summary.class_order", ("class_order",)),
    ("run_summary.test_metrics.class_order", ("test_metrics", "class_order")),
    ("run_summary.class_weights.class_order", ("class_weights", "class_order")),
)


def resolve_class_order(run_summary: Mapping, source_path: Path) -> tuple[tuple[str, ...], str]:
    """Return ``(class_order, where_it_came_from)`` or raise.

    Falls back to the declared label space of the recorded task only when no
    artefact states the order directly, and never invents one.
    """
    for label, keys in CLASS_ORDER_SOURCES:
        node: object = run_summary
        for key in keys:
            node = (node or {}).get(key) if isinstance(node, Mapping) else None
        if isinstance(node, (list, tuple)) and node:
            return tuple(str(name) for name in node), label

    task = run_summary.get("task") or (run_summary.get("config") or {}).get("task")
    if task:
        return tuple(get_label_space(str(task)).classes), f"label space {task!r}"

    num_classes = (run_summary.get("model") or {}).get("num_classes")
    raise ValueError(
        f"{source_path} declares no class order and no task, so the meaning of its "
        f"{num_classes} output positions is unknown. Re-run the baseline with the "
        f"current src.training.base_experiment, which records both."
    )


# ============================================================
# Prediction set
# ============================================================

@dataclass
class PredictionSet:
    """Per-sample outputs of one baseline on one split, plus its provenance.

    ``frame`` columns:

    ``sample_id``           str, the cross-modality identity
    ``dataset``             str, the contributing corpus
    ``true_class``          int, ground truth -- present for evaluation only
    ``predicted_class``     int
    ``logit_<c>``           float, one column per class in declared order
    ``prob_<c>``            float, softmax of the logits at the applied temperature
    ``confidence``          float, ``max_c p_c``
    ``predictive_entropy``  float, nats
    ``normalized_entropy``  float, in ``[0, 1]``
    ``margin``              float, top-1 minus top-2 probability
    ``latency_ms``          float, measured per-sample end-to-end
    ``inference_ms``        float, measured per-sample forward pass only
    """

    modality: str
    split: str
    class_order: tuple[str, ...]
    frame: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        missing = {"sample_id", "true_class", "predicted_class"} - set(self.frame.columns)
        if missing:
            raise ValueError(f"PredictionSet frame is missing columns: {sorted(missing)}")
        duplicated = self.frame["sample_id"].duplicated()
        if bool(duplicated.any()):
            raise ValueError(
                f"{self.modality}/{self.split} predictions contain "
                f"{int(duplicated.sum())} duplicate sample_id values"
            )

    # ------------------------------------------------------------- accessors

    @property
    def num_classes(self) -> int:
        return len(self.class_order)

    @property
    def prob_columns(self) -> list[str]:
        return [f"prob_{index}" for index in range(self.num_classes)]

    @property
    def logit_columns(self) -> list[str]:
        return [f"logit_{index}" for index in range(self.num_classes)]

    def probabilities(self) -> torch.Tensor:
        return torch.tensor(self.frame[self.prob_columns].to_numpy(), dtype=torch.float64)

    def logits(self) -> torch.Tensor:
        return torch.tensor(self.frame[self.logit_columns].to_numpy(), dtype=torch.float64)

    def labels(self) -> torch.Tensor:
        return torch.tensor(self.frame["true_class"].to_numpy(), dtype=torch.long)

    def predictions(self) -> torch.Tensor:
        return torch.tensor(self.frame["predicted_class"].to_numpy(), dtype=torch.long)

    def sample_ids(self) -> list[str]:
        return self.frame["sample_id"].astype(str).tolist()

    def restricted_to(self, sample_ids) -> "PredictionSet":
        """Return the same set narrowed to ``sample_ids``, in that exact order.

        Ordering by the requested ids -- rather than by manifest position -- is
        what lets two modalities be stacked row-for-row without ever trusting
        that their manifests happen to agree on order.
        """
        wanted = list(dict.fromkeys(str(identifier) for identifier in sample_ids))
        indexed = self.frame.set_index(self.frame["sample_id"].astype(str))
        unknown = [identifier for identifier in wanted if identifier not in indexed.index]
        if unknown:
            raise KeyError(
                f"{self.modality}/{self.split} has no prediction for "
                f"{len(unknown)} requested sample ids (e.g. {unknown[:5]})"
            )
        frame = indexed.loc[wanted].reset_index(drop=True)
        return PredictionSet(
            modality=self.modality, split=self.split, class_order=self.class_order,
            frame=frame, meta={**self.meta, "restricted_to_samples": len(wanted)},
        )

    def with_temperature(self, temperature: float) -> "PredictionSet":
        """Recompute probabilities and uncertainty at a different temperature.

        The logits are the model's frozen output; a temperature only changes
        how they are read.  The new set records which temperature produced it.
        """
        probabilities = probabilities_from_logits(self.logits(), temperature=temperature)
        frame = self.frame.copy()
        for index, column in enumerate(self.prob_columns):
            frame[column] = probabilities[:, index].numpy()
        for name, values in uncertainty_columns(probabilities).items():
            frame[name] = values.numpy()
        return PredictionSet(
            modality=self.modality, split=self.split, class_order=self.class_order,
            frame=frame, meta={**self.meta, "temperature": float(temperature),
                               "temperature_applied": temperature != 1.0},
        )

    # ------------------------------------------------------------------- io

    def save(self, parquet_path: Path, meta_path: Path | None = None) -> Path:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(parquet_path, index=False)
        target = Path(meta_path) if meta_path else parquet_path.with_suffix(".json")
        payload = {
            "schema": PREDICTION_SCHEMA_VERSION,
            "modality": self.modality,
            "split": self.split,
            "class_order": list(self.class_order),
            "samples": int(len(self.frame)),
            "predictions_path": str(parquet_path),
            **self.meta,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return parquet_path

    @classmethod
    def load(cls, parquet_path: Path, meta_path: Path | None = None) -> "PredictionSet":
        parquet_path = Path(parquet_path)
        target = Path(meta_path) if meta_path else parquet_path.with_suffix(".json")
        if not target.exists():
            raise FileNotFoundError(
                f"Prediction sidecar {target} is missing; a prediction frame without "
                f"its class order and provenance cannot be interpreted."
            )
        meta = json.loads(target.read_text(encoding="utf-8"))
        frame = pd.read_parquet(parquet_path)
        frame["sample_id"] = frame["sample_id"].astype(str)
        return cls(
            modality=meta["modality"], split=meta["split"],
            class_order=tuple(meta["class_order"]), frame=frame,
            meta={k: v for k, v in meta.items()
                  if k not in {"modality", "split", "class_order"}},
        )


# ============================================================
# Predictor
# ============================================================

class BaselinePredictor:
    """Run one frozen baseline over one split and emit a :class:`PredictionSet`."""

    def __init__(
        self,
        modality: str,
        experiment: str,
        iteration: str,
        root: Path | str = "experiments",
        device: str = "cpu",
    ):
        self.modality = modality
        self.experiment = experiment
        self.iteration = str(iteration)
        self.root = Path(root)
        self.layout = ExperimentLayout.from_name(experiment, self.root)
        self.device = torch.device(device)

        summary_path = self.layout.results_dir(self.iteration) / "run_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No run summary at {summary_path}; AHSEF rebuilds the architecture "
                f"from the recorded run, so the baseline must have been run first."
            )
        self.run_summary: dict = json.loads(summary_path.read_text(encoding="utf-8"))
        self.class_order, self.class_order_source = resolve_class_order(
            self.run_summary, summary_path
        )
        self.checkpoint_path = self.layout.best_checkpoint(self.iteration)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {self.checkpoint_path}")
        self.checkpoint_digest = file_digest(self.checkpoint_path)

        self._runner = self._build_runner()
        self.model = self._load_model()

    @classmethod
    def from_reference(
        cls, reference: BaselineRef, root: Path | str = "experiments", device: str = "cpu"
    ) -> "BaselinePredictor":
        return cls(
            modality=reference.modality, experiment=reference.experiment,
            iteration=reference.resolve_iteration(root), root=root, device=device,
        )

    @classmethod
    def for_modality(
        cls, modality: str, root: Path | str = "experiments",
        iteration: str | None = None, device: str = "cpu",
    ) -> "BaselinePredictor":
        reference = baseline(modality)
        if iteration is not None:
            reference = BaselineRef(
                reference.modality, reference.experiment, reference.label_column, str(iteration)
            )
        return cls.from_reference(reference, root=root, device=device)

    # --------------------------------------------------------------- set-up

    def _build_runner(self):
        """Instantiate the modality's own runner so its dataloader is reused verbatim."""
        runner_class = get_runner_class(self.modality)
        config_class = get_spec(self.modality).config_class
        recorded = dict(self.run_summary.get("config") or {})
        known = {item.name for item in fields(config_class)}
        unknown = sorted(set(recorded) - known)
        if unknown:
            raise ValueError(
                f"run_summary config for {self.experiment}/{self.iteration} carries "
                f"fields {unknown} that {config_class.__name__} does not declare; "
                f"the checkpoint and the code have diverged."
            )
        config = config_class(**{key: value for key, value in recorded.items() if key in known})
        # Inference must see every sample of the split, never a debug bound.
        config.max_train = config.max_val = config.max_test = None
        config.num_workers = 0
        runner = runner_class(config)

        if self.modality == "physiology":
            # The baseline standardised its features with statistics fitted on
            # the training windows.  Re-apply exactly those, from the artefact
            # the run wrote; refitting here would leak the evaluation split.
            normalisation = (
                self.layout.results_dir(self.iteration) / "feature_normalization.json"
            )
            if not normalisation.exists():
                raise FileNotFoundError(
                    f"Missing {normalisation}; the physiology baseline's train-fitted "
                    f"feature scaling is required to reproduce its inputs."
                )
            record = json.loads(normalisation.read_text(encoding="utf-8"))
            if record.get("computed_from_split") != "train":
                raise ValueError(
                    f"{normalisation} was not computed from the training split "
                    f"({record.get('computed_from_split')!r}); refusing to use it."
                )
            runner._scaling = (record["mean"], record["std"])
        return runner

    def _load_model(self) -> torch.nn.Module:
        model = build_model_from_record(self.run_summary)
        CheckpointManager.load(
            self.checkpoint_path, model=model, optimizer=None, device=self.device
        )
        return model.to(self.device).eval()

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def predict(self, split: str, temperature: float = 1.0) -> PredictionSet:
        """Score every sample of ``split`` and return the per-sample records."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        runner = self._runner
        loader = runner.build_loader(split, shuffle=False, max_samples=None)
        input_key = runner.INPUT_KEY
        lengths_key = runner.LENGTHS_KEY

        sample_ids: list[str] = []
        datasets: list[str] = []
        labels: list[int] = []
        logits_rows: list[torch.Tensor] = []
        load_ms: list[float] = []
        forward_ms: list[float] = []

        started = time.perf_counter()
        mark = time.perf_counter()
        for batch in loader:
            load_seconds = time.perf_counter() - mark
            features = batch[input_key].to(self.device)
            lengths = (
                batch[lengths_key].to(self.device)
                if lengths_key and batch.get(lengths_key) is not None
                else None
            )
            forward_started = time.perf_counter()
            logits = self.model(features) if lengths is None else self.model(features, lengths)
            forward_seconds = time.perf_counter() - forward_started

            size = int(logits.shape[0])
            sample_ids.extend(str(value) for value in batch["sample_id"])
            datasets.extend(str(value) for value in batch["dataset"])
            labels.extend(int(value) for value in batch["label"])
            logits_rows.append(logits.detach().cpu().to(torch.float64))
            load_ms.extend([load_seconds * 1000.0 / size] * size)
            forward_ms.extend([forward_seconds * 1000.0 / size] * size)
            mark = time.perf_counter()
        wall_seconds = time.perf_counter() - started

        if not logits_rows:
            raise RuntimeError(f"{self.modality}/{split} produced no predictions")
        after_digest = file_digest(self.checkpoint_path)
        if after_digest != self.checkpoint_digest:
            raise FrozenCheckpointError(
                f"{self.checkpoint_path} changed during inference "
                f"({self.checkpoint_digest} -> {after_digest}); a frozen baseline "
                f"must not be modified."
            )

        all_logits = torch.cat(logits_rows, dim=0)
        if all_logits.shape[1] != len(self.class_order):
            raise ValueError(
                f"{self.modality} emitted {all_logits.shape[1]} logits but its recorded "
                f"class order has {len(self.class_order)} entries"
            )
        probabilities = probabilities_from_logits(all_logits, temperature=temperature)

        frame = pd.DataFrame({
            "sample_id": sample_ids,
            "dataset": datasets,
            "modality": self.modality,
            "split": split,
            "true_class": labels,
        })
        for index in range(all_logits.shape[1]):
            frame[f"logit_{index}"] = all_logits[:, index].numpy()
        for index in range(probabilities.shape[1]):
            frame[f"prob_{index}"] = probabilities[:, index].numpy()
        for name, values in uncertainty_columns(probabilities).items():
            frame[name] = values.numpy()
        frame["latency_ms"] = [a + b for a, b in zip(load_ms, forward_ms)]
        frame["inference_ms"] = forward_ms
        frame["feature_ms"] = load_ms

        return PredictionSet(
            modality=self.modality, split=split, class_order=self.class_order,
            frame=frame, meta=self._meta(split, temperature, wall_seconds, frame),
        )

    def _meta(self, split: str, temperature: float, wall_seconds: float, frame: pd.DataFrame) -> dict:
        model_record = dict(self.run_summary.get("model") or {})
        return {
            "experiment": self.experiment,
            "iteration": self.iteration,
            "task": self.run_summary.get("task")
                    or (self.run_summary.get("config") or {}).get("task"),
            "class_order_source": self.class_order_source,
            "task_limitation": self.run_summary.get("task_limitation"),
            "metadata_dir": self.run_summary.get("metadata_dir"),
            "manifest": str(self._runner.manifest_path(split)),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_digest,
            "checkpoint_unchanged": True,
            "baseline_best_epoch": self.run_summary.get("best_epoch"),
            "baseline_best_val_macro_f1": self.run_summary.get("best_val_macro_f1"),
            "baseline_seed": (self.run_summary.get("config") or {}).get("seed"),
            "baseline_run_seed": (self.run_summary.get("config") or {}).get("run_seed"),
            # The whole architecture record, not just its name: the cost proxy
            # in src.ahsef.costs needs the input geometry (max_frames, n_mels,
            # image_size, num_frames, max_tokens, feature_dim) and reading it
            # from the export keeps the cost model independent of the baseline
            # directory layout.
            "model": model_record,
            "device": str(self.device),
            "temperature": float(temperature),
            "temperature_applied": temperature != 1.0,
            "uncertainty_definition": dict(UNCERTAINTY_DEFINITION),
            "latency": {
                "measured": True,
                "wall_seconds": round(wall_seconds, 4),
                "mean_latency_ms": float(frame["latency_ms"].mean()),
                "mean_inference_ms": float(frame["inference_ms"].mean()),
                "mean_feature_ms": float(frame["feature_ms"].mean()),
                "note": "Wall-clock on this host, batch time divided evenly across the "
                        "batch. Comparable across modalities only within one export run.",
            },
            "host": {"platform": platform.platform(), "torch": torch.__version__},
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "uses_labels_for_prediction": False,
        }


def load_predictions(paths: Mapping[str, Path]) -> dict[str, PredictionSet]:
    """Load several saved prediction sets, keyed by modality."""
    return {modality: PredictionSet.load(Path(path)) for modality, path in paths.items()}


def prediction_record(prediction_set: PredictionSet, index: int) -> dict[str, Any]:
    """The single-sample dictionary form named in the AHSEF specification."""
    row = prediction_set.frame.iloc[index]
    return {
        "modality": prediction_set.modality,
        "sample_id": str(row["sample_id"]),
        "logits": [float(row[column]) for column in prediction_set.logit_columns],
        "probabilities": [float(row[column]) for column in prediction_set.prob_columns],
        "predicted_class": int(row["predicted_class"]),
        "true_class": int(row["true_class"]),
        "confidence": float(row["confidence"]),
        "normalized_entropy": float(row["normalized_entropy"]),
        "latency_ms": float(row["latency_ms"]),
        "class_order": list(prediction_set.class_order),
    }
