"""Filesystem layout for fraction-sampled, multi-iteration experiments.

A single experiment is addressed by ``(modality, fraction)`` and expands to::

    experiments/<modality>/<fraction-label>/
        metadata/{train,validation,test}.parquet
        metadata/sampling_summary.json
        iteration_<label>/
            config.json
            checkpoints/{best,last}.pt
            results/*.json
            logs/training.log
        experiment_summary.json

The layout is data only: nothing here creates or deletes directories unless a
caller asks for it via :meth:`ExperimentLayout.prepare_iteration`.  Legacy
result locations (``results/`` and ``checkpoints/``) are untouched by design.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import ensure_dir


DEFAULT_EXPERIMENT_ROOT = Path("experiments")
SPLITS = ("train", "validation", "test")

#: Label used when an experiment consumes the entire eligible modality pool.
#: Text and physiology run at full scale, and ``text/full`` reads better than
#: ``text/100pct`` in both the tree and the experiment name.
FULL_LABEL = "full"

_LABEL_PATTERN = re.compile(r"^(?P<fraction>\d+(?:_\d+)?)pct$")
_NAME_PATTERN = re.compile(r"^(?P<modality>[a-z_]+)_(?P<label>\d+(?:_\d+)?pct|full)$")


def fraction_to_label(fraction: float) -> str:
    """Return a filesystem-safe label such as ``25pct``, ``12_5pct``, or ``full``."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return FULL_LABEL
    return f"{fraction * 100:g}".replace(".", "_") + "pct"


def label_to_fraction(label: str) -> float:
    """Inverse of :func:`fraction_to_label`."""
    if label == FULL_LABEL:
        return 1.0
    match = _LABEL_PATTERN.match(label)
    if match is None:
        raise ValueError(f"Unrecognised fraction label: {label!r}")
    return float(match.group("fraction").replace("_", ".")) / 100.0


@dataclass(frozen=True)
class ExperimentLayout:
    """Resolve every path belonging to one sampled experiment."""

    modality: str
    fraction: float
    root: Path = DEFAULT_EXPERIMENT_ROOT
    #: Overrides the label derived from ``fraction``.  Only used to round-trip
    #: an explicitly named experiment; the derived label is otherwise correct.
    label_override: str | None = None

    def __post_init__(self) -> None:
        if not self.modality:
            raise ValueError("modality must be a non-empty string")
        # Validates the fraction eagerly so a bad value fails at construction.
        derived = fraction_to_label(self.fraction)
        if self.label_override is not None:
            if label_to_fraction(self.label_override) != self.fraction:
                raise ValueError(
                    f"label_override {self.label_override!r} does not describe "
                    f"fraction {self.fraction} (expected {derived!r})"
                )
        object.__setattr__(self, "root", Path(self.root))

    # ---------------------------------------------------------------- naming

    @property
    def label(self) -> str:
        return self.label_override or fraction_to_label(self.fraction)

    @property
    def name(self) -> str:
        """Stable experiment identity, for example ``image_25pct``."""
        return f"{self.modality}_{self.label}"

    @classmethod
    def from_name(cls, name: str, root: Path | str = DEFAULT_EXPERIMENT_ROOT) -> "ExperimentLayout":
        match = _NAME_PATTERN.match(name)
        if match is None:
            raise ValueError(f"Unrecognised experiment name: {name!r} (expected e.g. 'image_25pct')")
        label = match.group("label")
        return cls(match.group("modality"), label_to_fraction(label), Path(root), label)

    # ------------------------------------------------------------ experiment

    @property
    def base(self) -> Path:
        return self.root / self.modality / self.label

    @property
    def metadata_dir(self) -> Path:
        return self.base / "metadata"

    def split_path(self, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return self.metadata_dir / f"{split}.parquet"

    @property
    def sampling_summary_path(self) -> Path:
        return self.metadata_dir / "sampling_summary.json"

    @property
    def validation_report_path(self) -> Path:
        return self.metadata_dir / "validation_report.json"

    @property
    def experiment_summary_path(self) -> Path:
        return self.base / "experiment_summary.json"

    # ------------------------------------------------------------- iteration

    @staticmethod
    def iteration_label(iteration: int | str) -> str:
        label = str(iteration).strip()
        if not label or not re.match(r"^[A-Za-z0-9_-]+$", label):
            raise ValueError(f"Invalid iteration label: {iteration!r}")
        return label

    def iteration_dir(self, iteration: int | str) -> Path:
        return self.base / f"iteration_{self.iteration_label(iteration)}"

    def checkpoints_dir(self, iteration: int | str) -> Path:
        return self.iteration_dir(iteration) / "checkpoints"

    def results_dir(self, iteration: int | str) -> Path:
        return self.iteration_dir(iteration) / "results"

    def logs_dir(self, iteration: int | str) -> Path:
        return self.iteration_dir(iteration) / "logs"

    def config_path(self, iteration: int | str) -> Path:
        return self.iteration_dir(iteration) / "config.json"

    def log_path(self, iteration: int | str) -> Path:
        return self.logs_dir(iteration) / "training.log"

    def best_checkpoint(self, iteration: int | str) -> Path:
        return self.checkpoints_dir(iteration) / "best.pt"

    def last_checkpoint(self, iteration: int | str) -> Path:
        return self.checkpoints_dir(iteration) / "last.pt"

    # ------------------------------------------------------------ filesystem

    def prepare_metadata(self) -> Path:
        ensure_dir(self.metadata_dir)
        return self.metadata_dir

    def prepare_iteration(self, iteration: int | str) -> Path:
        for directory in (
            self.iteration_dir(iteration),
            self.checkpoints_dir(iteration),
            self.results_dir(iteration),
            self.logs_dir(iteration),
        ):
            ensure_dir(directory)
        return self.iteration_dir(iteration)

    def existing_iterations(self) -> list[str]:
        """Return iteration labels present on disk, numeric ones sorted first."""
        if not self.base.exists():
            return []
        labels = [path.name[len("iteration_"):] for path in self.base.glob("iteration_*") if path.is_dir()]
        return sorted(labels, key=lambda label: (not label.isdigit(), int(label) if label.isdigit() else 0, label))
