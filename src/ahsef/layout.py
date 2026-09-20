"""Filesystem layout for one AHSEF run.

Deliberately parallel to :class:`~src.common.experiment_layout.ExperimentLayout`
but rooted at ``experiments/ahsef/`` so that no AHSEF command can ever write
into a frozen baseline's iteration directory::

    experiments/ahsef/<run>/
        provenance.json
        predictions/<modality>__<split>.parquet
        predictions/<modality>__<split>.json
        alignment/alignment_matrix.json
        alignment/alignment_report.json
        calibration/<modality>.json
        calibration/temperature.json
        fusion/<pair>/...
        reports/...

The layout is data only; directories are created only when a caller asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.utils.io import ensure_dir


DEFAULT_AHSEF_ROOT = Path("experiments") / "ahsef"

SPLITS = ("train", "validation", "test")

_RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class AhsefLayout:
    """Resolve every path belonging to one AHSEF run."""

    run: str = "stage1"
    root: Path = DEFAULT_AHSEF_ROOT

    def __post_init__(self) -> None:
        if not _RUN_PATTERN.match(self.run):
            raise ValueError(
                f"Invalid AHSEF run name {self.run!r}; expected letters, digits, "
                f"'_', '-', or '.' starting with an alphanumeric character"
            )
        object.__setattr__(self, "root", Path(self.root))

    # ---------------------------------------------------------------- paths

    @property
    def base(self) -> Path:
        return self.root / self.run

    @property
    def provenance_path(self) -> Path:
        return self.base / "provenance.json"

    @property
    def predictions_dir(self) -> Path:
        return self.base / "predictions"

    @property
    def alignment_dir(self) -> Path:
        return self.base / "alignment"

    @property
    def calibration_dir(self) -> Path:
        return self.base / "calibration"

    @property
    def fusion_dir(self) -> Path:
        return self.base / "fusion"

    @property
    def reports_dir(self) -> Path:
        return self.base / "reports"

    @staticmethod
    def _check_split(split: str) -> str:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return split

    def prediction_path(self, modality: str, split: str) -> Path:
        return self.predictions_dir / f"{modality}__{self._check_split(split)}.parquet"

    def prediction_meta_path(self, modality: str, split: str) -> Path:
        return self.predictions_dir / f"{modality}__{self._check_split(split)}.json"

    def calibration_path(self, modality: str) -> Path:
        return self.calibration_dir / f"{modality}.json"

    @property
    def temperature_path(self) -> Path:
        return self.calibration_dir / "temperature.json"

    @property
    def alignment_matrix_path(self) -> Path:
        return self.alignment_dir / "alignment_matrix.json"

    @property
    def alignment_report_path(self) -> Path:
        return self.alignment_dir / "alignment_report.json"

    @staticmethod
    def pair_label(modalities: tuple[str, ...] | list[str]) -> str:
        """Stable directory name for a modality set, e.g. ``audio+text``."""
        names = list(modalities)
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate modality in {names}")
        if not names:
            raise ValueError("A fusion set needs at least one modality")
        return "+".join(names)

    def fusion_pair_dir(
        self, modalities: tuple[str, ...] | list[str], variant: str | None = None
    ) -> Path:
        """``fusion/audio+text``, or ``fusion/audio+text__log_opinion_pool``.

        A ``variant`` keeps two fusion rules over the same pair from
        overwriting each other: they are different experiments and their
        artefacts have to be comparable side by side.
        """
        name = self.pair_label(modalities)
        if variant:
            if not _RUN_PATTERN.match(variant):
                raise ValueError(f"Invalid fusion variant name: {variant!r}")
            name = f"{name}__{variant}"
        return self.fusion_dir / name

    def fusion_metrics_path(self, modalities, split: str, variant: str | None = None) -> Path:
        return (
            self.fusion_pair_dir(modalities, variant)
            / f"{self._check_split(split)}_metrics.json"
        )

    def fusion_summary_path(self, modalities, variant: str | None = None) -> Path:
        """Both splits of one pair in a single record, weights included."""
        return self.fusion_pair_dir(modalities, variant) / "fusion_summary.json"

    def fusion_predictions_path(self, modalities, split: str, variant: str | None = None) -> Path:
        return (
            self.fusion_pair_dir(modalities, variant)
            / f"{self._check_split(split)}_predictions.parquet"
        )

    def delta_uncertainty_path(self, modalities, split: str, variant: str | None = None) -> Path:
        return (
            self.fusion_pair_dir(modalities, variant)
            / f"{self._check_split(split)}_delta_uncertainty.parquet"
        )

    def routing_log_path(self, name: str = "routing") -> Path:
        return self.reports_dir / f"{name}.jsonl"

    def report_path(self, name: str) -> Path:
        return self.reports_dir / name

    # ----------------------------------------------------------- filesystem

    def prepare(self) -> Path:
        for directory in (
            self.base, self.predictions_dir, self.alignment_dir,
            self.calibration_dir, self.fusion_dir, self.reports_dir,
        ):
            ensure_dir(directory)
        return self.base

    def existing_predictions(self) -> list[tuple[str, str]]:
        """Return ``(modality, split)`` pairs already exported for this run."""
        if not self.predictions_dir.exists():
            return []
        found = []
        for path in sorted(self.predictions_dir.glob("*__*.parquet")):
            modality, _, split = path.stem.partition("__")
            found.append((modality, split))
        return found
