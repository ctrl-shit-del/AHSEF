"""Filesystem layout for the Stage 3 Text->Audio run.

Wraps :class:`~src.ahsef.layout.AhsefLayout` and adds the directories the
Stage 3 brief names, so no command has to spell a path out by hand::

    experiments/ahsef/stage3_text_audio/
        alignment/      the co-split Text+Audio pool and its verification
        predictions/    per-modality prediction sets on exactly that pool
        hsig/           features, targets, fitted estimator, quality report
        fusion/         fused prediction sets and the weight selection record
        routing/        per-sample routing decisions and traces
        reports/        validation experiment, ablations, locked test
        transcripts/    the LLM transcripts this stage recorded
        frozen_config.json

Nothing here writes into ``experiments/<modality>/`` or into the Stage 1 and
Stage 2 run directories: those are read-only inputs to this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ahsef.layout import DEFAULT_AHSEF_ROOT, SPLITS, AhsefLayout
from src.utils.io import ensure_dir


DEFAULT_STAGE3_RUN = "stage3_text_audio"

#: Runs Stage 3 reads but must never write to.
READ_ONLY_RUNS = ("stage1", "stage2_llm")


class FrozenArtefactError(RuntimeError):
    """Raised when a Stage 3 command would write into a frozen earlier stage."""


@dataclass(frozen=True)
class Stage3Layout:
    """Resolve every path belonging to the Stage 3 run."""

    run: str = DEFAULT_STAGE3_RUN
    root: Path = DEFAULT_AHSEF_ROOT

    def __post_init__(self) -> None:
        if self.run in READ_ONLY_RUNS:
            raise FrozenArtefactError(
                f"{self.run!r} is a frozen earlier stage. Stage 3 reads it and never "
                f"writes to it; choose a different --run."
            )
        object.__setattr__(self, "root", Path(self.root))

    @property
    def ahsef(self) -> AhsefLayout:
        """The Stage 1 layout object, reused for predictions and fusion paths."""
        return AhsefLayout(run=self.run, root=self.root)

    @property
    def base(self) -> Path:
        return self.root / self.run

    # ------------------------------------------------------------ directories

    @property
    def alignment_dir(self) -> Path:
        return self.base / "alignment"

    @property
    def predictions_dir(self) -> Path:
        return self.base / "predictions"

    @property
    def hsig_dir(self) -> Path:
        return self.base / "hsig"

    @property
    def fusion_dir(self) -> Path:
        return self.base / "fusion"

    @property
    def routing_dir(self) -> Path:
        return self.base / "routing"

    @property
    def reports_dir(self) -> Path:
        return self.base / "reports"

    @property
    def transcripts_dir(self) -> Path:
        return self.base / "transcripts"

    # ------------------------------------------------------------------ files

    @staticmethod
    def _check_split(split: str) -> str:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return split

    @property
    def frozen_config_path(self) -> Path:
        return self.base / "frozen_config.json"

    def pool_path(self, split: str) -> Path:
        return self.alignment_dir / f"pool_{self._check_split(split)}.json"

    @property
    def alignment_report_path(self) -> Path:
        return self.alignment_dir / "alignment_report.json"

    def prediction_path(self, modality: str, split: str) -> Path:
        return self.predictions_dir / f"{modality}__{self._check_split(split)}.parquet"

    def prediction_meta_path(self, modality: str, split: str) -> Path:
        return self.predictions_dir / f"{modality}__{self._check_split(split)}.json"

    def transcript_path(self, split: str) -> Path:
        return self.transcripts_dir / f"transcript_{self._check_split(split)}.jsonl"

    def timing_path(self, split: str) -> Path:
        """Wall-clock timing of the LLM pass, kept apart from generation latency."""
        return self.transcripts_dir / f"wallclock_{self._check_split(split)}.json"

    def oracle_path(self, split: str) -> Path:
        return self.hsig_dir / f"oracle_{self._check_split(split)}.parquet"

    def oracle_report_path(self, split: str) -> Path:
        return self.reports_dir / f"oracle_gain_{self._check_split(split)}.json"

    def features_path(self, split: str) -> Path:
        return self.hsig_dir / f"features_{self._check_split(split)}.parquet"

    @property
    def hsig_model_path(self) -> Path:
        return self.hsig_dir / "hsig_model.json"

    @property
    def hsig_report_path(self) -> Path:
        return self.reports_dir / "hsig_quality.json"

    @property
    def fusion_spec_path(self) -> Path:
        return self.fusion_dir / "fusion_spec.json"

    def fused_path(self, split: str) -> Path:
        return self.fusion_dir / f"fused_{self._check_split(split)}.parquet"

    def routing_decisions_path(self, split: str) -> Path:
        return self.routing_dir / f"decisions_{self._check_split(split)}.jsonl"

    def routing_traces_path(self, split: str) -> Path:
        return self.routing_dir / f"traces_{self._check_split(split)}.jsonl"

    def report_path(self, name: str) -> Path:
        return self.reports_dir / name

    # ------------------------------------------------------------- filesystem

    def prepare(self) -> Path:
        for directory in (
            self.base, self.alignment_dir, self.predictions_dir, self.hsig_dir,
            self.fusion_dir, self.routing_dir, self.reports_dir, self.transcripts_dir,
        ):
            ensure_dir(directory)
        return self.base
