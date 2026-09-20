"""Which frozen baseline AHSEF reads for each modality.

The five baselines live at different fractions (``audio_25pct`` but
``text_full``), solve two different tasks, and were selected on validation
macro-F1 by :mod:`src.training.experiment_summary`.  AHSEF needs one place
that says, for each modality: which experiment, which iteration, which label
column, and which label space.

Iteration defaults to the one the experiment summary already declared best on
*validation*.  That choice is therefore inherited, not re-made here, and in
particular is not re-made using test results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import EMOTION_7CLASS, LabelSpace, get_label_space


#: The AHSEF modality order.  Audio is first because it is the anchor modality
#: of the first routing study, not because the code depends on the order.
MODALITY_ORDER: tuple[str, ...] = ("audio", "text", "image", "video", "physiology")


@dataclass(frozen=True)
class BaselineRef:
    """A frozen unimodal baseline AHSEF may read but never write."""

    modality: str
    experiment: str
    label_column: str = "canonical_emotion_id"
    #: Filled in from the experiment summary when not given explicitly.
    iteration: str | None = None

    def layout(self, root: Path | str = "experiments") -> ExperimentLayout:
        return ExperimentLayout.from_name(self.experiment, Path(root))

    def resolve_iteration(self, root: Path | str = "experiments") -> str:
        """Return the configured iteration, or the summary's validation-best one."""
        if self.iteration is not None:
            return self.iteration
        summary_path = self.layout(root).experiment_summary_path
        if not summary_path.exists():
            raise FileNotFoundError(
                f"No experiment_summary.json at {summary_path}; pass an explicit "
                f"iteration for {self.modality}."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best_iteration") or {}
        iteration = best.get("iteration")
        if iteration is None:
            raise ValueError(
                f"{summary_path} declares no best_iteration; pass an explicit "
                f"iteration for {self.modality}."
            )
        return str(iteration)

    def run_summary_path(self, root: Path | str = "experiments", iteration: str | None = None) -> Path:
        layout = self.layout(root)
        return layout.results_dir(iteration or self.resolve_iteration(root)) / "run_summary.json"

    def model_record(
        self, root: Path | str = "experiments", iteration: str | None = None
    ) -> dict:
        """The architecture record the run wrote.

        The authoritative source of the geometry the cost proxy prices, so a
        prediction export of any vintage is costed the same way.
        """
        path = self.run_summary_path(root, iteration)
        if not path.exists():
            return {}
        return dict(json.loads(path.read_text(encoding="utf-8")).get("model") or {})

    def task(self, root: Path | str = "experiments") -> str:
        summary_path = self.layout(root).experiment_summary_path
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("task"):
                return str(summary["task"])
        return "emotion_7class"

    def label_space(self, root: Path | str = "experiments") -> LabelSpace:
        try:
            return get_label_space(self.task(root))
        except (KeyError, ValueError):
            return EMOTION_7CLASS


#: The frozen baselines this project has trained.  ``iteration=None`` means
#: "whichever iteration the experiment summary selected on validation".
DEFAULT_BASELINES: dict[str, BaselineRef] = {
    "audio": BaselineRef("audio", "audio_25pct"),
    "text": BaselineRef("text", "text_full"),
    "image": BaselineRef("image", "image_25pct"),
    "video": BaselineRef("video", "video_25pct"),
    "physiology": BaselineRef(
        # WESAD windows carry a study-condition id, not a canonical emotion id.
        "physiology", "physiology_full", label_column="state_id",
    ),
}


def baseline(modality: str) -> BaselineRef:
    try:
        return DEFAULT_BASELINES[modality]
    except KeyError as error:
        raise ValueError(
            f"Unknown modality {modality!r}; AHSEF knows {sorted(DEFAULT_BASELINES)}"
        ) from error


def ordered_modalities(names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Return modality names in the declared AHSEF order."""
    if names is None:
        return list(MODALITY_ORDER)
    unknown = [name for name in names if name not in DEFAULT_BASELINES]
    if unknown:
        raise ValueError(f"Unknown modalities {unknown}; known: {sorted(DEFAULT_BASELINES)}")
    return [name for name in MODALITY_ORDER if name in set(names)]
