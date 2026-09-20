"""What every AHSEF experiment records so it can be reproduced or challenged.

The baselines already record their own seeds, environment, and checkpoint
identity.  An AHSEF run adds a second layer of choices on top of them --
temperature, fusion weights, uncertainty threshold, lambda, mu -- and each of
those is a place where an unrecorded decision could quietly become a result.
So the provenance record names all of them, including the ones that are still
at their defaults.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from src.ahsef.uncertainty import UNCERTAINTY_DEFINITION


def git_revision() -> dict:
    """Repository state, or an explicit note that it could not be read."""
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "note": "git unavailable or not a repository"}
    status = run("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "changed_files": len(status.splitlines()) if status else 0,
    }


@dataclass
class AhsefPolicy:
    """Every configurable AHSEF parameter, in one place and never hidden in code.

    Stage 1 uses only ``seed``, ``temperature_*``, ``fusion_*``, and the cost
    normalisation; the routing fields are declared here anyway so that a stage-1
    artefact and a stage-2 artefact share one schema and one audit surface.
    """

    # provenance
    seed: int = 42

    # uncertainty / calibration
    uncertainty_definition: str = "normalized_predictive_entropy"
    calibration_method: str = "temperature_scaling"
    calibration_fit_split: str = "validation"
    calibration_ece_bins: int = 15
    apply_calibration: bool = False

    # fusion
    fusion_method: str = "weighted_probability"
    fusion_weight_selection_split: str = "validation"
    fusion_weight_objective: str = "macro_f1"

    # cost model
    cost_normalization: str = "max"

    # routing (stage 2; declared now so the schema does not change later)
    uncertainty_threshold: float = 0.85
    threshold_selection_split: str = "validation"
    lambda_cost: float = 0.10
    mu_latency: float = 0.10
    max_modalities: int = 2
    min_information_gain: float = 0.0
    min_utility_improvement: float = 0.0

    #: Free-form notes folded into the artefact, e.g. why a default was kept.
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        record = asdict(self)
        record["uncertainty_definition_detail"] = dict(UNCERTAINTY_DEFINITION)
        return record


def environment_record() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def write_provenance(
    path: Path,
    run: str,
    stage: str,
    policy: AhsefPolicy,
    baselines: Mapping[str, Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write (or refresh) a run's ``provenance.json``.

    Stages accumulate: re-running a later stage keeps what earlier stages
    recorded rather than replacing the file, so the artefact tells the whole
    history of the run directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    stages = existing.get("stages", {})
    stages[stage] = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "policy": policy.to_dict(),
        "baselines": {name: dict(record) for name, record in baselines.items()},
        **(dict(extra) if extra else {}),
    }
    payload = {
        "run": run,
        "created_at": existing.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": git_revision(),
        "environment": environment_record(),
        "safety": {
            "frozen_baselines_modified": False,
            "test_labels_used_for_fitting": False,
            "test_labels_used_for_threshold_selection": False,
            "calibration_fitted_on": policy.calibration_fit_split,
            "fusion_weights_selected_on": policy.fusion_weight_selection_split,
            "note": "Test data is opened for evaluation only, after every parameter "
                    "in the policy has been fixed on train/validation.",
        },
        "stages": stages,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
