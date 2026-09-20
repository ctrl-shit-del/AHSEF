"""A single estimator family, so the Phase D comparison isolates the target.

Phase D asks which *routing target* to regress.  That question is only
answerable if everything else is held fixed, so every candidate target is fitted
with the same estimator, the same features, the same folds and the same seed.
If the estimator family varied with the target, a difference between two rows
would be a difference between two experiments rather than between two targets.

The family is **ridge regression on standardised features**.  Three reasons:

* Every candidate target is real-valued (``binary_correction`` is 0/1,
  ``signed_gain`` is −1/0/+1, ``macro_f1_marginal`` is a small signed float), and
  ridge handles all of them without a per-target choice of link function.
* Only the *ranking* of the predicted score is used -- the budget curve acquires
  top-k -- so a calibrated probability buys nothing here that a monotone score
  does not.
* It is linear, so the frozen artefact is a list of coefficients that can be read
  and audited, exactly as Stage 3's estimator was.

The regularisation strength is chosen once, on validation, by out-of-fold
Spearman correlation with the target, and the same value is used for every
target so the comparison stays controlled.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ahsef.stage3.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    assert_label_free,
)

MILESTONE_HSIG_SCHEMA = "ahsef.milestone.hsig.v1"

#: Splits an estimator may be fitted on. ``test`` is not one of them.
FIT_SPLITS = ("train", "validation")

#: Ridge strengths swept on validation. One value is chosen and shared by every
#: target, so the Phase D comparison varies only the target.
ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

DEFAULT_FOLDS = 5


class EstimatorLeakageError(RuntimeError):
    """Raised when an estimator would be fitted on evaluation data."""


def assert_fit_split(split: str) -> str:
    if split not in FIT_SPLITS:
        raise EstimatorLeakageError(
            f"A routing estimator may only be fitted on {FIT_SPLITS}; refusing to "
            f"fit on {split!r}."
        )
    return split


def _pipeline(alpha: float, seed: int):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=float(alpha), random_state=seed)),
    ])


def _matrix(
    features: pd.DataFrame, sample_ids: Sequence[str], feature_set: str
) -> tuple[np.ndarray, list[str]]:
    names = list(FEATURE_SETS[feature_set])
    assert_label_free(features, names)
    indexed = features.set_index(features["sample_id"].astype(str))
    unknown = [item for item in sample_ids if item not in indexed.index]
    if unknown:
        raise KeyError(f"No features for {len(unknown)} ids (e.g. {unknown[:5]})")
    matrix = indexed.loc[list(sample_ids), names].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        incomplete = int((~np.isfinite(matrix).all(axis=1)).sum())
        raise ValueError(
            f"{incomplete} rows have non-finite features. Nothing is imputed; "
            f"exclude them explicitly and report the exclusion."
        )
    return matrix, names


def _folds(size: int, folds: int, seed: int) -> list[np.ndarray]:
    """Seeded, contiguous-after-shuffle folds. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(size)
    return [part for part in np.array_split(order, min(folds, size)) if part.size]


def out_of_fold_scores(
    features: pd.DataFrame,
    sample_ids: Sequence[str],
    target: np.ndarray,
    split: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    alpha: float = 1.0,
    folds: int = DEFAULT_FOLDS,
    seed: int = 42,
) -> np.ndarray:
    """A score for every sample, from a model that never saw that sample.

    This is what the budget curve consumes. Using in-sample predictions here
    would make every policy look better than it is, and by an amount that grows
    with the number of features.
    """
    assert_fit_split(split)
    matrix, _ = _matrix(features, sample_ids, feature_set)
    target = np.asarray(target, dtype=float)
    if target.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"target has {target.shape[0]} rows but features have {matrix.shape[0]}"
        )
    scores = np.full(matrix.shape[0], np.nan)
    for held_out in _folds(matrix.shape[0], folds, seed):
        mask = np.ones(matrix.shape[0], dtype=bool)
        mask[held_out] = False
        if np.unique(target[mask]).size < 2:
            # A fold whose training half is constant cannot fit a direction;
            # predicting that constant is the honest answer, not an error.
            scores[held_out] = float(target[mask].mean()) if mask.any() else 0.0
            continue
        model = _pipeline(alpha, seed)
        model.fit(matrix[mask], target[mask])
        scores[held_out] = model.predict(matrix[held_out])
    return scores


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.size < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        raw = np.empty(values.size, dtype=float)
        raw[order] = np.arange(1, values.size + 1, dtype=float)
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, raw)
        return (sums / counts)[inverse]

    value = float(np.corrcoef(ranks(left), ranks(right))[0, 1])
    return None if np.isnan(value) else value


def select_alpha(
    features: pd.DataFrame,
    sample_ids: Sequence[str],
    target: np.ndarray,
    split: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    folds: int = DEFAULT_FOLDS,
    seed: int = 42,
    grid: Sequence[float] = ALPHA_GRID,
) -> dict:
    """Pick the ridge strength on validation by out-of-fold rank correlation."""
    assert_fit_split(split)
    scored = {}
    for alpha in grid:
        scores = out_of_fold_scores(
            features, sample_ids, target, split, feature_set, alpha, folds, seed
        )
        scored[float(alpha)] = spearman(scores, target)
    usable = {a: v for a, v in scored.items() if v is not None}
    best = max(usable, key=lambda a: usable[a]) if usable else float(grid[0])
    return {
        "selected_alpha": best,
        "grid": {str(a): v for a, v in scored.items()},
        "criterion": "out-of-fold Spearman correlation with the target, on validation",
        "selected_on_split": split,
        "uses_test_labels": False,
    }


# ============================================================
# The fitted, serialisable estimator
# ============================================================

@dataclass
class MilestoneHSIG:
    """A fitted ridge estimator, stored as auditable coefficients."""

    target_name: str
    feature_set: str = DEFAULT_FEATURE_SET
    alpha: float = 1.0
    seed: int = 42
    fitted_on_split: str = "validation"
    feature_names: list[str] = field(default_factory=list)
    scaler_mean: list[float] = field(default_factory=list)
    scaler_scale: list[float] = field(default_factory=list)
    coefficients: list[float] = field(default_factory=list)
    intercept: float = 0.0
    training_summary: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    created_at: str = ""

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        sample_ids: Sequence[str],
        target: np.ndarray,
        target_name: str,
        split: str,
        feature_set: str = DEFAULT_FEATURE_SET,
        alpha: float = 1.0,
        seed: int = 42,
    ) -> "MilestoneHSIG":
        assert_fit_split(split)
        matrix, names = _matrix(features, sample_ids, feature_set)
        target = np.asarray(target, dtype=float)
        model = _pipeline(alpha, seed)
        model.fit(matrix, target)
        scaler = model.named_steps["scale"]
        ridge = model.named_steps["model"]
        return cls(
            target_name=target_name, feature_set=feature_set, alpha=float(alpha),
            seed=seed, fitted_on_split=split, feature_names=list(names),
            scaler_mean=[float(v) for v in scaler.mean_],
            scaler_scale=[float(v) for v in scaler.scale_],
            coefficients=[float(v) for v in np.ravel(ridge.coef_)],
            intercept=float(np.ravel(ridge.intercept_)[0]),
            training_summary={
                "samples": int(matrix.shape[0]),
                "target_mean": float(target.mean()),
                "target_std": float(target.std(ddof=0)),
                "distinct_target_values": int(np.unique(target).size),
            },
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def predict(self, features: pd.DataFrame, sample_ids: Sequence[str]) -> np.ndarray:
        """Score from the stored coefficients alone -- no scikit-learn needed."""
        matrix, names = _matrix(features, sample_ids, self.feature_set)
        if names != self.feature_names:
            raise ValueError(
                f"Feature order changed since fitting: expected {self.feature_names}, "
                f"got {names}"
            )
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        scaled = (matrix - mean) / np.where(scale == 0, 1.0, scale)
        return scaled @ np.asarray(self.coefficients, dtype=float) + self.intercept

    def importance(self) -> dict:
        pairs = dict(zip(self.feature_names, self.coefficients))
        return {
            "coefficients": pairs,
            "ranked_by_absolute_weight": sorted(
                pairs, key=lambda name: abs(pairs[name]), reverse=True
            ),
            "intercept": self.intercept,
            "interpretation": (
                "Features are standardised before fitting, so a coefficient is the "
                "change in predicted routing value per standard deviation of that "
                "feature and the magnitudes are directly comparable."
            ),
        }

    def to_dict(self) -> dict:
        return {
            "schema": MILESTONE_HSIG_SCHEMA,
            "target_name": self.target_name,
            "estimator": "ridge",
            "feature_set": self.feature_set,
            "feature_names": list(self.feature_names),
            "alpha": self.alpha,
            "seed": self.seed,
            "fitted_on_split": self.fitted_on_split,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "training_summary": dict(self.training_summary),
            "quality": dict(self.quality),
            "created_at": self.created_at,
            "uses_test_labels": False,
            "serialisation_note": (
                "Stored as plain coefficients rather than a pickle so a routing "
                "decision can be audited, and replayed, without scikit-learn."
            ),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "MilestoneHSIG":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("schema") != MILESTONE_HSIG_SCHEMA:
            raise ValueError(
                f"{path} has schema {record.get('schema')!r}, expected "
                f"{MILESTONE_HSIG_SCHEMA!r}"
            )
        return cls(
            target_name=record["target_name"], feature_set=record["feature_set"],
            alpha=float(record["alpha"]), seed=int(record["seed"]),
            fitted_on_split=record["fitted_on_split"],
            feature_names=list(record["feature_names"]),
            scaler_mean=list(record["scaler_mean"]),
            scaler_scale=list(record["scaler_scale"]),
            coefficients=list(record["coefficients"]),
            intercept=float(record["intercept"]),
            training_summary=dict(record.get("training_summary") or {}),
            quality=dict(record.get("quality") or {}),
            created_at=record.get("created_at", ""),
        )

    def fingerprint(self) -> str:
        payload = json.dumps({
            "target": self.target_name, "feature_set": self.feature_set,
            "features": list(self.feature_names),
            "coefficients": list(self.coefficients), "intercept": self.intercept,
            "scaler_mean": list(self.scaler_mean), "scaler_scale": list(self.scaler_scale),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
