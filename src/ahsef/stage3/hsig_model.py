"""PHASE D -- a real, validation-trained HSIG estimator.

Stage 2 shipped only :class:`~src.ahsef.hsig.UnavailableHSIG`, which declines to
guess.  This module replaces it with an estimator that answers the question the
contract poses::

    gain(audio | text) = P(correct | text, audio) - P(correct | text)

and it estimates exactly that, rather than a convenient proxy for it.

Why the difference of two classifiers
-------------------------------------
The target is a *difference* of two probabilities, and its per-sample
realisation ``signed_gain = fused_correct - text_correct`` takes three values.
Two estimators are offered:

``paired_logistic`` (the shape the target has)
    fit ``P(text correct | x)`` and ``P(fused correct | x)`` separately and
    subtract.  This is the definition, written down.  It uses every sample, and
    it can predict a *negative* gain -- which matters, because the acquisitions
    a routing policy most needs to avoid are the ones that break an already
    correct prediction.

``fix_logistic`` (the convenient proxy)
    a single classifier on the binary event "audio fixed a wrong text
    prediction".  Simpler, but structurally blind to harm: its output is a
    probability in ``[0, 1]`` and can never say "acquiring will hurt".

Both are fitted, both are scored out of fold, and the choice is recorded.

What is *not* used
------------------
Delta-uncertainty.  Stage 1 measured that its sign flips with the anchor and the
fusion rule and that its peak per-sample correlation with real improvement was
0.143.  :func:`assert_target_not_delta_uncertainty` refuses a target column
whose name says otherwise, so the rejected target cannot creep back in.

Honesty about small data
------------------------
The aligned pool is 509 validation samples.  Every quality number reported here
is **out of fold** (seeded stratified K-fold), because an in-sample AUROC on 509
rows with ten features is optimistic by an amount that is not small.  Two
deliberately weak reference estimators -- a constant at the base rate, and a
logistic on uncertainty alone -- are scored the same way, so "HSIG works" is
always a comparison and never an assertion.

Serialisation is to JSON coefficients, not to a pickle: a routing decision has
to be auditable from the artefact, and a reader must be able to see the weight
on every feature.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ahsef.hsig import HSIG_TARGET, HSIG_TARGET_DEFINITION, GainEstimate, RoutingState
from src.ahsef.stage3.features import (
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    assert_label_free,
    feature_provenance,
)

#: Splits an HSIG estimator may be fitted on.  ``test`` is not one of them.
HSIG_FIT_SPLITS = ("train", "validation")

ESTIMATOR_TYPES = ("paired_logistic", "fix_logistic")

#: Reference estimators HSIG must beat before it may be called useful.
REFERENCE_TYPES = ("prior_constant", "uncertainty_only_logistic")

HSIG_MODEL_SCHEMA = "ahsef.stage3.hsig.v1"


class HSIGLeakageError(RuntimeError):
    """Raised when an HSIG estimator would be fitted on evaluation data."""


class HSIGTargetError(RuntimeError):
    """Raised when the rejected delta-uncertainty target is offered to HSIG."""


class HSIGNotFittedError(RuntimeError):
    """Raised when an estimate is requested from an unfitted estimator."""


def assert_target_not_delta_uncertainty(target_name: str) -> None:
    if "uncertainty" in target_name.lower() and "gain" not in target_name.lower():
        raise HSIGTargetError(
            f"Refusing to fit HSIG on {target_name!r}. Stage 1 rejected the "
            f"delta-uncertainty target: {HSIG_TARGET_DEFINITION['why_rejected']}"
        )


def assert_fit_split(split: str) -> str:
    if split not in HSIG_FIT_SPLITS:
        raise HSIGLeakageError(
            f"HSIG may only be fitted on {HSIG_FIT_SPLITS}; refusing to fit on "
            f"{split!r}. An estimator that has seen the evaluation labels makes "
            f"every routing number that follows meaningless."
        )
    return split


# ============================================================
# Targets
# ============================================================

@dataclass(frozen=True)
class GainTargets:
    """The supervision HSIG is fitted against, built from the oracle table."""

    sample_ids: list[str]
    text_correct: np.ndarray
    fused_correct: np.ndarray
    split: str

    def __post_init__(self) -> None:
        if not (len(self.sample_ids) == self.text_correct.size == self.fused_correct.size):
            raise ValueError("Target arrays and sample ids must be the same length")

    @property
    def signed_gain(self) -> np.ndarray:
        """``P(correct | A u {m}) - P(correct | A)``, realised per sample."""
        return self.fused_correct.astype(int) - self.text_correct.astype(int)

    @property
    def y_gain(self) -> np.ndarray:
        """Audio fixed a wrong text prediction."""
        return ((~self.text_correct.astype(bool)) & self.fused_correct.astype(bool)).astype(int)

    @property
    def y_harm(self) -> np.ndarray:
        return (self.text_correct.astype(bool) & (~self.fused_correct.astype(bool))).astype(int)

    def summary(self) -> dict:
        total = len(self.sample_ids)
        return {
            "split": self.split,
            "samples": total,
            "text_correct": int(self.text_correct.sum()),
            "fused_correct": int(self.fused_correct.sum()),
            "y_gain_positives": int(self.y_gain.sum()),
            "y_gain_rate": float(self.y_gain.mean()) if total else None,
            "y_harm_positives": int(self.y_harm.sum()),
            "y_harm_rate": float(self.y_harm.mean()) if total else None,
            "mean_signed_gain": float(self.signed_gain.mean()) if total else None,
            "class_imbalance_note": (
                "AUPRC is reported alongside AUROC because the positive event is a "
                "minority; on an imbalanced target AUROC alone flatters a model that "
                "would be useless at the operating point."
            ),
            **HSIG_TARGET_DEFINITION,
        }


def targets_from_oracle(oracle: pd.DataFrame, split: str) -> GainTargets:
    for column in ("sample_id", "text_correct", "fused_correct"):
        if column not in oracle.columns:
            raise ValueError(f"The oracle table is missing {column!r}")
    return GainTargets(
        sample_ids=oracle["sample_id"].astype(str).tolist(),
        text_correct=oracle["text_correct"].to_numpy(dtype=bool),
        fused_correct=oracle["fused_correct"].to_numpy(dtype=bool),
        split=split,
    )


# ============================================================
# The estimator
# ============================================================

def _pipeline(seed: int, penalty_c: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000, C=penalty_c, random_state=seed, solver="lbfgs",
        )),
    ])


def _linear_record(pipeline, names: Sequence[str]) -> dict:
    """Extract the fitted linear model as plain numbers, for JSON and for audit."""
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    return {
        "feature_names": list(names),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in model.coef_.ravel()],
        "intercept": float(model.intercept_.ravel()[0]),
        "classes": [int(value) for value in model.classes_],
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _apply_linear(record: Mapping, matrix: np.ndarray) -> np.ndarray:
    """Score a matrix from the serialised coefficients alone.

    Deliberately independent of scikit-learn: a frozen routing policy must be
    replayable from the JSON artefact, not from a library version.
    """
    mean = np.asarray(record["scaler_mean"], dtype=float)
    scale = np.asarray(record["scaler_scale"], dtype=float)
    coefficients = np.asarray(record["coefficients"], dtype=float)
    scaled = (matrix - mean) / np.where(scale == 0, 1.0, scale)
    probability = _sigmoid(scaled @ coefficients + float(record["intercept"]))
    # LogisticRegression.classes_ is sorted, so predict_proba[:, 1] is P(class=1)
    # only when 1 is the second class. A degenerate single-class fit is refused
    # upstream rather than silently inverted here.
    if list(record["classes"]) == [0, 1]:
        return probability
    if list(record["classes"]) == [1, 0]:  # pragma: no cover - sklearn sorts
        return 1.0 - probability
    raise HSIGNotFittedError(
        f"A binary component was fitted on classes {record['classes']}; a "
        f"single-class component cannot produce a probability."
    )


@dataclass
class Stage3HSIG:
    """A fitted expected-gain estimator satisfying the :class:`HSIG` protocol."""

    name: str = "stage3_text_audio_hsig"
    estimator_type: str = "paired_logistic"
    feature_set: str = DEFAULT_FEATURE_SET
    fitted_on_split: str = "validation"
    seed: int = 42
    penalty_c: float = 1.0
    #: Serialised linear components, keyed by role.
    components: dict = field(default_factory=dict)
    training_summary: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    created_at: str = ""

    # ------------------------------------------------------------------- fit

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        targets: GainTargets,
        feature_set: str = DEFAULT_FEATURE_SET,
        estimator_type: str = "paired_logistic",
        seed: int = 42,
        penalty_c: float = 1.0,
    ) -> "Stage3HSIG":
        assert_fit_split(targets.split)
        assert_target_not_delta_uncertainty(HSIG_TARGET)
        if estimator_type not in ESTIMATOR_TYPES:
            raise ValueError(
                f"estimator_type must be one of {ESTIMATOR_TYPES}, got {estimator_type!r}"
            )
        matrix, names = _aligned_matrix(features, targets.sample_ids, feature_set)

        components: dict[str, dict] = {}
        if estimator_type == "paired_logistic":
            for role, outcome in (
                ("text_correct", targets.text_correct.astype(int)),
                ("fused_correct", targets.fused_correct.astype(int)),
            ):
                components[role] = _fit_component(matrix, outcome, names, seed, penalty_c, role)
        else:
            components["y_gain"] = _fit_component(
                matrix, targets.y_gain, names, seed, penalty_c, "y_gain"
            )

        return cls(
            estimator_type=estimator_type, feature_set=feature_set,
            fitted_on_split=targets.split, seed=seed, penalty_c=penalty_c,
            components=components, training_summary=targets.summary(),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # -------------------------------------------------------------- predict

    def predict_gain(self, matrix: np.ndarray) -> np.ndarray:
        """Expected decision improvement for each row, in ``[-1, 1]``."""
        if not self.components:
            raise HSIGNotFittedError("This HSIG has no fitted components")
        if self.estimator_type == "paired_logistic":
            return (
                _apply_linear(self.components["fused_correct"], matrix)
                - _apply_linear(self.components["text_correct"], matrix)
            )
        return _apply_linear(self.components["y_gain"], matrix)

    def predict_frame(
        self, features: pd.DataFrame, sample_ids: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Per-sample gain, with incomplete rows marked rather than imputed."""
        ids = list(sample_ids) if sample_ids is not None \
            else features["sample_id"].astype(str).tolist()
        matrix, _ = _aligned_matrix(features, ids, self.feature_set, require_complete=False)
        complete = np.isfinite(matrix).all(axis=1)
        gains = np.full(matrix.shape[0], np.nan)
        if complete.any():
            gains[complete] = self.predict_gain(matrix[complete])
        return pd.DataFrame({
            "sample_id": ids,
            "predicted_gain": gains,
            "features_complete": complete,
        })

    # ------------------------------------------------------- HSIG protocol

    def estimate_gain(self, state: RoutingState, candidate_modality: str) -> GainEstimate:
        if candidate_modality != "audio":
            return GainEstimate(
                modality=candidate_modality, expected_improvement=None, available=False,
                basis=(
                    f"This estimator was fitted for the candidate 'audio' only; it has "
                    f"no evidence about {candidate_modality!r} and will not extrapolate."
                ),
            )
        names = list(FEATURE_SETS[self.feature_set])
        missing = [name for name in names if name not in state.features]
        row = np.array(
            [[float(state.features.get(name, np.nan)) for name in names]], dtype=float
        )
        if missing or not np.isfinite(row).all():
            return GainEstimate(
                modality=candidate_modality, expected_improvement=None, available=False,
                basis=(
                    f"Incomplete evidence state for sample {state.sample_id!r}: "
                    f"{missing or 'non-finite feature values'}. No estimate is given "
                    f"and nothing is imputed."
                ),
            )
        value = float(self.predict_gain(row)[0])
        return GainEstimate(
            modality=candidate_modality, expected_improvement=value, available=True,
            basis=(
                f"{self.estimator_type} over feature set {self.feature_set!r}, fitted on "
                f"{self.fitted_on_split} only; target {HSIG_TARGET}"
            ),
        )

    def estimate_all(
        self, state: RoutingState, candidates: Sequence[str]
    ) -> dict[str, GainEstimate]:
        return {name: self.estimate_gain(state, name) for name in candidates}

    def provenance(self) -> dict:
        return {
            "hsig": self.name,
            "implemented": True,
            "estimator_type": self.estimator_type,
            "fitted_on_split": self.fitted_on_split,
            "seed": self.seed,
            "penalty_c": self.penalty_c,
            "features": feature_provenance(self.feature_set),
            "coefficients": {
                role: dict(zip(record["feature_names"], record["coefficients"]))
                for role, record in self.components.items()
            },
            "training_summary": dict(self.training_summary),
            "quality": dict(self.quality),
            **HSIG_TARGET_DEFINITION,
        }

    # --------------------------------------------------------------- io

    def to_dict(self) -> dict:
        return {
            "schema": HSIG_MODEL_SCHEMA,
            "name": self.name,
            "estimator_type": self.estimator_type,
            "feature_set": self.feature_set,
            "features": list(FEATURE_SETS[self.feature_set]),
            "fitted_on_split": self.fitted_on_split,
            "seed": self.seed,
            "penalty_c": self.penalty_c,
            "components": self.components,
            "training_summary": dict(self.training_summary),
            "quality": dict(self.quality),
            "created_at": self.created_at,
            "target": HSIG_TARGET,
            "target_definition": HSIG_TARGET_DEFINITION,
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
    def load(cls, path: Path) -> "Stage3HSIG":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("schema") != HSIG_MODEL_SCHEMA:
            raise ValueError(
                f"{path} has schema {record.get('schema')!r}, expected {HSIG_MODEL_SCHEMA!r}"
            )
        return cls(
            name=record["name"], estimator_type=record["estimator_type"],
            feature_set=record["feature_set"], fitted_on_split=record["fitted_on_split"],
            seed=int(record["seed"]), penalty_c=float(record["penalty_c"]),
            components=record["components"],
            training_summary=dict(record.get("training_summary") or {}),
            quality=dict(record.get("quality") or {}),
            created_at=record.get("created_at", ""),
        )

    def fingerprint(self) -> str:
        """Hash of the coefficients -- proves the frozen estimator was not refitted."""
        import hashlib

        payload = json.dumps(
            {"type": self.estimator_type, "features": list(FEATURE_SETS[self.feature_set]),
             "components": self.components},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_component(
    matrix: np.ndarray, outcome: np.ndarray, names, seed: int, penalty_c: float, role: str
) -> dict:
    if np.unique(outcome).size < 2:
        raise HSIGNotFittedError(
            f"The {role!r} component has a single outcome value on this split; a "
            f"probability cannot be fitted from it, and a constant would be a "
            f"fabricated estimate."
        )
    pipeline = _pipeline(seed, penalty_c)
    pipeline.fit(matrix, outcome)
    return _linear_record(pipeline, names)


def _aligned_matrix(
    features: pd.DataFrame,
    sample_ids: Sequence[str],
    feature_set: str,
    require_complete: bool = True,
) -> tuple[np.ndarray, list[str]]:
    names = list(FEATURE_SETS[feature_set])
    assert_label_free(features, names)
    indexed = features.set_index(features["sample_id"].astype(str))
    unknown = [item for item in sample_ids if item not in indexed.index]
    if unknown:
        raise KeyError(
            f"No features for {len(unknown)} requested sample ids (e.g. {unknown[:5]})"
        )
    matrix = indexed.loc[list(sample_ids), names].to_numpy(dtype=float)
    if require_complete and not np.isfinite(matrix).all():
        incomplete = int((~np.isfinite(matrix).all(axis=1)).sum())
        raise ValueError(
            f"{incomplete} rows have non-finite features. HSIG does not impute; "
            f"exclude them explicitly and report the exclusion."
        )
    return matrix, names


# ============================================================
# Reference estimators
# ============================================================

def prior_constant_gain(targets: GainTargets, n: int) -> np.ndarray:
    """The base rate, repeated.  The floor any estimator must clear."""
    return np.full(n, float(targets.signed_gain.mean()))


def uncertainty_only_gain(
    features: pd.DataFrame, targets: GainTargets, seed: int = 42
) -> np.ndarray:
    """Out-of-fold gain from uncertainty alone -- Stage 2's implicit policy.

    Stage 2's conclusion was that the LLM's uncertainty gate is weak. This is
    that gate re-expressed as a gain estimate, so the Stage 3 estimator is
    compared against the thing it is meant to improve on.
    """
    matrix = _aligned_matrix(features, targets.sample_ids, "evidence_only")[0][:, :1]
    return _out_of_fold_paired(matrix, targets, seed)


# ============================================================
# Out-of-fold evaluation
# ============================================================

def _folds(stratify: np.ndarray, folds: int, seed: int):
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(stratify.astype(int))
    usable = int(min(folds, counts[counts > 0].min()))
    if usable < 2:
        raise HSIGNotFittedError(
            f"The rarest stratum has {counts[counts > 0].min()} member(s); "
            f"out-of-fold evaluation needs at least 2. The pool is too small to "
            f"report an honest quality number for this target."
        )
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed), usable


def _out_of_fold_paired(matrix: np.ndarray, targets: GainTargets, seed: int, folds: int = 5):
    splitter, _ = _folds(targets.y_gain, folds, seed)
    gains = np.full(matrix.shape[0], np.nan)
    for train_index, test_index in splitter.split(matrix, targets.y_gain):
        parts = {}
        for role, outcome in (
            ("text_correct", targets.text_correct.astype(int)),
            ("fused_correct", targets.fused_correct.astype(int)),
        ):
            fitted = _pipeline(seed, 1.0)
            fitted.fit(matrix[train_index], outcome[train_index])
            parts[role] = fitted.predict_proba(matrix[test_index])[:, 1]
        gains[test_index] = parts["fused_correct"] - parts["text_correct"]
    return gains


def out_of_fold_gain(
    features: pd.DataFrame,
    targets: GainTargets,
    feature_set: str = DEFAULT_FEATURE_SET,
    estimator_type: str = "paired_logistic",
    seed: int = 42,
    folds: int = 5,
    penalty_c: float = 1.0,
) -> np.ndarray:
    """Gain predicted for each sample by a model that never saw that sample.

    This -- not the in-sample fit -- is what the threshold is selected on and
    what every reported HSIG quality number is computed from.  At n=509 the
    difference between the two is large enough to change the conclusion.
    """
    assert_fit_split(targets.split)
    matrix, _ = _aligned_matrix(features, targets.sample_ids, feature_set)
    if estimator_type == "paired_logistic":
        return _out_of_fold_paired(matrix, targets, seed, folds)

    splitter, _ = _folds(targets.y_gain, folds, seed)
    gains = np.full(matrix.shape[0], np.nan)
    for train_index, test_index in splitter.split(matrix, targets.y_gain):
        fitted = _pipeline(seed, penalty_c)
        fitted.fit(matrix[train_index], targets.y_gain[train_index])
        gains[test_index] = fitted.predict_proba(matrix[test_index])[:, 1]
    return gains


def shallow_forest_gain(
    features: pd.DataFrame,
    targets: GainTargets,
    feature_set: str = DEFAULT_FEATURE_SET,
    seed: int = 42,
    folds: int = 5,
    max_depth: int = 3,
    n_estimators: int = 200,
) -> np.ndarray:
    """Out-of-fold gain from a shallow forest, as a non-linear reference point.

    Reported, never frozen: a forest cannot be serialised as auditable
    coefficients, and Stage 3 requires that every routing decision be readable
    from the artefact. If the forest wins materially, that is a finding to be
    stated, not a reason to swap it in silently.
    """
    from sklearn.ensemble import RandomForestClassifier

    matrix, _ = _aligned_matrix(features, targets.sample_ids, feature_set)
    splitter, _ = _folds(targets.y_gain, folds, seed)
    gains = np.full(matrix.shape[0], np.nan)
    for train_index, test_index in splitter.split(matrix, targets.y_gain):
        parts = {}
        for role, outcome in (
            ("text_correct", targets.text_correct.astype(int)),
            ("fused_correct", targets.fused_correct.astype(int)),
        ):
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=seed,
                min_samples_leaf=10, n_jobs=1,
            )
            model.fit(matrix[train_index], outcome[train_index])
            parts[role] = model.predict_proba(matrix[test_index])[:, 1]
        gains[test_index] = parts["fused_correct"] - parts["text_correct"]
    return gains
