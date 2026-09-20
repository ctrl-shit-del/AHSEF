"""PHASE D (inputs) -- the label-free description of the current evidence state.

Everything HSIG is allowed to condition on is declared here, in one place, so
that "does the estimator see the answer?" is a question about a list rather
than about the whole codebase.  Three rules hold without exception:

**No feature is derived from a label.**  Every column below is computable at
inference time from the LLM's own response.  :func:`assert_label_free` re-checks
the built frame against a deny-list of label-bearing column names, so a future
edit that reaches for ``true_class`` fails loudly instead of quietly inflating
every reported AUROC.

**No feature is imputed.**  A sample missing a feature gets no estimate: HSIG
returns "unavailable" and the router logs it.  Filling a gap with a mean or a
zero would make the estimator most confident exactly where it knows least.

**Class identity is optional and audited.**  Conditioning on the predicted class
is legitimate -- some classes genuinely benefit more from audio -- but it is
also the shortest path to a hard-coded routing rule wearing a learned model's
clothes.  So it lives in its own feature set, both sets are fitted, and the
choice between them is made on out-of-fold validation quality and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.ahsef.hsig import RoutingState
from src.common.labels import CANONICAL_EMOTION_CLASSES

#: Column names that carry ground truth.  A feature frame containing any of
#: these is rejected outright.
LABEL_BEARING_COLUMNS = frozenset({
    "true_class", "true_label", "canonical_emotion_id", "canonical_emotion",
    "emotion", "raw_emotion", "label", "y", "y_gain", "y_harm", "signed_gain",
    "text_correct", "audio_correct", "fused_correct", "correct",
    "fused_prediction", "audio_prediction",
})

#: Evidence-state features, all read off the LLM's own response.
EVIDENCE_FEATURES: tuple[str, ...] = (
    "uncertainty",
    "score_top1",
    "score_top2",
    "score_margin",
    "score_entropy",
    "score_mass_top2",
    "active_score_classes",
    "llm_confidence",
    "llm_ambiguity",
    "llm_evidence_strength",
)

#: One indicator per canonical class, for the class the LLM currently predicts.
CLASS_FEATURES: tuple[str, ...] = tuple(
    f"predicted_is_{name}" for name in CANONICAL_EMOTION_CLASSES
)

#: The simplest possible evidence state: Stage 2's routing signal, on its own.
#: It is a *candidate*, not merely a reference, because if a one-feature model
#: predicts the gain as well as a ten-feature one then the one-feature model is
#: what should be frozen -- and saying so is a result, not a disappointment.
UNCERTAINTY_ONLY: tuple[str, ...] = ("uncertainty",)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "uncertainty_only": UNCERTAINTY_ONLY,
    "evidence_only": EVIDENCE_FEATURES,
    "evidence_plus_class": EVIDENCE_FEATURES + CLASS_FEATURES,
}

DEFAULT_FEATURE_SET = "evidence_only"

#: Source columns each feature is read from, recorded so the frozen config can
#: state the feature definition rather than gesture at it.
FEATURE_DEFINITION = {
    "uncertainty": "routing uncertainty under the frozen policy (score_entropy -> "
                   "llm_normalized_score_entropy), in [0, 1]",
    "score_top1": "largest normalised self-reported class score",
    "score_top2": "second largest normalised self-reported class score",
    "score_margin": "score_top1 - score_top2",
    "score_entropy": "Shannon entropy of the normalised score vector, in nats",
    "score_mass_top2": "score_top1 + score_top2, how concentrated the answer is",
    "active_score_classes": "number of classes given a non-zero score",
    "llm_confidence": "the model's own self-reported confidence (NOT a posterior)",
    "llm_ambiguity": "the model's own self-reported ambiguity",
    "llm_evidence_strength": "the model's own self-reported evidence strength",
    "predicted_is_<class>": "indicator that the LLM currently predicts <class>",
    "note": "Every feature is computable at inference time from the LLM response "
            "alone. None is derived from a label, from the audio model, or from "
            "the fused outcome.",
}


class FeatureLeakageError(RuntimeError):
    """Raised when a feature frame carries a column that encodes ground truth."""


class MissingFeatureError(RuntimeError):
    """Raised when a feature cannot be computed and must not be imputed."""


def assert_label_free(frame: pd.DataFrame, feature_names: Sequence[str]) -> None:
    """Refuse a feature matrix that names a label-bearing column."""
    offending = sorted(set(feature_names) & LABEL_BEARING_COLUMNS)
    if offending:
        raise FeatureLeakageError(
            f"HSIG features may not include label-bearing columns, but "
            f"{offending} were requested. An estimator trained on these would "
            f"report a quality it cannot reproduce at routing time."
        )
    present = sorted(set(frame.columns) & LABEL_BEARING_COLUMNS & set(feature_names))
    if present:
        raise FeatureLeakageError(
            f"The feature matrix carries label-bearing columns {present}."
        )


def build_features(
    text_llm_frame: pd.DataFrame,
    uncertainty: Sequence[float],
    feature_set: str = DEFAULT_FEATURE_SET,
) -> pd.DataFrame:
    """Build the HSIG feature frame from an LLM prediction frame.

    ``uncertainty`` is passed in rather than read from a column because the
    routing uncertainty is a *policy* choice (Stage 2 froze ``score_entropy``)
    and the estimator must be fitted on exactly the series the router will use.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set must be one of {sorted(FEATURE_SETS)}, got {feature_set!r}"
        )
    values = np.asarray(uncertainty, dtype=float)
    if values.shape[0] != len(text_llm_frame):
        raise ValueError(
            f"uncertainty has {values.shape[0]} entries but the frame has "
            f"{len(text_llm_frame)} rows"
        )

    probability_columns = [
        f"prob_{index}" for index in range(len(CANONICAL_EMOTION_CLASSES))
    ]
    scores = text_llm_frame[probability_columns].to_numpy(dtype=float)
    ordered = np.sort(scores, axis=1)[:, ::-1]
    top1 = ordered[:, 0]
    top2 = ordered[:, 1] if ordered.shape[1] > 1 else np.zeros_like(top1)

    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(scores > 0, scores * np.log(scores), 0.0), axis=1)
    entropy = np.where(np.isnan(scores).any(axis=1), np.nan, entropy)

    built = pd.DataFrame({
        "sample_id": text_llm_frame["sample_id"].astype(str).to_numpy(),
        "uncertainty": values,
        "score_top1": top1,
        "score_top2": top2,
        "score_margin": top1 - top2,
        "score_entropy": entropy,
        "score_mass_top2": top1 + top2,
        "active_score_classes": np.where(
            np.isnan(scores).any(axis=1), np.nan, (scores > 0).sum(axis=1).astype(float)
        ),
        "llm_confidence": _column(text_llm_frame, "llm_confidence"),
        "llm_ambiguity": _column(text_llm_frame, "llm_ambiguity"),
        "llm_evidence_strength": _column(text_llm_frame, "llm_evidence_strength"),
    })

    predicted = text_llm_frame["predicted_class"].to_numpy(dtype=int)
    for index, name in enumerate(CANONICAL_EMOTION_CLASSES):
        built[f"predicted_is_{name}"] = (predicted == index).astype(float)

    built["predicted_class"] = predicted
    built["active_modalities"] = "text_llm"
    assert_label_free(built, FEATURE_SETS[feature_set])
    return built


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def feature_matrix(
    features: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET
) -> tuple[np.ndarray, list[str]]:
    """``(X, names)`` for a named feature set, without imputing anything."""
    names = list(FEATURE_SETS[feature_set])
    assert_label_free(features, names)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise MissingFeatureError(f"The feature frame is missing {missing}")
    return features[names].to_numpy(dtype=float), names


def complete_mask(features: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET):
    """Rows whose every feature is finite.  Incomplete rows are never imputed."""
    matrix, _ = feature_matrix(features, feature_set)
    return np.isfinite(matrix).all(axis=1)


def missing_feature_report(
    features: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET
) -> dict:
    matrix, names = feature_matrix(features, feature_set)
    finite = np.isfinite(matrix)
    return {
        "feature_set": feature_set,
        "features": names,
        "rows": int(matrix.shape[0]),
        "complete_rows": int(finite.all(axis=1).sum()),
        "incomplete_rows": int((~finite.all(axis=1)).sum()),
        "missing_per_feature": {
            name: int((~finite[:, index]).sum()) for index, name in enumerate(names)
        },
        "policy": "Incomplete rows receive no gain estimate. Nothing is imputed.",
    }


# ============================================================
# Routing state
# ============================================================

@dataclass(frozen=True)
class Stage3State:
    """The router's view of one sample, in the shape HSIG consumes."""

    sample_id: str
    predicted_class: int
    uncertainty: float
    confidence: float | None
    features: dict[str, float]

    def to_routing_state(self) -> RoutingState:
        return RoutingState(
            sample_id=self.sample_id,
            active_modalities=("text_llm",),
            uncertainty=float(np.clip(self.uncertainty, 0.0, 1.0)),
            confidence=self.confidence,
            predicted_class=self.predicted_class,
            features=dict(self.features),
        )


def states_from_features(
    features: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET
) -> list[Stage3State]:
    """One :class:`Stage3State` per row, carrying only label-free quantities."""
    names = list(FEATURE_SETS[feature_set])
    states = []
    for row in features.itertuples(index=False):
        record = row._asdict()
        confidence = record.get("llm_confidence")
        states.append(Stage3State(
            sample_id=str(record["sample_id"]),
            predicted_class=int(record["predicted_class"]),
            uncertainty=float(record["uncertainty"]),
            confidence=None if confidence is None or not np.isfinite(confidence)
            else float(confidence),
            features={name: float(record[name]) for name in names},
        ))
    return states


def feature_provenance(feature_set: str = DEFAULT_FEATURE_SET) -> Mapping:
    return {
        "feature_set": feature_set,
        "features": list(FEATURE_SETS[feature_set]),
        "definition": dict(FEATURE_DEFINITION),
        "label_free": True,
        "label_bearing_columns_denied": sorted(LABEL_BEARING_COLUMNS),
        "imputation": "none -- an incomplete row receives no estimate",
    }
