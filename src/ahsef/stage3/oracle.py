"""PHASE C -- the oracle acquisition value: what acquiring audio actually did.

For every aligned sample this module pays for audio, fuses, and records the
outcome.  That is the *empirical* quantity HSIG will later be asked to predict
without paying, and keeping the two apart is the whole point: the table here is
ground truth about the frozen models, not about the router.

The target is the one Stage 1's diagnostics forced::

    gain(m | A) = P(correct | A u {m}) - P(correct | A)

whose per-sample realisation is ``signed_gain = fused_correct - text_correct``,
taking values in ``{-1, 0, +1}``.  The binary ``y_gain`` (audio fixed a wrong
text prediction) is reported alongside it because the class imbalance question
is asked of that event, but the estimator's target is the signed quantity: an
estimator trained only on fixes would be blind to the acquisitions that *break*
a correct prediction, and those are the ones a routing policy most needs to
avoid.

Delta-uncertainty is computed and written out **as a diagnostic column only**.
Stage 1 measured that its sign flips with the anchor and the fusion rule and
that its largest per-sample correlation with actual improvement was 0.143.
:data:`REJECTED_TARGET` records that it is not, and may not become, the target.

Nothing in this module is consulted at routing time.  Labels enter here and
stay here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.ahsef.fusion import FusionSpec, fuse_prediction_sets, select_weights
from src.ahsef.hsig import HSIG_TARGET, HSIG_TARGET_DEFINITION
from src.ahsef.inference import PredictionSet
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

REJECTED_TARGET = dict(HSIG_TARGET_DEFINITION)

#: Splits on which a fusion weight may be chosen.  Mirrors
#: :data:`src.ahsef.fusion.WEIGHT_SELECTION_SPLITS`; restated so a Stage 3
#: reader does not have to go looking for it.
WEIGHT_SELECTION_SPLITS = ("train", "validation")


class OracleError(RuntimeError):
    """Raised when the oracle table cannot be built from the given inputs."""


def usable_text_ids(text_llm: PredictionSet) -> list[str]:
    """Pooled ids for which the LLM produced a canonical prediction.

    A sample the LLM refused has no initial evidence state, so it has no
    routing decision to make and no gain to estimate.  Such samples are
    excluded here and **counted** by :func:`coverage_note`; they are never
    given a fabricated prediction so that the pool looks larger.
    """
    frame = text_llm.frame
    return frame.loc[frame["llm_usable"].astype(bool), "sample_id"].astype(str).tolist()


def coverage_note(text_llm: PredictionSet, retained: Sequence[str]) -> dict:
    frame = text_llm.frame
    return {
        "pooled_samples": int(len(frame)),
        "usable_text_evidence": int(len(retained)),
        "excluded_no_text_evidence": int(len(frame) - len(retained)),
        "status_counts": {
            str(key): int(value) for key, value in frame["llm_status"].value_counts().items()
        },
        "rule": (
            "A sample the LLM refused, abstained on, or answered unmappably has no "
            "initial evidence state. It is excluded from the routing experiment and "
            "counted here, never assigned a filler prediction."
        ),
    }


# ============================================================
# Fusion weight
# ============================================================

def choose_fusion_spec(
    text_llm: PredictionSet,
    audio: PredictionSet,
    sample_ids: Sequence[str],
    split: str,
    method: str = "weighted_probability",
    objective: str = "macro_f1",
) -> FusionSpec:
    """Select the Text+Audio fusion weight on validation, by exhaustive scan.

    Refuses any split but train/validation.  The scan is one-dimensional and
    fully enumerated inside the returned spec, so there is nothing left to tune
    once it is frozen.
    """
    if split not in WEIGHT_SELECTION_SPLITS:
        raise OracleError(
            f"Fusion weights may only be selected on {WEIGHT_SELECTION_SPLITS}; "
            f"refusing to select on {split!r}. A weight chosen where the answer is "
            f"known is not a weight, it is a result quoted backwards."
        )
    return select_weights(
        {"text_llm": text_llm, "audio": audio}, sample_ids, split,
        method=method, objective=objective,
    )


def fuse(
    text_llm: PredictionSet,
    audio: PredictionSet,
    spec: FusionSpec,
    sample_ids: Sequence[str],
) -> PredictionSet:
    """Fuse over an explicitly vetted pool.  Never joins by position."""
    return fuse_prediction_sets(
        {"text_llm": text_llm, "audio": audio}, spec, sample_ids
    )


# ============================================================
# The oracle table
# ============================================================

ORACLE_COLUMNS = (
    "sample_id", "dataset", "split", "true_class",
    "text_prediction", "text_correct", "text_uncertainty", "text_confidence",
    "audio_prediction", "audio_correct", "audio_confidence", "audio_uncertainty",
    "fused_prediction", "fused_correct", "fused_confidence", "fused_uncertainty",
    "y_gain", "y_harm", "signed_gain", "prediction_changed",
    "delta_uncertainty_diagnostic_only", "audio_available",
)


def build_oracle_table(
    text_llm: PredictionSet,
    audio: PredictionSet,
    fused: PredictionSet,
    sample_ids: Sequence[str],
    uncertainty_column: str = "normalized_entropy",
) -> pd.DataFrame:
    """One row per aligned sample describing what acquiring audio actually did."""
    text = text_llm.restricted_to(sample_ids)
    sound = audio.restricted_to(sample_ids)
    both = fused.restricted_to(sample_ids)
    if not (text.sample_ids() == sound.sample_ids() == both.sample_ids()):
        raise OracleError(
            "The text, audio, and fused prediction sets did not reindex identically; "
            "rows would be paired by position, which is exactly the error the "
            "alignment machinery exists to prevent."
        )
    truth = text.labels().numpy()
    if not np.array_equal(truth, sound.labels().numpy()):
        raise OracleError(
            "Text and audio disagree about the true class of pooled samples; they are "
            "not the same samples."
        )

    text_pred = text.predictions().numpy()
    audio_pred = sound.predictions().numpy()
    fused_pred = both.predictions().numpy()
    text_correct = text_pred == truth
    audio_correct = audio_pred == truth
    fused_correct = fused_pred == truth

    frame = pd.DataFrame({
        "sample_id": text.sample_ids(),
        "dataset": text.frame["dataset"].tolist(),
        "split": text.split,
        "true_class": truth,
        "text_prediction": text_pred,
        "text_correct": text_correct,
        "text_uncertainty": text.frame[uncertainty_column].to_numpy(dtype=float),
        "text_confidence": text.frame["confidence"].to_numpy(dtype=float),
        "audio_prediction": audio_pred,
        "audio_correct": audio_correct,
        "audio_confidence": sound.frame["confidence"].to_numpy(dtype=float),
        "audio_uncertainty": sound.frame["normalized_entropy"].to_numpy(dtype=float),
        "fused_prediction": fused_pred,
        "fused_correct": fused_correct,
        "fused_confidence": both.frame["confidence"].to_numpy(dtype=float),
        "fused_uncertainty": both.frame["normalized_entropy"].to_numpy(dtype=float),
    })
    frame["y_gain"] = (~text_correct & fused_correct).astype(int)
    frame["y_harm"] = (text_correct & ~fused_correct).astype(int)
    frame["signed_gain"] = fused_correct.astype(int) - text_correct.astype(int)
    frame["prediction_changed"] = text_pred != fused_pred
    frame["delta_uncertainty_diagnostic_only"] = (
        frame["text_uncertainty"] - frame["fused_uncertainty"]
    )
    # Every row in this table came from the vetted pool, so audio was available
    # by construction. The column is carried anyway: the router consumes an
    # availability signal, and a table that silently assumed availability would
    # let an unavailable-audio bug pass unnoticed.
    frame["audio_available"] = True
    return frame


# ============================================================
# Reporting
# ============================================================

def _block(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int = 7) -> dict:
    metrics = classification_metrics(predictions, labels, num_classes)
    return {
        "samples": int(labels.numel()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": float(np.mean([
            value for value, support in zip(
                metrics["per_class_recall"], metrics["support"]
            ) if support > 0
        ])),
        "per_class_f1": {
            name: metrics["per_class_f1"][index]
            for index, name in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "support": {
            name: metrics["support"][index]
            for index, name in enumerate(CANONICAL_EMOTION_CLASSES)
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }


def _wilson(successes: int, total: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score interval -- correct at the small counts this pool produces."""
    if total <= 0:
        return None
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return [float(max(centre - half, 0.0)), float(min(centre + half, 1.0))]


def oracle_report(
    frame: pd.DataFrame,
    text_llm: PredictionSet,
    audio: PredictionSet,
    fused: PredictionSet,
    sample_ids: Sequence[str],
    spec: FusionSpec | None = None,
    coverage: Mapping | None = None,
) -> dict:
    """The Phase C report: does audio actually carry complementary information?"""
    text = text_llm.restricted_to(sample_ids)
    sound = audio.restricted_to(sample_ids)
    both = fused.restricted_to(sample_ids)
    labels = text.labels()

    total = int(len(frame))
    fixed = int(frame["y_gain"].sum())
    broken = int(frame["y_harm"].sum())

    per_class = {}
    for index, name in enumerate(CANONICAL_EMOTION_CLASSES):
        mask = frame["true_class"].to_numpy() == index
        count = int(mask.sum())
        per_class[name] = {
            "samples": count,
            "text_accuracy": float(frame.loc[mask, "text_correct"].mean()) if count else None,
            "fused_accuracy": float(frame.loc[mask, "fused_correct"].mean()) if count else None,
            "fixed_by_audio": int(frame.loc[mask, "y_gain"].sum()),
            "broken_by_audio": int(frame.loc[mask, "y_harm"].sum()),
            "net": int(frame.loc[mask, "signed_gain"].sum()),
        }

    return {
        "phase": "C -- oracle acquisition value",
        "split": str(frame["split"].iloc[0]) if total else None,
        "samples": total,
        "coverage": dict(coverage or {}),
        "fusion": spec.to_dict() if spec is not None else None,
        "systems": {
            "text_llm": _block(text.predictions(), labels),
            "audio": _block(sound.predictions(), labels),
            "text_llm+audio": _block(both.predictions(), labels),
        },
        "acquisition_outcome": {
            "text_correct": int(frame["text_correct"].sum()),
            "fused_correct": int(frame["fused_correct"].sum()),
            "fixed_by_audio": fixed,
            "fixed_fraction": fixed / total if total else None,
            "fixed_fraction_ci95": _wilson(fixed, total),
            "broken_by_audio": broken,
            "broken_fraction": broken / total if total else None,
            "broken_fraction_ci95": _wilson(broken, total),
            "net_improvement": fixed - broken,
            "net_improvement_rate": (fixed - broken) / total if total else None,
            "mean_signed_gain": float(frame["signed_gain"].mean()) if total else None,
            "prediction_change_rate": float(frame["prediction_changed"].mean())
            if total else None,
            "agreement_text_audio": float(
                (frame["text_prediction"] == frame["audio_prediction"]).mean()
            ) if total else None,
        },
        "per_class_improvement": per_class,
        "target": {
            **HSIG_TARGET_DEFINITION,
            "per_sample_realisation": "signed_gain = fused_correct - text_correct, in {-1, 0, +1}",
            "binary_event": "y_gain = 1 when audio fixed a wrong text prediction",
            "estimator_target": HSIG_TARGET,
        },
        "delta_uncertainty_diagnostic": {
            "mean": float(frame["delta_uncertainty_diagnostic_only"].mean()) if total else None,
            "correlation_with_signed_gain": _safe_corr(
                frame["delta_uncertainty_diagnostic_only"].to_numpy(dtype=float),
                frame["signed_gain"].to_numpy(dtype=float),
            ),
            "status": "REJECTED as an HSIG target by Stage 1; reported as a diagnostic only",
            "why_rejected": REJECTED_TARGET["why_rejected"],
        },
        "uncertainty": {
            "mean_text_uncertainty": float(frame["text_uncertainty"].mean()) if total else None,
            "mean_fused_uncertainty": float(frame["fused_uncertainty"].mean()) if total else None,
            "mean_text_uncertainty_when_audio_helps": (
                float(frame.loc[frame["y_gain"] == 1, "text_uncertainty"].mean())
                if fixed else None
            ),
            "mean_text_uncertainty_when_audio_harms": (
                float(frame.loc[frame["y_harm"] == 1, "text_uncertainty"].mean())
                if broken else None
            ),
        },
        "labels_used": "Ground truth is read here and only here. This table is built "
                       "after acquisition and is never consulted by a routing decision.",
    }


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return None if np.isnan(value) else value
