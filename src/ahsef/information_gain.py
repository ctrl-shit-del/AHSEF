"""Empirical per-sample information gain from acquiring a candidate modality.

For an active modality set ``A`` and an inactive candidate ``m``::

    dU(m | A) = U(A) - U(A u {m})

with ``U`` the normalised predictive entropy of the (fused) posterior.  A
positive ``dU`` means acquiring ``m`` sharpened the belief for that sample.

This is the *empirical* quantity: it is computed after actually paying for the
candidate, so it is what HSIG will later be asked to predict without paying.
Keeping the two apart matters -- a stage-2 HSIG trained on validation ``dU``
records is a predictor; the records themselves are ground truth about the
frozen models, not about the router.

Records are per sample, because AHSEF decides per sample.  A dataset-level mean
hides exactly the variation the router exists to exploit.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from src.ahsef.costs import CostModel
from src.ahsef.inference import PredictionSet


ROUTING_RECORD_COLUMNS = (
    "sample_id",
    "dataset",
    "split",
    "active_modalities",
    "candidate_modality",
    "uncertainty_before",
    "uncertainty_after",
    "delta_uncertainty",
    "confidence_before",
    "confidence_after",
    "predicted_before",
    "predicted_after",
    "prediction_changed",
    "latency_ms",
    "cost",
)


def delta_uncertainty_records(
    active: PredictionSet,
    fused: PredictionSet,
    candidate: str,
    sample_ids: Sequence[str],
    cost_model: CostModel | None = None,
    candidate_set: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per sample describing what acquiring ``candidate`` actually did.

    ``active`` is the posterior before acquisition, ``fused`` the posterior
    after.  Both are reindexed to ``sample_ids`` so no row is ever paired by
    position.

    ``latency_ms`` and ``cost`` are the *marginal* price of the candidate --
    what the router would have to spend to obtain this row -- not the running
    total, because that is what UGAPR trades against the gain.
    """
    before = active.restricted_to(sample_ids)
    after = fused.restricted_to(sample_ids)
    if before.sample_ids() != after.sample_ids():
        raise ValueError("Active and fused prediction sets did not reindex identically")

    active_modalities = list(before.meta.get("component_modalities") or [before.modality])
    latency = None
    cost = None
    if cost_model is not None and candidate in cost_model:
        latency = cost_model.normalized_latency(candidate, candidate_set)
        cost = cost_model.normalized_cost(candidate, candidate_set)

    frame = pd.DataFrame({
        "sample_id": before.sample_ids(),
        "dataset": before.frame["dataset"].tolist(),
        "split": before.split,
        "active_modalities": ["+".join(active_modalities)] * len(before.frame),
        "candidate_modality": candidate,
        "uncertainty_before": before.frame["normalized_entropy"].to_numpy(),
        "uncertainty_after": after.frame["normalized_entropy"].to_numpy(),
        "confidence_before": before.frame["confidence"].to_numpy(),
        "confidence_after": after.frame["confidence"].to_numpy(),
        "predicted_before": before.frame["predicted_class"].to_numpy(),
        "predicted_after": after.frame["predicted_class"].to_numpy(),
        "true_class": before.frame["true_class"].to_numpy(),
    })
    frame["delta_uncertainty"] = frame["uncertainty_before"] - frame["uncertainty_after"]
    frame["prediction_changed"] = frame["predicted_before"] != frame["predicted_after"]
    frame["measured_latency_ms"] = after.frame["latency_ms"].to_numpy() - before.frame[
        "latency_ms"
    ].to_numpy()
    frame["latency_ms"] = latency
    frame["cost"] = cost
    return frame


def summarise_delta_uncertainty(frame: pd.DataFrame) -> dict:
    """Dataset-level view of a per-sample ``dU`` table, with the spread kept.

    ``correct_before`` / ``correct_after`` are computed here and only here:
    they are diagnostic, reported after routing, and never consulted by a
    routing decision.
    """
    correct_before = frame["predicted_before"] == frame["true_class"]
    correct_after = frame["predicted_after"] == frame["true_class"]
    delta = frame["delta_uncertainty"]
    return {
        "samples": int(len(frame)),
        "active_modalities": sorted(set(frame["active_modalities"])),
        "candidate_modality": sorted(set(frame["candidate_modality"])),
        "mean_uncertainty_before": float(frame["uncertainty_before"].mean()),
        "mean_uncertainty_after": float(frame["uncertainty_after"].mean()),
        "mean_delta_uncertainty": float(delta.mean()),
        "median_delta_uncertainty": float(delta.median()),
        "std_delta_uncertainty": float(delta.std(ddof=0)),
        "min_delta_uncertainty": float(delta.min()),
        "max_delta_uncertainty": float(delta.max()),
        "positive_delta_rate": float((delta > 0).mean()),
        "prediction_change_rate": float(frame["prediction_changed"].mean()),
        "outcome": {
            "correct_before": int(correct_before.sum()),
            "correct_after": int(correct_after.sum()),
            "fixed_by_candidate": int((~correct_before & correct_after).sum()),
            "broken_by_candidate": int((correct_before & ~correct_after).sum()),
            "net_corrections": int((~correct_before & correct_after).sum())
            - int((correct_before & ~correct_after).sum()),
            "note": "Diagnostic only; correctness is read after routing, never during it.",
        },
        "normalized_latency": (
            float(frame["latency_ms"].iloc[0]) if frame["latency_ms"].notna().all() else None
        ),
        "normalized_cost": (
            float(frame["cost"].iloc[0]) if frame["cost"].notna().all() else None
        ),
        "mean_measured_marginal_latency_ms": float(frame["measured_latency_ms"].mean()),
    }


def gain_by_uncertainty_band(
    frame: pd.DataFrame, edges: Sequence[float] = (0.0, 0.6, 0.8, 0.9, 0.95, 1.0)
) -> list[dict]:
    """Mean ``dU`` grouped by the pre-acquisition uncertainty band.

    This is the evidence a stopping threshold has to be argued from: if gain is
    flat across bands, an uncertainty trigger buys nothing over acquiring
    always or never.
    """
    rows = []
    for lower, upper in zip(edges, edges[1:]):
        inside = (frame["uncertainty_before"] >= lower) & (frame["uncertainty_before"] < upper)
        if upper == edges[-1]:
            inside = (frame["uncertainty_before"] >= lower) & (
                frame["uncertainty_before"] <= upper
            )
        count = int(inside.sum())
        correct_before = frame.loc[inside, "predicted_before"] == frame.loc[inside, "true_class"]
        correct_after = frame.loc[inside, "predicted_after"] == frame.loc[inside, "true_class"]
        rows.append({
            "lower": float(lower),
            "upper": float(upper),
            "samples": count,
            "mean_delta_uncertainty": float(frame.loc[inside, "delta_uncertainty"].mean())
            if count else None,
            "positive_delta_rate": float((frame.loc[inside, "delta_uncertainty"] > 0).mean())
            if count else None,
            "accuracy_before": float(correct_before.mean()) if count else None,
            "accuracy_after": float(correct_after.mean()) if count else None,
        })
    return rows


def candidate_gain_table(frames: Mapping[str, pd.DataFrame]) -> dict:
    """Side-by-side ``dU`` summaries for several candidates over one active set.

    This is the empirical table a prototype HSIG is later fitted to; it is not
    itself HSIG, and it is built from the split it is labelled with.
    """
    summaries = {name: summarise_delta_uncertainty(frame) for name, frame in frames.items()}
    if not summaries:
        return {"candidates": {}, "ranked_by_mean_delta_uncertainty": []}
    ranked = sorted(
        summaries, key=lambda name: summaries[name]["mean_delta_uncertainty"], reverse=True
    )
    splits = sorted({split for frame in frames.values() for split in set(frame["split"])})
    return {
        "split": splits[0] if len(splits) == 1 else splits,
        "candidates": summaries,
        "ranked_by_mean_delta_uncertainty": ranked,
        "note": (
            "Empirical dU measured by actually fusing each candidate. HSIG's job in "
            "stage 2 is to predict these numbers without paying for the candidate."
        ),
    }
