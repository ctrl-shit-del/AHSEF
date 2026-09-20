"""Probability-level fusion of aligned unimodal prediction sets.

Deliberately simple.  The question this milestone has to answer is whether a
second modality carries *complementary evidence*, and a trained fusion network
would confound that question with its own capacity: a gain could come from the
modality or from the extra parameters.  A fixed, weight-parameterised
combination of two frozen posteriors has no capacity of its own, so any gain it
shows is attributable to the modality.

Two rules are enforced rather than documented:

* **Same class space.**  Two models are fusable only if they emit the same
  ordered class list.  The WESAD physiology model emits three study conditions;
  averaging those with seven emotion posteriors would produce a number with no
  referent, so it is refused.
* **Same sample.**  Fusion operates on an explicitly supplied, vetted pool of
  ``sample_id`` values (see :mod:`src.ahsef.identity`) and reindexes every
  participating set to that order.  Rows are never joined by position.

Fusion weights are a free parameter, so where they are chosen matters.
:func:`select_weights` searches on a named split and refuses to search on
``test``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd
import torch

from src.ahsef.inference import PredictionSet
from src.ahsef.uncertainty import normalize_probabilities, uncertainty_columns


FUSION_METHODS = ("weighted_probability", "log_opinion_pool")

#: Splits on which a fusion weight may be chosen.  ``test`` is not one of them.
WEIGHT_SELECTION_SPLITS = ("train", "validation")


class FusionCompatibilityError(RuntimeError):
    """Raised when two prediction sets cannot be fused as a matter of definition."""


class WeightSelectionError(RuntimeError):
    """Raised when fusion weights would be chosen using evaluation data."""


# ============================================================
# Weights
# ============================================================

@dataclass(frozen=True)
class FusionSpec:
    """A reproducible description of one fusion rule."""

    method: str = "weighted_probability"
    weights: Mapping[str, float] = field(default_factory=dict)
    selected_on_split: str | None = None
    selection: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in FUSION_METHODS:
            raise ValueError(f"method must be one of {FUSION_METHODS}, got {self.method!r}")
        if not self.weights:
            raise ValueError("A fusion spec needs at least one modality weight")
        if any(value < 0 for value in self.weights.values()):
            raise ValueError(f"Fusion weights must be non-negative, got {dict(self.weights)}")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Fusion weights must not all be zero")
        object.__setattr__(
            self, "weights",
            {name: float(value) / total for name, value in sorted(self.weights.items())},
        )

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.weights)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "weights": dict(self.weights),
            "weights_normalised": True,
            "selected_on_split": self.selected_on_split,
            "uses_test_labels": False,
            "selection": dict(self.selection),
        }

    @classmethod
    def uniform(cls, modalities: Sequence[str]) -> "FusionSpec":
        return cls(
            method="weighted_probability",
            weights={name: 1.0 for name in modalities},
            selected_on_split=None,
            selection={"method": "uniform", "note": "no data used to choose the weights"},
        )


# ============================================================
# Core combination
# ============================================================

def fuse_probabilities(
    probabilities: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    method: str = "weighted_probability",
) -> torch.Tensor:
    """Combine per-modality ``[N, K]`` posteriors into one ``[N, K]`` posterior.

    ``weighted_probability``
        ``normalize(sum_m w_m p_m)`` -- a linear opinion pool.  Robust: one
        modality's near-zero probability cannot veto a class.

    ``log_opinion_pool``
        ``normalize(prod_m p_m ^ w_m)`` -- a geometric pool.  Sharper, and a
        confident disagreement genuinely suppresses a class.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"method must be one of {FUSION_METHODS}, got {method!r}")
    if not probabilities:
        raise ValueError("Nothing to fuse")
    missing = set(probabilities) - set(weights)
    if missing:
        raise ValueError(f"No fusion weight given for {sorted(missing)}")

    shapes = {name: tuple(value.shape) for name, value in probabilities.items()}
    if len(set(shapes.values())) != 1:
        raise FusionCompatibilityError(f"Posterior shapes disagree: {shapes}")

    total_weight = sum(weights[name] for name in probabilities)
    if total_weight <= 0:
        raise ValueError("Fusion weights over the supplied modalities sum to zero")

    if method == "weighted_probability":
        stacked = sum(
            (weights[name] / total_weight) * value.to(torch.float64)
            for name, value in probabilities.items()
        )
        return normalize_probabilities(stacked)

    # Geometric pooling in log space; the clamp keeps a zero posterior from
    # driving the whole product to -inf.
    log_total = sum(
        (weights[name] / total_weight) * value.to(torch.float64).clamp(min=1e-12).log()
        for name, value in probabilities.items()
    )
    return normalize_probabilities((log_total - log_total.max(dim=-1, keepdim=True).values).exp())


# ============================================================
# Prediction-set level
# ============================================================

def assert_fusable(sets: Mapping[str, PredictionSet]) -> tuple[str, ...]:
    """Check the class space and split agree; return the shared class order."""
    if len(sets) < 2:
        raise ValueError("Fusion needs at least two prediction sets")
    orders = {name: tuple(item.class_order) for name, item in sets.items()}
    distinct = set(orders.values())
    if len(distinct) != 1:
        raise FusionCompatibilityError(
            "Refusing to fuse models with different class spaces: "
            + "; ".join(f"{name}={list(order)}" for name, order in sorted(orders.items()))
            + ". A posterior over WESAD study conditions is not a posterior over "
              "emotions, and averaging them would produce a number with no referent."
        )
    splits = {item.split for item in sets.values()}
    if len(splits) != 1:
        raise FusionCompatibilityError(
            f"Refusing to fuse prediction sets from different splits: {sorted(splits)}"
        )
    return next(iter(distinct))


def fuse_prediction_sets(
    sets: Mapping[str, PredictionSet],
    spec: FusionSpec,
    sample_ids: Sequence[str],
) -> PredictionSet:
    """Fuse aligned prediction sets over an explicitly vetted sample pool.

    ``sample_ids`` must come from :meth:`src.ahsef.identity.AlignmentIndex.fusion_pool`.
    Passing them explicitly -- rather than letting this function intersect what
    it happens to be handed -- is what makes an accidental fusion of unrelated
    records impossible to write by mistake.
    """
    class_order = assert_fusable(sets)
    if not sample_ids:
        raise ValueError("Refusing to fuse over an empty sample pool")
    missing = set(spec.modalities) - set(sets)
    if missing:
        raise ValueError(f"Fusion spec names {sorted(missing)} but no such prediction set was given")

    aligned = {name: sets[name].restricted_to(sample_ids) for name in spec.modalities}
    reference = aligned[spec.modalities[0]]

    # Two models can only be fused over a sample if they agree it is the same
    # sample carrying the same target.
    for name, item in aligned.items():
        if item.sample_ids() != reference.sample_ids():
            raise FusionCompatibilityError(f"{name} did not reindex to the requested pool order")
        disagreement = int((item.labels() != reference.labels()).sum())
        if disagreement:
            raise FusionCompatibilityError(
                f"{name} and {spec.modalities[0]} disagree about the true class of "
                f"{disagreement} pooled samples; they are not the same samples."
            )

    fused = fuse_probabilities(
        {name: item.probabilities() for name, item in aligned.items()},
        spec.weights, method=spec.method,
    )

    frame = pd.DataFrame({
        "sample_id": reference.sample_ids(),
        "dataset": reference.frame["dataset"].tolist(),
        "modality": "+".join(spec.modalities),
        "split": reference.split,
        "true_class": reference.frame["true_class"].tolist(),
    })
    for index in range(fused.shape[1]):
        frame[f"prob_{index}"] = fused[:, index].numpy()
        # The fused object has no logits of its own; log-probabilities keep the
        # column set uniform without pretending a network produced them.
        frame[f"logit_{index}"] = fused[:, index].clamp(min=1e-12).log().numpy()
    for name, values in uncertainty_columns(fused).items():
        frame[name] = values.numpy()
    frame["latency_ms"] = sum(item.frame["latency_ms"].to_numpy() for item in aligned.values())
    frame["inference_ms"] = sum(item.frame["inference_ms"].to_numpy() for item in aligned.values())
    frame["feature_ms"] = sum(item.frame["feature_ms"].to_numpy() for item in aligned.values())

    return PredictionSet(
        modality="+".join(spec.modalities), split=reference.split,
        class_order=class_order, frame=frame,
        meta={
            "kind": "fused",
            "component_modalities": list(spec.modalities),
            "fusion": spec.to_dict(),
            "pool_size": len(frame),
            "components": {
                name: {
                    "experiment": item.meta.get("experiment"),
                    "iteration": item.meta.get("iteration"),
                    "checkpoint_sha256": item.meta.get("checkpoint_sha256"),
                    "temperature": item.meta.get("temperature"),
                }
                for name, item in aligned.items()
            },
            "latency_note": "Sum of component latencies: acquiring a modality costs "
                            "its own inference, and AHSEF pays for both.",
            "uses_labels_for_prediction": False,
        },
    )


# ============================================================
# Weight selection
# ============================================================

def select_weights(
    sets: Mapping[str, PredictionSet],
    sample_ids: Sequence[str],
    split: str,
    method: str = "weighted_probability",
    grid: Sequence[float] = tuple(round(0.05 * step, 2) for step in range(21)),
    objective: str = "macro_f1",
) -> FusionSpec:
    """Choose a two-modality weight on ``split`` by scanning a fixed grid.

    Refuses to run on ``test``.  The scan is over the anchor's weight only, so
    the search space is one-dimensional, fully enumerated, and reported --
    there is nothing hidden to tune later.
    """
    if split not in WEIGHT_SELECTION_SPLITS:
        raise WeightSelectionError(
            f"Fusion weights may only be selected on {WEIGHT_SELECTION_SPLITS}; "
            f"refusing to select on {split!r}."
        )
    names = list(sets)
    if len(names) != 2:
        raise ValueError("select_weights currently handles exactly two modalities")
    if any(item.split != split for item in sets.values()):
        raise WeightSelectionError(
            f"Every prediction set must come from the {split!r} split; got "
            f"{ {name: item.split for name, item in sets.items()} }"
        )

    from src.ahsef.evaluation import evaluate_prediction_set

    anchor, candidate = names
    scanned = []
    best: tuple[float, float] | None = None
    for weight in grid:
        if weight <= 0 or weight >= 1:
            continue
        spec = FusionSpec(
            method=method, weights={anchor: weight, candidate: 1.0 - weight},
            selected_on_split=split,
        )
        metrics = evaluate_prediction_set(fuse_prediction_sets(sets, spec, sample_ids))
        score = float(metrics[objective])
        scanned.append({"weight_" + anchor: weight, objective: score})
        if best is None or score > best[1]:
            best = (weight, score)

    if best is None:
        raise ValueError("The weight grid contained no interior point in (0, 1)")
    weight, score = best
    return FusionSpec(
        method=method, weights={anchor: weight, candidate: 1.0 - weight},
        selected_on_split=split,
        selection={
            "method": "exhaustive_grid_scan",
            "objective": objective,
            "grid": [float(value) for value in grid],
            "anchor": anchor,
            "best_weight": weight,
            "best_score": score,
            "pool_size": len(sample_ids),
            "scanned": scanned,
        },
    )
