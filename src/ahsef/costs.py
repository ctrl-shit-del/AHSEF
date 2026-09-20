"""Measured acquisition cost and latency for each modality.

UGAPR trades expected information gain against cost, so a cost model invented
at the keyboard would decide the routing policy by fiat.  Everything here is
either measured during the prediction export or derived from the recorded
architecture:

``latency_ms``
    Mean measured per-sample wall clock -- feature extraction plus forward
    pass -- on the host that produced the predictions.

``compute_units``
    ``parameters x input_elements``, a multiply-accumulate-order proxy.  It is
    a proxy and is labelled as one: this project has no FLOP counter and
    inventing one would be worse than declaring the approximation.

Normalisation is explicit and configurable, because "cost 0.3" means nothing
until you know what it was divided by.  The normaliser is always computed over
the candidate set being ranked, and the record says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.ahsef.inference import PredictionSet


NORMALIZATIONS = ("max", "minmax", "sum")


@dataclass(frozen=True)
class ModalityCost:
    """What acquiring one modality costs for one sample."""

    modality: str
    latency_ms: float
    compute_units: float
    parameters: int
    input_elements: int
    measured_on: str
    samples: int

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "latency_ms": self.latency_ms,
            "compute_units": self.compute_units,
            "parameters": self.parameters,
            "input_elements": self.input_elements,
            "measured_on": self.measured_on,
            "samples": self.samples,
        }


#: Input tensor element count per sample, from the geometry each baseline
#: recorded in its run summary.  Anything absent falls back to 1 so the proxy
#: degenerates to the parameter count rather than to zero.
def _input_elements(model_record: Mapping) -> int:
    kind = model_record.get("class")
    if kind == "AudioEmotionBaseline":
        return int(model_record.get("max_frames", 1)) * int(model_record.get("n_mels", 1))
    if kind == "ImageEmotionBaseline":
        size = int(model_record.get("image_size", 1))
        return 3 * size * size
    if kind == "VideoEmotionBaseline":
        size = int(model_record.get("frame_size", 1))
        return int(model_record.get("num_frames", 1)) * 3 * size * size
    if kind == "TextEmotionBaseline":
        tokenizer = model_record.get("tokenizer") or {}
        return int(tokenizer.get("max_tokens", 1))
    if kind == "PhysiologyStateBaseline":
        return int(model_record.get("feature_dim", 1))
    return 1


def cost_from_predictions(
    prediction_set: PredictionSet, run_summary_model: Mapping | None = None
) -> ModalityCost:
    """Derive one modality's cost from a prediction set and its architecture record."""
    model_record = dict(run_summary_model or prediction_set.meta.get("model") or {})
    parameters = int(model_record.get("parameters") or 0)
    elements = _input_elements(model_record)
    return ModalityCost(
        modality=prediction_set.modality,
        latency_ms=float(prediction_set.frame["latency_ms"].mean()),
        compute_units=float(parameters * elements),
        parameters=parameters,
        input_elements=elements,
        measured_on=f"{prediction_set.split} split, {prediction_set.meta.get('device', 'cpu')}",
        samples=int(len(prediction_set.frame)),
    )


class CostModel:
    """Normalised cost and latency over one candidate set.

    ``lambda`` and ``mu`` in UGAPR multiply these normalised quantities, so the
    normaliser is part of the policy and is recorded with it.
    """

    def __init__(
        self,
        costs: Mapping[str, ModalityCost],
        normalization: str = "max",
    ):
        if normalization not in NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}"
            )
        if not costs:
            raise ValueError("A cost model needs at least one modality")
        self.costs = dict(costs)
        self.normalization = normalization

    def __contains__(self, modality: str) -> bool:
        return modality in self.costs

    def _normalise(self, values: Mapping[str, float], modality: str) -> float:
        numbers = [float(value) for value in values.values()]
        value = float(values[modality])
        if self.normalization == "max":
            top = max(numbers)
            return value / top if top > 0 else 0.0
        if self.normalization == "sum":
            total = sum(numbers)
            return value / total if total > 0 else 0.0
        low, high = min(numbers), max(numbers)
        # A single-candidate set has no spread; calling its cost 0 would make
        # the penalty vanish, so it is reported as the maximum instead.
        return (value - low) / (high - low) if high > low else 1.0

    def _subset(self, candidates: Sequence[str] | None) -> list[str]:
        names = list(candidates) if candidates else list(self.costs)
        unknown = [name for name in names if name not in self.costs]
        if unknown:
            raise KeyError(f"No measured cost for {unknown}; known: {sorted(self.costs)}")
        return names

    def normalized_latency(self, modality: str, candidates: Sequence[str] | None = None) -> float:
        names = self._subset(candidates)
        return self._normalise({name: self.costs[name].latency_ms for name in names}, modality)

    def normalized_cost(self, modality: str, candidates: Sequence[str] | None = None) -> float:
        names = self._subset(candidates)
        return self._normalise({name: self.costs[name].compute_units for name in names}, modality)

    def table(self, candidates: Sequence[str] | None = None) -> dict:
        names = self._subset(candidates)
        return {
            "normalization": self.normalization,
            "normalised_over": sorted(names),
            "modalities": {
                name: {
                    **self.costs[name].to_dict(),
                    "normalized_latency": self.normalized_latency(name, names),
                    "normalized_cost": self.normalized_cost(name, names),
                }
                for name in sorted(names)
            },
            "definition": {
                "latency_ms": "measured mean per-sample feature extraction + forward pass",
                "compute_units": "parameters x input_elements, a MAC-order proxy, not a FLOP count",
                "normalization": f"each quantity divided per the '{self.normalization}' rule "
                                 f"over the candidate set being ranked",
            },
        }

    @classmethod
    def from_prediction_sets(
        cls,
        sets: Mapping[str, PredictionSet],
        normalization: str = "max",
    ) -> "CostModel":
        return cls(
            {name: cost_from_predictions(item) for name, item in sets.items()},
            normalization=normalization,
        )
