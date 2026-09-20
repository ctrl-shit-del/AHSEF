"""PHASE E (cost side) -- what acquiring each modality actually costs.

UGAPR subtracts ``lambda * C(m) + mu * L(m)`` from the estimated gain, so a cost
model invented at the keyboard would decide the routing policy by fiat.  Stage 1
already measures per-modality latency and a MAC-order compute proxy; this module
adds the two things Stage 3 needs on top of it.

**The LLM has to be priced too.**  ``text_llm`` is not one of the five frozen
torch baselines, so :func:`src.ahsef.costs.cost_from_predictions` cannot derive
its geometry from a ``run_summary``.  Its compute proxy is built with the same
formula -- ``parameters x input_elements`` -- from the model's declared 32.7B
parameters and its measured mean input-token count, so the LLM and the audio CNN
sit on one scale rather than on two incomparable ones.

**Latency means wall clock, not generation time.**  Stage 2 measured generation
at roughly 29% of wall clock, with an 18x throughput swing inside a single run.
Pricing the LLM at generation time would understate it by a factor that itself
varies during the run, so :func:`llm_cost` takes the measured wall clock and
records both numbers plus their ratio.

**Normalisation is over the whole registry, not over the candidate set.**  With
one candidate, normalising over the candidate set alone would hand audio a
normalised cost of exactly 1.0 by construction, making the penalty an artefact
of there being nothing to compare it with.  Normalising over every modality
whose cost this project has measured gives audio a cost that means something.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from src.ahsef.costs import CostModel, ModalityCost, cost_from_predictions
from src.ahsef.inference import PredictionSet
from src.ahsef.layout import AhsefLayout
from src.ahsef.registry import DEFAULT_BASELINES

#: Declared parameter count of the Stage 2 text evidence source.  From the
#: Ollama model record (``parameter_size: 32.7B``), not estimated.
LLM_PARAMETERS = 32_700_000_000

COST_DEFINITION = {
    "latency_ms": (
        "measured mean per-sample wall clock. For the frozen torch baselines this is "
        "feature extraction plus forward pass; for text_llm it is end-to-end service "
        "time including queueing and cloud scheduling, NOT generation time."
    ),
    "compute_units": (
        "parameters x input_elements, a multiply-accumulate-order proxy. It is a "
        "proxy and is labelled as one: this project has no FLOP counter, and "
        "inventing one would be worse than declaring the approximation."
    ),
    "normalization": (
        "each quantity divided by the maximum over the FULL measured modality "
        "registry, not over the candidate set. Normalising over a one-member "
        "candidate set would make every normalised cost 1.0 by construction."
    ),
    "llm_latency_caveat": (
        "Ollama cloud throughput is not constant: Stage 2 recorded an 18x swing "
        "within one run. The recorded latency is the mean measured during the Stage 3 "
        "pass and is a point estimate of a varying quantity, not a service guarantee."
    ),
}


def llm_cost(
    predictions: PredictionSet,
    timing: Mapping,
    parameters: int = LLM_PARAMETERS,
) -> ModalityCost:
    """Price the LLM at measured wall clock, on the same proxy as the baselines."""
    wall = timing.get("wall_clock") or {}
    generation = timing.get("generation_latency_ms") or {}
    latency = wall.get("mean_ms_per_live_call")
    measured = wall.get("measured_this_invocation") and latency is not None
    if not measured:
        # Refusing to substitute generation time silently: it is a different
        # quantity and Stage 2 measured it at a fraction of the real cost.
        latency = float(generation.get("mean") or 0.0)
    input_tokens = predictions.frame.get("input_tokens")
    elements = int(input_tokens.mean()) if input_tokens is not None else 1
    return ModalityCost(
        modality="text_llm",
        latency_ms=float(latency),
        compute_units=float(parameters * max(elements, 1)),
        parameters=int(parameters),
        input_elements=max(elements, 1),
        measured_on=(
            f"{predictions.split} split, ollama wall clock"
            if measured else
            f"{predictions.split} split, GENERATION TIME ONLY -- wall clock was not "
            f"measured in this invocation, so this understates the real service cost"
        ),
        samples=int(len(predictions.frame)),
    )


def build_cost_model(
    llm_predictions: PredictionSet,
    llm_timing: Mapping,
    stage1_run: str = "stage1",
    root: Path | str = "experiments",
    split: str = "validation",
    normalization: str = "max",
) -> tuple[CostModel, dict]:
    """Assemble the Stage 3 registry cost model from measured Stage 1 exports.

    Every frozen baseline whose Stage 1 prediction export exists contributes its
    measured cost, so audio's normalised price is set against real alternatives
    rather than against itself.
    """
    layout = AhsefLayout(run=stage1_run, root=Path(root) / "ahsef")
    costs: dict[str, ModalityCost] = {}
    sources: dict[str, str] = {}
    for modality, reference in DEFAULT_BASELINES.items():
        path = layout.prediction_path(modality, split)
        if not path.exists():
            continue
        prediction_set = PredictionSet.load(
            path, layout.prediction_meta_path(modality, split)
        )
        costs[modality] = cost_from_predictions(
            prediction_set, reference.model_record(root)
        )
        sources[modality] = str(path)

    costs["text_llm"] = llm_cost(llm_predictions, llm_timing)
    sources["text_llm"] = "Stage 3 transcript wall clock + Ollama model record"

    if "audio" not in costs:
        raise FileNotFoundError(
            f"No Stage 1 audio prediction export for {split}; the audio acquisition "
            f"cost cannot be measured and Stage 3 will not invent one."
        )

    model = CostModel(costs, normalization=normalization)
    record = {
        **model.table(),
        "sources": sources,
        "definition": dict(COST_DEFINITION),
        "normalised_over_full_registry": sorted(costs),
        "candidate_set": ["audio"],
        "audio": {
            "normalized_cost": model.normalized_cost("audio"),
            "normalized_latency": model.normalized_latency("audio"),
        },
        "text_llm": {
            "normalized_cost": model.normalized_cost("text_llm"),
            "normalized_latency": model.normalized_latency("text_llm"),
        },
    }
    return model, record


def cost_model_from_frozen(frozen: Mapping) -> CostModel:
    """Rebuild the exact cost model the freeze recorded.

    The acquisition price is part of the frozen policy: UGAPR subtracts
    ``lambda*C + mu*L`` from the gain, so re-measuring C and L on the test split
    would shift the effective threshold without anything in the configuration
    appearing to change. The locked evaluation therefore prices audio at the
    numbers the freeze wrote down, and reports the test split's own measured
    latency separately as an observation rather than as a policy input.
    """
    record = frozen["costs"]
    costs = {
        name: ModalityCost(
            modality=name,
            latency_ms=float(entry["latency_ms"]),
            compute_units=float(entry["compute_units"]),
            parameters=int(entry["parameters"]),
            input_elements=int(entry["input_elements"]),
            measured_on=str(entry["measured_on"]),
            samples=int(entry["samples"]),
        )
        for name, entry in record["modalities"].items()
    }
    return CostModel(costs, normalization=record["normalization"])


def latency_pair(cost_model: CostModel) -> tuple[float, float]:
    """``(text_llm_ms, audio_ms)`` -- the two latencies a routed system pays."""
    return (
        float(cost_model.costs["text_llm"].latency_ms),
        float(cost_model.costs["audio"].latency_ms),
    )


def load_timing(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No LLM timing record at {path}. Run the llm stage for this split; the "
            f"cost model prices the LLM at measured wall clock and will not guess it."
        )
    return json.loads(path.read_text(encoding="utf-8"))
