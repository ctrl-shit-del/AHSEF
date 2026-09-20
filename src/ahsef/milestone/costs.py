"""PHASE E (cost side) -- the frozen cost configuration, read, never re-measured.

Phase E's cost requirement is specific: the strong audio expert's price must be
the one that was measured when the expert was built, and it must include the
wav2vec2 encoder rather than only the probe head that reads its cached output.

That distinction is worth three orders of magnitude.  The probe's measured
forward pass on this pool is 0.18 ms/sample because the encoder ran hours
earlier during feature extraction; the encoder itself costs 288.9 ms/clip on the
same host.  A budget curve priced at 0.18 ms would conclude that acquiring audio
is nearly free, and the whole AHSEF premise -- that modalities are expensive
enough to be worth withholding -- would evaporate as a measurement artefact.  So
the deployment cost is the sum, and :func:`load_frozen_costs` reads it from the
Phase B-1 artefact rather than timing anything on the evaluation pool.

Re-measuring here would be worse than merely redundant.  A cost measured on the
same pool the policy is selected on is a second quantity fitted to that pool,
and the acquisition rate a cost-aware policy chooses would then depend on the
host's load during this particular run.  The numbers below are inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

#: Where each frozen cost comes from.  Nothing in this module measures.
PHASE_B1_ARTEFACT = (
    Path("experiments") / "ahsef" / "analysis" / "audio_expert"
    / "phase_b1_comparison.json"
)
STAGE3_FROZEN_CONFIG = (
    Path("experiments") / "ahsef" / "stage3_text_audio" / "frozen_config.json"
)
STRONG_EXTRACTION_PROVENANCE = (
    Path("experiments") / "audio_strong" / "features" / "validation"
    / "extraction_provenance.json"
)

#: Phase E's brief names ~289 ms/sample for strong audio.  The artefact carries
#: the precise measured value and is always preferred; this constant exists so
#: that a missing artefact fails against a known number rather than silently
#: against zero.
DECLARED_AUDIO_STRONG_MS = 289.0

COST_POLICY = {
    "measured_on": "the Phase B-1 validation pass, not on the Phase D/E pool",
    "re_measurement": "prohibited -- Phase E reads costs, it does not time anything",
    "audio_strong_components": (
        "frozen WAV2VEC2_BASE encoder (dominant) plus the trainable "
        "LayerWeightedProbe head"
    ),
    "why_not_the_cached_head_latency": (
        "The probe reads features the encoder already produced. At deployment the "
        "encoder runs per request, so charging only the head would understate the "
        "price of acquiring audio by roughly three orders of magnitude and would "
        "make every budget on the curve look free."
    ),
    "compute_units": (
        "parameters x input_elements, the multiply-accumulate-order proxy used "
        "throughout this project. It is a proxy and is labelled as one."
    ),
    "text_llm_latency": (
        "measured end-to-end wall clock from the Stage 3 pass, including queueing "
        "and cloud scheduling -- not generation time, which Stage 2 measured at "
        "roughly a fifth of it."
    ),
}


class CostConfigurationError(RuntimeError):
    """Raised when a frozen cost cannot be read and would have to be invented."""


@dataclass(frozen=True)
class FrozenCosts:
    """The per-sample price of each modality, as frozen before Phase D began."""

    text_llm_latency_ms: float
    audio_strong_latency_ms: float
    text_llm_compute_units: float
    audio_strong_compute_units: float
    audio_strong_encoder_ms: float
    audio_strong_head_ms: float
    audio_strong_encoder_parameters: int
    audio_strong_head_parameters: int
    sources: dict
    frozen: bool = True

    def to_dict(self) -> dict:
        return {
            "text_llm": {
                "latency_ms_per_sample": self.text_llm_latency_ms,
                "compute_units_per_sample": self.text_llm_compute_units,
            },
            "audio_strong": {
                "latency_ms_per_sample": self.audio_strong_latency_ms,
                "compute_units_per_sample": self.audio_strong_compute_units,
                "encoder_ms_per_sample": self.audio_strong_encoder_ms,
                "head_ms_per_sample": self.audio_strong_head_ms,
                "encoder_parameters": self.audio_strong_encoder_parameters,
                "head_parameters": self.audio_strong_head_parameters,
                "includes_encoder": True,
                "includes_head": True,
            },
            "audio_strong_share_of_text_latency": (
                self.audio_strong_latency_ms / self.text_llm_latency_ms
                if self.text_llm_latency_ms else None
            ),
            "policy": dict(COST_POLICY),
            "sources": dict(self.sources),
            "frozen": self.frozen,
            "re_measured_on_evaluation_pool": False,
        }

    def fingerprint(self) -> str:
        payload = json.dumps({
            "text_llm_latency_ms": self.text_llm_latency_ms,
            "audio_strong_latency_ms": self.audio_strong_latency_ms,
            "text_llm_compute_units": self.text_llm_compute_units,
            "audio_strong_compute_units": self.audio_strong_compute_units,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path, what: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise CostConfigurationError(
            f"Phase E needs the frozen {what} at {path} and will not re-measure it. "
            f"Run the phase that produces it before running Phase D/E."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_costs(
    phase_b1: Path | str = PHASE_B1_ARTEFACT,
    stage3_config: Path | str = STAGE3_FROZEN_CONFIG,
    extraction: Path | str = STRONG_EXTRACTION_PROVENANCE,
) -> FrozenCosts:
    """Assemble the Phase D/E cost configuration from artefacts already on disk."""
    b1 = _read(Path(phase_b1), "Phase B-1 latency measurement")
    stage3 = _read(Path(stage3_config), "Stage 3 cost model")
    provenance = _read(Path(extraction), "wav2vec2 extraction provenance")

    strong = ((b1.get("latency") or {}).get("audio_strong")) or {}
    deployment = strong.get("deployment_ms_per_sample")
    encoder_ms = strong.get("encoder_ms_per_sample")
    head_ms = strong.get("measured_ms_per_sample")
    if deployment is None or encoder_ms is None:
        raise CostConfigurationError(
            f"{phase_b1} records no strong-audio deployment latency including the "
            f"encoder. Phase E refuses to price audio at the cached-head latency; "
            f"the declared figure is ~{DECLARED_AUDIO_STRONG_MS:.0f} ms/sample."
        )

    extractor = provenance.get("extractor") or {}
    encoder_parameters = int(extractor.get("parameters") or 0)
    encoder_elements = int(
        float(extractor.get("sample_rate") or 0)
        * float(extractor.get("max_seconds") or 0)
    )
    head_parameters = int(
        ((b1.get("resource_cost") or {}).get("audio_strong") or {})
        .get("trainable_parameters") or 0
    )
    head_elements = (
        int(extractor.get("num_layers") or 0) * int(extractor.get("feature_dim") or 0)
    )
    if not (encoder_parameters and encoder_elements):
        raise CostConfigurationError(
            f"{extraction} does not record the encoder geometry needed for the "
            f"compute proxy; nothing is assumed in its place."
        )

    llm = ((stage3.get("costs") or {}).get("modalities") or {}).get("text_llm") or {}
    if not llm.get("latency_ms"):
        raise CostConfigurationError(
            f"{stage3_config} records no measured text_llm latency."
        )

    return FrozenCosts(
        text_llm_latency_ms=float(llm["latency_ms"]),
        audio_strong_latency_ms=float(deployment),
        text_llm_compute_units=float(llm.get("compute_units") or 0.0),
        audio_strong_compute_units=float(
            encoder_parameters * encoder_elements + head_parameters * head_elements
        ),
        audio_strong_encoder_ms=float(encoder_ms),
        audio_strong_head_ms=float(head_ms or 0.0),
        audio_strong_encoder_parameters=encoder_parameters,
        audio_strong_head_parameters=head_parameters,
        sources={
            "audio_strong_latency": str(phase_b1),
            "audio_strong_geometry": str(extraction),
            "text_llm": str(stage3_config),
            "encoder": extractor.get("bundle"),
            "encoder_input_elements": encoder_elements,
            "head_input_elements": head_elements,
        },
    )


def assert_encoder_is_charged(costs: FrozenCosts, minimum_ms: float = 100.0) -> None:
    """Refuse a cost model that prices audio at the cached-head latency.

    The failure this guards against is silent: a budget curve computed with the
    head-only figure looks entirely reasonable and simply reports that audio
    costs almost nothing. The check is a floor rather than an equality, so a
    future faster encoder does not trip it -- but 0.18 ms does.
    """
    if costs.audio_strong_latency_ms < minimum_ms:
        raise CostConfigurationError(
            f"Strong audio is priced at {costs.audio_strong_latency_ms:.3f} ms/sample, "
            f"below the {minimum_ms:.0f} ms floor. That is the cached-head latency, "
            f"not the deployment cost: the wav2vec2 encoder is missing from it."
        )
    if costs.audio_strong_latency_ms <= costs.audio_strong_head_ms:
        raise CostConfigurationError(
            "The strong-audio deployment cost does not exceed its head-only cost, "
            "so the encoder has not been charged."
        )
