"""AHSEF Stage 3: dynamic Text -> Audio modality acquisition and fusion.

Stage 1 established the evidence layer and proved that the only defensible
co-split pair in this project is Text + Audio.  Stage 2 added an LLM as a text
evidence source and showed its self-reported uncertainty is a weak routing
signal on its own.  Stage 3 closes the loop: a validation-trained expected-gain
estimator (HSIG) and a cost/latency-aware utility policy (UGAPR) decide, per
sample and without seeing a label, whether acquiring audio is worth it.

Nothing here retrains a frozen baseline, rewrites a Stage 1/2 artefact, or
fabricates a modality.  Every sub-module states which split it is allowed to
learn from and raises when asked to learn from another.
"""

from __future__ import annotations

#: Stage 3's experiment identity, stamped into every artefact it writes.
STAGE3_EXPERIMENT_ID = "stage3_text_audio_v1"

#: The claim this stage is entitled to make.  Quoted verbatim in the reports so
#: it cannot drift into something stronger than the evidence supports.
STAGE3_CLAIM = (
    "AHSEF dynamically determines whether additional audio evidence is worth "
    "acquiring after an initial LLM-based text assessment, using a "
    "validation-trained expected-gain estimator and a cost/latency-aware "
    "utility policy."
)

STAGE3_CLAIM_LIMITS = (
    "This is a Text->Audio proof of concept. It is NOT a demonstration that "
    "AHSEF selects the best modality among all modalities: the available "
    "corpora provide no sample with two simultaneously available candidates, "
    "so the candidate set has exactly one member. Physiology is a separate "
    "3-class task and is never fused into the 7-class emotion task."
)
