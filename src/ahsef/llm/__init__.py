"""LLM-based semantic text reasoning as an AHSEF modality provider.

Stage 2 adds one new evidence source: a language model asked to name the
emotion in a piece of text.  It is deliberately kept at arm's length from the
five frozen statistical baselines, because it is not the same kind of thing:

* it emits words that must be mapped into the canonical label space rather
  than logits over it (:mod:`src.ahsef.llm.mapping`);
* its self-reported confidence is **not** a calibrated posterior, and the code
  never lets it be mistaken for one (:mod:`src.ahsef.llm.uncertainty`);
* it costs money and wall-clock per sample, and both are measured
  (:mod:`src.ahsef.llm.provider`).

What it shares with the other modalities is the output contract: a
:class:`~src.ahsef.inference.PredictionSet` with the same columns, so Stage 1's
evaluation, calibration, and routing-log machinery applies unchanged.
"""

from src.ahsef.llm.mapping import (
    CANONICAL_EMOTIONS,
    EmotionMappingError,
    map_emotion,
)
from src.ahsef.llm.prompts import PROMPT_VERSIONS, get_prompt
from src.ahsef.llm.schema import LLMEmotionResponse

__all__ = [
    "CANONICAL_EMOTIONS",
    "EmotionMappingError",
    "LLMEmotionResponse",
    "PROMPT_VERSIONS",
    "get_prompt",
    "map_emotion",
]
