"""The structured output contract for LLM emotion analysis.

The schema is deliberately honest about what a language model can and cannot
supply.  It asks for:

``emotion``
    One label, to be mapped into the canonical seven by
    :mod:`src.ahsef.llm.mapping`.

``class_scores``
    Seven non-negative numbers.  These are **graded self-assessments, not
    posterior probabilities.**  The field is named ``scores`` everywhere and
    normalising them yields a ``score distribution``, never a ``probability
    distribution``.  Nothing downstream calls them calibrated.

``confidence`` / ``evidence_strength`` / ``ambiguity``
    Self-reported scalars in ``[0, 1]``.  Same caveat, and they are carried
    under ``llm_``-prefixed names so no plotting or reporting code can confuse
    them with the statistical baselines' posterior confidence.

``abstain``
    An explicit refusal, which is a legitimate answer and is recorded as one
    rather than being coerced into ``neutral``.

The JSON Schema in :func:`response_json_schema` is what constrains the model
(``output_config.format``); :class:`LLMEmotionResponse` is what the rest of the
pipeline consumes after validation.  Parsing is strict: a response that does
not satisfy the contract raises rather than being patched up, because a
silently repaired response is an unlabelled measurement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.ahsef.llm.mapping import (
    CANONICAL_EMOTIONS,
    EmotionMappingError,
    MappingResult,
    classify_emotion_word,
    map_score_keys,
)


SCHEMA_VERSION = "ahsef.llm.response.v1"

#: Ranges every self-reported scalar must satisfy.
_UNIT_FIELDS = ("confidence", "evidence_strength", "ambiguity")


class LLMResponseError(ValueError):
    """Raised when an LLM response does not satisfy the output contract."""

    def __init__(self, reason: str, raw: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def response_json_schema() -> dict:
    """JSON Schema handed to the model as ``output_config.format``."""
    return {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "enum": list(CANONICAL_EMOTIONS),
                "description": "The single emotional state expressed by the speaker.",
            },
            "class_scores": {
                "type": "object",
                "description": (
                    "How strongly the text supports each label. Non-negative graded "
                    "judgement, not calibrated probabilities. All seven required."
                ),
                "properties": {
                    name: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    for name in CANONICAL_EMOTIONS
                },
                "required": list(CANONICAL_EMOTIONS),
                "additionalProperties": False,
            },
            "confidence": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "How strongly you back the chosen label.",
            },
            "evidence_strength": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "How much explicit affective evidence the text actually contains.",
            },
            "ambiguity": {
                "type": "number", "minimum": 0.0, "maximum": 1.0,
                "description": "How well a competing label would also fit the text.",
            },
            "evidence": {
                "type": "string", "maxLength": 240,
                "description": "The span or cue that drove the decision. Quote or describe briefly.",
            },
            "abstain": {
                "type": "boolean",
                "description": "True only if the text cannot support any of the seven labels.",
            },
        },
        "required": [
            "emotion", "class_scores", "confidence",
            "evidence_strength", "ambiguity", "evidence", "abstain",
        ],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class LLMEmotionResponse:
    """A validated LLM answer, already mapped into the canonical label space."""

    #: Canonical class name, or ``None`` when the answer was unusable.
    emotion: str | None
    class_id: int | None
    #: Canonical-keyed non-negative scores, or ``None`` when absent/invalid.
    class_scores: Mapping[str, float] | None
    llm_confidence: float | None
    evidence_strength: float | None
    ambiguity: float | None
    evidence: str
    abstain: bool
    #: How the raw emotion word resolved -- ``mapped``/``ambiguous``/``unknown``/``abstain``.
    mapping: MappingResult | None = None
    #: Populated when the response was rejected; ``usable`` is then False.
    error: str | None = None
    error_kind: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when this response yields a canonical prediction."""
        return self.error is None and self.class_id is not None and not self.abstain

    @property
    def has_scores(self) -> bool:
        return self.class_scores is not None

    def to_dict(self, include_raw: bool = False) -> dict:
        record = {
            "schema": SCHEMA_VERSION,
            "emotion": self.emotion,
            "class_id": self.class_id,
            "class_scores": dict(self.class_scores) if self.class_scores else None,
            "llm_confidence": self.llm_confidence,
            "evidence_strength": self.evidence_strength,
            "ambiguity": self.ambiguity,
            "evidence": self.evidence,
            "abstain": self.abstain,
            "usable": self.usable,
            "error": self.error,
            "error_kind": self.error_kind,
            "mapping": self.mapping.to_dict() if self.mapping else None,
            "scores_are_calibrated_probabilities": False,
        }
        if include_raw:
            record["raw"] = dict(self.raw)
        return record

    @classmethod
    def rejected(cls, reason: str, kind: str, raw: Any = None) -> "LLMEmotionResponse":
        """A response that failed the contract, recorded rather than discarded."""
        return cls(
            emotion=None, class_id=None, class_scores=None, llm_confidence=None,
            evidence_strength=None, ambiguity=None, evidence="", abstain=False,
            mapping=None, error=reason, error_kind=kind,
            raw=raw if isinstance(raw, Mapping) else {"raw": raw},
        )


#: A model told "no markdown fences" may still emit them. Stripping the fence
#: decodes the *envelope*; it does not alter the object inside. That distinction
#: is the line between decoding and repairing, and only the former is allowed.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def unwrap_json_text(text: str) -> tuple[str, bool]:
    """Return ``(json_text, was_fenced)``.

    Only a fence around the whole response is removed. Prose mixed with an
    object is left alone: guessing which braces were meant would be repair.
    """
    match = _FENCE.match(text)
    if match:
        return match.group("body"), True
    return text, False


def _unit(value: Any, name: str) -> float:
    number = float(value)
    if number != number or not 0.0 <= number <= 1.0:
        raise LLMResponseError(f"{name}={value!r} is not a number in [0, 1]")
    return number


def parse_response(payload: str | Mapping[str, Any]) -> LLMEmotionResponse:
    """Validate and canonicalise one raw LLM answer.

    Accepts either the decoded object or the raw JSON text.  Never raises for
    a *model* mistake -- a bad answer becomes a rejected response so the run
    can continue and the failure is counted.  It raises only for a programming
    error in the caller.
    """
    fenced = False
    if isinstance(payload, (str, bytes)):
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        text, fenced = unwrap_json_text(text)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            return LLMEmotionResponse.rejected(
                f"response is not valid JSON: {error}", "malformed_json", text,
            )
    elif isinstance(payload, Mapping):
        decoded = payload
    else:
        raise TypeError(f"parse_response expects JSON text or a mapping, got {type(payload)!r}")

    if not isinstance(decoded, Mapping):
        return LLMEmotionResponse.rejected(
            f"response decoded to {type(decoded).__name__}, not an object",
            "malformed_json", decoded,
        )

    missing = [key for key in ("emotion", "confidence", "abstain") if key not in decoded]
    if missing:
        return LLMEmotionResponse.rejected(
            f"response is missing required field(s) {missing}", "missing_field", decoded,
        )

    abstain = bool(decoded.get("abstain"))
    mapping = classify_emotion_word(decoded.get("emotion"))
    if mapping.kind == "abstain":
        abstain = True

    try:
        scalars = {
            name: (_unit(decoded[name], name) if decoded.get(name) is not None else None)
            for name in _UNIT_FIELDS if name in decoded
        }
    except LLMResponseError as error:
        return LLMEmotionResponse.rejected(error.reason, "out_of_range", decoded)

    # Scores are optional in practice: a model that omits or mangles them still
    # gives a usable label, and inventing a distribution to fill the gap is
    # exactly what this project refuses to do.
    scores: Mapping[str, float] | None = None
    score_error: str | None = None
    raw_scores = decoded.get("class_scores")
    if isinstance(raw_scores, Mapping) and raw_scores:
        try:
            mapped = map_score_keys(raw_scores)
        except EmotionMappingError as error:
            score_error = f"class_scores rejected: {error}"
        else:
            scores = mapped if sum(mapped.values()) > 0 else None
            if scores is None:
                score_error = "class_scores are all zero and cannot be normalised"
    elif raw_scores is not None:
        score_error = f"class_scores is {type(raw_scores).__name__}, not an object"

    if abstain:
        return LLMEmotionResponse(
            emotion=None, class_id=None, class_scores=scores,
            llm_confidence=scalars.get("confidence"),
            evidence_strength=scalars.get("evidence_strength"),
            ambiguity=scalars.get("ambiguity"),
            evidence=str(decoded.get("evidence") or ""), abstain=True, mapping=mapping,
            error=None, error_kind="abstain", raw=decoded,
        )

    if not mapping.ok:
        return LLMEmotionResponse(
            emotion=None, class_id=None, class_scores=scores,
            llm_confidence=scalars.get("confidence"),
            evidence_strength=scalars.get("evidence_strength"),
            ambiguity=scalars.get("ambiguity"),
            evidence=str(decoded.get("evidence") or ""), abstain=False, mapping=mapping,
            error=mapping.reason, error_kind=mapping.kind, raw=decoded,
        )

    return LLMEmotionResponse(
        emotion=mapping.canonical,
        class_id=mapping.class_id,
        class_scores=scores,
        llm_confidence=scalars.get("confidence"),
        evidence_strength=scalars.get("evidence_strength"),
        ambiguity=scalars.get("ambiguity"),
        evidence=str(decoded.get("evidence") or ""),
        abstain=False,
        mapping=mapping,
        error=None,
        error_kind=score_error and "scores_rejected",
        raw=decoded,
    )


def schema_provenance() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "json_schema": response_json_schema(),
        "score_semantics": (
            "class_scores are the model's graded self-assessment of how well each label "
            "fits the text. They are NOT calibrated probabilities and are never reported "
            "as such; normalising them yields a score distribution."
        ),
        "confidence_semantics": (
            "confidence, evidence_strength and ambiguity are self-reported scalars, "
            "carried under llm_-prefixed names throughout so they cannot be mistaken "
            "for a statistical model's posterior confidence."
        ),
        "rejection_policy": (
            "A response failing the contract is recorded as rejected with a reason and "
            "counted; it is never repaired, and an unmappable label is never folded "
            "into neutral."
        ),
    }
