"""Deterministic mapping from an LLM's emotion word into the canonical seven.

An LLM answers in language, and language has synonyms.  The canonical label
space is a scientific contract (``src.common.labels``), so the translation
between the two has to be explicit, total, and auditable -- never a fuzzy
match, and never a silent guess.

Three outcomes, and the difference between the last two matters:

``mapped``
    An unambiguous synonym of exactly one canonical class.

``ambiguous``
    A real emotion word this project refuses to assign, because reasonable
    taxonomies disagree about it.  ``excited`` is happy in one scheme and
    surprise in another; ``shocked`` is surprise or fear; ``contempt`` is a
    class of its own in Ekman-plus schemes and is not one of our seven.
    Guessing would fabricate a label, so these are rejected *by name* and
    counted separately.

``unknown``
    Anything else -- a malformed response, a hallucinated class, an empty
    string.

Only the first is a prediction.  The other two are logged as errors with the
offending text, and the sample is recorded as an abstention rather than being
quietly folded into ``neutral``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.common.labels import CANONICAL_EMOTION_CLASSES


#: Re-exported so LLM code never re-declares the class order.
CANONICAL_EMOTIONS: tuple[str, ...] = CANONICAL_EMOTION_CLASSES


class EmotionMappingError(ValueError):
    """Raised when an LLM emotion word cannot be mapped to a canonical class."""

    def __init__(self, raw: str, reason: str, kind: str):
        super().__init__(f"Cannot map {raw!r} to a canonical emotion: {reason}")
        self.raw = raw
        self.reason = reason
        #: ``"ambiguous"`` or ``"unknown"`` -- reported separately in the artefacts.
        self.kind = kind


# ============================================================
# The table
# ============================================================
#
# Every entry is a word whose canonical class is not in reasonable dispute.
# When in doubt, a word belongs in AMBIGUOUS below, not here.

_SYNONYMS: dict[str, tuple[str, ...]] = {
    "neutral": (
        "neutral", "calm", "indifferent", "impassive", "unemotional",
        "no emotion", "none", "no_emotion", "emotionless", "flat",
    ),
    "happy": (
        "happy", "happiness", "joy", "joyful", "joyous", "glad", "gladness",
        "pleased", "delighted", "delight", "cheerful", "elated", "content",
        "contentment", "amused", "amusement",
    ),
    "sad": (
        "sad", "sadness", "sorrow", "sorrowful", "unhappy", "unhappiness",
        "grief", "grieving", "downcast", "dejected", "despondent", "miserable",
    ),
    "angry": (
        "angry", "anger", "mad", "furious", "fury", "irate", "enraged", "rage",
        "annoyed", "annoyance", "irritated", "irritation", "indignant",
        "outraged", "outrage",
    ),
    "fear": (
        "fear", "fearful", "afraid", "scared", "frightened", "terrified",
        "terror", "panic", "panicked", "dread", "apprehensive", "apprehension",
    ),
    "disgust": (
        "disgust", "disgusted", "disgusting", "revulsion", "revolted",
        "repulsed", "repulsion", "repugnance", "loathing", "sickened",
    ),
    "surprise": (
        "surprise", "surprised", "surprising", "astonished", "astonishment",
        "amazed", "amazement", "astounded", "startled", "taken aback",
    ),
}

#: Words that name a real affective state but whose canonical class is
#: genuinely contested.  Rejected by name so the artefacts can distinguish
#: "the taxonomy does not cover this" from "the model produced nonsense".
AMBIGUOUS: Mapping[str, str] = MappingProxyType({
    "excited": "excitement is happy in some taxonomies and surprise in others",
    "excitement": "excitement is happy in some taxonomies and surprise in others",
    "anxious": "anxiety sits between fear and sadness and is not one of the seven",
    "anxiety": "anxiety sits between fear and sadness and is not one of the seven",
    "nervous": "nervousness sits between fear and neutral and is not one of the seven",
    "shocked": "shock reads as surprise or fear depending on valence",
    "shock": "shock reads as surprise or fear depending on valence",
    "contempt": "contempt is a separate class in Ekman-plus schemes, not one of the seven",
    "frustrated": "frustration spans anger and sadness",
    "frustration": "frustration spans anger and sadness",
    "confused": "confusion is an epistemic state, not one of the seven emotions",
    "confusion": "confusion is an epistemic state, not one of the seven emotions",
    "bored": "boredom is not one of the seven and is not reliably neutral",
    "boredom": "boredom is not one of the seven and is not reliably neutral",
    "sarcastic": "sarcasm is a pragmatic register, not an emotion class",
    "mixed": "a mixed emotion names no single class",
    "other": "'other' names no class in the canonical space",
    "unknown": "'unknown' names no class in the canonical space",
    "ambiguous": "'ambiguous' names no class in the canonical space",
})


def _build_lookup() -> Mapping[str, str]:
    table: dict[str, str] = {}
    for canonical, words in _SYNONYMS.items():
        if canonical not in CANONICAL_EMOTIONS:
            raise ValueError(
                f"Synonym table declares {canonical!r}, which is not a canonical class"
            )
        for word in words:
            key = _normalise(word)
            if key in table and table[key] != canonical:
                raise ValueError(
                    f"Synonym {word!r} is claimed by both {table[key]!r} and {canonical!r}"
                )
            table[key] = canonical
    missing = set(CANONICAL_EMOTIONS) - set(_SYNONYMS)
    if missing:
        raise ValueError(f"No synonym entry for canonical classes {sorted(missing)}")
    overlap = set(table) & {_normalise(word) for word in AMBIGUOUS}
    if overlap:
        raise ValueError(
            f"Words {sorted(overlap)} appear in both the synonym table and the "
            f"ambiguous list; a word cannot be both mappable and refused."
        )
    return MappingProxyType(table)


def _normalise(raw: str) -> str:
    """Casefold, strip accents and punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(raw))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.strip().casefold()
    # Underscores and hyphens are separators, not content: "no_emotion" and
    # "no emotion" are the same answer.
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^\w\s]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


LOOKUP: Mapping[str, str] = _build_lookup()

#: Words that mark a refusal to answer rather than a wrong answer.
ABSTENTION_TOKENS: frozenset[str] = frozenset({"abstain", "abstained", "decline", "declined"})


@dataclass(frozen=True)
class MappingResult:
    """The outcome of mapping one LLM emotion word."""

    raw: str
    normalized: str
    canonical: str | None
    class_id: int | None
    kind: str  # "mapped" | "ambiguous" | "unknown" | "abstain"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind == "mapped"

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "normalized": self.normalized,
            "canonical": self.canonical, "class_id": self.class_id,
            "kind": self.kind, "reason": self.reason,
        }


def classify_emotion_word(raw: str | None) -> MappingResult:
    """Map one word, reporting *why* when it cannot be mapped.

    Never raises -- callers that want an exception use :func:`map_emotion`.
    """
    if raw is None:
        return MappingResult("", "", None, None, "unknown", "no emotion word was returned")
    normalized = _normalise(raw)
    if not normalized:
        return MappingResult(str(raw), "", None, None, "unknown", "empty after normalisation")
    if normalized in ABSTENTION_TOKENS:
        return MappingResult(str(raw), normalized, None, None, "abstain",
                             "the model explicitly abstained")

    canonical = LOOKUP.get(normalized)
    if canonical is not None:
        return MappingResult(
            str(raw), normalized, canonical, CANONICAL_EMOTIONS.index(canonical), "mapped",
        )
    if normalized in AMBIGUOUS:
        return MappingResult(str(raw), normalized, None, None, "ambiguous", AMBIGUOUS[normalized])

    # A comma- or slash-separated list is a multi-label answer, which the task
    # forbids; say so specifically rather than calling it unknown.
    if re.search(r"[,;]| and | or ", str(raw)):
        return MappingResult(
            str(raw), normalized, None, None, "unknown",
            "the response names more than one emotion; the task requires exactly one",
        )
    return MappingResult(
        str(raw), normalized, None, None, "unknown",
        "not a recognised synonym of any canonical emotion",
    )


def map_emotion(raw: str | None) -> str:
    """Return the canonical class for ``raw`` or raise :class:`EmotionMappingError`."""
    result = classify_emotion_word(raw)
    if result.ok:
        return result.canonical  # type: ignore[return-value]
    raise EmotionMappingError(result.raw, result.reason or "unmappable", result.kind)


def emotion_id(raw: str | None) -> int:
    """Canonical class id for ``raw``, in the declared class order."""
    return CANONICAL_EMOTIONS.index(map_emotion(raw))


def map_score_keys(scores: Mapping[str, float]) -> dict[str, float]:
    """Map the keys of a per-class score dictionary into canonical names.

    Every canonical class must be present exactly once after mapping; a
    partial or duplicated set is an error rather than something to fill in,
    because a silently zero-filled class would look like a confident denial.
    """
    mapped: dict[str, float] = {}
    for key, value in scores.items():
        canonical = map_emotion(key)
        if canonical in mapped:
            raise EmotionMappingError(
                key, f"two score keys both map to {canonical!r}", "unknown"
            )
        number = float(value)
        if number < 0 or number != number:
            raise EmotionMappingError(key, f"score {value!r} is negative or NaN", "unknown")
        mapped[canonical] = number
    missing = [name for name in CANONICAL_EMOTIONS if name not in mapped]
    if missing:
        raise EmotionMappingError(
            ", ".join(sorted(scores)), f"no score for {missing}", "unknown"
        )
    return {name: mapped[name] for name in CANONICAL_EMOTIONS}


def mapping_provenance() -> dict:
    """The mapping table itself, recorded alongside every experiment."""
    return {
        "canonical_classes": list(CANONICAL_EMOTIONS),
        "synonyms": {name: list(words) for name, words in _SYNONYMS.items()},
        "ambiguous_refused": dict(AMBIGUOUS),
        "abstention_tokens": sorted(ABSTENTION_TOKENS),
        "normalisation": "NFKD, accent-stripped, casefolded, punctuation removed, "
                         "separators collapsed to single spaces",
        "policy": "Only unambiguous synonyms are mapped. Contested words are refused "
                  "by name and counted as 'ambiguous'; anything else is 'unknown'. "
                  "Neither is folded into a canonical class.",
    }
