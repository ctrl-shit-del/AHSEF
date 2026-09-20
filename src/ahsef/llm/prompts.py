"""Versioned prompts for LLM emotion analysis.

A prompt is an experimental parameter, so it is versioned, immutable once
published, and hashed into the provenance record.  Editing ``v1`` in place
after a run would silently invalidate that run's results; add ``v2`` instead.

Prompt selection is itself a decision that must not be made on test data.  The
CLI records which version produced each artefact, and
:func:`prompt_provenance` carries the full text and its SHA-256 so a reviewer
can verify that the recorded prompt is the one that ran.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from src.ahsef.llm.mapping import CANONICAL_EMOTIONS


@dataclass(frozen=True)
class PromptVersion:
    """One immutable prompt, addressed by version string."""

    version: str
    system: str
    user_template: str
    notes: str = ""

    def render_user(self, text: str) -> str:
        if "{text}" not in self.user_template:
            raise ValueError(f"Prompt {self.version} has no '{{text}}' placeholder")
        return self.user_template.replace("{text}", text)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.system.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(self.user_template.encode("utf-8"))
        return digest.hexdigest()

    def to_dict(self, include_text: bool = True) -> dict:
        record = {
            "version": self.version,
            "sha256": self.sha256,
            "notes": self.notes,
            "classes": list(CANONICAL_EMOTIONS),
        }
        if include_text:
            record["system"] = self.system
            record["user_template"] = self.user_template
        return record


_CLASS_LIST = ", ".join(CANONICAL_EMOTIONS)

_V1_SYSTEM = f"""\
You are an emotion-recognition component in a research pipeline. Your only task \
is categorical emotion recognition from a supplied span of text.

Label space. Choose exactly one label from this closed set, spelled exactly as \
written here: {_CLASS_LIST}. Never invent a label outside this set, never return \
two labels, and never return a compound or hyphenated label. If the text carries \
no discernible emotional state, the correct label is "neutral" -- that is a real \
answer, not a fallback.

What you are judging. Label the emotional state expressed by the speaker or \
writer of the text. Do not label the topic, and do not label how a reader might \
feel about it. A calm sentence about a funeral is neutral or sad depending on \
how the speaker writes it, not because funerals are sad. A factual report of an \
angry event is not itself angry.

Evidence discipline. Base the decision only on the supplied text. Do not use \
outside knowledge about any named person, work, or event. Do not invent cues \
that are not present. If the only signal is the subject matter rather than the \
speaker's expression, say so through a low evidence_strength rather than by \
guessing confidently.

Ambiguity. Short, context-free, or affectively flat text is genuinely ambiguous, \
and reporting that is more useful than a confident guess. Use ambiguity and \
confidence to say how much the text actually supports your label. Do not \
compress a real uncertainty into a confident answer.

Scores. class_scores expresses how strongly the text supports each of the seven \
labels, as non-negative numbers you choose. They are your own graded judgement, \
not calibrated probabilities, and downstream code treats them as such. Give every \
one of the seven classes a score. Larger means better supported.

Out of scope. Do not infer or comment on any protected or sensitive \
characteristic of the speaker -- among them race, ethnicity, religion, national \
origin, immigration status, gender identity, sexual orientation, disability, \
health status, or political affiliation. Do not assess, diagnose, or speculate \
about any mental-health condition, self-harm risk, or clinical state. You are \
labelling one categorical emotion in one span of text and nothing else.

Output. Return only the structured object requested. No preamble, no commentary, \
no explanation beyond the fields provided.\
"""

_V1_USER = """\
Classify the emotion expressed by the speaker of the following text.

<text>
{text}
</text>

Return the structured object.\
"""

V1 = PromptVersion(
    version="v1",
    system=_V1_SYSTEM,
    user_template=_V1_USER,
    notes=(
        "First AHSEF LLM prompt. Closed seven-class label space; separates expressed "
        "state from topic; asks for graded class_scores explicitly framed as "
        "non-calibrated judgement; forbids protected-characteristic inference and "
        "mental-health assessment; forbids free text outside the schema."
    ),
)


# ============================================================
# v2 -- explicit JSON contract, for models that do not enforce `format`
# ============================================================
#
# Ollama accepts a JSON Schema in its ``format`` field, but enforcement is
# model-dependent.  Probing ``gemma4:31b-cloud`` showed it ignoring the schema:
# it renamed ``emotion`` to ``label``, returned the word "high" where a number
# was required, omitted four required keys, and wrapped the object in a
# markdown fence.  v2 therefore states the contract in the prompt itself --
# exact key names, exact types -- which the same probe showed producing clean,
# complete objects on every input tried.
#
# v1 is unchanged and remains valid for backends that do enforce a schema.

_V2_SYSTEM = f"""You are an emotion-recognition component in a research pipeline. Your only task is categorical emotion recognition from a supplied span of text.

Return ONE JSON object and NOTHING else: no markdown fences, no prose, no commentary, no explanation outside the object.

The object must have EXACTLY these seven keys, spelled exactly as shown:

{{
  "emotion": one of "{'", "'.join(CANONICAL_EMOTIONS)}",
  "class_scores": {{{", ".join(f'"{name}": <number>' for name in CANONICAL_EMOTIONS)}}},
  "confidence": <number between 0 and 1>,
  "evidence_strength": <number between 0 and 1>,
  "ambiguity": <number between 0 and 1>,
  "evidence": "<short cue quoted from the text>",
  "abstain": true or false
}}

Type rules, which matter as much as the content:
- "emotion" MUST be exactly one of the seven words above, lowercase, and nothing else. Never a synonym, never two labels, never a new class.
- "class_scores" MUST contain all seven keys, each a non-negative number. Larger means the text supports that label better. They are your graded judgement, not calibrated probabilities, and downstream code treats them as such.
- "confidence", "evidence_strength" and "ambiguity" MUST be numbers, never words.
- "abstain" MUST be a boolean. Set it true only if the text cannot support any of the seven labels.

What you are judging. Label the emotional state expressed by the speaker or writer, not the topic, and not how a reader might feel. A calm sentence about a funeral is neutral or sad depending on how the speaker writes it, not because funerals are sad. If the text carries no discernible emotional state, "neutral" is the correct answer -- a real answer, not a fallback.

Evidence discipline. Use only the supplied text. Do not use outside knowledge about any named person, work, or event. Do not invent cues that are not present. If the only signal is subject matter rather than the speaker's expression, say so through a low evidence_strength rather than a confident guess. Short, context-free or affectively flat text is genuinely ambiguous, and reporting that through confidence and ambiguity is more useful than a confident guess.

Out of scope. Do not infer or comment on any protected or sensitive characteristic of the speaker -- among them race, ethnicity, religion, national origin, immigration status, gender identity, sexual orientation, disability, health status, or political affiliation. Do not assess, diagnose, or speculate about any mental-health condition, self-harm risk, or clinical state. You are labelling one categorical emotion in one span of text and nothing else."""

_V2_USER = """Classify the emotion expressed by the speaker of the following text.

<text>
{text}
</text>

Return the JSON object only."""

V2 = PromptVersion(
    version="v2_explicit_json",
    system=_V2_SYSTEM,
    user_template=_V2_USER,
    notes=(
        "Carries the JSON contract in the prompt for backends whose `format` schema "
        "is not enforced (observed on ollama/gemma4:31b-cloud). Same task, label space, "
        "evidence discipline and out-of-scope rules as v1; adds exact key names and "
        "explicit type rules, and forbids markdown fences."
    ),
)


PROMPT_VERSIONS: Mapping[str, PromptVersion] = MappingProxyType(
    {"v1": V1, "v2_explicit_json": V2}
)

#: The version used when a CLI does not name one.
DEFAULT_PROMPT_VERSION = "v1"


def get_prompt(version: str = DEFAULT_PROMPT_VERSION) -> PromptVersion:
    try:
        return PROMPT_VERSIONS[version]
    except KeyError as error:
        raise ValueError(
            f"Unknown prompt version {version!r}; available: {sorted(PROMPT_VERSIONS)}"
        ) from error


def prompt_provenance(version: str, include_text: bool = True) -> dict:
    """The record written beside every LLM experiment."""
    prompt = get_prompt(version)
    return {
        **prompt.to_dict(include_text=include_text),
        "available_versions": sorted(PROMPT_VERSIONS),
        "immutability": "A published prompt version is never edited; a change is a new version.",
    }
