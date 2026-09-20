"""Anthropic Claude provider for the AHSEF LLM modality.

Uses the official ``anthropic`` SDK with schema-constrained output
(``output_config.format``), so the response body is guaranteed to be JSON
matching :func:`~src.ahsef.llm.schema.response_json_schema` and the parser
never has to scrape prose.

Two properties of the current models matter for the experimental record and
are surfaced rather than hidden:

* **There is no sampling control.**  ``temperature``, ``top_p``, ``top_k`` are
  rejected by Opus 5 / Sonnet 5 and the 4.7+ family, and the API exposes no
  seed.  A run is therefore **not** bit-reproducible; ``deterministic`` is
  ``False`` and the provenance says so in words.  Reproducibility comes from
  the recorded transcript (:class:`~src.ahsef.llm.provider.ReplayProvider`),
  not from re-running.
* **A refusal is an HTTP 200.**  ``stop_reason == "refusal"`` is checked before
  the body is trusted, and is recorded as a call-level error rather than being
  parsed as an answer.

Effort is set to ``low`` by default: this is single-label classification at
scale, which is exactly the workload that does not repay deep reasoning.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from src.ahsef.llm.provider import LLMCall, LLMProvider, TokenUsage


#: Default model for the stage-2 study.
DEFAULT_MODEL = "claude-opus-5"

#: Classification emits a short JSON object; a large cap only invites truncation
#: to go unnoticed. 1024 leaves generous headroom over the ~150-token schema.
DEFAULT_MAX_TOKENS = 1024

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class AnthropicProvider(LLMProvider):
    """Schema-constrained Claude call, with usage and latency measured."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = "low",
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_system_prompt: bool = True,
        api_key: str | None = None,
    ):
        try:
            import anthropic
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The 'anthropic' package is required for the Anthropic provider. "
                "Install it with `pip install anthropic`, or run against a recorded "
                "transcript with --provider replay."
            ) from error
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")

        self._anthropic = anthropic
        self.model = model
        self.max_tokens = int(max_tokens)
        self.effort = effort
        self.cache_system_prompt = bool(cache_system_prompt)
        # No seed and no temperature control on the current models: the call is
        # nondeterministic and every artefact must say so.
        self.deterministic = False
        self._client = (
            anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
            if api_key
            else anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        )

    # ------------------------------------------------------------------ call

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> LLMCall:
        # The system prompt is identical for every sample, so caching it turns
        # the bulk of the input tokens into cache reads at a tenth of the price.
        system_block: Any = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if self.cache_system_prompt
            else system
        )
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_block,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": dict(schema)},
                },
            )
        except self._anthropic.APIStatusError as error:
            return self._failed(error, started, f"{type(error).__name__}: {error}")
        except self._anthropic.APIConnectionError as error:
            return self._failed(error, started, f"connection error: {error}")

        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = _usage_of(response)
        stop_reason = getattr(response, "stop_reason", None)

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return LLMCall(
                text="", model=self.model, usage=usage, latency_ms=latency_ms,
                stop_reason=stop_reason,
                stop_details=_as_mapping(details),
                deterministic=False, provider=self.name,
                error="the model declined this request (stop_reason=refusal)",
                request_id=getattr(response, "_request_id", None),
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        error = None
        if stop_reason == "max_tokens":
            # A truncated JSON body would fail parsing anyway; naming the cause
            # keeps the failure diagnosable instead of "malformed json".
            error = f"response truncated at max_tokens={self.max_tokens}"
        return LLMCall(
            text=text, model=self.model, usage=usage, latency_ms=latency_ms,
            stop_reason=stop_reason, deterministic=False, provider=self.name,
            error=error, request_id=getattr(response, "_request_id", None),
        )

    def _failed(self, error: Exception, started: float, message: str) -> LLMCall:
        return LLMCall(
            text="", model=self.model, usage=TokenUsage(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            stop_reason=None, deterministic=False, provider=self.name, error=message,
        )

    # ------------------------------------------------------------ provenance

    def provenance(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "effort": self.effort,
            "output_format": "json_schema via output_config.format",
            "prompt_caching": self.cache_system_prompt,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "seed": None,
            "deterministic": False,
            "nondeterminism_note": (
                "This model exposes no seed, and temperature/top_p/top_k are rejected "
                "by the API, so identical inputs may yield different outputs. Runs are "
                "NOT bit-reproducible. Reproducibility is provided by the recorded "
                "transcript, which ReplayProvider replays exactly."
            ),
        }


def _usage_of(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
    )


def _as_mapping(value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in ("type", "category", "explanation")
        if getattr(value, key, None) is not None
    } or None
