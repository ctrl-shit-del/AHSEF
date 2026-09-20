"""Ollama provider for the AHSEF LLM modality.

Talks to a local Ollama daemon over its REST API (``POST /api/chat``) using
only the standard library, so the pipeline gains no new dependency.

Three properties of this backend were established by probing the running
daemon rather than assumed, and each one is recorded in the provenance:

**Sampling is controllable, so runs are reproducible.**  Unlike the hosted
Claude models, Ollama accepts ``temperature`` and ``seed``.  At
``temperature=0`` with a fixed seed the provider reports ``deterministic=True``
-- a materially stronger reproducibility guarantee than Stage 2's original
design assumed.

**``format`` is not enforced for every model.**  Ollama accepts a JSON Schema
in ``format``, but the ``gemma4:31b-cloud`` cloud model was observed ignoring
it: it renamed keys, returned a word where a number was required, omitted
required fields, and wrapped the object in a markdown fence.  The schema is
still sent (it costs nothing and helps where it is honoured), but the *prompt*
carries the real contract, and ``schema_enforced`` is recorded as observed
rather than claimed.

**Execution may be local or cloud.**  A model whose ``/api/tags`` entry carries
a ``remote_host`` runs on Ollama's servers, not this machine.  That changes what
"latency" and "cost" mean, so :meth:`OllamaProvider.probe` reads it and the
provenance states which one applied.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from src.ahsef.llm.provider import LLMCall, LLMProvider, TokenUsage


DEFAULT_HOST = "http://localhost:11434"

#: Ollama reports durations in nanoseconds.
_NS_PER_MS = 1_000_000


class OllamaUnavailable(RuntimeError):
    """Raised when the Ollama daemon cannot be reached or lacks the model."""


def _post(url: str, payload: Mapping[str, Any], timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def probe_model(model: str, host: str = DEFAULT_HOST, timeout: float = 10.0) -> dict:
    """Return the daemon's own record for ``model``, or raise.

    Reading the tag list rather than trusting the caller is what lets the
    experiment record the exact model, its size, and -- decisively -- whether
    it executes locally or on Ollama's cloud.
    """
    try:
        tags = _get(f"{host.rstrip('/')}/api/tags", timeout)
    except (urllib.error.URLError, OSError) as error:
        raise OllamaUnavailable(
            f"Cannot reach the Ollama daemon at {host}: {error}. Start it with "
            f"`ollama serve`, or run against a recorded transcript with "
            f"--provider replay."
        ) from error

    models = {entry.get("name"): entry for entry in tags.get("models", [])}
    entry = models.get(model)
    if entry is None:
        raise OllamaUnavailable(
            f"Model {model!r} is not available to the daemon at {host}. "
            f"Present: {sorted(models)}. Pull it with `ollama pull {model}` -- "
            f"this run will not silently substitute a different model."
        )
    details = entry.get("details") or {}
    remote_host = entry.get("remote_host")
    return {
        "name": entry.get("name"),
        "remote_model": entry.get("remote_model"),
        "remote_host": remote_host,
        "execution": "cloud" if remote_host else "local",
        "digest": entry.get("digest"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "context_length": details.get("context_length"),
        "capabilities": entry.get("capabilities"),
        "modified_at": entry.get("modified_at"),
    }


class OllamaProvider(LLMProvider):
    """One schema-prompted chat completion against a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        temperature: float = 0.0,
        seed: int | None = 42,
        num_predict: int = 700,
        think: bool = False,
        send_format_schema: bool = True,
        timeout: float = 180.0,
        max_retries: int = 2,
        probe: bool = True,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = float(temperature)
        self.seed = seed
        self.num_predict = int(num_predict)
        self.think = bool(think)
        self.send_format_schema = bool(send_format_schema)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        # Greedy decoding with a fixed seed is reproducible; anything else is
        # not, and the record must not claim otherwise.
        self.deterministic = self.temperature == 0.0 and seed is not None
        self.model_record: dict = probe_model(model, self.host) if probe else {}
        #: Whether the first non-empty response was a bare JSON object. This
        #: records the shape actually returned; it does NOT isolate whether
        #: ``format`` was enforced, because the v2 prompt also demands bare JSON.
        self.bare_json_observed: bool | None = None

    # ------------------------------------------------------------------ call

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> LLMCall:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.seed is not None:
            payload["options"]["seed"] = int(self.seed)
        if self.send_format_schema:
            payload["format"] = dict(schema)

        url = f"{self.host}/api/chat"
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body = _post(url, payload, self.timeout)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                last_error = f"HTTP {error.code}: {detail}"
            except (urllib.error.URLError, OSError, TimeoutError) as error:
                last_error = f"{type(error).__name__}: {error}"
            except json.JSONDecodeError as error:
                last_error = f"daemon returned non-JSON: {error}"
            else:
                return self._call_from(body, started)
            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))

        return LLMCall(
            text="", model=self.model, usage=TokenUsage(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            provider=self.name, deterministic=self.deterministic,
            error=last_error or "unknown transport failure",
        )

    def _call_from(self, body: Mapping[str, Any], started: float) -> LLMCall:
        message = body.get("message") or {}
        text = str(message.get("content") or "")
        # Prefer the daemon's own measurement; fall back to wall clock.
        total_ns = body.get("total_duration")
        latency_ms = (
            float(total_ns) / _NS_PER_MS if total_ns
            else (time.perf_counter() - started) * 1000.0
        )
        usage = TokenUsage(
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
        )
        done_reason = body.get("done_reason")
        error = None
        if done_reason == "length":
            # A truncated body fails JSON parsing anyway; naming the cause keeps
            # the failure diagnosable rather than filed under "malformed".
            error = f"response truncated at num_predict={self.num_predict}"
        if not text.strip():
            error = error or "the model returned an empty message"

        if self.bare_json_observed is None and text.strip():
            self.bare_json_observed = text.strip().startswith("{")

        return LLMCall(
            text=text,
            # The daemon echoes the resolved remote name; the requested tag is
            # the identity that matters for reproduction, so both are kept.
            model=self.model,
            usage=usage,
            latency_ms=latency_ms,
            stop_reason=done_reason,
            deterministic=self.deterministic,
            provider=self.name,
            error=error,
            request_id=str(body.get("created_at") or "") or None,
        )

    # ------------------------------------------------------------ provenance

    def provenance(self) -> dict:
        return {
            "provider": self.name,
            "host": self.host,
            "model": self.model,
            "model_record": dict(self.model_record),
            "execution": self.model_record.get("execution", "unknown"),
            "remote_host": self.model_record.get("remote_host"),
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "think": self.think,
            "format_schema_sent": self.send_format_schema,
            "first_response_was_bare_json": self.bare_json_observed,
            "deterministic": self.deterministic,
            "determinism_note": (
                "Ollama accepts temperature and seed, so greedy decoding with a fixed "
                "seed is reproducible on the same daemon and model build. This is "
                "stronger than the hosted-API case, which exposes neither."
                if self.deterministic else
                "temperature > 0 or no seed: outputs are not reproducible; the recorded "
                "transcript is the reproducible artefact."
            ),
            "schema_note": (
                "The JSON Schema is sent in Ollama's `format` field, but enforcement is "
                "model-dependent. Probing gemma4:31b-cloud with prompt v1 showed it "
                "ignored: keys renamed, a word returned where a number was required, "
                "required keys omitted, and the object wrapped in a markdown fence. "
                "Prompt v2 therefore carries the authoritative contract. "
                "first_response_was_bare_json records the shape actually returned and "
                "does not by itself prove the schema was enforced."
            ),
            "cost_note": (
                "Ollama publishes no per-token price for cloud-backed models through this "
                "interface, so no monetary cost is reported. Token counts and measured "
                "latency are recorded instead; inventing a price would be fabrication."
            ),
        }
