"""The LLM provider boundary: one call in, one measured result out.

Everything provider-specific lives behind :class:`LLMProvider`.  The rest of
the stage-2 code sees only :class:`LLMCall` -- raw response text plus the
things an experiment has to record: token usage, measured latency, cost at
list price, and whether the call was deterministic.

Two implementations ship:

:class:`~src.ahsef.llm.anthropic_provider.AnthropicProvider`
    The real thing, via the official ``anthropic`` SDK.

:class:`ReplayProvider`
    Replays a recorded JSONL transcript.  This is not a convenience -- it is
    what makes an LLM experiment reproducible at all.  A run against a
    nondeterministic API can be re-analysed exactly, and the whole test suite
    runs with no network and no spend.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class TokenUsage:
    """Tokens billed for one call, as the provider reported them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMCall:
    """One provider call: what came back, and what it cost to get it."""

    text: str
    model: str
    usage: TokenUsage
    latency_ms: float
    #: ``end_turn``, ``max_tokens``, ``refusal`` -- checked before the text is trusted.
    stop_reason: str | None = None
    stop_details: Mapping[str, Any] | None = None
    #: False when the provider gives no seed/temperature control.
    deterministic: bool = False
    provider: str = ""
    error: str | None = None
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.stop_reason != "refusal" and bool(self.text)

    def to_dict(self) -> dict:
        return {
            "model": self.model, "provider": self.provider,
            "usage": self.usage.to_dict(), "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "stop_details": dict(self.stop_details) if self.stop_details else None,
            "deterministic": self.deterministic, "error": self.error,
            "request_id": self.request_id,
        }


#: Anthropic list prices, USD per million tokens, cached 2026-06-24.  Cost is a
#: *price* computed from measured tokens -- not an energy measurement, and the
#: artefacts say so.
PRICING_USD_PER_MTOK: Mapping[str, Mapping[str, float]] = {
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10},
}


def estimate_cost_usd(model: str, usage: TokenUsage) -> float | None:
    """List-price cost of one call, or ``None`` when the model is not in the table."""
    price = PRICING_USD_PER_MTOK.get(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price["input"]
        + usage.output_tokens * price["output"]
        + usage.cache_read_input_tokens * price.get("cache_read", price["input"])
    ) / 1_000_000


def text_key(text: str) -> str:
    """Stable content hash, used to key a replay transcript without storing text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LLMProvider:
    """One text-in, structured-text-out call, with its cost measured."""

    name: str = "abstract"
    model: str = ""
    deterministic: bool = False

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> LLMCall:
        raise NotImplementedError

    def provenance(self) -> dict:
        """Everything needed to describe this provider in an experiment record."""
        return {"provider": self.name, "model": self.model, "deterministic": self.deterministic}

    def close(self) -> None:  # pragma: no cover - providers that hold no resource
        return None


# ============================================================
# Replay
# ============================================================

class ReplayProvider(LLMProvider):
    """Serve recorded responses, keyed by the hash of the rendered user prompt.

    Recording once and replaying thereafter is the only way this project can
    offer a reproducible LLM experiment: the API is nondeterministic and has no
    seed, so "re-run it" is not a reproduction. The transcript is the artefact.
    """

    name = "replay"

    def __init__(
        self,
        transcript: Path | str | None = None,
        records: Mapping[str, list[Mapping[str, Any]]] | None = None,
        model: str = "replay",
        strict: bool = True,
    ):
        self.model = model
        self.deterministic = True
        self.strict = strict
        self.path = Path(transcript) if transcript else None
        self._records: dict[str, list[dict]] = {
            key: [dict(item) for item in value] for key, value in (records or {}).items()
        }
        self._cursor: dict[str, int] = {}
        self.misses: list[str] = []
        #: Line numbers dropped as unreadable; expected only as a final
        #: truncated record from an interrupted run.
        self.corrupt_lines: list[int] = []
        if self.path is not None:
            self._load(self.path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"No LLM transcript at {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        for position, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial final record. It is
                # incomplete data, so it is dropped and counted -- never
                # patched up into a response the model did not give.
                self.corrupt_lines.append(position + 1)
                continue
            if record.get("record") == "header":
                self.model = record.get("model", self.model)
                continue
            self._records.setdefault(record["prompt_key"], []).append(record)
        if self.corrupt_lines and self.corrupt_lines != [len(lines)]:
            raise ValueError(
                f"{path} has unreadable lines at positions {self.corrupt_lines}. Only a "
                f"truncated FINAL line is expected from an interrupted run; damage "
                f"elsewhere means the transcript cannot be trusted."
            )

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> LLMCall:
        key = text_key(user)
        bucket = self._records.get(key)
        if not bucket:
            self.misses.append(key)
            if self.strict:
                raise KeyError(
                    f"No recorded LLM response for prompt {key[:12]}...; the transcript "
                    f"does not cover this sample. Re-record rather than substituting a "
                    f"response."
                )
            return LLMCall(
                text="", model=self.model, usage=TokenUsage(), latency_ms=0.0,
                provider=self.name, deterministic=True, error="no recorded response",
            )
        # Repeated calls on one input walk the recorded repeats in order, then
        # cycle -- so a k-repeat analysis replays the k answers that were seen.
        index = self._cursor.get(key, 0)
        record = bucket[index % len(bucket)]
        self._cursor[key] = index + 1
        usage = record.get("usage") or {}
        return LLMCall(
            text=record.get("text", ""),
            model=record.get("model", self.model),
            usage=TokenUsage(
                int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                int(usage.get("cache_read_input_tokens", 0)),
                int(usage.get("cache_creation_input_tokens", 0)),
            ),
            latency_ms=float(record.get("latency_ms", 0.0)),
            stop_reason=record.get("stop_reason"),
            stop_details=record.get("stop_details"),
            deterministic=True,
            provider=self.name,
            error=record.get("error"),
            request_id=record.get("request_id"),
        )

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "transcript": str(self.path) if self.path else None,
            "recorded_prompts": len(self._records),
            "misses": len(self.misses),
            "corrupt_lines_dropped": list(self.corrupt_lines),
            "note": "Responses are replayed from a recorded transcript; no API call was made.",
        }


class TranscriptWriter:
    """Append-only JSONL record of every LLM call, for exact replay.

    Stores the *hash* of the prompt by default.  Raw text is written only when
    ``store_text`` is on, because a transcript is an artefact that outlives the
    run and the input is user data.
    """

    #: A transient OS lock (an antivirus scanner opening the growing file, an
    #: indexer, a sync client) must not destroy an hour of paid API calls. The
    #: write is retried with backoff before the run is allowed to fail.
    WRITE_ATTEMPTS = 6
    WRITE_BACKOFF_SECONDS = 0.25

    def __init__(
        self,
        path: Path | str,
        model: str,
        provider: str,
        prompt_version: str,
        store_text: bool = False,
        append: bool = False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_text = bool(store_text)
        # Appending onto an existing transcript continues a run; the header
        # already describes it and must not be duplicated.
        resuming = append and self.path.exists() and self.path.stat().st_size > 0
        self.appending = bool(resuming)
        self._handle = self.path.open("a" if resuming else "w", encoding="utf-8")
        if not resuming:
            self._handle.write(json.dumps({
                "record": "header",
                "model": model,
                "provider": provider,
                "prompt_version": prompt_version,
                "stores_raw_text": self.store_text,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }) + "\n")
        self.count = 0
        self.retries = 0

    def write(self, sample_id: str, user: str, call: LLMCall, repeat: int = 0) -> None:
        record = {
            "record": "call",
            "sample_id": sample_id,
            "prompt_key": text_key(user),
            "repeat": repeat,
            "text": call.text,
            "model": call.model,
            "usage": call.usage.to_dict(),
            "latency_ms": call.latency_ms,
            "stop_reason": call.stop_reason,
            "error": call.error,
            "request_id": call.request_id,
        }
        if self.store_text:
            record["user_text"] = user
        self._write_line(json.dumps(record) + "\n")
        self.count += 1

    def _write_line(self, line: str) -> None:
        """Write and flush, retrying a transient lock rather than aborting."""
        last: OSError | None = None
        for attempt in range(self.WRITE_ATTEMPTS):
            try:
                self._handle.write(line)
                # Flushing each record is what makes the transcript a usable
                # resume point after an abrupt failure.
                self._handle.flush()
                return
            except OSError as error:
                last = error
                self.retries += 1
                time.sleep(self.WRITE_BACKOFF_SECONDS * (attempt + 1))
                try:                      # the handle may have been invalidated
                    self._handle.close()
                except OSError:
                    pass
                try:
                    self._handle = self.path.open("a", encoding="utf-8")
                except OSError:
                    continue
        raise TranscriptWriteError(
            f"Could not append to {self.path} after {self.WRITE_ATTEMPTS} attempts "
            f"({last}). The calls already written are intact and the run can be "
            f"resumed from them."
        ) from last

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            # Losing the close on a flushed handle costs nothing; raising here
            # would mask whatever real error triggered the shutdown.
            pass

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ============================================================
# Scripted (tests)
# ============================================================

class ScriptedProvider(LLMProvider):
    """Return a fixed sequence of response bodies.  For tests only."""

    name = "scripted"

    def __init__(self, responses: list[str | Mapping[str, Any]], model: str = "scripted"):
        self.model = model
        self.deterministic = True
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []
        self._index = 0

    def complete(self, system: str, user: str, schema: Mapping[str, Any]) -> LLMCall:
        self.calls.append((system, user))
        if not self._responses:
            raise RuntimeError("ScriptedProvider ran out of responses")
        item = self._responses[self._index % len(self._responses)]
        self._index += 1
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMCall(
            text=text, model=self.model, usage=TokenUsage(100, 50),
            latency_ms=1.0, stop_reason="end_turn", deterministic=True, provider=self.name,
        )


@dataclass
class CallBudget:
    """A hard ceiling on calls and spend, checked before every request.

    An LLM evaluation loop that runs away costs real money, so the limit is
    enforced in code rather than trusted to the operator's arithmetic.
    """

    max_calls: int | None = None
    max_cost_usd: float | None = None
    calls: int = 0
    cost_usd: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    #: Calls whose model has no entry in the price table, so cost is unknown.
    unpriced_calls: int = 0

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(
                f"call budget exhausted after {self.calls} calls (limit {self.max_calls})"
            )
        if self.max_cost_usd is not None:
            if self.unpriced_calls:
                # A ceiling that cannot be computed is worse than no ceiling: the
                # operator believes they are protected while spend runs free.
                raise BudgetExceeded(
                    f"a cost ceiling of ${self.max_cost_usd:.2f} was set, but "
                    f"{self.unpriced_calls} call(s) used a model with no entry in the "
                    f"price table, so spend cannot be enforced. Add the model to "
                    f"PRICING_USD_PER_MTOK or use --max-calls instead."
                )
            if self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"cost budget exhausted at ${self.cost_usd:.4f} "
                    f"(limit ${self.max_cost_usd:.2f})"
                )

    def record(self, call: LLMCall) -> None:
        self.calls += 1
        self.usage = self.usage + call.usage
        cost = estimate_cost_usd(call.model, call.usage)
        if cost is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += cost

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "max_calls": self.max_calls,
            "estimated_cost_usd": round(self.cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "unpriced_calls": self.unpriced_calls,
            "cost_is_complete": self.unpriced_calls == 0,
            "usage": self.usage.to_dict(),
            "cost_basis": "provider list price per million tokens, applied to measured "
                          "token counts; not an energy measurement",
        }


class TranscriptWriteError(RuntimeError):
    """Raised when a transcript record cannot be persisted after retries."""


class BudgetExceeded(RuntimeError):
    """Raised when an LLM run reaches its configured call or cost ceiling."""


def recorded_sample_ids(path: Path | str) -> set[str]:
    """Sample ids a transcript already has at least one recorded call for.

    Used to resume an interrupted run without re-paying for work already done.
    A partial final line is skipped, so an abrupt kill costs at most one call.
    """
    path = Path(path)
    if not path.exists():
        return set()
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("record") == "call" and record.get("sample_id"):
            found.add(str(record["sample_id"]))
    return found


def iter_batched(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
