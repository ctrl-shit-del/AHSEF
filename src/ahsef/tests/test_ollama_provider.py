"""The Ollama provider: request shape, response handling, and provenance.

Every test here stubs the HTTP layer, so the suite needs no daemon and makes no
network call. The one test that touches a real daemon is marked ``integration``.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from src.ahsef.llm import ollama_provider as module
from src.ahsef.llm.ollama_provider import (
    OllamaProvider,
    OllamaUnavailable,
    probe_model,
)
from src.ahsef.llm.schema import parse_response, response_json_schema


CLOUD_TAG = {
    "models": [{
        "name": "gemma4:31b-cloud", "model": "gemma4:31b-cloud",
        "remote_model": "gemma4:31b", "remote_host": "https://ollama.com",
        "digest": "ef09f235", "details": {
            "parameter_size": "32.7B", "quantization_level": "BF16",
            "context_length": 262144,
        },
        "capabilities": ["completion", "thinking", "tools", "vision"],
    }]
}

LOCAL_TAG = {
    "models": [{
        "name": "llama3:8b", "model": "llama3:8b", "digest": "abc",
        "details": {"parameter_size": "8B"}, "capabilities": ["completion"],
    }]
}


def body(content: str = '{"emotion": "happy"}', **overrides) -> dict:
    record = {
        "model": "gemma4:31b",
        "created_at": "2026-08-29T12:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True, "done_reason": "stop",
        "total_duration": 1_200_000_000,   # 1200 ms in nanoseconds
        "prompt_eval_count": 400, "eval_count": 150,
    }
    record.update(overrides)
    return record


@pytest.fixture
def stub(monkeypatch):
    """Capture the outgoing payload and serve a canned response."""
    state = {"payload": None, "response": body(), "raise": None}

    def fake_get(url, timeout):
        return CLOUD_TAG

    def fake_post(url, payload, timeout):
        state["payload"] = payload
        state["url"] = url
        if state["raise"] is not None:
            raise state["raise"]
        return state["response"]

    monkeypatch.setattr(module, "_get", fake_get)
    monkeypatch.setattr(module, "_post", fake_post)
    return state


# ------------------------------------------------------------------- probe

def test_probe_reports_cloud_execution(monkeypatch):
    monkeypatch.setattr(module, "_get", lambda url, timeout: CLOUD_TAG)
    record = probe_model("gemma4:31b-cloud")
    assert record["execution"] == "cloud"
    assert record["remote_host"] == "https://ollama.com"
    assert record["parameter_size"] == "32.7B"


def test_probe_reports_local_execution(monkeypatch):
    monkeypatch.setattr(module, "_get", lambda url, timeout: LOCAL_TAG)
    assert probe_model("llama3:8b")["execution"] == "local"


def test_a_missing_model_is_refused_rather_than_substituted(monkeypatch):
    monkeypatch.setattr(module, "_get", lambda url, timeout: LOCAL_TAG)
    with pytest.raises(OllamaUnavailable, match="will not silently substitute"):
        probe_model("gemma4:31b-cloud")


def test_an_unreachable_daemon_says_how_to_proceed(monkeypatch):
    def boom(url, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(module, "_get", boom)
    with pytest.raises(OllamaUnavailable, match="--provider replay"):
        probe_model("gemma4:31b-cloud")


# ------------------------------------------------------------ request shape

def test_the_request_carries_the_exact_model_and_generation_config(stub):
    provider = OllamaProvider("gemma4:31b-cloud", temperature=0.0, seed=42, num_predict=700)
    provider.complete("sys", "user", response_json_schema())
    payload = stub["payload"]
    assert payload["model"] == "gemma4:31b-cloud"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 42
    assert payload["options"]["num_predict"] == 700
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"] == "user"


def test_the_json_schema_is_sent_in_the_format_field(stub):
    provider = OllamaProvider("gemma4:31b-cloud")
    provider.complete("sys", "user", response_json_schema())
    assert stub["payload"]["format"]["properties"]["emotion"]["enum"]


def test_the_schema_can_be_withheld(stub):
    provider = OllamaProvider("gemma4:31b-cloud", send_format_schema=False)
    provider.complete("sys", "user", response_json_schema())
    assert "format" not in stub["payload"]


def test_no_seed_is_sent_when_none(stub):
    provider = OllamaProvider("gemma4:31b-cloud", seed=None)
    provider.complete("sys", "user", {})
    assert "seed" not in stub["payload"]["options"]


# ----------------------------------------------------------------- response

def test_usage_and_latency_come_from_the_daemon(stub):
    call = OllamaProvider("gemma4:31b-cloud").complete("s", "u", {})
    assert call.usage.input_tokens == 400
    assert call.usage.output_tokens == 150
    assert call.latency_ms == pytest.approx(1200.0)   # 1.2e9 ns
    assert call.ok


def test_the_requested_tag_is_recorded_not_the_resolved_remote_name(stub):
    """`gemma4:31b-cloud` is the identity needed to reproduce the run."""
    call = OllamaProvider("gemma4:31b-cloud").complete("s", "u", {})
    assert call.model == "gemma4:31b-cloud"


def test_truncation_is_named_rather_than_left_as_malformed_json(stub):
    stub["response"] = body(content='{"emotion": "hap', done_reason="length")
    call = OllamaProvider("gemma4:31b-cloud", num_predict=700).complete("s", "u", {})
    assert "truncated at num_predict=700" in call.error


def test_an_empty_message_is_an_error(stub):
    stub["response"] = body(content="   ")
    assert "empty message" in OllamaProvider("gemma4:31b-cloud").complete("s", "u", {}).error


def test_transport_failures_are_retried_then_reported(stub, monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    stub["raise"] = urllib.error.URLError("boom")
    call = OllamaProvider("gemma4:31b-cloud", max_retries=2).complete("s", "u", {})
    assert not call.ok
    assert "URLError" in call.error
    assert call.usage.input_tokens == 0


def test_an_http_error_is_captured_with_its_body(stub, monkeypatch):
    class Fake(urllib.error.HTTPError):
        def __init__(self):
            super().__init__("u", 500, "err", {}, None)

        def read(self):
            return b"internal failure"

    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    stub["raise"] = Fake()
    call = OllamaProvider("gemma4:31b-cloud", max_retries=0).complete("s", "u", {})
    assert "HTTP 500" in call.error and "internal failure" in call.error


# -------------------------------------------------- structured-output path

def test_a_fenced_response_still_parses(stub):
    """Observed behaviour: the model wraps JSON in a markdown fence."""
    from src.ahsef.llm.mapping import CANONICAL_EMOTIONS

    payload = {
        "emotion": "happy",
        "class_scores": {name: 0.1 for name in CANONICAL_EMOTIONS},
        "confidence": 0.9, "evidence_strength": 0.8, "ambiguity": 0.1,
        "evidence": "glad", "abstain": False,
    }
    stub["response"] = body(content="```json\n" + json.dumps(payload) + "\n```")
    call = OllamaProvider("gemma4:31b-cloud").complete("s", "u", {})
    response = parse_response(call.text)
    assert response.usable and response.emotion == "happy"


def test_a_response_that_ignores_the_schema_is_rejected_not_repaired(stub):
    """The v1 probe returned `label` instead of `emotion`; that must not pass."""
    stub["response"] = body(content=json.dumps({"label": "happy", "evidence_strength": "high"}))
    call = OllamaProvider("gemma4:31b-cloud").complete("s", "u", {})
    response = parse_response(call.text)
    assert not response.usable
    assert response.error_kind == "missing_field"


def test_a_synonym_is_mapped_and_a_contested_word_is_not(stub):
    from src.ahsef.llm.mapping import CANONICAL_EMOTIONS

    base = {
        "class_scores": {name: 0.1 for name in CANONICAL_EMOTIONS},
        "confidence": 0.9, "evidence_strength": 0.8, "ambiguity": 0.1,
        "evidence": "c", "abstain": False,
    }
    stub["response"] = body(content=json.dumps({**base, "emotion": "joy"}))
    assert parse_response(
        OllamaProvider("gemma4:31b-cloud").complete("s", "u", {}).text
    ).emotion == "happy"

    stub["response"] = body(content=json.dumps({**base, "emotion": "excited"}))
    rejected = parse_response(OllamaProvider("gemma4:31b-cloud").complete("s", "u", {}).text)
    assert rejected.error_kind == "ambiguous"
    assert rejected.emotion is None


# --------------------------------------------------------------- provenance

def test_greedy_decoding_with_a_seed_is_reported_deterministic(stub):
    provider = OllamaProvider("gemma4:31b-cloud", temperature=0.0, seed=42)
    assert provider.deterministic is True
    assert "reproducible" in provider.provenance()["determinism_note"]


def test_sampling_without_a_seed_is_not_reported_deterministic(stub):
    provider = OllamaProvider("gemma4:31b-cloud", temperature=0.7, seed=None)
    assert provider.deterministic is False
    assert "not reproducible" in provider.provenance()["determinism_note"]


def test_provenance_records_cloud_execution_and_refuses_to_invent_a_price(stub):
    record = OllamaProvider("gemma4:31b-cloud").provenance()
    assert record["execution"] == "cloud"
    assert record["remote_host"] == "https://ollama.com"
    assert "no per-token price" in record["cost_note"]
    assert "inventing a price would be fabrication" in record["cost_note"]


def test_provenance_never_carries_a_credential(stub):
    record = json.dumps(OllamaProvider("gemma4:31b-cloud").provenance()).lower()
    for secret in ("api_key", "authorization", "bearer", "token="):
        assert secret not in record


def test_the_bare_json_flag_does_not_claim_schema_enforcement(stub):
    provider = OllamaProvider("gemma4:31b-cloud")
    provider.complete("s", "u", response_json_schema())
    assert provider.bare_json_observed is True
    assert "does not by itself prove" in provider.provenance()["schema_note"]


# -------------------------------------------------------------- integration

@pytest.mark.integration
def test_the_real_daemon_serves_the_pilot_model():
    """Skipped unless a daemon is up with the exact pilot model pulled."""
    try:
        record = probe_model("gemma4:31b-cloud")
    except OllamaUnavailable as error:
        pytest.skip(str(error))
    assert record["name"] == "gemma4:31b-cloud"
    assert record["execution"] in {"cloud", "local"}
