"""Providers, budgets, transcripts, and the end-to-end inference path."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ahsef.llm.inference import (
    LLM_MODALITY,
    LLMTextModality,
    build_prediction_set,
    coverage_report,
    routing_uncertainty_column,
)
from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm.provider import (
    BudgetExceeded,
    CallBudget,
    LLMCall,
    ReplayProvider,
    ScriptedProvider,
    TokenUsage,
    TranscriptWriter,
    estimate_cost_usd,
    text_key,
)


_KEEP = object()


def body(emotion="happy", confidence=0.8, scores=_KEEP, **extra) -> dict:
    """``scores=None`` means the model returned none; the default means it did."""
    record = {
        "emotion": emotion,
        "class_scores": (
            {name: 0.1 for name in CANONICAL_EMOTIONS} if scores is _KEEP else scores
        ),
        "confidence": confidence, "evidence_strength": 0.6, "ambiguity": 0.3,
        "evidence": "cue", "abstain": False,
    }
    record.update(extra)
    return record


def frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "dataset": ["MELD"] * n,
        "text": [f"utterance {i}" for i in range(n)],
        "canonical_emotion_id": [1] * n,
    })


# ------------------------------------------------------------------- usage

def test_token_usage_adds():
    total = TokenUsage(10, 5, 2, 1) + TokenUsage(20, 10, 3, 0)
    assert total.input_tokens == 30 and total.output_tokens == 15
    assert total.cache_read_input_tokens == 5


def test_cost_uses_the_published_price_table():
    cost = estimate_cost_usd("claude-opus-5", TokenUsage(1_000_000, 1_000_000))
    assert cost == pytest.approx(30.0)  # $5 in + $25 out


def test_cost_of_an_unknown_model_is_none_not_zero():
    assert estimate_cost_usd("some-other-model", TokenUsage(1000, 1000)) is None


# ------------------------------------------------------------------ budget

def test_budget_blocks_further_calls_once_exhausted():
    budget = CallBudget(max_calls=2)
    modality = LLMTextModality(ScriptedProvider([body()]), budget=budget)
    with pytest.raises(BudgetExceeded, match="call budget"):
        modality.predict_frame(frame(3))
    assert budget.calls == 2


def test_cost_budget_is_enforced():
    budget = CallBudget(max_cost_usd=1e-9)
    provider = ScriptedProvider([body()], model="claude-opus-5")
    modality = LLMTextModality(provider, budget=budget)
    with pytest.raises(BudgetExceeded, match="cost budget"):
        modality.predict_frame(frame(3))


def test_a_cost_ceiling_on_an_unpriced_model_fails_loudly():
    """A ceiling that cannot be computed must not read as protection."""
    budget = CallBudget(max_cost_usd=10.0)
    modality = LLMTextModality(ScriptedProvider([body()]), budget=budget)
    with pytest.raises(BudgetExceeded, match="no entry in the price table"):
        modality.predict_frame(frame(3))
    assert budget.to_dict()["cost_is_complete"] is False


def test_budget_record_labels_the_cost_basis():
    record = CallBudget().to_dict()
    assert "list price" in record["cost_basis"]
    assert "not an energy measurement" in record["cost_basis"]


# --------------------------------------------------------------- transcript

def test_transcript_round_trips_through_replay(tmp_path):
    path = tmp_path / "t.jsonl"
    provider = ScriptedProvider([body("happy"), body("sad")])
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        modality = LLMTextModality(provider, transcript=writer)
        first = modality.predict_frame(frame(2))

    replay = ReplayProvider(path)
    second = LLMTextModality(replay).predict_frame(frame(2))
    assert [r.response.emotion for r in first] == [r.response.emotion for r in second]


def test_transcript_stores_only_a_hash_by_default(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        modality = LLMTextModality(ScriptedProvider([body()]), transcript=writer)
        modality.predict_frame(frame(1))
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["stores_raw_text"] is False
    assert "user_text" not in lines[1]
    assert len(lines[1]["prompt_key"]) == 64


def test_transcript_stores_text_when_explicitly_asked(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1", store_text=True) as writer:
        modality = LLMTextModality(ScriptedProvider([body()]), transcript=writer)
        modality.predict_frame(frame(1))
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["stores_raw_text"] is True
    assert "utterance 0" in lines[1]["user_text"]


def test_replay_refuses_an_uncovered_prompt_rather_than_improvising(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(1))
    modality = LLMTextModality(ReplayProvider(path, strict=True))
    with pytest.raises(KeyError, match="No recorded LLM response"):
        modality.predict_frame(frame(3))


def test_replay_walks_repeats_in_recorded_order(tmp_path):
    path = tmp_path / "t.jsonl"
    provider = ScriptedProvider([body("happy"), body("sad"), body("angry")])
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(provider, repeats=3, transcript=writer).predict_frame(frame(1))
    result = LLMTextModality(ReplayProvider(path), repeats=3).predict("utterance 0", "s0")
    assert result.votes == ["happy", "sad", "angry"]


def test_replay_provenance_states_no_api_call_was_made(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(1))
    record = ReplayProvider(path).provenance()
    assert record["deterministic"] is True
    assert "no API call" in record["note"]


def test_text_key_is_stable():
    assert text_key("abc") == text_key("abc") != text_key("abd")


# ------------------------------------------------------------- call errors

def test_a_failed_call_becomes_an_unusable_row_not_a_crash():
    class Failing(ScriptedProvider):
        def complete(self, system, user, schema):
            return LLMCall(text="", model="m", usage=TokenUsage(), latency_ms=1.0,
                           provider="failing", error="boom")

    modality = LLMTextModality(Failing([body()]))
    results = modality.predict_frame(frame(2))
    assert all(result.status == "call_failed" for result in results)
    assert not any(result.response.usable for result in results)


def test_a_refusal_is_treated_as_a_failed_call():
    class Refusing(ScriptedProvider):
        def complete(self, system, user, schema):
            return LLMCall(text="", model="m", usage=TokenUsage(), latency_ms=1.0,
                           provider="refusing", stop_reason="refusal")

    result = LLMTextModality(Refusing([body()])).predict("x", "s0")
    assert result.status == "call_failed"


def test_empty_text_is_refused_before_a_call_is_made():
    modality = LLMTextModality(ScriptedProvider([body()]))
    bad = frame(1)
    bad.loc[0, "text"] = "   "
    with pytest.raises(ValueError, match="empty text"):
        modality.predict_frame(bad)


# ------------------------------------------------------------- repeats

def test_repeats_produce_a_majority_label_and_vote_entropy():
    provider = ScriptedProvider([body("happy"), body("happy"), body("sad")])
    result = LLMTextModality(provider, repeats=3).predict("x", "s0")
    assert result.response.emotion == "happy"
    assert result.uncertainty.repeats == 3
    assert result.uncertainty.llm_vote_agreement == pytest.approx(2 / 3)
    assert len(result.calls) == 3


def test_repeats_below_one_are_refused():
    with pytest.raises(ValueError, match="repeats must be at least 1"):
        LLMTextModality(ScriptedProvider([body()]), repeats=0)


# ----------------------------------------------------- prediction assembly

def test_prediction_set_matches_the_stage_one_contract():
    modality = LLMTextModality(ScriptedProvider([body()]))
    results = modality.predict_frame(frame(3))
    predictions = build_prediction_set(results, "validation", modality.provenance())
    assert predictions.modality == LLM_MODALITY
    assert predictions.class_order == tuple(CANONICAL_EMOTIONS)
    assert predictions.num_classes == 7
    for column in ("sample_id", "true_class", "predicted_class", "latency_ms"):
        assert column in predictions.frame.columns


def test_rows_without_scores_hold_nan_not_a_fabricated_distribution():
    modality = LLMTextModality(ScriptedProvider([body(scores=None)]))
    predictions = build_prediction_set(
        modality.predict_frame(frame(2)), "validation", modality.provenance()
    )
    assert predictions.frame["prob_0"].isna().all()
    assert predictions.meta["distribution_source"] == "none"


def test_meta_denies_that_prob_columns_are_posteriors():
    modality = LLMTextModality(ScriptedProvider([body()]))
    predictions = build_prediction_set(
        modality.predict_frame(frame(1)), "validation", modality.provenance()
    )
    assert predictions.meta["distribution_is_calibrated_posterior"] is False
    assert "SELF-REPORTED" in predictions.meta["prob_columns_note"]


def test_unusable_rows_are_kept_and_counted_not_dropped():
    provider = ScriptedProvider([body("happy"), body("excited"), "{bad"])
    modality = LLMTextModality(provider)
    predictions = build_prediction_set(
        modality.predict_frame(frame(3)), "validation", modality.provenance()
    )
    assert len(predictions.frame) == 3
    report = coverage_report(predictions)
    assert report["usable"] == 1
    assert report["coverage"] == pytest.approx(1 / 3)
    assert set(report["status_counts"]) == {"ok", "ambiguous", "malformed_json"}
    assert "excited" in report["unmappable_examples"]


def test_text_is_hashed_by_default_and_absent_when_asked():
    modality = LLMTextModality(ScriptedProvider([body()]), store_text="hash")
    data = frame(1)
    texts = dict(zip(data["sample_id"], data["text"]))
    hashed = build_prediction_set(
        modality.predict_frame(data), "validation", modality.provenance(),
        texts=texts, store_text="hash",
    )
    assert "text_sha256_16" in hashed.frame.columns
    assert "text" not in hashed.frame.columns

    none = build_prediction_set(
        LLMTextModality(ScriptedProvider([body()])).predict_frame(data),
        "validation", modality.provenance(), texts=texts, store_text="none",
    )
    assert "text_sha256_16" not in none.frame.columns
    assert "text" not in none.frame.columns


def test_routing_uncertainty_column_follows_the_named_policy():
    modality = LLMTextModality(ScriptedProvider([body(confidence=0.75)]))
    predictions = build_prediction_set(
        modality.predict_frame(frame(2)), "validation", modality.provenance()
    )
    assert routing_uncertainty_column(predictions, "llm_confidence").iloc[0] == \
        pytest.approx(0.25)
    assert routing_uncertainty_column(predictions, "score_entropy").iloc[0] == \
        pytest.approx(1.0)


def test_provenance_carries_prompt_schema_and_mapping():
    record = LLMTextModality(ScriptedProvider([body()])).provenance()
    assert record["prompt"]["version"] == "v1"
    assert record["schema"]["schema_version"].startswith("ahsef.llm.response")
    assert record["mapping"]["canonical_classes"] == list(CANONICAL_EMOTIONS)
    assert record["uncertainty_definition"]["calibrated"] is False


# ------------------------------------------- score_top1 recovery from scores

def test_score_top1_is_recomputed_from_the_stored_scores_when_the_column_is_absent():
    """Exports written before the column existed stay readable, exactly."""
    from src.ahsef.llm.uncertainty import UncertaintyPolicyError

    modality = LLMTextModality(ScriptedProvider([
        body(scores={n: (0.7 if n == "happy" else 0.05) for n in CANONICAL_EMOTIONS})
    ]))
    predictions = build_prediction_set(
        modality.predict_frame(frame(2)), "validation", modality.provenance()
    )
    expected = routing_uncertainty_column(predictions, "score_top1").tolist()

    predictions.frame = predictions.frame.drop(columns=["llm_score_top1_uncertainty"])
    recovered = routing_uncertainty_column(predictions, "score_top1").tolist()
    assert recovered == pytest.approx(expected)
    assert recovered[0] == pytest.approx(1 - 0.7 / (0.7 + 6 * 0.05))


def test_score_top1_without_scores_or_column_is_refused():
    from src.ahsef.llm.uncertainty import UncertaintyPolicyError

    modality = LLMTextModality(ScriptedProvider([body(scores=None)]))
    predictions = build_prediction_set(
        modality.predict_frame(frame(1)), "validation", modality.provenance()
    )
    predictions.frame = predictions.frame.drop(
        columns=["llm_score_top1_uncertainty"] + [f"prob_{i}" for i in range(7)]
    )
    with pytest.raises(UncertaintyPolicyError):
        routing_uncertainty_column(predictions, "score_top1")


# --------------------------------------------------- resume after a failure

def test_recorded_sample_ids_lists_what_a_transcript_covers(tmp_path):
    from src.ahsef.llm.provider import recorded_sample_ids

    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(3))
    assert recorded_sample_ids(path) == {"s0", "s1", "s2"}
    assert recorded_sample_ids(tmp_path / "absent.jsonl") == set()


def test_appending_continues_a_transcript_without_a_second_header(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(2))
    with TranscriptWriter(path, "m", "scripted", "v1", append=True) as writer:
        assert writer.appending is True
        LLMTextModality(
            ScriptedProvider([body("sad")]), transcript=writer
        ).predict_frame(frame(3).iloc[2:])

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(1 for r in records if r.get("record") == "header") == 1
    assert {r["sample_id"] for r in records if r.get("record") == "call"} == {"s0", "s1", "s2"}


def test_append_on_a_missing_file_writes_a_header(tmp_path):
    path = tmp_path / "fresh.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1", append=True) as writer:
        assert writer.appending is False
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["record"] == "header"


def test_a_resumed_transcript_replays_the_whole_frame(tmp_path):
    """The end state must not depend on how many invocations produced it."""
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(
            ScriptedProvider([body("happy")]), transcript=writer
        ).predict_frame(frame(3).iloc[:2])
    with TranscriptWriter(path, "m", "scripted", "v1", append=True) as writer:
        LLMTextModality(
            ScriptedProvider([body("sad")]), transcript=writer
        ).predict_frame(frame(3).iloc[2:])

    results = LLMTextModality(ReplayProvider(path)).predict_frame(frame(3))
    assert [r.response.emotion for r in results] == ["happy", "happy", "sad"]


def test_a_transient_write_lock_is_retried_not_fatal(tmp_path, monkeypatch):
    """An antivirus lock must not destroy an hour of paid API calls."""
    path = tmp_path / "t.jsonl"
    writer = TranscriptWriter(path, "m", "scripted", "v1")
    monkeypatch.setattr(TranscriptWriter, "WRITE_BACKOFF_SECONDS", 0.0)

    real_write = writer._handle.write
    state = {"fails": 2}

    def flaky(text):
        if state["fails"] > 0:
            state["fails"] -= 1
            raise PermissionError(13, "Permission denied")
        return real_write(text)

    monkeypatch.setattr(writer._handle, "write", flaky)
    writer.write("s0", "prompt", LLMCall("{}", "m", TokenUsage(), 1.0, provider="x"))
    writer.close()
    assert writer.retries >= 1
    assert writer.count == 1


def test_a_persistent_write_failure_is_raised_with_a_recovery_hint(tmp_path, monkeypatch):
    from src.ahsef.llm.provider import TranscriptWriteError

    path = tmp_path / "t.jsonl"
    writer = TranscriptWriter(path, "m", "scripted", "v1")
    monkeypatch.setattr(TranscriptWriter, "WRITE_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(
        writer, "_handle",
        type("Dead", (), {
            "write": lambda self, text: (_ for _ in ()).throw(PermissionError(13, "denied")),
            "flush": lambda self: None, "close": lambda self: None,
        })(),
    )
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: writer._handle)
    with pytest.raises(TranscriptWriteError, match="can be resumed"):
        writer.write("s0", "p", LLMCall("{}", "m", TokenUsage(), 1.0, provider="x"))


def test_a_truncated_final_line_is_dropped_and_counted(tmp_path):
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(2))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"record": "call", "sample_id": "s2", "prompt_k')

    replay = ReplayProvider(path)
    assert replay.corrupt_lines == [4]
    assert len(replay._records) == 2
    assert replay.provenance()["corrupt_lines_dropped"] == [4]


def test_damage_away_from_the_end_is_refused(tmp_path):
    """A partial final record is expected; anything else means real corruption."""
    path = tmp_path / "t.jsonl"
    with TranscriptWriter(path, "m", "scripted", "v1") as writer:
        LLMTextModality(ScriptedProvider([body()]), transcript=writer).predict_frame(frame(3))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = '{"broken'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be trusted"):
        ReplayProvider(path)
