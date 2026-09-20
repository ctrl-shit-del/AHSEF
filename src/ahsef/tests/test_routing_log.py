"""The routing trace format: structural invariants the stage-2 router must honour.

The stopping condition and candidate selection are *shapes* in stage 1 -- there
is no router yet -- so these tests pin the contract that shape has to satisfy:
a trace terminates, its steps are ordered, a selected modality is actually
active at the next step, and a stopping step does not also select something.
"""

from __future__ import annotations

import json

import pytest

from src.ahsef.routing_log import (
    CandidateScore,
    RoutingLogWriter,
    RoutingStep,
    RoutingTrace,
    read_routing_log,
    summarise_traces,
)


CLASSES = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]


def stopping_step(step=1, active=("audio",), uncertainty=0.2, reason="uncertainty_below_threshold"):
    return RoutingStep(
        step=step, active_modalities=list(active), predicted_class=1,
        predicted_label="happy", confidence=0.9, uncertainty=uncertainty,
        stop=True, stop_reason=reason,
    )


def acquiring_step(step=1, active=("audio",), selected="text"):
    return RoutingStep(
        step=step, active_modalities=list(active), predicted_class=0,
        predicted_label="neutral", confidence=0.3, uncertainty=0.95, stop=False,
        hsig={"text": 0.21, "image": 0.02, "video": 0.01},
        ugapr={
            "text": {"utility": 0.18, "estimated_delta_uncertainty": 0.21,
                     "normalized_cost": 0.2, "normalized_latency": 0.1},
            "image": {"utility": -0.05, "estimated_delta_uncertainty": 0.02,
                      "normalized_cost": 0.5, "normalized_latency": 0.4},
        },
        candidates=[
            CandidateScore("text", True, 0.21, 0.2, 0.1, 0.18),
            CandidateScore("image", True, 0.02, 0.5, 0.4, -0.05),
            CandidateScore("video", False, unavailable_reason="no aligned sample"),
        ],
        selected_modality=selected,
    )


# ------------------------------------------------------------------ steps

def test_a_stopping_step_may_not_also_select_a_modality():
    with pytest.raises(ValueError, match="must not also select"):
        RoutingStep(
            step=1, active_modalities=["audio"], predicted_class=0, predicted_label="neutral",
            confidence=0.9, uncertainty=0.1, stop=True,
            stop_reason="uncertainty_below_threshold", selected_modality="text",
        )


def test_a_continuing_step_must_select_something():
    with pytest.raises(ValueError, match="must select"):
        RoutingStep(
            step=1, active_modalities=["audio"], predicted_class=0, predicted_label="neutral",
            confidence=0.3, uncertainty=0.95, stop=False,
        )


def test_the_selected_modality_must_have_been_scored():
    with pytest.raises(ValueError, match="not among the candidates"):
        RoutingStep(
            step=1, active_modalities=["audio"], predicted_class=0, predicted_label="neutral",
            confidence=0.3, uncertainty=0.9, stop=False,
            candidates=[CandidateScore("image")], selected_modality="text",
        )


def test_unnormalised_uncertainty_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        stopping_step(uncertainty=1.4)


def test_unknown_stop_reason_is_refused():
    with pytest.raises(ValueError, match="stop_reason must be"):
        stopping_step(reason="felt_like_it")


def test_every_declared_stop_reason_is_accepted():
    from src.ahsef.routing_log import STOP_REASONS

    for reason in STOP_REASONS:
        assert stopping_step(reason=reason).stop_reason == reason


# ----------------------------------------------------------------- traces

def test_a_trace_must_terminate():
    with pytest.raises(ValueError, match="never stops"):
        RoutingTrace("s1", "MSP", "test", CLASSES, [acquiring_step()])


def test_a_trace_must_have_at_least_one_step():
    with pytest.raises(ValueError, match="empty routing trace"):
        RoutingTrace("s1", "MSP", "test", CLASSES, [])


def test_steps_must_be_consecutively_numbered():
    with pytest.raises(ValueError, match="out-of-order"):
        RoutingTrace("s1", "MSP", "test", CLASSES, [stopping_step(step=2)])


def test_an_acquired_modality_must_be_active_at_the_next_step():
    with pytest.raises(ValueError, match="does not have it active"):
        RoutingTrace("s1", "MSP", "test", CLASSES, [
            acquiring_step(step=1, selected="text"),
            stopping_step(step=2, active=("audio", "image")),
        ])


def test_a_two_step_trace_records_the_acquisition():
    trace = RoutingTrace("s1", "MSP", "test", CLASSES, [
        acquiring_step(step=1, selected="text"),
        stopping_step(step=2, active=("audio", "text")),
    ], true_class=1)
    assert trace.acquisitions == ["text"]
    assert trace.modalities_activated == 2
    assert trace.uncertainty_reduction == pytest.approx(0.95 - 0.2)


def test_a_single_step_trace_is_a_valid_stop_immediately_decision():
    trace = RoutingTrace("s1", "MSP", "test", CLASSES, [stopping_step()], true_class=1)
    assert trace.acquisitions == []
    assert trace.modalities_activated == 1
    assert trace.uncertainty_reduction == pytest.approx(0.0)


def test_serialised_trace_states_the_router_never_saw_the_label():
    trace = RoutingTrace("s1", "MSP", "test", CLASSES, [stopping_step()], true_class=1)
    record = trace.to_dict()
    assert record["router_saw_true_class"] is False
    assert record["true_label"] == "happy"


def test_rendered_trace_shows_hsig_ugapr_and_the_selection():
    trace = RoutingTrace("s1", "MSP", "test", CLASSES, [
        acquiring_step(step=1, selected="text"),
        stopping_step(step=2, active=("audio", "text")),
    ], true_class=1)
    rendered = trace.render()
    assert "Step 1:" in rendered and "Step 2:" in rendered
    assert "HSIG:" in rendered and "UGAPR:" in rendered
    assert "Selected    = text" in rendered
    assert "Stopping decision = TRUE" in rendered


# ------------------------------------------------------------------- log

def test_log_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "routing.jsonl"
    traces = [
        RoutingTrace("s1", "MSP", "test", CLASSES, [stopping_step()], true_class=1),
        RoutingTrace("s2", "MSP", "test", CLASSES, [
            acquiring_step(step=1), stopping_step(step=2, active=("audio", "text")),
        ], true_class=2),
    ]
    with RoutingLogWriter(path, policy={"uncertainty_threshold": 0.85}) as writer:
        writer.write_all(traces)

    header, loaded = read_routing_log(path)
    assert header["policy"]["uncertainty_threshold"] == 0.85
    assert [record["sample_id"] for record in loaded] == ["s1", "s2"]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["record"] == "header"


def test_summary_gives_the_headline_ahsef_metrics(tmp_path):
    traces = [
        RoutingTrace("s1", "MSP", "test", CLASSES, [stopping_step()], true_class=1).to_dict(),
        RoutingTrace("s2", "MSP", "test", CLASSES, [
            acquiring_step(step=1), stopping_step(step=2, active=("audio", "text")),
        ], true_class=2).to_dict(),
    ]
    summary = summarise_traces(traces)
    assert summary["samples"] == 2
    assert summary["average_modalities_activated"] == pytest.approx(1.5)
    assert summary["modality_activation_rate"]["audio"] == pytest.approx(1.0)
    assert summary["modality_activation_rate"]["text"] == pytest.approx(0.5)
    assert summary["stop_reasons"]["uncertainty_below_threshold"] == 2


def test_summary_refuses_an_empty_log():
    with pytest.raises(ValueError, match="No routing traces"):
        summarise_traces([])
