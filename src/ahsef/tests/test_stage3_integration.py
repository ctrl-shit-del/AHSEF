"""End-to-end: fuse, fit, price, threshold, route, score -- on one synthetic pool.

The unit tests check each component in isolation; this checks that the chain
holds together and that the properties the brief demands survive composition:
no label reaches the router, the policy is not a constant, the artefacts round
trip, and a second identical run produces identical decisions.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.ahsef.costs import CostModel, ModalityCost
from src.ahsef.stage3.experiment import (
    always_acquire,
    build_inputs,
    compare_systems,
    never_acquire,
    random_policy_reference,
)
from src.ahsef.stage3.features import build_features, states_from_features
from src.ahsef.stage3.hsig_model import Stage3HSIG, out_of_fold_gain, targets_from_oracle
from src.ahsef.stage3.objective import select_gate_threshold
from src.ahsef.stage3.oracle import build_oracle_table, choose_fusion_spec, fuse
from src.ahsef.stage3.router import (
    AudioAvailability,
    TextAudioRouter,
    attach_truth,
    decision_summary,
    decisions_frame,
)
from src.ahsef.routing_log import RoutingLogWriter, read_routing_log, summarise_traces
from src.ahsef.tests.stage3_fixtures import scenario
from src.ahsef.ugapr import UGAPR, UtilityWeights
from src.common.labels import CANONICAL_EMOTION_CLASSES

TEXT_MS, AUDIO_MS = 4000.0, 12.0


def cost_model() -> CostModel:
    return CostModel({
        "audio": ModalityCost("audio", AUDIO_MS, 1e8, 100_000, 1000, "synthetic", 10),
        "text_llm": ModalityCost("text_llm", TEXT_MS, 2e13, 32_700_000_000, 700,
                                 "synthetic", 10),
    })


def pipeline(seed: int = 31, n: int = 200):
    """The whole Stage 3 chain, validation-only, in the order the CLI runs it."""
    data = scenario(n=n, seed=seed)
    ids = data["ids"]

    # Phase C -- fusion weight on validation, then the oracle table.
    spec = choose_fusion_spec(data["text"], data["audio"], ids, "validation")
    fused = fuse(data["text"], data["audio"], spec, ids)
    oracle = build_oracle_table(data["text"], data["audio"], fused, ids)

    # Phase D -- features and a validation-fitted estimator.
    uncertainty = data["text"].frame["llm_normalized_score_entropy"].to_numpy()
    features = build_features(data["text"].frame, uncertainty)
    targets = targets_from_oracle(oracle, "validation")
    model = Stage3HSIG.fit(features, targets)

    # Phases E/F -- price the candidate, then choose tau on OUT-OF-FOLD gain.
    costs = cost_model()
    penalty = 0.1 * costs.normalized_cost("audio") + 0.1 * costs.normalized_latency("audio")
    oof = out_of_fold_gain(features, targets, folds=4)
    selection = select_gate_threshold(
        oof - penalty,
        oracle["text_prediction"].to_numpy(dtype=int),
        oracle["fused_prediction"].to_numpy(dtype=int),
        oracle["true_class"].to_numpy(dtype=int),
        split="validation", class_names=CANONICAL_EMOTION_CLASSES,
        text_latency_ms=TEXT_MS, audio_latency_ms=AUDIO_MS,
    )

    # Phase G -- route.
    router = TextAudioRouter(
        hsig=model,
        ugapr=UGAPR(
            UtilityWeights(lambda_cost=0.1, mu_latency=0.1, min_gain=-1.0,
                           min_utility=-2.0),
            cost_model=costs,
        ),
        threshold=selection.threshold,
        availability=AudioAvailability(data["audio"]),
        audio=data["audio"], fused=fused, cost_model=costs,
        text_latency_ms=TEXT_MS, audio_latency_ms=AUDIO_MS, split="validation",
    )
    decisions, traces = router.route_all(states_from_features(features))
    return {
        "data": data, "ids": ids, "spec": spec, "fused": fused, "oracle": oracle,
        "features": features, "targets": targets, "model": model,
        "selection": selection, "router": router, "decisions": decisions,
        "traces": traces, "uncertainty": uncertainty, "costs": costs,
    }


@pytest.fixture(scope="module")
def run():
    return pipeline()


# ------------------------------------------------------------------ the chain

def test_the_pipeline_completes(run):
    assert len(run["decisions"]) == len(run["ids"])
    assert len(run["traces"]) == len(run["ids"])


def test_the_policy_is_not_a_constant(run):
    summary = decision_summary(run["decisions"])
    assert 0 < summary["request_audio"] < summary["samples"], (
        "a policy that always or never acquires is not a routing policy"
    )


def test_every_decision_is_auditable(run):
    for decision in run["decisions"]:
        record = decision.to_dict()
        assert record["sample_id"]
        assert record["decision"] in ("STOP", "REQUEST_AUDIO")
        assert record["availability"] in (
            "AVAILABLE", "REQUESTED_BUT_UNAVAILABLE", "NOT_REQUESTED"
        )
        assert record["reason"]
        assert record["timestamp"]
        assert record["router_saw_true_class"] is False
        assert record["provenance"]["hsig"]


def test_no_label_reached_the_routing_path(run):
    """Decisions are made before truth exists anywhere in the record."""
    assert all(decision.true_class is None for decision in run["decisions"])
    assert "true_class" not in run["features"].columns
    for state in states_from_features(run["features"])[:5]:
        assert "true_class" not in state.features


def test_truth_can_be_attached_afterwards_for_evaluation(run):
    decisions = [d for d in run["decisions"]]
    labels = dict(zip(
        run["oracle"]["sample_id"].astype(str), run["oracle"]["true_class"].astype(int)
    ))
    attach_truth(decisions, labels)
    assert all(decision.final_correctness is not None for decision in decisions)


def test_final_prediction_is_text_when_stopped_and_fused_when_acquired(run):
    frame = decisions_frame(run["decisions"])
    stopped = frame[frame["decision"] == "STOP"]
    acquired = frame[frame["decision"] == "REQUEST_AUDIO"]
    assert (stopped["final_prediction"] == stopped["initial_prediction"]).all()
    assert (acquired["final_prediction"] == acquired["fused_prediction"]).all()


def test_deterministic_replay_of_the_whole_pipeline():
    left, right = pipeline(seed=31), pipeline(seed=31)
    assert left["model"].fingerprint() == right["model"].fingerprint()
    assert left["selection"].threshold == pytest.approx(right["selection"].threshold)
    for a, b in zip(left["decisions"], right["decisions"]):
        assert a.sample_id == b.sample_id
        assert a.decision == b.decision
        assert a.final_prediction == b.final_prediction


# -------------------------------------------------------------------- traces

def test_traces_round_trip_through_the_stage1_log(run, tmp_path):
    path = tmp_path / "traces.jsonl"
    with RoutingLogWriter(path, policy=run["router"].provenance()) as writer:
        writer.write_all(run["traces"])
    header, records = read_routing_log(path)
    assert header["policy"]["router_saw_true_class"] is False
    assert len(records) == len(run["traces"])
    summary = summarise_traces(records)
    assert 1.0 <= summary["average_modalities_activated"] <= 2.0
    assert set(summary["stop_reasons"]) <= {
        "no_candidate_above_min_utility", "no_candidate_available",
        "candidate_pool_exhausted", "requested_modality_unavailable",
    }


def test_a_two_step_trace_renders_hsig_and_ugapr(run):
    multi = [trace for trace in run["traces"] if len(trace.steps) == 2]
    assert multi, "no sample acquired audio"
    rendered = multi[0].render()
    assert "HSIG:" in rendered and "UGAPR:" in rendered
    assert "Selected    = audio" in rendered
    assert "Step 2:" in rendered


# ----------------------------------------------------------------- reporting

def test_system_comparison_covers_every_required_metric(run):
    inputs = build_inputs(
        run["ids"], run["data"]["text"], run["data"]["audio"], run["fused"],
        run["uncertainty"], TEXT_MS, AUDIO_MS,
    )
    frame = decisions_frame(run["decisions"])
    report = compare_systems(inputs, {
        "text_only": never_acquire(len(run["ids"])),
        "audio_only": always_acquire(len(run["ids"])),
        "always_fusion": always_acquire(len(run["ids"])),
        "ahsef_dynamic": (frame["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool),
    })
    for name, record in report["systems"].items():
        for metric in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            assert metric in record, (name, metric)
        assert len(record["per_class_f1"]) == 7
        assert record["latency"]["mean_latency_ms_per_sample"] > 0
        assert "audio_acquisition_rate" in record["acquisition"]
        assert "relative_to_always_fusion" in record
    assert report["central_result"]["performance_vs_modalities_activated"]
    assert report["central_result"]["performance_vs_latency"]


def test_text_only_and_always_fusion_bracket_the_acquisition_axis(run):
    inputs = build_inputs(
        run["ids"], run["data"]["text"], run["data"]["audio"], run["fused"],
        run["uncertainty"], TEXT_MS, AUDIO_MS,
    )
    frame = decisions_frame(run["decisions"])
    report = compare_systems(inputs, {
        "text_only": never_acquire(len(run["ids"])),
        "always_fusion": always_acquire(len(run["ids"])),
        "ahsef_dynamic": (frame["decision"] == "REQUEST_AUDIO").to_numpy(dtype=bool),
    })
    rate = report["systems"]["ahsef_dynamic"]["acquisition"]["audio_acquisition_rate"]
    assert 0.0 < rate < 1.0
    latencies = {
        name: record["latency"]["mean_latency_ms_per_sample"]
        for name, record in report["systems"].items()
    }
    assert latencies["text_only"] <= latencies["ahsef_dynamic"] <= latencies["always_fusion"]


def test_random_control_is_computed_at_the_same_rate(run):
    inputs = build_inputs(
        run["ids"], run["data"]["text"], run["data"]["audio"], run["fused"],
        run["uncertainty"], TEXT_MS, AUDIO_MS,
    )
    control = random_policy_reference(inputs, 0.4, repeats=30)
    assert control["acquisition_rate"] == 0.4
    assert control["accuracy"]["p2.5"] <= control["accuracy"]["mean"] <= \
        control["accuracy"]["p97.5"]


def test_audio_only_ignores_the_text_prediction(run):
    inputs = build_inputs(
        run["ids"], run["data"]["text"], run["data"]["audio"], run["fused"],
        run["uncertainty"], TEXT_MS, AUDIO_MS,
    )
    report = compare_systems(inputs, {
        "audio_only": always_acquire(len(run["ids"])),
        "always_fusion": always_acquire(len(run["ids"])),
    })
    audio_only = report["systems"]["audio_only"]
    assert audio_only["acquisition"]["average_modalities_activated"] == 1.0
    assert audio_only["latency"]["mean_latency_ms_per_sample"] == pytest.approx(AUDIO_MS)


# ------------------------------------------------------------- serialisation

def test_artefacts_round_trip(run, tmp_path):
    model_path = run["model"].save(tmp_path / "hsig_model.json")
    reloaded = Stage3HSIG.load(model_path)
    assert reloaded.fingerprint() == run["model"].fingerprint()

    record = run["selection"].to_dict()
    assert json.loads(json.dumps(record))["uses_test_labels"] is False

    spec = json.loads(json.dumps(run["spec"].to_dict()))
    assert spec["uses_test_labels"] is False
    assert spec["selected_on_split"] == "validation"


def test_decisions_serialise_to_json(run):
    payload = [decision.to_dict() for decision in run["decisions"]]
    assert json.loads(json.dumps(payload))[0]["router_saw_true_class"] is False
