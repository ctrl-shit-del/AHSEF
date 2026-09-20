"""End-to-end CLI behaviour, driven from a synthetic transcript (no network).

These cover the guards that only exist at the command boundary: the locked-test
rule, the frozen-threshold rule, and the uncertainty-policy consistency check.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ahsef.cli.run_llm_text import stratified_sample
from src.ahsef.layout import AhsefLayout
from src.ahsef.llm.inference import LLMTextModality, build_prediction_set
from src.ahsef.llm.mapping import CANONICAL_EMOTIONS
from src.ahsef.llm.provider import ScriptedProvider


# --------------------------------------------------------------- sampling

def frame_of(labels) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(len(labels))],
        "dataset": ["MELD"] * len(labels),
        "text": [f"t{i}" for i in range(len(labels))],
        "canonical_emotion_id": list(labels),
    })


def test_stratified_sample_hits_the_requested_size_exactly():
    frame = frame_of([0] * 500 + [1] * 300 + [2] * 200)
    assert len(stratified_sample(frame, 100, seed=42)) == 100


def test_stratified_sample_keeps_class_proportions():
    frame = frame_of([0] * 500 + [1] * 300 + [2] * 200)
    counts = stratified_sample(frame, 100, 42)["canonical_emotion_id"].value_counts()
    assert counts[0] == pytest.approx(50, abs=2)
    assert counts[1] == pytest.approx(30, abs=2)


def test_a_rare_class_still_appears():
    """A class with two members must not vanish from the subsample."""
    frame = frame_of([0] * 998 + [5, 6])
    sample = stratified_sample(frame, 50, 42)
    assert {5, 6} <= set(sample["canonical_emotion_id"])


def test_sampling_is_deterministic_under_a_seed():
    frame = frame_of([0] * 200 + [1] * 200)
    first = stratified_sample(frame, 60, 7)["sample_id"].tolist()
    second = stratified_sample(frame, 60, 7)["sample_id"].tolist()
    assert first == second


def test_requesting_more_than_available_returns_everything():
    frame = frame_of([0] * 10)
    assert len(stratified_sample(frame, 999, 42)) == 10


# ------------------------------------------------------------- router CLI

def build_run(tmp_path, confidences, split="validation") -> AhsefLayout:
    """Write an LLM prediction set into a fresh AHSEF run directory."""
    layout = AhsefLayout(run="t", root=tmp_path)
    layout.prepare()
    provider = ScriptedProvider([
        {"emotion": "happy", "class_scores": {n: 0.1 for n in CANONICAL_EMOTIONS},
         "confidence": value, "evidence_strength": 0.5, "ambiguity": 1.0 - value,
         "evidence": "c", "abstain": False}
        for value in confidences
    ])
    modality = LLMTextModality(provider, uncertainty_policy="llm_confidence")
    labels = [1 if value > 0.5 else 2 for value in confidences]
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(len(confidences))],
        "dataset": ["MELD"] * len(confidences),
        "text": [f"t{i}" for i in range(len(confidences))],
        "canonical_emotion_id": labels,
    })
    predictions = build_prediction_set(
        modality.predict_frame(frame), split, modality.provenance()
    )
    predictions.save(
        layout.prediction_path("text_llm", split),
        layout.prediction_meta_path("text_llm", split),
    )
    return layout


def run_router(layout, argv):
    from src.ahsef.cli.run_llm_router import main

    return main([
        "--run", layout.run, "--ahsef-root", str(layout.root),
        "--uncertainty-policy", "llm_confidence", "--candidates", "audio", *argv,
    ])


def test_router_selects_a_threshold_on_validation_and_writes_it(tmp_path, capsys):
    layout = build_run(tmp_path, [0.9, 0.85, 0.2, 0.15])
    assert run_router(layout, ["--split", "validation"]) == 0
    selection = json.loads(
        (layout.base / "llm" / "threshold_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selected_on_split"] == "validation"
    assert selection["uses_test_labels"] is False
    assert (layout.base / "llm" / "routing_records_validation.parquet").exists()
    assert layout.routing_log_path("llm_routing_validation").exists()


def test_router_refuses_test_before_a_threshold_is_frozen(tmp_path):
    layout = build_run(tmp_path, [0.9, 0.2], split="test")
    with pytest.raises(SystemExit, match="No frozen threshold"):
        run_router(layout, ["--split", "test"])


def test_router_reuses_the_frozen_threshold_on_test(tmp_path):
    layout = build_run(tmp_path, [0.9, 0.85, 0.2, 0.15], split="validation")
    run_router(layout, ["--split", "validation"])
    tau = json.loads(
        (layout.base / "llm" / "threshold_selection.json").read_text(encoding="utf-8")
    )["threshold"]

    build_run(tmp_path, [0.95, 0.3], split="test")
    assert run_router(layout, ["--split", "test"]) == 0
    report = json.loads(
        layout.report_path("llm_routing_test.json").read_text(encoding="utf-8")
    )
    assert report["threshold"] == pytest.approx(tau)
    assert "loaded frozen" in report["threshold_source"]
    assert report["threshold_selection"] is None


def test_router_refuses_a_threshold_selected_under_another_policy(tmp_path):
    from src.ahsef.cli.run_llm_router import main

    layout = build_run(tmp_path, [0.9, 0.85, 0.2, 0.15], split="validation")
    run_router(layout, ["--split", "validation"])
    build_run(tmp_path, [0.9, 0.2], split="test")
    with pytest.raises(SystemExit, match="different quantities"):
        main([
            "--run", layout.run, "--ahsef-root", str(layout.root),
            "--uncertainty-policy", "score_entropy", "--candidates", "audio",
            "--split", "test",
        ])


def test_router_report_refuses_the_stronger_scientific_claim(tmp_path):
    layout = build_run(tmp_path, [0.9, 0.85, 0.2, 0.15])
    run_router(layout, ["--split", "validation"])
    report = json.loads(
        layout.report_path("llm_routing_validation.json").read_text(encoding="utf-8")
    )
    claim = report["scientific_claim"]
    assert "NOT evidence that AHSEF improves" in claim
    assert report["policy"]["acquisition_implemented"] is False


def test_router_reports_that_no_modality_was_acquirable(tmp_path):
    """With no alignment index the honest answer is 'nothing available'."""
    layout = build_run(tmp_path, [0.9, 0.9, 0.2, 0.2])
    run_router(layout, ["--split", "validation", "--root", str(tmp_path / "absent")])
    report = json.loads(
        layout.report_path("llm_routing_validation.json").read_text(encoding="utf-8")
    )
    availability = report["availability"]
    assert availability["requests"] > 0
    assert availability["requests_with_no_available_modality"] == availability["requests"]


def test_an_operator_supplied_threshold_is_labelled_as_such(tmp_path):
    layout = build_run(tmp_path, [0.9, 0.2])
    run_router(layout, ["--split", "validation", "--threshold", "0.5"])
    report = json.loads(
        layout.report_path("llm_routing_validation.json").read_text(encoding="utf-8")
    )
    assert report["threshold"] == pytest.approx(0.5)
    assert "operator-supplied" in report["threshold_source"]


def test_router_fails_clearly_when_predictions_are_missing(tmp_path):
    layout = AhsefLayout(run="t", root=tmp_path)
    layout.prepare()
    with pytest.raises(SystemExit, match="No LLM predictions"):
        run_router(layout, ["--split", "validation"])
