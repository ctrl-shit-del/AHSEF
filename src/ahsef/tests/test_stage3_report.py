"""The generated results document: it must be generated, and it must be complete.

A deliverable that silently loses a section is worse than one that is obviously
missing it, so the renderer is checked for the sections the brief names, and for
the property that a missing artefact renders as *absent* rather than as nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ahsef.stage3.layout import Stage3Layout
from src.ahsef.stage3.render import render_reports

DOCS = Path("docs")
RESULTS = DOCS / "ahsef_stage3_results.md"
METHOD = DOCS / "ahsef_stage3_text_audio.md"
STAGE3 = Path("experiments/ahsef/stage3_text_audio")


def test_a_missing_artefact_renders_as_absent(tmp_path):
    layout = Stage3Layout(run="stage3_empty", root=tmp_path)
    layout.prepare()
    written = render_reports(layout, docs=tmp_path / "docs")
    text = written[0].read_text(encoding="utf-8")
    assert "*Not run" in text
    # The claim section does not depend on any artefact and must still appear.
    assert "AHSEF dynamically determines whether additional audio evidence" in text


def test_the_claim_section_never_overstates(tmp_path):
    layout = Stage3Layout(run="stage3_empty", root=tmp_path)
    layout.prepare()
    text = render_reports(layout, docs=tmp_path / "docs")[0].read_text(encoding="utf-8")
    assert "not** claimed" in text or "Specifically **not** claimed" in text
    assert "best modality among all modalities" in text
    assert "MSP-Podcast" in text


@pytest.mark.skipif(not METHOD.exists(), reason="method document not written")
def test_method_document_states_the_limits():
    text = METHOD.read_text(encoding="utf-8")
    for phrase in (
        "not** a demonstration that AHSEF selects the best modality",
        "REQUESTED_BUT_UNAVAILABLE",
        "Known limitations",
        "target_stop_accuracy",
        "509",
    ):
        assert phrase in text, phrase


@pytest.mark.skipif(not RESULTS.exists(), reason="run --stage report first")
def test_results_document_carries_every_required_section():
    text = RESULTS.read_text(encoding="utf-8")
    for heading in (
        "The aligned Text+Audio pool",
        "Does audio actually help?",
        "HSIG: can the gain be predicted?",
        "The frozen configuration",
        "Validation experiment",
        "Ablations",
        "The scientific claim",
    ):
        assert heading in text, heading


@pytest.mark.skipif(not RESULTS.exists(), reason="run --stage report first")
def test_results_document_reports_the_required_quantities():
    text = RESULTS.read_text(encoding="utf-8")
    for phrase in (
        "Macro-F1", "Weighted-F1", "Balanced acc.", "Acquisition rate",
        "Modalities/sample", "Latency", "AUROC", "AUPRC", "Spearman",
        "Confusion matrices", "ECE", "Brier",
        "Text wrong → fusion right", "Text right → fusion wrong",
        "REQUESTED_BUT_UNAVAILABLE",
        "acquiring at random at the same rate",
    ):
        assert phrase in text, phrase


@pytest.mark.skipif(not RESULTS.exists(), reason="run --stage report first")
def test_results_document_matches_its_artefacts():
    """Spot-check that the rendered numbers came from the JSON, not from a keyboard."""
    text = RESULTS.read_text(encoding="utf-8")
    oracle = json.loads(
        (STAGE3 / "reports" / "oracle_gain_validation.json").read_text(encoding="utf-8")
    )
    fused = oracle["systems"]["text_llm+audio"]
    assert f"{fused['macro_f1']:.4f}" in text
    assert f"{fused['accuracy']:.4f}" in text
    assert str(oracle["acquisition_outcome"]["fixed_by_audio"]) in text

    hsig = json.loads((STAGE3 / "reports" / "hsig_quality.json").read_text(encoding="utf-8"))
    selected = hsig["selected"]["variant"]
    auroc = hsig["variants"][selected]["identifying_useful_acquisitions"]["auroc"]
    assert f"{auroc:.4f}" in text


@pytest.mark.skipif(
    not (STAGE3 / "reports" / "hsig_quality.json").exists(), reason="run --stage hsig first"
)
def test_the_hsig_verdict_is_stated_not_buried():
    """Whether HSIG beat the simple baseline must appear in the document."""
    hsig = json.loads((STAGE3 / "reports" / "hsig_quality.json").read_text(encoding="utf-8"))
    verdict = hsig["comparison"]["verdict"]
    assert verdict is not None
    if RESULTS.exists():
        assert verdict["statement"] in RESULTS.read_text(encoding="utf-8")
