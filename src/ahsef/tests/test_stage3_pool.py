"""PHASE A: the aligned pool, its verification, and its refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ahsef.identity import SampleAlignmentError
from src.ahsef.registry import DEFAULT_BASELINES
from src.ahsef.stage3.layout import FrozenArtefactError, Stage3Layout
from src.ahsef.stage3.pool import (
    MINIMUM_POOL,
    STAGE3_MODALITIES,
    AlignedPool,
    PoolTooSmallError,
    _availability,
    _text_present,
    alignment_report,
    assert_pools_disjoint,
    build_aligned_pool,
)

EXPERIMENTS = Path("experiments")
POOL_FILE = EXPERIMENTS / "ahsef" / "stage3_text_audio" / "alignment" / "pool_validation.json"

needs_manifests = pytest.mark.skipif(
    not (DEFAULT_BASELINES["audio"].layout(EXPERIMENTS).split_path("validation").exists()
         and DEFAULT_BASELINES["text"].layout(EXPERIMENTS).split_path("validation").exists()),
    reason="requires the frozen audio and text experiment manifests",
)


# ------------------------------------------------------------------- layout

def test_layout_refuses_to_write_into_a_frozen_stage():
    for run in ("stage1", "stage2_llm"):
        with pytest.raises(FrozenArtefactError, match="frozen earlier stage"):
            Stage3Layout(run=run)


def test_layout_rejects_an_unknown_split():
    layout = Stage3Layout()
    with pytest.raises(ValueError, match="split must be one of"):
        layout.pool_path("holdout")


# --------------------------------------------------------------- availability

def test_text_availability_follows_the_file_backed_transcript(tmp_path):
    """MSP-Podcast text lives in files; the column is empty and that is not absence."""
    transcript = tmp_path / "a.txt"
    transcript.write_text("hello", encoding="utf-8")
    frame = pd.DataFrame({
        "sample_id": ["a", "b", "c"],
        "text_source": ["file", "file", "inline"],
        "text": [None, None, "written here"],
        "text_path": ["a.txt", "missing.txt", None],
    })
    present = _text_present(frame, tmp_path)
    assert list(present) == [True, False, True]


def test_audio_availability_requires_the_file_on_disk(tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"0")
    frame = pd.DataFrame({
        "sample_id": ["a", "b"],
        "has_audio": [True, True],
        "audio_path": ["clip.wav", "gone.wav"],
    })
    record = _availability(frame, "audio", tmp_path)
    assert record["available"] == 1
    assert record["unavailable"] == 1
    assert record["declared_but_asset_missing"] == 1
    assert "Nothing is fabricated" in record["rule"]


def test_availability_is_record_level_not_registry_level(tmp_path):
    """A corpus that can carry audio says nothing about whether this row does."""
    (tmp_path / "clip.wav").write_bytes(b"0")
    frame = pd.DataFrame({
        "sample_id": ["a", "b"],
        "has_audio": [True, False],
        "audio_path": ["clip.wav", "clip.wav"],
    })
    assert _availability(frame, "audio", tmp_path)["available"] == 1


# ------------------------------------------------------------------ the pool

@needs_manifests
def test_pool_is_co_split_and_verified():
    pool = build_aligned_pool("validation", minimum=1)
    assert pool.size > 0
    assert pool.checks["identical_sample_ids"] is True
    assert pool.checks["identical_split_assignment"] is True
    assert pool.checks["train_contamination"] == {}
    assert pool.checks["label_agreement_verified"] == pool.size
    assert pool.checks["pool_expanded_across_splits"] is False
    assert pool.modalities == STAGE3_MODALITIES


@needs_manifests
def test_pool_ids_appear_in_the_same_split_of_both_manifests():
    pool = build_aligned_pool("validation", minimum=1)
    for modality in STAGE3_MODALITIES:
        path = DEFAULT_BASELINES[modality].layout(EXPERIMENTS).split_path("validation")
        ids = set(pd.read_parquet(path, columns=["sample_id"])["sample_id"].astype(str))
        assert set(pool.sample_ids) <= ids


@needs_manifests
def test_pool_ids_are_absent_from_every_training_split():
    """The contamination rule, checked directly rather than trusted."""
    pool = build_aligned_pool("validation", minimum=1)
    for modality in STAGE3_MODALITIES:
        path = DEFAULT_BASELINES[modality].layout(EXPERIMENTS).split_path("train")
        ids = set(pd.read_parquet(path, columns=["sample_id"])["sample_id"].astype(str))
        assert not (set(pool.sample_ids) & ids)


@needs_manifests
def test_validation_and_test_pools_are_disjoint():
    validation = build_aligned_pool("validation", minimum=1)
    test = build_aligned_pool("test", minimum=1)
    assert_pools_disjoint(validation, test)
    assert not set(validation.sample_ids) & set(test.sample_ids)


@needs_manifests
def test_pool_too_small_stops_rather_than_inventing_alignment():
    with pytest.raises(PoolTooSmallError, match="rather than manufacturing"):
        build_aligned_pool("validation", minimum=10 ** 9)


@needs_manifests
def test_pool_is_reproducible():
    first = build_aligned_pool("validation", minimum=1)
    second = build_aligned_pool("validation", minimum=1)
    assert first.fingerprint == second.fingerprint
    assert first.sample_ids == second.sample_ids


# ------------------------------------------------------------- serialisation

def _pool(ids):
    return AlignedPool(
        split="validation", sample_ids=list(ids),
        labels={item: 0 for item in ids}, checks={"identical_sample_ids": True},
    )


def test_pool_round_trips(tmp_path):
    pool = _pool(["a", "b", "c"])
    reloaded = AlignedPool.load(pool.save(tmp_path / "pool.json"))
    assert reloaded.sample_ids == pool.sample_ids
    assert reloaded.fingerprint == pool.fingerprint


def test_an_edited_pool_file_is_refused(tmp_path):
    """A fingerprint that no longer matches means the run is not reproducible."""
    path = _pool(["a", "b", "c"]).save(tmp_path / "pool.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["sample_ids"].append("smuggled")
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SampleAlignmentError, match="fingerprint"):
        AlignedPool.load(path)


def test_overlapping_pools_are_refused():
    with pytest.raises(SampleAlignmentError, match="contaminated"):
        assert_pools_disjoint(
            _pool(["a", "b"]),
            AlignedPool(split="test", sample_ids=["b", "c"]),
        )


def test_report_states_what_was_and_was_not_done():
    report = alignment_report({"validation": _pool(["a", "b"])}, minimum=MINIMUM_POOL)
    assert any("co-split" in item for item in report["verified"])
    assert any("NOT expanded" in item for item in report["not_done"])
    assert any("fabricated" in item for item in report["not_done"])
    assert "MSP-Podcast" in report["limitation"]


# ------------------------------------------------- the produced artefact

@pytest.mark.skipif(not POOL_FILE.exists(), reason="run --stage align first")
def test_written_pool_matches_a_fresh_build():
    written = AlignedPool.load(POOL_FILE)
    assert written.fingerprint == build_aligned_pool("validation", minimum=1).fingerprint
    assert written.availability["audio"]["unavailable"] == 0
    assert written.availability["text"]["unavailable"] == 0
