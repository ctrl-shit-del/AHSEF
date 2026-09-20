"""Deterministic sample selection, manifests, and split isolation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.ahsef.llm.samples import (
    SELECTION_METHOD,
    SampleManifest,
    assert_disjoint,
    restrict,
    select_samples,
)


def population(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sample_id": [f"id{i:05d}" for i in range(n)],
        "dataset": rng.choice(["MSP-Podcast", "MELD"], n, p=[0.94, 0.06]),
        "canonical_emotion_id": rng.choice(
            range(7), n, p=[0.38, 0.27, 0.11, 0.17, 0.01, 0.02, 0.04]
        ),
    })


# --------------------------------------------------------------- determinism

def test_the_same_seed_yields_the_identical_ordered_id_list():
    frame = population()
    first = select_samples(frame, 500, 42, "validation", "exp")
    second = select_samples(frame, 500, 42, "validation", "exp")
    assert first.sample_ids == second.sample_ids
    assert first.fingerprint == second.fingerprint


def test_a_different_seed_yields_a_different_selection():
    frame = population()
    assert (
        select_samples(frame, 500, 42, "validation", "exp").sample_ids
        != select_samples(frame, 500, 43, "validation", "exp").sample_ids
    )


def test_selection_does_not_depend_on_input_row_order():
    """Otherwise the manifest would not be reproducible from the same data."""
    frame = population()
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert (
        select_samples(frame, 400, 42, "validation", "exp").sample_ids
        == select_samples(shuffled, 400, 42, "validation", "exp").sample_ids
    )


def test_the_requested_size_is_hit_exactly():
    frame = population()
    for size in (10, 137, 500, 1000):
        assert select_samples(frame, size, 42, "validation", "exp").size == size


def test_requesting_more_than_the_population_returns_everything():
    frame = population(50)
    assert select_samples(frame, 999, 42, "validation", "exp").size == 50


# -------------------------------------------------------------- proportions

def test_the_sample_keeps_the_population_class_balance():
    frame = population(5000)
    manifest = select_samples(frame, 1000, 42, "validation", "exp")
    for name, count in manifest.class_counts.items():
        expected = manifest.population_class_counts[name] / manifest.population * 1000
        assert abs(count - expected) <= max(2, 0.05 * expected)


def test_a_rare_class_is_not_dropped():
    frame = pd.DataFrame({
        "sample_id": [f"id{i}" for i in range(1000)],
        "dataset": ["X"] * 1000,
        "canonical_emotion_id": [0] * 996 + [4, 5, 5, 6],
    })
    manifest = select_samples(frame, 100, 42, "validation", "exp")
    assert {"fear", "disgust", "surprise"} <= set(manifest.class_counts)


def test_the_manifest_records_that_labels_did_not_gate_inclusion():
    record = select_samples(population(), 200, 42, "test", "exp").to_dict()
    assert record["labels_used_for_inclusion"] is False
    assert record["rebalanced"] is False
    assert record["selection_method"] == SELECTION_METHOD


# ---------------------------------------------------------------- manifests

def test_a_manifest_round_trips_through_disk(tmp_path):
    manifest = select_samples(population(), 300, 42, "validation", "exp")
    path = manifest.save(tmp_path / "m.json")
    restored = SampleManifest.load(path)
    assert restored.sample_ids == manifest.sample_ids
    assert restored.fingerprint == manifest.fingerprint
    assert restored.seed == manifest.seed


def test_an_edited_manifest_is_refused(tmp_path):
    """The fingerprint is what makes 'these were the samples' checkable."""
    manifest = select_samples(population(), 100, 42, "validation", "exp")
    path = manifest.save(tmp_path / "m.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["sample_ids"][0] = "tampered"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="no longer matches its recorded fingerprint"):
        SampleManifest.load(path)


def test_the_fingerprint_is_order_sensitive():
    frame = population()
    manifest = select_samples(frame, 100, 42, "validation", "exp")
    reversed_manifest = SampleManifest(
        experiment_id="e", split="validation", seed=42, method=SELECTION_METHOD,
        requested=100, sample_ids=list(reversed(manifest.sample_ids)),
    )
    assert reversed_manifest.fingerprint != manifest.fingerprint


# ------------------------------------------------------------- isolation

def test_validation_and_test_manifests_must_be_disjoint():
    frame = population()
    manifest = select_samples(frame, 100, 42, "validation", "exp")
    with pytest.raises(ValueError, match="share 100 sample ids"):
        assert_disjoint(manifest, manifest)


def test_disjoint_manifests_pass():
    left = select_samples(population(500, seed=1), 100, 42, "validation", "exp")
    right = SampleManifest(
        experiment_id="e", split="test", seed=42, method=SELECTION_METHOD,
        requested=2, sample_ids=["other-a", "other-b"],
    )
    assert_disjoint(left, right)


# ------------------------------------------------------------- restriction

def test_restrict_returns_the_manifest_order():
    frame = population()
    manifest = select_samples(frame, 200, 42, "validation", "exp")
    narrowed = restrict(frame, manifest)
    assert narrowed["sample_id"].tolist() == manifest.sample_ids
    assert len(narrowed) == 200


def test_restrict_refuses_a_frame_missing_manifest_ids():
    frame = population()
    manifest = select_samples(frame, 200, 42, "validation", "exp")
    with pytest.raises(KeyError, match="absent from the frame"):
        restrict(frame.iloc[:10], manifest)


def test_a_frame_without_the_label_column_is_refused():
    frame = population().drop(columns=["canonical_emotion_id"])
    with pytest.raises(ValueError, match="canonical_emotion_id"):
        select_samples(frame, 10, 42, "validation", "exp")
