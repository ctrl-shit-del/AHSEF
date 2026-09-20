"""The ``audio_strong`` expert: feature cache, probe head, manifests, registry.

These tests never load wav2vec2 and never read an audio file. The encoder is a
frozen external artefact; what this project is responsible for is the cache
around it, the head on top of it, and the manifest discipline that keeps the
comparison against ``audio_25pct`` honest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.data.cached_feature_dataset import (
    CachedFeatureDataset,
    create_cached_feature_dataloader,
)
from src.data.features.wav2vec2_features import (
    SUPPORTED_BUNDLES,
    FeatureExtractionError,
    FeatureStore,
    Wav2Vec2Config,
    manifest_fingerprint,
)
from src.models.strong_experts import LayerWeightedProbe
from src.training.audio_strong_manifest import (
    DEFAULT_CAP,
    MANIFEST_RECORD,
    SOURCE_EXPERIMENT,
    STRONG_EXPERIMENT,
    ManifestError,
    build_manifests,
    cap_training_partition,
)

LAYERS, DIM = 12, 768


# ============================================================
# Extractor configuration
# ============================================================

def test_fingerprint_tracks_what_changes_the_numbers():
    base = Wav2Vec2Config()
    assert Wav2Vec2Config().fingerprint == base.fingerprint
    assert Wav2Vec2Config(max_seconds=8.0).fingerprint != base.fingerprint
    assert Wav2Vec2Config(bundle="HUBERT_BASE").fingerprint != base.fingerprint
    assert Wav2Vec2Config(normalize_waveform=True).fingerprint != base.fingerprint


def test_fingerprint_ignores_what_only_changes_the_speed():
    """Batch size and thread count change runtime, not the stored features."""
    base = Wav2Vec2Config()
    assert Wav2Vec2Config(batch_size=64).fingerprint == base.fingerprint
    assert Wav2Vec2Config(num_threads=1).fingerprint == base.fingerprint


def test_an_unknown_bundle_is_refused():
    with pytest.raises(ValueError, match="bundle must be one of"):
        Wav2Vec2Config(bundle="whisper-large")
    assert "WAV2VEC2_BASE" in SUPPORTED_BUNDLES


def test_config_declares_the_encoder_is_frozen():
    record = Wav2Vec2Config().to_dict()
    assert record["frozen_encoder"] is True
    assert "unpadded" in record["pooling"]


# ============================================================
# Feature store
# ============================================================

def make_store(tmp_path, fingerprint="fp") -> FeatureStore:
    return FeatureStore.open(tmp_path / "cache", fingerprint, LAYERS, DIM, shard_size=4)


def test_store_round_trips_in_requested_order(tmp_path):
    store = make_store(tmp_path)
    features = np.arange(3 * LAYERS * DIM, dtype=np.float32).reshape(3, LAYERS, DIM)
    store.append(["a", "b", "c"], features)

    loaded = store.load(["c", "a"])
    assert loaded.shape == (2, LAYERS, DIM)
    np.testing.assert_allclose(loaded[0], features[2].astype(np.float16), rtol=1e-2)
    np.testing.assert_allclose(loaded[1], features[0].astype(np.float16), rtol=1e-2)


def test_store_reports_what_is_missing(tmp_path):
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    assert store.missing(["a", "b", "c", "d"]) == ["c", "d"]
    assert "a" in store and "z" not in store
    assert len(store) == 2


def test_loading_a_missing_sample_raises_rather_than_substituting(tmp_path):
    """The core integrity rule of the cache."""
    store = make_store(tmp_path)
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    with pytest.raises(FeatureExtractionError, match="never substituted"):
        store.load(["a", "ghost"])


def test_store_resumes_across_reopen(tmp_path):
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    reopened = FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM)
    assert len(reopened) == 2
    assert reopened.missing(["a", "b", "c"]) == ["c"]
    # and it keeps appending into a fresh shard rather than clobbering
    reopened.append(["c"], np.ones((1, LAYERS, DIM), dtype=np.float32))
    assert len(FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM)) == 3


def test_a_cache_from_another_configuration_is_refused(tmp_path):
    """Mixing two representations in one matrix is the worst silent corruption."""
    store = make_store(tmp_path, "fingerprint-one")
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    with pytest.raises(FeatureExtractionError, match="different extractor"):
        FeatureStore.open(tmp_path / "cache", "fingerprint-two", LAYERS, DIM)


def test_store_refuses_mismatched_shapes(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(FeatureExtractionError, match="ids but"):
        store.append(["a", "b"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    with pytest.raises(FeatureExtractionError, match="Expected"):
        store.append(["a"], np.zeros((1, 3, DIM), dtype=np.float32))


def test_manifest_fingerprint_is_order_sensitive():
    assert manifest_fingerprint(["a", "b"]) == manifest_fingerprint(["a", "b"])
    assert manifest_fingerprint(["a", "b"]) != manifest_fingerprint(["b", "a"])
    assert manifest_fingerprint(["a", "b"]) != manifest_fingerprint(["a", "b", "c"])


# ============================================================
# Probe head
# ============================================================

def test_probe_maps_layers_to_classes():
    model = LayerWeightedProbe(num_layers=LAYERS, feature_dim=DIM, num_classes=7)
    logits = model(torch.randn(5, LAYERS, DIM))
    assert logits.shape == (5, 7)


def test_layer_weights_are_a_distribution_and_start_uniform():
    model = LayerWeightedProbe(num_layers=LAYERS, feature_dim=DIM)
    weights = model.layer_weights().detach()
    assert weights.shape == (LAYERS,)
    assert float(weights.sum()) == pytest.approx(1.0)
    # Uniform at initialisation: the fitted weighting must be a finding.
    assert torch.allclose(weights, torch.full((LAYERS,), 1.0 / LAYERS), atol=1e-6)


def test_probe_rejects_the_wrong_number_of_layers():
    model = LayerWeightedProbe(num_layers=LAYERS, feature_dim=DIM)
    with pytest.raises(ValueError, match="Expected 12 encoder layers"):
        model(torch.randn(2, 6, DIM))
    with pytest.raises(ValueError, match=r"Expected \[batch, num_layers"):
        model(torch.randn(2, DIM))


def test_probe_accepts_and_ignores_lengths():
    """The shared trainer passes lengths; cached features are already pooled."""
    model = LayerWeightedProbe(num_layers=LAYERS, feature_dim=DIM)
    features = torch.randn(3, LAYERS, DIM)
    model.eval()
    with torch.inference_mode():
        assert torch.allclose(
            model(features), model(features, torch.tensor([1, 2, 3]))
        )


def test_probe_describes_itself_as_a_frozen_encoder_head():
    record = LayerWeightedProbe().describe()
    assert record["class"] == "LayerWeightedProbe"
    assert record["encoder_frozen"] is True
    assert record["encoder_trained_by_this_project"] is False
    # The head must be small: this is a probe, not a second model.
    assert record["trainable_parameters"] < 1_000_000


def test_probe_trains_only_the_head():
    model = LayerWeightedProbe()
    assert all(p.requires_grad for p in model.parameters())
    assert model.describe()["parameters"] == model.describe()["trainable_parameters"]


# ============================================================
# Manifests
# ============================================================

def synthetic_manifest(counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for class_id, count in counts.items():
        for index in range(count):
            rows.append({
                "sample_id": f"c{class_id}_s{index:05d}",
                "dataset": "SYNTH",
                "canonical_emotion_id": class_id,
                "audio_path": f"x/{class_id}_{index}.wav",
            })
    return pd.DataFrame(rows)


def test_cap_keeps_rare_classes_whole():
    frame = synthetic_manifest({0: 500, 1: 50, 2: 10})
    capped, report = cap_training_partition(frame, cap=100, seed=42)
    assert report["neutral"]["kept"] == 100
    assert report["happy"]["kept"] == 50
    assert report["sad"]["kept"] == 10
    assert report["happy"]["action"] == "kept whole"
    assert "capped from 500" in report["neutral"]["action"]
    assert len(capped) == 160


def test_cap_is_deterministic():
    frame = synthetic_manifest({0: 500, 1: 300})
    first, _ = cap_training_partition(frame, cap=100, seed=42)
    second, _ = cap_training_partition(frame, cap=100, seed=42)
    assert first["sample_id"].tolist() == second["sample_id"].tolist()


def test_cap_is_independent_of_input_row_order():
    frame = synthetic_manifest({0: 200, 1: 200})
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    first, _ = cap_training_partition(frame, cap=50, seed=42)
    second, _ = cap_training_partition(shuffled, cap=50, seed=42)
    assert first["sample_id"].tolist() == second["sample_id"].tolist()


def test_a_different_seed_draws_a_different_subset():
    frame = synthetic_manifest({0: 500})
    first, _ = cap_training_partition(frame, cap=50, seed=1)
    second, _ = cap_training_partition(frame, cap=50, seed=2)
    assert first["sample_id"].tolist() != second["sample_id"].tolist()


def test_cap_must_be_positive():
    with pytest.raises(ManifestError, match="cap must be at least 1"):
        cap_training_partition(synthetic_manifest({0: 5}), cap=0, seed=42)


def test_strong_experiment_is_not_the_baseline():
    """The single most important guard: nothing lands in experiments/audio/."""
    assert STRONG_EXPERIMENT != SOURCE_EXPERIMENT
    strong = ExperimentLayout.from_name(STRONG_EXPERIMENT, Path("experiments"))
    baseline = ExperimentLayout.from_name(SOURCE_EXPERIMENT, Path("experiments"))
    assert strong.base != baseline.base
    assert strong.modality == "audio_strong"
    assert baseline.modality == "audio"
    assert "audio_strong" in str(strong.base)


# ============================================================
# Cached dataset
# ============================================================

def write_manifest(path: Path, ids, labels) -> Path:
    pd.DataFrame({
        "sample_id": list(ids),
        "dataset": ["SYNTH"] * len(ids),
        "canonical_emotion_id": list(labels),
    }).to_parquet(path, index=False)
    return path


def test_dataset_orders_rows_by_manifest_not_by_shard(tmp_path):
    store = make_store(tmp_path)
    features = np.stack([np.full((LAYERS, DIM), value, dtype=np.float32)
                         for value in (1.0, 2.0, 3.0)])
    store.append(["c", "a", "b"], features)          # deliberately out of order

    path = write_manifest(tmp_path / "m.parquet", ["a", "b", "c"], [0, 1, 2])
    dataset = CachedFeatureDataset(path, store)
    assert dataset.manifest["sample_id"].tolist() == ["a", "b", "c"]
    assert float(dataset[0]["features"][0, 0]) == pytest.approx(2.0)
    assert float(dataset[1]["features"][0, 0]) == pytest.approx(3.0)
    assert float(dataset[2]["features"][0, 0]) == pytest.approx(1.0)


def test_dataset_refuses_an_uncached_sample(tmp_path):
    store = make_store(tmp_path)
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    path = write_manifest(tmp_path / "m.parquet", ["a", "b"], [0, 1])
    with pytest.raises(KeyError, match="never replaced with zeros"):
        CachedFeatureDataset(path, store)


def test_dataloader_batches_the_expected_shapes(tmp_path):
    store = make_store(tmp_path)
    ids = [f"s{i}" for i in range(6)]
    store.append(ids, np.random.rand(6, LAYERS, DIM).astype(np.float32))
    path = write_manifest(tmp_path / "m.parquet", ids, [0, 1, 2, 3, 4, 5])
    loader = create_cached_feature_dataloader(path, store, batch_size=4)
    batch = next(iter(loader))
    assert batch["features"].shape == (4, LAYERS, DIM)
    assert batch["label"].shape == (4,)
    assert batch["features"].dtype == torch.float32


def test_dataset_exposes_geometry_for_the_model_builder(tmp_path):
    store = make_store(tmp_path)
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    path = write_manifest(tmp_path / "m.parquet", ["a"], [0])
    dataset = CachedFeatureDataset(path, store)
    assert dataset.num_layers == LAYERS
    assert dataset.feature_dim == DIM


# ============================================================
# Registry
# ============================================================

def test_audio_strong_is_registered_without_displacing_audio():
    from src.training.modalities import (
        EXPERIMENT_SPECS,
        FORWARD_PROBES,
        MODEL_BUILDERS,
        RUNNER_CLASSES,
    )
    for registry in (EXPERIMENT_SPECS, RUNNER_CLASSES):
        assert "audio" in registry and "audio_strong" in registry
        assert registry["audio"] is not registry["audio_strong"]
    assert "AudioEmotionBaseline" in MODEL_BUILDERS
    assert "LayerWeightedProbe" in MODEL_BUILDERS
    assert "LayerWeightedProbe" in FORWARD_PROBES


def test_the_probe_rebuilds_from_a_run_summary_record():
    """AHSEF reloads architectures from the recorded run summary, not from code."""
    from src.training.modalities import build_model_from_record

    model = build_model_from_record({
        "model": {
            "class": "LayerWeightedProbe", "num_layers": LAYERS, "feature_dim": DIM,
            "hidden_dim": 128, "num_classes": 7, "dropout": 0.1,
        },
        "config": {},
    })
    assert isinstance(model, LayerWeightedProbe)
    assert model.num_layers == LAYERS and model.hidden_dim == 128
    assert model(torch.randn(2, LAYERS, DIM)).shape == (2, 7)


def test_audio_baseline_still_rebuilds_unchanged():
    """A regression guard: registering a new expert must not disturb the old one."""
    from src.models.baselines import AudioEmotionBaseline
    from src.training.modalities import build_model_from_record

    model = build_model_from_record({
        "model": {"class": "AudioEmotionBaseline", "n_mels": 64, "hidden_dim": 256,
                  "num_classes": 7},
        "config": {},
    })
    assert isinstance(model, AudioEmotionBaseline)


# ============================================================
# The produced artefacts
# ============================================================

STRONG_META = Path("experiments/audio_strong/full/metadata")


@pytest.mark.skipif(
    not (STRONG_META / MANIFEST_RECORD).exists(),
    reason="audio_strong manifests not built",
)
def test_written_manifests_keep_evaluation_identical_to_the_baseline():
    summary = json.loads(
        (STRONG_META / MANIFEST_RECORD).read_text(encoding="utf-8")
    )
    assert summary["evaluation_splits_identical_to_source"] is True
    assert summary["splits_disjoint"] is True
    assert summary["class_cap"] == DEFAULT_CAP

    baseline = ExperimentLayout.from_name(SOURCE_EXPERIMENT, Path("experiments"))
    for split in ("validation", "test"):
        left = pd.read_parquet(baseline.split_path(split), columns=["sample_id"])
        assert summary["splits"][split]["fingerprint_sha256"] == manifest_fingerprint(
            left["sample_id"].astype(str).tolist()
        )
        assert summary["splits"][split]["samples"] == len(left)


@pytest.mark.skipif(
    not (STRONG_META / MANIFEST_RECORD).exists(),
    reason="audio_strong manifests not built",
)
def test_written_training_manifest_is_capped_and_keeps_rare_classes():
    summary = json.loads(
        (STRONG_META / MANIFEST_RECORD).read_text(encoding="utf-8")
    )
    train = summary["splits"]["train"]
    assert train["samples"] < train["source_samples"]
    report = train["class_cap_report"]
    for name in ("fear", "disgust", "surprise"):
        assert report[name]["action"] == "kept whole", name
        assert report[name]["kept"] == report[name]["available"]
    for name in ("neutral", "happy", "sad", "angry"):
        assert report[name]["kept"] == DEFAULT_CAP, name
    assert set(train["class_counts"]) <= set(CANONICAL_EMOTION_CLASSES)


@pytest.mark.skipif(
    not (STRONG_META / MANIFEST_RECORD).exists(),
    reason="audio_strong manifests not built",
)
def test_the_cap_is_declared_as_a_training_only_label_use():
    summary = json.loads(
        (STRONG_META / MANIFEST_RECORD).read_text(encoding="utf-8")
    )
    assert summary["labels_used_for_inclusion"] is True
    assert "training partition only" in summary["labels_note"]
    assert "no evaluation sample is selected or excluded" in summary["labels_note"]


# ============================================================
# End-to-end training smoke
# ============================================================

@pytest.mark.smoke
def test_audio_strong_trains_end_to_end_on_a_synthetic_cache(tmp_path):
    """The whole runner path: cache -> loader -> probe -> checkpoint -> summary.

    Deliberately given a learnable synthetic signal planted in one encoder
    layer, so a failure here is an integration failure rather than a modelling
    one. Marked ``smoke`` because it trains a real (tiny) model.
    """
    from src.training.audio_strong_experiment import (
        AudioStrongExperimentConfig,
        run_iteration,
    )

    root = tmp_path / "experiments"
    meta = root / "audio_strong" / "full" / "metadata"
    meta.mkdir(parents=True)
    cache_root = tmp_path / "features"
    fingerprint = Wav2Vec2Config().fingerprint
    rng = np.random.default_rng(0)

    for split, count in (("train", 240), ("validation", 120), ("test", 120)):
        ids = [f"{split}_{index:04d}" for index in range(count)]
        labels = rng.integers(0, 7, count)
        features = rng.normal(size=(count, LAYERS, DIM)).astype(np.float32) * 0.1
        for row, klass in enumerate(labels):
            features[row, 5, int(klass) * 10:(int(klass) + 1) * 10] += 3.0
        pd.DataFrame({
            "sample_id": ids, "dataset": "SYNTH", "canonical_emotion_id": labels,
        }).to_parquet(meta / f"{split}.parquet", index=False)
        FeatureStore.open(cache_root / split, fingerprint, LAYERS, DIM).append(ids, features)

    summary = run_iteration(AudioStrongExperimentConfig(
        experiment_root=str(root), cache_root=str(cache_root),
        epochs=6, batch_size=32, min_epochs=2, patience=3, iteration="1",
    ))

    assert summary["model"]["class"] == "LayerWeightedProbe"
    assert summary["model"]["encoder_frozen"] is True
    assert summary["model"]["encoder_trained_by_this_project"] is False
    assert summary["model"]["front_end"]["bundle"] == "WAV2VEC2_BASE"
    assert summary["model"]["front_end"]["cache_fingerprint"] == fingerprint
    # The planted signal is trivially learnable; anything near chance means the
    # features are not reaching the head.
    assert summary["best_val_macro_f1"] > 0.8
    assert summary["checkpoints"]["reload_verified"] is True
    assert summary["class_weights"]["policy"] == "inverse_frequency_from_train"
    assert (root / "audio_strong" / "full" / "experiment_summary.json").exists()


@pytest.mark.smoke
def test_a_missing_feature_cache_fails_with_instructions(tmp_path):
    from src.training.audio_strong_experiment import (
        AudioStrongExperimentConfig,
        AudioStrongExperimentRunner,
    )

    root = tmp_path / "experiments"
    meta = root / "audio_strong" / "full" / "metadata"
    meta.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        pd.DataFrame({
            "sample_id": ["a"], "dataset": ["S"], "canonical_emotion_id": [0],
        }).to_parquet(meta / f"{split}.parquet", index=False)

    runner = AudioStrongExperimentRunner(AudioStrongExperimentConfig(
        experiment_root=str(root), cache_root=str(tmp_path / "nothing"),
    ))
    with pytest.raises(FileNotFoundError, match="extract_audio_features"):
        runner.build_model()


# ============================================================
# Cache integrity: the concurrent-writer race and its repair
# ============================================================

def test_shard_numbering_comes_from_disk_not_the_counter(tmp_path):
    """Two writers must not choose the same shard number and clobber each other."""
    store = make_store(tmp_path)
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))

    # A second view opened before the first wrote again still picks a fresh shard.
    other = FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM)
    first = store.append(["b"], np.ones((1, LAYERS, DIM), dtype=np.float32))
    second = other.append(["c"], np.full((1, LAYERS, DIM), 2.0, dtype=np.float32))
    assert first != second
    assert len(list((tmp_path / "cache").glob("shard_*.npz"))) == 3


def test_verify_reports_a_healthy_cache(tmp_path):
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    health = store.verify()
    assert health["healthy"] is True
    assert health["misattributed"] == 0
    assert health["extracted_but_unindexed"] == 0
    assert health["index_entries"] == 2


def test_verify_detects_entries_stranded_by_a_lost_index_update(tmp_path):
    """The exact failure this project hit: shards written, index overwritten."""
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    store.append(["c", "d"], np.ones((2, LAYERS, DIM), dtype=np.float32))
    # Simulate a stale writer flushing its older view over the newer index.
    store.index["entries"] = {"a": [0, 0], "b": [0, 1]}
    store.flush()

    reopened = FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM)
    health = reopened.verify()
    assert health["misattributed"] == 0, "no feature is attached to the wrong sample"
    assert health["extracted_but_unindexed"] == 2
    assert health["repairable"] is True
    assert health["healthy"] is True


def test_rebuild_index_recovers_stranded_samples_without_re_extracting(tmp_path):
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    store.append(["c", "d"], np.full((2, LAYERS, DIM), 5.0, dtype=np.float32))
    store.index["entries"] = {"a": [0, 0]}
    store.flush()

    reopened = FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM)
    repair = reopened.rebuild_index()
    assert repair["entries_after"] == 4
    assert repair["recovered"] == 3
    # and the recovered features are the right ones
    np.testing.assert_allclose(
        reopened.load(["c"])[0], np.full((LAYERS, DIM), 5.0), rtol=1e-2
    )


def test_rebuild_keeps_one_row_per_duplicated_id(tmp_path):
    """Two writers extracting the same sample is waste, not corruption."""
    store = make_store(tmp_path)
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    store.append(["a"], np.zeros((1, LAYERS, DIM), dtype=np.float32))
    repair = store.rebuild_index()
    assert repair["rows_scanned"] == 2
    assert repair["entries_after"] == 1
    assert repair["duplicate_rows_dropped"] == 1


def test_verify_flags_a_genuinely_misattributed_entry(tmp_path):
    """The one finding that means the data, not the bookkeeping, is wrong."""
    store = make_store(tmp_path)
    store.append(["a", "b"], np.zeros((2, LAYERS, DIM), dtype=np.float32))
    store.index["entries"]["a"] = [0, 1]          # now points at b's row
    store.flush()
    health = FeatureStore.open(tmp_path / "cache", "fp", LAYERS, DIM).verify()
    assert health["misattributed"] == 1
    assert health["healthy"] is False
    assert health["repairable"] is False


def test_only_one_extractor_may_hold_a_cache(tmp_path):
    from src.data.features.wav2vec2_features import CacheBusyError, ExtractionLock

    held = ExtractionLock(tmp_path).acquire()
    try:
        with pytest.raises(CacheBusyError, match="held by pid"):
            ExtractionLock(tmp_path).acquire()
    finally:
        held.release()
    # released locks are reclaimable without force
    ExtractionLock(tmp_path).acquire().release()


def test_a_lock_can_be_broken_deliberately(tmp_path):
    from src.data.features.wav2vec2_features import ExtractionLock

    ExtractionLock(tmp_path).acquire()            # leaked, as a crash would leave it
    broken = ExtractionLock(tmp_path).acquire(force=True)
    assert broken.acquired is True
    broken.release()


def test_the_lock_does_not_probe_liveness_with_os_kill():
    """``os.kill(pid, 0)`` TERMINATES the target on Windows, this project's platform.

    Checked on the parsed syntax tree rather than on the text, because the module
    legitimately *names* the function in the comment explaining why it is avoided.
    """
    import ast

    from src.data.features import wav2vec2_features

    tree = ast.parse(Path(wav2vec2_features.__file__).read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "kill"
    ]
    assert not calls, "os.kill would terminate the running extractor on Windows"
