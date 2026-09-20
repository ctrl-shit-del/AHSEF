"""Tests for the HSEN phase.

These are about the claims the phase makes, not about line coverage.  Each one
corresponds to something a section of the brief asserts and that would be
expensive to discover was false after a long training run: that the two profiles
build the same model, that a missing modality does not crash or get zero-filled
into significance, that no normalisation is fitted, that the loss masks absent
targets, and that the audit actually catches leakage rather than merely printing
about it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.labels import get_label_space
from src.hsen.audit import (
    audit_manifests,
    check_no_cross_split_ids,
    check_speaker_independence,
)
from src.hsen.data import HSENSample, collate_hsen, stratified_subset
from src.hsen.features.store import FeatureCacheError, SequenceFeatureStore, config_fingerprint
from src.hsen.labels.iemocap import IEMOCAPLabelAdapter, normalise_vad
from src.hsen.labels.mosei import (
    MOSEIEmotionLabelsUnavailable,
    MOSEILabelAdapter,
    sentiment_bucket,
)
from src.hsen.models.hsen import HSENConfig, build_hsen
from src.hsen.training.hsen_trainer import assert_same_architecture, resolve_device
from src.hsen.training.losses import (
    HSENLoss,
    LossConfig,
    balanced_class_weights,
    focal_loss,
    regression_loss,
)
from src.hsen.training.metrics import classification_metrics, concordance_correlation

CLASSES = get_label_space("iemocap_erc6").classes


# ======================================================================
# Section 23 -- the two profiles build the same model
# ======================================================================

def test_cpu_and_cuda_profiles_build_identical_architectures():
    """The central claim of section 23, checked directly rather than assumed."""
    from src.hsen.cli import CPU25_PROFILE, CUDA_PROFILE, build_parser, configs_from_args

    built = {}
    for profile in (CPU25_PROFILE, CUDA_PROFILE):
        parser = build_parser(profile, "test")
        args = parser.parse_args(["--dataset", "iemocap"])
        _, _, modalities, overrides = configs_from_args(args, profile)
        built[profile.name] = build_hsen(
            HSENConfig(modalities=modalities, num_classes=6, **overrides)
        )

    left, right = built["cpu25"], built["full_cuda"]
    assert_same_architecture(left, right)
    assert left.parameter_counts()["total"] == right.parameter_counts()["total"]
    assert left.config.to_dict() == right.config.to_dict()


def test_profiles_differ_only_in_compute_and_data():
    """A profile may change the device, fraction, batch size, workers and AMP.

    If a future edit lets a profile move a model field, this fails -- which is
    the point, because the CPU run stops being a preview of the CUDA one the
    moment it trains something else.
    """
    from src.hsen.cli import CPU25_PROFILE, CUDA_PROFILE

    allowed = {"name", "device", "amp", "data_fraction", "batch_size",
               "num_workers", "epochs"}
    assert set(vars(CPU25_PROFILE)) == allowed
    assert set(vars(CUDA_PROFILE)) == allowed


@pytest.mark.parametrize("extra_argv", [
    [],
    ["--config", "src/hsen/configs/hsen_full_cuda.yaml"],
], ids=["flags-only", "yaml-profile"])
def test_phase16_cuda_iemocap_resolves_to_audio_text_concat_without_video(
    tmp_path, monkeypatch, extra_argv,
):
    """Phase 16: the documented full-data CUDA invocation carries the Phase 15
    selection -- audio+text, concat -- and never asks for a video cache.

    Video is deferred, so IEMOCAP has audio and text caches only.  The CUDA
    launcher once defaulted ``--modalities`` to ``all`` and failed at
    ``open_feature_sources`` with "No feature cache for ['video/train']" --
    after the audit had passed, on the machine with the GPU.  This pins both
    halves: what the arguments resolve to, and which caches that resolution
    actually opens.
    """
    from src.hsen import experiment as experiment_module
    from src.hsen.cli import CUDA_PROFILE, apply_yaml, build_parser, configs_from_args

    parser = build_parser(CUDA_PROFILE, "test")
    argv = ["--dataset", "iemocap", "--data_fraction", "1.0", *extra_argv]
    args = apply_yaml(parser, parser.parse_args(argv))
    trainer, _, modalities, _ = configs_from_args(args, CUDA_PROFILE)

    assert trainer.experiment == "iemocap_erc6"
    assert trainer.dataset == "IEMOCAP"
    assert trainer.data_fraction == 1.0
    assert trainer.device == "cuda" and trainer.amp
    assert tuple(modalities) == ("audio", "text")
    assert args.fusion == "concat"

    # A feature root holding exactly what exists on disk: audio and text, no video.
    for split in ("train", "validation"):
        for modality in ("audio", "text"):
            root = experiment_module.cache_dir("iemocap_erc6", modality, split, tmp_path)
            root.mkdir(parents=True)
            (root / "index.json").write_text(
                json.dumps({"fingerprint": "f", "feature_dim": 768}), encoding="utf-8",
            )

    requested: list[str] = []
    real_cache_dir = experiment_module.cache_dir

    def recording_cache_dir(experiment, modality, split, root=None):
        requested.append(f"{modality}/{split}")
        return real_cache_dir(experiment, modality, split, root)

    monkeypatch.setattr(experiment_module, "cache_dir", recording_cache_dir)
    monkeypatch.setattr(
        experiment_module.SequenceFeatureStore, "open",
        staticmethod(lambda root, fingerprint, modality, dim: modality),
    )
    for split in ("train", "validation"):
        sources = experiment_module.open_feature_sources(
            "iemocap_erc6", split, modalities, tmp_path,
        )
        assert set(sources) == {"audio", "text"}

    assert "video/train" not in requested
    assert "video/validation" not in requested
    assert set(requested) == {"audio/train", "text/train", "audio/validation", "text/validation"}


def test_cuda_device_refuses_to_fall_back_to_cpu():
    """Section 20: fail clearly, never silently substitute a device."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is present, so the failure path cannot be exercised")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device("cuda")
    assert resolve_device("cpu").type == "cpu"


# ======================================================================
# Model
# ======================================================================

def _batch(batch_size=4, lengths=(60, 12, 8)):
    audio, text, video = lengths
    features = {
        "audio": torch.randn(batch_size, audio, 768),
        "text": torch.randn(batch_size, text, 768),
        "video": torch.randn(batch_size, video, 576),
    }
    masks = {k: torch.ones(batch_size, v.shape[1], dtype=torch.bool)
             for k, v in features.items()}
    available = {k: torch.ones(batch_size, dtype=torch.bool) for k in features}
    return features, masks, available


def test_forward_exposes_every_intermediate_section_24_needs():
    model = build_hsen()
    outputs = model(*_batch())
    for key in ("logits", "valence", "arousal", "fused", "fused_sequence",
                "projected", "modality_sequences", "modality_vectors"):
        assert key in outputs, f"{key} is part of the interface later phases read"
    assert outputs["fused"].shape == (4, 256)
    assert set(outputs["modality_vectors"]) == {"audio", "video", "text"}


def test_missing_modality_neither_crashes_nor_contributes():
    """A sample with no video must produce the same output as if video were absent.

    Zero-filling would make "observed and silent" and "never acquired"
    indistinguishable, which is precisely the distinction the routing phase
    exists to act on.
    """
    torch.manual_seed(0)
    model = build_hsen().eval()
    features, masks, available = _batch(batch_size=2)

    with torch.no_grad():
        available["video"][0] = False
        without = model(features, masks, available)["logits"][0].clone()
        # Change the absent modality's content entirely; the output must not move.
        features["video"][0] = torch.randn_like(features["video"][0]) * 100
        again = model(features, masks, available)["logits"][0]
    assert torch.allclose(without, again, atol=1e-5)


def test_all_modalities_absent_is_refused():
    model = build_hsen()
    features, masks, available = _batch(batch_size=2)
    for modality in available:
        available[modality][1] = False
    with pytest.raises(ValueError, match="no available modality"):
        model(features, masks, available)


def test_padding_does_not_leak_into_real_frames():
    torch.manual_seed(0)
    model = build_hsen(HSENConfig(modalities=("text",))).eval()
    features = {"text": torch.randn(1, 20, 768)}
    masks = {"text": torch.ones(1, 20, dtype=torch.bool)}
    masks["text"][0, 10:] = False
    available = {"text": torch.ones(1, dtype=torch.bool)}

    with torch.no_grad():
        before = model(features, masks, available)["logits"].clone()
        features["text"][0, 10:] = 999.0
        after = model(features, masks, available)["logits"]
    assert torch.allclose(before, after, atol=1e-4)


def test_no_fitted_normalisation_anywhere_in_the_model():
    """Section 7: no statistic learned from training data may scale evaluation data.

    ``LayerNorm`` is per-sample and ``BatchNorm`` is not, so the presence of any
    running-statistics buffer is the mechanical form of the question.
    """
    model = build_hsen()
    offenders = [name for name, _ in model.named_buffers()
                 if "running_mean" in name or "running_var" in name]
    assert offenders == [], f"fitted normalisation statistics found: {offenders}"


@pytest.mark.parametrize("fusion", ["husformer", "self_attention", "concat"])
def test_every_fusion_variant_runs_and_reports_its_size(fusion):
    model = build_hsen(HSENConfig(fusion=fusion))
    outputs = model(*_batch())
    assert outputs["logits"].shape == (4, 6)
    assert torch.isfinite(outputs["logits"]).all()
    description = model.fusion.describe()
    assert description["parameters"] > 0
    assert description["class"].lower().startswith(fusion.split("_")[0][:4])


def test_husformer_is_the_default_and_carries_the_survey_geometry():
    config = HSENConfig()
    assert (config.fusion, config.model_dim, config.num_layers, config.num_heads) == (
        "husformer", 256, 2, 4,
    )


# ======================================================================
# Labels
# ======================================================================

def test_iemocap_erc6_keeps_excited_and_frustrated():
    frame = pd.DataFrame({
        "sample_id": list("abcdefgh"),
        "emotion": ["neutral", "happy", "sad", "angry", "excited",
                    "frustrated", "fear", None],
        "valence": [3.0, 4.5, 1.5, 2.0, 4.0, 2.5, 3.0, 3.0],
        "arousal": [3.0, 4.0, 2.0, 4.5, 4.5, 3.5, 3.0, 3.0],
    })
    adapter = IEMOCAPLabelAdapter(protocol="erc6")
    admitted = adapter.selects(frame)
    assert admitted.sum() == 6, "fear and the unlabelled row are outside the protocol"

    labels = adapter.build(frame[admitted].reset_index(drop=True))
    assert labels.class_counts() == {
        "neutral": 1, "happy": 1, "sad": 1, "angry": 1, "excited": 1, "frustrated": 1,
    }


def test_iemocap_ser4_merges_excited_into_happy():
    frame = pd.DataFrame({
        "sample_id": list("abcd"),
        "emotion": ["happy", "excited", "neutral", "frustrated"],
        "valence": [4.0] * 4, "arousal": [4.0] * 4,
    })
    adapter = IEMOCAPLabelAdapter(protocol="ser4")
    admitted = adapter.selects(frame)
    assert admitted.tolist() == [True, True, True, False]
    labels = adapter.build(frame[admitted].reset_index(drop=True))
    assert labels.class_counts()["happy"] == 2


def test_vad_transform_is_fixed_not_fitted():
    """The same rating maps to the same target regardless of what it sits beside."""
    alone = normalise_vad(pd.Series([1.0, 3.0, 5.0]))
    assert np.allclose(alone, [-1.0, 0.0, 1.0])
    # A different surrounding distribution must not move the mapping.
    with_others = normalise_vad(pd.Series([1.0, 3.0, 5.0, 5.0, 5.0, 5.0]))
    assert np.allclose(alone, with_others[:3])
    # Out-of-scale ratings are unlabelled rather than clipped into range.
    assert not np.isfinite(normalise_vad(pd.Series([7.0])))[0]


def test_mosei_sentiment_buckets_match_the_acc7_definition():
    scores = np.array([-3.0, -2.4, -0.5, 0.0, 0.4, 2.0, 3.0, np.nan])
    buckets = sentiment_bucket(scores)
    assert buckets.tolist() == [0, 1, 3, 3, 3, 5, 6, -1]


def test_mosei_refuses_to_read_sentiment_derived_emotion_as_emotion():
    """The emotion column for CMU-MOSEI is derived from polarity, not annotated."""
    adapter = MOSEILabelAdapter(protocol="emotion6", emotion_label_path="does/not/exist.csv")
    frame = pd.DataFrame({
        "sample_id": ["a"], "sentiment_score": [1.0],
        "emotion": ["happy"], "feature_id": ["v$_$1"],
    })
    with pytest.raises(MOSEIEmotionLabelsUnavailable, match="NOT a substitute"):
        adapter.build(frame)


def test_mosei_leaves_arousal_unsupervised():
    adapter = MOSEILabelAdapter(protocol="sentiment")
    frame = pd.DataFrame({"sample_id": ["a", "b"], "sentiment_score": [1.5, -2.0]})
    labels = adapter.build(frame)
    assert np.isnan(labels.arousal).all(), "CMU-MOSEI does not annotate arousal"
    assert np.allclose(labels.valence, [0.5, -2.0 / 3.0])


# ======================================================================
# Loss
# ======================================================================

def test_focal_loss_reduces_to_weighted_cross_entropy_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(16, 6)
    targets = torch.randint(0, 6, (16,))
    weights = balanced_class_weights(targets, 6)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    reference = (
        torch.nn.functional.nll_loss(log_probabilities, targets, reduction="none")
        * weights[targets]
    ).mean()
    assert torch.allclose(focal_loss(logits, targets, gamma=0.0, alpha=weights), reference)


def test_loss_masks_absent_targets_rather_than_training_toward_them():
    torch.manual_seed(0)
    loss = HSENLoss(
        LossConfig(lambda_valence=1.0, lambda_arousal=1.0), num_classes=6,
    )
    outputs = {
        "logits": torch.randn(4, 6, requires_grad=True),
        "valence": torch.rand(4) * 2 - 1,
        "arousal": torch.rand(4) * 2 - 1,
    }
    targets = {
        "class_id": torch.tensor([0, 1, -1, -1]),
        "valence": torch.tensor([0.5, float("nan"), 0.1, float("nan")]),
        "arousal": torch.full((4,), float("nan")),
    }
    components = loss(outputs, targets)
    assert float(components["arousal"]) == 0.0, "a fully unsupervised head contributes nothing"
    assert float(components["valence"]) > 0.0
    assert torch.isfinite(components["total"])
    components["total"].backward()
    assert outputs["logits"].grad is not None


def test_class_weights_come_from_training_labels_only():
    train = torch.tensor([0, 0, 0, 0, 1, 2])
    weights = balanced_class_weights(train, 3)
    assert weights[0] < weights[1], "the frequent class must be down-weighted"
    assert pytest.approx(float(weights.mean()), abs=1e-5) == 1.0, "normalised to mean 1"
    # An absent class gets weight 1, not infinity.
    assert float(balanced_class_weights(torch.tensor([0, 0]), 3)[2]) > 0


def test_regression_loss_variants_all_run_and_mask():
    prediction = torch.tensor([0.2, -0.4, 0.9])
    target = torch.tensor([0.1, float("nan"), 0.8])
    for kind in ("mse", "huber", "ccc"):
        value = regression_loss(prediction, target, kind)
        assert torch.isfinite(value), kind


# ======================================================================
# Metrics
# ======================================================================

def test_per_class_metrics_are_in_declared_order_never_sorted():
    logits = torch.eye(6)
    targets = torch.arange(6)
    metrics = classification_metrics(logits, targets, CLASSES)
    assert list(metrics["per_class"]) == list(CLASSES)
    assert metrics["accuracy"] == 1.0
    assert len(metrics["confusion_matrix"]) == 6


def test_ccc_catches_a_model_that_predicts_the_mean():
    """A constant prediction has a respectable MAE and a CCC of zero."""
    target = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert concordance_correlation(np.full(5, target.mean()), target) == pytest.approx(0.0)
    assert concordance_correlation(target, target) == pytest.approx(1.0)


def test_unlabelled_rows_are_excluded_from_metrics():
    logits = torch.eye(6)
    targets = torch.tensor([0, 1, 2, 3, -1, -1])
    metrics = classification_metrics(logits, targets, CLASSES)
    assert metrics["support"] == 4


# ======================================================================
# Data
# ======================================================================

def test_collate_builds_both_masks_and_pads_to_the_batch_maximum():
    batch = [
        HSENSample("a", {"text": np.ones((5, 768), np.float32)}, {"text": True, "audio": False},
                   0, 0.5, 0.2),
        HSENSample("b", {"text": np.ones((9, 768), np.float32), "audio": np.ones((3, 768), np.float32)},
                   {"text": True, "audio": True}, 1, float("nan"), 0.1),
    ]
    collated = collate_hsen(batch, ("audio", "text"))
    assert collated["features"]["text"].shape == (2, 9, 768)
    assert collated["masks"]["text"][0].sum() == 5
    assert collated["available"]["audio"].tolist() == [False, True]
    assert not torch.isnan(collated["targets"]["valence"][0])
    assert torch.isnan(collated["targets"]["valence"][1])


def test_fractional_subset_is_stratified_seeded_and_keeps_every_class():
    frame = pd.DataFrame({"class_id": [0] * 100 + [1] * 40 + [2] * 3})
    selected = stratified_subset(frame, 0.25, seed=42)
    counts = frame.iloc[selected]["class_id"].value_counts().to_dict()
    assert set(counts) == {0, 1, 2}, "a rare class must survive the draw"
    assert counts[2] >= 1
    assert np.array_equal(selected, stratified_subset(frame, 0.25, seed=42))
    assert not np.array_equal(selected, stratified_subset(frame, 0.25, seed=7))


# ======================================================================
# Feature cache
# ======================================================================

def test_cache_refuses_a_different_encoder_configuration(tmp_path):
    store = SequenceFeatureStore.open(tmp_path, config_fingerprint({"a": 1}), "text", 4)
    store.append(["x"], [np.ones((3, 4), np.float32)])
    with pytest.raises(FeatureCacheError, match="different extractor"):
        SequenceFeatureStore.open(tmp_path, config_fingerprint({"a": 2}), "text", 4)


def test_cache_round_trips_ragged_sequences_in_request_order(tmp_path):
    store = SequenceFeatureStore.open(tmp_path, "fp", "audio", 4)
    store.append(["a", "b", "c"], [np.full((n, 4), n, np.float32) for n in (2, 5, 3)])
    loaded = store.load(["c", "a"])
    assert [item.shape[0] for item in loaded] == [3, 2]
    assert np.allclose(loaded[0], 3.0) and np.allclose(loaded[1], 2.0)
    assert store.verify()["healthy"]


def test_cache_refuses_non_finite_features(tmp_path):
    store = SequenceFeatureStore.open(tmp_path, "fp", "audio", 4)
    broken = np.ones((2, 4), np.float32)
    broken[0, 0] = np.nan
    with pytest.raises(FeatureCacheError, match="NaN"):
        store.append(["a"], [broken])


def test_missing_feature_is_an_error_not_a_zero(tmp_path):
    store = SequenceFeatureStore.open(tmp_path, "fp", "audio", 4)
    store.append(["a"], [np.ones((2, 4), np.float32)])
    with pytest.raises(FeatureCacheError, match="never substituted"):
        store.load(["a", "missing"])


def test_index_rebuilds_from_shards_after_an_interrupted_run(tmp_path):
    store = SequenceFeatureStore.open(tmp_path, "fp", "audio", 4)
    store.append(["a", "b"], [np.ones((2, 4), np.float32)] * 2)
    store.index["entries"] = {}          # simulate a lost index
    assert store.rebuild_index()["recovered"] == 2
    assert store.verify()["healthy"]


# ======================================================================
# Audit -- section 17
# ======================================================================

def _frames(train_ids, validation_ids, test_ids, speakers=None):
    def make(ids, speaker):
        return pd.DataFrame({
            "sample_id": ids,
            "class_id": [index % 6 for index in range(len(ids))],
            "speaker_id": [speaker] * len(ids),
            "valence_target": [0.0] * len(ids),
            "arousal_target": [0.0] * len(ids),
        })
    speakers = speakers or ("s1", "s2", "s3")
    return {
        "train": make(train_ids, speakers[0]),
        "validation": make(validation_ids, speakers[1]),
        "test": make(test_ids, speakers[2]),
    }


def test_audit_catches_a_sample_id_in_two_splits():
    frames = _frames(list("abcdef"), list("ghijkl"), list("mnopqf"))
    check = check_no_cross_split_ids(frames)
    assert not check.passed
    assert "train|test" in check.evidence


def test_audit_catches_speaker_overlap():
    frames = _frames(list("abcdef"), list("ghijkl"), list("mnopqr"),
                     speakers=("s1", "s1", "s3"))
    check = check_speaker_independence(frames)
    assert not check.passed


def test_audit_reports_an_unverifiable_speaker_check_as_unverifiable():
    """An empty speaker column must not be reported as a passed check."""
    frames = _frames(list("abcdef"), list("ghijkl"), list("mnopqr"))
    for frame in frames.values():
        frame["speaker_id"] = None
    check = check_speaker_independence(frames, required=False)
    assert check.evidence["verifiable"] is False
    assert "cannot be verified" in check.detail


def test_audit_catches_a_class_missing_from_a_split():
    frames = _frames(list("abcdef"), list("ghijkl"), list("mnopqr"))
    frames["validation"]["class_id"] = 0
    report = audit_manifests("test", frames, CLASSES, speaker_check_required=True)
    assert not report.passed
    failed = [check.name for check in report.checks if not check.passed]
    assert "every_class_present_in_every_split" in failed


def test_a_clean_manifest_set_passes_every_check():
    frames = _frames(list("abcdefghijkl"), list("mnopqrstuvwx"), list("ABCDEFGHIJKL"))
    report = audit_manifests("test", frames, CLASSES, speaker_check_required=True)
    assert report.passed, [c.name for c in report.checks if not c.passed]


def test_a_batch_with_no_sample_carrying_a_modality_still_runs():
    """The placeholder stream must have the modality's real width, not width 1.

    Only fires when a batch happens to contain no sample with a given modality,
    which on a mostly-complete corpus can be thousands of steps into a run.
    """
    batch = [
        HSENSample(str(index), {"text": np.ones((4, 768), np.float32)},
                   {"text": True, "video": False}, index % 6, 0.1, 0.2)
        for index in range(3)
    ]
    collated = collate_hsen(batch, ("video", "text"), {"video": 576, "text": 768})
    assert collated["features"]["video"].shape == (3, 1, 576)
    assert not collated["available"]["video"].any()

    model = build_hsen(HSENConfig(modalities=("video", "text"))).eval()
    with torch.no_grad():
        outputs = model(collated["features"], collated["masks"], collated["available"])
    assert outputs["logits"].shape == (3, 6)
    assert torch.isfinite(outputs["logits"]).all()


def test_shards_are_decompressed_at_most_once(tmp_path, monkeypatch):
    """The dataloader reads one sample at a time; shards must not re-inflate per read.

    Without the shard cache this is not a slow path, it is a different
    complexity class: every epoch decompresses each shard once per sample in it.
    Measured at 520 s per epoch on the IEMOCAP text cache against roughly 30 s.
    """
    store = SequenceFeatureStore.open(tmp_path, "fp", "text", 4, shard_size=4)
    store.append([f"s{i}" for i in range(4)], [np.ones((2, 4), np.float32)] * 4)

    real_load = np.load
    calls = []

    def counting_load(*args, **kwargs):
        calls.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    for identifier in ("s0", "s1", "s2", "s3", "s0"):
        store.load([identifier])
    assert len(calls) == 1, f"shard was decompressed {len(calls)} times, expected once"

    store.release()
    store.load(["s0"])
    assert len(calls) == 2, "release() must drop held shards"


# ======================================================================
# Concurrency and checkpoint ordering
# ======================================================================

def test_best_value_is_updated_before_last_is_written(tmp_path):
    """A resumed run must not restore a best that is one epoch stale.

    Writing last.pt before the comparison stamps it with the *previous* best,
    so a resume restores that and can then overwrite a genuinely better best.pt
    with a worse epoch -- and reports nothing, because "no improvement" is what
    an ordinary epoch looks like.
    """
    import torch.nn as nn

    from src.hsen.training.checkpoint import HSENCheckpointManager

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = HSENCheckpointManager(tmp_path, monitor="val_weighted_f1", mode="max")

    assert manager.update(model, optimizer, None, 1, {"val_weighted_f1": 0.30}, [])
    assert manager.update(model, optimizer, None, 2, {"val_weighted_f1": 0.50}, [])

    payload = manager.load(manager.last_path)
    assert payload["best_value"] == 0.50, "last.pt carries a stale best"
    assert payload["best_epoch"] == 2

    resumed = HSENCheckpointManager(tmp_path, monitor="val_weighted_f1", mode="max")
    resumed.resume(model, optimizer, None)
    assert resumed.best_value == 0.50
    # The epoch that follows must not displace a better best.
    assert not resumed.update(model, optimizer, None, 3, {"val_weighted_f1": 0.45}, [])
    assert resumed.load(resumed.best_path)["epoch"] == 2


def test_a_nan_metric_never_wins_the_checkpoint(tmp_path):
    import torch.nn as nn

    from src.hsen.training.checkpoint import HSENCheckpointManager

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = HSENCheckpointManager(tmp_path, monitor="val_weighted_f1", mode="max")
    assert not manager.update(model, optimizer, None, 1, {"val_weighted_f1": float("nan")}, [])
    assert not manager.best_path.exists()


def test_a_second_trainer_is_refused_the_same_run_directory(tmp_path):
    """The failure this guards against actually happened during this phase."""
    from src.hsen.training.runlock import RunDirectoryBusy, RunLock

    first = RunLock(tmp_path).acquire()
    try:
        with pytest.raises(RunDirectoryBusy, match="already claimed"):
            RunLock(tmp_path).acquire()
        # --force-lock is the escape hatch, for a holder confirmed gone.
        RunLock(tmp_path).acquire(force=True).release()
    finally:
        first.release()

    RunLock(tmp_path).acquire().release()   # free again once released


def test_saved_metrics_preserve_declared_class_order(tmp_path):
    """Serialising with sort_keys would silently alphabetise the per-class arrays.

    Class order is a scientific contract in this project (see src.common.labels).
    A sorted JSON diffs more cleanly and rotates every per-class array and every
    figure drawn from one -- which is exactly what happened before this test.
    """
    import json

    logits = torch.eye(6)
    targets = torch.arange(6)
    metrics = classification_metrics(logits, targets, CLASSES)

    path = tmp_path / "validation_metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert list(reloaded["per_class"]) == list(CLASSES)
    assert list(reloaded["per_class"]) != sorted(CLASSES), (
        "the declared order happens to be alphabetical, so this test proves nothing"
    )


def test_fusion_arm_names_survive_the_round_trip_through_a_directory_name(tmp_path):
    """'self_attention' must not become 'self+attention' and vanish from the table.

    Modality arms flatten '+' to '_' for the directory name and need translating
    back; fusion arm names contain literal underscores. Translating both the
    same way silently drops the self-attention row -- the one arm that isolates
    the cross-modal mechanism, and therefore the whole point of the study.
    """
    import json

    from src.hsen.ablations import arm_name, collect_runs

    def write(study: str, arm: str, score: float) -> None:
        directory = tmp_path / "iemocap_erc6" / arm_name(study, arm, "cpu25")
        directory.mkdir(parents=True)
        (directory / "run_summary.json").write_text(json.dumps({
            "primary_metric": "weighted_f1",
            "validation": {"weighted_f1": score, "macro_f1": score - 0.02,
                           "accuracy": score},
            "model_config": {"modalities": ["audio", "text"], "fusion": arm},
            "parameter_counts": {"total": 10, "fusion": 5},
            "best_epoch": 1, "epochs_run": 1, "training_seconds": 1.0,
        }), encoding="utf-8")

    write("fusion", "husformer", 0.50)
    write("fusion", "self_attention", 0.49)
    write("fusion", "concat", 0.48)
    fusion = collect_runs(tmp_path, "fusion", "cpu25")
    assert set(fusion["arm"]) == {"husformer", "self_attention", "concat"}

    write("modality", "audio+text", 0.50)
    modality = collect_runs(tmp_path, "modality", "cpu25")
    assert set(modality["arm"]) == {"audio+text"}
