"""End-to-end tests for the audio, text, video, and physiology baselines.

Every fixture is a handful of quarter-second tones, one-line transcripts,
sixteen-pixel clips, or synthetic window vectors in ``tmp_path``.  Nothing here
touches the real datasets, the real sampled subsets, or the completed image
baseline under ``experiments/image/25pct``.

The assertions are about the *protocol* rather than about accuracy: artefact
completeness, canonical class ordering, train-only class weights, iteration
isolation, and the guarantee that the test partition stays closed until
training has finished.
"""

import json

import pandas as pd
import pytest
import torch

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import EMOTION_7CLASS, WESAD_STATE_3CLASS
from src.training.audio_experiment import AudioExperimentConfig
from src.training.audio_experiment import run_iteration as run_audio
from src.training.experiment_summary import build_experiment_summary
from src.training.physiology_experiment import PhysiologyExperimentConfig
from src.training.physiology_experiment import run_iteration as run_physiology
from src.training.tests.synthetic import (
    SPLIT_CLASSES,
    write_audio_experiment,
    write_leaky_physiology_experiment,
    write_physiology_experiment,
    write_text_experiment,
    write_video_experiment,
)
from src.training.text_experiment import TextExperimentConfig
from src.training.text_experiment import run_iteration as run_text
from src.training.verify_artifacts import verify_experiment_summary, verify_iteration
from src.training.video_experiment import VideoExperimentConfig
from src.training.video_experiment import run_iteration as run_video


RESULT_FILES = (
    "training_history.json", "validation_metrics.json", "test_metrics.json",
    "confusion_matrix.json", "run_summary.json", "class_weights.json",
)
EXPECTED_SAMPLES = {split: len(labels) for split, labels in SPLIT_CLASSES.items()}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def audio(tmp_path, monkeypatch):
    layout = ExperimentLayout("audio", 0.25, tmp_path / "experiments")
    datasets_root = tmp_path / "datasets"
    write_audio_experiment(datasets_root, layout.metadata_dir)
    monkeypatch.setattr("src.data.audio_dataset.DATASETS_DIR", datasets_root)
    return layout


@pytest.fixture
def text(tmp_path, monkeypatch):
    layout = ExperimentLayout("text", 1.0, tmp_path / "experiments")
    datasets_root = tmp_path / "datasets"
    write_text_experiment(datasets_root, layout.metadata_dir)
    monkeypatch.setattr("src.data.text_dataset.DATASETS_DIR", datasets_root)
    return layout


@pytest.fixture
def video(tmp_path, monkeypatch):
    layout = ExperimentLayout("video", 0.25, tmp_path / "experiments")
    datasets_root = tmp_path / "datasets"
    if not write_video_experiment(datasets_root, layout.metadata_dir):
        pytest.skip("This OpenCV build cannot write a test clip.")
    monkeypatch.setattr("src.data.video_dataset.DATASETS_DIR", datasets_root)
    return layout


@pytest.fixture
def physiology(tmp_path):
    layout = ExperimentLayout("physiology", 1.0, tmp_path / "experiments")
    write_physiology_experiment(layout.metadata_dir)
    return layout


def _audio_config(layout, **overrides):
    defaults = dict(
        experiment_name=layout.name, experiment_root=str(layout.root), iteration="1",
        sample_rate=8_000, n_fft=128, hop_length=64, n_mels=8, max_seconds=0.25,
        hidden_dim=8, batch_size=6, epochs=2, min_epochs=1, patience=2, device="cpu", seed=42,
    )
    defaults.update(overrides)
    return AudioExperimentConfig(**defaults)


def _text_config(layout, **overrides):
    defaults = dict(
        experiment_name=layout.name, experiment_root=str(layout.root), iteration="1",
        vocab_size=256, max_tokens=12, embedding_dim=8, hidden_dim=8,
        batch_size=6, epochs=2, min_epochs=1, patience=2, device="cpu", seed=42,
    )
    defaults.update(overrides)
    return TextExperimentConfig(**defaults)


def _video_config(layout, **overrides):
    defaults = dict(
        experiment_name=layout.name, experiment_root=str(layout.root), iteration="1",
        num_frames=3, frame_size=8, hidden_dim=8,
        batch_size=4, epochs=2, min_epochs=1, patience=2, device="cpu", seed=42,
    )
    defaults.update(overrides)
    return VideoExperimentConfig(**defaults)


def _physiology_config(layout, **overrides):
    defaults = dict(
        experiment_name=layout.name, experiment_root=str(layout.root), iteration="1",
        hidden_dim=8, batch_size=8, epochs=2, min_epochs=1, patience=2, device="cpu", seed=42,
    )
    defaults.update(overrides)
    return PhysiologyExperimentConfig(**defaults)


MODALITIES = ("audio", "text", "video", "physiology")
RUNNERS = {
    "audio": (run_audio, _audio_config),
    "text": (run_text, _text_config),
    "video": (run_video, _video_config),
    "physiology": (run_physiology, _physiology_config),
}


@pytest.fixture
def experiment(request):
    """Parameterised access to one modality's layout, runner, and config."""
    layout = request.getfixturevalue(request.param)
    runner, builder = RUNNERS[request.param]
    return request.param, layout, runner, builder


# ============================================================
# Artefacts
# ============================================================

@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_every_modality_writes_the_full_artefact_set(experiment):
    modality, layout, run, config = experiment
    summary = run(config(layout))

    results = layout.results_dir("1")
    for name in RESULT_FILES:
        assert (results / name).exists(), f"{modality}: {name}"
    assert layout.config_path("1").exists()
    assert layout.log_path("1").exists()
    assert layout.best_checkpoint("1").exists()
    assert layout.last_checkpoint("1").exists()
    assert layout.experiment_summary_path.exists()

    assert summary["experiment"] == layout.name
    assert summary["modality"] == modality
    assert summary["iteration"] == "1"


@pytest.mark.parametrize("experiment", ["audio", "text", "video"], indirect=True)
def test_emotion_modalities_use_the_canonical_seven_class_order(experiment):
    _, layout, run, config = experiment
    summary = run(config(layout))

    assert summary["task"] == "emotion_7class"
    assert summary["class_order"] == list(EMOTION_7CLASS.classes)
    assert summary["class_order"] != sorted(summary["class_order"])
    assert summary["samples"] == EXPECTED_SAMPLES

    for name in ("validation_metrics", "test_metrics"):
        metrics = summary[name]
        assert len(metrics["confusion_matrix"]) == 7
        assert all(len(row) == 7 for row in metrics["confusion_matrix"])
        assert list(metrics["per_class"]) == list(EMOTION_7CLASS.classes)
        assert len(metrics["per_class_f1"]) == 7

    confusion = json.loads((layout.results_dir("1") / "confusion_matrix.json").read_text())
    assert confusion["class_order"] == list(EMOTION_7CLASS.classes)


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_metrics_cover_every_required_quantity(experiment):
    _, layout, run, config = experiment
    summary = run(config(layout))
    for name in ("validation_metrics", "test_metrics"):
        metrics = summary[name]
        for key in (
            "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "per_class_precision", "per_class_recall", "per_class_f1", "support",
            "confusion_matrix", "samples",
        ):
            assert key in metrics, f"{name} missing {key}"


# ============================================================
# Protocol guarantees
# ============================================================

@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_class_weights_come_from_the_training_partition_only(experiment):
    modality, layout, run, config = experiment
    run(config(layout))

    record = json.loads((layout.results_dir("1") / "class_weights.json").read_text())
    assert record["computed_from_split"] == "train"
    assert record["uses_validation_labels"] is False
    assert record["uses_test_labels"] is False
    assert record["computed_from"].endswith("train.parquet")

    label_column = record["label_column"]
    train = pd.read_parquet(layout.metadata_dir / "train.parquet")
    expected = train[label_column].astype(int).value_counts().to_dict()
    for index, name in enumerate(record["class_order"]):
        assert record["training_distribution"][name] == expected.get(index, 0)
    assert len(record["weights"]) == len(record["class_order"])


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_the_test_partition_is_opened_only_after_training(experiment):
    _, layout, run, config = experiment
    summary = run(config(layout))

    events = summary["loader_events"]
    assert [event["split"] for event in events] == ["train", "validation", "test"]
    assert all(event["phase"] != "training" for event in events)
    assert next(event for event in events if event["split"] == "test")["phase"] == "evaluation"
    assert summary["test_used_for_model_selection"] is False


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_best_checkpoint_matches_the_highest_validation_macro_f1(experiment):
    _, layout, run, config = experiment
    summary = run(config(layout, epochs=3))

    history = json.loads((layout.results_dir("1") / "training_history.json").read_text())
    best = max(history, key=lambda row: row["val_macro_f1"])
    assert summary["best_epoch"] == best["epoch"]
    assert summary["best_val_macro_f1"] == pytest.approx(best["val_macro_f1"])

    checkpoint = torch.load(layout.best_checkpoint("1"), map_location="cpu", weights_only=False)
    assert checkpoint["monitor"] == "macro_f1"
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint["extra"]["experiment"] == layout.name
    assert checkpoint["extra"]["class_order"] == summary["class_order"]


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_checkpoint_round_trips_into_the_recorded_architecture(experiment):
    _, layout, run, config = experiment
    summary = run(config(layout))
    assert summary["checkpoints"]["reload_verified"] is True

    report = verify_iteration(layout, "1")
    assert report["passed"], [check for check in report["checks"] if not check["passed"]]
    names = {check["check"] for check in report["checks"]}
    assert "reloaded_model_emits_num_classes_logits" in names
    assert "confusion_matrices_are_square_in_the_declared_class_space" in names


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_two_iterations_stay_completely_isolated(experiment):
    _, layout, run, config = experiment
    first = run(config(layout, iteration="1", seed=42))
    second = run(config(layout, iteration="2", seed=42, run_seed=43))

    assert layout.iteration_dir("1") != layout.iteration_dir("2")
    assert first["log_file"] != second["log_file"]
    assert first["checkpoints"]["best"] != second["checkpoints"]["best"]
    assert first["config"]["run_seed"] is None
    assert second["config"]["run_seed"] == 43

    # Iteration 1's artefacts must survive iteration 2 untouched.
    reloaded = json.loads((layout.results_dir("1") / "run_summary.json").read_text())
    assert reloaded["best_val_macro_f1"] == first["best_val_macro_f1"]
    assert "iteration_2" not in layout.log_path("1").read_text(encoding="utf-8")
    assert "iteration_1" not in layout.log_path("2").read_text(encoding="utf-8")

    # The data split is shared; only the run seed differs.
    assert first["samples"] == second["samples"]


@pytest.mark.parametrize("experiment", MODALITIES, indirect=True)
def test_experiment_summary_reports_both_iterations(experiment):
    _, layout, run, config = experiment
    run(config(layout, iteration="1", seed=42))
    run(config(layout, iteration="2", seed=42, run_seed=43))

    summary = build_experiment_summary(layout)
    assert sorted(record["iteration"] for record in summary["iterations"]) == ["1", "2"]
    assert summary["iteration_seeds"] == {"1": 42, "2": 43}
    for metric in ("best_val_macro_f1", "test_accuracy", "test_macro_f1",
                   "test_weighted_f1", "test_loss"):
        block = summary["comparison"][metric]
        assert block["n"] == 2
        assert block["min"] <= block["mean"] <= block["max"]
    assert summary["best_iteration"]["iteration"] in {"1", "2"}
    assert verify_experiment_summary(layout)["passed"]


# ============================================================
# Modality-specific behaviour
# ============================================================

def test_audio_batches_carry_fixed_width_features_and_real_lengths(audio):
    from src.training.audio_experiment import AudioExperimentRunner

    runner = AudioExperimentRunner(_audio_config(audio))
    loader = runner.build_loader("train", shuffle=False, max_samples=None)
    batch = next(iter(loader))
    assert batch["audio"].shape[0] == batch["label"].shape[0]
    assert batch["audio"].shape[2] == 8                      # n_mels
    assert batch["audio_lengths"].dtype == torch.long
    assert bool((batch["audio_lengths"] > 0).all())
    assert bool((batch["audio_lengths"] <= batch["audio"].shape[1]).all())


def test_text_reads_both_metadata_and_file_backed_transcripts(text):
    from src.data.text_dataset import EmotionTextDataset

    dataset = EmotionTextDataset(
        text.metadata_dir / "train.parquet",
        columns=list(EmotionTextDataset.MINIMAL_COLUMNS),
        datasets_root=text.metadata_dir.parents[3] / "datasets",
    )
    sources = set(dataset.manifest["text_source"])
    assert sources == {"metadata", "file"}
    for index in range(len(dataset)):
        row = dataset.row(index)
        assert dataset.read_text(row).strip(), row["sample_id"]


def test_video_decodes_the_requested_number_of_frames(video):
    from src.training.video_experiment import VideoExperimentRunner

    runner = VideoExperimentRunner(_video_config(video))
    loader = runner.build_loader("validation", shuffle=False, max_samples=None)
    batch = next(iter(loader))
    assert batch["video"].shape[1:] == (3, 3, 8, 8)           # frames, channels, H, W
    assert bool((batch["video_lengths"] > 0).all())
    assert bool(torch.isfinite(batch["video"]).all())


def test_physiology_declares_its_label_space_and_records_the_limitation(physiology):
    summary = run_physiology(_physiology_config(physiology))

    assert summary["task"] == "wesad_state_3class"
    assert summary["class_order"] == list(WESAD_STATE_3CLASS.classes)
    assert summary["model"]["num_classes"] == 3
    assert "physiological_state" in summary["task_limitation"]
    assert len(summary["test_metrics"]["confusion_matrix"]) == 3
    assert summary["subject_disjoint_splits"] is True
    assert set(summary["subjects"]["train"]) == {"S2", "S3", "S4"}
    assert set(summary["subjects"]["test"]) == {"S6"}


def test_physiology_standardisation_is_fitted_on_training_windows_only(physiology):
    run_physiology(_physiology_config(physiology))

    record = json.loads(
        (physiology.results_dir("1") / "feature_normalization.json").read_text()
    )
    assert record["computed_from_split"] == "train"
    assert record["uses_validation_windows"] is False
    assert record["uses_test_windows"] is False

    train = pd.read_parquet(physiology.metadata_dir / "train.parquet")
    features = torch.tensor([list(value) for value in train["features"]])
    assert record["mean"] == pytest.approx(features.mean(dim=0).tolist(), abs=1e-5)
    assert record["std"] == pytest.approx(features.std(dim=0, unbiased=False).tolist(), abs=1e-5)


def test_physiology_refuses_to_train_on_a_subject_that_spans_partitions(tmp_path):
    layout = ExperimentLayout("physiology", 1.0, tmp_path / "experiments")
    write_leaky_physiology_experiment(layout.metadata_dir)
    with pytest.raises(ValueError, match="both the training and validation"):
        run_physiology(_physiology_config(layout))
