"""End-to-end tests for the image experiment runner on tiny synthetic data.

Nothing here touches the real datasets, the real 25% subset, or the existing
``results/`` and ``checkpoints/`` baseline artefacts.  Every fixture is a
handful of 8x8 PNGs in ``tmp_path``.
"""

import json

import pandas as pd
import pytest
import torch
from PIL import Image

from src.common.experiment_layout import ExperimentLayout
from src.models.multimodal import ImageEmotionBaseline
from src.training.checkpoint import CheckpointManager
from src.training.experiment_summary import build_experiment_summary
from src.training.image_experiment import (
    DEBUG_OVERRIDES,
    CLASS_NAMES,
    ImageExperimentConfig,
    ImageExperimentRunner,
    debug_config,
    run_iteration,
)
from src.training.run_image_experiment import build_parser, config_from_args
from src.training.verify_artifacts import verify_iteration


SPLIT_CLASSES = {
    # Deliberately different distributions so a weight computed from the wrong
    # split is detectable.
    "train": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 6, 1, 1, 0, 2, 3, 4, 5, 6, 1, 0],
    "validation": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4],
    "test": [6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2],
}
DATASETS = ("AffectNet+", "FERPlus", "RAF-DB")


def _write_manifest(datasets_root, metadata_dir, split, labels):
    rows = []
    for index, label in enumerate(labels):
        dataset = DATASETS[index % len(DATASETS)]
        relative = f"{dataset}/{split}/{index}.png"
        path = datasets_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(label * 30 % 256, 40, 200 - label * 20)).save(path)
        rows.append({
            "sample_id": f"{split}-{index}",
            "dataset": dataset,
            "canonical_emotion_id": label,
            "has_image": True,
            "image_source": "file",
            "image_path": relative,
            "training_split": "train",
            "experiment_split": split,
        })
    metadata_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(metadata_dir / f"{split}.parquet", index=False)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    root = tmp_path / "experiments"
    layout = ExperimentLayout("image", 0.25, root)
    for split, labels in SPLIT_CLASSES.items():
        _write_manifest(datasets_root, layout.metadata_dir, split, labels)
    monkeypatch.setattr("src.data.image_dataset.DATASETS_DIR", datasets_root)
    return layout


def _config(layout, iteration="1", **overrides):
    defaults = dict(
        experiment_name=layout.name,
        experiment_root=str(layout.root),
        iteration=iteration,
        image_size=8,
        hidden_dim=8,
        batch_size=6,
        epochs=3,
        patience=2,
        min_epochs=2,
        device="cpu",
        seed=42,
    )
    defaults.update(overrides)
    return ImageExperimentConfig(**defaults)


# ============================================================
# Artefacts
# ============================================================

def test_iteration_produces_the_full_artefact_set(experiment):
    layout = experiment
    summary = run_iteration(_config(layout))

    results = layout.results_dir("1")
    for name in (
        "training_history.json", "validation_metrics.json", "test_metrics.json",
        "confusion_matrix.json", "run_summary.json", "class_weights.json",
    ):
        assert (results / name).exists(), name
    assert layout.config_path("1").exists()
    assert layout.log_path("1").exists()
    assert layout.best_checkpoint("1").exists()
    assert layout.last_checkpoint("1").exists()
    assert layout.experiment_summary_path.exists()

    assert summary["experiment"] == "image_25pct"
    assert summary["iteration"] == "1"
    assert summary["samples"] == {"train": 24, "validation": 12, "test": 12}
    assert summary["description"].startswith("Image-only modality-filtered")


def test_metrics_cover_every_required_quantity(experiment):
    layout = experiment
    run_iteration(_config(layout))
    metrics = json.loads((layout.results_dir("1") / "test_metrics.json").read_text())

    for key in (
        "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "per_class_precision", "per_class_recall", "per_class_f1",
        "support", "confusion_matrix", "samples",
    ):
        assert key in metrics, key
    assert len(metrics["confusion_matrix"]) == 7
    assert all(len(row) == 7 for row in metrics["confusion_matrix"])
    assert list(metrics["per_class"]) == list(CLASS_NAMES)
    assert sum(metrics["support"]) == 12

    confusion = json.loads((layout.results_dir("1") / "confusion_matrix.json").read_text())
    assert confusion["class_order"] == list(CLASS_NAMES)
    assert len(confusion["validation"]) == 7 and len(confusion["test"]) == 7


def test_model_emits_seven_logits(experiment):
    model = ImageEmotionBaseline(image_size=8, hidden_dim=8, num_classes=7)
    assert model(torch.randn(4, 3, 8, 8)).shape == (4, 7)


# ============================================================
# Class weights come from training data only
# ============================================================

def test_class_weights_are_computed_from_the_training_split_only(experiment):
    layout = experiment
    run_iteration(_config(layout))
    record = json.loads((layout.results_dir("1") / "class_weights.json").read_text())

    expected = {name: 0 for name in CLASS_NAMES}
    for label in SPLIT_CLASSES["train"]:
        expected[CLASS_NAMES[label]] += 1

    assert record["training_distribution"] == expected
    assert record["computed_from_split"] == "train"
    assert record["uses_validation_labels"] is False
    assert record["uses_test_labels"] is False
    assert record["class_order"] == list(CLASS_NAMES)
    assert len(record["weights"]) == 7

    # A weight vector derived from validation or test would differ.
    validation_distribution = {name: 0 for name in CLASS_NAMES}
    for label in SPLIT_CLASSES["validation"]:
        validation_distribution[CLASS_NAMES[label]] += 1
    assert record["training_distribution"] != validation_distribution


# ============================================================
# Model selection, checkpoints, early stopping
# ============================================================

def test_best_checkpoint_matches_the_highest_validation_macro_f1(experiment):
    layout = experiment
    summary = run_iteration(_config(layout, epochs=3))
    history = json.loads((layout.results_dir("1") / "training_history.json").read_text())

    best = max(history, key=lambda row: row["val_macro_f1"])
    assert summary["best_epoch"] == best["epoch"]
    assert summary["best_val_macro_f1"] == pytest.approx(best["val_macro_f1"])

    checkpoint = torch.load(layout.best_checkpoint("1"), map_location="cpu", weights_only=False)
    assert checkpoint["monitor"] == "macro_f1"
    assert checkpoint["epoch"] == best["epoch"]
    assert checkpoint["monitor_value"] == pytest.approx(best["val_macro_f1"])


def test_checkpoint_round_trips_and_carries_experiment_identity(experiment):
    layout = experiment
    summary = run_iteration(_config(layout))

    model = ImageEmotionBaseline(image_size=8, hidden_dim=8, num_classes=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = CheckpointManager.load(
        layout.best_checkpoint("1"), model=model, optimizer=optimizer, device="cpu",
    )

    assert "model_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint
    extra = checkpoint["extra"]
    assert extra["experiment"] == "image_25pct"
    assert extra["iteration"] == "1"
    assert len(extra["class_weights"]) == 7
    assert extra["config"]["seed"] == 42
    assert extra["class_mapping"]["neutral"] == 0
    assert summary["checkpoints"]["reload_verified"] is True

    logits = model(torch.randn(2, 3, 8, 8))
    assert logits.shape == (2, 7)


def test_early_stopping_state_is_recorded(experiment):
    layout = experiment
    summary = run_iteration(_config(layout, epochs=5, patience=1, min_epochs=1))
    stopping = summary["early_stopping"]

    assert stopping["monitor"] == "val_macro_f1"
    assert stopping["mode"] == "max"
    assert stopping["patience"] == 1
    assert summary["stop_reason"]
    assert summary["epochs_completed"] <= 5
    if stopping["stopped_early"]:
        assert summary["epochs_completed"] == stopping["stopped_epoch"]
        assert "early stopping" in summary["stop_reason"]


# ============================================================
# Test-set protection
# ============================================================

def test_the_test_partition_is_opened_only_after_model_selection(experiment):
    layout = experiment
    runner = ImageExperimentRunner(_config(layout))
    summary = runner.run()

    events = [(event["phase"], event["split"]) for event in runner.loader_events]
    assert ("setup", "test") not in events
    assert ("training", "test") not in events
    assert events[-1] == ("evaluation", "test")
    assert [event for event in events if event[1] == "test"] == [("evaluation", "test")]
    assert summary["test_used_for_model_selection"] is False
    assert summary["loader_events"][-1]["phase"] == "evaluation"


def test_missing_partition_fails_loudly(experiment):
    layout = experiment
    (layout.metadata_dir / "test.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Experiment metadata missing"):
        run_iteration(_config(layout, iteration="missing"))


def test_manifest_without_a_usable_image_modality_is_rejected(experiment, tmp_path):
    layout = experiment
    manifest = pd.read_parquet(layout.metadata_dir / "train.parquet")
    manifest["has_image"] = False
    manifest.to_parquet(layout.metadata_dir / "train.parquet", index=False)

    with pytest.raises(ValueError, match="No valid file-backed image records"):
        run_iteration(_config(layout, iteration="broken"))


# ============================================================
# Iterations
# ============================================================

def test_iterations_are_isolated_and_summarised(experiment):
    layout = experiment
    first = run_iteration(_config(layout, iteration="1", epochs=2))
    second = run_iteration(_config(layout, iteration="2", epochs=2, run_seed=43))

    assert layout.iteration_dir("1") != layout.iteration_dir("2")
    assert layout.log_path("1").exists() and layout.log_path("2").exists()

    first_log = layout.log_path("1").read_text(encoding="utf-8")
    second_log = layout.log_path("2").read_text(encoding="utf-8")
    assert "iteration 1" in first_log and "iteration 2" not in first_log
    assert "iteration 2" in second_log
    for fragment in ("Class weights (train only)", "Parameters", "Device", "Seeds", "Epoch 1/2"):
        assert fragment in first_log

    assert first["config"]["run_seed"] is None
    assert second["config"]["run_seed"] == 43

    summary = build_experiment_summary(layout)
    assert summary["iteration_count"] == 2
    assert summary["comparable_iterations"] == ["1", "2"]
    assert summary["best_iteration"]["iteration"] in {"1", "2"}
    assert "best_val_macro_f1" in summary["comparison"]
    assert summary["comparison"]["best_val_macro_f1"]["n"] == 2
    assert "stdev" in summary["comparison"]["test_macro_f1"]
    assert summary["sampling_summary"] is None  # no sampling summary in this fixture


def test_reruns_append_to_the_iteration_log_instead_of_truncating(experiment):
    layout = experiment
    run_iteration(_config(layout, iteration="1", epochs=1, min_epochs=1))
    first_length = len(layout.log_path("1").read_text(encoding="utf-8"))
    run_iteration(_config(layout, iteration="1", epochs=1, min_epochs=1))
    assert len(layout.log_path("1").read_text(encoding="utf-8")) > first_length


# ============================================================
# Configuration surface
# ============================================================

def test_debug_config_is_small_and_bounded():
    config = debug_config(ImageExperimentConfig())
    assert config.debug is True
    for field, value in DEBUG_OVERRIDES.items():
        assert getattr(config, field) == value
    assert config.max_train and config.max_train <= 512


def test_cli_arguments_override_debug_defaults():
    args = build_parser().parse_args(
        ["--experiment", "image_25pct", "--debug", "--epochs", "4", "--device", "cpu"]
    )
    config = config_from_args(args)
    assert config.debug is True
    assert config.iteration == "debug"
    assert config.epochs == 4  # explicit CLI beats the debug override
    assert config.batch_size == DEBUG_OVERRIDES["batch_size"]
    assert config.device == "cpu"


def test_artifact_verification_passes_for_a_completed_iteration(experiment):
    layout = experiment
    run_iteration(_config(layout, iteration="1", epochs=2))

    report = verify_iteration(layout, "1")
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    names = {check["check"] for check in report["checks"]}
    assert "best_checkpoint_is_the_highest_validation_macro_f1" in names
    assert "class_weights_derived_from_train_only" in names
    assert "test_partition_opened_only_after_training" in names


def test_artifact_verification_detects_a_missing_checkpoint(experiment):
    layout = experiment
    run_iteration(_config(layout, iteration="1", epochs=1, min_epochs=1))
    layout.best_checkpoint("1").unlink()

    report = verify_iteration(layout, "1")
    assert not report["passed"]
    failed = {check["check"] for check in report["checks"] if not check["passed"]}
    assert "checkpoints/best.pt" in failed


def test_cli_defaults_to_iteration_one():
    config = config_from_args(build_parser().parse_args(["--experiment", "image_25pct"]))
    assert config.iteration == "1"
    assert config.debug is False
    assert config.epochs == 5
    assert config.seed == 42
    assert config.effective_run_seed == 42
