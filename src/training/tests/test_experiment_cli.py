"""Tests for the shared modality CLI surface and the experiment summary.

The five baselines are only comparable if their command lines mean the same
thing, so these tests pin the flag set, the precedence rules, and the summary
statistics that the mentor-facing comparison table is built from.
"""

import json
import statistics

import pytest

from src.common.experiment_layout import ExperimentLayout
from src.training.base_experiment import DEBUG_OVERRIDES
from src.training.experiment_cli import COMMON_ARGUMENTS, ExperimentSpec
from src.training.experiment_summary import build_experiment_summary, write_experiment_summary
from src.training.modalities import EXPERIMENT_SPECS, build_model_from_record


MODALITIES = ("image", "audio", "text", "video", "physiology")


# ============================================================
# Flag surface
# ============================================================

@pytest.mark.parametrize("modality", MODALITIES)
def test_every_modality_exposes_the_shared_protocol_flags(modality):
    parser = EXPERIMENT_SPECS[modality].build_parser()
    options = {action.option_strings[0] for action in parser._actions if action.option_strings}
    for flag, _ in COMMON_ARGUMENTS:
        assert flag in options, f"{modality} is missing {flag}"
    for flag in ("--experiment", "--iteration", "--debug", "--summarize", "--run-seed"):
        assert flag in options, f"{modality} is missing {flag}"


@pytest.mark.parametrize("modality,experiment", [
    ("image", "image_25pct"),
    ("audio", "audio_25pct"),
    ("text", "text_full"),
    ("video", "video_25pct"),
    ("physiology", "physiology_full"),
])
def test_defaults_follow_the_five_epoch_baseline_protocol(modality, experiment):
    spec = EXPERIMENT_SPECS[modality]
    config = spec.config_from_args(spec.build_parser().parse_args([]))

    assert config.experiment_name == experiment
    assert config.iteration == "1"
    assert config.debug is False
    assert config.epochs == 5
    assert config.seed == 42
    assert config.effective_run_seed == 42
    assert config.min_epochs == 2
    assert config.patience == 2
    assert config.min_delta == pytest.approx(1e-4)
    assert config.num_workers == 0
    assert ExperimentLayout.from_name(config.experiment_name).modality == modality


@pytest.mark.parametrize("modality", MODALITIES)
def test_second_iteration_shares_the_split_and_changes_only_the_run_seed(modality):
    spec = EXPERIMENT_SPECS[modality]
    first = spec.config_from_args(spec.build_parser().parse_args(["--iteration", "1"]))
    second = spec.config_from_args(
        spec.build_parser().parse_args(["--iteration", "2", "--run-seed", "43"])
    )
    assert first.effective_run_seed == 42
    assert second.effective_run_seed == 43
    # The sampling seed -- and therefore the data split -- is untouched.
    assert first.seed == second.seed == 42
    assert first.experiment_name == second.experiment_name


@pytest.mark.parametrize("modality", MODALITIES)
def test_explicit_flags_beat_debug_overrides(modality):
    spec = EXPERIMENT_SPECS[modality]
    config = spec.config_from_args(
        spec.build_parser().parse_args(["--debug", "--epochs", "4", "--device", "cpu"])
    )
    assert config.debug is True
    assert config.iteration == "debug"
    assert config.epochs == 4
    assert config.batch_size == DEBUG_OVERRIDES["batch_size"]
    assert config.max_train == DEBUG_OVERRIDES["max_train"]
    assert config.device == "cpu"


def test_physiology_declares_its_own_label_space_on_the_command_line():
    spec = EXPERIMENT_SPECS["physiology"]
    config = spec.config_from_args(spec.build_parser().parse_args([]))
    assert config.task == "wesad_state_3class"
    assert config.num_classes == 3
    assert config.label_space.limitation


def test_a_spec_cannot_declare_a_flag_the_config_has_no_field_for():
    spec = EXPERIMENT_SPECS["image"]
    with pytest.raises(ValueError, match="no field"):
        ExperimentSpec(
            modality="image",
            default_experiment="image_25pct",
            description="x",
            config_class=spec.config_class,
            run_iteration=spec.run_iteration,
            extra_arguments=(("--nonexistent-knob", {"type": int}),),
        )


def test_num_classes_must_agree_with_the_declared_label_space():
    spec = EXPERIMENT_SPECS["image"]
    with pytest.raises(ValueError, match="contradicts label space"):
        spec.config_class(num_classes=5)


# ============================================================
# Model reconstruction from a recorded summary
# ============================================================

@pytest.mark.parametrize("record,expected", [
    ({"class": "ImageEmotionBaseline", "image_size": 16, "hidden_dim": 8, "num_classes": 7}, 7),
    ({"class": "AudioEmotionBaseline", "n_mels": 8, "hidden_dim": 8, "num_classes": 7}, 7),
    ({"class": "TextEmotionBaseline", "vocab_size": 64, "embedding_dim": 8,
      "hidden_dim": 8, "num_classes": 7}, 7),
    ({"class": "VideoEmotionBaseline", "frame_size": 8, "hidden_dim": 8,
      "num_classes": 7, "temporal": "gru"}, 7),
    ({"class": "PhysiologyStateBaseline", "feature_dim": 10, "hidden_dim": 8,
      "num_classes": 3}, 3),
])
def test_architectures_rebuild_from_their_recorded_geometry(record, expected):
    model = build_model_from_record({"model": record, "config": {}})
    assert model.classifier.out_features == expected


def test_an_unknown_architecture_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="No model builder"):
        build_model_from_record({"model": {"class": "SomethingElse"}, "config": {}})


def test_physiology_reconstruction_requires_the_recorded_feature_dimension():
    with pytest.raises(ValueError, match="feature_dim"):
        build_model_from_record({"model": {"class": "PhysiologyStateBaseline"}, "config": {}})


# ============================================================
# Experiment summary statistics
# ============================================================

def _fake_iteration(layout: ExperimentLayout, label: str, seed: int, values: dict) -> None:
    layout.prepare_iteration(label)
    (layout.results_dir(label) / "run_summary.json").write_text(json.dumps({
        "experiment": layout.name,
        "iteration": label,
        "modality": layout.modality,
        "task": "emotion_7class",
        "class_order": ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"],
        "debug": False,
        "config": {"seed": seed, "run_seed": seed, "num_classes": 7},
        "epochs_configured": 5,
        "epochs_completed": 5,
        "best_epoch": values["best_epoch"],
        "best_val_macro_f1": values["best_val_macro_f1"],
        "samples": {"train": 80, "validation": 10, "test": 10},
        "total_seconds": values["total_seconds"],
        "validation_metrics": {"macro_f1": values["best_val_macro_f1"]},
        "test_metrics": {
            "loss": values["test_loss"],
            "accuracy": values["test_accuracy"],
            "macro_f1": values["test_macro_f1"],
            "weighted_f1": values["test_weighted_f1"],
        },
    }), encoding="utf-8")


ITERATION_ONE = {
    "best_val_macro_f1": 0.367793, "test_accuracy": 0.518689, "test_macro_f1": 0.378946,
    "test_weighted_f1": 0.541005, "test_loss": 1.5678, "best_epoch": 5, "total_seconds": 354.7,
}
ITERATION_TWO = {
    "best_val_macro_f1": 0.365269, "test_accuracy": 0.492903, "test_macro_f1": 0.356118,
    "test_weighted_f1": 0.520093, "test_loss": 1.5945, "best_epoch": 4, "total_seconds": 317.2,
}


def test_summary_statistics_match_the_textbook_definitions(tmp_path):
    layout = ExperimentLayout("image", 0.25, tmp_path / "experiments")
    _fake_iteration(layout, "1", 42, ITERATION_ONE)
    _fake_iteration(layout, "2", 43, ITERATION_TWO)

    summary = build_experiment_summary(layout)
    for metric in ("best_val_macro_f1", "test_accuracy", "test_macro_f1",
                   "test_weighted_f1", "test_loss"):
        values = [ITERATION_ONE[metric], ITERATION_TWO[metric]]
        block = summary["comparison"][metric]
        assert block["n"] == 2
        assert block["values"] == values
        assert block["mean"] == pytest.approx(statistics.fmean(values))
        assert block["stdev"] == pytest.approx(statistics.stdev(values))
        assert block["min"] == pytest.approx(min(values))
        assert block["max"] == pytest.approx(max(values))


def test_summary_records_what_a_comparison_table_needs(tmp_path):
    layout = ExperimentLayout("image", 0.25, tmp_path / "experiments")
    _fake_iteration(layout, "1", 42, ITERATION_ONE)
    _fake_iteration(layout, "2", 43, ITERATION_TWO)

    summary = build_experiment_summary(layout)
    assert summary["modality"] == "image"
    assert summary["fraction"] == 0.25
    assert summary["fraction_label"] == "25pct"
    assert summary["task"] == "emotion_7class"
    assert summary["class_names"][0] == "neutral"
    assert summary["iteration_seeds"] == {"1": 42, "2": 43}
    assert summary["best_epochs"] == {"1": 5, "2": 4}
    assert summary["durations_seconds"] == {"1": 354.7, "2": 317.2}
    assert summary["split_counts"] == {"train": 80, "validation": 10, "test": 10}
    # The higher validation macro-F1 wins; test scores never decide.
    assert summary["best_iteration"]["iteration"] == "1"


def test_debug_iterations_are_listed_but_excluded_from_the_statistics(tmp_path):
    layout = ExperimentLayout("image", 0.25, tmp_path / "experiments")
    _fake_iteration(layout, "1", 42, ITERATION_ONE)
    _fake_iteration(layout, "2", 43, ITERATION_TWO)
    _fake_iteration(layout, "debug", 42, ITERATION_ONE)
    summary_path = layout.results_dir("debug") / "run_summary.json"
    payload = json.loads(summary_path.read_text())
    payload["debug"] = True
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = build_experiment_summary(layout)
    assert {record["iteration"] for record in summary["iterations"]} == {"1", "2", "debug"}
    assert summary["comparable_iterations"] == ["1", "2"]
    assert summary["comparison"]["test_macro_f1"]["n"] == 2


def test_write_experiment_summary_round_trips(tmp_path):
    layout = ExperimentLayout("audio", 0.25, tmp_path / "experiments")
    _fake_iteration(layout, "1", 42, ITERATION_ONE)
    path = write_experiment_summary(layout)
    assert path == layout.experiment_summary_path
    assert json.loads(path.read_text())["experiment"] == "audio_25pct"
