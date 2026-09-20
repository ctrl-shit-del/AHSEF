import pytest

from src.common.experiment_layout import (
    ExperimentLayout,
    fraction_to_label,
    label_to_fraction,
)


def test_fraction_label_round_trip():
    assert fraction_to_label(0.25) == "25pct"
    assert fraction_to_label(0.125) == "12_5pct"
    assert label_to_fraction("25pct") == pytest.approx(0.25)
    assert label_to_fraction("12_5pct") == pytest.approx(0.125)


def test_layout_paths_are_iteration_scoped(tmp_path):
    layout = ExperimentLayout("image", 0.25, tmp_path / "experiments")
    assert layout.name == "image_25pct"
    assert layout.base == tmp_path / "experiments" / "image" / "25pct"
    assert layout.split_path("train").name == "train.parquet"
    assert layout.metadata_dir == layout.base / "metadata"

    for iteration in ("1", "2"):
        assert layout.iteration_dir(iteration) == layout.base / f"iteration_{iteration}"

    # Iteration artefacts must never collide across iterations.
    assert layout.log_path("1") != layout.log_path("2")
    assert layout.best_checkpoint("1") != layout.best_checkpoint("2")
    assert layout.results_dir("1") != layout.results_dir("2")


def test_layout_from_name_and_existing_iterations(tmp_path):
    layout = ExperimentLayout.from_name("image_25pct", tmp_path / "experiments")
    assert layout.modality == "image"
    assert layout.fraction == pytest.approx(0.25)

    assert layout.existing_iterations() == []
    for iteration in ("2", "1", "debug"):
        layout.prepare_iteration(iteration)
    assert layout.existing_iterations() == ["1", "2", "debug"]


def test_layout_rejects_invalid_input(tmp_path):
    with pytest.raises(ValueError):
        ExperimentLayout("image", 0.0, tmp_path)
    with pytest.raises(ValueError):
        ExperimentLayout.from_name("not-an-experiment", tmp_path)
    with pytest.raises(ValueError):
        ExperimentLayout("image", 0.25, tmp_path).iteration_dir("../escape")
    with pytest.raises(ValueError):
        ExperimentLayout("image", 0.25, tmp_path).split_path("holdout")
