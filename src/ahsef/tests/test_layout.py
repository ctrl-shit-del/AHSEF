"""AHSEF run layout, and the guarantee that it never addresses a baseline directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ahsef.layout import DEFAULT_AHSEF_ROOT, AhsefLayout
from src.ahsef.registry import DEFAULT_BASELINES, baseline, ordered_modalities


def test_every_path_lives_under_the_run_directory(tmp_path):
    layout = AhsefLayout(run="stage1", root=tmp_path)
    paths = [
        layout.provenance_path,
        layout.prediction_path("audio", "test"),
        layout.calibration_path("audio"),
        layout.alignment_matrix_path,
        layout.fusion_metrics_path(("audio", "text"), "test"),
        layout.routing_log_path(),
        layout.report_path("summary.json"),
    ]
    for path in paths:
        assert Path(tmp_path / "stage1") in path.parents or path.parent == tmp_path / "stage1"


def test_the_default_root_is_inside_experiments_ahsef():
    assert DEFAULT_AHSEF_ROOT == Path("experiments") / "ahsef"
    # A baseline experiment directory is a sibling, never a parent.
    layout = AhsefLayout(run="stage1")
    assert "ahsef" in layout.base.parts


def test_invalid_run_names_are_refused(tmp_path):
    for name in ("", "../escape", "a/b", "-leading"):
        with pytest.raises(ValueError, match="Invalid AHSEF run name"):
            AhsefLayout(run=name, root=tmp_path)


def test_split_names_are_validated(tmp_path):
    layout = AhsefLayout(run="stage1", root=tmp_path)
    with pytest.raises(ValueError, match="split must be one of"):
        layout.prediction_path("audio", "holdout")


def test_a_fusion_variant_gets_its_own_directory(tmp_path):
    """Two fusion rules over one pair are different experiments."""
    layout = AhsefLayout(run="stage1", root=tmp_path)
    default = layout.fusion_pair_dir(("audio", "text"))
    geometric = layout.fusion_pair_dir(("audio", "text"), "log_opinion_pool")
    assert default.name == "audio+text"
    assert geometric.name == "audio+text__log_opinion_pool"
    assert default != geometric
    assert layout.delta_uncertainty_path(("audio", "text"), "test", "log_opinion_pool") != \
        layout.delta_uncertainty_path(("audio", "text"), "test")


def test_an_invalid_variant_name_is_refused(tmp_path):
    layout = AhsefLayout(run="stage1", root=tmp_path)
    with pytest.raises(ValueError, match="Invalid fusion variant"):
        layout.fusion_pair_dir(("audio", "text"), "../escape")


def test_pair_label_is_stable_and_rejects_duplicates():
    assert AhsefLayout.pair_label(["audio", "text"]) == "audio+text"
    with pytest.raises(ValueError, match="Duplicate modality"):
        AhsefLayout.pair_label(["audio", "audio"])
    with pytest.raises(ValueError, match="at least one modality"):
        AhsefLayout.pair_label([])


def test_prepare_creates_every_subdirectory(tmp_path):
    layout = AhsefLayout(run="stage1", root=tmp_path)
    layout.prepare()
    for directory in (layout.predictions_dir, layout.alignment_dir,
                      layout.calibration_dir, layout.fusion_dir, layout.reports_dir):
        assert directory.is_dir()


def test_existing_predictions_are_discovered(tmp_path):
    layout = AhsefLayout(run="stage1", root=tmp_path)
    layout.prepare()
    layout.prediction_path("audio", "test").write_bytes(b"")
    layout.prediction_path("text", "validation").write_bytes(b"")
    assert set(layout.existing_predictions()) == {("audio", "test"), ("text", "validation")}


# --------------------------------------------------------------- registry

def test_registry_covers_the_five_trained_baselines():
    assert sorted(DEFAULT_BASELINES) == [
        "audio", "image", "physiology", "text", "video"
    ]


def test_physiology_is_registered_with_its_own_label_column():
    assert baseline("physiology").label_column == "state_id"
    assert baseline("audio").label_column == "canonical_emotion_id"


def test_unknown_modality_is_refused():
    with pytest.raises(ValueError, match="Unknown modality"):
        baseline("olfactory")


def test_modality_order_is_declared_not_alphabetical():
    assert ordered_modalities() == ["audio", "text", "image", "video", "physiology"]
    assert ordered_modalities(["video", "audio"]) == ["audio", "video"]
