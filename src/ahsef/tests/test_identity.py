"""Cross-modality sample identity: co-split pools and the guards around them."""

from __future__ import annotations

import pytest

from src.ahsef.identity import (
    AlignmentIndex,
    ModalitySplits,
    SampleAlignmentError,
    SplitContaminationError,
)


def splits(modality: str, train, validation, test, labels=None, task="emotion_7class"):
    ids = {
        "train": frozenset(train), "validation": frozenset(validation), "test": frozenset(test),
    }
    label_map = labels or {}
    return ModalitySplits(
        modality=modality, experiment=f"{modality}_synthetic",
        metadata_dir=f"synthetic/{modality}", ids=ids,
        labels={split: {i: label_map.get(i, 0) for i in members}
                for split, members in ids.items()},
        task=task,
    )


@pytest.fixture
def index() -> AlignmentIndex:
    """audio and text share MSP-like ids; image shares nothing."""
    return AlignmentIndex({
        "audio": splits("audio", ["a1", "a2", "shared_train"],
                        ["v1", "shared_val"], ["t1", "shared_test", "cross"]),
        "text": splits("text", ["shared_train", "b1", "cross"],
                       ["shared_val", "b2"], ["shared_test", "b3"]),
        "image": splits("image", ["i1"], ["i2"], ["i3"]),
        "physiology": splits("physiology", ["p1"], ["p2"], ["p3"],
                             task="wesad_state_3class"),
    })


# ------------------------------------------------------------- the matrix

def test_matrix_reports_both_co_split_and_cross_split_overlap(index):
    pair = index.alignment_matrix()["pairs"]["audio|text"]
    assert pair["co_split"] == {"train": 1, "validation": 1, "test": 1}
    # "cross" is audio-test but text-train: shared, and dangerous.
    assert pair["cross_split_shared"] == 1
    assert pair["matrix"]["test"]["train"] == 1


def test_a_disjoint_pair_is_reported_as_sharing_nothing(index):
    pair = index.alignment_matrix()["pairs"]["audio|image"]
    assert pair["shares_any_sample"] is False
    assert pair["fusable_splits"] == []


def test_matrix_carries_each_modality_task(index):
    modalities = index.alignment_matrix()["modalities"]
    assert modalities["physiology"]["task"] == "wesad_state_3class"
    assert modalities["audio"]["task"] == "emotion_7class"


# ---------------------------------------------------------------- pooling

def test_co_split_pool_is_the_same_split_intersection(index):
    assert index.co_split_pool(["audio", "text"], "test") == ["shared_test"]
    assert index.co_split_pool(["audio", "text"], "validation") == ["shared_val"]


def test_cross_split_ids_never_enter_the_pool(index):
    """'cross' is audio-test and text-train; fusing it would score memorisation."""
    assert "cross" not in index.co_split_pool(["audio", "text"], "test")
    assert index.contamination(["audio", "text"], "test") == {}


def test_pool_of_disjoint_modalities_is_empty(index):
    assert index.co_split_pool(["audio", "image"], "test") == []


def test_fusion_pool_refuses_a_pool_below_the_minimum(index):
    with pytest.raises(SampleAlignmentError, match="different corpora"):
        index.fusion_pool(["audio", "image"], "test", minimum=1)
    with pytest.raises(SampleAlignmentError, match="share only 1"):
        index.fusion_pool(["audio", "text"], "test", minimum=30)


def test_fusion_pool_accepts_a_clean_pair(index):
    assert index.fusion_pool(["audio", "text"], "test", minimum=1) == ["shared_test"]


def test_pool_needs_at_least_two_modalities(index):
    with pytest.raises(ValueError, match="at least two"):
        index.co_split_pool(["audio"], "test")


def test_unknown_split_is_refused(index):
    with pytest.raises(ValueError, match="split must be one of"):
        index.co_split_pool(["audio", "text"], "holdout")


# ------------------------------------------------------- label agreement

def test_disagreeing_labels_mean_the_ids_are_not_the_same_sample():
    conflicting = AlignmentIndex({
        "audio": splits("audio", [], [], ["x"], labels={"x": 1}),
        "text": splits("text", [], [], ["x"], labels={"x": 4}),
    })
    with pytest.raises(SampleAlignmentError, match="disagree about the label"):
        conflicting.fusion_pool(["audio", "text"], "test", minimum=1)


def test_agreeing_labels_pass(index):
    assert index.assert_label_agreement(["audio", "text"], "test") == 1


def test_contamination_is_raised_when_manifests_disagree_about_split():
    """A malformed index where one id is both test and train for a participant."""
    broken = AlignmentIndex({
        "audio": splits("audio", [], [], ["x"]),
        "text": splits("text", ["x"], [], ["x"]),
    })
    with pytest.raises(SplitContaminationError, match="training split"):
        broken.fusion_pool(["audio", "text"], "test", minimum=1)


# ------------------------------------------------------------ fusability

def test_fusability_report_explains_every_refusal(index):
    report = index.fusability_report("audio", ["text", "image", "physiology"], "test", minimum=30)
    assert report["candidates"]["image"]["fusable"] is False
    assert report["candidates"]["image"]["shares_any_sample"] is False
    assert report["candidates"]["text"]["fusable"] is False  # only 1 < 30
    assert "only 1 co-split test samples" in report["candidates"]["text"]["blockers"][0]
    physiology = report["candidates"]["physiology"]
    assert physiology["same_task"] is False
    assert any("label spaces differ" in blocker for blocker in physiology["blockers"])


def test_fusability_report_lists_what_survives(index):
    report = index.fusability_report("audio", ["text", "image"], "test", minimum=1)
    assert report["fusable_candidates"] == ["text"]
