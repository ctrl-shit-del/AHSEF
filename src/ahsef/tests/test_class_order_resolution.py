"""Where a baseline's class order is read from, and when it is refused.

The image baseline was trained before ``run_summary['class_order']`` existed.
Reading its order from the artefact that does carry it is fine; *guessing* it
is not, because a wrong order silently rotates every per-class number and the
confusion matrix without raising anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ahsef.inference import resolve_class_order


EMOTION = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
WESAD = ["baseline", "stress", "amusement"]
PATH = Path("synthetic/run_summary.json")


def test_top_level_class_order_wins():
    order, source = resolve_class_order(
        {"class_order": WESAD, "class_weights": {"class_order": EMOTION}}, PATH
    )
    assert order == tuple(WESAD)
    assert source == "run_summary.class_order"


def test_falls_back_to_the_class_weights_record():
    """This is the image baseline's situation."""
    order, source = resolve_class_order(
        {"class_weights": {"class_order": EMOTION}, "model": {"num_classes": 7}}, PATH
    )
    assert order == tuple(EMOTION)
    assert source == "run_summary.class_weights.class_order"


def test_falls_back_to_the_test_metrics_record():
    order, source = resolve_class_order({"test_metrics": {"class_order": EMOTION}}, PATH)
    assert order == tuple(EMOTION)
    assert source == "run_summary.test_metrics.class_order"


def test_falls_back_to_the_declared_label_space_of_the_recorded_task():
    order, source = resolve_class_order({"config": {"task": "wesad_state_3class"}}, PATH)
    assert order == tuple(WESAD)
    assert "wesad_state_3class" in source


def test_a_null_class_order_does_not_satisfy_the_chain():
    """The image summary has ``task: null``; a None must not be accepted."""
    order, source = resolve_class_order(
        {"class_order": None, "test_metrics": {"class_order": None},
         "class_weights": {"class_order": EMOTION}},
        PATH,
    )
    assert order == tuple(EMOTION)
    assert source == "run_summary.class_weights.class_order"


def test_a_summary_with_no_order_and_no_task_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="declares no class order and no task"):
        resolve_class_order({"model": {"num_classes": 7}}, PATH)


def test_an_unknown_task_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="Unknown label space"):
        resolve_class_order({"task": "emotion_99class"}, PATH)


@pytest.mark.integration
def test_the_real_image_baseline_resolves_to_the_canonical_emotion_order():
    import json

    path = Path("experiments/image/25pct/iteration_1/results/run_summary.json")
    if not path.exists():
        pytest.skip("the frozen image baseline is not present")
    order, source = resolve_class_order(
        json.loads(path.read_text(encoding="utf-8")), path
    )
    assert list(order) == EMOTION
    assert source == "run_summary.class_weights.class_order"
