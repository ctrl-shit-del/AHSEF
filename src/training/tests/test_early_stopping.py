import pytest

from src.training.early_stopping import EarlyStopping


def _run(values, **kwargs):
    stopper = EarlyStopping(**kwargs)
    states = []
    for epoch, value in enumerate(values, start=1):
        state = stopper.update(value, epoch)
        states.append(state)
        if state.should_stop:
            break
    return stopper, states


def test_a_single_bad_epoch_does_not_stop_training():
    stopper, states = _run([0.30, 0.20, 0.35], patience=2, min_epochs=2)
    assert [state.should_stop for state in states] == [False, False, False]
    assert stopper.best_epoch == 3
    assert stopper.best_value == pytest.approx(0.35)
    assert stopper.stopped_epoch is None


def test_patience_is_exhausted_by_consecutive_non_improvement():
    stopper, states = _run([0.30, 0.29, 0.28, 0.40], patience=2, min_epochs=2)
    assert len(states) == 3, "training must stop once patience is exhausted"
    assert states[-1].should_stop is True
    assert stopper.best_epoch == 1
    assert stopper.stopped_epoch == 3
    assert "degraded" in stopper.reason


def test_a_plateau_is_reported_as_no_improvement_not_degradation():
    stopper, states = _run([0.30, 0.30, 0.30], patience=2, min_epochs=2)
    assert states[-1].should_stop is True
    assert "did not improve" in stopper.reason
    assert stopper.best_epoch == 1


def test_min_delta_requires_meaningful_improvement():
    stopper, states = _run([0.30, 0.3000001, 0.3000002], patience=2, min_delta=1e-4, min_epochs=2)
    assert states[-1].should_stop is True
    assert stopper.best_epoch == 1


def test_min_epochs_prevents_an_immediate_stop():
    stopper, states = _run([0.30, 0.10, 0.05], patience=1, min_epochs=3)
    assert [state.should_stop for state in states] == [False, False, True]
    assert stopper.stopped_epoch == 3


def test_completed_records_the_terminal_reason_when_no_early_stop():
    stopper, states = _run([0.10, 0.20, 0.30, 0.40, 0.50], patience=2, min_epochs=2)
    assert all(not state.should_stop for state in states)
    reason = stopper.completed(epochs_run=5, max_epochs=5)
    assert "maximum of 5 epochs" in reason
    assert stopper.summary()["stopped_early"] is False
    assert stopper.summary()["best_epoch"] == 5
    assert stopper.summary()["monitored_history"] == [0.10, 0.20, 0.30, 0.40, 0.50]


def test_summary_is_serialisable_and_complete():
    stopper, _ = _run([0.30, 0.20, 0.10], patience=2, min_epochs=2)
    summary = stopper.summary()
    assert summary["monitor"] == "val_macro_f1"
    assert summary["mode"] == "max"
    assert summary["stopped_early"] is True
    assert summary["reason"]
    assert set(summary) >= {
        "monitor", "mode", "patience", "min_delta", "min_epochs",
        "best_value", "best_epoch", "epochs_without_improvement",
        "stopped_early", "stopped_epoch", "reason", "monitored_history",
    }


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=-1)
    with pytest.raises(ValueError):
        EarlyStopping(mode="sideways")
    with pytest.raises(ValueError):
        EarlyStopping(min_epochs=0)
