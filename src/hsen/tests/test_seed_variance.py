"""Phase 15 -- regression tests for seed-variance aggregation.

Each of these guards a way the study's table could be wrong while looking
ordinary: a run filed under the wrong variant, a std that is 0.0 because only
one seed ran, per-class arrays in alphabetical rather than declared order, a
run that trained on a different quarter of the data, or a checkpoint whose
score is absent from its own history.  None of them needs a trained model.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.common.labels import get_label_space
from src.hsen.seed_variance import (
    REFERENCE_VARIANT,
    STUDY_SEEDS,
    STUDY_VARIANTS,
    SeedStudyError,
    StudyPlan,
    aggregate,
    collect_rows,
    margin_versus_variance,
    paired_comparison,
    render_table,
    run_row,
    verdict,
)

CLASSES = tuple(get_label_space("iemocap_erc6").classes)


def make_summary(variant: str, seed: int, wf1: float, macro: float = None,
                 accuracy: float = None, seconds: float = 100.0, best_epoch: int = 5,
                 params: int = None, subset_seed: int = 42, class_order=CLASSES,
                 history_matches: bool = True, train_samples: int = 1062) -> dict:
    """A run_summary.json shaped exactly as HSENTrainer._finalise writes it."""
    macro = wf1 - 0.03 if macro is None else macro
    accuracy = wf1 + 0.01 if accuracy is None else accuracy
    params = {"husformer": 5_992_456, "self_attention": 2_830_344,
              "concat": 1_448_968}[variant] if params is None else params
    history = [
        {"epoch": e, "val_weighted_f1": (wf1 if e == best_epoch else wf1 - 0.05)}
        for e in range(1, best_epoch + 3)
    ]
    if not history_matches:
        history = [{"epoch": e, "val_weighted_f1": wf1 - 0.05} for e in range(1, best_epoch + 3)]
    return {
        "experiment": "iemocap_erc6",
        "profile": f"phase15_seed_variance/fusion_{variant}_seed{seed}",
        "trainer_config": {"seed": seed, "subset_seed": subset_seed, "data_fraction": 0.25},
        "model_config": {"modalities": ["audio", "text"], "fusion": variant},
        "run_record": {"dataset_counts": {"train": {"samples": train_samples},
                                          "validation": {"samples": 1512}}},
        "primary_metric": "weighted_f1",
        "best_epoch": best_epoch,
        "best_val_weighted_f1": wf1,
        "epochs_run": best_epoch + 2,
        "training_seconds": seconds,
        "stop_reason": "completed",
        "parameter_counts": {"total": params, "fusion": params // 2},
        "history": history,
        "validation": {
            "accuracy": accuracy, "weighted_f1": wf1, "macro_f1": macro,
            "per_class": {name: {"f1": 0.5} for name in class_order},
        },
    }


def write_run(plan: StudyPlan, summary: dict) -> None:
    variant = summary["model_config"]["fusion"]
    seed = summary["trainer_config"]["seed"]
    directory = plan.run_dir(variant, seed)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


#: Three variants x three seeds, with a deliberately non-trivial pattern:
#: husformer leads on two seeds and loses to concat on the third.
GRID = {
    ("husformer", 42): 0.5036, ("husformer", 43): 0.4980, ("husformer", 44): 0.5100,
    ("concat", 42): 0.4928, ("concat", 43): 0.5010, ("concat", 44): 0.4950,
    ("self_attention", 42): 0.4911, ("self_attention", 43): 0.4890, ("self_attention", 44): 0.4930,
}


@pytest.fixture
def plan(tmp_path) -> StudyPlan:
    return StudyPlan(output_root=tmp_path)


@pytest.fixture
def grid_frame(plan) -> pd.DataFrame:
    for (variant, seed), wf1 in GRID.items():
        write_run(plan, make_summary(variant, seed, wf1, seconds={"husformer": 3600,
                                                                  "concat": 480,
                                                                  "self_attention": 1800}[variant]))
    subset = {"count": 1062, "sha256": "x", "subset_seed": 42, "data_fraction": 0.25}
    return collect_rows(plan, CLASSES, subset)


# ---------------------------------------------------------------- rows

def test_every_seed_of_every_variant_becomes_one_row(grid_frame):
    assert len(grid_frame) == 9
    assert set(grid_frame["variant"]) == set(STUDY_VARIANTS)
    assert set(grid_frame["seed"]) == set(STUDY_SEEDS)
    # Ordered by the study's declared variant order, then seed -- not alphabetically.
    assert grid_frame["variant"].tolist()[:3] == ["husformer"] * 3
    assert grid_frame["seed"].tolist()[:3] == [42, 43, 44]
    for column in ("best_epoch", "val_WF1", "val_macro_F1", "accuracy",
                   "train_time", "parameter_count"):
        assert column in grid_frame.columns


def test_a_row_is_filed_under_the_variant_the_run_actually_trained():
    """Directory name and model_config must agree; the model_config wins or it errors."""
    summary = make_summary("concat", 43, 0.49)
    assert run_row(summary)["variant"] == "concat"
    assert run_row(summary, "concat", 43)["seed"] == 43
    with pytest.raises(SeedStudyError, match="labelled 'husformer' but"):
        run_row(summary, "husformer", 43)
    with pytest.raises(SeedStudyError, match="labelled seed 42"):
        run_row(summary, "concat", 42)


def test_an_unknown_variant_name_is_refused():
    summary = make_summary("concat", 42, 0.49, params=10)
    summary["model_config"]["fusion"] = "self+attention"
    with pytest.raises(SeedStudyError, match="Unknown fusion variant"):
        run_row(summary)
    with pytest.raises(SeedStudyError, match="Unknown fusion variant"):
        StudyPlan(variants=("husformer", "self+attention"))


def test_a_run_that_opened_the_test_split_is_refused():
    summary = make_summary("concat", 42, 0.49)
    summary["test"] = {"weighted_f1": 0.5}
    with pytest.raises(SeedStudyError, match="validation-only"):
        run_row(summary)


def test_a_best_score_absent_from_its_own_history_is_refused():
    """The RunLock incident, as a check on the artefact rather than the process."""
    with pytest.raises(SeedStudyError, match="does not appear in the run's own history"):
        run_row(make_summary("husformer", 42, 0.50, history_matches=False))


def test_class_order_is_the_declared_order_never_sorted(plan):
    """A per_class block in alphabetical order is a rotated per-class array."""
    assert list(CLASSES) != sorted(CLASSES), "declared order is alphabetical; test is vacuous"
    write_run(plan, make_summary("husformer", 42, 0.50, class_order=tuple(sorted(CLASSES))))
    with pytest.raises(SeedStudyError, match="not the declared"):
        collect_rows(plan, CLASSES)
    # Without a declared order to check against the row is still readable and
    # carries whatever order it found, so a later check can still catch it.
    row = collect_rows(plan)
    assert tuple(row.iloc[0]["class_order"]) == tuple(sorted(CLASSES))


def test_a_run_on_a_different_training_subset_is_refused(plan):
    write_run(plan, make_summary("concat", 43, 0.49, subset_seed=43))
    with pytest.raises(SeedStudyError, match="subset_seed=43, not the study's 42"):
        collect_rows(plan, CLASSES)


def test_a_run_with_a_different_training_count_is_refused(plan):
    write_run(plan, make_summary("concat", 43, 0.49, train_samples=4246))
    subset = {"count": 1062, "sha256": "x", "subset_seed": 42, "data_fraction": 0.25}
    with pytest.raises(SeedStudyError, match="trained on 4246 samples"):
        collect_rows(plan, CLASSES, subset)


def test_a_partial_run_without_a_summary_is_not_a_row(plan):
    (plan.run_dir("husformer", 42) / "checkpoints").mkdir(parents=True)
    assert collect_rows(plan, CLASSES).empty


# ----------------------------------------------------------- aggregate

def test_mean_std_min_max_over_multiple_seeds(grid_frame):
    stats = aggregate(grid_frame)
    hus = stats["husformer"]
    values = [GRID[("husformer", s)] for s in STUDY_SEEDS]
    mean = sum(values) / 3
    assert hus["n_seeds"] == 3 and hus["seeds"] == [42, 43, 44]
    assert hus["mean_WF1"] == pytest.approx(mean)
    # Sample std, ddof=1 -- the honest choice at n=3, and stated in the output.
    assert hus["std_WF1"] == pytest.approx(math.sqrt(sum((v - mean) ** 2 for v in values) / 2))
    assert hus["std_ddof"] == 1
    assert hus["min_WF1"] == pytest.approx(0.4980)
    assert hus["max_WF1"] == pytest.approx(0.5100)
    assert hus["mean_macro_F1"] == pytest.approx(mean - 0.03)
    assert hus["std_macro_F1"] == pytest.approx(hus["std_WF1"])
    assert hus["mean_training_time"] == pytest.approx(3600.0)
    assert stats["concat"]["mean_training_time"] == pytest.approx(480.0)
    assert hus["parameter_count"] == 5_992_456
    assert list(stats) == list(STUDY_VARIANTS)


def test_std_matches_pandas_sample_std(grid_frame):
    stats = aggregate(grid_frame)
    for variant in STUDY_VARIANTS:
        expected = grid_frame.loc[grid_frame["variant"] == variant, "val_WF1"].std(ddof=1)
        assert stats[variant]["std_WF1"] == pytest.approx(expected)


def test_a_single_seed_reports_nan_std_not_zero(plan):
    write_run(plan, make_summary("concat", 42, 0.49))
    stats = aggregate(collect_rows(plan, CLASSES))
    assert stats["concat"]["n_seeds"] == 1
    assert math.isnan(stats["concat"]["std_WF1"])
    assert stats["concat"]["mean_WF1"] == pytest.approx(0.49)


def test_seeds_that_trained_different_architectures_cannot_be_averaged(plan):
    write_run(plan, make_summary("husformer", 42, 0.50))
    write_run(plan, make_summary("husformer", 43, 0.50, params=123))
    with pytest.raises(SeedStudyError, match="disagree on parameter_count"):
        aggregate(collect_rows(plan, CLASSES))


# -------------------------------------------------------------- pairing

def test_paired_comparison_is_seed_by_seed(grid_frame):
    pairs = paired_comparison(grid_frame, REFERENCE_VARIANT)
    concat = pairs["concat"]
    assert concat["seeds"] == [42, 43, 44]
    assert concat["delta_per_seed"]["42"] == pytest.approx(0.5036 - 0.4928)
    assert concat["delta_per_seed"]["43"] == pytest.approx(0.4980 - 0.5010)
    assert concat["reference_wins"] == 2 and concat["n_pairs"] == 3
    assert concat["consistent"] is False
    sa = pairs["self_attention"]
    assert sa["reference_wins"] == 3 and sa["consistent"] is True
    assert sa["min_delta"] > 0


def test_pairing_uses_only_seeds_both_variants_ran(plan):
    write_run(plan, make_summary("husformer", 42, 0.50))
    write_run(plan, make_summary("husformer", 43, 0.51))
    write_run(plan, make_summary("concat", 42, 0.49))
    pairs = paired_comparison(collect_rows(plan, CLASSES))
    assert pairs["concat"]["seeds"] == [42]
    assert pairs["concat"]["n_pairs"] == 1


def test_margin_is_compared_against_both_spreads(grid_frame):
    stats = aggregate(grid_frame)
    pairs = paired_comparison(grid_frame)
    margins = margin_versus_variance(stats, pairs)
    m = margins["concat"]
    assert m["larger_seed_std"] == pytest.approx(max(stats["husformer"]["std_WF1"],
                                                     stats["concat"]["std_WF1"]))
    assert m["margin_exceeds_larger_seed_std"] is (m["mean_margin"] > m["larger_seed_std"])
    assert m["ranges_overlap"] is True          # 0.4980 <= 0.5010


# -------------------------------------------------------------- verdict

def test_verdict_never_claims_significance(grid_frame):
    stats = aggregate(grid_frame)
    pairs = paired_comparison(grid_frame)
    decision = verdict(stats, pairs, margin_versus_variance(stats, pairs))
    text = json.dumps(decision).lower()
    assert "significan" not in text.replace("statistical significance in either direction", "")
    assert "three seeds" in text or "3 seeds" in text
    assert decision["complete"] is True
    assert decision["ranking_by_mean_WF1"][0] == "husformer"
    assert decision["recommended_fusion"] == "husformer"
    assert "flips with the seed" in decision["reason"]


def test_verdict_flags_an_incomplete_grid(plan):
    write_run(plan, make_summary("husformer", 42, 0.50))
    write_run(plan, make_summary("concat", 42, 0.49))
    frame = collect_rows(plan, CLASSES)
    stats = aggregate(frame)
    decision = verdict(stats, paired_comparison(frame), {}, n_seeds_planned=3)
    assert decision["complete"] is False
    assert decision["statements"][0].startswith("INCOMPLETE")


def test_verdict_recommends_the_higher_mean_when_reference_loses(plan):
    for seed, (h, c) in zip(STUDY_SEEDS, [(0.49, 0.50), (0.48, 0.51), (0.50, 0.505)]):
        write_run(plan, make_summary("husformer", seed, h))
        write_run(plan, make_summary("concat", seed, c))
    frame = collect_rows(plan, CLASSES)
    stats = aggregate(frame)
    pairs = paired_comparison(frame)
    decision = verdict(stats, pairs, margin_versus_variance(stats, pairs), n_seeds_planned=3)
    assert decision["recommended_fusion"] == "concat"
    assert "did not hold its section-13 lead" in decision["reason"]


# ------------------------------------------------------------ rendering

def test_table_has_one_column_per_seed_and_mean_pm_std(grid_frame):
    stats = aggregate(grid_frame)
    table = render_table(grid_frame, stats)
    lines = table.splitlines()
    assert lines[0].startswith("| variant | seed42 | seed43 | seed44 | mean ± std")
    assert lines[0].rstrip().endswith("| params |")
    body = lines[2:]
    assert [line.split("|")[1].strip() for line in body] == list(STUDY_VARIANTS)
    hus = body[0]
    assert "0.5036" in hus and "0.4980" in hus and "0.5100" in hus
    assert f"{stats['husformer']['mean_WF1']:.4f} ± {stats['husformer']['std_WF1']:.4f}" in hus
    assert "5,992,456" in hus


def test_table_marks_a_missing_seed_rather_than_dropping_the_row(plan):
    write_run(plan, make_summary("husformer", 42, 0.50))
    write_run(plan, make_summary("husformer", 44, 0.51))
    frame = collect_rows(plan, CLASSES)
    table = render_table(frame, aggregate(frame))
    row = [line for line in table.splitlines() if line.startswith("| husformer")][0]
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[1:4] == ["0.5000", "--", "0.5100"]


# --------------------------------------------------------------- plan

def test_plan_pins_the_subset_seed_and_never_opens_test(plan):
    config = plan.trainer_config("concat", 44)
    assert config.seed == 44
    assert config.subset_seed == 42
    assert config.evaluate_test is False
    assert config.run_dir == plan.study_dir / "fusion_concat_seed44"
    # The section-13 settings, exactly.
    assert (config.device, config.data_fraction, config.batch_size, config.epochs) == \
        ("cpu", 0.25, 8, 15)
    assert (config.learning_rate, config.optimizer, config.scheduler, config.patience,
            config.min_epochs) == (1e-4, "adamw", "cosine_warmup", 5, 3)


def test_plan_refuses_duplicate_seeds():
    with pytest.raises(SeedStudyError, match="distinct"):
        StudyPlan(seeds=(42, 42, 44))


def test_each_run_directory_is_its_own_lock(plan):
    """Two runs of the study may not share a directory; the same run may not run twice."""
    from src.hsen.training.runlock import RunDirectoryBusy, RunLock

    first = RunLock(plan.trainer_config("husformer", 42).run_dir).acquire()
    try:
        # A different seed of the same variant is a different directory: allowed.
        RunLock(plan.trainer_config("husformer", 43).run_dir).acquire().release()
        # The same (variant, seed) is refused.
        with pytest.raises(RunDirectoryBusy):
            RunLock(plan.trainer_config("husformer", 42).run_dir).acquire()
    finally:
        first.release()
