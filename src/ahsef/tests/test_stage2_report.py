"""The Stage 2 report renderer, and the routing-trace record shape."""

from __future__ import annotations

import json

import pytest

from src.ahsef.cli.report_stage2 import class_table, render, routing_tables, table_a, table_b
from src.ahsef.layout import AhsefLayout
from src.common.labels import CANONICAL_EMOTION_CLASSES


def per_class(value: float) -> dict:
    return {
        name: {"precision": value, "recall": value, "f1": value, "support": 10}
        for name in CANONICAL_EMOTION_CLASSES
    }


def analysis(split: str = "validation", with_routing: bool = False) -> dict:
    record = {
        "split": split,
        "coverage": {"samples": 1000, "usable": 980, "coverage": 0.98,
                     "status_counts": {"ok": 980, "unknown": 20},
                     "with_class_scores": 975, "score_coverage": 0.975},
        "llm_metrics": {"accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.45,
                        "macro_precision": 0.42, "macro_recall": 0.41,
                        "per_class": per_class(0.4)},
        "text_baseline_metrics_same_samples": {
            "accuracy": 0.38, "macro_f1": 0.28, "weighted_f1": 0.40,
            "macro_precision": 0.28, "macro_recall": 0.34, "per_class": per_class(0.28)},
        "uncertainty": {"policy": "score_entropy", "mean": 0.3, "median": 0.25,
                        "std": 0.2, "mean_top1_prob": 0.8, "median_top1_prob": 0.85,
                        "mean_margin": 0.6, "mean_top1_score": 0.9,
                        "definition_note": "AHSEF-derived, not a posterior."},
        "uncertainty_bins": [
            {"bin": "[0.0, 0.1)", "samples": 400, "accuracy": 0.7, "error_rate": 0.3,
             "mean_uncertainty": 0.05},
            {"bin": "[0.9, 1.0]", "samples": 100, "accuracy": 0.2, "error_rate": 0.8,
             "mean_uncertainty": 0.95},
            {"bin": "[0.5, 0.6)", "samples": 0, "accuracy": None, "error_rate": None,
             "mean_uncertainty": None},
        ],
        "uncertainty_vs_correctness": {
            "spearman_rho": 0.42, "rho_ci95": [0.36, 0.48],
            "auroc_uncertainty_predicts_error": 0.71, "n": 980},
        "latency": {"total_seconds": 1200.0, "mean_ms": 1200.0, "median_ms": 1100.0,
                    "p95_ms": 2000.0, "total_ms": 1_200_000.0, "min_ms": 800.0,
                    "max_ms": 3000.0},
        "tokens": {"input_tokens": 400000, "output_tokens": 150000,
                   "mean_input_tokens": 400.0, "mean_output_tokens": 150.0},
        "cost": {"monetary_cost_usd": None, "reason": "no published price", "execution": "cloud"},
    }
    if with_routing:
        record["routing"] = {
            "threshold": 0.35, "threshold_source": "frozen on validation",
            "stop": 600, "request": 380, "stop_pct": 0.6122, "request_pct": 0.3878,
            "conditional": [
                {"group": "All", "samples": 980, "accuracy": 0.5, "macro_f1": 0.4,
                 "error_rate": 0.5},
                {"group": "STOP", "samples": 600, "accuracy": 0.65, "macro_f1": 0.5,
                 "error_rate": 0.35},
                {"group": "REQUEST", "samples": 380, "accuracy": 0.26, "macro_f1": 0.2,
                 "error_rate": 0.74},
            ],
            "limitation": "REQUEST means only that textual evidence was insufficient.",
        }
    return record


# ------------------------------------------------------------------ tables

def test_table_a_lists_both_systems_with_all_five_metrics():
    lines = table_a(analysis(), "validation")
    body = [line for line in lines if line.startswith("| ")]
    assert any("Existing Text Baseline" in line for line in body)
    assert any("Gemma LLM" in line for line in body)
    assert body[0].count("|") == 7  # header: system + 5 metrics


def test_the_class_table_shows_signed_deltas_for_f1_and_recall():
    lines = class_table(analysis(), "validation")
    row = next(line for line in lines if line.startswith("| neutral"))
    assert "+0.1200" in row       # 0.40 LLM - 0.28 baseline
    assert row.count("|") == 9


def test_the_class_table_covers_every_canonical_emotion():
    lines = class_table(analysis(), "validation")
    for name in CANONICAL_EMOTION_CLASSES:
        assert any(line.startswith(f"| {name} ") for line in lines)


def test_table_b_omits_empty_bins_and_reports_the_association():
    lines = table_b(analysis(), "validation")
    assert any("[0.0, 0.1)" in line for line in lines)
    assert not any("[0.5, 0.6)" in line for line in lines)
    assert any("0.4200" in line and "0.7100" in line for line in lines)


def test_table_b_renders_a_missing_association_as_not_available():
    record = analysis()
    record["uncertainty_vs_correctness"] = {
        "spearman_rho": None, "auroc_uncertainty_predicts_error": None, "n": 3}
    assert any("n/a" in line for line in table_b(record, "validation"))


def test_routing_tables_carry_the_limitation_statement():
    lines = routing_tables(analysis(with_routing=True))
    assert any("| STOP | 600 |" in line for line in lines)
    assert any("REQUEST means only" in line for line in lines)
    assert any(line.startswith("| STOP |") and "0.6500" in line for line in lines)


# ------------------------------------------------------------------ render

def test_render_says_which_stages_have_not_run(tmp_path):
    layout = AhsefLayout(run="r", root=tmp_path)
    layout.prepare()
    text = render("r", str(tmp_path))
    assert "not yet run" in text


def write_analysis(tmp_path, split, record):
    base = tmp_path / "ahsef" / "r"
    (base / "analysis").mkdir(parents=True, exist_ok=True)
    (base / "analysis" / f"{split}_analysis.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    return base


def test_render_includes_every_required_section(tmp_path):
    base = write_analysis(tmp_path, "validation", analysis())
    write_analysis(tmp_path, "test", analysis("test", with_routing=True))
    (base / "frozen_config.json").write_text(json.dumps({
        "model": "gemma4:31b-cloud", "host": "http://localhost:11434",
        "prompt_version": "v2_explicit_json", "temperature": 0.0, "llm_seed": 42,
        "num_predict": 700, "uncertainty_policy": "score_entropy", "threshold": 0.35,
        "threshold_selection": {"objective": "target_stop_accuracy",
                                "selected_on_split": "validation"},
        "calibration_decision": {"reason": "not justified"},
        "frozen_at": "2026-08-29T18:00:00", "git_commit": "abc123",
    }), encoding="utf-8")

    text = render("r", str(tmp_path))
    for expected in ("TABLE A", "TABLE B", "TABLE C", "TABLE D", "Class-wise",
                     "gemma4:31b-cloud", "v2_explicit_json", "τ = 0.350000",
                     "Cost and latency"):
        assert expected in text, expected


def test_render_states_that_no_monetary_cost_is_reported(tmp_path):
    write_analysis(tmp_path, "validation", analysis())
    text = render("r", str(tmp_path))
    assert "monetary cost: **not reported**" in text
    assert "no published price" in text


def test_render_never_invents_a_missing_number(tmp_path):
    """A section whose artefact is absent must not appear with plausible values."""
    write_analysis(tmp_path, "validation", analysis())
    text = render("r", str(tmp_path))
    assert "TABLE C" not in text          # no test routing yet
    assert "test: not yet run" in text
