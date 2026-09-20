"""Render the Stage 2 pilot report from the artefacts on disk.

    python -m src.ahsef.cli.report_stage2 --run stage2_llm

Reads only what the pilot stages wrote, so the report cannot drift from the
experiment: every number it prints is traceable to a file, and it refuses to
render a section whose artefact is missing rather than leaving a plausible gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ahsef.layout import AhsefLayout
from src.common.labels import CANONICAL_EMOTION_CLASSES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="stage2_llm")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--out", default=None, help="Write markdown here as well.")
    return parser


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table_a(record: dict, split: str) -> list[str]:
    llm = record["llm_metrics"]
    base = record["text_baseline_metrics_same_samples"]
    lines = [
        f"### TABLE A — {split}: LLM vs existing text baseline (paired, identical samples)",
        "",
        "| System | Accuracy | Macro-F1 | Weighted-F1 | Macro Precision | Macro Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in (("Existing Text Baseline", base), ("Gemma LLM", llm)):
        lines.append(
            f"| {name} | {block['accuracy']:.4f} | {block['macro_f1']:.4f} | "
            f"{block['weighted_f1']:.4f} | {block['macro_precision']:.4f} | "
            f"{block['macro_recall']:.4f} |"
        )
    lines.append("")
    return lines


def class_table(record: dict, split: str) -> list[str]:
    llm = record["llm_metrics"]["per_class"]
    base = record["text_baseline_metrics_same_samples"]["per_class"]
    lines = [
        f"### Class-wise — {split}",
        "",
        "| Emotion | Baseline F1 | LLM F1 | Δ F1 | Baseline Recall | LLM Recall | Δ Recall | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANONICAL_EMOTION_CLASSES:
        b, l = base[name], llm[name]
        lines.append(
            f"| {name} | {b['f1']:.4f} | {l['f1']:.4f} | {l['f1'] - b['f1']:+.4f} | "
            f"{b['recall']:.4f} | {l['recall']:.4f} | {l['recall'] - b['recall']:+.4f} | "
            f"{b['support']} |"
        )
    lines.append("")
    return lines


def table_b(record: dict, split: str) -> list[str]:
    lines = [
        f"### TABLE B — {split}: uncertainty reliability",
        "",
        "| Uncertainty Bin | Samples | Accuracy | Error Rate | Mean U |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in record["uncertainty_bins"]:
        if not row["samples"]:
            continue
        lines.append(
            f"| {row['bin']} | {row['samples']} | {row['accuracy']:.4f} | "
            f"{row['error_rate']:.4f} | {row['mean_uncertainty']:.4f} |"
        )
    assoc = record["uncertainty_vs_correctness"]
    lines += [
        "",
        f"Spearman ρ(uncertainty, wrong) = **{_fmt(assoc['spearman_rho'])}**"
        + (f" (95% CI {_fmt(assoc['rho_ci95'][0])} to {_fmt(assoc['rho_ci95'][1])})"
           if assoc.get("rho_ci95") else "")
        + f"; AUROC = **{_fmt(assoc['auroc_uncertainty_predicts_error'])}**"
        f" over n={assoc['n']}.",
        "",
    ]
    return lines


def routing_tables(record: dict) -> list[str]:
    routing = record["routing"]
    lines = [
        "### TABLE C — AHSEF text gate (τ frozen on validation)",
        "",
        "| Routing Outcome | Count | Percentage |",
        "|---|---:|---:|",
        f"| STOP | {routing['stop']} | {routing['stop_pct']:.1%} |",
        f"| REQUEST | {routing['request']} | {routing['request_pct']:.1%} |",
        "",
        "### TABLE D — Conditional performance",
        "",
        "| Group | Samples | Accuracy | Macro-F1 | Error Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in routing["conditional"]:
        if row.get("samples"):
            lines.append(
                f"| {row['group']} | {row['samples']} | {row['accuracy']:.4f} | "
                f"{row['macro_f1']:.4f} | {row['error_rate']:.4f} |"
            )
    lines += ["", f"> {routing['limitation']}", ""]
    return lines


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render(run: str, root: str) -> str:
    layout = AhsefLayout(run=run, root=Path(root) / "ahsef")
    base = layout.base
    validation = _load(base / "analysis" / "validation_analysis.json")
    test = _load(base / "analysis" / "test_analysis.json")
    frozen = _load(base / "frozen_config.json")
    selection = _load(base / "manifests" / "selection_summary.json")

    lines = ["# AHSEF Stage 2 — LLM text pilot results", ""]
    if selection:
        lines += ["## Sample selection", ""]
        for split, entry in selection["splits"].items():
            lines.append(
                f"- **{split}**: {entry['selected']} samples, seed {entry['seed']}, "
                f"fingerprint `{entry['fingerprint_sha256'][:16]}`, "
                f"classes {entry['class_counts']}"
            )
        lines += [
            f"- method: `{selection['selection_method']}`; "
            f"labels used for inclusion: {selection['labels_used_for_inclusion']}; "
            f"splits disjoint: {selection['splits_disjoint']}",
            "",
        ]

    if frozen:
        lines += [
            "## Frozen configuration", "",
            f"- model: `{frozen['model']}` (provider ollama, host `{frozen['host']}`)",
            f"- prompt: `{frozen['prompt_version']}`",
            f"- temperature {frozen['temperature']}, seed {frozen['llm_seed']}, "
            f"num_predict {frozen['num_predict']}",
            f"- uncertainty policy: `{frozen['uncertainty_policy']}`",
            f"- **τ = {frozen['threshold']:.6f}** "
            f"(objective `{frozen['threshold_selection']['objective']}`, "
            f"selected on {frozen['threshold_selection']['selected_on_split']})",
            f"- calibration decision: {frozen['calibration_decision']['reason']}",
            f"- frozen at {frozen['frozen_at']}, git `{frozen.get('git_commit')}`",
            "",
        ]

    for split, record in (("validation", validation), ("test", test)):
        if record is None:
            lines += [f"## {split}: not yet run", ""]
            continue
        lines += [f"## {split.capitalize()} results", ""]
        coverage = record["coverage"]
        lines += [
            f"- parsed/usable: {coverage['usable']}/{coverage['samples']} "
            f"({coverage['coverage']:.1%}); statuses {coverage['status_counts']}",
            f"- responses carrying class scores: {coverage['with_class_scores']} "
            f"({coverage['score_coverage']:.1%})",
            "",
        ]
        lines += table_a(record, split)
        lines += class_table(record, split)
        unc = record["uncertainty"]
        lines += [
            f"### Uncertainty — {split}", "",
            f"- policy `{unc['policy']}`: mean {unc['mean']:.4f}, "
            f"median {unc['median']:.4f}, std {unc['std']:.4f}",
            f"- mean top-1 score {unc['mean_top1_prob']:.4f}, "
            f"median {unc['median_top1_prob']:.4f}, mean margin {unc['mean_margin']:.4f}",
            f"- {unc['definition_note']}",
            "",
        ]
        lines += table_b(record, split)
        if record.get("llm_confidence_calibration"):
            cal = record["llm_confidence_calibration"]
            lines += [
                f"### Calibration — {split}", "",
                f"- self-reported confidence: ECE {cal['ece']:.4f}, "
                f"MCE {cal['mce']:.4f}, Brier {cal['brier']:.4f}, "
                f"mean confidence {cal['mean_confidence']:.4f} vs "
                f"accuracy {cal['accuracy']:.4f} "
                f"(overconfidence {cal['overconfidence']:+.4f})",
                f"- discrimination: AUROC {_fmt(cal['discrimination']['auroc'])}, "
                f"{cal['discrimination']['distinct_values']} distinct values, "
                f"separation {_fmt(cal['confidence_vs_correctness']['separation'])}",
            ]
            if record.get("score_distribution_calibration"):
                sd = record["score_distribution_calibration"]
                lines.append(
                    f"- score distribution: ECE {sd['ece']:.4f}, NLL {sd['nll']:.4f}, "
                    f"Brier {sd['brier']:.4f} — {sd['caveat']}"
                )
            lines.append("")
        latency, tokens, cost = record["latency"], record["tokens"], record["cost"]
        lines += [
            f"### Cost and latency — {split}", "",
            f"- total {latency['total_seconds']:.1f} s; per sample mean "
            f"{latency['mean_ms']:.0f} ms, median {latency['median_ms']:.0f} ms, "
            f"p95 {latency['p95_ms']:.0f} ms",
            f"- tokens: {tokens['input_tokens']} in / {tokens['output_tokens']} out "
            f"(mean {tokens['mean_input_tokens']:.0f} / "
            f"{tokens['mean_output_tokens']:.0f} per sample)",
            f"- monetary cost: **not reported** — {cost['reason']}",
            f"- execution: {cost.get('execution')}",
            "",
        ]
        if record.get("routing"):
            lines += routing_tables(record)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = render(args.run, args.root)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
