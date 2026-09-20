"""Render ``docs/ahsef_stage3_results.md`` from the Stage 3 artefacts.

The results document is *generated*, never typed.  Every number in it is read
from a JSON artefact produced by the pipeline, so the report cannot drift away
from the run that produced it, and a reader who doubts a figure can open the
file it came from.  Sections whose artefact is missing say so rather than being
quietly omitted -- an absent locked-test table must look absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from src.ahsef.stage3 import STAGE3_CLAIM, STAGE3_CLAIM_LIMITS, STAGE3_EXPERIMENT_ID
from src.ahsef.stage3.layout import Stage3Layout
from src.common.labels import CANONICAL_EMOTION_CLASSES

DOCS = Path("docs")

SYSTEM_LABELS = {
    "text_only": "Text only (Gemma LLM)",
    "audio_only": "Audio only",
    "always_fusion": "Always Text+Audio",
    "stage2_gate_plus_audio": "Stage-2 gate + Audio",
    "ahsef_dynamic": "AHSEF dynamic (HSIG + UGAPR)",
    "D_uncertainty_threshold_only": "D. Uncertainty threshold only",
    "E_hsig_without_ugapr": "E. HSIG without UGAPR",
    "F_ugapr_with_oracle_gain": "F. UGAPR with oracle gain",
    "G_hsig_plus_ugapr": "G. HSIG + UGAPR",
}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 4, dash: str = "--") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value) * 100:.{digits}f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


# ============================================================
# Sections
# ============================================================

def _alignment(report: Mapping | None) -> list[str]:
    lines = ["## 1. The aligned Text+Audio pool (Phase A)", ""]
    if not report:
        return lines + ["*Not run: `--stage align` has not been executed.*", ""]
    lines += [
        "Stage 1 established that the only defensible co-split pair in this project is",
        "Text + Audio. Phase A turns that pair into an exact sample list and records the",
        "evidence that vetted it.", "",
    ]
    rows = []
    for split, block in report["splits"].items():
        rows.append([
            split, str(block["size"]), block["fingerprint_sha256"][:16],
            ", ".join(f"{k}={v}" for k, v in block["datasets"].items()),
            str(block["checks"]["label_agreement_verified"]),
            "none" if not block["checks"]["train_contamination"] else "FOUND",
        ])
    lines += _table(
        ["Split", "Samples", "Fingerprint", "Datasets", "Labels agreed", "Train contamination"],
        rows,
    )
    lines += ["", "Verified:", ""]
    lines += [f"- {item}" for item in report["verified"]]
    lines += ["", "Explicitly not done:", ""]
    lines += [f"- {item}" for item in report["not_done"]]
    lines += ["", f"**Limitation.** {report['limitation']}", ""]

    first = next(iter(report["splits"].values()))
    counts = first["class_counts"]
    lines += ["Validation class balance:", ""]
    lines += _table(
        ["Class"] + list(CANONICAL_EMOTION_CLASSES),
        [["n"] + [str(counts.get(name, 0)) for name in CANONICAL_EMOTION_CLASSES]],
    )
    lines += [
        "",
        "The three rarest classes carry single-digit support. Every macro-averaged",
        "figure in this document is therefore dominated at the margin by a handful of",
        "samples, and per-class F1 for `fear`, `disgust`, and `surprise` should be read",
        "as an indication rather than a measurement.",
        "",
    ]
    return lines


def _oracle(report: Mapping | None, split: str) -> list[str]:
    lines = [f"## 2. Does audio actually help? (Phase C, {split})", ""]
    if not report:
        return lines + [f"*Not run for {split}.*", ""]
    rows = []
    for name, block in report["systems"].items():
        rows.append([
            name, str(block["samples"]), _fmt(block["accuracy"]),
            _fmt(block["macro_f1"]), _fmt(block["weighted_f1"]),
            _fmt(block["balanced_accuracy"]),
        ])
    lines += _table(
        ["System", "n", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc."], rows
    )
    outcome = report["acquisition_outcome"]
    lines += ["", "Acquisition outcome, measured by paying for audio on every sample:", ""]
    lines += _table(
        ["Quantity", "Count", "Fraction", "95% CI"],
        [
            ["Text wrong → fusion right (`y_gain`)", str(outcome["fixed_by_audio"]),
             _pct(outcome["fixed_fraction"]),
             f"[{_fmt(outcome['fixed_fraction_ci95'][0], 3)}, "
             f"{_fmt(outcome['fixed_fraction_ci95'][1], 3)}]"],
            ["Text right → fusion wrong (`y_harm`)", str(outcome["broken_by_audio"]),
             _pct(outcome["broken_fraction"]),
             f"[{_fmt(outcome['broken_fraction_ci95'][0], 3)}, "
             f"{_fmt(outcome['broken_fraction_ci95'][1], 3)}]"],
            ["Net improvement", f"{outcome['net_improvement']:+d}",
             _pct(outcome["net_improvement_rate"]), "--"],
            ["Text/audio prediction agreement", "--",
             _pct(outcome["agreement_text_audio"]), "--"],
        ],
    )
    lines += ["", "Per-class:", ""]
    rows = []
    for name, row in report["per_class_improvement"].items():
        if not row["samples"]:
            continue
        rows.append([
            name, str(row["samples"]), _fmt(row["text_accuracy"]),
            _fmt(row["fused_accuracy"]), str(row["fixed_by_audio"]),
            str(row["broken_by_audio"]), f"{row['net']:+d}",
        ])
    lines += _table(
        ["Class", "n", "Text acc.", "Fused acc.", "Fixed", "Broken", "Net"], rows
    )
    delta = report["delta_uncertainty_diagnostic"]
    lines += [
        "",
        f"**The rejected target, re-measured.** Delta-uncertainty correlates with the "
        f"realised signed gain at r = {_fmt(delta['correlation_with_signed_gain'], 3)} "
        f"on this pool. {delta['status']}.",
        "",
    ]
    if report.get("fusion"):
        fusion = report["fusion"]
        lines += [
            f"Fusion rule: `{fusion['method']}`, weights "
            + ", ".join(f"{k}={v:.2f}" for k, v in fusion["weights"].items())
            + f", selected on {fusion['selected_on_split']}.",
            "",
        ]
    return lines


def _hsig(report: Mapping | None) -> list[str]:
    lines = ["## 3. HSIG: can the gain be predicted? (Phase D)", ""]
    if not report:
        return lines + ["*Not run.*", ""]
    target = report["target"]
    lines += [
        f"Target: `{target['formula']}`, estimated on {target['estimated_on']}.",
        f"Positive events (`y_gain`): {target['y_gain_positives']} of "
        f"{target['samples']} ({_pct(target['y_gain_rate'])}). "
        f"Harmful events (`y_harm`): {target['y_harm_positives']} "
        f"({_pct(target['y_harm_rate'])}).",
        "",
        "**Every figure below is out of fold.** At this pool size an in-sample AUROC",
        "would be optimistic by an amount that changes the conclusion, so the model is",
        f"scored by {report['honesty']['folds']}-fold seeded stratified cross-validation,",
        "and every reference -- a constant at the base rate, a shallow forest, and the",
        "single-feature variants -- is scored by the same code path.",
        "",
    ]
    rows = []
    for name, entry in report["variants"].items():
        block = entry.get("identifying_useful_acquisitions")
        if not block:
            rows.append([f"`{name}`", "unavailable", "--", "--", "--", "--"])
            continue
        interval = block["auroc_ci95"]
        rows.append([
            f"`{name}`", _fmt(block["auroc"]),
            f"[{_fmt(interval[0], 3)}, {_fmt(interval[1], 3)}]" if interval else "--",
            _fmt(block["auprc"]), _fmt(block["base_rate"], 3),
            _fmt(entry["predicted_vs_observed"]["spearman_rho"], 3),
        ])
    lines += _table(
        ["Estimator", "AUROC", "AUROC 95% CI", "AUPRC", "AUPRC baseline", "Spearman ρ"],
        rows,
    )
    verdict = report["comparison"].get("verdict")
    if verdict:
        lines += ["", f"**Verdict.** {verdict['statement']}", ""]
    selected = report["selected"]
    lines += [
        f"Frozen estimator: `{selected['estimator_type']}` over feature set "
        f"`{selected['feature_set']}`, fingerprint `{selected['fingerprint_sha256'][:16]}`.",
        "",
    ]
    rule = report.get("selection_rule")
    if rule:
        lines += [
            "### How that variant was chosen", "",
            f"- **Primary rule.** {rule['primary']}. Best by that rule: "
            f"`{rule['best_by_auroc']}` at AUROC {_fmt(rule['best_auroc'])}.",
            f"- **Tie-break.** {rule['tie_break']}",
            f"- Variants this pool cannot distinguish from the best: "
            + ", ".join(f"`{name}`" for name in rule["indistinguishable_from_best"]),
            f"- Selected `{rule['selected']}` at AUROC {_fmt(rule['selected_auroc'])}, "
            f"conceding {_fmt(rule['auroc_conceded_to_the_tie_break'], 4)} AUROC to the "
            f"tie-break.",
            "",
            rule["why_paired_wins_ties"],
            "",
        ]
    direction = report["feature_importance"].get("gain_direction")
    if direction:
        ranked = direction["ranked_by_absolute_weight"][:6]
        lines += ["Features driving the acquisition decision (standardised weights on the",
                  "difference of the two components -- positive means acquiring looks more",
                  "attractive as the feature rises):", ""]
        lines += _table(
            ["Feature", "Weight"],
            [[f"`{name}`", _fmt(direction["coefficients"][name], 3)] for name in ranked],
        )
        lines += [""]
    audit = report["class_rule_audit"]
    lines += [
        "**Is this a hard-coded class rule in disguise?**",
        f"Within-class spread of predicted gain: {_fmt(audit['mean_within_class_std'], 4)}; "
        f"between-class spread: {_fmt(audit['between_class_std'], 4)}.",
        f"{audit['verdict']}",
        "",
        f"**Caveat.** {report['honesty']['caveat']}",
        "",
    ]
    return lines


def _frozen(config: Mapping | None) -> list[str]:
    lines = ["## 4. The frozen configuration (Phases E, F, J)", ""]
    if not config:
        return lines + ["*Not frozen.*", ""]
    gate = config["gate"]
    ugapr = config["ugapr"]
    lines += [
        f"Frozen at `{config['frozen_at']}`, git commit `{(config.get('git_commit') or '')[:12]}`.",
        "",
        "### Gate objective",
        "",
        f"    {gate['selection']['objective_formula']}",
        "",
        f"Selected objective: `{gate['objective']}`, "
        f"α = {gate['alpha_acquisition']}, β = {gate['beta_latency']}, "
        f"τ = {_fmt(gate['threshold'], 6)}.",
        "",
        gate["objective_rationale"]["why"],
        "",
        "The alternatives were swept and are recorded so the choice is auditable:",
        "",
    ]
    candidates = gate["selection"]["objective_comparison"]["candidates"]
    lines += _table(
        ["Objective", "τ", "Acquisition rate", "Accuracy", "Macro-F1", "Selected"],
        [
            [f"`{name}`", _fmt(entry["selected_threshold"], 4),
             _pct(entry["acquisition_rate"]), _fmt(entry["accuracy"]),
             _fmt(entry["macro_f1"]), "**yes**" if entry["selected_for_stage3"] else "no"]
            for name, entry in candidates.items()
        ],
    )
    lines += [
        "", "### UGAPR", "",
        f"    {ugapr['formula']}",
        "",
        f"λ = {ugapr['lambda_cost']}, μ = {ugapr['mu_latency']}, "
        f"candidate registry = {ugapr['candidate_registry']}, "
        f"audio penalty = {_fmt(ugapr['audio_penalty'], 6)}.",
        "",
    ]
    costs = config["costs"]["modalities"]
    lines += _table(
        ["Modality", "Latency (ms)", "Compute units", "Normalised cost",
         "Normalised latency"],
        [
            [name, _fmt(entry["latency_ms"], 1), f"{entry['compute_units']:.3g}",
             _fmt(entry["normalized_cost"], 4), _fmt(entry["normalized_latency"], 4)]
            for name, entry in costs.items()
        ],
    )
    lines += [
        "",
        config["costs"]["definition"]["llm_latency_caveat"],
        "",
    ]
    sensitivity = gate.get("penalty_sensitivity")
    if sensitivity:
        lines += [
            "### Sensitivity to the frozen penalties", "",
            "α and β are a policy choice, not a measurement, so what the policy would",
            "have been at other prices is recorded rather than left implicit:",
            "",
        ]
        lines += _table(
            ["α", "β", "τ", "Acquisition rate", "Macro-F1", "Accuracy"],
            [
                [_fmt(row["alpha"], 2), _fmt(row["beta"], 2), _fmt(row["threshold"], 4),
                 _pct(row["acquisition_rate"]), _fmt(row["macro_f1"]),
                 _fmt(row["accuracy"])]
                for row in sensitivity
            ],
        )
        lines += [""]
    comparison = gate.get("out_of_fold_comparison")
    if comparison:
        point = comparison["selected_point"]
        lines += [
            "### Threshold selection: which scores τ was chosen against", "",
            gate.get("why_not_out_of_fold", ""), "",
        ]
        lines += _table(
            ["Scores τ was chosen on", "τ", "Acquisition rate", "Macro-F1", "Accuracy"],
            [
                ["**Deployed estimator (frozen)**", _fmt(gate["threshold"], 6),
                 _pct(gate["selection"]["selected_point"]["acquisition_rate"]),
                 _fmt(gate["selection"]["selected_point"]["macro_f1"]),
                 _fmt(gate["selection"]["selected_point"]["accuracy"])],
                ["Out-of-fold (comparison only)", _fmt(comparison["threshold"], 6),
                 _pct(point["acquisition_rate"]), _fmt(point["macro_f1"]),
                 _fmt(point["accuracy"])],
            ],
        )
        lines += ["", comparison["note"], ""]
    lines += ["### What the freeze fixes", ""]
    lines += _table(
        ["Item", "Value"],
        [
            ["LLM model", f"`{config['text_llm']['model']}`"],
            ["Prompt version", f"`{config['text_llm']['prompt_version']}`"],
            ["Decoding", f"temperature={config['text_llm']['temperature']}, "
                         f"seed={config['text_llm']['llm_seed']}"],
            ["Audio checkpoint", f"`{(config['audio_baseline']['checkpoint_sha256'] or '')[:16]}`"],
            ["Uncertainty policy", f"`{config['uncertainty_policy']}`"],
            ["HSIG estimator", f"`{config['hsig']['estimator_type']}` / "
                               f"`{config['hsig']['feature_set']}`"],
            ["HSIG fingerprint", f"`{config['hsig']['fingerprint_sha256'][:16]}`"],
            ["HSIG target", f"`{config['hsig']['target_definition']['formula']}`"],
            ["Fusion rule", f"`{config['fusion']['method']}` "
                            + ", ".join(f"{k}={v:.2f}" for k, v in config['fusion']['weights'].items())],
            ["Gate objective", f"`{gate['objective']}`"],
            ["Threshold τ", _fmt(gate["threshold"], 6)],
            ["λ / μ", f"{ugapr['lambda_cost']} / {ugapr['mu_latency']}"],
            ["α / β", f"{gate['alpha_acquisition']} / {gate['beta_latency']}"],
            ["Validation pool", f"`{config['alignment']['validation_pool_fingerprint_sha256'][:16]}` "
                                f"(n={config['alignment']['validation_pool_size']})"],
            ["Test pool", f"`{config['alignment']['test_pool_fingerprint_sha256'][:16]}` "
                          f"(n={config['alignment']['test_pool_size']})"],
            ["Seeds", ", ".join(f"{k}={v}" for k, v in config["seeds"].items())],
            ["Python", config["environment"]["python"].split()[0]],
            ["scikit-learn", config["environment"]["scikit_learn"]],
        ],
    )
    lines += ["", config["lock_note"], ""]
    return lines


def _systems(report: Mapping | None, title: str, order: Sequence[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not report:
        return lines + ["*Not run.*", ""]
    lines += [f"n = {report['samples']}.", ""]
    rows = []
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        rows.append([
            SYSTEM_LABELS.get(name, name), _fmt(record["accuracy"]),
            _fmt(record["macro_f1"]), _fmt(record["weighted_f1"]),
            _fmt(record["balanced_accuracy"]),
        ])
    lines += _table(
        ["System", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc."], rows
    )
    lines += ["", "Cost of each policy:", ""]
    rows = []
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        relative = record.get("relative_to_always_fusion") or {}
        rows.append([
            SYSTEM_LABELS.get(name, name),
            _fmt(record["acquisition"]["average_modalities_activated"], 2),
            _pct(record["acquisition"]["audio_acquisition_rate"]),
            _fmt(record["latency"]["mean_latency_ms_per_sample"], 1),
            _fmt(record["latency"]["latency_per_correct_prediction_ms"], 1),
            _pct(relative.get("macro_f1_retained")) if relative.get("macro_f1_retained")
            else "--",
        ])
    lines += _table(
        ["System", "Modalities/sample", "Acquisition rate", "Latency (ms)",
         "Latency per correct (ms)", "Macro-F1 retained vs full fusion"],
        rows,
    )
    lines += ["", "Per-class F1:", ""]
    rows = []
    for name in order:
        record = report["systems"].get(name)
        if not record:
            continue
        rows.append(
            [SYSTEM_LABELS.get(name, name)]
            + [_fmt(record["per_class_f1"][cls], 3) for cls in CANONICAL_EMOTION_CLASSES]
        )
    lines += _table(["System"] + list(CANONICAL_EMOTION_CLASSES), rows)

    control = report.get("random_acquisition_control")
    dynamic = report["systems"].get("ahsef_dynamic") or report["systems"].get(
        "G_hsig_plus_ugapr"
    )
    if control and dynamic:
        lines += [
            "", "### Control: acquiring at random at the same rate", "",
            f"The router acquired audio for {_pct(control['acquisition_rate'])} of samples.",
            f"Acquiring a random {_pct(control['acquisition_rate'])} instead, over "
            f"{control['repeats']} draws:",
            "",
        ]
        lines += _table(
            ["Policy", "Accuracy", "Macro-F1"],
            [
                ["Random acquisition (mean)", _fmt(control["accuracy"]["mean"]),
                 _fmt(control["macro_f1"]["mean"])],
                ["Random acquisition (95% range)",
                 f"[{_fmt(control['accuracy']['p2.5'])}, {_fmt(control['accuracy']['p97.5'])}]",
                 f"[{_fmt(control['macro_f1']['p2.5'])}, {_fmt(control['macro_f1']['p97.5'])}]"],
                ["**AHSEF dynamic**", f"**{_fmt(dynamic['accuracy'])}**",
                 f"**{_fmt(dynamic['macro_f1'])}**"],
            ],
        )
        lines += ["", control["purpose"], ""]

    routing = report.get("routing")
    if routing:
        lines += ["", "### Routing decisions", ""]
        lines += _table(
            ["Outcome", "Count", "Share"],
            [
                ["STOP", str(routing["stop"]),
                 _pct(1 - (routing["acquisition_rate"] or 0))],
                ["REQUEST_AUDIO", str(routing["request_audio"]),
                 _pct(routing["acquisition_rate"])],
                ["… audio acquired", str(routing["audio_acquired"]), "--"],
                ["… REQUESTED_BUT_UNAVAILABLE",
                 str(routing["requested_but_unavailable"]), "--"],
            ],
        )
        before = routing["uncertainty_before_acquisition"]
        after = routing["uncertainty_after_acquisition"]
        lines += ["", "Uncertainty around the decision:", ""]
        lines += _table(
            ["Quantity", "Value"],
            [
                ["Mean uncertainty, all samples", _fmt(before["all"])],
                ["Mean uncertainty, STOP set", _fmt(before["stopped"])],
                ["Mean uncertainty, REQUEST set (before)", _fmt(before["requested"])],
                ["Mean uncertainty, REQUEST set (after fusion)",
                 _fmt(after["requested_and_acquired"])],
                ["Mean reduction among acquired", _fmt(after["mean_reduction"])],
            ],
        )
        improvement = (dynamic or {}).get("improvement_among_requested") or {}
        if improvement.get("requested"):
            lines += ["", "Did the router spend well?", ""]
            lines += _table(
                ["Quantity", "Value"],
                [
                    ["Acquisitions paid for", str(improvement["requested"])],
                    ["… that fixed a wrong text answer", str(improvement["fixed_by_audio"])],
                    ["… that broke a correct text answer", str(improvement["broken_by_audio"])],
                    ["Net", f"{improvement['net_improvement']:+d}"],
                    ["Improvement rate among paid acquisitions",
                     _pct(improvement["actual_improvement_rate"])],
                ],
            )
    conditional = report.get("conditional_performance")
    if conditional:
        lines += ["", "Conditional performance:", ""]
        lines += _table(
            ["Group", "n", "Accuracy", "Macro-F1"],
            [
                [row["group"], str(row["samples"]), _fmt(row.get("accuracy")),
                 _fmt(row.get("macro_f1"))]
                for row in conditional["groups"] if row.get("samples")
            ],
        )
    quality = report.get("hsig_selection_quality") or report.get(
        "hsig_predicted_vs_observed"
    )
    if quality:
        lines += ["", "HSIG selection quality:", ""]
        lines += _table(
            ["Quantity", "Value"],
            [
                ["AUROC (predicted gain vs `y_gain`)",
                 _fmt(quality["auroc_predicted_gain_vs_y_gain"])],
                ["AUPRC", _fmt(quality["auprc"])],
                ["Spearman ρ (predicted vs observed)",
                 _fmt(quality["spearman_predicted_vs_observed"], 3)],
                ["Mean predicted gain", _fmt(quality["mean_predicted_gain"])],
                ["Mean observed gain", _fmt(quality["mean_observed_gain"])],
                ["Useful acquisitions captured",
                 str(quality["useful_acquisitions_captured"])],
                ["Useful acquisitions missed", str(quality["useful_acquisitions_missed"])],
                ["Recall of useful acquisitions",
                 _pct(quality["recall_of_useful_acquisitions"])],
                ["Precision of acquisition", _pct(quality["precision_of_acquisition"])],
            ],
        )
        lines += ["", quality["note"], ""]

    calibrated = [
        (name, report["systems"][name]["calibration"]) for name in order
        if report["systems"].get(name, {}).get("calibration")
    ]
    if calibrated:
        lines += ["", "Calibration of the routed system's score vector:", ""]
        lines += _table(
            ["System", "ECE", "MCE", "Brier", "NLL", "Mean confidence", "Accuracy"],
            [
                [SYSTEM_LABELS.get(name, name), _fmt(entry["ece"]), _fmt(entry["mce"]),
                 _fmt(entry["brier"]), _fmt(entry["nll"]),
                 _fmt(entry["mean_confidence"]), _fmt(entry["accuracy"])]
                for name, entry in calibrated
            ],
        )
        lines += ["", calibrated[0][1]["caveat"], ""]

    lines += _confusion_matrices(report, order)

    central = report.get("central_result")
    if central:
        lines += ["", "### Performance against modalities activated", ""]
        lines += _table(
            ["System", "Modalities/sample", "Latency (ms)", "Accuracy", "Macro-F1"],
            [
                [SYSTEM_LABELS.get(row["system"], row["system"]),
                 _fmt(row["average_modalities_activated"], 2),
                 _fmt(row["mean_latency_ms"], 1), _fmt(row["accuracy"]),
                 _fmt(row["macro_f1"])]
                for row in central["performance_vs_modalities_activated"]
            ],
        )
        lines += ["", central["reading"], ""]
    return lines


def _confusion_matrices(report: Mapping, order: Sequence[str]) -> list[str]:
    """One matrix per system, rows = true class, columns = predicted class."""
    wanted = [name for name in ("text_only", "always_fusion", "ahsef_dynamic")
              if name in order and report["systems"].get(name)]
    if not wanted:
        return []
    lines = ["", "Confusion matrices (rows = true class, columns = predicted):", ""]
    header = ["true \\ pred"] + [name[:7] for name in CANONICAL_EMOTION_CLASSES]
    for name in wanted:
        matrix = report["systems"][name]["confusion_matrix"]
        lines += ["", f"*{SYSTEM_LABELS.get(name, name)}*", ""]
        lines += _table(
            header,
            [
                [CANONICAL_EMOTION_CLASSES[index]] + [str(int(value)) for value in row]
                for index, row in enumerate(matrix)
            ],
        )
    return lines


def _ablations(report: Mapping | None) -> list[str]:
    order = [
        "text_only", "audio_only", "always_fusion", "D_uncertainty_threshold_only",
        "E_hsig_without_ugapr", "F_ugapr_with_oracle_gain", "G_hsig_plus_ugapr",
    ]
    lines = _systems(report, "6. Ablations (Phase I, validation only)", order)
    if not report:
        return lines
    lines += ["", f"**Reading.** {report['reading']}", ""]

    equivalence = report.get("policy_equivalence")
    if equivalence:
        lines += ["### Which of these are the same policy?", ""]
        lines += _table(
            ["Pair", "Identical", "Agreement", "Both", "Left only", "Right only"],
            [
                [pair, "**yes**" if row["identical"] else "no",
                 _pct(row["agreement"]), str(row["acquired_by_both"]),
                 str(row["acquired_by_left_only"]), str(row["acquired_by_right_only"])]
                for pair, row in equivalence["pairwise"].items()
            ],
        )
        lines += ["", f"**{equivalence['interpretation']}**", ""]

    fusion = report.get("H_fusion_rules") or {}
    available = {k: v for k, v in fusion.items() if isinstance(v, dict) and v.get("available")}
    if available:
        lines += ["### H. Fusion-rule sensitivity", ""]
        lines += _table(
            ["Rule", "Weights", "Always-fusion macro-F1", "Dynamic macro-F1",
             "Dynamic acquisition rate", "Fixed", "Broken"],
            [
                [f"`{name}`"
                 + (" (frozen)" if name == fusion.get("frozen_rule") else ""),
                 ", ".join(f"{k}={v:.2f}" for k, v in entry["weights"].items()),
                 _fmt(entry["always_fusion"]["macro_f1"]),
                 _fmt(entry["dynamic"]["macro_f1"]),
                 _pct(entry["dynamic"]["acquisition_rate"]),
                 str(entry["fixed_by_audio"]), str(entry["broken_by_audio"])]
                for name, entry in available.items()
            ],
        )
        lines += ["", fusion["note"], ""]
    unavailable = report.get("unavailable_audio_experiment") or {}
    if unavailable.get("available"):
        lines += ["### REQUESTED_BUT_UNAVAILABLE, on real audio-less samples", ""]
        lines += [
            f"Source: {unavailable['source']}. n = {unavailable['samples']}.", "",
        ]
        lines += _table(
            ["Outcome", "Count"],
            [
                ["STOP", str(unavailable["stop"])],
                ["REQUEST_AUDIO", str(unavailable["requested"])],
                ["REQUESTED_BUT_UNAVAILABLE",
                 str(unavailable["requested_but_unavailable"])],
                ["Fused predictions produced", str(unavailable["fused_predictions_produced"])],
            ],
        )
        lines += ["", "Verification:", ""]
        for key, value in unavailable["verification"].items():
            lines += [f"- `{key}`: {value}"]
        lines += ["", f"**{unavailable['conclusion']}**", ""]
    return lines


def _claim(test_report: Mapping | None, hsig: Mapping | None) -> list[str]:
    lines = ["## 8. The scientific claim (Phase L)", "", "The defensible claim is:", "",
             f"> {STAGE3_CLAIM}", "", f"{STAGE3_CLAIM_LIMITS}", "",
             "Specifically **not** claimed:", "",
             "- that AHSEF selects the best modality among all modalities. The candidate",
             "  set has one member, so no selection among modalities was tested.",
             "- that the result generalises beyond conversational podcast speech. Both",
             "  splits are drawn entirely from MSP-Podcast, the only corpus in this project",
             "  carrying audio and text on the same record within one split.",
             "- that the LLM's class scores are posteriors. Stage 2 measured their NLL as",
             "  worse than uniform; they are normalised self-reports throughout.",
             ""]
    if hsig and hsig.get("comparison", {}).get("verdict"):
        lines += [
            "On whether the learned estimator beats a simple baseline:", "",
            f"> {hsig['comparison']['verdict']['statement']}", "",
        ]
    if test_report and test_report.get("verdict"):
        verdict = test_report["verdict"]
        lines += [
            "On what the locked test established:", "",
            f"> {verdict['targeting']['answer']}", "",
            f"> {verdict['system_level']['answer']}", "",
            "Read together, the two findings say that the *targeting* component works "
            "and the *end-to-end benefit over a random budget* is not established on "
            "this pool. Both belong in the claim, and neither cancels the other.",
            "",
        ]
    return lines


# ============================================================
# Entry point
# ============================================================

def render_reports(layout: Stage3Layout, docs: Path = DOCS) -> list[Path]:
    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)

    alignment = _load(layout.alignment_report_path)
    oracle_validation = _load(layout.oracle_report_path("validation"))
    oracle_test = _load(layout.oracle_report_path("test"))
    hsig = _load(layout.hsig_report_path)
    frozen = _load(layout.frozen_config_path)
    validation = _load(layout.report_path("validation_experiment.json"))
    ablations = _load(layout.report_path("ablations.json"))
    locked = _load(layout.report_path("locked_test.json"))

    lines = [
        "# AHSEF Stage 3 — results: dynamic Text → Audio acquisition",
        "",
        f"*Experiment `{STAGE3_EXPERIMENT_ID}`. Every number in this document is read "
        f"from a JSON artefact under `experiments/ahsef/{layout.run}/`; the document is "
        f"generated by `python -m src.ahsef.cli.run_stage3_text_audio --stage report` and "
        f"is not written by hand.*",
        "",
        "Method and design decisions: `docs/ahsef_stage3_text_audio.md`.",
        "",
    ]
    lines += _alignment(alignment)
    lines += _oracle(oracle_validation, "validation")
    lines += _hsig(hsig)
    lines += _frozen(frozen)
    lines += _systems(
        validation, "5. Validation experiment (Phase H)",
        ["text_only", "audio_only", "always_fusion", "stage2_gate_plus_audio",
         "ahsef_dynamic"],
    )
    lines += _ablations(ablations)
    lines += _oracle(oracle_test, "test")
    lines += _systems(
        locked, "7. Locked test (Phase K)",
        ["text_only", "audio_only", "always_fusion", "stage2_gate_plus_audio",
         "ahsef_dynamic"],
    )
    if locked and locked.get("verdict"):
        verdict = locked["verdict"]
        lines += ["", "### What the locked test does and does not establish", ""]
        for key, heading in (
            ("targeting", "Did the router acquire on better-than-random samples?"),
            ("system_level",
             "Did that targeting beat acquiring at random at the same rate?"),
            ("against_the_endpoints", "How does it sit against text-only and full fusion?"),
        ):
            block = verdict[key]
            lines += [f"**{heading}**", "", block["answer"], ""]
        lines += [
            "**Validation → test.** " + verdict["validation_to_test_gap"]["note"], "",
        ]
    if locked and locked.get("frozen_config"):
        record = locked["frozen_config"]
        lines += ["", "Lock verification:", ""]
        lines += _table(
            ["Check", "Value"],
            [
                ["Frozen at", record["frozen_at"]],
                ["Threshold reselected on test", _fmt(record["reselected_on_test"])],
                ["HSIG refitted on test", _fmt(record["hsig_refitted_on_test"])],
                ["Fusion weights tuned on test", _fmt(record["fusion_weights_tuned_on_test"])],
                ["Configuration drift checked", _fmt(record["drift_checked"])],
                ["HSIG fingerprint", f"`{record['hsig_fingerprint_sha256'][:16]}`"],
            ],
        )
        lines += [""]
    lines += _claim(locked, hsig)

    path = docs / "ahsef_stage3_results.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return [path]
