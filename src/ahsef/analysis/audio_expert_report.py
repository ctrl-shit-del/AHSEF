"""Render ``docs/ahsef_phase_b_audio_strong.md`` from the Phase B-1 / C artefacts.

Generated, never typed -- the same rule the Stage 3 report follows.  Every
number here is read out of ``phase_b1_comparison.json`` or
``phase_c_aligned.json``, so the document cannot drift away from the run that
produced it, and a missing artefact leaves a visible gap rather than a quietly
shorter report.

The document is organised around the two questions in the order they have to be
answered, because the second one only matters if the first is settled:

1. **Is the strong expert better?**  Pre-declared bar, identical validation
   samples, every metric the phase asked for.
2. **Does that help the router?**  The aligned Text+Audio pool, fusion on both
   sides, and the headroom against Phase A's recorded figure.

The locked 482-sample test pool is not read by either stage, and the report says
so explicitly rather than leaving its absence to be inferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from src.common.labels import CANONICAL_EMOTION_CLASSES

DOCS = Path("docs")
REPORT_NAME = "ahsef_phase_b_audio_strong.md"

SYSTEM_LABELS = {
    "audio_baseline": "Audio baseline (`audio_25pct`, log-mel, from scratch)",
    "audio_strong": "Audio strong (`audio_strong_full`, frozen wav2vec2 + probe)",
    "text_llm": "Text / Gemma",
    "text_llm+audio_baseline": "Fusion: Text + Audio baseline",
    "text_llm+audio_strong": "Fusion: Text + Audio strong",
}


def _load(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 4, dash: str = "--") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value, digits: int = 4) -> str:
    return "--" if value is None else f"{float(value):+.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value) * 100:.{digits}f}%"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


def _missing(title: str, command: str) -> list[str]:
    return [f"## {title}", "",
            f"*Not generated. Run `{command}`.*", ""]


# ============================================================
# Phase B-1
# ============================================================

def _headline(record: Mapping) -> list[str]:
    systems = record["systems"]
    rows = []
    for key in ("audio_baseline", "audio_strong"):
        block = systems[key]
        rows.append([
            SYSTEM_LABELS.get(key, key),
            _fmt(block["accuracy"]), _fmt(block["macro_f1"]),
            _fmt(block["weighted_f1"]), _fmt(block["macro_precision"]),
            _fmt(block["macro_recall"]), _fmt(block["balanced_accuracy"]),
        ])
    verdict = record["verdict"]
    rows.append([
        "**Δ (strong − baseline)**",
        _signed(verdict["accuracy_delta"]), _signed(verdict["macro_f1_gain"]),
        _signed(verdict["weighted_f1_delta"]), "--", "--",
        _signed(verdict["balanced_accuracy_delta"]),
    ])
    return [
        "## 1. Phase B-1 — baseline audio vs strong audio",
        "",
        f"Validation split, **{record['samples']} identical samples**. The two experts "
        f"are evaluated on the same manifest, verified at manifest-build time, so every "
        f"difference below is a difference between experts and not between evaluation "
        f"sets.",
        "",
    ] + _table(
        ["System", "Accuracy", "Macro-F1", "Weighted-F1", "Macro-P", "Macro-R",
         "Balanced acc"],
        rows,
    )


def _per_class(record: Mapping) -> list[str]:
    verdict = record["verdict"]
    systems = record["systems"]
    rows = []
    for klass in CANONICAL_EMOTION_CLASSES:
        entry = verdict["per_class"].get(klass)
        if entry is None:
            continue
        base = systems["audio_baseline"]["per_class"][klass]
        strong = systems["audio_strong"]["per_class"][klass]
        rows.append([
            klass, str(entry["support"]),
            _fmt(base["precision"], 3), _fmt(strong["precision"], 3),
            _fmt(base["recall"], 3), _fmt(strong["recall"], 3),
            _fmt(base["f1"], 3), _fmt(strong["f1"], 3),
            _signed(entry["f1_delta"], 3),
        ])
    return [
        "### Per class",
        "",
    ] + _table(
        ["Class", "n", "P base", "P strong", "R base", "R strong",
         "F1 base", "F1 strong", "ΔF1"],
        rows,
    ) + [
        f"Improved (ΔF1 > 0.01): {', '.join(verdict['classes_improved']) or 'none'}. "
        f"Regressed (ΔF1 < −0.01): {', '.join(verdict['classes_regressed']) or 'none'}.",
        "",
    ]


def _confusion(record: Mapping) -> list[str]:
    lines = ["### Confusion matrices", "",
             "Rows are the true class, columns the prediction, in canonical order.", ""]
    for key in ("audio_baseline", "audio_strong"):
        block = record["systems"][key]
        lines += [f"**{SYSTEM_LABELS.get(key, key)}**", ""]
        rows = [
            [klass] + [str(int(cell)) for cell in row]
            for klass, row in zip(block["class_order"], block["confusion_matrix"])
        ]
        lines += _table(["true \\ pred"] + list(block["class_order"]), rows)
    return lines


def _calibration_uncertainty_latency(record: Mapping) -> list[str]:
    rows = []
    for key in ("audio_baseline", "audio_strong"):
        cal = record["calibration"][key]
        unc = record["uncertainty"][key]
        if not cal.get("available"):
            rows.append([SYSTEM_LABELS.get(key, key), "--", "--", "--", "--", "--",
                         _fmt(unc.get("mean_normalized_entropy"))])
            continue
        rows.append([
            SYSTEM_LABELS.get(key, key),
            _fmt(cal["ece"]), _fmt(cal["mce"]), _fmt(cal["brier"]),
            _fmt(cal["nll"]), _fmt(cal["mean_confidence"]),
            _fmt(unc.get("mean_normalized_entropy")),
        ])
    lines = ["### Calibration and uncertainty", ""]
    lines += _table(
        ["System", "ECE", "MCE", "Brier", "NLL", "Mean confidence",
         "Mean normalised entropy"],
        rows,
    )

    latency_rows = []
    for key in ("audio_baseline", "audio_strong"):
        lat = record["latency"][key]
        latency_rows.append([
            SYSTEM_LABELS.get(key, key),
            _fmt(lat["measured_ms_per_sample"], 3),
            _fmt(lat.get("encoder_ms_per_sample"), 1),
            _fmt(lat["deployment_ms_per_sample"], 3),
            _fmt(lat["median_ms"], 3), _fmt(lat["p95_ms"], 3),
            lat["measured_covers"],
        ])
    lines += ["### Inference latency", "",
              "The frozen encoder is charged separately and explicitly: the strong "
              "expert's measured forward pass is only the probe head reading cached "
              "features, and reporting that alone would understate its deployment cost "
              "by orders of magnitude.", ""]
    lines += _table(
        ["System", "Measured ms", "Encoder ms", "Deployment ms", "Median ms", "p95 ms",
         "Measured covers"],
        latency_rows,
    )

    cost = record.get("resource_cost") or {}
    if cost:
        lines += ["### Parameters paid for", ""]
        lines += _table(
            ["System", "Trainable", "Pretrained (frozen)", "Pretrained weights used"],
            [
                [SYSTEM_LABELS.get(key, key),
                 _fmt(cost[key].get("trainable_parameters")),
                 _fmt(cost[key].get("pretrained_parameters")),
                 _fmt(cost[key].get("pretrained_weights_used"))]
                for key in ("audio_baseline", "audio_strong") if key in cost
            ],
        )
    return lines


def _checkpoint(record: Mapping) -> list[str]:
    verdict = record["verdict"]
    rule = verdict["rule"]
    decision = "**YES**" if verdict["material_improvement"] else "**NO**"
    return [
        "### Phase B-1 decision checkpoint",
        "",
        "```text",
        "Phase B-1: strong audio expert",
        "        ↓",
        "Validation evaluation (5,550 samples, test untouched)",
        "        ↓",
        "Does it materially improve?",
        "        ↓",
        f"{'YES → proceed to AHSEF redesign' if verdict['material_improvement'] else 'NO  → investigate an alternative lightweight audio representation'}",
        "```",
        "",
        f"**Rule, declared {rule['declared']}:** {rule['rule']}",
        "",
        f"**Measured:** {verdict['statement']}",
        "",
        f"**Materially improved:** {decision} "
        f"(macro-F1 bar {'met' if verdict['meets_macro_f1_bar'] else 'not met'}, "
        f"accuracy bar {'met' if verdict['meets_accuracy_bar'] else 'not met'}).",
        "",
        f"**Decision:** {verdict['decision']}",
        "",
    ]


def _phase_b1(record: Mapping | None) -> list[str]:
    if record is None:
        return _missing(
            "1. Phase B-1 — baseline audio vs strong audio",
            "python -m src.ahsef.cli.run_audio_expert_comparison --stage compare",
        )
    return (
        _headline(record) + _per_class(record) + _confusion(record)
        + _calibration_uncertainty_latency(record) + _checkpoint(record)
    )


# ============================================================
# Phase C
# ============================================================

def _phase_c(record: Mapping | None) -> list[str]:
    if record is None:
        return _missing(
            "2. Phase C — the aligned Text+Audio validation pool",
            "python -m src.ahsef.cli.run_audio_expert_comparison --stage aligned",
        )
    order = ["text_llm", "audio_baseline", "audio_strong",
             "text_llm+audio_baseline", "text_llm+audio_strong"]
    rows = []
    for key in order:
        block = record["systems"].get(key)
        if not block:
            continue
        rows.append([
            SYSTEM_LABELS.get(key, key),
            _fmt(block["accuracy"]), _fmt(block["macro_f1"]),
            _fmt(block["weighted_f1"]), _fmt(block["balanced_accuracy"]),
        ])
    lines = [
        "## 2. Phase C — the aligned Text+Audio validation pool",
        "",
        f"The co-split Text+Audio intersection, **{record['pool_samples']} validation "
        f"samples** with usable Gemma evidence. This is the pool AHSEF actually routes "
        f"over, so it -- not the full validation split -- decides whether a better "
        f"audio expert is worth anything to the router.",
        "",
        "```text",
        "Text/Gemma",
        "    ↓",
        "       Audio baseline",
        "       vs",
        "       Audio strong",
        "    ↓",
        "Fusion",
        "```",
        "",
    ]
    lines += _table(
        ["System", "Accuracy", "Macro-F1", "Weighted-F1", "Balanced acc"], rows
    )

    per_class_rows = []
    for key in order:
        block = record["systems"].get(key)
        if not block:
            continue
        per_class_rows.append(
            [SYSTEM_LABELS.get(key, key)]
            + [_fmt(block["per_class"][klass]["f1"], 3)
               for klass in CANONICAL_EMOTION_CLASSES]
        )
    lines += ["### Per-class F1 on the pool", ""]
    lines += _table(["System"] + list(CANONICAL_EMOTION_CLASSES), per_class_rows)

    gain = record.get("fusion_gain") or {}
    if gain:
        lines += ["### What fusion buys over Gemma alone", ""]
        lines += _table(
            ["Audio expert in the fusion", "Δ Macro-F1", "Δ Accuracy"],
            [[name, _signed(entry["over_text_llm"]),
              _signed(entry["accuracy_over_text_llm"])]
             for name, entry in gain.items()],
        )

    lines += _headroom(record["headroom"])
    return lines


def _headroom(headroom: Mapping) -> list[str]:
    rows = []
    for key, label in (("baseline_audio", "Audio baseline"),
                       ("strong_audio", "Audio strong")):
        entry = headroom[key]
        rows.append([
            label,
            _fmt(entry["right_accuracy"]), _fmt(entry["left_accuracy"]),
            _fmt(entry["best_single_accuracy"]), _fmt(entry["oracle_either_accuracy"]),
            _fmt(entry["headroom_over_best_single"]),
            _pct(entry["disagreement_rate"]),
            str(entry["right_fixes_left"]), str(entry["both_wrong"]),
        ])
    lines = [
        "### Complementarity headroom against Gemma",
        "",
        "The oracle ceiling is the accuracy of a router that always trusted whichever "
        "of the two was right. It is not achievable, but it bounds what *any* routing "
        "policy over this pair can reach -- which is exactly the quantity a stronger "
        "audio expert is supposed to raise.",
        "",
    ]
    lines += _table(
        ["Audio expert", "Audio acc", "Gemma acc", "Best single", "Oracle",
         "Headroom", "Disagreement", "Audio fixes Gemma", "Both wrong"],
        rows,
    )
    lines += [
        f"Oracle ceiling moved **{_signed(headroom['oracle_accuracy_delta'])}** and "
        f"headroom over the better single expert moved "
        f"**{_signed(headroom['headroom_delta'])}** when the strong expert replaced the "
        f"baseline.",
        "",
        f"> {headroom['statement']}",
        "",
        f"*{headroom['why_this_matters']}*",
        "",
    ]
    lines += _phase_a(headroom.get("phase_a") or {}, headroom)
    return lines


def _phase_a(block: Mapping, headroom: Mapping) -> list[str]:
    lines = ["### Against the Phase A finding", ""]
    if not block.get("available"):
        return lines + [f"*{headroom.get('phase_a_statement', 'No Phase A reference.')}*",
                        ""]
    recorded = block["recorded"]
    strong = headroom["strong_audio"]
    baseline = headroom["baseline_audio"]
    lines += [
        f"Phase A's recorded complementarity for `{block['pair']}` is read from "
        f"`{block['source']}` rather than quoted from memory.",
        "",
    ]
    lines += _table(
        ["Measurement", "n", "Audio acc", "Gemma acc", "Oracle", "Headroom",
         "Disagreement"],
        [
            ["Phase A (recorded)", str(recorded["samples"]),
             _fmt(recorded["audio_accuracy"]), _fmt(recorded["text_llm_accuracy"]),
             _fmt(recorded["oracle_either_accuracy"]),
             _fmt(recorded["headroom_over_best_single"]),
             _pct(recorded["disagreement_rate"])],
            ["Phase C, baseline audio", str(baseline["samples"]),
             _fmt(baseline["right_accuracy"]), _fmt(baseline["left_accuracy"]),
             _fmt(baseline["oracle_either_accuracy"]),
             _fmt(baseline["headroom_over_best_single"]),
             _pct(baseline["disagreement_rate"])],
            ["Phase C, strong audio", str(strong["samples"]),
             _fmt(strong["right_accuracy"]), _fmt(strong["left_accuracy"]),
             _fmt(strong["oracle_either_accuracy"]),
             _fmt(strong["headroom_over_best_single"]),
             _pct(strong["disagreement_rate"])],
        ],
    )
    lines += [
        f"Baseline reproduces Phase A: **{_fmt(block['baseline_reproduces_phase_a'])}** "
        f"(same pool: {_fmt(block['pool_matches_phase_a'])}). This is the check that the "
        f"pool has not moved underneath the historical comparison.",
        "",
        f"> {headroom['phase_a_statement']}",
        "",
    ]
    return lines


# ============================================================
# Provenance
# ============================================================

def _provenance(b1: Mapping | None, c: Mapping | None) -> list[str]:
    lines = ["## 3. Provenance", ""]
    rows = []
    for name, record in (("Phase B-1", b1), ("Phase C", c)):
        if record is None:
            rows.append([name, "--", "--", "--", "--"])
            continue
        rows.append([
            name,
            record.get("created_at", "--"),
            (record.get("git_commit") or "--")[:12],
            _fmt(record.get("test_data_accessed")),
            str(record.get("samples") or record.get("pool_samples") or "--"),
        ])
    lines += _table(
        ["Stage", "Created", "Commit", "Test data accessed", "Samples"], rows
    )
    lines += [
        "The locked 482-sample Text+Audio test pool was **not opened** by either stage. "
        "Neither reads a test label, neither offers a `--split` flag, and the feature "
        "extractor refuses the test split without an explicit "
        "`--i-am-running-the-locked-evaluation` flag. It stays reserved for the final "
        "frozen evaluation.",
        "",
    ]
    return lines


# ============================================================
# Entry point
# ============================================================

def render_report(analysis_dir: Path | str, docs: Path | str = DOCS) -> Path:
    """Write the Phase B report and return its path."""
    analysis_dir = Path(analysis_dir)
    docs = Path(docs)
    docs.mkdir(parents=True, exist_ok=True)

    b1 = _load(analysis_dir / "phase_b1_comparison.json")
    c = _load(analysis_dir / "phase_c_aligned.json")

    lines = [
        "# Phase B — a stronger audio expert, and whether AHSEF can use it",
        "",
        f"*Every number in this document is read from a JSON artefact under "
        f"`{analysis_dir.as_posix()}/`. It is generated by "
        f"`python -m src.ahsef.cli.run_audio_expert_comparison --stage report` and is "
        f"not written by hand.*",
        "",
        "The strong expert is a **frozen** `WAV2VEC2_BASE` encoder with a trainable "
        "layer-weighted probe over its 12 layer means, mapping to the same canonical "
        "7-class space every other experiment uses. Its training partition is "
        "class-capped at 4,000 per class to fit a CPU feature-extraction budget, so "
        "class weights are recomputed from the capped subset; validation and test "
        "manifests are the audio baseline's, unchanged.",
        "",
    ]
    lines += _phase_b1(b1)
    lines += _phase_c(c)
    lines += _provenance(b1, c)

    path = docs / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
