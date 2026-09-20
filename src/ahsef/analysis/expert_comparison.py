"""PHASE B-1 / PHASE C -- is the strong audio expert actually better, and does
it give AHSEF more to work with?

Two questions, deliberately separated because they can have different answers.

**B-1: is the expert better?**  Baseline audio vs strong audio on the *identical*
5,550-sample validation split.  The bar for "materially better" is declared in
:data:`MATERIAL_IMPROVEMENT` **before** the numbers are looked at, because a bar
chosen afterwards is not a bar.

**C: does it help the router?**  A better expert is only useful to AHSEF if it
changes what routing can buy on the 509-sample aligned Text+Audio pool.  An
expert could improve in isolation and add nothing there -- if its new correct
answers are the ones Gemma already gets right, the oracle ceiling does not move
and there is no more headroom to exploit.  So Phase C measures the headroom
directly and compares it against Phase A's baseline figure.

One latency subtlety is handled explicitly rather than glossed.  The strong
expert's *measured* forward pass is only the probe head reading cached features;
the wav2vec2 encoder ran hours earlier during extraction.  Reporting that head
latency as the expert's cost would understate it by orders of magnitude, so
:func:`latency_profile` adds the measured per-clip encoder cost from the
extraction provenance and reports both components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.ahsef.calibration import calibration_report
from src.ahsef.evaluation import uncertainty_summary
from src.ahsef.inference import PredictionSet
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

#: The B-1 decision rule, fixed in advance.  Macro-F1 is primary because the
#: split is heavily imbalanced and Stage 2 already showed that accuracy on this
#: corpus rewards predicting the majority class.
MATERIAL_IMPROVEMENT = {
    "primary_metric": "macro_f1",
    "min_macro_f1_gain": 0.03,
    "max_accuracy_regression": 0.0,
    "rule": (
        "The strong expert counts as a material improvement when validation "
        "macro-F1 rises by at least 0.03 absolute AND accuracy does not fall. "
        "Macro-F1 leads because this split is heavily imbalanced (neutral is 37% "
        "of it) and an accuracy-led rule would reward exactly the majority-class "
        "behaviour Stage 2 identified as the problem."
    ),
    "declared": "before any strong-expert result was computed",
    "if_not_met": (
        "Do not proceed to a second expensive encoder. Investigate a lighter "
        "alternative representation, or accept the baseline expert and spend the "
        "remaining effort on the routing policy, which Phase A showed has large "
        "unexploited headroom."
    ),
}


def metric_block(prediction_set: PredictionSet, name: str) -> dict:
    """Every per-class and aggregate number Phase B-1 asks for."""
    frame = prediction_set.frame
    usable = frame["predicted_class"] >= 0
    scored = prediction_set.restricted_to(
        frame.loc[usable, "sample_id"].astype(str).tolist()
    )
    classes = list(prediction_set.class_order)
    metrics = classification_metrics(scored.predictions(), scored.labels(), len(classes))
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    return {
        "system": name,
        "samples": int(len(scored.frame)),
        "excluded_unusable": int((~usable).sum()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "per_class": {
            klass: {
                "support": metrics["support"][index],
                "precision": metrics["per_class_precision"][index],
                "recall": metrics["per_class_recall"][index],
                "f1": metrics["per_class_f1"][index],
            }
            for index, klass in enumerate(classes)
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "class_order": classes,
    }


def calibration_block(prediction_set: PredictionSet, name: str, bins: int = 10) -> dict:
    probabilities = prediction_set.probabilities()
    valid = ~torch.isnan(probabilities).any(dim=1)
    if not bool(valid.any()):
        return {"system": name, "available": False, "reason": "no usable probabilities"}
    return {
        "system": name, "available": True,
        **calibration_report(
            probabilities[valid], prediction_set.labels()[valid], bins, name
        ),
    }


def latency_profile(
    prediction_set: PredictionSet,
    encoder_ms_per_clip: float | None = None,
    encoder_name: str | None = None,
) -> dict:
    """Measured cost per sample, with the frozen encoder's share made explicit.

    For a from-scratch baseline the measured forward pass *is* the cost. For a
    frozen-encoder expert it is not: the encoder ran during extraction and its
    cost is real at deployment even though no training-time timer sees it.
    """
    frame = prediction_set.frame
    measured = float(frame["latency_ms"].mean())
    record = {
        "measured_ms_per_sample": measured,
        "measured_covers": (
            "probe head over cached features only" if encoder_ms_per_clip is not None
            else "feature extraction plus forward pass"
        ),
        "encoder_ms_per_sample": encoder_ms_per_clip,
        "encoder": encoder_name,
        "deployment_ms_per_sample": (
            measured + encoder_ms_per_clip if encoder_ms_per_clip is not None else measured
        ),
        "median_ms": float(frame["latency_ms"].median()),
        "p95_ms": float(np.percentile(frame["latency_ms"].to_numpy(dtype=float), 95)),
    }
    if encoder_ms_per_clip is not None:
        record["note"] = (
            "The frozen encoder is the dominant cost and it is charged here even "
            "though it is cached during experiments. Reporting only the head's "
            "forward pass would understate this expert's real per-sample cost by "
            "roughly three orders of magnitude."
        )
    return record


def encoder_cost_from_provenance(path: Path | str) -> tuple[float | None, str | None]:
    """Measured milliseconds per clip for the frozen encoder, from extraction."""
    path = Path(path)
    if not path.exists():
        return None, None
    record = json.loads(path.read_text(encoding="utf-8"))
    rate = (record.get("this_invocation") or {}).get("rate_per_second")
    name = (record.get("extractor") or {}).get("bundle")
    if not rate:
        return None, name
    return float(1000.0 / rate), name


# ============================================================
# B-1 verdict
# ============================================================

def improvement_verdict(baseline: Mapping, strong: Mapping) -> dict:
    """Apply :data:`MATERIAL_IMPROVEMENT` and say plainly what follows."""
    macro_gain = strong["macro_f1"] - baseline["macro_f1"]
    accuracy_delta = strong["accuracy"] - baseline["accuracy"]
    # A gain of exactly the bar must pass. Subtracting two floats that differ by
    # the bar can land a few ulps below it (0.29 - 0.26 = 0.0299999...), and a
    # pre-declared threshold that rejects the value it names is a bug, not a rule.
    tolerance = 1e-9
    meets_macro = macro_gain >= MATERIAL_IMPROVEMENT["min_macro_f1_gain"] - tolerance
    meets_accuracy = (
        accuracy_delta >= -MATERIAL_IMPROVEMENT["max_accuracy_regression"] - tolerance
    )
    material = bool(meets_macro and meets_accuracy)

    per_class = {}
    for klass in baseline["per_class"]:
        left, right = baseline["per_class"][klass], strong["per_class"][klass]
        per_class[klass] = {
            "support": right["support"],
            "baseline_f1": left["f1"], "strong_f1": right["f1"],
            "f1_delta": right["f1"] - left["f1"],
            "baseline_recall": left["recall"], "strong_recall": right["recall"],
            "recall_delta": right["recall"] - left["recall"],
        }
    improved = [k for k, v in per_class.items() if v["f1_delta"] > 0.01]
    regressed = [k for k, v in per_class.items() if v["f1_delta"] < -0.01]

    return {
        "rule": dict(MATERIAL_IMPROVEMENT),
        "macro_f1_gain": macro_gain,
        "accuracy_delta": accuracy_delta,
        "weighted_f1_delta": strong["weighted_f1"] - baseline["weighted_f1"],
        "balanced_accuracy_delta": (
            strong["balanced_accuracy"] - baseline["balanced_accuracy"]
        ),
        "meets_macro_f1_bar": bool(meets_macro),
        "meets_accuracy_bar": bool(meets_accuracy),
        "material_improvement": material,
        "classes_improved": improved,
        "classes_regressed": regressed,
        "per_class": per_class,
        "decision": (
            "PROCEED -- the strong expert clears the pre-declared bar; carry it into "
            "the AHSEF routing redesign."
            if material else
            "DO NOT PROCEED to another expensive encoder. " +
            MATERIAL_IMPROVEMENT["if_not_met"]
        ),
        "statement": (
            f"Validation macro-F1 moved {macro_gain:+.4f} "
            f"({baseline['macro_f1']:.4f} -> {strong['macro_f1']:.4f}) and accuracy "
            f"{accuracy_delta:+.4f} ({baseline['accuracy']:.4f} -> "
            f"{strong['accuracy']:.4f}) on {strong['samples']} identical validation "
            f"samples. The pre-declared bar was +"
            f"{MATERIAL_IMPROVEMENT['min_macro_f1_gain']:.2f} macro-F1 with no "
            f"accuracy regression, so this "
            + ("MEETS" if material else "DOES NOT MEET") + " it."
        ),
    }


# ============================================================
# Phase C: does the router gain anything?
# ============================================================

def pair_headroom(
    left: PredictionSet, right: PredictionSet, sample_ids: Sequence[str]
) -> dict:
    """Oracle ceiling and correction opportunities for one aligned pair."""
    a, b = left.restricted_to(sample_ids), right.restricted_to(sample_ids)
    truth = a.labels().numpy()
    if not np.array_equal(truth, b.labels().numpy()):
        raise ValueError("The two prediction sets disagree about the true class")
    left_ok = a.predictions().numpy() == truth
    right_ok = b.predictions().numpy() == truth
    best_single = max(float(left_ok.mean()), float(right_ok.mean()))
    oracle = float((left_ok | right_ok).mean())
    return {
        "samples": len(sample_ids),
        "left_accuracy": float(left_ok.mean()),
        "right_accuracy": float(right_ok.mean()),
        "best_single_accuracy": best_single,
        "oracle_either_accuracy": oracle,
        "headroom_over_best_single": oracle - best_single,
        "disagreement_rate": float(
            (a.predictions().numpy() != b.predictions().numpy()).mean()
        ),
        "left_fixes_right": int((left_ok & ~right_ok).sum()),
        "right_fixes_left": int((right_ok & ~left_ok).sum()),
        "both_wrong": int((~left_ok & ~right_ok).sum()),
    }


#: Where Phase A recorded the complementarity it measured. Phase C reads its
#: reference from this artefact rather than carrying a copied constant, so the
#: comparison cannot silently drift away from the number Phase A actually wrote.
PHASE_A_ARTEFACT = (
    Path("experiments") / "ahsef" / "analysis" / "classwise" / "classwise_validation.json"
)
PHASE_A_PAIR = "audio+text_llm"


def load_phase_a_headroom(
    path: Path | str = PHASE_A_ARTEFACT, pair: str = PHASE_A_PAIR
) -> dict | None:
    """The Phase A complementarity block for one pair, or ``None`` if absent.

    Returning ``None`` rather than raising is deliberate: Phase C's own
    measurements stand on their own, and a missing Phase A artefact should cost
    the report its historical comparison, not the whole run.
    """
    path = Path(path)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    block = ((record.get("complementarity") or {}).get("pairs") or {}).get(pair)
    return dict(block) if block else None


def _phase_a_block(
    phase_a: Mapping | None, baseline_pair: Mapping, strong_pair: Mapping
) -> dict:
    """Compare both experts against what Phase A recorded on this pool.

    The baseline side is a *reproduction check*: Phase C recomputes the same
    quantity Phase A measured, so if the two disagree the pool changed under us
    and every historical statement in the report is suspect. It is reported
    either way rather than assumed.
    """
    if phase_a is None:
        return {
            "available": False,
            "reason": f"no Phase A artefact at {PHASE_A_ARTEFACT}",
        }
    # Phase A named the pair audio+text_llm, so 'left' is audio and 'right' Gemma.
    same_pool = int(phase_a["samples"]) == int(strong_pair["samples"])
    reproduced = same_pool and abs(
        baseline_pair["oracle_either_accuracy"] - phase_a["oracle_either_accuracy"]
    ) < 1e-6
    return {
        "available": True,
        "pair": PHASE_A_PAIR,
        "source": str(PHASE_A_ARTEFACT),
        "recorded": {
            "samples": int(phase_a["samples"]),
            "audio_accuracy": phase_a["left_accuracy"],
            "text_llm_accuracy": phase_a["right_accuracy"],
            "oracle_either_accuracy": phase_a["oracle_either_accuracy"],
            "headroom_over_best_single": phase_a["headroom_over_best_single"],
            "disagreement_rate": phase_a["disagreement_rate"],
        },
        "pool_matches_phase_a": bool(same_pool),
        "baseline_reproduces_phase_a": bool(reproduced),
        "strong_vs_phase_a": {
            "oracle_delta": (
                strong_pair["oracle_either_accuracy"] - phase_a["oracle_either_accuracy"]
            ),
            "headroom_delta": (
                strong_pair["headroom_over_best_single"]
                - phase_a["headroom_over_best_single"]
            ),
            "disagreement_delta": (
                strong_pair["disagreement_rate"] - phase_a["disagreement_rate"]
            ),
            "audio_accuracy_delta": (
                strong_pair["left_accuracy"] - phase_a["left_accuracy"]
            ),
        },
        "caveat": None if same_pool else (
            f"Phase A measured {int(phase_a['samples'])} samples and Phase C measured "
            f"{int(strong_pair['samples'])}; the deltas against Phase A are across "
            f"different pools and are indicative only. The baseline-vs-strong "
            f"comparison in this same record is on identical samples and is the one "
            f"to rely on."
        ),
    }


def headroom_comparison(
    baseline_pair: Mapping, strong_pair: Mapping, phase_a: Mapping | None = None
) -> dict:
    """Did swapping in the strong expert give the router more to exploit?"""
    delta_oracle = (
        strong_pair["oracle_either_accuracy"] - baseline_pair["oracle_either_accuracy"]
    )
    delta_headroom = (
        strong_pair["headroom_over_best_single"]
        - baseline_pair["headroom_over_best_single"]
    )
    phase_a_block = _phase_a_block(phase_a, baseline_pair, strong_pair)
    return {
        "baseline_audio": dict(baseline_pair),
        "strong_audio": dict(strong_pair),
        "oracle_accuracy_delta": delta_oracle,
        "headroom_delta": delta_headroom,
        "phase_a": phase_a_block,
        "more_room_for_routing": bool(delta_oracle > 0),
        "statement": (
            f"Against Gemma on the aligned pool, the oracle ceiling moved "
            f"{delta_oracle:+.4f} ({baseline_pair['oracle_either_accuracy']:.4f} -> "
            f"{strong_pair['oracle_either_accuracy']:.4f}) and the headroom over the "
            f"better single expert moved {delta_headroom:+.4f}. "
            + (
                "There is more for the router to exploit than in Phase A."
                if delta_oracle > 0 else
                "There is NOT more for the router to exploit: the strong expert's new "
                "correct answers largely overlap the ones Gemma already gets right, so "
                "a better expert does not by itself widen the routing opportunity."
            )
        ),
        "phase_a_statement": _phase_a_statement(phase_a_block, strong_pair),
        "why_this_matters": (
            "AHSEF can only convert disagreement into accuracy. An expert that "
            "improves in isolation but agrees with Gemma more often gives the router "
            "LESS to do, not more, so expert quality and routing opportunity have to "
            "be measured separately."
        ),
    }


def _phase_a_statement(block: Mapping, strong_pair: Mapping) -> str:
    """Answer the Phase A question in one sentence, in the report itself."""
    if not block.get("available"):
        return "No Phase A artefact was found, so no historical comparison is made."
    recorded = block["recorded"]
    delta = block["strong_vs_phase_a"]
    verdict = (
        "the strong expert widens it"
        if delta["oracle_delta"] > 0 else
        "the strong expert does NOT widen it"
    )
    caveat = f" {block['caveat']}" if block.get("caveat") else ""
    reproduction = (
        "" if block["baseline_reproduces_phase_a"] else
        " NOTE: the recomputed baseline headroom does not reproduce Phase A's, so the "
        "pool or the exports have changed since Phase A and the historical delta "
        "should not be quoted without checking why."
    )
    return (
        f"Phase A recorded oracle {recorded['oracle_either_accuracy']:.4f} against "
        f"Gemma's {recorded['text_llm_accuracy']:.4f} with "
        f"{recorded['disagreement_rate']:.1%} disagreement on "
        f"{recorded['samples']} samples. With the strong expert the oracle is "
        f"{strong_pair['oracle_either_accuracy']:.4f} "
        f"({delta['oracle_delta']:+.4f}) and disagreement is "
        f"{strong_pair['disagreement_rate']:.1%} "
        f"({delta['disagreement_delta']:+.1%}), so on the Phase A question -- does a "
        f"stronger audio expert increase the complementarity headroom -- "
        f"{verdict}.{caveat}{reproduction}"
    )
