"""PHASE A -- class-wise modality analysis.

The question this module exists to answer is *modality specialization*: is there
a class that one modality handles and another does not?  That question is easy
to answer badly, and the bad answer is the one a naive table produces.

**The trap.** The five frozen baselines were sampled independently from
different corpora.  The image baseline is scored on AffectNet+ / FERPlus /
RAF-DB faces; the audio baseline on 93% MSP-Podcast speech; the video baseline
on MELD clips.  Putting their per-class F1 scores in one table and reading off
"image is best at happy" compares *datasets*, not modalities -- the image model
might simply have been given an easier corpus for that class.  Stage 1 measured
that most modality pairs share no sample at all, so for most of the table there
is no shared ground to stand on.

So this module reports two things, and keeps them apart on purpose:

:func:`modality_profiles`
    Per-modality, per-class precision / recall / F1 / support / confusion
    matrix, each on its **own** evaluation split.  Every record carries
    ``comparable_across_modalities: False`` and names the corpora it came from.
    These profiles describe what each expert does on the data it was given.

:func:`aligned_comparison`
    Head-to-head, per class, on the **genuinely aligned** co-split pools only
    (Stage 1 found exactly two: audio+text and text+video).  Here the samples
    are the same, the labels are the same, and a claim like "audio is stronger
    than text on angry" has a referent.  This is also where complementarity --
    A-right/B-wrong, both-wrong, correction opportunity -- is computed, because
    those quantities are undefined for samples the two models never shared.

A specialization claim is emitted only from the second function, and
:func:`specialization_verdicts` refuses to rank a class it has no aligned
evidence for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from src.ahsef.inference import PredictionSet
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.training.metrics import classification_metrics

#: Modalities that solve the 7-class emotion task and may therefore appear in
#: the same table. Physiology is excluded by task, not by preference.
EMOTION_MODALITIES: tuple[str, ...] = ("audio", "text", "image", "video")

#: Pools Stage 1 verified as genuinely co-split. Declared, not discovered, so a
#: future edit that adds a pair has to justify it.
ALIGNED_PAIRS: tuple[tuple[str, str], ...] = (("audio", "text"), ("text", "video"))

NON_COMPARABLE_NOTE = (
    "These per-class figures come from this modality's OWN evaluation split. The "
    "five baselines were sampled independently from different corpora, so a "
    "per-class score here is not comparable with the same class's score under "
    "another modality -- that comparison would measure the corpora, not the "
    "modalities. Head-to-head class comparison lives in the aligned-pair section, "
    "which is the only place this project has shared samples."
)


class AnalysisError(RuntimeError):
    """Raised when an analysis would compare quantities that are not comparable."""


# ============================================================
# Per-modality profiles
# ============================================================

def modality_profile(prediction_set: PredictionSet, name: str | None = None) -> dict:
    """Per-class precision / recall / F1 / support and the confusion matrix.

    Reported on this modality's own split, and labelled as such.
    """
    frame = prediction_set.frame
    scored = frame["predicted_class"] >= 0
    if not bool(scored.any()):
        raise AnalysisError(
            f"{prediction_set.modality}/{prediction_set.split} has no usable "
            f"prediction; there is nothing to profile."
        )
    subset = prediction_set.restricted_to(
        frame.loc[scored, "sample_id"].astype(str).tolist()
    )
    classes = list(prediction_set.class_order)
    metrics = classification_metrics(
        subset.predictions(), subset.labels(), len(classes)
    )
    datasets = (
        subset.frame["dataset"].value_counts().sort_index().to_dict()
        if "dataset" in subset.frame.columns else {}
    )
    return {
        "modality": name or prediction_set.modality,
        "split": prediction_set.split,
        "task": "wesad_state_3class" if len(classes) == 3 else "emotion_7class",
        "class_order": classes,
        "samples": int(len(subset.frame)),
        "excluded_unusable": int((~scored).sum()),
        "datasets": {str(k): int(v) for k, v in datasets.items()},
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": _balanced_accuracy(metrics),
        "per_class": {
            name_: {
                "support": metrics["support"][index],
                "precision": metrics["per_class_precision"][index],
                "recall": metrics["per_class_recall"][index],
                "f1": metrics["per_class_f1"][index],
            }
            for index, name_ in enumerate(classes)
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "comparable_across_modalities": False,
        "note": NON_COMPARABLE_NOTE,
        "provenance": {
            key: prediction_set.meta.get(key)
            for key in ("experiment", "iteration", "checkpoint_sha256", "kind", "model")
            if key in prediction_set.meta
        },
    }


def _balanced_accuracy(metrics: Mapping) -> float:
    recalls = [
        value for value, support in zip(metrics["per_class_recall"], metrics["support"])
        if support > 0
    ]
    return float(np.mean(recalls)) if recalls else 0.0


def modality_profiles(sets: Mapping[str, PredictionSet]) -> dict:
    """Profiles for every supplied modality, grouped by the task each solves."""
    profiles = {name: modality_profile(item, name) for name, item in sets.items()}
    by_task: dict[str, list[str]] = {}
    for name, record in profiles.items():
        by_task.setdefault(record["task"], []).append(name)
    return {
        "profiles": profiles,
        "tasks": {task: sorted(names) for task, names in by_task.items()},
        "comparable_across_modalities": False,
        "why_not": NON_COMPARABLE_NOTE,
        "separate_label_spaces": {
            task: sorted(names) for task, names in by_task.items() if len(by_task) > 1
        },
    }


# ============================================================
# Aligned head-to-head
# ============================================================

@dataclass(frozen=True)
class PairOutcome:
    """The four-way outcome split for one modality pair over shared samples."""

    both_correct: int
    left_only: int
    right_only: int
    both_wrong: int

    @property
    def total(self) -> int:
        return self.both_correct + self.left_only + self.right_only + self.both_wrong

    def to_dict(self) -> dict:
        total = self.total or 1
        return {
            "both_correct": self.both_correct,
            "left_correct_right_wrong": self.left_only,
            "right_correct_left_wrong": self.right_only,
            "both_wrong": self.both_wrong,
            "samples": self.total,
            "both_correct_rate": self.both_correct / total,
            "left_only_rate": self.left_only / total,
            "right_only_rate": self.right_only / total,
            "both_wrong_rate": self.both_wrong / total,
        }


def _outcome(left_correct: np.ndarray, right_correct: np.ndarray) -> PairOutcome:
    return PairOutcome(
        both_correct=int((left_correct & right_correct).sum()),
        left_only=int((left_correct & ~right_correct).sum()),
        right_only=int((right_correct & ~left_correct).sum()),
        both_wrong=int((~left_correct & ~right_correct).sum()),
    )


def aligned_comparison(
    left: PredictionSet,
    right: PredictionSet,
    sample_ids: Sequence[str],
    left_name: str | None = None,
    right_name: str | None = None,
) -> dict:
    """Head-to-head class-wise comparison on samples both models actually saw.

    ``sample_ids`` must come from
    :meth:`src.ahsef.identity.AlignmentIndex.fusion_pool`, so the pool is
    already co-split, contamination-checked and label-agreed.  Every quantity
    below is defined only because the two models are describing the same
    recordings.
    """
    if tuple(left.class_order) != tuple(right.class_order):
        raise AnalysisError(
            f"Refusing to compare {left.modality!r} and {right.modality!r}: their "
            f"class spaces differ ({list(left.class_order)} vs "
            f"{list(right.class_order)}). A 'stress' F1 is not an 'angry' F1."
        )
    if not sample_ids:
        raise AnalysisError("Refusing to compare over an empty aligned pool")

    a, b = left.restricted_to(sample_ids), right.restricted_to(sample_ids)
    if a.sample_ids() != b.sample_ids():
        raise AnalysisError("The two prediction sets did not reindex identically")
    truth = a.labels().numpy()
    if not np.array_equal(truth, b.labels().numpy()):
        raise AnalysisError(
            "The two prediction sets disagree about the true class of pooled "
            "samples; they are not describing the same recordings."
        )

    left_pred, right_pred = a.predictions().numpy(), b.predictions().numpy()
    left_ok, right_ok = left_pred == truth, right_pred == truth
    classes = list(left.class_order)

    left_metrics = classification_metrics(a.predictions(), a.labels(), len(classes))
    right_metrics = classification_metrics(b.predictions(), b.labels(), len(classes))

    per_class = {}
    for index, name in enumerate(classes):
        mask = truth == index
        count = int(mask.sum())
        outcome = _outcome(left_ok[mask], right_ok[mask]) if count else None
        per_class[name] = {
            "support": count,
            "left_f1": left_metrics["per_class_f1"][index],
            "right_f1": right_metrics["per_class_f1"][index],
            "left_recall": left_metrics["per_class_recall"][index],
            "right_recall": right_metrics["per_class_recall"][index],
            "f1_difference": (
                left_metrics["per_class_f1"][index] - right_metrics["per_class_f1"][index]
            ),
            "outcome": outcome.to_dict() if outcome else None,
            # The only samples another modality could rescue: this one is wrong
            # and the other is right. Anything else is not an opportunity.
            "correction_opportunity_for_left": outcome.right_only if outcome else 0,
            "correction_opportunity_for_right": outcome.left_only if outcome else 0,
            "stronger": _stronger(
                left_metrics["per_class_f1"][index],
                right_metrics["per_class_f1"][index],
                count, left_name or left.modality, right_name or right.modality,
            ),
        }

    overall = _outcome(left_ok, right_ok)
    return {
        "left": left_name or left.modality,
        "right": right_name or right.modality,
        "split": a.split,
        "samples": len(sample_ids),
        "aligned": True,
        "class_order": classes,
        "overall": {
            "left": {
                "accuracy": left_metrics["accuracy"],
                "macro_f1": left_metrics["macro_f1"],
                "weighted_f1": left_metrics["weighted_f1"],
            },
            "right": {
                "accuracy": right_metrics["accuracy"],
                "macro_f1": right_metrics["macro_f1"],
                "weighted_f1": right_metrics["weighted_f1"],
            },
            "outcome": overall.to_dict(),
            "disagreement_rate": float((left_pred != right_pred).mean()),
            "agreement_rate": float((left_pred == right_pred).mean()),
            "correction_opportunity_for_left": overall.right_only,
            "correction_opportunity_for_right": overall.left_only,
            "ceiling_if_either_were_trusted": float(
                (left_ok | right_ok).mean()
            ),
        },
        "per_class": per_class,
        "reading": (
            "'Correction opportunity for X' counts samples where X is wrong and the "
            "other modality is right -- the only samples an acquisition could rescue. "
            "'Ceiling if either were trusted' is the accuracy of an oracle that always "
            "picked the correct one of the two; it bounds what any routing policy over "
            "this pair can reach and is NOT achievable by a real router."
        ),
    }


def _stronger(left_f1: float, right_f1: float, support: int, left: str, right: str) -> dict:
    """Name the stronger modality for one class, or decline when support is thin."""
    # Below this, a single sample moves F1 by more than the gap usually is, so a
    # winner would be an artefact of the support rather than a finding.
    MIN_SUPPORT = 20
    difference = left_f1 - right_f1
    if support < MIN_SUPPORT:
        return {
            "winner": None,
            "reason": (
                f"support is {support} (< {MIN_SUPPORT}); a single sample moves F1 by "
                f"more than the observed gap, so no winner is declared"
            ),
        }
    if abs(difference) < 0.02:
        return {
            "winner": None,
            "reason": f"F1 differs by only {difference:+.4f}; treated as a tie",
        }
    return {
        "winner": left if difference > 0 else right,
        "margin": abs(difference),
        "reason": f"F1 {left_f1:.4f} vs {right_f1:.4f} on {support} shared samples",
    }


# ============================================================
# Specialization
# ============================================================

def specialization_verdicts(comparisons: Mapping[str, dict]) -> dict:
    """Per class, which modality wins -- using aligned evidence only.

    Each aligned pair contributes at most one directed edge ``A beats B`` for a
    class.  Those edges form a partial order, not a vote: "text beats video" and
    "text_llm beats text" are a consistent *chain*, not a disagreement, and an
    earlier version of this function wrongly read them as one.  So the verdict is
    the unique **undefeated** modality -- one that wins at least one comparison
    and loses none.  Anything else (no edges, several undefeated modalities, or a
    genuine cycle) is reported as undecided with the graph attached.

    A class with no aligned pool, or with too little support in the pools that
    exist, gets ``None`` and a reason.  That is the honest answer for most of
    this project's classes, and printing it is the point.
    """
    verdicts: dict[str, dict] = {}
    for name in CANONICAL_EMOTION_CLASSES:
        evidence, edges = [], []
        participants: set[str] = set()
        for pair, record in comparisons.items():
            entry = record["per_class"].get(name)
            if not entry:
                continue
            left, right = record["left"], record["right"]
            participants.update({left, right})
            winner = entry["stronger"]["winner"]
            evidence.append({
                "pair": pair,
                "split": record["split"],
                "support": entry["support"],
                "winner": winner,
                "reason": entry["stronger"]["reason"],
                "left": left, "left_f1": entry["left_f1"],
                "right": right, "right_f1": entry["right_f1"],
            })
            if winner:
                edges.append({"beats": winner, "beaten": right if winner == left else left})

        beaten = {edge["beaten"] for edge in edges}
        winners = {edge["beats"] for edge in edges}
        undefeated = sorted(winners - beaten)
        chain = "; ".join(f"{edge['beats']} > {edge['beaten']}" for edge in edges)

        if len(undefeated) == 1:
            reason = (
                f"{undefeated[0]} wins at least one aligned comparison for this class "
                f"and loses none ({chain})"
            )
        elif not edges:
            reason = "no aligned comparison has usable support for this class"
        elif len(undefeated) > 1:
            reason = (
                f"several modalities are undefeated ({', '.join(undefeated)}); the "
                f"aligned pairs do not rank them against each other ({chain})"
            )
        else:
            reason = f"the aligned comparisons form a cycle ({chain}); none is undefeated"

        verdicts[name] = {
            "evidence": evidence,
            "comparisons_with_a_winner": len(edges),
            "win_edges": [f"{edge['beats']} > {edge['beaten']}" for edge in edges],
            "participants": sorted(participants),
            "undefeated": undefeated,
            "verdict": undefeated[0] if len(undefeated) == 1 else None,
            "reason": reason,
        }
    covered = [name for name, record in verdicts.items() if record["verdict"]]
    return {
        "per_class": verdicts,
        "classes_with_a_verdict": covered,
        "classes_without_a_verdict": [
            name for name in CANONICAL_EMOTION_CLASSES if name not in covered
        ],
        "evidence_rule": (
            "Each aligned pair contributes at most one directed 'A beats B' edge per "
            "class, and only when the class has at least 20 shared samples in that "
            "pool and the F1 gap exceeds 0.02. The verdict is the unique modality that "
            "wins at least one comparison and loses none. Edges from different pairs "
            "form a partial order, not a vote: 'text > video' and 'text_llm > text' "
            "are a consistent chain, not a contradiction. Classes with no edges, "
            "several undefeated modalities, or a cycle are reported as undecided."
        ),
        "coverage_caveat": (
            "This project has exactly two aligned pairs (audio+text, text+video), so "
            "image and physiology appear in NO head-to-head comparison. Nothing here "
            "says anything about image's class-wise strength relative to the others, "
            "and the per-modality profiles must not be read as if it did."
        ),
    }


# ============================================================
# Complementarity summary
# ============================================================

def complementarity_summary(comparisons: Mapping[str, dict]) -> dict:
    """How much room a router actually has, per aligned pair."""
    rows = {}
    for pair, record in comparisons.items():
        overall = record["overall"]
        left_accuracy = overall["left"]["accuracy"]
        right_accuracy = overall["right"]["accuracy"]
        best_single = max(left_accuracy, right_accuracy)
        rows[pair] = {
            "split": record["split"],
            "samples": record["samples"],
            "left": record["left"], "right": record["right"],
            "left_accuracy": left_accuracy,
            "right_accuracy": right_accuracy,
            "best_single_accuracy": best_single,
            "oracle_either_accuracy": overall["ceiling_if_either_were_trusted"],
            "headroom_over_best_single": (
                overall["ceiling_if_either_were_trusted"] - best_single
            ),
            "disagreement_rate": overall["disagreement_rate"],
            "correction_opportunity_for_left": overall["correction_opportunity_for_left"],
            "correction_opportunity_for_right": overall["correction_opportunity_for_right"],
            "both_wrong_rate": overall["outcome"]["both_wrong_rate"],
        }
    return {
        "pairs": rows,
        "reading": (
            "'Headroom over best single' is how much accuracy an oracle that always "
            "trusted the right modality would add over simply always using the better "
            "one. It is the ceiling on what routing over this pair can buy. "
            "'Both wrong' is the fraction no amount of routing between these two can "
            "fix, and it bounds the pair from the other side."
        ),
    }
