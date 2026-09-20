"""Phase 15 -- is the section-13 fusion ranking robust to the seed?

    # train the 3 x 3 grid, then aggregate (resumable; finished runs are skipped)
    python -m src.hsen.seed_variance --study phase15_seed_variance

    # aggregate finished runs into the table without training anything
    python -m src.hsen.seed_variance --study phase15_seed_variance --report-only

Section 13 put Husformer +0.0107 weighted F1 over concatenation and +0.0125 over
self-attention, from one seed each, and the phase document already said the
margin is one "a seed change could plausibly cover".  Before a day of GPU time
is spent on the fusion that ranking chose, this study asks the one question
that ranking cannot answer about itself: does it survive a change of seed?

What varies: the training seed -- initialisation, dropout masks, batch order.
Nothing else.  The stratified 25% training draw is pinned to the section-13
subset (``subset_seed=42``) rather than following the training seed, so every
run trains on the same 1,062 utterances and "seed variance" is not quietly
"seed plus data-subset variance".  The study records a digest of those 1,062
ids and refuses to aggregate a run whose training set differs.

What this study does *not* do.  It does not open the test split -- every number
is validation, selected on validation, exactly as section 13's were.  It does
not claim significance: three seeds bound the spread, they do not estimate a
distribution, and the verdict is worded accordingly.  It does not touch the
architecture, the schedule or the loss.  Each run's directory is claimed by
:class:`~src.hsen.training.runlock.RunLock` for the life of the run, and the
study directory by a second lock for the life of the driver, so two drivers
cannot interleave their tables any more than two trainers can interleave
their checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.hsen.ablations import PROFILES
from src.hsen.cli import MODALITY_SETS
from src.hsen.experiment import ExperimentError, run_experiment
from src.hsen.models.fusion_variants import FUSION_VARIANTS
from src.hsen.training.hsen_trainer import TrainerConfig
from src.hsen.training.losses import LossConfig
from src.hsen.training.runlock import RunLock

#: The three fusion trunks section 13 compared, in the order the brief lists them.
STUDY_VARIANTS: tuple[str, ...] = ("husformer", "concat", "self_attention")

#: Consecutive seeds, declared once. 42 reproduces section 13 exactly, which
#: doubles as a determinism check on the CPU build.
STUDY_SEEDS: tuple[int, ...] = (42, 43, 44)

#: The section-13 training subset. Every run draws this quarter of the data.
SUBSET_SEED = 42

DEFAULT_STUDY = "phase15_seed_variance"

#: The first fusion in this order is the one section 13 chose and Phase 15 is
#: auditing. The others are compared against it, seed by seed.
REFERENCE_VARIANT = "husformer"


class SeedStudyError(RuntimeError):
    """Raised when a run cannot be admitted to the study's aggregate."""


# =====================================================================
# Run configuration
# =====================================================================

@dataclass(frozen=True)
class StudyPlan:
    """Everything held fixed across the grid, declared in one place."""

    study: str = DEFAULT_STUDY
    experiment: str = "iemocap_erc6"
    profile: str = "cpu25"
    modalities: str = "audio+text"
    variants: tuple[str, ...] = STUDY_VARIANTS
    seeds: tuple[int, ...] = STUDY_SEEDS
    subset_seed: int = SUBSET_SEED
    output_root: Path = Path("results/hsen")

    def __post_init__(self) -> None:
        unknown = [v for v in self.variants if v not in FUSION_VARIANTS]
        if unknown:
            raise SeedStudyError(
                f"Unknown fusion variant(s) {unknown}; expected a subset of {list(FUSION_VARIANTS)}"
            )
        if len(set(self.seeds)) != len(self.seeds):
            raise SeedStudyError(f"Seeds must be distinct, got {list(self.seeds)}")
        if self.profile not in PROFILES:
            raise SeedStudyError(f"Unknown profile {self.profile!r}; expected one of {sorted(PROFILES)}")
        object.__setattr__(self, "output_root", Path(self.output_root))

    @property
    def study_dir(self) -> Path:
        return self.output_root / self.experiment / self.study

    def run_name(self, variant: str, seed: int) -> str:
        return f"fusion_{variant}_seed{seed}"

    def run_dir(self, variant: str, seed: int) -> Path:
        return self.study_dir / self.run_name(variant, seed)

    def trainer_config(self, variant: str, seed: int, resume: bool = False,
                       force_lock: bool = False) -> TrainerConfig:
        """The section-13 configuration with the seed swapped in and the subset pinned.

        ``PROFILES[profile]`` and ``TrainerConfig``'s defaults are exactly what
        :func:`src.hsen.ablations.run_arm` used, so the seed-42 row of this study
        is section 13's run re-executed, not a near relative of it.
        """
        return TrainerConfig(
            experiment=self.experiment,
            profile=f"{self.study}/{self.run_name(variant, seed)}",
            output_root=self.output_root,
            seed=seed,
            subset_seed=self.subset_seed,
            evaluate_test=False,          # never: model selection is validation-only
            resume=resume,
            force_lock=force_lock,
            **PROFILES[self.profile],
        )

    def command(self, variant: str, seed: int) -> str:
        """The single-run equivalent, for the record."""
        settings = PROFILES[self.profile]
        return (
            f"python -m src.hsen.seed_variance --study {self.study} "
            f"--experiment {self.experiment} --profile {self.profile} "
            f"--modalities {self.modalities} --variants {variant} --seeds {seed} "
            f"--subset-seed {self.subset_seed}"
            f"  # device={settings['device']} data_fraction={settings['data_fraction']} "
            f"batch_size={settings['batch_size']} epochs={settings['epochs']}"
        )

    def to_dict(self) -> dict:
        settings = PROFILES[self.profile]
        template = self.trainer_config(self.variants[0], self.seeds[0])
        return {
            "study": self.study,
            "experiment": self.experiment,
            "profile": self.profile,
            "profile_settings": settings,
            "modalities": list(MODALITY_SETS[self.modalities]),
            "variants": list(self.variants),
            "seeds": list(self.seeds),
            "subset_seed": self.subset_seed,
            "reference_variant": REFERENCE_VARIANT,
            "evaluate_test": False,
            "checkpoint_selection": "best validation weighted_f1",
            "trainer_config_template": {
                k: v for k, v in template.to_dict().items()
                if k not in ("profile", "seed", "resume", "force_lock")
            },
            "loss_config": LossConfig().to_dict(),
            "commands": {
                self.run_name(v, s): self.command(v, s)
                for v in self.variants for s in self.seeds
            },
        }


def training_subset_digest(experiment: str, data_fraction: float, subset_seed: int) -> dict:
    """SHA-256 over the sorted training ids the study draws, plus their count.

    Recorded once in ``study_config.json`` and checked against every run's
    dataset count, so a run that trained on a different quarter of the data
    cannot enter the aggregate looking like the others.
    """
    from src.hsen.data import stratified_subset
    from src.hsen.manifests import load_all

    train = load_all(experiment)["train"]
    selected = stratified_subset(train, data_fraction, subset_seed)
    ids = sorted(train["sample_id"].astype(str).iloc[selected].tolist())
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return {"count": len(ids), "sha256": digest, "subset_seed": subset_seed,
            "data_fraction": data_fraction}


# =====================================================================
# Per-run rows
# =====================================================================

#: The columns the brief asks for, in its order, then the provenance that makes
#: a row checkable.
ROW_COLUMNS = (
    "variant", "seed", "best_epoch", "val_WF1", "val_macro_F1", "accuracy",
    "train_time", "parameter_count",
    "fusion_parameters", "epochs_run", "stop_reason", "subset_seed",
    "train_samples", "validation_samples", "class_order",
)


def run_row(summary: dict, variant: str | None = None, seed: int | None = None) -> dict:
    """Flatten one ``run_summary.json`` into the study's row.

    The variant and seed are read from the summary, and if the caller says
    what it *expects* them to be, a mismatch is an error rather than a relabel:
    a row filed under the wrong variant is the most damaging mistake an
    aggregate can contain, and the cheapest to catch here.
    """
    recorded_variant = summary["model_config"]["fusion"]
    recorded_seed = int(summary["trainer_config"]["seed"])
    if variant is not None and recorded_variant != variant:
        raise SeedStudyError(
            f"Run is labelled {variant!r} but its model_config says fusion="
            f"{recorded_variant!r}"
        )
    if seed is not None and recorded_seed != seed:
        raise SeedStudyError(
            f"Run is labelled seed {seed} but its trainer_config says seed={recorded_seed}"
        )
    if recorded_variant not in FUSION_VARIANTS:
        raise SeedStudyError(
            f"Unknown fusion variant {recorded_variant!r} in run summary; "
            f"expected one of {list(FUSION_VARIANTS)}"
        )
    if summary.get("test"):
        raise SeedStudyError(
            "Run summary carries test metrics; the seed-variance study is "
            "validation-only and refuses a run that opened the test split"
        )

    validation = summary["validation"]
    primary = summary["primary_metric"]
    best = summary.get(f"best_val_{primary}")
    # The shipped checkpoint's validation score must be the score that selected
    # it. A run whose best_val_* is absent from its own history is the exact
    # corruption RunLock was written after.
    history_values = [row.get(f"val_{primary}") for row in summary.get("history", [])]
    if best is not None and not any(
        isinstance(v, float) and math.isclose(v, best, rel_tol=0, abs_tol=1e-9)
        for v in history_values
    ):
        raise SeedStudyError(
            f"best_val_{primary}={best} does not appear in the run's own history; "
            f"the artefacts are jointly incoherent and the run is refused"
        )
    if not math.isclose(validation[primary], best, rel_tol=0, abs_tol=1e-6):
        raise SeedStudyError(
            f"Restored-best validation {primary}={validation[primary]} differs from "
            f"the selected best_val_{primary}={best}"
        )

    counts = summary["run_record"].get("dataset_counts", {})
    trainer = summary["trainer_config"]
    return {
        "variant": recorded_variant,
        "seed": recorded_seed,
        "best_epoch": int(summary["best_epoch"]),
        "val_WF1": float(validation["weighted_f1"]),
        "val_macro_F1": float(validation["macro_f1"]),
        "accuracy": float(validation["accuracy"]),
        "train_time": float(summary["training_seconds"]),
        "parameter_count": int(summary["parameter_counts"]["total"]),
        "fusion_parameters": int(summary["parameter_counts"]["fusion"]),
        "epochs_run": int(summary["epochs_run"]),
        "stop_reason": summary.get("stop_reason"),
        "subset_seed": trainer.get("subset_seed"),
        "train_samples": counts.get("train", {}).get("samples"),
        "validation_samples": counts.get("validation", {}).get("samples"),
        "class_order": tuple(validation["per_class"]),
    }


def collect_rows(plan: StudyPlan, class_order: tuple[str, ...] | None = None,
                 subset: dict | None = None) -> pd.DataFrame:
    """Read every finished run in the study directory into one frame.

    ``run_summary.json`` is written only at the end of a run, so its presence is
    the completion marker; a directory holding only checkpoints is a partial
    run and is not a row.  When the declared class order or the training-subset
    record is given, every row is checked against it.
    """
    rows = []
    for variant in plan.variants:
        for seed in plan.seeds:
            path = plan.run_dir(variant, seed) / "run_summary.json"
            if not path.exists():
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            row = run_row(summary, variant, seed)
            assert_row_admissible(row, plan, class_order, subset)
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(ROW_COLUMNS))
    if not frame.empty:
        frame = frame.sort_values(["variant", "seed"], key=_variant_sort_key(plan)).reset_index(drop=True)
    return frame


def _variant_sort_key(plan: StudyPlan):
    order = {v: i for i, v in enumerate(plan.variants)}

    def key(column: pd.Series) -> pd.Series:
        return column.map(order) if column.name == "variant" else column
    return key


def assert_row_admissible(row: dict, plan: StudyPlan, class_order: tuple[str, ...] | None,
                          subset: dict | None) -> None:
    """The checks that make a row comparable with the others."""
    if row["variant"] not in plan.variants:
        raise SeedStudyError(f"{row['variant']!r} is not one of the study's variants {list(plan.variants)}")
    if row["seed"] not in plan.seeds:
        raise SeedStudyError(f"seed {row['seed']} is not one of the study's seeds {list(plan.seeds)}")
    if row["subset_seed"] != plan.subset_seed:
        raise SeedStudyError(
            f"{row['variant']} seed {row['seed']} drew its training subset with "
            f"subset_seed={row['subset_seed']!r}, not the study's {plan.subset_seed}; "
            f"it trained on different data and cannot enter the aggregate"
        )
    if class_order is not None and tuple(row["class_order"]) != tuple(class_order):
        raise SeedStudyError(
            f"{row['variant']} seed {row['seed']} reports per-class metrics in order "
            f"{list(row['class_order'])}, not the declared {list(class_order)}"
        )
    if subset is not None and row["train_samples"] != subset["count"]:
        raise SeedStudyError(
            f"{row['variant']} seed {row['seed']} trained on {row['train_samples']} "
            f"samples; the study's subset has {subset['count']}"
        )


# =====================================================================
# Aggregation
# =====================================================================

def _std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); NaN below two values rather than 0.

    A single run has no spread to report, and printing 0.0 for it would read
    as "perfectly stable" -- the opposite of what one run can show.
    """
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def aggregate(frame: pd.DataFrame, variants: tuple[str, ...] = STUDY_VARIANTS) -> dict[str, dict]:
    """Per-variant summary statistics over whatever seeds have finished."""
    out: dict[str, dict] = {}
    for variant in variants:
        rows = frame[frame["variant"] == variant].sort_values("seed")
        if rows.empty:
            continue
        wf1 = rows["val_WF1"].astype(float).tolist()
        mf1 = rows["val_macro_F1"].astype(float).tolist()
        params = sorted(set(rows["parameter_count"].astype(int).tolist()))
        if len(params) != 1:
            raise SeedStudyError(
                f"{variant} runs disagree on parameter_count {params}; the seeds "
                f"did not train the same architecture"
            )
        out[variant] = {
            "n_seeds": int(len(rows)),
            "seeds": rows["seed"].astype(int).tolist(),
            "mean_WF1": sum(wf1) / len(wf1),
            "std_WF1": _std(wf1),
            "min_WF1": min(wf1),
            "max_WF1": max(wf1),
            "mean_macro_F1": sum(mf1) / len(mf1),
            "std_macro_F1": _std(mf1),
            "mean_accuracy": float(rows["accuracy"].mean()),
            "mean_training_time": float(rows["train_time"].mean()),
            "mean_best_epoch": float(rows["best_epoch"].mean()),
            "parameter_count": params[0],
            "fusion_parameters": int(rows["fusion_parameters"].iloc[0]),
            "std_ddof": 1,
        }
    return out


def paired_comparison(frame: pd.DataFrame, reference: str = REFERENCE_VARIANT,
                      metric: str = "val_WF1") -> dict[str, dict]:
    """Reference minus each other variant, seed by seed, on the seeds both ran.

    Pairing by seed is what the study design buys: two variants under the same
    seed share initialisation noise in the projections and heads and the same
    batch order, so their difference is closer to the fusion's own effect than
    a difference of two unpaired means would be.
    """
    reference_rows = frame[frame["variant"] == reference].set_index("seed")[metric]
    out: dict[str, dict] = {}
    for variant in sorted(set(frame["variant"]) - {reference},
                          key=lambda v: STUDY_VARIANTS.index(v) if v in STUDY_VARIANTS else 99):
        other = frame[frame["variant"] == variant].set_index("seed")[metric]
        seeds = sorted(set(reference_rows.index) & set(other.index))
        if not seeds:
            continue
        deltas = [float(reference_rows[s] - other[s]) for s in seeds]
        wins = sum(d > 0 for d in deltas)
        out[variant] = {
            "metric": metric,
            "seeds": seeds,
            "delta_per_seed": dict(zip((str(s) for s in seeds), deltas)),
            "reference_wins": int(wins),
            "n_pairs": len(seeds),
            "consistent": wins == len(seeds),
            "mean_delta": sum(deltas) / len(deltas),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "std_delta": _std(deltas),
        }
    return out


def margin_versus_variance(aggregates: dict[str, dict], pairs: dict[str, dict],
                           reference: str = REFERENCE_VARIANT) -> dict[str, dict]:
    """Is the mean margin larger than the seed spread on either side of it?

    Two yardsticks, both reported: the larger of the two variants' own
    across-seed standard deviations, and the spread of the paired differences.
    A margin smaller than either is inside the noise these seeds can show.
    """
    out: dict[str, dict] = {}
    for variant, pair in pairs.items():
        if reference not in aggregates or variant not in aggregates:
            continue
        ref_std = aggregates[reference]["std_WF1"]
        other_std = aggregates[variant]["std_WF1"]
        seed_std = max(s for s in (ref_std, other_std) if not math.isnan(s)) \
            if not (math.isnan(ref_std) and math.isnan(other_std)) else float("nan")
        margin = pair["mean_delta"]
        out[variant] = {
            "mean_margin": margin,
            "reference_std_WF1": ref_std,
            "other_std_WF1": other_std,
            "larger_seed_std": seed_std,
            "paired_delta_std": pair["std_delta"],
            "margin_exceeds_larger_seed_std": bool(margin > seed_std) if not math.isnan(seed_std) else None,
            "margin_exceeds_paired_std": bool(margin > pair["std_delta"]) if not math.isnan(pair["std_delta"]) else None,
            "ranges_overlap": bool(
                aggregates[reference]["min_WF1"] <= aggregates[variant]["max_WF1"]
            ),
        }
    return out


def verdict(aggregates: dict[str, dict], pairs: dict[str, dict], margins: dict[str, dict],
            reference: str = REFERENCE_VARIANT, n_seeds_planned: int = len(STUDY_SEEDS)) -> dict:
    """The recommendation, worded for what three seeds can and cannot show.

    Three seeds bound a spread; they do not estimate a distribution, so no
    sentence here says "significant".  The recommendation is the variant with
    the highest mean validation weighted F1 across the seeds that ran, with the
    consistency and margin-versus-spread facts stated beside it rather than
    folded into it.
    """
    lines: list[str] = []
    complete = all(a["n_seeds"] == n_seeds_planned for a in aggregates.values())
    if not complete:
        lines.append(
            f"INCOMPLETE: not every variant has {n_seeds_planned} finished seeds; "
            f"the statements below cover only the seeds that ran."
        )
    ranking = sorted(aggregates, key=lambda v: aggregates[v]["mean_WF1"], reverse=True)
    best = ranking[0] if ranking else None

    for variant, pair in pairs.items():
        m = margins.get(variant, {})
        n, wins = pair["n_pairs"], pair["reference_wins"]
        head = f"{reference} vs {variant}: {reference} is ahead on {wins} of {n} seeds"
        if pair["consistent"]:
            head += " (consistent across these seeds)"
        elif wins == 0:
            head += f" ({variant} is ahead on every seed that ran)"
        else:
            head += " (the ordering flips with the seed)"
        lines.append(
            f"{head}; mean margin {pair['mean_delta']:+.4f} WF1 "
            f"[{pair['min_delta']:+.4f}, {pair['max_delta']:+.4f}]."
        )
        if m:
            seed_std = m["larger_seed_std"]
            comparator = (
                "larger than" if m["margin_exceeds_larger_seed_std"] else "not larger than"
            ) if m["margin_exceeds_larger_seed_std"] is not None else "not comparable with"
            lines.append(
                f"    That margin is {comparator} the larger across-seed std "
                f"({seed_std:.4f}) and "
                f"{'larger than' if m['margin_exceeds_paired_std'] else 'not larger than'} "
                f"the paired-difference std ({m['paired_delta_std']:.4f}); "
                f"the two variants' WF1 ranges "
                f"{'overlap' if m['ranges_overlap'] else 'do not overlap'}."
            )

    if best is not None:
        b = aggregates[best]
        lines.append(
            f"Highest mean validation WF1 across these seeds: {best} "
            f"({b['mean_WF1']:.4f} +/- {b['std_WF1']:.4f}, "
            f"{b['parameter_count']:,} parameters, "
            f"{b['mean_training_time']:.0f} s mean training time)."
        )
    stable = min(aggregates, key=lambda v: aggregates[v]["std_WF1"]) if aggregates else None
    if stable is not None and not math.isnan(aggregates[stable]["std_WF1"]):
        lines.append(
            f"Smallest across-seed spread: {stable} (std {aggregates[stable]['std_WF1']:.4f})."
        )

    recommendation = best
    reason = "highest mean validation weighted F1 across the seeds that ran"
    if best is not None and best != reference:
        reason += f"; {reference} did not hold its section-13 lead across seeds"
    elif best is not None:
        consistent = all(p["consistent"] for p in pairs.values()) if pairs else False
        clear = all(m.get("margin_exceeds_larger_seed_std") for m in margins.values()) if margins else False
        if consistent and clear:
            reason += ", ahead on every seed against every alternative, by more than the seed spread"
        elif consistent:
            reason += (", ahead on every seed against every alternative, but by a margin "
                       "inside the seed spread -- the ordering is more stable across these "
                       "seeds than the magnitude is")
        else:
            reason += (", but the ordering against at least one alternative flips with the "
                       "seed, so this is a preference on mean, not a robust ranking")

    return {
        "recommended_fusion": recommendation,
        "reason": reason,
        "ranking_by_mean_WF1": ranking,
        "complete": complete,
        "seeds_per_variant": {v: a["n_seeds"] for v, a in aggregates.items()},
        "statements": lines,
        "caveat": (
            f"{n_seeds_planned} seeds bound the spread; they do not support a claim of "
            f"statistical significance in either direction."
        ),
    }


# =====================================================================
# Rendering
# =====================================================================

def render_table(frame: pd.DataFrame, aggregates: dict[str, dict],
                 seeds: tuple[int, ...] = STUDY_SEEDS,
                 variants: tuple[str, ...] = STUDY_VARIANTS,
                 metric: str = "val_WF1", label: str = "val WF1") -> str:
    """``variant | seed42 | seed43 | seed44 | mean ± std | params`` as Markdown."""
    header = f"| variant | " + " | ".join(f"seed{s}" for s in seeds) + f" | mean ± std ({label}) | params |"
    rule = "|---|" + "---:|" * len(seeds) + "---:|---:|"
    lines = [header, rule]
    for variant in variants:
        if variant not in aggregates:
            continue
        rows = frame[frame["variant"] == variant].set_index("seed")[metric]
        cells = [f"{rows[s]:.4f}" if s in rows.index else "--" for s in seeds]
        a = aggregates[variant]
        key = "std_WF1" if metric == "val_WF1" else "std_macro_F1"
        mean_key = "mean_WF1" if metric == "val_WF1" else "mean_macro_F1"
        std = a[key]
        spread = f"{a[mean_key]:.4f} ± {std:.4f}" if not math.isnan(std) else f"{a[mean_key]:.4f} ± --"
        lines.append(
            f"| {variant} | " + " | ".join(cells) + f" | {spread} | {a['parameter_count']:,} |"
        )
    return "\n".join(lines)


def render_report(plan: StudyPlan, frame: pd.DataFrame, aggregates: dict, pairs: dict,
                  margins: dict, decision: dict) -> str:
    lines = [
        "=" * 96,
        f"PHASE 15 -- seed-variance study  ({plan.experiment}, {plan.profile}, "
        f"{'+'.join(MODALITY_SETS[plan.modalities])}, subset seed {plan.subset_seed})",
        "=" * 96,
        "",
        "Per-run (validation only; the test split was not opened):",
        "",
        f"{'variant':<16}{'seed':>6}{'best_ep':>9}{'val_WF1':>10}{'val_mF1':>10}"
        f"{'acc':>9}{'train s':>10}{'params':>12}{'epochs':>8}",
        "-" * 96,
    ]
    for _, r in frame.iterrows():
        lines.append(
            f"{r['variant']:<16}{int(r['seed']):>6}{int(r['best_epoch']):>9}"
            f"{r['val_WF1']:>10.4f}{r['val_macro_F1']:>10.4f}{r['accuracy']:>9.4f}"
            f"{r['train_time']:>10.0f}{int(r['parameter_count']):>12,}{int(r['epochs_run']):>8}"
        )
    lines += ["", "Aggregate (std is the sample std, ddof=1):", ""]
    lines.append(
        f"{'variant':<16}{'n':>3}{'mean_WF1':>10}{'std_WF1':>9}{'min_WF1':>9}{'max_WF1':>9}"
        f"{'mean_mF1':>10}{'std_mF1':>9}{'mean s':>9}{'params':>12}"
    )
    lines.append("-" * 96)
    for variant, a in aggregates.items():
        def f(v, w=9, p=4):
            return f"{v:>{w}.{p}f}" if not math.isnan(v) else f"{'--':>{w}}"
        lines.append(
            f"{variant:<16}{a['n_seeds']:>3}{f(a['mean_WF1'], 10)}{f(a['std_WF1'])}"
            f"{f(a['min_WF1'])}{f(a['max_WF1'])}{f(a['mean_macro_F1'], 10)}"
            f"{f(a['std_macro_F1'])}{a['mean_training_time']:>9.0f}{a['parameter_count']:>12,}"
        )
    lines += ["", render_table(frame, aggregates, plan.seeds, plan.variants), ""]
    lines += [render_table(frame, aggregates, plan.seeds, plan.variants,
                           metric="val_macro_F1", label="val macro-F1"), ""]
    lines.append(f"Paired against {REFERENCE_VARIANT}, seed by seed:")
    for statement in decision["statements"]:
        lines.append(f"  {statement}")
    lines += [
        "",
        f"Recommended fusion for the full-data CUDA run: {decision['recommended_fusion']}",
        f"  because: {decision['reason']}",
        f"  caveat:  {decision['caveat']}",
        "=" * 96,
    ]
    return "\n".join(lines)


# =====================================================================
# Driver
# =====================================================================

def declared_class_order(experiment: str) -> tuple[str, ...]:
    from src.hsen.labels.base import resolve_label_adapter
    from src.hsen.manifests import EXPERIMENTS

    return tuple(resolve_label_adapter(EXPERIMENTS[experiment].label_protocol).label_space().classes)


def run_grid(plan: StudyPlan, resume: bool, force_lock: bool, log) -> list[dict]:
    """Train every (variant, seed) that has no ``run_summary.json`` yet."""
    loss_config = LossConfig()
    outcomes: list[dict] = []
    total = len(plan.variants) * len(plan.seeds)
    index = 0
    for seed in plan.seeds:
        for variant in plan.variants:
            index += 1
            name = plan.run_name(variant, seed)
            if (plan.run_dir(variant, seed) / "run_summary.json").exists():
                log(f"[{index}/{total}] {name}: already complete, skipped")
                outcomes.append({"run": name, "status": "skipped_complete"})
                continue
            log(f"\n{'#' * 96}\n# [{index}/{total}] {name}\n{'#' * 96}")
            started = time.time()
            try:
                run_experiment(
                    trainer_config=plan.trainer_config(variant, seed, resume, force_lock),
                    modalities=MODALITY_SETS[plan.modalities],
                    fusion=variant,
                    loss_config=loss_config,
                )
            except ExperimentError as error:
                log(f"[skip] {name}: {str(error).splitlines()[0]}")
                outcomes.append({"run": name, "status": "skipped_error", "error": str(error)})
                continue
            outcomes.append({"run": name, "status": "trained",
                             "wall_clock_seconds": round(time.time() - started, 1)})
            log(f"[done] {name} in {time.time() - started:,.0f} s")
    return outcomes


def write_study_artifacts(plan: StudyPlan, frame: pd.DataFrame, aggregates: dict, pairs: dict,
                          margins: dict, decision: dict, subset: dict, class_order: tuple,
                          outcomes: list[dict] | None = None) -> None:
    study_dir = plan.study_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    config = plan.to_dict() | {
        "training_subset": subset,
        "class_order": list(class_order),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (study_dir / "study_config.json").write_text(json.dumps(config, indent=2, default=str),
                                                 encoding="utf-8")
    csv = frame.copy()
    csv["class_order"] = csv["class_order"].map(lambda order: "|".join(order))
    csv.to_csv(study_dir / "seed_variance_runs.csv", index=False)
    summary = {
        "study": plan.study,
        "experiment": plan.experiment,
        "profile": plan.profile,
        "modalities": list(MODALITY_SETS[plan.modalities]),
        "seeds": list(plan.seeds),
        "subset_seed": plan.subset_seed,
        "class_order": list(class_order),
        "reference_variant": REFERENCE_VARIANT,
        "runs": [
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")
        ],
        "aggregate": aggregates,
        "paired_vs_reference": pairs,
        "margin_versus_seed_variance": margins,
        "verdict": decision,
        "outcomes": outcomes or [],
    }
    (study_dir / "seed_variance_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    table = [
        f"# Phase 15 seed-variance study -- {plan.experiment}, {plan.profile}",
        "",
        render_table(frame, aggregates, plan.seeds, plan.variants),
        "",
        render_table(frame, aggregates, plan.seeds, plan.variants,
                     metric="val_macro_F1", label="val macro-F1"),
        "",
        *(f"- {s}" for s in decision["statements"]),
        "",
        f"**Recommended fusion:** `{decision['recommended_fusion']}` -- {decision['reason']}.",
        "",
        f"_{decision['caveat']}_",
        "",
    ]
    (study_dir / "seed_variance_table.md").write_text("\n".join(table), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default=DEFAULT_STUDY,
                        help="Study directory name under results/hsen/<experiment>/.")
    parser.add_argument("--experiment", default="iemocap_erc6")
    parser.add_argument("--profile", default="cpu25", choices=sorted(PROFILES))
    parser.add_argument("--modalities", default="audio+text", choices=sorted(MODALITY_SETS))
    parser.add_argument("--variants", default=",".join(STUDY_VARIANTS),
                        help="Comma-separated fusion variants.")
    parser.add_argument("--seeds", default=",".join(str(s) for s in STUDY_SEEDS),
                        help="Comma-separated training seeds.")
    parser.add_argument("--subset-seed", "--subset_seed", dest="subset_seed", type=int,
                        default=SUBSET_SEED,
                        help="Seed of the stratified training-fraction draw, held "
                             "fixed across every run.")
    parser.add_argument("--output-root", default="results/hsen")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a partially-trained run from its last.pt.")
    parser.add_argument("--force-lock", action="store_true",
                        help="Break stale locks. Only after confirming the holders are gone.")
    parser.add_argument("--report-only", action="store_true",
                        help="Aggregate finished runs without training.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = StudyPlan(
        study=args.study, experiment=args.experiment, profile=args.profile,
        modalities=args.modalities,
        variants=tuple(v.strip() for v in args.variants.split(",") if v.strip()),
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        subset_seed=args.subset_seed, output_root=Path(args.output_root),
    )
    plan.study_dir.mkdir(parents=True, exist_ok=True)
    log_path = plan.study_dir / "study.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    class_order = declared_class_order(plan.experiment)
    subset = training_subset_digest(plan.experiment, PROFILES[plan.profile]["data_fraction"],
                                    plan.subset_seed)
    log(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] study {plan.study}: "
        f"variants={list(plan.variants)} seeds={list(plan.seeds)} "
        f"subset_seed={plan.subset_seed} -> {subset['count']} training ids "
        f"(sha256 {subset['sha256'][:16]}); class order {list(class_order)}")

    # One driver per study directory, on top of one trainer per run directory.
    study_lock = RunLock(plan.study_dir).acquire(force=args.force_lock)
    try:
        outcomes = None
        if not args.report_only:
            outcomes = run_grid(plan, args.resume, args.force_lock, log)
        frame = collect_rows(plan, class_order, subset)
        aggregates = aggregate(frame, plan.variants)
        pairs = paired_comparison(frame, REFERENCE_VARIANT) if REFERENCE_VARIANT in aggregates else {}
        margins = margin_versus_variance(aggregates, pairs)
        decision = verdict(aggregates, pairs, margins, n_seeds_planned=len(plan.seeds))
        write_study_artifacts(plan, frame, aggregates, pairs, margins, decision,
                              subset, class_order, outcomes)
        log("\n" + render_report(plan, frame, aggregates, pairs, margins, decision))
        log(f"\nArtefacts written to {plan.study_dir}")
    finally:
        study_lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
