"""Reusable representative fraction sampler for modality-filtered experiments.

Design contract
---------------
* The standardized table is immutable input; records are copied verbatim and
  only an ``experiment_split`` column is appended.  Source paths are never
  rewritten and no media file is duplicated.
* Two bounded streaming passes are used.  The first pass keeps only integer
  row ordinals per stratum, the second re-reads the source and writes the
  three split partitions.  Neither pass materialises the whole table.
* Selection is stratified by ``stratify_by`` (default ``dataset`` x
  ``canonical_emotion_id``) so a large dataset cannot dominate the subset.
* Official split boundaries are respected: under the default
  ``official_aware`` policy a record whose standardized ``training_split`` is
  ``test`` can only land in the experiment test split, and a record whose
  official split is ``validation`` can only land in validation or test.
  Nothing from an official holdout is ever promoted into experiment training.
* Every deviation from exact proportionality is recorded in the summary
  instead of failing silently.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Hashable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.sampling.allocation import (
    Cursor,
    allocate_fraction,
    deterministic_shuffle,
    largest_remainder,
)
from src.preprocessing.sampling.rules import get_modality_rule, get_task_rule, is_missing
from src.utils.io import ensure_dir


SPLITS = ("train", "validation", "test")
SPLIT_COLUMN = "experiment_split"

FREE = "free"
LOCKED_VALIDATION = "locked_validation"
LOCKED_TEST = "locked_test"
LOCK_GROUPS = (FREE, LOCKED_VALIDATION, LOCKED_TEST)

DEFAULT_SOURCE = Path("metadata/standardized/standardized.parquet")
DEFAULT_BATCH_SIZE = 16_384
MISSING = "__missing__"
SPLIT_POLICIES = ("official_aware", "free")
HOLDOUT_OVERFLOW = ("drop", "absorb")


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SamplerConfig:
    """Fully reproducible sampling configuration."""

    modality: str = "image"
    task: str = "emotion_7class"
    fraction: float = 0.25
    seed: int = 42
    stratify_by: tuple[str, ...] = ("dataset", "canonical_emotion_id")
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    datasets: tuple[str, ...] | None = None
    split_policy: str = "official_aware"
    #: What to do with selected records that no split quota can hold, which
    #: happens when a dataset's official holdout is larger than the experiment
    #: quota for the same split.  ``drop`` keeps the split ratios exact and
    #: leaves those records unselected; ``absorb`` places them in a split they
    #: are permitted to occupy, which uses the whole selected budget at the
    #: cost of slightly larger holdout splits.  Neither ever moves an official
    #: holdout record into experiment training.
    holdout_overflow: str = "drop"
    minimum_per_stratum: int = 1
    batch_size: int = DEFAULT_BATCH_SIZE
    source: str = str(DEFAULT_SOURCE)
    max_strata_in_summary: int = 512

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must be (train, validation, test)")
        if any(ratio < 0 for ratio in self.split_ratios):
            raise ValueError("split_ratios must be non-negative")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}")
        if self.split_policy not in SPLIT_POLICIES:
            raise ValueError(f"split_policy must be one of {SPLIT_POLICIES}")
        if self.holdout_overflow not in HOLDOUT_OVERFLOW:
            raise ValueError(f"holdout_overflow must be one of {HOLDOUT_OVERFLOW}")
        if not self.stratify_by:
            raise ValueError("stratify_by must contain at least one column")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        # Normalise sequences so equality and serialisation stay stable.
        object.__setattr__(self, "stratify_by", tuple(self.stratify_by))
        object.__setattr__(self, "split_ratios", tuple(float(ratio) for ratio in self.split_ratios))
        if self.datasets is not None:
            object.__setattr__(self, "datasets", tuple(self.datasets))


# ============================================================
# Intermediate results
# ============================================================

@dataclass
class _Pool:
    """Eligible-record ordinals grouped by stratum and lock group."""

    groups: dict[tuple, dict[str, list[int]]] = field(default_factory=dict)
    dataset_counts: Counter = field(default_factory=Counter)
    class_counts: Counter = field(default_factory=Counter)
    dataset_class_counts: Counter = field(default_factory=Counter)
    official_counts: Counter = field(default_factory=Counter)
    dataset_official_counts: Counter = field(default_factory=Counter)
    exclusions: Counter = field(default_factory=Counter)
    scanned_rows: int = 0
    eligible_total: int = 0


@dataclass
class SamplingPlan:
    """Row-ordinal to split assignment plus the audit trail behind it."""

    assignments: dict[int, str]
    strata: list[dict]
    quota_exceptions: list[dict]
    minimum_adjustments: list[dict]
    target_total: int
    digest: str
    #: Records the allocation selected but no permitted split could hold.
    unplaced_records: int = 0

    @property
    def selected_total(self) -> int:
        return len(self.assignments)

    def counts(self) -> dict[str, int]:
        counts = Counter(self.assignments.values())
        return {split: int(counts[split]) for split in SPLITS}


# ============================================================
# Sampler
# ============================================================

class ExperimentSampler:
    """Select a representative fraction of a modality pool and split it."""

    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        self.modality_rule = get_modality_rule(self.config.modality)
        self.task_rule = get_task_rule(self.config.task)
        self.source = Path(self.config.source)
        self.datasets = self.config.datasets or self.modality_rule.default_datasets

    # ------------------------------------------------------------- plumbing

    @property
    def scan_columns(self) -> list[str]:
        columns = {"dataset", "training_split"}
        columns.update(self.modality_rule.required_columns)
        columns.update(self.task_rule.required_columns)
        columns.update(self.config.stratify_by)
        return sorted(columns)

    def _open_source(self) -> pq.ParquetFile:
        if not self.source.exists():
            raise FileNotFoundError(f"Sampling source not found: {self.source}")
        parquet = pq.ParquetFile(self.source)
        missing = set(self.scan_columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(
                f"Sampling source {self.source} is missing columns required by the "
                f"{self.config.modality!r} modality rule: {sorted(missing)}. "
                f"Regenerate the standardized metadata with "
                f"'python -m src.preprocessing.standardization.run_standardization' "
                f"so the current schema is materialised."
            )
        return parquet

    @staticmethod
    def _rows(batch: pa.RecordBatch, columns: list[str]) -> Iterator[dict[str, Any]]:
        """Yield plain dicts for one bounded batch.

        Materialising a batch (not the table) keeps peak memory proportional to
        ``batch_size`` while avoiding version-sensitive compute kernels.
        """
        values = {name: batch.column(name).to_pylist() for name in columns}
        for index in range(batch.num_rows):
            yield {name: values[name][index] for name in columns}

    @staticmethod
    def _normalise(value: Any) -> Any:
        """Make a stratification value hashable and stable across dtypes."""
        if is_missing(value):
            return MISSING
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, float):
            # An integral float and its int are the same stratum.
            return int(value) if value.is_integer() else value
        if isinstance(value, (int, str)):
            return value
        return str(value)

    def _stratum_key(self, row: Mapping[str, Any]) -> tuple:
        return tuple(self._normalise(row.get(column)) for column in self.config.stratify_by)

    def _lock_group(self, official_split: Any) -> str:
        if self.config.split_policy == "free":
            return FREE
        if official_split == "test":
            return LOCKED_TEST
        if official_split == "validation":
            return LOCKED_VALIDATION
        return FREE

    # ------------------------------------------------------------ pass one

    def scan(self) -> _Pool:
        """Stream the source and index eligible row ordinals per stratum."""
        parquet = self._open_source()
        columns = self.scan_columns
        allowed = set(self.datasets) if self.datasets else None
        pool = _Pool()
        ordinal = 0

        for batch in parquet.iter_batches(batch_size=self.config.batch_size, columns=columns):
            for row in self._rows(batch, columns):
                current, ordinal = ordinal, ordinal + 1
                pool.scanned_rows += 1
                dataset = row.get("dataset")
                if allowed is not None and dataset not in allowed:
                    pool.exclusions["dataset_not_contributing"] += 1
                    continue
                if not self.task_rule.is_eligible(row):
                    pool.exclusions["invalid_or_missing_target"] += 1
                    continue
                if not self.modality_rule.is_eligible(row):
                    pool.exclusions["modality_unavailable"] += 1
                    continue

                official = MISSING if is_missing(row.get("training_split")) else str(row["training_split"])
                key = self._stratum_key(row)
                bucket = pool.groups.setdefault(key, {group: [] for group in LOCK_GROUPS})
                bucket[self._lock_group(official)].append(current)

                pool.eligible_total += 1
                pool.dataset_counts[str(dataset)] += 1
                pool.official_counts[official] += 1
                pool.dataset_official_counts[(str(dataset), official)] += 1
                label = self.task_rule.label_of(row)
                if label is not None:
                    pool.class_counts[label] += 1
                    pool.dataset_class_counts[(str(dataset), label)] += 1

        if pool.eligible_total == 0:
            raise ValueError(
                f"No eligible {self.config.modality} records for task {self.config.task!r} "
                f"in {self.source}"
            )
        return pool

    # ------------------------------------------------------------ pass two

    def plan(self, pool: _Pool | None = None) -> SamplingPlan:
        """Turn an eligible pool into a deterministic ordinal to split map."""
        pool = pool if pool is not None else self.scan()
        sizes = {key: sum(len(items) for items in groups.values()) for key, groups in pool.groups.items()}
        allocation = allocate_fraction(sizes, self.config.fraction, self.config.minimum_per_stratum)

        assignments: dict[int, str] = {}
        strata: list[dict] = []
        exceptions: list[dict] = []
        unplaced = 0

        for key in sorted(pool.groups, key=repr):
            selected = allocation.counts[key]
            cursors = {
                group: Cursor(
                    deterministic_shuffle(
                        pool.groups[key][group], self.config.seed, self.config.modality, key, group
                    )
                )
                for group in LOCK_GROUPS
            }
            quota = dict(zip(SPLITS, largest_remainder(selected, self.config.split_ratios)))
            chosen = self._assign_stratum(
                cursors, quota, selected, self.config.holdout_overflow
            )

            for split, ordinals in chosen.items():
                for value in ordinals:
                    assignments[value] = split

            assigned = {split: len(chosen[split]) for split in SPLITS}
            placed = sum(assigned.values())
            if placed < selected:
                unplaced += selected - placed
            if assigned != quota:
                exceptions.append({
                    "stratum": self._stratum_record(key),
                    "reason": (
                        "insufficient unlocked records to honour the exact split quota"
                        if placed < selected
                        else "official holdout records absorbed into the splits they may occupy"
                    ),
                    "requested": quota,
                    "assigned": assigned,
                    "selected": selected,
                    "unplaced": selected - placed,
                    "holdout_overflow": self.config.holdout_overflow,
                    "pool_by_lock_group": {group: len(pool.groups[key][group]) for group in LOCK_GROUPS},
                })

            strata.append({
                "stratum": self._stratum_record(key),
                "pool": sizes[key],
                "pool_by_lock_group": {group: len(pool.groups[key][group]) for group in LOCK_GROUPS},
                "selected": sum(assigned.values()),
                "quota": quota,
                "assigned": assigned,
            })

        digest = hashlib.blake2b(digest_size=16)
        for ordinal in sorted(assignments):
            digest.update(f"{ordinal}:{assignments[ordinal]}|".encode("utf-8"))

        return SamplingPlan(
            assignments=assignments,
            strata=strata,
            quota_exceptions=exceptions,
            minimum_adjustments=allocation.minimum_adjustments,
            target_total=allocation.target_total,
            digest=digest.hexdigest(),
            unplaced_records=unplaced,
        )

    @staticmethod
    def _assign_stratum(
        cursors: dict[str, Cursor],
        quota: dict[str, int],
        selected: int | None = None,
        holdout_overflow: str = "drop",
    ) -> dict[str, list[int]]:
        """Fill split quotas, honouring lock groups and never leaking holdouts.

        Locked records are consumed first for the split they belong to so that
        official holdout material is preferred for the matching experiment
        split; the remainder is filled from unlocked (official-train or
        unassigned) records.  Leftover locked-validation records may fall back
        to test, never to train.

        When a dataset's official test partition is proportionally larger than
        the experiment's test quota, some selected records fit nowhere.  With
        ``holdout_overflow='drop'`` they stay unselected, which keeps the split
        ratios exact; with ``'absorb'`` they are placed in a split they are
        permitted to occupy, which consumes the whole selected budget at the
        cost of a slightly larger holdout split.  Training never receives an
        official holdout record under either setting.
        """
        chosen: dict[str, list[int]] = {split: [] for split in SPLITS}

        chosen["test"].extend(cursors[LOCKED_TEST].take(quota["test"]))
        chosen["validation"].extend(cursors[LOCKED_VALIDATION].take(quota["validation"]))
        chosen["test"].extend(cursors[FREE].take(quota["test"] - len(chosen["test"])))
        chosen["validation"].extend(cursors[FREE].take(quota["validation"] - len(chosen["validation"])))
        chosen["train"].extend(cursors[FREE].take(quota["train"]))

        # Degenerate strata (for example a stratum made only of official test
        # records) cannot fill every quota from unlocked records.  Top up the
        # holdout splits from leftover locked material rather than leaking it.
        chosen["validation"].extend(
            cursors[LOCKED_VALIDATION].take(quota["validation"] - len(chosen["validation"]))
        )
        chosen["test"].extend(cursors[LOCKED_TEST].take(quota["test"] - len(chosen["test"])))
        chosen["test"].extend(cursors[LOCKED_VALIDATION].take(quota["test"] - len(chosen["test"])))

        if holdout_overflow == "absorb" and selected is not None:
            placed = sum(len(ordinals) for ordinals in chosen.values())
            for group, split in (
                (LOCKED_TEST, "test"),
                (LOCKED_VALIDATION, "validation"),
                (FREE, "train"),
            ):
                if placed >= selected:
                    break
                taken = cursors[group].take(selected - placed)
                chosen[split].extend(taken)
                placed += len(taken)
        return chosen

    def _stratum_record(self, key: tuple) -> dict[str, Any]:
        return {column: value for column, value in zip(self.config.stratify_by, key)}

    # ---------------------------------------------------------------- write

    def write(self, plan: SamplingPlan, output_dir: Path | str) -> "_WriteReport":
        """Stream the source again and materialise the three split files."""
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        parquet = self._open_source()
        source_names = [name for name in parquet.schema_arrow.names if name != SPLIT_COLUMN]
        schema = pa.schema([parquet.schema_arrow.field(name) for name in source_names])
        schema = schema.append(pa.field(SPLIT_COLUMN, pa.string()))

        report = _WriteReport(
            datasets=self.datasets, modality_rule=self.modality_rule, task_rule=self.task_rule
        )
        writers = {
            split: pq.ParquetWriter(output_dir / f"{split}.parquet", schema, compression="snappy")
            for split in SPLITS
        }
        assignments = plan.assignments
        ordinal = 0
        try:
            for batch in parquet.iter_batches(batch_size=self.config.batch_size):
                table = pa.Table.from_batches([batch]).select(source_names)
                rows = table.num_rows
                labels = [assignments.get(ordinal + index) for index in range(rows)]
                ordinal += rows
                if not any(labels):
                    continue
                for split in SPLITS:
                    mask = pa.array([label == split for label in labels], type=pa.bool_())
                    selected = table.filter(mask)
                    if not selected.num_rows:
                        continue
                    selected = selected.append_column(
                        SPLIT_COLUMN, pa.array([split] * selected.num_rows, type=pa.string())
                    )
                    report.observe(selected, split)
                    writers[split].write_table(selected)
        finally:
            for writer in writers.values():
                writer.close()

        report.finalise()
        return report

    # ------------------------------------------------------------------ run

    def run(self, output_dir: Path | str) -> dict:
        """Scan, plan, write, and persist ``sampling_summary.json``."""
        output_dir = Path(output_dir)
        pool = self.scan()
        plan = self.plan(pool)
        report = self.write(plan, output_dir)
        summary = self.summarise(pool, plan, report, output_dir)
        ensure_dir(output_dir)
        (output_dir / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    # -------------------------------------------------------------- summary

    def summarise(
        self, pool: _Pool, plan: SamplingPlan, report: "_WriteReport", output_dir: Path
    ) -> dict:
        counts = plan.counts()
        selected_total = sum(counts.values())
        strata = plan.strata if len(plan.strata) <= self.config.max_strata_in_summary else []
        return {
            "experiment": {
                "modality": self.config.modality,
                "task": self.config.task,
                "description": (
                    f"{self.config.modality.capitalize()}-only modality-filtered "
                    f"{self.config.task} representative subset."
                ),
                "output_dir": str(output_dir),
            },
            "sampling_method": (
                "stratified proportional selection by "
                f"{list(self.config.stratify_by)} using largest-remainder allocation, "
                "seeded per-stratum shuffling, and quota-driven "
                f"{'/'.join(format(ratio, 'g') for ratio in self.config.split_ratios)} "
                f"split assignment under the '{self.config.split_policy}' official-split policy"
            ),
            "config": asdict(self.config),
            "seed": self.config.seed,
            "source": str(self.source),
            "contributing_datasets": (
                list(self.datasets) if self.datasets else sorted(pool.dataset_counts)
            ),
            "stratification_fields": list(self.config.stratify_by),
            "eligibility": {
                "modality_rule": {
                    "availability_column": self.modality_rule.availability_column,
                    "source_column": self.modality_rule.source_column,
                    "accepted_sources": list(self.modality_rule.accepted_sources),
                    "path_column": self.modality_rule.path_column,
                },
                "task_rule": {
                    "target_type": self.task_rule.target_type,
                    "validity_column": self.task_rule.validity_column,
                    "label_column": self.task_rule.label_column,
                    "valid_label_ids": list(self.task_rule.valid_label_ids or ()),
                },
                "scanned_rows": pool.scanned_rows,
                "excluded": dict(pool.exclusions),
            },
            "pool": {
                "eligible_records": pool.eligible_total,
                "dataset_counts": dict(sorted(pool.dataset_counts.items())),
                "class_counts": {str(key): value for key, value in sorted(pool.class_counts.items())},
                "dataset_class_counts": _pair_counter(pool.dataset_class_counts),
                "official_split_counts": dict(sorted(pool.official_counts.items())),
                "dataset_official_split_counts": _pair_counter(pool.dataset_official_counts),
            },
            "selection": {
                "requested_fraction": self.config.fraction,
                "target_records": plan.target_total,
                "selected_records": selected_total,
                "actual_fraction": (
                    selected_total / pool.eligible_total if pool.eligible_total else 0.0
                ),
                "dataset_counts": dict(sorted(report.dataset_counts.items())),
                "class_counts": {
                    str(key): value for key, value in sorted(report.class_counts.items())
                },
                "dataset_class_counts": _pair_counter(report.dataset_class_counts),
                "official_split_counts": dict(sorted(report.official_counts.items())),
                "plan_digest": plan.digest,
            },
            "splits": {
                "ratios": dict(zip(SPLITS, self.config.split_ratios)),
                "counts": counts,
                "actual_ratios": {
                    split: (counts[split] / selected_total if selected_total else 0.0)
                    for split in SPLITS
                },
                "dataset_counts": {
                    split: dict(sorted(report.split_datasets[split].items())) for split in SPLITS
                },
                "class_counts": {
                    split: {str(key): value for key, value in sorted(report.split_classes[split].items())}
                    for split in SPLITS
                },
                "official_split_counts": {
                    split: dict(sorted(report.split_official[split].items())) for split in SPLITS
                },
                "sample_id_digests": report.digests,
            },
            "representativeness": {
                "dataset_share_before": _shares(pool.dataset_counts),
                "dataset_share_after": _shares(report.dataset_counts),
                "class_share_before": _shares(pool.class_counts),
                "class_share_after": _shares(report.class_counts),
                "max_dataset_share_drift": _max_drift(pool.dataset_counts, report.dataset_counts),
                "max_class_share_drift": _max_drift(pool.class_counts, report.class_counts),
            },
            "integrity": {
                "duplicate_sample_ids": report.duplicate_ids,
                "cross_split_sample_ids": report.cross_split_ids,
                "invalid_label_records": report.invalid_labels,
                "records_outside_contributing_datasets": report.foreign_datasets,
                "records_with_missing_modality_source": report.missing_modality,
                "official_holdout_in_train": report.official_holdout_in_train,
                "source_paths_rewritten": 0,
                "source_path_column": self.modality_rule.path_column,
                "source_path_examples": report.path_examples,
                "leakage_checks_passed": report.leakage_ok,
            },
            "deviations": {
                "minimum_per_stratum_adjustments": plan.minimum_adjustments,
                "quota_exceptions": plan.quota_exceptions,
                "holdout_overflow": self.config.holdout_overflow,
                "unplaced_eligible_records": plan.unplaced_records,
                "unplaced_reason": (
                    "Selected records whose only permitted split was already full. This "
                    "happens when a dataset's official test or validation partition is "
                    "proportionally larger than the experiment quota for that split. Pass "
                    "--holdout-overflow absorb to place them instead of dropping them."
                    if plan.unplaced_records else None
                ),
                "policy": (
                    "Strata whose proportional share rounds below "
                    f"{self.config.minimum_per_stratum} are raised to that floor, which can make "
                    "the realised fraction exceed the requested one. Strata that cannot fill a "
                    "split quota from unlocked records keep the shortfall rather than moving "
                    "official holdout records into training."
                ),
            },
            "strata": strata,
            "strata_omitted": len(plan.strata) if not strata else 0,
        }


# ============================================================
# Write-time verification
# ============================================================

class _WriteReport:
    """Accumulate integrity evidence while the split files are written."""

    def __init__(self, datasets, modality_rule, task_rule):
        self.datasets = set(datasets) if datasets else None
        self.modality_rule = modality_rule
        self.task_rule = task_rule
        self.dataset_counts: Counter = Counter()
        self.class_counts: Counter = Counter()
        self.dataset_class_counts: Counter = Counter()
        self.official_counts: Counter = Counter()
        self.split_datasets = {split: Counter() for split in SPLITS}
        self.split_classes = {split: Counter() for split in SPLITS}
        self.split_official = {split: Counter() for split in SPLITS}
        self.split_ids: dict[str, list[str]] = defaultdict(list)
        self.duplicate_ids = 0
        self.cross_split_ids = 0
        self.invalid_labels = 0
        self.foreign_datasets = 0
        self.missing_modality = 0
        self.official_holdout_in_train = 0
        self.path_examples: list[dict] = []
        self.digests: dict[str, str] = {}
        self.leakage_ok = False
        self._seen: dict[str, str] = {}

    def observe(self, table: pa.Table, split: str) -> None:
        columns = {"sample_id", "dataset", "training_split"}
        columns.update(self.modality_rule.required_columns)
        if self.task_rule.label_column:
            columns.add(self.task_rule.label_column)
        available = [name for name in sorted(columns) if name in table.column_names]
        values = {name: table.column(name).to_pylist() for name in available}

        for index in range(table.num_rows):
            row = {name: values[name][index] for name in available}
            sample_id = str(row.get("sample_id"))
            previous = self._seen.get(sample_id)
            if previous is not None:
                self.duplicate_ids += 1
                if previous != split:
                    self.cross_split_ids += 1
            else:
                self._seen[sample_id] = split
            self.split_ids[split].append(sample_id)

            dataset = str(row.get("dataset"))
            official = (
                MISSING if is_missing(row.get("training_split")) else str(row["training_split"])
            )
            self.dataset_counts[dataset] += 1
            self.official_counts[official] += 1
            self.split_datasets[split][dataset] += 1
            self.split_official[split][official] += 1
            if self.datasets is not None and dataset not in self.datasets:
                self.foreign_datasets += 1
            if split == "train" and official in ("validation", "test"):
                self.official_holdout_in_train += 1
            if not self.modality_rule.is_eligible(row):
                self.missing_modality += 1
            if self.task_rule.label_column:
                label = self.task_rule.label_of(row)
                if label is None:
                    self.invalid_labels += 1
                else:
                    self.class_counts[label] += 1
                    self.dataset_class_counts[(dataset, label)] += 1
                    self.split_classes[split][label] += 1
            path_column = self.modality_rule.path_column
            if path_column and len(self.path_examples) < 3 and not is_missing(row.get(path_column)):
                self.path_examples.append({"sample_id": sample_id, path_column: row[path_column]})

    def finalise(self) -> None:
        for split in SPLITS:
            digest = hashlib.blake2b(digest_size=16)
            for sample_id in sorted(self.split_ids[split]):
                digest.update(sample_id.encode("utf-8"))
                digest.update(b"|")
            self.digests[split] = digest.hexdigest()
        self.leakage_ok = (
            self.duplicate_ids == 0
            and self.cross_split_ids == 0
            and self.invalid_labels == 0
            and self.foreign_datasets == 0
            and self.missing_modality == 0
            and self.official_holdout_in_train == 0
        )


# ============================================================
# Summary helpers
# ============================================================

def _pair_counter(counter: Counter) -> dict[str, int]:
    return {f"{left}|{right}": value for (left, right), value in sorted(counter.items(), key=repr)}


def _shares(counter: Mapping[Hashable, int]) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {str(key): counter[key] / total for key in sorted(counter, key=repr)}


def _max_drift(before: Mapping[Hashable, int], after: Mapping[Hashable, int]) -> float:
    left, right = _shares(before), _shares(after)
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
