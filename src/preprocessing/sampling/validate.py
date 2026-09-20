"""Independent verification of a generated experiment subset.

The sampler already asserts integrity while writing.  This module re-derives
the same guarantees from the artefacts alone, so a subset can be audited long
after it was produced -- including determinism, which is checked by replanning
from the immutable source and comparing plan digests.

Every check is streaming and bounded; only the selected sample identifiers and
their modality paths are retained.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.common.paths import DATASETS_DIR
from src.preprocessing.sampling.rules import get_modality_rule, get_task_rule, is_missing
from src.preprocessing.sampling.sampler import (
    MISSING,
    SPLIT_COLUMN,
    SPLITS,
    ExperimentSampler,
    SamplerConfig,
)


DEFAULT_FILE_CHECKS = 200
SHARE_TOLERANCE = 0.02
RATIO_TOLERANCE = 0.02
FRACTION_TOLERANCE = 0.01


@dataclass
class ValidationReport:
    """Collected check results; ``passed`` is the conjunction of all of them."""

    metadata_dir: str = ""
    checks: list[dict] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"check": name, "passed": bool(passed), "detail": detail})

    @property
    def passed(self) -> bool:
        return all(check["passed"] for check in self.checks)

    @property
    def failures(self) -> list[dict]:
        return [check for check in self.checks if not check["passed"]]

    def to_dict(self) -> dict:
        return {
            "metadata_dir": self.metadata_dir,
            "passed": self.passed,
            "counts": self.counts,
            "checks": self.checks,
            "failures": self.failures,
            "notes": self.notes,
        }


def _load_summary(metadata_dir: Path) -> dict:
    path = metadata_dir / "sampling_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"sampling_summary.json not found in {metadata_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_summary(summary: dict, source: Path | str | None = None) -> SamplerConfig:
    """Rebuild the exact sampler configuration recorded at generation time."""
    payload = dict(summary["config"])
    if payload.get("datasets") is not None:
        payload["datasets"] = tuple(payload["datasets"])
    payload["stratify_by"] = tuple(payload["stratify_by"])
    payload["split_ratios"] = tuple(payload["split_ratios"])
    if source is not None:
        payload["source"] = str(source)
    return SamplerConfig(**payload)


def validate_experiment(
    metadata_dir: Path | str,
    source: Path | str | None = None,
    check_files: int = DEFAULT_FILE_CHECKS,
    verify_determinism: bool = False,
    datasets_root: Path | str = DATASETS_DIR,
    batch_size: int = 16_384,
    ratio_tolerance: float = RATIO_TOLERANCE,
    share_tolerance: float = SHARE_TOLERANCE,
    fraction_tolerance: float = FRACTION_TOLERANCE,
) -> ValidationReport:
    """Verify split integrity, representativeness, and leakage safety."""
    metadata_dir = Path(metadata_dir)
    summary = _load_summary(metadata_dir)
    config = _config_from_summary(summary, source)
    modality_rule = get_modality_rule(config.modality)
    task_rule = get_task_rule(config.task)
    path_column = modality_rule.path_column

    report = ValidationReport(metadata_dir=str(metadata_dir))

    # ---------------------------------------------------------- read splits
    columns = {"sample_id", "dataset", "training_split", SPLIT_COLUMN}
    columns.update(modality_rule.required_columns)
    if task_rule.label_column:
        columns.add(task_rule.label_column)

    owner: dict[str, str] = {}
    paths: dict[str, str] = {}
    duplicates: dict[str, int] = {split: 0 for split in SPLITS}
    cross_split = 0
    invalid_labels = 0
    ineligible = 0
    foreign_datasets = 0
    holdout_in_train = 0
    mislabelled_split = 0
    path_backed = 0
    payload_backed = 0
    split_counts: dict[str, int] = {}
    split_datasets = {split: Counter() for split in SPLITS}
    split_classes = {split: Counter() for split in SPLITS}
    dataset_counts: Counter = Counter()
    class_counts: Counter = Counter()
    split_ids = {split: [] for split in SPLITS}
    allowed = set(config.datasets) if config.datasets else None

    for split in SPLITS:
        path = metadata_dir / f"{split}.parquet"
        if not path.exists():
            report.record(f"{split}_partition_exists", False, str(path))
            continue
        report.record(f"{split}_partition_exists", True, str(path))
        parquet = pq.ParquetFile(path)
        available = [name for name in sorted(columns) if name in parquet.schema_arrow.names]
        seen_in_split: set[str] = set()
        rows = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=available):
            values = {name: batch.column(name).to_pylist() for name in available}
            for index in range(batch.num_rows):
                rows += 1
                row = {name: values[name][index] for name in available}
                sample_id = str(row.get("sample_id"))
                if sample_id in seen_in_split:
                    duplicates[split] += 1
                seen_in_split.add(sample_id)
                previous = owner.get(sample_id)
                if previous is not None and previous != split:
                    cross_split += 1
                owner[sample_id] = split
                split_ids[split].append(sample_id)

                if row.get(SPLIT_COLUMN) not in (None, split):
                    mislabelled_split += 1

                dataset = str(row.get("dataset"))
                dataset_counts[dataset] += 1
                split_datasets[split][dataset] += 1
                if allowed is not None and dataset not in allowed:
                    foreign_datasets += 1

                official = (
                    MISSING if is_missing(row.get("training_split")) else str(row["training_split"])
                )
                if split == "train" and official in ("validation", "test"):
                    holdout_in_train += 1

                if not modality_rule.is_eligible(row):
                    ineligible += 1
                if path_column and not is_missing(row.get(path_column)):
                    paths[sample_id] = str(row[path_column])
                    path_backed += 1
                elif path_column:
                    payload_backed += 1

                if task_rule.label_column:
                    label = task_rule.label_of(row)
                    if label is None:
                        invalid_labels += 1
                    else:
                        class_counts[label] += 1
                        split_classes[split][label] += 1
        split_counts[split] = rows

    total = sum(split_counts.values())
    report.counts = {**split_counts, "total": total}

    # --------------------------------------------------------------- checks
    report.record("no_duplicate_sample_ids_within_split", sum(duplicates.values()) == 0, duplicates)
    report.record("no_sample_id_across_splits", cross_split == 0, {"cross_split_sample_ids": cross_split})
    report.record(
        "experiment_split_column_matches_partition", mislabelled_split == 0,
        {"mismatched_rows": mislabelled_split},
    )
    report.record(
        "canonical_labels_valid", invalid_labels == 0,
        {"invalid_label_records": invalid_labels, "valid_ids": list(task_rule.valid_label_ids or ())},
    )
    report.record(
        f"{config.modality}_modality_available_for_every_record", ineligible == 0,
        {"records_failing_modality_rule": ineligible},
    )
    report.record(
        "dataset_identity_preserved", foreign_datasets == 0,
        {"records_outside_contributing_datasets": foreign_datasets,
         "contributing_datasets": sorted(allowed) if allowed else None},
    )
    report.record(
        "official_holdout_never_in_train", holdout_in_train == 0,
        {"official_validation_or_test_records_in_train": holdout_in_train,
         "split_policy": config.split_policy},
    )

    # Split proportions.
    ratios = dict(zip(SPLITS, config.split_ratios))
    actual = {split: (split_counts.get(split, 0) / total if total else 0.0) for split in SPLITS}
    ratio_ok = all(abs(actual[split] - ratios[split]) <= ratio_tolerance for split in SPLITS)
    report.record(
        "split_ratios_within_tolerance", ratio_ok,
        {"expected": ratios, "actual": actual, "tolerance": ratio_tolerance},
    )

    # Realised fraction of the eligible pool.  Enforcing a minimum per stratum
    # can push it above the request, so the check is one-sided below and
    # tolerant above, and the deviation is always reported.
    eligible = summary["pool"]["eligible_records"]
    requested = float(summary["selection"]["requested_fraction"])
    realised = (total / eligible) if eligible else 0.0
    fraction_ok = (
        abs(realised - requested) <= fraction_tolerance
        or (realised > requested and summary["deviations"]["minimum_per_stratum_adjustments"])
    )
    report.record(
        "selected_fraction_matches_request", bool(fraction_ok),
        {"requested_fraction": requested, "realised_fraction": realised,
         "eligible_records": eligible, "selected_records": total,
         "tolerance": fraction_tolerance,
         "minimum_per_stratum_adjustments":
             len(summary["deviations"]["minimum_per_stratum_adjustments"])},
    )
    report.record(
        "deviations_are_documented",
        isinstance(summary["deviations"].get("policy"), str)
        and bool(summary["deviations"]["policy"].strip()),
        {"quota_exceptions": len(summary["deviations"]["quota_exceptions"]),
         "minimum_per_stratum_adjustments":
             len(summary["deviations"]["minimum_per_stratum_adjustments"])},
    )

    # Representation drift against the pool recorded at sampling time.
    pool_dataset_share = summary["representativeness"]["dataset_share_before"]
    pool_class_share = summary["representativeness"]["class_share_before"]
    dataset_drift = _drift(pool_dataset_share, _share(dataset_counts))
    class_drift = _drift(pool_class_share, _share(class_counts))
    report.record(
        "dataset_representation_preserved", dataset_drift <= share_tolerance,
        {"max_share_drift": dataset_drift, "tolerance": share_tolerance},
    )
    report.record(
        "class_representation_preserved", class_drift <= share_tolerance,
        {"max_share_drift": class_drift, "tolerance": share_tolerance},
    )

    # Counts and digests recorded by the sampler.
    recorded_counts = summary["splits"]["counts"]
    report.record(
        "counts_match_sampling_summary",
        all(split_counts.get(split, 0) == recorded_counts.get(split) for split in SPLITS),
        {"recorded": recorded_counts, "observed": split_counts},
    )
    digests = {split: _digest(split_ids[split]) for split in SPLITS}
    recorded_digests = summary["splits"].get("sample_id_digests", {})
    report.record(
        "sample_id_digests_match_sampling_summary",
        all(digests[split] == recorded_digests.get(split) for split in SPLITS),
        {"recorded": recorded_digests, "observed": digests},
    )

    # Source path preservation.  A modality can be legitimately path-free for a
    # whole partition -- MELD text lives in the metadata column, not a file --
    # so an empty path set is a note, not a failure.
    if path_column and paths:
        report.record(*_check_paths(config, path_column, owner, paths))
    elif path_column:
        report.notes.append(
            f"No record carries {path_column!r}; every selected {config.modality} record is "
            f"payload-backed ({payload_backed} records). Path check skipped."
        )
    else:
        report.notes.append(f"Modality {config.modality!r} is not file-backed; path check skipped.")

    # Bounded existence check on the real media files.
    if path_column and paths and check_files:
        report.record(*_check_files(paths, Path(datasets_root), check_files, config.seed))
    elif path_column and paths:
        report.notes.append("File-existence check disabled (check_files=0).")
    if path_column:
        report.counts["path_backed"] = path_backed
        report.counts["payload_backed"] = payload_backed

    # Determinism: replan from the immutable source and compare digests.
    if verify_determinism:
        plan = ExperimentSampler(config).plan()
        recorded = summary["selection"].get("plan_digest")
        report.record(
            "sampling_deterministic_for_seed", plan.digest == recorded,
            {"seed": config.seed, "recorded_plan_digest": recorded, "replanned_digest": plan.digest},
        )
    else:
        report.notes.append("Determinism replan skipped (pass verify_determinism=True to enable).")

    return report


def _share(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {str(key): value / total for key, value in counter.items()}


def _drift(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return max(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys)


def _digest(sample_ids: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for sample_id in sorted(sample_ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _check_paths(
    config: SamplerConfig, path_column: str, owner: dict[str, str], paths: dict[str, str]
) -> tuple[str, bool, dict]:
    """Compare every retained modality path against the standardized source."""
    source = Path(config.source)
    if not source.exists():
        return ("source_paths_unchanged", False, {"error": f"source not found: {source}"})
    parquet = pq.ParquetFile(source)
    columns = [name for name in ("sample_id", path_column) if name in parquet.schema_arrow.names]
    if len(columns) != 2:
        return ("source_paths_unchanged", False, {"error": f"source lacks {path_column}"})
    mismatched = 0
    matched = 0
    missing = set(paths)
    for batch in parquet.iter_batches(batch_size=config.batch_size, columns=columns):
        ids = batch.column("sample_id").to_pylist()
        values = batch.column(path_column).to_pylist()
        for sample_id, value in zip(ids, values):
            key = str(sample_id)
            if key not in paths:
                continue
            missing.discard(key)
            if str(value) == paths[key]:
                matched += 1
            else:
                mismatched += 1
    passed = mismatched == 0 and not missing
    return (
        "source_paths_unchanged",
        passed,
        {
            "path_column": path_column,
            "matched": matched,
            "mismatched": mismatched,
            "selected_ids_absent_from_source": len(missing),
        },
    )


def _check_files(
    paths: dict[str, str], datasets_root: Path, limit: int, seed: int
) -> tuple[str, bool, dict]:
    """Check a deterministic bounded sample of media files actually exist."""
    keys = sorted(paths)
    if not keys:
        return ("modality_files_exist", False, {"error": "no modality paths recorded"})
    if limit < 0 or limit >= len(keys):
        sampled = keys
    else:
        sampled = sorted(random.Random(seed).sample(keys, limit))
    missing = [paths[key] for key in sampled if not (datasets_root / paths[key]).exists()]
    return (
        "modality_files_exist",
        not missing,
        {
            "datasets_root": str(datasets_root),
            "checked": len(sampled),
            "of_total": len(keys),
            "missing": missing[:10],
            "missing_count": len(missing),
        },
    )
