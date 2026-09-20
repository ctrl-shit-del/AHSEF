"""Independent verification of the WESAD physiological-window experiment.

The generic subset validator audits record-level sampling from the standardized
table.  Physiology is built differently -- subject recordings expanded into
windows -- so it needs its own audit with one extra, load-bearing property:
**no subject may appear in more than one partition**.  Everything else mirrors
the generic checks so both reports read the same way.

Every check is derived from the written artefacts, not from the builder's
in-memory state, so an experiment can be audited long after it was produced.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.common.labels import get_label_space
from src.common.paths import DATASETS_DIR
from src.preprocessing.sampling.physiology_windows import (
    FEATURE_NAMES,
    SPLIT_COLUMN,
    SPLITS,
)
from src.preprocessing.sampling.validate import ValidationReport


#: Whole subjects are indivisible, so realised ratios deviate much further from
#: the request than record-level sampling ever would.
SUBJECT_RATIO_TOLERANCE = 0.25


def _load_summary(metadata_dir: Path) -> dict:
    path = metadata_dir / "sampling_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"sampling_summary.json not found in {metadata_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(sample_ids: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for sample_id in sorted(sample_ids):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def validate_physiology_experiment(
    metadata_dir: Path | str,
    check_files: int = 1,
    datasets_root: Path | str = DATASETS_DIR,
    batch_size: int = 16_384,
    ratio_tolerance: float = SUBJECT_RATIO_TOLERANCE,
) -> ValidationReport:
    """Verify window integrity, subject disjointness, and label validity."""
    metadata_dir = Path(metadata_dir)
    summary = _load_summary(metadata_dir)
    config = summary["config"]
    label_space = get_label_space(config["task"])

    report = ValidationReport(metadata_dir=str(metadata_dir))

    columns = [
        "sample_id", "dataset", "subject", "state", "state_id", "features",
        "physiology_path", SPLIT_COLUMN,
    ]

    owner: dict[str, str] = {}
    subject_splits: dict[str, set[str]] = defaultdict(set)
    duplicates = {split: 0 for split in SPLITS}
    cross_split = 0
    invalid_labels = 0
    mismatched_state = 0
    mislabelled_split = 0
    foreign_datasets = 0
    feature_widths: set[int] = set()
    split_counts: dict[str, int] = {}
    split_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    split_classes = {split: Counter() for split in SPLITS}
    class_counts: Counter = Counter()
    recordings: dict[str, str] = {}

    for split in SPLITS:
        path = metadata_dir / f"{split}.parquet"
        if not path.exists():
            report.record(f"{split}_partition_exists", False, str(path))
            continue
        report.record(f"{split}_partition_exists", True, str(path))
        parquet = pq.ParquetFile(path)
        available = [name for name in columns if name in parquet.schema_arrow.names]
        missing = set(columns) - set(available)
        if missing:
            report.record(f"{split}_partition_schema", False, {"missing_columns": sorted(missing)})
            continue
        seen: set[str] = set()
        rows = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=available):
            values = {name: batch.column(name).to_pylist() for name in available}
            for index in range(batch.num_rows):
                rows += 1
                row = {name: values[name][index] for name in available}
                sample_id = str(row["sample_id"])
                if sample_id in seen:
                    duplicates[split] += 1
                seen.add(sample_id)
                previous = owner.get(sample_id)
                if previous is not None and previous != split:
                    cross_split += 1
                owner[sample_id] = split
                split_ids[split].append(sample_id)

                if row.get(SPLIT_COLUMN) not in (None, split):
                    mislabelled_split += 1
                if row.get("dataset") != config["dataset"]:
                    foreign_datasets += 1

                subject_splits[str(row["subject"])].add(split)

                state_id = row.get("state_id")
                if state_id is None or int(state_id) not in label_space.valid_ids:
                    invalid_labels += 1
                else:
                    state_id = int(state_id)
                    class_counts[state_id] += 1
                    split_classes[split][state_id] += 1
                    if row.get("state") != label_space.name_of(state_id):
                        mismatched_state += 1

                features = row.get("features")
                feature_widths.add(len(features) if features is not None else -1)
                if row.get("physiology_path"):
                    recordings[str(row["subject"])] = str(row["physiology_path"])
        split_counts[split] = rows

    total = sum(split_counts.values())
    report.counts = {**split_counts, "total": total, "subjects": len(subject_splits)}

    # ------------------------------------------------------------- integrity
    report.record("no_duplicate_sample_ids_within_split", sum(duplicates.values()) == 0, duplicates)
    report.record(
        "no_sample_id_across_splits", cross_split == 0, {"cross_split_sample_ids": cross_split}
    )
    report.record(
        "experiment_split_column_matches_partition", mislabelled_split == 0,
        {"mismatched_rows": mislabelled_split},
    )
    report.record(
        "dataset_identity_preserved", foreign_datasets == 0,
        {"records_outside_contributing_datasets": foreign_datasets,
         "contributing_datasets": [config["dataset"]]},
    )

    overlapping = sorted(
        subject for subject, splits in subject_splits.items() if len(splits) > 1
    )
    report.record(
        "subjects_are_disjoint_across_splits", not overlapping,
        {"subjects_in_multiple_splits": overlapping,
         "subjects_per_split": {
             split: sorted(s for s, v in subject_splits.items() if split in v) for split in SPLITS
         }},
    )

    report.record(
        "state_labels_valid", invalid_labels == 0,
        {"invalid_label_records": invalid_labels,
         "label_space": label_space.name,
         "valid_ids": list(label_space.valid_ids)},
    )
    report.record(
        "state_names_match_declared_class_order", mismatched_state == 0,
        {"mismatched_rows": mismatched_state, "class_order": list(label_space.classes)},
    )
    report.record(
        "every_class_is_represented",
        set(class_counts) == set(label_space.valid_ids),
        {"present": sorted(class_counts), "expected": list(label_space.valid_ids)},
    )
    report.record(
        "feature_width_is_consistent",
        feature_widths == {len(FEATURE_NAMES)},
        {"observed_widths": sorted(feature_widths), "expected": len(FEATURE_NAMES)},
    )

    # ---------------------------------------------------------------- splits
    ratios = dict(zip(SPLITS, config["split_ratios"]))
    actual = {split: (split_counts.get(split, 0) / total if total else 0.0) for split in SPLITS}
    report.record(
        "split_ratios_within_subject_granularity_tolerance",
        all(abs(actual[split] - ratios[split]) <= ratio_tolerance for split in SPLITS),
        {"expected": ratios, "actual": actual, "tolerance": ratio_tolerance,
         "note": "Whole subjects are indivisible; deviation is expected and recorded."},
    )
    report.record(
        "counts_match_sampling_summary",
        all(split_counts.get(split, 0) == summary["splits"]["counts"].get(split) for split in SPLITS),
        {"recorded": summary["splits"]["counts"], "observed": split_counts},
    )
    digests = {split: _digest(split_ids[split]) for split in SPLITS}
    report.record(
        "sample_id_digests_match_sampling_summary",
        all(digests[split] == summary["splits"]["sample_id_digests"].get(split) for split in SPLITS),
        {"recorded": summary["splits"]["sample_id_digests"], "observed": digests},
    )
    report.record(
        "full_pool_is_used",
        abs(float(summary["selection"]["actual_fraction"]) - 1.0) < 1e-9,
        {"actual_fraction": summary["selection"]["actual_fraction"]},
    )
    report.record(
        "deviations_are_documented",
        bool((summary.get("deviations") or {}).get("policy")),
        {"subject_split_rounding": (summary.get("deviations") or {}).get("subject_split_rounding")},
    )

    # ------------------------------------------------------------ recordings
    if check_files:
        datasets_root = Path(datasets_root)
        wanted = sorted(recordings)[:check_files] if check_files > 0 else sorted(recordings)
        missing_files = [
            recordings[subject] for subject in wanted
            if not (datasets_root / recordings[subject]).exists()
        ]
        report.record(
            "source_recordings_exist", not missing_files,
            {"datasets_root": str(datasets_root), "checked": len(wanted),
             "of_total": len(recordings), "missing": missing_files[:10]},
        )
    else:
        report.notes.append("Recording-existence check disabled (check_files=0).")

    report.notes.append(
        f"WESAD carries no canonical emotion target; the declared label space is "
        f"{label_space.name} ({', '.join(label_space.classes)})."
    )
    return report
