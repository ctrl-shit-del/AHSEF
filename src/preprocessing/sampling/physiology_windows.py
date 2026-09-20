"""Expand WESAD subject recordings into a subject-disjoint window experiment.

Why this exists instead of the generic sampler
----------------------------------------------
The standardized metadata holds **fifteen** WESAD rows -- one per subject, each
pointing at a multi-hour ``S*.pkl``.  That is a pointer index, not a dataset:
there is nothing for the generic record-level sampler to stratify, and nothing
a dataloader could turn into a batch.  This module performs the one expansion
step WESAD needs, once, and writes ordinary experiment split Parquet that the
rest of the training stack consumes exactly like any other modality.

Two properties are non-negotiable and are enforced here rather than downstream:

*Subject-aware splitting.*  Windows from one subject are highly autocorrelated;
letting two windows of the same recording land in train and test would measure
memorisation, not generalisation.  The split is therefore made over *subjects*
and windows inherit it, so no subject appears in more than one partition.

*No emotion labels.*  WESAD annotates study conditions, not the canonical
7-class emotion taxonomy (``target_type == 'physiological_state'``,
``canonical_emotion_valid == False`` for every row).  The declared label space
is :data:`~src.common.labels.WESAD_STATE_3CLASS`; nothing here invents an
emotion.

Memory: one subject's recording is loaded at a time, converted to ``float32``,
reduced to per-window statistics, and released before the next subject opens.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.labels import WESAD_STATE_3CLASS, LabelSpace, get_label_space
from src.common.paths import DATASETS_DIR
from src.preprocessing.sampling.allocation import deterministic_shuffle, largest_remainder
from src.preprocessing.sampling.rules import get_modality_rule, is_missing
from src.utils.io import ensure_dir


SPLITS = ("train", "validation", "test")
SPLIT_COLUMN = "experiment_split"
DEFAULT_SOURCE = Path("metadata/standardized/standardized.parquet")

#: WESAD protocol condition codes.  0/5/6/7 are transition or ignore markers
#: and never become windows.
WESAD_LABEL_STATES: dict[int, str] = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

#: RespiBAN chest channels, all sampled at 700 Hz.  ``ACC`` contributes three
#: axes, so the channel list below is the flattened, ordered signal set.
CHEST_SAMPLE_RATE = 700
CHEST_CHANNELS: tuple[str, ...] = (
    "ACC_x", "ACC_y", "ACC_z", "ECG", "EMG", "EDA", "Temp", "Resp",
)

#: Per-channel statistics, in a fixed order: the feature vector's meaning is
#: positional and is recorded alongside the windows.
FEATURE_STATISTICS: tuple[str, ...] = (
    "mean", "std", "min", "max", "p25", "p75", "mean_abs_diff",
)

FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{channel}_{statistic}" for channel in CHEST_CHANNELS for statistic in FEATURE_STATISTICS
)


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class PhysiologyWindowConfig:
    """Fully reproducible window-extraction and splitting configuration."""

    dataset: str = "WESAD"
    modality: str = "physiology"
    task: str = "wesad_state_3class"
    window_seconds: float = 60.0
    stride_seconds: float = 10.0
    sample_rate: int = CHEST_SAMPLE_RATE
    signal_group: str = "chest"
    seed: int = 42
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    subject_aware: bool = True
    source: str = str(DEFAULT_SOURCE)
    datasets_root: str = str(DATASETS_DIR)

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive")
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride_seconds must not exceed window_seconds")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if len(self.split_ratios) != 3:
            raise ValueError("split_ratios must be (train, validation, test)")
        if abs(sum(self.split_ratios) - 1.0) > 1e-9:
            raise ValueError(f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}")
        if self.signal_group != "chest":
            raise ValueError(
                "Only the 'chest' (RespiBAN) signal group is supported; its channels "
                "share one sample rate, which the wrist sensors do not."
            )
        object.__setattr__(self, "split_ratios", tuple(float(r) for r in self.split_ratios))
        # Fails fast on an unknown label space.
        get_label_space(self.task)

    @property
    def label_space(self) -> LabelSpace:
        return get_label_space(self.task)

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def stride_samples(self) -> int:
        return max(1, int(round(self.stride_seconds * self.sample_rate)))


# ============================================================
# Feature extraction
# ============================================================

def window_features(window: np.ndarray) -> np.ndarray:
    """Reduce a ``[samples, channels]`` window to the fixed statistic vector.

    Statistics are computed per channel in :data:`FEATURE_STATISTICS` order and
    flattened channel-major, matching :data:`FEATURE_NAMES`.
    """
    if window.ndim != 2:
        raise ValueError(f"window must be [samples, channels], got {window.shape}")
    values = window.astype(np.float32, copy=False)
    quartiles = np.percentile(values, [25, 75], axis=0)
    difference = np.abs(np.diff(values, axis=0)) if values.shape[0] > 1 else np.zeros_like(values)
    statistics = np.stack(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.min(axis=0),
            values.max(axis=0),
            quartiles[0],
            quartiles[1],
            difference.mean(axis=0),
        ],
        axis=1,
    )  # [channels, statistics]
    return np.ascontiguousarray(statistics.reshape(-1), dtype=np.float32)


def _stack_chest_signals(signal: dict) -> np.ndarray:
    """Return chest signals as ``[samples, len(CHEST_CHANNELS)]`` float32."""
    chest = signal.get("chest")
    if chest is None:
        raise ValueError("WESAD recording has no 'chest' signal group")
    columns: list[np.ndarray] = []
    for name in ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp"):
        if name not in chest:
            raise ValueError(f"WESAD chest recording is missing the {name!r} channel")
        array = np.asarray(chest[name], dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        columns.append(array)
    stacked = np.concatenate(columns, axis=1)
    if stacked.shape[1] != len(CHEST_CHANNELS):
        raise ValueError(
            f"Expected {len(CHEST_CHANNELS)} chest channels, got {stacked.shape[1]}"
        )
    return stacked


def iter_subject_windows(
    recording_path: Path, config: PhysiologyWindowConfig
) -> Iterator[dict]:
    """Yield one record per label-homogeneous window of a subject recording.

    Windows spanning a condition boundary are dropped rather than assigned to
    whichever label happens to dominate: a mixed window has no single truth.
    """
    with open(recording_path, "rb") as handle:
        # WESAD's pickles were written under Python 2; latin-1 is the documented
        # encoding for reading them back.
        payload = pickle.load(handle, encoding="latin1")

    labels = np.asarray(payload["label"]).reshape(-1)
    signals = _stack_chest_signals(payload["signal"])
    del payload

    usable = min(labels.shape[0], signals.shape[0])
    labels, signals = labels[:usable], signals[:usable]

    wanted = {
        code: name
        for code, name in WESAD_LABEL_STATES.items()
        if name in config.label_space.classes
    }
    window, stride = config.window_samples, config.stride_samples

    for index, start in enumerate(range(0, max(usable - window + 1, 0), stride)):
        end = start + window
        segment_labels = labels[start:end]
        first = int(segment_labels[0])
        if first not in wanted:
            continue
        if not np.all(segment_labels == first):
            continue
        state = wanted[first]
        yield {
            "window_index": index,
            "start_sample": int(start),
            "end_sample": int(end),
            "start_second": round(start / config.sample_rate, 3),
            "end_second": round(end / config.sample_rate, 3),
            "state": state,
            "state_id": config.label_space.id_of(state),
            "features": window_features(signals[start:end]),
        }


# ============================================================
# Builder
# ============================================================

class PhysiologyWindowBuilder:
    """Turn WESAD subject pointers into a subject-disjoint window experiment."""

    def __init__(self, config: PhysiologyWindowConfig | None = None):
        self.config = config or PhysiologyWindowConfig()
        self.label_space = self.config.label_space
        self.modality_rule = get_modality_rule(self.config.modality)
        self.source = Path(self.config.source)
        self.datasets_root = Path(self.config.datasets_root)

    # ----------------------------------------------------------- subjects

    def subjects(self) -> list[dict]:
        """Read the eligible subject pointer records from standardized metadata."""
        if not self.source.exists():
            raise FileNotFoundError(f"Standardized metadata not found: {self.source}")
        parquet = pq.ParquetFile(self.source)
        columns = sorted({
            "sample_id", "dataset", "speaker", "training_split", "target_type",
            *self.modality_rule.required_columns,
        } & set(parquet.schema_arrow.names))

        records: list[dict] = []
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            values = {name: batch.column(name).to_pylist() for name in columns}
            for index in range(batch.num_rows):
                row = {name: values[name][index] for name in columns}
                if row.get("dataset") != self.config.dataset:
                    continue
                if not self.modality_rule.is_eligible(row):
                    continue
                subject = row.get("speaker")
                if is_missing(subject):
                    raise ValueError(
                        f"{self.config.dataset} record {row.get('sample_id')} has no subject "
                        f"identity; subject-aware splitting cannot proceed without it."
                    )
                records.append({
                    "sample_id": str(row.get("sample_id")),
                    "subject": str(subject),
                    "physiology_path": str(row[self.modality_rule.path_column]),
                    "training_split": row.get("training_split"),
                    "target_type": row.get("target_type"),
                })

        if not records:
            raise ValueError(
                f"No eligible {self.config.dataset} physiology records in {self.source}"
            )
        records.sort(key=lambda record: record["subject"])
        return records

    def split_subjects(self, subjects: list[str]) -> dict[str, str]:
        """Assign whole subjects to splits deterministically.

        With fifteen subjects an exact 80/10/10 is impossible, so validation
        and test are guaranteed at least one subject each and the deviation is
        reported rather than hidden.
        """
        if not self.config.subject_aware:
            raise ValueError(
                "Subject-aware splitting is mandatory for WESAD; windows of one "
                "subject must never span partitions."
            )
        ordered = sorted(set(subjects))
        if len(ordered) < len(SPLITS):
            raise ValueError(
                f"Need at least {len(SPLITS)} subjects for a train/validation/test "
                f"split; got {len(ordered)}"
            )
        positions = deterministic_shuffle(
            list(range(len(ordered))), self.config.seed, self.config.dataset, "subjects"
        )
        shuffled = [ordered[position] for position in positions]

        counts = dict(zip(SPLITS, largest_remainder(len(ordered), self.config.split_ratios)))
        # Guarantee a non-empty holdout before honouring the train remainder.
        for split in ("validation", "test"):
            if counts[split] == 0:
                counts[split] = 1
                counts["train"] -= 1
        if counts["train"] < 1:
            raise ValueError("Not enough subjects to populate all three partitions")

        assignment: dict[str, str] = {}
        cursor = 0
        for split in SPLITS:
            for subject in shuffled[cursor:cursor + counts[split]]:
                assignment[subject] = split
            cursor += counts[split]
        return assignment

    # -------------------------------------------------------------- build

    def run(self, output_dir: Path | str) -> dict:
        """Extract windows, write the three partitions, and record provenance."""
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        config = self.config

        pointers = self.subjects()
        assignment = self.split_subjects([record["subject"] for record in pointers])

        rows: list[dict] = []
        per_subject: dict[str, int] = {}
        skipped: list[dict] = []

        for pointer in pointers:
            recording = self.datasets_root / pointer["physiology_path"]
            if not recording.exists():
                raise FileNotFoundError(
                    f"WESAD recording not found: {recording} "
                    f"(subject {pointer['subject']}, from {pointer['sample_id']})"
                )
            split = assignment[pointer["subject"]]
            count = 0
            for window in iter_subject_windows(recording, config):
                rows.append({
                    "sample_id": f"WESAD|{pointer['subject']}|{window['window_index']:06d}",
                    "source_sample_id": pointer["sample_id"],
                    "dataset": config.dataset,
                    "subject": pointer["subject"],
                    "state": window["state"],
                    "state_id": window["state_id"],
                    "window_index": window["window_index"],
                    "start_second": window["start_second"],
                    "end_second": window["end_second"],
                    "features": window["features"].tolist(),
                    "physiology_path": pointer["physiology_path"],
                    "has_physiology": True,
                    "physiology_source": "file",
                    "training_split": pointer["training_split"],
                    SPLIT_COLUMN: split,
                })
                count += 1
            per_subject[pointer["subject"]] = count
            if count == 0:
                skipped.append({
                    "subject": pointer["subject"],
                    "reason": "no label-homogeneous window matched the declared label space",
                })

        if not rows:
            raise ValueError(
                "No windows extracted; check window_seconds/stride_seconds against "
                "the recording lengths and the configured label space."
            )

        frame = pd.DataFrame(rows)
        for split in SPLITS:
            partition = frame[frame[SPLIT_COLUMN] == split].reset_index(drop=True)
            if partition.empty:
                raise ValueError(
                    f"Partition {split!r} is empty; the subject split produced no windows "
                    f"for it."
                )
            partition.to_parquet(output_dir / f"{split}.parquet", index=False)

        summary = self._summarise(frame, assignment, per_subject, skipped, output_dir)
        (output_dir / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    # ------------------------------------------------------------ summary

    def _summarise(
        self,
        frame: pd.DataFrame,
        assignment: dict[str, str],
        per_subject: dict[str, int],
        skipped: list[dict],
        output_dir: Path,
    ) -> dict:
        config = self.config
        total = len(frame)
        counts = {split: int((frame[SPLIT_COLUMN] == split).sum()) for split in SPLITS}
        class_counts = Counter(frame["state_id"].tolist())
        subject_counts = Counter(frame["subject"].tolist())
        split_subjects = {
            split: sorted(subject for subject, value in assignment.items() if value == split)
            for split in SPLITS
        }
        digests = {
            split: _digest(frame.loc[frame[SPLIT_COLUMN] == split, "sample_id"].tolist())
            for split in SPLITS
        }
        overlap = sum(
            1 for subject in subject_counts
            if sum(1 for split in SPLITS if subject in split_subjects[split]) > 1
        )

        return {
            "experiment": {
                "modality": config.modality,
                "task": config.task,
                "description": (
                    "WESAD physiological-state windows, split by subject. WESAD carries no "
                    "canonical emotion target, so the declared label space is "
                    f"{self.label_space.name}."
                ),
                "output_dir": str(output_dir),
            },
            "sampling_method": (
                f"fixed {config.window_seconds:g}s windows at a {config.stride_seconds:g}s "
                f"stride over the {config.signal_group} channels, keeping only "
                f"label-homogeneous windows, split subject-wise "
                f"{'/'.join(format(r, 'g') for r in config.split_ratios)} with a seeded "
                f"subject shuffle"
            ),
            "config": asdict(config),
            "seed": config.seed,
            "source": str(self.source),
            "contributing_datasets": [config.dataset],
            "stratification_fields": ["subject"],
            "label_space": self.label_space.to_dict(),
            "features": {
                "channels": list(CHEST_CHANNELS),
                "statistics": list(FEATURE_STATISTICS),
                "names": list(FEATURE_NAMES),
                "dimension": len(FEATURE_NAMES),
                "sample_rate": config.sample_rate,
                "normalisation": "fitted on the training partition at training time",
            },
            "eligibility": {
                "modality_rule": {
                    "availability_column": self.modality_rule.availability_column,
                    "source_column": self.modality_rule.source_column,
                    "accepted_sources": list(self.modality_rule.accepted_sources),
                    "path_column": self.modality_rule.path_column,
                },
                "subjects_scanned": len(per_subject),
                "subjects_without_windows": skipped,
            },
            "pool": {
                "eligible_records": total,
                "dataset_counts": {config.dataset: total},
                "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
                "subject_counts": {key: int(value) for key, value in sorted(subject_counts.items())},
                "official_split_counts": {},
            },
            "selection": {
                "requested_fraction": 1.0,
                "target_records": total,
                "selected_records": total,
                "actual_fraction": 1.0,
                "dataset_counts": {config.dataset: total},
                "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
                "plan_digest": _digest(frame["sample_id"].tolist()),
            },
            "splits": {
                "ratios": dict(zip(SPLITS, config.split_ratios)),
                "counts": counts,
                "actual_ratios": {
                    split: (counts[split] / total if total else 0.0) for split in SPLITS
                },
                "subjects": split_subjects,
                "subject_counts": {split: len(split_subjects[split]) for split in SPLITS},
                "class_counts": {
                    split: {
                        str(key): int(value)
                        for key, value in sorted(
                            Counter(
                                frame.loc[frame[SPLIT_COLUMN] == split, "state_id"].tolist()
                            ).items()
                        )
                    }
                    for split in SPLITS
                },
                "sample_id_digests": digests,
            },
            "representativeness": {
                "class_share_before": _shares(class_counts),
                "class_share_after": _shares(class_counts),
                "dataset_share_before": {config.dataset: 1.0},
                "dataset_share_after": {config.dataset: 1.0},
                "max_dataset_share_drift": 0.0,
                "max_class_share_drift": 0.0,
                "note": (
                    "The full eligible window pool is used, so the selected distribution "
                    "is the pool distribution by construction."
                ),
            },
            "integrity": {
                "duplicate_sample_ids": int(total - frame["sample_id"].nunique()),
                "cross_split_sample_ids": 0,
                "subjects_in_multiple_splits": overlap,
                "invalid_label_records": int(
                    (~frame["state_id"].isin(list(self.label_space.valid_ids))).sum()
                ),
                "records_outside_contributing_datasets": int(
                    (frame["dataset"] != config.dataset).sum()
                ),
                "records_with_missing_modality_source": int(
                    frame["physiology_path"].isna().sum()
                ),
                "official_holdout_in_train": 0,
                "source_paths_rewritten": 0,
                "source_path_column": self.modality_rule.path_column,
                "leakage_checks_passed": (
                    overlap == 0
                    and total == frame["sample_id"].nunique()
                    and bool(frame["state_id"].isin(list(self.label_space.valid_ids)).all())
                ),
            },
            "deviations": {
                "subject_split_rounding": {
                    "subjects": len(assignment),
                    "requested_ratios": dict(zip(SPLITS, config.split_ratios)),
                    "subject_counts": {split: len(split_subjects[split]) for split in SPLITS},
                    "note": (
                        "Whole subjects are indivisible, so realised split ratios deviate "
                        "from the requested ones; validation and test are guaranteed at "
                        "least one subject each."
                    ),
                },
                "windows_per_subject": {key: int(value) for key, value in sorted(per_subject.items())},
                "policy": (
                    "Windows spanning a condition boundary are discarded rather than "
                    "assigned a dominant label; conditions outside the declared label "
                    "space are never mapped onto it."
                ),
            },
        }


def _digest(sample_ids: list[str]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for sample_id in sorted(sample_ids):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"|")
    return digest.hexdigest()


def _shares(counter: Counter) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    return {str(key): value / total for key, value in sorted(counter.items())}
