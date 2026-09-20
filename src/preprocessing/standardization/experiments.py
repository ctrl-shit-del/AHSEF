"""Memory-bounded, reproducible task-specific experiment split builders.

The standardized table is immutable input.  Builders scan it in bounded Arrow
batches and write experiment partitions directly to Parquet, avoiding a full
table dataframe and its filtered copies in memory.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.preprocessing.standardization.targets import CANONICAL_EMOTIONS
from src.utils.io import ensure_dir


STANDARDIZED_PATH = Path("metadata/standardized/standardized.parquet")
EXPERIMENTS_DIR = Path("metadata/experiments")
SPLITS = ("train", "validation", "test")
# A 16K-row batch keeps peak memory low even when ``extras`` contains large
# serialized metadata.  It is configurable for storage with smaller row groups.
DEFAULT_BATCH_SIZE = 16_384


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration recorded verbatim beside every experiment split."""

    name: str
    seed: int = 42
    split_policy: str = "official_only"
    held_out_session: str = "Session5"
    validation_session: str | None = None


class _SummaryAccumulator:
    """Aggregate the report without retaining experiment records."""

    def __init__(self) -> None:
        self.counts = Counter()
        self.datasets = {split: Counter() for split in SPLITS}
        self.labels = {split: Counter() for split in SPLITS}
        self.modalities = Counter()
        self.seen_ids: set[str] = set()

    def add(self, table: pa.Table, split: str) -> None:
        ids = table.column("sample_id").to_pylist()
        duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
        duplicates.extend(sample_id for sample_id in ids if sample_id in self.seen_ids)
        if duplicates:
            raise ValueError(f"Experiment contains duplicate sample IDs (for example {duplicates[0]!r})")
        self.seen_ids.update(ids)
        self.counts[split] += table.num_rows
        self.datasets[split].update(str(value) for value in table.column("dataset").to_pylist())
        if "canonical_emotion" in table.column_names:
            self.labels[split].update(
                str(value) for value in table.column("canonical_emotion").to_pylist() if value is not None
            )
        for column in ("has_audio", "has_video", "has_image", "has_text", "has_physiology"):
            if column in table.column_names:
                self.modalities[column] += sum(bool(value) for value in table.column(column).fill_null(False).to_pylist())


def _source_schema(source: Path, with_experiment_split: bool) -> pa.Schema:
    schema = pq.ParquetFile(source).schema_arrow
    return schema.append(pa.field("experiment_split", pa.string())) if with_experiment_split else schema


def _validate_source(source: Path) -> pq.ParquetFile:
    if not source.exists():
        raise FileNotFoundError(f"Standardized metadata not found: {source}")
    parquet = pq.ParquetFile(source)
    required = {"sample_id", "dataset", "target_type", "training_split"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Standardized metadata missing columns: {sorted(missing)}")
    return parquet


def _mask(table: pa.Table, predicate: Callable[[pa.Table], pa.Array]) -> pa.Table:
    mask = predicate(table)
    return table.filter(pc.fill_null(mask, False))


def _validate_selected(table: pa.Table, task: str) -> None:
    if task == "emotion":
        valid = pc.is_in(table.column("canonical_emotion_id"), value_set=pa.array(list(CANONICAL_EMOTIONS.values())))
        if not bool(pc.all(pc.fill_null(valid, False)).as_py()):
            raise ValueError("Emotion experiment contains an invalid canonical emotion ID")
    elif task == "sentiment":
        for column in ("feature_file", "feature_split", "feature_id"):
            if bool(pc.any(pc.is_null(table.column(column))).as_py()):
                raise ValueError(f"CMU-MOSEI experiment has a missing {column}")
    elif task == "physiology":
        valid = pc.and_(
            pc.fill_null(table.column("has_physiology"), False),
            pc.and_(pc.equal(table.column("physiology_source"), "file"), pc.invert(pc.is_null(table.column("physiology_path")))),
        )
        if not bool(pc.all(pc.fill_null(valid, False)).as_py()):
            raise ValueError("WESAD experiment has an invalid physiology source")


def _write_streamed_partitions(
    source: Path,
    output_dir: Path,
    task: str,
    predicate: Callable[[pa.Table], pa.Array],
    split_for: Callable[[pa.Table], dict[str, pa.Array]],
    config: ExperimentConfig,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[dict[str, int], _SummaryAccumulator, int]:
    """Filter and partition source records without ever loading all rows."""
    parquet = _validate_source(source)
    ensure_dir(output_dir)
    schema = _source_schema(source, with_experiment_split=True)
    writers = {split: pq.ParquetWriter(output_dir / f"{split}.parquet", schema, compression="snappy") for split in SPLITS}
    summary = _SummaryAccumulator()
    eligible_count = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            eligible = _mask(pa.Table.from_batches([batch]), predicate)
            eligible_count += eligible.num_rows
            for split, split_mask in split_for(eligible).items():
                selected = eligible.filter(pc.fill_null(split_mask, False))
                if not selected.num_rows:
                    continue
                _validate_selected(selected, task)
                selected = selected.append_column("experiment_split", pa.array([split] * selected.num_rows, type=pa.string()))
                summary.add(selected, split)
                writers[split].write_table(selected)
    finally:
        for writer in writers.values():
            writer.close()
    return {split: int(summary.counts[split]) for split in SPLITS}, summary, eligible_count


def _save_summary(summary: dict, output_dir: Path) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _experiment_summary(
    accumulator: _SummaryAccumulator, config: ExperimentConfig, source: Path, counts: dict[str, int], excluded: int = 0,
) -> dict:
    return {
        "source": str(source), "config": asdict(config),
        "target_definition": "canonical_emotion_id" if config.name.startswith(("emotion", "iemocap")) else "sentiment_score",
        "canonical_label_mapping": CANONICAL_EMOTIONS, "counts": counts, "excluded_by_policy": int(excluded),
        "dataset_by_split": {split: dict(accumulator.datasets[split]) for split in SPLITS},
        "label_by_split": {split: dict(accumulator.labels[split]) for split in SPLITS},
        "modality_availability": dict(accumulator.modalities),
        "batch_size": DEFAULT_BATCH_SIZE,
    }


def _categorical_emotion(table: pa.Table) -> pa.Array:
    return pc.and_(pc.equal(table.column("target_type"), "categorical_emotion"), pc.fill_null(table.column("emotion_target_valid"), False))


def build_emotion_7class(standardized_path: Path = STANDARDIZED_PATH, output_dir: Path = EXPERIMENTS_DIR / "emotion_7class", config: ExperimentConfig | None = None) -> dict:
    """Build official-splits-only seven-class metadata.

    Datasets with non-canonical source partitions are excluded deliberately;
    IEMOCAP is handled by ``build_iemocap_session_experiment``.
    """
    config = config or ExperimentConfig(name="emotion_7class")
    if config.split_policy != "official_only":
        raise ValueError("emotion_7class currently supports only split_policy='official_only'")
    source, destination = Path(standardized_path), Path(output_dir)
    counts, accumulator, eligible = _write_streamed_partitions(
        source, destination, "emotion", _categorical_emotion,
        lambda table: {split: pc.equal(table.column("training_split"), split) for split in SPLITS}, config,
    )
    result = _experiment_summary(accumulator, config, source, counts, excluded=eligible - sum(counts.values()))
    _save_summary(result, destination)
    return result


def build_iemocap_session_experiment(held_out_session: str = "Session5", validation_session: str | None = None, standardized_path: Path = STANDARDIZED_PATH, output_dir: Path | None = None, seed: int = 42) -> dict:
    """Build a leakage-safe IEMOCAP experiment with whole-session partitions."""
    sessions = [f"Session{i}" for i in range(1, 6)]
    if held_out_session not in sessions:
        raise ValueError(f"Unknown IEMOCAP test session: {held_out_session}")
    if validation_session is None:
        validation_session = next(session for session in reversed(sessions) if session != held_out_session)
    if validation_session not in sessions or validation_session == held_out_session:
        raise ValueError("validation_session must be a session different from held_out_session")
    config = ExperimentConfig(f"iemocap_{held_out_session.lower()}", seed, "iemocap_cross_session", held_out_session, validation_session)
    source, destination = Path(standardized_path), Path(output_dir) if output_dir else EXPERIMENTS_DIR / config.name
    counts, accumulator, _ = _write_streamed_partitions(
        source, destination, "emotion",
        lambda table: pc.and_(pc.equal(table.column("dataset"), "IEMOCAP"), _categorical_emotion(table)),
        lambda table: {
            "train": pc.and_(pc.not_equal(table.column("evaluation_group"), held_out_session), pc.not_equal(table.column("evaluation_group"), validation_session)),
            "validation": pc.equal(table.column("evaluation_group"), validation_session),
            "test": pc.equal(table.column("evaluation_group"), held_out_session),
        }, config,
    )
    if not all(counts.values()):
        raise ValueError("IEMOCAP session policy produced an empty train, validation, or test partition")
    result = _experiment_summary(accumulator, config, source, counts)
    _save_summary(result, destination)
    return result


def build_sentiment(standardized_path: Path = STANDARDIZED_PATH, output_dir: Path = EXPERIMENTS_DIR / "sentiment", seed: int = 42) -> dict:
    """Build CMU-MOSEI official sentiment-regression partitions."""
    config = ExperimentConfig(name="sentiment", seed=seed, split_policy="official_only")
    source, destination = Path(standardized_path), Path(output_dir)
    predicate = lambda table: pc.and_(pc.equal(table.column("dataset"), "CMU-MOSEI"), pc.and_(pc.equal(table.column("target_type"), "sentiment"), pc.fill_null(table.column("sentiment_target_valid"), False)))
    counts, accumulator, eligible = _write_streamed_partitions(source, destination, "sentiment", predicate, lambda table: {split: pc.equal(table.column("training_split"), split) for split in SPLITS}, config)
    result = _experiment_summary(accumulator, config, source, counts, excluded=eligible - sum(counts.values()))
    _save_summary(result, destination)
    return result


def build_physiology_metadata(standardized_path: Path = STANDARDIZED_PATH, output_dir: Path = EXPERIMENTS_DIR / "physiology", seed: int = 42, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
    """Save WESAD subject metadata only; continuous recordings are not batched."""
    source, destination = Path(standardized_path), Path(output_dir)
    config = ExperimentConfig(name="physiology", seed=seed, split_policy="subject_protocol_unsegmented")
    parquet = _validate_source(source)
    ensure_dir(destination)
    writer = pq.ParquetWriter(destination / "metadata.parquet", parquet.schema_arrow, compression="snappy")
    records, seen_ids = 0, set()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            selected = _mask(table, lambda item: pc.and_(pc.equal(item.column("dataset"), "WESAD"), pc.equal(item.column("target_type"), "physiological_state")))
            if selected.num_rows:
                _validate_selected(selected, "physiology")
                ids = selected.column("sample_id").to_pylist()
                if any(sample_id in seen_ids for sample_id in ids):
                    raise ValueError("Physiology experiment contains duplicate sample IDs")
                seen_ids.update(ids)
                writer.write_table(selected)
                records += selected.num_rows
    finally:
        writer.close()
    result = {"source": str(source), "config": asdict(config), "records": int(records), "target_definition": "WESAD protocol states; segmentation into windows is deferred", "ignored_protocol_states": ["transition", "ignore"], "batch_size": batch_size}
    _save_summary(result, destination)
    return result
