from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.preprocessing.standardization.targets import (
    CANONICAL_EMOTIONS,
)


REQUIRED_COLUMNS = [
    "sample_id",
    "dataset",
    "emotion",
    "canonical_emotion",
    "canonical_emotion_id",
    "target_type",
    "emotion_target_valid",
    "vad_target_valid",
    "sentiment_target_valid",
    "training_split",
    "evaluation_group",
    "has_audio",
    "has_video",
    "has_image",
    "has_text",
    "has_physiology",
    "audio_source",
    "video_source",
    "image_source",
    "text_source",
    "physiology_source",
    "feature_file",
    "feature_split",
    "feature_id",
]


def validate_parquet(
    path: str | Path,
    batch_size: int = 16_384,
) -> dict:
    """Validate standardized metadata without materializing it in pandas.

    This is the validation entry point for the large 792k-row metadata file.
    It retains only aggregate counters and the set of sample IDs needed for
    duplicate detection, rather than a second full copy of the table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Standardized metadata not found: {path}")
    parquet = pq.ParquetFile(path)
    missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    seen_ids: set[str] = set()
    rows = 0
    dataset_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    valid_ids = pa.array(list(CANONICAL_EMOTIONS.values()))

    for batch in parquet.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        rows += table.num_rows
        sample_ids = table.column("sample_id").to_pylist()
        if len(sample_ids) != len(set(sample_ids)) or any(sample_id in seen_ids for sample_id in sample_ids):
            raise ValueError("Duplicate sample IDs found.")
        seen_ids.update(sample_ids)

        canonical = table.column("canonical_emotion")
        valid_canonical = pc.or_(pc.is_null(canonical), pc.is_in(canonical, value_set=pa.array(list(CANONICAL_EMOTIONS))))
        if not bool(pc.all(valid_canonical).as_py()):
            raise ValueError("Invalid canonical emotion detected.")
        canonical_id = table.column("canonical_emotion_id")
        valid_id = pc.or_(pc.is_null(canonical), pc.is_in(canonical_id, value_set=valid_ids))
        if not bool(pc.all(pc.fill_null(valid_id, False)).as_py()):
            raise ValueError("Canonical emotion has an invalid or missing ID.")

        split = table.column("training_split")
        valid_split = pc.or_(pc.is_null(split), pc.is_in(split, value_set=pa.array(["train", "validation", "test"])))
        if not bool(pc.all(valid_split).as_py()):
            raise ValueError("Invalid training split detected.")

        for modality in ("audio", "video", "image", "text", "physiology"):
            available = pc.fill_null(table.column(f"has_{modality}"), False)
            source = table.column(f"{modality}_source")
            if bool(pc.any(pc.and_(available, pc.is_null(source))).as_py()):
                raise ValueError(f"{modality} is available but has no modality source.")
            if bool(pc.any(pc.and_(pc.invert(available), pc.equal(source, "feature_container"))).as_py()):
                raise ValueError("Feature container declared for an unavailable modality.")
            modality_counts[f"has_{modality}"] += sum(bool(value) for value in available.to_pylist())

        mosei = table.filter(pc.equal(table.column("dataset"), "CMU-MOSEI"))
        if mosei.num_rows:
            valid_mosei = pc.and_(
                pc.fill_null(mosei.column("has_audio"), False),
                pc.and_(
                    pc.fill_null(mosei.column("has_video"), False),
                    pc.and_(
                        pc.fill_null(mosei.column("has_text"), False),
                        pc.and_(
                            pc.equal(mosei.column("audio_source"), "feature_container"),
                            pc.and_(
                                pc.equal(mosei.column("video_source"), "feature_container"),
                                pc.and_(
                                    pc.equal(mosei.column("text_source"), "metadata"),
                                    pc.and_(
                                        pc.invert(pc.is_null(mosei.column("feature_file"))),
                                        pc.and_(
                                            pc.is_in(mosei.column("feature_split"), value_set=pa.array(["train", "valid", "test"])),
                                            pc.invert(pc.is_null(mosei.column("feature_id"))),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            if not bool(pc.all(pc.fill_null(valid_mosei, False)).as_py()):
                raise ValueError("CMU-MOSEI feature-backed modality contract failed.")

        wesad = table.filter(pc.equal(table.column("dataset"), "WESAD"))
        if wesad.num_rows:
            valid_wesad = pc.and_(
                pc.fill_null(wesad.column("has_physiology"), False),
                pc.and_(pc.equal(wesad.column("physiology_source"), "file"), pc.invert(pc.is_null(wesad.column("physiology_path")))),
            )
            if not bool(pc.all(pc.fill_null(valid_wesad, False)).as_py()):
                raise ValueError("WESAD physiology modality contract failed.")

        dataset_counts.update(str(value) for value in table.column("dataset").to_pylist())
        split_counts.update(str(value) if value is not None else "unspecified" for value in split.to_pylist())

    return {
        "path": str(path), "rows": rows, "datasets": dict(dataset_counts),
        "training_splits": dict(split_counts), "modality_availability": dict(modality_counts),
        "batch_size": batch_size,
    }


def validate(df: pd.DataFrame):

    print("=" * 70)
    print("STANDARDIZED METADATA VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Basic
    # ---------------------------------------------------------

    print("\n1. BASIC")

    print("Rows:", len(df))

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns: PASS")

    # ---------------------------------------------------------
    # Duplicate IDs
    # ---------------------------------------------------------

    print("\n2. DUPLICATES")

    duplicate_ids = df[
        df["sample_id"].duplicated(
            keep=False
        )
    ]

    print(
        "Duplicate sample IDs:",
        len(duplicate_ids),
    )

    if len(duplicate_ids):
        raise ValueError(
            "Duplicate sample IDs found."
        )

    # ---------------------------------------------------------
    # Canonical labels
    # ---------------------------------------------------------

    print("\n3. CANONICAL EMOTIONS")

    valid_labels = set(
        CANONICAL_EMOTIONS.keys()
    )

    invalid = df[
        df["canonical_emotion"].notna()
        & ~df["canonical_emotion"].isin(
            valid_labels
        )
    ]

    print(
        "Invalid canonical labels:",
        len(invalid),
    )

    if len(invalid):
        raise ValueError(
            "Invalid canonical emotion detected."
        )

    print("\nDistribution:")

    print(
        df["canonical_emotion"]
        .value_counts(dropna=False)
    )

    # ---------------------------------------------------------
    # Canonical IDs
    # ---------------------------------------------------------

    print("\n4. CANONICAL IDS")

    invalid_ids = df[
        df["canonical_emotion"].notna()
        & df["canonical_emotion_id"].isna()
    ]

    print(
        "Missing canonical IDs:",
        len(invalid_ids),
    )

    if len(invalid_ids):
        raise ValueError(
            "Canonical emotion has no ID."
        )

    # ---------------------------------------------------------
    # Dataset target summary
    # ---------------------------------------------------------

    print("\n5. TARGETS BY DATASET")

    summary = (
        df.groupby("dataset")
        .agg(
            samples=("sample_id", "size"),
            emotion_targets=(
                "emotion_target_valid",
                "sum",
            ),
            vad_targets=(
                "vad_target_valid",
                "sum",
            ),
            sentiment_targets=(
                "sentiment_target_valid",
                "sum",
            ),
        )
        .sort_index()
    )

    print(summary)

    # ---------------------------------------------------------
    # Split summary
    # ---------------------------------------------------------

    print("\n6. TRAINING SPLITS")

    print(
        df["training_split"]
        .value_counts(dropna=False)
    )

    print("\n7. DATASET × TRAINING SPLIT")

    print(
        pd.crosstab(
            df["dataset"],
            df["training_split"],
            dropna=False,
        )
    )

    # ---------------------------------------------------------
    # Modality summary
    # ---------------------------------------------------------

    print("\n8. MODALITY AVAILABILITY")

    modality_columns = [
        "has_audio",
        "has_video",
        "has_image",
        "has_text",
        "has_physiology",
    ]

    print(
        df[modality_columns].sum()
    )

    # ---------------------------------------------------------
    # Modality/source contract
    # ---------------------------------------------------------

    print("\n9. MODALITY SOURCE CONTRACT")

    sources = {
        "audio": "audio_source",
        "video": "video_source",
        "image": "image_source",
        "text": "text_source",
        "physiology": "physiology_source",
    }

    for modality, source_column in sources.items():
        has_column = f"has_{modality}"
        invalid = df[
            df[has_column].fillna(False)
            & df[source_column].isna()
        ]
        print(f"{modality:12} available_without_source={len(invalid)}")
        if len(invalid):
            raise ValueError(
                f"{modality} is available but has no modality source."
            )

    impossible_feature_source = df[
        (~df["has_audio"].fillna(False))
        & df["audio_source"].eq("feature_container")
        | (~df["has_video"].fillna(False))
        & df["video_source"].eq("feature_container")
    ]

    print(
        "Feature source for unavailable modality:",
        len(impossible_feature_source),
    )

    if len(impossible_feature_source):
        raise ValueError(
            "Feature container declared for an unavailable modality."
        )

    # CMU-MOSEI's audio/video are feature-backed even though file paths
    # are intentionally absent.
    mosei = df[df["dataset"] == "CMU-MOSEI"]
    invalid_mosei = mosei[
        ~(
            mosei["has_audio"].fillna(False)
            & mosei["has_video"].fillna(False)
            & mosei["has_text"].fillna(False)
            & mosei["audio_source"].eq("feature_container")
            & mosei["video_source"].eq("feature_container")
            & mosei["text_source"].eq("metadata")
            & mosei["feature_file"].notna()
            & mosei["feature_split"].notna()
            & mosei["feature_id"].notna()
        )
    ]
    print("Invalid CMU-MOSEI feature references:", len(invalid_mosei))
    if len(invalid_mosei):
        raise ValueError("CMU-MOSEI feature-backed modality contract failed.")

    wesad = df[df["dataset"] == "WESAD"]
    invalid_wesad = wesad[
        ~(
            wesad["has_physiology"].fillna(False)
            & wesad["physiology_source"].eq("file")
            & wesad["physiology_path"].notna()
        )
    ]
    print("Invalid WESAD physiology references:", len(invalid_wesad))
    if len(invalid_wesad):
        raise ValueError("WESAD physiology modality contract failed.")

    print("\nVALIDATION PASSED")
