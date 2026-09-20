from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.preprocessing.standardization.targets import (
    get_canonical_emotion,
    get_canonical_emotion_id,
)

from src.preprocessing.standardization.splits import (
    get_training_split,
    get_evaluation_group,
)

from src.utils.io import ensure_dir


MASTER_PATH = Path(
    "metadata/master/master.parquet"
)

OUTPUT_DIR = Path(
    "metadata/standardized"
)


# =============================================================
# Feature-backed datasets
# =============================================================
#
# Some datasets do not store individual audio/video files.
# Instead, their modality representations are stored inside
# a feature container.
#
# CMU-MOSEI:
#   Processed/aligned_50.pkl
#
# Therefore:
#
#   audio_path == None
#   video_path == None
#
# does NOT mean:
#
#   audio unavailable
#   video unavailable
#
# The master metadata already records the modalities correctly.
#
# =============================================================

FEATURE_BACKED_DATASETS = {
    "CMU-MOSEI",
}


# =============================================================
# Standardization
# =============================================================

def _extract_extras_field(df, key: str):
    """Pull one field out of the serialized ``extras`` column, row by row.

    ``extras`` is stored as a JSON string (occasionally as a real dict when the
    frame has not round-tripped through Parquet).  Anything unparsable yields
    ``None`` rather than an exception, because a missing optional path is a
    legitimate state, not a corruption.
    """
    import json

    if "extras" not in df.columns:
        return [None] * len(df)

    values = []
    for raw in df["extras"]:
        if isinstance(raw, dict):
            value = raw.get(key)
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                parsed = None
            value = parsed.get(key) if isinstance(parsed, dict) else None
        else:
            value = None
        values.append(value if value else None)
    return values


def standardize(
    input_path: Path = MASTER_PATH,
    output_dir: Path = OUTPUT_DIR,
):

    ensure_dir(output_dir)

    print("=" * 70)
    print("TARGET / SPLIT STANDARDIZATION")
    print("=" * 70)

    print("\nLoading master metadata...")

    df = pd.read_parquet(input_path)

    print("Input rows:", len(df))

    # ---------------------------------------------------------
    # Canonical emotion
    # ---------------------------------------------------------

    print("\nGenerating canonical emotion targets...")

    df["canonical_emotion"] = [
        get_canonical_emotion(
            dataset,
            emotion,
        )
        for dataset, emotion
        in zip(
            df["dataset"],
            df["emotion"],
        )
    ]

    df["canonical_emotion_id"] = [
        get_canonical_emotion_id(
            emotion
        )
        for emotion in df["canonical_emotion"]
    ]

    df["canonical_emotion_valid"] = (
        df["canonical_emotion"].notna()
    )

    # ---------------------------------------------------------
    # Target type
    # ---------------------------------------------------------

    def target_type(row):

        dataset = row["dataset"]

        # CMU-MOSEI provides continuous sentiment
        # from -3 to +3.
        if dataset == "CMU-MOSEI":
            return "sentiment"

        # WESAD contains physiological states rather
        # than conventional emotion labels.
        if dataset == "WESAD":
            return "physiological_state"

        # Canonical categorical emotion.
        if pd.notna(
            row["canonical_emotion"]
        ):
            return "categorical_emotion"

        # Continuous affect dimensions.
        if (
            pd.notna(row["valence"])
            or pd.notna(row["arousal"])
            or pd.notna(row["dominance"])
        ):
            return "continuous_affect"

        # Dataset-specific target that has not yet
        # been mapped into the unified target space.
        return "dataset_specific"

    df["target_type"] = df.apply(
        target_type,
        axis=1,
    )

    # ---------------------------------------------------------
    # Target availability
    # ---------------------------------------------------------

    df["emotion_target_valid"] = (
        df["canonical_emotion"].notna()
    )

    df["vad_target_valid"] = (
        df[
            [
                "valence",
                "arousal",
                "dominance",
            ]
        ].notna().any(axis=1)
    )

    df["sentiment_target_valid"] = (
        df["sentiment_score"].notna()
    )

    # ---------------------------------------------------------
    # Dataset split preservation
    # ---------------------------------------------------------

    df["training_split"] = [
        get_training_split(
            dataset,
            split,
        )
        for dataset, split
        in zip(
            df["dataset"],
            df["split"],
        )
    ]

    df["evaluation_group"] = [
        get_evaluation_group(
            dataset,
            split,
        )
        for dataset, split
        in zip(
            df["dataset"],
            df["split"],
        )
    ]

    # ---------------------------------------------------------
    # Modality availability
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Do not determine modality availability solely from
    # *_path columns.
    #
    # Example:
    #
    # CMU-MOSEI:
    #
    #   modalities = audio,video,text
    #   audio_path = None
    #   video_path = None
    #
    # Its audio and vision representations are stored in:
    #
    #   Processed/aligned_50.pkl
    #
    # Therefore modality availability is determined from
    # the canonical `modalities` field.
    #
    # ---------------------------------------------------------

    modalities = (
        df["modalities"]
        .fillna("")
        .astype(str)
    )

    df["has_audio"] = modalities.str.contains(
        r"(?:^|,)audio(?:,|$)",
        regex=True,
    )

    df["has_video"] = modalities.str.contains(
        r"(?:^|,)video(?:,|$)",
        regex=True,
    )

    df["has_image"] = modalities.str.contains(
        r"(?:^|,)image(?:,|$)",
        regex=True,
    )

    df["has_text"] = modalities.str.contains(
        r"(?:^|,)text(?:,|$)",
        regex=True,
    )

    df["has_physiology"] = modalities.str.contains(
        r"(?:^|,)physiology(?:,|$)",
        regex=True,
    )

    # ---------------------------------------------------------
    # Modality source
    # ---------------------------------------------------------
    #
    # Distinguishes:
    #
    #   file-backed modality
    #
    # from:
    #
    #   feature-container-backed modality
    #
    # This is particularly important for CMU-MOSEI.
    # ---------------------------------------------------------

    df["audio_source"] = None
    df["video_source"] = None
    df["image_source"] = None
    df["text_source"] = None
    df["physiology_source"] = None

    # ---------------------------------------------------------
    # File-backed modalities
    # ---------------------------------------------------------

    df.loc[
        df["audio_path"].notna(),
        "audio_source",
    ] = "file"

    df.loc[
        df["video_path"].notna(),
        "video_source",
    ] = "file"

    df.loc[
        df["image_path"].notna(),
        "image_source",
    ] = "file"

    df.loc[
        df["text"].notna(),
        "text_source",
    ] = "metadata"

    # MSP-Podcast stores transcript paths in ``extras`` rather than copying
    # every transcript into the metadata text field.  The modality is real,
    # but is file-backed and must be resolved through that explicit source,
    # so the path is promoted into a first-class ``text_path`` column below.
    df["text_path"] = _extract_extras_field(df, "transcript_path")

    df.loc[
        (df["dataset"] == "MSP-Podcast")
        & df["has_text"]
        & df["text_path"].notna(),
        "text_source",
    ] = "file"

    df.loc[
        df["physiology_path"].notna(),
        "physiology_source",
    ] = "file"

    # ---------------------------------------------------------
    # CMU-MOSEI feature-container modalities
    # ---------------------------------------------------------

    mosei_mask = (
        df["dataset"] == "CMU-MOSEI"
    )

    df.loc[
        mosei_mask & df["has_audio"],
        "audio_source",
    ] = "feature_container"

    df.loc[
        mosei_mask & df["has_video"],
        "video_source",
    ] = "feature_container"

    # CMU-MOSEI text is already available in the master
    # metadata through label.csv.
    df.loc[
        mosei_mask & df["has_text"],
        "text_source",
    ] = "metadata"

    # ---------------------------------------------------------
    # Feature file
    # ---------------------------------------------------------
    #
    # Keep this explicit in standardized metadata so downstream
    # feature loading does not need to rediscover the PKL file.
    #
    # We do NOT copy or modify the PKL.
    # ---------------------------------------------------------

    df["feature_file"] = None

    df.loc[
        mosei_mask,
        "feature_file",
    ] = (
        "CMU-MOSEI/Processed/aligned_50.pkl"
    )

    # ---------------------------------------------------------
    # Feature split
    # ---------------------------------------------------------
    #
    # The canonical training split uses:
    #
    #   validation
    #
    # while the original CMU-MOSEI PKL uses:
    #
    #   valid
    #
    # Preserve both.
    # ---------------------------------------------------------

    df["feature_split"] = None

    df.loc[
        mosei_mask,
        "feature_split",
    ] = (
        df.loc[
            mosei_mask,
            "split",
        ]
        .map(
            {
                "train": "train",
                "validation": "valid",
                "test": "test",
            }
        )
    )

    # ---------------------------------------------------------
    # Feature ID
    # ---------------------------------------------------------
    #
    # CMU-MOSEI PKL uses:
    #
    #   video_id$_$clip_id
    #
    # The values are already stored in `extras`.
    #
    # Extract them where possible without modifying the
    # immutable master metadata.
    # ---------------------------------------------------------

    df["feature_id"] = None

    if "extras" in df.columns:

        def extract_feature_id(
            value,
        ):

            if pd.isna(value):
                return None

            try:

                if isinstance(value, dict):
                    video_id = value.get(
                        "video_id"
                    )

                    clip_id = value.get(
                        "clip_id"
                    )

                    if (
                        video_id is not None
                        and clip_id is not None
                    ):
                        return (
                            f"{video_id}$_${clip_id}"
                        )

                # Serialized extras are JSON strings.
                if isinstance(value, str):

                    import json

                    data = json.loads(value)

                    video_id = data.get(
                        "video_id"
                    )

                    clip_id = data.get(
                        "clip_id"
                    )

                    if (
                        video_id is not None
                        and clip_id is not None
                    ):
                        return (
                            f"{video_id}$_${clip_id}"
                        )

            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return None

            return None

        df["feature_id"] = [
            extract_feature_id(value)
            if dataset == "CMU-MOSEI"
            else None
            for dataset, value
            in zip(
                df["dataset"],
                df["extras"],
            )
        ]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    csv_path = (
        output_dir /
        "standardized.csv"
    )

    parquet_path = (
        output_dir /
        "standardized.parquet"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    df.to_parquet(
        parquet_path,
        index=False,
        compression="snappy",
    )

    print("\nSaved:")
    print(" ", csv_path)
    print(" ", parquet_path)

    return df


def repair_msp_podcast_text_sources(
    parquet_path: Path = OUTPUT_DIR / "standardized.parquet",
    csv_path: Path = OUTPUT_DIR / "standardized.csv",
    batch_size: int = 16_384,
) -> int:
    """Stream a targeted source-contract repair into replacement metadata.

    Older standardized metadata marked MSP-Podcast transcripts as available
    while leaving ``text_source`` null.  This migration changes only those
    rows to ``file`` and regenerates both serialized metadata forms without
    loading the 792k-row table into memory.
    """
    parquet_path = Path(parquet_path)
    csv_path = Path(csv_path)
    source = pq.ParquetFile(parquet_path)
    required = {"dataset", "has_text", "text_source"}
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"Cannot repair missing columns: {sorted(missing)}")

    temporary_parquet = parquet_path.with_suffix(".parquet.repairing")
    temporary_csv = csv_path.with_suffix(".csv.repairing")
    source_index = source.schema_arrow.get_field_index("text_source")
    changed = 0

    parquet_writer = pq.ParquetWriter(
        temporary_parquet, source.schema_arrow, compression="snappy"
    )
    csv_sink = pa.OSFile(str(temporary_csv), "wb")
    csv_writer = pacsv.CSVWriter(csv_sink, source.schema_arrow)
    try:
        for batch in source.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            repair_mask = pc.and_(
                pc.equal(table.column("dataset"), "MSP-Podcast"),
                pc.and_(
                    pc.fill_null(table.column("has_text"), False),
                    pc.is_null(table.column("text_source")),
                ),
            )
            changed += int(pc.sum(pc.cast(repair_mask, pa.int64())).as_py() or 0)
            repaired_source = pc.if_else(
                repair_mask, pa.scalar("file", type=pa.string()), table.column("text_source")
            )
            table = table.set_column(source_index, "text_source", repaired_source)
            parquet_writer.write_table(table)
            csv_writer.write(table)
    finally:
        parquet_writer.close()
        csv_writer.close()
        csv_sink.close()

    temporary_parquet.replace(parquet_path)
    temporary_csv.replace(csv_path)
    return changed


if __name__ == "__main__":
    standardize()
