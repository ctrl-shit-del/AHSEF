from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


STANDARDIZED_PATH = Path(
    "metadata/standardized/standardized.parquet"
)

OUTPUT_DIR = Path(
    "metadata/manifests"
)


EMOTION_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "training_split",
    "evaluation_group",
    "modalities",
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
    "raw_emotion",
    "emotion",
    "canonical_emotion",
    "canonical_emotion_id",
    "canonical_emotion_valid",
    "emotion_target_valid",
    "text",
    "audio_path",
    "video_path",
    "image_path",
    "physiology_path",
    "speaker",
    "gender",
    "duration",
    "segment_start",
    "segment_end",
    "extras",
    "feature_file",
    "feature_split",
    "feature_id",
]


VAD_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "training_split",
    "evaluation_group",
    "modalities",
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
    "valence",
    "arousal",
    "dominance",
    "vad_target_valid",
    "text",
    "audio_path",
    "video_path",
    "image_path",
    "physiology_path",
    "speaker",
    "gender",
    "duration",
    "segment_start",
    "segment_end",
    "extras",
    "feature_file",
    "feature_split",
    "feature_id",
]


SENTIMENT_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "training_split",
    "evaluation_group",
    "modalities",
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
    "sentiment_score",
    "sentiment_target_valid",
    "text",
    "audio_path",
    "video_path",
    "image_path",
    "physiology_path",
    "speaker",
    "gender",
    "duration",
    "segment_start",
    "segment_end",
    "extras",
    "feature_file",
    "feature_split",
    "feature_id",
]


def _save_task_manifests(
    df: pd.DataFrame,
    task: str,
    columns: list[str],
    output_dir: Path,
) -> None:

    task_dir = output_dir / task

    ensure_dir(task_dir)

    for split in ["train", "validation", "test"]:

        split_df = df[
            df["training_split"] == split
        ].copy()

        output_path = (
            task_dir /
            f"{split}.parquet"
        )

        split_df[columns].to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

        print(
            f"  {task:10} {split:10} "
            f"{len(split_df):,} -> {output_path}"
        )

    unsplit_df = df[
        df["training_split"].isna()
    ].copy()

    if len(unsplit_df) > 0:

        output_path = (
            task_dir /
            "unspecified.parquet"
        )

        unsplit_df[columns].to_parquet(
            output_path,
            index=False,
            compression="snappy",
        )

        print(
            f"  {task:10} {'unspecified':10} "
            f"{len(unsplit_df):,} -> {output_path}"
        )
def generate_manifests(
    standardized_path: Path = STANDARDIZED_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict:

    ensure_dir(output_dir)

    print("=" * 70)
    print("TASK MANIFEST GENERATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load standardized metadata
    # ---------------------------------------------------------

    print("\nLoading standardized metadata...")

    df = pd.read_parquet(
        standardized_path
    )

    print(
        f"Input rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Emotion task
    # ---------------------------------------------------------

    canonical_emotions = {
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    }

    emotion_df = df[
        df["emotion_target_valid"].fillna(False)
        & df["canonical_emotion"].isin(
            canonical_emotions
        )
    ].copy()

    # ---------------------------------------------------------
    # VAD task
    # ---------------------------------------------------------

    vad_df = df[
        df["vad_target_valid"].fillna(False)
    ].copy()

    # ---------------------------------------------------------
    # Sentiment task
    # ---------------------------------------------------------

    sentiment_df = df[
        (df["dataset"] == "CMU-MOSEI")
        & df["sentiment_target_valid"].fillna(False)
    ].copy()

    # ---------------------------------------------------------
    # Task counts
    # ---------------------------------------------------------

    print("\nTASK COUNTS")

    print(
        f"Emotion    : {len(emotion_df):,}"
    )

    print(
        f"VAD        : {len(vad_df):,}"
    )

    print(
        f"Sentiment  : {len(sentiment_df):,}"
    )

    # ---------------------------------------------------------
    # Generate manifests
    # ---------------------------------------------------------

    print("\nGenerating manifests...")

    _save_task_manifests(
        emotion_df,
        "emotion",
        EMOTION_COLUMNS,
        output_dir,
    )

    _save_task_manifests(
        vad_df,
        "vad",
        VAD_COLUMNS,
        output_dir,
    )

    _save_task_manifests(
        sentiment_df,
        "sentiment",
        SENTIMENT_COLUMNS,
        output_dir,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "source": str(standardized_path),
        "input_rows": int(len(df)),
        "tasks": {
            "emotion": int(len(emotion_df)),
            "vad": int(len(vad_df)),
            "sentiment": int(len(sentiment_df)),
        },
        "emotion_distribution": (
            emotion_df[
                "canonical_emotion"
            ]
            .value_counts()
            .to_dict()
        ),
        "emotion_dataset_distribution": (
            emotion_df[
                "dataset"
            ]
            .value_counts()
            .to_dict()
        ),
        "vad_dataset_distribution": (
            vad_df[
                "dataset"
            ]
            .value_counts()
            .to_dict()
        ),
        "sentiment_dataset_distribution": (
            sentiment_df[
                "dataset"
            ]
            .value_counts()
            .to_dict()
        ),
    }

    import json

    summary_path = (
        output_dir /
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\nSaved summary:")
    print(
        f"  {summary_path}"
    )

    print(
        "\nManifest generation completed."
    )

    return summary


if __name__ == "__main__":
    generate_manifests()