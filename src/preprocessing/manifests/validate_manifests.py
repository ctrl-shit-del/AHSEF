from pathlib import Path

import pandas as pd


STANDARDIZED_PATH = Path(
    "metadata/standardized/standardized.parquet"
)

MANIFEST_DIR = Path(
    "metadata/manifests"
)


CANONICAL_EMOTIONS = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
}

def load_manifest(task, split):
    path = (
        MANIFEST_DIR
        / task
        / f"{split}.parquet"
    )

    # An unspecified partition is optional.
    # If no records belong to it, the manifest generator
    # intentionally does not create the file.
    if not path.exists():

        if split == "unspecified":
            return pd.DataFrame(
                columns=["sample_id"]
            )

        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    return pd.read_parquet(path)


def check_duplicates(
    df: pd.DataFrame,
    name: str,
) -> None:

    duplicates = (
        df["sample_id"]
        .duplicated()
        .sum()
    )

    print(
        f"{name:35} "
        f"duplicates={duplicates}"
    )

    if duplicates != 0:
        raise AssertionError(
            f"{name} contains duplicate sample IDs."
        )


def check_against_standardized(
    manifest: pd.DataFrame,
    standardized_ids: set,
    name: str,
) -> None:

    manifest_ids = set(
        manifest["sample_id"]
    )

    missing = (
        manifest_ids
        - standardized_ids
    )

    print(
        f"{name:35} "
        f"missing_from_standardized={len(missing)}"
    )

    if missing:
        print(
            "First missing IDs:",
            list(missing)[:10],
        )

        raise AssertionError(
            f"{name} contains IDs not present "
            f"in standardized metadata."
        )


def check_partition_overlap(
    task: str,
) -> None:

    split_names = [
        "train",
        "validation",
        "test",
        "unspecified",
    ]

    split_ids = {}

    for split in split_names:

        df = load_manifest(
            task,
            split,
        )

        split_ids[split] = set(
            df["sample_id"]
        )

    print(
        f"\n{task.upper()} PARTITION OVERLAP"
    )

    for i, split_a in enumerate(
        split_names
    ):

        for split_b in split_names[
            i + 1:
        ]:

            overlap = (
                split_ids[split_a]
                & split_ids[split_b]
            )

            print(
                f"{split_a:12} × "
                f"{split_b:12} : "
                f"{len(overlap)}"
            )

            if overlap:
                raise AssertionError(
                    f"{task}: overlap between "
                    f"{split_a} and {split_b}."
                )


def validate_emotion(
    standardized: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("EMOTION MANIFEST")
    print("=" * 70)

    expected = standardized[
        standardized[
            "emotion_target_valid"
        ].fillna(False)
        & standardized[
            "canonical_emotion"
        ].isin(CANONICAL_EMOTIONS)
    ]

    print(
        "Expected records:",
        len(expected),
    )

    standardized_ids = set(
        standardized["sample_id"]
    )

    manifests = []

    for split in [
        "train",
        "validation",
        "test",
        "unspecified",
    ]:

        df = load_manifest(
            "emotion",
            split,
        )

        manifests.append(df)

        print(
            f"{split:12}: {len(df):,}"
        )

        check_duplicates(
            df,
            f"emotion/{split}",
        )

        check_against_standardized(
            df,
            standardized_ids,
            f"emotion/{split}",
        )

        invalid = df[
            ~df[
                "canonical_emotion"
            ].isin(CANONICAL_EMOTIONS)
        ]

        if len(invalid) != 0:

            raise AssertionError(
                f"emotion/{split} contains "
                f"invalid canonical emotions."
            )

    combined = pd.concat(
        manifests,
        ignore_index=True,
    )

    check_duplicates(
        combined,
        "emotion/all partitions",
    )

    if len(combined) != len(expected):

        raise AssertionError(
            "Emotion manifest total does not "
            "match standardized target count."
        )

    if set(combined["sample_id"]) != set(
        expected["sample_id"]
    ):

        raise AssertionError(
            "Emotion manifest IDs do not exactly "
            "match expected emotion IDs."
        )

    print(
        "Emotion manifest: PASSED"
    )


def validate_vad(
    standardized: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("VAD MANIFEST")
    print("=" * 70)

    expected = standardized[
        standardized[
            "vad_target_valid"
        ].fillna(False)
    ]

    print(
        "Expected records:",
        len(expected),
    )

    standardized_ids = set(
        standardized["sample_id"]
    )

    manifests = []

    for split in [
        "train",
        "validation",
        "test",
        "unspecified",
    ]:

        df = load_manifest(
            "vad",
            split,
        )

        manifests.append(df)

        print(
            f"{split:12}: {len(df):,}"
        )

        check_duplicates(
            df,
            f"vad/{split}",
        )

        check_against_standardized(
            df,
            standardized_ids,
            f"vad/{split}",
        )

        invalid = df[
            ~df[
                "vad_target_valid"
            ].fillna(False)
        ]

        if len(invalid) != 0:

            raise AssertionError(
                f"vad/{split} contains "
                f"invalid VAD targets."
            )

    combined = pd.concat(
        manifests,
        ignore_index=True,
    )

    check_duplicates(
        combined,
        "vad/all partitions",
    )

    if len(combined) != len(expected):

        raise AssertionError(
            "VAD manifest total does not "
            "match standardized target count."
        )

    if set(combined["sample_id"]) != set(
        expected["sample_id"]
    ):

        raise AssertionError(
            "VAD manifest IDs do not exactly "
            "match expected VAD IDs."
        )

    print(
        "VAD manifest: PASSED"
    )


def validate_sentiment(
    standardized: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("SENTIMENT MANIFEST")
    print("=" * 70)

    expected = standardized[
        (standardized["dataset"] == "CMU-MOSEI")
        & standardized[
            "sentiment_target_valid"
        ].fillna(False)
    ]

    print(
        "Expected records:",
        len(expected),
    )

    standardized_ids = set(
        standardized["sample_id"]
    )

    manifests = []

    for split in [
        "train",
        "validation",
        "test",
    ]:

        df = load_manifest(
            "sentiment",
            split,
        )

        manifests.append(df)

        print(
            f"{split:12}: {len(df):,}"
        )

        check_duplicates(
            df,
            f"sentiment/{split}",
        )

        check_against_standardized(
            df,
            standardized_ids,
            f"sentiment/{split}",
        )

        wrong_dataset = df[
            df["dataset"] != "CMU-MOSEI"
        ]

        if len(wrong_dataset) != 0:

            raise AssertionError(
                f"sentiment/{split} contains "
                f"non-CMU-MOSEI records."
            )

        invalid = df[
            ~df[
                "sentiment_target_valid"
            ].fillna(False)
        ]

        if len(invalid) != 0:

            raise AssertionError(
                f"sentiment/{split} contains "
                f"invalid sentiment targets."
            )

    combined = pd.concat(
        manifests,
        ignore_index=True,
    )

    check_duplicates(
        combined,
        "sentiment/all partitions",
    )

    if len(combined) != len(expected):

        raise AssertionError(
            "Sentiment manifest total does not "
            "match standardized target count."
        )

    if set(combined["sample_id"]) != set(
        expected["sample_id"]
    ):

        raise AssertionError(
            "Sentiment manifest IDs do not exactly "
            "match expected sentiment IDs."
        )

    # CMU-MOSEI split verification.
    expected_split_counts = (
        expected["training_split"]
        .value_counts()
        .to_dict()
    )

    actual_split_counts = (
        combined["training_split"]
        .value_counts()
        .to_dict()
    )

    if expected_split_counts != actual_split_counts:

        raise AssertionError(
            "CMU-MOSEI sentiment split counts "
            "do not match standardized metadata."
        )

    print(
        "Sentiment manifest: PASSED"
    )


def validate_iemocap_sessions(
    manifest: pd.DataFrame,
) -> None:

    iemocap = manifest[
        manifest["dataset"] == "IEMOCAP"
    ]

    if iemocap.empty:
        return

    print("\nIEMOCAP EVALUATION GROUPS")

    print(
        iemocap[
            "evaluation_group"
        ]
        .value_counts()
        .sort_index()
    )

    missing_groups = iemocap[
        iemocap[
            "evaluation_group"
        ].isna()
    ]

    if len(missing_groups) != 0:

        raise AssertionError(
            "IEMOCAP records are missing "
            "evaluation_group."
        )


def validate_mosei(
    sentiment_manifest: pd.DataFrame,
) -> None:

    print("\nCMU-MOSEI FEATURE LINKAGE")

    mosei = sentiment_manifest[
        sentiment_manifest["dataset"]
        == "CMU-MOSEI"
    ]

    print(
        "Records:",
        len(mosei),
    )

    feature_file_count = (
        mosei["feature_file"]
        .notna()
        .sum()
        if "feature_file" in mosei.columns
        else 0
    )

    feature_id_count = (
        mosei["feature_id"]
        .notna()
        .sum()
        if "feature_id" in mosei.columns
        else 0
    )

    print(
        "Feature files:",
        feature_file_count,
    )

    print(
        "Feature IDs:",
        feature_id_count,
    )

    if feature_file_count != len(mosei):

        raise AssertionError(
            "Some CMU-MOSEI records are missing "
            "feature_file."
        )

    if feature_id_count != len(mosei):

        raise AssertionError(
            "Some CMU-MOSEI records are missing "
            "feature_id."
        )


def main() -> None:

    print("=" * 70)
    print("TASK MANIFEST VALIDATION")
    print("=" * 70)

    print("\nLoading standardized metadata...")

    standardized = pd.read_parquet(
        STANDARDIZED_PATH
    )

    print(
        "Standardized rows:",
        f"{len(standardized):,}",
    )

    standardized_ids = set(
        standardized["sample_id"]
    )

    print(
        "Standardized unique IDs:",
        f"{len(standardized_ids):,}",
    )

    if len(standardized_ids) != len(
        standardized
    ):

        raise AssertionError(
            "Standardized metadata contains "
            "duplicate sample IDs."
        )

    validate_emotion(
        standardized
    )

    validate_vad(
        standardized
    )

    validate_sentiment(
        standardized
    )

    # ---------------------------------------------------------
    # Partition validation
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PARTITION VALIDATION")
    print("=" * 70)

    for task in [
        "emotion",
        "vad",
        "sentiment",
    ]:

        check_partition_overlap(
            task
        )

    # ---------------------------------------------------------
    # IEMOCAP
    # ---------------------------------------------------------

    emotion_all = pd.concat(
        [
            load_manifest(
                "emotion",
                split,
            )
            for split in [
                "train",
                "validation",
                "test",
                "unspecified",
            ]
        ],
        ignore_index=True,
    )

    vad_all = pd.concat(
        [
            load_manifest(
                "vad",
                split,
            )
            for split in [
                "train",
                "validation",
                "test",
                "unspecified",
            ]
        ],
        ignore_index=True,
    )

    sentiment_all = pd.concat(
        [
            load_manifest(
                "sentiment",
                split,
            )
            for split in [
                "train",
                "validation",
                "test",
            ]
        ],
        ignore_index=True,
    )

    validate_iemocap_sessions(
        emotion_all
    )

    validate_iemocap_sessions(
        vad_all
    )

    validate_mosei(
        sentiment_all
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "TASK MANIFEST VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()