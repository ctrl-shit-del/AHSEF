"""Command-line entry point for conservative default experiment metadata."""

from src.preprocessing.standardization.experiments import (
    build_emotion_7class,
    build_physiology_metadata,
    build_sentiment,
)
from src.preprocessing.standardization.validator import validate_parquet


def main() -> None:
    validation = validate_parquet("metadata/standardized/standardized.parquet")
    print(f"Validated {validation['rows']:,} standardized records in bounded batches.")
    for build in (build_emotion_7class, build_sentiment, build_physiology_metadata):
        summary = build()
        print(f"{summary['config']['name']}: {summary.get('counts', summary.get('records'))}")


if __name__ == "__main__":
    main()
