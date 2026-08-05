from src.analysis.statistics import DatasetStatistics
from src.common.paths import (
    RAW_METADATA_DIR,
    PROCESSED_METADATA_DIR,
    REPORTS_DIR,
)

from src.preprocessing.core.serializer import MetadataSerializer
from src.preprocessing.core.validator import MetadataValidator


class MetadataPipeline:
    """
    Complete preprocessing pipeline for a single dataset.
    """

    def __init__(self, adapter):

        self.adapter = adapter

    def run(self):

        dataset = self.adapter.DATASET_NAME

        self.adapter.logger.info(f"Starting {dataset}")

        try:

            # ---------------------------------
            # Scan dataset
            # ---------------------------------

            self.adapter.logger.info("Scanning dataset...")

            samples = self.adapter.scan()

            self.adapter.logger.info(
                f"Found {len(samples)} samples."
            )

            # ---------------------------------
            # Validate
            # ---------------------------------

            self.adapter.logger.info("Validating metadata...")

            MetadataValidator.validate(samples)

            self.adapter.logger.info(
                "Validation completed."
            )

            # ---------------------------------
            # Statistics
            # ---------------------------------

            self.adapter.logger.info(
                "Computing statistics..."
            )

            statistics = DatasetStatistics.summarize(
                samples
            )

            # ---------------------------------
            # Save metadata
            # ---------------------------------

            self.adapter.logger.info(
                "Saving CSV..."
            )

            MetadataSerializer.save_csv(
                samples,
                RAW_METADATA_DIR / f"{dataset}.csv",
            )

            self.adapter.logger.info(
                "Saving Parquet..."
            )

            MetadataSerializer.save_parquet(
                samples,
                PROCESSED_METADATA_DIR / f"{dataset}.parquet",
            )

            self.adapter.logger.info(
                "Saving statistics..."
            )

            MetadataSerializer.save_json(
                statistics,
                REPORTS_DIR / f"{dataset}.json",
            )

            self.adapter.logger.info(
                f"{dataset} completed successfully."
            )

            return samples, statistics

        except Exception as e:

            self.adapter.logger.exception(
                f"{dataset} failed: {e}"
            )

            raise