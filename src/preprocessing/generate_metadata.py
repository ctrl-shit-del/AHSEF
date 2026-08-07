from src.common.paths import MASTER_METADATA_DIR

from src.preprocessing.core.master_builder import MasterMetadataBuilder
from src.preprocessing.core.pipeline import MetadataPipeline
from src.preprocessing.core.serializer import MetadataSerializer
from src.preprocessing.registry import AdapterRegistry


def main():

    master = MasterMetadataBuilder()

    for dataset in AdapterRegistry.enabled_datasets():

        adapter = AdapterRegistry.create(dataset)

        samples, stats = MetadataPipeline(
            adapter
        ).run()

        master.add(samples)

    master.save_csv(
        MASTER_METADATA_DIR / "master.csv"
    )

    master.save_parquet(
        MASTER_METADATA_DIR / "master.parquet"
    )

    MetadataSerializer.save_json(
        master.summary(),
        MASTER_METADATA_DIR / "summary.json",
    )


if __name__ == "__main__":

    main()