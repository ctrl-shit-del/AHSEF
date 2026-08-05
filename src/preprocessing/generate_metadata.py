from src.preprocessing.core.pipeline import MetadataPipeline
from src.preprocessing.registry import AdapterRegistry


def main():

    for dataset in AdapterRegistry.enabled_datasets():

        adapter = AdapterRegistry.create(dataset)

        MetadataPipeline(adapter).run()


if __name__ == "__main__":

    main()