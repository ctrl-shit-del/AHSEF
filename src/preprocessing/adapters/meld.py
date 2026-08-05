from src.preprocessing.core.base_adapter import BaseAdapter


class MELDAdapter(BaseAdapter):

    DATASET_NAME = "MELD"

    def scan(self):
        raise NotImplementedError