from src.preprocessing.core.base_adapter import BaseAdapter


class WESADAdapter(BaseAdapter):

    DATASET_NAME = "WESAD"

    def scan(self):
        raise NotImplementedError