from src.preprocessing.core.base_adapter import BaseAdapter


class MOSEIAdapter(BaseAdapter):

    DATASET_NAME = "MOSEI"

    def scan(self):
        raise NotImplementedError