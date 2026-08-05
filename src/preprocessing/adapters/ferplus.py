from src.preprocessing.core.base_adapter import BaseAdapter


class FERPlusAdapter(BaseAdapter):

    DATASET_NAME = "FER+"

    def scan(self):
        raise NotImplementedError