from src.preprocessing.core.base_adapter import BaseAdapter


class IEMOCAPAdapter(BaseAdapter):

    DATASET_NAME = "IEMOCAP"

    def scan(self):
        raise NotImplementedError