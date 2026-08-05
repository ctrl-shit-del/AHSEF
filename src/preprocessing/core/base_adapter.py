from abc import ABC, abstractmethod
from src.utils.logger import get_logger

from src.common.paths import DATASETS_DIR
from src.common.models import EmotionRecord

from pathlib import Path


class BaseAdapter(ABC):

    DATASET_NAME = ""

    def __init__(self):

        self.dataset_dir = DATASETS_DIR / self.DATASET_NAME

        self.samples: list[EmotionRecord] = []

        self.logger = get_logger(self.DATASET_NAME)

    def add(self, sample: EmotionRecord):

        self.samples.append(sample)

    def clear(self):

        self.samples.clear()

    def dataset_exists(self) -> bool:

        return self.dataset_dir.exists()

    def relative_path(self, path: Path) -> str:
        
        return str(path.relative_to(DATASETS_DIR))

    def create_record(self, **kwargs):

        defaults = {

            "dataset": self.DATASET_NAME,

            "split": None,

            "modalities": [],

            "emotion": None,

            "valence": None,

            "arousal": None,

            "dominance": None,

            "audio_path": None,

            "video_path": None,

            "image_path": None,

            "text": None,

            "physiology_path": None,

            "speaker": None,

            "gender": None,

            "duration": None,

            "extras": {},
        }

        defaults.update(kwargs)

        return EmotionRecord(**defaults)

    def make_sample_id(self, *parts: str) -> str:
        """
        Create a globally unique sample identifier.
        """

        return "_".join(
            str(part)
            for part in (self.DATASET_NAME, *parts)
            if part not in (None, "")
        )

    @abstractmethod
    def scan(self):
        """
        Scan dataset and return EmotionRecord objects.
        """
        pass