from src.preprocessing.adapters.affectnet import AffectNetAdapter
from src.preprocessing.adapters.cremad import CREMADAdapter
from src.preprocessing.adapters.ferplus import FERPlusAdapter
from src.preprocessing.adapters.iemocap import IEMOCAPAdapter
from src.preprocessing.adapters.meld import MELDAdapter
from src.preprocessing.adapters.mosei import MOSEIAdapter
from src.preprocessing.adapters.rafdb import RAFDBAdapter
from src.preprocessing.adapters.ravdess import RAVDESSAdapter
from src.preprocessing.adapters.wesad import WESADAdapter

class AdapterRegistry:
    """
    Central registry for all dataset adapters.
    """

    _registry = {

        "RAVDESS": {
            "adapter": RAVDESSAdapter,
            "enabled": True,
            "modalities": ("audio",),
            "priority": 1,
        },

        "CREMA-D": {
            "adapter": CREMADAdapter,
            "enabled": True,
            "modalities": ("audio",),
            "priority": 2,
        },

        "IEMOCAP": {
            "adapter": IEMOCAPAdapter,
            "enabled": True,
            "modalities": ("audio", "video", "text"),
            "priority": 3,
        },

        "MELD": {
            "adapter": MELDAdapter,
            "enabled": True,
            "modalities": ("audio", "video", "text"),
            "priority": 4,
        },

        "CMU-MOSEI": {
            "adapter": MOSEIAdapter,
            "enabled": True,
            "modalities": ("audio", "video", "text"),
            "priority": 5,
        },

        "RAF-DB": {
            "adapter": RAFDBAdapter,
            "enabled": True,
            "modalities": ("image",),
            "priority": 7,
        },

        "AffectNet+": {
            "adapter": AffectNetAdapter,
            "enabled": True,
            "modalities": ("image",),
            "priority": 8,
        },

        "WESAD": {
            "adapter": WESADAdapter,
            "enabled": True,
            "modalities": ("physiology",),
            "priority": 9,
        },
        "FERPlus": {
            "adapter": FERPlusAdapter,
            "enabled": True,
            "modalities": ("image",),
            "priority": 6,
        },
    }

    @classmethod
    def create(cls, dataset: str):
        """
        Create an instance of the requested adapter.
        """
        return cls._registry[dataset]["adapter"]()

    @classmethod
    def info(cls, dataset: str) -> dict:
        """
        Return metadata associated with a dataset.
        """
        return cls._registry[dataset]

    @classmethod
    def exists(cls, dataset: str) -> bool:
        """
        Check whether a dataset exists in the registry.
        """
        return dataset in cls._registry

    @classmethod
    def datasets(cls) -> list[str]:
        """
        Return all registered datasets sorted by priority.
        """
        return [
            name
            for name, _ in sorted(
                cls._registry.items(),
                key=lambda item: item[1]["priority"],
            )
        ]

    @classmethod
    def enabled_datasets(cls) -> list[str]:
        """
        Return enabled datasets sorted by priority.
        """
        return [
            name
            for name, info in sorted(
                cls._registry.items(),
                key=lambda item: item[1]["priority"],
            )
            if info["enabled"]
        ]

    @classmethod
    def registry(cls) -> dict:
        """
        Return a shallow copy of the registry.
        """
        return cls._registry.copy()

    @classmethod
    def count(cls) -> int:
        """
        Return the number of registered datasets.
        """
        return len(cls.datasets())