from dataclasses import dataclass
from typing import Type

from src.preprocessing.core.base_adapter import BaseAdapter


@dataclass(frozen=True)
class DatasetInfo:

    adapter: Type[BaseAdapter]

    enabled: bool

    modalities: tuple[str, ...]

    priority: int