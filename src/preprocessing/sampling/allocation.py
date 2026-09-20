"""Deterministic proportional allocation helpers used by the sampler.

Every function here is pure and seed-stable: given identical inputs it returns
identical output in any process, which is what makes an experiment subset
reproducible.  ``hash()`` is deliberately avoided because Python randomises
string hashing per process.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Sequence


def derive_seed(seed: int, *parts: Any) -> int:
    """Derive a stable 32-bit sub-seed for a named group."""
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    digest = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return (int(seed) ^ digest) & 0xFFFF_FFFF


def deterministic_shuffle(items: Sequence[int], seed: int, *parts: Any) -> list[int]:
    """Shuffle ``items`` reproducibly using a seed derived from ``parts``."""
    shuffled = list(items)
    random.Random(derive_seed(seed, *parts)).shuffle(shuffled)
    return shuffled


def largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` into integer parts proportional to ``weights``.

    Ties are broken by ascending index so the result never depends on the
    iteration order of a hash table.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if total <= 0:
        return [0] * len(weights)
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("weights must not sum to zero")
    exact = [total * weight / weight_sum for weight in weights]
    parts = [int(value) for value in exact]
    leftover = total - sum(parts)
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - parts[index]), index))
    for index in order[:leftover]:
        parts[index] += 1
    return parts


@dataclass
class FractionAllocation:
    """Per-stratum selection counts plus every deviation from proportionality."""

    counts: dict[Hashable, int] = field(default_factory=dict)
    target_total: int = 0
    minimum_adjustments: list[dict] = field(default_factory=list)

    @property
    def selected_total(self) -> int:
        return sum(self.counts.values())


def allocate_fraction(
    sizes: Mapping[Hashable, int],
    fraction: float,
    minimum_per_group: int = 1,
) -> FractionAllocation:
    """Allocate ``fraction`` of the pool across strata by largest remainder.

    Small strata whose proportional share rounds below ``minimum_per_group``
    are raised to that floor; every such adjustment is reported rather than
    silently applied, because it makes the realised fraction exceed the
    requested one.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if minimum_per_group < 0:
        raise ValueError("minimum_per_group must be non-negative")

    keys = sorted(sizes, key=repr)
    total = sum(sizes[key] for key in keys)
    target = int(round(total * fraction))
    counts = {key: min(int(sizes[key] * fraction), sizes[key]) for key in keys}

    leftover = target - sum(counts.values())
    if leftover > 0:
        order = sorted(keys, key=lambda key: (-(sizes[key] * fraction - counts[key]), repr(key)))
        for key in order:
            if leftover <= 0:
                break
            if counts[key] < sizes[key]:
                counts[key] += 1
                leftover -= 1

    adjustments: list[dict] = []
    for key in keys:
        size = sizes[key]
        if size <= 0:
            continue
        floor = min(minimum_per_group, size)
        if counts[key] < floor:
            adjustments.append({
                "stratum": list(key) if isinstance(key, tuple) else key,
                "pool": size,
                "proportional": counts[key],
                "enforced_minimum": floor,
                "reason": "proportional share rounded below the configured minimum",
            })
            counts[key] = floor

    return FractionAllocation(counts=counts, target_total=target, minimum_adjustments=adjustments)


class Cursor:
    """Consume a deterministic ordering of record ordinals without copying."""

    __slots__ = ("_items", "_position")

    def __init__(self, items: Iterable[int]):
        self._items = list(items)
        self._position = 0

    def take(self, count: int) -> list[int]:
        if count <= 0:
            return []
        end = min(self._position + count, len(self._items))
        taken = self._items[self._position:end]
        self._position = end
        return taken

    @property
    def remaining(self) -> int:
        return len(self._items) - self._position

    def __len__(self) -> int:
        return len(self._items)
