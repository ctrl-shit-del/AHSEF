"""Deterministic, reproducible sample selection for the stage-2 pilot.

The pilot scores 1,000 validation and 1,000 test samples, and the *same* ids
must be used by the LLM and by the frozen text baseline so the comparison is
paired.  That makes selection an experimental parameter in its own right: it is
seeded, recorded in a manifest, and re-derivable without re-running anything.

Two properties are enforced rather than intended:

**Selection never looks at a label to decide inclusion.**  Stratification uses
the class column only to allocate quotas -- proportionally to the population, so
the sampled split keeps the population's class balance rather than being
rebalanced into one the evaluation would flatter.  Within a stratum the choice
is a seeded shuffle.  Test is never rebalanced.

**Validation and test never mix.**  They are drawn from separate manifests and
:func:`assert_disjoint` re-checks it, because the two pools do share a corpus
and an accidental overlap would silently invalidate the locked evaluation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.common.labels import CANONICAL_EMOTION_CLASSES


SELECTION_METHOD = "seeded_stratified_largest_remainder_v1"
LABEL_COLUMN = "canonical_emotion_id"


@dataclass
class SampleManifest:
    """The exact samples one split of the pilot ran on."""

    experiment_id: str
    split: str
    seed: int
    method: str
    requested: int
    sample_ids: list[str] = field(default_factory=list)
    datasets: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    population: int = 0
    population_class_counts: dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    source_manifest: str = ""

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    @property
    def fingerprint(self) -> str:
        """Hash of the ordered id list -- one value that proves the set matched."""
        digest = hashlib.sha256()
        for identifier in self.sample_ids:
            digest.update(identifier.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "split": self.split,
            "seed": self.seed,
            "selection_method": self.method,
            "requested": self.requested,
            "selected": self.size,
            "fingerprint_sha256": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "datasets": dict(self.datasets),
            "class_counts": dict(self.class_counts),
            "population": self.population,
            "population_class_counts": dict(self.population_class_counts),
            "source_manifest": self.source_manifest,
            "created_at": self.created_at,
            "labels_used_for_inclusion": False,
            "rebalanced": False,
            "notes": (
                "Stratified proportionally to the population class distribution, so the "
                "sample keeps the split's natural balance. Labels allocate quotas only; "
                "membership within a stratum is a seeded shuffle. Nothing is rebalanced."
            ),
            "environment": {"platform": platform.platform()},
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SampleManifest":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls(
            experiment_id=record["experiment_id"], split=record["split"],
            seed=int(record["seed"]), method=record["selection_method"],
            requested=int(record["requested"]), sample_ids=list(record["sample_ids"]),
            datasets=dict(record.get("datasets") or {}),
            class_counts=dict(record.get("class_counts") or {}),
            population=int(record.get("population") or 0),
            population_class_counts=dict(record.get("population_class_counts") or {}),
            created_at=record.get("created_at", ""),
            source_manifest=record.get("source_manifest", ""),
        )
        if manifest.fingerprint != record["fingerprint_sha256"]:
            raise ValueError(
                f"{path} has been edited: the id list no longer matches its recorded "
                f"fingerprint. The run it describes cannot be reproduced from it."
            )
        return manifest


def select_samples(
    frame: pd.DataFrame,
    size: int,
    seed: int,
    split: str,
    experiment_id: str,
    source_manifest: str = "",
) -> SampleManifest:
    """Choose ``size`` samples deterministically, keeping the class balance.

    Largest-remainder allocation over the class column, mirroring the sampler
    the frozen baselines used, then a seeded per-stratum shuffle.  Running this
    twice with the same seed and frame yields the identical ordered id list.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    for column in ("sample_id", LABEL_COLUMN):
        if column not in frame.columns:
            raise ValueError(f"Frame is missing required column {column!r}")

    work = frame.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    # Sorting first makes selection independent of the order rows arrived in.
    work = work.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    population = len(work)
    population_counts = _class_counts(work)

    if size >= population:
        chosen = work
    else:
        groups = {int(key): part for key, part in work.groupby(LABEL_COLUMN)}
        exact = {key: len(part) * size / population for key, part in groups.items()}
        counts = {key: min(int(value), len(groups[key])) for key, value in exact.items()}
        # Every class present in the population keeps at least one sample, so a
        # rare class does not silently vanish from the pilot.
        for key in counts:
            if counts[key] == 0 and len(groups[key]) > 0:
                counts[key] = 1
        _balance(counts, exact, groups, size)
        parts = [
            groups[key].sample(n=counts[key], random_state=seed + key)
            for key in sorted(counts) if counts[key] > 0
        ]
        chosen = pd.concat(parts)

    chosen = chosen.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    return SampleManifest(
        experiment_id=experiment_id, split=split, seed=seed, method=SELECTION_METHOD,
        requested=size, sample_ids=chosen["sample_id"].tolist(),
        datasets=(
            chosen["dataset"].value_counts().sort_index().to_dict()
            if "dataset" in chosen.columns else {}
        ),
        class_counts=_class_counts(chosen),
        population=population, population_class_counts=population_counts,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_manifest=source_manifest,
    )


def _balance(counts: dict, exact: dict, groups: dict, size: int) -> None:
    """Top up or trim the allocation until it sums to ``size`` exactly."""
    guard = 0
    while sum(counts.values()) != size:
        guard += 1
        if guard > 10 * max(size, 1):  # pragma: no cover - structural safety net
            raise RuntimeError("Stratified allocation failed to converge")
        if sum(counts.values()) < size:
            candidates = [key for key in counts if counts[key] < len(groups[key])]
            if not candidates:
                return
            key = max(candidates, key=lambda k: exact[k] - counts[k])
            counts[key] += 1
        else:
            candidates = [key for key in counts if counts[key] > 1]
            if not candidates:
                candidates = [key for key in counts if counts[key] > 0]
            key = max(candidates, key=lambda k: counts[k] - exact[k])
            counts[key] -= 1


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame[LABEL_COLUMN].astype(int).value_counts().sort_index().to_dict()
    return {
        CANONICAL_EMOTION_CLASSES[int(key)]: int(value)
        for key, value in counts.items()
        if 0 <= int(key) < len(CANONICAL_EMOTION_CLASSES)
    }


def assert_disjoint(left: SampleManifest, right: SampleManifest) -> None:
    """Validation and test must share no sample; a leak invalidates the lock."""
    overlap = set(left.sample_ids) & set(right.sample_ids)
    if overlap:
        raise ValueError(
            f"{left.split} and {right.split} manifests share {len(overlap)} sample ids "
            f"(e.g. {sorted(overlap)[:5]}). The locked test evaluation would be "
            f"contaminated by the split threshold selection ran on."
        )


def restrict(frame: pd.DataFrame, manifest: SampleManifest) -> pd.DataFrame:
    """Narrow a frame to the manifest's ids, in the manifest's order."""
    work = frame.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    indexed = work.set_index("sample_id")
    missing = [item for item in manifest.sample_ids if item not in indexed.index]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest ids are absent from the frame "
            f"(e.g. {missing[:5]}); the pilot cannot be reproduced against it."
        )
    return indexed.loc[manifest.sample_ids].reset_index()


def manifest_summary(manifests: Sequence[SampleManifest]) -> dict:
    return {
        "splits": {
            manifest.split: {
                "selected": manifest.size,
                "fingerprint_sha256": manifest.fingerprint,
                "seed": manifest.seed,
                "class_counts": dict(manifest.class_counts),
                "datasets": dict(manifest.datasets),
            }
            for manifest in manifests
        },
        "selection_method": SELECTION_METHOD,
        "labels_used_for_inclusion": False,
        "splits_disjoint": True,
    }
