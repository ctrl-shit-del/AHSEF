"""Reading cached features into HSEN batches.

Training never touches an encoder.  A batch is assembled from three shard
caches, padded to the batch's own maximum length per modality, and handed to the
trunk with an explicit frame mask and an explicit availability flag.

Two masks, not one, and the difference is load-bearing:

``mask``       ``[B, L]``  which *frames* are real rather than padding.
``available``  ``[B]``     whether the sample has the modality **at all**.

Collapsing them would make "a sample whose audio is one frame long" and "a
sample with no audio" the same object.  The whole later routing question --
should this modality be acquired? -- is about the second case, so it gets its
own channel from the start.

No fitted normalisation anywhere.  The projections use ``LayerNorm``, which is a
per-sample statistic, so nothing about the training distribution is baked into
how a validation or test sample is scaled.  That is a deliberate choice over a
fitted feature mean/variance, which would be a genuine leak dressed up as
preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.hsen.labels.base import MISSING_CLASS_ID, LabelSet

MODALITIES: tuple[str, ...] = ("audio", "video", "text")

#: Last-resort widths, used only when a batch has no sample carrying a modality
#: *and* no dataset was consulted.  The dataset's own ``feature_dims`` is always
#: preferred, because CMU-MOSEI's provided features are 74-d and 35-d and these
#: defaults would be wrong for it.
MODALITY_FEATURE_DIMS: dict[str, int] = {"audio": 768, "video": 576, "text": 768}


@runtime_checkable
class FeatureSource(Protocol):
    """Anything that can return a ``[T, D]`` sequence for a sample id.

    :class:`~src.hsen.features.store.SequenceFeatureStore` satisfies this, and
    so will the CMU-MOSEI adapter that reads COVAREP and FACET rows out of
    ``aligned_50.pkl`` -- which is why the dataset depends on this shape rather
    than on the store.
    """

    feature_dim: int

    def __contains__(self, sample_id: str) -> bool: ...

    def load(self, sample_ids: Sequence[str]) -> list[np.ndarray]: ...


class MissingFeatureError(RuntimeError):
    """Raised when a manifest promises a modality the cache does not hold."""


@dataclass
class HSENSample:
    """One assembled training example."""

    sample_id: str
    features: dict[str, np.ndarray]
    available: dict[str, bool]
    class_id: int
    valence: float
    arousal: float
    multilabel: np.ndarray | None = None


class HSENDataset(Dataset):
    """Manifest rows plus cached features, with per-modality availability."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        sources: dict[str, FeatureSource],
        labels: LabelSet,
        modalities: Sequence[str] = MODALITIES,
        strict: bool = True,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.sources = {m: source for m, source in sources.items() if m in modalities}
        self.modalities = tuple(m for m in modalities if m in self.sources)
        self.labels = labels
        if not self.modalities:
            raise ValueError(
                f"No feature source supplied for any of {list(modalities)}; "
                f"HSEN cannot train on a manifest alone"
            )
        if len(labels.sample_ids) != len(self.manifest):
            raise ValueError(
                f"{len(labels.sample_ids)} label rows for "
                f"{len(self.manifest)} manifest rows"
            )

        self.sample_ids = self.manifest["sample_id"].astype(str).to_numpy()
        self._availability = self._resolve_availability(strict)
        #: Each modality's width, taken from its cache. The collate function
        #: needs this: a batch in which *no* sample has a given modality has no
        #: feature array to infer the width from, and guessing would hand the
        #: projection a tensor of the wrong shape.
        self.feature_dims = {
            modality: int(source.feature_dim) for modality, source in self.sources.items()
        }

    def _resolve_availability(self, strict: bool) -> dict[str, np.ndarray]:
        """Which samples actually have each modality, cache and manifest agreeing.

        A modality the manifest declares but the cache does not hold is an error
        under ``strict``, not a silently dropped modality: quietly demoting it
        would make an ablation table report "audio+text" for a run that was
        partly text-only, and nothing in the output would say so.
        """
        availability: dict[str, np.ndarray] = {}
        for modality in self.modalities:
            source = self.sources[modality]
            declared = (
                self.manifest[f"has_{modality}"].fillna(False).to_numpy(dtype=bool)
                if f"has_{modality}" in self.manifest.columns
                else np.ones(len(self.manifest), dtype=bool)
            )
            cached = np.fromiter(
                (str(sample_id) in source for sample_id in self.sample_ids),
                dtype=bool, count=len(self.sample_ids),
            )
            missing = declared & ~cached
            if missing.any():
                examples = self.sample_ids[missing][:5].tolist()
                message = (
                    f"{int(missing.sum())} samples declare {modality} but it is not "
                    f"in the feature cache (e.g. {examples}).\n"
                    f"Extract them first:\n"
                    f"    python -m src.hsen.extract --modality {modality} --split <split>"
                )
                if strict:
                    raise MissingFeatureError(message)
                print(f"[warn] {message}")
            availability[modality] = declared & cached
        return availability

    def availability_summary(self) -> dict[str, int]:
        return {m: int(flags.sum()) for m, flags in self._availability.items()}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> HSENSample:
        sample_id = str(self.sample_ids[index])
        features: dict[str, np.ndarray] = {}
        available: dict[str, bool] = {}
        for modality in self.modalities:
            present = bool(self._availability[modality][index])
            available[modality] = present
            if present:
                features[modality] = self.sources[modality].load([sample_id])[0]

        labels = self.labels
        return HSENSample(
            sample_id=sample_id,
            features=features,
            available=available,
            class_id=int(labels.class_id[index]) if labels.class_id is not None else MISSING_CLASS_ID,
            valence=float(labels.valence[index]) if labels.valence is not None else float("nan"),
            arousal=float(labels.arousal[index]) if labels.arousal is not None else float("nan"),
            multilabel=labels.multilabel[index] if labels.multilabel is not None else None,
        )


def collate_hsen(
    batch: Sequence[HSENSample],
    modalities: Sequence[str] = MODALITIES,
    feature_dims: dict[str, int] | None = None,
) -> dict:
    """Pad each modality to the batch maximum and build both masks.

    A modality no sample in the batch has still gets a length-1 all-padding
    tensor rather than being dropped, so the trunk sees the same set of streams
    in every batch and a modality's absence never changes the shape of the
    graph.  Its availability flags are all ``False``, so it contributes nothing.

    ``feature_dims`` is what makes that safe.  Such a batch has no array to read
    the width from, and a placeholder of the wrong width reaches the projection
    as a shape error -- on a rare batch, deep into a run, for a reason that
    looks nothing like "this batch happened to contain no video".
    """
    feature_dims = feature_dims or {}
    features: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    available: dict[str, torch.Tensor] = {}

    for modality in modalities:
        present = [sample.features[modality] for sample in batch if modality in sample.features]
        dimension = (
            present[0].shape[1] if present
            else feature_dims.get(modality, MODALITY_FEATURE_DIMS.get(modality, 1))
        )
        longest = max((array.shape[0] for array in present), default=1)

        padded = np.zeros((len(batch), longest, dimension), dtype=np.float32)
        mask = np.zeros((len(batch), longest), dtype=bool)
        flags = np.zeros(len(batch), dtype=bool)
        for row, sample in enumerate(batch):
            array = sample.features.get(modality)
            flags[row] = bool(sample.available.get(modality, False)) and array is not None
            if array is None:
                continue
            padded[row, :array.shape[0]] = array
            mask[row, :array.shape[0]] = True

        features[modality] = torch.from_numpy(padded)
        masks[modality] = torch.from_numpy(mask)
        available[modality] = torch.from_numpy(flags)

    targets = {
        "class_id": torch.tensor([sample.class_id for sample in batch], dtype=torch.long),
        "valence": torch.tensor([sample.valence for sample in batch], dtype=torch.float32),
        "arousal": torch.tensor([sample.arousal for sample in batch], dtype=torch.float32),
    }
    if batch[0].multilabel is not None:
        targets["multilabel"] = torch.from_numpy(
            np.stack([sample.multilabel for sample in batch]).astype(np.float32)
        )

    return {
        "sample_ids": [sample.sample_id for sample in batch],
        "features": features,
        "masks": masks,
        "available": available,
        "targets": targets,
    }


# ======================================================================
# Fractional subsets
# ======================================================================

def stratified_subset(
    manifest: pd.DataFrame, fraction: float, seed: int, label_column: str = "class_id",
) -> np.ndarray:
    """Row positions of a seeded, class-stratified fraction of a manifest.

    Stratified rather than uniform, because the CPU profile's 25% draw would
    otherwise change the class balance the loss is weighted against -- a 25%
    run and a 100% run would then differ in two ways at once and the fast
    profile would stop predicting the slow one.

    Largest-remainder allocation, matching the policy the project's Layer-2
    sampler already uses, so a class never rounds down to zero while another
    rounds up twice.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return np.arange(len(manifest))

    rng = np.random.default_rng(seed)
    labels = manifest[label_column].to_numpy()
    classes, counts = np.unique(labels, return_counts=True)

    exact = counts * fraction
    allocation = np.floor(exact).astype(int)
    # Never drop a class entirely: a class with no samples has no class weight,
    # no per-class F1, and silently leaves the macro average.
    allocation = np.maximum(allocation, np.minimum(1, counts))
    shortfall = int(round(len(manifest) * fraction)) - int(allocation.sum())
    if shortfall > 0:
        order = np.argsort(-(exact - np.floor(exact)))
        for position in order[:shortfall]:
            allocation[position] = min(allocation[position] + 1, counts[position])

    selected: list[np.ndarray] = []
    for class_value, take in zip(classes, allocation):
        positions = np.flatnonzero(labels == class_value)
        selected.append(rng.permutation(positions)[:take])
    return np.sort(np.concatenate(selected))


def build_dataloader(
    dataset: HSENDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """A DataLoader with a seeded generator, so shuffling is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_hsen(batch, dataset.modalities, dataset.feature_dims),
        generator=generator if shuffle else None,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
