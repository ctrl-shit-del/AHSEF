from pathlib import Path
from collections import Counter
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ExperimentDataset, MOSEIDataset
from src.data.image_dataset import EmotionImageDataset


DEFAULT_FEATURE_PATH = Path(
    "datasets/CMU-MOSEI/Processed/aligned_50.pkl"
)

MANIFEST_DIR = Path(
    "metadata/experiments/sentiment"
)


def create_mosei_dataloader(
    split: str,
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    feature_path: str | Path = DEFAULT_FEATURE_PATH,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a CMU-MOSEI sentiment split.

    Parameters
    ----------
    split:
        One of: train, validation, test.

    batch_size:
        Number of samples per batch.

    shuffle:
        Whether to shuffle samples.
        Defaults to True for training and False otherwise.

    num_workers:
        Number of DataLoader worker processes.

    feature_path:
        Path to aligned_50.pkl.
    """

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    if split not in valid_splits:
        raise ValueError(
            f"Invalid split '{split}'. "
            f"Expected one of {sorted(valid_splits)}."
        )

    manifest_path = (
        MANIFEST_DIR /
        f"{split}.parquet"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{manifest_path}"
        )

    if shuffle is None:
        shuffle = split == "train"

    dataset = MOSEIDataset(
        manifest_path=manifest_path,
        feature_path=feature_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    return loader


def experiment_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Safely collate mixed-modality samples without hiding absence.

    A modality is padded only when every item provides a tensor.  Otherwise it
    remains a per-sample list containing explicit ``None`` entries.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "dataset": [sample["dataset"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "label": torch.stack([sample["label"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }
    for modality in ("audio", "video", "image", "text", "physiology"):
        values = [sample[modality] for sample in samples]
        if all(torch.is_tensor(value) for value in values):
            if len({tuple(value.shape) for value in values}) == 1:
                batch[modality] = torch.stack(values)
            elif all(value.ndim >= 1 for value in values):
                batch[modality] = pad_sequence(values, batch_first=True)
                batch[f"{modality}_lengths"] = torch.tensor([value.shape[0] for value in values])
            else:
                batch[modality] = values
        else:
            batch[modality] = values
    return batch


def create_experiment_dataloader(
    manifest_path: str | Path,
    target_column: str,
    batch_size: int = 8,
    sampling: str = "natural",
    shuffle: bool | None = None,
    seed: int = 42,
    num_workers: int = 0,
    load_modalities: bool = True,
) -> DataLoader:
    """Create a reproducible generic experiment dataloader.

    Supported initial sampling strategies are ``natural`` and
    ``balanced_by_class``. Dataset-aware sampling remains intentionally absent
    until a baseline establishes the appropriate comparison protocol.
    """
    if sampling not in {"natural", "balanced_by_class"}:
        raise ValueError("sampling must be 'natural' or 'balanced_by_class'")
    dataset = ExperimentDataset(
        manifest_path=manifest_path,
        target_column=target_column,
        load_modalities=load_modalities,
    )
    if shuffle is None:
        shuffle = Path(manifest_path).stem == "train"
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sampling == "balanced_by_class":
        labels = dataset.manifest[target_column].tolist()
        counts = Counter(labels)
        weights = torch.as_tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=experiment_collate,
        generator=generator,
    )


def create_emotion_image_dataloader(
    manifest_path,
    batch_size=32,
    image_size=32,
    max_samples=None,
    seed=42,
    shuffle=None,
    columns=None,
    num_workers=0,
):
    """Create a lazy, metadata-backed image dataloader.

    ``columns`` restricts the manifest columns read into memory; images are
    always loaded lazily per item, never preloaded.
    """
    dataset = EmotionImageDataset(
        manifest_path, image_size=image_size, max_samples=max_samples, seed=seed, columns=columns
    )
    if shuffle is None:
        shuffle = Path(manifest_path).stem == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )


# ============================================================
# Modality experiment dataloaders
# ============================================================
#
# One factory per baseline modality.  They share the same contract as
# ``create_emotion_image_dataloader``: the manifest column set is bounded, the
# media is opened lazily per item, ``num_workers`` defaults to 0 so no worker
# process duplicates the manifest, and the shuffle generator is seeded.


def _loader(dataset, manifest_path, batch_size, shuffle, seed, num_workers):
    if shuffle is None:
        shuffle = Path(manifest_path).stem == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def create_emotion_audio_dataloader(
    manifest_path,
    batch_size=32,
    max_seconds=4.0,
    feature_config=None,
    max_samples=None,
    seed=42,
    shuffle=None,
    columns=None,
    num_workers=0,
    datasets_root=None,
):
    """Create a lazy log-mel audio dataloader over one experiment split."""
    from src.data.audio_dataset import EmotionAudioDataset

    dataset = EmotionAudioDataset(
        manifest_path,
        max_seconds=max_seconds,
        feature_config=feature_config,
        max_samples=max_samples,
        seed=seed,
        columns=columns,
        datasets_root=datasets_root,
    )
    return _loader(dataset, manifest_path, batch_size, shuffle, seed, num_workers)


def create_emotion_text_dataloader(
    manifest_path,
    batch_size=64,
    tokenizer_config=None,
    max_samples=None,
    seed=42,
    shuffle=None,
    columns=None,
    num_workers=0,
    datasets_root=None,
    cache_text=True,
):
    """Create a lazy hashed-token text dataloader over one experiment split."""
    from src.data.text_dataset import EmotionTextDataset

    dataset = EmotionTextDataset(
        manifest_path,
        tokenizer_config=tokenizer_config,
        max_samples=max_samples,
        seed=seed,
        columns=columns,
        datasets_root=datasets_root,
        cache_text=cache_text,
    )
    return _loader(dataset, manifest_path, batch_size, shuffle, seed, num_workers)


def create_emotion_video_dataloader(
    manifest_path,
    batch_size=8,
    num_frames=8,
    frame_size=48,
    max_samples=None,
    seed=42,
    shuffle=None,
    columns=None,
    num_workers=0,
    datasets_root=None,
):
    """Create a lazy, sparsely decoded video dataloader over one split."""
    from src.data.video_dataset import EmotionVideoDataset

    dataset = EmotionVideoDataset(
        manifest_path,
        num_frames=num_frames,
        frame_size=frame_size,
        max_samples=max_samples,
        seed=seed,
        columns=columns,
        datasets_root=datasets_root,
    )
    return _loader(dataset, manifest_path, batch_size, shuffle, seed, num_workers)


def create_physiology_window_dataloader(
    manifest_path,
    batch_size=64,
    label_space=None,
    mean=None,
    std=None,
    max_samples=None,
    seed=42,
    shuffle=None,
    columns=None,
    num_workers=0,
):
    """Create a physiological window dataloader with train-fitted scaling."""
    from src.common.labels import WESAD_STATE_3CLASS
    from src.data.physiology_dataset import PhysiologyWindowDataset

    dataset = PhysiologyWindowDataset(
        manifest_path,
        label_space=label_space or WESAD_STATE_3CLASS,
        mean=mean,
        std=std,
        max_samples=max_samples,
        seed=seed,
        columns=columns,
    )
    return _loader(dataset, manifest_path, batch_size, shuffle, seed, num_workers)
