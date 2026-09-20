from typing import Optional


OFFICIAL_SPLIT_MAPPING = {

    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "test": "test",
}


def get_training_split(
    dataset: str,
    dataset_split: Optional[str],
) -> Optional[str]:

    if dataset_split is None:
        return None

    split = str(dataset_split).strip()

    # Official train/validation/test splits.
    if split in OFFICIAL_SPLIT_MAPPING:

        return OFFICIAL_SPLIT_MAPPING[split]

    # Dataset-specific partitions are intentionally preserved
    # rather than incorrectly converted into train/test.
    return None


def get_evaluation_group(
    dataset: str,
    dataset_split: Optional[str],
) -> Optional[str]:

    if dataset_split is None:
        return None

    split = str(dataset_split).strip()

    if dataset == "IEMOCAP":
        return split

    if dataset == "RAVDESS":
        return split

    return None