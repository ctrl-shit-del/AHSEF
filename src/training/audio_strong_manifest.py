"""Build the ``audio_strong`` experiment manifests from the frozen audio splits.

``audio_strong`` reuses the audio baseline's *sample space* exactly -- the same
splits, the same official-split-aware policy, the same leakage guarantees -- and
differs from it in only one respect: the training partition is **class-capped**.

Why cap.  Extracting wav2vec2 features costs about 3 clips/s on this CPU, so the
full 44,348-clip training partition is roughly four hours before any head is
trained.  Capping the four common classes at 4,000 samples cuts that to about
1.7 h while keeping every sample of the three rare classes.  This is a compute
decision, not a scientific one, and it is recorded as such: the cap changes the
training distribution, so class weights are recomputed from the actual capped
subset rather than inherited from the baseline.

What is *not* capped: validation and test.  Both are copied through unchanged,
so every number reported against ``audio_25pct`` is computed on the same
evaluation samples and the comparison is honest.

The selection is a seeded, stratified, deterministic draw and the resulting id
list is fingerprinted, so the same command reproduces the same manifest.

Naming.  The experiment is ``audio_strong_full``: the project's layout contract
is ``<modality>_<label>``, and ``audio_strong`` is treated as its own modality
identity so nothing it writes can land in ``experiments/audio/``. ``full``
records that both evaluation splits are complete.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.common.experiment_layout import ExperimentLayout
from src.common.labels import CANONICAL_EMOTION_CLASSES
from src.data.features.wav2vec2_features import manifest_fingerprint

#: The experiment this module writes. Never ``audio_25pct``.
STRONG_EXPERIMENT = "audio_strong_full"
SOURCE_EXPERIMENT = "audio_25pct"

LABEL_COLUMN = "canonical_emotion_id"
DEFAULT_CAP = 4_000
SELECTION_METHOD = "seeded_stratified_class_cap_v1"

#: Where this module records what it built. See the note in build_manifests.
MANIFEST_RECORD = "audio_strong_manifest.json"


class ManifestError(RuntimeError):
    """Raised when the strong-audio manifests cannot be built safely."""


def cap_training_partition(
    frame: pd.DataFrame, cap: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Keep at most ``cap`` samples of each class, deterministically.

    Classes smaller than the cap are kept whole -- the rare emotions are exactly
    the ones a capped run can least afford to thin.
    """
    if cap < 1:
        raise ManifestError("cap must be at least 1")
    work = frame.copy()
    work["sample_id"] = work["sample_id"].astype(str)
    # Sorting first makes the draw independent of the order rows arrived in.
    work = work.sort_values("sample_id", kind="mergesort").reset_index(drop=True)

    kept, report = [], {}
    for class_id, group in work.groupby(work[LABEL_COLUMN].astype(int)):
        name = (
            CANONICAL_EMOTION_CLASSES[class_id]
            if 0 <= class_id < len(CANONICAL_EMOTION_CLASSES) else str(class_id)
        )
        if len(group) <= cap:
            chosen = group
            action = "kept whole"
        else:
            chosen = group.sample(n=cap, random_state=seed + int(class_id))
            action = f"capped from {len(group)}"
        kept.append(chosen)
        report[name] = {
            "available": int(len(group)), "kept": int(len(chosen)), "action": action,
        }

    capped = (
        pd.concat(kept).sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    )
    return capped, report


def build_manifests(
    cap: int = DEFAULT_CAP,
    seed: int = 42,
    root: Path | str = "experiments",
    source_experiment: str = SOURCE_EXPERIMENT,
    experiment: str = STRONG_EXPERIMENT,
) -> dict:
    """Write ``audio_strong`` manifests and return the sampling summary."""
    source = ExperimentLayout.from_name(source_experiment, Path(root))
    target = ExperimentLayout.from_name(experiment, Path(root))
    if target.base.resolve() == source.base.resolve():
        raise ManifestError(
            f"Refusing to write {experiment!r} into the frozen baseline's directory "
            f"{source.base}."
        )
    target.metadata_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment": experiment,
        "source_experiment": source_experiment,
        "source_metadata_dir": str(source.metadata_dir),
        "selection_method": SELECTION_METHOD,
        "class_cap": cap,
        "seed": seed,
        "class_order": list(CANONICAL_EMOTION_CLASSES),
        "splits": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "labels_used_for_inclusion": True,
        "labels_note": (
            "The cap is applied per class, so class labels decide how many TRAINING "
            "samples are kept. This is a training-set construction choice, applied to "
            "the training partition only. Validation and test are copied through "
            "untouched, so no evaluation sample is selected or excluded by its label."
        ),
    }

    for split in ("train", "validation", "test"):
        source_path = source.split_path(split)
        if not source_path.exists():
            raise ManifestError(f"Missing source manifest: {source_path}")
        frame = pd.read_parquet(source_path)
        frame["sample_id"] = frame["sample_id"].astype(str)

        if split == "train":
            frame, class_report = cap_training_partition(frame, cap, seed)
            summary["splits"][split] = {"class_cap_report": class_report}
        else:
            summary["splits"][split] = {"class_cap_report": None, "copied_unchanged": True}

        target_path = target.split_path(split)
        frame.to_parquet(target_path, index=False)
        identifiers = frame["sample_id"].tolist()
        summary["splits"][split].update({
            "path": str(target_path),
            "samples": int(len(frame)),
            "source_samples": int(len(pd.read_parquet(source_path, columns=["sample_id"]))),
            "fingerprint_sha256": manifest_fingerprint(identifiers),
            "class_counts": {
                CANONICAL_EMOTION_CLASSES[int(key)]: int(value)
                for key, value in
                frame[LABEL_COLUMN].astype(int).value_counts().sort_index().items()
                if 0 <= int(key) < len(CANONICAL_EMOTION_CLASSES)
            },
            "datasets": frame["dataset"].value_counts().sort_index().to_dict()
            if "dataset" in frame.columns else {},
        })

    _assert_splits_disjoint(target)
    summary["splits_disjoint"] = True
    summary["evaluation_splits_identical_to_source"] = _assert_evaluation_unchanged(
        source, target
    )
    # Deliberately NOT sampling_summary.json. That filename belongs to the
    # stratified sampler and has a fixed shape that experiment_summary.py reads;
    # writing a different record there would either crash the summary builder or,
    # worse, be silently misread as a sampler run this experiment never did.
    (target.metadata_dir / MANIFEST_RECORD).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _assert_splits_disjoint(layout: ExperimentLayout) -> None:
    sets = {
        split: set(
            pd.read_parquet(layout.split_path(split), columns=["sample_id"])["sample_id"]
            .astype(str)
        )
        for split in ("train", "validation", "test")
    }
    for left in ("train", "validation"):
        for right in ("validation", "test"):
            if left >= right:
                continue
            overlap = sets[left] & sets[right]
            if overlap:
                raise ManifestError(
                    f"{left} and {right} share {len(overlap)} sample ids "
                    f"(e.g. {sorted(overlap)[:5]}); the capped manifest is leaking."
                )


def _assert_evaluation_unchanged(
    source: ExperimentLayout, target: ExperimentLayout
) -> bool:
    """Validation and test must be the baseline's, sample for sample.

    If they were not, every ``audio_strong`` vs ``audio_25pct`` number would be
    computed on different data and the comparison would be meaningless.
    """
    for split in ("validation", "test"):
        left = pd.read_parquet(source.split_path(split), columns=["sample_id"])
        right = pd.read_parquet(target.split_path(split), columns=["sample_id"])
        if manifest_fingerprint(left["sample_id"].astype(str).tolist()) != \
           manifest_fingerprint(right["sample_id"].astype(str).tolist()):
            raise ManifestError(
                f"The {split} manifest differs from {source.split_path(split)}. The "
                f"strong and baseline audio experts must be evaluated on identical "
                f"samples or their comparison means nothing."
            )
    return True
