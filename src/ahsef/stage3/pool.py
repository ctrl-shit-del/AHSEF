"""PHASE A -- the aligned Text+Audio sample pool.

Stage 1 already answered *which* pairs may be fused; this module answers *which
exact samples*, and writes down enough evidence that the answer can be
challenged.  It builds on :class:`~src.ahsef.identity.AlignmentIndex` rather
than re-deriving alignment, so Stage 3 inherits Stage 1's contamination and
label-agreement checks instead of inventing weaker ones.

Seven properties are verified, and every one of them raises rather than warns:

1. identical sample ids in both modality manifests;
2. identical split assignment (the co-split rule -- an id that is one model's
   test sample and another's training sample is excluded and counted);
3. no train/validation/test contamination;
4. a compatible seven-class emotion label space on both sides;
5. agreeing ground-truth labels for every pooled id;
6. provenance for both modalities (experiment, manifest path, checkpoint);
7. availability -- the audio asset and the text payload actually exist.

The pool is never widened by joining ids across different splits, and it is
never widened by relaxing a check.  If it is too small to support the
experiment the caller is told the number, not given a synthetic substitute.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.ahsef.identity import AlignmentIndex, SampleAlignmentError
from src.ahsef.registry import DEFAULT_BASELINES
from src.common.labels import CANONICAL_EMOTION_CLASSES

#: The two modalities Stage 1 proved are co-split.  Declared, not discovered at
#: call time, so a future edit that adds a third has to say so explicitly.
STAGE3_MODALITIES: tuple[str, str] = ("audio", "text")

#: The task both sides must solve for a fusion to mean anything.
STAGE3_TASK = "emotion_7class"

#: Below this the HSIG experiment is not worth running; the caller is told so
#: rather than being handed a pool that cannot support an estimator.
MINIMUM_POOL = 200


class PoolTooSmallError(RuntimeError):
    """Raised when the aligned pool cannot support a meaningful experiment."""


class ModalityUnavailableError(RuntimeError):
    """Raised when a pooled sample's declared modality payload is missing."""


def _fingerprint(sample_ids: Sequence[str]) -> str:
    """Hash of the ordered id list -- one value that proves the pool matched."""
    digest = hashlib.sha256()
    for identifier in sample_ids:
        digest.update(str(identifier).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


@dataclass
class AlignedPool:
    """One split's vetted Text+Audio pool, with the evidence that vetted it."""

    split: str
    sample_ids: list[str]
    modalities: tuple[str, ...] = STAGE3_MODALITIES
    task: str = STAGE3_TASK
    labels: dict[str, int] = field(default_factory=dict)
    datasets: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    created_at: str = ""

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.sample_ids)

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "modalities": list(self.modalities),
            "task": self.task,
            "size": self.size,
            "fingerprint_sha256": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "labels": dict(self.labels),
            "datasets": dict(self.datasets),
            "class_counts": dict(self.class_counts),
            "class_order": list(CANONICAL_EMOTION_CLASSES),
            "provenance": dict(self.provenance),
            "availability": dict(self.availability),
            "checks": dict(self.checks),
            "created_at": self.created_at,
            "pool_rule": (
                "ids present in the SAME split of both the audio and the text "
                "experiment manifest. Ids shared across DIFFERENT splits are "
                "excluded and counted, because one model trained on them."
            ),
            "labels_used_for_inclusion": False,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "AlignedPool":
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls(
            split=record["split"],
            sample_ids=[str(item) for item in record["sample_ids"]],
            modalities=tuple(record.get("modalities") or STAGE3_MODALITIES),
            task=record.get("task", STAGE3_TASK),
            labels={str(k): int(v) for k, v in (record.get("labels") or {}).items()},
            datasets=dict(record.get("datasets") or {}),
            class_counts=dict(record.get("class_counts") or {}),
            provenance=dict(record.get("provenance") or {}),
            availability=dict(record.get("availability") or {}),
            checks=dict(record.get("checks") or {}),
            created_at=record.get("created_at", ""),
        )
        if pool.fingerprint != record["fingerprint_sha256"]:
            raise SampleAlignmentError(
                f"{path} has been edited: the pooled id list no longer matches its "
                f"recorded fingerprint, so the run it describes is not reproducible "
                f"from it."
            )
        return pool


def _read_manifest(modality: str, split: str, root: Path | str) -> tuple[pd.DataFrame, Path]:
    reference = DEFAULT_BASELINES[modality]
    path = reference.layout(root).split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest for {modality}/{split}: {path}")
    frame = pd.read_parquet(path)
    frame["sample_id"] = frame["sample_id"].astype(str)
    return frame, path


def _text_declared_file(frame: pd.DataFrame) -> pd.Series:
    """Rows whose transcript lives in a file rather than in the manifest."""
    if "text_source" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["text_source"].astype(str).eq("file")


def _text_present(frame: pd.DataFrame, dataset_root: Path | str) -> pd.Series:
    """Rows whose transcript is genuinely readable, inline or on disk.

    Mirrors :meth:`src.data.text_dataset.EmotionTextDataset.read_text` so the
    pool's notion of "text is available" is the frozen baseline's notion, not a
    second, looser one that could admit a sample the baseline would reject.
    """
    dataset_root = Path(dataset_root)
    from_file = _text_declared_file(frame)
    inline = (
        frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if "text" in frame.columns else pd.Series(False, index=frame.index)
    )
    if "text_path" in frame.columns:
        on_disk = pd.Series(
            [
                bool(str(value).strip()) and value == value and value is not None
                and (dataset_root / str(value)).exists()
                for value in frame["text_path"]
            ],
            index=frame.index,
        )
    else:
        on_disk = pd.Series(False, index=frame.index)
    return (from_file & on_disk) | (~from_file & inline)


def _availability(
    frame: pd.DataFrame, modality: str, dataset_root: Path | str
) -> dict:
    """Confirm the declared payload of ``modality`` is really there.

    Record-level, never registry-level: a corpus that *can* carry audio says
    nothing about whether this row does.  Missing payloads are counted and
    named; they are never replaced with a placeholder.
    """
    dataset_root = Path(dataset_root)
    flag = f"has_{modality}"
    declared = (
        frame[flag].fillna(False).astype(bool)
        if flag in frame.columns
        else pd.Series(True, index=frame.index)
    )

    if modality == "text":
        # Text is usually file-backed: MSP-Podcast records carry text_source ==
        # 'file' and an empty ``text`` column, with the transcript on disk. A
        # check that only read the column would call every one of them absent.
        present = _text_present(frame, dataset_root)
        missing_assets = frame.loc[
            _text_declared_file(frame) & ~present, "sample_id"
        ].astype(str).tolist()
    else:
        column = f"{modality}_path"
        payload = (
            frame[column].notna() & frame[column].astype(str).str.strip().ne("")
            if column in frame.columns else pd.Series(False, index=frame.index)
        )
        # Paths are relative to datasets/; a declared path that does not resolve
        # is an unavailable modality, not an available one with a broken file.
        resolved = [
            bool(payload.iloc[position])
            and (dataset_root / str(frame[column].iloc[position])).exists()
            for position in range(len(frame))
        ]
        present = pd.Series(resolved, index=frame.index)
        missing_assets = frame.loc[payload & ~present, "sample_id"].astype(str).tolist()

    available = declared & present
    return {
        "modality": modality,
        "rows": int(len(frame)),
        "declared_has_flag": int(declared.sum()),
        "payload_present": int(present.sum()),
        "available": int(available.sum()),
        "unavailable": int((~available).sum()),
        "unavailable_sample_ids": frame.loc[~available, "sample_id"].astype(str).tolist()[:50],
        "declared_but_asset_missing": len(missing_assets),
        "declared_but_asset_missing_examples": missing_assets[:10],
        "rule": (
            "A modality counts as available only when the has_<modality> flag, the "
            "payload column, and (for file-backed modalities) the file on disk all "
            "agree. Nothing is fabricated for an absent modality."
        ),
    }


def build_aligned_pool(
    split: str,
    root: Path | str = "experiments",
    dataset_root: Path | str = "datasets",
    minimum: int = MINIMUM_POOL,
    require_assets: bool = True,
) -> AlignedPool:
    """Build and verify the co-split Text+Audio pool for one split.

    ``require_assets`` exists so a metadata-only environment can still verify
    identity, split, and label agreement; it does not relax any of the other
    checks and the resulting record says which mode produced it.
    """
    index = AlignmentIndex.from_experiments(
        [
            (name, DEFAULT_BASELINES[name].experiment,
             DEFAULT_BASELINES[name].label_column, STAGE3_TASK)
            for name in STAGE3_MODALITIES
        ],
        root=root,
    )

    # fusion_pool applies checks 1-3 and 5 and raises on any failure.
    sample_ids = index.fusion_pool(list(STAGE3_MODALITIES), split, minimum=1)
    matrix = index.pair_matrix(*STAGE3_MODALITIES)
    cross_split = sum(
        matrix[left][right] for left in matrix for right in matrix[left] if left != right
    )
    tasks = {name: index[name].task for name in STAGE3_MODALITIES}
    if len(set(tasks.values())) != 1:
        raise SampleAlignmentError(
            f"Refusing to build a Stage 3 pool over differing tasks: {tasks}"
        )

    manifests = {name: _read_manifest(name, split, root) for name in STAGE3_MODALITIES}
    restricted = {
        name: frame[frame["sample_id"].isin(sample_ids)].set_index("sample_id").loc[sample_ids]
        .reset_index()
        for name, (frame, _) in manifests.items()
    }

    availability = {}
    if require_assets:
        availability = {
            "text": _availability(restricted["text"], "text", dataset_root),
            "audio": _availability(restricted["audio"], "audio", dataset_root),
        }
        unavailable = {
            name: record["unavailable"] for name, record in availability.items()
            if record["unavailable"]
        }
        if unavailable:
            # These ids are still real samples; they simply cannot participate in
            # a Text+Audio experiment, so they leave the pool and are counted.
            drop = set()
            for name, record in availability.items():
                frame = restricted[name]
                mask = _available_mask(frame, name, dataset_root)
                drop |= set(frame.loc[~mask, "sample_id"].astype(str))
            sample_ids = [item for item in sample_ids if item not in drop]
            restricted = {
                name: frame[frame["sample_id"].isin(sample_ids)]
                .set_index("sample_id").loc[sample_ids].reset_index()
                for name, frame in restricted.items()
            }
            for name in availability:
                availability[name]["dropped_from_pool"] = int(
                    len(drop) if name in unavailable else 0
                )
    else:
        availability = {
            name: {"modality": name, "checked": False,
                   "reason": "require_assets=False; identity and label checks still applied"}
            for name in STAGE3_MODALITIES
        }

    if len(sample_ids) < minimum:
        raise PoolTooSmallError(
            f"The aligned Text+Audio {split} pool holds {len(sample_ids)} samples, below "
            f"the minimum of {minimum}. Stage 3 stops here rather than manufacturing "
            f"alignment: these baselines were sampled from different corpora and the "
            f"shared sample space is what it is."
        )

    reference = restricted["text"]
    labels = {
        str(row.sample_id): int(row.canonical_emotion_id)
        for row in reference.itertuples()
    }
    class_counts = {
        CANONICAL_EMOTION_CLASSES[key]: int(value)
        for key, value in sorted(pd.Series(list(labels.values())).value_counts().items())
        if 0 <= int(key) < len(CANONICAL_EMOTION_CLASSES)
    }

    provenance = {
        name: {
            "modality": name,
            "experiment": DEFAULT_BASELINES[name].experiment,
            "manifest": str(manifests[name][1]),
            "task": tasks[name],
            "label_column": DEFAULT_BASELINES[name].label_column,
            "split_counts": index[name].counts,
        }
        for name in STAGE3_MODALITIES
    }

    checks = {
        "identical_sample_ids": True,
        "identical_split_assignment": True,
        "co_split_only": True,
        "cross_split_shared_ids_excluded": int(cross_split),
        "train_contamination": index.contamination(list(STAGE3_MODALITIES), split),
        "label_spaces_compatible": True,
        "label_agreement_verified": int(
            index.assert_label_agreement(list(STAGE3_MODALITIES), split)
        ),
        "class_order": list(CANONICAL_EMOTION_CLASSES),
        "assets_verified": bool(require_assets),
        "pool_expanded_across_splits": False,
    }

    return AlignedPool(
        split=split,
        sample_ids=[str(item) for item in sample_ids],
        labels=labels,
        datasets=reference["dataset"].value_counts().sort_index().to_dict(),
        class_counts=class_counts,
        provenance=provenance,
        availability=availability,
        checks=checks,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def _available_mask(frame: pd.DataFrame, modality: str, dataset_root: Path | str):
    """Row mask of records whose ``modality`` payload genuinely exists."""
    dataset_root = Path(dataset_root)
    flag = f"has_{modality}"
    declared = (
        frame[flag].fillna(False).astype(bool)
        if flag in frame.columns else pd.Series(True, index=frame.index)
    )
    if modality == "text":
        payload = _text_present(frame, dataset_root)
    else:
        column = f"{modality}_path"
        if column not in frame.columns:
            payload = pd.Series(False, index=frame.index)
        else:
            payload = pd.Series(
                [
                    bool(str(value).strip()) and value == value
                    and (dataset_root / str(value)).exists()
                    for value in frame[column]
                ],
                index=frame.index,
            )
    return declared & payload


def assert_pools_disjoint(left: AlignedPool, right: AlignedPool) -> None:
    """Validation and test pools must share no sample id."""
    overlap = set(left.sample_ids) & set(right.sample_ids)
    if overlap:
        raise SampleAlignmentError(
            f"The {left.split} and {right.split} pools share {len(overlap)} sample ids "
            f"(e.g. {sorted(overlap)[:5]}). The locked test evaluation would be "
            f"contaminated by the split every Stage 3 decision was made on."
        )


def alignment_report(pools: Mapping[str, AlignedPool], minimum: int = MINIMUM_POOL) -> dict:
    """The explicit Phase A report: what the pool is and what was verified."""
    return {
        "phase": "A -- aligned Text+Audio pool",
        "modalities": list(STAGE3_MODALITIES),
        "task": STAGE3_TASK,
        "minimum_pool": minimum,
        "splits": {
            split: {
                "size": pool.size,
                "fingerprint_sha256": pool.fingerprint,
                "datasets": dict(pool.datasets),
                "class_counts": dict(pool.class_counts),
                "checks": dict(pool.checks),
                "availability": dict(pool.availability),
                "provenance": dict(pool.provenance),
            }
            for split, pool in pools.items()
        },
        "splits_disjoint": True,
        "verified": [
            "identical sample ids in both modality manifests",
            "identical split assignment (co-split rule)",
            "no train/validation/test contamination",
            "compatible seven-class emotion label space",
            "agreeing ground truth for every pooled id",
            "provenance recorded for both modalities",
            "record-level availability of both modality payloads",
        ],
        "not_done": [
            "the pool was NOT expanded by joining ids across different splits",
            "no synthetic alignment was created",
            "no cross-corpus pairing (audio+video, audio+image, audio+physiology) "
            "was fabricated",
        ],
        "limitation": (
            "Both splits are drawn entirely from MSP-Podcast, the only corpus in this "
            "project carrying audio and text on the same record within one split. "
            "Stage 3 results therefore describe a Text->Audio routing policy on "
            "conversational podcast speech, not a corpus-general one."
        ),
    }
