"""Cross-modality sample identity: what may legitimately be fused with what.

The single most dangerous assumption available to a multimodal fusion study is
that "sample 37" of the audio test set and "sample 37" of the text test set
describe the same recording.  In this project they usually do not: the five
baselines were sampled independently from *different* corpora, and only some
corpora carry more than one modality.

This module answers three questions from the experiment manifests themselves,
never from position:

1. Which ``sample_id`` values does each ``(modality, split)`` actually contain?
2. For a pair of modalities, how do their splits intersect -- including the
   dangerous off-diagonal cells, where one model's *test* sample is another
   model's *training* sample?
3. Given a set of modalities and a split, which sample ids may be fused?

The answer to (3) is the ``co-split pool``: ids present in the *same* split of
every participating modality.  An id that is evaluation data for one model and
training data for another is excluded and reported, because fusing it would
measure the second model's memorisation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.common.experiment_layout import SPLITS, ExperimentLayout


class SampleAlignmentError(RuntimeError):
    """Raised when a requested fusion would join records that are not the same sample."""


class SplitContaminationError(RuntimeError):
    """Raised when a fusion pool contains a sample another model trained on."""


@dataclass(frozen=True)
class ModalitySplits:
    """The sample ids of one modality experiment, per split."""

    modality: str
    experiment: str
    metadata_dir: str
    ids: Mapping[str, frozenset[str]]
    labels: Mapping[str, Mapping[str, int]]
    task: str

    def split_ids(self, split: str) -> frozenset[str]:
        return self.ids.get(split, frozenset())

    @property
    def counts(self) -> dict[str, int]:
        return {split: len(self.ids.get(split, ())) for split in SPLITS}


def _read_manifest_ids(path: Path, label_column: str) -> tuple[frozenset[str], dict[str, int]]:
    try:
        frame = pd.read_parquet(path, columns=["sample_id", label_column])
    except (KeyError, ValueError):
        # A pool whose label column differs (physiology carries a WESAD state
        # id, not a canonical emotion id) still has usable sample identities;
        # -1 marks "no comparable label" so label agreement simply skips it.
        frame = pd.read_parquet(path, columns=["sample_id"])
        frame[label_column] = -1
    identifiers = frame["sample_id"].astype(str)
    duplicated = identifiers[identifiers.duplicated()].tolist()
    if duplicated:
        raise SampleAlignmentError(
            f"{path} contains duplicate sample_id values ({duplicated[:5]}...); "
            f"a sample identity that is not unique cannot anchor a fusion."
        )
    labels = pd.to_numeric(frame[label_column], errors="coerce").fillna(-1).astype(int)
    return frozenset(identifiers), dict(zip(identifiers, labels))


def load_modality_splits(
    modality: str,
    experiment: str,
    label_column: str = "canonical_emotion_id",
    task: str = "emotion_7class",
    root: Path | str = "experiments",
) -> ModalitySplits:
    """Read the ``sample_id`` sets of one baseline experiment's three splits."""
    layout = ExperimentLayout.from_name(experiment, Path(root))
    ids: dict[str, frozenset[str]] = {}
    labels: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        path = layout.split_path(split)
        if not path.exists():
            raise FileNotFoundError(f"Missing experiment manifest: {path}")
        ids[split], labels[split] = _read_manifest_ids(path, label_column)
    return ModalitySplits(
        modality=modality, experiment=experiment,
        metadata_dir=str(layout.metadata_dir), ids=ids, labels=labels, task=task,
    )


@dataclass(frozen=True)
class AlignmentIndex:
    """Sample identity across every registered modality experiment."""

    modalities: Mapping[str, ModalitySplits]

    @classmethod
    def from_experiments(
        cls, specs: Sequence[tuple[str, str, str, str]], root: Path | str = "experiments"
    ) -> "AlignmentIndex":
        """Build from ``(modality, experiment, label_column, task)`` tuples."""
        return cls({
            modality: load_modality_splits(modality, experiment, label_column, task, root)
            for modality, experiment, label_column, task in specs
        })

    def __getitem__(self, modality: str) -> ModalitySplits:
        try:
            return self.modalities[modality]
        except KeyError as error:
            raise KeyError(
                f"Unknown modality {modality!r}; the index holds "
                f"{sorted(self.modalities)}"
            ) from error

    @property
    def names(self) -> list[str]:
        return sorted(self.modalities)

    # -------------------------------------------------------------- matrices

    def intersection(self, left: str, left_split: str, right: str, right_split: str) -> set[str]:
        return set(self[left].split_ids(left_split) & self[right].split_ids(right_split))

    def pair_matrix(self, left: str, right: str) -> dict[str, dict[str, int]]:
        """``{left_split: {right_split: shared_count}}`` for one modality pair."""
        return {
            left_split: {
                right_split: len(self.intersection(left, left_split, right, right_split))
                for right_split in SPLITS
            }
            for left_split in SPLITS
        }

    def alignment_matrix(self) -> dict:
        """The full pairwise split-by-split overlap picture, ready to serialise."""
        pairs = {}
        for left, right in itertools.combinations(self.names, 2):
            matrix = self.pair_matrix(left, right)
            total = sum(sum(row.values()) for row in matrix.values())
            co_split = {split: matrix[split][split] for split in SPLITS}
            cross_split = total - sum(co_split.values())
            pairs[f"{left}|{right}"] = {
                "left": left,
                "right": right,
                "matrix": matrix,
                "co_split": co_split,
                "cross_split_shared": cross_split,
                "shares_any_sample": total > 0,
                "fusable_splits": [split for split, count in co_split.items() if count > 0],
            }
        return {
            "modalities": {
                name: {
                    "experiment": item.experiment,
                    "metadata_dir": item.metadata_dir,
                    "task": item.task,
                    "counts": item.counts,
                }
                for name, item in sorted(self.modalities.items())
            },
            "split_order": list(SPLITS),
            "pairs": pairs,
            "definition": {
                "co_split": "ids present in the SAME split of both modalities; the only "
                            "pool a fused evaluation may use",
                "cross_split_shared": "ids shared but in different splits; excluded from "
                                      "fusion because one model trained on them",
            },
        }

    # ------------------------------------------------------------ fusion pool

    def co_split_pool(self, modalities: Sequence[str], split: str) -> list[str]:
        """Sorted ids present in ``split`` of *every* named modality."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if len(modalities) < 2:
            raise ValueError("A fusion pool needs at least two modalities")
        pool: set[str] | None = None
        for modality in modalities:
            ids = set(self[modality].split_ids(split))
            pool = ids if pool is None else pool & ids
        return sorted(pool or ())

    def contamination(self, modalities: Sequence[str], split: str) -> dict[str, dict[str, int]]:
        """Ids in the pool that another participating model saw during *training*.

        The co-split pool cannot contain such an id by construction (an id is
        in exactly one split per modality), so a non-empty result means the
        manifests disagree about split membership and must be fixed upstream.
        """
        pool = set(self.co_split_pool(modalities, split))
        found: dict[str, dict[str, int]] = {}
        for modality in modalities:
            leaked = pool & set(self[modality].split_ids("train"))
            if leaked:
                found[modality] = {"train_overlap": len(leaked)}
        return found

    def assert_label_agreement(self, modalities: Sequence[str], split: str) -> int:
        """Verify every modality assigns the same class to each pooled id.

        A disagreement means the two manifests are describing different things
        under one identifier, which invalidates the pool outright.
        """
        pool = self.co_split_pool(modalities, split)
        if not pool:
            return 0
        reference = self[modalities[0]].labels[split]
        for modality in modalities[1:]:
            other = self[modality].labels[split]
            mismatched = [
                identifier for identifier in pool
                if reference.get(identifier, -1) >= 0 and other.get(identifier, -1) >= 0
                and reference[identifier] != other[identifier]
            ]
            if mismatched:
                raise SampleAlignmentError(
                    f"{modalities[0]} and {modality} disagree about the label of "
                    f"{len(mismatched)} pooled {split} samples (e.g. {mismatched[:5]}); "
                    f"they cannot be the same samples."
                )
        return len(pool)

    def fusion_pool(
        self, modalities: Sequence[str], split: str, minimum: int = 1
    ) -> list[str]:
        """The vetted pool: co-split, uncontaminated, label-consistent.

        Raises rather than returning a smaller-than-useful pool silently.
        """
        pool = self.co_split_pool(modalities, split)
        if len(pool) < minimum:
            raise SampleAlignmentError(
                f"Modalities {list(modalities)} share only {len(pool)} co-split "
                f"{split} samples (minimum {minimum}). These baselines were sampled "
                f"from different corpora; fusing them would join unrelated records."
            )
        contaminated = self.contamination(modalities, split)
        if contaminated:
            raise SplitContaminationError(
                f"The {split} fusion pool for {list(modalities)} contains samples that "
                f"appear in another participant's training split: {contaminated}"
            )
        self.assert_label_agreement(modalities, split)
        return pool

    def fusability_report(
        self, anchor: str, candidates: Iterable[str], split: str, minimum: int = 1
    ) -> dict:
        """For an anchor modality, say plainly which candidates can be fused and why not."""
        entries = {}
        for candidate in candidates:
            pair = (anchor, candidate)
            pool = self.co_split_pool(pair, split)
            same_task = self[anchor].task == self[candidate].task
            blockers = []
            if len(pool) < minimum:
                blockers.append(
                    f"only {len(pool)} co-split {split} samples "
                    f"(shared across any split: "
                    f"{sum(sum(row.values()) for row in self.pair_matrix(*pair).values())})"
                )
            if not same_task:
                blockers.append(
                    f"label spaces differ: {anchor} solves {self[anchor].task!r} but "
                    f"{candidate} solves {self[candidate].task!r}"
                )
            entries[candidate] = {
                "co_split_samples": len(pool),
                "shares_any_sample": bool(
                    sum(sum(row.values()) for row in self.pair_matrix(*pair).values())
                ),
                "same_task": same_task,
                "fusable": not blockers,
                "blockers": blockers,
            }
        return {
            "anchor": anchor,
            "split": split,
            "minimum_pool": minimum,
            "candidates": entries,
            "fusable_candidates": sorted(k for k, v in entries.items() if v["fusable"]),
        }
