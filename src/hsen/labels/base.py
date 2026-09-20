"""The supervision container every HSEN dataset is reduced to.

One shape, three optional channels.  A dataset that supervises only some of them
leaves the rest absent, and "absent" is represented explicitly -- ``-1`` for a
categorical id, ``NaN`` for a regression target -- so a missing label can never
be silently trained on as if it were a real one.  The loss masks on exactly
these sentinels; see :mod:`src.hsen.training.losses`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.common.labels import LabelSpace

#: Categorical id for a sample whose class is outside the protocol or unlabelled.
#: Chosen rather than ``NaN`` so the array stays integral and indexable, and
#: rather than a real class id so it can never be confused with one.
MISSING_CLASS_ID: int = -1


@dataclass(frozen=True)
class LabelSet:
    """Supervision for one split, aligned row-for-row with its manifest.

    ``class_id`` and ``multilabel`` are mutually exclusive: a dataset is either
    single-label categorical or multi-label, never both, and the head built for
    it follows from which one is populated.
    """

    sample_ids: np.ndarray
    label_space: LabelSpace | None = None
    #: ``[N]`` int64, ``MISSING_CLASS_ID`` where unsupervised.
    class_id: np.ndarray | None = None
    #: ``[N, C]`` float32 in ``[0, 1]``, ``NaN`` rows where unsupervised.
    multilabel: np.ndarray | None = None
    #: ``[N]`` float32 in ``[-1, 1]``, ``NaN`` where unsupervised.
    valence: np.ndarray | None = None
    #: ``[N]`` float32 in ``[-1, 1]``, ``NaN`` where unsupervised.
    arousal: np.ndarray | None = None
    #: Free-form record of how these targets were derived, saved with the run.
    provenance: dict | None = None

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        for name in ("class_id", "multilabel", "valence", "arousal"):
            value = getattr(self, name)
            if value is not None and len(value) != count:
                raise ValueError(
                    f"LabelSet.{name} has {len(value)} rows but there are "
                    f"{count} sample ids; targets must align row-for-row with "
                    f"the manifest they were built from"
                )
        if self.class_id is not None and self.multilabel is not None:
            raise ValueError(
                "A LabelSet carries either a single-label class_id or a "
                "multilabel matrix, not both"
            )
        if self.class_id is not None and self.label_space is None:
            raise ValueError("A categorical LabelSet must declare its label space")

    @property
    def task(self) -> str:
        if self.multilabel is not None:
            return "multi_label"
        if self.class_id is not None:
            return "single_label"
        return "regression_only"

    @property
    def num_classes(self) -> int:
        if self.multilabel is not None:
            return int(self.multilabel.shape[1])
        if self.label_space is not None:
            return self.label_space.num_classes
        return 0

    @property
    def has_valence(self) -> bool:
        return self.valence is not None and bool(np.isfinite(self.valence).any())

    @property
    def has_arousal(self) -> bool:
        return self.arousal is not None and bool(np.isfinite(self.arousal).any())

    def supervised_mask(self) -> np.ndarray:
        """Rows carrying at least one usable target."""
        mask = np.zeros(len(self.sample_ids), dtype=bool)
        if self.class_id is not None:
            mask |= self.class_id != MISSING_CLASS_ID
        if self.multilabel is not None:
            mask |= np.isfinite(self.multilabel).all(axis=1)
        for value in (self.valence, self.arousal):
            if value is not None:
                mask |= np.isfinite(value)
        return mask

    def class_counts(self) -> dict[str, int]:
        """Per-class sample counts in declared class order, never sorted."""
        if self.class_id is None or self.label_space is None:
            return {}
        return {
            name: int((self.class_id == index).sum())
            for index, name in enumerate(self.label_space.classes)
        }

    def describe(self) -> dict:
        return {
            "task": self.task,
            "samples": len(self.sample_ids),
            "supervised": int(self.supervised_mask().sum()),
            "label_space": self.label_space.to_dict() if self.label_space else None,
            "class_counts": self.class_counts(),
            "valence_supervised": int(np.isfinite(self.valence).sum())
            if self.valence is not None else 0,
            "arousal_supervised": int(np.isfinite(self.arousal).sum())
            if self.arousal is not None else 0,
            "provenance": self.provenance or {},
        }


@runtime_checkable
class LabelAdapter(Protocol):
    """Turns one dataset's metadata frame into a :class:`LabelSet`."""

    name: str

    def label_space(self) -> LabelSpace | None:
        """The categorical space, or ``None`` for a purely regression task."""

    def selects(self, frame: pd.DataFrame) -> pd.Series:
        """Boolean mask of the rows this protocol admits.

        Applied *before* splitting, so a protocol that excludes a class excludes
        it identically from train, validation and test.
        """

    def build(self, frame: pd.DataFrame) -> LabelSet:
        """Targets for ``frame``, aligned to its row order."""


def label_set_from_manifest(
    frame: pd.DataFrame, label_space: LabelSpace | None, provenance: dict | None = None,
) -> LabelSet:
    """Read targets straight out of a built manifest.

    The manifest is the *materialised output* of a label adapter: the adapter
    defines the protocol, the manifest builder applies it once, and the audit
    signs off on the result.  Re-running the adapter at training time would
    apply its transforms a second time to values that already carry them --
    IEMOCAP's valence would be rescaled from [-1, 1] as though it were still a
    1-to-5 rating -- so the targets are read, not recomputed.
    """
    def column(name: str) -> np.ndarray | None:
        if name not in frame.columns:
            return None
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(np.float32)

    multilabel = None
    if label_space is not None and all(
        f"label_{name}" in frame.columns for name in label_space.classes
    ):
        multilabel = np.stack(
            [column(f"label_{name}") for name in label_space.classes], axis=1
        ).astype(np.float32)

    class_id = None
    if multilabel is None and "class_id" in frame.columns:
        class_id = frame["class_id"].to_numpy(np.int64)

    return LabelSet(
        sample_ids=frame["sample_id"].astype(str).to_numpy(),
        label_space=label_space,
        class_id=class_id,
        multilabel=multilabel,
        valence=column("valence_target"),
        arousal=column("arousal_target"),
        provenance=(provenance or {}) | {"source": "built manifest; targets read, not recomputed"},
    )


def resolve_label_adapter(name: str, **kwargs) -> LabelAdapter:
    """Look up a label adapter by protocol name.

    Imported lazily to keep :mod:`src.hsen.labels.base` free of dependencies on
    the concrete adapters that implement its protocol.
    """
    from src.hsen.labels.iemocap import IEMOCAPLabelAdapter
    from src.hsen.labels.mosei import MOSEILabelAdapter

    builders = {
        "iemocap_erc6": lambda: IEMOCAPLabelAdapter(protocol="erc6", **kwargs),
        "iemocap_ser4": lambda: IEMOCAPLabelAdapter(protocol="ser4", **kwargs),
        "mosei_sentiment": lambda: MOSEILabelAdapter(protocol="sentiment", **kwargs),
        "mosei_emotion6": lambda: MOSEILabelAdapter(protocol="emotion6", **kwargs),
    }
    try:
        return builders[name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown label protocol {name!r}; expected one of {sorted(builders)}"
        ) from error
