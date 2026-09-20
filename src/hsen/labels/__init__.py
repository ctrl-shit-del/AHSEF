"""Per-dataset label adapters.

Each dataset supervises a different thing, and the differences are real: IEMOCAP
gives one categorical label per utterance plus dimensional valence/arousal;
CMU-MOSEI gives a continuous sentiment score and, in its full release, six
non-exclusive emotion intensities.  A model that hardcodes either shape cannot
train on the other.

So the shape lives here instead.  An adapter turns a metadata frame into a
:class:`~src.hsen.labels.base.LabelSet` -- a fixed container the trainer, the
loss and the metrics all understand -- and declares which of its fields are
actually supervised.  Nothing downstream asks "which dataset is this?".
"""

from src.hsen.labels.base import (
    LabelAdapter,
    LabelSet,
    MISSING_CLASS_ID,
    label_set_from_manifest,
    resolve_label_adapter,
)
from src.hsen.labels.iemocap import IEMOCAPLabelAdapter
from src.hsen.labels.mosei import MOSEILabelAdapter

__all__ = [
    "LabelAdapter",
    "LabelSet",
    "MISSING_CLASS_ID",
    "label_set_from_manifest",
    "IEMOCAPLabelAdapter",
    "MOSEILabelAdapter",
    "resolve_label_adapter",
]
