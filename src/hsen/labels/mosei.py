"""CMU-MOSEI label adapter: sentiment now, six-way multi-label emotion when it exists.

What is actually on disk.  ``datasets/CMU-MOSEI/label.csv`` and
``Processed/aligned_50.pkl`` supervise **sentiment only** -- one continuous
score per utterance in ``[-3, 3]`` plus a Positive/Negative string derived from
it.  CMU-MOSEI's six non-exclusive emotion intensities live in
``CMU_MOSEI_Labels.csd`` in the CMU-MultimodalSDK release, which this repository
does not hold.

Why that matters enough to say twice.  The standardized metadata carries an
``emotion`` column for CMU-MOSEI reading ``happy`` / ``sad`` / ``neutral``.  It
is derived from the sentiment polarity, not annotated: 11,264 ``happy`` is
exactly the count of positive-scored clips.  Training a categorical emotion head
on it would be training on relabelled sentiment and reporting it as emotion
recognition, so this adapter refuses to touch that column.

Two protocols therefore:

``sentiment``
    The Acc7 / Acc2 / MAE / correlation protocol.  The categorical head predicts
    the seven ordinal sentiment buckets; the valence head regresses the score
    rescaled to ``[-1, 1]``.  Arousal is left unsupervised -- CMU-MOSEI does not
    annotate it, and a zero-filled target would be a fabricated label.

``emotion6``
    The LDDU / CARAT / TAILOR multi-label protocol the survey's 0.496 accuracy /
    0.587 micro-F1 reference points come from.  Fully implemented behind
    :meth:`MOSEILabelAdapter.build`, and it raises with a precise instruction
    until the annotation file is present, rather than silently substituting
    something weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.labels import LabelSpace, get_label_space
from src.hsen.labels.base import MISSING_CLASS_ID, LabelSet

#: The range CMU-MOSEI sentiment scores are annotated on.
SENTIMENT_SCALE = 3.0

#: Where the six-way emotion annotations would have to be for ``emotion6`` to
#: run.  Either the CMU-MultimodalSDK computational-sequence file, or a CSV with
#: one column per emotion keyed by ``video_id`` and ``clip_id``.
EMOTION6_SOURCES: tuple[str, ...] = (
    "datasets/CMU-MOSEI/CMU_MOSEI_Labels.csd",
    "datasets/CMU-MOSEI/emotion_labels.csv",
)

#: Columns whose CMU-MOSEI values are derived from sentiment rather than
#: annotated, and must never be read as emotion supervision.
DERIVED_EMOTION_COLUMNS: frozenset[str] = frozenset({"emotion", "canonical_emotion",
                                                     "canonical_emotion_id", "raw_emotion"})


class MOSEIEmotionLabelsUnavailable(FileNotFoundError):
    """Raised when the six-way emotion protocol is requested without its labels."""


def sentiment_bucket(scores: np.ndarray) -> np.ndarray:
    """Map ``[-3, 3]`` sentiment scores onto the seven ordinal Acc7 buckets.

    ``round`` then shift, which is the standard definition.  Scores are clipped
    to the annotation range first so a stray out-of-range value lands in the
    extreme bucket instead of indexing off the end of the head; anything
    non-finite stays unlabelled.
    """
    buckets = np.full(scores.shape, MISSING_CLASS_ID, dtype=np.int64)
    finite = np.isfinite(scores)
    clipped = np.clip(scores[finite], -SENTIMENT_SCALE, SENTIMENT_SCALE)
    # numpy rounds halves to even; MOSEI scores are averages of integer ratings
    # so exact halves do occur, and round-half-even keeps this deterministic and
    # symmetric about neutral rather than biased toward the positive bucket.
    buckets[finite] = np.rint(clipped).astype(np.int64) + int(SENTIMENT_SCALE)
    return buckets


@dataclass
class MOSEILabelAdapter:
    """Build CMU-MOSEI targets under the sentiment or the multi-label protocol."""

    protocol: str = "sentiment"
    score_column: str = "sentiment_score"
    emotion_label_path: Path | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.protocol not in ("sentiment", "emotion6"):
            raise ValueError(
                f"Unknown CMU-MOSEI protocol {self.protocol!r}; "
                f"expected 'sentiment' or 'emotion6'"
            )
        self.name = f"mosei_{self.protocol}"

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def emotion_labels_available(self) -> Path | None:
        """The six-way annotation file, if one is present."""
        candidates = (
            [Path(self.emotion_label_path)] if self.emotion_label_path
            else [Path(source) for source in EMOTION6_SOURCES]
        )
        return next((path for path in candidates if path.exists()), None)

    def label_space(self) -> LabelSpace:
        if self.protocol == "emotion6":
            return get_label_space("mosei_emotion6")
        return get_label_space("mosei_sentiment7")

    def selects(self, frame: pd.DataFrame) -> pd.Series:
        """Rows carrying a usable sentiment score.

        The same admission rule for both protocols: CMU-MOSEI's emotion
        annotations cover the sentiment-annotated clips, so a row without a
        score is a row without supervision either way.
        """
        scores = pd.to_numeric(frame[self.score_column], errors="coerce")
        return scores.notna()

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def build(self, frame: pd.DataFrame) -> LabelSet:
        if self.protocol == "emotion6":
            return self._build_emotion6(frame)
        return self._build_sentiment(frame)

    def _build_sentiment(self, frame: pd.DataFrame) -> LabelSet:
        scores = pd.to_numeric(frame[self.score_column], errors="coerce").to_numpy(np.float64)

        # Sentiment is a valence measurement, so it supervises the valence head
        # directly once rescaled to the same [-1, 1] range IEMOCAP's ratings use.
        # Arousal stays NaN: CMU-MOSEI does not annotate it, and the loss masks
        # NaN targets rather than driving the head toward an invented value.
        valence = np.full(scores.shape, np.nan, dtype=np.float32)
        finite = np.isfinite(scores)
        valence[finite] = np.clip(
            scores[finite] / SENTIMENT_SCALE, -1.0, 1.0
        ).astype(np.float32)

        return LabelSet(
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            label_space=self.label_space(),
            class_id=sentiment_bucket(scores),
            valence=valence,
            arousal=np.full(scores.shape, np.nan, dtype=np.float32),
            provenance={
                "adapter": "MOSEILabelAdapter",
                "protocol": "sentiment",
                "score_column": self.score_column,
                "sentiment_scale": [-SENTIMENT_SCALE, SENTIMENT_SCALE],
                "categorical_definition": "round(score) + 3, the standard Acc7 buckets",
                "valence_definition": "score / 3 -> [-1, 1]; fixed, not fitted",
                "arousal": "not annotated by CMU-MOSEI; left unsupervised",
                "emotion_columns_ignored": sorted(DERIVED_EMOTION_COLUMNS),
                "why_emotion_columns_ignored": (
                    "CMU-MOSEI's emotion column in the standardized metadata is "
                    "derived from sentiment polarity, not annotated"
                ),
            },
        )

    def _build_emotion6(self, frame: pd.DataFrame) -> LabelSet:
        path = self.emotion_labels_available()
        if path is None:
            raise MOSEIEmotionLabelsUnavailable(
                "CMU-MOSEI's six-way emotion annotations are not present.\n"
                "This repository holds sentiment supervision only "
                "(label.csv and Processed/aligned_50.pkl).\n"
                "The emotion column in the standardized metadata is derived from "
                "sentiment polarity and is NOT a substitute.\n\n"
                "To enable this protocol, place one of:\n"
                + "".join(f"    {source}\n" for source in EMOTION6_SOURCES)
                + "\nThe .csd comes from CMU-MultimodalSDK "
                "(CMU_MOSEI_Labels); the .csv form needs columns "
                "video_id, clip_id, happy, sad, angry, fear, disgust, surprise "
                "with intensities in [0, 3].\n"
                "Until then, use --label-protocol mosei_sentiment."
            )

        if path.suffix == ".csd":
            raise MOSEIEmotionLabelsUnavailable(
                f"Found {path}, but reading CMU-MultimodalSDK computational-sequence "
                f"files needs the mmsdk package, which is not a dependency of this "
                f"project.\nExport it to {EMOTION6_SOURCES[1]} with columns "
                f"video_id, clip_id, happy, sad, angry, fear, disgust, surprise."
            )

        space = self.label_space()
        annotations = pd.read_csv(path)
        missing = [name for name in ("video_id", "clip_id", *space.classes)
                   if name not in annotations.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        keys = self._sample_keys(frame)
        annotations["_key"] = (
            annotations["video_id"].astype(str) + "$_$" + annotations["clip_id"].astype(str)
        )
        indexed = annotations.set_index("_key")

        # Present-versus-absent at intensity > 0 is the binarisation LDDU, CARAT
        # and TAILOR all use; the raw intensities are kept in provenance so a
        # later phase can supervise them directly, which LDDU notes nobody does.
        matrix = np.full((len(frame), space.num_classes), np.nan, dtype=np.float32)
        found = indexed.reindex(keys)
        for index, name in enumerate(space.classes):
            values = pd.to_numeric(found[name], errors="coerce").to_numpy(np.float64)
            present = np.isfinite(values)
            matrix[present, index] = (values[present] > 0).astype(np.float32)

        scores = pd.to_numeric(frame[self.score_column], errors="coerce").to_numpy(np.float64)
        valence = np.full(scores.shape, np.nan, dtype=np.float32)
        finite = np.isfinite(scores)
        valence[finite] = np.clip(scores[finite] / SENTIMENT_SCALE, -1.0, 1.0).astype(np.float32)

        return LabelSet(
            sample_ids=frame["sample_id"].astype(str).to_numpy(),
            label_space=space,
            multilabel=matrix,
            valence=valence,
            arousal=np.full(scores.shape, np.nan, dtype=np.float32),
            provenance={
                "adapter": "MOSEILabelAdapter",
                "protocol": "emotion6",
                "annotation_source": str(path),
                "binarisation": "intensity > 0 counts as present",
                "matched": int(np.isfinite(matrix).all(axis=1).sum()),
                "valence_definition": "sentiment score / 3 -> [-1, 1]",
                "arousal": "not annotated by CMU-MOSEI; left unsupervised",
            },
        )

    @staticmethod
    def _sample_keys(frame: pd.DataFrame) -> np.ndarray:
        """``<video_id>$_$<clip_id>`` for each row, the SDK's own key form.

        Taken from the ``feature_id`` the standardizer already extracted, so the
        join uses the same key the feature container is indexed by rather than a
        second, independently parsed one.
        """
        if "feature_id" not in frame.columns:
            raise ValueError(
                "CMU-MOSEI frames need a feature_id column to join emotion "
                "annotations; it is written by the standardizer"
            )
        return frame["feature_id"].astype(str).to_numpy()
