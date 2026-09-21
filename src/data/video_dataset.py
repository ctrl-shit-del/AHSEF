"""Lazy, modality-filtered video dataset for categorical emotion experiments.

MELD is the only corpus that reaches this dataset: its ``video_path`` is
file-backed and its annotation is a canonical categorical emotion.  CMU-MOSEI
declares video, but as ``feature_container`` provenance behind
``aligned_50.pkl`` and with a sentiment target, so it satisfies neither the
file-backed modality rule nor the 7-class task rule.

Each item decodes a handful of uniformly spaced frames rather than the clip, so
memory stays proportional to ``num_frames``, not to clip duration.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.common.paths import DATASETS_DIR
from src.data.loaders.video import VideoLoader
from src.data.modality_dataset import ModalityManifestDataset


class EmotionVideoDataset(ModalityManifestDataset):
    """File-backed video only; frames are decoded lazily and sparsely."""

    MODALITY = "video"
    LABEL_COLUMN = "canonical_emotion_id"

    #: Smallest column set the dataset can operate on.
    MINIMAL_COLUMNS = (
        "sample_id", "dataset", "canonical_emotion_id", "has_video", "video_source", "video_path",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        num_frames: int = 8,
        frame_size: int = 48,
        max_samples: int | None = None,
        seed: int = 42,
        columns: list[str] | tuple[str, ...] | None = None,
        datasets_root: Path | None = None,
    ):
        super().__init__(
            manifest_path=manifest_path,
            max_samples=max_samples,
            seed=seed,
            columns=columns,
        )
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if frame_size < 1:
            raise ValueError("frame_size must be at least 1")
        self.num_frames = int(num_frames)
        self.frame_size = int(frame_size)
        self.loader = VideoLoader(Path(datasets_root) if datasets_root else DATASETS_DIR)

    @property
    def frame_dim(self) -> int:
        return 3 * self.frame_size * self.frame_size

    def __getitem__(self, index: int) -> dict:
        row = self.row(index)
        frames, valid = self.loader.load_frames(
            row["video_path"], num_frames=self.num_frames, frame_size=self.frame_size
        )
        return {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "video": torch.from_numpy(frames),
            "video_lengths": torch.tensor(max(valid, 1), dtype=torch.long),
            "label": torch.tensor(int(row[self.LABEL_COLUMN]), dtype=torch.long),
        }
