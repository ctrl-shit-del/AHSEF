from pathlib import Path

import numpy as np


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "opencv-python is required for video frame decoding. "
            "Install it with: pip install opencv-python"
        ) from exc
    return cv2


def uniform_indices(total_frames: int, wanted: int) -> list[int]:
    """Pick ``wanted`` frame positions spread evenly across ``total_frames``.

    Deterministic by construction: no sampling, no randomness, so two runs of
    the same experiment decode exactly the same frames.
    """
    if wanted < 1:
        raise ValueError("wanted must be at least 1")
    if total_frames <= 0:
        return []
    if total_frames <= wanted:
        return list(range(total_frames))
    step = total_frames / wanted
    return [min(total_frames - 1, int(index * step + step / 2)) for index in range(wanted)]


class VideoLoader:
    """
    Resolve a video file and decode a bounded set of frames from it.

    Decoding is deliberately sparse: a baseline never needs every frame, and
    reading a whole clip into RAM would defeat the memory budget this project
    runs under.
    """

    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root

    def resolve(self, relative_path: str) -> Path:
        if not relative_path:
            raise ValueError(
                "Video path is empty."
            )

        path = self.datasets_root / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Video not found: {path}"
            )

        return path

    def load(self, relative_path: str):

        return {
            "path": str(self.resolve(relative_path)),
        }

    def load_frames(
        self,
        relative_path: str,
        num_frames: int = 8,
        frame_size: int = 48,
    ) -> tuple[np.ndarray, int]:
        """Return ``([num_frames, 3, S, S] float32, decoded_frame_count)``.

        Frames are uniformly spaced over the clip, resized to ``frame_size``,
        RGB-ordered, and scaled to ``[-1, 1]`` to match the image baseline's
        normalisation.  Short or unreadable clips are zero-padded and report
        the number of real frames so the model can pool over those only.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        if frame_size < 1:
            raise ValueError("frame_size must be at least 1")

        cv2 = _require_cv2()
        path = self.resolve(relative_path)
        frames = np.zeros((num_frames, 3, frame_size, frame_size), dtype=np.float32)

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            wanted = uniform_indices(total, num_frames) if total > 0 else list(range(num_frames))
            wanted_set = set(wanted)
            # Sequential decode is both cheaper and far more reliable than
            # per-frame seeking on the codecs these corpora ship.
            limit = max(wanted) if wanted else num_frames
            decoded: list[np.ndarray] = []
            position = 0
            while position <= limit and len(decoded) < num_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if position in wanted_set or total <= 0:
                    resized = cv2.resize(
                        frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA
                    )
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    decoded.append(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32))
                position += 1
        finally:
            capture.release()

        for index, frame in enumerate(decoded[:num_frames]):
            frames[index] = (frame / 255.0 - 0.5) / 0.5
        return frames, min(len(decoded), num_frames)
