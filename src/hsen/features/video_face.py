"""The frozen video branch: face detection, crop, then MobileNetV3 per frame.

The survey is unusually direct about how much to spend here.  ESED Table 6 puts
video last of the three modalities (50.13 weighted F1 alone against text's
69.74) and worth about 0.84 points on top of text, so the recommendation is
MobileNetV3 or EfficientNet-B0 and explicitly *not* a ViT-B budget.  MobileNetV3-
Small is the lighter of the two and is what this implements.

Face crop before the CNN, not the whole frame.  An emotion model fed a full
conversational frame spends most of its receptive field on furniture, and on
IEMOCAP in particular the frame contains *two* people, only one of whom is
speaking.  Detection is done with OpenCV's Haar cascade by default because
``opencv-python`` is already a dependency and the alternative pulls a narrower
torch pin; MTCNN is available via ``detector='mtcnn'`` when ``facenet-pytorch``
is installed, and is the better detector where it can be.

When no face is found in a frame, that frame is skipped rather than replaced by
the raw frame or by zeros.  A clip in which no frame yields a face produces no
feature at all, and the sample is then recorded as having no usable video --
which is a true statement about the data and exactly the kind of per-sample
modality availability the later routing work needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.hsen.features.base import EncoderSpec, assert_frozen

#: MobileNetV3-Small's final feature map has 576 channels; MobileNetV3-Large has
#: 960. The value is read from the built model rather than assumed, but the
#: default is declared here so a config can be written before a model is built.
BACKBONE_DIMS = {"mobilenet_v3_small": 576, "mobilenet_v3_large": 960, "efficientnet_b0": 1280}

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_MAX_FRAMES = 16
CROP_SIZE = 224

# ImageNet statistics, because the backbone is an ImageNet checkpoint. These are
# constants of the pretrained weights, not statistics of our data, so they carry
# no information between splits.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VideoEncodingError(RuntimeError):
    """Raised when a clip cannot be read or contains no usable face."""


def uniform_frame_indices(total: int, wanted: int) -> list[int]:
    """Evenly spaced frame positions, deterministically.

    No random sampling anywhere: two runs of the same extraction must decode the
    same frames, or the cache is not reproducible and neither is anything
    trained on it.
    """
    if wanted < 1:
        raise ValueError("wanted must be at least 1")
    if total <= 0:
        return []
    if total <= wanted:
        return list(range(total))
    step = total / wanted
    return [min(total - 1, int(index * step + step / 2)) for index in range(wanted)]


class HaarFaceDetector:
    """OpenCV's frontal-face cascade.  Fast, dependency-free, and adequate here."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        import cv2

        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not path.exists():
            raise VideoEncodingError(f"OpenCV cascade not found at {path}")
        self._cascade = cv2.CascadeClassifier(str(path))
        self.scale_factor, self.min_neighbors = scale_factor, min_neighbors
        self.name = "opencv_haar_frontalface"

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """The largest face box in one BGR frame, or ``None``."""
        import cv2

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            grey, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
        )
        if len(faces) == 0:
            return None
        # Largest box: in a two-speaker IEMOCAP frame this is a heuristic, not a
        # speaker-identification method, and the extractor records which half of
        # the frame it searched so the choice is not silent.
        return max(faces, key=lambda box: int(box[2]) * int(box[3]))


class MTCNNFaceDetector:
    """facenet-pytorch's MTCNN.  Better, and an optional dependency."""

    def __init__(self, device: str = "cpu"):
        try:
            from facenet_pytorch import MTCNN
        except ImportError as error:  # pragma: no cover - optional dependency
            raise VideoEncodingError(
                "facenet-pytorch is required for the MTCNN detector.\n"
                "    pip install facenet-pytorch\n"
                "Or use --detector haar, which needs nothing extra."
            ) from error
        self._mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
        self.name = "mtcnn"

    def detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        import cv2

        boxes, _ = self._mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if boxes is None or len(boxes) == 0:
            return None
        left, top, right, bottom = boxes[0]
        return int(left), int(top), int(right - left), int(bottom - top)


def build_detector(kind: str, device: str = "cpu"):
    if kind == "haar":
        return HaarFaceDetector()
    if kind == "mtcnn":
        return MTCNNFaceDetector(device)
    raise ValueError(f"Unknown detector {kind!r}; expected 'haar' or 'mtcnn'")


class FaceVideoEncoder:
    """Frozen MobileNetV3 over detected face crops, cached as ``[T, D]``."""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        detector: str = "haar",
        max_frames: int = DEFAULT_MAX_FRAMES,
        device: str = "cpu",
        #: ``left`` / ``right`` restrict detection to one half of the frame,
        #: which is what IEMOCAP's two-speaker dialog video needs.
        frame_half: str | None = None,
    ):
        if backbone not in BACKBONE_DIMS:
            raise ValueError(
                f"Unknown backbone {backbone!r}; expected one of {sorted(BACKBONE_DIMS)}"
            )
        self.backbone_name = backbone
        self.detector_name = detector
        self.max_frames = int(max_frames)
        self.device = torch.device(device)
        self.frame_half = frame_half
        self._model = None
        self._detector = None
        self.frames_without_face = 0
        self.clips_without_face = 0
        self.spec = EncoderSpec(
            modality="video",
            model=backbone,
            feature_dim=BACKBONE_DIMS[backbone],
            max_frames=self.max_frames,
            layer=-1,
            options={
                "detector": detector,
                "crop_size": CROP_SIZE,
                "frame_half": frame_half,
                "frame_selection": "uniform, deterministic",
            },
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        from torchvision import models

        builders = {
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
            "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
        }
        builder, weights = builders[self.backbone_name]
        model = builder(weights=weights.IMAGENET1K_V1)
        # The classifier is dropped: what is cached is the pooled feature map,
        # not 1,000 ImageNet logits.
        model.classifier = torch.nn.Identity()
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert_frozen(model)
        self._model = model
        self._detector = build_detector(self.detector_name, str(self.device))

    # ---------------------------------------------------------------- encode

    def _crop(self, frame: np.ndarray) -> np.ndarray | None:
        import cv2

        height, width = frame.shape[:2]
        offset = 0
        search = frame
        if self.frame_half == "left":
            search = frame[:, : width // 2]
        elif self.frame_half == "right":
            search, offset = frame[:, width // 2:], width // 2

        box = self._detector.detect(search)
        if box is None:
            return None
        left, top, box_width, box_height = box
        left += offset
        # A 20% margin: a tight detector box cuts the forehead and chin, and
        # both carry expression.
        margin_x, margin_y = int(0.2 * box_width), int(0.2 * box_height)
        left, top = max(0, left - margin_x), max(0, top - margin_y)
        right = min(width, left + box_width + 2 * margin_x)
        bottom = min(height, top + box_height + 2 * margin_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

    @torch.inference_mode()
    def encode(self, payloads: Sequence[Path | str]) -> list[np.ndarray]:
        """Clips to ``[T, D]`` float32 face-frame features, one per clip."""
        import cv2

        self._load()
        out: list[np.ndarray] = []
        for payload in payloads:
            path = Path(payload)
            if not path.exists():
                raise VideoEncodingError(f"Video file not found: {path}")

            capture = cv2.VideoCapture(str(path))
            try:
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                wanted = set(uniform_frame_indices(total, self.max_frames))
                crops, position = [], 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if position in wanted:
                        crop = self._crop(frame)
                        if crop is None:
                            self.frames_without_face += 1
                        else:
                            crops.append(crop)
                    position += 1
            finally:
                capture.release()

            if not crops:
                self.clips_without_face += 1
                raise VideoEncodingError(
                    f"No face detected in any sampled frame of {path.name}. "
                    f"This clip has no usable video; record it as video-unavailable "
                    f"rather than caching a fabricated feature."
                )

            stack = np.stack(crops).astype(np.float32)[..., ::-1] / 255.0   # BGR -> RGB
            stack = (stack - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(np.ascontiguousarray(stack.transpose(0, 3, 1, 2)))
            features = self._model(tensor.to(self.device)).float().cpu().numpy()
            out.append(np.ascontiguousarray(features, dtype=np.float32))
        return out

    def describe(self) -> dict:
        return self.spec.describe() | {
            "class": "FaceVideoEncoder",
            "backbone": self.backbone_name,
            "detector": self.detector_name,
            "frames_without_face": int(self.frames_without_face),
            "clips_without_face": int(self.clips_without_face),
            "device": str(self.device),
            "frozen": True,
            "trained_by_this_project": False,
            "pretraining": "ImageNet; not yet fine-tuned on AffectNet+/RAF-DB",
        }
