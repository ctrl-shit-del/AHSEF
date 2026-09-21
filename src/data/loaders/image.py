from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ImageLoader:
    """
    Load an image from a dataset-relative path.
    """

    def __init__(self, datasets_root: Path):
        self.datasets_root = datasets_root

    def load(self, relative_path: str):
        if not relative_path:
            raise ValueError(
                "Image path is empty."
            )

        path = self.datasets_root / relative_path

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        return Image.open(path).convert("RGB")

    def load_tensor(self, relative_path: str, image_size: int = 32) -> torch.Tensor:
        """Load a file-backed RGB image as a normalized ``[3, H, W]`` tensor."""
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        image = self.load(relative_path).resize((image_size, image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5
