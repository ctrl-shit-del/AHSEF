import torch
from PIL import Image
import pytest

from src.data.loaders.image import ImageLoader


def test_image_loader_tensor_contract(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (5, 7), color=(255, 128, 0)).save(image_path)
    tensor = ImageLoader(tmp_path).load_tensor("image.png", image_size=8)
    assert tensor.shape == (3, 8, 8)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_image_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError, match="Image not found"):
        ImageLoader(__import__("pathlib").Path("/tmp")).load_tensor("missing-image.png")
