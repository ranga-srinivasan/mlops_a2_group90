import torch
from PIL import Image
import numpy as np

from src.inference.model import transform


def test_image_preprocessing_shape():
    """
    Unit test for image preprocessing.

    Validates:
    - PIL image can be transformed
    - Output is a torch.Tensor
    - Shape matches (3, 224, 224)

    This test is lightweight and CI-safe (no GPU, no model).
    """

    # Create a dummy RGB image (random pixels)
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    tensor = transform(dummy_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
