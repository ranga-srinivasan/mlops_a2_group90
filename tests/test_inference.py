import torch
from PIL import Image
import numpy as np

from src.inference.model import predict_image, CLASS_NAMES


class DummyModel(torch.nn.Module):
    """
    Minimal dummy model for CI testing.
    Always returns fixed logits.
    """
    def forward(self, x):
        # batch_size x num_classes
        return torch.tensor([[1.0, 2.0]])


def test_predict_image_output_format():
    """
    Unit test for inference utility.

    Validates:
    - predict_image returns expected keys
    - label is valid
    - probabilities sum to ~1
    """

    model = DummyModel()

    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    result = predict_image(model, dummy_image)

    assert "label" in result
    assert "probabilities" in result
    assert result["label"] in CLASS_NAMES

    probs = result["probabilities"]
    assert isinstance(probs, dict)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
