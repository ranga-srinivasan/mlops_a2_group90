from pathlib import Path

# PyTorch core imports
import torch
import torch.nn as nn

# torchvision provides pretrained architectures and image transforms
from torchvision import models, transforms

# PIL Image is used for compatibility with FastAPI uploads
from PIL import Image


# ======================================================
# Configuration
# ======================================================
# Default model path used during local inference.
# In production / FastAPI usage, the model path is typically
# injected via environment variables and passed to load_model().
MODEL_PATH = Path("models/model_local.pt")

# Device selection:
# - CUDA if available (local GPU experimentation)
# - CPU otherwise (Linux VM, Docker, CI/CD)
#
# This makes the code portable across environments without changes.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels must match the training-time label order exactly.
# Any mismatch here would silently corrupt predictions.
CLASS_NAMES = ["cat", "dog"]


# ======================================================
# Model loader
# ======================================================
def load_model():
    """
    Loads the trained MobileNetV2 model for inference.

    Key design points:
    - Architecture definition must exactly match training
    - Weights are loaded explicitly from disk
    - Model is set to eval() mode for inference
    """

    # Instantiate MobileNetV2 architecture.
    # pretrained=False is intentional because:
    # - we load our own trained weights
    # - avoids downloading weights during CI/CD or container startup
    model = models.mobilenet_v2(pretrained=False)

    # Replace the final classification layer.
    # MobileNetV2 classifier structure:
    # classifier[1] is the final Linear layer.
    #
    # Output dimension must match the number of classes.
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES)
    )

    # Load trained weights from disk.
    # map_location ensures compatibility across CPU/GPU environments.
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    # Move model to selected device (CPU/GPU)
    model.to(DEVICE)

    # Set model to evaluation mode:
    # - disables dropout
    # - uses running statistics for batch normalization
    model.eval()

    return model


# ======================================================
# Image preprocessing
# ======================================================
# Define inference-time preprocessing pipeline.
#
# IMPORTANT:
# This must match the preprocessing used during training,
# otherwise train–serve skew can degrade performance.
transform = transforms.Compose([
    # Resize input image to MobileNetV2 expected resolution
    transforms.Resize((224, 224)),

    # Convert PIL Image → PyTorch tensor
    # Output shape: [C, H, W], values in [0, 1]
    transforms.ToTensor(),
])


# ======================================================
# Prediction
# ======================================================
def predict_image(model, image: Image.Image):
    """
    Runs inference on a single image.

    Steps:
    1. Ensure RGB format
    2. Apply preprocessing
    3. Forward pass through model
    4. Convert logits → probabilities
    5. Return human-readable output
    """

    # Ensure image has 3 channels (RGB).
    # This guards against grayscale or RGBA inputs.
    image = image.convert("RGB")

    # Apply preprocessing and add batch dimension:
    # Shape: [1, C, H, W]
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Disable gradient computation for inference:
    # - reduces memory usage
    # - improves performance
    with torch.no_grad():
        outputs = model(tensor)

        # Convert raw logits → probabilities
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Get predicted class index
    predicted_idx = probs.argmax()

    # Map index → class label
    predicted_label = CLASS_NAMES[predicted_idx]

    # Return structured, JSON-serializable output
    # This format is API-friendly and easy to log.
    return {
        "label": predicted_label,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }
    }
