from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = Path("models/model_local.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["cat", "dog"]

# -----------------------------
# Model loader
# -----------------------------
def load_model():
    model = models.mobilenet_v2(pretrained=False)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -----------------------------
# Prediction
# -----------------------------
def predict_image(model, image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = probs.argmax()
    predicted_label = CLASS_NAMES[predicted_idx]

    return {
        "label": predicted_label,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }
    }
