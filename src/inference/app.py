from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import logging

from src.inference.model import load_model, predict_image

# -----------------------
# Logging configuration
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="CNN-based image classification service",
    version="1.0.0"
)

# -----------------------
# Safe model loading
# -----------------------
MODEL_PATH = os.getenv("MODEL_PATH")

model = None
if MODEL_PATH and os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    logger.warning(
        "MODEL_PATH not set or model file not found. "
        "Running in health-only mode (CI/CD)."
    )

# -----------------------
# Endpoints
# -----------------------
@app.get("/health")
def health_check():
    logger.info("Health check requested.")
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Prediction unavailable."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image")

    result = predict_image(model, image)

    logger.info(
        f"Prediction made | label={result['label']} | probabilities={result['probabilities']}"
    )

    return result

