# ======================================================
# FastAPI imports for building the inference service
# ======================================================
# - FastAPI: main application object
# - File, UploadFile: handle multipart file uploads (images)
# - HTTPException: ensures proper HTTP error responses instead of crashes
from fastapi import FastAPI, File, UploadFile, HTTPException

# PIL is used for image decoding and manipulation.
# This mirrors the preprocessing logic used during training,
# ensuring train–serve consistency.
from PIL import Image

# io is required to convert raw uploaded bytes into a file-like object
# that PIL can read.
import io

# os is used for:
# - environment variable access
# - filesystem existence checks
# This is essential for containerized and CI/CD-safe configuration.
import os

# logging is preferred over print() for production services.
# Logs are automatically collected by Docker and CI/CD systems
# and can later be shipped to centralized logging tools.
import logging

# time is used for inference latency measurement (M5 requirement)
import time

# Project-specific inference utilities:
# - load_model: loads the trained PyTorch model from disk
# - predict_image: handles preprocessing, inference, and postprocessing
from src.inference.model import load_model, predict_image


# ======================================================
# Logging configuration
# ======================================================
# Configure global logging behavior:
# - INFO level captures key operational events
#   (startup, health checks, predictions)
# - Timestamped logs are critical for debugging in containers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a module-level logger.
# Using __name__ enables log filtering per module
# as the application scales.
logger = logging.getLogger(__name__)


# ======================================================
# Basic monitoring metrics (M5 requirement)
# ======================================================
# Simple in-app counters are sufficient for this assignment.
# These values are emitted via logs and captured by Docker/CI.
REQUEST_COUNT = 0


# ======================================================
# FastAPI application definition
# ======================================================
# Application metadata improves:
# - auto-generated OpenAPI documentation (/docs)
# - clarity during demos, grading, and reviews
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="CNN-based image classification service",
    version="1.0.0"
)


# ======================================================
# Safe model loading (CI/CD-aware design)
# ======================================================
# The model path is injected via an environment variable
# instead of being hardcoded.
#
# This is a core MLOps best practice:
# - keeps Docker images lightweight
# - avoids committing large artifacts to Git
# - allows different models per environment (dev/staging/prod)
MODEL_PATH = os.getenv("MODEL_PATH")

# Initialize the model reference.
# It is intentionally allowed to remain None during CI/CD
# where only container startup and health checks are required.
model = None

# Load the model only if:
# 1. MODEL_PATH is defined
# 2. The model file exists on disk
if MODEL_PATH and os.path.exists(MODEL_PATH):
    logger.info(f"Loading model from {MODEL_PATH}")

    # Model loading happens once at application startup.
    # This avoids repeated loading per request,
    # which would severely impact latency.
    model = load_model(MODEL_PATH)

else:
    # This branch is expected during:
    # - CI pipelines
    # - early CD stages
    # where the container is validated but the model artifact
    # is not yet mounted.
    #
    # The service still starts successfully and serves /health.
    logger.warning(
        "MODEL_PATH not set or model file not found. "
        "Running in health-only mode (CI/CD)."
    )


# ======================================================
# API Endpoints
# ======================================================
@app.get("/health")
def health_check():
    """
    Health endpoint.

    Used for:
    - container liveness checks
    - CI/CD smoke tests
    - load balancer readiness probes

    IMPORTANT:
    This endpoint must succeed even if the model is NOT loaded.
    """
    logger.info("Health check requested.")
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Inference endpoint.

    Accepts:
    - multipart/form-data image upload

    Returns:
    - predicted class label
    - class probabilities
    - inference latency (ms)
    """

    global REQUEST_COUNT
    REQUEST_COUNT += 1

    # If the model is not loaded, return a clear service-level error.
    if model is None:
        logger.warning("Prediction requested but model not loaded (CI/CD mode).")
        return {
            "label": "unavailable",
            "probabilities": {},
            "latency_ms": None
        }

    # Validate that the uploaded file is an image.
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    try:
        # Start latency timer (includes preprocessing + inference)
        start_time = time.time()

        # Read raw bytes from the uploaded file
        image_bytes = file.file.read()

        # Convert raw bytes → PIL Image
        image = Image.open(io.BytesIO(image_bytes))

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read image"
        )

    # Run inference:
    # - preprocessing
    # - model forward pass
    # - postprocessing
    result = predict_image(model, image)

    # Compute latency in milliseconds
    latency_ms = (time.time() - start_time) * 1000

    # Log monitoring metrics (M5 requirement)
    logger.info(
        f"Request #{REQUEST_COUNT} | "
        f"label={result['label']} | "
        f"latency={latency_ms:.2f} ms"
    )

    # Add latency to response (useful for demo & debugging)
    result["latency_ms"] = round(latency_ms, 2)

    return result
