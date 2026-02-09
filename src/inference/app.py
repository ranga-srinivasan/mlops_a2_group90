from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from src.inference.model import load_model, predict_image

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="CNN-based image classification service",
    version="1.0.0"
)

# Load model at startup
model = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image")

    result = predict_image(model, image)
    return result
