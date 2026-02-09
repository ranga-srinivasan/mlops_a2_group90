# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy inference requirements
# -----------------------------
COPY requirements.inference.txt .

RUN pip install --no-cache-dir -r requirements.inference.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY src/ src/
COPY models/ models/

# -----------------------------
# Expose port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Run FastAPI app
# -----------------------------
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
