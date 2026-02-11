# MLOps Assignment 2 – Group 90

## Overview
This project demonstrates an end-to-end MLOps workflow for a binary image classification problem (Cats vs Dogs) designed for a pet adoption platform. The pipeline covers data versioning, model development, experiment tracking, model packaging, containerization, CI/CD-based deployment, and post-deployment monitoring using open-source tools.

The assignment is structured across five milestones (M1–M5), each addressing a critical stage in the machine learning lifecycle.

---

## 1. Project Architecture
A complete end-to-end MLOps pipeline is implemented involving dataset versioning with DVC, model training and experiment tracking using MLflow, inference service development using FastAPI, containerization with Docker, automated CI/CD using GitHub Actions, deployment via Docker Compose, and basic monitoring and logging.

(Architecture diagram is included in the main report.)

---

## 2. Setup & Installation Instructions

### 2.1 Prerequisites
- Python 3.10
- Conda or venv
- Docker & Docker Compose
- DVC
- GitHub account
- (Optional) GPU for local training

---

### 2.2 Environment Setup

conda create -n mlops_a2_group90 python=3.10 -y  
conda activate mlops_a2_group90  
pip install -r requirements.txt  

---

### 2.3 Dataset Setup (DVC)

The dataset is versioned using DVC:

- Raw dataset: `data/raw.dvc`
- Processed dataset (224×224 splits): `data/processed.dvc`

If a DVC remote is configured:

dvc pull

Otherwise, ensure the processed dataset is available locally under:

data/processed/

---

## 3. Model Development & Experiment Tracking (M1)

### 3.1 Model Training

Run model training locally:

python src/models/train.py

This performs:
- A baseline model run (MobileNetV2 with frozen backbone)
- A final model run (with data augmentation)
- Logging of parameters, metrics, confusion matrix, and model artifacts to MLflow

---

### 3.2 MLflow UI

Start MLflow UI:

mlflow ui

Access MLflow at:

http://localhost:5000

Screenshots of experiments and artifacts are included in the report.

---

## 4. Model Packaging & Inference Service (M2)

### 4.1 FastAPI Inference Service

The trained model is served using FastAPI with the following endpoints:

- GET /health – Health check endpoint  
- POST /predict – Image classification endpoint returning label and probabilities  

The model is loaded at runtime using an environment variable (MODEL_PATH), enabling CI/CD-safe execution.

---

### 4.2 Run Inference Locally

uvicorn src.inference.app:app --reload

---

### 4.3 Docker Containerization

Build Docker image:

docker build -t cats-dogs-inference .

Run container:

docker run -p 8000:8000 cats-dogs-inference

Test health endpoint:

curl http://localhost:8000/health

---

## 5. Continuous Integration Pipeline (M3)

### 5.1 CI Pipeline

CI is implemented using GitHub Actions and is automatically triggered on:
- Push to main branch
- Pull requests

CI steps include:
- Dependency installation
- Unit tests (preprocessing and inference utilities)
- Docker image build validation

All CI configuration is available under:

.github/workflows/

---

## 6. Continuous Deployment & Deployment Target (M4)

### 6.1 Deployment via Docker Compose

The inference service is deployed on a Linux VM using Docker Compose.

Start deployment:

docker compose up -d

This:
- Starts the inference container
- Injects the trained model via volume mount
- Exposes the service for inference

---

### 6.2 Smoke Tests

Health check:

curl http://localhost:8000/health

(Optional) Prediction test:

curl -X POST http://localhost:8000/predict -F file=@sample.jpg

---

## 7. Monitoring & Logging (M5)

### 7.1 Logging

- Logging is implemented using Python logging module
- Logs include:
  - Health check requests
  - Prediction requests
  - Errors and exceptions

View logs:

docker compose logs --timestamps

---

### 7.2 Post-Deployment Evaluation

A small batch of known images is used to:
- Send inference requests to the deployed service
- Compare predictions against true labels
- Compute basic accuracy metrics

This validates deployed model behavior.

---

## 8. Documentation & Reporting

- This README provides execution and reproducibility instructions
- A detailed report (PDF/DOCX) includes:
  - Architecture diagram
  - MLflow screenshots
  - CI/CD screenshots
  - Confusion matrices
- A separate requirement checklist maps assignment criteria to report sections

---

## 9. Deliverables

- GitHub repository with source code
- DVC-tracked dataset metadata
- MLflow experiment logs and artifacts
- FastAPI inference service
- Dockerfile and Docker Compose configuration
- GitHub Actions CI/CD workflows
- Logs and post-deployment evaluation
- Final report (PDF/DOCX)
- [End-to-end demo video (**from Google Drive**)](https://drive.google.com/file/d/1HeFcT734VZ5lUkgfKX_ysCCS2JL974P_/view?usp=drive_link)

---

## Repository Link
[Git Repo](https://github.com/ranga-srinivasan/mlops_a2_group90)

---

Group: 90

Note: This README focuses on execution and reproducibility.  
Detailed explanations, diagrams, and screenshots are included in the final report.
