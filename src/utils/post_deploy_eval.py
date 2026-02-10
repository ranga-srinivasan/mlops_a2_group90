"""
Post-deployment model evaluation script (M5 requirement)

This script:
1. Sends a small batch of images to the deployed FastAPI service
2. Uses known ground-truth labels
3. Computes and prints post-deployment accuracy

Run this ONLY after the inference service is up and running.
"""

import requests
from pathlib import Path

# ======================================================
# Configuration
# ======================================================

API_URL = "http://localhost:8000/predict"

# Matches your actual dataset structure
# data/processed/val/
# ├── cat/
# └── dog/
TEST_DATA_DIR = Path("data/processed/val")

SAMPLES_PER_CLASS = 5


# ======================================================
# Helper function
# ======================================================
def collect_samples():
    """
    Collect a small, balanced set of samples
    with known ground-truth labels.
    """
    samples = []

    cat_dir = TEST_DATA_DIR / "cat"
    dog_dir = TEST_DATA_DIR / "dog"

    if cat_dir.exists():
        cat_images = list(cat_dir.glob("*"))[:SAMPLES_PER_CLASS]
        for img_path in cat_images:
            samples.append((img_path, "cat"))

    if dog_dir.exists():
        dog_images = list(dog_dir.glob("*"))[:SAMPLES_PER_CLASS]
        for img_path in dog_images:
            samples.append((img_path, "dog"))

    return samples


# ======================================================
# Main evaluation logic
# ======================================================
def main():
    samples = collect_samples()

    if not samples:
        raise RuntimeError(
            "No evaluation images found. "
            "Check TEST_DATA_DIR and dataset structure."
        )

    correct = 0
    total = 0

    print("Starting post-deployment evaluation...\n")

    for img_path, true_label in samples:
        with open(img_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        img_path.name,
                        f,
                        "image/jpeg",
                    )
                },
            )

        if response.status_code != 200:
            print(
                f"[ERROR] Request failed for {img_path.name} "
                f"(status={response.status_code})"
            )
            continue

        result = response.json()
        predicted_label = result.get("label")

        is_correct = predicted_label == true_label
        correct += int(is_correct)
        total += 1

        print(
            f"Image: {img_path.name:<20} | "
            f"True: {true_label:<3} | "
            f"Pred: {predicted_label:<3} | "
            f"Correct: {is_correct}"
        )

    accuracy = correct / total if total > 0 else 0.0

    print("\n===================================")
    print(f"Post-deployment accuracy: {accuracy:.2f}")
    print(f"Total samples evaluated: {total}")
    print("===================================")


# ======================================================
# Entry point
# ======================================================
if __name__ == "__main__":
    main()
