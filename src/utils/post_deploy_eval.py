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

# URL of the deployed inference service
API_URL = "http://localhost:8000/predict"

# Path to test dataset (adjust if needed)
# Expected structure:
# data/test/
# ├── cats/
# └── dogs/
TEST_DATA_DIR = Path("data/test")

# Number of samples per class (keep small, rubric-friendly)
SAMPLES_PER_CLASS = 5


# ======================================================
# Helper function
# ======================================================
def collect_samples():
    """
    Collect a small, balanced set of test samples
    with known ground-truth labels.
    """
    samples = []

    cat_images = list((TEST_DATA_DIR / "cats").glob("*"))[:SAMPLES_PER_CLASS]
    dog_images = list((TEST_DATA_DIR / "dogs").glob("*"))[:SAMPLES_PER_CLASS]

    for img_path in cat_images:
        samples.append((img_path, "cat"))

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
            "No test images found. "
            "Check TEST_DATA_DIR path and dataset structure."
        )

    correct = 0
    total = 0

    print("Starting post-deployment evaluation...\n")

    for img_path, true_label in samples:
        with open(img_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": f}
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
