import os
import shutil
import random
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
RAW_DATA_DIR = Path("data/raw/PetImages")
PROCESSED_DATA_DIR = Path("data/processed")

IMAGE_SIZE = (224, 224)
SEED = 42

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

CLASSES = {
    "Cat": "cat",
    "Dog": "dog"
}

# -----------------------------
# Utility functions
# -----------------------------
def is_image_valid(img_path: Path) -> bool:
    """
    Check if an image can be opened.
    Returns False for corrupted images.
    """
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def process_and_save_image(src_path: Path, dst_path: Path):
    """
    Convert image to RGB, resize, and save.
    """
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)


# -----------------------------
# Main preprocessing logic
# -----------------------------
def main():
    random.seed(SEED)

    all_images = []
    corrupted_images = 0

    print("Scanning raw dataset...")

    for raw_class, class_name in CLASSES.items():
        class_dir = RAW_DATA_DIR / raw_class
        for img_path in class_dir.glob("*.jpg"):
            if is_image_valid(img_path):
                all_images.append((img_path, class_name))
            else:
                corrupted_images += 1

    print(f"Valid images found: {len(all_images)}")
    print(f"Corrupted images skipped: {corrupted_images}")

    # Separate paths and labels
    image_paths = [x[0] for x in all_images]
    labels = [x[1] for x in all_images]

    # Train split
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=(1 - TRAIN_SPLIT),
        stratify=labels,
        random_state=SEED
    )

    # Validation + Test split
    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=SEED
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    print("📁 Creating processed dataset...")

    for split_name, (paths, labels) in splits.items():
        for img_path, label in zip(paths, labels):
            dst_path = PROCESSED_DATA_DIR / split_name / label / img_path.name
            process_and_save_image(img_path, dst_path)

    print("\nDataset split summary:")
    print(f"Train: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
