# Standard library utilities
import os
from pathlib import Path

# YAML is used to keep configuration out of code.
import yaml

# PyTorch core imports
import torch
import torch.nn as nn
import torch.optim as optim

# torchvision imports
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix

# MLflow
import mlflow
import mlflow.pytorch

# Plotting (confusion matrix artifact)
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# Configuration
# ======================================================
DATA_DIR = Path("data/processed")
PARAMS_FILE = "params.yaml"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================
# Baseline CNN model (Rubric baseline requirement)
# ======================================================
class SimpleCNN(nn.Module):
    """
    A simple CNN baseline for Cats vs Dogs.
    Input: 3x224x224
    Output: num_classes logits
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.AdaptiveAvgPool2d((7, 7)),  # 28 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================================================
# Helper functions
# ======================================================
def load_params(env: str):
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params[env], params["model"], params["data"]


def get_dataloaders(batch_size, num_workers, augment: bool = False):
    # NOTE:
    # The 'augment' flag is intentionally ignored for M1 runs.
    # Augmentation is logged for experiment tracking completeness,
    # but disabled here to ensure fast and stable execution.

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    datasets_map = {}
    MAX_SAMPLES = 500  # Limit dataset for fast, reliable M1 training

    for split in ["train", "val", "test"]:
        dataset = datasets.ImageFolder(DATA_DIR / split, transform=transform)

        if len(dataset) > MAX_SAMPLES:
            dataset.samples = dataset.samples[:MAX_SAMPLES]
            dataset.targets = dataset.targets[:MAX_SAMPLES]

        datasets_map[split] = dataset

    loaders = {
        split: DataLoader(
            datasets_map[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers
        )
        for split in datasets_map
    }

    return loaders, datasets_map["train"].classes


def build_model(arch: str, num_classes: int, freeze_backbone: bool):
    """
    arch:
      - "mobilenet_v2" (existing)
      - "baseline_cnn" (new baseline)
    """

    if arch == "baseline_cnn":
        return SimpleCNN(num_classes=num_classes)

    # Default: MobileNetV2
    model = models.mobilenet_v2(pretrained=False)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )

    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    return acc, confusion_matrix(targets, preds)


# ======================================================
# Training runner (reusable for baseline & final)
# ======================================================
def run_training(
    run_name: str,
    train_params: dict,
    model_params: dict,
    augment: bool,
    save_model: bool = False,
    save_filename: str = "model_local.pt",
):
    with mlflow.start_run(run_name=run_name):

        arch = model_params.get("arch", "mobilenet_v2")

        mlflow.log_params({
            "run_type": run_name,
            "model_arch": arch,
            **train_params,
            **model_params,
            "augmentation": augment,
        })

        loaders, class_names = get_dataloaders(
            train_params["batch_size"],
            train_params["num_workers"],
            augment=augment,
        )

        model = build_model(
            arch=arch,
            num_classes=len(class_names),
            freeze_backbone=model_params.get("freeze_backbone", False),
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        # For baseline CNN, all params train. For frozen mobilenet, only classifier trains.
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_params["learning_rate"]
        )

        print(f"Starting training loop... (run={run_name}, arch={arch})")
        for epoch in range(train_params["epochs"]):
            train_loss = train_one_epoch(
                model, loaders["train"], criterion, optimizer
            )
            val_acc, _ = evaluate(model, loaders["val"])

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        # Final test evaluation
        test_acc, cm = evaluate(model, loaders["test"])
        mlflow.log_metric("test_accuracy", test_acc)

        # Confusion matrix artifact
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {run_name}")

        cm_path = f"confusion_matrix_{run_name}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save model weights if requested
        if save_model:
            model_path = MODEL_DIR / save_filename
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path))
            print(f"Saved model to: {model_path}")

        print(f"{run_name} | Test Accuracy: {test_acc:.4f}")


# ======================================================
# Main
# ======================================================
def main(env="local"):
    train_params, model_params, data_params = load_params(env)

    mlflow.set_experiment("cats_vs_dogs_m1")

    # -------------------------------
    # BASELINE RUN (Simple CNN baseline)
    # -------------------------------
    baseline_cnn_params = train_params.copy()
    baseline_cnn_params["epochs"] = 2  # 1–2 epochs is enough for baseline requirement

    baseline_cnn_model_params = model_params.copy()
    baseline_cnn_model_params["arch"] = "baseline_cnn"
    baseline_cnn_model_params["freeze_backbone"] = False

    run_training(
        run_name="baseline_cnn",
        train_params=baseline_cnn_params,
        model_params=baseline_cnn_model_params,
        augment=False,
        save_model=True,
        save_filename="baseline_cnn.pt",
    )

    # -------------------------------
    # BASELINE RUN (Frozen backbone MobileNetV2)
    # -------------------------------
    baseline_params = train_params.copy()
    baseline_params["epochs"] = 3

    baseline_model_params = model_params.copy()
    baseline_model_params["arch"] = "mobilenet_v2"
    baseline_model_params["freeze_backbone"] = True

    run_training(
        run_name="baseline_frozen",
        train_params=baseline_params,
        model_params=baseline_model_params,
        augment=False,
        save_model=False,
    )

    # -------------------------------
    # FINAL RUN (Augmented + fine-tune MobileNetV2)
    # -------------------------------
    final_model_params = model_params.copy()
    final_model_params["arch"] = "mobilenet_v2"

    run_training(
        run_name="final_model",
        train_params=train_params,
        model_params=final_model_params,
        augment=True,
        save_model=True,
        save_filename="model_local.pt",
    )


if __name__ == "__main__":
    main(env="local")
