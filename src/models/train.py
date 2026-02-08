import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data/processed")
PARAMS_FILE = "params.yaml"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helper functions
# -----------------------------
def load_params(env: str):
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)
    return params[env], params["model"], params["data"]


def get_dataloaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    datasets_map = {
        split: datasets.ImageFolder(DATA_DIR / split, transform=transform)
        for split in ["train", "val", "test"]
    }

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


def build_model(num_classes, freeze_backbone=True):
    model = models.mobilenet_v2(pretrained=True)

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


# -----------------------------
# Main training logic
# -----------------------------
def main(env="local"):
    train_params, model_params, data_params = load_params(env)

    mlflow.set_experiment("cats_vs_dogs_m1")

    with mlflow.start_run(run_name=f"baseline_{env}"):

        # Log parameters
        mlflow.log_params({
            "env": env,
            **train_params,
            **model_params,
            **data_params
        })

        loaders, class_names = get_dataloaders(
            train_params["batch_size"],
            train_params["num_workers"]
        )

        model = build_model(
            num_classes=len(class_names),
            freeze_backbone=model_params["freeze_backbone"]
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_params["learning_rate"]
        )

        val_accuracies = []

        for epoch in range(train_params["epochs"]):
            train_loss = train_one_epoch(
                model, loaders["train"], criterion, optimizer
            )
            val_acc, _ = evaluate(model, loaders["val"])
            val_accuracies.append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch [{epoch+1}/{train_params['epochs']}], "
                f"Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        # Final evaluation
        test_acc, cm = evaluate(model, loaders["test"])
        mlflow.log_metric("test_accuracy", test_acc)

        # Save confusion matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # Save model
        model_path = MODEL_DIR / f"model_{env}.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        print(f"\nTraining complete. Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main(env="local")
