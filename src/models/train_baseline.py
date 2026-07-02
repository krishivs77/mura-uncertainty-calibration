from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.data.mura_dataset import MuraImageDataset


MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")
IMAGE_ROOT = Path("data/raw/MURA-v1.1")

CHECKPOINT_DIR = Path("outputs/checkpoints")
METRICS_DIR = Path("outputs/metrics")

backbone_name = "resnet50"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(backbone_name="resnet18"):
    if backbone_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    return model


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, valid_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels, _metadata in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = epoch_loss

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels, _metadata in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().reshape(-1))

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = epoch_loss

    return metrics


def compute_metrics(labels, probs):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
    }


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_transform, valid_transform = get_transforms()

    train_dataset = MuraImageDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="train",
        transform=train_transform,
    )

    valid_dataset = MuraImageDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="valid",
        transform=valid_transform,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(backbone_name=backbone_name).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 3
    best_val_auroc = -1.0

    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - start_time

        row = {
            "backbone": backbone_name,
            "epoch": epoch,
            "epoch_time_sec": epoch_time,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }

        history.append(row)

        print(
            f"train loss={train_metrics['loss']:.4f}, "
            f"train AUROC={train_metrics['auroc']:.4f}, "
            f"train F1={train_metrics['f1']:.4f}"
        )

        print(
            f"valid loss={valid_metrics['loss']:.4f}, "
            f"valid AUROC={valid_metrics['auroc']:.4f}, "
            f"valid F1={valid_metrics['f1']:.4f}, "
            f"valid acc={valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["auroc"] > best_val_auroc:
            best_val_auroc = valid_metrics["auroc"]

            checkpoint_path = CHECKPOINT_DIR / f"baseline_{backbone_name}_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "valid_auroc": best_val_auroc,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")

        history_df = pd.DataFrame(history)
        history_df.to_csv(METRICS_DIR / f"baseline_{backbone_name}_history.csv", index=False)

        with open(METRICS_DIR / f"baseline_{backbone_name}_latest.json", "w") as f:
            json.dump(row, f, indent=2)

    print("\nTraining complete.")
    print(f"Best valid AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()