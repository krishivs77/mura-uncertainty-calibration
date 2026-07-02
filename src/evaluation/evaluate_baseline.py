from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.data.mura_dataset import MuraImageDataset


MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")
IMAGE_ROOT = Path("data/raw/MURA-v1.1")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MURA baseline model.")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone model to evaluate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional custom checkpoint path. If omitted, uses outputs/checkpoints/baseline_<backbone>_best.pt.",
    )

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(backbone_name):
    if backbone_name == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def compute_metrics(labels, probs):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)
    preds = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, preds


@torch.no_grad()
def main():
    args = parse_args()

    backbone_name = args.backbone
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(f"outputs/checkpoints/baseline_{backbone_name}_best.pt")
    )

    output_dir = Path(f"outputs/evaluation/{backbone_name}")
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Backbone: {backbone_name}")
    print(f"Checkpoint: {checkpoint_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = MuraImageDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="valid",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(backbone_name).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_logits = []
    all_probs = []
    all_labels = []
    all_metadata = []

    for images, labels, metadata in tqdm(loader, desc="Evaluating"):
        images = images.to(device)

        logits = model(images)
        logits_np = logits.cpu().numpy().reshape(-1)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        all_logits.extend(logits_np)
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().reshape(-1))

        batch_size = len(labels)
        for i in range(batch_size):
            all_metadata.append({
                "relative_path": metadata["relative_path"][i],
                "body_part": metadata["body_part"][i],
                "patient_id": metadata["patient_id"][i],
                "study_uid": metadata["study_uid"][i],
                "label_name": metadata["label_name"][i],
            })

    assert len(all_labels) == len(all_logits) == len(all_probs) == len(all_metadata), (
        len(all_labels),
        len(all_logits),
        len(all_probs),
        len(all_metadata),
    )

    metrics, preds = compute_metrics(all_labels, all_probs)

    print(json.dumps(metrics, indent=2))

    pred_df = pd.DataFrame(all_metadata)
    pred_df["label"] = np.array(all_labels).astype(int)
    pred_df["logit"] = np.array(all_logits)
    pred_df["prob_abnormal"] = np.array(all_probs)
    pred_df["pred"] = preds
    pred_df["pred_label_name"] = np.where(pred_df["pred"] == 1, "abnormal", "normal")
    pred_df["correct"] = pred_df["pred"] == pred_df["label"]

    pred_path = output_dir / "valid_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions: {pred_path}")

    metrics_path = output_dir / "valid_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["normal", "abnormal"],
    )

    disp.plot(values_format="d")
    plt.title(f"{backbone_name.upper()} Confusion Matrix on MURA Validation Set")
    plt.tight_layout()

    cm_path = figure_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {metrics['auroc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title(f"{backbone_name.upper()} ROC Curve on MURA Validation Set")
    plt.legend()
    plt.tight_layout()

    roc_path = figure_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved: {roc_path}")


if __name__ == "__main__":
    main()