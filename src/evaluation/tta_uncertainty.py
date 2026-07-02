from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models
import torchvision.transforms.functional as TF

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


BACKBONES = ["resnet18", "resnet50"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test-time augmentation uncertainty for MURA classifiers."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=BACKBONES,
        help="Backbone architecture.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained checkpoint.",
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/manifests/mura_manifest.csv",
        help="Path to MURA manifest CSV.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw/MURA-v1.1",
        help="Root directory containing the MURA-v1.1 dataset.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Split to evaluate. Usually valid.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(backbone):
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
        ]:
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]

        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]

        cleaned[new_key] = value

    return cleaned


def load_model(backbone, checkpoint_path, device):
    model = build_model(backbone)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


class MuraImageDataset(Dataset):
    def __init__(self, manifest_path, split, data_root, max_samples=None):
        self.df = pd.read_csv(manifest_path)
        self.data_root = Path(data_root)

        if "split" not in self.df.columns:
            raise ValueError("Manifest must contain a `split` column.")

        self.df = self.df[self.df["split"] == split].copy()

        if max_samples is not None:
            self.df = self.df.head(max_samples).copy()

        self.df = self.df.reset_index(drop=True)

        possible_image_path_cols = [
            "image_path",
            "full_path",
            "path",
            "filepath",
            "file_path",
            "relative_path",
            "filename",
            "image_file",
        ]

        self.image_path_col = None

        for col in possible_image_path_cols:
            if col in self.df.columns:
                self.image_path_col = col
                break

        if self.image_path_col is None:
            raise ValueError(
                "Manifest must contain one image path column. "
                f"Tried: {possible_image_path_cols}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

        required_cols = [
            "label",
            "label_name",
            "body_part",
            "patient_id",
            "study_uid",
        ]

        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}. "
                f"Available columns: {self.df.columns.tolist()}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_path = Path(str(row[self.image_path_col]))

        if raw_path.is_absolute() and raw_path.exists():
            image_path = raw_path

        elif raw_path.exists():
            image_path = raw_path

        elif (self.data_root / raw_path).exists():
            # Handles manifest paths like:
            # valid/XR_SHOULDER/patient11770/study1_negative/image1.png
            image_path = self.data_root / raw_path

        else:
            # Handles manifests that only store image1.png.
            split = str(row["split"])
            body_part = str(row["body_part"])
            patient_id = str(row["patient_id"])
            study_uid = str(row["study_uid"])

            image_path = (
                self.data_root
                / split
                / body_part
                / patient_id
                / study_uid
                / raw_path.name
            )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Could not find image. Tried: {image_path}. "
                f"Original manifest value: {raw_path}"
            )

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "image_path": str(image_path),
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "body_part": row["body_part"],
            "patient_id": row["patient_id"],
            "study_uid": row["study_uid"],
        }


def collate_fn(batch):
    return batch


def normalize_tensor(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor - mean) / std


def preprocess_pil(image):
    image = TF.resize(image, [224, 224])
    tensor = TF.to_tensor(image)
    tensor = normalize_tensor(tensor)
    return tensor


def build_tta_views(image):
    """
    Deterministic, anatomy-safe TTA views.

    We avoid horizontal flips because laterality/anatomical orientation can matter.
    These transforms are mild acquisition-style perturbations.
    """
    base = TF.resize(image, [224, 224])

    views = []

    views.append(base)

    views.append(TF.adjust_brightness(base, 0.90))
    views.append(TF.adjust_brightness(base, 1.10))

    views.append(TF.adjust_contrast(base, 0.90))
    views.append(TF.adjust_contrast(base, 1.10))

    views.append(TF.rotate(base, angle=-5, fill=0))
    views.append(TF.rotate(base, angle=5, fill=0))

    # Slight center crop and resize back.
    cropped = TF.center_crop(base, output_size=[210, 210])
    cropped = TF.resize(cropped, [224, 224])
    views.append(cropped)

    tensors = [normalize_tensor(TF.to_tensor(view)) for view in views]
    return torch.stack(tensors, dim=0)


def binary_entropy(prob):
    prob = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(-(prob * math.log(prob) + (1 - prob) * math.log(1 - prob)))


def expected_calibration_error(labels, probs, n_bins=10):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        prop = np.mean(in_bin)

        if prop > 0:
            bin_acc = np.mean(correct[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += prop * abs(bin_acc - bin_conf)

    return float(ece)


def compute_classification_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    confidences = np.maximum(probs, 1 - probs)
    correct = preds == labels

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=[0, 1])),
        "ece_10_bins": float(expected_calibration_error(labels, probs, n_bins=10)),
        "mean_confidence": float(confidences.mean()),
        "overconfidence_gap": float(confidences.mean() - correct.mean()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "num_samples": int(len(labels)),
    }


def compute_uncertainty_metrics(df):
    labels = df["label"].astype(int).values
    probs = df["tta_mean_prob_abnormal"].astype(float).values
    preds = (probs >= 0.5).astype(int)
    errors = (preds != labels).astype(int)

    uncertainty_columns = [
        "tta_prob_std",
        "tta_entropy",
        "tta_margin_uncertainty",
    ]

    metrics = {
        "num_samples": int(len(df)),
        "num_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
    }

    for col in uncertainty_columns:
        values = df[col].astype(float).values

        correct_values = values[errors == 0]
        incorrect_values = values[errors == 1]

        metrics[f"{col}_mean_correct"] = (
            float(correct_values.mean()) if len(correct_values) > 0 else None
        )
        metrics[f"{col}_mean_incorrect"] = (
            float(incorrect_values.mean()) if len(incorrect_values) > 0 else None
        )
        metrics[f"{col}_gap_incorrect_minus_correct"] = (
            float(incorrect_values.mean() - correct_values.mean())
            if len(correct_values) > 0 and len(incorrect_values) > 0
            else None
        )

        if len(np.unique(errors)) == 2:
            metrics[f"{col}_error_detection_auroc"] = float(
                roc_auc_score(errors, values)
            )
            metrics[f"{col}_error_detection_average_precision"] = float(
                average_precision_score(errors, values)
            )
        else:
            metrics[f"{col}_error_detection_auroc"] = None
            metrics[f"{col}_error_detection_average_precision"] = None

    return metrics


@torch.no_grad()
def evaluate_tta(model, dataloader, device):
    rows = []

    for batch in dataloader:
        for sample in batch:
            image = sample["image"]

            tta_tensor = build_tta_views(image).to(device)

            logits = model(tta_tensor).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            mean_prob = float(np.mean(probs))
            std_prob = float(np.std(probs))
            min_prob = float(np.min(probs))
            max_prob = float(np.max(probs))

            pred = int(mean_prob >= 0.5)
            label = int(sample["label"])
            correct = int(pred == label)

            confidence = float(max(mean_prob, 1 - mean_prob))
            margin_uncertainty = float(1 - abs(mean_prob - 0.5) * 2)

            rows.append({
                "image_path": sample["image_path"],
                "patient_id": sample["patient_id"],
                "study_uid": sample["study_uid"],
                "body_part": sample["body_part"],
                "label": label,
                "label_name": sample["label_name"],
                "tta_mean_prob_abnormal": mean_prob,
                "tta_prob_std": std_prob,
                "tta_prob_min": min_prob,
                "tta_prob_max": max_prob,
                "tta_prob_range": max_prob - min_prob,
                "tta_entropy": binary_entropy(mean_prob),
                "tta_margin_uncertainty": margin_uncertainty,
                "prediction": pred,
                "confidence": confidence,
                "correct": correct,
                "num_tta_views": int(len(probs)),
            })

    return pd.DataFrame(rows)


def round_numeric_columns(df, digits=4):
    rounded = df.copy()

    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(digits)

    return rounded


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = load_model(
        backbone=args.backbone,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    dataset = MuraImageDataset(
        manifest_path=args.manifest_path,
        split=args.split,
        data_root=args.data_root,
        max_samples=args.max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Evaluating {len(dataset)} images with TTA.")

    tta_df = evaluate_tta(model, dataloader, device)

    output_dir = Path(f"outputs/evaluation/{args.backbone}/tta_uncertainty")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "tta_predictions.csv"
    tta_df.to_csv(predictions_path, index=False)
    print(f"Saved TTA predictions: {predictions_path}")

    labels = tta_df["label"].values
    probs = tta_df["tta_mean_prob_abnormal"].values

    classification_metrics = compute_classification_metrics(labels, probs)
    uncertainty_metrics = compute_uncertainty_metrics(tta_df)

    metrics = {
        "backbone": args.backbone,
        "split": args.split,
        "checkpoint_path": args.checkpoint_path,
        "num_tta_views": int(tta_df["num_tta_views"].iloc[0]),
        "classification_metrics": classification_metrics,
        "uncertainty_metrics": uncertainty_metrics,
    }

    metrics_path = output_dir / "tta_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved TTA metrics: {metrics_path}")

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "backbone": args.backbone,
        "split": args.split,
        "num_tta_views": int(tta_df["num_tta_views"].iloc[0]),
        "accuracy": classification_metrics["accuracy"],
        "auroc": classification_metrics["auroc"],
        "f1": classification_metrics["f1"],
        "sensitivity": classification_metrics["recall_sensitivity"],
        "specificity": classification_metrics["specificity"],
        "ece_10_bins": classification_metrics["ece_10_bins"],
        "brier_score": classification_metrics["brier_score"],
        "nll": classification_metrics["negative_log_likelihood"],
        "mean_confidence": classification_metrics["mean_confidence"],
        "overconfidence_gap": classification_metrics["overconfidence_gap"],
        "error_rate": uncertainty_metrics["error_rate"],
        "std_mean_correct": uncertainty_metrics["tta_prob_std_mean_correct"],
        "std_mean_incorrect": uncertainty_metrics["tta_prob_std_mean_incorrect"],
        "std_gap_incorrect_minus_correct": uncertainty_metrics[
            "tta_prob_std_gap_incorrect_minus_correct"
        ],
        "std_error_detection_auroc": uncertainty_metrics[
            "tta_prob_std_error_detection_auroc"
        ],
        "entropy_mean_correct": uncertainty_metrics["tta_entropy_mean_correct"],
        "entropy_mean_incorrect": uncertainty_metrics["tta_entropy_mean_incorrect"],
        "entropy_gap_incorrect_minus_correct": uncertainty_metrics[
            "tta_entropy_gap_incorrect_minus_correct"
        ],
        "entropy_error_detection_auroc": uncertainty_metrics[
            "tta_entropy_error_detection_auroc"
        ],
        "margin_uncertainty_mean_correct": uncertainty_metrics[
            "tta_margin_uncertainty_mean_correct"
        ],
        "margin_uncertainty_mean_incorrect": uncertainty_metrics[
            "tta_margin_uncertainty_mean_incorrect"
        ],
        "margin_uncertainty_gap_incorrect_minus_correct": uncertainty_metrics[
            "tta_margin_uncertainty_gap_incorrect_minus_correct"
        ],
        "margin_uncertainty_error_detection_auroc": uncertainty_metrics[
            "tta_margin_uncertainty_error_detection_auroc"
        ],
    }

    summary_df = pd.DataFrame([summary_row])

    summary_csv_path = reports_dir / f"tta_uncertainty_{args.backbone}.csv"
    summary_md_path = reports_dir / f"tta_uncertainty_{args.backbone}.md"

    summary_df.to_csv(summary_csv_path, index=False)

    rounded = round_numeric_columns(summary_df, digits=4)

    with open(summary_md_path, "w") as f:
        f.write(f"# TTA Uncertainty Summary: {args.backbone}\n\n")
        f.write(rounded.to_markdown(index=False))
        f.write("\n")

    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary markdown: {summary_md_path}")

    print("\nTTA summary:")
    print(rounded.to_string(index=False))


if __name__ == "__main__":
    main()