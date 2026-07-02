from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")
IMAGE_ROOT = Path("data/raw/MURA-v1.1")


CORRUPTIONS = [
    "clean",
    "gaussian_noise",
    "blur",
    "brightness_down",
    "brightness_up",
    "contrast_down",
    "contrast_up",
]

SEVERITY_LEVELS = [0, 1, 2, 3, 4]


def parse_args():
    parser = argparse.ArgumentParser(description="Run corruption stress tests for a MURA model.")

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
        "--max-samples",
        type=int,
        default=None,
        help="Optional subset size for fast debugging. Use None/full command for full validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for optional subset sampling.",
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


def apply_corruption(img, corruption, severity):
    """
    Applies a deterministic corruption type at severity 0-4.

    img: PIL RGB image.
    corruption: corruption name.
    severity: integer from 0 to 4.
    """
    if corruption == "clean" or severity == 0:
        return img

    if corruption == "gaussian_noise":
        arr = np.array(img).astype(np.float32) / 255.0

        sigma_map = {
            1: 0.03,
            2: 0.06,
            3: 0.10,
            4: 0.15,
        }

        sigma = sigma_map[severity]
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1)
        arr = (arr * 255).astype(np.uint8)

        return Image.fromarray(arr)

    if corruption == "blur":
        radius_map = {
            1: 0.75,
            2: 1.5,
            3: 2.5,
            4: 4.0,
        }

        return img.filter(ImageFilter.GaussianBlur(radius=radius_map[severity]))

    if corruption == "brightness_down":
        factor_map = {
            1: 0.85,
            2: 0.70,
            3: 0.55,
            4: 0.40,
        }

        return ImageEnhance.Brightness(img).enhance(factor_map[severity])

    if corruption == "brightness_up":
        factor_map = {
            1: 1.15,
            2: 1.35,
            3: 1.60,
            4: 2.00,
        }

        return ImageEnhance.Brightness(img).enhance(factor_map[severity])

    if corruption == "contrast_down":
        factor_map = {
            1: 0.85,
            2: 0.70,
            3: 0.50,
            4: 0.30,
        }

        return ImageEnhance.Contrast(img).enhance(factor_map[severity])

    if corruption == "contrast_up":
        factor_map = {
            1: 1.15,
            2: 1.35,
            3: 1.70,
            4: 2.20,
        }

        return ImageEnhance.Contrast(img).enhance(factor_map[severity])

    raise ValueError(f"Unknown corruption: {corruption}")


class CorruptedMuraDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        image_root,
        split,
        transform,
        corruption="clean",
        severity=0,
        max_samples=None,
        seed=42,
    ):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            self.df = self.df.sample(
                n=min(max_samples, len(self.df)),
                random_state=seed,
            ).reset_index(drop=True)

        self.image_root = Path(image_root)
        self.transform = transform
        self.corruption = corruption
        self.severity = severity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row["relative_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img = apply_corruption(img, self.corruption, self.severity)

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(row["label"], dtype=torch.float32)

        metadata = {
            "relative_path": row["relative_path"],
            "body_part": row["body_part"],
            "patient_id": row["patient_id"],
            "study_uid": row["study_uid"],
            "label_name": row["label_name"],
        }

        return img, label, metadata


def expected_calibration_error(labels, probs, n_bins=10):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correctness = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_accuracy = np.mean(correctness[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += prop_in_bin * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    preds = (probs >= 0.5).astype(int)

    confidences = np.maximum(probs, 1 - probs)
    correct = preds == labels

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "ece_10_bins": float(expected_calibration_error(labels, probs, n_bins=10)),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=[0, 1])),
        "mean_confidence": float(np.mean(confidences)),
        "mean_confidence_correct": float(np.mean(confidences[correct])) if np.any(correct) else None,
        "mean_confidence_incorrect": float(np.mean(confidences[~correct])) if np.any(~correct) else None,
        "overconfidence_gap": float(np.mean(confidences) - accuracy_score(labels, preds)),
        "num_samples": int(len(labels)),
        "num_correct": int(np.sum(correct)),
        "num_incorrect": int(np.sum(~correct)),
    }

    return metrics, preds


@torch.no_grad()
def evaluate_condition(
    model,
    device,
    transform,
    corruption,
    severity,
    batch_size,
    max_samples=None,
    seed=42,
):
    dataset = CorruptedMuraDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="valid",
        transform=transform,
        corruption=corruption,
        severity=severity,
        max_samples=max_samples,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_probs = []
    all_labels = []

    for images, labels, _metadata in tqdm(
        loader,
        desc=f"{corruption} severity={severity}",
        leave=False,
    ):
        images = images.to(device)

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        all_probs.extend(probs)
        all_labels.extend(labels.numpy().reshape(-1))

    metrics, _preds = compute_metrics(all_labels, all_probs)

    metrics["corruption"] = corruption
    metrics["severity"] = int(severity)

    return metrics


def plot_metric_vs_severity(results_df, metric, output_path, backbone_name):
    plt.figure(figsize=(9, 6))

    for corruption in results_df["corruption"].unique():
        subset = results_df[results_df["corruption"] == corruption].sort_values("severity")
        plt.plot(
            subset["severity"],
            subset[metric],
            marker="o",
            label=corruption,
        )

    plt.xlabel("Corruption severity")
    plt.ylabel(metric)
    plt.title(f"{backbone_name.upper()} {metric} vs Corruption Severity")
    plt.xticks(SEVERITY_LEVELS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_corruption_examples(figure_dir):
    """
    Saves visual examples of one image across corruptions/severities.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)
    sample = df[df["split"] == "valid"].sample(1, random_state=7).iloc[0]
    img_path = IMAGE_ROOT / sample["relative_path"]
    img = Image.open(img_path).convert("RGB")

    selected_corruptions = [
        "clean",
        "gaussian_noise",
        "blur",
        "brightness_down",
        "contrast_down",
    ]

    plt.figure(figsize=(14, 8))

    plot_idx = 1
    for corruption in selected_corruptions:
        for severity in [0, 2, 4]:
            if corruption == "clean" and severity != 0:
                continue

            corrupted = apply_corruption(img, corruption, severity)

            ax = plt.subplot(len(selected_corruptions), 3, plot_idx)
            ax.imshow(corrupted, cmap="gray")
            ax.axis("off")
            ax.set_title(f"{corruption}\nseverity={severity}", fontsize=9)

            plot_idx += 1

    plt.suptitle("Example MURA Corruptions", fontsize=16)
    plt.tight_layout()

    output_path = figure_dir / "corruption_examples.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    backbone_name = args.backbone
    checkpoint_path = Path(f"outputs/checkpoints/baseline_{backbone_name}_best.pt")
    output_dir = Path(f"outputs/evaluation/{backbone_name}/stress_tests")
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Backbone: {backbone_name}")
    print(f"Checkpoint: {checkpoint_path}")

    if args.max_samples is not None:
        print(f"Using max_samples={args.max_samples} for debugging")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model = build_model(backbone_name).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_results = []

    for corruption in CORRUPTIONS:
        for severity in SEVERITY_LEVELS:
            if corruption == "clean" and severity != 0:
                continue

            if corruption != "clean" and severity == 0:
                continue

            metrics = evaluate_condition(
                model=model,
                device=device,
                transform=transform,
                corruption=corruption,
                severity=severity,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                seed=args.seed,
            )

            all_results.append(metrics)

            print(
                f"{corruption:16s} severity={severity} "
                f"acc={metrics['accuracy']:.4f} "
                f"auroc={metrics['auroc']:.4f} "
                f"ece={metrics['ece_10_bins']:.4f} "
                f"conf={metrics['mean_confidence']:.4f}"
            )

    results_df = pd.DataFrame(all_results)

    results_path = output_dir / "corruption_stress_test_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")

    json_path = output_dir / "corruption_stress_test_metrics.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results: {json_path}")

    for metric in [
        "accuracy",
        "auroc",
        "f1",
        "brier_score",
        "ece_10_bins",
        "mean_confidence",
        "mean_confidence_incorrect",
        "overconfidence_gap",
    ]:
        plot_metric_vs_severity(
            results_df,
            metric,
            figure_dir / f"{metric}_vs_corruption_severity.png",
            backbone_name=backbone_name,
        )

    save_corruption_examples(figure_dir)

    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()