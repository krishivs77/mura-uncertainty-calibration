from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibration analysis for a MURA model.")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone model to analyze.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of confidence bins for ECE/reliability analysis.",
    )

    return parser.parse_args()


def compute_ece(labels, probs, n_bins=10):
    labels = np.array(labels).astype(int)
    probs = np.array(probs)

    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        bin_count = int(in_bin.sum())

        if bin_count > 0:
            bin_confidence = float(confidences[in_bin].mean())
            bin_accuracy = float(correct[in_bin].mean())
            bin_gap = abs(bin_accuracy - bin_confidence)
            bin_weight = bin_count / len(labels)
            ece += bin_weight * bin_gap
        else:
            bin_confidence = None
            bin_accuracy = None
            bin_gap = None
            bin_weight = 0.0

        bin_rows.append({
            "bin": i,
            "lower": float(lower),
            "upper": float(upper),
            "count": bin_count,
            "weight": float(bin_weight),
            "accuracy": bin_accuracy,
            "confidence": bin_confidence,
            "abs_gap": bin_gap,
        })

    return float(ece), pd.DataFrame(bin_rows)


def plot_reliability_diagram(bin_df, output_path, title):
    plot_df = bin_df.dropna(subset=["accuracy", "confidence"])

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.scatter(plot_df["confidence"], plot_df["accuracy"], s=80, label="Model bins")

    for _, row in plot_df.iterrows():
        plt.text(
            row["confidence"],
            row["accuracy"],
            str(int(row["count"])),
            fontsize=8,
            ha="center",
            va="bottom",
        )

    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confidence_histogram(confidences, correct, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.hist(confidences[correct == 1], bins=20, alpha=0.7, label="Correct")
    plt.hist(confidences[correct == 0], bins=20, alpha=0.7, label="Incorrect")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of samples")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_accuracy_by_confidence_bin(bin_df, output_path, title):
    plot_df = bin_df.dropna(subset=["accuracy", "confidence"]).copy()
    plot_df["bin_center"] = (plot_df["lower"] + plot_df["upper"]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(plot_df["bin_center"], plot_df["accuracy"], width=0.08, label="Accuracy")
    plt.plot(plot_df["bin_center"], plot_df["confidence"], marker="o", label="Confidence")

    plt.xlabel("Confidence bin")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()

    backbone_name = args.backbone
    pred_path = Path(f"outputs/evaluation/{backbone_name}/valid_predictions.csv")
    output_dir = Path(f"outputs/evaluation/{backbone_name}/calibration")
    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n"
            f"Run evaluate_baseline.py first:\n"
            f"python -m src.evaluation.evaluate_baseline --backbone {backbone_name}"
        )

    pred_df = pd.read_csv(pred_path)

    labels = pred_df["label"].values.astype(int)
    probs = pred_df["prob_abnormal"].values.astype(float)
    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(int)

    ece, bin_df = compute_ece(labels, probs, n_bins=args.n_bins)

    metrics = {
        f"ece_{args.n_bins}_bins": float(ece),
        "brier_score": float(brier_score_loss(labels, probs)),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=[0, 1])),
        "mean_confidence": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[correct == 1].mean()),
        "mean_confidence_incorrect": float(confidences[correct == 0].mean()),
        "overconfidence_gap": float(confidences.mean() - correct.mean()),
        "num_samples": int(len(labels)),
        "num_correct": int(correct.sum()),
        "num_incorrect": int((1 - correct).sum()),
    }

    print(json.dumps(metrics, indent=2))

    metrics_path = output_dir / "calibration_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    bin_path = output_dir / "calibration_bins.csv"
    bin_df.to_csv(bin_path, index=False)
    print(f"Saved bins: {bin_path}")

    plot_reliability_diagram(
        bin_df,
        figure_dir / "reliability_diagram.png",
        f"{backbone_name.upper()} Reliability Diagram",
    )

    plot_confidence_histogram(
        confidences,
        correct,
        figure_dir / "confidence_histogram.png",
        f"{backbone_name.upper()} Confidence Histogram",
    )

    plot_accuracy_by_confidence_bin(
        bin_df,
        figure_dir / "accuracy_by_confidence_bin.png",
        f"{backbone_name.upper()} Accuracy by Confidence Bin",
    )

    print(f"Saved figures to: {figure_dir}")


if __name__ == "__main__":
    main()