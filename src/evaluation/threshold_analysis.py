from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run threshold analysis for MURA abnormality classification."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone model to analyze.",
    )

    parser.add_argument(
        "--use-temp-scaled",
        action="store_true",
        help="Use temperature-scaled probabilities instead of vanilla probabilities.",
    )

    return parser.parse_args()


def compute_threshold_metrics(labels, probs, threshold):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sensitivity = recall_score(labels, preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def plot_metric_vs_threshold(df, metric, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.plot(df["threshold"], df[metric], marker="o")
    plt.xlabel("Decision threshold")
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(np.arange(0.05, 1.00, 0.10))
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sensitivity_specificity(df, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["threshold"],
        df["recall_sensitivity"],
        marker="o",
        label="Sensitivity / Recall",
    )
    plt.plot(
        df["threshold"],
        df["specificity"],
        marker="o",
        label="Specificity",
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title(title)
    plt.xticks(np.arange(0.05, 1.00, 0.10))
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_false_pos_false_neg(df, output_path, title):
    plt.figure(figsize=(8, 6))
    plt.plot(
        df["threshold"],
        df["false_positive"],
        marker="o",
        label="False positives",
    )
    plt.plot(
        df["threshold"],
        df["false_negative"],
        marker="o",
        label="False negatives",
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(np.arange(0.05, 1.00, 0.10))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_markdown_summary(df, output_path, backbone_name, probability_type):
    best_f1_row = df.loc[df["f1"].idxmax()]
    high_sensitivity_candidates = df[df["recall_sensitivity"] >= 0.90]

    with open(output_path, "w") as f:
        f.write(f"# Threshold Analysis: {backbone_name}\n\n")
        f.write(f"Probability type: `{probability_type}`\n\n")

        f.write("## Best F1 threshold\n\n")
        f.write(
            f"- Threshold: {best_f1_row['threshold']:.2f}\n"
            f"- F1: {best_f1_row['f1']:.4f}\n"
            f"- Accuracy: {best_f1_row['accuracy']:.4f}\n"
            f"- Sensitivity: {best_f1_row['recall_sensitivity']:.4f}\n"
            f"- Specificity: {best_f1_row['specificity']:.4f}\n"
            f"- False positives: {int(best_f1_row['false_positive'])}\n"
            f"- False negatives: {int(best_f1_row['false_negative'])}\n\n"
        )

        if not high_sensitivity_candidates.empty:
            # Among thresholds with sensitivity >= 0.90, choose highest specificity.
            high_sens_row = high_sensitivity_candidates.loc[
                high_sensitivity_candidates["specificity"].idxmax()
            ]

            f.write("## Highest-specificity threshold with sensitivity >= 0.90\n\n")
            f.write(
                f"- Threshold: {high_sens_row['threshold']:.2f}\n"
                f"- F1: {high_sens_row['f1']:.4f}\n"
                f"- Accuracy: {high_sens_row['accuracy']:.4f}\n"
                f"- Sensitivity: {high_sens_row['recall_sensitivity']:.4f}\n"
                f"- Specificity: {high_sens_row['specificity']:.4f}\n"
                f"- False positives: {int(high_sens_row['false_positive'])}\n"
                f"- False negatives: {int(high_sens_row['false_negative'])}\n\n"
            )
        else:
            f.write("## High-sensitivity threshold\n\n")
            f.write("No threshold in the sweep reached sensitivity >= 0.90.\n\n")

        f.write("## Full threshold table\n\n")
        f.write(df.round(4).to_markdown(index=False))
        f.write("\n")


def main():
    args = parse_args()

    backbone_name = args.backbone

    if args.use_temp_scaled:
        pred_path = Path(
            f"outputs/evaluation/{backbone_name}/temperature_scaling/"
            "temperature_scaled_predictions.csv"
        )
        prob_column = "temperature_scaled_prob_abnormal"
        output_dir = Path(
            f"outputs/evaluation/{backbone_name}/threshold_analysis_temperature_scaled"
        )
        probability_type = "temperature_scaled"
    else:
        pred_path = Path(f"outputs/evaluation/{backbone_name}/valid_predictions.csv")
        prob_column = "prob_abnormal"
        output_dir = Path(f"outputs/evaluation/{backbone_name}/threshold_analysis")
        probability_type = "vanilla"

    figure_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)

    if prob_column not in pred_df.columns:
        raise ValueError(
            f"Expected probability column `{prob_column}` not found in {pred_path}"
        )

    labels = pred_df["label"].values.astype(int)
    probs = pred_df[prob_column].values.astype(float)

    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    rows = []
    for threshold in thresholds:
        rows.append(compute_threshold_metrics(labels, probs, threshold))

    threshold_df = pd.DataFrame(rows)

    csv_path = output_dir / "threshold_metrics.csv"
    threshold_df.to_csv(csv_path, index=False)
    print(f"Saved threshold metrics: {csv_path}")

    json_path = output_dir / "threshold_metrics.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved threshold metrics: {json_path}")

    summary_path = output_dir / "threshold_summary.md"
    save_markdown_summary(
        threshold_df,
        summary_path,
        backbone_name=backbone_name,
        probability_type=probability_type,
    )
    print(f"Saved summary: {summary_path}")

    plot_sensitivity_specificity(
        threshold_df,
        figure_dir / "sensitivity_specificity_vs_threshold.png",
        f"{backbone_name.upper()} Sensitivity/Specificity vs Threshold",
    )

    plot_false_pos_false_neg(
        threshold_df,
        figure_dir / "false_positive_false_negative_vs_threshold.png",
        f"{backbone_name.upper()} FP/FN vs Threshold",
    )

    for metric in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1"]:
        plot_metric_vs_threshold(
            threshold_df,
            metric,
            figure_dir / f"{metric}_vs_threshold.png",
            f"{backbone_name.upper()} {metric} vs Threshold",
        )

    best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]

    print("\nBest F1 threshold:")
    print(best_f1_row.round(4).to_string())

    high_sensitivity_candidates = threshold_df[threshold_df["recall_sensitivity"] >= 0.90]
    if not high_sensitivity_candidates.empty:
        high_sens_row = high_sensitivity_candidates.loc[
            high_sensitivity_candidates["specificity"].idxmax()
        ]

        print("\nHighest-specificity threshold with sensitivity >= 0.90:")
        print(high_sens_row.round(4).to_string())
    else:
        print("\nNo threshold in the sweep reached sensitivity >= 0.90.")

    print(f"\nSaved figures to: {figure_dir}")


if __name__ == "__main__":
    main()