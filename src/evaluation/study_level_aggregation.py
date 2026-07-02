from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


BACKBONES = ["resnet18", "resnet50"]
AGGREGATION_METHODS = ["mean", "max", "top2_mean"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate image-level MURA predictions to study-level predictions."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=BACKBONES,
        help="Optional single backbone to process. If omitted, processes all backbones.",
    )

    parser.add_argument(
        "--use-temp-scaled",
        action="store_true",
        help="Use temperature-scaled probabilities.",
    )

    return parser.parse_args()


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
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += prop * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(labels, probs, threshold=0.5):
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
        "num_studies": int(len(labels)),
    }


def top_k_mean(values, k=2):
    values = np.asarray(values, dtype=float)

    if len(values) <= k:
        return float(values.mean())

    return float(np.sort(values)[-k:].mean())


def aggregate_studies(pred_df, prob_col, method):
    group_cols = ["patient_id", "study_uid"]

    rows = []

    for (patient_id, study_uid), group in pred_df.groupby(group_cols):
        labels = group["label"].astype(int).unique()

        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent labels for patient={patient_id}, study={study_uid}: {labels}"
            )

        probs = group[prob_col].astype(float).values

        if method == "mean":
            study_prob = float(np.mean(probs))
        elif method == "max":
            study_prob = float(np.max(probs))
        elif method == "top2_mean":
            study_prob = top_k_mean(probs, k=2)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        rows.append({
            "patient_id": patient_id,
            "study_uid": study_uid,
            "body_part": group["body_part"].iloc[0],
            "label": int(labels[0]),
            "label_name": group["label_name"].iloc[0],
            "num_images": int(len(group)),
            "study_prob_abnormal": study_prob,
        })

    return pd.DataFrame(rows)


def get_prediction_file(backbone, use_temp_scaled):
    if use_temp_scaled:
        return (
            Path(f"outputs/evaluation/{backbone}/temperature_scaling/temperature_scaled_predictions.csv"),
            "temperature_scaled_prob_abnormal",
            "temperature_scaled",
        )

    return (
        Path(f"outputs/evaluation/{backbone}/valid_predictions.csv"),
        "prob_abnormal",
        "vanilla",
    )


def process_backbone(backbone, use_temp_scaled):
    pred_path, prob_col, probability_type = get_prediction_file(backbone, use_temp_scaled)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)

    if prob_col not in pred_df.columns:
        raise ValueError(f"Missing probability column `{prob_col}` in {pred_path}")

    output_dir = Path(f"outputs/evaluation/{backbone}/study_level_{probability_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for method in AGGREGATION_METHODS:
        study_df = aggregate_studies(pred_df, prob_col=prob_col, method=method)

        metrics = compute_metrics(
            labels=study_df["label"].values,
            probs=study_df["study_prob_abnormal"].values,
            threshold=0.5,
        )

        metrics["backbone"] = backbone
        metrics["probability_type"] = probability_type
        metrics["aggregation_method"] = method

        summary_rows.append(metrics)

        study_preds_path = output_dir / f"study_predictions_{method}.csv"
        study_df.to_csv(study_preds_path, index=False)
        print(f"Saved study predictions: {study_preds_path}")

        metrics_path = output_dir / f"study_metrics_{method}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved study metrics: {metrics_path}")

    return summary_rows


def round_numeric_columns(df, digits=4):
    rounded = df.copy()

    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(digits)

    return rounded


def main():
    args = parse_args()

    backbones = [args.backbone] if args.backbone is not None else BACKBONES

    all_rows = []

    for backbone in backbones:
        rows = process_backbone(backbone, use_temp_scaled=args.use_temp_scaled)
        all_rows.extend(rows)

    probability_type = "temperature_scaled" if args.use_temp_scaled else "vanilla"

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(all_rows)

    csv_path = reports_dir / f"study_level_summary_{probability_type}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV: {csv_path}")

    rounded_df = round_numeric_columns(summary_df, digits=4)

    md_path = reports_dir / f"study_level_summary_{probability_type}.md"
    with open(md_path, "w") as f:
        f.write(f"# Study-Level Aggregation Summary: {probability_type}\n\n")
        f.write(rounded_df.to_markdown(index=False))
        f.write("\n")

    print(f"Saved summary markdown: {md_path}")

    print("\nStudy-level summary:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()