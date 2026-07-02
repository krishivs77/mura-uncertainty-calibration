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


AGGREGATION_METHODS = ["mean", "max", "top2_mean"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn study-level temperature scaling for MURA predictions."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["resnet18", "resnet50"],
        help="Backbone model to calibrate.",
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="top2_mean",
        choices=AGGREGATION_METHODS,
        help="Study-level aggregation method.",
    )

    parser.add_argument(
        "--min-temp",
        type=float,
        default=0.5,
        help="Minimum temperature for grid search.",
    )

    parser.add_argument(
        "--max-temp",
        type=float,
        default=5.0,
        help="Maximum temperature for grid search.",
    )

    parser.add_argument(
        "--num-temps",
        type=int,
        default=451,
        help="Number of temperatures to test.",
    )

    return parser.parse_args()


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))


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


def aggregate_probs(probs, method):
    probs = np.asarray(probs, dtype=float)

    if method == "mean":
        return float(np.mean(probs))

    if method == "max":
        return float(np.max(probs))

    if method == "top2_mean":
        return top_k_mean(probs, k=2)

    raise ValueError(f"Unknown aggregation method: {method}")


def build_study_predictions(pred_df, temperature, aggregation_method):
    rows = []

    for (patient_id, study_uid), group in pred_df.groupby(["patient_id", "study_uid"]):
        labels = group["label"].astype(int).unique()

        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent labels for patient={patient_id}, study={study_uid}: {labels}"
            )

        logits = group["logit"].astype(float).values
        probs = sigmoid(logits / temperature)
        study_prob = aggregate_probs(probs, aggregation_method)

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


def find_best_temperature(pred_df, aggregation_method, min_temp, max_temp, num_temps):
    temperatures = np.linspace(min_temp, max_temp, num_temps)

    rows = []

    for temperature in temperatures:
        study_df = build_study_predictions(
            pred_df=pred_df,
            temperature=temperature,
            aggregation_method=aggregation_method,
        )

        labels = study_df["label"].values
        probs = study_df["study_prob_abnormal"].values
        probs = np.clip(probs, 1e-7, 1 - 1e-7)

        nll = log_loss(labels, probs, labels=[0, 1])
        brier = brier_score_loss(labels, probs)
        ece = expected_calibration_error(labels, probs, n_bins=10)

        rows.append({
            "temperature": float(temperature),
            "negative_log_likelihood": float(nll),
            "brier_score": float(brier),
            "ece_10_bins": float(ece),
        })

    search_df = pd.DataFrame(rows)

    # Choose temperature by NLL, which is the standard objective for temp scaling.
    best_row = search_df.loc[search_df["negative_log_likelihood"].idxmin()]

    return float(best_row["temperature"]), search_df


def main():
    args = parse_args()

    backbone = args.backbone
    aggregation_method = args.aggregation

    pred_path = Path(f"outputs/evaluation/{backbone}/valid_predictions.csv")

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)

    if "logit" not in pred_df.columns:
        raise ValueError(
            f"{pred_path} does not contain a `logit` column. "
            "Re-run evaluate_baseline.py first."
        )

    output_dir = Path(
        f"outputs/evaluation/{backbone}/study_level_temperature_scaling/{aggregation_method}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    best_temperature, search_df = find_best_temperature(
        pred_df=pred_df,
        aggregation_method=aggregation_method,
        min_temp=args.min_temp,
        max_temp=args.max_temp,
        num_temps=args.num_temps,
    )

    print(f"Best study-level temperature: {best_temperature:.4f}")

    search_path = output_dir / "temperature_grid_search.csv"
    search_df.to_csv(search_path, index=False)
    print(f"Saved temperature search: {search_path}")

    vanilla_study_df = build_study_predictions(
        pred_df=pred_df,
        temperature=1.0,
        aggregation_method=aggregation_method,
    )

    scaled_study_df = build_study_predictions(
        pred_df=pred_df,
        temperature=best_temperature,
        aggregation_method=aggregation_method,
    )

    vanilla_metrics = compute_metrics(
        labels=vanilla_study_df["label"].values,
        probs=vanilla_study_df["study_prob_abnormal"].values,
    )

    scaled_metrics = compute_metrics(
        labels=scaled_study_df["label"].values,
        probs=scaled_study_df["study_prob_abnormal"].values,
    )

    results = {
        "backbone": backbone,
        "aggregation_method": aggregation_method,
        "study_level_temperature": best_temperature,
        "vanilla": vanilla_metrics,
        "study_temperature_scaled": scaled_metrics,
    }

    print(json.dumps(results, indent=2))

    metrics_path = output_dir / "study_temperature_scaling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    vanilla_study_df.to_csv(output_dir / "study_predictions_vanilla.csv", index=False)
    scaled_study_df.to_csv(
        output_dir / "study_predictions_temperature_scaled.csv",
        index=False,
    )

    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "backbone": backbone,
        "aggregation_method": aggregation_method,
        "study_level_temperature": best_temperature,
        "vanilla_accuracy": vanilla_metrics["accuracy"],
        "scaled_accuracy": scaled_metrics["accuracy"],
        "vanilla_auroc": vanilla_metrics["auroc"],
        "scaled_auroc": scaled_metrics["auroc"],
        "vanilla_f1": vanilla_metrics["f1"],
        "scaled_f1": scaled_metrics["f1"],
        "vanilla_ece_10_bins": vanilla_metrics["ece_10_bins"],
        "scaled_ece_10_bins": scaled_metrics["ece_10_bins"],
        "vanilla_brier_score": vanilla_metrics["brier_score"],
        "scaled_brier_score": scaled_metrics["brier_score"],
        "vanilla_nll": vanilla_metrics["negative_log_likelihood"],
        "scaled_nll": scaled_metrics["negative_log_likelihood"],
        "vanilla_overconfidence_gap": vanilla_metrics["overconfidence_gap"],
        "scaled_overconfidence_gap": scaled_metrics["overconfidence_gap"],
    }

    summary_path = reports_dir / (
        f"study_level_temperature_scaling_{backbone}_{aggregation_method}.csv"
    )
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()