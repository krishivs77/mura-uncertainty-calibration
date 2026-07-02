from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description="Apply temperature scaling to a MURA model.")

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone model to calibrate.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for ECE calculation.",
    )

    return parser.parse_args()


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def compute_ece(labels, probs, n_bins=10):
    labels = np.array(labels).astype(int)
    probs = np.array(probs).astype(float)

    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if in_bin.sum() > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_weight = in_bin.mean()
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(labels, probs, n_bins=10):
    labels = np.array(labels).astype(int)
    probs = np.array(probs).astype(float)

    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=[0, 1])),
        f"ece_{n_bins}_bins": float(compute_ece(labels, probs, n_bins=n_bins)),
        "mean_confidence": float(confidences.mean()),
        "mean_confidence_correct": float(confidences[correct == 1].mean()),
        "mean_confidence_incorrect": float(confidences[correct == 0].mean()),
        "overconfidence_gap": float(confidences.mean() - correct.mean()),
    }


def learn_temperature(logits, labels):
    logits_tensor = torch.tensor(logits, dtype=torch.float32).view(-1, 1)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    temperature = torch.nn.Parameter(torch.ones(1))

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits_tensor / temperature.clamp(min=1e-6)
        loss = criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    return float(temperature.detach().item())


def main():
    args = parse_args()

    backbone_name = args.backbone
    pred_path = Path(f"outputs/evaluation/{backbone_name}/valid_predictions.csv")
    output_dir = Path(f"outputs/evaluation/{backbone_name}/temperature_scaling")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}\n"
            f"Run evaluation first:\n"
            f"python -m src.evaluation.evaluate_baseline --backbone {backbone_name}"
        )

    pred_df = pd.read_csv(pred_path)

    if "logit" not in pred_df.columns:
        raise ValueError(
            f"{pred_path} does not contain a 'logit' column. "
            "Re-run evaluate_baseline.py with the updated version that saves logits."
        )

    labels = pred_df["label"].values.astype(int)
    logits = pred_df["logit"].values.astype(float)

    vanilla_probs = sigmoid_np(logits)

    temperature = learn_temperature(logits, labels)

    temp_scaled_logits = logits / temperature
    temp_scaled_probs = sigmoid_np(temp_scaled_logits)

    vanilla_metrics = compute_metrics(labels, vanilla_probs, n_bins=args.n_bins)
    temp_scaled_metrics = compute_metrics(labels, temp_scaled_probs, n_bins=args.n_bins)

    results = {
        "backbone": backbone_name,
        "temperature": temperature,
        "vanilla": vanilla_metrics,
        "temperature_scaled": temp_scaled_metrics,
    }

    print(json.dumps(results, indent=2))

    metrics_path = output_dir / "temperature_scaling_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics: {metrics_path}")

    output_df = pred_df.copy()
    output_df["vanilla_prob_abnormal"] = vanilla_probs
    output_df["temperature_scaled_logit"] = temp_scaled_logits
    output_df["temperature_scaled_prob_abnormal"] = temp_scaled_probs
    output_df["temperature_scaled_pred"] = (temp_scaled_probs >= 0.5).astype(int)
    output_df["temperature_scaled_correct"] = (
        output_df["temperature_scaled_pred"] == output_df["label"]
    )

    pred_out_path = output_dir / "temperature_scaled_predictions.csv"
    output_df.to_csv(pred_out_path, index=False)

    print(f"Saved predictions: {pred_out_path}")


if __name__ == "__main__":
    main()