from pathlib import Path
import json

import pandas as pd


BACKBONES = ["resnet18", "resnet50"]
OUTPUT_DIR = Path("outputs/reports")


def load_json(path):
    if not path.exists():
        print(f"Warning: missing file: {path}")
        return None

    with open(path, "r") as f:
        return json.load(f)


def safe_get(dictionary, key, default=None):
    if dictionary is None:
        return default
    return dictionary.get(key, default)


def build_model_row(backbone):
    eval_dir = Path(f"outputs/evaluation/{backbone}")

    clean_metrics_path = eval_dir / "valid_metrics.json"
    calibration_metrics_path = eval_dir / "calibration" / "calibration_metrics.json"
    temperature_metrics_path = (
        eval_dir / "temperature_scaling" / "temperature_scaling_metrics.json"
    )

    clean_metrics = load_json(clean_metrics_path)
    calibration_metrics = load_json(calibration_metrics_path)
    temperature_metrics = load_json(temperature_metrics_path)

    vanilla_temp_metrics = None
    temp_scaled_metrics = None

    if temperature_metrics is not None:
        vanilla_temp_metrics = temperature_metrics.get("vanilla", {})
        temp_scaled_metrics = temperature_metrics.get("temperature_scaled", {})

    row = {
        "backbone": backbone,

        # Clean classification metrics
        "accuracy": safe_get(clean_metrics, "accuracy"),
        "auroc": safe_get(clean_metrics, "auroc"),
        "precision": safe_get(clean_metrics, "precision"),
        "recall_sensitivity": safe_get(clean_metrics, "recall_sensitivity"),
        "specificity": safe_get(clean_metrics, "specificity"),
        "f1": safe_get(clean_metrics, "f1"),
        "true_negative": safe_get(clean_metrics, "true_negative"),
        "false_positive": safe_get(clean_metrics, "false_positive"),
        "false_negative": safe_get(clean_metrics, "false_negative"),
        "true_positive": safe_get(clean_metrics, "true_positive"),

        # Vanilla calibration metrics
        "ece_10_bins": safe_get(calibration_metrics, "ece_10_bins"),
        "brier_score": safe_get(calibration_metrics, "brier_score"),
        "negative_log_likelihood": safe_get(
            calibration_metrics,
            "negative_log_likelihood",
        ),
        "mean_confidence": safe_get(calibration_metrics, "mean_confidence"),
        "mean_confidence_correct": safe_get(
            calibration_metrics,
            "mean_confidence_correct",
        ),
        "mean_confidence_incorrect": safe_get(
            calibration_metrics,
            "mean_confidence_incorrect",
        ),
        "overconfidence_gap": safe_get(calibration_metrics, "overconfidence_gap"),

        # Temperature scaling
        "temperature": safe_get(temperature_metrics, "temperature"),

        # Temp-scaled metrics
        "temp_scaled_accuracy": safe_get(temp_scaled_metrics, "accuracy"),
        "temp_scaled_auroc": safe_get(temp_scaled_metrics, "auroc"),
        "temp_scaled_f1": safe_get(temp_scaled_metrics, "f1"),
        "temp_scaled_ece_10_bins": safe_get(temp_scaled_metrics, "ece_10_bins"),
        "temp_scaled_brier_score": safe_get(temp_scaled_metrics, "brier_score"),
        "temp_scaled_negative_log_likelihood": safe_get(
            temp_scaled_metrics,
            "negative_log_likelihood",
        ),
        "temp_scaled_mean_confidence": safe_get(
            temp_scaled_metrics,
            "mean_confidence",
        ),
        "temp_scaled_overconfidence_gap": safe_get(
            temp_scaled_metrics,
            "overconfidence_gap",
        ),
    }

    return row


def round_numeric_columns(df, digits=4):
    rounded = df.copy()

    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(digits)

    return rounded


def save_markdown_table(df, path):
    markdown = df.to_markdown(index=False)

    with open(path, "w") as f:
        f.write("# Model Comparison\n\n")
        f.write(markdown)
        f.write("\n")

    print(f"Saved markdown table: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [build_model_row(backbone) for backbone in BACKBONES]
    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    rounded_df = round_numeric_columns(df, digits=4)

    md_path = OUTPUT_DIR / "model_comparison.md"
    save_markdown_table(rounded_df, md_path)

    print("\nModel comparison:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()