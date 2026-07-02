from pathlib import Path

import pandas as pd


BACKBONES = ["resnet18", "resnet50"]

SELECTED_CONDITIONS = [
    ("clean", 0),
    ("gaussian_noise", 4),
    ("blur", 4),
    ("brightness_up", 4),
    ("contrast_up", 4),
]


OUTPUT_DIR = Path("outputs/reports")


def load_csv(path):
    if not path.exists():
        print(f"Warning: missing file: {path}")
        return None

    return pd.read_csv(path)


def get_condition_row(df, corruption, severity):
    subset = df[
        (df["corruption"] == corruption)
        & (df["severity"] == severity)
    ]

    if subset.empty:
        return None

    return subset.iloc[0]


def build_rows_for_backbone(backbone):
    stress_path = Path(
        f"outputs/evaluation/{backbone}/stress_tests/corruption_stress_test_metrics.csv"
    )
    temp_stress_path = Path(
        f"outputs/evaluation/{backbone}/stress_tests_temperature_scaled/"
        "temperature_scaled_corruption_metrics.csv"
    )

    stress_df = load_csv(stress_path)
    temp_stress_df = load_csv(temp_stress_path)

    rows = []

    for corruption, severity in SELECTED_CONDITIONS:
        stress_row = (
            get_condition_row(stress_df, corruption, severity)
            if stress_df is not None
            else None
        )

        temp_row = (
            get_condition_row(temp_stress_df, corruption, severity)
            if temp_stress_df is not None
            else None
        )

        if stress_row is None and temp_row is None:
            continue

        row = {
            "backbone": backbone,
            "corruption": corruption,
            "severity": severity,
        }

        if stress_row is not None:
            row.update({
                "accuracy": stress_row.get("accuracy"),
                "auroc": stress_row.get("auroc"),
                "f1": stress_row.get("f1"),
                "ece_10_bins": stress_row.get("ece_10_bins"),
                "brier_score": stress_row.get("brier_score"),
                "nll": stress_row.get("negative_log_likelihood"),
                "mean_confidence": stress_row.get("mean_confidence"),
                "overconfidence_gap": stress_row.get("overconfidence_gap"),
            })

        if temp_row is not None:
            row.update({
                "vanilla_ece_from_temp_run": temp_row.get("vanilla_ece_10_bins"),
                "temp_scaled_ece_10_bins": temp_row.get("temp_scaled_ece_10_bins"),
                "temp_scaled_brier_score": temp_row.get("temp_scaled_brier_score"),
                "temp_scaled_nll": temp_row.get("temp_scaled_negative_log_likelihood"),
                "temp_scaled_mean_confidence": temp_row.get("temp_scaled_mean_confidence"),
                "temp_scaled_overconfidence_gap": temp_row.get(
                    "temp_scaled_overconfidence_gap"
                ),
                "ece_delta_temp_minus_vanilla": temp_row.get(
                    "ece_delta_temp_minus_vanilla"
                ),
                "brier_delta_temp_minus_vanilla": temp_row.get(
                    "brier_delta_temp_minus_vanilla"
                ),
                "nll_delta_temp_minus_vanilla": temp_row.get(
                    "nll_delta_temp_minus_vanilla"
                ),
            })

        rows.append(row)

    return rows


def round_numeric_columns(df, digits=4):
    rounded = df.copy()

    for col in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(digits)

    return rounded


def save_markdown_table(df, path):
    markdown = df.to_markdown(index=False)

    with open(path, "w") as f:
        f.write("# Robustness Summary\n\n")
        f.write(markdown)
        f.write("\n")

    print(f"Saved markdown table: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for backbone in BACKBONES:
        rows.extend(build_rows_for_backbone(backbone))

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "robustness_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    rounded_df = round_numeric_columns(df, digits=4)

    md_path = OUTPUT_DIR / "robustness_summary.md"
    save_markdown_table(rounded_df, md_path)

    print("\nRobustness summary:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()