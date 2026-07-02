from pathlib import Path

import pandas as pd


BACKBONES = ["resnet18", "resnet50"]
PROBABILITY_TYPES = ["vanilla", "temperature_scaled"]

OUTPUT_DIR = Path("outputs/reports")


def get_threshold_path(backbone, probability_type):
    if probability_type == "vanilla":
        return Path(f"outputs/evaluation/{backbone}/threshold_analysis/threshold_metrics.csv")

    if probability_type == "temperature_scaled":
        return Path(
            f"outputs/evaluation/{backbone}/threshold_analysis_temperature_scaled/"
            "threshold_metrics.csv"
        )

    raise ValueError(f"Unknown probability type: {probability_type}")


def load_threshold_df(backbone, probability_type):
    path = get_threshold_path(backbone, probability_type)

    if not path.exists():
        print(f"Warning: missing file: {path}")
        return None

    return pd.read_csv(path)


def get_best_f1_row(df):
    return df.loc[df["f1"].idxmax()]


def get_high_sensitivity_row(df, min_sensitivity=0.90):
    candidates = df[df["recall_sensitivity"] >= min_sensitivity]

    if candidates.empty:
        return None

    # Among thresholds that reach target sensitivity, choose the one with highest specificity.
    return candidates.loc[candidates["specificity"].idxmax()]


def get_default_threshold_row(df):
    subset = df[df["threshold"].round(4) == 0.50]

    if subset.empty:
        return None

    return subset.iloc[0]


def row_to_summary(backbone, probability_type, operating_point, row):
    if row is None:
        return {
            "backbone": backbone,
            "probability_type": probability_type,
            "operating_point": operating_point,
            "threshold": None,
            "accuracy": None,
            "precision": None,
            "recall_sensitivity": None,
            "specificity": None,
            "f1": None,
            "true_negative": None,
            "false_positive": None,
            "false_negative": None,
            "true_positive": None,
        }

    return {
        "backbone": backbone,
        "probability_type": probability_type,
        "operating_point": operating_point,
        "threshold": row["threshold"],
        "accuracy": row["accuracy"],
        "precision": row["precision"],
        "recall_sensitivity": row["recall_sensitivity"],
        "specificity": row["specificity"],
        "f1": row["f1"],
        "true_negative": row["true_negative"],
        "false_positive": row["false_positive"],
        "false_negative": row["false_negative"],
        "true_positive": row["true_positive"],
    }


def build_summary_rows():
    rows = []

    for backbone in BACKBONES:
        for probability_type in PROBABILITY_TYPES:
            df = load_threshold_df(backbone, probability_type)

            if df is None:
                continue

            default_row = get_default_threshold_row(df)
            best_f1_row = get_best_f1_row(df)
            high_sens_row = get_high_sensitivity_row(df, min_sensitivity=0.90)

            rows.append(
                row_to_summary(
                    backbone=backbone,
                    probability_type=probability_type,
                    operating_point="default_0.50",
                    row=default_row,
                )
            )

            rows.append(
                row_to_summary(
                    backbone=backbone,
                    probability_type=probability_type,
                    operating_point="best_f1",
                    row=best_f1_row,
                )
            )

            rows.append(
                row_to_summary(
                    backbone=backbone,
                    probability_type=probability_type,
                    operating_point="sensitivity_ge_0.90_best_specificity",
                    row=high_sens_row,
                )
            )

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
        f.write("# Threshold Summary\n\n")
        f.write(markdown)
        f.write("\n")

    print(f"Saved markdown table: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_summary_rows()
    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "threshold_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    rounded_df = round_numeric_columns(df, digits=4)

    md_path = OUTPUT_DIR / "threshold_summary.md"
    save_markdown_table(rounded_df, md_path)

    print("\nThreshold summary:")
    print(rounded_df.to_string(index=False))


if __name__ == "__main__":
    main()