from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("outputs/evaluation/resnet18/stress_tests/corruption_stress_test_metrics.csv")
FIGURE_DIR = Path("outputs/evaluation/resnet18/stress_tests/figures_cleaned")

METRICS_TO_PLOT = [
    "accuracy",
    "auroc",
    "f1",
    "brier_score",
    "ece_10_bins",
    "mean_confidence",
    "mean_confidence_incorrect",
    "overconfidence_gap",
]


def build_plot_df(results_df):
    """
    Convert:
      clean severity 0 as one row
      corruption severity 1-4 rows

    Into:
      every corruption gets the same clean row at severity 0
    """
    clean_row = results_df[results_df["corruption"] == "clean"].iloc[0]

    corruption_names = [
        c for c in results_df["corruption"].unique()
        if c != "clean"
    ]

    plot_rows = []

    for corruption in corruption_names:
        clean_copy = clean_row.copy()
        clean_copy["corruption"] = corruption
        clean_copy["severity"] = 0
        plot_rows.append(clean_copy)

        corruption_rows = results_df[results_df["corruption"] == corruption]
        for _, row in corruption_rows.iterrows():
            plot_rows.append(row)

    return pd.DataFrame(plot_rows)


def add_overconfidence_gap(results_df):
    if "overconfidence_gap" not in results_df.columns:
        results_df["overconfidence_gap"] = (
            results_df["mean_confidence"] - results_df["accuracy"]
        )
    return results_df


def plot_metric(plot_df, metric):
    plt.figure(figsize=(9, 6))

    for corruption in sorted(plot_df["corruption"].unique()):
        subset = plot_df[plot_df["corruption"] == corruption].sort_values("severity")

        plt.plot(
            subset["severity"],
            subset[metric],
            marker="o",
            label=corruption,
        )

    plt.xlabel("Corruption severity")
    plt.ylabel(metric)
    plt.title(f"{metric} vs corruption severity")
    plt.xticks([0, 1, 2, 3, 4])
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / f"{metric}_vs_corruption_severity_cleaned.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(RESULTS_PATH)
    results_df = add_overconfidence_gap(results_df)

    # Save updated CSV with overconfidence_gap included
    updated_path = RESULTS_PATH.with_name("corruption_stress_test_metrics_with_gap.csv")
    results_df.to_csv(updated_path, index=False)
    print(f"Saved: {updated_path}")

    plot_df = build_plot_df(results_df)

    for metric in METRICS_TO_PLOT:
        if metric in plot_df.columns:
            plot_metric(plot_df, metric)


if __name__ == "__main__":
    main()