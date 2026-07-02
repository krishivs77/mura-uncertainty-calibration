from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


IMAGE_ROOT = Path("data/raw/MURA-v1.1")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize representative success and failure cases."
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="Backbone model to visualize.",
    )

    parser.add_argument(
        "--use-temp-scaled",
        action="store_true",
        help="Use temperature-scaled probabilities.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of examples per panel.",
    )

    return parser.parse_args()


def get_paths(backbone, use_temp_scaled):
    if use_temp_scaled:
        pred_path = Path(
            f"outputs/evaluation/{backbone}/temperature_scaling/"
            "temperature_scaled_predictions.csv"
        )
        prob_col = "temperature_scaled_prob_abnormal"
        pred_col = "temperature_scaled_pred"
        output_dir = Path(
            f"outputs/evaluation/{backbone}/failure_cases_temperature_scaled"
        )
        probability_type = "temperature_scaled"
    else:
        pred_path = Path(f"outputs/evaluation/{backbone}/valid_predictions.csv")
        prob_col = "prob_abnormal"
        pred_col = "pred"
        output_dir = Path(f"outputs/evaluation/{backbone}/failure_cases")
        probability_type = "vanilla"

    return pred_path, prob_col, pred_col, output_dir, probability_type


def add_confidence_columns(df, prob_col, pred_col):
    df = df.copy()

    df["prob_abnormal_used"] = df[prob_col].astype(float)
    df["pred_used"] = df[pred_col].astype(int)

    df["confidence"] = np.maximum(
        df["prob_abnormal_used"],
        1 - df["prob_abnormal_used"],
    )

    df["correct_used"] = df["pred_used"] == df["label"]

    return df


def sample_top(df, condition, num_examples):
    subset = df[condition].copy()

    if subset.empty:
        return subset

    return subset.sort_values("confidence", ascending=False).head(num_examples)


def sample_uncertain(df, condition, num_examples):
    subset = df[condition].copy()

    if subset.empty:
        return subset

    subset["uncertainty_distance"] = np.abs(subset["prob_abnormal_used"] - 0.5)
    return subset.sort_values("uncertainty_distance", ascending=True).head(num_examples)


def load_image(relative_path):
    img_path = IMAGE_ROOT / relative_path

    if not img_path.exists():
        raise FileNotFoundError(f"Missing image: {img_path}")

    return Image.open(img_path).convert("L")


def plot_panel(rows, title, output_path, num_examples):
    if rows.empty:
        print(f"Skipping empty panel: {title}")
        return

    n = min(len(rows), num_examples)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))

    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, rows.head(n).iterrows()):
        img = load_image(row["relative_path"])

        ax.imshow(img, cmap="gray")
        ax.axis("off")

        true_name = "abnormal" if int(row["label"]) == 1 else "normal"
        pred_name = "abnormal" if int(row["pred_used"]) == 1 else "normal"

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {pred_name}\n"
            f"p(abn): {row['prob_abnormal_used']:.3f}\n"
            f"conf: {row['confidence']:.3f}",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def save_case_csv(rows, output_path):
    rows.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    args = parse_args()

    pred_path, prob_col, pred_col, output_dir, probability_type = get_paths(
        args.backbone,
        args.use_temp_scaled,
    )

    figure_dir = output_dir / "figures"
    csv_dir = output_dir / "case_csvs"

    figure_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_df = pd.read_csv(pred_path)

    if prob_col not in pred_df.columns:
        raise ValueError(f"Missing probability column: {prob_col}")

    if pred_col not in pred_df.columns:
        raise ValueError(f"Missing prediction column: {pred_col}")

    df = add_confidence_columns(pred_df, prob_col, pred_col)

    cases = {
        "confident_correct_normal": sample_top(
            df,
            (df["label"] == 0) & (df["pred_used"] == 0),
            args.num_examples,
        ),
        "confident_correct_abnormal": sample_top(
            df,
            (df["label"] == 1) & (df["pred_used"] == 1),
            args.num_examples,
        ),
        "confident_false_negative": sample_top(
            df,
            (df["label"] == 1) & (df["pred_used"] == 0),
            args.num_examples,
        ),
        "confident_false_positive": sample_top(
            df,
            (df["label"] == 0) & (df["pred_used"] == 1),
            args.num_examples,
        ),
        "uncertain_correct": sample_uncertain(
            df,
            df["correct_used"] == True,
            args.num_examples,
        ),
        "uncertain_wrong": sample_uncertain(
            df,
            df["correct_used"] == False,
            args.num_examples,
        ),
    }

    for case_name, rows in cases.items():
        title = (
            f"{args.backbone.upper()} {probability_type.replace('_', ' ')}: "
            f"{case_name.replace('_', ' ')}"
        )

        plot_panel(
            rows,
            title=title,
            output_path=figure_dir / f"{case_name}.png",
            num_examples=args.num_examples,
        )

        save_case_csv(
            rows,
            output_path=csv_dir / f"{case_name}.csv",
        )

    print(f"\nSaved failure-case figures to: {figure_dir}")
    print(f"Saved case CSVs to: {csv_dir}")


if __name__ == "__main__":
    main()