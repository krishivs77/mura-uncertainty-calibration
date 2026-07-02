from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIGURE_DIR = Path("outputs/figures")
MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")


def save_bar_plot(counts, title, xlabel, ylabel, output_name, rotation=0):
    plt.figure(figsize=(8,5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()

    output_path = FIGURE_DIR / output_name
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    print("Manifest shape:", df.shape)
    print()
    print(df.head())
    print()

    # 1. Label distribution
    label_counts = df["label_name"].value_counts()
    save_bar_plot(
        label_counts,
        title="MURA Image Label Distribution",
        xlabel="Label",
        ylabel="Number of Images",
        output_name="label_distribution.png",
    )

    # 2. Split distribution
    split_counts = df["split"].value_counts()
    save_bar_plot(
        split_counts,
        title="MURA Split Distribution",
        xlabel="Split",
        ylabel="Number of Images",
        output_name="split_distribution.png",
    )

    # 3. Body part distribution
    body_counts = df["body_part"].value_counts()
    save_bar_plot(
        body_counts,
        title="MURA Body Part Distribution",
        xlabel="Body Part",
        ylabel="Number of Images",
        output_name="body_part_distribution.png",
        rotation=45,
    )

    # 4. Label distribution by body part
    label_by_body = pd.crosstab(df["body_part"], df["label_name"])
    label_by_body = label_by_body.loc[body_counts.index]

    plt.figure(figsize=(10,6))
    label_by_body.plot(kind="bar", figsize=(10,6))
    plt.title("Normal vs Abnormal Images by Body Part")
    plt.xlabel("Body Part")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "label_by_body_part.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

    # 5. Images per study
    images_per_study = df.groupby("study_uid").size()

    plt.figure(figsize=(8,5))
    images_per_study.hist(bins=30)
    plt.title("Number of Images per Study")
    plt.xlabel("Images per Study")
    plt.ylabel("Number of Studies")
    plt.tight_layout()

    output_path = FIGURE_DIR / "images_per_study_distribution.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")

    # Useful printed summaries
    print()
    print("Images per study summary:")
    print(images_per_study.describe())
    print()

    print("Label distribution by split:")
    print(pd.crosstab(df["split"], df["label_name"]))
    print()

    print("Label distribution by body part:")
    print(label_by_body)


if __name__ == "__main__":
    main()