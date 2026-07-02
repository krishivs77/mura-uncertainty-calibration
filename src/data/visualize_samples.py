from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")
IMAGE_ROOT = Path("data/raw/MURA-v1.1")
FIGURE_DIR = Path("outputs/figures")


def load_image(relative_path):
    image_path = IMAGE_ROOT / relative_path

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    img = Image.open(image_path).convert("L")
    return img


def plot_sample_grid(df, title, output_name, n=16):
    sample_df = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    rows = 4
    cols = 4

    plt.figure(figsize=(12, 12))

    for i, row in sample_df.iterrows():
        img = load_image(row["relative_path"])

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        label = row["label_name"]
        body_part = row["body_part"]
        split = row["split"]

        ax.set_title(f"{label}\n{body_part}, {split}", fontsize=9)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    output_path = FIGURE_DIR / output_name
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST_PATH)

    # Verify that all manifest paths exist locally
    print("Checking local image paths...")
    df["exists"] = df["relative_path"].apply(lambda p: (IMAGE_ROOT / p).exists())

    missing_count = (~df["exists"]).sum()
    print(f"Missing images: {missing_count}")

    if missing_count > 0:
        print(df.loc[~df["exists"], "relative_path"].head(10).to_string(index=False))
        raise RuntimeError("Some images from the manifest are missing.")

    print("All manifest images found.")

    # Normal and abnormal grids
    plot_sample_grid(
        df[df["label_name"] == "normal"],
        title="Random Normal MURA X-rays",
        output_name="sample_normal_xrays.png",
    )

    plot_sample_grid(
        df[df["label_name"] == "abnormal"],
        title="Random Abnormal MURA X-rays",
        output_name="sample_abnormal_xrays.png",
    )

    # One grid with mixed body parts and labels
    plot_sample_grid(
        df,
        title="Random MURA X-rays",
        output_name="sample_mixed_xrays.png",
    )

    # One example per body part, normal and abnormal if possible
    selected_rows = []

    for body_part in sorted(df["body_part"].unique()):
        for label_name in ["normal", "abnormal"]:
            subset = df[(df["body_part"] == body_part) & (df["label_name"] == label_name)]
            if len(subset) > 0:
                selected_rows.append(subset.sample(1, random_state=42))

    body_label_df = pd.concat(selected_rows).reset_index(drop=True)

    plt.figure(figsize=(14, 10))

    rows = 4
    cols = 4

    for i, row in body_label_df.iterrows():
        img = load_image(row["relative_path"])

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        ax.set_title(f"{row['body_part']}\n{row['label_name']}", fontsize=9)

    plt.suptitle("Example MURA X-rays by Body Part and Label", fontsize=16)
    plt.tight_layout()

    output_path = FIGURE_DIR / "sample_by_body_part_and_label.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()