from pathlib import Path
import pandas as pd


def parse_mura_path(file_name: str) -> dict:
    """
    Example path:
    train/XR_ELBOW/patient00011/study1_negative/image1.png
    """
    parts = Path(file_name).parts

    if len(parts) < 5:
        raise ValueError(f"Unexpected path format: {file_name}")
    
    split = parts[0]
    body_part = parts[1]
    patient_id = parts[2]
    study_folder = parts[3]
    image_file = parts[4]

    if "positive" in study_folder:
        label = 1
        label_name = "abnormal"
    elif "negative" in study_folder:
        label = 0
        label_name = "normal"
    else:
        raise ValueError(f"Could not infer label from: {file_name}")
    
    study_id = study_folder.split("_")[0]

    return {
        "file_name": file_name,
        "split": split,
        "body_part": body_part,
        "patient_id": patient_id,
        "study_folder": study_folder,
        "study_id": study_id,
        "image_file": image_file,
        "label": label,
        "label_name": label_name,
    }


def main():
    metadata_path = Path("data/metadata/mura_v1_1.csv")
    output_path = Path("data/manifests/mura_manifest.csv")

    df = pd.read_csv(metadata_path)

    if "file_name" not in df.columns:
        raise ValueError(f"Expected column 'file_name'. Found: {df.columns.tolist()}")

    print("Original metadata rows:", len(df))

    # Keep only real image files
    image_extensions = (".png", ".jpg", ".jpeg")
    df["file_name"] = df["file_name"].astype(str)

    df = df[df["file_name"].str.lower().str.endswith(image_extensions)].copy()

    # Exclude macOS AppleDouble/resource-fork files like ._image3.png
    def is_real_image_path(path_str):
        parts = Path(path_str).parts
        return not any(
            part.startswith("._") or part == ".DS_Store"
            for part in parts
        )

    df = df[df["file_name"].apply(is_real_image_path)].copy()

    print("Image rows after filtering real images:", len(df))

    print("Image rows after filtering:", len(df))

    parsed_rows = []
    skipped = []

    for file_name in df["file_name"]:
        try:
            parsed_rows.append(parse_mura_path(file_name))
        except ValueError as e:
            skipped.append(str(e))

    if skipped:
        print()
        print(f"Skipped {len(skipped)} unexpected image paths.")
        print("First few skipped examples:")
        for item in skipped[:10]:
            print("  ", item)

    manifest = pd.DataFrame(parsed_rows)

    # Add useful IDs
    manifest["study_uid"] = (
        manifest["split"] + "/" +
        manifest["body_part"] + "/" +
        manifest["patient_id"] + "/" +
        manifest["study_folder"]
    )

    manifest["relative_path"] = manifest["file_name"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print()
    print(f"Saved manifest to: {output_path}")
    print()
    print("Shape:", manifest.shape)
    print()
    print("Splits:")
    print(manifest["split"].value_counts())
    print()
    print("Labels:")
    print(manifest["label_name"].value_counts())
    print()
    print("Body parts:")
    print(manifest["body_part"].value_counts())
    print()
    print("Unique studies:", manifest["study_uid"].nunique())
    print("Unique patients:", manifest["patient_id"].nunique())
    

if __name__ == "__main__":
    main()