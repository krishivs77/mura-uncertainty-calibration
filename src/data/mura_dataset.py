from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MuraImageDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        image_root,
        split,
        transform=None,
        body_part=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_root = Path(image_root)
        self.transform = transform

        self.df = pd.read_csv(self.manifest_path)

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        if body_part is not None:
            self.df = self.df[self.df["body_part"] == body_part].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split={split}, body_part={body_part}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_root / row["relative_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        metadata = {
            "relative_path": row["relative_path"],
            "body_part": row["body_part"],
            "patient_id": row["patient_id"],
            "study_uid": row["study_uid"],
            "label_name": row["label_name"],
        }

        return image, label, metadata