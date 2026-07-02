from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from mura_dataset import MuraImageDataset


MANIFEST_PATH = Path("data/manifests/mura_manifest.csv")
IMAGE_ROOT = Path("data/raw/MURA-v1.1")
FIGURE_DIR = Path("outputs/figures")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = MuraImageDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="train",
        transform=transform,
    )

    valid_dataset = MuraImageDataset(
        manifest_path=MANIFEST_PATH,
        image_root=IMAGE_ROOT,
        split="valid",
        transform=transform,
    )

    print("Train samples:", len(train_dataset))
    print("Valid samples:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    images, labels, metadata = next(iter(train_loader))

    print("Batch image tensor shape:", images.shape)
    print("Batch label tensor shape:", labels.shape)
    print("Labels:", labels.tolist())
    print("Body parts:", metadata["body_part"][:5])
    print("Study UIDs:", metadata["study_uid"][:2])

    # Unnormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    plt.figure(figsize=(12, 12))

    for i in range(min(16, images.size(0))):
        img = images[i].cpu() * std + mean
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0)

        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(img)
        ax.axis("off")

        label_name = metadata["label_name"][i]
        body_part = metadata["body_part"][i]

        ax.set_title(f"{label_name}\n{body_part}", fontsize=9)

    plt.suptitle("PyTorch DataLoader Batch Sanity Check", fontsize=16)
    plt.tight_layout()

    output_path = FIGURE_DIR / "dataloader_batch_sanity_check.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()