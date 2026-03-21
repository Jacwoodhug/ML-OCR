"""PyTorch Dataset wrapping the synthetic generator with augmentation and collation."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.alphabet import encode
from src.data.augmentations import apply_augmentation, get_augmentation_pipeline
from src.data.synth_generator import SynthGenerator


class SynthOCRDataset(Dataset):
    """Infinite-length synthetic OCR dataset for training.

    Each __getitem__ call generates a fresh synthetic sample on the fly.
    The dataset length is set to a configurable value to define an "epoch"
    for the DataLoader, but data is always freshly generated.
    """

    def __init__(
        self,
        generator: SynthGenerator,
        epoch_size: int = 100_000,
        augment: bool = True,
        aug_config: dict | None = None,
    ):
        self.generator = generator
        self.epoch_size = epoch_size
        self.augment = augment

        if augment:
            cfg = aug_config or {}
            self.aug_pipeline = get_augmentation_pipeline(**cfg)
        else:
            self.aug_pipeline = None

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _idx: int) -> dict:
        """Generate a single sample.

        Returns dict with:
            - image: (C, H, W) float32 tensor, normalized to [0, 1]
            - target: 1D int64 tensor of encoded character indices
            - target_length: int, length of target sequence
            - text: str, ground truth text
        """
        img, text = self.generator.generate()

        # Convert to numpy HWC uint8
        img_np = np.array(img, dtype=np.uint8)

        # Apply augmentation
        if self.augment and self.aug_pipeline is not None:
            img_np = apply_augmentation(img_np, self.aug_pipeline)

        # Convert to tensor: HWC uint8 -> CHW float32 [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # Encode text
        target = encode(text)

        return {
            "image": img_tensor,
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "text": text,
        }


class PregenOCRDataset(Dataset):
    """Pre-generated dataset loaded from disk.

    Expects a directory with:
        - images/ containing .png files named 000000.png, 000001.png, ...
        - labels.txt with one ground-truth text per line

    When augment=True, augmentations are applied at load time (for training).
    """

    def __init__(self, data_dir: str, augment: bool = False, aug_config: dict | None = None):
        from PIL import Image

        self.data_dir = data_dir
        labels_path = os.path.join(data_dir, "labels.txt")
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.rstrip("\n") for line in f]

        self.image_dir = os.path.join(data_dir, "images")
        self.augment = augment
        if augment:
            cfg = aug_config or {}
            self.aug_pipeline = get_augmentation_pipeline(**cfg)
        else:
            self.aug_pipeline = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        base = os.path.join(self.image_dir, f"{idx:06d}")
        img_path = base + ".jpg" if os.path.exists(base + ".jpg") else base + ".png"
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8)

        if self.augment and self.aug_pipeline is not None:
            img_np = apply_augmentation(img_np, self.aug_pipeline)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        text = self.labels[idx]
        target = encode(text)

        return {
            "image": img_tensor,
            "target": torch.tensor(target, dtype=torch.long),
            "target_length": len(target),
            "text": text,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for variable-width images.

    Pads all images in the batch to the maximum width with zeros.
    Packs targets into a single 1D tensor for CTC loss.
    """
    images = [item["image"] for item in batch]
    targets = [item["target"] for item in batch]
    target_lengths = [item["target_length"] for item in batch]
    texts = [item["text"] for item in batch]

    # Pad images to max width in this batch
    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        # img shape: (C, H, W)
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            padding = torch.zeros(img.shape[0], img.shape[1], pad_w, dtype=img.dtype)
            img = torch.cat([img, padding], dim=2)
        padded.append(img)

    image_batch = torch.stack(padded, dim=0)  # (B, C, H, W)

    # Concatenate all targets into a single 1D tensor (CTC format)
    target_concat = torch.cat(targets, dim=0)  # (sum of target_lengths,)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    # Input lengths: number of timesteps per image after backbone
    # This depends on the model stride — computed externally, but we store image widths
    image_widths = torch.tensor([img.shape[2] for img in images], dtype=torch.long)

    return {
        "images": image_batch,
        "targets": target_concat,
        "target_lengths": target_lengths_tensor,
        "image_widths": image_widths,
        "texts": texts,
    }
