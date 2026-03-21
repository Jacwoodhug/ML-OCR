"""Training entry point for the OCR model.

Usage:
    python scripts/train.py [--config config/default.yaml] [--resume checkpoints/step_XXXXX.pt]
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alphabet import NUM_CLASSES
from src.data.dataset import LMDBOCRDataset, PregenOCRDataset, SynthOCRDataset, collate_fn
from src.data.synth_generator import SynthGenerator
from src.model.crnn import CRNN
from src.training.trainer import Trainer


def generate_validation_set(generator: SynthGenerator, val_dir: str, val_size: int):
    """Pre-generate a fixed validation set for consistent evaluation."""
    images_dir = os.path.join(val_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_path = os.path.join(val_dir, "labels.txt")

    # Check if already generated
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if existing >= val_size:
            print(f"Validation set already exists ({existing} samples)")
            return

    print(f"Generating {val_size} validation samples...")
    labels = []
    for i in range(val_size):
        img, text = generator.generate()
        img.save(os.path.join(images_dir, f"{i:06d}.png"))
        labels.append(text)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{val_size}")

    with open(labels_path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")

    print(f"Saved validation set to {val_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train OCR model")
    parser.add_argument("--config", default="config/default.yaml", help="Config file path")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--steps", type=int, default=None, help="Override max training iterations")
    parser.add_argument("--data-dir", default=None, help="Pre-generated training data directory (skips on-the-fly generation)")
    parser.add_argument("--lmdb", default=None, help="LMDB training dataset path (faster than --data-dir for large datasets)")
    parser.add_argument("--no-augment", action="store_true", help="Disable runtime augmentation (use when data was pre-augmented)")
    parser.add_argument("--bw", action="store_true", help="Use grayscale, high-contrast generator for validation set")
    parser.add_argument("--tag", default=None, help="Tag appended to checkpoint filenames, e.g. 'grayscale' -> best_grayscale.pt")
    parser.add_argument("--reset-lr", action="store_true", help="Reset LR schedule when resuming (for fine-tuning). --steps becomes the number of NEW training steps")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.steps is not None:
        config.setdefault("training", {})["max_iterations"] = args.steps

    if args.tag:
        tag = args.tag
        config.setdefault("training", {})["best_model_path"] = f"checkpoints/best_{tag}.pt"
        config.setdefault("training", {})["checkpoint_tag"] = tag

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    # Seed
    seed = config.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)

    # Create generator
    generator = SynthGenerator(
        fonts_json=data_cfg.get("fonts_cache", "data/fonts/fonts.json"),
        backgrounds_dir=data_cfg.get("backgrounds_dir", "data/backgrounds"),
        img_height=data_cfg.get("img_height", 32),
        img_min_width=data_cfg.get("img_min_width", 32),
        img_max_width=data_cfg.get("img_max_width", 800),
        min_text_len=data_cfg.get("min_text_len", 1),
        max_text_len=data_cfg.get("max_text_len", 50),
        word_mode_prob=data_cfg.get("word_mode_prob", 0.7),
        bg_solid_prob=data_cfg.get("bg_solid_prob", 0.3),
        bg_gradient_prob=data_cfg.get("bg_gradient_prob", 0.3),
        bg_texture_prob=data_cfg.get("bg_texture_prob", 0.4),
        bw=args.bw,
    )

    # Generate validation set
    val_dir = "data/val"
    val_size = train_cfg.get("val_size", 10000)
    generate_validation_set(generator, val_dir, val_size)

    # Create datasets
    aug_config = data_cfg.get("augmentation", {})
    aug_kwargs = {k: v for k, v in aug_config.items() if k != "enabled"}
    augment = aug_config.get("enabled", True) and not args.no_augment

    if args.no_augment:
        print("Runtime augmentation: disabled")

    if args.lmdb:
        print(f"Using LMDB training data from {args.lmdb}")
        train_dataset = LMDBOCRDataset(args.lmdb, augment=augment, aug_config=aug_kwargs)
    elif args.data_dir:
        print(f"Using pre-generated training data from {args.data_dir}")
        train_dataset = PregenOCRDataset(args.data_dir, augment=augment, aug_config=aug_kwargs)
    else:
        train_dataset = SynthOCRDataset(
            generator=generator,
            epoch_size=train_cfg.get("batch_size", 256) * 1000,
            augment=augment,
            aug_config=aug_kwargs,
        )
    val_dataset = PregenOCRDataset(val_dir)

    # Create data loaders
    batch_size = train_cfg.get("batch_size", 256)
    num_workers = train_cfg.get("num_workers", 4)

    prefetch_factor = train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    model = CRNN(
        num_classes=NUM_CLASSES,
        backbone_pretrained=model_cfg.get("backbone_pretrained", True),
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_num_layers=model_cfg.get("lstm_num_layers", 2),
        lstm_dropout=model_cfg.get("lstm_dropout", 0.1),
    )

    if model_cfg.get("compile", True):
        import platform
        if platform.system() == "Windows":
            print("torch.compile skipped (Triton not available on Windows)")
        else:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Num classes (incl. blank): {NUM_CLASSES}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume, reset_lr=args.reset_lr)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
