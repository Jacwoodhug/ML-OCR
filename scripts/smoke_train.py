"""Quick 50-step smoke test to validate the training pipeline."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data.synth_generator import SynthGenerator
from src.data.dataset import SynthOCRDataset, PregenOCRDataset, collate_fn
from src.data.alphabet import NUM_CLASSES
from src.model.crnn import CRNN
from src.training.trainer import Trainer
from scripts.train import generate_validation_set


def main():
    with open("config/default.yaml") as f:
        config = yaml.safe_load(f)

    # Override for smoke test
    config["training"]["max_iterations"] = 50
    config["training"]["val_interval"] = 25
    config["training"]["log_interval"] = 10
    config["training"]["checkpoint_interval"] = 50
    config["training"]["val_size"] = 100
    config["training"]["batch_size"] = 16
    config["training"]["num_workers"] = 0
    config["training"]["tensorboard_dir"] = "runs/smoke"

    gen = SynthGenerator(
        fonts_json="data/fonts/fonts.json",
        backgrounds_dir="data/backgrounds",
        img_height=32,
    )

    val_dir = "data/val_smoke"
    generate_validation_set(gen, val_dir, 100)

    train_ds = SynthOCRDataset(gen, epoch_size=160, augment=True)
    val_ds = PregenOCRDataset(val_dir)

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=0,
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=True,
    )

    model = CRNN(NUM_CLASSES, backbone_pretrained=False)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train()
    print("Smoke training test PASSED!")


if __name__ == "__main__":
    main()
