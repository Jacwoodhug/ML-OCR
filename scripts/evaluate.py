"""Evaluate a trained OCR model on the validation set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt [--config config/default.yaml]
"""

import argparse
import os
import sys

import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alphabet import NUM_CLASSES
from src.data.dataset import PregenOCRDataset, collate_fn
from src.model.crnn import CRNN
from src.model.ctc_decoder import beam_search_decode, greedy_decode
from src.training.metrics import batch_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--val-dir", default="data/val", help="Validation data directory")
    parser.add_argument("--beam-width", type=int, default=0, help="Beam width (0=greedy)")
    parser.add_argument("--show-samples", type=int, default=20, help="Number of sample predictions to print")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CRNN(
        num_classes=NUM_CLASSES,
        backbone_pretrained=False,
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_num_layers=model_cfg.get("lstm_num_layers", 2),
        lstm_dropout=model_cfg.get("lstm_dropout", 0.1),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint.get('step', '?')}")
    print(f"Checkpoint best CER: {checkpoint.get('best_cer', '?')}")

    # Load validation data
    val_dataset = PregenOCRDataset(args.val_dir)
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Evaluating on {len(val_dataset)} samples...")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            image_widths = batch["image_widths"].to(device)
            texts = batch["texts"]

            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                log_probs = model(images)

            if args.beam_width > 0:
                preds = beam_search_decode(log_probs.float().cpu(), beam_width=args.beam_width)
            else:
                preds = greedy_decode(log_probs.float())

            all_preds.extend(preds)
            all_targets.extend(texts)

    # Compute metrics
    metrics = batch_metrics(all_preds, all_targets)
    print(f"\nResults ({len(all_preds)} samples):")
    print(f"  CER:          {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"  WER:          {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"  Seq Accuracy: {metrics['seq_acc']:.4f} ({metrics['seq_acc']*100:.2f}%)")

    # Show sample predictions
    if args.show_samples > 0:
        print(f"\nSample predictions ({args.show_samples}):")
        print("-" * 80)
        for i in range(min(args.show_samples, len(all_preds))):
            match = "OK" if all_preds[i] == all_targets[i] else "XX"
            print(f"  [{match}] Target: {all_targets[i]!r}")
            print(f"       Pred:   {all_preds[i]!r}")
            print()


if __name__ == "__main__":
    main()
