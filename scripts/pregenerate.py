"""Pre-generate a synthetic training dataset to disk.

Usage:
    python scripts/pregenerate.py [--count 500000] [--output data/train]

Generates images + labels.txt so training reads from disk instead of
rendering on-the-fly, eliminating the CPU data-generation bottleneck.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from src.data.synth_generator import SynthGenerator


def main():
    parser = argparse.ArgumentParser(description="Pre-generate synthetic training data")
    parser.add_argument("--count", type=int, default=500_000, help="Number of samples to generate")
    parser.add_argument("--output", default="data/train", help="Output directory")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})

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
    )

    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_path = os.path.join(args.output, "labels.txt")

    # Check for existing partial generation to allow resuming
    start_idx = 0
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)
        if start_idx >= args.count:
            print(f"Already have {start_idx} samples (requested {args.count}). Nothing to do.")
            return
        print(f"Resuming from sample {start_idx} (found existing labels)")

    print(f"Generating {args.count - start_idx} samples to {args.output}/ ...")

    with open(labels_path, "a", encoding="utf-8") as f:
        for i in range(start_idx, args.count):
            img, text = generator.generate()
            img.save(os.path.join(images_dir, f"{i:06d}.png"))
            f.write(text + "\n")

            if (i + 1) % 10_000 == 0:
                print(f"  {i + 1:,} / {args.count:,}")
                f.flush()

    print(f"Done. Saved {args.count:,} samples to {args.output}/")


if __name__ == "__main__":
    main()
