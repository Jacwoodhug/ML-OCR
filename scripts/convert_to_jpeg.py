"""Convert a pre-generated dataset from PNG to JPEG in-place.

Usage:
    python scripts/convert_to_jpeg.py --data-dir data/train-augment [--quality 90] [--workers 8]

Replaces all .png files in the images/ directory with .jpg equivalents.
The labels.txt file is left untouched.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image


def convert_one(png_path: str, quality: int) -> str:
    jpg_path = png_path[:-4] + ".jpg"
    if os.path.exists(jpg_path):
        os.remove(png_path)
        return "skipped"
    img = Image.open(png_path).convert("RGB")
    img.save(jpg_path, "JPEG", quality=quality, optimize=True)
    os.remove(png_path)
    return "converted"


def main():
    parser = argparse.ArgumentParser(description="Convert dataset PNGs to JPEG")
    parser.add_argument("--data-dir", required=True, help="Dataset directory (contains images/)")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality (default: 90)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel conversion threads")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "images")
    png_files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith(".png")
    ]

    if not png_files:
        print("No PNG files found — already converted or wrong directory.")
        return

    print(f"Converting {len(png_files):,} PNG files to JPEG (quality={args.quality}) using {args.workers} threads...")

    converted = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, p, args.quality): p for p in png_files}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            converted += 1
            if converted % 10_000 == 0:
                print(f"  {converted:,} / {len(png_files):,}")

    print(f"Done. Converted {converted:,} files.")


if __name__ == "__main__":
    main()
