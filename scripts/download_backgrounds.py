"""Download the DTD (Describable Textures Dataset) for background images.

Usage:
    python scripts/download_backgrounds.py [--output data/backgrounds]

Downloads and extracts the DTD dataset (~600MB) which provides 5640 texture
images across 47 categories — used as realistic backgrounds for synthetic
OCR training data.
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_FILENAME = "dtd-r1.0.1.tar.gz"


def download_file(url: str, dest: str) -> None:
    """Download a file with progress reporting."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()  # newline after progress


def extract_images(tar_path: str, output_dir: str) -> int:
    """Extract image files from the DTD tar.gz archive.

    Flattens the directory structure so all images end up directly in output_dir.
    Returns the number of images extracted.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    count = 0

    print(f"Extracting images to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            ext = Path(member.name).suffix.lower()
            if ext not in image_extensions:
                continue

            # Flatten: use just the filename, prefixed with category to avoid collisions
            parts = Path(member.name).parts
            # Typical path: dtd/images/category/filename.jpg
            if len(parts) >= 3:
                category = parts[-2]
                filename = f"{category}_{parts[-1]}"
            else:
                filename = parts[-1]

            dest_path = os.path.join(output_dir, filename)
            # Extract file content
            f = tar.extractfile(member)
            if f is None:
                continue
            with open(dest_path, "wb") as out_f:
                out_f.write(f.read())
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Download DTD texture dataset for OCR backgrounds")
    parser.add_argument("--output", default="data/backgrounds", help="Output directory for images")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded tar.gz file")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    if len(existing) > 1000:
        print(f"Background directory already contains {len(existing)} images, skipping download.")
        return

    # Download
    archive_path = str(output_dir / DTD_FILENAME)
    if not os.path.exists(archive_path):
        download_file(DTD_URL, archive_path)
    else:
        print(f"Archive already exists: {archive_path}")

    # Extract
    count = extract_images(archive_path, str(output_dir))
    print(f"Extracted {count} images to {output_dir}")

    # Clean up archive
    if not args.keep_archive and os.path.exists(archive_path):
        os.remove(archive_path)
        print(f"Removed archive: {archive_path}")


if __name__ == "__main__":
    main()
