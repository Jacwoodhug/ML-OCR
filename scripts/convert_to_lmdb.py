"""Convert a pre-generated image dataset (individual JPG/PNG files) to LMDB.

The resulting LMDB has:
    image-{idx:08d}  ->  raw JPEG bytes (re-encoded at given quality)
    label-{idx:08d}  ->  UTF-8 text label
    num_samples      ->  ASCII decimal count

Usage:
    python scripts/convert_to_lmdb.py --data-dir data/train-simple --out data/train-simple.lmdb
    python scripts/convert_to_lmdb.py --data-dir data/val          --out data/val.lmdb
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import lmdb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_image_bytes(args):
    """Read one image and re-encode as JPEG bytes. Returns jpeg_bytes."""
    img_path, jpeg_quality = args
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read {img_path}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"Failed to encode {img_path}")
    return buf.tobytes()


def convert(data_dir: str, out_path: str, jpeg_quality: int, num_workers: int,
            commit_every: int, map_size_gb: int):
    labels_path = os.path.join(data_dir, "labels.txt")
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f]

    n = len(labels)
    image_dir = os.path.join(data_dir, "images")
    first_jpg = os.path.join(image_dir, "000000.jpg")
    ext = ".jpg" if os.path.exists(first_jpg) else ".png"

    map_size = map_size_gb * 1024 ** 3
    print(f"Samples: {n:,}  |  map size: {map_size_gb} GB  |  workers: {num_workers}")

    env = lmdb.open(out_path, map_size=map_size, writemap=False)

    # Build tasks in index order — executor.map preserves order, so LMDB
    # receives sequential keys (image-00000000, image-00000001, ...) which
    # are pure appends on the B-tree and avoid page-split slowdown.
    task_args = [(os.path.join(image_dir, f"{i:06d}{ext}"), jpeg_quality) for i in range(n)]

    written = 0
    txn = env.begin(write=True)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pbar = tqdm(total=n, desc="Converting", unit="img")
        for idx, jpeg_bytes in enumerate(pool.map(read_image_bytes, task_args, chunksize=64)):
            txn.put(f"image-{idx:08d}".encode(), jpeg_bytes)
            txn.put(f"label-{idx:08d}".encode(), labels[idx].encode("utf-8"))
            written += 1
            pbar.update(1)

            if written % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

    txn.put(b"num_samples", str(n).encode())
    txn.commit()
    pbar.close()
    env.close()

    db_size = sum(
        os.path.getsize(os.path.join(out_path, f))
        for f in os.listdir(out_path)
    )
    print(f"Done. LMDB written to {out_path}  ({db_size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to LMDB")
    parser.add_argument("--data-dir", required=True, help="Source dataset directory")
    parser.add_argument("--out", required=True, help="Output LMDB path")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for re-encoding (default: 95)")
    parser.add_argument("--workers", type=int, default=8, help="Reader threads (default: 8)")
    parser.add_argument("--commit-every", type=int, default=10000, help="Commit transaction every N writes")
    parser.add_argument("--map-size-gb", type=int, default=20, help="LMDB map size in GB (default: 20)")
    args = parser.parse_args()

    if os.path.exists(args.out):
        print(f"Output path {args.out} already exists. Delete it first.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    convert(args.data_dir, args.out, args.jpeg_quality, args.workers, args.commit_every, args.map_size_gb)


if __name__ == "__main__":
    main()
