"""Pre-generate a synthetic training dataset to disk or LMDB.

Usage:
    python scripts/pregenerate.py [--count 500000] [--output data/train] [--augment]
    python scripts/pregenerate.py --bw --bg-ratio 20 --count 500000 --output data/train
    python scripts/pregenerate.py --lmdb data/train.lmdb --count 500000 --bw
    python scripts/pregenerate.py --font-file path/to/font.ttf --count 10000
    python scripts/pregenerate.py --font-dir path/to/fonts/ --count 10000

Generates images + labels.txt so training reads from disk instead of
rendering on-the-fly, eliminating the CPU data-generation bottleneck.

Use --lmdb to write directly to LMDB format, skipping intermediate files.

Use --augment to bake augmentations into the saved images. Combined with
--no-augment in train.py, this removes all per-sample CPU work from the
training loop so workers just decode PNGs.
"""

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm
from src.data.synth_generator import SynthGenerator

# ---------------------------------------------------------------------------
# Per-worker state (populated by _worker_init, isolated per process)
# ---------------------------------------------------------------------------
_worker_state: dict = {}


def _worker_init(generator_kwargs: dict, aug_kwargs: dict | None) -> None:
    import random
    from src.data.augmentations import apply_augmentation, get_augmentation_pipeline
    random.seed(os.getpid())
    _worker_state["generator"] = SynthGenerator(**generator_kwargs)
    _worker_state["apply_augmentation"] = apply_augmentation
    _worker_state["aug_pipeline"] = (
        get_augmentation_pipeline(**aug_kwargs) if aug_kwargs is not None else None
    )


def _generate_and_save(task: tuple) -> str:
    """Generate one sample, save image to disk, return label text."""
    i, images_dir, fmt, quality = task
    img, text = _worker_state["generator"].generate()
    aug = _worker_state["aug_pipeline"]
    if aug is not None:
        img_np = np.array(img, dtype=np.uint8)
        img_np = _worker_state["apply_augmentation"](img_np, aug)
        img = Image.fromarray(img_np)
    img_file = os.path.join(images_dir, f"{i:06d}.{fmt}")
    if fmt == "jpg":
        img.save(img_file, "JPEG", quality=quality)
    else:
        img.save(img_file)
    return text


def _generate_lmdb_sample(task: tuple) -> tuple[bytes, str]:
    """Generate one sample, return (jpeg_bytes, label)."""
    _idx, jpeg_quality = task
    img, text = _worker_state["generator"].generate()
    aug = _worker_state["aug_pipeline"]
    if aug is not None:
        img_np = np.array(img, dtype=np.uint8)
        img_np = _worker_state["apply_augmentation"](img_np, aug)
        img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=jpeg_quality)
    return buf.getvalue(), text


def _build_generator_kwargs(args, data_cfg):
    """Build generator kwargs from CLI args and config."""
    # Resolve font source: explicit files/dir > google-fonts > config default
    font_paths = None
    fonts_json = None
    if args.font_file or args.font_dir:
        font_paths = list(args.font_file)
        if args.font_dir:
            font_exts = {".ttf", ".otf", ".TTF", ".OTF"}
            if not os.path.isdir(args.font_dir):
                print(f"Error: --font-dir '{args.font_dir}' is not a directory")
                sys.exit(1)
            font_paths.extend(
                os.path.join(args.font_dir, f)
                for f in sorted(os.listdir(args.font_dir))
                if os.path.splitext(f)[1] in font_exts
            )
        if not font_paths:
            print("Error: no font files found from --font-file / --font-dir")
            sys.exit(1)
        print(f"Using {len(font_paths)} custom font(s)")
    else:
        fonts_json = "data/fonts/google_fonts.json" if args.google_fonts else data_cfg.get("fonts_cache", "data/fonts/fonts.json")

    # Compute background probabilities from --bg-ratio
    bg_ratio = max(0, min(100, args.bg_ratio)) / 100.0
    remaining = 1.0 - bg_ratio
    bg_solid_prob = remaining * 0.5
    bg_gradient_prob = remaining * 0.5
    bg_texture_prob = bg_ratio

    return dict(
        fonts_json=fonts_json,
        font_paths=font_paths,
        backgrounds_dir=data_cfg.get("backgrounds_dir", "data/backgrounds"),
        img_height=data_cfg.get("img_height", 32),
        img_min_width=data_cfg.get("img_min_width", 32),
        img_max_width=data_cfg.get("img_max_width", 800),
        min_text_len=data_cfg.get("min_text_len", 1),
        max_text_len=data_cfg.get("max_text_len", 50),
        word_mode_prob=data_cfg.get("word_mode_prob", 0.7),
        bg_solid_prob=bg_solid_prob,
        bg_gradient_prob=bg_gradient_prob,
        bg_texture_prob=bg_texture_prob,
        bw=args.bw,
    )


def _generate_to_files(args, generator_kwargs, aug_kwargs):
    """Generate samples as individual image files + labels.txt."""
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)
    labels_path = os.path.join(args.output, "labels.txt")

    # Check for existing partial generation to allow resuming
    start_idx = 0
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content:
            lines = content.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            else:
                print("Warning: truncated last label detected, discarding it")
                lines = lines[:-1]
                with open(labels_path, "w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")
            start_idx = len(lines)

        if start_idx >= args.count:
            print(f"Already have {start_idx} samples (requested {args.count}). Nothing to do.")
            return
        if start_idx > 0:
            print(f"Resuming from sample {start_idx} (found existing labels)")

    n_workers = args.workers or min(os.cpu_count() or 1, 8)
    print(f"Generating {args.count - start_idx} samples to {args.output}/ (workers={n_workers}) ...")

    tasks = (
        (i, images_dir, args.format, args.jpeg_quality)
        for i in range(start_idx, args.count)
    )

    completed = 0
    with open(labels_path, "a", encoding="utf-8") as f:
        executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(generator_kwargs, aug_kwargs),
        )
        try:
            with tqdm(total=args.count, initial=start_idx, unit="img", dynamic_ncols=True) as pbar:
                for completed, text in enumerate(executor.map(_generate_and_save, tasks, chunksize=64), 1):
                    f.write(text + "\n")
                    pbar.update(1)
                    if (start_idx + completed) % 10_000 == 0:
                        f.flush()
        except KeyboardInterrupt:
            print("\nInterrupted — stopping workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            f.flush()
            print(f"Saved {start_idx + completed:,} samples. Resume with the same command.")
            return
        else:
            executor.shutdown(wait=True)

    print(f"Done. Saved {args.count:,} samples to {args.output}/")


def _generate_to_lmdb(args, generator_kwargs, aug_kwargs):
    """Generate samples directly into LMDB."""
    import lmdb

    lmdb_path = args.lmdb
    commit_every = 10_000
    map_size = args.map_size_gb * 1024 ** 3

    # Check for existing partial LMDB to allow resuming
    start_idx = 0
    if os.path.exists(lmdb_path):
        try:
            env = lmdb.open(lmdb_path, readonly=True)
            with env.begin() as txn:
                val = txn.get(b"num_samples")
                if val is not None:
                    existing = int(val.decode())
                    if existing >= args.count:
                        print(f"Already have {existing} samples (requested {args.count}). Nothing to do.")
                        env.close()
                        return
                    start_idx = existing
                    print(f"Resuming from sample {start_idx} (found existing LMDB)")
            env.close()
        except Exception:
            pass
    else:
        os.makedirs(lmdb_path, exist_ok=True)

    n_workers = args.workers or min(os.cpu_count() or 1, 8)
    print(f"Generating {args.count - start_idx} samples to {lmdb_path} (workers={n_workers}, map_size={args.map_size_gb}GB) ...")

    tasks = ((i, args.jpeg_quality) for i in range(start_idx, args.count))

    env = lmdb.open(lmdb_path, map_size=map_size, writemap=False)
    txn = env.begin(write=True)
    written = 0
    completed = 0

    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_worker_init,
        initargs=(generator_kwargs, aug_kwargs),
    )
    try:
        with tqdm(total=args.count, initial=start_idx, unit="img", dynamic_ncols=True) as pbar:
            for completed, (jpeg_bytes, text) in enumerate(
                executor.map(_generate_lmdb_sample, tasks, chunksize=64), 1
            ):
                idx = start_idx + completed - 1
                txn.put(f"image-{idx:08d}".encode(), jpeg_bytes)
                txn.put(f"label-{idx:08d}".encode(), text.encode("utf-8"))
                written += 1
                pbar.update(1)

                if written % commit_every == 0:
                    txn.put(b"num_samples", str(idx + 1).encode())
                    txn.commit()
                    txn = env.begin(write=True)
    except KeyboardInterrupt:
        print("\nInterrupted — committing progress...")
        total_written = start_idx + completed
        txn.put(b"num_samples", str(total_written).encode())
        txn.commit()
        executor.shutdown(wait=False, cancel_futures=True)
        env.close()
        db_size = sum(os.path.getsize(os.path.join(lmdb_path, f)) for f in os.listdir(lmdb_path))
        print(f"Saved {total_written:,} samples ({db_size / 1e9:.2f} GB). Resume with the same command.")
        return
    else:
        executor.shutdown(wait=True)

    total = start_idx + completed
    txn.put(b"num_samples", str(total).encode())
    txn.commit()
    env.close()

    db_size = sum(os.path.getsize(os.path.join(lmdb_path, f)) for f in os.listdir(lmdb_path))
    print(f"Done. Saved {total:,} samples to {lmdb_path} ({db_size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Pre-generate synthetic training data")
    parser.add_argument("--count", type=int, default=500_000, help="Number of samples to generate")
    parser.add_argument("--output", default="data/train", help="Output directory (for file-based output)")
    parser.add_argument("--lmdb", default=None, help="Write directly to LMDB at this path (skips intermediate files)")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--augment", action="store_true", help="Bake augmentations into saved images")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Image format (default: jpg)")
    parser.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality (default: 90)")
    parser.add_argument("--google-fonts", action="store_true", help="Use Google Fonts instead of system fonts (data/fonts/google_fonts.json)")
    parser.add_argument("--bw", action="store_true", help="Grayscale output with text/bg luminance contrast enforced")
    parser.add_argument("--bg-ratio", type=int, default=0, metavar="PCT", help="Percentage of images that use random backgrounds from data/backgrounds (default: 0 = none)")
    parser.add_argument("--font-file", action="append", default=[], help="Path to a specific font file (.ttf/.otf) to use. Can be repeated.")
    parser.add_argument("--font-dir", default=None, help="Path to a directory of font files (.ttf/.otf) to use instead of the fonts cache")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: min(CPU count, 8))")
    parser.add_argument("--map-size-gb", type=int, default=20, help="LMDB map size in GB (default: 20)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    generator_kwargs = _build_generator_kwargs(args, data_cfg)

    aug_kwargs = None
    if args.augment:
        aug_config = config.get("data", {}).get("augmentation", {})
        aug_kwargs = {k: v for k, v in aug_config.items() if k != "enabled"}
        print("Augmentation: enabled (baking into images)")
    else:
        print("Augmentation: disabled (clean images)")

    if args.lmdb:
        _generate_to_lmdb(args, generator_kwargs, aug_kwargs)
    else:
        _generate_to_files(args, generator_kwargs, aug_kwargs)


if __name__ == "__main__":
    main()
