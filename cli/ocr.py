"""Lightweight CLI for OCR inference using an ONNX model.

Only depends on: onnxruntime, numpy, Pillow.

Usage:
    python cli/ocr.py --model exports/model.onnx image.png
    python cli/ocr.py --model exports/model.onnx img1.png img2.png img3.png
    python cli/ocr.py --model exports/model.onnx --no-bw image.png
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Alphabet (inlined for zero-dependency on src/) ────────────────────────────

BLANK_IDX = 0
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?-:;'\"()@#$%&+=/_~[] {}\\|<>^`"
)
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}


# ── Image helpers ─────────────────────────────────────────────────────────────

def convert_to_bw(image: Image.Image) -> Image.Image:
    """Convert to grayscale using the RGB channel with the most contrast."""
    img_np = np.array(image.convert("RGB"), dtype=np.float32)
    stds = [img_np[:, :, c].std() for c in range(3)]
    best = int(np.argmax(stds))
    channel = img_np[:, :, best].astype(np.uint8)
    return Image.fromarray(channel, mode="L").convert("RGB")


def preprocess(image: Image.Image, img_height: int = 32, img_max_width: int = 800) -> np.ndarray:
    """Resize and normalise a PIL image to a (1, 3, H, W) float32 tensor."""
    image = image.convert("RGB")
    w, h = image.size
    aspect = w / max(h, 1)
    new_w = int(img_height * aspect)
    new_w = max(32, min(img_max_width, new_w))
    new_w = (new_w // 4) * 4
    if new_w < 4:
        new_w = 4

    image = image.resize((new_w, img_height), Image.BILINEAR)
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img_np, 0)


# ── CTC decode ────────────────────────────────────────────────────────────────

def ctc_decode(log_probs: np.ndarray) -> list[str]:
    """Greedy CTC decoding. log_probs shape: (T, B, C)."""
    preds = np.argmax(log_probs, axis=2).T  # (B, T)
    results = []
    for seq in preds:
        decoded = []
        prev = BLANK_IDX
        for idx in seq:
            idx = int(idx)
            if idx != BLANK_IDX and idx != prev:
                ch = IDX_TO_CHAR.get(idx)
                if ch is not None:
                    decoded.append(ch)
            prev = idx
        results.append("".join(decoded))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_default_model() -> str | None:
    """Return the first .onnx file found next to this script, or None."""
    models = sorted(glob.glob(os.path.join(_SCRIPT_DIR, "*.onnx")))
    return models[0] if models else None


def main() -> None:
    default_model = _find_default_model()

    parser = argparse.ArgumentParser(description="Lightweight ONNX OCR inference")
    parser.add_argument("images", nargs="+", help="Image file path(s)")
    parser.add_argument(
        "--model",
        default=default_model,
        help="Path to ONNX model (default: auto-detect in cli/ folder)",
    )
    parser.add_argument("--no-bw", action="store_true", help="Skip automatic BW conversion")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA execution provider")
    args = parser.parse_args()

    if args.model is None:
        parser.error("No .onnx model found in cli/ folder. Specify one with --model.")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)

    for path in args.images:
        try:
            image = Image.open(path)
        except Exception as exc:
            print(f"{path}: ERROR - {exc}", file=sys.stderr)
            continue

        if not args.no_bw:
            image = convert_to_bw(image)

        t0 = time.perf_counter()
        inp = preprocess(image)
        log_probs = session.run(None, {"image": inp})[0]
        text = ctc_decode(log_probs)[0]
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"{path}: {text}  ({elapsed:.1f}ms)")


if __name__ == "__main__":
    main()
