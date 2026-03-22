"""Benchmark cli/ocr.py (custom ONNX OCR) vs popular OCR libraries.

Compares against: EasyOCR, Tesseract

Usage:
    python benchmark.py [--n 1000] [--model cli/model.onnx] [--gpu]
"""

import argparse
import os
import sys
import time
import importlib.util

import numpy as np
from PIL import Image
import onnxruntime as ort

# ── Custom OCR imports (from cli/ocr.py) ─────────────────────────────────────
_cli_ocr_spec = importlib.util.spec_from_file_location(
    "cli_ocr", os.path.join(os.path.dirname(__file__), "cli", "ocr.py")
)
_cli_ocr = importlib.util.module_from_spec(_cli_ocr_spec)
_cli_ocr_spec.loader.exec_module(_cli_ocr)
convert_to_bw = _cli_ocr.convert_to_bw
preprocess = _cli_ocr.preprocess
ctc_decode = _cli_ocr.ctc_decode


def load_val_set(val_dir: str, n: int | None = None):
    """Load (image_path, label) pairs from a val/ directory."""
    labels_path = os.path.join(val_dir, "labels.txt")
    images_dir = os.path.join(val_dir, "images")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f]

    pairs = []
    for i, label in enumerate(labels):
        img_path = os.path.join(images_dir, f"{i:06d}.jpg")
        if os.path.exists(img_path):
            pairs.append((img_path, label))
        if n and len(pairs) >= n:
            break
    return pairs


def edit_distance(pred: str, gt: str) -> int:
    """Levenshtein edit distance."""
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_metrics(predictions, pairs):
    """Compute exact match %, mean char accuracy %, and mean CER."""
    exact = 0
    char_accs = []
    for pred, (_, gt) in zip(predictions, pairs):
        if pred == gt:
            exact += 1
        dist = edit_distance(pred, gt)
        max_len = max(len(gt), 1)
        cer = dist / max_len
        char_accs.append(max(0.0, 1.0 - cer))
    n = len(pairs)
    return exact / n * 100, np.mean(char_accs) * 100


# ── Benchmark runners ────────────────────────────────────────────────────────

def benchmark_custom_ocr(pairs, model_path: str, use_gpu: bool):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(model_path, providers=providers)

    predictions = []
    t0 = time.perf_counter()
    for img_path, _ in pairs:
        image = Image.open(img_path)
        image = convert_to_bw(image)
        inp = preprocess(image)
        log_probs = session.run(None, {"image": inp})[0]
        text = ctc_decode(log_probs)[0]
        predictions.append(text)
    elapsed = time.perf_counter() - t0
    return predictions, elapsed


def benchmark_easyocr(pairs, use_gpu: bool):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)

    predictions = []
    t0 = time.perf_counter()
    for img_path, _ in pairs:
        result = reader.readtext(img_path, detail=0)
        text = " ".join(result) if result else ""
        predictions.append(text)
    elapsed = time.perf_counter() - t0
    return predictions, elapsed



def benchmark_tesseract(pairs):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    predictions = []
    t0 = time.perf_counter()
    for img_path, _ in pairs:
        image = Image.open(img_path)
        text = pytesseract.image_to_string(image, config="--psm 7").strip()
        predictions.append(text)
    elapsed = time.perf_counter() - t0
    return predictions, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark custom OCR vs popular libraries")
    parser.add_argument(
        "--val-dir",
        default=os.path.join("data", "train-bg80.lmdb", "val"),
        help="Path to validation directory",
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("--model", default=os.path.join("cli", "model.onnx"), help="ONNX model path")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    pairs = load_val_set(args.val_dir, args.n)
    print(f"Loaded {len(pairs)} validation samples\n")

    results = {}

    # ── Custom ONNX OCR ──────────────────────────────────────────────────────
    print("[ 1/3 ] Custom ONNX OCR ...")
    preds, elapsed = benchmark_custom_ocr(pairs, args.model, args.gpu)
    exact, char_acc = compute_metrics(preds, pairs)
    results["Custom ONNX"] = (preds, elapsed, exact, char_acc)

    # ── EasyOCR ──────────────────────────────────────────────────────────────
    print("[ 2/3 ] EasyOCR ...")
    preds, elapsed = benchmark_easyocr(pairs, args.gpu)
    exact, char_acc = compute_metrics(preds, pairs)
    results["EasyOCR"] = (preds, elapsed, exact, char_acc)

    # ── Tesseract ────────────────────────────────────────────────────────────
    print("[ 3/3 ] Tesseract ...")
    try:
        preds, elapsed = benchmark_tesseract(pairs)
        exact, char_acc = compute_metrics(preds, pairs)
        results["Tesseract"] = (preds, elapsed, exact, char_acc)
    except Exception as e:
        print(f"         Tesseract failed: {e}")

    # ── Results table ────────────────────────────────────────────────────────
    names = list(results.keys())
    col_w = max(15, max(len(n) for n in names) + 2)
    header = f"{'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(header)
    print(sep)

    row = f"{'Exact match (%)':<25}"
    for n in names:
        row += f"{results[n][2]:>{col_w - 1}.1f}%"
    print(row)

    row = f"{'Char accuracy (%)':<25}"
    for n in names:
        row += f"{results[n][3]:>{col_w - 1}.1f}%"
    print(row)

    row = f"{'Total time (s)':<25}"
    for n in names:
        row += f"{results[n][1]:>{col_w}.2f}"
    print(row)

    row = f"{'Avg per image (ms)':<25}"
    for n in names:
        row += f"{results[n][1] / len(pairs) * 1000:>{col_w}.1f}"
    print(row)

    # Speed multiplier vs custom
    custom_time = results["Custom ONNX"][1]
    row = f"{'Speed vs Custom':<25}"
    for n in names:
        mult = results[n][1] / custom_time
        row += f"{mult:>{col_w - 1}.1f}x"
    print(row)

    print(f"{'=' * len(header)}")

    # ── Sample comparisons ───────────────────────────────────────────────────
    print(f"\nSample predictions (first 10):")
    trunc = 30
    col = max(trunc + 2, 15)
    h = f"{'GT':<{col}}" + "".join(f"{n:<{col}}" for n in names)
    print(h)
    print("-" * len(h))
    for i in range(min(10, len(pairs))):
        gt = pairs[i][1][:trunc]
        line = f"{gt:<{col}}"
        for n in names:
            p = results[n][0][i][:trunc]
            line += f"{p:<{col}}"
        print(line)


if __name__ == "__main__":
    main()
