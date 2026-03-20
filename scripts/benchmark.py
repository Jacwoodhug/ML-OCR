"""Benchmark inference speed for the OCR model (both PyTorch and ONNX).

Usage:
    python scripts/benchmark.py --checkpoint checkpoints/best.pt --onnx exports/model.onnx
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alphabet import NUM_CLASSES


def benchmark_pytorch(checkpoint_path: str, config: dict, device_name: str = "cpu"):
    """Benchmark PyTorch model inference."""
    from src.model.crnn import CRNN

    model_cfg = config.get("model", {})
    device = torch.device(device_name)

    model = CRNN(
        num_classes=NUM_CLASSES,
        backbone_pretrained=False,
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_num_layers=model_cfg.get("lstm_num_layers", 2),
        lstm_dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_widths = [64, 128, 256, 512]
    warmup = 10
    repeats = 100

    print(f"\nPyTorch ({device_name}):")
    print(f"{'Width':>8} {'Mean (ms)':>10} {'Std (ms)':>10} {'FPS':>8}")
    print("-" * 40)

    for w in test_widths:
        dummy = torch.randn(1, 3, 32, w, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        fps = 1000.0 / mean_ms
        print(f"{w:>8} {mean_ms:>10.2f} {std_ms:>10.2f} {fps:>8.1f}")


def benchmark_onnx(onnx_path: str, use_gpu: bool = False):
    """Benchmark ONNX Runtime inference."""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    active_provider = session.get_providers()[0]

    test_widths = [64, 128, 256, 512]
    warmup = 10
    repeats = 100

    print(f"\nONNX Runtime ({active_provider}):")
    print(f"{'Width':>8} {'Mean (ms)':>10} {'Std (ms)':>10} {'FPS':>8}")
    print("-" * 40)

    for w in test_widths:
        dummy = np.random.randn(1, 3, 32, w).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            session.run(None, {"image": dummy})

        # Benchmark
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            session.run(None, {"image": dummy})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        fps = 1000.0 / mean_ms
        print(f"{w:>8} {mean_ms:>10.2f} {std_ms:>10.2f} {fps:>8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR model inference speed")
    parser.add_argument("--checkpoint", default=None, help="PyTorch checkpoint path")
    parser.add_argument("--onnx", default=None, help="ONNX model path")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    args = parser.parse_args()

    if not args.checkpoint and not args.onnx:
        print("Provide at least one of --checkpoint or --onnx")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        benchmark_pytorch(args.checkpoint, config, "cpu")
        if torch.cuda.is_available():
            benchmark_pytorch(args.checkpoint, config, "cuda")

    if args.onnx:
        benchmark_onnx(args.onnx, use_gpu=False)
        try:
            benchmark_onnx(args.onnx, use_gpu=True)
        except Exception:
            print("GPU ONNX Runtime not available, skipping.")


if __name__ == "__main__":
    main()
