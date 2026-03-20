"""Export trained CRNN model to ONNX format with dynamic axes.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt [--output exports/model.onnx]
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.alphabet import NUM_CLASSES
from src.model.crnn import CRNN


def export_onnx(
    model: CRNN,
    output_path: str,
    img_height: int = 32,
    opset_version: int = 17,
):
    """Export model to ONNX with dynamic batch and width axes."""
    model.eval()

    # Dummy input: batch=1, channels=3, height=32, width=128
    dummy_input = torch.randn(1, 3, img_height, 128)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={
            "image": {0: "batch", 3: "width"},
            "log_probs": {0: "time", 1: "batch"},
        },
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model to {output_path} ({file_size:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export OCR model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="config/default.yaml", help="Config file")
    parser.add_argument("--output", default="exports/model.onnx", help="ONNX output path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})

    model = CRNN(
        num_classes=NUM_CLASSES,
        backbone_pretrained=False,
        lstm_hidden_size=model_cfg.get("lstm_hidden_size", 256),
        lstm_num_layers=model_cfg.get("lstm_num_layers", 2),
        lstm_dropout=0.0,  # no dropout at inference
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from step {checkpoint.get('step', '?')}")

    export_onnx(model, args.output)

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(args.output)
        dummy = np.random.randn(1, 3, 32, 128).astype(np.float32)
        outputs = session.run(None, {"image": dummy})
        print(f"ONNX verification passed. Output shape: {outputs[0].shape}")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


if __name__ == "__main__":
    main()
