"""ONNX Runtime inference wrapper for the OCR model.

Handles image preprocessing, model inference, and CTC decoding.
Supports both CPU and GPU execution providers.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.data.alphabet import BLANK_IDX, IDX_TO_CHAR


class OCRPredictor:
    """High-level OCR inference using an ONNX model."""

    def __init__(
        self,
        onnx_path: str,
        img_height: int = 32,
        img_max_width: int = 800,
        use_gpu: bool = False,
    ):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.img_height = img_height
        self.img_max_width = img_max_width

        # Log active provider
        active = self.session.get_providers()
        print(f"ONNX Runtime providers: {active}")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL image for the model.

        Resizes to target height preserving aspect ratio, converts to float32 [0,1].

        Returns:
            (1, 3, H, W) numpy array
        """
        image = image.convert("RGB")
        w, h = image.size

        # Resize to target height, preserving aspect ratio
        aspect = w / max(h, 1)
        new_w = int(self.img_height * aspect)
        new_w = max(32, min(self.img_max_width, new_w))
        # Ensure width is divisible by 4 (backbone stride)
        new_w = (new_w // 4) * 4
        if new_w < 4:
            new_w = 4

        image = image.resize((new_w, self.img_height), Image.BILINEAR)

        # Convert to CHW float32 [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        img_np = np.expand_dims(img_np, 0)  # add batch dim

        return img_np

    def decode_output(self, log_probs: np.ndarray) -> list[str]:
        """Greedy CTC decoding on ONNX output.

        Args:
            log_probs: (T, B, C) numpy array

        Returns:
            List of B decoded strings
        """
        predictions = np.argmax(log_probs, axis=2)  # (T, B)
        predictions = predictions.T  # (B, T)

        results = []
        for seq in predictions:
            decoded = []
            prev = BLANK_IDX
            for idx in seq:
                idx = int(idx)
                if idx != BLANK_IDX and idx != prev:
                    if idx in IDX_TO_CHAR:
                        decoded.append(IDX_TO_CHAR[idx])
                prev = idx
            results.append("".join(decoded))

        return results

    def predict(self, image: Image.Image) -> str:
        """Run OCR on a single PIL image. Returns the predicted text."""
        img_np = self.preprocess(image)
        log_probs = self.session.run(None, {"image": img_np})[0]
        texts = self.decode_output(log_probs)
        return texts[0]

    def predict_batch(self, images: list[Image.Image]) -> list[str]:
        """Run OCR on a batch of images.

        Images are preprocessed individually and padded to the same width
        before batching.
        """
        if not images:
            return []

        processed = [self.preprocess(img) for img in images]
        max_w = max(p.shape[3] for p in processed)

        # Pad to max width
        padded = []
        for p in processed:
            pad_w = max_w - p.shape[3]
            if pad_w > 0:
                padding = np.zeros((1, 3, self.img_height, pad_w), dtype=np.float32)
                p = np.concatenate([p, padding], axis=3)
            padded.append(p)

        batch = np.concatenate(padded, axis=0)  # (B, 3, H, W)
        log_probs = self.session.run(None, {"image": batch})[0]
        return self.decode_output(log_probs)
