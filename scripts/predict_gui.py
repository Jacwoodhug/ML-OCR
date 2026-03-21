"""Browser-based GUI for OCR inference using Gradio.

Upload or paste a pre-cropped image and the model will return the recognised text.

Usage:
    python scripts/predict_gui.py --model exports/model.onnx
    python scripts/predict_gui.py          # browse for model in the UI
"""

import argparse
import os
import sys

import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import OCRPredictor


def build_app(model_path: str | None) -> gr.Blocks:
    predictor: list[OCRPredictor | None] = [None]  # mutable container for closure

    if model_path:
        predictor[0] = OCRPredictor(model_path)

    def load_model(path: str) -> str:
        path = path.strip()
        if not path:
            return "Enter the path to an ONNX model file."
        if not os.path.isfile(path):
            return f"File not found: {path}"
        try:
            predictor[0] = OCRPredictor(path)
            return f"Model loaded: {os.path.basename(path)}"
        except Exception as exc:
            return f"Failed to load model: {exc}"

    def convert_to_bw(image: Image.Image | None) -> Image.Image | None:
        if image is None:
            return None
        img_np = np.array(image.convert("RGB"), dtype=np.float32)
        stds = [img_np[:, :, c].std() for c in range(3)]
        best = int(np.argmax(stds))
        channel = img_np[:, :, best].astype(np.uint8)
        return Image.fromarray(channel, mode="L").convert("RGB")

    def run_ocr(image: Image.Image | None) -> str:
        if predictor[0] is None:
            return "No model loaded. Enter a model path above and click Load."
        if image is None:
            return "No image provided."
        try:
            return predictor[0].predict(image)
        except Exception as exc:
            return f"Inference failed: {exc}"

    initial_status = f"Model loaded: {os.path.basename(model_path)}" if model_path else "No model loaded."

    with gr.Blocks(title="OCR Predictor") as app:
        gr.Markdown("# OCR Predictor")
        gr.Markdown("Upload or paste a **pre-cropped** image of a single text line and click **Run OCR**.")

        with gr.Row():
            model_input = gr.Textbox(
                label="ONNX model path",
                value=model_path or "",
                placeholder="e.g. exports/model.onnx",
                scale=4,
            )
            load_btn = gr.Button("Load model", scale=1)

        model_status = gr.Textbox(
            label="Model status",
            value=initial_status,
            interactive=False,
        )

        image_input = gr.Image(
            label="Input image (upload or paste)",
            type="pil",
            sources=["upload", "clipboard"],
        )

        with gr.Row():
            bw_btn = gr.Button("Convert to BW")
            run_btn = gr.Button("▶  Run OCR", variant="primary")

        result = gr.Textbox(
            label="Predicted text",
            interactive=False,
        )

        load_btn.click(fn=load_model, inputs=model_input, outputs=model_status)
        bw_btn.click(fn=convert_to_bw, inputs=image_input, outputs=image_input)
        run_btn.click(fn=run_ocr, inputs=image_input, outputs=result)

    return app


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR GUI")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to ONNX model (e.g. exports/model.onnx).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on (default: 7860).",
    )
    args = parser.parse_args()

    app = build_app(model_path=args.model)
    app.launch(server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
