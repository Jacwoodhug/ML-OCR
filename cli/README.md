# OCR CLI

Lightweight command-line OCR inference using an ONNX model. Self-contained — no dependency on the parent `src/` package.

## Requirements

- **Python 3.10+** (uses `str | None` union syntax)
- Packages listed in `requirements.txt`:
  ```
  onnxruntime
  numpy
  Pillow
  ```

Install with:
```bash
pip install -r cli/requirements.txt
```

For GPU acceleration, install `onnxruntime-gpu` instead of `onnxruntime`.

## Setup

Place your `.onnx` model file in the `cli/` folder and it will be detected automatically. Otherwise, pass a path with `--model`.

## Usage

```bash
# auto-detects model in cli/ folder
python cli/ocr.py image.png

# multiple images
python cli/ocr.py img1.png img2.png img3.png

# all images in a folder
python cli/ocr.py images/*.png

# explicit model path
python cli/ocr.py --model path/to/model.onnx image.png

# skip automatic black-and-white conversion
python cli/ocr.py --no-bw image.png

# use GPU
python cli/ocr.py --gpu image.png
```

## Output

One line per image with the predicted text and inference time:
```
image.png: Hello World  (12.3ms)
```

## Black-and-white conversion

By default, color images are converted to grayscale by selecting the RGB channel with the highest contrast (standard deviation). This matches the behaviour of the Gradio GUI. Use `--no-bw` to skip this step if your images are already preprocessed.
