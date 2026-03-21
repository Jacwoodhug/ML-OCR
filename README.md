# ML-OCR

A from-scratch OCR system trained entirely on synthetic data. Uses a CRNN architecture (MobileNetV3-Small + BiLSTM + CTC) for fast inference on both CPU and GPU.

## Setup

### 1. Create virtual environment (Python 3.11)

```powershell
& "C:\Users\jacwo\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

PyTorch with CUDA 12.1 (for 3090):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Prepare Data

### 3. Collect fonts

**Option A — Windows system fonts** (no API key needed)

Scans Windows fonts, filters out symbol/icon fonts (Wingdings, Webdings, etc.) via blocklist + automated render test, and caches the approved font list.

```powershell
python scripts/collect_fonts.py
```

Output: `data/fonts/fonts.json`

**Option B — Google Fonts** (recommended for more variety)

Downloads the most popular Latin-subset fonts from Google Fonts. Requires a free API key — get one at [console.cloud.google.com](https://console.cloud.google.com/) under APIs & Services → Credentials (no billing required).

Add your key to `.env`:

```
GOOGLE_FONTS_API_KEY=your_api_key_here
```

Then run:

```powershell
python scripts/download_google_fonts.py --top-n 200
```

Output: `data/fonts/google/` (.ttf files) + `data/fonts/google_fonts.json`

Pass `--out-json data/fonts/fonts.json` to write directly to the path `SynthGenerator` expects, or keep both and merge them manually.

Options:
- `--top-n N` — Number of top fonts to fetch by popularity (default: 200)
- `--output DIR` — Directory to save .ttf files (default: `data/fonts/google`)
- `--out-json PATH` — Output JSON path (default: `data/fonts/google_fonts.json`)

### 4. Download background textures

Downloads the DTD (Describable Textures Dataset, ~600MB) for background variety.

```powershell
python scripts/download_backgrounds.py
```

Output: `data/backgrounds/` (~5640 texture images)

## Train

### 5. Pre-generate training data (recommended)

Pre-generating avoids the CPU bottleneck of on-the-fly rendering during training.

Generate clean images (augmentation applied at training time):

```powershell
python scripts/pregenerate.py --count 500000 --output data/train
```

Or bake augmentations in at generation time for maximum training speed (workers just decode PNGs):

```powershell
python scripts/pregenerate.py --count 500000 --output data/train --augment
```

> Generation is resumable — if interrupted, re-run the same command and it picks up where it left off.

### 6. Start training

With pre-generated clean data (augmentation runs at load time):

```powershell
python scripts/train.py --data-dir data/train
```

With pre-augmented data (fastest — no per-sample CPU work during training):

```powershell
python scripts/train.py --data-dir data/train --no-augment
```

Or on-the-fly generation (no disk space needed but slowest):

```powershell
python scripts/train.py
```

Options:
- `--config CONFIG` — Config file path (default: `config/default.yaml`)
- `--resume CHECKPOINT` — Resume from a saved checkpoint
- `--steps N` — Override max training iterations (default: 500K)
- `--data-dir DIR` — Pre-generated training data directory
- `--no-augment` — Disable runtime augmentation (use when data was pre-generated with `--augment`)

The default config trains for 500K iterations with batch size 256, mixed precision (FP16) on GPU. Training data is generated on-the-fly — no dataset to download.

**Monitor training** with TensorBoard:

```powershell
tensorboard --logdir runs
```

Checkpoints are saved to `checkpoints/`. The best model (by validation CER) is saved as `checkpoints/best.pt`.

### Config overrides

Edit `config/default.yaml` to change:
- `training.batch_size` — reduce if you hit OOM (try 128 or 64)
- `training.max_iterations` — total training steps
- `training.val_interval` — how often to validate (in steps)
- `data.img_height` — training image height (default 32)
- `model.lstm_hidden_size` — LSTM width (default 256)

## Evaluate

```powershell
python scripts/evaluate.py --checkpoint checkpoints/best.pt
```

Options:
- `--config CONFIG` — Config file (default: `config/default.yaml`)
- `--val-dir DIR` — Custom validation data directory
- `--beam-width N` — Beam search width (default 0 = greedy decoding)
- `--show-samples N` — Number of sample predictions to print (default 20)

Reports Character Error Rate (CER), Word Error Rate (WER), and sequence-level accuracy.

## Export & Deploy

### Export to ONNX

First install the `onnx` package if you haven't already:

```powershell
pip install onnx
```

```powershell
python src/inference/export_onnx.py --checkpoint checkpoints/best.pt --output exports/model.onnx
```

Options:
- `--output PATH` — ONNX output path (default: `exports/model.onnx`)
- `--config CONFIG` — Config file

### GUI Inference

Install Gradio if you haven't already:

```powershell
pip install gradio
```

Launch the graphical interface to run OCR on any image interactively:

```powershell
python scripts/predict_gui.py --model exports/model.onnx
```

The `--model` flag is optional — you can also browse for the ONNX file inside the app.

**Loading an image:**
- **Open image…** — file picker (PNG, JPG, BMP, TIFF, WebP)
- **Paste (Ctrl+V)** — paste a screenshot or copied image directly from the clipboard
- **Click the preview area** — also opens the file picker

**Running inference:**
- Click **▶ Run OCR** to recognise the loaded image
- The predicted text appears in the result box at the bottom
- Click **Copy** to copy the result to the clipboard

> Images should be pre-cropped to the text region. The model expects a single line of text.

### Benchmark inference speed

```powershell
python scripts/benchmark.py --checkpoint checkpoints/best.pt
python scripts/benchmark.py --onnx exports/model.onnx
```

Measures latency across various input widths on both CPU and GPU.

## Project Structure

```
ML-OCR/
├── config/default.yaml          # All hyperparameters
├── data/
│   ├── fonts/fonts.json         # Cached approved font list
│   ├── backgrounds/             # DTD texture images
│   └── val/                     # Pre-generated validation set
├── src/
│   ├── data/
│   │   ├── alphabet.py          # Character set (a-z, A-Z, 0-9, punctuation)
│   │   ├── augmentations.py     # Albumentations pipeline
│   │   ├── dataset.py           # PyTorch Dataset + variable-width collate
│   │   └── synth_generator.py   # On-the-fly synthetic image generation
│   ├── model/
│   │   ├── backbone.py          # MobileNetV3-Small feature extractor
│   │   ├── sequence.py          # BiLSTM layers
│   │   ├── crnn.py              # Full CRNN model
│   │   └── ctc_decoder.py       # Greedy + beam search decoders
│   ├── training/
│   │   ├── trainer.py           # Training loop (AMP, checkpointing)
│   │   └── metrics.py           # CER, WER, accuracy
│   └── inference/
│       ├── export_onnx.py       # PyTorch → ONNX export
│       └── predictor.py         # ONNX Runtime inference wrapper
├── scripts/
│   ├── collect_fonts.py         # Font enumeration & filtering (Windows system fonts)
│   ├── download_google_fonts.py # Download top fonts from Google Fonts API
│   ├── download_backgrounds.py  # DTD dataset download
│   ├── pregenerate.py           # Pre-generate training data to disk
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── predict_gui.py           # GUI for interactive inference
│   └── benchmark.py             # Inference speed benchmarking
├── .env                         # API keys (not committed)
├── requirements.txt
└── PLAN.md                      # Full project plan & design decisions
```

## Quick Smoke Test

To verify everything works before committing to a long training run:

```powershell
python scripts/smoke_train.py
```

Runs 50 training steps with a small batch size. You should see the loss drop from ~11 to ~4.
