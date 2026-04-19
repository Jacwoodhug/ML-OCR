# ML-OCR

A from-scratch OCR system trained entirely on synthetic data using rendered fonts (not handwriting). Recognises single-line English text only. Uses a CRNN architecture (MobileNetV3-Small + BiLSTM + CTC) for fast inference on both CPU and GPU.

**Supported characters:** `a-z A-Z 0-9 . , ! ? - : ; ' " ( ) @ # $ % & + = / _ ~ [ ] { } \ | < > ^ ``

## Benchmark (n=100, CPU)

| Metric | This Model | EasyOCR | Tesseract |
|---|---|---|---|
| Exact match (%) | **87.0%** | 40.0% | 33.0% |
| Char accuracy (%) | **98.4%** | 81.8% | 70.1% |
| Avg per image (ms) | **5.9** | 134.9 | 125.4 |
| Speed vs Custom | **1.0x** | 22.7x | 21.1x |

> **Caveat:** This benchmark was tested with black and white synthetic data created using the same method as the training data, with 80% photo backgrounds and 20% color backgrounds. Results on real-world images may differ.

## Using the Model

To use the pre-trained model, **download the latest release** from the [Releases page](../../releases/latest). The release includes the ONNX model, a lightweight CLI script, and its own README with instructions for running inference.

Everything below this section is only needed if you want to train the model yourself.

---

## Training Setup

### 1. Create virtual environment (Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

PyTorch with CUDA 12.1:

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

Or bake augmentations in at generation time for maximum training speed (workers just decode images):

```powershell
python scripts/pregenerate.py --count 500000 --output data/train --augment
```

By default, images use solid/gradient backgrounds only. Use `--bg-ratio` to include a percentage of real background textures from `data/backgrounds/`:

```powershell
python scripts/pregenerate.py --count 500000 --output data/train --bg-ratio 20
```

For grayscale, high-contrast output (ideal for documents/receipts):

```powershell
python scripts/pregenerate.py --count 500000 --output data/train --bw
```

Or generate directly to LMDB (skips intermediate files, faster for large datasets):

```powershell
python scripts/pregenerate.py --lmdb data/train.lmdb --count 500000 --bw --bg-ratio 60
```

A validation set (10K samples by default) is automatically generated in a `val/` subdirectory alongside the training data, which `train.py` auto-detects:

```powershell
python scripts/pregenerate.py --lmdb data/train.lmdb --count 500000 --bw --bg-ratio 60
```

Options:
- `--count N` — Number of training samples to generate
- `--val-count N` — Number of validation samples to generate (default: 10000, saved to `val/` subdirectory)
- `--no-val` — Skip validation set generation
- `--output DIR` — Output directory (for file-based output)
- `--lmdb PATH` — Write directly to LMDB format (skips intermediate files)
- `--config CONFIG` — Config file (default: `config/default.yaml`)
- `--augment` — Bake augmentations into saved images
- `--format {jpg,png}` — Image format (default: `jpg`)
- `--jpeg-quality N` — JPEG quality when `--format=jpg` (default: 90)
- `--google-fonts` — Use Google Fonts instead of system fonts
- `--bw` — Grayscale output with text/bg luminance contrast enforced
- `--bg-ratio PCT` — Percentage of images that use random backgrounds from `data/backgrounds/` (default: 0 = none)
- `--font-file PATH` — Path to a specific font file (.ttf/.otf). Can be repeated
- `--font-dir DIR` — Path to a directory of font files to use instead of the fonts cache
- `--workers N` — Number of worker processes (default: min(CPU count, 8))
- `--map-size-gb N` — LMDB map size in GB (default: 20, only used with `--lmdb`)

> Generation is resumable — if interrupted, re-run the same command and it picks up where it left off (both file-based and LMDB modes).

### 6. Convert to LMDB (optional, for file-based datasets)

If you already generated data as individual files, convert to LMDB for faster training I/O:

```powershell
python scripts/convert_to_lmdb.py --input data/train --output data/train.lmdb
```

### 7. Start training

With pre-generated data (augmentation runs at load time):

```powershell
python scripts/train.py --data-dir data/train
```

With an LMDB dataset (fastest I/O for large datasets):

```powershell
python scripts/train.py --lmdb data/train.lmdb
```

With pre-augmented data (no per-sample CPU work during training):

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
- `--steps N` — Override max training iterations (default: 500K). When used with `--reset-lr`, this is the number of **new** steps to train
- `--data-dir DIR` — Pre-generated training data directory
- `--lmdb PATH` — LMDB training dataset path (faster than `--data-dir` for large datasets)
- `--no-augment` — Disable runtime augmentation (use when data was pre-generated with `--augment`)
- `--bw` — Use grayscale, high-contrast generator for the validation set
- `--tag NAME` — Tag appended to checkpoint filenames (e.g. `--tag grayscale` → `best_grayscale.pt`)
- `--reset-lr` — Reset the learning rate schedule when resuming (for fine-tuning on new data). Loads model weights only, creates a fresh optimizer and LR schedule
- `--lr FLOAT` — Override max learning rate (default: 1e-3). Use a lower value for fine-tuning (see recommendations below)
- `--val-dir DIR` — Use a pre-generated validation set instead of auto-generating one. Generate with `pregenerate.py` to control background mix, fonts, etc.

The default config trains for 500K iterations with batch size 256, mixed precision (FP16) on GPU.

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

## Fine-Tuning for Specific Fonts

If you need high accuracy on a particular set of fonts (e.g. a specific document typeface, receipt font, or display font), you can fine-tune a pretrained model on data generated exclusively from those fonts.

### 1. Generate font-specific data

Use `--font-file` and/or `--font-dir` to restrict data generation to your target fonts. Combine with `--bw` to produce clean, grayscale, high-contrast images — ideal for documents and receipts where backgrounds are plain:

```powershell
# Single font, grayscale
python scripts/pregenerate.py --font-file path/to/consola.ttf --bw --count 20000 --output data/finetune_consolas

# Multiple specific fonts
python scripts/pregenerate.py --font-file font1.ttf --font-file font2.ttf --bw --count 20000 --output data/finetune_custom

# All fonts in a directory
python scripts/pregenerate.py --font-dir path/to/my-fonts/ --bw --count 20000 --output data/finetune_custom
```

The `--bw` flag produces grayscale images with enforced luminance contrast between text and background. This is well-suited for fine-tuning on document-like data where color and textured backgrounds aren't relevant.

Use `--bg-ratio` to include a percentage of real background textures if your target domain has more visual variety (e.g. `--bg-ratio 20` for 20% textured backgrounds).

### 2. Fine-tune from a pretrained checkpoint

Resume from your best general model and train on the font-specific data. Use `--reset-lr` to get a fresh learning rate schedule and `--tag` to keep the fine-tuned checkpoint separate:

```powershell
python scripts/train.py --data-dir data/finetune_consolas --resume checkpoints/best.pt --steps 50000 --tag consolas --bw --reset-lr
```

The `--reset-lr` flag loads only the model weights from the checkpoint and creates a fresh optimizer + LR schedule for the specified number of new steps. Without it, the scheduler state from the original training run is restored, which typically means a near-zero learning rate — too low for the model to adapt to new data.

**Important:** When fine-tuning, use `--lr` to lower the max learning rate. The default (1e-3) is designed for training from scratch and will destroy the pretrained weights:

```powershell
# Fine-tune with a lower learning rate
python scripts/train.py --resume checkpoints/best.pt --steps 50000 --tag consolas --bw --reset-lr --lr 1e-4
```

**Recommended learning rates:**

| Scenario | `--lr` | Notes |
|---|---|---|
| Training from scratch | `1e-3` (default) | Full OneCycle schedule |
| Fine-tuning (similar data) | `1e-4` | Small domain shift, e.g. adding backgrounds |
| Fine-tuning (different data) | `2e-4` – `5e-4` | Larger domain shift, e.g. new font families |
| Light adaptation | `5e-5` | Minor adjustments, low risk of forgetting |

Tips:
- **Dataset size:** 10K–50K samples is usually enough for fine-tuning on a handful of fonts.
- **Steps:** 20K–50K fine-tuning steps is a reasonable starting point; monitor CER on TensorBoard.
- **Matching validation to training data:** Use `--val-count` when generating data so the val set matches the training distribution. `train.py` auto-detects the `val/` subdirectory:
  ```powershell
  # Generate training + val data with 60% textured backgrounds
  python scripts/pregenerate.py --lmdb data/train-bg60.lmdb --count 500000 --bw --bg-ratio 60

  # Fine-tune — val set is auto-detected from the LMDB directory
  python scripts/train.py --resume checkpoints/best.pt --lmdb data/train-bg60.lmdb --reset-lr --lr 1e-4 --bw --tag bg-finetune
  ```
- **Combining with `--augment`:** You can pre-bake augmentations (`--augment` on `pregenerate.py`) and then use `--no-augment` during training for maximum throughput.
- **Mixing data:** For best results, consider mixing font-specific data with some general data to avoid catastrophic forgetting. Generate both datasets and combine them into one directory.

The fine-tuned checkpoint will be saved as `checkpoints/best_consolas.pt` (matching the `--tag`).

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
- Click **Run OCR** to recognise the loaded image
- The predicted text appears in the result box at the bottom
- Click **Copy** to copy the result to the clipboard

> Images should be pre-cropped to the text region. The model expects a single line of text.

### CLI Inference

A lightweight standalone script (no dependency on `src/`) is provided in `cli/`:

```powershell
python cli/ocr.py image.png
```

Auto-detects `.onnx` models in the `cli/` folder. Supports `--gpu` for CUDA inference and batch processing of multiple images.

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
│   │   ├── dataset.py           # PyTorch Datasets (on-the-fly, disk, LMDB)
│   │   └── synth_generator.py   # Synthetic image generation engine
│   ├── model/
│   │   ├── backbone.py          # MobileNetV3-Small feature extractor
│   │   ├── sequence.py          # BiLSTM layers
│   │   ├── crnn.py              # Full CRNN model
│   │   └── ctc_decoder.py       # Greedy + beam search decoders
│   ├── training/
│   │   ├── trainer.py           # Training loop (AMP, checkpointing, torch.compile)
│   │   └── metrics.py           # CER, WER, accuracy
│   └── inference/
│       ├── export_onnx.py       # PyTorch → ONNX export
│       └── predictor.py         # ONNX Runtime inference wrapper
├── scripts/
│   ├── collect_fonts.py         # Font enumeration & filtering (Windows system fonts)
│   ├── download_google_fonts.py # Download top fonts from Google Fonts API
│   ├── download_backgrounds.py  # DTD dataset download
│   ├── pregenerate.py           # Pre-generate training data to disk
│   ├── convert_to_lmdb.py      # Convert dataset to LMDB format
│   ├── convert_to_jpeg.py       # Batch convert PNG dataset to JPEG
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   ├── predict_gui.py           # Gradio GUI for interactive inference
│   ├── smoke_train.py           # Quick 50-step smoke test
│   └── benchmark.py             # Inference speed benchmarking
├── cli/
│   └── ocr.py                   # Lightweight standalone CLI for inference
├── .env                         # API keys (not committed)
├── requirements.txt
└── PLAN.md                      # Project plan & design decisions
```

## Quick Smoke Test

To verify everything works before committing to a long training run:

```powershell
python scripts/smoke_train.py
```

Runs 50 training steps with a small batch size. You should see the loss drop from ~11 to ~4.
