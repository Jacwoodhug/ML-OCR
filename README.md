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

### 3. Collect system fonts

Scans Windows fonts, filters out symbol/icon fonts (Wingdings, Webdings, etc.) via blocklist + automated render test, and caches the approved font list.

```powershell
python scripts/collect_fonts.py
```

Output: `data/fonts/fonts.json`

### 4. Download background textures

Downloads the DTD (Describable Textures Dataset, ~600MB) for background variety.

```powershell
python scripts/download_backgrounds.py
```

Output: `data/backgrounds/` (~5640 texture images)

## Train

### 5. Pre-generate training data (recommended)

Pre-generating avoids the CPU bottleneck of on-the-fly rendering during training:

```powershell
python scripts/pregenerate.py --count 500000 --output data/train
```

This is resumable — if interrupted, re-run the same command and it picks up where it left off.

### 6. Start training

With pre-generated data (faster):

```powershell
python scripts/train.py --data-dir data/train
```

Or on-the-fly generation (no disk space needed but slower):

```powershell
python scripts/train.py
```

Options:
- `--config CONFIG` — Config file path (default: `config/default.yaml`)
- `--resume CHECKPOINT` — Resume from a saved checkpoint
- `--steps N` — Override max training iterations (default: 500K)
- `--data-dir DIR` — Pre-generated training data directory

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

```powershell
python -m src.inference.export_onnx --checkpoint checkpoints/best.pt
```

Options:
- `--output PATH` — ONNX output path (default: `exports/model.onnx`)
- `--config CONFIG` — Config file

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
│   ├── collect_fonts.py         # Font enumeration & filtering
│   ├── download_backgrounds.py  # DTD dataset download
│   ├── pregenerate.py           # Pre-generate training data to disk
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   └── benchmark.py             # Inference speed benchmarking
├── requirements.txt
└── PLAN.md                      # Full project plan & design decisions
```

## Quick Smoke Test

To verify everything works before committing to a long training run:

```powershell
python scripts/smoke_train.py
```

Runs 50 training steps with a small batch size. You should see the loss drop from ~11 to ~4.
