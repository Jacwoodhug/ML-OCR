## Plan: Synthetic OCR Model from Scratch

Build a fast, from-scratch OCR system trained entirely on synthetic data. Phase 1 delivers a single-line text recognition model. Phase 2 adds text detection for multi-line support.

**Architecture**: CRNN (MobileNetV3-Small backbone + BiLSTM + CTC decoder) in PyTorch, exported to ONNX for fast cross-platform inference.

---

### Phase 1: Single-Line Text Recognition

**Step 1 — Project scaffolding & dependencies** (no dependencies)
- Create project directory structure (see below)
- `requirements.txt`: torch, torchvision, Pillow, albumentations, onnxruntime, pyyaml, tqdm, tensorboard, editdistance
- Config system (`config/default.yaml`) defining charset, model hyperparams, training schedules, data generation params

**Step 2 — Font collection** (no dependencies)
- Script `scripts/collect_fonts.py`: enumerate Windows system fonts from `C:\Windows\Fonts\`, filter to `.ttf`/`.otf`
- **Hard blocklist** (by font family name substring match): Wingdings, Webdings, Symbol, Marlett, Segoe MDL2 Assets, HoloLens MDL2 Assets, Segoe Fluent Icons, Segoe UI Emoji, Segoe Chess, Bookshelf Symbol, Directions MT, Holidays MT, Keystrokes MT, Map Symbols, Parties MT, Vacation MT, Transport MT, RefSpecialty, Cariadings, Almanac MT, McZee, Temp Installer Font
- **Automated render test**: render "ABCabc123!?@#" with each candidate font, compute per-glyph similarity — reject fonts where glyphs are blank, identical, or don't vary (catches symbol fonts the blocklist misses, plus broken/corrupt fonts)
- Categorize fonts by style (serif, sans-serif, monospace, decorative) for balanced sampling
- Optionally download a subset of Google Fonts (Noto family for coverage)

**Step 3 — Background image dataset** (no dependencies, *parallel with steps 1-2*)
- Script `scripts/download_backgrounds.py`: download DTD (Describable Textures Dataset, ~600MB, 5640 images, freely available) as the stock-photo/texture source
- Store in `data/backgrounds/`

**Step 4 — Synthetic data generator** (depends on steps 2-3)
- `src/data/synth_generator.py`: on-the-fly generator (no pre-generation, infinite variety)
- **Character set** (`src/data/alphabet.py`): a-z, A-Z, 0-9, and common punctuation `. , ! ? - : ; ' " ( ) @ # $ % & + = / \ [ ] { } _ ~` (approximately 85-90 classes + CTC blank)
- **Text generation**: random strings 1-50 chars, sampled from charset with configurable word/random-char mix (to produce realistic-ish word patterns)
- **Background pipeline** (randomly chosen per sample):
  - Solid: random RGB
  - Gradient: linear or radial, 2-3 random colors, random angle
  - Texture: random crop from DTD dataset, resized to target dimensions
- **Rendering**: Pillow `ImageDraw.text()`, random font, random font size (scaled to image height), random text color (any RGB)
- **Augmentation** (`src/data/augmentations.py`, using albumentations):
  - Gaussian blur, motion blur
  - Gaussian noise, JPEG compression artifacts
  - Slight rotation (±5°), slight perspective warp
  - Color jitter (brightness, contrast, saturation)
  - Random shadows / lighting overlays
- **Output**: images resized to fixed height=32px, variable width (preserving aspect ratio, clamped to min 32px / max 800px wide)
- PyTorch Dataset (`src/data/dataset.py`) wrapping the generator with a custom collate function that pads variable-width images within each batch

**Step 5 — CRNN model** (parallel with step 4)
- `src/model/backbone.py`: MobileNetV3-Small (from torchvision), modified to output features at stride 4 (giving W/4 timesteps). Remove classification head, keep feature maps. Output shape: `(B, C, 1, W')`.
- `src/model/sequence.py`: 2-layer BiLSTM, input dim = backbone channels, hidden size 256 → output 512-dim per timestep
- `src/model/crnn.py`: Compose backbone → squeeze height → BiLSTM → Linear(512, num_classes+1). Forward returns log-softmax over classes per timestep.
- `src/model/ctc_decoder.py`: Greedy decoder (argmax → collapse repeats → remove blanks) and optional beam search decoder

**Step 6 — Training pipeline** (depends on steps 4-5)
- `scripts/train.py`: Main training script
- `src/training/trainer.py`: Training loop with:
  - Loss: `torch.nn.CTCLoss(blank=0, zero_infinity=True)`
  - Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
  - LR schedule: OneCycleLR (max_lr=1e-3, warmup 5% of training)
  - Mixed precision: `torch.cuda.amp.GradScaler` for FP16 training on 3090
  - Batch size: start with 256, adjust based on VRAM usage
  - Iterations: ~500K-1M steps (synthetic data is infinite, no "epochs")
  - Checkpointing: save best model by validation CER, periodic snapshots
  - TensorBoard logging: loss curves, CER/accuracy, sample predictions
- `src/training/metrics.py`: Character Error Rate (edit distance / target length), Word-level accuracy, Sequence-level accuracy
- **Validation set**: Pre-generate a fixed 5K-10K sample validation set at training start for consistent metrics
- **Curriculum learning** (optional but recommended): start with simple backgrounds (solid colors, clean fonts, larger text), gradually introduce more augmentation and harder backgrounds over training

**Step 7 — Evaluation** (depends on step 6)
- `scripts/evaluate.py`: Run model on validation set, report CER, WER, sequence accuracy
- Qualitative inspection: visualize predictions vs ground truth on random samples
- Optional: test on public STR benchmarks (IIIT5K, SVT, IC13, IC15) to gauge real-world performance

**Step 8 — ONNX export & speed optimization** (depends on step 6)
- `src/inference/export_onnx.py`: Export trained model to ONNX with dynamic axes for variable width
- `src/inference/predictor.py`: ONNX Runtime inference wrapper (supports both CPU and GPU via CUDAExecutionProvider)
- INT8 quantization via ONNX Runtime for CPU inference speed
- FP16 for GPU inference
- `scripts/benchmark.py`: Measure latency on CPU and GPU for various input widths

---

### Phase 2: Multi-Line / Text Detection (Future extension)

**Step 9 — Text detection model** (after Phase 1 is validated)
- Lightweight segmentation approach (simplified DBNet or CRAFT-like architecture)
- Also trained on synthetic data: render multi-line text on backgrounds, output segmentation masks for text regions
- Detects text line bounding boxes in arbitrary images

**Step 10 — Full pipeline** (depends on step 9)
- Detection → crop text lines → recognition model from Phase 1
- NMS and line ordering (top-to-bottom, left-to-right)
- End-to-end inference wrapper

---

### Project Structure

```
ML-OCR/
├── config/
│   └── default.yaml
├── data/
│   ├── fonts/                  # Cached font list/metadata
│   ├── backgrounds/            # DTD dataset images
│   └── val/                    # Pre-generated validation set
├── src/
│   ├── data/
│   │   ├── alphabet.py         # Character set definition + encoding
│   │   ├── dataset.py          # PyTorch Dataset + collate
│   │   ├── synth_generator.py  # On-the-fly synthetic image generator
│   │   └── augmentations.py    # Albumentations pipeline
│   ├── model/
│   │   ├── backbone.py         # MobileNetV3-Small feature extractor
│   │   ├── sequence.py         # BiLSTM layers
│   │   ├── crnn.py             # Full CRNN assembly
│   │   └── ctc_decoder.py      # Greedy + beam search decoders
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── metrics.py          # CER, WER, accuracy
│   └── inference/
│       ├── predictor.py        # ONNX Runtime inference wrapper
│       └── export_onnx.py      # PyTorch → ONNX export
├── scripts/
│   ├── collect_fonts.py        # Font enumeration & filtering
│   ├── download_backgrounds.py # DTD dataset download
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation entry point
│   └── benchmark.py            # Inference speed benchmarking
├── requirements.txt
└── README.md
```

### Verification
1. **Data sanity check**: Visually inspect 50+ synthetic samples — verify text is readable in "easy" mode, augmentation variety is sufficient, text labels match rendered text
2. **Model unit test**: Forward pass with random tensor, verify output shape is `(W', B, num_classes+1)` with correct W' for given input width
3. **Training smoke test**: Train for 1000 steps, verify loss decreases and CER improves on validation
4. **Overfitting test**: Train on a tiny fixed dataset (100 samples) until near-zero CER — confirms model has enough capacity and pipeline is correct
5. **ONNX parity test**: Compare ONNX model output vs PyTorch model output on same inputs (should be numerically close)
6. **Benchmark**: Measure inference latency — target <10ms per image on GPU, <50ms on CPU for typical inputs
7. **Full evaluation**: CER < 5% on synthetic validation set before moving to Phase 2

### Decisions
- **CRNN + CTC** chosen over attention/transformer models for fastest inference on CPU+GPU. CTC has no autoregressive decoding step (single forward pass).
- **MobileNetV3-Small** backbone chosen for speed. Can upgrade to ResNet-31 or EfficientNet if accuracy is insufficient.
- **On-the-fly data generation** instead of pre-generating to disk — infinite variety, no storage cost, minimal overhead.
- **DTD (Describable Textures Dataset)** for backgrounds — freely available, good texture variety, lightweight download.
- **Phase 2 (multi-line/detection)** deferred — get single-line recognition working well first, then add detection.
- **Scope exclusions**: No handwriting, no scene text detection in Phase 1, no language modeling / dictionary correction (can be added later as post-processing).

### Further Considerations
1. **Language model post-processing**: After Phase 1 training, we could add an optional dictionary/n-gram language model to correct predictions. This would improve real-world accuracy significantly but adds inference latency. Recommend deferring to after base model works.
2. **Font diversity**: Windows system fonts may be limited (~200 fonts). Downloading Google Fonts would add 1000+ fonts for much better generalization. Recommend doing this but it adds setup complexity.
3. **Reparameterization for speed**: After training, the BiLSTM could potentially be replaced with a lighter 1D convolution stack (RepVGG-style) for even faster inference. This is an advanced optimization to consider if speed targets aren't met.
