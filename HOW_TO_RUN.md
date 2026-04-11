# 🧠 CS318 – GAN-Based Image Generation
## Complete How-to-Run Guide

> **Authors:** Faqre Alam (23/CS/150) · Ekansh Agrawal (23/CS/149)
> **Models:** Vanilla GAN · DCGAN
> **Dataset:** MNIST (auto-downloaded)

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Training](#4-training)
5. [Monitoring Training Progress](#5-monitoring-training-progress)
6. [Generating Samples](#6-generating-samples)
7. [Evaluation – FID Score](#7-evaluation--fid-score)
8. [Visualization](#8-visualization)
9. [Quick-Run Reference](#9-quick-run-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or later |
| pip | Latest recommended |
| GPU (optional) | CUDA-capable (training is ~10× faster on GPU) |
| Disk space | ~500 MB (model checkpoints + generated images) |

> **CPU is supported** — training will just take longer (~15–30 min/epoch on CPU vs ~1–2 min on GPU).

---

## 2. Project Structure

```
dl-project/
├── gan_mnist.py          ← Main training script (Vanilla GAN + DCGAN)
├── generate_samples.py   ← Generate fake images from a saved model
├── compute_fid.py        ← FID score evaluation
├── visualize.py          ← Standalone visualization tool
├── requirements.txt      ← Python dependencies
├── configs/
│   ├── dcgan.yaml        ← DCGAN hyperparameters
│   └── vanilla.yaml      ← Vanilla GAN hyperparameters
├── data/                 ← MNIST dataset (auto-downloaded)
│   └── real_mnist/       ← Exported real images for FID (auto-generated)
└── results/
    ├── vanilla/
    │   ├── samples/          ← Image grids saved every 10 epochs
    │   ├── checkpoints/      ← .pth files saved every 25 epochs
    │   ├── generator.pth     ← Final saved generator
    │   ├── discriminator.pth ← Final saved discriminator
    │   ├── losses.png        ← G vs D loss curve
    │   ├── progression.png   ← 4-panel training progression
    │   ├── g_losses.npy      ← Raw loss arrays
    │   └── d_losses.npy
    ├── dcgan/
    │   └── (same structure as vanilla/)
    └── comparison.png        ← Side-by-side comparison (after both trained)
```

---

## 3. Environment Setup

### Step 1 — Clone / Navigate to the project

```bash
cd /path/to/dl-project
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
```

### Step 3 — Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear in your terminal prompt.

### Step 4 — Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `torch` + `torchvision` (PyTorch deep learning framework)
- `pytorch-fid` (FID score computation)
- `matplotlib`, `numpy`, `tqdm`, `Pillow`, `pyyaml`

> **GPU users (CUDA):** If `pip install torch` installs a CPU-only build, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and use the appropriate install command for your CUDA version.

### Step 5 — Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output (example):
```
PyTorch: 2.x.x
CUDA: True   ← or False if using CPU
```

---

## 4. Training

### 4a. Train DCGAN (Recommended — better image quality)

**Option A — Use the YAML config (recommended):**
```bash
python gan_mnist.py --config configs/dcgan.yaml
```

**Option B — Use CLI flags directly:**
```bash
python gan_mnist.py --model dcgan --epochs 100
```

**Option C — Quick test run (fewer epochs):**
```bash
python gan_mnist.py --model dcgan --epochs 10
```

---

### 4b. Train Vanilla GAN (Fully-Connected MLP)

**Using the YAML config:**
```bash
python gan_mnist.py --config configs/vanilla.yaml
```

**Or via CLI:**
```bash
python gan_mnist.py --model vanilla --epochs 100
```

---

### 4c. All CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | `vanilla` / `dcgan` | `dcgan` | Architecture to train |
| `--epochs` | int | `100` | Number of training epochs |
| `--batch_size` | int | `128` | Batch size |
| `--config` | path | None | Path to YAML config file |

**Example — override batch size:**
```bash
python gan_mnist.py --model dcgan --epochs 50 --batch_size 64
```

---

### 4d. What Happens During Training

On first run, MNIST (~11 MB) is automatically downloaded to `./data/`.

Every **10 epochs**, a sample image grid (8×8 = 64 images) is saved:
```
results/dcgan/samples/epoch_0001.png
results/dcgan/samples/epoch_0010.png
...
```

Every **25 epochs**, a checkpoint is saved (resumable):
```
results/dcgan/checkpoints/epoch_0025.pth
results/dcgan/checkpoints/epoch_0050.pth
...
```

Console output per epoch looks like:
```
Epoch [  10/100]  D loss: 0.3821  G loss: 1.4502  (12.3s)
  → Sample grid saved → results/dcgan/samples/epoch_0010.png
```

At the end of training:
```
[DONE] Models saved → results/dcgan/generator.pth
[DONE] Loss curve   → results/dcgan/losses.png
  → Progression figure saved → results/dcgan/progression.png
[INFO] FID note      → results/dcgan/fid_note.txt
```

---

## 5. Monitoring Training Progress

While training runs, you can open the sample grids as they are being saved:

```bash
# macOS — open the latest sample grid
open results/dcgan/samples/

# Or view a specific epoch
open results/dcgan/samples/epoch_0010.png
```

After training completes, a `progression.png` is auto-saved showing epoch 1 → 10 → 50 → 100 side-by-side.

---

## 6. Generating Samples

After training is complete, run this script to generate **10,000 fake images** from your trained generator. These images are used as input for the FID score computation.

### Step-by-step

**Step 1 — Generate from DCGAN:**
```bash
python generate_samples.py --model dcgan --n 10000
```

**Step 2 — Generate from Vanilla GAN:**
```bash
python generate_samples.py --model vanilla --n 10000
```

You should see output like:
```
[INFO] Generating 10000 images from DCGAN generator …
100%|████████████████████| 10000/10000 [00:45<00:00, 220img/s]
[DONE] 10000 images saved → results/dcgan/generated/
```

Images are saved to:
```
results/dcgan/generated/
├── 00000.png
├── 00001.png
├── ...
└── 09999.png

results/vanilla/generated/
├── 00000.png
└── ...
```

### Available Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `dcgan` | Which trained model to load (`vanilla` or `dcgan`) |
| `--n` | `10000` | Number of images to generate |

> ⚠️ **Requires training to be complete first.** The script loads `results/<model>/generator.pth`.

---

## 7. Evaluation – FID Score

**Fréchet InceptionDistance (FID)** measures how close the generated images are to real MNIST images. **Lower is better.**

> ✅ **`compute_fid.py` (FIXED VERSION)** — Uses the `pytorch-fid` Python API directly (no subprocess). Real MNIST images are **exported automatically** before computing FID — you do not need to run `--export_real` separately.

### Step 1 — Generate fake images (if not done already)

```bash
python generate_samples.py --model dcgan --n 10000
python generate_samples.py --model vanilla --n 10000
```

### Step 2 — Compute FID score

**For DCGAN** (auto-exports real images on first run):
```bash
python compute_fid.py --model dcgan
```

**For Vanilla GAN:**
```bash
python compute_fid.py --model vanilla
```

Expected terminal output:
```
[INFO] Real MNIST already exists → data/real_mnist/
[INFO] Computing FID for DCGAN...
[RESULT] FID = 18.5423
[DONE] FID saved → results/dcgan/fid_score.txt
```

The FID score is printed to the terminal and saved to:
```
results/dcgan/fid_score.txt
results/vanilla/fid_score.txt
```

Contents of `fid_score.txt`:
```
Model : DCGAN
FID   : 18.5423
Real  : data/real_mnist (10000 images)
Fake  : results/dcgan/generated (10000 images)
```

### (Optional) Export real images only — one-time manual setup:

```bash
python compute_fid.py --export_real
```

This is **not required** — it happens automatically. Only run this explicitly if you want to pre-populate `data/real_mnist/` before training.

### One-liner (full FID pipeline for both models):
```bash
# Generate samples
python generate_samples.py --model dcgan --n 10000
python generate_samples.py --model vanilla --n 10000

# Compute FID (real images exported automatically on first call)
python compute_fid.py --model dcgan
python compute_fid.py --model vanilla
```

### Interpreting FID Scores

| FID Range | Meaning |
|-----------|---------|
| < 10 | Excellent — images very close to real |
| 10 – 50 | Good quality |
| 50 – 150 | Moderate — visible but imperfect |
| > 150 | Poor — training may need more epochs |

> 📊 DCGAN typically achieves **FID ≈ 10–30** on MNIST after 100 epochs.
> Vanilla GAN typically achieves **FID ≈ 50–100** after 100 epochs.

---

## 8. Visualization

Use `visualize.py` to generate publication-quality plots from a finished training run.

### Loss Curves + G/D Ratio (for DCGAN):
```bash
python visualize.py --model dcgan
```

Outputs:
- `results/dcgan/losses_detailed.png` — Smoothed G/D loss + ratio panel

### Training Progression Figure:
```bash
python visualize.py --model dcgan --milestones 1 10 50 100
```

Output:
- `results/dcgan/progression.png` — 4-panel figure showing image quality improvement

### Side-by-side Vanilla vs DCGAN Comparison:

> ⚠️ Requires **both** models to be fully trained first.

```bash
python visualize.py --compare
```

Output:
- `results/comparison.png` — Final samples from both architectures side-by-side

### All Visualization Flags

| Flag | Description |
|------|-------------|
| `--model vanilla\|dcgan` | Which model to visualize |
| `--compare` | Generate Vanilla vs DCGAN comparison |
| `--milestones 1 10 50 100` | Epoch checkpoints for progression figure |

---

## 9. Quick-Run Reference

### Full pipeline — DCGAN from scratch:

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Train DCGAN for 100 epochs
python gan_mnist.py --config configs/dcgan.yaml

# 3. Generate 10,000 fake images
python generate_samples.py --model dcgan --n 10000

# 4. Compute FID score (auto-exports real images on first run)
python compute_fid.py --model dcgan

# 5. Create detailed visualizations
python visualize.py --model dcgan
```

---

### Full pipeline — Both models + comparison:

```bash
source venv/bin/activate

# 1. Train both models
python gan_mnist.py --config configs/dcgan.yaml
python gan_mnist.py --config configs/vanilla.yaml

# 2. Generate 10,000 fake images from each model
python generate_samples.py --model dcgan --n 10000
python generate_samples.py --model vanilla --n 10000

# 3. Compute FID for both
#    (real MNIST images are exported automatically on first call)
python compute_fid.py --model dcgan
python compute_fid.py --model vanilla

# 4. Visualize both + generate comparison figure
python visualize.py --model dcgan
python visualize.py --model vanilla
python visualize.py --compare
```

---

### Quick test (5 epochs only — just to verify everything works):

```bash
source venv/bin/activate
python gan_mnist.py --model dcgan --epochs 5
```

---

## 10. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'torch'`
> You forgot to activate the virtual environment.
```bash
source venv/bin/activate
```

---

### ❌ `FileNotFoundError: No saved generator found at 'results/dcgan/generator.pth'`
> Training hasn't completed yet. Run training first:
```bash
python gan_mnist.py --model dcgan --epochs 100
```

---

### ❌ `No generated images found at 'results/dcgan/generated'`
> You need to generate samples before computing FID:
```bash
python generate_samples.py --model dcgan --n 10000
```

---

### ❌ CUDA out of memory
> Reduce batch size:
```bash
python gan_mnist.py --model dcgan --epochs 100 --batch_size 64
```

---

### ❌ Training is very slow (CPU)
> CPU training is expected to be slow. ~15–30 minutes per epoch for 100 epochs total.
> You can reduce epochs for testing:
```bash
python gan_mnist.py --model dcgan --epochs 20
```

---

### ❌ `RuntimeError: ... num_workers`
> On some macOS systems, multiprocessing causes issues. Edit `configs/dcgan.yaml` and set:
```yaml
num_workers: 0
```

---

### 📁 Output Files Summary

| File | Description |
|------|-------------|
| `results/<model>/generator.pth` | Final trained generator weights |
| `results/<model>/discriminator.pth` | Final trained discriminator weights |
| `results/<model>/checkpoints/epoch_XXXX.pth` | Intermediate checkpoints |
| `results/<model>/samples/epoch_XXXX.png` | Image grids saved every 10 epochs |
| `results/<model>/losses.png` | G vs D loss curve |
| `results/<model>/losses_detailed.png` | Detailed loss + ratio figure |
| `results/<model>/progression.png` | Training progression (4 panels) |
| `results/<model>/generated/` | 10k fake images for FID |
| `results/<model>/fid_score.txt` | FID score result |
| `results/comparison.png` | Vanilla vs DCGAN side-by-side |
| `data/real_mnist/` | Exported real MNIST images for FID |

---

*Happy training! 🚀*
