# CS318 – GAN-Based Image Generation

**Authors:** Faqre Alam (23/CS/150) · Ekansh Agrawal (23/CS/149)  
**Course:** CS318 · Deep Learning  
**Dataset:** MNIST  

---

## Project Overview

End-to-end pipeline implementing **Vanilla GAN** (fully-connected) and **DCGAN** (convolutional) on MNIST. The pipeline covers:

- Training with label smoothing, tqdm logging, and periodic checkpointing
- Image grid sampling every 10 epochs
- Loss curve visualisation (G vs D)
- Training progression figure (epoch 1 → 10 → 50 → 100)
- Fréchet Inception Distance (FID) computation

---

## Repository Structure

```
dl-project/
├── gan_mnist.py          ← main training script
├── generate_samples.py   ← generate N fake images from checkpoint
├── compute_fid.py        ← FID computation pipeline
├── visualize.py          ← plots & comparison figures
├── requirements.txt
├── configs/
│   ├── vanilla.yaml
│   └── dcgan.yaml
├── results/              ← auto-created by training
│   ├── vanilla/
│   │   ├── samples/      ← epoch_0001.png … epoch_0100.png
│   │   ├── checkpoints/  ← periodic .pth files
│   │   ├── losses.png
│   │   ├── progression.png
│   │   └── fid_score.txt
│   └── dcgan/            ← same structure
└── paper/
    └── report.md
```

---

## Setup

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick Smoke-Test (5 epochs)

Verify everything works before the full run:

```bash
python gan_mnist.py --model vanilla --epochs 5
python gan_mnist.py --model dcgan   --epochs 5
```

Outputs created in `results/vanilla/` and `results/dcgan/`.

---

## Full Training (100 epochs)

### Option A — CLI flags
```bash
python gan_mnist.py --model vanilla --epochs 100
python gan_mnist.py --model dcgan   --epochs 100
```

### Option B — YAML config (recommended for reproducibility)
```bash
python gan_mnist.py --config configs/vanilla.yaml
python gan_mnist.py --config configs/dcgan.yaml
```

Training output example:
```
Epoch [   1/100]  D loss: 0.6821  G loss: 0.7432  (12.3s)
  → Sample grid saved → results/dcgan/samples/epoch_0001.png
Epoch [  10/100]  D loss: 0.4912  G loss: 1.1234  (11.8s)
  → Sample grid saved → results/dcgan/samples/epoch_0010.png
  → Checkpoint saved   → results/dcgan/checkpoints/epoch_0025.pth
...
```

---

## FID Score

FID (Fréchet Inception Distance) — lower is better.

```bash
# Step 1: Generate 10k fake images
python generate_samples.py --model dcgan --n 10000

# Step 2 & 3: Export real images + compute FID (automated)
python compute_fid.py --model dcgan
```

Score written to `results/dcgan/fid_score.txt`.

---

## Visualisation

```bash
# Individual model: loss curve + progression panels
python visualize.py --model dcgan
python visualize.py --model vanilla

# Side-by-side Vanilla vs DCGAN comparison (needs both trained)
python visualize.py --compare
```

---

## Expected Results

| Model | FID (↓) | Visual Quality |
|-------|---------|----------------|
| Vanilla GAN | ~60–80 | Blurry digits, some mode collapse |
| DCGAN | ~15–35 | Sharp, well-formed digits |

> FID values are approximate. Actual values depend on hardware and random seed.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dim | 100 |
| Batch size | 128 |
| Learning rate | 0.0002 |
| Optimiser | Adam (β₁=0.5, β₂=0.999) |
| Label smoothing | 0.9 |
| Epochs | 100 |
| Sample interval | every 10 epochs |
| Checkpoint interval | every 25 epochs |
# gan-stability-study
