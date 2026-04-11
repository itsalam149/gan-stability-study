"""
visualize.py
────────────
CS318 GAN – Standalone Visualisation Tool

Generates publication-quality figures from completed training runs:
  1. Loss curves (G vs D) with smoothing
  2. Training progression panels (epoch 1 → 10 → 50 → 100)
  3. Side-by-side Vanilla vs DCGAN comparison

Usage:
    python visualize.py --model dcgan
    python visualize.py --model vanilla
    python visualize.py --compare          # needs BOTH models trained
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ─── Helpers ──────────────────────────────────────────────────────────────────

def smooth(values, window=5):
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def load_losses(model_type: str):
    """Re-read loss values from the losses.png is not possible;
    instead, we store them as a simple .npy during training.
    Falls back to plotting from existing losses.png if .npy absent."""
    npy_g = f"results/{model_type}/g_losses.npy"
    npy_d = f"results/{model_type}/d_losses.npy"
    if os.path.exists(npy_g) and os.path.exists(npy_d):
        return np.load(npy_g), np.load(npy_d)
    return None, None


def plot_loss_curve(model_type: str):
    g, d = load_losses(model_type)
    if g is None:
        print(f"[WARN] No .npy loss arrays found for {model_type}. "
              f"Losses are already saved as losses.png during training.")
        return

    out = f"results/{model_type}/losses_detailed.png"
    epochs = np.arange(1, len(g) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Panel 1: raw + smoothed ──
    ax = axes[0]
    ax.plot(epochs, g, alpha=0.2, color="#e74c3c")
    ax.plot(epochs, smooth(g), label="G Loss (smoothed)", color="#e74c3c", lw=2)
    ax.plot(epochs, d, alpha=0.2, color="#2980b9")
    ax.plot(epochs, smooth(d), label="D Loss (smoothed)", color="#2980b9", lw=2)
    ax.set_title(f"{model_type.upper()} – G vs D Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Panel 2: G/D ratio ──
    ax2 = axes[1]
    ratio = np.array(g) / (np.array(d) + 1e-8)
    ax2.plot(epochs, smooth(ratio), color="#8e44ad", lw=2)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Equilibrium")
    ax2.set_title("G/D Loss Ratio", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Ratio")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{model_type.upper()} GAN – Training Analysis", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Detailed loss curve → {out}")


def plot_progression(model_type: str, milestones=(1, 10, 50, 100)):
    sample_dir = f"results/{model_type}/samples"
    out_path   = f"results/{model_type}/progression.png"

    found = []
    for ep in milestones:
        p = os.path.join(sample_dir, f"epoch_{ep:04d}.png")
        if os.path.exists(p):
            found.append((ep, p))

    if not found:
        print(f"[WARN] No sample images found in {sample_dir}/")
        return

    n = len(found)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (ep, p) in zip(axes, found):
        img = np.array(Image.open(p).convert("RGB"))
        ax.imshow(img)
        ax.set_title(f"Epoch {ep}", fontsize=13, fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle(f"{model_type.upper()} GAN – Training Progression (MNIST)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Progression figure → {out_path}")


def plot_comparison():
    """Side-by-side final samples: Vanilla vs DCGAN."""
    v_samples = [f for f in sorted(os.listdir("results/vanilla/samples"))
                 if f.endswith(".png")]
    d_samples = [f for f in sorted(os.listdir("results/dcgan/samples"))
                 if f.endswith(".png")]

    if not v_samples or not d_samples:
        print("[WARN] Need both vanilla and dcgan sample directories populated.")
        return

    v_img = np.array(Image.open(f"results/vanilla/samples/{v_samples[-1]}").convert("RGB"))
    d_img = np.array(Image.open(f"results/dcgan/samples/{d_samples[-1]}").convert("RGB"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(v_img)
    axes[0].set_title("Vanilla GAN", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(d_img)
    axes[1].set_title("DCGAN", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Vanilla GAN vs DCGAN – Final Generated Samples",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    out = "results/comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DONE] Comparison figure → {out}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation tool for GAN results")
    parser.add_argument("--model",   choices=["vanilla", "dcgan"], default="dcgan")
    parser.add_argument("--compare", action="store_true",
                        help="Generate side-by-side Vanilla vs DCGAN comparison")
    parser.add_argument("--milestones", nargs="+", type=int, default=[1, 10, 50, 100],
                        help="Epoch checkpoints for progression figure")
    args = parser.parse_args()

    if args.compare:
        plot_comparison()
    else:
        plot_loss_curve(args.model)
        plot_progression(args.model, milestones=args.milestones)
