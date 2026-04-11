"""
generate_samples.py
────────────────────
CS318 GAN – Image Generation
Generate N fake images from a saved generator checkpoint.
Used as input for FID computation.

Usage:
    python generate_samples.py --model dcgan --n 10000
    python generate_samples.py --model vanilla --n 10000
"""

import argparse
import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm

# ── Import model classes from the training script ──────────────────────────────
from gan_mnist import (
    VanillaGenerator, DCGenerator, DEFAULT_CFG, device
)

# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(model_type: str, n: int = 10000, cfg: dict = None):
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    out_dir = f"results/{model_type}/generated"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load generator ────────────────────────────────────────────────────────
    ckpt_path = f"results/{model_type}/generator.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No saved generator found at '{ckpt_path}'.\n"
            f"Run training first:  python gan_mnist.py --model {model_type}"
        )

    G = (VanillaGenerator(cfg) if model_type == "vanilla" else DCGenerator(cfg)).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()

    print(f"[INFO] Generating {n} images from {model_type.upper()} generator …")
    batch_size = 256
    saved = 0

    with torch.no_grad():
        pbar = tqdm(total=n, unit="img", ncols=80)
        while saved < n:
            this_batch = min(batch_size, n - saved)
            z    = torch.randn(this_batch, cfg["latent_dim"], device=device)
            imgs = G(z)
            for img in imgs:
                save_image(img, f"{out_dir}/{saved:05d}.png", normalize=True)
                saved += 1
            pbar.update(this_batch)
        pbar.close()

    print(f"[DONE] {saved} images saved → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake MNIST images for FID")
    parser.add_argument("--model", choices=["vanilla", "dcgan"], default="dcgan")
    parser.add_argument("--n",     type=int, default=10000,
                        help="Number of images to generate (default: 10000)")
    args = parser.parse_args()
    generate_samples(args.model, args.n)
