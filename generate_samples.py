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
    VanillaGenerator, DCGenerator, cDCGenerator, DEFAULT_CFG, device
)

# ─────────────────────────────────────────────────────────────────────────────

def generate_samples(model_type: str, dataset_name: str = "mnist", n: int = 10000, class_label: int = None, cfg: dict = None):
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    out_dir = f"results/{model_type}_{dataset_name}/generated" if dataset_name == "fashion_mnist" else f"results/{model_type}/generated"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load generator ────────────────────────────────────────────────────────
    ckpt_path = f"results/{model_type}_{dataset_name}/generator.pth" if dataset_name == "fashion_mnist" else f"results/{model_type}/generator.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No saved generator found at '{ckpt_path}'.\n"
            f"Run training first:  python gan_mnist.py --model {model_type} --dataset {dataset_name}"
        )

    if model_type == "vanilla":
        G = VanillaGenerator(cfg).to(device)
    elif model_type == "cdcgan":
        G = cDCGenerator(cfg).to(device)
    else:
        G = DCGenerator(cfg).to(device)
        
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.eval()

    print(f"[INFO] Generating {n} images from {model_type.upper()} generator …")
    batch_size = 256
    saved = 0

    with torch.no_grad():
        pbar = tqdm(total=n, unit="img", ncols=80)
        while saved < n:
            this_batch = min(batch_size, n - saved)
            z = torch.randn(this_batch, cfg["latent_dim"], device=device)
            
            if model_type == "cdcgan":
                if class_label is not None:
                    labels = torch.full((this_batch,), class_label, dtype=torch.long, device=device)
                else:
                    labels = torch.randint(0, cfg["num_classes"], (this_batch,), device=device)
                imgs = G(z, labels)
            else:
                imgs = G(z)
                
            for img in imgs:
                save_image(img, f"{out_dir}/{saved:05d}.png", normalize=True)
                saved += 1
            pbar.update(this_batch)
        pbar.close()

    print(f"[DONE] {saved} images saved → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fake images for FID")
    parser.add_argument("--model", choices=["vanilla", "dcgan", "cdcgan"], default="dcgan")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--n",     type=int, default=10000,
                        help="Number of images to generate (default: 10000)")
    parser.add_argument("--class_label", type=int, default=None,
                        help="Generate only this specific class (0-9) if using cDCGAN")
    args = parser.parse_args()
    generate_samples(args.model, args.dataset, args.n, args.class_label)
