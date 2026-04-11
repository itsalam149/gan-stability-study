"""
compute_fid.py
──────────────
CS318 GAN – FID Score Computation (FIXED VERSION)

Uses pytorch-fid (Python API) to compute Fréchet Inception Distance.

Steps:
  1. Export real MNIST images → data/real_mnist/
  2. Load generated images → results/<model>/generated/
  3. Compute FID directly (no subprocess parsing issues)
  4. Save score → results/<model>/fid_score.txt

Usage:
    # Export real MNIST images only (one-time setup):
    python compute_fid.py --export_real

    # Compute FID for dcgan (needs trained model + generated samples):
    python compute_fid.py --model dcgan

    # Compute FID for vanilla:
    python compute_fid.py --model vanilla
"""

import argparse
import os
import sys
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

# ✅ Direct FID import (Python API — no subprocess parsing issues)
from pytorch_fid import fid_score


REAL_DIR = "data/real_mnist"
N_REAL   = 10000


# ---------------------------------------------------------
# Export real MNIST images
# ---------------------------------------------------------
def export_real_mnist(n: int = N_REAL):
    if os.path.exists(REAL_DIR) and len(os.listdir(REAL_DIR)) >= n:
        print(f"[INFO] Real MNIST already exists → {REAL_DIR}/")
        return

    os.makedirs(REAL_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print(f"[INFO] Exporting {n} real MNIST images → {REAL_DIR}/")

    pbar = tqdm(total=n, unit="img", ncols=80)

    for i, (img, _) in enumerate(dataset):
        if i >= n:
            break

        save_image(img, f"{REAL_DIR}/{i:05d}.png", normalize=True)
        pbar.update(1)

    pbar.close()
    print(f"[DONE] Saved {n} real images.")


# ---------------------------------------------------------
# Compute FID
# ---------------------------------------------------------
def compute_fid(model_type: str):
    # ✅ FIXED PATH (IMPORTANT)
    gen_dir = f"results/{model_type}/generated"
    out_txt = f"results/{model_type}/fid_score.txt"

    if not os.path.exists(gen_dir) or len(os.listdir(gen_dir)) == 0:
        print(f"[ERROR] No generated images found at '{gen_dir}'")
        print(f"Run first: python generate_samples.py --model {model_type}")
        sys.exit(1)

    if not os.path.exists(REAL_DIR) or len(os.listdir(REAL_DIR)) == 0:
        print("[ERROR] Real MNIST images not found")
        print("Run: python compute_fid.py --export_real")
        sys.exit(1)

    print(f"[INFO] Computing FID for {model_type.upper()}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ BEST METHOD — direct Python API (no subprocess parsing issues)
    fid_value = fid_score.calculate_fid_given_paths(
        [REAL_DIR, gen_dir],
        batch_size=50,
        device=device,
        dims=2048
    )

    print(f"[RESULT] FID = {fid_value:.4f}")

    # Save result
    with open(out_txt, "w") as f:
        f.write(f"Model : {model_type.upper()}\n")
        f.write(f"FID   : {fid_value:.4f}\n")
        f.write(f"Real  : {REAL_DIR} ({len(os.listdir(REAL_DIR))} images)\n")
        f.write(f"Fake  : {gen_dir} ({len(os.listdir(gen_dir))} images)\n")

    print(f"[DONE] FID saved → {out_txt}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for GAN models")

    parser.add_argument(
        "--model",
        choices=["vanilla", "dcgan"],
        default="dcgan"
    )

    parser.add_argument(
        "--export_real",
        action="store_true",
        help="Export real MNIST images only"
    )

    parser.add_argument(
        "--n_real",
        type=int,
        default=N_REAL
    )

    args = parser.parse_args()

    if args.export_real:
        export_real_mnist(args.n_real)
    else:
        export_real_mnist(args.n_real)  # safe to call again — skips if already done
        compute_fid(args.model)