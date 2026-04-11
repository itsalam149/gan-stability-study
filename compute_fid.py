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


N_REAL   = 10000


# ---------------------------------------------------------
# Export real images
# ---------------------------------------------------------
def export_real_images(n: int = N_REAL, dataset_name: str = "mnist"):
    real_dir = f"data/real_{dataset_name}"
    if os.path.exists(real_dir) and len(os.listdir(real_dir)) >= n:
        print(f"[INFO] Real {dataset_name} already exists → {real_dir}/")
        return real_dir

    os.makedirs(real_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if dataset_name == "fashion_mnist":
        dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

    print(f"[INFO] Exporting {n} real {dataset_name} images → {real_dir}/")

    pbar = tqdm(total=n, unit="img", ncols=80)

    for i, (img, _) in enumerate(dataset):
        if i >= n:
            break

        save_image(img, f"{real_dir}/{i:05d}.png", normalize=True)
        pbar.update(1)

    pbar.close()
    print(f"[DONE] Saved {n} real images.")
    return real_dir


# ---------------------------------------------------------
# Compute FID
# ---------------------------------------------------------
def compute_fid(model_type: str, dataset_name: str = "mnist"):
    # ✅ FIXED PATH (IMPORTANT)
    gen_dir = f"results/{model_type}_{dataset_name}/generated" if dataset_name == "fashion_mnist" else f"results/{model_type}/generated"
    out_txt = f"results/{model_type}_{dataset_name}/fid_score.txt" if dataset_name == "fashion_mnist" else f"results/{model_type}/fid_score.txt"
    real_dir = f"data/real_{dataset_name}"

    if not os.path.exists(gen_dir) or len(os.listdir(gen_dir)) == 0:
        print(f"[ERROR] No generated images found at '{gen_dir}'")
        print(f"Run first: python generate_samples.py --model {model_type} --dataset {dataset_name}")
        sys.exit(1)

    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
        print(f"[ERROR] Real {dataset_name} images not found")
        print(f"Run: python compute_fid.py --export_real --dataset {dataset_name}")
        sys.exit(1)

    print(f"[INFO] Computing FID for {model_type.upper()}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ BEST METHOD — direct Python API (no subprocess parsing issues)
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir],
        batch_size=50,
        device=device,
        dims=2048
    )

    print(f"[RESULT] FID = {fid_value:.4f}")

    # Save result
    with open(out_txt, "w") as f:
        f.write(f"Model   : {model_type.upper()}\n")
        f.write(f"Dataset : {dataset_name.upper()}\n")
        f.write(f"FID     : {fid_value:.4f}\n")
        f.write(f"Real    : {real_dir} ({len(os.listdir(real_dir))} images)\n")
        f.write(f"Fake    : {gen_dir} ({len(os.listdir(gen_dir))} images)\n")

    print(f"[DONE] FID saved → {out_txt}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for GAN models")

    parser.add_argument(
        "--model",
        choices=["vanilla", "dcgan", "cdcgan"],
        default="dcgan"
    )

    parser.add_argument(
        "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="mnist"
    )

    parser.add_argument(
        "--export_real",
        action="store_true",
        help="Export real images only"
    )

    parser.add_argument(
        "--n_real",
        type=int,
        default=N_REAL
    )

    args = parser.parse_args()

    if args.export_real:
        export_real_images(args.n_real, args.dataset)
    else:
        export_real_images(args.n_real, args.dataset)  # safe to call again — skips if already done
        compute_fid(args.model, args.dataset)