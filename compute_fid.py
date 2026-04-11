"""
compute_fid.py
──────────────
CS318 GAN – FID Score Computation
Uses pytorch-fid to compute Fréchet Inception Distance between
real MNIST images and generated samples.

Steps performed automatically:
  1. Export 10 k real MNIST images to data/real_mnist/  (once)
  2. Load generated images from results/<model>/generated/
  3. Run pytorch-fid and save score to results/<model>/fid_score.txt

Usage:
    # Export real MNIST images only (one-time setup):
    python compute_fid.py --export_real

    # Compute FID for dcgan (needs trained model + generated samples):
    python compute_fid.py --model dcgan

    # Compute FID for both models:
    python compute_fid.py --model vanilla
    python compute_fid.py --model dcgan
"""

import argparse
import os
import subprocess
import sys
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

REAL_DIR = "data/real_mnist"
N_REAL   = 10000


def export_real_mnist(n: int = N_REAL):
    """Save n real MNIST training images to REAL_DIR as PNG files."""
    if os.path.exists(REAL_DIR) and len(os.listdir(REAL_DIR)) >= n:
        print(f"[INFO] Real MNIST images already exported ({REAL_DIR}/)  – skipping.")
        return

    os.makedirs(REAL_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(root="./data", train=True,
                             download=True, transform=transform)

    print(f"[INFO] Exporting {n} real MNIST images → {REAL_DIR}/ …")
    pbar = tqdm(total=n, unit="img", ncols=80)
    for i, (img, _) in enumerate(dataset):
        if i >= n:
            break
        save_image(img, f"{REAL_DIR}/{i:05d}.png", normalize=True)
        pbar.update(1)
    pbar.close()
    print(f"[DONE] {n} real images saved → {REAL_DIR}/")


def compute_fid(model_type: str):
    """Run pytorch-fid and write the score to a txt file."""
    gen_dir = f"results/{model_type}/generated"
    out_txt = f"results/{model_type}/fid_score.txt"

    if not os.path.exists(gen_dir) or len(os.listdir(gen_dir)) == 0:
        print(f"[ERROR] No generated images found at '{gen_dir}'.")
        print(f"  Run first:  python generate_samples.py --model {model_type}")
        sys.exit(1)

    if not os.path.exists(REAL_DIR) or len(os.listdir(REAL_DIR)) == 0:
        print("[ERROR] Real MNIST images not exported yet.")
        print("  Run:  python compute_fid.py --export_real")
        sys.exit(1)

    print(f"[INFO] Computing FID for {model_type.upper()} …")
    result = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", REAL_DIR, gen_dir,
         "--device", "cuda" if torch.cuda.is_available() else "cpu"],
        capture_output=True, text=True
    )

    output = result.stdout + result.stderr
    print(output)

    # Extract FID value from output
    fid_value = None
    for line in output.splitlines():
        if "FID" in line:
            try:
                fid_value = float(line.strip().split()[-1])
            except ValueError:
                pass

    if fid_value is not None:
        with open(out_txt, "w") as f:
            f.write(f"Model : {model_type.upper()}\n")
            f.write(f"FID   : {fid_value:.4f}\n")
            f.write(f"Real  : {REAL_DIR}  ({len(os.listdir(REAL_DIR))} images)\n")
            f.write(f"Fake  : {gen_dir}  ({len(os.listdir(gen_dir))} images)\n")
        print(f"[DONE] FID = {fid_value:.4f}  →  saved to {out_txt}")
    else:
        print("[WARN] Could not parse FID value from output. Check output above.")
        with open(out_txt, "w") as f:
            f.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for GAN models")
    parser.add_argument("--model",       choices=["vanilla", "dcgan"], default="dcgan")
    parser.add_argument("--export_real", action="store_true",
                        help="Only export real MNIST images (one-time setup)")
    parser.add_argument("--n_real",      type=int, default=N_REAL,
                        help=f"Number of real images to export (default: {N_REAL})")
    args = parser.parse_args()

    if args.export_real:
        export_real_mnist(args.n_real)
    else:
        export_real_mnist(args.n_real)   # idempotent
        compute_fid(args.model)
