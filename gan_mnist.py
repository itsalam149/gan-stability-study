"""
CS318 – GAN-Based Image Generation
Authors: Faqre Alam (23/CS/150), Ekansh Agrawal (23/CS/149)

Implements:
  - Vanilla GAN  (fully-connected)
  - DCGAN        (convolutional)
on MNIST.

Usage:
    python gan_mnist.py --model vanilla --epochs 100
    python gan_mnist.py --model dcgan   --epochs 100
    python gan_mnist.py --config configs/dcgan.yaml

Outputs saved to:
    results/
        vanilla/  or  dcgan/
            samples/        – image grids every 10 epochs
            checkpoints/    – model .pth every 25 epochs
            losses.png      – G vs D loss curve
            progression.png – 4-panel training progression
            fid_note.txt    – instructions for FID computation
"""

import argparse
import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ─── Defaults (overridden by YAML or CLI args) ─────────────────────────────────
DEFAULT_CFG = dict(
    latent_dim=100,
    img_size=28,
    channels=1,
    batch_size=128,
    epochs=100,
    lr=0.0002,
    beta1=0.5,
    beta2=0.999,
    label_smoothing=0.9,
    sample_interval=10,
    checkpoint_interval=25,
    num_workers=2,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] Using device: {device}")


# ═══════════════════════════════════════════════════════════════════════════════
#  VANILLA GAN  (fully-connected)
# ═══════════════════════════════════════════════════════════════════════════════

class VanillaGenerator(nn.Module):
    """MLP Generator: z → flattened image."""
    def __init__(self, cfg):
        super().__init__()
        self.img_shape = (cfg["channels"], cfg["img_size"], cfg["img_size"])
        flat = cfg["channels"] * cfg["img_size"] * cfg["img_size"]
        self.net = nn.Sequential(
            nn.Linear(cfg["latent_dim"], 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, flat),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(img.size(0), *self.img_shape)


class VanillaDiscriminator(nn.Module):
    """MLP Discriminator: flattened image → real/fake score."""
    def __init__(self, cfg):
        super().__init__()
        flat = cfg["channels"] * cfg["img_size"] * cfg["img_size"]
        self.net = nn.Sequential(
            nn.Linear(flat, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        flat = img.view(img.size(0), -1)
        return self.net(flat)


# ═══════════════════════════════════════════════════════════════════════════════
#  DCGAN  (convolutional)
# ═══════════════════════════════════════════════════════════════════════════════

class DCGenerator(nn.Module):
    """
    Convolutional Generator.
    z (100,) → Linear → reshape (128,7,7) → ConvTranspose → (1,28,28)
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["latent_dim"], 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            # 7×7 → 14×14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 14×14 → 28×28
            nn.ConvTranspose2d(64, cfg["channels"], kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)


class DCDiscriminator(nn.Module):
    """
    Convolutional Discriminator.
    (1,28,28) → Conv → Conv → Linear → sigmoid
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            # 28×28 → 14×14
            nn.Conv2d(cfg["channels"], 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 14×14 → 7×7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        features = self.net(img).view(img.size(0), -1)
        return self.fc(features)


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def weights_init(m):
    """DCGAN-style weight initialisation (zero-mean normal, small std)."""
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_dataloader(cfg):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # → [-1, 1]
    ])
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )


def save_loss_curve(g_losses, d_losses, path):
    """Plot and save G/D loss curves with a light smoothing."""
    def smooth(vals, w=5):
        w = min(w, len(vals))   # don't smooth wider than the data
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="same")

    epochs = np.arange(1, len(g_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, g_losses, alpha=0.3, color="#e74c3c")
    ax.plot(epochs, smooth(g_losses), label="Generator Loss",
            color="#e74c3c", linewidth=2)
    ax.plot(epochs, d_losses, alpha=0.3, color="#2980b9")
    ax.plot(epochs, smooth(d_losses), label="Discriminator Loss",
            color="#2980b9", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_progression(sample_dir, out_path, milestones):
    """
    Build a 4-panel figure showing samples at given epoch milestones.
    Silently skips missing files.
    """
    found = []
    for ep in milestones:
        p = os.path.join(sample_dir, f"epoch_{ep:04d}.png")
        if os.path.exists(p):
            found.append((ep, p))

    if not found:
        return

    n = len(found)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (ep, p) in zip(axes, found):
        img = Image.open(p)
        ax.imshow(np.array(img), cmap="gray" if img.mode == "L" else None)
        ax.set_title(f"Epoch {ep}", fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Training Progression", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Progression figure saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(model_type: str, cfg: dict):
    out_dir    = f"results/{model_type}"
    sample_dir = f"{out_dir}/samples"
    ckpt_dir   = f"{out_dir}/checkpoints"
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    epochs = cfg["epochs"]

    # ── Build models ─────────────────────────────────────────────────────────
    if model_type == "vanilla":
        G = VanillaGenerator(cfg).to(device)
        D = VanillaDiscriminator(cfg).to(device)
    else:
        G = DCGenerator(cfg).to(device)
        D = DCDiscriminator(cfg).to(device)
        G.apply(weights_init)
        D.apply(weights_init)

    adversarial_loss = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=cfg["lr"],
                       betas=(cfg["beta1"], cfg["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr"],
                       betas=(cfg["beta1"], cfg["beta2"]))

    dataloader = get_dataloader(cfg)
    fixed_z    = torch.randn(64, cfg["latent_dim"], device=device)

    g_losses, d_losses = [], []

    smooth = cfg["label_smoothing"]   # 0.9 → real labels

    print(f"\n{'═'*60}")
    print(f"  Training {model_type.upper()} GAN  |  {epochs} epochs  |  {device}")
    print(f"{'═'*60}\n")

    for epoch in range(1, epochs + 1):
        epoch_g, epoch_d = 0.0, 0.0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:>4}/{epochs}",
                    leave=False, ncols=90)

        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)

            # Label smoothing: real → smooth, fake → 0
            real_labels = torch.full((B, 1), smooth, device=device)
            fake_labels = torch.zeros(B, 1, device=device)

            # ── Train Discriminator ──────────────────────────────────────────
            opt_D.zero_grad()
            z        = torch.randn(B, cfg["latent_dim"], device=device)
            fake_imgs = G(z).detach()

            loss_real = adversarial_loss(D(real_imgs), real_labels)
            loss_fake = adversarial_loss(D(fake_imgs), fake_labels)
            loss_D    = (loss_real + loss_fake) / 2
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────────────────────────
            opt_G.zero_grad()
            z       = torch.randn(B, cfg["latent_dim"], device=device)
            gen_imgs = G(z)
            # Fool discriminator: generated images should look real (label=1)
            loss_G  = adversarial_loss(D(gen_imgs),
                                       torch.ones(B, 1, device=device))
            loss_G.backward()
            opt_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()
            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}")

        avg_g = epoch_g / len(dataloader)
        avg_d = epoch_d / len(dataloader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:>4}/{epochs}]  "
              f"D loss: {avg_d:.4f}  G loss: {avg_g:.4f}  "
              f"({elapsed:.1f}s)")

        # ── Save sample grid ─────────────────────────────────────────────────
        if epoch % cfg["sample_interval"] == 0 or epoch == 1:
            with torch.no_grad():
                samples = G(fixed_z)
            path = f"{sample_dir}/epoch_{epoch:04d}.png"
            save_image(samples, path, nrow=8, normalize=True)
            print(f"  → Sample grid saved → {path}")

        # ── Save checkpoint ──────────────────────────────────────────────────
        if epoch % cfg["checkpoint_interval"] == 0:
            ckpt = {
                "epoch":   epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "opt_G":   opt_G.state_dict(),
                "opt_D":   opt_D.state_dict(),
                "cfg":     cfg,
            }
            ckpt_path = f"{ckpt_dir}/epoch_{epoch:04d}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"  → Checkpoint saved   → {ckpt_path}")

    # ── Final model save ─────────────────────────────────────────────────────
    torch.save(G.state_dict(), f"{out_dir}/generator.pth")
    torch.save(D.state_dict(), f"{out_dir}/discriminator.pth")
    print(f"\n[DONE] Models saved → {out_dir}/generator.pth")

    # ── Save raw loss arrays (for visualize.py) ───────────────────────────────
    np.save(f"{out_dir}/g_losses.npy", np.array(g_losses))
    np.save(f"{out_dir}/d_losses.npy", np.array(d_losses))

    # ── Loss curve ───────────────────────────────────────────────────────────
    save_loss_curve(g_losses, d_losses, f"{out_dir}/losses.png")
    print(f"[DONE] Loss curve   → {out_dir}/losses.png")

    # ── Training progression ─────────────────────────────────────────────────
    milestones = [1, 10, 50, 100]
    save_progression(sample_dir, f"{out_dir}/progression.png", milestones)

    # ── FID instructions ─────────────────────────────────────────────────────
    with open(f"{out_dir}/fid_note.txt", "w") as f:
        f.write(
            "To compute FID:\n"
            "  pip install pytorch-fid\n\n"
            "1. Generate fake images:\n"
            f"   python generate_samples.py --model {model_type} --n 10000\n\n"
            "2. Export real MNIST images:\n"
            "   python compute_fid.py --export_real\n\n"
            "3. Compute FID:\n"
            f"   python compute_fid.py --model {model_type}\n"
        )
    print(f"[INFO] FID note      → {out_dir}/fid_note.txt\n")

    return g_losses, d_losses


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(args) -> dict:
    cfg = DEFAULT_CFG.copy()

    # 1. Load YAML if provided
    if args.config:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update({k: v for k, v in yaml_cfg.items() if k in cfg})
        if "model" in yaml_cfg and not args.model:
            args.model = yaml_cfg["model"]

    # 2. CLI overrides
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS318 GAN – MNIST")
    parser.add_argument("--model",      choices=["vanilla", "dcgan"], default="dcgan",
                        help="GAN architecture (default: dcgan)")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--config",     type=str, default=None,
                        help="Path to YAML config file (e.g. configs/dcgan.yaml)")
    args = parser.parse_args()

    cfg = load_config(args)
    print(f"[CFG] model={args.model}  epochs={cfg['epochs']}  "
          f"batch={cfg['batch_size']}  lr={cfg['lr']}")

    train(args.model, cfg)
