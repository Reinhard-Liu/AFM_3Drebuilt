"""
cGAN baseline training (simplified pix2pix).

Generates 2D molecular projection images from AFM stacks.
This is a 2D-only baseline — cannot produce 3D coordinates.
Used for comparison in the paper to show 3D reconstruction advantage.

Usage:
    python -m src.train_cgan --config config_cgan.json
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import QUAMAFMDataset, create_dataloaders, ATOM_TYPES
from src.models.baselines import UNetGenerator, PatchDiscriminator


def render_2d_projection(coords, atom_types, mask, img_size=128):
    """Render a simple 2D molecular projection from 3D coordinates.

    Creates an RGB image where each atom is drawn as a colored circle
    at its XY position (projected from above, like AFM).

    Args:
        coords: (N, 3) coordinates in normalized space [-1, 1]
        atom_types: (N,) type indices
        mask: (N,) atom mask

    Returns:
        (3, img_size, img_size) RGB image in [0, 1]
    """
    # Atom colors (RGB, normalized to [0,1])
    COLORS = {
        0: (1.0, 1.0, 1.0),     # H: white
        1: (0.2, 0.2, 0.2),     # C: dark gray
        2: (0.19, 0.31, 0.97),  # N: blue
        3: (1.0, 0.05, 0.05),   # O: red
        4: (0.56, 0.88, 0.31),  # F: green
        5: (1.0, 1.0, 0.19),    # S: yellow
        6: (1.0, 0.50, 0.0),    # P: orange
        7: (0.12, 0.94, 0.12),  # Cl: bright green
        8: (0.65, 0.16, 0.16),  # Br: brown
        9: (0.58, 0.0, 0.58),   # I: purple
    }
    RADII = {0: 2, 1: 4, 2: 4, 3: 3, 4: 3, 5: 5, 6: 5, 7: 4, 8: 5, 9: 5}

    img = np.zeros((3, img_size, img_size), dtype=np.float32)
    valid = mask > 0

    for i in range(len(coords)):
        if not valid[i]:
            continue
        # Map from [-1,1] to pixel coordinates
        px = int((coords[i, 0] + 1) / 2 * (img_size - 1))
        py = int((coords[i, 1] + 1) / 2 * (img_size - 1))
        px = max(0, min(px, img_size - 1))
        py = max(0, min(py, img_size - 1))

        t = int(atom_types[i])
        color = COLORS.get(t, (0.5, 0.5, 0.5))
        r = RADII.get(t, 3)

        # Draw circle
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    x, y = px + dx, py + dy
                    if 0 <= x < img_size and 0 <= y < img_size:
                        for c in range(3):
                            img[c, y, x] = color[c]

    return img


def train_cgan(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config["param_key"],
        img_size=config["img_size"],
        min_corrugation=config.get("min_corrugation", 1.25),
        augment_rotation=config.get("augment_rotation", True),
        require_ring=config.get("require_ring", False),
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Models
    generator = UNetGenerator(in_channels=10, out_channels=3).to(device)
    discriminator = PatchDiscriminator(in_channels=13).to(device)  # 10 AFM + 3 RGB

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator: {g_params/1e6:.2f}M, Discriminator: {d_params/1e6:.2f}M")

    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Losses
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    lambda_l1 = 100.0

    save_dir = config.get("save_dir", "experiments/cgan/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    epochs = config.get("epochs", 50)
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        pbar = tqdm(train_loader, desc=f"cGAN [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)  # (B, 10, H, W)
            coords = batch["coords"].cpu().numpy()
            types = batch["atom_types"].cpu().numpy()
            mask = batch["atom_mask"].cpu().numpy()

            # Render GT 2D projections
            B = afm.shape[0]
            real_imgs = []
            for b in range(B):
                proj = render_2d_projection(coords[b], types[b], mask[b], config["img_size"])
                real_imgs.append(proj)
            real_imgs = torch.tensor(np.stack(real_imgs)).to(device)  # (B, 3, H, W)

            # --- Train Discriminator ---
            fake_imgs = generator(afm)
            d_real = discriminator(real_imgs, afm)
            d_fake = discriminator(fake_imgs.detach(), afm)
            d_loss_real = criterion_gan(d_real, torch.ones_like(d_real))
            d_loss_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            fake_imgs = generator(afm)
            d_fake = discriminator(fake_imgs, afm)
            g_loss_gan = criterion_gan(d_fake, torch.ones_like(d_fake))
            g_loss_l1 = criterion_l1(fake_imgs, real_imgs) * lambda_l1
            g_loss = g_loss_gan + g_loss_l1

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            pbar.set_postfix(g=f"{np.mean(g_losses[-10:]):.3f}", d=f"{np.mean(d_losses[-10:]):.3f}")

        print(f"Epoch {epoch}/{epochs} | G_loss: {np.mean(g_losses):.4f} | D_loss: {np.mean(d_losses):.4f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
            }, os.path.join(save_dir, f"cgan_epoch_{epoch}.pt"))

    # Save final
    torch.save({
        "epoch": epochs,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }, os.path.join(save_dir, "cgan_final.pt"))
    print(f"cGAN training complete. Saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    if config.get("data_root") == "auto":
        config["data_root"] = os.path.join(base_dir, "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = os.path.join(base_dir, "experiments", "cgan", "checkpoints")

    train_cgan(config)
