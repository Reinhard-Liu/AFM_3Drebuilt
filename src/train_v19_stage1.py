"""
V19 Stage 1:
AFM stack -> structured 2D molecular map

This is a minimal executable implementation of the new V19 direction.
It does not attempt full 3D reconstruction. It first learns:
- 2D atom occupancy
- 2D bond map
- 2D atom-type maps
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import create_dataloaders
from src.models.baselines import PatchDiscriminator, UNetGenerator
from src.utils.mol2d import (
    V19_2D_TARGET_CHANNELS,
    batch_render_v19_2d_targets,
    project_xy_to_pixels,
    structure_map_to_rgb,
)


def build_targets(batch: dict, img_size: int, device: torch.device) -> torch.Tensor:
    coords = batch["coords"].cpu().numpy()
    types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()
    target = batch_render_v19_2d_targets(coords, types, mask, img_size=img_size)
    target = torch.from_numpy(target).to(device)
    return target


def _macro_f1_from_lists(gt_labels, pred_labels, n_classes=10):
    f1s = []
    for cls in range(n_classes):
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cls and p != cls)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def _local_window(arr2d, x, y, radius):
    h, w = arr2d.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    return arr2d[y0:y1, x0:x1], x0, y0


def compute_v19_stage1_metrics(pred_01, batch, img_size):
    """Continuous metrics that tolerate small spatial deviations."""
    pred_np = pred_01.detach().cpu().numpy()
    coords = batch["coords"].cpu().numpy()
    atom_types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()

    atom_mae_list = []
    bond_mae_list = []
    type_mae_list = []
    center_score_list = []
    typed_center_score_list = []
    type_acc_list = []
    gt_type_all = []
    pred_type_all = []
    pred_ch_count = 0
    pred_total_count = 0
    gt_ch_count = 0
    gt_total_count = 0

    gt_targets = batch_render_v19_2d_targets(coords, atom_types, mask, img_size=img_size)
    for b in range(pred_np.shape[0]):
        pred = pred_np[b]
        real = gt_targets[b]
        atom_mae_list.append(float(np.abs(pred[0] - real[0]).mean()))
        bond_mae_list.append(float(np.abs(pred[1] - real[1]).mean()))
        type_mae_list.append(float(np.abs(pred[2:] - real[2:]).mean()))

        pix = project_xy_to_pixels(coords[b], img_size)
        valid_idx = np.where(mask[b] > 0.5)[0]
        for idx in valid_idx:
            gt_t = int(atom_types[b, idx])
            if gt_t < 0 or gt_t >= 10:
                continue
            x, y = map(int, pix[idx])
            occ_patch, x0, y0 = _local_window(pred[0], x, y, radius=3)
            type_patch = pred[2:, y0:y0 + occ_patch.shape[0], x0:x0 + occ_patch.shape[1]]
            occ_max = float(occ_patch.max()) if occ_patch.size else 0.0
            center_score_list.append(occ_max)

            if occ_patch.size:
                flat_idx = int(np.argmax(occ_patch))
                oy, ox = np.unravel_index(flat_idx, occ_patch.shape)
                local_type_vec = type_patch[:, oy, ox]
                pred_t = int(np.argmax(local_type_vec))
                typed_center_score_list.append(float(occ_patch[oy, ox] * type_patch[gt_t, oy, ox]))
            else:
                pred_t = 1  # default carbon-like collapse
                typed_center_score_list.append(0.0)

            gt_type_all.append(gt_t)
            pred_type_all.append(pred_t)
            type_acc_list.append(1.0 if pred_t == gt_t else 0.0)
            pred_total_count += 1
            gt_total_count += 1
            if pred_t in (0, 1):
                pred_ch_count += 1
            if gt_t in (0, 1):
                gt_ch_count += 1

    gt_ch_fraction = gt_ch_count / max(gt_total_count, 1)
    pred_ch_fraction = pred_ch_count / max(pred_total_count, 1)
    ch_collapse_rate = float(np.clip((pred_ch_fraction - gt_ch_fraction) / max(1.0 - gt_ch_fraction, 1e-6), 0.0, 1.0))

    return {
        "atom_xy_mae": float(np.mean(atom_mae_list)) if atom_mae_list else 0.0,
        "bond_map_mae": float(np.mean(bond_mae_list)) if bond_mae_list else 0.0,
        "type_map_mae": float(np.mean(type_mae_list)) if type_mae_list else 0.0,
        "atom_center_score_r3": float(np.mean(center_score_list)) if center_score_list else 0.0,
        "typed_center_score_r3": float(np.mean(typed_center_score_list)) if typed_center_score_list else 0.0,
        "type_top1_local_acc_r3": float(np.mean(type_acc_list)) if type_acc_list else 0.0,
        "atom_type_macro_f1_2d": _macro_f1_from_lists(gt_type_all, pred_type_all, n_classes=10),
        "ch_collapse_rate_2d": ch_collapse_rate,
    }


def save_preview(generator, val_loader, config, device, save_path: Path):
    generator.eval()
    batch = next(iter(val_loader))
    afm = batch["afm_stack"].to(device)
    real = build_targets(batch, config["img_size"], device)
    with torch.no_grad():
        pred = generator(afm)
    pred = ((pred + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy()
    real = real.cpu().numpy()
    afm = afm.cpu().numpy()

    n_show = min(4, pred.shape[0])
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n_show):
        axes[i, 0].imshow(afm[i, afm.shape[1] // 2], cmap="afmhot", vmin=0, vmax=1)
        axes[i, 0].set_title("AFM mid-slice")
        axes[i, 1].imshow(np.transpose(structure_map_to_rgb(real[i]), (1, 2, 0)))
        axes[i, 1].set_title("GT 2D structure")
        axes[i, 2].imshow(np.transpose(structure_map_to_rgb(pred[i]), (1, 2, 0)))
        axes[i, 2].set_title("Pred 2D structure")
        for j in range(3):
            axes[i, j].axis("off")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate(generator, val_loader, config, device):
    generator.eval()
    agg = {}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            afm = batch["afm_stack"].to(device)
            pred = generator(afm)
            pred = (pred + 1.0) * 0.5  # tanh -> [0,1]
            metrics = compute_v19_stage1_metrics(pred, batch, config["img_size"])
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n_batches += 1
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in agg.items()}


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config.get("param_key", "K-1"),
        img_size=config["img_size"],
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=config.get("augment_rotation", True),
        require_ring=config.get("require_ring", False),
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    generator = UNetGenerator(in_channels=10, out_channels=V19_2D_TARGET_CHANNELS).to(device)
    discriminator = PatchDiscriminator(in_channels=10 + V19_2D_TARGET_CHANNELS).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=config.get("lr", 2e-4), betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.get("lr", 2e-4), betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    lambda_atom = config.get("lambda_atom", 20.0)
    lambda_bond = config.get("lambda_bond", 10.0)
    lambda_type = config.get("lambda_type", 20.0)

    save_dir = Path(config.get("save_dir", "experiments/v19_stage1_2d/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    epochs = int(config.get("epochs", 1))
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_losses = []
        d_losses = []

        pbar = tqdm(train_loader, desc=f"V19-2D [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)
            real = build_targets(batch, config["img_size"], device)
            real_tanh = real * 2.0 - 1.0

            fake = generator(afm)

            d_real = discriminator(real_tanh, afm)
            d_fake = discriminator(fake.detach(), afm)
            d_loss_real = criterion_gan(d_real, torch.ones_like(d_real))
            d_loss_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            fake = generator(afm)
            d_fake = discriminator(fake, afm)
            g_gan = criterion_gan(d_fake, torch.ones_like(d_fake))
            fake_01 = (fake + 1.0) * 0.5
            g_atom = criterion_l1(fake_01[:, 0:1], real[:, 0:1]) * lambda_atom
            g_bond = criterion_l1(fake_01[:, 1:2], real[:, 1:2]) * lambda_bond
            g_type = criterion_l1(fake_01[:, 2:], real[:, 2:]) * lambda_type
            g_loss = g_gan + g_atom + g_bond + g_type

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))
            pbar.set_postfix(g=f"{np.mean(g_losses[-10:]):.3f}", d=f"{np.mean(d_losses[-10:]):.3f}")

        val_metrics = evaluate(generator, val_loader, config, device)
        train_metrics = {
            "g_loss": float(np.mean(g_losses)) if g_losses else 0.0,
            "d_loss": float(np.mean(d_losses)) if d_losses else 0.0,
        }
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"G {train_metrics['g_loss']:.4f} | D {train_metrics['d_loss']:.4f} | "
            f"atom_xy_mae {val_metrics['atom_xy_mae']:.4f} | "
            f"bond_map_mae {val_metrics['bond_map_mae']:.4f} | "
            f"type_map_mae {val_metrics['type_map_mae']:.4f} | "
            f"center_r3 {val_metrics['atom_center_score_r3']:.4f} | "
            f"typed_center_r3 {val_metrics['typed_center_score_r3']:.4f} | "
            f"type_acc_r3 {val_metrics['type_top1_local_acc_r3']:.4f} | "
            f"macro_f1_2d {val_metrics['atom_type_macro_f1_2d']:.4f} | "
            f"collapse_2d {val_metrics['ch_collapse_rate_2d']:.4f}"
        )

        if val_metrics["type_map_mae"] < best_val:
            best_val = val_metrics["type_map_mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "val_metrics": val_metrics,
                    "history": history,
                    "config": config,
                },
                save_dir / "best_v19_stage1.pt",
            )
            save_preview(generator, val_loader, config, device, save_dir / "best_preview.png")

    with open(save_dir / "history_v19_stage1.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": epochs,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "history": history,
            "config": config,
        },
        save_dir / "last_v19_stage1.pt",
    )

    print(f"Saved checkpoints to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    base_dir = Path(__file__).resolve().parents[1]
    if config.get("data_root") == "auto":
        config["data_root"] = str(base_dir / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = str(base_dir / "experiments" / "v19_stage1_2d_debug" / "checkpoints")

    train(config)
