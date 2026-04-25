"""
V19.1 minimal joint training:
AFM stack -> structured 2D molecular map + z-map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.models.baselines import PatchDiscriminator
from src.models.v19_center_type_head import CenterConditionedTypeHead
from src.models.gnn_type_classifier import GNNTypeClassifier
from src.models.v19_joint_model import V19JointUNet
from src.models.video_vit import VideoViTEncoder
from src.utils.mol2d import (
    V19_JOINT_TARGET_CHANNELS,
    batch_render_v19_joint_targets,
    project_xy_to_pixels,
    structure_map_to_rgb,
    z_map_to_rgb,
)


TYPE_CLASS_WEIGHTS = torch.tensor(
    [0.041, 0.039, 0.077, 0.098, 0.666, 0.207, 6.348, 0.688, 0.921, 0.914],
    dtype=torch.float32,
)


def build_targets(batch: dict, img_size: int, device: torch.device) -> torch.Tensor:
    coords = batch["coords"].cpu().numpy()
    types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()
    target = batch_render_v19_joint_targets(coords, types, mask, img_size=img_size)
    return torch.from_numpy(target).to(device)


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
        f1s.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def _discrete_type_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = (mask > 0) & (gt >= 0)
    if valid.sum() == 0:
        return {
            "center_type_acc_gtcoord": 0.0,
            "center_macro_f1_gtcoord": 0.0,
            "center_hetero_f1_gtcoord": 0.0,
            "center_ch_collapse_rate_gtcoord": 0.0,
        }

    pred_np = pred[valid].detach().cpu().numpy().astype(np.int64)
    gt_np = gt[valid].detach().cpu().numpy().astype(np.int64)

    type_acc = float((pred_np == gt_np).mean())
    macro_f1 = _macro_f1_from_lists(gt_np.tolist(), pred_np.tolist(), n_classes=10)

    pred_ch_fraction = float(np.isin(pred_np, [0, 1]).mean()) if len(pred_np) else 0.0
    gt_ch_fraction = float(np.isin(gt_np, [0, 1]).mean()) if len(gt_np) else 0.0
    ch_collapse = float(np.clip((pred_ch_fraction - gt_ch_fraction) / max(1.0 - gt_ch_fraction, 1e-6), 0.0, 1.0))

    pred_het = ~np.isin(pred_np, [0, 1])
    gt_het = ~np.isin(gt_np, [0, 1])
    tp = int((pred_het & gt_het).sum())
    fp = int((pred_het & ~gt_het).sum())
    fn = int((~pred_het & gt_het).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    hetero_f1 = 0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec)

    return {
        "center_type_acc_gtcoord": type_acc,
        "center_macro_f1_gtcoord": float(macro_f1),
        "center_hetero_f1_gtcoord": float(hetero_f1),
        "center_ch_collapse_rate_gtcoord": ch_collapse,
    }


def build_type_teacher(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]
    encoder = VideoViTEncoder(
        img_size=cfg["img_size"],
        num_frames=cfg.get("num_frames", 10),
        patch_size=cfg.get("patch_size", 16),
        temporal_patch_size=cfg.get("temporal_patch_size", 2),
        embed_dim=cfg.get("embed_dim", 256),
        depth=cfg.get("encoder_depth", 4),
        num_heads=cfg.get("num_heads", 4),
        drop_rate=cfg.get("drop_rate", 0.1),
    ).to(device)
    classifier = GNNTypeClassifier(
        cond_dim=cfg.get("embed_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_gnn_layers=cfg.get("num_gnn_layers", 4),
        num_types=10,
        num_heads=cfg.get("num_heads", 4),
        bond_threshold=cfg.get("bond_threshold", 0.20),
    ).to(device)
    encoder.load_state_dict(state["encoder"])
    classifier.load_state_dict(state["classifier"])
    encoder.eval()
    classifier.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = False
    return encoder, classifier, cfg


def _local_window(arr2d, x, y, radius):
    h, w = arr2d.shape
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    return arr2d[y0:y1, x0:x1], x0, y0


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def scheduled_weight(epoch: int, total_epochs: int, final_weight: float, start_weight: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 1:
        return float(final_weight)
    if epoch <= 1:
        return float(start_weight)
    if epoch >= warmup_epochs:
        return float(final_weight)
    alpha = float(epoch - 1) / float(max(warmup_epochs - 1, 1))
    return float(start_weight + alpha * (final_weight - start_weight))


def gather_gt_center_predictions(
    raw_pred: torch.Tensor,
    batch: dict,
    img_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather raw predictions at GT atom centers.

    Returns:
      center_scores: (M,)
      type_scores: (M, 10)
      gt_types: (M,)
    """
    coords = batch["coords"]
    atom_types = batch["atom_types"]
    atom_mask = batch["atom_mask"]
    device = raw_pred.device

    centers = []
    type_scores = []
    gt_labels = []

    for b in range(raw_pred.shape[0]):
        valid_idx = torch.nonzero(atom_mask[b] > 0.5, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        xy = coords[b, valid_idx, :2]
        px = ((xy[:, 0] + 1.0) * 0.5 * (img_size - 1)).round().long().clamp(0, img_size - 1)
        py = ((xy[:, 1] + 1.0) * 0.5 * (img_size - 1)).round().long().clamp(0, img_size - 1)
        centers.append(raw_pred[b, 0, py, px])
        type_scores.append(raw_pred[b, 2:12, py, px].transpose(0, 1))
        gt_labels.append(atom_types[b, valid_idx].long().to(device))

    if not centers:
        return (
            torch.zeros(0, device=device),
            torch.zeros((0, 10), device=device),
            torch.zeros(0, dtype=torch.long, device=device),
        )

    return torch.cat(centers, dim=0), torch.cat(type_scores, dim=0), torch.cat(gt_labels, dim=0)


def compute_v19_joint_metrics(pred_01: torch.Tensor, batch: dict, img_size: int) -> dict:
    pred_np = pred_01.detach().cpu().numpy()
    coords = batch["coords"].cpu().numpy()
    atom_types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()

    gt_targets = batch_render_v19_joint_targets(coords, atom_types, mask, img_size=img_size)

    atom_mae_list = []
    bond_mae_list = []
    type_mae_list = []
    z_mae_list = []
    center_score_list = []
    typed_center_score_list = []
    type_acc_list = []
    atom_z_err_list = []
    gt_type_all = []
    pred_type_all = []
    pred_ch_count = 0
    pred_total_count = 0
    gt_ch_count = 0
    gt_total_count = 0

    for b in range(pred_np.shape[0]):
        pred = pred_np[b]
        real = gt_targets[b]
        atom_mae_list.append(float(np.abs(pred[0] - real[0]).mean()))
        bond_mae_list.append(float(np.abs(pred[1] - real[1]).mean()))
        type_mae_list.append(float(np.abs(pred[2:12] - real[2:12]).mean()))

        occ_mask = real[0] > 0.05
        if occ_mask.any():
            z_mae_list.append(float(np.abs(pred[12][occ_mask] - real[12][occ_mask]).mean()))
        else:
            z_mae_list.append(0.0)

        pix = project_xy_to_pixels(coords[b], img_size)
        valid_idx = np.where(mask[b] > 0.5)[0]
        for idx in valid_idx:
            gt_t = int(atom_types[b, idx])
            if gt_t < 0 or gt_t >= 10:
                continue
            x, y = map(int, pix[idx])
            occ_patch, x0, y0 = _local_window(pred[0], x, y, radius=3)
            type_patch = pred[2:12, y0:y0 + occ_patch.shape[0], x0:x0 + occ_patch.shape[1]]
            z_patch = pred[12, y0:y0 + occ_patch.shape[0], x0:x0 + occ_patch.shape[1]]
            occ_max = float(occ_patch.max()) if occ_patch.size else 0.0
            center_score_list.append(occ_max)

            gt_z01 = float(np.clip((coords[b, idx, 2] + 1.0) * 0.5, 0.0, 1.0))
            if occ_patch.size:
                flat_idx = int(np.argmax(occ_patch))
                oy, ox = np.unravel_index(flat_idx, occ_patch.shape)
                local_type_vec = type_patch[:, oy, ox]
                pred_t = int(np.argmax(local_type_vec))
                typed_center_score_list.append(float(occ_patch[oy, ox] * type_patch[gt_t, oy, ox]))
                pred_z01 = float(z_patch[oy, ox])
                atom_z_err_list.append(abs(pred_z01 - gt_z01) * 24.0)
            else:
                pred_t = 1
                typed_center_score_list.append(0.0)
                atom_z_err_list.append(24.0 * gt_z01)

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
    ch_collapse_rate = float(
        np.clip((pred_ch_fraction - gt_ch_fraction) / max(1.0 - gt_ch_fraction, 1e-6), 0.0, 1.0)
    )

    return {
        "atom_xy_mae": float(np.mean(atom_mae_list)) if atom_mae_list else 0.0,
        "bond_map_mae": float(np.mean(bond_mae_list)) if bond_mae_list else 0.0,
        "type_map_mae": float(np.mean(type_mae_list)) if type_mae_list else 0.0,
        "z_map_mae": float(np.mean(z_mae_list)) if z_mae_list else 0.0,
        "atom_center_score_r3": float(np.mean(center_score_list)) if center_score_list else 0.0,
        "typed_center_score_r3": float(np.mean(typed_center_score_list)) if typed_center_score_list else 0.0,
        "type_top1_local_acc_r3": float(np.mean(type_acc_list)) if type_acc_list else 0.0,
        "atom_type_macro_f1_2d": _macro_f1_from_lists(gt_type_all, pred_type_all, n_classes=10),
        "ch_collapse_rate_2d": ch_collapse_rate,
        "atom_z_mae_r3": float(np.mean(atom_z_err_list)) if atom_z_err_list else 0.0,
    }


def save_preview(model, val_loader, config, device, save_path: Path):
    model.eval()
    batch = next(iter(val_loader))
    afm = batch["afm_stack"].to(device)
    real = build_targets(batch, config["img_size"], device)
    with torch.no_grad():
        pred = model(afm)
    pred = ((pred + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy()
    real = real.cpu().numpy()
    afm = afm.cpu().numpy()

    n_show = min(4, pred.shape[0])
    fig, axes = plt.subplots(n_show, 5, figsize=(15, 3 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_show):
        axes[i, 0].imshow(afm[i, afm.shape[1] // 2], cmap="afmhot", vmin=0, vmax=1)
        axes[i, 0].set_title("AFM mid-slice")
        axes[i, 1].imshow(np.transpose(structure_map_to_rgb(real[i, :12]), (1, 2, 0)))
        axes[i, 1].set_title("GT 2D structure")
        axes[i, 2].imshow(np.transpose(structure_map_to_rgb(pred[i, :12]), (1, 2, 0)))
        axes[i, 2].set_title("Pred 2D structure")
        axes[i, 3].imshow(np.transpose(z_map_to_rgb(real[i, 12], real[i, 0]), (1, 2, 0)))
        axes[i, 3].set_title("GT z-map")
        axes[i, 4].imshow(np.transpose(z_map_to_rgb(pred[i, 12], pred[i, 0]), (1, 2, 0)))
        axes[i, 4].set_title("Pred z-map")
        for j in range(5):
            axes[i, j].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate(model, center_type_head, val_loader, config, device):
    model.eval()
    if center_type_head is not None:
        center_type_head.eval()
    agg = {}
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            afm = batch["afm_stack"].to(device)
            pred, features = model.forward_with_features(afm)
            pred = (pred + 1.0) * 0.5
            metrics = compute_v19_joint_metrics(pred, batch, config["img_size"])
            if center_type_head is not None:
                _, type_logits = center_type_head.compute_loss(
                    batch["coords"].to(device),
                    features["enc1"],
                    afm,
                    batch["atom_types"].to(device),
                    batch["atom_mask"].to(device),
                    class_weight=None,
                )
                type_pred = type_logits.argmax(dim=-1)
                type_metrics = _discrete_type_metrics(
                    type_pred,
                    batch["atom_types"].to(device),
                    batch["atom_mask"].to(device),
                )
                metrics.update(type_metrics)
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

    model = V19JointUNet(in_channels=10, base_ch=config.get("base_ch", 64)).to(device)
    lambda_center_type_aux = config.get("lambda_center_type_aux", 0.0)
    center_type_head = None
    if lambda_center_type_aux > 0:
        center_type_head = CenterConditionedTypeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)
    teacher_encoder = None
    teacher_classifier = None
    lambda_teacher_type_distill = config.get("lambda_teacher_type_distill", 0.0)
    teacher_temperature = float(config.get("teacher_temperature", 1.5))
    teacher_ckpt = config.get("teacher_type_checkpoint", "")
    if teacher_ckpt:
        teacher_encoder, teacher_classifier, _ = build_type_teacher(teacher_ckpt, device)
    discriminator = PatchDiscriminator(in_channels=10 + V19_JOINT_TARGET_CHANNELS).to(device)

    g_params = list(model.parameters())
    if center_type_head is not None:
        g_params += list(center_type_head.parameters())
    opt_g = optim.Adam(g_params, lr=config.get("lr", 2e-4), betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=config.get("lr", 2e-4), betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    lambda_atom = config.get("lambda_atom", 20.0)
    lambda_bond = config.get("lambda_bond", 10.0)
    lambda_type = config.get("lambda_type", 20.0)
    lambda_z = config.get("lambda_z", 20.0)
    lambda_z_start = config.get("lambda_z_start", lambda_z)
    z_warmup_epochs = int(config.get("z_warmup_epochs", 1))
    lambda_center_local = config.get("lambda_center_local", 0.0)
    lambda_type_local = config.get("lambda_type_local", 0.0)
    type_class_weights = TYPE_CLASS_WEIGHTS.to(device)

    save_dir = Path(config.get("save_dir", "experiments/v19_joint_debug/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_key = None
    epochs = int(config.get("epochs", 1))
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        discriminator.train()
        g_losses = []
        d_losses = []
        current_lambda_z = scheduled_weight(epoch, epochs, lambda_z, lambda_z_start, z_warmup_epochs)

        pbar = tqdm(train_loader, desc=f"V19.1-joint [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)
            real = build_targets(batch, config["img_size"], device)
            real_tanh = real * 2.0 - 1.0

            fake = model(afm)

            d_real = discriminator(real_tanh, afm)
            d_fake = discriminator(fake.detach(), afm)
            d_loss_real = criterion_gan(d_real, torch.ones_like(d_real))
            d_loss_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            fake, features = model.forward_with_features(afm)
            d_fake = discriminator(fake, afm)
            g_gan = criterion_gan(d_fake, torch.ones_like(d_fake))
            fake_01 = (fake + 1.0) * 0.5

            g_atom = criterion_l1(fake_01[:, 0:1], real[:, 0:1]) * lambda_atom
            g_bond = criterion_l1(fake_01[:, 1:2], real[:, 1:2]) * lambda_bond
            g_type = criterion_l1(fake_01[:, 2:12], real[:, 2:12]) * lambda_type
            z_mask = real[:, 0:1]
            g_z = masked_l1(fake_01[:, 12:13], real[:, 12:13], z_mask) * current_lambda_z
            center_scores, local_type_scores, gt_local_types = gather_gt_center_predictions(fake, batch, config["img_size"])
            if center_scores.numel() > 0:
                g_center_local = ((1.0 - torch.sigmoid(center_scores)).mean()) * lambda_center_local
                g_type_local = nn.functional.cross_entropy(
                    local_type_scores,
                    gt_local_types,
                    weight=type_class_weights,
                ) * lambda_type_local
            else:
                g_center_local = torch.tensor(0.0, device=device)
                g_type_local = torch.tensor(0.0, device=device)

            if center_type_head is not None:
                g_center_type_aux, center_logits = center_type_head.compute_loss(
                    batch["coords"].to(device),
                    features["enc1"],
                    afm,
                    batch["atom_types"].to(device),
                    batch["atom_mask"].to(device),
                    class_weight=type_class_weights,
                )
                g_center_type_aux = g_center_type_aux * lambda_center_type_aux
            else:
                g_center_type_aux = torch.tensor(0.0, device=device)
                center_logits = None

            if teacher_encoder is not None and teacher_classifier is not None and center_logits is not None and lambda_teacher_type_distill > 0:
                with torch.no_grad():
                    _, teacher_patches = teacher_encoder(afm)
                    teacher_logits = teacher_classifier(
                        batch["coords"].to(device),
                        teacher_patches,
                        batch["atom_mask"].to(device),
                        afm_stack=afm,
                    )
                valid = (batch["atom_mask"].to(device) > 0) & (batch["atom_types"].to(device) >= 0)
                if valid.sum() > 0:
                    s = center_logits[valid] / teacher_temperature
                    t = teacher_logits[valid] / teacher_temperature
                    g_teacher_distill = (
                        F.kl_div(
                            F.log_softmax(s, dim=-1),
                            F.softmax(t, dim=-1),
                            reduction="batchmean",
                        )
                        * (teacher_temperature ** 2)
                        * lambda_teacher_type_distill
                    )
                else:
                    g_teacher_distill = torch.tensor(0.0, device=device)
            else:
                g_teacher_distill = torch.tensor(0.0, device=device)

            g_loss = g_gan + g_atom + g_bond + g_type + g_z + g_center_local + g_type_local + g_center_type_aux + g_teacher_distill

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))
            pbar.set_postfix(g=f"{np.mean(g_losses[-10:]):.3f}", d=f"{np.mean(d_losses[-10:]):.3f}")

        train_metrics = {
            "g_loss": float(np.mean(g_losses)) if g_losses else 0.0,
            "d_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "lambda_z": float(current_lambda_z),
        }
        history["train"].append(train_metrics)
        val_metrics = evaluate(model, center_type_head, val_loader, config, device)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"G {train_metrics['g_loss']:.4f} | D {train_metrics['d_loss']:.4f} | z_w {train_metrics['lambda_z']:.2f} | "
            f"atom_xy_mae {val_metrics['atom_xy_mae']:.4f} | "
            f"bond_map_mae {val_metrics['bond_map_mae']:.4f} | "
            f"type_map_mae {val_metrics['type_map_mae']:.4f} | "
            f"z_map_mae {val_metrics['z_map_mae']:.4f} | "
            f"center_r3 {val_metrics['atom_center_score_r3']:.4f} | "
            f"typed_center_r3 {val_metrics['typed_center_score_r3']:.4f} | "
            f"type_acc_r3 {val_metrics['type_top1_local_acc_r3']:.4f} | "
            f"macro_f1_2d {val_metrics['atom_type_macro_f1_2d']:.4f} | "
            f"z_mae_r3 {val_metrics['atom_z_mae_r3']:.4f}"
        )
        if center_type_head is not None:
            print(
                f"  center_type_acc_gtcoord {val_metrics['center_type_acc_gtcoord']:.4f} | "
                f"center_macro_f1_gtcoord {val_metrics['center_macro_f1_gtcoord']:.4f} | "
                f"center_hetero_f1_gtcoord {val_metrics['center_hetero_f1_gtcoord']:.4f} | "
                f"center_collapse_gtcoord {val_metrics['center_ch_collapse_rate_gtcoord']:.4f}"
            )

        current_key = (
            float(val_metrics.get("center_macro_f1_gtcoord", 0.0)),
            float(val_metrics.get("center_type_acc_gtcoord", 0.0)),
            float(val_metrics["typed_center_score_r3"]),
            float(val_metrics["atom_center_score_r3"]),
            -float(val_metrics["z_map_mae"]),
            -float(val_metrics["atom_z_mae_r3"]),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "center_type_head": center_type_head.state_dict() if center_type_head is not None else None,
                    "discriminator": discriminator.state_dict(),
                    "val_metrics": val_metrics,
                },
                save_dir / "best_v19_joint.pt",
            )
            save_preview(model, val_loader, config, device, save_dir / "best_preview.png")

    with open(save_dir / "history_v19_joint.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "center_type_head": center_type_head.state_dict() if center_type_head is not None else None,
            "discriminator": discriminator.state_dict(),
        },
        save_dir / "last_v19_joint.pt",
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
        config["save_dir"] = str(base_dir / "experiments" / "v19_joint_debug" / "checkpoints")

    train(config)
