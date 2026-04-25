"""
Systematic review and visualization for a V19 object-joint training run.

Outputs:
- curve plots from history
- sample-level visual review for best / median / worst validation samples
- markdown + json summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import create_dataloaders
from src.data.dataset import ATOM_TYPES
from src.models.v19_center_edge_head import CenterConditionedEdgeHead
from src.models.v19_center_type_head import CenterConditionedTypeHead
from src.models.v19_joint_model import V19JointUNet
from src.train_v19_object_joint import (
    _edge_metrics_named,
    _object_score,
    _type_metrics_named,
    build_edge_labels,
    build_peak_center_coords,
    build_targets,
)
from src.utils.mol2d import structure_map_to_rgb, z_map_to_rgb
from src.utils.visualize import ATOM_COLORS, ATOM_SIZES


@dataclass
class SampleReview:
    dataset_index: int
    peak_object_score: float
    gt_object_score: float
    atom_center_score_r3: float
    typed_center_score_r3: float
    atom_type_macro_f1_2d: float
    atom_z_mae_r3: float
    peak_center_type_acc: float
    peak_center_macro_f1: float
    peak_center_hetero_f1: float
    peak_center_edge_f1: float
    peak_center_shift_px: float
    gt_center_type_acc: float
    gt_center_macro_f1: float
    gt_center_hetero_f1: float
    gt_center_edge_f1: float
    figure_path: str = ""


def _per_sample_dense_metrics(
    pred_01: torch.Tensor,
    center_map_01: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
    img_size: int,
) -> dict[str, float]:
    pred = pred_01.detach().cpu().numpy()
    center_map = center_map_01.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    coords_np = coords.detach().cpu().numpy()
    types_np = atom_types.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    atom_xy_mae = float(np.abs(pred[0] - target_np[0]).mean())
    occ_mask = target_np[0] > 0.05
    z_map_mae = float(np.abs(pred[12][occ_mask] - target_np[12][occ_mask]).mean()) if occ_mask.any() else 0.0

    from src.utils.mol2d import project_xy_to_pixels
    from src.train_v19_object_joint import _macro_f1_from_lists

    pix = project_xy_to_pixels(coords_np, img_size)
    valid_idx = np.where(mask_np > 0.5)[0]
    center_scores = []
    typed_scores = []
    z_errs = []
    gt_type_all = []
    pred_type_all = []
    for idx in valid_idx:
        gt_t = int(types_np[idx])
        if gt_t < 0 or gt_t >= 10:
            continue
        x, y = map(int, pix[idx])
        x0 = max(0, x - 3)
        x1 = min(img_size, x + 4)
        y0 = max(0, y - 3)
        y1 = min(img_size, y + 4)
        occ_patch = center_map[0, y0:y1, x0:x1]
        type_patch = pred[2:12, y0:y1, x0:x1]
        z_patch = pred[12, y0:y1, x0:x1]
        occ_max = float(occ_patch.max()) if occ_patch.size else 0.0
        center_scores.append(occ_max)
        gt_z01 = float(np.clip((coords_np[idx, 2] + 1.0) * 0.5, 0.0, 1.0))
        if occ_patch.size:
            flat_idx = int(np.argmax(occ_patch))
            oy, ox = np.unravel_index(flat_idx, occ_patch.shape)
            pred_t = int(np.argmax(type_patch[:, oy, ox]))
            typed_scores.append(float(occ_patch[oy, ox] * type_patch[gt_t, oy, ox]))
            pred_z01 = float(z_patch[oy, ox])
            z_errs.append(abs(pred_z01 - gt_z01) * 24.0)
        else:
            pred_t = 1
            typed_scores.append(0.0)
            z_errs.append(24.0 * gt_z01)
        gt_type_all.append(gt_t)
        pred_type_all.append(pred_t)

    return {
        "atom_xy_mae": atom_xy_mae,
        "atom_center_score_r3": float(np.mean(center_scores)) if center_scores else 0.0,
        "typed_center_score_r3": float(np.mean(typed_scores)) if typed_scores else 0.0,
        "atom_type_macro_f1_2d": _macro_f1_from_lists(gt_type_all, pred_type_all, n_classes=10),
        "z_map_mae": z_map_mae,
        "atom_z_mae_r3": float(np.mean(z_errs)) if z_errs else 0.0,
    }


def _plot_curves(history: dict, output_path: Path) -> None:
    train = history["train"]
    val = history["val"]
    epochs = np.arange(1, len(train) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(epochs, [x["loss"] for x in train], marker="o", label="train_loss")
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [x["peak_object_score"] for x in val], marker="o", label="peak")
    ax.plot(epochs, [x["gt_object_score"] for x in val], marker="o", label="gt")
    ax.set_title("Object Score")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, [x["peak_center_type_acc"] for x in val], marker="o", label="peak_type_acc")
    ax.plot(epochs, [x["peak_center_macro_f1"] for x in val], marker="o", label="peak_macro_f1")
    ax.plot(epochs, [x["peak_center_hetero_f1"] for x in val], marker="o", label="peak_hetero_f1")
    ax.set_title("Pred-Center Type")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [x["peak_center_edge_f1"] for x in val], marker="o", label="peak_edge_f1")
    ax.plot(epochs, [x["peak_center_shift_px"] for x in val], marker="o", label="peak_shift_px")
    ax.set_title("Pred-Center Edge / Shift")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [x["atom_center_score_r3"] for x in val], marker="o", label="center_score_r3")
    ax.plot(epochs, [x["typed_center_score_r3"] for x in val], marker="o", label="typed_center_score_r3")
    ax.plot(epochs, [x["atom_type_macro_f1_2d"] for x in val], marker="o", label="type_macro_f1_2d")
    ax.set_title("Dense 2D Signals")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, [x["atom_z_mae_r3"] for x in val], marker="o", label="atom_z_mae_r3")
    ax.plot(epochs, [x["z_map_mae"] for x in val], marker="o", label="z_map_mae")
    ax.set_title("Z Error")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_schedule(history: dict, output_path: Path) -> None:
    train = history["train"]
    epochs = np.arange(1, len(train) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs, [x["center_alpha"] for x in train], marker="o", label="center_alpha")
    ax.plot(epochs, [x["lambda_type_peak"] for x in train], marker="o", label="lambda_type_peak")
    ax.plot(epochs, [x["lambda_edge_peak"] for x in train], marker="o", label="lambda_edge_peak")
    ax.set_title("Curriculum Weights")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [x["lambda_peak_consistency"] for x in train], marker="o", label="lambda_peak_consistency")
    ax.plot(epochs, [x["lr"] for x in train], marker="o", label="lr")
    ax.set_title("Consistency / LR")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _load_model(checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    model = V19JointUNet(in_channels=10, base_ch=config.get("base_ch", 64)).to(device)
    type_head = CenterConditionedTypeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)
    edge_head = CenterConditionedEdgeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)
    model.load_state_dict(state["model"], strict=False)
    type_head.load_state_dict(state["type_head"], strict=False)
    edge_head.load_state_dict(state["edge_head"], strict=False)
    model.eval()
    type_head.eval()
    edge_head.eval()
    return model, type_head, edge_head, config, state


def _make_sample_figure(
    afm: np.ndarray,
    gt_target: np.ndarray,
    pred_01: np.ndarray,
    center_map_01: np.ndarray,
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    pred_edge_adj: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.20)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(afm[afm.shape[0] // 2], cmap="afmhot", vmin=0, vmax=1)
    ax.set_title("AFM Mid Slice")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.transpose(structure_map_to_rgb(gt_target[:12]), (1, 2, 0)))
    ax.set_title("GT Dense 2D")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    pred_struct = pred_01[:12].copy()
    pred_struct[0] = center_map_01[0]
    ax.imshow(np.transpose(structure_map_to_rgb(pred_struct), (1, 2, 0)))
    ax.set_title("Pred Dense 2D")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    z_rgb = np.concatenate(
        [
            np.transpose(z_map_to_rgb(gt_target[12], gt_target[0]), (1, 2, 0)),
            np.transpose(z_map_to_rgb(pred_01[12], pred_01[0]), (1, 2, 0)),
        ],
        axis=1,
    )
    ax.imshow(z_rgb)
    ax.set_title("GT Z | Pred Z")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    _plot_object_2d(ax, gt_coords, gt_types, gt_mask, gt_edge_adj, "GT Object 2D")
    ax = fig.add_subplot(gs[1, 1])
    _plot_object_2d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred Object 2D")

    ax = fig.add_subplot(gs[1, 2], projection="3d")
    _plot_object_3d(ax, gt_coords, gt_types, gt_mask, gt_edge_adj, "GT Object 3D")
    ax = fig.add_subplot(gs[1, 3], projection="3d")
    _plot_object_3d(ax, pred_coords, pred_types, pred_mask, pred_edge_adj, "Pred Object 3D")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _adjacency_from_logits(logits: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    pred = probs > threshold
    adj = np.zeros_like(pred, dtype=np.int32)
    for i in range(pred.shape[0]):
        if not mask_np[i]:
            continue
        for j in range(i + 1, pred.shape[1]):
            if not mask_np[j]:
                continue
            if pred[i, j]:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def _plot_object_2d(ax, coords: np.ndarray, atom_types: np.ndarray, mask: np.ndarray, edge_adj: np.ndarray, title: str) -> None:
    valid = mask > 0.5
    c = coords[valid]
    t = atom_types[valid]
    if c.shape[0] == 0:
        ax.set_title(title)
        ax.axis("off")
        return
    valid_idx = np.where(valid)[0]
    idx_to_local = {int(idx): i for i, idx in enumerate(valid_idx.tolist())}
    for i_idx in valid_idx:
        for j_idx in valid_idx:
            if j_idx <= i_idx:
                continue
            if edge_adj[i_idx, j_idx] > 0:
                i = idx_to_local[int(i_idx)]
                j = idx_to_local[int(j_idx)]
                ax.plot([c[i, 0], c[j, 0]], [c[i, 1], c[j, 1]], color="#666666", linewidth=1.2, alpha=0.85)
    for i in range(len(c)):
        idx = int(t[i])
        elem = ATOM_TYPES[idx] if 0 <= idx < len(ATOM_TYPES) else "C"
        color = ATOM_COLORS.get(elem, "#999999")
        size = max(30, ATOM_SIZES.get(elem, 50))
        ax.scatter(c[i, 0], c[i, 1], c=color, s=size, edgecolors="black", linewidths=0.5, alpha=0.95)
        ax.text(c[i, 0], c[i, 1], elem, fontsize=7, ha="center", va="center")
    mins = c[:, :2].min(axis=0)
    maxs = c[:, :2].max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = max((maxs - mins).max() / 2.0, 0.5) + 0.2
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(alpha=0.25)


def _plot_object_3d(ax, coords: np.ndarray, atom_types: np.ndarray, mask: np.ndarray, edge_adj: np.ndarray, title: str) -> None:
    valid = mask > 0.5
    c = coords[valid]
    t = atom_types[valid]
    if c.shape[0] == 0:
        ax.set_title(title)
        return
    valid_idx = np.where(valid)[0]
    idx_to_local = {int(idx): i for i, idx in enumerate(valid_idx.tolist())}
    for i_idx in valid_idx:
        for j_idx in valid_idx:
            if j_idx <= i_idx:
                continue
            if edge_adj[i_idx, j_idx] > 0:
                i = idx_to_local[int(i_idx)]
                j = idx_to_local[int(j_idx)]
                ax.plot(
                    [c[i, 0], c[j, 0]],
                    [c[i, 1], c[j, 1]],
                    [c[i, 2], c[j, 2]],
                    color="#666666",
                    linewidth=1.2,
                    alpha=0.85,
                )
    for i in range(len(c)):
        idx = int(t[i])
        elem = ATOM_TYPES[idx] if 0 <= idx < len(ATOM_TYPES) else "C"
        color = ATOM_COLORS.get(elem, "#999999")
        size = ATOM_SIZES.get(elem, 50)
        ax.scatter(c[i, 0], c[i, 1], c[i, 2], c=color, s=size, edgecolors="black", linewidths=0.5, alpha=0.95)
    mins = c.min(axis=0)
    maxs = c.max(axis=0)
    center = (mins + maxs) / 2.0
    half_range = max((maxs - mins).max() / 2.0, 0.5) + 0.2
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _sample_reviews(model, type_head, edge_head, val_loader, config: dict, device: torch.device, output_dir: Path) -> list[SampleReview]:
    reviews: list[SampleReview] = []
    img_size = int(config["img_size"])
    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Review", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, img_size, device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)

            for bi in range(afm.shape[0]):
                dense = _per_sample_dense_metrics(
                    pred_01[bi],
                    center_map_01[bi],
                    targets[bi],
                    coords[bi],
                    atom_types[bi],
                    mask[bi],
                    img_size,
                )

                sample_mask = mask[bi : bi + 1]
                sample_edge_labels = edge_labels[bi : bi + 1]
                sample_coords = coords[bi : bi + 1]
                sample_types = atom_types[bi : bi + 1]
                sample_afm = afm[bi : bi + 1]
                sample_feat = features["enc1"][bi : bi + 1]
                sample_pred = pred_01[bi : bi + 1]
                sample_center_map = center_map_01[bi : bi + 1]

                _, sample_gt_type_logits = type_head.compute_loss(sample_coords, sample_feat, sample_afm, sample_types, sample_mask, class_weight=None)
                sample_gt_type_pred = sample_gt_type_logits.argmax(dim=-1)
                sample_gt_type = _type_metrics_named(sample_gt_type_pred, sample_types, sample_mask, prefix="gt_center")
                _, sample_gt_edge_logits = edge_head.compute_loss(sample_coords, sample_feat, sample_afm, sample_mask, sample_edge_labels)
                sample_gt_edge = _edge_metrics_named(sample_gt_edge_logits, sample_edge_labels, sample_mask, prefix="gt_center")
                sample_gt_edge_adj = sample_edge_labels[0].detach().cpu().numpy().astype(np.int32)

                sample_peak_coords, sample_shift = build_peak_center_coords(
                    sample_center_map,
                    sample_pred[:, 12:13],
                    sample_coords,
                    sample_mask,
                    img_size,
                    alpha=1.0,
                    search_radius=int(config.get("center_search_radius", 3)),
                )
                _, sample_peak_type_logits = type_head.compute_loss(sample_peak_coords, sample_feat, sample_afm, sample_types, sample_mask, class_weight=None)
                sample_peak_type_pred = sample_peak_type_logits.argmax(dim=-1)
                sample_peak_type = _type_metrics_named(sample_peak_type_pred, sample_types, sample_mask, prefix="peak_center")
                _, sample_peak_edge_logits = edge_head.compute_loss(sample_peak_coords, sample_feat, sample_afm, sample_mask, sample_edge_labels)
                sample_peak_edge = _edge_metrics_named(sample_peak_edge_logits, sample_edge_labels, sample_mask, prefix="peak_center")
                sample_peak_edge_adj = _adjacency_from_logits(sample_peak_edge_logits[0], sample_mask[0])

                gt_object_score = _object_score(
                    sample_gt_type["gt_center_type_acc"],
                    sample_gt_type["gt_center_macro_f1"],
                    sample_gt_type["gt_center_hetero_f1"],
                    sample_gt_edge["gt_center_edge_f1"],
                    dense["atom_center_score_r3"],
                    dense["atom_z_mae_r3"],
                    0.0,
                )
                peak_object_score = _object_score(
                    sample_peak_type["peak_center_type_acc"],
                    sample_peak_type["peak_center_macro_f1"],
                    sample_peak_type["peak_center_hetero_f1"],
                    sample_peak_edge["peak_center_edge_f1"],
                    dense["atom_center_score_r3"],
                    dense["atom_z_mae_r3"],
                    sample_shift,
                )

                reviews.append(
                    SampleReview(
                        dataset_index=sample_index,
                        peak_object_score=float(peak_object_score),
                        gt_object_score=float(gt_object_score),
                        atom_center_score_r3=float(dense["atom_center_score_r3"]),
                        typed_center_score_r3=float(dense["typed_center_score_r3"]),
                        atom_type_macro_f1_2d=float(dense["atom_type_macro_f1_2d"]),
                        atom_z_mae_r3=float(dense["atom_z_mae_r3"]),
                        peak_center_type_acc=float(sample_peak_type["peak_center_type_acc"]),
                        peak_center_macro_f1=float(sample_peak_type["peak_center_macro_f1"]),
                        peak_center_hetero_f1=float(sample_peak_type["peak_center_hetero_f1"]),
                        peak_center_edge_f1=float(sample_peak_edge["peak_center_edge_f1"]),
                        peak_center_shift_px=float(sample_shift),
                        gt_center_type_acc=float(sample_gt_type["gt_center_type_acc"]),
                        gt_center_macro_f1=float(sample_gt_type["gt_center_macro_f1"]),
                        gt_center_hetero_f1=float(sample_gt_type["gt_center_hetero_f1"]),
                        gt_center_edge_f1=float(sample_gt_edge["gt_center_edge_f1"]),
                    )
                )
                sample_index += 1

    reviews.sort(key=lambda x: x.peak_object_score, reverse=True)
    if not reviews:
        return reviews

    selected = {
        "best": reviews[0],
        "median": reviews[len(reviews) // 2],
        "worst": reviews[-1],
    }
    idx_to_rank = {v.dataset_index: k for k, v in selected.items()}

    sample_index = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Review figs", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, img_size, device)
            edge_labels = build_edge_labels(batch, device)
            pred, features = model.forward_with_features(afm)
            pred_01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            center_map_01 = torch.sigmoid(features["center_logits"]).clamp(0.0, 1.0)
            for bi in range(afm.shape[0]):
                if sample_index in idx_to_rank:
                    review = next(r for r in reviews if r.dataset_index == sample_index)
                    rank_name = idx_to_rank[sample_index]
                    title = (
                        f"{rank_name.upper()} | idx={sample_index} | peak_score={review.peak_object_score:.3f} | "
                        f"type_acc={review.peak_center_type_acc:.3f} | macro_f1={review.peak_center_macro_f1:.3f} | "
                        f"hetero_f1={review.peak_center_hetero_f1:.3f} | edge_f1={review.peak_center_edge_f1:.3f} | "
                        f"shift={review.peak_center_shift_px:.2f}px | z_mae={review.atom_z_mae_r3:.3f}"
                    )
                    fig_path = output_dir / "samples" / f"{rank_name}_sample_{sample_index:04d}.png"
                    sample_mask = mask[bi : bi + 1]
                    sample_edge_labels = edge_labels[bi : bi + 1]
                    sample_coords = coords[bi : bi + 1]
                    sample_types = atom_types[bi : bi + 1]
                    sample_afm = afm[bi : bi + 1]
                    sample_feat = features["enc1"][bi : bi + 1]
                    sample_pred = pred_01[bi : bi + 1]
                    sample_center_map = center_map_01[bi : bi + 1]

                    sample_gt_edge_adj = sample_edge_labels[0].detach().cpu().numpy().astype(np.int32)
                    sample_peak_coords, _ = build_peak_center_coords(
                        sample_center_map,
                        sample_pred[:, 12:13],
                        sample_coords,
                        sample_mask,
                        img_size,
                        alpha=1.0,
                        search_radius=int(config.get("center_search_radius", 3)),
                    )
                    _, sample_peak_type_logits = type_head.compute_loss(
                        sample_peak_coords, sample_feat, sample_afm, sample_types, sample_mask, class_weight=None
                    )
                    sample_peak_type_pred = sample_peak_type_logits.argmax(dim=-1)
                    _, sample_peak_edge_logits = edge_head.compute_loss(
                        sample_peak_coords, sample_feat, sample_afm, sample_mask, sample_edge_labels
                    )
                    sample_peak_edge_adj = _adjacency_from_logits(sample_peak_edge_logits[0], sample_mask[0])
                    _make_sample_figure(
                        afm[bi].detach().cpu().numpy(),
                        targets[bi].detach().cpu().numpy(),
                        pred_01[bi].detach().cpu().numpy(),
                        center_map_01[bi].detach().cpu().numpy(),
                        sample_coords[0].detach().cpu().numpy(),
                        sample_types[0].detach().cpu().numpy(),
                        sample_mask[0].detach().cpu().numpy(),
                        sample_gt_edge_adj,
                        sample_peak_coords[0].detach().cpu().numpy(),
                        sample_peak_type_pred[0].detach().cpu().numpy(),
                        sample_mask[0].detach().cpu().numpy(),
                        sample_peak_edge_adj,
                        title,
                        fig_path,
                    )
                    review.figure_path = str(fig_path)
                sample_index += 1
    return reviews


def _build_report(history: dict, state: dict, reviews: list[SampleReview], report_title: str) -> tuple[dict, str]:
    best_epoch = int(state["epoch"])
    best_val = history["val"][best_epoch - 1]
    first_val = history["val"][0]

    summary = {
        "best_epoch": best_epoch,
        "best_metrics": best_val,
        "epoch1_metrics": first_val,
        "improvements": {
            "peak_object_score_delta": float(best_val["peak_object_score"] - first_val["peak_object_score"]),
            "peak_center_type_acc_delta": float(best_val["peak_center_type_acc"] - first_val["peak_center_type_acc"]),
            "peak_center_macro_f1_delta": float(best_val["peak_center_macro_f1"] - first_val["peak_center_macro_f1"]),
            "peak_center_hetero_f1_delta": float(best_val["peak_center_hetero_f1"] - first_val["peak_center_hetero_f1"]),
            "peak_center_edge_f1_delta": float(best_val["peak_center_edge_f1"] - first_val["peak_center_edge_f1"]),
            "peak_center_shift_px_delta": float(best_val["peak_center_shift_px"] - first_val["peak_center_shift_px"]),
            "atom_z_mae_r3_delta": float(best_val["atom_z_mae_r3"] - first_val["atom_z_mae_r3"]),
        },
        "top_samples": [review.__dict__ for review in reviews[:5]],
        "bottom_samples": [review.__dict__ for review in reviews[-5:]],
    }

    metric_desc = [
        ("best_epoch", "最佳轮次"),
        ("peak_object_score", "预测中心闭环对象级总分，越高越好"),
        ("gt_object_score", "真实中心条件下的对象级总分，表示当前上限参考，越高越好"),
        ("atom_center_score_r3", "原子中心命中分数，统计真实中心半径3像素内的预测响应，越高越好"),
        ("peak_center_type_acc", "预测中心条件下的原子类型准确率，越高越好"),
        ("peak_center_macro_f1", "预测中心条件下的原子类型宏平均F1，越高越好"),
        ("peak_center_hetero_f1", "预测中心条件下的杂原子F1，越高越好"),
        ("peak_center_edge_f1", "预测中心条件下的对象级边F1，越高越好"),
        ("peak_center_shift_px", "预测中心相对真实中心的平均偏移，单位像素，越低越好"),
        ("atom_z_mae_r3", "真实中心附近的z轴平均绝对误差，越低越好"),
        ("typed_center_score_r3", "半径3像素内位置与类型同时正确的软分数，越高越好"),
        ("atom_type_macro_f1_2d", "稠密2D类型图的原子类型宏平均F1，越高越好"),
        ("atom_xy_mae", "稠密2D原子图平均绝对误差，越低越好"),
        ("z_map_mae", "稠密z图平均绝对误差，越低越好"),
        ("gt_center_type_acc", "真实中心条件下的原子类型准确率，越高越好"),
        ("gt_center_macro_f1", "真实中心条件下的原子类型宏平均F1，越高越好"),
        ("gt_center_hetero_f1", "真实中心条件下的杂原子F1，越高越好"),
        ("gt_center_edge_f1", "真实中心条件下的对象级边F1，越高越好"),
    ]

    epoch_table_metrics = [
        "peak_object_score",
        "gt_object_score",
        "atom_center_score_r3",
        "peak_center_type_acc",
        "peak_center_macro_f1",
        "peak_center_hetero_f1",
        "peak_center_edge_f1",
        "peak_center_shift_px",
        "atom_z_mae_r3",
        "typed_center_score_r3",
        "atom_type_macro_f1_2d",
        "atom_xy_mae",
        "z_map_mae",
        "gt_center_type_acc",
        "gt_center_macro_f1",
        "gt_center_hetero_f1",
        "gt_center_edge_f1",
    ]

    md = []
    md.append(f"# {report_title}")
    md.append("")
    md.append("## 一、最佳结果")
    md.append(f"- 字段名 `best_epoch`：最佳轮次 = `{best_epoch}`")
    md.append(f"- 字段名 `peak_object_score`：预测中心闭环对象级总分 = `{best_val['peak_object_score']:.4f}`")
    md.append(f"- 字段名 `gt_object_score`：真实中心条件对象级总分 = `{best_val['gt_object_score']:.4f}`")
    md.append(f"- 字段名 `atom_center_score_r3`：原子中心命中分数 = `{best_val['atom_center_score_r3']:.4f}`")
    md.append(f"- 字段名 `peak_center_type_acc`：预测中心原子类型准确率 = `{best_val['peak_center_type_acc']:.4f}`")
    md.append(f"- 字段名 `peak_center_macro_f1`：预测中心原子类型宏平均F1 = `{best_val['peak_center_macro_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_hetero_f1`：预测中心杂原子F1 = `{best_val['peak_center_hetero_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_edge_f1`：预测中心对象级边F1 = `{best_val['peak_center_edge_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_shift_px`：预测中心平均偏移(像素) = `{best_val['peak_center_shift_px']:.4f}`")
    md.append(f"- 字段名 `atom_z_mae_r3`：z轴平均绝对误差 = `{best_val['atom_z_mae_r3']:.4f}`")
    md.append(f"- 字段名 `typed_center_score_r3`：位置与类型同时正确的软分数 = `{best_val['typed_center_score_r3']:.4f}`")
    md.append(f"- 字段名 `atom_type_macro_f1_2d`：稠密2D类型图宏平均F1 = `{best_val['atom_type_macro_f1_2d']:.4f}`")
    md.append("")
    md.append("## 二、训练趋势")
    md.append(f"- 字段名 `peak_object_score`：`{first_val['peak_object_score']:.4f}` -> `{best_val['peak_object_score']:.4f}`")
    md.append(f"- 字段名 `peak_center_type_acc`：`{first_val['peak_center_type_acc']:.4f}` -> `{best_val['peak_center_type_acc']:.4f}`")
    md.append(f"- 字段名 `peak_center_macro_f1`：`{first_val['peak_center_macro_f1']:.4f}` -> `{best_val['peak_center_macro_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_hetero_f1`：`{first_val['peak_center_hetero_f1']:.4f}` -> `{best_val['peak_center_hetero_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_edge_f1`：`{first_val['peak_center_edge_f1']:.4f}` -> `{best_val['peak_center_edge_f1']:.4f}`")
    md.append(f"- 字段名 `peak_center_shift_px`：`{first_val['peak_center_shift_px']:.4f}` -> `{best_val['peak_center_shift_px']:.4f}`")
    md.append(f"- 字段名 `atom_z_mae_r3`：`{first_val['atom_z_mae_r3']:.4f}` -> `{best_val['atom_z_mae_r3']:.4f}`")
    md.append("")
    md.append("## 三、核心结论")
    md.append("- 原子中心分支已经基本学稳，`atom_center_score_r3` 已接近饱和。")
    md.append("- 在预测中心闭环条件下，原子类型、杂原子、对象级边和 z 轴误差都出现了持续改善。")
    md.append("- `gt_object_score` 仍明显高于 `peak_object_score`，说明当前主瓶颈仍然是“预测中心到对象级类型/边的迁移损失”，而不是类型头和边头本身完全不会。")
    md.append("- 稠密 `2D` 类型图仍然偏弱，说明后续仍应优先发展对象级类型/边头，而不是继续把主要希望放在稠密 `type map` 上。")
    md.append("")
    md.append("## 四、代表样本复盘")
    for name, review in [("最佳样本", reviews[0]), ("中位样本", reviews[len(reviews)//2]), ("最差样本", reviews[-1])]:
        md.append(
            f"- {name}：样本编号 `dataset_index={review.dataset_index}`；"
            f"`peak_object_score={review.peak_object_score:.4f}`，"
            f"`peak_center_type_acc={review.peak_center_type_acc:.4f}`，"
            f"`peak_center_macro_f1={review.peak_center_macro_f1:.4f}`，"
            f"`peak_center_hetero_f1={review.peak_center_hetero_f1:.4f}`，"
            f"`peak_center_edge_f1={review.peak_center_edge_f1:.4f}`，"
            f"`peak_center_shift_px={review.peak_center_shift_px:.2f}`，"
            f"`atom_z_mae_r3={review.atom_z_mae_r3:.4f}`"
        )
    md.append("")
    md.append("## 五、指标说明")
    for field_name, zh_desc in metric_desc:
        md.append(f"- 字段名 `{field_name}`：{zh_desc}")
    md.append("")
    md.append("## 六、各 Epoch 验证指标变化表")
    md.append("- 说明：`Epoch 1` 显示完整数值；从 `Epoch 2` 开始，各指标显示相对上一轮验证结果的变化值。")
    md.append("")
    md.append("| Epoch | " + " | ".join([f"`{m}`" for m in epoch_table_metrics]) + " |")
    md.append("|---|" + "|".join(["---:"] * len(epoch_table_metrics)) + "|")
    val_hist = history["val"]
    for i, cur in enumerate(val_hist):
        row = [str(i + 1)]
        if i == 0:
            for m in epoch_table_metrics:
                row.append(f"`{cur[m]:.4f}`")
        else:
            prev = val_hist[i - 1]
            for m in epoch_table_metrics:
                row.append(f"`{(cur[m] - prev[m]):+.4f}`")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    return summary, "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--report_title", type=str, default="V19 对象级联合训练复盘报告")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    history_path = Path(args.history)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = json.load(open(history_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, type_head, edge_head, config, state = _load_model(checkpoint_path, device)

    _plot_curves(history, output_dir / "plots" / "curves.png")
    _plot_schedule(history, output_dir / "plots" / "schedule.png")

    train_loader, val_loader, _, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config.get("param_key", "K-1"),
        img_size=config["img_size"],
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        require_ring=config.get("require_ring", False),
        batch_size=args.batch_size,
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
    )
    del train_loader

    reviews = _sample_reviews(model, type_head, edge_head, val_loader, config, device, output_dir)
    summary, report_md = _build_report(history, state, reviews, args.report_title)

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "review_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(reports_dir / "review_summary.md", "w") as f:
        f.write(report_md)

    print(json.dumps({
        "output_dir": str(output_dir),
        "best_epoch": summary["best_epoch"],
        "peak_object_score": summary["best_metrics"]["peak_object_score"],
        "peak_center_type_acc": summary["best_metrics"]["peak_center_type_acc"],
        "peak_center_macro_f1": summary["best_metrics"]["peak_center_macro_f1"],
        "peak_center_hetero_f1": summary["best_metrics"]["peak_center_hetero_f1"],
        "peak_center_edge_f1": summary["best_metrics"]["peak_center_edge_f1"],
        "peak_center_shift_px": summary["best_metrics"]["peak_center_shift_px"],
        "atom_z_mae_r3": summary["best_metrics"]["atom_z_mae_r3"],
    }, indent=2))


if __name__ == "__main__":
    main()
