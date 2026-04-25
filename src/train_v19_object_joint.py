"""
V19 object-level joint training:
AFM stack -> atom center map + object-level type/edge heads + z-map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MAX_ATOMS, create_dataloaders
from src.models.gnn_type_classifier import GNNTypeClassifier
from src.models.v19_center_edge_head import CenterConditionedEdgeHead
from src.models.v19_center_type_head import CenterConditionedTypeHead
from src.models.v19_joint_model import V19JointUNet
from src.models.v20_ablation_heads import LegacyGNNTypeHeadAdapter, ZeroEdgeHead
from src.models.video_vit import VideoViTEncoder
from src.utils.metrics import _hungarian_match_numpy, _macro_type_f1, _safe_f1
from src.utils.mol2d import (
    TYPE_COLORS,
    batch_render_v19_joint_targets,
    infer_bonds_from_coords,
    project_xy_to_pixels,
    structure_map_to_rgb,
    z_map_to_rgb,
)


TYPE_CLASS_WEIGHTS = torch.tensor(
    [0.041, 0.039, 0.077, 0.098, 0.666, 0.207, 6.348, 0.688, 0.921, 0.914],
    dtype=torch.float32,
)


def maybe_strip_z_coords(coords: torch.Tensor, disable_z: bool) -> torch.Tensor:
    if not disable_z:
        return coords
    out = coords.clone()
    out[..., 2] = 0.0
    return out


def maybe_neutralize_z_map(pred_01: torch.Tensor, disable_z: bool) -> torch.Tensor:
    if not disable_z:
        return pred_01
    out = pred_01.clone()
    out[:, 12:13] = 0.5
    return out


def build_targets(batch: dict, img_size: int, device: torch.device) -> torch.Tensor:
    coords = batch["coords"].cpu().numpy()
    types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()
    target = batch_render_v19_joint_targets(coords, types, mask, img_size=img_size)
    return torch.from_numpy(target).to(device)


def build_edge_labels(batch: dict, device: torch.device) -> torch.Tensor:
    coords = batch["coords"].cpu().numpy()
    atom_types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()
    bsz, n_atoms, _ = coords.shape
    labels = np.zeros((bsz, n_atoms, n_atoms), dtype=np.float32)
    for b in range(bsz):
        bonds = infer_bonds_from_coords(coords[b].astype(np.float32) * 12.0, atom_types[b], mask[b])
        for i, j in bonds:
            labels[b, i, j] = 1.0
            labels[b, j, i] = 1.0
    return torch.from_numpy(labels).to(device)


def scheduled_weight(epoch: int, final_weight: float, start_weight: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 1:
        return float(final_weight)
    if epoch <= 1:
        return float(start_weight)
    if epoch >= warmup_epochs:
        return float(final_weight)
    alpha = float(epoch - 1) / float(max(warmup_epochs - 1, 1))
    return float(start_weight + alpha * (final_weight - start_weight))


def build_peak_center_coords(
    center_map_01: torch.Tensor,
    pred_z_01: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    img_size: int,
    alpha: float,
    search_radius: int = 3,
) -> tuple[torch.Tensor, float]:
    """Replace GT centers with local predicted peaks near each GT center.

    This is a curriculum step toward fully predicted centers:
    - alpha = 0.0 -> pure GT coords
    - alpha = 1.0 -> local peak coords from predicted atom/z maps
    """
    if alpha <= 0:
        return coords, 0.0

    atom_map = center_map_01[:, 0].detach()
    z_map = pred_z_01[:, 0].detach()
    out = coords.clone()
    shifts = []

    for b in range(coords.shape[0]):
        valid_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        px = ((coords[b, valid_idx, 0] + 1.0) * 0.5 * (img_size - 1)).round().long().clamp(0, img_size - 1)
        py = ((coords[b, valid_idx, 1] + 1.0) * 0.5 * (img_size - 1)).round().long().clamp(0, img_size - 1)

        for local_i, atom_i in enumerate(valid_idx.tolist()):
            x = int(px[local_i].item())
            y = int(py[local_i].item())
            x0 = max(0, x - search_radius)
            x1 = min(img_size, x + search_radius + 1)
            y0 = max(0, y - search_radius)
            y1 = min(img_size, y + search_radius + 1)
            patch = atom_map[b, y0:y1, x0:x1]
            if patch.numel() == 0:
                continue
            flat = int(torch.argmax(patch).item())
            patch_w = patch.shape[1]
            oy = flat // patch_w
            ox = flat % patch_w
            peak_x = x0 + ox
            peak_y = y0 + oy

            pred_x = (float(peak_x) / float(max(img_size - 1, 1))) * 2.0 - 1.0
            pred_y = (float(peak_y) / float(max(img_size - 1, 1))) * 2.0 - 1.0
            pred_z = float(z_map[b, peak_y, peak_x].item()) * 2.0 - 1.0

            out[b, atom_i, 0] = (1.0 - alpha) * coords[b, atom_i, 0] + alpha * pred_x
            out[b, atom_i, 1] = (1.0 - alpha) * coords[b, atom_i, 1] + alpha * pred_y
            out[b, atom_i, 2] = (1.0 - alpha) * coords[b, atom_i, 2] + alpha * pred_z
            shifts.append(float(((peak_x - x) ** 2 + (peak_y - y) ** 2) ** 0.5))

    return out, float(np.mean(shifts)) if shifts else 0.0


def build_predicted_type_training_batch(
    center_map_01: torch.Tensor,
    pred_01: torch.Tensor,
    count_logits: torch.Tensor,
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
    img_size: int,
    peak_threshold: float = 0.45,
    min_distance_px: int = 2,
    max_objects: int = MAX_ATOMS,
    match_radius_px: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build matched predicted-center supervision for object type closure."""
    peaks_batch, _ = decode_object_peaks(
        center_map_01,
        count_logits=count_logits,
        atom_map_01=pred_01[:, 0:1],
        peak_threshold=peak_threshold,
        min_distance_px=min_distance_px,
        max_objects=max_objects,
    )

    bsz, n_atoms, _ = coords.shape
    device = coords.device
    pred_coords = torch.zeros_like(coords)
    pred_types = torch.full_like(atom_types, -1)
    pred_mask = torch.zeros_like(mask)
    pred_gt_index = torch.full_like(atom_types, -1)

    match_radius_norm = 2.0 * float(match_radius_px) / float(max(img_size - 1, 1))

    for b in range(bsz):
        peaks = peaks_batch[b]
        n_pred = min(len(peaks), n_atoms)
        if n_pred == 0:
            continue

        z_np = pred_01[b, 12].detach().cpu().numpy()
        center_np = center_map_01[b, 0].detach().cpu().numpy()
        pred_xyz = []
        for y, x, _ in peaks[:n_pred]:
            y_ref, x_ref = _refine_peak_xy(center_np, y, x, refine_radius_px=2)
            x_norm = (float(x_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
            y_norm = (float(y_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
            z_norm = _bilinear_sample_np(z_np, y_ref, x_ref) * 2.0 - 1.0
            pred_xyz.append([x_norm, y_norm, z_norm])
        pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
        pred_coords[b, :n_pred] = torch.from_numpy(pred_xyz).to(device=device, dtype=coords.dtype)

        gt_valid = torch.nonzero(mask[b] > 0.5, as_tuple=False).squeeze(-1)
        if gt_valid.numel() == 0:
            continue

        gt_xy = coords[b, gt_valid, :2].detach().cpu().numpy().astype(np.float32)
        pred_xy = pred_xyz[:, :2]
        row_ind, col_ind, cost = _hungarian_match_numpy(pred_xy, gt_xy)

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[r, c] > match_radius_norm:
                continue
            gt_i = int(gt_valid[c].item())
            pred_types[b, r] = atom_types[b, gt_i]
            pred_mask[b, r] = 1.0
            pred_gt_index[b, r] = gt_i

    return pred_coords, pred_types, pred_mask, pred_gt_index


def build_predicted_edge_training_labels(
    pred_gt_index: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_edge_labels: torch.Tensor,
) -> torch.Tensor:
    """Project GT edge labels onto matched predicted proposals.

    Only proposals with a valid GT assignment participate in the label graph.
    Unmatched proposals remain masked out by ``pred_mask`` during loss.
    """
    bsz, n_atoms = pred_gt_index.shape
    device = gt_edge_labels.device
    pred_edge_labels = torch.zeros((bsz, n_atoms, n_atoms), device=device, dtype=gt_edge_labels.dtype)

    for b in range(bsz):
        valid = torch.nonzero((pred_mask[b] > 0.5) & (pred_gt_index[b] >= 0), as_tuple=False).squeeze(-1)
        if valid.numel() <= 1:
            continue
        gt_idx = pred_gt_index[b, valid].long()
        pred_edge_labels[b][valid.unsqueeze(1), valid.unsqueeze(0)] = gt_edge_labels[b][
            gt_idx.unsqueeze(1),
            gt_idx.unsqueeze(0),
        ]

    return pred_edge_labels


def _count_similarity(n_pred: int, n_gt: int) -> float:
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(n_pred - n_gt) / max(n_pred, n_gt)))


def _proposal_support_map(center_map: np.ndarray, atom_map: np.ndarray | None, atom_support_weight: float) -> np.ndarray:
    if atom_map is None or atom_support_weight <= 0:
        return center_map
    support = center_map * ((1.0 - atom_support_weight) + atom_support_weight * np.clip(atom_map, 0.0, 1.0))
    return np.clip(support, 0.0, 1.0)


def _refine_peak_xy(center_map: np.ndarray, y: int, x: int, refine_radius_px: int = 2) -> tuple[float, float]:
    if refine_radius_px <= 0:
        return float(y), float(x)

    h, w = center_map.shape
    x0 = max(0, x - refine_radius_px)
    x1 = min(w, x + refine_radius_px + 1)
    y0 = max(0, y - refine_radius_px)
    y1 = min(h, y + refine_radius_px + 1)
    patch = center_map[y0:y1, x0:x1]
    if patch.size == 0:
        return float(y), float(x)

    patch = np.asarray(patch, dtype=np.float32)
    weights = np.clip(patch - 0.5 * float(patch.max(initial=0.0)), 0.0, None)
    if float(weights.sum()) < 1e-8:
        weights = np.clip(patch, 0.0, None)
    if float(weights.sum()) < 1e-8:
        return float(y), float(x)

    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )
    y_ref = float((weights * yy).sum() / weights.sum())
    x_ref = float((weights * xx).sum() / weights.sum())
    return y_ref, x_ref


def _bilinear_sample_np(arr: np.ndarray, y: float, x: float) -> float:
    h, w = arr.shape
    x = float(np.clip(x, 0.0, max(w - 1, 0)))
    y = float(np.clip(y, 0.0, max(h - 1, 0)))
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    v00 = float(arr[y0, x0])
    v01 = float(arr[y0, x1])
    v10 = float(arr[y1, x0])
    v11 = float(arr[y1, x1])
    top = v00 * (1.0 - wx) + v01 * wx
    bot = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bot * wy


def _decode_single_center_map(
    center_map: np.ndarray,
    target_k: int | None,
    peak_threshold: float,
    min_distance_px: int,
    max_objects: int,
) -> list[tuple[int, int, float]]:
    local_max = maximum_filter(center_map, size=min_distance_px * 2 + 1, mode="nearest")
    peak_mask = center_map >= (local_max - 1e-8)
    ys, xs = np.nonzero(peak_mask)
    scores = center_map[ys, xs]
    order = np.argsort(-scores)

    peaks: list[tuple[int, int, float]] = []

    def _try_add(y: int, x: int, score: float) -> bool:
        for py, px, _ in peaks:
            if (py - y) ** 2 + (px - x) ** 2 < min_distance_px ** 2:
                return False
        peaks.append((int(y), int(x), float(score)))
        return True

    if target_k is None:
        for idx in order.tolist():
            y = int(ys[idx])
            x = int(xs[idx])
            score = float(scores[idx])
            if score < peak_threshold:
                continue
            _try_add(y, x, score)
            if len(peaks) >= max_objects:
                break
        if not peaks:
            y, x = np.unravel_index(int(np.argmax(center_map)), center_map.shape)
            peaks = [(int(y), int(x), float(center_map[y, x]))]
        return peaks

    target_k = int(np.clip(target_k, 1, max_objects))
    for idx in order.tolist():
        y = int(ys[idx])
        x = int(xs[idx])
        score = float(scores[idx])
        _try_add(y, x, score)
        if len(peaks) >= target_k:
            return peaks

    flat_order = np.argsort(-center_map.reshape(-1))
    h, w = center_map.shape
    for flat_idx in flat_order.tolist():
        y = flat_idx // w
        x = flat_idx % w
        score = float(center_map[y, x])
        _try_add(y, x, score)
        if len(peaks) >= target_k:
            break

    if not peaks:
        y, x = np.unravel_index(int(np.argmax(center_map)), center_map.shape)
        peaks = [(int(y), int(x), float(center_map[y, x]))]
    return peaks


def decode_object_peaks(
    center_map_01: torch.Tensor,
    count_logits: torch.Tensor | None,
    atom_map_01: torch.Tensor | None = None,
    peak_threshold: float = 0.45,
    min_distance_px: int = 2,
    max_objects: int = MAX_ATOMS,
) -> tuple[list[list[tuple[int, int, float]]], list[int]]:
    center_np = center_map_01.detach().cpu().numpy()
    atom_np = atom_map_01.detach().cpu().numpy() if atom_map_01 is not None else None
    if count_logits is not None:
        count_pred = count_logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
    else:
        count_pred = np.full(center_np.shape[0], -1, dtype=np.int64)

    all_peaks: list[list[tuple[int, int, float]]] = []
    all_counts: list[int] = []
    for b in range(center_np.shape[0]):
        target_k = None if int(count_pred[b]) < 0 else int(count_pred[b])
        proposal_map = _proposal_support_map(
            center_np[b, 0],
            None if atom_np is None else atom_np[b, 0],
            atom_support_weight=0.35,
        )
        peaks = _decode_single_center_map(
            proposal_map,
            target_k=target_k,
            peak_threshold=float(peak_threshold),
            min_distance_px=int(min_distance_px),
            max_objects=int(max_objects),
        )
        all_peaks.append(peaks)
        all_counts.append(len(peaks))
    return all_peaks, all_counts


def _pad_object_arrays(coords: np.ndarray, atom_types: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    padded_coords = np.zeros((1, MAX_ATOMS, 3), dtype=np.float32)
    padded_types = np.full((1, MAX_ATOMS), -1, dtype=np.int64)
    mask = np.zeros((1, MAX_ATOMS), dtype=np.float32)
    n = min(len(coords), MAX_ATOMS)
    if n > 0:
        padded_coords[0, :n] = coords[:n]
        padded_types[0, :n] = atom_types[:n]
        mask[0, :n] = 1.0
    return torch.from_numpy(padded_coords), torch.from_numpy(mask)


def _xy_from_pixels(x: int, y: int, img_size: int) -> tuple[float, float]:
    x_norm = (float(x) / float(max(img_size - 1, 1))) * 2.0 - 1.0
    y_norm = (float(y) / float(max(img_size - 1, 1))) * 2.0 - 1.0
    return x_norm, y_norm


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


def extract_predicted_objects(
    center_map_01: torch.Tensor,
    pred_01: torch.Tensor,
    shared_feat: torch.Tensor,
    afm_stack: torch.Tensor,
    type_head: torch.nn.Module,
    edge_head: torch.nn.Module,
    device: torch.device,
    img_size: int,
    count_logits: torch.Tensor | None = None,
    peak_threshold: float = 0.45,
    min_distance_px: int = 2,
    max_objects: int = MAX_ATOMS,
) -> dict:
    peaks_batch, count_batch = decode_object_peaks(
        center_map_01,
        count_logits=count_logits,
        atom_map_01=pred_01[:, 0:1],
        peak_threshold=peak_threshold,
        min_distance_px=min_distance_px,
        max_objects=max_objects,
    )
    peaks = peaks_batch[0]
    z_np = pred_01[0, 12].detach().cpu().numpy()
    center_np = center_map_01[0, 0].detach().cpu().numpy()

    coords = []
    refined_xy = []
    for y, x, _ in peaks:
        y_ref, x_ref = _refine_peak_xy(center_np, y, x, refine_radius_px=2)
        x_norm = (float(x_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
        y_norm = (float(y_ref) / float(max(img_size - 1, 1))) * 2.0 - 1.0
        z_norm = _bilinear_sample_np(z_np, y_ref, x_ref) * 2.0 - 1.0
        coords.append([x_norm, y_norm, z_norm])
        refined_xy.append([x_ref, y_ref])
    coords_np = np.asarray(coords, dtype=np.float32)

    coords_pad, mask_pad = _pad_object_arrays(coords_np, np.zeros(len(coords_np), dtype=np.int64))
    coords_pad = coords_pad.to(device)
    mask_pad = mask_pad.to(device)

    with torch.no_grad():
        fine_logits, _, _ = type_head.forward(
            coords_pad,
            shared_feat,
            afm_stack,
            mask_pad,
            center_map=center_map_01,
        )
        pred_types = fine_logits.argmax(dim=-1)
        edge_logits = edge_head.forward(coords_pad, shared_feat, afm_stack, mask_pad)

    return {
        "coords": coords_pad[0].detach().cpu().numpy(),
        "types": pred_types[0].detach().cpu().numpy().astype(np.int64),
        "mask": mask_pad[0].detach().cpu().numpy().astype(np.float32),
        "edge_adj": _adjacency_from_logits(edge_logits[0], mask_pad[0]),
        "peaks": peaks,
        "refined_xy_px": np.asarray(refined_xy, dtype=np.float32),
        "pred_count": int(count_batch[0]),
        "mean_center_score": float(np.mean([s for _, _, s in peaks])) if peaks else 0.0,
    }


def _infer_edge_set_from_adj(adj: np.ndarray, mask: np.ndarray) -> set[tuple[int, int]]:
    n = int((mask > 0.5).sum())
    out = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                out.add((i, j))
    return out


def _matched_edge_f1(
    pred_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
    pred_to_gt: dict[int, int],
) -> tuple[float, float, float]:
    mapped_pred = set()
    for i, j in pred_edges:
        if i not in pred_to_gt or j not in pred_to_gt:
            continue
        mapped_pred.add(tuple(sorted((pred_to_gt[i], pred_to_gt[j]))))

    tp = len(mapped_pred & gt_edges)
    fp = len(mapped_pred - gt_edges)
    fn = len(gt_edges - mapped_pred)
    return _safe_f1(tp, fp, fn)


def _matched_edge_f1_with_gate(
    pred_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
    pred_to_gt: dict[int, int],
    accepted_pred_nodes: set[int],
) -> tuple[float, float, float]:
    mapped_pred = set()
    for i, j in pred_edges:
        if i not in accepted_pred_nodes or j not in accepted_pred_nodes:
            continue
        if i not in pred_to_gt or j not in pred_to_gt:
            continue
        mapped_pred.add(tuple(sorted((pred_to_gt[i], pred_to_gt[j]))))

    accepted_gt_nodes = {pred_to_gt[i] for i in accepted_pred_nodes if i in pred_to_gt}
    gt_edges_gated = {
        edge for edge in gt_edges
        if edge[0] in accepted_gt_nodes and edge[1] in accepted_gt_nodes
    }

    tp = len(mapped_pred & gt_edges_gated)
    fp = len(mapped_pred - gt_edges_gated)
    fn = len(gt_edges_gated - mapped_pred)
    return _safe_f1(tp, fp, fn)


def compute_pred_object_metrics(
    pred_obj: dict,
    gt_coords: np.ndarray,
    gt_types: np.ndarray,
    gt_mask: np.ndarray,
    gt_edge_adj: np.ndarray,
    img_size: int = 128,
    edge_match_radius_px: float = 3.0,
) -> dict[str, float]:
    pred_coords = pred_obj["coords"]
    pred_types = pred_obj["types"]
    pred_mask = pred_obj["mask"]
    n_pred = int((pred_mask > 0.5).sum())
    n_gt = int((gt_mask > 0.5).sum())

    count_mae = float(abs(n_pred - n_gt))
    count_score = _count_similarity(n_pred, n_gt)
    center_score = float(pred_obj.get("mean_center_score", 0.0))

    if n_pred == 0 or n_gt == 0:
        return {
            "pred_object_count_mae": count_mae,
            "pred_object_count_score": count_score,
            "pred_object_center_score": center_score,
            "pred_object_type_acc": 0.0,
            "pred_object_macro_f1": 0.0,
            "pred_object_hetero_f1": 0.0,
            "pred_object_edge_f1": 0.0,
            "pred_object_edge_f1_robust": 0.0,
            "pred_object_match_coverage_robust": 0.0,
            "pred_object_graph_score": 0.0,
            "pred_object_heavy_rmsd": 9.999,
            "pred_object_z_mae": 12.0,
            "pred_object_score": 0.0,
            "pred_object_3d_score": 0.0,
        }

    pc = pred_coords[:n_pred].astype(np.float32)
    pt = pred_types[:n_pred].astype(np.int64)
    gc = gt_coords[:n_gt].astype(np.float32)
    gt = gt_types[:n_gt].astype(np.int64)

    row_ind, col_ind, cost = _hungarian_match_numpy(pc, gc)
    if len(row_ind) == 0:
        return {
            "pred_object_count_mae": count_mae,
            "pred_object_count_score": count_score,
            "pred_object_center_score": center_score,
            "pred_object_type_acc": 0.0,
            "pred_object_macro_f1": 0.0,
            "pred_object_hetero_f1": 0.0,
            "pred_object_edge_f1": 0.0,
            "pred_object_edge_f1_robust": 0.0,
            "pred_object_match_coverage_robust": 0.0,
            "pred_object_graph_score": 0.0,
            "pred_object_heavy_rmsd": 9.999,
            "pred_object_z_mae": 12.0,
            "pred_object_score": 0.0,
            "pred_object_3d_score": 0.0,
        }

    pred_match_types = pt[row_ind]
    gt_match_types = gt[col_ind]
    type_acc = float((pred_match_types == gt_match_types).mean())
    macro_f1 = float(_macro_type_f1(pred_match_types, gt_match_types, pt, gt))

    pred_het = ~np.isin(pt, [0, 1])
    gt_het = ~np.isin(gt, [0, 1])
    tp_het = int((((pred_match_types != 0) & (pred_match_types != 1)) & ((gt_match_types != 0) & (gt_match_types != 1))).sum())
    fp_het = int(pred_het.sum() - tp_het)
    fn_het = int(gt_het.sum() - tp_het)
    _, _, hetero_f1 = _safe_f1(tp_het, fp_het, fn_het)

    heavy_match = gt_match_types != 0
    if heavy_match.any():
        heavy_rmsd = float(np.sqrt(np.mean(cost[row_ind, col_ind][heavy_match] ** 2)))
    else:
        heavy_rmsd = float(np.sqrt(np.mean(cost[row_ind, col_ind] ** 2)))

    z_mae = float(np.mean(np.abs(pc[row_ind, 2] - gc[col_ind, 2])) * 12.0)

    pred_edges = _infer_edge_set_from_adj(pred_obj["edge_adj"], pred_mask)
    gt_edges = _infer_edge_set_from_adj(gt_edge_adj, gt_mask)
    pred_to_gt = {int(p): int(g) for p, g in zip(row_ind.tolist(), col_ind.tolist())}
    _, _, edge_f1 = _matched_edge_f1(pred_edges, gt_edges, pred_to_gt)

    edge_match_radius_norm = 2.0 * float(edge_match_radius_px) / float(max(img_size - 1, 1))
    accepted_pred_nodes = {
        int(p)
        for p, g in zip(row_ind.tolist(), col_ind.tolist())
        if float(cost[p, g]) <= edge_match_radius_norm
    }
    robust_match_coverage = float(len(accepted_pred_nodes) / max(n_gt, 1))
    _, _, edge_f1_robust = _matched_edge_f1_with_gate(
        pred_edges,
        gt_edges,
        pred_to_gt,
        accepted_pred_nodes,
    )

    graph_score = (
        0.35 * edge_f1
        + 0.25 * count_score
        + 0.20 * macro_f1
        + 0.20 * hetero_f1
    )
    coord_score = float(np.clip(1.0 - heavy_rmsd / 0.35, 0.0, 1.0))
    z_score = float(np.clip(1.0 - z_mae / 1.0, 0.0, 1.0))
    pred_object_score = (
        0.25 * type_acc
        + 0.20 * macro_f1
        + 0.15 * hetero_f1
        + 0.20 * edge_f1
        + 0.15 * count_score
        + 0.05 * center_score
    )
    pred_object_3d_score = (
        0.35 * coord_score
        + 0.20 * z_score
        + 0.15 * edge_f1
        + 0.15 * type_acc
        + 0.10 * count_score
        + 0.05 * center_score
    )

    return {
        "pred_object_count_mae": count_mae,
        "pred_object_count_score": float(count_score),
        "pred_object_center_score": center_score,
        "pred_object_type_acc": float(type_acc),
        "pred_object_macro_f1": float(macro_f1),
        "pred_object_hetero_f1": float(hetero_f1),
        "pred_object_edge_f1": float(edge_f1),
        "pred_object_edge_f1_robust": float(edge_f1_robust),
        "pred_object_match_coverage_robust": float(robust_match_coverage),
        "pred_object_graph_score": float(graph_score),
        "pred_object_heavy_rmsd": float(heavy_rmsd),
        "pred_object_z_mae": float(z_mae),
        "pred_object_score": float(pred_object_score),
        "pred_object_3d_score": float(pred_object_3d_score),
    }


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
        }

    pred_np = pred[valid].detach().cpu().numpy().astype(np.int64)
    gt_np = gt[valid].detach().cpu().numpy().astype(np.int64)
    type_acc = float((pred_np == gt_np).mean())
    macro_f1 = _macro_f1_from_lists(gt_np.tolist(), pred_np.tolist(), n_classes=10)

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
    }


def _edge_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits) > 0.5)
    valid_pair = (mask.unsqueeze(2) > 0) & (mask.unsqueeze(1) > 0)
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid_pair & (~eye)
    upper = torch.triu(torch.ones_like(valid_pair, dtype=torch.bool), diagonal=1)
    valid_pair = valid_pair & upper

    if valid_pair.sum() == 0:
        return {
            "center_edge_precision_gtcoord": 0.0,
            "center_edge_recall_gtcoord": 0.0,
            "center_edge_f1_gtcoord": 0.0,
        }

    pred_np = pred[valid_pair].detach().cpu().numpy().astype(np.int64)
    gt_np = labels[valid_pair].detach().cpu().numpy().astype(np.int64)
    tp = int(((pred_np == 1) & (gt_np == 1)).sum())
    fp = int(((pred_np == 1) & (gt_np == 0)).sum())
    fn = int(((pred_np == 0) & (gt_np == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec)
    return {
        "center_edge_precision_gtcoord": float(prec),
        "center_edge_recall_gtcoord": float(rec),
        "center_edge_f1_gtcoord": float(f1),
    }


def _edge_metrics_named(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict[str, float]:
    base = _edge_metrics(logits, labels, mask)
    return {
        f"{prefix}_edge_precision": base["center_edge_precision_gtcoord"],
        f"{prefix}_edge_recall": base["center_edge_recall_gtcoord"],
        f"{prefix}_edge_f1": base["center_edge_f1_gtcoord"],
    }


def _type_metrics_named(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict[str, float]:
    base = _discrete_type_metrics(pred, gt, mask)
    return {
        f"{prefix}_type_acc": base["center_type_acc_gtcoord"],
        f"{prefix}_macro_f1": base["center_macro_f1_gtcoord"],
        f"{prefix}_hetero_f1": base["center_hetero_f1_gtcoord"],
    }


def _object_score(type_acc: float, macro_f1: float, hetero_f1: float, edge_f1: float, center_score: float, z_mae: float, shift_px: float) -> float:
    z_score = float(np.clip(1.0 - z_mae / 0.5, 0.0, 1.0))
    shift_score = float(np.clip(1.0 - shift_px / 5.0, 0.0, 1.0))
    return (
        0.22 * float(type_acc)
        + 0.22 * float(macro_f1)
        + 0.12 * float(hetero_f1)
        + 0.20 * float(edge_f1)
        + 0.12 * float(center_score)
        + 0.07 * z_score
        + 0.05 * shift_score
    )


def maybe_load_warm_start(
    checkpoint_path: str | None,
    model: torch.nn.Module,
    type_head: torch.nn.Module,
    edge_head: torch.nn.Module,
    device: torch.device,
):
    if not checkpoint_path:
        return None
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"warm_start checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        model.load_state_dict(state["model"], strict=False)
    if "type_head" in state:
        type_head.load_state_dict(state["type_head"], strict=False)
    if "edge_head" in state:
        edge_head.load_state_dict(state["edge_head"], strict=False)
    return state.get("epoch")


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
    return encoder, classifier


def compute_object_metrics(pred_01: torch.Tensor, center_map_01: torch.Tensor, batch: dict, img_size: int) -> dict[str, float]:
    pred_np = pred_01.detach().cpu().numpy()
    center_np = center_map_01.detach().cpu().numpy()
    coords = batch["coords"].cpu().numpy()
    atom_types = batch["atom_types"].cpu().numpy()
    mask = batch["atom_mask"].cpu().numpy()
    gt_targets = batch_render_v19_joint_targets(coords, atom_types, mask, img_size=img_size)

    atom_mae_list = []
    z_mae_list = []
    center_score_list = []
    typed_center_score_list = []
    gt_type_all = []
    pred_type_all = []
    atom_z_err_list = []

    for b in range(pred_np.shape[0]):
        pred = pred_np[b]
        real = gt_targets[b]
        atom_mae_list.append(float(np.abs(pred[0] - real[0]).mean()))

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
            x0 = max(0, x - 3)
            x1 = min(img_size, x + 4)
            y0 = max(0, y - 3)
            y1 = min(img_size, y + 4)
            occ_patch = center_np[b, 0, y0:y1, x0:x1]
            type_patch = pred[2:12, y0:y1, x0:x1]
            z_patch = pred[12, y0:y1, x0:x1]
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

    return {
        "atom_xy_mae": float(np.mean(atom_mae_list)) if atom_mae_list else 0.0,
        "atom_center_score_r3": float(np.mean(center_score_list)) if center_score_list else 0.0,
        "typed_center_score_r3": float(np.mean(typed_center_score_list)) if typed_center_score_list else 0.0,
        "atom_type_macro_f1_2d": _macro_f1_from_lists(gt_type_all, pred_type_all, n_classes=10),
        "z_map_mae": float(np.mean(z_mae_list)) if z_mae_list else 0.0,
        "atom_z_mae_r3": float(np.mean(atom_z_err_list)) if atom_z_err_list else 0.0,
    }


def save_preview(model, val_loader, config, device, save_path: Path):
    model.eval()
    batch = next(iter(val_loader))
    afm = batch["afm_stack"].to(device)
    real = build_targets(batch, config["img_size"], device)
    with torch.no_grad():
        pred, features = model.forward_with_features(afm)
        center_map = torch.sigmoid(features["center_logits"])
    pred = ((pred + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy()
    center_map = center_map.clamp(0.0, 1.0).cpu().numpy()
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
        pred_struct = pred[i, :12].copy()
        pred_struct[0] = center_map[i, 0]
        axes[i, 2].imshow(np.transpose(structure_map_to_rgb(pred_struct), (1, 2, 0)))
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


def evaluate(model, type_head, edge_head, val_loader, config, device):
    model.eval()
    type_head.eval()
    edge_head.eval()
    agg = {}
    n_batches = 0
    disable_z_for_object_heads = bool(config.get("disable_z_for_object_heads", False))
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", leave=False):
            afm = batch["afm_stack"].to(device)
            pred, features = model.forward_with_features(afm)
            pred_01 = (pred + 1.0) * 0.5
            pred_01_heads = maybe_neutralize_z_map(pred_01, disable_z_for_object_heads)
            center_map_01 = torch.sigmoid(features["center_logits"])
            metrics = compute_object_metrics(pred_01_heads, center_map_01, batch, config["img_size"])

            coords = batch["coords"].to(device)
            coords_obj = maybe_strip_z_coords(coords, disable_z_for_object_heads)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            edge_labels = build_edge_labels(batch, device)

            _, type_logits = type_head.compute_loss(
                coords_obj,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=None,
                center_map=center_map_01,
            )
            type_pred = type_logits.argmax(dim=-1)
            metrics.update(_type_metrics_named(type_pred, atom_types, mask, prefix="gt_center"))

            _, edge_logits = edge_head.compute_loss(coords_obj, features["enc1"], afm, mask, edge_labels)
            metrics.update(_edge_metrics_named(edge_logits, edge_labels, mask, prefix="gt_center"))

            peak_coords, peak_shift = build_peak_center_coords(
                center_map_01,
                pred_01_heads[:, 12:13],
                coords_obj,
                mask,
                config["img_size"],
                alpha=1.0,
                search_radius=int(config.get("center_search_radius", 3)),
            )
            _, peak_type_logits = type_head.compute_loss(
                peak_coords,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=None,
                center_map=center_map_01,
            )
            peak_type_pred = peak_type_logits.argmax(dim=-1)
            metrics.update(_type_metrics_named(peak_type_pred, atom_types, mask, prefix="peak_center"))

            _, peak_edge_logits = edge_head.compute_loss(peak_coords, features["enc1"], afm, mask, edge_labels)
            metrics.update(_edge_metrics_named(peak_edge_logits, edge_labels, mask, prefix="peak_center"))
            metrics["peak_center_shift_px"] = peak_shift
            metrics["gt_object_score"] = _object_score(
                metrics["gt_center_type_acc"],
                metrics["gt_center_macro_f1"],
                metrics["gt_center_hetero_f1"],
                metrics["gt_center_edge_f1"],
                metrics["atom_center_score_r3"],
                metrics["atom_z_mae_r3"],
                0.0,
            )
            metrics["peak_object_score"] = _object_score(
                metrics["peak_center_type_acc"],
                metrics["peak_center_macro_f1"],
                metrics["peak_center_hetero_f1"],
                metrics["peak_center_edge_f1"],
                metrics["atom_center_score_r3"],
                metrics["atom_z_mae_r3"],
                metrics["peak_center_shift_px"],
            )

            pred_object_metrics = []
            for bi in range(afm.shape[0]):
                pred_obj = extract_predicted_objects(
                    center_map_01[bi : bi + 1],
                    pred_01_heads[bi : bi + 1],
                    features["enc1"][bi : bi + 1],
                    afm[bi : bi + 1],
                    type_head,
                    edge_head,
                    device,
                    img_size=int(config["img_size"]),
                    count_logits=features.get("count_logits", None)[bi : bi + 1] if features.get("count_logits", None) is not None else None,
                    peak_threshold=float(config.get("proposal_peak_threshold", 0.45)),
                    min_distance_px=int(config.get("proposal_min_distance_px", 2)),
                    max_objects=int(config.get("proposal_max_objects", MAX_ATOMS)),
                )
                gt_coords_np = batch["coords"][bi].detach().cpu().numpy()
                gt_types_np = batch["atom_types"][bi].detach().cpu().numpy()
                gt_mask_np = batch["atom_mask"][bi].detach().cpu().numpy()
                gt_edge_adj = edge_labels[bi].detach().cpu().numpy().astype(np.int32)
                pred_object_metrics.append(
                    compute_pred_object_metrics(
                        pred_obj,
                        gt_coords_np,
                        gt_types_np,
                        gt_mask_np,
                        gt_edge_adj,
                        img_size=int(config["img_size"]),
                        edge_match_radius_px=float(config.get("pred_object_edge_match_radius_px", 3.0)),
                    )
                )
            if pred_object_metrics:
                for key in pred_object_metrics[0].keys():
                    metrics[key] = float(np.mean([m[key] for m in pred_object_metrics]))

            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in agg.items()}


def train(config: dict):
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
    type_head_variant = str(config.get("type_head_variant", "center"))
    if type_head_variant == "legacy_gnn":
        type_head = LegacyGNNTypeHeadAdapter(
            shared_feat_dim=config.get("base_ch", 64),
            hidden_dim=config.get("type_hidden_dim", 192),
            num_types=10,
            num_gnn_layers=config.get("legacy_type_num_gnn_layers", 4),
            num_heads=config.get("legacy_type_num_heads", 4),
            bond_threshold=config.get("legacy_type_bond_threshold", 0.20),
            token_grid_size=config.get("legacy_type_token_grid_size", 16),
            label_smoothing=config.get("type_label_smoothing", 0.0),
        ).to(device)
    else:
        type_head = CenterConditionedTypeHead(
            shared_feat_dim=config.get("base_ch", 64),
            hidden_dim=config.get("type_hidden_dim", 192),
            coarse_lambda=config.get("type_coarse_lambda", 0.35),
            hetero_lambda=config.get("type_hetero_lambda", 0.25),
            focal_gamma=config.get("type_focal_gamma", 1.5),
            label_smoothing=config.get("type_label_smoothing", 0.02),
            afm_radius_px=config.get("type_afm_radius_px", 2.0),
            feat_radius_px=config.get("type_feat_radius_px", 1.0),
            center_radius_px=config.get("type_center_radius_px", 2.0),
        ).to(device)

    edge_head_variant = str(config.get("edge_head_variant", "center"))
    if edge_head_variant == "zero":
        edge_head = ZeroEdgeHead().to(device)
    else:
        edge_head = CenterConditionedEdgeHead(shared_feat_dim=config.get("base_ch", 64)).to(device)

    teacher_encoder = None
    teacher_classifier = None
    teacher_ckpt = config.get("teacher_type_checkpoint", "")
    lambda_teacher = float(config.get("lambda_teacher_type_distill", 0.0))
    teacher_temp = float(config.get("teacher_temperature", 1.5))
    if teacher_ckpt:
        teacher_encoder, teacher_classifier = build_type_teacher(teacher_ckpt, device)

    params = list(model.parameters()) + list(type_head.parameters()) + list(edge_head.parameters())
    optimizer = optim.AdamW(params, lr=config.get("lr", 2e-4), weight_decay=config.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config.get("epochs", 1)), 1),
        eta_min=config.get("min_lr", 1e-5),
    )

    type_class_weights = TYPE_CLASS_WEIGHTS.to(device)
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_key = None
    epochs = int(config.get("epochs", 10))
    start_epoch = 1

    resume_ckpt = config.get("resume_from_checkpoint", "")
    if resume_ckpt:
        resume_path = Path(resume_ckpt)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        type_head.load_state_dict(state["type_head"], strict=False)
        edge_head.load_state_dict(state["edge_head"], strict=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        history = state.get("history", history)
        loaded_best = state.get("best_key")
        best_key = tuple(loaded_best) if loaded_best is not None else None
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch - 1}: {resume_path}")
    else:
        warm_start_epoch = maybe_load_warm_start(config.get("warm_start_checkpoint"), model, type_head, edge_head, device)
        if warm_start_epoch is not None:
            print(f"Warm-started from epoch {warm_start_epoch}: {config.get('warm_start_checkpoint')}")

    lambda_center = float(config.get("lambda_center", 20.0))
    lambda_atom_aux_start = float(config.get("lambda_atom_aux_start", config.get("lambda_atom_aux", 5.0)))
    lambda_atom_aux_final = float(config.get("lambda_atom_aux_final", 1.0))
    lambda_z_start = float(config.get("lambda_z_start", config.get("lambda_z", 8.0)))
    lambda_z_final = float(config.get("lambda_z_final", lambda_z_start))
    lambda_type_obj_gt = float(config.get("lambda_type_obj_gt", 1.5))
    lambda_type_obj_peak_start = float(config.get("lambda_type_obj_peak_start", 0.2))
    lambda_type_obj_peak_final = float(config.get("lambda_type_obj_peak_final", 2.5))
    lambda_type_obj_pred_start = float(config.get("lambda_type_obj_pred_start", 0.1))
    lambda_type_obj_pred_final = float(config.get("lambda_type_obj_pred_final", 2.0))
    lambda_edge_obj_gt = float(config.get("lambda_edge_obj_gt", 1.5))
    lambda_edge_obj_peak_start = float(config.get("lambda_edge_obj_peak_start", 0.2))
    lambda_edge_obj_peak_final = float(config.get("lambda_edge_obj_peak_final", 2.5))
    lambda_edge_obj_pred_start = float(config.get("lambda_edge_obj_pred_start", 0.0))
    lambda_edge_obj_pred_final = float(config.get("lambda_edge_obj_pred_final", 0.0))
    lambda_type_map_aux_start = float(config.get("lambda_type_map_aux_start", config.get("lambda_type_map_aux", 1.0)))
    lambda_type_map_aux_final = float(config.get("lambda_type_map_aux_final", 0.2))
    lambda_bond_map_aux_start = float(config.get("lambda_bond_map_aux_start", config.get("lambda_bond_map_aux", 1.0)))
    lambda_bond_map_aux_final = float(config.get("lambda_bond_map_aux_final", 0.2))
    lambda_peak_consistency_start = float(config.get("lambda_peak_consistency_start", 0.0))
    lambda_peak_consistency_final = float(config.get("lambda_peak_consistency_final", 0.5))
    lambda_pred_type_consistency_start = float(config.get("lambda_pred_type_consistency_start", 0.0))
    lambda_pred_type_consistency_final = float(config.get("lambda_pred_type_consistency_final", 0.5))
    lambda_object_count = float(config.get("lambda_object_count", 1.0))
    lambda_object_count_mae = float(config.get("lambda_object_count_mae", 0.15))
    lambda_teacher_type_pred = float(config.get("lambda_teacher_type_pred_distill", 0.0))
    pred_train_match_radius_px = float(config.get("pred_train_match_radius_px", 4.0))
    consistency_temperature = float(config.get("consistency_temperature", 1.5))
    aux_decay_epochs = int(config.get("aux_decay_epochs", epochs))
    loss_warmup_epochs = int(config.get("loss_warmup_epochs", epochs))
    center_alpha_final = float(config.get("center_curriculum_alpha_final", 1.0))
    center_alpha_start = float(config.get("center_curriculum_alpha_start", 0.0))
    center_alpha_warmup_epochs = int(config.get("center_curriculum_warmup_epochs", epochs))
    center_search_radius = int(config.get("center_search_radius", 3))
    disable_z_for_object_heads = bool(config.get("disable_z_for_object_heads", False))

    if start_epoch > epochs:
        print(f"resume checkpoint already reached target epochs ({epochs}); nothing to do.")
        return

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        type_head.train()
        edge_head.train()
        losses = []
        current_center_alpha = scheduled_weight(
            epoch,
            final_weight=center_alpha_final,
            start_weight=center_alpha_start,
            warmup_epochs=center_alpha_warmup_epochs,
        )

        pbar = tqdm(train_loader, desc=f"V19-object [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, config["img_size"], device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = (pred + 1.0) * 0.5
            pred_01_heads = maybe_neutralize_z_map(pred_01, disable_z_for_object_heads)
            center_logits = features["center_logits"]
            center_map_01 = torch.sigmoid(center_logits)
            count_logits = features["count_logits"]

            target_center = targets[:, 0:1]
            count_targets = mask.sum(dim=1).long().clamp(min=0, max=MAX_ATOMS)
            pos = target_center.sum().item()
            neg = target_center.numel() - pos
            center_pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0), device=device)
            center_bce = F.binary_cross_entropy_with_logits(center_logits, target_center, pos_weight=center_pos_weight)
            center_l1 = F.l1_loss(center_map_01, target_center)
            center_loss = (0.75 * center_bce + 0.25 * center_l1) * lambda_center
            count_ce = F.cross_entropy(count_logits, count_targets)
            count_pred = count_logits.argmax(dim=-1)
            count_mae = (count_pred.float() - count_targets.float()).abs().mean()
            count_loss = lambda_object_count * count_ce + lambda_object_count_mae * count_mae

            current_atom_aux = scheduled_weight(epoch, lambda_atom_aux_final, lambda_atom_aux_start, aux_decay_epochs)
            current_z = 0.0 if disable_z_for_object_heads else scheduled_weight(epoch, lambda_z_final, lambda_z_start, loss_warmup_epochs)
            current_type_map_aux = scheduled_weight(epoch, lambda_type_map_aux_final, lambda_type_map_aux_start, aux_decay_epochs)
            current_bond_map_aux = scheduled_weight(epoch, lambda_bond_map_aux_final, lambda_bond_map_aux_start, aux_decay_epochs)
            current_type_peak = scheduled_weight(epoch, lambda_type_obj_peak_final, lambda_type_obj_peak_start, loss_warmup_epochs)
            current_type_pred = scheduled_weight(epoch, lambda_type_obj_pred_final, lambda_type_obj_pred_start, loss_warmup_epochs)
            current_edge_peak = scheduled_weight(epoch, lambda_edge_obj_peak_final, lambda_edge_obj_peak_start, loss_warmup_epochs)
            current_edge_pred = scheduled_weight(epoch, lambda_edge_obj_pred_final, lambda_edge_obj_pred_start, loss_warmup_epochs)
            current_peak_consistency = scheduled_weight(epoch, lambda_peak_consistency_final, lambda_peak_consistency_start, loss_warmup_epochs)
            current_pred_type_consistency = scheduled_weight(
                epoch,
                lambda_pred_type_consistency_final,
                lambda_pred_type_consistency_start,
                loss_warmup_epochs,
            )

            atom_map_aux = F.l1_loss(pred_01[:, 0:1], target_center) * current_atom_aux
            z_mask = targets[:, 0:1]
            z_loss = ((pred_01[:, 12:13] - targets[:, 12:13]).abs() * z_mask).sum() / z_mask.sum().clamp(min=1.0)
            z_loss = z_loss * current_z
            coords_obj = maybe_strip_z_coords(coords, disable_z_for_object_heads)

            type_map_aux = F.l1_loss(pred_01[:, 2:12], targets[:, 2:12]) * current_type_map_aux
            bond_map_aux = F.l1_loss(pred_01[:, 1:2], targets[:, 1:2]) * current_bond_map_aux

            gt_type_obj_loss, gt_center_logits = type_head.compute_loss(
                coords_obj,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=type_class_weights,
                center_map=center_map_01,
            )
            gt_type_obj_loss = gt_type_obj_loss * lambda_type_obj_gt

            gt_edge_obj_loss, gt_edge_logits = edge_head.compute_loss(coords_obj, features["enc1"], afm, mask, edge_labels)
            gt_edge_obj_loss = gt_edge_obj_loss * lambda_edge_obj_gt

            object_coords, center_shift_px = build_peak_center_coords(
                center_map_01,
                pred_01_heads[:, 12:13],
                coords_obj,
                mask,
                config["img_size"],
                alpha=current_center_alpha,
                search_radius=center_search_radius,
            )

            peak_type_obj_loss, peak_center_logits = type_head.compute_loss(
                object_coords,
                features["enc1"],
                afm,
                atom_types,
                mask,
                class_weight=type_class_weights,
                center_map=center_map_01,
            )
            peak_type_obj_loss = peak_type_obj_loss * current_type_peak

            peak_edge_obj_loss, peak_edge_logits = edge_head.compute_loss(object_coords, features["enc1"], afm, mask, edge_labels)
            peak_edge_obj_loss = peak_edge_obj_loss * current_edge_peak

            pred_train_coords, pred_train_types, pred_train_mask, pred_train_gt_index = build_predicted_type_training_batch(
                center_map_01,
                pred_01_heads,
                count_logits,
                coords_obj,
                atom_types,
                mask,
                config["img_size"],
                max_objects=MAX_ATOMS,
                match_radius_px=pred_train_match_radius_px,
            )
            pred_type_obj_loss, pred_center_logits = type_head.compute_loss(
                pred_train_coords,
                features["enc1"],
                afm,
                pred_train_types,
                pred_train_mask,
                class_weight=type_class_weights,
                center_map=center_map_01,
            )
            pred_type_obj_loss = pred_type_obj_loss * current_type_pred

            pred_edge_labels = build_predicted_edge_training_labels(
                pred_train_gt_index,
                pred_train_mask,
                edge_labels,
            )
            pred_edge_obj_loss, _ = edge_head.compute_loss(
                pred_train_coords,
                features["enc1"],
                afm,
                pred_train_mask,
                pred_edge_labels,
            )
            pred_edge_obj_loss = pred_edge_obj_loss * current_edge_pred

            if teacher_encoder is not None and teacher_classifier is not None and lambda_teacher > 0:
                with torch.no_grad():
                    _, teacher_patches = teacher_encoder(afm)
                    teacher_logits = teacher_classifier(coords_obj, teacher_patches, mask, afm_stack=afm)
                valid = (mask > 0) & (atom_types >= 0)
                if valid.sum() > 0:
                    student = gt_center_logits[valid] / teacher_temp
                    teacher = teacher_logits[valid] / teacher_temp
                    teacher_loss = (
                        F.kl_div(
                            F.log_softmax(student, dim=-1),
                            F.softmax(teacher, dim=-1),
                            reduction="batchmean",
                        )
                        * (teacher_temp ** 2)
                        * lambda_teacher
                    )
                else:
                    teacher_loss = torch.tensor(0.0, device=device)
            else:
                teacher_loss = torch.tensor(0.0, device=device)

            if teacher_encoder is not None and teacher_classifier is not None and lambda_teacher_type_pred > 0:
                with torch.no_grad():
                    _, teacher_patches = teacher_encoder(afm)
                    teacher_pred_logits = teacher_classifier(
                        pred_train_coords,
                        teacher_patches,
                        pred_train_mask,
                        afm_stack=afm,
                    )
                pred_valid = (pred_train_mask > 0) & (pred_train_types >= 0)
                if pred_valid.sum() > 0:
                    student = pred_center_logits[pred_valid] / teacher_temp
                    teacher = teacher_pred_logits[pred_valid] / teacher_temp
                    pred_teacher_loss = (
                        F.kl_div(
                            F.log_softmax(student, dim=-1),
                            F.softmax(teacher, dim=-1),
                            reduction="batchmean",
                        )
                        * (teacher_temp ** 2)
                        * lambda_teacher_type_pred
                    )
                else:
                    pred_teacher_loss = torch.tensor(0.0, device=device)
            else:
                pred_teacher_loss = torch.tensor(0.0, device=device)

            valid = (mask > 0) & (atom_types >= 0)
            if valid.sum() > 0 and current_peak_consistency > 0:
                peak_student = peak_center_logits[valid] / consistency_temperature
                gt_teacher = gt_center_logits[valid].detach() / consistency_temperature
                peak_consistency_loss = (
                    F.kl_div(
                        F.log_softmax(peak_student, dim=-1),
                        F.softmax(gt_teacher, dim=-1),
                        reduction="batchmean",
                    )
                    * (consistency_temperature ** 2)
                    * current_peak_consistency
                )
            else:
                peak_consistency_loss = torch.tensor(0.0, device=device)

            pred_valid = (pred_train_mask > 0) & (pred_train_gt_index >= 0)
            if pred_valid.sum() > 0 and current_pred_type_consistency > 0:
                gather_index = pred_train_gt_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, gt_center_logits.shape[-1])
                pred_teacher_logits = torch.gather(gt_center_logits.detach(), 1, gather_index)
                pred_student = pred_center_logits[pred_valid] / consistency_temperature
                pred_teacher = pred_teacher_logits[pred_valid] / consistency_temperature
                pred_consistency_loss = (
                    F.kl_div(
                        F.log_softmax(pred_student, dim=-1),
                        F.softmax(pred_teacher, dim=-1),
                        reduction="batchmean",
                    )
                    * (consistency_temperature ** 2)
                    * current_pred_type_consistency
                )
            else:
                pred_consistency_loss = torch.tensor(0.0, device=device)

            loss = (
                center_loss
                + count_loss
                + atom_map_aux
                + z_loss
                + type_map_aux
                + bond_map_aux
                + gt_type_obj_loss
                + gt_edge_obj_loss
                + peak_type_obj_loss
                + peak_edge_obj_loss
                + pred_type_obj_loss
                + pred_edge_obj_loss
                + teacher_loss
                + pred_teacher_loss
                + peak_consistency_loss
                + pred_consistency_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(
                loss=f"{np.mean(losses[-10:]):.3f}",
                alpha=f"{current_center_alpha:.2f}",
                tpk=f"{current_type_peak:.2f}",
                tpr=f"{current_type_pred:.2f}",
                epk=f"{current_edge_peak:.2f}",
                epr=f"{current_edge_pred:.2f}",
                cnt=f"{count_mae.item():.2f}",
                shift=f"{center_shift_px:.2f}",
            )

        scheduler.step()
        train_metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "center_alpha": float(current_center_alpha),
            "lambda_type_peak": float(current_type_peak),
            "lambda_type_pred": float(current_type_pred),
            "lambda_edge_peak": float(current_edge_peak),
            "lambda_edge_pred": float(current_edge_pred),
            "lambda_peak_consistency": float(current_peak_consistency),
            "lambda_pred_type_consistency": float(current_pred_type_consistency),
            "count_mae_head": float(count_mae.item()),
        }
        val_metrics = evaluate(model, type_head, edge_head, val_loader, config, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss {train_metrics['loss']:.4f} | alpha {train_metrics['center_alpha']:.2f} | "
            f"peak_score {val_metrics['peak_object_score']:.4f} | gt_score {val_metrics['gt_object_score']:.4f} | "
            f"pred_score {val_metrics.get('pred_object_score', 0.0):.4f} | "
            f"atom_center_r3 {val_metrics['atom_center_score_r3']:.4f} | "
            f"typed_center_r3 {val_metrics['typed_center_score_r3']:.4f} | "
            f"type_acc_peak {val_metrics['peak_center_type_acc']:.4f} | "
            f"type_acc_pred {val_metrics.get('pred_object_type_acc', 0.0):.4f} | "
            f"macro_f1_peak {val_metrics['peak_center_macro_f1']:.4f} | "
            f"macro_f1_pred {val_metrics.get('pred_object_macro_f1', 0.0):.4f} | "
            f"hetero_f1_peak {val_metrics['peak_center_hetero_f1']:.4f} | "
            f"edge_f1_peak {val_metrics['peak_center_edge_f1']:.4f} | "
            f"edge_f1_pred {val_metrics.get('pred_object_edge_f1', 0.0):.4f} | "
            f"count_mae_pred {val_metrics.get('pred_object_count_mae', 0.0):.4f} | "
            f"shift_px {val_metrics['peak_center_shift_px']:.4f} | "
            f"z_mae_r3 {val_metrics['atom_z_mae_r3']:.4f}"
        )

        current_key = (
            float(val_metrics.get("pred_object_score", 0.0)),
            float(val_metrics.get("pred_object_3d_score", 0.0)),
            -float(val_metrics.get("pred_object_count_mae", 999.0)),
            float(val_metrics.get("pred_object_macro_f1", 0.0)),
            float(val_metrics.get("pred_object_edge_f1", 0.0)),
            float(val_metrics["peak_object_score"]),
            float(val_metrics["peak_center_edge_f1"]),
            float(val_metrics["peak_center_macro_f1"]),
            float(val_metrics["peak_center_type_acc"]),
            float(val_metrics["peak_center_hetero_f1"]),
            -float(val_metrics["peak_center_shift_px"]),
            -float(val_metrics["atom_z_mae_r3"]),
            float(val_metrics["gt_object_score"]),
        )
        latest_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "type_head": type_head.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key) if best_key is not None else None,
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(latest_state, save_dir / "latest_v19_object_joint.pt")

        if best_key is None or current_key > best_key:
            best_key = current_key
            torch.save(latest_state, save_dir / "best_v19_object_joint.pt")
            save_preview(model, val_loader, config, device, save_dir / "best_preview.png")

        with open(save_dir / "history_v19_object_joint.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "type_head": type_head.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key) if best_key is not None else None,
            "config": config,
        },
        save_dir / "last_v19_object_joint.pt",
    )
    print(f"Saved checkpoints to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.resume_checkpoint:
        config["resume_from_checkpoint"] = args.resume_checkpoint

    base_dir = Path(__file__).resolve().parents[1]
    if config.get("data_root") == "auto":
        config["data_root"] = str(base_dir / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = str(base_dir / "experiments" / "v19_object_joint_medium" / "checkpoints")

    train(config)
