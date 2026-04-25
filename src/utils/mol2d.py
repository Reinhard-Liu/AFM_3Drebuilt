"""
V19 Stage 1 utilities:
- infer simple 2D bonds from projected coordinates
- render structured 2D molecular targets from 3D coords + atom types
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


V19_2D_TARGET_CHANNELS = 12
V19_JOINT_TARGET_CHANNELS = 13
TYPE_COLORS = np.array([
    [1.00, 1.00, 1.00],  # H
    [0.20, 0.20, 0.20],  # C
    [0.19, 0.31, 0.97],  # N
    [1.00, 0.05, 0.05],  # O
    [0.56, 0.88, 0.31],  # F
    [1.00, 1.00, 0.19],  # S
    [1.00, 0.50, 0.00],  # P
    [0.12, 0.94, 0.12],  # Cl
    [0.65, 0.16, 0.16],  # Br
    [0.58, 0.00, 0.58],  # I
], dtype=np.float32)

# H, C, N, O, F, S, P, Cl, Br, I
COVALENT_RADII = np.array([0.31, 0.76, 0.71, 0.66, 0.57, 1.05, 1.07, 1.02, 1.20, 1.39], dtype=np.float32)
ATOM_RADII_PX = np.array([2, 4, 4, 4, 3, 5, 5, 4, 5, 5], dtype=np.int32)


def _draw_disk(img: np.ndarray, cx: int, cy: int, radius: int, value: float = 1.0) -> None:
    h, w = img.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x = cx + dx
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = max(img[y, x], value)


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, thickness: int = 1, value: float = 1.0) -> None:
    h, w = img.shape
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, steps + 1)
    ys = np.linspace(y0, y1, steps + 1)
    for xf, yf in zip(xs, ys):
        xi = int(round(float(xf)))
        yi = int(round(float(yf)))
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx * dx + dy * dy > thickness * thickness:
                    continue
                x = xi + dx
                y = yi + dy
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = max(img[y, x], value)


def infer_bonds_from_coords(coords_ang: np.ndarray, atom_types: np.ndarray, mask: np.ndarray, scale: float = 1.20) -> List[Tuple[int, int]]:
    """Infer a simple bond list from pairwise distances and covalent radii.

    This is intentionally lightweight for Stage 1 2D supervision. It is not
    meant to replace chemically exact bond perception.
    """
    valid_idx = np.where(mask > 0.5)[0]
    bonds: List[Tuple[int, int]] = []
    if valid_idx.size < 2:
        return bonds

    coords = coords_ang[valid_idx]
    types = atom_types[valid_idx]
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    for ii in range(len(valid_idx)):
        ti = int(types[ii])
        if ti < 0 or ti >= len(COVALENT_RADII):
            continue
        for jj in range(ii + 1, len(valid_idx)):
            tj = int(types[jj])
            if tj < 0 or tj >= len(COVALENT_RADII):
                continue
            dist = float(dmat[ii, jj])
            if dist < 0.45:
                continue
            cutoff = scale * float(COVALENT_RADII[ti] + COVALENT_RADII[tj])
            if dist <= cutoff:
                bonds.append((int(valid_idx[ii]), int(valid_idx[jj])))
    return bonds


def project_xy_to_pixels(coords_norm: np.ndarray, img_size: int) -> np.ndarray:
    """Project normalized XY coords in [-1,1] to pixel coordinates."""
    xy = coords_norm[:, :2].copy()
    px = ((xy[:, 0] + 1.0) * 0.5 * (img_size - 1)).round().astype(np.int32)
    py = ((xy[:, 1] + 1.0) * 0.5 * (img_size - 1)).round().astype(np.int32)
    px = np.clip(px, 0, img_size - 1)
    py = np.clip(py, 0, img_size - 1)
    return np.stack([px, py], axis=-1)


def render_v19_2d_target(
    coords_norm: np.ndarray,
    atom_types: np.ndarray,
    mask: np.ndarray,
    img_size: int = 128,
) -> np.ndarray:
    """Render a structured 2D molecular target with 12 channels.

    Channel layout:
    - 0: atom occupancy
    - 1: bond map
    - 2..11: per-type atom maps for 10 atom types
    """
    target = np.zeros((V19_2D_TARGET_CHANNELS, img_size, img_size), dtype=np.float32)
    valid = mask > 0.5
    if valid.sum() == 0:
        return target

    coords_ang = coords_norm.astype(np.float32) * 12.0
    pix = project_xy_to_pixels(coords_norm, img_size)
    bonds = infer_bonds_from_coords(coords_ang, atom_types, mask)

    bond_map = target[1]
    for i, j in bonds:
        x0, y0 = pix[i]
        x1, y1 = pix[j]
        _draw_line(bond_map, int(x0), int(y0), int(x1), int(y1), thickness=1, value=1.0)

    atom_map = target[0]
    for idx in np.where(valid)[0]:
        t = int(atom_types[idx])
        if t < 0 or t >= 10:
            continue
        x, y = pix[idx]
        radius = int(ATOM_RADII_PX[t])
        _draw_disk(atom_map, int(x), int(y), radius, value=1.0)
        _draw_disk(target[2 + t], int(x), int(y), radius, value=1.0)

    return target


def render_v19_joint_target(
    coords_norm: np.ndarray,
    atom_types: np.ndarray,
    mask: np.ndarray,
    img_size: int = 128,
) -> np.ndarray:
    """Render a structured 2D target plus a soft z-map.

    Channel layout:
    - 0: atom occupancy
    - 1: bond map
    - 2..11: per-type atom maps for 10 atom types
    - 12: z-map, storing normalized z in [0, 1] near atom centers
    """
    target2d = render_v19_2d_target(coords_norm, atom_types, mask, img_size=img_size)
    target = np.zeros((V19_JOINT_TARGET_CHANNELS, img_size, img_size), dtype=np.float32)
    target[:V19_2D_TARGET_CHANNELS] = target2d

    valid = mask > 0.5
    if valid.sum() == 0:
        return target

    pix = project_xy_to_pixels(coords_norm, img_size)
    z_map = np.zeros((img_size, img_size), dtype=np.float32)
    z_weight = np.zeros((img_size, img_size), dtype=np.float32)
    z_vals = np.clip((coords_norm[:, 2].astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)

    for idx in np.where(valid)[0]:
        t = int(atom_types[idx])
        if t < 0 or t >= 10:
            continue
        x, y = pix[idx]
        radius = int(ATOM_RADII_PX[t])
        z_val = float(z_vals[idx])
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                xx = int(x + dx)
                yy = int(y + dy)
                if 0 <= xx < img_size and 0 <= yy < img_size:
                    z_map[yy, xx] += z_val
                    z_weight[yy, xx] += 1.0

    nz = z_weight > 0
    z_map[nz] /= z_weight[nz]
    target[12] = z_map
    return target


def batch_render_v19_2d_targets(
    coords_batch: np.ndarray,
    types_batch: np.ndarray,
    mask_batch: np.ndarray,
    img_size: int = 128,
) -> np.ndarray:
    outs = []
    for coords, atom_types, mask in zip(coords_batch, types_batch, mask_batch):
        outs.append(render_v19_2d_target(coords, atom_types, mask, img_size=img_size))
    return np.stack(outs, axis=0).astype(np.float32)


def batch_render_v19_joint_targets(
    coords_batch: np.ndarray,
    types_batch: np.ndarray,
    mask_batch: np.ndarray,
    img_size: int = 128,
) -> np.ndarray:
    outs = []
    for coords, atom_types, mask in zip(coords_batch, types_batch, mask_batch):
        outs.append(render_v19_joint_target(coords, atom_types, mask, img_size=img_size))
    return np.stack(outs, axis=0).astype(np.float32)


def structure_map_to_rgb(struct_map: np.ndarray) -> np.ndarray:
    """Convert a 12-channel structure map into an RGB preview.

    Input may be either a hard target map or a soft prediction map in [0,1].
    Returns an RGB image in [0,1].
    """
    if struct_map.ndim != 3 or struct_map.shape[0] != V19_2D_TARGET_CHANNELS:
        raise ValueError(f"expected (12,H,W), got {struct_map.shape}")

    atom_occ = np.clip(struct_map[0], 0.0, 1.0)
    bond_map = np.clip(struct_map[1], 0.0, 1.0)
    type_maps = np.clip(struct_map[2:], 0.0, 1.0)  # (10, H, W)

    rgb = np.zeros((3, struct_map.shape[1], struct_map.shape[2]), dtype=np.float32)
    rgb += bond_map[None, ...] * 0.55

    for ti in range(10):
        rgb += TYPE_COLORS[ti][:, None, None] * type_maps[ti][None, ...]

    rgb = np.clip(rgb, 0.0, 1.0)
    rgb *= np.clip(0.35 + 0.65 * atom_occ[None, ...] + 0.35 * bond_map[None, ...], 0.0, 1.0)
    return np.clip(rgb, 0.0, 1.0)


def z_map_to_rgb(z_map: np.ndarray, atom_occ: np.ndarray | None = None) -> np.ndarray:
    """Convert a z-map in [0,1] to an RGB preview."""
    z = np.clip(z_map, 0.0, 1.0)
    rgb = np.zeros((3, z.shape[0], z.shape[1]), dtype=np.float32)
    rgb[0] = z
    rgb[1] = 0.25 + 0.75 * z
    rgb[2] = 1.0 - z
    if atom_occ is not None:
        alpha = np.clip(atom_occ, 0.0, 1.0)[None, ...]
        rgb *= 0.2 + 0.8 * alpha
    return np.clip(rgb, 0.0, 1.0)
