"""
Prepare EDAFM experimental cases into a SUP-03-ready real-AFM dataset.

Output layout:
  output_root/
    manifest.json
    README.md
    cases/
      <case_id>/
        afm_stack.npy
        afm_stack_inverted.npy
        afm_stack_raw.npy
        coords_ang_centered.npy
        coords_norm.npy
        atom_types.npy
        metadata.json
        mol.xyz
        mol.png                (if present)

The processed fields are intentionally aligned with the current V20 input
expectation:
- AFM stack shape is (10, 128, 128)
- values are normalized to [0, 1]
- coordinates are centered and normalized to [-1, 1]-like scale by /12 Å
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import ATOM_TO_IDX, center_coords, parse_xyz


def load_afm_stack(npz_path: Path) -> tuple[np.ndarray, float, float]:
    data = np.load(npz_path, allow_pickle=True)
    stack = np.asarray(data["data"], dtype=np.float32)  # (H, W, D)
    length_x = float(data["lengthX"])
    length_y = float(data["lengthY"])
    stack = np.transpose(stack, (2, 0, 1))  # -> (D, H, W)
    return stack, length_x, length_y


def resize_stack(stack: np.ndarray, target_depth: int, target_img_size: int) -> np.ndarray:
    tensor = torch.from_numpy(stack).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    resized = F.interpolate(
        tensor,
        size=(target_depth, target_img_size, target_img_size),
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def normalize_stack(stack: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(stack, q_low))
    hi = float(np.percentile(stack, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(stack.min())
        hi = float(stack.max())
    if hi <= lo:
        return np.zeros_like(stack, dtype=np.float32)
    clipped = np.clip(stack, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    return norm.astype(np.float32)


def save_preview(stack: np.ndarray, out_path: Path, title: str):
    d = stack.shape[0]
    idxs = [0, d // 2, d - 1]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, i, name in zip(axes, idxs, ["low", "mid", "high"]):
        ax.imshow(stack[i], cmap="afmhot", vmin=0.0, vmax=1.0)
        ax.set_title(f"{name} slice")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def prepare_case(
    case_dir: Path,
    output_cases_root: Path,
    target_depth: int,
    target_img_size: int,
    q_low: float,
    q_high: float,
) -> dict:
    molecule_name = case_dir.name
    xyz_candidates = sorted(case_dir.glob("*.xyz"))
    if not xyz_candidates:
        raise FileNotFoundError(f"no xyz found in {case_dir}")

    xyz_path = None
    for candidate in xyz_candidates:
        if candidate.name == "mol.xyz":
            xyz_path = candidate
            break
    if xyz_path is None:
        xyz_path = xyz_candidates[0]

    coords_ang, elements = parse_xyz(str(xyz_path))
    coords_ang = center_coords(coords_ang)
    coords_norm = (coords_ang / 12.0).astype(np.float32)
    atom_types = np.asarray([ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"]) for e in elements], dtype=np.int64)

    records = []
    for npz_path in sorted(case_dir.glob("*_exp.npz")):
        stack_native, length_x, length_y = load_afm_stack(npz_path)
        stack_resized = resize_stack(stack_native, target_depth=target_depth, target_img_size=target_img_size)
        stack_norm = normalize_stack(stack_resized, q_low=q_low, q_high=q_high)
        stack_inverted = (1.0 - stack_norm).astype(np.float32)

        tip = "CO" if "_CO_" in npz_path.name else "Xe" if "_Xe_" in npz_path.name else "unknown"
        case_id = f"edafm_{molecule_name}_{tip}_exp"
        out_dir = output_cases_root / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "afm_stack_raw.npy", stack_resized.astype(np.float32))
        np.save(out_dir / "afm_stack.npy", stack_norm)
        np.save(out_dir / "afm_stack_inverted.npy", stack_inverted)
        np.save(out_dir / "coords_ang_centered.npy", coords_ang.astype(np.float32))
        np.save(out_dir / "coords_norm.npy", coords_norm)
        np.save(out_dir / "atom_types.npy", atom_types)

        shutil.copy2(xyz_path, out_dir / "mol.xyz")
        mol_png = case_dir / "mol.png"
        if mol_png.exists():
            shutil.copy2(mol_png, out_dir / "mol.png")
        save_preview(stack_norm, out_dir / "preview.png", title=case_id)

        metadata = {
            "case_id": case_id,
            "molecule_name": molecule_name,
            "tip": tip,
            "is_experimental": True,
            "source_npz": str(npz_path),
            "source_xyz": str(xyz_path),
            "image_shape_native_dhw": list(stack_native.shape),
            "image_shape_processed_dhw": list(stack_norm.shape),
            "length_x_ang": length_x,
            "length_y_ang": length_y,
            "n_atoms": int(len(elements)),
            "elements": list(elements),
            "normalization": {
                "mode": "percentile_clip_minmax",
                "q_low": q_low,
                "q_high": q_high,
            },
            "default_contrast_variant": "normal",
            "available_contrast_variants": ["normal", "inverted"],
            "files": {
                "afm_stack": "afm_stack.npy",
                "afm_stack_inverted": "afm_stack_inverted.npy",
                "afm_stack_raw": "afm_stack_raw.npy",
                "coords_norm": "coords_norm.npy",
                "coords_ang_centered": "coords_ang_centered.npy",
                "atom_types": "atom_types.npy",
                "mol_xyz": "mol.xyz",
                "preview": "preview.png",
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        records.append(
            {
                "case_id": case_id,
                "case_dir": str(Path("cases") / case_id),
                "molecule_name": molecule_name,
                "tip": tip,
                "source_npz": str(npz_path),
                "source_xyz": str(xyz_path),
                "n_atoms": int(len(elements)),
                "elements": list(elements),
                "has_gt_structure": True,
                "image_shape_native_dhw": list(stack_native.shape),
                "image_shape_processed_dhw": list(stack_norm.shape),
                "length_x_ang": length_x,
                "length_y_ang": length_y,
                "default_contrast_variant": "normal",
                "available_contrast_variants": ["normal", "inverted"],
            }
        )

    return records


def write_readme(output_root: Path):
    text = """# EDAFM SUP-03 Prepared Cases

This directory contains real-AFM experimental cases converted into a V20/SUP-03
compatible format.

Key design choices:
- AFM stacks are resampled to `(10, 128, 128)` to match the current model input.
- Values are normalized to `[0, 1]` with percentile clipping.
- Both `normal` and `inverted` contrast variants are stored.
- Molecule coordinates are centered and normalized by `/12.0`, matching the
  current QUAM-AFM training convention.

Use `manifest.json` as the index.
"""
    (output_root / "README.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edafm_root",
        type=str,
        default="/root/autodl-tmp/real_afm_datasets/edafm_zenodo_10609676/edafm-data/edafm-data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/micro/real_afm/edafm_sup03_cases",
    )
    parser.add_argument("--target_depth", type=int, default=10)
    parser.add_argument("--target_img_size", type=int, default=128)
    parser.add_argument("--q_low", type=float, default=1.0)
    parser.add_argument("--q_high", type=float, default=99.0)
    args = parser.parse_args()

    edafm_root = Path(args.edafm_root)
    output_root = Path(args.output_root)
    output_cases_root = output_root / "cases"
    output_cases_root.mkdir(parents=True, exist_ok=True)

    all_cases = []
    for case_dir in sorted(p for p in edafm_root.iterdir() if p.is_dir()):
        exp_files = list(case_dir.glob("*_exp.npz"))
        if not exp_files:
            continue
        all_cases.extend(
            prepare_case(
                case_dir=case_dir,
                output_cases_root=output_cases_root,
                target_depth=int(args.target_depth),
                target_img_size=int(args.target_img_size),
                q_low=float(args.q_low),
                q_high=float(args.q_high),
            )
        )

    manifest = {
        "dataset_name": "EDAFM experimental cases for SUP-03",
        "source_root": str(edafm_root),
        "num_cases": len(all_cases),
        "target_depth": int(args.target_depth),
        "target_img_size": int(args.target_img_size),
        "cases": all_cases,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_readme(output_root)
    print(json.dumps({"output_root": str(output_root), "num_cases": len(all_cases)}, indent=2))


if __name__ == "__main__":
    main()
