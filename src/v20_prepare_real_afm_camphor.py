"""
Prepare public experimental camphor AFM images for SUP-03.

The source dataset provides real AFM stacks as `.npy` files with shape
`(1, H, W, 10)`. This script converts them into the same case-directory layout
used by the prepared EDAFM real-AFM set, but without GT coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def load_afm_stack(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected camphor AFM shape at {path}: {arr.shape}")
    # (H, W, D) -> (D, H, W)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def resize_stack(stack: np.ndarray, target_depth: int, target_img_size: int) -> np.ndarray:
    tensor = torch.from_numpy(stack).unsqueeze(0).unsqueeze(0)
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


def save_preview(stack: np.ndarray, out_path: Path, title: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camphor_root",
        type=str,
        default="/root/autodl-tmp/real_afm_datasets/camphor_zenodo_10562769/afm_camphor/afm_camphor",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/root/autodl-tmp/micro/real_afm/camphor_sup03_cases",
    )
    parser.add_argument("--target_depth", type=int, default=10)
    parser.add_argument("--target_img_size", type=int, default=128)
    parser.add_argument("--q_low", type=float, default=1.0)
    parser.add_argument("--q_high", type=float, default=99.0)
    args = parser.parse_args()

    camphor_root = Path(args.camphor_root)
    output_root = Path(args.output_root)
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    records = []
    for npy_path in sorted(camphor_root.glob("*.npy")):
        case_id = f"camphor_exp_{npy_path.stem}"
        case_dir = cases_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        stack_native = load_afm_stack(npy_path)
        stack_resized = resize_stack(
            stack_native,
            target_depth=int(args.target_depth),
            target_img_size=int(args.target_img_size),
        )
        stack_norm = normalize_stack(stack_resized, q_low=float(args.q_low), q_high=float(args.q_high))
        stack_inverted = (1.0 - stack_norm).astype(np.float32)

        np.save(case_dir / "afm_stack_raw.npy", stack_resized.astype(np.float32))
        np.save(case_dir / "afm_stack.npy", stack_norm)
        np.save(case_dir / "afm_stack_inverted.npy", stack_inverted)
        save_preview(stack_norm, case_dir / "preview.png", title=case_id)

        metadata = {
            "case_id": case_id,
            "molecule_name": "camphor",
            "molecule_label": "camphor",
            "tip": "unknown",
            "is_experimental": True,
            "has_gt_structure": False,
            "gt_structure_compatible": False,
            "source_file": str(npy_path),
            "image_shape_native_dhw": list(stack_native.shape),
            "image_shape_processed_dhw": list(stack_norm.shape),
            "normalization": {
                "mode": "percentile_clip_minmax",
                "q_low": float(args.q_low),
                "q_high": float(args.q_high),
            },
            "default_contrast_variant": "normal",
            "available_contrast_variants": ["normal", "inverted"],
            "files": {
                "afm_stack": "afm_stack.npy",
                "afm_stack_inverted": "afm_stack_inverted.npy",
                "afm_stack_raw": "afm_stack_raw.npy",
                "preview": "preview.png",
            },
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        records.append(
            {
                "case_id": case_id,
                "case_dir": str(Path("cases") / case_id),
                "molecule_name": "camphor",
                "molecule_label": "camphor",
                "tip": "unknown",
                "source_npz": str(npy_path),
                "n_atoms": 0,
                "elements": [],
                "has_gt_structure": False,
                "gt_structure_compatible": False,
                "image_shape_native_dhw": list(stack_native.shape),
                "image_shape_processed_dhw": list(stack_norm.shape),
                "default_contrast_variant": "normal",
                "available_contrast_variants": ["normal", "inverted"],
            }
        )

    manifest = {
        "dataset_name": "camphor experimental cases for SUP-03",
        "source_root": str(camphor_root),
        "num_cases": len(records),
        "target_depth": int(args.target_depth),
        "target_img_size": int(args.target_img_size),
        "cases": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_root / "README.md").write_text(
        "# Camphor SUP-03 Prepared Cases\n\n"
        "Prepared public experimental AFM images of 1S-camphor without GT coordinates.\n"
    )
    print(json.dumps({"output_root": str(output_root), "num_cases": len(records)}, indent=2))


if __name__ == "__main__":
    main()
