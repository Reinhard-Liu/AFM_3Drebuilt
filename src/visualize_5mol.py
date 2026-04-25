"""
Generate 5-molecule 3D comparison plots for V16b validation samples.

Each output image contains 5 columns of 3D scatter plots:
    GT | Predicted | Top-1 | Top-2 | Top-3

Usage:
    python -m src.visualize_5mol \
        --checkpoint experiments/v16b/checkpoints/best_diffusion.pt \
        --num_samples 15 \
        --output_dir experiments/v16b/visual_compar
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data.dataset import QUAMAFMDataset, MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES
from src.models.baselines import ResNet3DRegression
from src.utils.metrics import compute_rmsd, compute_structure_fidelity
from src.utils.visualize import ATOM_COLORS, ATOM_SIZES
from src.visualize_val import (
    load_model, retrieve_by_3d_structure,
)


def _elem_str_from_types(types, mask):
    """Return element string like 'CHNO' from atom type array."""
    n = int((mask > 0).sum())
    if n == 0:
        return "?"
    t = types[:n]
    # Convert type indices to element symbols
    elems = set()
    for idx in t:
        i = int(idx)
        if 0 <= i < len(ATOM_TYPES):
            elems.add(ATOM_TYPES[i])
    return "".join(sorted(elems))


def _n_atoms_from_mask(mask):
    return int((mask > 0).sum())


def plot_molecule_3d(ax, coords, atom_types, mask, title="", elev=30, azim=45):
    """Plot a single 3D molecule on the given axis."""
    valid = mask > 0
    c = coords[valid]
    t = atom_types[valid]

    for i in range(len(c)):
        idx = int(t[i])
        elem = ATOM_TYPES[idx] if 0 <= idx < len(ATOM_TYPES) else "C"
        color = ATOM_COLORS.get(elem, "#999999")
        size = ATOM_SIZES.get(elem, 50)
        ax.scatter(c[i, 0], c[i, 1], c[i, 2],
                   c=color, s=size, edgecolors="black",
                   linewidths=0.5, alpha=0.9, depthshade=True)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.set_zlabel("Z", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=6)

    # Equal aspect ratio
    if len(c) > 0:
        mins = c.min(axis=0)
        maxs = c.max(axis=0)
        center = (mins + maxs) / 2.0
        half_range = max((maxs - mins).max() / 2.0, 0.5) + 0.3
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)


def generate_5mol_comparison(
    gt_coords, gt_types, gt_mask,
    pred_coords, pred_types, pred_mask,
    top_mols_data,  # list of (cid_str, sim, coords, types, db_mask, sim_dict) or None
    save_path,
    sample_idx,
    rmsd_val=None,
    fidelity_score=None,
    heavy_rmsd=None,
    atom_type_acc=None,
    ring_complete=None,
    gt_rank=None,
):
    """Generate a single figure with 5 3D molecule plots.

    V16c fix: GT and Pred use SEPARATE masks.

    Layout: GT | Predicted | Top-1 | Top-2 | Top-3
    """
    n_cols = 5
    fig = plt.figure(figsize=(22, 5))

    gt_n = _n_atoms_from_mask(gt_mask)
    pred_n = _n_atoms_from_mask(pred_mask)
    gt_elem = _elem_str_from_types(gt_types, gt_mask)
    pred_elem = _elem_str_from_types(pred_types, pred_mask)

    # Column 0: GT
    ax0 = fig.add_subplot(1, n_cols, 1, projection='3d')
    plot_molecule_3d(ax0, gt_coords, gt_types, gt_mask,
                     title=f"GT\n{gt_n} atoms | {gt_elem}")

    # Column 1: Predicted (uses pred_mask, NOT gt_mask)
    ax1 = fig.add_subplot(1, n_cols, 2, projection='3d')
    rmsd_str = f", HRMSD={heavy_rmsd:.3f}" if heavy_rmsd is not None else ""
    sim_str = f", fidelity={fidelity_score:.3f}" if fidelity_score is not None else ""
    plot_molecule_3d(ax1, pred_coords, pred_types, pred_mask,
                     title=f"Predicted\n{pred_n} atoms | {pred_elem}{rmsd_str}{sim_str}")

    # Columns 3-5: Top-1, Top-2, Top-3
    for col_i, mol_data in enumerate(top_mols_data):
        ax = fig.add_subplot(1, n_cols, col_i + 3, projection='3d')
        cid_str, sim, db_coords, db_types, db_mask, sim_dict = mol_data

        if db_coords is not None:
            m_n = _n_atoms_from_mask(db_mask)
            m_elem = _elem_str_from_types(db_types, db_mask)
            title = (
                f"Top-{col_i+1}\n"
                f"CID: {cid_str}\n"
                f"{m_n} atoms | {m_elem}\n"
                f"sim={sim:.3f}"
            )
            plot_molecule_3d(ax, db_coords, db_types, db_mask, title)
        else:
            title = f"Top-{col_i+1}\nCID: {cid_str}\nNot found"
            ax.set_title(title, fontsize=9, fontweight="bold", color="red")
            ax.axis("off")

    # Add info bar at the bottom
    if gt_rank is not None:
        fig.text(0.5, 0.02,
                 f"Sample #{sample_idx} | GT Rank: {gt_rank} | "
                 f"HeavyRMSD: {heavy_rmsd:.4f} | Fidelity: {fidelity_score:.3f} | "
                 f"Type: {atom_type_acc:.3f} | Ring: {ring_complete:.3f}",
                 ha='center', fontsize=10, style='italic',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    fig.suptitle(
        f"V16b 5-Molecule Comparison — Sample #{sample_idx}",
        fontsize=13, fontweight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate 5-molecule 3D comparison plots")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--output_dir", type=str,
                        default="experiments/v16b/visual_compar")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--gnn_checkpoint", type=str, default=None)
    parser.add_argument("--use_gt_count", action="store_true")
    parser.add_argument("--disable_guidance", action="store_true")
    parser.add_argument("--disable_ring_snap", action="store_true")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--ddim_steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seeds for deterministic inference
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    np.random.seed(seed)

    # Load model
    model, config, model_type = load_model(args.checkpoint, device)

    # Resolve data_root
    data_root = args.data_root or config.get("data_root")
    if data_root is None:
        data_root = os.path.join(
            os.path.dirname(__file__), "..",
            "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM"
        )

    # Load validation dataset
    dataset = QUAMAFMDataset(
        data_root=data_root,
        param_key=config.get("param_key", "K-1"),
        img_size=config.get("img_size", 128),
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        require_ring=config.get("require_ring", False),
        split="val",
        val_size=config.get("val_size", 0),
        max_samples=config.get("max_samples", 0),
    )
    print(f"Validation set: {len(dataset)} samples")

    # Build CID index mapping
    idx_to_cid = {v: k for k, v in dataset.cid_to_idx.items()}
    print(f"Number of unique CIDs: {len(idx_to_cid)}")

    total = len(dataset)
    if total == 0:
        print("Validation set is empty.")
        return

    num_samples = min(args.num_samples, total)
    os.makedirs(args.output_dir, exist_ok=True)

    # Evenly-spaced sample indices
    indices = np.linspace(0, total - 1, num_samples, dtype=int).tolist()

    print(f"\nGenerating {num_samples} comparison images → {args.output_dir}\n")

    for count, idx in enumerate(indices):
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        true_cid_idx = sample.get("cid_idx", -1)
        if isinstance(true_cid_idx, torch.Tensor):
            true_cid_idx = true_cid_idx.item()
        true_cid = idx_to_cid.get(true_cid_idx, f'idx_{true_cid_idx}')

        # Inference
        with torch.no_grad():
            if model_type == "diffusion":
                result = model.generate(
                    batch,
                    use_gt_count=args.use_gt_count,
                    use_ddim=(args.sampler == "ddim"),
                    ddim_steps=args.ddim_steps,
                    disable_guidance=args.disable_guidance,
                    disable_ring_snap=args.disable_ring_snap,
                    sampler=args.sampler,
                )
                coords_pred = result["coords"]
                type_logits = result["type_logits"]
            else:
                result = model.generate(batch)
                coords_pred = result["coords"]
                type_logits = result["type_logits"]
            pred_types = type_logits.argmax(dim=-1)

        # CPU numpy
        gt_coords = batch["coords"][0].cpu().numpy()
        gt_types = batch["atom_types"][0].cpu().numpy()
        mask = batch["atom_mask"][0].cpu().numpy()
        pred_coords_np = coords_pred[0].cpu().numpy()
        pred_types_np = pred_types[0].cpu().numpy()

        # V16c fix: build pred_mask from n_atoms_pred
        N = pred_coords_np.shape[0]
        n_atoms_pred = result.get("n_atoms_pred", None)
        if n_atoms_pred is not None:
            n_pred_int = int(n_atoms_pred[0].item())
            pred_mask_np = np.zeros(N, dtype=np.float32)
            pred_mask_np[:min(n_pred_int, N)] = 1.0
        else:
            pred_mask_np = mask.copy()

        # CID Retrieval (pred side uses pred_mask)
        exclude_cid = true_cid_idx
        top3_results, gt_sim_score, gt_sim_rank = retrieve_by_3d_structure(
            pred_coords_np, pred_types_np, pred_mask_np,
            dataset, idx_to_cid, top_k=3, exclude_cid_idx=exclude_cid
        )

        # Look up Top-1/2/3 3D structures from dataset
        top_mols_data = []
        for (cid, sim, n_pred, n_db, sim_dict) in top3_results:
            cid_str = str(cid)
            if cid_str in dataset.cid_to_idx:
                db_idx = dataset.cid_to_idx[cid_str]
                db_sample = dataset[db_idx]
                db_coords = db_sample['coords'].numpy()
                db_types = db_sample['atom_types'].numpy()
                db_mask = db_sample['atom_mask'].numpy()
                top_mols_data.append((cid_str, sim, db_coords, db_types, db_mask, sim_dict))
            else:
                top_mols_data.append((cid_str, sim, None, None, None, sim_dict))

        # RMSD
        rmsd_val = compute_rmsd(
            coords_pred.cpu(), batch["coords"].cpu(), batch["atom_mask"].cpu()
        ).item()

        fidelity = compute_structure_fidelity(
            coords_pred.cpu(), batch["coords"].cpu(),
            pred_types.cpu(), batch["atom_types"].cpu(),
            batch["atom_mask"].cpu(),
            n_atoms_pred=result.get("n_atoms_pred", None).cpu() if result.get("n_atoms_pred", None) is not None else None,
            scaffold_labels=batch if "scaffold_n_ring_systems" in batch else None,
        )
        fidelity_score = fidelity["struct_fidelity_score_mean"]
        heavy_rmsd = fidelity["matched_heavy_atom_rmsd_mean"]
        atom_type_acc = fidelity["atom_type_acc_mean"]
        ring_complete = fidelity["ring_complete_rate_mean"]

        # Save figure
        save_path = os.path.join(args.output_dir, f"sample_{idx:05d}_5mol.png")
        generate_5mol_comparison(
            gt_coords, gt_types, mask,
            pred_coords_np, pred_types_np, pred_mask_np,
            top_mols_data,
            save_path=save_path,
            sample_idx=idx,
            rmsd_val=rmsd_val,
            fidelity_score=fidelity_score,
            heavy_rmsd=heavy_rmsd,
            atom_type_acc=atom_type_acc,
            ring_complete=ring_complete,
            gt_rank=gt_sim_rank,
        )
        print(f"  [{count+1}/{num_samples}] Sample {idx}: "
              f"HeavyRMSD={heavy_rmsd:.4f}, fidelity={fidelity_score:.3f}, "
              f"GT_rank={gt_sim_rank} → {save_path}")

    print(f"\nDone. {num_samples} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
