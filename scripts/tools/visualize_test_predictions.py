"""
Visualize the test set predictions that correspond to predictions_diffusion.json

This generates visualizations for the same samples saved in predictions_diffusion.json,
so you can see which visual corresponds to which prediction data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from src.data.dataset import QUAMAFMDataset, MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES
from src.models.diffusion import SE3EquivariantDenoiser, ConditionalDDPM
from src.models.baselines import ResNet3DRegression
from src.utils.metrics import compute_rmsd
from src.utils.visualize import ATOM_COLORS, ATOM_SIZES


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    model_type = config["model_type"]

    if model_type == "diffusion":
        from src.train import AFM3DReconModel
        model = AFM3DReconModel(config).to(device)
    else:
        model = ResNet3DRegression(
            img_size=config["img_size"],
            num_frames=config.get("num_frames", 10),
            max_atoms=MAX_ATOMS,
            num_atom_types=NUM_ATOM_TYPES,
        ).to(device)

    model.load_state_dict(state["model"])
    model.eval()
    return model, config, model_type


def plot_molecule(ax, coords, atom_types, mask, title=""):
    """Plot a single 3D molecule."""
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

    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")


def set_equal_axes(ax, coords, padding=0.5):
    """Set equal aspect ratio for 3D axes."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, 0.5) + padding
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def visualize_sample(gt_coords, pred_coords, gt_types, pred_types,
                     mask, afm_stack, rmsd_val, sample_idx, save_path):
    """Generate comparison figure for one test sample."""
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 10, figure=fig, hspace=0.3, wspace=0.3)

    # Top row: AFM Z-slices
    afm_np = afm_stack.cpu().numpy()  # (num_frames, H, W)
    slice_indices = [0, 2, 4, 6, 9]
    for i, z_idx in enumerate(slice_indices):
        ax = fig.add_subplot(gs[0, i * 2:i * 2 + 2])
        z_slice = afm_np[z_idx]
        im = ax.imshow(z_slice, cmap="viridis", origin="lower")
        ax.set_title(f"Z-slice {z_idx}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: 3D structures
    # GT
    ax_gt = fig.add_subplot(gs[1, :5], projection="3d")
    plot_molecule(ax_gt, gt_coords, gt_types, mask, title="Ground Truth")
    set_equal_axes(ax_gt, gt_coords[mask > 0])

    # Predicted
    ax_pred = fig.add_subplot(gs[1, 5:], projection="3d")
    plot_molecule(ax_pred, pred_coords, pred_types, mask,
                  title=f"Predicted (RMSD: {rmsd_val:.2f} Å)")
    set_equal_axes(ax_pred, pred_coords[mask > 0])

    plt.suptitle(f"Test Sample #{sample_idx}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize test set predictions corresponding to predictions_diffusion.json"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_diffusion.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of test samples to visualize (default: 10)")
    parser.add_argument("--output_dir", type=str, default="visualizations/test_predictions",
                        help="Directory to save visualization images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, config, model_type = load_model(args.checkpoint, device)
    print(f"Loaded {model_type} model from: {args.checkpoint}")

    # Load test dataset
    data_root = config.get("data_root")
    if data_root is None or data_root == "auto":
        data_root = os.path.join(os.path.dirname(__file__),
                                 "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM")

    dataset = QUAMAFMDataset(
        data_root=data_root,
        param_key=config.get("param_key", "K-1"),
        img_size=config.get("img_size", 128),
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        split="test",  # Use TEST set!
        val_size=config.get("val_size", 0),
        max_samples=config.get("max_samples", 0),
    )
    print(f"Test set: {len(dataset)} samples")

    num_samples = min(args.num_samples, len(dataset))
    os.makedirs(args.output_dir, exist_ok=True)

    # Create index mapping file
    index_mapping = []

    print(f"\nGenerating visualizations for first {num_samples} test samples...")
    print("=" * 70)

    for idx in range(num_samples):
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        # Inference
        with torch.no_grad():
            result = model.generate(batch)

        # Extract data
        gt_coords = batch["coords"][0].cpu().numpy()  # (max_atoms, 3)
        gt_types = batch["atom_types"][0].cpu().numpy()  # (max_atoms,)
        n_atoms = batch["n_atoms"][0].item()
        mask = batch["atom_mask"][0].cpu().numpy()  # (max_atoms,)
        afm_stack = batch["afm_stack"][0]  # (num_frames, H, W)

        pred_coords = result["coords"][0].cpu().numpy()
        pred_type_logits = result["type_logits"][0].cpu().numpy()
        pred_types = pred_type_logits.argmax(axis=-1)

        # Compute RMSD
        pred_coords_t = torch.from_numpy(pred_coords[:n_atoms]).unsqueeze(0)  # (1, n, 3)
        gt_coords_t = torch.from_numpy(gt_coords[:n_atoms]).unsqueeze(0)  # (1, n, 3)
        mask_t = torch.from_numpy(mask[:n_atoms]).unsqueeze(0)  # (1, n)
        rmsd_val = compute_rmsd(pred_coords_t, gt_coords_t, mask_t).item()

        # Save visualization
        save_path = os.path.join(args.output_dir, f"test_sample_{idx:05d}.png")
        visualize_sample(
            gt_coords, pred_coords, gt_types, pred_types,
            mask, afm_stack, rmsd_val, idx, save_path
        )

        # Record mapping
        index_mapping.append({
            "visualization_file": f"test_sample_{idx:05d}.png",
            "test_set_index": idx,
            "predictions_json_index": idx,
            "rmsd": float(rmsd_val),
            "n_atoms": int(n_atoms)
        })

        print(f"[{idx+1}/{num_samples}] test_sample_{idx:05d}.png  →  "
              f"Test sample #{idx}  (RMSD: {rmsd_val:.2f} Å)")

    # Save index mapping
    mapping_file = os.path.join(args.output_dir, "index_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(index_mapping, f, indent=2)

    print("=" * 70)
    print(f"\n✓ Generated {num_samples} visualization images")
    print(f"✓ Saved to: {args.output_dir}/")
    print(f"✓ Index mapping saved to: {mapping_file}")
    print(f"\nNOTE: These visualizations correspond to the first {num_samples} entries")
    print(f"      in predictions_diffusion.json (test set indices 0-{num_samples-1})")


if __name__ == "__main__":
    main()
