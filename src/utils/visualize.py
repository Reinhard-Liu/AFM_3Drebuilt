"""
Visualization utilities:
- 3D molecular structure comparison (GT vs Predicted)
- Training curves
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.data.dataset import ATOM_TYPES

# Color map for atoms
ATOM_COLORS = {
    "H": "#FFFFFF",
    "C": "#333333",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "S": "#FFFF30",
    "P": "#FF8000",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
}

ATOM_SIZES = {
    "H": 25, "C": 70, "N": 65, "O": 60,
    "F": 50, "S": 100, "P": 100, "Cl": 80,
    "Br": 90, "I": 110,
}


def plot_molecule_3d(
    coords: np.ndarray,
    atom_types: np.ndarray,
    mask: np.ndarray,
    ax=None,
    title: str = "",
):
    """Plot a 3D molecular structure.

    Args:
        coords: (N, 3) atom coordinates
        atom_types: (N,) atom type indices
        mask: (N,) atom mask
        ax: matplotlib 3D axis
        title: plot title
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    valid = mask > 0
    c = coords[valid]
    t = atom_types[valid]

    for i in range(len(c)):
        idx = int(t[i])
        if 0 <= idx < len(ATOM_TYPES):
            elem = ATOM_TYPES[idx]
        else:
            elem = "C"
        color = ATOM_COLORS.get(elem, "#999999")
        size = ATOM_SIZES.get(elem, 50)
        ax.scatter(c[i, 0], c[i, 1], c[i, 2], c=color, s=size,
                   edgecolors="black", linewidths=0.5, alpha=0.9)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    return ax


def plot_comparison(
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    gt_types: np.ndarray,
    pred_types: np.ndarray,
    mask: np.ndarray,
    save_path: str = None,
    sample_idx: int = 0,
):
    """Plot GT vs Predicted molecular structures side by side."""
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_molecule_3d(gt_coords, gt_types, mask, ax=ax1,
                     title=f"Ground Truth (Sample {sample_idx})")

    ax2 = fig.add_subplot(122, projection="3d")
    plot_molecule_3d(pred_coords, pred_types, mask, ax=ax2,
                     title=f"Predicted (Sample {sample_idx})")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(history_path: str, save_path: str = None):
    """Plot training and validation loss curves (all 5 components).

    Backward compatible: handles old history files with only 3 losses.
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    # Always present: total, coord, type
    train_loss = [m["loss"] for m in history["train"]]
    val_loss = [m["loss"] for m in history["val"]]
    train_coord = [m["coord_loss"] for m in history["train"]]
    val_coord = [m["coord_loss"] for m in history["val"]]
    train_type = [m["type_loss"] for m in history["train"]]
    val_type = [m["type_loss"] for m in history["val"]]

    # Optional: count_loss and retrieval_loss (may not exist in old files)
    train_count = [m.get("count_loss", 0.0) for m in history["train"]]
    val_count = [m.get("count_loss", 0.0) for m in history["val"]]
    train_retrieval = [m.get("retrieval_loss", 0.0) for m in history["train"]]
    val_retrieval = [m.get("retrieval_loss", 0.0) for m in history["val"]]

    epochs = range(1, len(train_loss) + 1)

    # Check if new losses exist (all non-zero)
    has_new_losses = any(train_count) or any(train_retrieval)

    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1, Col 1: Total loss
    axes[0, 0].plot(epochs, train_loss, label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label="Val", linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("Total Loss", fontsize=10)
    axes[0, 0].set_title("Total Loss", fontsize=11, fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Row 1, Col 2: Coordinate loss
    axes[0, 1].plot(epochs, train_coord, label="Train", linewidth=2)
    axes[0, 1].plot(epochs, val_coord, label="Val", linewidth=2)
    axes[0, 1].set_xlabel("Epoch", fontsize=10)
    axes[0, 1].set_ylabel("Coord Loss (MSE)", fontsize=10)
    axes[0, 1].set_title("Coordinate Loss", fontsize=11, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Row 1, Col 3: Type loss
    axes[0, 2].plot(epochs, train_type, label="Train", linewidth=2)
    axes[0, 2].plot(epochs, val_type, label="Val", linewidth=2)
    axes[0, 2].set_xlabel("Epoch", fontsize=10)
    axes[0, 2].set_ylabel("Type Loss (CE)", fontsize=10)
    axes[0, 2].set_title("Atom Type Loss", fontsize=11, fontweight="bold")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2, Col 1: Count loss
    if has_new_losses:
        axes[1, 0].plot(epochs, train_count, label="Train", linewidth=2)
        axes[1, 0].plot(epochs, val_count, label="Val", linewidth=2)
        axes[1, 0].set_xlabel("Epoch", fontsize=10)
        axes[1, 0].set_ylabel("Count Loss", fontsize=10)
        axes[1, 0].set_title("Atom Count Loss (weight=0.5)", fontsize=11, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, "Count Loss\n(Not available in old history file)",
                       ha='center', va='center', fontsize=12, color='gray')

    # Row 2, Col 2: Retrieval loss
    if has_new_losses:
        axes[1, 1].plot(epochs, train_retrieval, label="Train", linewidth=2)
        axes[1, 1].plot(epochs, val_retrieval, label="Val", linewidth=2)
        axes[1, 1].set_xlabel("Epoch", fontsize=10)
        axes[1, 1].set_ylabel("Retrieval Loss", fontsize=10)
        axes[1, 1].set_title("Molecule Retrieval Loss (weight=0.05)", fontsize=11, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
        axes[1, 1].text(0.5, 0.5, "Retrieval Loss\n(Not available in old history file)",
                       ha='center', va='center', fontsize=12, color='gray')

    # Row 2, Col 3: Summary text
    axes[1, 2].axis('off')
    if has_new_losses:
        summary_text = f"""Loss Components:

Total = coord_loss
      + 0.1  × type_loss
      + 0.5  × count_loss
      + 0.05 × retrieval_loss

Final Epoch ({len(epochs)}):
  Total:     {train_loss[-1]:.4f}
  Coord:     {train_coord[-1]:.4f}
  Type:      {train_type[-1]:.4f}
  Count:     {train_count[-1]:.4f}
  Retrieval: {train_retrieval[-1]:.4f}
"""
    else:
        summary_text = f"""Loss Components (Old Format):

Total = coord_loss
      + 0.1 × type_loss

Final Epoch ({len(epochs)}):
  Total: {train_loss[-1]:.4f}
  Coord: {train_coord[-1]:.4f}
  Type:  {train_type[-1]:.4f}

Note: This is an old history file.
Run new training to see all 5 losses.
"""
    axes[1, 2].text(0.1, 0.5, summary_text,
                    fontsize=10, family='monospace',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    if has_new_losses:
        title = "Training Loss Curves (All Components)"
    else:
        title = "Training Loss Curves (Old Format - 3 Components Only)"
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:  # Only create dir if path contains a directory
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_afm_stack(afm_stack: np.ndarray, save_path: str = None):
    """Visualize the 10-layer AFM image stack."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(afm_stack[i], cmap="gray")
        ax.set_title(f"Z-slice {i}")
        ax.axis("off")

    plt.suptitle("AFM Image Stack (10 Z-slices)", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
