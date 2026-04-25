"""
Visualize GT vs Predicted molecular coordinates on validation samples.

Load trained weights, pick 10 samples from the validation set, run inference,
and save side-by-side 3D comparison plots.

Usage:
    python -m src.visualize_val \
        --checkpoint ./micro/checkpoints/best_diffusion.pt \
        --num_samples 10 \
        --output_dir micro/val_visualizations
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.data.dataset import (
    QUAMAFMDataset, MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES,
)
from src.models.diffusion import SE3EquivariantDenoiser, ConditionalDDPM
from src.models.baselines import ResNet3DRegression
from src.utils.metrics import compute_rmsd, compute_structure_fidelity
from src.utils.visualize import ATOM_COLORS, ATOM_SIZES

# Store retrieval head for CID lookup
RETRIEVAL_HEAD = None
MOL_EMBEDDINGS = None
CID_LIST = None  # List of CID strings in order
VAL_DATASET = None  # Store validation dataset for 3D structure retrieval

# Atomic numbers for Coulomb matrix (matches metrics.py)
_ATOMIC_NUMBERS = np.array([1, 6, 7, 8, 9, 16, 15, 17, 35, 53], dtype=np.float64)


# ============================================================
# CORRECTED CID Retrieval: Based on Training Metrics
# ============================================================




def _coulomb_eigenvalues(coords: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Sorted Coulomb matrix eigenvalues (rotation-invariant descriptor).

    C_ij = Z_i * Z_j / |r_i - r_j| for i != j
    """
    n = len(coords)
    if n == 0:
        return np.array([0.0])
    Z = _ATOMIC_NUMBERS[types]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                C[i, i] = 0.5 * Z[i] ** 2.4
            else:
                r_ij = np.linalg.norm(coords[i] - coords[j])
                if r_ij > 1e-8:
                    C[i, j] = Z[i] * Z[j] / r_ij
    eigenvalues = np.linalg.eigvalsh(C)
    return np.sort(eigenvalues)[::-1]


def compute_molecular_similarity_for_retrieval(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    db_coords: np.ndarray,
    db_types: np.ndarray,
    db_mask: np.ndarray,
    TYPE_COST: float = 5.0,
) -> dict:
    """Compute molecular similarity for CID retrieval.

    This method uses the same principles as compute_structure_similarity():
      1. Hungarian matching (spatial + element type cost matrix)
      2. Atom-type accuracy after matching
      3. Coulomb matrix eigenvalue similarity
      4. Element composition cosine similarity
      5. Atom count exact match

    Returns:
        dict with type_acc, coulomb_sim, elem_sim, count_sim, overall, n_pred, n_db
    """
    from scipy.optimize import linear_sum_assignment

    n_pred = int((pred_mask > 0).sum())
    n_db = int((db_mask > 0).sum())

    if n_pred == 0 or n_db == 0:
        return {
            "type_acc": 0.0, "coulomb_sim": 0.0,
            "elem_sim": 0.0, "count_sim": 0.0, "overall": 0.0,
            "n_pred": n_pred, "n_db": n_db,
        }

    pc = pred_coords[:n_pred]
    pt = pred_types[:n_pred]
    gc = db_coords[:n_db]
    gt_t = db_types[:n_db]

    # =============================================
    # Step 1: Hungarian matching
    # Cost = spatial_distance + TYPE_COST * (type_mismatch)
    # =============================================
    diff = pc[:, None, :] - gc[None, :, :]  # (n_pred, n_db, 3)
    spatial_cost = np.sqrt((diff ** 2).sum(axis=-1))  # (n_pred, n_db)

    type_cost = np.abs(pt[:, None].astype(int) - gt_t[None, :].astype(int))
    type_cost = TYPE_COST * (type_cost > 0).astype(float)  # 0 or TYPE_COST

    total_cost = spatial_cost + type_cost  # (n_pred, n_db)

    row_ind, col_ind = linear_sum_assignment(total_cost)
    n_matched = len(row_ind)

    pc_matched = pc[row_ind]
    gc_matched = gc[col_ind]
    pt_matched = pt[row_ind]
    gt_t_matched = gt_t[col_ind]

    # =============================================
    # Step 2: Atom-type accuracy (after matching)
    # =============================================
    if n_matched > 0:
        type_correct = (pt_matched == gt_t_matched).sum()
        type_acc = float(type_correct / n_matched)
    else:
        type_acc = 0.0

    # =============================================
    # Step 3: Coulomb matrix eigenvalue similarity
    # =============================================
    if n_matched >= 2:
        eig_pred = _coulomb_eigenvalues(pc_matched, pt_matched)
        eig_gt = _coulomb_eigenvalues(gc_matched, gt_t_matched)
        max_len = max(len(eig_pred), len(eig_gt))
        eig_pred_pad = np.zeros(max_len)
        eig_gt_pad = np.zeros(max_len)
        eig_pred_pad[:len(eig_pred)] = eig_pred
        eig_gt_pad[:len(eig_gt)] = eig_gt
        l2 = np.linalg.norm(eig_gt_pad - eig_pred_pad)
        norm = max(np.linalg.norm(eig_gt_pad), 1e-8)
        coulomb_sim = max(0.0, 1.0 - l2 / norm)
    else:
        coulomb_sim = 0.0

    # =============================================
    # Step 4: Element composition cosine similarity
    # =============================================
    c_pred = np.bincount(pt, minlength=NUM_ATOM_TYPES).astype(float)
    c_gt = np.bincount(gt_t, minlength=NUM_ATOM_TYPES).astype(float)
    dot = (c_pred * c_gt).sum()
    norm = np.linalg.norm(c_pred) * np.linalg.norm(c_gt)
    elem_sim = dot / norm if norm > 0 else 0.0

    # =============================================
    # Step 5: Atom count similarity
    # =============================================
    count_sim = 1.0 - abs(n_pred - n_db) / max(n_pred, n_db)

    # =============================================
    # Step 6: Overall score
    # =============================================
    overall = (
        0.20 * type_acc
        + 0.15 * coulomb_sim
        + 0.20 * elem_sim
        + 0.10 * count_sim
        + 0.15 * count_sim
    )
    overall = max(0.0, min(1.0, overall))

    return {
        "type_acc": type_acc,
        "coulomb_sim": coulomb_sim,
        "elem_sim": elem_sim,
        "count_sim": count_sim,
        "overall": overall,
        "n_pred": n_pred,
        "n_db": n_db,
        "n_matched": n_matched,
    }


def retrieve_by_3d_structure(
    pred_coords: np.ndarray,
    pred_types: np.ndarray,
    pred_mask: np.ndarray,
    dataset,
    idx_to_cid: dict,
    top_k: int = 3,
    exclude_cid_idx: int = None,
) -> tuple:
    """Retrieve top-k similar molecules from dataset based on molecular similarity.

    Uses Hungarian matching for CID retrieval.

    Returns:
        top_results: list of (cid, sim, n_atoms, n_db_atoms) for top-k matches
        gt_sim: overall similarity score of GT molecule (None if exclude_cid_idx is None)
        gt_rank: 1-based rank of GT molecule among all candidates
    """
    similarities = []
    gt_sim = None
    gt_idx = None

    for db_idx in range(len(dataset)):
        db_sample = dataset[db_idx]
        db_cid_idx = db_sample.get('cid_idx', -1)
        if hasattr(db_cid_idx, 'item'):
            db_cid_idx = db_cid_idx.item()
        elif hasattr(db_cid_idx, 'numpy'):
            db_cid_idx = db_cid_idx.item()

        db_coords = db_sample['coords'].numpy()
        db_types = db_sample['atom_types'].numpy()
        db_mask = db_sample['atom_mask'].numpy()

        sim_dict = compute_molecular_similarity_for_retrieval(
            pred_coords, pred_types, pred_mask,
            db_coords, db_types, db_mask,
        )

        # Record GT similarity separately
        if exclude_cid_idx is not None and db_cid_idx == exclude_cid_idx:
            gt_sim = sim_dict["overall"]
            gt_idx = db_idx
            continue

        # Get CID from cid_idx
        cid = idx_to_cid.get(db_cid_idx, f'idx_{db_cid_idx}')
        n_pred = sim_dict["n_pred"]
        n_db = sim_dict["n_db"]
        similarities.append((sim_dict["overall"], cid, db_cid_idx, n_pred, n_db, sim_dict))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Compute GT rank (where would GT place among all other molecules)
    gt_rank = None
    if gt_sim is not None:
        gt_rank = sum(1 for s, _, _, _, _, _ in similarities if s > gt_sim) + 1

    # Return top-k
    top_results = []
    for s, cid, cidx, n_pred, n_db, sim_dict in similarities[:top_k]:
        top_results.append((cid, s, n_pred, n_db, sim_dict))

    return top_results, gt_sim, gt_rank


# ============================================================
# Legacy buggy method (kept for reference)
# ============================================================

def compute_distance_histogram_similarity(
    coords1, types1, mask1, coords2, types2, mask2,
    n_bins=20, max_dist=5.0
):
    """BUGGY: density=True causes different-atom-count molecules to get sim≈1.0.

    DO NOT USE. Kept for reference only.
    """
    from scipy.spatial.distance import cdist

    valid1 = mask1 > 0
    valid2 = mask2 > 0
    c1 = coords1[valid1]
    c2 = coords2[valid2]

    n1, n2 = len(c1), len(c2)
    if n1 < 2 or n2 < 2:
        return 0.0

    d1 = cdist(c1, c1)
    d2 = cdist(c2, c2)
    d1 = d1[np.triu_indices(n1, k=1)]
    d2 = d2[np.triu_indices(n2, k=1)]

    bins = np.linspace(0, max_dist, n_bins + 1)
    # BUG: density=True makes integral=1.0 regardless of atom count
    h1, _ = np.histogram(d1, bins=bins, density=True)
    h2, _ = np.histogram(d2, bins=bins, density=True)

    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return np.dot(h1, h2) / (norm1 * norm2)


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint, auto-detect model type from saved config."""
    global RETRIEVAL_HEAD, MOL_EMBEDDINGS, CID_LIST

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
    print(f"Loaded {model_type} model from: {checkpoint_path}")
    print(f"  Epoch: {state.get('epoch', '?')}, Val loss: {state.get('val_loss', 'N/A')}")

    # Setup retrieval components if available
    if hasattr(model, 'mol_embeddings') and model.mol_embeddings is not None:
        RETRIEVAL_HEAD = model.retrieval_head
        MOL_EMBEDDINGS = model.mol_embeddings
        print(f"  Retrieval head available: {MOL_EMBEDDINGS.weight.shape}")

    return model, config, model_type


# ============================================================
# Plotting utilities
# ============================================================

def plot_molecule(ax, coords, atom_types, mask, title=""):
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

    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")


def set_equal_axes(ax, coords, padding=0.5):
    """Set equal aspect ratio for 3D axes based on coordinate range."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, 0.5) + padding
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)


def visualize_sample(
    gt_coords, pred_coords, gt_types, pred_types,
    gt_mask, pred_mask, afm_stack, rmsd_val, sample_idx, save_path,
    fidelity_metrics=None, n_pred=None, true_cid=None,
    pred_3d_cids=None, pred_3d_scores=None,
    gt_sim_score=None, gt_sim_rank=None,
):
    """Generate a full comparison figure for one sample.

    V16c fix: GT and Pred now use SEPARATE masks.
    gt_mask: (N,) float, from batch["atom_mask"]
    pred_mask: (N,) float, constructed from n_atoms_pred

    Layout:
        Top row    – 5 selected AFM Z-slices (0, 2, 4, 6, 9)
        Bottom row – GT 3D | Predicted 3D | Retrieval panel
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: AFM slices
    slice_indices = [0, 2, 4, 6, 9]
    for i, si in enumerate(slice_indices):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        ax.imshow(afm_stack[si], cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(f"Z-slice {si}", fontsize=9)
        ax.axis("off")

    gt_valid = gt_mask > 0
    n_gt_valid = int(gt_valid.sum())
    pred_valid = pred_mask > 0
    n_pred_valid = int(pred_valid.sum())
    n_pred_display = n_pred if n_pred is not None else n_pred_valid

    # Bottom-left: Ground Truth
    ax_gt = fig.add_subplot(gs[1, 0:5], projection="3d")
    plot_molecule(ax_gt, gt_coords, gt_types, gt_mask,
                  title=f"Ground Truth ({n_gt_valid} atoms)")
    if n_gt_valid > 0:
        set_equal_axes(ax_gt, gt_coords[gt_valid])

    # Bottom-middle: Predicted (uses pred_mask, NOT gt_mask)
    ax_pred = fig.add_subplot(gs[1, 5:10], projection="3d")
    plot_molecule(ax_pred, pred_coords, pred_types, pred_mask,
                  title=f"Predicted ({n_pred_display} atoms, RMSD={rmsd_val:.3f})")
    if n_pred_valid > 0:
        set_equal_axes(ax_pred, pred_coords[pred_valid])

    # Bottom-right: Metrics + Retrieval panel
    ax_info = fig.add_subplot(gs[1, 10:12])
    ax_info.axis("off")

    if fidelity_metrics is not None:
        fidelity_score = fidelity_metrics.get("struct_fidelity_score_mean", 0.0)
        count_exact = fidelity_metrics.get("atom_count_exact_mean", 0.0)
        heavy_rmsd = fidelity_metrics.get("matched_heavy_atom_rmsd_mean", 0.0)
        atom_type_acc = fidelity_metrics.get("atom_type_acc_mean", 0.0)
        hetero_f1 = fidelity_metrics.get("heteroatom_f1_mean", 0.0)
        ring_complete = fidelity_metrics.get("ring_complete_rate_mean", 0.0)
        attachment_f1 = fidelity_metrics.get("attachment_edge_f1_mean", 0.0)
        bond_valid = fidelity_metrics.get("local_chem_score_mean", 0.0)

        if fidelity_score >= 0.7:
            score_color = "#2ecc71"
        elif fidelity_score >= 0.5:
            score_color = "#f39c12"
        else:
            score_color = "#e74c3c"

        info_text = (
            f"Structure Fidelity\n"
            f"{'=' * 26}\n\n"
            f"Fidelity:  {fidelity_score:.3f}\n"
            f"CountEq:   {count_exact:.3f}\n"
            f"HeavyRMSD: {heavy_rmsd:.3f}\n"
            f"Type Acc:  {atom_type_acc:.3f}\n"
            f"HeteroF1:  {hetero_f1:.3f}\n"
            f"RingComp:  {ring_complete:.3f}\n"
            f"AttachF1:  {attachment_f1:.3f}\n"
            f"BondPred:  {bond_valid:.3f}\n\n"
            f"Atoms: {n_pred_display} / {n_gt_valid}\n"
            f"RMSD:  {rmsd_val:.4f}"
        )

        ax_info.text(0.05, 0.95, info_text,
                     transform=ax_info.transAxes,
                     fontsize=10, family='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round,pad=0.5',
                               facecolor=score_color, alpha=0.2,
                               edgecolor=score_color, linewidth=2))

        metrics = ['Type', 'Hetero', 'Ring', 'Bond']
        values = [atom_type_acc, hetero_f1, ring_complete, bond_valid]
        colors = ['#e67e22', '#9b59b6', '#2ecc71', '#3498db']

        ax_bar = fig.add_subplot(gs[0, 10:12])
        bars = ax_bar.barh(metrics, values, color=colors, edgecolor='white', height=0.6)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_title(f"Fidelity = {fidelity_score:.3f}", fontsize=10, fontweight='bold',
                         color=score_color)
        ax_bar.tick_params(axis='y', labelsize=8)
        ax_bar.tick_params(axis='x', labelsize=7)
        for bar, val in zip(bars, values):
            ax_bar.text(min(val + 0.02, 0.92), bar.get_y() + bar.get_height() / 2,
                        f'{val:.2f}', va='center', fontsize=7)

    # CID Retrieval panel (row 2)
    if true_cid is not None or pred_3d_cids is not None:
        ax_cid = fig.add_subplot(gs[2, :])
        ax_cid.axis("off")

        cid_info_lines = []
        cid_info_lines.append("CID Retrieval (Hungarian Matching)")
        cid_info_lines.append("=" * 50)

        if true_cid is not None:
            cid_info_lines.append(f"GT CID:  {true_cid}")

        if pred_3d_cids is not None:
            cid_info_lines.append("")
            cid_info_lines.append("Top-3 by Molecular Similarity:")
            for i, (cid, score, n_pred, n_db, sim_dict) in enumerate(pred_3d_cids[:3]):
                cid_info_lines.append(
                    f"  {i+1}. {cid}  (sim={score:.3f}, "
                    f"type_acc={sim_dict['type_acc']:.3f}, "
                    f"coulomb={sim_dict['coulomb_sim']:.3f}, "
                    f"elem_sim={sim_dict['elem_sim']:.3f}, "
                    f"n={n_pred}/{n_db})"
                )

        if gt_sim_score is not None:
            cid_info_lines.append("")
            rank_str = f"rank {gt_sim_rank}" if gt_sim_rank is not None else "N/A"
            cid_info_lines.append(
                f"GT Similarity:  {gt_sim_score:.3f}  ({rank_str})"
            )

        cid_text = "\n".join(cid_info_lines)
        ax_cid.text(0.02, 0.95, cid_text,
                    transform=ax_cid.transAxes,
                    fontsize=11, family='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='#3498db', alpha=0.1,
                              edgecolor='#3498db', linewidth=1))

    fig.suptitle(f"Validation Sample #{sample_idx}",
                 fontsize=14, fontweight="bold", y=0.98)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT vs Predicted molecular coordinates on validation samples"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to QUAM data root (default: use config in checkpoint)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of validation samples to visualize")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save output images")
    parser.add_argument("--gnn_checkpoint", type=str, default=None,
                        help="Path to GNN TypeClassifier checkpoint")
    parser.add_argument("--use_gt_count", action="store_true",
                        help="Use GT atom count instead of predicted count.")
    parser.add_argument("--disable_guidance", action="store_true",
                        help="Disable all physics guidance during sampling.")
    parser.add_argument("--disable_ring_snap", action="store_true",
                        help="Disable all ring snapping during sampling.")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"],
                        help="Sampler type: ddim or ddpm.")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM steps.")
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

    # Load GNN if provided
    gnn_model = None
    if args.gnn_checkpoint:
        from src.models.gnn_type_classifier import GNNTypeClassifier
        from src.eval_phase1 import chemical_type_correction
        gnn_model = GNNTypeClassifier().to(device)
        gnn_state = torch.load(args.gnn_checkpoint, map_location=device, weights_only=False)
        gnn_model.load_state_dict(gnn_state.get("gnn", gnn_state.get("model")))
        gnn_model.eval()
        print(f"GNN loaded: {args.gnn_checkpoint} (epoch {gnn_state.get('epoch','?')}, val_acc={gnn_state.get('val_acc','?'):.4f})")

    # Resolve data_root
    data_root = args.data_root or config.get("data_root")
    if data_root is None:
        data_root = os.path.join(os.path.dirname(__file__), "..",
                                 "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM")

    # Resolve output_dir
    output_dir = args.output_dir or os.path.join(
        config.get("save_dir", "micro/checkpoints"), "val_visualizations"
    )

    # Load validation dataset (use config parameters exactly)
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

    global VAL_DATASET
    VAL_DATASET = dataset

    # Build CID index to string mapping
    idx_to_cid = {v: k for k, v in dataset.cid_to_idx.items()}
    print(f"Number of unique CIDs: {len(idx_to_cid)}")

    total = len(dataset)
    if total == 0:
        print("Validation set is empty, nothing to visualize.")
        return

    num_samples = min(args.num_samples, total)
    os.makedirs(output_dir, exist_ok=True)

    # Pick evenly-spaced indices for diversity
    if num_samples >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_samples, dtype=int).tolist()

    print(f"Generating comparison plots for {num_samples} samples...\n")

    all_rmsd = []
    all_struct_sim = []
    all_fidelity_scores = []

    for count, idx in enumerate(indices):
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}

        # Get true CID before batch conversion
        true_cid_idx = sample.get("cid_idx", -1)
        if isinstance(true_cid_idx, int):
            true_cid = idx_to_cid.get(true_cid_idx, f'idx_{true_cid_idx}')
        else:
            true_cid = idx_to_cid.get(true_cid_idx.item(), f'idx_{true_cid_idx.item()}')

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
                n_atoms_pred = result.get("n_atoms_pred", None)

            else:
                result = model.generate(batch)
                coords_pred = result["coords"]
                type_logits = result["type_logits"]
                n_atoms_pred = None
            pred_types = type_logits.argmax(dim=-1)

            # V16b mainline: keep visualization on the real default inference path.
            # GNN ensemble remains an optional side experiment and is not applied here.

        # To CPU numpy
        gt_coords = batch["coords"][0].cpu().numpy()
        gt_types = batch["atom_types"][0].cpu().numpy()
        mask = batch["atom_mask"][0].cpu().numpy()
        afm_stack = batch["afm_stack"][0].cpu().numpy()
        pred_coords_np = coords_pred[0].cpu().numpy()
        pred_types_np = pred_types[0].cpu().numpy()

        # CORRECTED CID Retrieval: use pred_mask (not GT mask) for pred side
        # Build pred_mask from n_atoms_pred
        N = pred_coords_np.shape[0]
        if n_atoms_pred is not None:
            n_pred_int = int(n_atoms_pred[0].item())
            pred_mask_np = np.zeros(N, dtype=np.float32)
            pred_mask_np[:min(n_pred_int, N)] = 1.0
        else:
            pred_mask_np = mask.copy()  # fallback to GT mask if no pred count
        if isinstance(true_cid_idx, torch.Tensor):
            exclude_cid = true_cid_idx.item()
        else:
            exclude_cid = true_cid_idx

        top3_results, gt_sim_score, gt_sim_rank = retrieve_by_3d_structure(
            pred_coords_np, pred_types_np, pred_mask_np,
            VAL_DATASET, idx_to_cid, top_k=3, exclude_cid_idx=exclude_cid
        )
        pred_3d_cids = [(cid, s, n_pred, n_db, sim_dict) for cid, s, n_pred, n_db, sim_dict in top3_results]
        pred_3d_scores = [s for cid, s, n_pred, n_db, sim_dict in top3_results]

        # Compute RMSD
        rmsd_val = compute_rmsd(
            coords_pred.cpu(), batch["coords"].cpu(), batch["atom_mask"].cpu()
        ).item()
        all_rmsd.append(rmsd_val)

        # Compute structure similarity (pred side uses pred_mask)
        sim_dict = compute_molecular_similarity_for_retrieval(
            pred_coords_np, pred_types_np, pred_mask_np,
            gt_coords, gt_types, mask,
        )
        all_struct_sim.append(sim_dict["overall"])
        fidelity_metrics = compute_structure_fidelity(
            coords_pred.cpu(), batch["coords"].cpu(),
            pred_types.cpu(), batch["atom_types"].cpu(),
            batch["atom_mask"].cpu(),
            n_atoms_pred=n_atoms_pred.cpu() if n_atoms_pred is not None else None,
            scaffold_labels=batch if "scaffold_n_ring_systems" in batch else None,
        )
        all_fidelity_scores.append(fidelity_metrics["struct_fidelity_score_mean"])

        n_pred_val = int(n_atoms_pred[0].item()) if n_atoms_pred is not None else None

        # Save figure
        save_path = os.path.join(output_dir, f"val_sample_{idx:05d}.png")
        visualize_sample(
            gt_coords, pred_coords_np,
            gt_types, pred_types_np,
            mask, pred_mask_np, afm_stack,
            rmsd_val, idx, save_path,
            fidelity_metrics=fidelity_metrics,
            n_pred=n_pred_val,
            true_cid=true_cid,
            pred_3d_cids=pred_3d_cids,
            pred_3d_scores=pred_3d_scores,
            gt_sim_score=gt_sim_score,
            gt_sim_rank=gt_sim_rank,
        )
        retrieval_info = f" | Top1: {pred_3d_cids[0][0] if pred_3d_cids else 'N/A'}"
        gt_rank_info = f" | GT rank={gt_sim_rank}" if gt_sim_rank is not None else ""
        print(f"  [{count+1}/{num_samples}] Sample {idx}: "
              f"HeavyRMSD={fidelity_metrics['matched_heavy_atom_rmsd_mean']:.4f} "
              f"Fidelity={fidelity_metrics['struct_fidelity_score_mean']:.3f}"
              f"{retrieval_info}{gt_rank_info} -> {save_path}")

    # Summary
    mean_rmsd = np.mean(all_rmsd)
    std_rmsd = np.std(all_rmsd)
    mean_sim = np.mean(all_struct_sim)
    mean_fidelity = np.mean(all_fidelity_scores) if all_fidelity_scores else 0.0
    print(f"\n{'='*60}")
    print(f"Results: {num_samples} validation samples")
    print(f"  RMSD:                {mean_rmsd:.4f} +/- {std_rmsd:.4f}")
    print(f"  Structure Fidelity:  {mean_fidelity:.4f}")
    print(f"  Structure Similarity: {mean_sim:.4f}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
