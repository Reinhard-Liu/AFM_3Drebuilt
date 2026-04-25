"""
Phase 1 评估：化学约束后处理 + 新指标（PMI形状/环保持/条件type_acc）
使用 V10 checkpoint，不需要重训。

Usage:
    python -m src.eval_phase1 --checkpoint experiments/v10/checkpoints/best_diffusion.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.data.dataset import (
    QUAMAFMDataset, create_dataloaders,
    MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES,
)
from src.models.diffusion import SE3EquivariantDenoiser, ConditionalDDPM, compute_shape_descriptors
from src.utils.metrics import (
    compute_rmsd, compute_structure_similarity, compute_formula_similarity,
    compute_atom_count_accuracy, compute_bond_validity,
)


# ============================================================
# 化学约束后处理：用化合价规则修正 type
# ============================================================

# Covalent radii in normalized space (Angstrom / 12.0)
_COVALENT_RADII = {
    0: 0.0258, 1: 0.0642, 2: 0.0608, 3: 0.0550, 4: 0.0533,
    5: 0.0867, 6: 0.0892, 7: 0.0825, 8: 0.0950, 9: 0.1108,
}


def infer_bonds(coords, types, mask, tolerance=0.035):
    """Infer bonds from coordinates using covalent radii."""
    n = int(mask.sum().item())
    coords_np = coords[:n].detach().cpu().numpy()
    types_np = types[:n].detach().cpu().numpy()

    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords_np[i] - coords_np[j])
            r_sum = _COVALENT_RADII.get(int(types_np[i]), 0.07) + _COVALENT_RADII.get(int(types_np[j]), 0.07)
            if d < r_sum + tolerance:
                adj[i, j] = True
                adj[j, i] = True

    valences = adj.sum(axis=1)
    return valences, adj


def chemical_type_correction(coords, type_logits, mask):
    """Use valence rules to correct atom type predictions.

    Rules:
      valence=4 → must be C(sp3)
      valence=1 → most likely H (95%+), or F/Cl/Br
      valence=0 → likely H (isolated) or error
      valence=2 → O most common, or S, N(sp2)
      valence=3 → C(sp2) most common, or N(sp3)

    Only corrects high-confidence cases.
    """
    B = coords.shape[0]
    corrected_types = type_logits.argmax(dim=-1).clone()  # (B, N)

    NAMES = ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I']
    # type indices: H=0, C=1, N=2, O=3, F=4, S=5, P=6, Cl=7, Br=8, I=9

    for b in range(B):
        n = int(mask[b].sum().item())
        if n < 2:
            continue

        pred_types = corrected_types[b, :n]
        valences, adj = infer_bonds(coords[b], pred_types, mask[b])

        for i in range(n):
            v = valences[i]
            orig_type = pred_types[i].item()
            probs = torch.softmax(type_logits[b, i], dim=-1)

            if v == 4:
                # Must be C (sp3) — only C has valence 4
                corrected_types[b, i] = 1  # C
            elif v == 0 or v == 1:
                # Most likely H, or halogen (F/Cl/Br/I)
                # If model predicted C or N, correct to H
                if orig_type in [1, 2, 3]:  # C, N, O with 0-1 bonds is wrong
                    # Check if H probability is reasonable
                    if probs[0] > 0.05:  # H has some probability
                        corrected_types[b, i] = 0  # H
                    elif v == 1 and probs[4] + probs[7] + probs[8] + probs[9] > 0.1:
                        # Halogen
                        halogen_probs = torch.tensor([probs[4], probs[7], probs[8], probs[9]])
                        corrected_types[b, i] = [4, 7, 8, 9][halogen_probs.argmax()]
            elif v == 3:
                # C(sp2) or N(sp3) — if predicted O/H, correct
                if orig_type in [0, 3]:  # H or O with 3 bonds is wrong
                    if probs[1] > probs[2]:
                        corrected_types[b, i] = 1  # C
                    else:
                        corrected_types[b, i] = 2  # N

    return corrected_types


# ============================================================
# PMI 形状相似度
# ============================================================

def compute_pmi_similarity(pred_coords, gt_coords, mask, n_atoms_pred=None):
    """Compute shape similarity using Principal Moments of Inertia ratios."""
    B = pred_coords.shape[0]
    sims = []

    for b in range(B):
        n_gt = int(mask[b].sum().item())
        n_pred = int(n_atoms_pred[b].item()) if n_atoms_pred is not None else n_gt
        if n_gt < 3 or n_pred < 3:
            sims.append(1.0)
            continue

        # GT PMI
        g = gt_coords[b, :n_gt].detach().cpu().numpy().astype(np.float64)
        g = g - g.mean(0)
        S_gt = g.T @ g / n_gt
        eig_gt = np.sort(np.linalg.eigvalsh(S_gt))[::-1]  # descending

        # Pred PMI
        p = pred_coords[b, :n_pred].detach().cpu().numpy().astype(np.float64)
        p = p - p.mean(0)
        S_pred = p.T @ p / n_pred
        eig_pred = np.sort(np.linalg.eigvalsh(S_pred))[::-1]

        # Normalize to ratios: [I1/I1, I2/I1, I3/I1]
        if eig_gt[0] > 1e-10 and eig_pred[0] > 1e-10:
            ratio_gt = eig_gt / eig_gt[0]
            ratio_pred = eig_pred / eig_pred[0]
            # Similarity = 1 - L2(ratios) / sqrt(2)
            dist = np.linalg.norm(ratio_gt - ratio_pred)
            sim = max(0.0, 1.0 - dist / np.sqrt(2))
        else:
            sim = 0.0
        sims.append(sim)

    return {"pmi_similarity": np.array(sims), "pmi_similarity_mean": float(np.mean(sims))}


# ============================================================
# 环保持率
# ============================================================

def detect_rings_from_numpy(coords, types, n, bond_threshold=0.20):
    """Detect 5/6-membered rings from coordinates."""
    if n < 5:
        return []

    dists = cdist(coords[:n], coords[:n])
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < bond_threshold:
                adj[i].append(j)
                adj[j].append(i)

    rings = []
    for start in range(n):
        if len(adj[start]) < 2:
            continue
        for ring_size in [6, 5]:
            for nb in adj[start]:
                path = [start, nb]
                ring = _find_ring(adj, path, start, ring_size)
                if ring is not None:
                    ring_sorted = tuple(sorted(ring))
                    if ring_sorted not in [tuple(sorted(r)) for r in rings]:
                        rings.append(ring)
                        if len(rings) >= 10:
                            return rings
    return rings


def _find_ring(adj, path, target, max_len):
    if len(path) == max_len:
        if target in adj[path[-1]]:
            return list(path)
        return None
    current = path[-1]
    for nb in adj[current]:
        if nb == target and len(path) >= 3:
            continue
        if nb in path[1:]:
            continue
        path.append(nb)
        result = _find_ring(adj, path, target, max_len)
        if result is not None:
            return result
        path.pop()
    return None


def compute_ring_preservation(pred_coords, gt_coords, pred_types, gt_types, mask, n_atoms_pred=None):
    """Compare ring structures between predicted and ground truth molecules."""
    B = pred_coords.shape[0]
    ring_scores = []

    for b in range(B):
        n_gt = int(mask[b].sum().item())
        n_pred = int(n_atoms_pred[b].item()) if n_atoms_pred is not None else n_gt

        g = gt_coords[b].detach().cpu().numpy()
        p = pred_coords[b].detach().cpu().numpy()
        gt_t = gt_types[b].detach().cpu().numpy()
        pt_t = pred_types[b].detach().cpu().numpy()

        gt_rings = detect_rings_from_numpy(g, gt_t, n_gt)
        pred_rings = detect_rings_from_numpy(p, pt_t, n_pred)

        n_gt_rings = len(gt_rings)
        n_pred_rings = len(pred_rings)

        if n_gt_rings == 0:
            # No rings in GT
            ring_scores.append(1.0 if n_pred_rings == 0 else 0.5)
            continue

        # Count matching: same number and size of rings
        gt_sizes = sorted([len(r) for r in gt_rings])
        pred_sizes = sorted([len(r) for r in pred_rings])

        # Ring count similarity
        count_sim = 1.0 - abs(n_gt_rings - n_pred_rings) / max(n_gt_rings, 1)

        # Ring size distribution match
        max_len = max(len(gt_sizes), len(pred_sizes))
        gt_padded = gt_sizes + [0] * (max_len - len(gt_sizes))
        pred_padded = pred_sizes + [0] * (max_len - len(pred_sizes))
        size_match = sum(1 for a, b in zip(gt_padded, pred_padded) if a == b) / max(max_len, 1)

        ring_scores.append(0.5 * count_sim + 0.5 * size_match)

    return {"ring_preservation": np.array(ring_scores), "ring_preservation_mean": float(np.mean(ring_scores))}


# ============================================================
# 条件 Type Accuracy（只在匹配正确的原子上评估）
# ============================================================

def compute_conditional_type_acc(pred_coords, gt_coords, pred_types, gt_types, mask,
                                  n_atoms_pred=None, distance_threshold=0.20):
    """Type accuracy only on correctly matched atoms (distance < threshold)."""
    B = pred_coords.shape[0]
    cond_accs = []

    for b in range(B):
        n_gt = int(mask[b].sum().item())
        n_pred = int(n_atoms_pred[b].item()) if n_atoms_pred is not None else n_gt
        if n_gt == 0 or n_pred == 0:
            cond_accs.append(0.0)
            continue

        p = pred_coords[b, :n_pred].detach().cpu().numpy()
        g = gt_coords[b, :n_gt].detach().cpu().numpy()
        pt = pred_types[b, :n_pred].detach().cpu().numpy()
        gt = gt_types[b, :n_gt].detach().cpu().numpy()

        cost = cdist(p, g)
        row_ind, col_ind = linear_sum_assignment(cost)
        dists = cost[row_ind, col_ind]

        # Only count atoms matched within threshold
        close = dists < distance_threshold
        if close.sum() == 0:
            cond_accs.append(0.0)
            continue

        correct = sum(1 for r, c, ok in zip(row_ind, col_ind, close)
                      if ok and pt[r] == gt[c])
        cond_accs.append(correct / close.sum())

    return {
        "conditional_type_acc": np.array(cond_accs),
        "conditional_type_acc_mean": float(np.mean(cond_accs)),
    }


# ============================================================
# Ring head metrics
# ============================================================

def compute_ring_detection_metrics(predicted_rings, batch):
    """Evaluate ring count / center / type quality from RingDetectionHead outputs."""
    gt_n_rings = batch["n_rings"]
    gt_centers = batch["ring_centers"][:, :, :2]
    gt_types = batch["ring_types"]
    gt_valid = batch["ring_valid"]

    pred_n_rings = predicted_rings["n_rings"]
    pred_centers = predicted_rings["ring_centers"][:, :, :2]
    pred_types = predicted_rings["ring_types"]
    pred_valid = predicted_rings["ring_valid"] > 0.5

    count_exact = []
    count_mae = []
    center_mae = []
    type_acc = []

    B = pred_n_rings.shape[0]
    for b in range(B):
        gt_n = int(gt_n_rings[b].item())
        pred_n = int(pred_n_rings[b].item())
        count_exact.append(1.0 if gt_n == pred_n else 0.0)
        count_mae.append(abs(gt_n - pred_n))

        gt_mask = gt_valid[b] > 0.5
        pred_mask = pred_valid[b]
        gt_xy = gt_centers[b][gt_mask]
        pred_xy = pred_centers[b][pred_mask]
        gt_t = gt_types[b][gt_mask].long()
        pred_t = pred_types[b][pred_mask].long()

        if gt_xy.shape[0] == 0 and pred_xy.shape[0] == 0:
            center_mae.append(0.0)
            type_acc.append(1.0)
            continue
        if gt_xy.shape[0] == 0 or pred_xy.shape[0] == 0:
            center_mae.append(1.0)
            type_acc.append(0.0)
            continue

        cost = torch.cdist(pred_xy, gt_xy)
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        matched_cost = cost[row_ind, col_ind]
        center_mae.append(float(matched_cost.mean().item()))

        pred_match = pred_t[row_ind]
        gt_match = gt_t[col_ind]
        type_acc.append(float((pred_match == gt_match).float().mean().item()))

    return {
        "ring_count_exact": float(np.mean(count_exact)) if count_exact else 0.0,
        "ring_count_mae": float(np.mean(count_mae)) if count_mae else 0.0,
        "ring_center_mae": float(np.mean(center_mae)) if center_mae else 0.0,
        "ring_type_acc": float(np.mean(type_acc)) if type_acc else 0.0,
    }


# ============================================================
# Main evaluation
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--use_chem_correction", action="store_true", default=False,
                        help="Enable chemical type correction post-processing. Off by default for clean eval.")
    parser.add_argument("--no_chem_correction", action="store_true", default=False,
                        help="Explicitly disable chem correction (overrides --use_chem_correction).")
    parser.add_argument("--oracle_ring_info", action="store_true",
                        help="Use GT ring templates during sampling. Default is false for true end-to-end eval.")
    parser.add_argument("--use_gt_count", action="store_true",
                        help="Use GT atom count instead of predicted count.")
    parser.add_argument("--disable_guidance", action="store_true",
                        help="Disable all physics guidance during sampling.")
    parser.add_argument("--disable_ring_snap", action="store_true",
                        help="Disable all ring snapping during sampling.")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"],
                        help="Sampler type: ddim or ddpm.")
    args = parser.parse_args()
    if args.no_chem_correction:
        args.use_chem_correction = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    config["num_cids"] = 0

    from src.train import AFM3DReconModel
    model = AFM3DReconModel(config).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"Loaded: {args.checkpoint} (epoch {state.get('epoch', '?')})")

    _, _, test_loader, _ = create_dataloaders(
        data_root=config["data_root"], param_key=config["param_key"],
        img_size=config["img_size"], min_corrugation=config["min_corrugation"],
        augment_rotation=False, require_ring=config.get("require_ring", False),
        batch_size=config["batch_size"],
        num_workers=0, max_samples=config["max_samples"], val_size=config["val_size"],
    )
    print(f"Test set: {len(test_loader.dataset)} samples")

    all_pmi = []
    all_ring = []
    all_cond_type = []
    all_cond_type_corrected = []
    all_type_match = []
    all_type_match_corrected = []
    all_rmsd = []
    all_bond_valid = []
    all_count_exact = []
    all_count_mae = []
    all_ring_count_exact = []
    all_ring_count_mae = []
    all_ring_center_mae = []
    all_ring_type_acc = []
    count = 0

    all_gt_types_flat = []
    all_pred_types_flat = []

    per_sample_corrugation = []
    per_sample_rmsd = []
    per_sample_type_match = []
    per_sample_cond_type = []
    per_sample_pmi = []
    per_sample_ring = []

    def _to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    pbar = tqdm(test_loader, desc=f"Phase1 Eval (DDIM-{args.ddim_steps})")
    for batch in pbar:
        if count >= args.num_samples:
            break
        batch = _to_device(batch)
        B_cur = batch["afm_stack"].shape[0]

        gen = model.generate(
            batch,
            use_gt_count=args.use_gt_count,
            use_ddim=(args.sampler == "ddim"),
            ddim_steps=args.ddim_steps,
            use_gt_ring_info=args.oracle_ring_info,
            disable_guidance=args.disable_guidance,
            disable_ring_snap=args.disable_ring_snap,
            sampler=args.sampler,
        )
        coords_pred = gen["coords"]
        type_logits = gen["type_logits"]
        pred_types = type_logits.argmax(dim=-1)
        n_pred = gen["n_atoms_pred"]

        with torch.no_grad():
            c_global, c_patches = model.encoder(batch["afm_stack"])
            predicted_rings = model.ring_head.predict(c_global, c_patches)
            ring_det = compute_ring_detection_metrics(predicted_rings, batch)
        all_ring_count_exact.append(ring_det["ring_count_exact"])
        all_ring_count_mae.append(ring_det["ring_count_mae"])
        all_ring_center_mae.append(ring_det["ring_center_mae"])
        all_ring_type_acc.append(ring_det["ring_type_acc"])

        if args.use_chem_correction:
            corrected_types = chemical_type_correction(coords_pred, type_logits, batch["atom_mask"])
        else:
            corrected_types = pred_types

        for i in range(B_cur):
            n_valid = int(batch["atom_mask"][i].sum().item())
            gt_t = batch["atom_types"][i, :n_valid].cpu().numpy()
            pred_t = corrected_types[i, :n_valid].cpu().numpy()
            all_gt_types_flat.extend(gt_t.tolist())
            all_pred_types_flat.extend(pred_t.tolist())

        rmsd = compute_rmsd(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
        all_rmsd.append(rmsd.mean().item())

        pmi = compute_pmi_similarity(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
        all_pmi.append(pmi["pmi_similarity_mean"])

        ring = compute_ring_preservation(coords_pred, batch["coords"], corrected_types, batch["atom_types"],
                                         batch["atom_mask"], n_atoms_pred=n_pred)
        all_ring.append(ring["ring_preservation_mean"])

        cond = compute_conditional_type_acc(coords_pred, batch["coords"], pred_types, batch["atom_types"],
                                            batch["atom_mask"], n_atoms_pred=n_pred)
        all_cond_type.append(cond["conditional_type_acc_mean"])

        cond_corr = compute_conditional_type_acc(coords_pred, batch["coords"], corrected_types, batch["atom_types"],
                                                 batch["atom_mask"], n_atoms_pred=n_pred)
        all_cond_type_corrected.append(cond_corr["conditional_type_acc_mean"])

        struct_sim = compute_structure_similarity(coords_pred, batch["coords"], pred_types, batch["atom_types"],
                                                  batch["atom_mask"], n_atoms_pred=n_pred)
        all_type_match.append(struct_sim["type_match_rate_mean"])

        struct_sim_corr = compute_structure_similarity(coords_pred, batch["coords"], corrected_types, batch["atom_types"],
                                                       batch["atom_mask"], n_atoms_pred=n_pred)
        all_type_match_corrected.append(struct_sim_corr["type_match_rate_mean"])

        bond_valid = compute_bond_validity(coords_pred, corrected_types, batch["atom_mask"])
        all_bond_valid.append(bond_valid.mean().item())

        count_metrics = compute_atom_count_accuracy(n_pred, batch["n_atoms"])
        all_count_exact.append(count_metrics["exact_match"])
        all_count_mae.append(count_metrics["mae"])

        if "corrugation" in batch:
            for i in range(B_cur):
                per_sample_corrugation.append(batch["corrugation"][i].item())
                per_sample_rmsd.append(rmsd[i].item() if rmsd.dim() > 0 else rmsd.item())
            per_sample_pmi.extend(pmi["pmi_similarity"][:B_cur].tolist() if hasattr(pmi["pmi_similarity"], "tolist") else [pmi["pmi_similarity_mean"]] * B_cur)
            per_sample_ring.extend(ring["ring_preservation"][:B_cur].tolist() if hasattr(ring["ring_preservation"], "tolist") else [ring["ring_preservation_mean"]] * B_cur)
            per_sample_cond_type.extend(cond_corr["conditional_type_acc"][:B_cur].tolist() if hasattr(cond_corr["conditional_type_acc"], "tolist") else [cond_corr["conditional_type_acc_mean"]] * B_cur)
            per_sample_type_match.extend(struct_sim_corr["type_match_rate"][:B_cur].tolist() if hasattr(struct_sim_corr["type_match_rate"], "tolist") else [struct_sim_corr["type_match_rate_mean"]] * B_cur)

        count += B_cur
        pbar.set_postfix(rmsd=f"{np.mean(all_rmsd):.3f}", type_acc=f"{np.mean(all_cond_type_corrected):.3f}")

    print() 
    print("=" * 70)
    print("Phase 1 Evaluation Results")
    print("=" * 70)
    print(f"Samples evaluated:              {count}")
    print(f"RMSD:                          {np.mean(all_rmsd):.4f}")
    print(f"Bond Validity:                 {np.mean(all_bond_valid):.4f}")
    print(f"Count Exact Accuracy:          {np.mean(all_count_exact):.4f}")
    print(f"Count MAE:                     {np.mean(all_count_mae):.4f}")
    print(f"Ring Preservation:             {np.mean(all_ring):.4f}")
    print(f"Ring Count Exact:              {np.mean(all_ring_count_exact):.4f}")
    print(f"Ring Count MAE:                {np.mean(all_ring_count_mae):.4f}")
    print(f"Ring Center MAE:               {np.mean(all_ring_center_mae):.4f}")
    print(f"Ring Type Accuracy:            {np.mean(all_ring_type_acc):.4f}")
    print()
    print("--- Type Accuracy ---")
    print(f"type_match (raw):              {np.mean(all_type_match):.4f}")
    print(f"type_match (corrected):        {np.mean(all_type_match_corrected):.4f}")
    print(f"conditional_type_acc (raw):    {np.mean(all_cond_type):.4f}")
    print(f"conditional_type_acc (corr):   {np.mean(all_cond_type_corrected):.4f}")
    print()
    print("--- Auxiliary ---")
    print(f"PMI shape similarity:          {np.mean(all_pmi):.4f}")
    print("=" * 70)

    if all_gt_types_flat:
        gt_arr = np.array(all_gt_types_flat, dtype=int)
        pred_arr = np.array(all_pred_types_flat, dtype=int)
        n_classes = len(ATOM_TYPES)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for g, p in zip(gt_arr, pred_arr):
            if 0 <= g < n_classes and 0 <= p < n_classes:
                cm[g, p] += 1
        print()
        print("=" * 70)
        print("Type Confusion Matrix (rows=GT, cols=Pred)")
        print("=" * 70)
        header = "     " + "".join(f"{ATOM_TYPES[j]:>6s}" for j in range(n_classes))
        print(header)
        for i in range(n_classes):
            row_total = cm[i].sum()
            acc = 0.0 if row_total == 0 else cm[i, i] / row_total
            row_str = f"{ATOM_TYPES[i]:>4s} " + "".join(f"{cm[i,j]:>6d}" for j in range(n_classes))
            row_str += f"  | acc={acc:.1%} (n={row_total})"
            print(row_str)
        total_correct = sum(cm[i, i] for i in range(n_classes))
        total = cm.sum()
        print(f"Overall accuracy: {total_correct}/{total} = {total_correct/max(total,1):.1%}")
        print("=" * 70)

    if per_sample_corrugation:
        corr_arr = np.array(per_sample_corrugation)
        n_total = len(corr_arr)
        t1 = np.percentile(corr_arr, 33)
        t2 = np.percentile(corr_arr, 67)
        groups = {
            f"low (<{t1:.2f}A, P0-33)": corr_arr < t1,
            f"mid ({t1:.2f}-{t2:.2f}A, P33-67)": (corr_arr >= t1) & (corr_arr < t2),
            f"high (>{t2:.2f}A, P67-100)": corr_arr >= t2,
        }
        print()
        print("=" * 70)
        print("Corrugation Group Analysis")
        print("=" * 70)
        rmsd_arr = np.array(per_sample_rmsd[:n_total])
        type_arr = np.array(per_sample_type_match[:n_total])
        cond_arr = np.array(per_sample_cond_type[:n_total])
        pmi_arr = np.array(per_sample_pmi[:n_total])
        ring_arr = np.array(per_sample_ring[:n_total])
        for name, gmask in groups.items():
            n = gmask.sum()
            if n == 0:
                print(f"  {name}: no samples")
                continue
            print(f"  {name} (n={n}):")
            print(f"    RMSD:           {rmsd_arr[gmask].mean():.4f}")
            print(f"    Type Match:     {type_arr[gmask].mean():.4f}")
            print(f"    Cond Type Acc:  {cond_arr[gmask].mean():.4f}")
            print(f"    PMI:            {pmi_arr[gmask].mean():.4f}")
            print(f"    Ring:           {ring_arr[gmask].mean():.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
