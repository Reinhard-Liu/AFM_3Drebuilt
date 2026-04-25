"""
Evaluation metrics:
- RMSD: Root Mean Square Deviation for 3D coordinate reconstruction
- Bottom Atom Recall: recall of occluded (bottom-layer) atoms
- Atom Count Accuracy: exact match and MAE for predicted atom count
- Bond Validity: fraction of bonds with valid lengths
- Ring Preservation: fraction of GT rings correctly reconstructed
- CID Accuracy: Top-1 and Top-5 retrieval accuracy
- Bottom Atom RMSD: RMSD specifically for bottom atoms
- Composite Score: weighted combination of all metrics
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.dataset import NUM_ATOM_TYPES, ATOM_TYPES
from src.models.constraints import (
    IDEAL_BOND_LENGTHS, MAX_BOND_DIST,
    BOND_VALIDITY_TOLERANCE, NUM_ATOM_TYPES as _NUM_ATOM_TYPES,
)
from src.models.ring_detection import (
    build_molecular_graph, find_rings, compute_ring_system_scaffold_labels,
)


def compute_rmsd(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    n_atoms_pred: torch.Tensor = None,
) -> torch.Tensor:
    """Compute per-sample RMSD between predicted and ground-truth coordinates.

    Uses Hungarian matching on the rectangular cost matrix (n_pred x n_gt).

    V7 fix: uses min(n_pred, n_gt) to avoid matching zero-padded coords.

    Args:
        pred_coords: (B, N, 3) predicted coordinates
        gt_coords: (B, N, 3) ground truth coordinates
        mask: (B, N) atom mask
        n_atoms_pred: (B,) predicted atom counts (if None, uses n_gt)

    Returns:
        rmsd: (B,) RMSD per sample in coordinate units
    """
    B = pred_coords.shape[0]
    rmsd_list = []

    for b in range(B):
        n_gt = int(mask[b].bool().sum().item())
        if n_gt == 0:
            rmsd_list.append(0.0)
            continue

        n_pred = int(n_atoms_pred[b].item()) if n_atoms_pred is not None else n_gt
        n_pred = max(1, min(n_pred, pred_coords.shape[1]))

        p = pred_coords[b, :n_pred].detach().cpu().numpy()
        g = gt_coords[b, :n_gt].detach().cpu().numpy()

        # Rectangular cost matrix (n_pred x n_gt)
        diff = p[:, None, :] - g[None, :, :]  # (n_pred, n_gt, 3)
        cost = np.sqrt((diff ** 2).sum(axis=-1))

        # Hungarian matching on rectangular matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        n_matched = len(row_ind)

        matched_diff = p[row_ind] - g[col_ind]
        msd = (matched_diff ** 2).sum() / max(n_matched, 1)
        rmsd_list.append(float(np.sqrt(msd)))

    return torch.tensor(rmsd_list)


def _hungarian_match_numpy(pred_coords: np.ndarray, gt_coords: np.ndarray):
    """Return Hungarian matching indices for a rectangular distance matrix."""
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.zeros((len(pred_coords), len(gt_coords)))

    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=-1))
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind, cost


def _safe_mean(values, default=0.0):
    if len(values) == 0:
        return float(default)
    return float(np.mean(values))


def _safe_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall < 1e-8:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _compute_reference_distance(coords: np.ndarray, types: np.ndarray) -> float:
    """Estimate a stable per-molecule reference length for soft hit thresholds."""
    if len(coords) < 2:
        return 0.24

    heavy_idx = np.where(types != 0)[0]
    use_idx = heavy_idx if len(heavy_idx) >= 2 else np.arange(len(coords))
    if len(use_idx) < 2:
        return 0.24

    use_coords = coords[use_idx]
    diff = use_coords[:, None, :] - use_coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    nearest = dists.min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 0.24
    return float(np.median(nearest))


def _get_soft_hit_thresholds(coords: np.ndarray, types: np.ndarray) -> tuple[float, float, float, float]:
    """Return (reference, tight, medium, loose) coordinate tolerances."""
    ref = _compute_reference_distance(coords, types)
    tight = float(np.clip(0.45 * ref, 0.08, 0.16))
    medium = float(np.clip(0.70 * ref, 0.12, 0.24))
    loose = float(np.clip(1.00 * ref, 0.18, 0.32))
    return ref, tight, medium, loose


def _coverage_rate(gt_points: np.ndarray, pred_points: np.ndarray, threshold: float) -> float:
    """Fraction of GT points covered by a prediction within a tolerance."""
    if len(gt_points) == 0:
        return 1.0 if len(pred_points) == 0 else 0.0
    if len(pred_points) == 0:
        return 0.0

    hits = 0
    for pt in gt_points:
        dists = np.sqrt(((pred_points - pt[None, :]) ** 2).sum(axis=-1))
        if float(dists.min()) <= threshold:
            hits += 1
    return hits / max(len(gt_points), 1)


def _macro_type_f1(
    pred_match_types: np.ndarray,
    gt_match_types: np.ndarray,
    pred_all_types: np.ndarray,
    gt_all_types: np.ndarray,
) -> float:
    """Macro F1 across atom classes present in GT or prediction."""
    pred_all_types = np.asarray(pred_all_types, dtype=np.int64)
    gt_all_types = np.asarray(gt_all_types, dtype=np.int64)
    pred_match_types = np.asarray(pred_match_types, dtype=np.int64)
    gt_match_types = np.asarray(gt_match_types, dtype=np.int64)

    classes = sorted(
        c for c in set(pred_all_types.tolist()) | set(gt_all_types.tolist())
        if c >= 0
    )
    if not classes:
        return 1.0

    f1_list = []
    for cls in classes:
        tp = int(((pred_match_types == cls) & (gt_match_types == cls)).sum())
        fp = int((pred_all_types == cls).sum() - tp)
        fn = int((gt_all_types == cls).sum() - tp)
        _, _, f1 = _safe_f1(tp, fp, fn)
        f1_list.append(f1)
    return _safe_mean(f1_list, default=0.0)


def _infer_bond_edges(coords: np.ndarray, types: np.ndarray) -> set[tuple[int, int]]:
    """Infer undirected bond edges with the same distance table as bond validity."""
    n = len(coords)
    edges = set()
    ideal = IDEAL_BOND_LENGTHS.numpy()
    max_dist_table = MAX_BOND_DIST.numpy()

    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = int(types[i]), int(types[j])
            if ti < 0 or tj < 0 or ti >= _NUM_ATOM_TYPES or tj >= _NUM_ATOM_TYPES:
                continue
            if ideal[ti, tj] < 1e-8:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < float(max_dist_table[ti, tj]):
                edges.add((i, j))
    return edges


def _detect_ring_sizes(coords: np.ndarray, types: np.ndarray) -> list[int]:
    """Detect 5/6-membered ring sizes via the repo ring detector."""
    if len(coords) < 5:
        return []
    elements = []
    for t in types:
        idx = int(t)
        if 0 <= idx < len(ATOM_TYPES):
            elements.append(ATOM_TYPES[idx])
        else:
            elements.append("C")
    try:
        adj = build_molecular_graph(coords, elements)
        # Skip exhaustive cycle search on obviously non-physical dense graphs.
        # In bad predictions this graph can become near-clique-like, and the
        # DFS-based ring enumeration explodes even though such samples should
        # not receive ring credit anyway.
        edge_count = sum(len(v) for v in adj.values()) // 2
        max_degree = max((len(v) for v in adj.values()), default=0)
        if edge_count > max(3 * len(coords), 24) or max_degree > 4:
            return []
        rings = find_rings(adj, max_size=6)
    except Exception:
        return []
    return sorted(len(r) for r in rings)


def _types_to_elements(types: np.ndarray) -> list[str]:
    elements = []
    for t in types:
        idx = int(t)
        if 0 <= idx < len(ATOM_TYPES):
            elements.append(ATOM_TYPES[idx])
        else:
            elements.append("C")
    return elements


def _extract_label_edge_set(edge_array, n_edges: int) -> set[tuple[int, int]]:
    edges = set()
    limit = min(int(n_edges), len(edge_array))
    for i in range(limit):
        a = int(edge_array[i][0])
        b = int(edge_array[i][1])
        if a < 0 or b < 0:
            continue
        edges.add(tuple(sorted((a, b))))
    return edges


def compute_structure_fidelity(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    pred_types: torch.Tensor,
    gt_types: torch.Tensor,
    mask: torch.Tensor,
    n_atoms_pred: torch.Tensor | None = None,
    pred_mask: torch.Tensor | None = None,
    bond_validity_pred: torch.Tensor | None = None,
    scaffold_labels: dict | None = None,
) -> dict:
    """Compute V18 Phase-1 atom-level structure fidelity metrics.

    This metric bundle contains two layers:
    1. Early-direction soft metrics:
       - tolerance-based atom hit rates
       - ring-site coverage
       - soft attachment-site accuracy
       - macro atom-type F1
    2. Final strict metrics:
       - exact atom count
       - strict ring/connectivity proxies
       - strict structure fidelity pass criteria
    """
    B = pred_coords.shape[0]

    atom_count_exact_list = []
    atom_count_abs_error_list = []
    atom_count_score_list = []
    matched_atom_rmsd_list = []
    matched_heavy_atom_rmsd_list = []
    matched_atom_mae_list = []
    atom_position_score_list = []
    heavy_atom_hit_rate_tight_list = []
    heavy_atom_hit_rate_medium_list = []
    heavy_atom_hit_rate_loose_list = []
    typed_heavy_atom_hit_rate_tight_list = []
    typed_heavy_atom_hit_rate_medium_list = []
    typed_heavy_atom_hit_rate_loose_list = []
    heteroatom_hit_rate_medium_list = []
    macro_type_f1_list = []
    ch_collapse_rate_list = []
    atom_type_acc_list = []
    hetero_precision_list = []
    hetero_recall_list = []
    hetero_f1_list = []
    atom_semantic_score_list = []
    ring_count_exact_list = []
    ring_complete_rate_list = []
    ring_size_acc_list = []
    ring_integrity_score_list = []
    scaffold_local_edge_recall_list = []
    scaffold_local_edge_f1_list = []
    ring_site_coverage_medium_list = []
    attachment_edge_f1_list = []
    soft_attachment_site_acc_medium_list = []
    scaffold_relation_f1_list = []
    connectivity_score_list = []
    local_chem_score_list = []
    global_shape_aux_score_list = []
    atom_match_coverage_list = []
    soft_recon_score_list = []
    struct_fidelity_score_list = []
    struct_fidelity_pass_list = []

    if pred_mask is None:
        N = pred_coords.shape[1]
        if n_atoms_pred is None:
            pred_mask = mask.float()
        else:
            idx = torch.arange(N, device=pred_coords.device, dtype=torch.float32).unsqueeze(0)
            pred_mask = (idx < n_atoms_pred.float().unsqueeze(1)).float()

    if bond_validity_pred is None:
        bond_validity_pred = compute_bond_validity(pred_coords, pred_types, pred_mask)

    for b in range(B):
        n_gt = int(mask[b].bool().sum().item())
        if n_atoms_pred is not None:
            n_pred = int(n_atoms_pred[b].item())
        else:
            n_pred = int(pred_mask[b].bool().sum().item())
        n_pred = max(0, min(n_pred, pred_coords.shape[1]))

        p = pred_coords[b, :n_pred].detach().cpu().numpy()
        g = gt_coords[b, :n_gt].detach().cpu().numpy()
        pt = pred_types[b, :n_pred].detach().cpu().numpy()
        gt_t = gt_types[b, :n_gt].detach().cpu().numpy()
        _, tol_tight, tol_medium, tol_loose = _get_soft_hit_thresholds(g, gt_t)

        atom_count_exact = 1.0 if n_pred == n_gt else 0.0
        atom_count_abs_error = float(abs(n_pred - n_gt))
        count_norm = max(max(n_pred, n_gt), 1)
        atom_count_score = 0.7 * atom_count_exact + 0.3 * max(0.0, 1.0 - atom_count_abs_error / count_norm)

        row_ind, col_ind, cost = _hungarian_match_numpy(p, g)
        n_matched = len(row_ind)
        atom_match_coverage = n_matched / max(n_gt, 1)

        if n_matched > 0:
            matched_dists = cost[row_ind, col_ind]
            matched_atom_mae = float(matched_dists.mean())
            matched_atom_rmsd = float(np.sqrt(np.mean(matched_dists ** 2)))

            pred_match_types = pt[row_ind]
            gt_match_types = gt_t[col_ind]
            pred_heavy = pred_match_types != 0
            gt_heavy = gt_match_types != 0
            heavy_mask = pred_heavy | gt_heavy
            if heavy_mask.any():
                heavy_dists = matched_dists[heavy_mask]
                matched_heavy_atom_rmsd = float(np.sqrt(np.mean(heavy_dists ** 2)))
            else:
                matched_heavy_atom_rmsd = matched_atom_rmsd

            atom_type_acc = float((pred_match_types == gt_match_types).sum() / max(n_matched, 1))
        else:
            matched_atom_mae = 10.0
            matched_atom_rmsd = 10.0
            matched_heavy_atom_rmsd = 10.0
            pred_match_types = np.array([], dtype=np.int64)
            gt_match_types = np.array([], dtype=np.int64)
            atom_type_acc = 0.0

        gt_heavy_total = int((gt_t != 0).sum())
        gt_hetero_total = int(np.isin(gt_t, [2, 3, 4, 5, 6, 7, 8, 9]).sum())
        if n_matched > 0:
            gt_heavy_match = gt_match_types != 0
            typed_match = pred_match_types == gt_match_types
            heavy_atom_hit_rate_tight = float(((matched_dists <= tol_tight) & gt_heavy_match).sum() / max(gt_heavy_total, 1))
            heavy_atom_hit_rate_medium = float(((matched_dists <= tol_medium) & gt_heavy_match).sum() / max(gt_heavy_total, 1))
            heavy_atom_hit_rate_loose = float(((matched_dists <= tol_loose) & gt_heavy_match).sum() / max(gt_heavy_total, 1))
            typed_heavy_atom_hit_rate_tight = float(((matched_dists <= tol_tight) & gt_heavy_match & typed_match).sum() / max(gt_heavy_total, 1))
            typed_heavy_atom_hit_rate_medium = float(((matched_dists <= tol_medium) & gt_heavy_match & typed_match).sum() / max(gt_heavy_total, 1))
            typed_heavy_atom_hit_rate_loose = float(((matched_dists <= tol_loose) & gt_heavy_match & typed_match).sum() / max(gt_heavy_total, 1))
            gt_hetero_match = np.isin(gt_match_types, [2, 3, 4, 5, 6, 7, 8, 9])
            heteroatom_hit_rate_medium = (
                float(((matched_dists <= tol_medium) & gt_hetero_match & typed_match).sum() / max(gt_hetero_total, 1))
                if gt_hetero_total > 0 else
                (1.0 if int(np.isin(pt, [2, 3, 4, 5, 6, 7, 8, 9]).sum()) == 0 else 0.0)
            )
        else:
            heavy_atom_hit_rate_tight = 0.0 if gt_heavy_total > 0 else 1.0
            heavy_atom_hit_rate_medium = 0.0 if gt_heavy_total > 0 else 1.0
            heavy_atom_hit_rate_loose = 0.0 if gt_heavy_total > 0 else 1.0
            typed_heavy_atom_hit_rate_tight = 0.0 if gt_heavy_total > 0 else 1.0
            typed_heavy_atom_hit_rate_medium = 0.0 if gt_heavy_total > 0 else 1.0
            typed_heavy_atom_hit_rate_loose = 0.0 if gt_heavy_total > 0 else 1.0
            heteroatom_hit_rate_medium = 0.0 if gt_hetero_total > 0 else 1.0

        macro_type_f1 = _macro_type_f1(pred_match_types, gt_match_types, pt, gt_t)
        gt_ch_fraction = float(np.isin(gt_t, [0, 1]).mean()) if len(gt_t) > 0 else 1.0
        pred_ch_fraction = float(np.isin(pt, [0, 1]).mean()) if len(pt) > 0 else 1.0
        if gt_ch_fraction >= 0.999:
            ch_collapse_rate = 0.0
        else:
            ch_collapse_rate = float(np.clip((pred_ch_fraction - gt_ch_fraction) / max(1.0 - gt_ch_fraction, 1e-6), 0.0, 1.0))

        atom_position_score = max(0.0, 1.0 - matched_heavy_atom_rmsd / 2.0)
        global_shape_aux_score = max(0.0, 1.0 - matched_atom_rmsd / 2.0)

        pred_is_hetero = np.array([int(t not in (0, 1)) for t in pt], dtype=np.int64)
        gt_is_hetero = np.array([int(t not in (0, 1)) for t in gt_t], dtype=np.int64)
        pred_match_hetero = np.array([int(t not in (0, 1)) for t in pred_match_types], dtype=np.int64)
        gt_match_hetero = np.array([int(t not in (0, 1)) for t in gt_match_types], dtype=np.int64)
        tp_hetero = int(((pred_match_hetero == 1) & (gt_match_hetero == 1)).sum())
        fp_hetero = int(pred_is_hetero.sum() - tp_hetero)
        fn_hetero = int(gt_is_hetero.sum() - tp_hetero)
        hetero_precision, hetero_recall, hetero_f1 = _safe_f1(tp_hetero, fp_hetero, fn_hetero)
        atom_semantic_score = 0.65 * atom_type_acc + 0.35 * hetero_f1

        pred_ring_sizes = _detect_ring_sizes(p, pt)
        gt_ring_sizes = _detect_ring_sizes(g, gt_t)
        gt_ring_set = list(gt_ring_sizes)
        pred_ring_pool = list(pred_ring_sizes)
        matched_ring_sizes = 0
        for size in gt_ring_set:
            if size in pred_ring_pool:
                pred_ring_pool.remove(size)
                matched_ring_sizes += 1
        approx_ring_complete_rate = matched_ring_sizes / max(len(gt_ring_set), 1) if len(gt_ring_set) > 0 else (1.0 if len(pred_ring_sizes) == 0 else 0.0)
        ring_size_acc = approx_ring_complete_rate

        pred_elements = _types_to_elements(pt)
        pred_scaffold = compute_ring_system_scaffold_labels(p, pred_elements, max_atoms=max(len(p), 1))

        if scaffold_labels is not None:
            gt_n_ring_systems = int(scaffold_labels["scaffold_n_ring_systems"][b].item())
            gt_local_edges = _extract_label_edge_set(
                scaffold_labels["scaffold_local_edges"][b].detach().cpu().numpy(),
                int(scaffold_labels["scaffold_n_local_edges"][b].item()),
            )
            gt_sidechain_edges = _extract_label_edge_set(
                scaffold_labels["scaffold_sidechain_edges"][b].detach().cpu().numpy(),
                int(scaffold_labels["scaffold_n_sidechain_edges"][b].item()),
            )
            gt_scaffold_flags = scaffold_labels["scaffold_atom_is_scaffold"][b, :n_gt].detach().cpu().numpy() > 0.5
            gt_attachment_anchor_flags = scaffold_labels["scaffold_atom_is_attachment_anchor"][b, :n_gt].detach().cpu().numpy() > 0.5
        else:
            gt_elements = _types_to_elements(gt_t)
            gt_scaffold = compute_ring_system_scaffold_labels(g, gt_elements, max_atoms=max(len(g), 1))
            gt_n_ring_systems = int(gt_scaffold["scaffold_n_ring_systems"])
            gt_local_edges = _extract_label_edge_set(
                gt_scaffold["scaffold_local_edges"],
                int(gt_scaffold["scaffold_n_local_edges"]),
            )
            gt_sidechain_edges = _extract_label_edge_set(
                gt_scaffold["scaffold_sidechain_edges"],
                int(gt_scaffold["scaffold_n_sidechain_edges"]),
            )
            gt_scaffold_flags = gt_scaffold["scaffold_atom_is_scaffold"][:n_gt] > 0.5
            gt_attachment_anchor_flags = gt_scaffold["scaffold_atom_is_attachment_anchor"][:n_gt] > 0.5

        pred_n_ring_systems = int(pred_scaffold["scaffold_n_ring_systems"])
        ring_count_exact = 1.0 if pred_n_ring_systems == gt_n_ring_systems else 0.0

        pred_local_edges = _extract_label_edge_set(
            pred_scaffold["scaffold_local_edges"],
            int(pred_scaffold["scaffold_n_local_edges"]),
        )
        pred_sidechain_edges = _extract_label_edge_set(
            pred_scaffold["scaffold_sidechain_edges"],
            int(pred_scaffold["scaffold_n_sidechain_edges"]),
        )
        pred_scaffold_flags = pred_scaffold["scaffold_atom_is_scaffold"][:len(p)] > 0.5
        pred_attachment_anchor_flags = pred_scaffold["scaffold_atom_is_attachment_anchor"][:len(p)] > 0.5

        pred_to_gt = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
        mapped_pred_local_edges = set()
        for i, j in pred_local_edges:
            if i in pred_to_gt and j in pred_to_gt:
                mapped_pred_local_edges.add(tuple(sorted((pred_to_gt[i], pred_to_gt[j]))))
        mapped_pred_sidechain_edges = set()
        for i, j in pred_sidechain_edges:
            if i in pred_to_gt and j in pred_to_gt:
                mapped_pred_sidechain_edges.add(tuple(sorted((pred_to_gt[i], pred_to_gt[j]))))

        local_tp = len(mapped_pred_local_edges & gt_local_edges)
        local_fp = len(mapped_pred_local_edges - gt_local_edges)
        local_fn = len(gt_local_edges - mapped_pred_local_edges)
        _, scaffold_local_edge_recall, scaffold_local_edge_f1 = _safe_f1(local_tp, local_fp, local_fn)
        ring_complete_rate = (
            0.75 * scaffold_local_edge_recall + 0.25 * approx_ring_complete_rate
        )
        gt_ring_site_coords = g[gt_scaffold_flags] if gt_scaffold_flags.any() else np.zeros((0, 3), dtype=np.float32)
        pred_ring_site_coords = p[pred_scaffold_flags] if pred_scaffold_flags.any() else np.zeros((0, 3), dtype=np.float32)
        ring_site_coverage_medium = _coverage_rate(gt_ring_site_coords, pred_ring_site_coords, tol_medium)
        ring_integrity_score = (
            0.30 * ring_count_exact
            + 0.40 * ring_complete_rate
            + 0.30 * scaffold_local_edge_f1
        )

        pred_edges = _infer_bond_edges(p, pt)
        gt_edges = _infer_bond_edges(g, gt_t)
        mapped_pred_edges = set()
        for i, j in pred_edges:
            if i in pred_to_gt and j in pred_to_gt:
                a, c = pred_to_gt[i], pred_to_gt[j]
                mapped_pred_edges.add(tuple(sorted((a, c))))
        tp_edges = len(mapped_pred_edges & gt_edges)
        fp_edges = len(mapped_pred_edges - gt_edges)
        fn_edges = len(gt_edges - mapped_pred_edges)
        _, _, edge_f1 = _safe_f1(tp_edges, fp_edges, fn_edges)

        gt_attachment_edges = set()
        for u, v in gt_sidechain_edges:
            u_scaffold = bool(gt_scaffold_flags[u]) if 0 <= u < len(gt_scaffold_flags) else False
            v_scaffold = bool(gt_scaffold_flags[v]) if 0 <= v < len(gt_scaffold_flags) else False
            if u_scaffold ^ v_scaffold:
                gt_attachment_edges.add(tuple(sorted((u, v))))
        pred_attachment_edges = set()
        for u, v in pred_sidechain_edges:
            u_scaffold = bool(pred_scaffold_flags[u]) if 0 <= u < len(pred_scaffold_flags) else False
            v_scaffold = bool(pred_scaffold_flags[v]) if 0 <= v < len(pred_scaffold_flags) else False
            if u_scaffold ^ v_scaffold:
                pred_attachment_edges.add((u, v))

        mapped_pred_attachment_edges = set()
        for i, j in pred_attachment_edges:
            if i in pred_to_gt and j in pred_to_gt:
                mapped_pred_attachment_edges.add(tuple(sorted((pred_to_gt[i], pred_to_gt[j]))))

        side_tp = len(mapped_pred_attachment_edges & gt_attachment_edges)
        side_fp = len(mapped_pred_attachment_edges - gt_attachment_edges)
        side_fn = len(gt_attachment_edges - mapped_pred_attachment_edges)
        _, _, attachment_edge_f1 = _safe_f1(side_tp, side_fp, side_fn)
        gt_anchor_coords = g[gt_attachment_anchor_flags] if gt_attachment_anchor_flags.any() else np.zeros((0, 3), dtype=np.float32)
        pred_anchor_coords = p[pred_attachment_anchor_flags] if pred_attachment_anchor_flags.any() else np.zeros((0, 3), dtype=np.float32)
        soft_attachment_site_acc_medium = _coverage_rate(gt_anchor_coords, pred_anchor_coords, tol_medium)
        scaffold_relation_f1 = scaffold_local_edge_f1
        connectivity_score = 0.6 * attachment_edge_f1 + 0.4 * scaffold_local_edge_f1

        bond_valid = float(bond_validity_pred[b].item()) if torch.is_tensor(bond_validity_pred) else float(bond_validity_pred[b])
        local_chem_score = bond_valid
        soft_recon_score = (
            0.25 * typed_heavy_atom_hit_rate_medium
            + 0.20 * heavy_atom_hit_rate_medium
            + 0.15 * heteroatom_hit_rate_medium
            + 0.10 * macro_type_f1
            + 0.10 * ring_site_coverage_medium
            + 0.10 * soft_attachment_site_acc_medium
            + 0.05 * atom_count_score
            + 0.05 * local_chem_score
        )

        struct_fidelity_score = (
            0.15 * atom_count_score
            + 0.25 * atom_position_score
            + 0.20 * atom_semantic_score
            + 0.15 * ring_integrity_score
            + 0.10 * connectivity_score
            + 0.10 * local_chem_score
            + 0.05 * global_shape_aux_score
        )

        struct_fidelity_pass = float(
            (atom_count_exact >= 1.0)
            and (matched_heavy_atom_rmsd <= 0.30)
            and (atom_type_acc >= 0.70)
            and (hetero_f1 >= 0.50)
            and (ring_complete_rate >= 0.70)
            and (attachment_edge_f1 >= 0.55)
            and (bond_valid >= 0.60)
        )

        atom_count_exact_list.append(atom_count_exact)
        atom_count_abs_error_list.append(atom_count_abs_error)
        atom_count_score_list.append(atom_count_score)
        matched_atom_rmsd_list.append(matched_atom_rmsd)
        matched_heavy_atom_rmsd_list.append(matched_heavy_atom_rmsd)
        matched_atom_mae_list.append(matched_atom_mae)
        atom_position_score_list.append(atom_position_score)
        heavy_atom_hit_rate_tight_list.append(heavy_atom_hit_rate_tight)
        heavy_atom_hit_rate_medium_list.append(heavy_atom_hit_rate_medium)
        heavy_atom_hit_rate_loose_list.append(heavy_atom_hit_rate_loose)
        typed_heavy_atom_hit_rate_tight_list.append(typed_heavy_atom_hit_rate_tight)
        typed_heavy_atom_hit_rate_medium_list.append(typed_heavy_atom_hit_rate_medium)
        typed_heavy_atom_hit_rate_loose_list.append(typed_heavy_atom_hit_rate_loose)
        heteroatom_hit_rate_medium_list.append(heteroatom_hit_rate_medium)
        macro_type_f1_list.append(macro_type_f1)
        ch_collapse_rate_list.append(ch_collapse_rate)
        atom_type_acc_list.append(atom_type_acc)
        hetero_precision_list.append(hetero_precision)
        hetero_recall_list.append(hetero_recall)
        hetero_f1_list.append(hetero_f1)
        atom_semantic_score_list.append(atom_semantic_score)
        ring_count_exact_list.append(ring_count_exact)
        ring_complete_rate_list.append(ring_complete_rate)
        ring_size_acc_list.append(ring_size_acc)
        ring_integrity_score_list.append(ring_integrity_score)
        scaffold_local_edge_recall_list.append(scaffold_local_edge_recall)
        scaffold_local_edge_f1_list.append(scaffold_local_edge_f1)
        ring_site_coverage_medium_list.append(ring_site_coverage_medium)
        attachment_edge_f1_list.append(attachment_edge_f1)
        soft_attachment_site_acc_medium_list.append(soft_attachment_site_acc_medium)
        scaffold_relation_f1_list.append(scaffold_relation_f1)
        connectivity_score_list.append(connectivity_score)
        local_chem_score_list.append(local_chem_score)
        global_shape_aux_score_list.append(global_shape_aux_score)
        atom_match_coverage_list.append(atom_match_coverage)
        soft_recon_score_list.append(soft_recon_score)
        struct_fidelity_score_list.append(struct_fidelity_score)
        struct_fidelity_pass_list.append(struct_fidelity_pass)

    return {
        "atom_count_exact": torch.tensor(atom_count_exact_list),
        "atom_count_abs_error": torch.tensor(atom_count_abs_error_list),
        "atom_count_score": torch.tensor(atom_count_score_list),
        "matched_atom_rmsd": torch.tensor(matched_atom_rmsd_list),
        "matched_heavy_atom_rmsd": torch.tensor(matched_heavy_atom_rmsd_list),
        "matched_atom_mae": torch.tensor(matched_atom_mae_list),
        "atom_position_score": torch.tensor(atom_position_score_list),
        "heavy_atom_hit_rate_tight": torch.tensor(heavy_atom_hit_rate_tight_list),
        "heavy_atom_hit_rate_medium": torch.tensor(heavy_atom_hit_rate_medium_list),
        "heavy_atom_hit_rate_loose": torch.tensor(heavy_atom_hit_rate_loose_list),
        "typed_heavy_atom_hit_rate_tight": torch.tensor(typed_heavy_atom_hit_rate_tight_list),
        "typed_heavy_atom_hit_rate_medium": torch.tensor(typed_heavy_atom_hit_rate_medium_list),
        "typed_heavy_atom_hit_rate_loose": torch.tensor(typed_heavy_atom_hit_rate_loose_list),
        "heteroatom_hit_rate_medium": torch.tensor(heteroatom_hit_rate_medium_list),
        "macro_type_f1": torch.tensor(macro_type_f1_list),
        "ch_collapse_rate": torch.tensor(ch_collapse_rate_list),
        "atom_type_acc": torch.tensor(atom_type_acc_list),
        "heteroatom_precision": torch.tensor(hetero_precision_list),
        "heteroatom_recall": torch.tensor(hetero_recall_list),
        "heteroatom_f1": torch.tensor(hetero_f1_list),
        "atom_semantic_score": torch.tensor(atom_semantic_score_list),
        "ring_count_exact": torch.tensor(ring_count_exact_list),
        "ring_complete_rate": torch.tensor(ring_complete_rate_list),
        "ring_size_acc": torch.tensor(ring_size_acc_list),
        "ring_integrity_score": torch.tensor(ring_integrity_score_list),
        "scaffold_local_edge_recall": torch.tensor(scaffold_local_edge_recall_list),
        "scaffold_local_edge_f1": torch.tensor(scaffold_local_edge_f1_list),
        "ring_site_coverage_medium": torch.tensor(ring_site_coverage_medium_list),
        "attachment_edge_f1": torch.tensor(attachment_edge_f1_list),
        "soft_attachment_site_acc_medium": torch.tensor(soft_attachment_site_acc_medium_list),
        "scaffold_relation_f1": torch.tensor(scaffold_relation_f1_list),
        "connectivity_score": torch.tensor(connectivity_score_list),
        "local_chem_score": torch.tensor(local_chem_score_list),
        "global_shape_aux_score": torch.tensor(global_shape_aux_score_list),
        "atom_match_coverage": torch.tensor(atom_match_coverage_list),
        "soft_recon_score": torch.tensor(soft_recon_score_list),
        "struct_fidelity_score": torch.tensor(struct_fidelity_score_list),
        "struct_fidelity_pass": torch.tensor(struct_fidelity_pass_list),
        "atom_count_exact_mean": _safe_mean(atom_count_exact_list),
        "atom_count_abs_error_mean": _safe_mean(atom_count_abs_error_list),
        "atom_count_score_mean": _safe_mean(atom_count_score_list),
        "matched_atom_rmsd_mean": _safe_mean(matched_atom_rmsd_list),
        "matched_heavy_atom_rmsd_mean": _safe_mean(matched_heavy_atom_rmsd_list),
        "matched_atom_mae_mean": _safe_mean(matched_atom_mae_list),
        "atom_position_score_mean": _safe_mean(atom_position_score_list),
        "heavy_atom_hit_rate_tight_mean": _safe_mean(heavy_atom_hit_rate_tight_list),
        "heavy_atom_hit_rate_medium_mean": _safe_mean(heavy_atom_hit_rate_medium_list),
        "heavy_atom_hit_rate_loose_mean": _safe_mean(heavy_atom_hit_rate_loose_list),
        "typed_heavy_atom_hit_rate_tight_mean": _safe_mean(typed_heavy_atom_hit_rate_tight_list),
        "typed_heavy_atom_hit_rate_medium_mean": _safe_mean(typed_heavy_atom_hit_rate_medium_list),
        "typed_heavy_atom_hit_rate_loose_mean": _safe_mean(typed_heavy_atom_hit_rate_loose_list),
        "heteroatom_hit_rate_medium_mean": _safe_mean(heteroatom_hit_rate_medium_list),
        "macro_type_f1_mean": _safe_mean(macro_type_f1_list),
        "ch_collapse_rate_mean": _safe_mean(ch_collapse_rate_list),
        "atom_type_acc_mean": _safe_mean(atom_type_acc_list),
        "heteroatom_precision_mean": _safe_mean(hetero_precision_list),
        "heteroatom_recall_mean": _safe_mean(hetero_recall_list),
        "heteroatom_f1_mean": _safe_mean(hetero_f1_list),
        "atom_semantic_score_mean": _safe_mean(atom_semantic_score_list),
        "ring_count_exact_mean": _safe_mean(ring_count_exact_list),
        "ring_complete_rate_mean": _safe_mean(ring_complete_rate_list),
        "ring_size_acc_mean": _safe_mean(ring_size_acc_list),
        "ring_integrity_score_mean": _safe_mean(ring_integrity_score_list),
        "scaffold_local_edge_recall_mean": _safe_mean(scaffold_local_edge_recall_list),
        "scaffold_local_edge_f1_mean": _safe_mean(scaffold_local_edge_f1_list),
        "ring_site_coverage_medium_mean": _safe_mean(ring_site_coverage_medium_list),
        "attachment_edge_f1_mean": _safe_mean(attachment_edge_f1_list),
        "soft_attachment_site_acc_medium_mean": _safe_mean(soft_attachment_site_acc_medium_list),
        "scaffold_relation_f1_mean": _safe_mean(scaffold_relation_f1_list),
        "connectivity_score_mean": _safe_mean(connectivity_score_list),
        "local_chem_score_mean": _safe_mean(local_chem_score_list),
        "global_shape_aux_score_mean": _safe_mean(global_shape_aux_score_list),
        "atom_match_coverage_mean": _safe_mean(atom_match_coverage_list),
        "soft_recon_score_mean": _safe_mean(soft_recon_score_list),
        "struct_fidelity_score_mean": _safe_mean(struct_fidelity_score_list),
        "struct_fidelity_pass_rate": _safe_mean(struct_fidelity_pass_list),
    }


def compute_bottom_atom_recall(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    pred_types: torch.Tensor,
    gt_types: torch.Tensor,
    mask: torch.Tensor,
    z_threshold_ratio: float = 0.3,
    distance_threshold: float = 0.1,
) -> torch.Tensor:
    """Compute recall for bottom-layer (occluded) atoms.

    Bottom atoms: atoms whose Z coordinate is in the lower z_threshold_ratio
    of the molecule's Z range.

    An atom is considered "recalled" if there exists a predicted atom
    within distance_threshold (in normalized units) with correct type.

    Args:
        pred_coords: (B, N, 3) predicted
        gt_coords: (B, N, 3) ground truth
        pred_types: (B, N) predicted atom type indices
        gt_types: (B, N) ground truth atom type indices
        mask: (B, N) atom mask
        z_threshold_ratio: bottom fraction to consider
        distance_threshold: matching distance threshold

    Returns:
        recall: (B,) per-sample recall
    """
    B = pred_coords.shape[0]
    recalls = []

    for b in range(B):
        m = mask[b].bool()
        n = m.sum().item()
        if n == 0:
            recalls.append(0.0)
            continue

        g = gt_coords[b, :n].detach().cpu().numpy()
        p = pred_coords[b, :n].detach().cpu().numpy()
        gt_t = gt_types[b, :n].detach().cpu().numpy()
        pt_t = pred_types[b, :n].detach().cpu().numpy()

        # Identify bottom atoms by Z coordinate
        z_min, z_max = g[:, 2].min(), g[:, 2].max()
        z_range = z_max - z_min
        if z_range < 1e-6:
            recalls.append(1.0)
            continue

        z_cutoff = z_min + z_range * z_threshold_ratio
        bottom_mask = g[:, 2] <= z_cutoff
        n_bottom = bottom_mask.sum()

        if n_bottom == 0:
            recalls.append(1.0)
            continue

        # For each bottom GT atom, check if matched
        recalled = 0
        for i in range(n):
            if not bottom_mask[i]:
                continue
            # Find nearest predicted atom
            dists = np.sqrt(((p - g[i:i+1]) ** 2).sum(axis=-1))
            j = dists.argmin()
            if dists[j] < distance_threshold and pt_t[j] == gt_t[i]:
                recalled += 1

        recalls.append(recalled / n_bottom)

    return torch.tensor(recalls)


# ============================================================
# Bond validity constants (shared with constraints.py)
# IDEAL_BOND_LENGTHS and MAX_BOND_DIST are imported from constraints.py.
# ============================================================


def compute_atom_count_accuracy(
    pred_n: torch.Tensor,
    gt_n: torch.Tensor,
) -> dict:
    """Compute atom count prediction accuracy.

    Args:
        pred_n: (B,) predicted atom counts
        gt_n: (B,) ground truth atom counts

    Returns:
        dict with 'exact_match' (fraction), 'mae' (mean absolute error)
    """
    exact_match = (pred_n == gt_n).float().mean().item()
    mae = (pred_n.float() - gt_n.float()).abs().mean().item()
    return {"exact_match": exact_match, "mae": mae}


def compute_bond_validity(
    pred_coords: torch.Tensor,
    pred_types: torch.Tensor,
    mask: torch.Tensor,
    tolerance: float = None,
) -> torch.Tensor:
    """Compute fraction of detected bonds with valid lengths.

    Bond definition is now unified with constraints.py:
    - Candidate bond: dist < MAX_BOND_DIST[ti,tj] (= IDEAL_BOND_LENGTHS * 1.3)
    - Valid bond: |dist - IDEAL_BOND_LENGTHS[ti,tj]| / IDEAL_BOND_LENGTHS < BOND_VALIDITY_TOLERANCE

    Args:
        pred_coords: (B, N, 3)
        pred_types: (B, N) atom type indices
        mask: (B, N) atom mask (1=valid, 0=padding). Use n_atoms_pred for pred-masked version.
        tolerance: if None, uses BOND_VALIDITY_TOLERANCE from constraints.py (0.25 = 25%).

    Returns:
        validity: (B,) fraction of valid bonds per sample
    """
    if tolerance is None:
        tolerance = BOND_VALIDITY_TOLERANCE  # 0.25

    ideal = IDEAL_BOND_LENGTHS.numpy()
    max_dist_table = MAX_BOND_DIST.numpy()

    B = pred_coords.shape[0]
    results = []

    for b in range(B):
        m = mask[b].bool()
        n = m.sum().item()
        if n < 2:
            results.append(1.0)
            continue

        coords = pred_coords[b, :n].detach().cpu().numpy()
        types = pred_types[b, :n].detach().cpu().numpy()

        # Pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))

        n_bonds = 0
        n_valid = 0

        for i in range(n):
            for j in range(i + 1, n):
                ti, tj = int(types[i]), int(types[j])
                if ti < 0 or tj < 0 or ti >= _NUM_ATOM_TYPES or tj >= _NUM_ATOM_TYPES:
                    continue
                ideal_len = ideal[ti, tj]
                if ideal_len < 1e-8:
                    continue
                max_d = max_dist_table[ti, tj]
                if dists[i, j] < max_d:
                    n_bonds += 1
                    if abs(dists[i, j] - ideal_len) / ideal_len < tolerance:
                        n_valid += 1

        results.append(n_valid / max(n_bonds, 1))

    return torch.tensor(results)


def compute_bottom_atom_rmsd(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    mask: torch.Tensor,
    z_threshold_ratio: float = 0.3,
) -> torch.Tensor:
    """Compute RMSD specifically for bottom-layer atoms.

    Args:
        pred_coords: (B, N, 3)
        gt_coords: (B, N, 3)
        mask: (B, N)
        z_threshold_ratio: bottom fraction to consider

    Returns:
        rmsd: (B,) bottom-atom RMSD per sample
    """
    B = pred_coords.shape[0]
    results = []

    for b in range(B):
        m = mask[b].bool()
        n = m.sum().item()
        if n == 0:
            results.append(0.0)
            continue

        g = gt_coords[b, :n].detach().cpu().numpy()
        p = pred_coords[b, :n].detach().cpu().numpy()

        z_min, z_max = g[:, 2].min(), g[:, 2].max()
        z_range = z_max - z_min
        if z_range < 1e-6:
            results.append(0.0)
            continue

        z_cutoff = z_min + z_range * z_threshold_ratio
        bottom_mask = g[:, 2] <= z_cutoff
        n_bottom = bottom_mask.sum()

        if n_bottom == 0:
            results.append(0.0)
            continue

        # Hungarian matching on full molecule, then extract bottom atoms
        diff = p[:, None, :] - g[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        row_ind, col_ind = linear_sum_assignment(cost)

        # Filter to bottom GT atoms
        bottom_diffs = []
        for r, c in zip(row_ind, col_ind):
            if bottom_mask[c]:
                bottom_diffs.append(p[r] - g[c])

        if len(bottom_diffs) == 0:
            results.append(0.0)
            continue

        bottom_diffs = np.array(bottom_diffs)
        msd = (bottom_diffs ** 2).sum() / len(bottom_diffs)
        results.append(float(np.sqrt(msd)))

    return torch.tensor(results)




# Atomic numbers for Coulomb matrix (index: H=0,C=1,N=2,O=3,F=4,S=5,P=6,Cl=7,Br=8,I=9)
_ATOMIC_NUMBERS = np.array([1, 6, 7, 8, 9, 16, 15, 17, 35, 53], dtype=np.float64)

# Max valence per atom type for valence validity check
_MAX_VALENCE = {0: 1, 1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5, 7: 1, 8: 1, 9: 1}

# Min valence per atom type (V3: detect broken bonds and isolated atoms)
_MIN_VALENCE = {0: 1, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}

# Covalent radii in normalized space (Angstrom / 12.0) for bond detection
_COVALENT_RADII = {
    0: 0.0258,   # H: 0.31 A
    1: 0.0642,   # C: 0.77 A
    2: 0.0608,   # N: 0.73 A
    3: 0.0550,   # O: 0.66 A
    4: 0.0533,   # F: 0.64 A
    5: 0.0867,   # S: 1.04 A
    6: 0.0892,   # P: 1.07 A
    7: 0.0825,   # Cl: 0.99 A
    8: 0.0950,   # Br: 1.14 A
    9: 0.1108,   # I: 1.33 A
}


def _coulomb_matrix_eigenvalues(coords: np.ndarray, types: np.ndarray) -> np.ndarray:
    """Compute sorted Coulomb matrix eigenvalues (rotation-invariant molecular descriptor).

    C_ij = Z_i * Z_j / |r_i - r_j| for i != j
    C_ii = 0.5 * Z_i^2.4

    Args:
        coords: (n, 3) atom coordinates
        types: (n,) atom type indices
    Returns:
        Sorted eigenvalues (descending) as 1D array
    """
    n = len(coords)
    Z = np.array([_ATOMIC_NUMBERS[int(t)] for t in types])
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = 0.5 * Z[i] ** 2.4
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 1e-8:
                d = 1e-8
            C[i, j] = Z[i] * Z[j] / d
            C[j, i] = C[i, j]
    eigenvalues = np.sort(np.linalg.eigvalsh(C))[::-1]
    return eigenvalues


def _pairwise_distance_histogram(coords: np.ndarray, n_bins: int = 50,
                                  max_dist: float = 2.0) -> np.ndarray:
    """Compute normalized histogram of all pairwise interatomic distances.

    Args:
        coords: (n, 3) atom coordinates (normalized space)
        n_bins: number of histogram bins
        max_dist: maximum distance for histogram range
    Returns:
        Normalized histogram (sums to 1)
    """
    n = len(coords)
    if n < 2:
        return np.ones(n_bins) / n_bins
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(coords[i] - coords[j]))
    hist, _ = np.histogram(dists, bins=n_bins, range=(0, max_dist), density=False)
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        hist = np.ones(n_bins) / n_bins
    return hist


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions."""
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = (p * np.log(p / m)).sum()
    kl_qm = (q * np.log(q / m)).sum()
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _valence_validity(coords: np.ndarray, types: np.ndarray,
                       tolerance: float = 0.0333) -> float:
    """Check valence validity: fraction of atoms with chemically reasonable valence.

    V3: bidirectional check — both min AND max valence must be satisfied.
    Also checks molecular connectivity (single connected component).

    Args:
        coords: (n, 3)
        types: (n,) atom type indices
        tolerance: bond detection tolerance in normalized space (~0.4 A / 12)
    Returns:
        Fraction of atoms with valid valence in [0, 1]
    """
    n = len(coords)
    if n < 2:
        return 1.0
    valences = np.zeros(n, dtype=int)
    # Build adjacency for connectivity check
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            r_sum = _COVALENT_RADII.get(int(types[i]), 0.07) + _COVALENT_RADII.get(int(types[j]), 0.07)
            if d < r_sum + tolerance:
                valences[i] += 1
                valences[j] += 1
                adj[i].append(j)
                adj[j].append(i)

    # Bidirectional valence check
    valid = 0
    for i in range(n):
        max_v = _MAX_VALENCE.get(int(types[i]), 4)
        min_v = _MIN_VALENCE.get(int(types[i]), 1)
        if min_v <= valences[i] <= max_v:
            valid += 1
    valence_score = valid / n

    # Connectivity check: BFS to count connected components
    visited = [False] * n
    components = 0
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

    # Connectivity penalty: 1.0 if single component, decreasing with more fragments
    connectivity_score = 1.0 / components if components > 0 else 0.0

    # Combined: 70% valence + 30% connectivity
    return 0.7 * valence_score + 0.3 * connectivity_score


def compute_structure_similarity(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    pred_types: torch.Tensor,
    gt_types: torch.Tensor,
    mask: torch.Tensor,
    n_atoms_pred: torch.Tensor = None,
) -> dict:
    """Compute molecular structure similarity.

    Sub-metrics (all use Hungarian matching for per-atom correspondence):
      1. Type accuracy — element correctness after matching
      2. Coulomb matrix similarity — rotation-invariant descriptor
      3. Valence validity — chemically valid bond count
      4. Atom count similarity — |N_pred - N_gt| / max
      5. Formula similarity — element composition cosine

    Overall score:
        0.25 * type_acc + 0.25 * coulomb_sim + 0.15 * (1 - JS)
        + 0.10 * valence + 0.10 * count_sim

    Args:
        pred_coords: (B, N, 3)
        gt_coords: (B, N, 3)
        pred_types: (B, N)
        gt_types: (B, N)
        mask: (B, N) ground truth atom mask
        n_atoms_pred: (B,) predicted atom counts. If None, uses n_gt.

    Returns:
        dict with per-sample tensors and mean values for all sub-metrics
    """
    B = pred_coords.shape[0]
    type_accs = []
    coulomb_sims = []
    valence_vals = []
    count_sims = []
    formula_sims = []
    overall_sims = []

    for b in range(B):
        m = mask[b].bool()
        n_gt = m.sum().item()
        if n_gt == 0:
            type_accs.append(1.0)
            coulomb_sims.append(1.0)
            valence_vals.append(1.0)
            count_sims.append(1.0)
            formula_sims.append(1.0)
            overall_sims.append(1.0)
            continue

        g = gt_coords[b, :n_gt].detach().cpu().numpy()
        gt_t = gt_types[b, :n_gt].detach().cpu().numpy()
        # Use predicted atom count if available, otherwise fall back to n_gt
        if n_atoms_pred is not None:
            n_pred = int(n_atoms_pred[b].item())
            n_pred = max(1, min(n_pred, pred_coords.shape[1]))  # clamp to valid range
        else:
            n_pred = n_gt
        p = pred_coords[b, :n_pred].detach().cpu().numpy()
        pt_t = pred_types[b, :n_pred].detach().cpu().numpy()

        # --- (1) Hungarian matching + Type accuracy ---
        n_match = min(n_pred, n_gt)
        if n_match >= 1:
            diff = p[:, None, :] - g[None, :, :]  # (n_pred, n_gt, 3)
            cost = np.sqrt((diff ** 2).sum(axis=-1))  # (n_pred, n_gt)
            row_ind, col_ind = linear_sum_assignment(cost)
            p_matched = p[row_ind]
            g_matched = g[col_ind]
            pt_matched = pt_t[row_ind]
            gt_matched = gt_t[col_ind]
        else:
            row_ind, col_ind = np.array([]), np.array([])
            p_matched = p[:0]
            g_matched = g[:0]
            pt_matched = pt_t[:0]
            gt_matched = gt_t[:0]

        if len(row_ind) > 0:
            type_acc = float((pt_matched == gt_matched).sum() / len(row_ind))
        else:
            type_acc = 0.0

        # --- (2) Coulomb matrix similarity ---
        if n_gt >= 2 and n_pred >= 2:
            eig_gt = _coulomb_matrix_eigenvalues(g, gt_t)
            eig_pred = _coulomb_matrix_eigenvalues(p, pt_t)
            max_len = max(len(eig_gt), len(eig_pred))
            eig_gt_pad = np.zeros(max_len)
            eig_pred_pad = np.zeros(max_len)
            eig_gt_pad[:len(eig_gt)] = eig_gt
            eig_pred_pad[:len(eig_pred)] = eig_pred
            l2 = np.linalg.norm(eig_gt_pad - eig_pred_pad)
            norm = max(np.linalg.norm(eig_gt_pad), 1e-8)
            coulomb_sim = max(0.0, 1.0 - l2 / norm)
        else:
            coulomb_sim = 0.0

        # --- (3) Valence validity ---
        valence_val = _valence_validity(p, pt_t)

        # --- (4) Atom count similarity ---
        n_p = max(n_pred, 1)
        n_g = max(n_gt, 1)
        count_sim = 1.0 - abs(n_p - n_g) / max(n_p, n_g)

        # --- (5) Formula similarity (element composition cosine) ---
        # Hungarian-matched subset for atom correspondence
        if len(row_ind) > 0:
            c_pred = np.bincount(pt_matched, minlength=NUM_ATOM_TYPES).astype(float)
            c_gt = np.bincount(gt_matched, minlength=NUM_ATOM_TYPES).astype(float)
        else:
            c_pred = np.zeros(NUM_ATOM_TYPES)
            c_gt = np.zeros(NUM_ATOM_TYPES)
        dot = (c_pred * c_gt).sum()
        norm_pred = np.linalg.norm(c_pred)
        norm_gt = np.linalg.norm(c_gt)
        formula_sim = dot / (norm_pred * norm_gt + 1e-8) if norm_pred > 0 and norm_gt > 0 else 0.0

        # --- Overall weighted score (V16) ---
        overall = (
            0.25 * type_acc
            + 0.25 * coulomb_sim
            + 0.15 * (1 - _js_divergence(_pairwise_distance_histogram(p), _pairwise_distance_histogram(g)))
            + 0.10 * valence_val
            + 0.10 * count_sim
        )

        type_accs.append(type_acc)
        coulomb_sims.append(coulomb_sim)
        valence_vals.append(valence_val)
        count_sims.append(count_sim)
        formula_sims.append(formula_sim)
        overall_sims.append(overall)

    return {
        "type_match_rate": torch.tensor(type_accs),
        "coulomb_similarity": torch.tensor(coulomb_sims),
        "valence_validity": torch.tensor(valence_vals),
        "count_similarity": torch.tensor(count_sims),
        "formula_similarity": torch.tensor(formula_sims),
        "overall_similarity": torch.tensor(overall_sims),
        "type_match_rate_mean": float(np.mean(type_accs)),
        "coulomb_similarity_mean": float(np.mean(coulomb_sims)),
        "valence_validity_mean": float(np.mean(valence_vals)),
        "count_similarity_mean": float(np.mean(count_sims)),
        "formula_similarity_mean": float(np.mean(formula_sims)),
        "overall_similarity_mean": float(np.mean(overall_sims)),
    }


def compute_cid_accuracy(
    pred_indices: torch.Tensor,
    gt_cid_idx: torch.Tensor,
) -> dict:
    """Compute CID retrieval accuracy.

    Args:
        pred_indices: (B, K) top-K predicted CID indices
        gt_cid_idx: (B,) ground truth CID indices

    Returns:
        dict with 'top1_acc', 'top5_acc'
    """
    gt = gt_cid_idx.unsqueeze(-1)  # (B, 1)
    top1 = (pred_indices[:, :1] == gt).any(dim=-1).float().mean().item()
    top5 = (pred_indices == gt).any(dim=-1).float().mean().item()
    return {"top1_acc": top1, "top5_acc": top5}


def compute_composite_score(
    rmsd: float,
    bottom_atom_score: float,
    bond_validity: float,
    ring_preservation: float,
    atom_count_accuracy: float,
    structure_similarity: float = 0.0,
    rmsd_max: float = 2.0,
) -> float:
    """Compute weighted composite evaluation score.

    All sub-scores should be in [0, 1], higher is better.

    Args:
        rmsd: mean RMSD (will be converted to 1 - rmsd/rmsd_max)
        bottom_atom_score: bottom atom recall or (1 - bottom_rmsd/rmsd_max)
        bond_validity: fraction of valid bonds
        ring_preservation: fraction of preserved rings
        atom_count_accuracy: exact match rate
        structure_similarity: 3D structure similarity score
        rmsd_max: max RMSD for normalization

    Returns:
        composite score in [0, 1]
    """
    rmsd_score = max(0.0, 1.0 - rmsd / rmsd_max)

    # Composite: RMSD + Bottom + Bond + Ring + Count + Structure (V16)
    composite = (
        0.30 * rmsd_score
        + 0.20 * bottom_atom_score
        + 0.15 * bond_validity
        + 0.15 * ring_preservation
        + 0.10 * atom_count_accuracy
        + 0.10 * structure_similarity
    )
    return float(composite)


# ============================================================
# V7: Molecule-level metrics
# ============================================================

def compute_formula_similarity(
    pred_types: torch.Tensor,
    gt_types: torch.Tensor,
    mask: torch.Tensor,
    n_atoms_pred: torch.Tensor = None,
    num_types: int = 10,
) -> dict:
    """Compute molecule-level element composition similarity.

    Compares the element count vectors (molecular formula) using cosine similarity.
    Does NOT require per-atom correspondence — only checks if the overall
    composition is correct (e.g., "6C, 6H, 1O" vs "6C, 5H, 1N, 1O").

    Also computes type distribution JS-divergence for proportion matching.

    Args:
        pred_types: (B, N) predicted atom type indices
        gt_types: (B, N) ground truth atom type indices
        mask: (B, N) ground truth atom mask
        n_atoms_pred: (B,) predicted atom counts
        num_types: number of atom types

    Returns:
        dict with formula_similarity, type_distribution_match per sample
    """
    B = pred_types.shape[0]
    formula_sims = []
    type_dist_matches = []

    for b in range(B):
        n_gt = int(mask[b].bool().sum().item())
        if n_gt == 0:
            formula_sims.append(1.0)
            type_dist_matches.append(1.0)
            continue

        n_pred = int(n_atoms_pred[b].item()) if n_atoms_pred is not None else n_gt
        n_pred = max(1, min(n_pred, pred_types.shape[1]))

        gt_t = gt_types[b, :n_gt].detach().cpu()
        pt_t = pred_types[b, :n_pred].detach().cpu()

        # Element count vectors
        gt_counts = torch.zeros(num_types)
        pred_counts = torch.zeros(num_types)
        for i in range(num_types):
            gt_counts[i] = (gt_t == i).sum().float()
            pred_counts[i] = (pt_t == i).sum().float()

        # Cosine similarity of formula vectors
        dot = (gt_counts * pred_counts).sum()
        norm_gt = gt_counts.norm()
        norm_pred = pred_counts.norm()
        if norm_gt > 0 and norm_pred > 0:
            formula_sim = float(dot / (norm_gt * norm_pred))
        else:
            formula_sim = 0.0
        formula_sims.append(formula_sim)

        # Type distribution JS-divergence
        gt_dist = gt_counts / gt_counts.sum().clamp(min=1)
        pred_dist = pred_counts / pred_counts.sum().clamp(min=1)
        gt_np = gt_dist.numpy() + 1e-10
        pred_np = pred_dist.numpy() + 1e-10
        gt_np = gt_np / gt_np.sum()
        pred_np = pred_np / pred_np.sum()
        m = 0.5 * (gt_np + pred_np)
        js = 0.5 * (gt_np * np.log(gt_np / m)).sum() + 0.5 * (pred_np * np.log(pred_np / m)).sum()
        type_dist_matches.append(float(max(0.0, 1.0 - js)))

    return {
        "formula_similarity": torch.tensor(formula_sims),
        "formula_similarity_mean": float(np.mean(formula_sims)),
        "type_distribution_match": torch.tensor(type_dist_matches),
        "type_distribution_match_mean": float(np.mean(type_dist_matches)),
    }
