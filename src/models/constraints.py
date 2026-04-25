"""
Differentiable physics-informed constraint losses for molecular structure generation.

All coordinates are in NORMALIZED space (divided by 12.0, mapping to [-1, 1]).
Atom type indices follow dataset.py: H=0, C=1, N=2, O=3, F=4, S=5, P=6, Cl=7, Br=8, I=9.

Provides:
    - bond_length_penalty: penalizes deviations from ideal bond lengths
    - bond_angle_penalty: penalizes deviations from ideal bond angles
    - planarity_penalty: penalizes out-of-plane deviations for ring atoms
    - compute_all_constraints: convenience wrapper returning weighted sum
"""

import math
import torch
import torch.nn.functional as F

NUM_ATOM_TYPES = 10

# Atom type indices (mirroring dataset.py)
H, C, N, O, F_, S, P, Cl, Br, I = range(NUM_ATOM_TYPES)

# ---------------------------------------------------------------------------
# 1. Bond Length Lookup Table
# ---------------------------------------------------------------------------

# Ideal bond lengths in normalized space (Angstrom / 12.0).
_BOND_LENGTH_PAIRS = {
    frozenset({C, C}): 0.12,    # average of single/double C-C
    frozenset({C, H}): 0.0908,
    frozenset({C, N}): 0.1225,
    frozenset({C, O}): 0.1192,
    frozenset({C, S}): 0.1517,
    frozenset({C, F_}): 0.1125,
    frozenset({C, Cl}): 0.1475,
    frozenset({C, Br}): 0.1617,
    frozenset({C, I}): 0.1783,
    frozenset({N, H}): 0.0842,
    frozenset({O, H}): 0.0800,
    frozenset({N, N}): 0.1208,
    frozenset({N, O}): 0.1167,
    frozenset({S, H}): 0.1117,
}

# Build (NUM_ATOM_TYPES, NUM_ATOM_TYPES) tensors for fast lookup.
_ideal_matrix = torch.zeros(NUM_ATOM_TYPES, NUM_ATOM_TYPES)
for pair, length in _BOND_LENGTH_PAIRS.items():
    atoms = list(pair)
    if len(atoms) == 1:
        # same-element bond (e.g. C-C)
        _ideal_matrix[atoms[0], atoms[0]] = length
    else:
        _ideal_matrix[atoms[0], atoms[1]] = length
        _ideal_matrix[atoms[1], atoms[0]] = length

IDEAL_BOND_LENGTHS: torch.Tensor = _ideal_matrix
"""(NUM_ATOM_TYPES, NUM_ATOM_TYPES) ideal bond lengths in normalized space."""

MAX_BOND_DIST: torch.Tensor = _ideal_matrix * 1.3
"""(NUM_ATOM_TYPES, NUM_ATOM_TYPES) maximum distance considered a potential bond."""

# ---------------------------------------------------------------------------
# B. Bond Validity Thresholds (shared across training, guidance, evaluation)
# ---------------------------------------------------------------------------

BOND_TOLERANCE: float = 0.0125
"""Training penalty threshold: deviations below this (normalized space) are not penalized (~0.15 Å)."""

BOND_VALIDITY_TOLERANCE: float = 0.25
"""Relative tolerance for bond validity: valid if |d - ideal| / ideal < tolerance."""

# VDW radii in Ångstrom — used ONLY for repulsion/connectivity guidance.
# Training constraints use IDEAL_BOND_LENGTHS exclusively.
VDW_RADII_ANGSTROM: list[float] = [
    1.20,  # H
    1.70,  # C
    1.55,  # N
    1.52,  # O
    1.47,  # F
    1.80,  # S
    1.80,  # P
    1.75,  # Cl
    1.85,  # Br
    1.98,  # I
]
VDW_RADII: torch.Tensor = torch.tensor(VDW_RADII_ANGSTROM) / 12.0
"""(NUM_ATOM_TYPES,) VDW radii normalized to coordinate space."""


def _ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a module-level tensor to the target device (cached)."""
    if tensor.device != device:
        return tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# 2. bond_length_penalty
# ---------------------------------------------------------------------------

def bond_length_penalty(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
    tolerance: float = 0.0125,
) -> torch.Tensor:
    """Penalize deviations of bonded-atom distances from ideal bond lengths.

    Args:
        coords: (B, N, 3) predicted coordinates in normalized space.
        atom_types: (B, N) int64 atom type indices (may contain -1 for padding).
        mask: (B, N) float32 binary mask (1 = valid atom).
        tolerance: float, deviations below this are not penalized
                   (default 0.0125 ~ 0.15 Angstrom).

    Returns:
        Scalar loss (mean squared excess deviation over all bonded pairs).
    """
    device = coords.device
    B, N, _ = coords.shape

    ideal_table = _ensure_device(IDEAL_BOND_LENGTHS, device)
    max_table = _ensure_device(MAX_BOND_DIST, device)

    # Pairwise distances: (B, N, N)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)    # (B, N, N)

    # Valid pair mask: both atoms valid AND atom_types >= 0
    type_valid = (atom_types >= 0).float() * mask      # (B, N)
    pair_valid = type_valid.unsqueeze(2) * type_valid.unsqueeze(1)  # (B, N, N)

    # Remove self-pairs
    eye = torch.eye(N, device=device).unsqueeze(0)
    pair_valid = pair_valid * (1.0 - eye)

    # Clamp atom_types to [0, NUM_ATOM_TYPES-1] for indexing (padding stays valid
    # because pair_valid already masks them out).
    safe_types = atom_types.clamp(min=0, max=NUM_ATOM_TYPES - 1)  # (B, N)

    # Look up ideal bond lengths for every (i, j) pair
    # Flatten to use advanced indexing on the table
    ti = safe_types.unsqueeze(2).expand(B, N, N)  # (B, N, N)
    tj = safe_types.unsqueeze(1).expand(B, N, N)  # (B, N, N)
    ideal = ideal_table[ti, tj]                    # (B, N, N)
    max_d = max_table[ti, tj]                      # (B, N, N)

    # Identify bonded pairs: distance < max_bond_dist AND ideal > 0
    bonded = (dist < max_d) & (ideal > 0)
    bonded = bonded.float() * pair_valid           # (B, N, N)

    num_bonded = bonded.sum()
    if num_bonded < 1.0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Penalty: mean of max(0, |dist - ideal| - tolerance)^2
    deviation = (dist - ideal).abs() - tolerance
    penalty = F.relu(deviation) ** 2
    loss = (penalty * bonded).sum() / num_bonded

    return loss


# ---------------------------------------------------------------------------
# 3. bond_angle_penalty
# ---------------------------------------------------------------------------

def bond_angle_penalty(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviations of bond angles from ideal values.

    For each central atom *j* bonded to at least two neighbours *i* and *k*,
    compute the angle i-j-k and compare to the ideal angle determined by
    atom j's type and coordination number.

    Args:
        coords: (B, N, 3) predicted coordinates in normalized space.
        atom_types: (B, N) int64 atom type indices (may contain -1 for padding).
        mask: (B, N) float32 binary mask.

    Returns:
        Scalar loss (mean of (cos(angle) - cos(ideal))^2 over sampled triplets).
    """
    device = coords.device
    B, N, _ = coords.shape
    MAX_TRIPLETS = 500

    ideal_table = _ensure_device(IDEAL_BOND_LENGTHS, device)
    max_table = _ensure_device(MAX_BOND_DIST, device)

    # --- Identify bonds (same logic as bond_length_penalty) ---
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(-1) + 1e-12)

    type_valid = (atom_types >= 0).float() * mask
    pair_valid = type_valid.unsqueeze(2) * type_valid.unsqueeze(1)
    eye = torch.eye(N, device=device).unsqueeze(0)
    pair_valid = pair_valid * (1.0 - eye)

    safe_types = atom_types.clamp(min=0, max=NUM_ATOM_TYPES - 1)
    ti = safe_types.unsqueeze(2).expand(B, N, N)
    tj = safe_types.unsqueeze(1).expand(B, N, N)
    ideal_bl = ideal_table[ti, tj]
    max_d = max_table[ti, tj]

    bonded = ((dist < max_d) & (ideal_bl > 0)).float() * pair_valid  # (B, N, N)

    # --- Collect triplets (i, j, k) where j is the central atom ---
    # bonded[b, j, i] == 1 means i is bonded to j
    # We need j with >= 2 bonds.

    # Work per-batch to collect triplets
    all_cos_angles = []
    all_cos_ideals = []

    for b in range(B):
        bond_mat = bonded[b]  # (N, N)
        num_bonds = bond_mat.sum(dim=1)  # (N,)

        # Atoms with >= 2 bonds
        central_mask = num_bonds >= 2
        central_indices = central_mask.nonzero(as_tuple=False).squeeze(-1)

        if central_indices.numel() == 0:
            continue

        triplets_i = []
        triplets_j = []
        triplets_k = []

        for j_idx in central_indices:
            j = j_idx.item()
            neighbours = bond_mat[j].nonzero(as_tuple=False).squeeze(-1)
            n_neigh = neighbours.numel()
            if n_neigh < 2:
                continue
            # All pairs of neighbours
            for a in range(n_neigh):
                for c in range(a + 1, n_neigh):
                    triplets_i.append(neighbours[a].item())
                    triplets_j.append(j)
                    triplets_k.append(neighbours[c].item())

        if len(triplets_j) == 0:
            continue

        # Convert to tensors
        ti_t = torch.tensor(triplets_i, device=device, dtype=torch.long)
        tj_t = torch.tensor(triplets_j, device=device, dtype=torch.long)
        tk_t = torch.tensor(triplets_k, device=device, dtype=torch.long)

        # Sample if too many
        n_triplets = ti_t.size(0)
        if n_triplets > MAX_TRIPLETS:
            perm = torch.randperm(n_triplets, device=device)[:MAX_TRIPLETS]
            ti_t = ti_t[perm]
            tj_t = tj_t[perm]
            tk_t = tk_t[perm]

        # Compute angles using vectors from j to i and j to k
        coords_b = coords[b]  # (N, 3)
        vec_ji = coords_b[ti_t] - coords_b[tj_t]  # (T, 3)
        vec_jk = coords_b[tk_t] - coords_b[tj_t]  # (T, 3)

        cos_angle = F.cosine_similarity(vec_ji, vec_jk, dim=-1)  # (T,)
        cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Determine ideal angle for each central atom j
        # Count bonds for each j in this triplet set
        j_types = safe_types[b][tj_t]  # (T,)
        j_num_bonds = num_bonds[tj_t]  # (T,)

        # Default: 109.47 degrees (tetrahedral)
        ideal_angle = torch.full_like(cos_angle, math.radians(109.47))

        # Carbon with 3+ bonds but not 4 -> sp2 -> 120 degrees
        is_carbon = (j_types == C)
        is_sp2 = is_carbon & (j_num_bonds >= 3) & (j_num_bonds < 4)
        ideal_angle = torch.where(is_sp2, torch.tensor(math.radians(120.0), device=device), ideal_angle)

        cos_ideal = torch.cos(ideal_angle)

        all_cos_angles.append(cos_angle)
        all_cos_ideals.append(cos_ideal)

    if len(all_cos_angles) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    all_cos_angles = torch.cat(all_cos_angles)
    all_cos_ideals = torch.cat(all_cos_ideals)

    loss = ((all_cos_angles - all_cos_ideals) ** 2).mean()
    return loss


# ---------------------------------------------------------------------------
# 4. planarity_penalty
# ---------------------------------------------------------------------------

def planarity_penalty(
    coords: torch.Tensor,
    ring_atom_indices: torch.Tensor,
    ring_valid: torch.Tensor,
) -> torch.Tensor:
    """Penalize out-of-plane deviations for atoms in planar rings.

    Args:
        coords: (B, N, 3) predicted coordinates in normalized space.
        ring_atom_indices: (B, MAX_RINGS, MAX_RING_SIZE) int64, -1 for padding.
        ring_valid: (B, MAX_RINGS) float32 binary mask for valid rings.

    Returns:
        Scalar loss (mean of squared distances from best-fit plane).
    """
    device = coords.device
    B, MAX_RINGS, MAX_RING_SIZE = ring_atom_indices.shape

    total_penalty = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        for r in range(MAX_RINGS):
            if ring_valid[b, r].item() < 0.5:
                continue

            # Get valid ring atom indices
            indices = ring_atom_indices[b, r]          # (MAX_RING_SIZE,)
            valid = indices >= 0
            valid_indices = indices[valid]              # (K,)
            K = valid_indices.size(0)
            if K < 3:
                continue

            # Extract ring atom coords
            ring_coords = coords[b][valid_indices]     # (K, 3)

            # Compute centroid
            centroid = ring_coords.mean(dim=0, keepdim=True)  # (1, 3)
            centered = ring_coords - centroid                  # (K, 3)

            # SVD to find plane normal (smallest singular value direction)
            # centered: (K, 3), U @ diag(S) @ V^T = centered
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            normal = Vh[-1]  # (3,) direction of smallest singular value

            # Distance of each atom from the best-fit plane
            dists = (centered * normal.unsqueeze(0)).sum(dim=-1)  # (K,)
            total_penalty = total_penalty + (dists ** 2).mean()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_penalty / count


# ---------------------------------------------------------------------------
# 5. compute_all_constraints
# ---------------------------------------------------------------------------

def compute_all_constraints(
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    mask: torch.Tensor,
    ring_atom_indices: torch.Tensor | None = None,
    ring_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all applicable physics-informed constraint losses.

    Args:
        coords: (B, N, 3) predicted coordinates in normalized space.
        atom_types: (B, N) int64 atom type indices (may contain -1 for padding).
        mask: (B, N) float32 binary mask.
        ring_atom_indices: optional (B, MAX_RINGS, MAX_RING_SIZE) int64.
        ring_valid: optional (B, MAX_RINGS) float32 binary mask.

    Returns:
        Dict with keys:
            - ``"bond_length_loss"``: bond length penalty.
            - ``"bond_angle_loss"``: bond angle penalty (if enough bonds found).
            - ``"planarity_loss"``: planarity penalty (if ring info provided).
            - ``"total_constraint_loss"``: weighted sum
              (1.0 * bond + 0.5 * angle + 0.3 * planarity).
    """
    device = coords.device
    results: dict[str, torch.Tensor] = {}

    # Bond length loss (always computed)
    bl_loss = bond_length_penalty(coords, atom_types, mask)
    results["bond_length_loss"] = bl_loss

    # Bond angle loss
    ba_loss = bond_angle_penalty(coords, atom_types, mask)
    results["bond_angle_loss"] = ba_loss

    # Planarity loss (only if ring info provided)
    if ring_atom_indices is not None and ring_valid is not None:
        pl_loss = planarity_penalty(coords, ring_atom_indices, ring_valid)
        results["planarity_loss"] = pl_loss
    else:
        pl_loss = torch.tensor(0.0, device=device)

    # Weighted total
    total = 1.0 * bl_loss + 0.5 * ba_loss + 0.3 * pl_loss
    results["total_constraint_loss"] = total

    return results
