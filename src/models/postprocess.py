"""
Post-processing module for generated molecular structures.

Provides RDKit-based force field relaxation (MMFF94 / UFF) to refine
predicted atom coordinates.  If RDKit is not installed the module
degrades gracefully: ``rdkit_relaxation`` returns coordinates unchanged.

All coordinates are expected in NORMALIZED space (Angstroms / 12.0)
unless otherwise specified.
"""

import warnings
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Try to import RDKit
# ---------------------------------------------------------------------------

RDKIT_AVAILABLE = True

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn(
        "RDKit is not installed. rdkit_relaxation() will return coordinates "
        "unchanged. Install with: conda install -c conda-forge rdkit",
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_TYPES = ["H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I"]
COORD_SCALE = 12.0  # normalization factor (Angstroms -> normalized)

# Bond-length lookup table in Angstroms (single-bond maximums).
# Used with a 1.3x tolerance to infer bonds from pairwise distances.
_BOND_TOLERANCE = 1.3

_BOND_LENGTHS_ANG = {
    frozenset({"C", "C"}):  1.54,
    frozenset({"C", "H"}):  1.09,
    frozenset({"C", "N"}):  1.47,
    frozenset({"C", "O"}):  1.43,
    frozenset({"C", "S"}):  1.82,
    frozenset({"C", "F"}):  1.35,
    frozenset({"C", "Cl"}): 1.77,
    frozenset({"C", "Br"}): 1.94,
    frozenset({"C", "I"}):  2.14,
    frozenset({"C", "P"}):  1.84,
    frozenset({"N", "H"}):  1.01,
    frozenset({"N", "N"}):  1.45,
    frozenset({"N", "O"}):  1.40,
    frozenset({"O", "H"}):  0.96,
    frozenset({"O", "S"}):  1.58,
    frozenset({"S", "H"}):  1.34,
    frozenset({"S", "S"}):  2.05,
    frozenset({"P", "O"}):  1.63,
    frozenset({"P", "N"}):  1.68,
    frozenset({"P", "H"}):  1.44,
}

# ---------------------------------------------------------------------------
# coords_to_mol
# ---------------------------------------------------------------------------


def coords_to_mol(coords, atom_types, mask):
    """Convert predicted coordinates and atom types to an RDKit Mol object.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Atom positions in **normalized** space (will be multiplied by
        ``COORD_SCALE`` to recover Angstroms).
    atom_types : np.ndarray, shape (N,)
        Integer indices into :data:`ATOM_TYPES`.
    mask : np.ndarray, shape (N,)
        Binary mask indicating valid atoms (> 0 means valid).

    Returns
    -------
    mol : rdkit.Chem.RWMol or None
        The constructed molecule with 3D coordinates, or ``None`` if
        construction fails or RDKit is unavailable.
    """
    if not RDKIT_AVAILABLE:
        return None

    try:
        coords = np.asarray(coords, dtype=np.float64)
        atom_types = np.asarray(atom_types, dtype=np.int64)
        mask = np.asarray(mask, dtype=np.float64)

        # Filter to valid atoms
        valid = (mask > 0) & (atom_types >= 0) & (atom_types < len(ATOM_TYPES))
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            return None

        # Convert to Angstrom space
        coords_ang = coords[valid_indices] * COORD_SCALE
        elements = [ATOM_TYPES[atom_types[i]] for i in valid_indices]

        # Create RWMol and add atoms
        mol = Chem.RWMol()
        for elem in elements:
            atom = Chem.Atom(elem)
            mol.AddAtom(atom)

        # Infer bonds from pairwise distances
        n_atoms = len(valid_indices)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pair = frozenset({elements[i], elements[j]})
                max_len = _BOND_LENGTHS_ANG.get(pair)
                if max_len is None:
                    continue
                dist = np.linalg.norm(coords_ang[i] - coords_ang[j])
                if dist < _BOND_TOLERANCE * max_len:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Set 3D coordinates via a Conformer
        conf = Chem.Conformer(n_atoms)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, coords_ang[i].tolist())
        mol.AddConformer(conf, assignId=True)

        return mol

    except Exception:
        return None


# ---------------------------------------------------------------------------
# rdkit_relaxation
# ---------------------------------------------------------------------------


def rdkit_relaxation(coords, atom_types, mask, max_iters=200,
                     max_displacement=0.3):
    """Relax molecular structures using RDKit force fields.

    Attempts MMFF94 optimization first; falls back to UFF if MMFF94 fails.
    Per-atom displacements are capped at *max_displacement* Angstroms to
    prevent large distortions.

    Parameters
    ----------
    coords : torch.Tensor, shape (B, N, 3)
        Atom positions in **normalized** space.
    atom_types : torch.Tensor, shape (B, N)
        Integer atom-type indices (int64).
    mask : torch.Tensor, shape (B, N)
        Binary validity mask (float32).
    max_iters : int, optional
        Maximum force-field optimization iterations (default 200).
    max_displacement : float, optional
        Maximum allowed per-atom displacement in Angstroms (default 0.3).

    Returns
    -------
    relaxed : torch.Tensor, shape (B, N, 3)
        Relaxed coordinates in normalized space, on the same device as
        the input *coords*.
    """
    if not RDKIT_AVAILABLE:
        return coords

    try:
        device = coords.device
        coords_np = coords.detach().cpu().numpy().copy()
        types_np = atom_types.detach().cpu().numpy()
        mask_np = mask.detach().cpu().float().numpy()

        B, N, _ = coords_np.shape
        result = coords_np.copy()

        for b in range(B):
            try:
                mol = coords_to_mol(coords_np[b], types_np[b], mask_np[b])
                if mol is None:
                    continue

                # Try MMFF94 first, then UFF
                optimized = False
                try:
                    ret = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
                    if ret != -1:  # -1 means setup failure
                        optimized = True
                except Exception:
                    pass

                if not optimized:
                    try:
                        ret = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
                        if ret != -1:
                            optimized = True
                    except Exception:
                        pass

                if not optimized:
                    continue

                # Extract optimized coordinates
                conf = mol.GetConformer()
                valid = ((mask_np[b] > 0)
                         & (types_np[b] >= 0)
                         & (types_np[b] < len(ATOM_TYPES)))
                valid_indices = np.where(valid)[0]

                new_coords_ang = np.array(
                    [list(conf.GetAtomPosition(i))
                     for i in range(conf.GetNumAtoms())],
                    dtype=np.float64,
                )

                # Original coords in Angstrom space for the valid atoms
                orig_coords_ang = coords_np[b, valid_indices] * COORD_SCALE

                # Compute displacement and cap if necessary
                displacement = new_coords_ang - orig_coords_ang
                per_atom_dist = np.linalg.norm(displacement, axis=1)
                max_dist = per_atom_dist.max() if len(per_atom_dist) > 0 else 0.0

                if max_dist > max_displacement and max_dist > 0:
                    scale = max_displacement / max_dist
                    displacement *= scale

                # Write back in normalized space
                result[b, valid_indices] = (orig_coords_ang + displacement) / COORD_SCALE

            except Exception:
                # Keep original coordinates for this sample
                continue

        return torch.tensor(result, dtype=coords.dtype, device=device)

    except Exception:
        return coords
