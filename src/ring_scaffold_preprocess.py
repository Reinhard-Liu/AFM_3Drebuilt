"""
V17 Ring Scaffold Preprocessing

Extracts rich ring scaffold labels from XYZ molecular structures using RDKit.
Produces per-atom and per-ring annotations suitable for PyTorch training.

Usage:
    python -m src.ring_scaffold_preprocess --data_root /path/to/QUAM --num_samples 100
"""

import argparse
import json
import os
import pickle
import numpy as np
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_ATOMS = 85
MAX_RINGS = 10
MAX_RING_SIZE = 6
MAX_RING_MEMBERSHIP = 3  # max rings per atom (fused systems)
MAX_FUSIONS = 10

ELEMENT_TO_IDX = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "S": 5, "P": 6, "Cl": 7, "Br": 8, "I": 9}

RING_TYPE_MAP = {
    # 6-membered
    ("C", "C", "C", "C", "C", "C"): "benzene",        # all C, aromatic
    ("C", "C", "C", "C", "C", "N"): "pyridine",        # one N
    ("C", "C", "C", "C", "N", "N"): "pyrimidine",      # two N
    ("C", "C", "C", "N", "N", "N"): "triazine",
    # 5-membered
    ("C", "C", "C", "C", "O"): "furan",
    ("C", "C", "C", "C", "S"): "thiophene",
    ("C", "C", "C", "C", "N"): "pyrrole",
    ("C", "C", "C", "N", "N"): "imidazole",
    ("C", "C", "C", "N", "O"): "oxazole",
    ("C", "C", "C", "N", "S"): "thiazole",
}


def classify_ring(elements_in_ring, is_aromatic):
    """Classify a ring by its element composition and aromaticity."""
    sorted_elems = tuple(sorted(elements_in_ring))
    size = len(sorted_elems)

    if size == 6:
        if sorted_elems == ("C", "C", "C", "C", "C", "C"):
            return "benzene" if is_aromatic else "cyclohexane"
        name = RING_TYPE_MAP.get(sorted_elems, "other_6")
        return name
    elif size == 5:
        if sorted_elems == ("C", "C", "C", "C", "C"):
            return "cyclopentane" if not is_aromatic else "cyclopentadienyl"
        name = RING_TYPE_MAP.get(sorted_elems, "other_5")
        return name
    return f"other_{size}"


RING_TYPE_TO_IDX = {
    "benzene": 0, "pyridine": 1, "pyrimidine": 2, "triazine": 3,
    "furan": 4, "thiophene": 5, "pyrrole": 6, "imidazole": 7,
    "oxazole": 8, "thiazole": 9,
    "cyclohexane": 10, "cyclopentane": 11, "cyclopentadienyl": 12,
    "other_5": 13, "other_6": 14,
}


def parse_xyz_file(xyz_path):
    """Parse XYZ file, return elements list and coords (N, 3) in Angstrom."""
    with open(xyz_path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    elements = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.array(coords, dtype=np.float64)


def xyz_to_rdkit_mol(elements, coords):
    """Build RDKit Mol from elements + 3D coordinates using DetermineBonds."""
    n = len(elements)
    rwmol = Chem.RWMol()
    conf = Chem.Conformer(n)
    for i, (elem, c) in enumerate(zip(elements, coords)):
        atom = Chem.Atom(elem)
        rwmol.AddAtom(atom)
        conf.SetAtomPosition(i, c.tolist())
    rwmol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineBonds(rwmol)
    except Exception:
        rdDetermineBonds.DetermineBonds(rwmol, allowChargedFragments=True)
    return rwmol.GetMol()


def extract_ring_scaffold(mol, coords_norm):
    """Extract full ring scaffold labels from an RDKit mol.

    Args:
        mol: RDKit Mol with bonds determined
        coords_norm: (N, 3) normalized coordinates (Angstrom / 12.0)

    Returns:
        dict with all ring scaffold fields, padded to MAX_* constants.
    """
    n_atoms = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    atom_rings_raw = ri.AtomRings()

    # Filter to 5/6-membered rings only
    atom_rings = [list(r) for r in atom_rings_raw if len(r) in (5, 6)]
    n_rings = min(len(atom_rings), MAX_RINGS)
    atom_rings = atom_rings[:n_rings]

    # --- Per-ring fields ---
    ring_size = np.zeros(MAX_RINGS, dtype=np.int64)
    ring_aromaticity = np.zeros(MAX_RINGS, dtype=np.float32)
    ring_type = np.full(MAX_RINGS, -1, dtype=np.int64)
    ring_center = np.zeros((MAX_RINGS, 3), dtype=np.float32)
    ring_normal = np.zeros((MAX_RINGS, 3), dtype=np.float32)
    ring_valid = np.zeros(MAX_RINGS, dtype=np.float32)
    ring_atom_indices = np.full((MAX_RINGS, MAX_RING_SIZE), -1, dtype=np.int64)
    ring_attachment = np.zeros((MAX_RINGS, MAX_RING_SIZE), dtype=np.float32)

    for ri_idx, ring_atoms in enumerate(atom_rings):
        ring_valid[ri_idx] = 1.0
        size = len(ring_atoms)
        ring_size[ri_idx] = size

        # Fill ring_atom_indices
        for si, ai in enumerate(ring_atoms[:MAX_RING_SIZE]):
            ring_atom_indices[ri_idx, si] = ai

        # Ring element composition + aromaticity
        elems = [mol.GetAtomWithIdx(a).GetSymbol() for a in ring_atoms]
        is_arom = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring_atoms)
        ring_aromaticity[ri_idx] = float(is_arom)

        # Classify ring type
        rtype = classify_ring(elems, is_arom)
        ring_type[ri_idx] = RING_TYPE_TO_IDX.get(rtype, 14)

        # Ring center (in normalized space)
        ring_coords = coords_norm[ring_atoms]
        center = ring_coords.mean(axis=0)
        ring_center[ri_idx] = center

        # Ring normal via SVD
        centered = ring_coords - center
        if len(ring_atoms) >= 3:
            _, _, Vh = np.linalg.svd(centered, full_matrices=False)
            normal = Vh[-1]
            # Ensure consistent orientation (normal pointing +Z)
            if normal[2] < 0:
                normal = -normal
            ring_normal[ri_idx] = normal

        # Attachment sites: ring atoms bonded to non-ring atoms
        ring_set = set(ring_atoms)
        for si, ai in enumerate(ring_atoms[:MAX_RING_SIZE]):
            atom = mol.GetAtomWithIdx(ai)
            for nb in atom.GetNeighbors():
                if nb.GetIdx() not in ring_set:
                    ring_attachment[ri_idx, si] = 1.0
                    break

    # --- Per-atom fields ---
    atom_to_ring_ids = np.full((MAX_ATOMS, MAX_RING_MEMBERSHIP), -1, dtype=np.int64)
    ring_site_index = np.full(MAX_ATOMS, -1, dtype=np.int64)
    is_ring_atom = np.zeros(MAX_ATOMS, dtype=np.float32)
    is_scaffold_atom = np.zeros(MAX_ATOMS, dtype=np.float32)

    for ri_idx, ring_atoms in enumerate(atom_rings):
        for site_idx, atom_idx in enumerate(ring_atoms):
            if atom_idx >= MAX_ATOMS:
                continue
            is_ring_atom[atom_idx] = 1.0
            is_scaffold_atom[atom_idx] = 1.0
            # Add ring membership
            for slot in range(MAX_RING_MEMBERSHIP):
                if atom_to_ring_ids[atom_idx, slot] == -1:
                    atom_to_ring_ids[atom_idx, slot] = ri_idx
                    break
            # Site index: position within the ring (first ring wins for fused atoms)
            if ring_site_index[atom_idx] == -1:
                ring_site_index[atom_idx] = site_idx

    # --- Ring fusion edges ---
    ring_fusion_edges = np.full((MAX_FUSIONS, 2), -1, dtype=np.int64)
    ring_fusion_atoms = np.full((MAX_FUSIONS, 2), -1, dtype=np.int64)
    n_fusions = 0

    for i in range(n_rings):
        for j in range(i + 1, n_rings):
            shared = set(atom_rings[i]) & set(atom_rings[j])
            if shared and n_fusions < MAX_FUSIONS:
                ring_fusion_edges[n_fusions] = [i, j]
                shared_list = sorted(shared)
                ring_fusion_atoms[n_fusions, 0] = shared_list[0]
                if len(shared_list) > 1:
                    ring_fusion_atoms[n_fusions, 1] = shared_list[1]
                # Mark fusion atoms as scaffold atoms
                for a in shared:
                    if a < MAX_ATOMS:
                        is_scaffold_atom[a] = 1.0
                n_fusions += 1

    return {
        # Per-ring
        "n_rings": n_rings,
        "ring_size": ring_size,
        "ring_aromaticity": ring_aromaticity,
        "ring_type": ring_type,
        "ring_center": ring_center,
        "ring_normal": ring_normal,
        "ring_valid": ring_valid,
        "ring_atom_indices": ring_atom_indices,
        "ring_attachment": ring_attachment,
        # Per-atom
        "atom_to_ring_ids": atom_to_ring_ids,
        "ring_site_index": ring_site_index,
        "is_ring_atom": is_ring_atom,
        "is_scaffold_atom": is_scaffold_atom,
        # Topology
        "ring_fusion_edges": ring_fusion_edges,
        "ring_fusion_atoms": ring_fusion_atoms,
        "n_fusions": n_fusions,
    }


def process_molecule(xyz_path):
    """Full pipeline: XYZ → RDKit mol → ring scaffold labels."""
    elements, coords = parse_xyz_file(xyz_path)
    coords_norm = coords / 12.0

    try:
        mol = xyz_to_rdkit_mol(elements, coords)
    except Exception as e:
        return None, str(e)

    scaffold = extract_ring_scaffold(mol, coords_norm)
    scaffold["elements"] = elements
    scaffold["n_atoms"] = len(elements)
    return scaffold, None


def main():
    parser = argparse.ArgumentParser(description="V17 Ring Scaffold Preprocessing")
    parser.add_argument("--data_root", default="/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output", default="experiments/v17_ring_scaffold/scaffold_samples.pkl")
    args = parser.parse_args()

    xyz_dir = Path(args.data_root) / "XYZ_FILES"
    cid_dirs = sorted(os.listdir(xyz_dir))[:args.num_samples]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    errors = 0
    for i, cid in enumerate(cid_dirs):
        xyz_path = xyz_dir / cid / f"{cid}.xyz"
        if not xyz_path.exists():
            continue
        scaffold, err = process_molecule(str(xyz_path))
        if err:
            errors += 1
            if errors <= 5:
                print(f"  Error on {cid}: {err}")
            continue

        scaffold["cid"] = cid
        results.append(scaffold)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(cid_dirs)}, {len(results)} OK, {errors} errors")

    print(f"\nDone: {len(results)} molecules processed, {errors} errors")

    # Save
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved to {args.output}")

    # Print summary stats
    n_rings_list = [s["n_rings"] for s in results]
    n_fusions_list = [s["n_fusions"] for s in results]
    n_ring_atoms_list = [int(s["is_ring_atom"][:s["n_atoms"]].sum()) for s in results]
    n_scaffold_list = [int(s["is_scaffold_atom"][:s["n_atoms"]].sum()) for s in results]

    print(f"\n--- Ring Scaffold Statistics ---")
    print(f"  Molecules: {len(results)}")
    print(f"  Avg rings/mol: {np.mean(n_rings_list):.1f} (max {max(n_rings_list)})")
    print(f"  Avg fusions/mol: {np.mean(n_fusions_list):.1f}")
    print(f"  Avg ring atoms/mol: {np.mean(n_ring_atoms_list):.1f}")
    print(f"  Avg scaffold atoms/mol: {np.mean(n_scaffold_list):.1f}")
    print(f"  Molecules with 0 rings: {sum(1 for n in n_rings_list if n == 0)}")
    print(f"  Molecules with fused rings: {sum(1 for n in n_fusions_list if n > 0)}")

    # Ring type distribution
    type_counts = {}
    for s in results:
        for ri in range(s["n_rings"]):
            rtype = s["ring_type"][ri]
            type_counts[rtype] = type_counts.get(rtype, 0) + 1
    idx_to_name = {v: k for k, v in RING_TYPE_TO_IDX.items()}
    print(f"\n  Ring type distribution:")
    for idx in sorted(type_counts.keys()):
        name = idx_to_name.get(idx, f"type_{idx}")
        print(f"    {name:<20}: {type_counts[idx]}")

    # Export a few samples as JSON for inspection
    json_path = args.output.replace(".pkl", "_sample.json")
    sample_out = []
    for s in results[:5]:
        entry = {
            "cid": s["cid"],
            "n_atoms": s["n_atoms"],
            "elements": s["elements"],
            "n_rings": s["n_rings"],
            "n_fusions": s["n_fusions"],
            "ring_sizes": s["ring_size"][:s["n_rings"]].tolist(),
            "ring_aromaticity": s["ring_aromaticity"][:s["n_rings"]].tolist(),
            "ring_types": [idx_to_name.get(int(s["ring_type"][i]), "?") for i in range(s["n_rings"])],
            "ring_atom_indices": [s["ring_atom_indices"][i][:s["ring_size"][i]].tolist() for i in range(s["n_rings"])],
            "ring_fusion_pairs": s["ring_fusion_edges"][:s["n_fusions"]].tolist(),
            "ring_fusion_shared_atoms": s["ring_fusion_atoms"][:s["n_fusions"]].tolist(),
            "atom_ring_membership": {
                str(a): s["atom_to_ring_ids"][a][s["atom_to_ring_ids"][a] >= 0].tolist()
                for a in range(s["n_atoms"]) if s["is_ring_atom"][a] > 0
            },
        }
        sample_out.append(entry)
    with open(json_path, "w") as f:
        json.dump(sample_out, f, indent=2)
    print(f"  Sample JSON: {json_path}")


if __name__ == "__main__":
    main()
