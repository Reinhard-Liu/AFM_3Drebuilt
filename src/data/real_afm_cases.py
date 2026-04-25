"""
Real AFM case dataset for SUP-03.

This loader reads processed real-AFM case directories and returns samples in a
field layout that is intentionally close to QUAMAFMDataset, so later
evaluation/inference scripts can reuse the current V20 pipeline with minimal
adaptation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import MAX_ATOMS, compute_corrugation
from src.models.ring_detection import MAX_RING_SIZE, MAX_RINGS, detect_rings, pad_ring_info


class RealAFMCaseDataset(Dataset):
    def __init__(self, root: str | Path, contrast_variant: str = "normal"):
        self.root = Path(root)
        self.contrast_variant = str(contrast_variant)
        if self.contrast_variant not in {"normal", "inverted"}:
            raise ValueError("contrast_variant must be 'normal' or 'inverted'")

        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        self.cases: List[Dict] = list(self.manifest.get("cases", []))
        if not self.cases:
            raise ValueError(f"no cases found in manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.cases)

    @staticmethod
    def _empty_ring_info() -> Dict:
        return {
            "ring_centers": np.zeros((MAX_RINGS, 3), dtype=np.float32),
            "ring_normals": np.zeros((MAX_RINGS, 3), dtype=np.float32),
            "ring_types": np.zeros((MAX_RINGS,), dtype=np.int64),
            "ring_atom_indices": np.full((MAX_RINGS, MAX_RING_SIZE), -1, dtype=np.int64),
            "ring_templates": np.zeros((MAX_RINGS, MAX_RING_SIZE, 3), dtype=np.float32),
            "ring_valid": np.zeros((MAX_RINGS,), dtype=np.float32),
            "n_rings": 0,
        }

    def __getitem__(self, idx: int) -> Dict:
        case = self.cases[idx]
        case_dir = self.root / case["case_dir"]

        if self.contrast_variant == "inverted":
            afm_path = case_dir / "afm_stack_inverted.npy"
        else:
            afm_path = case_dir / "afm_stack.npy"

        afm_stack = np.load(afm_path).astype(np.float32)
        coords_path = case_dir / "coords_norm.npy"
        types_path = case_dir / "atom_types.npy"
        has_gt_files = coords_path.exists() and types_path.exists()
        elements = case.get("elements", [])

        raw_n_atoms = int(case.get("n_atoms", 0))
        gt_structure_compatible = False
        padded_coords = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
        padded_types = np.full(MAX_ATOMS, -1, dtype=np.int64)
        atom_mask = np.zeros(MAX_ATOMS, dtype=np.float32)
        padded_ring = self._empty_ring_info()
        corrugation = 0.0

        if has_gt_files:
            coords_norm = np.load(coords_path).astype(np.float32)
            atom_types = np.load(types_path).astype(np.int64)
            raw_n_atoms = int(atom_types.shape[0])
            if raw_n_atoms <= MAX_ATOMS:
                atom_mask[:raw_n_atoms] = 1.0
                padded_coords[:raw_n_atoms] = coords_norm
                padded_types[:raw_n_atoms] = atom_types
                ring_info = detect_rings(coords_norm, elements, normalized=True)
                padded_ring = pad_ring_info(ring_info, MAX_RINGS, MAX_RING_SIZE, MAX_ATOMS)
                corrugation = compute_corrugation(coords_norm * 12.0)
                gt_structure_compatible = True

        return {
            "afm_stack": torch.from_numpy(afm_stack),
            "coords": torch.from_numpy(padded_coords),
            "atom_types": torch.from_numpy(padded_types),
            "atom_mask": torch.from_numpy(atom_mask),
            "n_atoms": torch.tensor(raw_n_atoms, dtype=torch.long),
            "corrugation": torch.tensor(corrugation, dtype=torch.float32),
            "cid_idx": torch.tensor(-1, dtype=torch.long),
            "ring_centers": torch.from_numpy(padded_ring["ring_centers"]),
            "ring_normals": torch.from_numpy(padded_ring["ring_normals"]),
            "ring_types": torch.from_numpy(padded_ring["ring_types"]),
            "ring_atom_indices": torch.from_numpy(padded_ring["ring_atom_indices"]),
            "ring_templates": torch.from_numpy(padded_ring["ring_templates"]),
            "ring_valid": torch.from_numpy(padded_ring["ring_valid"]),
            "n_rings": torch.tensor(padded_ring["n_rings"], dtype=torch.long),
            "case_id": case["case_id"],
            "molecule_name": case["molecule_name"],
            "molecule_label": case.get("molecule_label", case["molecule_name"]),
            "tip": case["tip"],
            "source_npz": case["source_npz"],
            "has_gt_structure": bool(has_gt_files),
            "gt_structure_compatible": bool(gt_structure_compatible),
        }
