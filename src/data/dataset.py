"""
QUAM-AFM Dataset: loads AFM image stacks and molecular XYZ coordinates.
Each sample = (10-frame AFM image stack, atom coordinates + types).
"""

import os
import glob
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError

from src.models.ring_detection import (
    detect_rings, pad_ring_info, compute_ring_system_scaffold_labels,
    MAX_RINGS, MAX_RING_SIZE,
)

# Atom type mapping (elements present in QUAM-AFM)
ATOM_TYPES = ["H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I"]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Max atoms per molecule (for padding)
MAX_ATOMS = 85


def parse_xyz(xyz_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Parse an extended XYZ file.

    Returns:
        coords: (N, 3) float array of atomic coordinates
        atomic_numbers: (N,) int array
        elements: list of element symbols
    """
    coords = []
    elements = []
    with open(xyz_path, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    # lines[1] is the comment/header line
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        elem = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        elements.append(elem)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=np.float32)
    return coords, elements


def compute_corrugation(coords: np.ndarray) -> float:
    """Compute the Z-axis corrugation (max_z - min_z) of a molecule."""
    return float(coords[:, 2].max() - coords[:, 2].min())


def center_coords(coords: np.ndarray) -> np.ndarray:
    """Center coordinates to have zero mean."""
    return coords - coords.mean(axis=0, keepdims=True)


def random_xy_rotation_angle() -> float:
    """Generate a random rotation angle around Z-axis (XY-plane rotation).

    XY-plane rotation is the only augmentation consistent with AFM top-down images:
    rotating both the AFM image and coordinates by the same angle preserves
    the input-output correspondence. 3D tilts would require PPM re-rendering.
    """
    return np.random.uniform(0, 2 * np.pi)


def apply_xy_rotation(coords: np.ndarray, angle: float) -> np.ndarray:
    """Rotate coordinates around Z-axis by given angle."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return coords @ R.T


def rotate_afm_stack(afm_stack: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate each AFM image slice by the same angle (PIL rotation)."""
    from scipy.ndimage import rotate as ndimage_rotate
    rotated = np.empty_like(afm_stack)
    for i in range(afm_stack.shape[0]):
        rotated[i] = ndimage_rotate(afm_stack[i], angle_deg, reshape=False,
                                     order=1, mode='constant', cval=0.0)
    return rotated


class QUAMAFMDataset(Dataset):
    """Dataset for QUAM-AFM: AFM image stacks + molecular coordinates.

    Args:
        data_root: path to SUBMIT_QUAM-AFM/QUAM/
        param_key: parameter folder name, e.g. "K-7"
        img_size: resize images to this size
        min_corrugation: minimum corrugation threshold in Angstrom
        augment_rotation: whether to apply 3D rotation augmentation
        require_ring: if True, only keep molecules with 5/6-membered rings
        split: 'train' or 'val' or 'test'
        train_ratio: fraction for training
        val_ratio: fraction for validation
    """

    def __init__(
        self,
        data_root: str,
        param_key: str = "K-1",
        img_size: int = 128,
        min_corrugation: float = 0.0,
        augment_rotation: bool = False,
        require_ring: bool = False,
        split: str = "train",
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        val_size: int = 0,
        max_samples: int = 0,
        cid_to_idx: Optional[Dict[str, int]] = None,
        return_v17_bridge_labels: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.param_key = param_key
        self.img_size = img_size
        self.min_corrugation = min_corrugation
        self.augment_rotation = augment_rotation
        self.require_ring = require_ring
        self.split = split
        self.return_v17_bridge_labels = return_v17_bridge_labels

        self.afm_dir = self.data_root / param_key
        self.xyz_dir = self.data_root / "XYZ_FILES"

        # Collect all valid molecule CIDs (with pkl cache)
        cache_path = self.data_root / f"samples_cache_{param_key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            # Rebase cached paths so they are valid under the current data_root
            self._rebase_paths()
            print(f"Loaded {len(self.samples)} samples from cache: {cache_path}")
        else:
            self.samples = self._collect_samples()
            with open(cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            print(f"Saved {len(self.samples)} samples to cache: {cache_path}")

        # Filter by corrugation
        if min_corrugation > 0:
            self.samples = [
                s for s in self.samples
                if s["corrugation"] >= min_corrugation
            ]

        # Filter by ring presence (only keep molecules with 5/6-membered rings)
        if require_ring:
            self.samples = self._filter_by_ring()

        # Sort for reproducibility
        self.samples.sort(key=lambda s: s["cid"])

        # Split
        n = len(self.samples)
        if val_size > 0:
            n_val = val_size
            n_test = val_size
            n_train = n - n_val - n_test
        else:
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:n_train + n_val]
        else:  # test
            self.samples = self.samples[n_train + n_val:]

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        # Build or use provided CID-to-index mapping
        if cid_to_idx is not None:
            self.cid_to_idx = cid_to_idx
        else:
            # Build mapping from this split's samples
            self.cid_to_idx = {s["cid"]: i for i, s in enumerate(self.samples)}

    def _rebase_paths(self):
        """Rebase xyz_path and afm_folder stored in cache to current data_root.

        The cache may have been built with a different (e.g. relative) data_root.
        We extract the CID from each entry and reconstruct absolute paths using
        the current self.xyz_dir / self.afm_dir.
        """
        first = self.samples[0] if self.samples else None
        if first is None:
            return
        # Check if the stored path already resolves; if so, skip rebasing
        if Path(first["xyz_path"]).exists():
            return
        for s in self.samples:
            cid = s["cid"]
            s["xyz_path"] = str(self.xyz_dir / cid / f"{cid}.xyz")
            # Reconstruct afm_folder from the original basename
            afm_basename = Path(s["afm_folder"]).name
            s["afm_folder"] = str(self.afm_dir / afm_basename)

    def _filter_by_ring(self) -> List[Dict]:
        """Filter samples to only keep molecules containing 5/6-membered rings.

        Uses a separate ring cache file to avoid re-scanning every time.
        """
        ring_cache_path = self.data_root / f"ring_cache_{self.param_key}.pkl"

        if ring_cache_path.exists():
            with open(ring_cache_path, "rb") as f:
                ring_cache = pickle.load(f)
            print(f"Loaded ring cache: {len(ring_cache)} entries from {ring_cache_path}")
        else:
            ring_cache = {}

        # Check which samples need ring detection
        need_detect = [s for s in self.samples if s["cid"] not in ring_cache]
        if need_detect:
            print(f"Detecting rings for {len(need_detect)} molecules (one-time cost)...")
            for i, s in enumerate(need_detect):
                try:
                    xyz_path = s["xyz_path"]
                    if not Path(xyz_path).exists():
                        # Try rebased path
                        xyz_path = str(self.xyz_dir / s["cid"] / f"{s['cid']}.xyz")
                    coords, elements = parse_xyz(xyz_path)
                    coords_norm = coords / 12.0
                    ring_info = detect_rings(coords_norm, elements, normalized=True)
                    ring_cache[s["cid"]] = ring_info["n_rings"] > 0
                except Exception:
                    ring_cache[s["cid"]] = False

                if (i + 1) % 10000 == 0:
                    print(f"  Ring detection: {i+1}/{len(need_detect)} ...")

            # Save updated ring cache
            with open(ring_cache_path, "wb") as f:
                pickle.dump(ring_cache, f)
            print(f"Saved ring cache: {len(ring_cache)} entries to {ring_cache_path}")

        # Filter
        before = len(self.samples)
        filtered = [s for s in self.samples if ring_cache.get(s["cid"], False)]
        print(f"Ring filter: {before} → {len(filtered)} molecules "
              f"({100*len(filtered)/before:.1f}% have 5/6-membered rings)")
        return filtered

    def _collect_samples(self) -> List[Dict]:
        """Scan the dataset and collect valid samples.
        Only stores paths and corrugation — coords/elements are loaded lazily.
        """
        # Build AFM folder index: CID -> folder path (avoid repeated glob)
        afm_index = {}
        for name in os.listdir(self.afm_dir):
            # Format: Conformer3D_CID_{CID}_K040_Amp040
            parts = name.split("_")
            if len(parts) >= 4 and parts[0] == "Conformer3D" and parts[1] == "CID":
                cid = parts[2]
                afm_index[cid] = str(self.afm_dir / name)
        print(f"AFM index built: {len(afm_index)} entries")

        samples = []
        xyz_dirs = sorted(os.listdir(self.xyz_dir))
        n_total = len(xyz_dirs)

        for i, cid_name in enumerate(xyz_dirs):
            if i % 50000 == 0:
                print(f"  Scanning XYZ: {i}/{n_total} ...")

            # Quick lookup instead of glob
            if cid_name not in afm_index:
                continue

            xyz_path = self.xyz_dir / cid_name / f"{cid_name}.xyz"
            if not xyz_path.exists():
                continue

            # Only parse to get corrugation, don't keep coords in memory
            try:
                coords, elements = parse_xyz(str(xyz_path))
            except Exception:
                continue

            corrugation = compute_corrugation(coords)

            samples.append({
                "cid": cid_name,
                "xyz_path": str(xyz_path),
                "afm_folder": afm_index[cid_name],
                "corrugation": corrugation,
            })

        print(f"Collected {len(samples)} valid samples")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # --- Load AFM image stack (10 frames) ---
        afm_folder = Path(sample["afm_folder"])
        images = []
        for i in range(10):
            # Files are named *_df_000.jpg to *_df_009.jpg
            pattern = list(afm_folder.glob(f"*_df_{i:03d}.jpg"))
            if not pattern:
                pattern = list(afm_folder.glob(f"*_df_{i:03d}.png"))
            if pattern:
                try:
                    with Image.open(pattern[0]) as img:
                        img = img.convert("L")
                        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                        img_arr = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_arr)
                except (UnidentifiedImageError, OSError, ValueError):
                    # Keep long-running training robust against a small number of
                    # corrupted AFM frames in the raw dataset.
                    images.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))
            else:
                images.append(np.zeros((self.img_size, self.img_size), dtype=np.float32))

        # Stack: (D=10, H, W)
        afm_stack = np.stack(images, axis=0)

        # --- Coordinates and atom types (lazy load from xyz file) ---
        coords, elements = parse_xyz(sample["xyz_path"])

        # Apply consistent XY rotation augmentation (rotates both AFM and coords)
        if self.augment_rotation and self.split == "train":
            angle = random_xy_rotation_angle()
            coords = center_coords(coords)
            coords = apply_xy_rotation(coords, angle)
            afm_stack = rotate_afm_stack(afm_stack, np.degrees(angle))

        # Center coordinates
        coords = center_coords(coords)

        # Normalize coordinates to [-1, 1] range (based on cell size 24 Å)
        coords = coords / 12.0  # half of 24 Å cell

        # Atom type indices
        atom_types = np.array(
            [ATOM_TO_IDX.get(e, ATOM_TO_IDX["C"]) for e in elements],
            dtype=np.int64,
        )

        n_atoms = len(elements)

        # Ring detection on normalized coordinates
        ring_info = detect_rings(coords, elements, normalized=True)
        padded_ring = pad_ring_info(ring_info, MAX_RINGS, MAX_RING_SIZE, MAX_ATOMS)

        # V17-Bridge: optional ring-system scaffold labels (kept off by default
        # so existing training and evaluation paths are unaffected).
        scaffold_labels = None
        if self.return_v17_bridge_labels:
            scaffold_labels = compute_ring_system_scaffold_labels(coords, elements, MAX_ATOMS)

        # Pad to MAX_ATOMS
        padded_coords = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
        padded_types = np.full(MAX_ATOMS, -1, dtype=np.int64)  # -1 = padding
        atom_mask = np.zeros(MAX_ATOMS, dtype=np.float32)

        padded_coords[:n_atoms] = coords
        padded_types[:n_atoms] = atom_types
        atom_mask[:n_atoms] = 1.0

        # CID index for retrieval (-1 if not in mapping)
        cid = sample["cid"]
        cid_idx = self.cid_to_idx.get(cid, -1)

        out = {
            "afm_stack": torch.from_numpy(afm_stack),          # (10, H, W)
            "coords": torch.from_numpy(padded_coords),          # (MAX_ATOMS, 3)
            "atom_types": torch.from_numpy(padded_types),        # (MAX_ATOMS,)
            "atom_mask": torch.from_numpy(atom_mask),            # (MAX_ATOMS,)
            "n_atoms": torch.tensor(n_atoms, dtype=torch.long),
            "corrugation": torch.tensor(sample["corrugation"], dtype=torch.float32),
            "cid_idx": torch.tensor(cid_idx, dtype=torch.long),
            # Ring info
            "ring_centers": torch.from_numpy(padded_ring["ring_centers"]),        # (MAX_RINGS, 3)
            "ring_normals": torch.from_numpy(padded_ring["ring_normals"]),        # (MAX_RINGS, 3)
            "ring_types": torch.from_numpy(padded_ring["ring_types"]),            # (MAX_RINGS,)
            "ring_atom_indices": torch.from_numpy(padded_ring["ring_atom_indices"]),  # (MAX_RINGS, MAX_RING_SIZE)
            "ring_templates": torch.from_numpy(padded_ring["ring_templates"]),    # (MAX_RINGS, MAX_RING_SIZE, 3)
            "ring_valid": torch.from_numpy(padded_ring["ring_valid"]),            # (MAX_RINGS,)
            "n_rings": torch.tensor(padded_ring["n_rings"], dtype=torch.long),
        }

        if scaffold_labels is not None:
            for key, value in scaffold_labels.items():
                if isinstance(value, np.ndarray):
                    out[key] = torch.from_numpy(value)
                elif isinstance(value, (int, np.integer)):
                    out[key] = torch.tensor(value, dtype=torch.long)
                else:
                    out[key] = torch.tensor(value)

        return out


def create_dataloaders(
    data_root: str,
    param_key: str = "K-1",
    img_size: int = 128,
    min_corrugation: float = 0.0,
    augment_rotation: bool = True,
    require_ring: bool = False,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: int = 0,
    val_size: int = 0,
    return_v17_bridge_labels: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Create train/val/test dataloaders.

    Returns:
        train_loader, val_loader, test_loader, num_cids
    """

    # Build training set first to establish CID mapping
    train_ds = QUAMAFMDataset(
        data_root, param_key, img_size, min_corrugation,
        augment_rotation=augment_rotation, require_ring=require_ring,
        split="train",
        max_samples=max_samples, val_size=val_size,
        return_v17_bridge_labels=return_v17_bridge_labels,
    )
    cid_to_idx = train_ds.cid_to_idx
    num_cids = len(cid_to_idx)

    # Pass training CID mapping to val/test sets
    val_ds = QUAMAFMDataset(
        data_root, param_key, img_size, min_corrugation,
        augment_rotation=False, require_ring=require_ring,
        split="val",
        max_samples=max_samples, val_size=val_size,
        cid_to_idx=cid_to_idx,
        return_v17_bridge_labels=return_v17_bridge_labels,
    )
    test_ds = QUAMAFMDataset(
        data_root, param_key, img_size, min_corrugation,
        augment_rotation=False, require_ring=require_ring,
        split="test",
        max_samples=max_samples, val_size=val_size,
        cid_to_idx=cid_to_idx,
        return_v17_bridge_labels=return_v17_bridge_labels,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, num_cids
