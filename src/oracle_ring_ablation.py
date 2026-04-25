"""
V17 Oracle Ring Scaffold Ablation

Compares generation quality with and without GT ring scaffold information.
Uses V16c best_gen.pt as the base model.

Configs:
  Baseline: C_pred_guided (pred count, physics guidance, no ring snap)
  Oracle A: C_pred_guided + GT ring Procrustes snap every 5 DDIM steps
  Oracle B: C_pred_guided + GT ring planarity + distance constraint (soft)
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, ".")
from src.train import AFM3DReconModel, evaluate_generation, run_sanity_evals
from src.data.dataset import QUAMAFMDataset
from src.utils.metrics import (
    compute_rmsd, compute_bond_validity, compute_atom_count_accuracy,
    compute_bottom_atom_recall, compute_bottom_atom_rmsd,
    compute_structure_similarity, compute_composite_score,
)
from src.eval_phase1 import compute_ring_preservation


def oracle_ring_snap_generate(model, batch, ddim_steps=50):
    """Generate with GT ring Procrustes snapping during DDIM sampling.

    This bypasses the ring_head entirely and uses GT ring_atom_indices
    + ring_templates to project ring atoms to standard geometry every 5 steps.
    """
    afm = batch["afm_stack"]
    c_global, c_patches = model.encoder(afm)
    n_atoms = model.count_head.predict(c_global)  # pred count

    # Use GT ring info for snapping
    ring_info = None
    if "ring_atom_indices" in batch and "ring_templates" in batch and "ring_valid" in batch:
        ring_info = {
            "ring_atom_indices": batch["ring_atom_indices"],
            "ring_templates": batch["ring_templates"],
            "ring_valid": batch["ring_valid"],
        }

    use_physics = model.config.get("physics_guidance", True)

    coords, type_logits = model.ddpm.sample(
        c_global, c_patches, n_atoms, max_atoms=85,
        ring_info=ring_info,           # GT ring info for Procrustes snap
        predicted_rings=None,
        use_ddim=True, ddim_steps=ddim_steps,
        use_physics_guidance=use_physics,
        target_shape=None,
        disable_guidance=False,
        disable_ring_snap=False,       # Enable ring snap
    )

    return {
        "coords": coords,
        "type_logits": type_logits,
        "n_atoms_pred": n_atoms,
    }


def oracle_ring_constraint_generate(model, batch, ddim_steps=50):
    """Generate with soft GT ring planarity constraint during sampling.

    Instead of Procrustes snap, applies softer per-step corrections:
    1. Project ring atoms toward the ring plane (planarity)
    2. Pull ring atom distances toward ideal ring bond lengths
    """
    afm = batch["afm_stack"]
    c_global, c_patches = model.encoder(afm)
    n_atoms = model.count_head.predict(c_global)

    use_physics = model.config.get("physics_guidance", True)

    # First generate normally with physics guidance but no ring snap
    coords, type_logits = model.ddpm.sample(
        c_global, c_patches, n_atoms, max_atoms=85,
        ring_info=None,
        predicted_rings=None,
        use_ddim=True, ddim_steps=ddim_steps,
        use_physics_guidance=use_physics,
        target_shape=None,
        disable_guidance=False,
        disable_ring_snap=True,
    )

    # Post-hoc soft ring correction using GT ring info
    if "ring_atom_indices" in batch and "ring_valid" in batch:
        coords = _apply_soft_ring_correction(
            coords, type_logits, batch, n_atoms
        )

    return {
        "coords": coords,
        "type_logits": type_logits,
        "n_atoms_pred": n_atoms,
    }


def _apply_soft_ring_correction(coords, type_logits, batch, n_atoms):
    """Apply soft planarity + distance correction to ring atoms using GT info."""
    B, N, _ = coords.shape
    device = coords.device
    coords = coords.clone()

    ring_atom_indices = batch["ring_atom_indices"]  # (B, MAX_RINGS, MAX_RING_SIZE)
    ring_valid = batch["ring_valid"]                # (B, MAX_RINGS)

    IDEAL_RING_BOND = 0.128  # C-C aromatic bond in normalized space

    for b in range(B):
        for ri in range(ring_atom_indices.shape[1]):
            if ring_valid[b, ri].item() < 0.5:
                continue

            idx = ring_atom_indices[b, ri].long()
            valid = idx >= 0
            valid_idx = idx[valid]
            n_ring = valid.sum().item()
            if n_ring < 5:
                continue

            # Check if atoms are within n_atoms
            n_pred = int(n_atoms[b].item()) if n_atoms is not None else N
            valid_idx = valid_idx[valid_idx < n_pred]
            if len(valid_idx) < 5:
                continue

            ring_coords = coords[b, valid_idx]  # (K, 3)

            # 1. Planarity correction: project to best-fit plane
            center = ring_coords.mean(0, keepdim=True)
            centered = ring_coords - center
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            normal = Vh[-1]  # plane normal
            projections = (centered * normal).sum(-1, keepdim=True)  # distance from plane
            # Move 30% toward the plane
            ring_coords = ring_coords - 0.3 * projections * normal

            # 2. Distance correction: pull adjacent ring atoms toward ideal bond length
            for k in range(len(valid_idx)):
                k_next = (k + 1) % len(valid_idx)
                diff = ring_coords[k_next] - ring_coords[k]
                dist = diff.norm() + 1e-8
                correction = 0.1 * (dist - IDEAL_RING_BOND) / dist * diff
                ring_coords[k] = ring_coords[k] + correction
                ring_coords[k_next] = ring_coords[k_next] - correction

            coords[b, valid_idx] = ring_coords

    return coords


def evaluate_config(model, loader, device, generate_fn, name, num_samples=200):
    """Evaluate a generation configuration."""
    model.eval()
    all_rmsd, all_bond_gt, all_bond_pred = [], [], []
    all_count_exact, all_type_match, all_bottom_recall = [], [], []
    all_ring_pres = []
    count = 0

    pbar = tqdm(loader, desc=name, leave=False)
    with torch.no_grad():
        for batch in pbar:
            if count >= num_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            result = generate_fn(model, batch)
            coords_pred = result["coords"]
            type_logits = result["type_logits"]
            pred_types = type_logits.argmax(dim=-1)
            n_pred = result["n_atoms_pred"]
            B = coords_pred.shape[0]
            N = coords_pred.shape[1]

            rmsd = compute_rmsd(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
            bond_gt = compute_bond_validity(coords_pred, pred_types, batch["atom_mask"])
            idx = torch.arange(N, device=device).unsqueeze(0).float()
            pred_mask = (idx < n_pred.float().unsqueeze(1)).float()
            bond_pred = compute_bond_validity(coords_pred, pred_types, pred_mask)
            count_acc = compute_atom_count_accuracy(n_pred, batch["n_atoms"])
            bottom_recall = compute_bottom_atom_recall(
                coords_pred, batch["coords"], pred_types, batch["atom_types"], batch["atom_mask"])
            struct_sim = compute_structure_similarity(
                coords_pred, batch["coords"], pred_types, batch["atom_types"],
                batch["atom_mask"], n_atoms_pred=n_pred)
            ring_pres = compute_ring_preservation(
                coords_pred, batch["coords"], pred_types, batch["atom_types"],
                batch["atom_mask"], n_atoms_pred=n_pred)

            all_rmsd.append(rmsd)
            all_bond_gt.append(bond_gt)
            all_bond_pred.append(bond_pred)
            all_count_exact.append(count_acc["exact_match"])
            all_type_match.append(struct_sim["type_match_rate_mean"])
            all_bottom_recall.append(bottom_recall)
            all_ring_pres.append(ring_pres["ring_preservation_mean"])
            count += B

    all_rmsd = torch.cat(all_rmsd)
    all_bond_gt = torch.cat(all_bond_gt)
    all_bond_pred = torch.cat(all_bond_pred)
    all_bottom_recall = torch.cat(all_bottom_recall)

    rmsd_mean = all_rmsd.mean().item()
    bond_gt_mean = all_bond_gt.mean().item()
    bond_pred_mean = all_bond_pred.mean().item()
    count_mean = np.mean(all_count_exact)
    type_mean = np.mean(all_type_match)
    bottom_mean = all_bottom_recall.mean().item()
    ring_mean = np.mean(all_ring_pres)

    composite = compute_composite_score(
        rmsd=rmsd_mean, bottom_atom_score=bottom_mean,
        bond_validity=bond_gt_mean, ring_preservation=ring_mean,
        atom_count_accuracy=count_mean, structure_similarity=type_mean)

    return {
        "rmsd": rmsd_mean,
        "bond_gt": bond_gt_mean,
        "bond_pred": bond_pred_mean,
        "count": count_mean,
        "type": type_mean,
        "bottom_recall": bottom_mean,
        "ring": ring_mean,
        "composite": composite,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/v16c_train/checkpoints/best_gen.pt")
    parser.add_argument("--config", default="config_v16c_train.json")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output", default="experiments/v17_ring_scaffold/oracle_results.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    config = json.load(open(args.config))

    # Resolve data_root
    if config.get("data_root") == "auto":
        config["data_root"] = "/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM"

    model = AFM3DReconModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    dataset = QUAMAFMDataset(
        data_root=config["data_root"],
        param_key=config["param_key"],
        img_size=config["img_size"],
        split="val",
        min_corrugation=config.get("min_corrugation", 1.25),
        require_ring=True,
        augment_rotation=False,
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 1000),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # Run three configs
    configs = {
        "baseline_C_pred_guided": lambda m, b: m.generate(
            b, use_gt_count=False, disable_guidance=False,
            disable_ring_snap=True, sampler="ddim"),
        "oracle_A_ring_snap": lambda m, b: oracle_ring_snap_generate(m, b, ddim_steps=50),
        "oracle_B_soft_correction": lambda m, b: oracle_ring_constraint_generate(m, b, ddim_steps=50),
    }

    results = {}
    for name, gen_fn in configs.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        r = evaluate_config(model, loader, device, gen_fn, name, num_samples=args.num_samples)
        results[name] = r
        print(f"  RMSD={r['rmsd']:.4f} Bond(gt)={r['bond_gt']:.4f} Bond(pred)={r['bond_pred']:.4f} "
              f"Count={r['count']:.4f} Type={r['type']:.4f} "
              f"Bottom={r['bottom_recall']:.4f} Ring={r['ring']:.4f} Comp={r['composite']:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'='*80}")
    print("ORACLE ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<30} {'RMSD':>6} {'Bond(gt)':>9} {'Bond(pred)':>10} {'Count':>6} {'Type':>6} {'Bottom':>7} {'Ring':>6} {'Comp':>6}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<30} {r['rmsd']:>6.4f} {r['bond_gt']:>9.4f} {r['bond_pred']:>10.4f} "
              f"{r['count']:>6.4f} {r['type']:>6.4f} {r['bottom_recall']:>7.4f} {r['ring']:>6.4f} {r['composite']:>6.4f}")


if __name__ == "__main__":
    main()
