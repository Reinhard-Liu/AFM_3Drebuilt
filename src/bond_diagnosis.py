"""
V16c Bond Diagnosis Script

Runs detailed bond-level diagnostics on existing checkpoints WITHOUT retraining.
Computes per-sample bond statistics, type-pair breakdowns, and zero-bond rates.

Usage:
    python -m src.bond_diagnosis --checkpoint experiments/v16c_debug2/checkpoints/best_gen.pt
"""

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from src.models.diffusion import ConditionalDDPM, SE3EquivariantDenoiser
from src.models.video_vit import VideoViTEncoder
from src.models.prediction_heads import AtomCountHead, MoleculeRetrievalHead, RingDetectionHead
from src.models.constraints import IDEAL_BOND_LENGTHS, MAX_BOND_DIST, NUM_ATOM_TYPES
from src.utils.metrics import compute_bond_validity


# --- Atom type names ---
TYPE_NAMES = ['H', 'C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br', 'I']


def build_model(config, device):
    """Build the full model from config."""
    encoder = VideoViTEncoder(
        img_size=config.get('img_size', 128),
        num_frames=config.get('num_frames', 10),
        patch_size=config.get('patch_size', 16),
        temporal_patch_size=config.get('temporal_patch_size', 2),
        embed_dim=config.get('embed_dim', 512),
        depth=config.get('encoder_depth', 8),
        num_heads=config.get('num_heads', 8),
        drop_rate=config.get('drop_rate', 0.1),
    )

    denoiser = SE3EquivariantDenoiser(
        max_atoms=85,
        coord_dim=3,
        num_atom_types=10,
        cond_dim=config.get('embed_dim', 512),
        hidden_dim=config.get('denoiser_hidden_dim', 256),
        num_layers=config.get('denoiser_depth', 6),
        num_heads=config.get('num_heads', 8),
    )
    ddpm = ConditionalDDPM(
        denoiser=denoiser,
        timesteps=config.get('diffusion_steps', 1000),
    )

    count_head = AtomCountHead(
        embed_dim=config.get('embed_dim', 512),
        max_count=85,
    )

    retrieval_head = MoleculeRetrievalHead(
        embed_dim=config.get('embed_dim', 512),
        proj_dim=128,
        temperature=0.07,
    )

    ring_head = RingDetectionHead(
        embed_dim=config.get('embed_dim', 512),
    )

    shape_head = torch.nn.Sequential(
        torch.nn.Linear(config.get('embed_dim', 512), 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 3),
    )

    class AFM3DReconModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.ddpm = ddpm
            self.count_head = count_head
            self.retrieval_head = retrieval_head
            self.ring_head = ring_head
            self.shape_head = shape_head
            self.config = config
            self.mol_embeddings = None

        def forward(self, batch):
            return self.generate(batch)

        def generate(self, batch, use_gt_count=False, use_ddim=True, ddim_steps=50,
                     disable_guidance=False, disable_ring_snap=False, sampler='ddim'):
            afm = batch["afm_stack"]
            c_global, c_patches = self.encoder(afm)

            if use_gt_count and "n_atoms" in batch:
                n_atoms = batch["n_atoms"]
            else:
                n_atoms = self.count_head.predict(c_global)

            ring_info = None
            predicted_rings = None
            target_shape = None
            use_physics = self.config.get("physics_guidance", True) and not disable_guidance

            is_ddim = (sampler == "ddim") or use_ddim
            coords, type_logits = self.ddpm.sample(
                c_global, c_patches, n_atoms, max_atoms=85,
                ring_info=ring_info,
                predicted_rings=predicted_rings,
                use_ddim=is_ddim, ddim_steps=ddim_steps,
                use_physics_guidance=use_physics,
                target_shape=target_shape,
                disable_guidance=disable_guidance,
                disable_ring_snap=disable_ring_snap,
            )
            return {
                "coords": coords,
                "type_logits": type_logits,
                "n_atoms_pred": n_atoms,
            }

    model = AFM3DReconModel()
    return model


def compute_detailed_bond_stats(pred_coords, pred_types, mask, n_pred=None):
    """Compute detailed per-sample bond statistics.

    Returns per-sample dicts with:
      - n_candidate_bonds
      - n_valid_bonds
      - n_bonds_zero (fraction of samples with 0 bonds)
      - mean/median candidate bond length
      - per_type_pair counts
    """
    ideal = IDEAL_BOND_LENGTHS.numpy()
    max_dist_table = MAX_BOND_DIST.numpy()
    TOLERANCE = 0.25  # 25% relative tolerance (BOND_VALIDITY_TOLERANCE)

    B = pred_coords.shape[0]
    results = []

    for b in range(B):
        if n_pred is not None:
            n = int(n_pred[b].item())
            # Use pred count mask
            m = torch.zeros(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)
            m[:n] = True
        else:
            m = mask[b].bool()
            n = m.sum().item()

        if n < 2:
            results.append({
                "n_candidate_bonds": 0,
                "n_valid_bonds": 0,
                "validity": 1.0,
                "mean_candidate_dist": 0.0,
                "median_candidate_dist": 0.0,
                "type_pair_candidates": {},
                "type_pair_valid": {},
            })
            continue

        coords = pred_coords[b, :n].detach().cpu().numpy()
        types = pred_types[b, :n].detach().cpu().numpy()

        # Pairwise distances
        diff = coords[:, None, :] - coords[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))

        n_candidates = 0
        n_valid = 0
        candidate_dists = []
        type_pair_candidates = {}
        type_pair_valid = {}

        for i in range(n):
            for j in range(i + 1, n):
                ti, tj = int(types[i]), int(types[j])
                if ti < 0 or tj < 0 or ti >= NUM_ATOM_TYPES or tj >= NUM_ATOM_TYPES:
                    continue
                ideal_len = ideal[ti, tj]
                if ideal_len < 1e-8:
                    continue
                max_d = max_dist_table[ti, tj]
                d = dists[i, j]
                pair_key = f"{TYPE_NAMES[ti]}-{TYPE_NAMES[tj]}"

                if d < max_d:
                    n_candidates += 1
                    candidate_dists.append(d)
                    type_pair_candidates[pair_key] = type_pair_candidates.get(pair_key, 0) + 1
                    if abs(d - ideal_len) / ideal_len < TOLERANCE:
                        n_valid += 1
                        type_pair_valid[pair_key] = type_pair_valid.get(pair_key, 0) + 1

        validity = n_valid / max(n_candidates, 1)
        mean_dist = float(np.mean(candidate_dists)) if candidate_dists else 0.0
        median_dist = float(np.median(candidate_dists)) if candidate_dists else 0.0

        results.append({
            "n_candidate_bonds": n_candidates,
            "n_valid_bonds": n_valid,
            "validity": validity,
            "mean_candidate_dist": mean_dist,
            "median_candidate_dist": median_dist,
            "type_pair_candidates": type_pair_candidates,
            "type_pair_valid": type_pair_valid,
        })

    return results


def run_diagnosis(checkpoint_path, config_path, num_samples=200,
                  use_gt_count=True, disable_guidance=True, disable_ring_snap=True,
                  device='cuda'):
    """Run bond diagnostics on a checkpoint."""
    import sys
    sys.path.insert(0, '/root/autodl-tmp/micro')
    from src.data.dataset import QUAMAFMDataset
    from torch.utils.data import DataLoader

    # Load config
    if config_path:
        import json
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default minimal config
        config = {
            'img_size': 128, 'num_frames': 10, 'patch_size': 16,
            'temporal_patch_size': 2, 'embed_dim': 512, 'encoder_depth': 8,
            'num_heads': 8, 'drop_rate': 0.1,
            'denoiser_hidden_dim': 256, 'denoiser_depth': 6,
            'diffusion_steps': 1000,
            'physics_guidance': True,
        }

    # Build model
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Checkpoint may have nested structure {'model': state_dict, ...} or flat state_dict
    if 'model' in checkpoint and isinstance(checkpoint['model'], dict) and 'ddpm' in checkpoint['model']:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Load dataset
    data_root = config.get('data_root', 'auto')
    if data_root == 'auto':
        # Auto-resolved: mirrors train.py's data_root resolution
        data_root = '/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM'
    val_size = config.get('val_size', 1000)

    dataset = QUAMAFMDataset(
        data_root=data_root,
        param_key=config.get('param_key', 'K-1'),
        img_size=config.get('img_size', 128),
        split='val',
        min_corrugation=config.get('min_corrugation', 1.25),
        require_ring=config.get('require_ring', True),
        augment_rotation=config.get('augment_rotation', False),
        max_samples=val_size + num_samples,
    )

    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        num_workers=min(config.get('num_workers', 4), 4),
        prefetch_factor=2,
    )

    # Run generation
    all_bond_stats = []
    all_gt_bond_stats = []
    all_validity_gt = []
    all_validity_pred = []
    all_rmsd = []
    all_count_acc = []

    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Bond diagnosis"):
            if count >= num_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            gen_result = model.generate(
                batch,
                use_gt_count=use_gt_count,
                disable_guidance=disable_guidance,
                disable_ring_snap=disable_ring_snap,
                sampler='ddim',
            )

            coords_pred = gen_result["coords"]
            type_logits = gen_result["type_logits"]
            pred_types = type_logits.argmax(dim=-1)
            n_pred = gen_result["n_atoms_pred"]

            # GT-masked bond validity (using ground truth atom count)
            bond_validity_gt = compute_bond_validity(
                coords_pred, pred_types, batch["atom_mask"],
            )

            # Pred-masked bond validity (using predicted atom count)
            N = coords_pred.shape[1]
            idx = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
            pred_mask = (idx < n_pred.float().unsqueeze(1)).float()
            bond_validity_pred = compute_bond_validity(
                coords_pred, pred_types, pred_mask,
            )

            # Detailed stats (pred-masked)
            stats_pred = compute_detailed_bond_stats(
                coords_pred, pred_types, batch["atom_mask"], n_pred=n_pred
            )
            # Detailed stats (GT-masked)
            stats_gt = compute_detailed_bond_stats(
                coords_pred, pred_types, batch["atom_mask"], n_pred=None
            )

            all_bond_stats.extend(stats_pred)
            all_gt_bond_stats.extend(stats_gt)
            all_validity_gt.append(bond_validity_gt)
            all_validity_pred.append(bond_validity_pred)

            count += batch["afm_stack"].shape[0]

    all_validity_gt = torch.cat(all_validity_gt)
    all_validity_pred = torch.cat(all_validity_pred)

    # Aggregate statistics
    n_samples = len(all_bond_stats)
    n_candidate_list = [s["n_candidate_bonds"] for s in all_bond_stats]
    n_valid_list = [s["n_valid_bonds"] for s in all_bond_stats]
    validity_list = [s["validity"] for s in all_bond_stats]
    mean_dist_list = [s["mean_candidate_dist"] for s in all_bond_stats if s["n_candidate_bonds"] > 0]

    # Aggregate type pair counts across all samples
    global_type_pair_candidates = {}
    global_type_pair_valid = {}
    for s in all_bond_stats:
        for k, v in s["type_pair_candidates"].items():
            global_type_pair_candidates[k] = global_type_pair_candidates.get(k, 0) + v
        for k, v in s["type_pair_valid"].items():
            global_type_pair_valid[k] = global_type_pair_valid.get(k, 0) + v

    # Per-sample zero-bond rates
    n_zero_bonds_samples = sum(1 for n in n_candidate_list if n == 0)
    n_zero_valid_samples = sum(1 for v in validity_list if v == 0.0 and sum(n_candidate_list) > 0)

    report = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "num_samples": n_samples,
        "evaluation_config": {
            "use_gt_count": use_gt_count,
            "disable_guidance": disable_guidance,
            "disable_ring_snap": disable_ring_snap,
        },
        "bond_constants": {
            "tolerance_relative": 0.25,
            "ideal_bond_lengths": {k: float(v) for k, v in
                zip([f"{TYPE_NAMES[i]}-{TYPE_NAMES[j]}" for i in range(NUM_ATOM_TYPES) for j in range(i, NUM_ATOM_TYPES)],
                    [IDEAL_BOND_LENGTHS[i, j].item() for i in range(NUM_ATOM_TYPES) for j in range(i, NUM_ATOM_TYPES) if IDEAL_BOND_LENGTHS[i, j] > 0])},
            "source": "src/models/constraints.py (unified)",
        },
        "aggregate": {
            "bond_validity_mean_gt_masked": float(all_validity_gt.mean().item()),
            "bond_validity_std_gt_masked": float(all_validity_gt.std().item()),
            "bond_validity_mean_pred_masked": float(all_validity_pred.mean().item()),
            "bond_validity_std_pred_masked": float(all_validity_pred.std().item()),
            "mean_candidate_bonds_per_sample": float(np.mean(n_candidate_list)),
            "median_candidate_bonds_per_sample": float(np.median(n_candidate_list)),
            "mean_valid_bonds_per_sample": float(np.mean(n_valid_list)),
            "median_valid_bonds_per_sample": float(np.median(n_valid_list)),
            "samples_with_zero_candidate_bonds": n_zero_bonds_samples,
            "samples_with_zero_candidate_bonds_pct": n_zero_bonds_samples / max(n_samples, 1),
            "samples_with_zero_valid_bonds": n_zero_valid_samples,
            "samples_with_zero_valid_bonds_pct": n_zero_valid_samples / max(n_samples, 1),
            "mean_candidate_bond_distance": float(np.mean(mean_dist_list)) if mean_dist_list else 0.0,
            "median_candidate_bond_distance": float(np.median(mean_dist_list)) if mean_dist_list else 0.0,
        },
        "type_pair_candidates": dict(sorted(global_type_pair_candidates.items(), key=lambda x: -x[1])),
        "type_pair_valid": dict(sorted(global_type_pair_valid.items(), key=lambda x: -x[1])),
        "type_pair_validity_rate": {
            k: global_type_pair_valid.get(k, 0) / max(global_type_pair_candidates.get(k, 1), 1)
            for k in global_type_pair_candidates
        },
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="V16c Bond Diagnosis")
    parser.add_argument("--checkpoint", type=str,
                       default="experiments/v16c_debug2/checkpoints/best_gen.pt",
                       help="Path to checkpoint to diagnose")
    parser.add_argument("--config", type=str,
                       default="config_v16c_debug2.json",
                       help="Path to config JSON")
    parser.add_argument("--num_samples", type=int, default=200,
                       help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON path (default: auto)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Resolve paths
    import os
    os.chdir('/root/autodl-tmp/micro')

    checkpoint_path = args.checkpoint
    config_path = args.config if os.path.exists(args.config) else None

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return

    # Run all three configurations
    configs = [
        ("A_gt_noguidance", True, True, True),
        ("B_pred_noguidance", False, True, True),
        ("C_pred_mainline", False, False, False),
    ]

    all_reports = {}
    for name, use_gt_count, disable_guidance, disable_ring_snap in configs:
        print(f"\n{'='*60}")
        print(f"Running config: {name}")
        print(f"  use_gt_count={use_gt_count}, disable_guidance={disable_guidance}, "
              f"disable_ring_snap={disable_ring_snap}")
        print('='*60)
        report = run_diagnosis(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            num_samples=args.num_samples,
            use_gt_count=use_gt_count,
            disable_guidance=disable_guidance,
            disable_ring_snap=disable_ring_snap,
            device=args.device,
        )
        all_reports[name] = report

    # Save output
    if args.output:
        output_path = args.output
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        report_dir = os.path.join(ckpt_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        output_path = os.path.join(report_dir, 'bond_diagnosis.json')

    with open(output_path, 'w') as f:
        json.dump(all_reports, f, indent=2)

    print(f"\nBond diagnosis saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("BOND DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"{'Config':<25} {'Bond(gt_mask)':>14} {'Bond(pred_mask)':>15} "
          f"{'MeanCand':>10} {'Cand=0%':>8} {'Valid=0%':>8}")
    print("-"*80)
    for name, report in all_reports.items():
        agg = report["aggregate"]
        print(f"{name:<25} {agg['bond_validity_mean_gt_masked']:>14.4f} "
              f"{agg['bond_validity_mean_pred_masked']:>15.4f} "
              f"{agg['mean_candidate_bonds_per_sample']:>10.1f} "
              f"{agg['samples_with_zero_candidate_bonds_pct']*100:>7.1f}% "
              f"{agg['samples_with_zero_valid_bonds_pct']*100:>7.1f}%")

    print("\nType Pair Bond Statistics:")
    print("-"*60)
    report0 = list(all_reports.values())[0]
    tp_validity = report0["type_pair_validity_rate"]
    tp_candidates = report0["type_pair_candidates"]
    for pair in sorted(tp_validity.keys(), key=lambda x: -tp_candidates.get(x, 0))[:10]:
        n_cand = tp_candidates.get(pair, 0)
        rate = tp_validity.get(pair, 0.0)
        print(f"  {pair:<10} cand={n_cand:>5}  valid_rate={rate:.3f}")


if __name__ == "__main__":
    main()
