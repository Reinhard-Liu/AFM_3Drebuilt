"""
V17-Bridge ablation:

- Baseline: current guided generator, no scaffold bridge
- Bridge A: GT scaffold token cross-attention only
- Bridge B: GT scaffold token cross-attention + low-noise soft scaffold constraint
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import AFM3DReconModel
from src.data.dataset import QUAMAFMDataset
from src.utils.metrics import (
    compute_rmsd,
    compute_bond_validity,
    compute_atom_count_accuracy,
    compute_bottom_atom_recall,
    compute_structure_similarity,
    compute_composite_score,
)
from src.eval_phase1 import compute_ring_preservation


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("data_root") == "auto":
        config["data_root"] = "/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM"
    return config


def build_loader(config: dict, num_samples: int, batch_size: int) -> DataLoader:
    dataset = QUAMAFMDataset(
        data_root=config["data_root"],
        param_key=config["param_key"],
        img_size=config["img_size"],
        split="val",
        min_corrugation=config.get("min_corrugation", 0.0),
        require_ring=True,
        augment_rotation=False,
        max_samples=num_samples,
        val_size=max(num_samples, config.get("val_size", 1000)),
        return_v17_bridge_labels=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def generate_baseline(model, batch, ddim_steps: int):
    return model.generate(
        batch,
        use_gt_count=False,
        disable_guidance=False,
        disable_ring_snap=True,
        sampler="ddim",
        ddim_steps=ddim_steps,
    )


def generate_bridge_a(model, batch, ddim_steps: int):
    return model.generate(
        batch,
        use_gt_count=False,
        disable_guidance=False,
        disable_ring_snap=True,
        sampler="ddim",
        ddim_steps=ddim_steps,
        use_gt_scaffold_tokens=True,
    )


def generate_bridge_b(
    model,
    batch,
    ddim_steps: int,
    time_threshold: int,
    pos_scale: float,
    plane_scale: float,
    edge_scale: float,
    sidechain_edge_scale: float,
):
    return model.generate(
        batch,
        use_gt_count=False,
        disable_guidance=False,
        disable_ring_snap=True,
        sampler="ddim",
        ddim_steps=ddim_steps,
        use_gt_scaffold_tokens=True,
        use_gt_scaffold_soft_constraint=True,
        scaffold_constraint_time_threshold=time_threshold,
        scaffold_constraint_scale=pos_scale,
        scaffold_plane_scale=plane_scale,
        scaffold_edge_scale=edge_scale,
        scaffold_sidechain_edge_scale=sidechain_edge_scale,
    )


@torch.no_grad()
def evaluate_config(model, loader, device, name: str, gen_fn):
    model.eval()
    all_rmsd, all_bond_gt, all_bond_pred = [], [], []
    all_count_exact, all_type_match, all_bottom_recall = [], [], []
    all_ring_pres = []

    pbar = tqdm(loader, desc=name, leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        result = gen_fn(model, batch)

        coords_pred = result["coords"]
        type_logits = result["type_logits"]
        pred_types = type_logits.argmax(dim=-1)
        n_pred = result["n_atoms_pred"]
        n_atoms = coords_pred.shape[1]

        rmsd = compute_rmsd(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
        bond_gt = compute_bond_validity(coords_pred, pred_types, batch["atom_mask"])
        idx = torch.arange(n_atoms, device=device).unsqueeze(0).float()
        pred_mask = (idx < n_pred.float().unsqueeze(1)).float()
        bond_pred = compute_bond_validity(coords_pred, pred_types, pred_mask)
        count_acc = compute_atom_count_accuracy(n_pred, batch["n_atoms"])
        bottom_recall = compute_bottom_atom_recall(
            coords_pred, batch["coords"], pred_types, batch["atom_types"], batch["atom_mask"]
        )
        struct_sim = compute_structure_similarity(
            coords_pred, batch["coords"], pred_types, batch["atom_types"], batch["atom_mask"], n_atoms_pred=n_pred
        )
        ring_pres = compute_ring_preservation(
            coords_pred, batch["coords"], pred_types, batch["atom_types"], batch["atom_mask"], n_atoms_pred=n_pred
        )

        all_rmsd.append(rmsd)
        all_bond_gt.append(bond_gt)
        all_bond_pred.append(bond_pred)
        all_count_exact.append(count_acc["exact_match"])
        all_type_match.append(struct_sim["type_match_rate_mean"])
        all_bottom_recall.append(bottom_recall)
        all_ring_pres.append(ring_pres["ring_preservation_mean"])

    all_rmsd = torch.cat(all_rmsd)
    all_bond_gt = torch.cat(all_bond_gt)
    all_bond_pred = torch.cat(all_bond_pred)
    all_bottom_recall = torch.cat(all_bottom_recall)
    rmsd_mean = all_rmsd.mean().item()
    bond_gt_mean = all_bond_gt.mean().item()
    bond_pred_mean = all_bond_pred.mean().item()
    count_mean = float(np.mean(all_count_exact))
    type_mean = float(np.mean(all_type_match))
    bottom_mean = all_bottom_recall.mean().item()
    ring_mean = float(np.mean(all_ring_pres))
    composite = compute_composite_score(
        rmsd=rmsd_mean,
        bottom_atom_score=bottom_mean,
        bond_validity=bond_gt_mean,
        ring_preservation=ring_mean,
        atom_count_accuracy=count_mean,
        structure_similarity=type_mean,
    )
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
    parser.add_argument("--checkpoint", default="experiments/v16d1_debug/checkpoints/best_gen.pt")
    parser.add_argument("--config", default="config_v16c_train.json")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--constraint_time_threshold", type=int, default=200)
    parser.add_argument("--constraint_scale", type=float, default=0.12)
    parser.add_argument("--plane_scale", type=float, default=0.08)
    parser.add_argument("--edge_scale", type=float, default=0.0)
    parser.add_argument("--sidechain_edge_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/v17_bridge/bridge_ablation_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    config["use_v17_bridge_gt_scaffold_tokens"] = True
    config["v17_return_bridge_labels"] = True

    model = AFM3DReconModel(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    loader = build_loader(config, args.num_samples, args.batch_size)

    configs = {
        "baseline_guided": lambda m, b: generate_baseline(m, b, args.ddim_steps),
        "bridge_A_gt_tokens": lambda m, b: generate_bridge_a(m, b, args.ddim_steps),
        "bridge_B_gt_tokens_soft": lambda m, b: generate_bridge_b(
            m, b, args.ddim_steps, args.constraint_time_threshold, args.constraint_scale, args.plane_scale, args.edge_scale, args.sidechain_edge_scale
        ),
    }

    results = {}
    for name, fn in configs.items():
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
        r = evaluate_config(model, loader, device, name, fn)
        results[name] = r
        print(
            f"  RMSD={r['rmsd']:.4f} Bond(gt)={r['bond_gt']:.4f} Bond(pred)={r['bond_pred']:.4f} "
            f"Count={r['count']:.4f} Type={r['type']:.4f} Bottom={r['bottom_recall']:.4f} "
            f"Ring={r['ring']:.4f} Comp={r['composite']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "ddim_steps": args.ddim_steps,
                "num_samples": args.num_samples,
                "constraint_time_threshold": args.constraint_time_threshold,
                "constraint_scale": args.constraint_scale,
                "plane_scale": args.plane_scale,
                "edge_scale": args.edge_scale,
                "sidechain_edge_scale": args.sidechain_edge_scale,
                "seed": args.seed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
