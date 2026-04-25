from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.train import AFM3DReconModel, load_config
from src.utils.metrics import (
    compute_bottom_atom_recall,
    compute_bond_validity,
    compute_rmsd,
    compute_structure_similarity,
)
from src.eval_phase1 import compute_ring_preservation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def matched_subset_rmsd(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    n_atoms_pred: torch.Tensor,
    subset_mask: torch.Tensor,
) -> torch.Tensor:
    vals = []
    for b in range(pred_coords.shape[0]):
        n_gt = int(gt_mask[b].bool().sum().item())
        n_pred = int(n_atoms_pred[b].item())
        n_pred = max(1, min(n_pred, pred_coords.shape[1]))
        p = pred_coords[b, :n_pred].detach().cpu().numpy()
        g = gt_coords[b, :n_gt].detach().cpu().numpy()
        subset = subset_mask[b, :n_gt].detach().cpu().numpy().astype(bool)
        diff = p[:, None, :] - g[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_sq = ((p[row_ind] - g[col_ind]) ** 2).sum(axis=-1)
        chosen = subset[col_ind]
        if chosen.any():
            vals.append(float(np.sqrt(matched_sq[chosen].mean())))
        else:
            vals.append(0.0)
    return torch.tensor(vals)


def scaffold_edge_mae(
    pred_coords: torch.Tensor,
    local_edges: torch.Tensor,
    local_edge_lengths: torch.Tensor,
    n_local_edges: torch.Tensor,
    n_atoms_pred: torch.Tensor,
) -> torch.Tensor:
    vals = []
    for b in range(pred_coords.shape[0]):
        n_edges = int(n_local_edges[b].item())
        n_pred = int(n_atoms_pred[b].item())
        n_pred = max(1, min(n_pred, pred_coords.shape[1]))
        coords = pred_coords[b]
        errs = []
        for ei in range(n_edges):
            i = int(local_edges[b, ei, 0].item())
            j = int(local_edges[b, ei, 1].item())
            if i < 0 or j < 0 or i >= n_pred or j >= n_pred:
                continue
            pred_len = (coords[i] - coords[j]).norm().item()
            errs.append(abs(pred_len - float(local_edge_lengths[b, ei].item())))
        vals.append(float(np.mean(errs)) if errs else 0.0)
    return torch.tensor(vals)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def generate(model, batch, bridge: bool, common: dict) -> dict:
    return model.generate(
        batch,
        use_gt_count=False,
        disable_guidance=False,
        disable_ring_snap=True,
        sampler="ddim",
        ddim_steps=common["ddim_steps"],
        use_gt_scaffold_tokens=bridge,
        use_gt_scaffold_soft_constraint=bridge,
        scaffold_constraint_time_threshold=common["time_threshold"],
        scaffold_constraint_scale=common["constraint_scale"],
        scaffold_plane_scale=common["plane_scale"],
        scaffold_edge_scale=common["edge_scale"],
        scaffold_sidechain_edge_scale=common["sidechain_edge_scale"],
        guidance_step_size=common["guidance_step_size"],
        guidance_time_threshold=common["guidance_time_threshold"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v17_bridge_b_eval.json")
    parser.add_argument("--checkpoint", default="experiments/v17_bridge_b_debug/checkpoints/best_gen.pt")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/v17_bridge_b_formal_eval/reports/rmsd_diagnosis_test_seed42.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config["param_key"],
        img_size=config["img_size"],
        min_corrugation=config["min_corrugation"],
        augment_rotation=config["augment_rotation"],
        require_ring=config.get("require_ring", False),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_samples=config["max_samples"],
        val_size=config["val_size"],
        return_v17_bridge_labels=config.get("v17_return_bridge_labels", False),
    )
    loader = val_loader if args.split == "val" else test_loader

    model = AFM3DReconModel(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    common = {
        "ddim_steps": config.get("eval_ddim_steps", 100),
        "time_threshold": config.get("bridge_eval_scaffold_constraint_time_threshold", 200),
        "constraint_scale": config.get("bridge_eval_scaffold_constraint_scale", 0.12),
        "plane_scale": config.get("bridge_eval_scaffold_plane_scale", 0.08),
        "edge_scale": config.get("bridge_eval_scaffold_edge_scale", 0.0),
        "sidechain_edge_scale": config.get("bridge_eval_scaffold_sidechain_edge_scale", 0.0),
        "guidance_step_size": config.get("guidance_step_size", 0.002),
        "guidance_time_threshold": config.get("guidance_time_threshold", 500),
    }

    records = []
    global_idx = 0
    for batch_idx, batch in enumerate(loader):
        cids = [loader.dataset.samples[global_idx + i]["cid"] for i in range(batch["coords"].shape[0])]
        global_idx += batch["coords"].shape[0]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        set_seed(args.seed * 1000 + batch_idx)
        baseline = generate(model, batch, bridge=False, common=common)
        set_seed(args.seed * 1000 + batch_idx)
        bridge = generate(model, batch, bridge=True, common=common)

        pred_base = baseline["type_logits"].argmax(dim=-1)
        pred_bridge = bridge["type_logits"].argmax(dim=-1)
        gt_mask = batch["atom_mask"]
        gt_coords = batch["coords"]
        scaffold_mask = batch["scaffold_atom_is_scaffold"]
        non_scaffold_mask = (gt_mask > 0) & (scaffold_mask < 0.5)
        attach_mask = batch["scaffold_atom_is_attachment_anchor"]

        rmsd_base = compute_rmsd(baseline["coords"], gt_coords, gt_mask, baseline["n_atoms_pred"])
        rmsd_bridge = compute_rmsd(bridge["coords"], gt_coords, gt_mask, bridge["n_atoms_pred"])
        bond_base = compute_bond_validity(baseline["coords"], pred_base, gt_mask)
        bond_bridge = compute_bond_validity(bridge["coords"], pred_bridge, gt_mask)
        bottom_base = compute_bottom_atom_recall(baseline["coords"], gt_coords, pred_base, batch["atom_types"], gt_mask)
        bottom_bridge = compute_bottom_atom_recall(bridge["coords"], gt_coords, pred_bridge, batch["atom_types"], gt_mask)
        struct_base = compute_structure_similarity(
            baseline["coords"], gt_coords, pred_base, batch["atom_types"], gt_mask, n_atoms_pred=baseline["n_atoms_pred"]
        )
        struct_bridge = compute_structure_similarity(
            bridge["coords"], gt_coords, pred_bridge, batch["atom_types"], gt_mask, n_atoms_pred=bridge["n_atoms_pred"]
        )
        ring_base = compute_ring_preservation(
            baseline["coords"], gt_coords, pred_base, batch["atom_types"], gt_mask, n_atoms_pred=baseline["n_atoms_pred"]
        )["ring_preservation"]
        ring_bridge = compute_ring_preservation(
            bridge["coords"], gt_coords, pred_bridge, batch["atom_types"], gt_mask, n_atoms_pred=bridge["n_atoms_pred"]
        )["ring_preservation"]

        scaffold_rmsd_base = matched_subset_rmsd(
            baseline["coords"], gt_coords, gt_mask, baseline["n_atoms_pred"], scaffold_mask > 0.5
        )
        scaffold_rmsd_bridge = matched_subset_rmsd(
            bridge["coords"], gt_coords, gt_mask, bridge["n_atoms_pred"], scaffold_mask > 0.5
        )
        non_scaffold_rmsd_base = matched_subset_rmsd(
            baseline["coords"], gt_coords, gt_mask, baseline["n_atoms_pred"], non_scaffold_mask
        )
        non_scaffold_rmsd_bridge = matched_subset_rmsd(
            bridge["coords"], gt_coords, gt_mask, bridge["n_atoms_pred"], non_scaffold_mask
        )
        attach_rmsd_base = matched_subset_rmsd(
            baseline["coords"], gt_coords, gt_mask, baseline["n_atoms_pred"], attach_mask > 0.5
        )
        attach_rmsd_bridge = matched_subset_rmsd(
            bridge["coords"], gt_coords, gt_mask, bridge["n_atoms_pred"], attach_mask > 0.5
        )
        edge_mae_base = scaffold_edge_mae(
            baseline["coords"], batch["scaffold_local_edges"], batch["scaffold_local_edge_lengths"],
            batch["scaffold_n_local_edges"], baseline["n_atoms_pred"]
        )
        edge_mae_bridge = scaffold_edge_mae(
            bridge["coords"], batch["scaffold_local_edges"], batch["scaffold_local_edge_lengths"],
            batch["scaffold_n_local_edges"], bridge["n_atoms_pred"]
        )

        exact_count = (baseline["n_atoms_pred"] == batch["n_atoms"]).float()

        for i, cid in enumerate(cids):
            records.append({
                "cid": cid,
                "rmsd_base": float(rmsd_base[i].item()),
                "rmsd_bridge": float(rmsd_bridge[i].item()),
                "bond_base": float(bond_base[i].item()),
                "bond_bridge": float(bond_bridge[i].item()),
                "bottom_base": float(bottom_base[i].item()),
                "bottom_bridge": float(bottom_bridge[i].item()),
                "type_base": float(struct_base["type_match_rate"][i].item()),
                "type_bridge": float(struct_bridge["type_match_rate"][i].item()),
                "ring_base": float(ring_base[i].item()),
                "ring_bridge": float(ring_bridge[i].item()),
                "scaffold_rmsd_base": float(scaffold_rmsd_base[i].item()),
                "scaffold_rmsd_bridge": float(scaffold_rmsd_bridge[i].item()),
                "non_scaffold_rmsd_base": float(non_scaffold_rmsd_base[i].item()),
                "non_scaffold_rmsd_bridge": float(non_scaffold_rmsd_bridge[i].item()),
                "attachment_rmsd_base": float(attach_rmsd_base[i].item()),
                "attachment_rmsd_bridge": float(attach_rmsd_bridge[i].item()),
                "edge_mae_base": float(edge_mae_base[i].item()),
                "edge_mae_bridge": float(edge_mae_bridge[i].item()),
                "exact_count": float(exact_count[i].item()),
                "n_atoms_gt": int(batch["n_atoms"][i].item()),
                "n_atoms_pred": int(baseline["n_atoms_pred"][i].item()),
                "n_scaffold_atoms": int(scaffold_mask[i].sum().item()),
            })

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in records], dtype=np.float64)

    delta = {
        "rmsd": arr("rmsd_bridge") - arr("rmsd_base"),
        "bond": arr("bond_bridge") - arr("bond_base"),
        "bottom": arr("bottom_bridge") - arr("bottom_base"),
        "type": arr("type_bridge") - arr("type_base"),
        "ring": arr("ring_bridge") - arr("ring_base"),
        "scaffold_rmsd": arr("scaffold_rmsd_bridge") - arr("scaffold_rmsd_base"),
        "non_scaffold_rmsd": arr("non_scaffold_rmsd_bridge") - arr("non_scaffold_rmsd_base"),
        "attachment_rmsd": arr("attachment_rmsd_bridge") - arr("attachment_rmsd_base"),
        "edge_mae": arr("edge_mae_bridge") - arr("edge_mae_base"),
    }

    improve_bond_worse_rmsd = (delta["bond"] > 0) & (delta["rmsd"] > 0)
    exact_mask = arr("exact_count") > 0.5

    summary = {
        "split": args.split,
        "seed": args.seed,
        "n_samples": len(records),
        "mean_delta": {k: float(v.mean()) for k, v in delta.items()},
        "median_delta": {k: float(np.median(v)) for k, v in delta.items()},
        "corr_delta_rmsd": {
            "bond": corr(delta["rmsd"], delta["bond"]),
            "bottom": corr(delta["rmsd"], delta["bottom"]),
            "type": corr(delta["rmsd"], delta["type"]),
            "ring": corr(delta["rmsd"], delta["ring"]),
            "scaffold_rmsd": corr(delta["rmsd"], delta["scaffold_rmsd"]),
            "non_scaffold_rmsd": corr(delta["rmsd"], delta["non_scaffold_rmsd"]),
            "attachment_rmsd": corr(delta["rmsd"], delta["attachment_rmsd"]),
            "edge_mae": corr(delta["rmsd"], delta["edge_mae"]),
        },
        "bond_up_rmsd_worse_fraction": float(improve_bond_worse_rmsd.mean()),
        "bond_up_rmsd_worse_stats": {
            "count": int(improve_bond_worse_rmsd.sum()),
            "mean_delta_non_scaffold_rmsd": float(delta["non_scaffold_rmsd"][improve_bond_worse_rmsd].mean()) if improve_bond_worse_rmsd.any() else 0.0,
            "mean_delta_scaffold_rmsd": float(delta["scaffold_rmsd"][improve_bond_worse_rmsd].mean()) if improve_bond_worse_rmsd.any() else 0.0,
            "mean_delta_edge_mae": float(delta["edge_mae"][improve_bond_worse_rmsd].mean()) if improve_bond_worse_rmsd.any() else 0.0,
            "mean_delta_bottom": float(delta["bottom"][improve_bond_worse_rmsd].mean()) if improve_bond_worse_rmsd.any() else 0.0,
        },
        "exact_count_vs_not": {
            "exact_count_fraction": float(exact_mask.mean()),
            "mean_rmsd_delta_exact": float(delta["rmsd"][exact_mask].mean()) if exact_mask.any() else 0.0,
            "mean_rmsd_delta_inexact": float(delta["rmsd"][~exact_mask].mean()) if (~exact_mask).any() else 0.0,
            "mean_bond_delta_exact": float(delta["bond"][exact_mask].mean()) if exact_mask.any() else 0.0,
            "mean_bond_delta_inexact": float(delta["bond"][~exact_mask].mean()) if (~exact_mask).any() else 0.0,
        },
    }

    out = {
        "checkpoint": args.checkpoint,
        "config": {
            "edge_scale": common["edge_scale"],
            "sidechain_edge_scale": common["sidechain_edge_scale"],
            "plane_scale": common["plane_scale"],
            "constraint_scale": common["constraint_scale"],
            "time_threshold": common["time_threshold"],
            "ddim_steps": common["ddim_steps"],
        },
        "summary": summary,
        "records": records,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
