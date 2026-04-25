from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.train import AFM3DReconModel, evaluate_generation, load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def aggregate(records: list[dict]) -> dict:
    keys = [
        "rmsd_mean",
        "bottom_recall_mean",
        "bottom_rmsd_mean",
        "bond_validity_gt_masked",
        "bond_validity_pred_masked",
        "count_exact_match",
        "count_mae",
        "type_match_rate",
        "ring_preservation",
        "composite_score",
    ]
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in records], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v17_bridge_b_eval.json")
    parser.add_argument("--checkpoint", default="experiments/v17_bridge_b_debug/checkpoints/best_gen.pt")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--output", default="experiments/v17_bridge_b_formal_eval/reports/repeat_eval_summary.json")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, test_loader, num_cids = create_dataloaders(
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

    model = AFM3DReconModel(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    common = dict(
        use_gt_scaffold_tokens=config.get("bridge_eval_use_gt_scaffold_tokens", False),
        use_gt_scaffold_soft_constraint=config.get("bridge_eval_use_gt_scaffold_soft_constraint", False),
        scaffold_constraint_time_threshold=config.get("bridge_eval_scaffold_constraint_time_threshold", 200),
        scaffold_constraint_scale=config.get("bridge_eval_scaffold_constraint_scale", 0.12),
        scaffold_plane_scale=config.get("bridge_eval_scaffold_plane_scale", 0.08),
        scaffold_edge_scale=config.get("bridge_eval_scaffold_edge_scale", 0.0),
        scaffold_sidechain_edge_scale=config.get("bridge_eval_scaffold_sidechain_edge_scale", 0.0),
        guidance_step_size=config.get("guidance_step_size", 0.002),
        guidance_time_threshold=config.get("guidance_time_threshold", 500),
        use_ddim=True,
        ddim_steps=config.get("eval_ddim_steps", 100),
    )

    results = {
        "checkpoint": args.checkpoint,
        "seeds": seeds,
        "config": {
            "ddim_steps": common["ddim_steps"],
            "time_threshold": common["scaffold_constraint_time_threshold"],
            "constraint_scale": common["scaffold_constraint_scale"],
            "plane_scale": common["scaffold_plane_scale"],
            "edge_scale": common["scaffold_edge_scale"],
            "sidechain_edge_scale": common["scaffold_sidechain_edge_scale"],
            "guidance_step_size": common["guidance_step_size"],
            "guidance_time_threshold": common["guidance_time_threshold"],
        },
        "val": [],
        "test": [],
    }
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint} (epoch {state.get('epoch', '?')})")
    print(f"Val={len(val_loader.dataset)} Test={len(test_loader.dataset)} CIDs={num_cids}")
    print(f"Bridge-B eval config: edge_scale={common['scaffold_edge_scale']}, "
          f"sidechain_edge_scale={common['scaffold_sidechain_edge_scale']}, "
          f"pos={common['scaffold_constraint_scale']}, plane={common['scaffold_plane_scale']}, "
          f"time<{common['scaffold_constraint_time_threshold']}, ddim_steps={common['ddim_steps']}")

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)
        val_result = evaluate_generation(model, val_loader, device, num_samples=len(val_loader.dataset), **common)
        results["val"].append({"seed": seed, **val_result})
        print(f"VAL  RMSD={val_result['rmsd_mean']:.4f} Bond(gt)={val_result['bond_validity_gt_masked']:.4f} "
              f"Type={val_result['type_match_rate']:.4f} Bottom={val_result['bottom_recall_mean']:.4f} "
              f"Comp={val_result['composite_score']:.4f}")

        set_seed(seed)
        test_result = evaluate_generation(model, test_loader, device, num_samples=len(test_loader.dataset), **common)
        results["test"].append({"seed": seed, **test_result})
        print(f"TEST RMSD={test_result['rmsd_mean']:.4f} Bond(gt)={test_result['bond_validity_gt_masked']:.4f} "
              f"Type={test_result['type_match_rate']:.4f} Bottom={test_result['bottom_recall_mean']:.4f} "
              f"Comp={test_result['composite_score']:.4f}")

    results["summary"] = {
        "val": aggregate(results["val"]),
        "test": aggregate(results["test"]),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
