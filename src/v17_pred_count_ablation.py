from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_dataloaders
from src.train import AFM3DReconModel, evaluate_generation, get_default_config, load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def summarize(result: dict) -> dict:
    return {
        "rmsd": result["rmsd_mean"],
        "bond_gt": result["bond_validity_gt_masked"],
        "bond_pred": result["bond_validity_pred_masked"],
        "count": result["count_exact_match"],
        "count_mae": result["count_mae"],
        "type": result["type_match_rate"],
        "bottom": result["bottom_recall_mean"],
        "ring": result["ring_preservation"],
        "composite": result["composite_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v17_pred_count_eval.json")
    parser.add_argument("--checkpoint", default="experiments/v17_pred_count_debug/checkpoints/best_gen.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/v17_pred_count_eval/reports/pred_count_ablation.json")
    args = parser.parse_args()

    set_seed(args.seed)
    config = get_default_config()
    config.update(load_config(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, _ = create_dataloaders(
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
        return_v17_bridge_labels=True,
    )
    del train_loader
    loader = val_loader if args.split == "val" else test_loader

    model = AFM3DReconModel(config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    eval_common = dict(
        num_samples=min(args.num_samples, len(loader.dataset)),
        use_ddim=True,
        ddim_steps=config.get("eval_ddim_steps", 100),
        use_gt_scaffold_tokens=config.get("bridge_eval_use_gt_scaffold_tokens", False),
        use_predicted_relation_tokens=config.get("bridge_eval_use_predicted_relation_tokens", False),
        use_gt_scaffold_soft_constraint=config.get("bridge_eval_use_gt_scaffold_soft_constraint", False),
        scaffold_constraint_time_threshold=config.get("bridge_eval_scaffold_constraint_time_threshold", 150),
        scaffold_constraint_scale=config.get("bridge_eval_scaffold_constraint_scale", 0.08),
        scaffold_plane_scale=config.get("bridge_eval_scaffold_plane_scale", 0.04),
        scaffold_edge_scale=config.get("bridge_eval_scaffold_edge_scale", 0.12),
        scaffold_sidechain_edge_scale=config.get("bridge_eval_scaffold_sidechain_edge_scale", 0.15),
        scaffold_post_guidance_scale=config.get("bridge_eval_scaffold_post_guidance_scale", 0.0),
        guidance_step_size=config.get("guidance_step_size", 0.002),
        guidance_time_threshold=config.get("guidance_time_threshold", 500),
    )

    cases = {
        "no_comp": {
            "v17_scaffold_count_compensation": False,
        },
        "gt_edges": {
            "v17_scaffold_count_compensation": True,
            "v17_count_comp_source": "gt_edges",
            "v17_count_comp_mode": "blend",
            "v17_count_comp_blend_alpha": 0.85,
            "v17_count_comp_sidechain_ratio": 1.0,
        },
        "predicted_structure": {
            "v17_scaffold_count_compensation": True,
            "v17_count_comp_source": "predicted_structure",
            "v17_count_comp_mode": "blend",
            "v17_count_comp_blend_alpha": config.get("v17_count_comp_blend_alpha", 0.6),
            "v17_count_comp_sidechain_ratio": config.get("v17_count_comp_sidechain_ratio", 1.0),
        },
        "hybrid": {
            "v17_scaffold_count_compensation": True,
            "v17_count_comp_source": "hybrid",
            "v17_count_comp_mode": "blend",
            "v17_count_comp_blend_alpha": config.get("v17_count_comp_blend_alpha", 0.6),
            "v17_count_comp_hybrid_alpha": config.get("v17_count_comp_hybrid_alpha", 0.5),
            "v17_count_comp_sidechain_ratio": config.get("v17_count_comp_sidechain_ratio", 1.0),
        },
    }

    original_cfg = copy.deepcopy(model.config)
    results = {
        "checkpoint": args.checkpoint,
        "epoch": state.get("epoch"),
        "split": args.split,
        "num_samples": eval_common["num_samples"],
        "seed": args.seed,
        "cases": {},
    }

    for name, updates in cases.items():
        print(f"\n=== {name} ===")
        model.config = copy.deepcopy(original_cfg)
        model.config.update(updates)
        set_seed(args.seed)
        raw = evaluate_generation(model, loader, device, **eval_common)
        results["cases"][name] = {
            "config": updates,
            "metrics": summarize(raw),
            "raw": raw,
        }
        m = results["cases"][name]["metrics"]
        print(
            f"RMSD={m['rmsd']:.4f} Bond(gt)={m['bond_gt']:.4f} Count={m['count']:.4f} "
            f"CountMAE={m['count_mae']:.4f} Type={m['type']:.4f} Bottom={m['bottom']:.4f} "
            f"Comp={m['composite']:.4f}"
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
