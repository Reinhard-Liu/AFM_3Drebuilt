"""
Compare major checkpoints under V18 Phase-1 structure fidelity metrics.

Usage:
    python -m src.v18_compare_checkpoints --num_samples 64
"""

import os
import json
import argparse

import torch

from src.train import AFM3DReconModel, evaluate_generation
from src.data.dataset import create_dataloaders


DEFAULT_CHECKPOINTS = {
    "v16c_train": "experiments/v16c_train/checkpoints/best_gen.pt",
    "v16d1_debug": "experiments/v16d1_debug/checkpoints/best_gen.pt",
    "v17_bridge_b": "experiments/v17_bridge_b_debug/checkpoints/best_gen.pt",
    "v17_bridge_closed_loop_comp": "experiments/v17_bridge_token_closed_loop_debug/checkpoints/best_gen.pt",
    "v17_pred_count_v2": "experiments/v17_pred_count_debug_v2/checkpoints/best_gen.pt",
}


def _load_checkpoint(path: str, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    config = dict(state["config"])
    config["num_workers"] = 0
    config["v18_eval_use_structure_labels"] = True
    config["v17_return_bridge_labels"] = True
    model = AFM3DReconModel(config).to(device)
    current_state = model.state_dict()
    filtered_state = {}
    for key, value in state["model"].items():
        if key not in current_state:
            continue
        if current_state[key].shape != value.shape:
            continue
        filtered_state[key] = value
    load_res = model.load_state_dict(filtered_state, strict=False)
    return model, config, state, load_res


def _get_loaders(config: dict):
    return create_dataloaders(
        data_root=config["data_root"],
        param_key=config.get("param_key", "K-1"),
        img_size=config.get("img_size", 128),
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=False,
        require_ring=config.get("require_ring", False),
        batch_size=min(config.get("batch_size", 8), 8),
        num_workers=0,
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
        return_v17_bridge_labels=True,
    )


def _run_eval(model, loader, device, config, num_samples: int):
    return evaluate_generation(
        model, loader, device, num_samples=num_samples,
        use_ddim=True,
        ddim_steps=config.get("eval_ddim_steps", 100),
        use_gt_count=False,
        disable_guidance=False,
        disable_ring_snap=False,
        sampler="ddim",
        use_gt_scaffold_tokens=config.get("bridge_eval_use_gt_scaffold_tokens", False),
        use_predicted_relation_tokens=config.get("bridge_eval_use_predicted_relation_tokens", False),
        use_gt_scaffold_soft_constraint=config.get("bridge_eval_use_gt_scaffold_soft_constraint", False),
        scaffold_constraint_time_threshold=config.get("bridge_eval_scaffold_constraint_time_threshold", 200),
        scaffold_constraint_scale=config.get("bridge_eval_scaffold_constraint_scale", 0.12),
        scaffold_plane_scale=config.get("bridge_eval_scaffold_plane_scale", 0.08),
        scaffold_edge_scale=config.get("bridge_eval_scaffold_edge_scale", 0.0),
        scaffold_sidechain_edge_scale=config.get("bridge_eval_scaffold_sidechain_edge_scale", 0.0),
        scaffold_post_guidance_scale=config.get("bridge_eval_scaffold_post_guidance_scale", 0.0),
        guidance_step_size=config.get("guidance_step_size", 0.002),
        guidance_time_threshold=config.get("guidance_time_threshold", 500),
    )


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints with V18 structure fidelity metrics")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output_dir", type=str, default="experiments/v18_visual_compare")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reports"), exist_ok=True)

    results = []

    for name, path in DEFAULT_CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue

        print(f"\n=== {name} ===")
        model, config, state, load_res = _load_checkpoint(path, device)
        train_loader, val_loader, test_loader, _ = _get_loaders(config)
        loader = val_loader if args.split == "val" else test_loader
        metrics = _run_eval(model, loader, device, config, args.num_samples)

        row = {
            "name": name,
            "checkpoint": path,
            "epoch": int(state.get("epoch", -1)),
            "struct_fidelity_pass_rate": metrics["struct_fidelity_pass_rate"],
            "struct_fidelity_score": metrics["struct_fidelity_score"],
            "atom_count_exact": metrics["atom_count_exact"],
            "atom_count_abs_error": metrics["atom_count_abs_error"],
            "matched_heavy_atom_rmsd": metrics["matched_heavy_atom_rmsd"],
            "atom_type_acc": metrics["atom_type_acc"],
            "heteroatom_f1": metrics["heteroatom_f1"],
            "ring_complete_rate": metrics["ring_complete_rate"],
            "scaffold_local_edge_f1": metrics.get("scaffold_relation_f1", 0.0),
            "attachment_edge_f1": metrics["attachment_edge_f1"],
            "bond_validity_pred": metrics["bond_validity_pred_masked"],
            "local_chem_score": metrics["local_chem_score"],
            "legacy_composite_score": metrics["composite_score"],
        }
        print(
            f"pass={row['struct_fidelity_pass_rate']:.4f} "
            f"fidelity={row['struct_fidelity_score']:.4f} "
            f"heavy_rmsd={row['matched_heavy_atom_rmsd']:.4f} "
            f"type={row['atom_type_acc']:.4f} hetero={row['heteroatom_f1']:.4f} "
            f"ring={row['ring_complete_rate']:.4f} attach={row['attachment_edge_f1']:.4f} "
            f"bond={row['bond_validity_pred']:.4f}"
        )
        results.append(row)

    results.sort(
        key=lambda r: (
            r["struct_fidelity_pass_rate"],
            r["struct_fidelity_score"],
            -r["matched_heavy_atom_rmsd"],
        ),
        reverse=True,
    )

    json_path = os.path.join(args.output_dir, "reports", f"checkpoint_compare_{args.split}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.join(args.output_dir, "reports", f"checkpoint_compare_{args.split}_summary.md")
    with open(md_path, "w") as f:
        f.write(f"# V18 {args.split} Checkpoint Compare\n\n")
        f.write("| Rank | Name | Pass | Fidelity | HeavyRMSD | CountExact | Type | HeteroF1 | Ring | Attach | Bond |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for i, r in enumerate(results, start=1):
            f.write(
                f"| {i} | {r['name']} | {r['struct_fidelity_pass_rate']:.4f} | "
                f"{r['struct_fidelity_score']:.4f} | {r['matched_heavy_atom_rmsd']:.4f} | "
                f"{r['atom_count_exact']:.4f} | {r['atom_type_acc']:.4f} | "
                f"{r['heteroatom_f1']:.4f} | {r['ring_complete_rate']:.4f} | "
                f"{r['attachment_edge_f1']:.4f} | {r['bond_validity_pred']:.4f} |\n"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
