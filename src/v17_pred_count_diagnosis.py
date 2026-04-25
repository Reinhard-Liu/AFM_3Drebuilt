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
from src.train import AFM3DReconModel, get_default_config, load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def acc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def bias(a: np.ndarray, b: np.ndarray) -> float:
    return float((a - b).mean())


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v17_pred_count_eval.json")
    parser.add_argument("--checkpoint", default="experiments/v17_pred_count_debug/checkpoints/best_gen.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/v17_pred_count_eval/reports/pred_count_diagnosis.json")
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

    records = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        c_global, c_patches = model.encoder(batch["afm_stack"])
        base_count = model.count_head.predict(c_global)
        structure_pred = model.scaffold_count_head.predict(c_global)
        relation_pred = None
        if model.config.get("v17_count_comp_use_relation_signal", True):
            relation_pred = model.scaffold_relation_head(c_patches)
        compensated = model.compensate_atom_count(
            batch,
            base_count,
            predicted_relations=relation_pred,
            predicted_structure_counts=structure_pred,
        )

        B = base_count.shape[0]
        for i in range(B):
            records.append({
                "n_atoms_gt": int(batch["n_atoms"][i].item()),
                "n_atoms_base": int(base_count[i].item()),
                "n_atoms_comp": int(compensated[i].item()),
                "ring_system_gt": int(batch["scaffold_n_ring_systems"][i].item()),
                "ring_system_pred": int(structure_pred["ring_system_count"][i].item()),
                "scaffold_atoms_gt": int(batch["scaffold_total_scaffold_atoms"][i].item()),
                "scaffold_atoms_pred": int(structure_pred["scaffold_atom_count"][i].item()),
                "non_scaffold_gt": int(batch["scaffold_total_non_scaffold_atoms"][i].item()),
                "non_scaffold_pred": int(structure_pred["non_scaffold_atom_count"][i].item()),
                "anchors_gt": int(batch["scaffold_total_attachment_anchors"][i].item()),
                "anchors_pred": int(structure_pred["attachment_anchor_count"][i].item()),
                "side_edges_gt": int(batch["scaffold_total_sidechain_edges"][i].item()),
                "side_edges_pred": int(structure_pred["sidechain_edge_count"][i].item()),
            })

    def arr(key: str) -> np.ndarray:
        return np.array([r[key] for r in records], dtype=np.float64)

    summary = {
        "base_count": {
            "mae": mae(arr("n_atoms_base"), arr("n_atoms_gt")),
            "accuracy": acc(arr("n_atoms_base"), arr("n_atoms_gt")),
            "bias": bias(arr("n_atoms_base"), arr("n_atoms_gt")),
        },
        "comp_count": {
            "mae": mae(arr("n_atoms_comp"), arr("n_atoms_gt")),
            "accuracy": acc(arr("n_atoms_comp"), arr("n_atoms_gt")),
            "bias": bias(arr("n_atoms_comp"), arr("n_atoms_gt")),
        },
        "structure_targets": {
            "ring_system_count_mae": mae(arr("ring_system_pred"), arr("ring_system_gt")),
            "scaffold_atom_count_mae": mae(arr("scaffold_atoms_pred"), arr("scaffold_atoms_gt")),
            "non_scaffold_atom_count_mae": mae(arr("non_scaffold_pred"), arr("non_scaffold_gt")),
            "attachment_anchor_count_mae": mae(arr("anchors_pred"), arr("anchors_gt")),
            "sidechain_edge_count_mae": mae(arr("side_edges_pred"), arr("side_edges_gt")),
        },
        "correlation": {
            "total_count_error_vs_non_scaffold_error": corr(
                arr("n_atoms_comp") - arr("n_atoms_gt"),
                arr("non_scaffold_pred") - arr("non_scaffold_gt"),
            ),
            "base_count_error_vs_non_scaffold_error": corr(
                arr("n_atoms_base") - arr("n_atoms_gt"),
                arr("non_scaffold_pred") - arr("non_scaffold_gt"),
            ),
        },
    }

    out = {
        "checkpoint": args.checkpoint,
        "epoch": state.get("epoch"),
        "split": args.split,
        "seed": args.seed,
        "summary": summary,
        "records": records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
