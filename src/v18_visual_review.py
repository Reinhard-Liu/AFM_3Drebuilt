"""
V18 visual review on a fixed validation split and fixed sample indices.

Generates side-by-side visual comparisons for:
    GT | baseline | v18 slot-hard | v18 slot-graph

This script intentionally uses one common validation dataset definition so the
same samples are compared across checkpoints.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import QUAMAFMDataset
from src.utils.metrics import compute_rmsd, compute_structure_fidelity
from src.visualize_val import plot_molecule, set_equal_axes


DEFAULT_DATA_ROOT = "/root/autodl-tmp/micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM"
DEFAULT_CHECKPOINTS = {
    "v16d1": "/root/autodl-tmp/micro/experiments/v16d1_debug/checkpoints/best_gen.pt",
    "v18_slot_hard": "/root/autodl-tmp/micro/experiments/v18_slot_hard_bridge_debug/checkpoints/best_gen.pt",
    "v18_slot_graph": "/root/autodl-tmp/micro/experiments/v18_slot_graph_bridge_debug/checkpoints/best_gen.pt",
}
DEFAULT_SPLIT_CFG = {
    "param_key": "K-1",
    "img_size": 128,
    "min_corrugation": 1.25,
    "require_ring": True,
    "split": "val",
    "val_size": 256,
    "max_samples": 4096,
}


def _pred_mask_from_result(result, gt_mask, n_slots):
    n_atoms_pred = result.get("n_atoms_pred", None)
    if n_atoms_pred is None:
        return gt_mask.astype(np.float32)
    n_pred = int(n_atoms_pred[0].item())
    pred_mask = np.zeros(n_slots, dtype=np.float32)
    pred_mask[:min(n_pred, n_slots)] = 1.0
    return pred_mask


def _load_model_compat(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    from src.train import AFM3DReconModel
    model = AFM3DReconModel(config).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"Loaded {checkpoint_path}")
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    return model, config, "diffusion"


def _run_checkpoint_on_samples(name, checkpoint_path, dataset, indices, device, ddim_steps):
    model, _, model_type = _load_model_compat(checkpoint_path, device)
    assert model_type == "diffusion", f"{name} is not a diffusion checkpoint"

    outputs = []
    for idx in indices:
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
        with torch.no_grad():
            result = model.generate(
                batch,
                use_gt_count=False,
                use_ddim=True,
                ddim_steps=ddim_steps,
                disable_guidance=False,
                disable_ring_snap=False,
                sampler="ddim",
            )
            coords_pred = result["coords"]
            type_logits = result["type_logits"]
            pred_types = type_logits.argmax(dim=-1)

        gt_mask_np = batch["atom_mask"][0].cpu().numpy()
        pred_mask_np = _pred_mask_from_result(result, gt_mask_np, coords_pred.shape[1])
        fidelity = compute_structure_fidelity(
            coords_pred.cpu(),
            batch["coords"].cpu(),
            pred_types.cpu(),
            batch["atom_types"].cpu(),
            batch["atom_mask"].cpu(),
            n_atoms_pred=result.get("n_atoms_pred", None).cpu() if result.get("n_atoms_pred", None) is not None else None,
            scaffold_labels=batch if "scaffold_n_ring_systems" in batch else None,
        )
        rmsd_val = compute_rmsd(coords_pred.cpu(), batch["coords"].cpu(), batch["atom_mask"].cpu()).item()
        outputs.append({
            "sample_idx": idx,
            "pred_coords": coords_pred[0].cpu().numpy(),
            "pred_types": pred_types[0].cpu().numpy(),
            "pred_mask": pred_mask_np,
            "n_atoms_pred": int(result["n_atoms_pred"][0].item()) if result.get("n_atoms_pred", None) is not None else int(pred_mask_np.sum()),
            "rmsd": rmsd_val,
            "metrics": {
                "struct_fidelity_score": float(fidelity["struct_fidelity_score_mean"]),
                "atom_count_exact": float(fidelity["atom_count_exact_mean"]),
                "matched_heavy_atom_rmsd": float(fidelity["matched_heavy_atom_rmsd_mean"]),
                "atom_type_acc": float(fidelity["atom_type_acc_mean"]),
                "heteroatom_f1": float(fidelity["heteroatom_f1_mean"]),
                "ring_complete_rate": float(fidelity["ring_complete_rate_mean"]),
                "attachment_edge_f1": float(fidelity["attachment_edge_f1_mean"]),
                "bond_validity_pred_masked": float(fidelity["local_chem_score_mean"]),
            },
        })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def _draw_review_figure(sample_idx, sample, model_rows, save_path):
    import matplotlib.gridspec as gridspec

    gt_coords = sample["coords"].numpy()
    gt_types = sample["atom_types"].numpy()
    gt_mask = sample["atom_mask"].numpy()
    afm_stack = sample["afm_stack"].numpy()

    fig = plt.figure(figsize=(22, 11))
    gs = gridspec.GridSpec(3, 12, figure=fig, hspace=0.28, wspace=0.28)

    slice_indices = [0, 2, 4, 6]
    for i, si in enumerate(slice_indices):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        ax.imshow(afm_stack[si], cmap="afmhot", vmin=0, vmax=1)
        ax.set_title(f"AFM Z={si}", fontsize=9)
        ax.axis("off")

    ax_gt = fig.add_subplot(gs[1, 0:3], projection="3d")
    plot_molecule(ax_gt, gt_coords, gt_types, gt_mask, title=f"GT ({int((gt_mask>0).sum())} atoms)")
    if int((gt_mask > 0).sum()) > 0:
        set_equal_axes(ax_gt, gt_coords[gt_mask > 0])

    panel_spans = [(3, 6), (6, 9), (9, 12)]
    text_lines = [
        f"Sample #{sample_idx}",
        "=" * 72,
        "字段: fidelity | count_exact | heavy_rmsd | type_acc | hetero_f1 | ring_complete | attach_edge_f1 | bond_pred",
        "",
    ]

    for (name, row), (c0, c1) in zip(model_rows.items(), panel_spans):
        ax = fig.add_subplot(gs[1, c0:c1], projection="3d")
        plot_molecule(
            ax,
            row["pred_coords"],
            row["pred_types"],
            row["pred_mask"],
            title=f"{name}\nPred ({row['n_atoms_pred']} atoms)",
        )
        if int((row["pred_mask"] > 0).sum()) > 0:
            set_equal_axes(ax, row["pred_coords"][row["pred_mask"] > 0])

        m = row["metrics"]
        text_lines.append(
            f"{name:>14}: "
            f"{m['struct_fidelity_score']:.3f} | "
            f"{m['atom_count_exact']:.3f} | "
            f"{m['matched_heavy_atom_rmsd']:.3f} | "
            f"{m['atom_type_acc']:.3f} | "
            f"{m['heteroatom_f1']:.3f} | "
            f"{m['ring_complete_rate']:.3f} | "
            f"{m['attachment_edge_f1']:.3f} | "
            f"{m['bond_validity_pred_masked']:.3f}"
        )

    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")
    ax_text.text(
        0.01, 0.98, "\n".join(text_lines),
        transform=ax_text.transAxes,
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f8", edgecolor="#95a5a6", alpha=0.95),
    )

    fig.suptitle("V18 Directed Visual Review", fontsize=14, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="V18 directed visual review on common validation samples")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/micro/experiments/v18_visual_review")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--ddim_steps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = QUAMAFMDataset(
        data_root=args.data_root,
        param_key=DEFAULT_SPLIT_CFG["param_key"],
        img_size=DEFAULT_SPLIT_CFG["img_size"],
        min_corrugation=DEFAULT_SPLIT_CFG["min_corrugation"],
        augment_rotation=False,
        require_ring=DEFAULT_SPLIT_CFG["require_ring"],
        split=DEFAULT_SPLIT_CFG["split"],
        val_size=DEFAULT_SPLIT_CFG["val_size"],
        max_samples=DEFAULT_SPLIT_CFG["max_samples"],
    )
    total = len(dataset)
    indices = np.linspace(0, total - 1, min(args.num_samples, total), dtype=int).tolist()

    outputs_by_model = {}
    for name, ckpt in DEFAULT_CHECKPOINTS.items():
        print(f"Running {name}: {ckpt}")
        outputs_by_model[name] = _run_checkpoint_on_samples(name, ckpt, dataset, indices, device, args.ddim_steps)

    img_dir = os.path.join(args.output_dir, "images")
    rep_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    per_sample = []
    for pos, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        model_rows = {name: outputs_by_model[name][pos] for name in DEFAULT_CHECKPOINTS}
        save_path = os.path.join(img_dir, f"sample_{sample_idx:05d}.png")
        _draw_review_figure(sample_idx, sample, model_rows, save_path)
        per_sample.append({
            "sample_idx": sample_idx,
            "image_path": save_path,
            "models": {name: model_rows[name]["metrics"] | {"n_atoms_pred": model_rows[name]["n_atoms_pred"]} for name in model_rows},
        })
        print(f"  saved {save_path}")

    summary = {}
    for name, rows in outputs_by_model.items():
        keys = rows[0]["metrics"].keys()
        summary[name] = {k: float(np.mean([r["metrics"][k] for r in rows])) for k in keys}
        summary[name]["mean_n_atoms_pred"] = float(np.mean([r["n_atoms_pred"] for r in rows]))

    json_path = os.path.join(rep_dir, "v18_visual_review_summary.json")
    Path(json_path).write_text(json.dumps({"indices": indices, "summary": summary, "per_sample": per_sample}, indent=2))

    md_lines = [
        "# V18 Directed Visual Review",
        "",
        "Fixed validation split: `val_size=256`, `max_samples=4096`, `require_ring=true`",
        "",
        "| Model | fidelity | count_exact | heavy_rmsd | type_acc | hetero_f1 | ring_complete | attach_edge_f1 | bond_pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summary.items():
        md_lines.append(
            f"| {name} | {s['struct_fidelity_score']:.4f} | {s['atom_count_exact']:.4f} | "
            f"{s['matched_heavy_atom_rmsd']:.4f} | {s['atom_type_acc']:.4f} | "
            f"{s['heteroatom_f1']:.4f} | {s['ring_complete_rate']:.4f} | "
            f"{s['attachment_edge_f1']:.4f} | {s['bond_validity_pred_masked']:.4f} |"
        )
    md_lines.extend([
        "",
        "Images are stored in `images/`.",
    ])
    md_path = os.path.join(rep_dir, "v18_visual_review_summary.md")
    Path(md_path).write_text("\n".join(md_lines))

    print(f"Summary JSON: {json_path}")
    print(f"Summary MD:   {md_path}")


if __name__ == "__main__":
    main()
