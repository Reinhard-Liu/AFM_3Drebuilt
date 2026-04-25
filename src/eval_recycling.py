"""
V12 Recycling Inference: Iterative coordinate-type refinement.

Flow:
  Round 1: DDIM → GNN → type_logits
  Round 2: High-confidence types → MMFF94 force field → refined coords
  Round 3: Refined coords → GNN → final types

Usage:
    python -m src.eval_recycling \
        --diffusion_checkpoint experiments/v12/checkpoints/best_diffusion.pt \
        --gnn_checkpoint experiments/v12/gnn_checkpoints/best_gnn.pt \
        --num_samples 200 --num_rounds 2
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.dataset import create_dataloaders, MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES
from src.train import AFM3DReconModel
from src.models.gnn_type_classifier import GNNTypeClassifier
from src.models.postprocess import rdkit_relaxation, RDKIT_AVAILABLE
from src.utils.metrics import (
    compute_rmsd, compute_structure_similarity, compute_formula_similarity,
    compute_composite_score,
)
from src.eval_phase1 import (
    compute_conditional_type_acc, compute_pmi_similarity,
    compute_ring_preservation, chemical_type_correction,
)


def relaxation_with_confidence(coords, type_logits, mask, confidence_threshold=0.8):
    """Apply MMFF94 relaxation only using high-confidence atom types.

    Low-confidence atoms keep their coordinates unchanged.
    """
    if not RDKIT_AVAILABLE:
        return coords

    B = coords.shape[0]
    probs = torch.softmax(type_logits, dim=-1)
    max_probs, pred_types = probs.max(dim=-1)
    relaxed_coords = coords.clone()

    for b in range(B):
        n = int(mask[b].sum().item())
        if n < 3:
            continue

        # Only relax if enough high-confidence atoms
        high_conf = (max_probs[b, :n] > confidence_threshold)
        if high_conf.sum() < 3:
            continue

        c = coords[b, :n].detach().cpu().numpy()
        t = pred_types[b, :n].detach().cpu().numpy()
        elements = [ATOM_TYPES[int(ti)] for ti in t]

        try:
            relaxed = rdkit_relaxation(c, elements, max_displacement=0.03)
            if relaxed is not None:
                relaxed_t = torch.tensor(relaxed, device=coords.device, dtype=coords.dtype)
                relaxed_coords[b, :n] = relaxed_t
        except Exception:
            pass

    return relaxed_coords


@torch.no_grad()
def recycling_inference(diffusion_model, gnn, batch, device,
                         ddim_steps=100, num_rounds=2, confidence_threshold=0.8,
                         ensemble_alpha=0.6):
    """Optional side-branch evaluation: DDIM + GNN refinement. Not part of the V16b mainline."""

    afm = batch["afm_stack"]
    c_global, c_patches = diffusion_model.encoder(afm)
    n_atoms = diffusion_model.count_head.predict(c_global)

    # V16b side-branch evaluation still uses the real default diffusion path first.
    gen = diffusion_model.generate(batch, use_gt_count=False, use_ddim=True, ddim_steps=ddim_steps)
    coords = gen["coords"]
    denoiser_type_logits = gen["type_logits"]
    n_atoms = gen["n_atoms_pred"]

    B = coords.shape[0]
    mask = torch.zeros(B, MAX_ATOMS, device=device)
    for i in range(B):
        mask[i, :n_atoms[i]] = 1.0

    # GNN predict types (multiple rounds)
    for round_idx in range(num_rounds):
        gnn_type_logits = gnn(coords, c_patches, mask, afm_stack=afm)

    # V14 M1: Ensemble — weighted fusion of denoiser and GNN type probabilities
    if denoiser_type_logits is not None and ensemble_alpha < 1.0:
        denoiser_probs = F.softmax(denoiser_type_logits, dim=-1)
        gnn_probs = F.softmax(gnn_type_logits, dim=-1)
        fused_probs = ensemble_alpha * denoiser_probs + (1.0 - ensemble_alpha) * gnn_probs
        type_logits = torch.log(fused_probs.clamp(min=1e-8))
    else:
        type_logits = gnn_type_logits

    # Final chemical correction
    corrected_types = chemical_type_correction(coords, type_logits, mask)

    return {
        "coords": coords,
        "type_logits": type_logits,
        "pred_types": corrected_types,
        "n_atoms_pred": n_atoms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)
    parser.add_argument("--gnn_checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.8)
    parser.add_argument("--ensemble_alpha", type=float, default=0.6,
                        help="V14: denoiser weight in ensemble (1.0=denoiser only, 0.0=GNN only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load diffusion model
    state = torch.load(args.diffusion_checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    config["num_cids"] = 0
    diff_model = AFM3DReconModel(config).to(device)
    diff_model.load_state_dict(state["model"], strict=False)
    diff_model.eval()
    print(f"Diffusion model loaded (epoch {state.get('epoch', '?')})")

    # Load GNN
    gnn = GNNTypeClassifier(cond_dim=config["embed_dim"]).to(device)
    gnn_state = torch.load(args.gnn_checkpoint, map_location=device, weights_only=False)
    gnn.load_state_dict(gnn_state["gnn"])
    gnn.eval()
    print(f"GNN loaded (epoch {gnn_state.get('epoch', '?')}, val_acc={gnn_state.get('val_acc', '?')})")

    # Data
    _, _, test_loader, _ = create_dataloaders(
        data_root=config["data_root"], param_key=config["param_key"],
        img_size=config["img_size"], min_corrugation=config["min_corrugation"],
        augment_rotation=False, require_ring=config.get("require_ring", False),
        batch_size=config["batch_size"],
        num_workers=0, max_samples=config["max_samples"], val_size=config["val_size"],
    )

    # Evaluate
    all_type_match = []
    all_cond_type = []
    all_rmsd = []
    all_pmi = []
    all_ring = []
    all_formula = []
    all_coulomb = []
    all_bond = []
    all_bottom = []
    all_count_exact = []
    all_count_mae = []
    all_valence = []
    all_kabsch = []
    all_jsdiv = []
    count = 0

    def _to_device(batch):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    pbar = tqdm(test_loader, desc=f"Recycling ({args.num_rounds} rounds)")
    for batch in pbar:
        if count >= args.num_samples:
            break
        batch = _to_device(batch)

        result = recycling_inference(
            diff_model, gnn, batch, device,
            ddim_steps=args.ddim_steps,
            num_rounds=args.num_rounds,
            confidence_threshold=args.confidence_threshold,
            ensemble_alpha=args.ensemble_alpha,
        )

        coords_pred = result["coords"]
        pred_types = result["pred_types"]
        n_pred = result["n_atoms_pred"]

        rmsd = compute_rmsd(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
        all_rmsd.append(rmsd.mean().item())

        struct_sim = compute_structure_similarity(
            coords_pred, batch["coords"], pred_types, batch["atom_types"],
            batch["atom_mask"], n_atoms_pred=n_pred,
        )
        all_type_match.append(struct_sim["type_match_rate_mean"])

        cond = compute_conditional_type_acc(
            coords_pred, batch["coords"], pred_types, batch["atom_types"],
            batch["atom_mask"], n_atoms_pred=n_pred,
        )
        all_cond_type.append(cond["conditional_type_acc_mean"])

        pmi = compute_pmi_similarity(coords_pred, batch["coords"], batch["atom_mask"], n_atoms_pred=n_pred)
        all_pmi.append(pmi["pmi_similarity_mean"])

        ring = compute_ring_preservation(
            coords_pred, batch["coords"], pred_types, batch["atom_types"],
            batch["atom_mask"], n_atoms_pred=n_pred,
        )
        all_ring.append(ring["ring_preservation_mean"])

        formula = compute_formula_similarity(pred_types, batch["atom_types"], batch["atom_mask"], n_atoms_pred=n_pred)
        all_formula.append(formula["formula_similarity_mean"])

        # Additional metrics
        all_coulomb.append(struct_sim["coulomb_similarity_mean"])
        all_kabsch.append(struct_sim["kabsch_score_mean"])
        all_valence.append(struct_sim["valence_validity_mean"])
        all_jsdiv.append(struct_sim["distance_js_divergence_mean"])

        from src.utils.metrics import compute_bond_validity, compute_bottom_atom_recall, compute_atom_count_accuracy
        bond = compute_bond_validity(coords_pred, pred_types, batch["atom_mask"])
        all_bond.append(bond.mean().item())

        recall = compute_bottom_atom_recall(coords_pred, batch["coords"], pred_types, batch["atom_types"], batch["atom_mask"])
        all_bottom.append(recall.mean().item())

        count_acc = compute_atom_count_accuracy(n_pred, batch["n_atoms"])
        all_count_exact.append(count_acc["exact_match"])
        all_count_mae.append(count_acc["mae"])

        count += batch["afm_stack"].shape[0]
        pbar.set_postfix(type=f"{np.mean(all_cond_type):.3f}")

    # V14: Compute Composite score
    composite = compute_composite_score(
        rmsd=np.mean(all_rmsd),
        bottom_atom_score=np.mean(all_bottom),
        bond_validity=np.mean(all_bond),
        ring_preservation=np.mean(all_ring),
        atom_count_accuracy=np.mean(all_count_exact),
        structure_similarity=np.mean(all_kabsch),
    )

    print()
    print("=" * 70)
    print(f"V14 Recycling Inference ({args.num_rounds} rounds, ensemble_alpha={args.ensemble_alpha})")
    print("=" * 70)
    print(f"RMSD:                    {np.mean(all_rmsd):.4f}")
    print(f"Kabsch Score:            {np.mean(all_kabsch):.4f}")
    print(f"Type Match:              {np.mean(all_type_match):.4f}")
    print(f"Conditional Type Acc:    {np.mean(all_cond_type):.4f}")
    print(f"Coulomb Similarity:      {np.mean(all_coulomb):.4f}")
    print(f"Dist JS-Div:             {np.mean(all_jsdiv):.4f}")
    print(f"Bond Validity:           {np.mean(all_bond):.4f}")
    print(f"Valence Validity:        {np.mean(all_valence):.4f}")
    print(f"Count Accuracy:          {np.mean(all_count_exact):.4f} (MAE: {np.mean(all_count_mae):.4f})")
    print(f"Bottom Recall:           {np.mean(all_bottom):.4f}")
    print(f"Formula Similarity:      {np.mean(all_formula):.4f}")
    print(f"PMI Shape Similarity:    {np.mean(all_pmi):.4f}")
    print(f"Ring Preservation:       {np.mean(all_ring):.4f}")
    print(f"Composite Score:         {composite:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
