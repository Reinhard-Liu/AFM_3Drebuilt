"""
Quick sanity test: verify all modules can be imported and work together.
Runs a minimal forward/backward pass on CPU with tiny config.
"""

import sys
import numpy as np
import torch

from src.data.dataset import QUAMAFMDataset, MAX_ATOMS, NUM_ATOM_TYPES
from src.models.video_vit import VideoViTEncoder
from src.models.diffusion import SE3EquivariantDenoiser, ConditionalDDPM
from src.models.baselines import ResNet3DRegression
from src.models.prediction_heads import AtomCountHead, MoleculeRetrievalHead
from src.models.ring_detection import detect_rings, pad_ring_info, RING_TEMPLATES
from src.models.constraints import bond_length_penalty, compute_all_constraints
from src.utils.metrics import (
    compute_rmsd, compute_bottom_atom_recall,
    compute_atom_count_accuracy, compute_bond_validity,
    compute_bottom_atom_rmsd, compute_composite_score,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Test Dataset ---
    print("\n[1] Testing Dataset...")
    try:
        ds = QUAMAFMDataset(
            "micro/dataverse_files/SUBMIT_QUAM-AFM/QUAM",
            param_key="K-1", img_size=64, split="train",
        )
        print(f"  Dataset size: {len(ds)}")
        sample = ds[0]
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")
    except Exception as e:
        print(f"  Dataset not available (expected if no data): {e}")

    # --- Test Video ViT ---
    print("\n[2] Testing Video ViT Encoder...")
    encoder = VideoViTEncoder(
        img_size=64, num_frames=10, patch_size=16,
        temporal_patch_size=2, embed_dim=64, depth=2, num_heads=4,
    ).to(device)
    x = torch.randn(1, 10, 64, 64, device=device)
    c_global, c_patches = encoder(x)
    print(f"  Input: {x.shape} -> c_global: {c_global.shape}, c_patches: {c_patches.shape}")

    # --- Test Diffusion ---
    print("\n[3] Testing Conditional DDPM...")
    denoiser = SE3EquivariantDenoiser(
        max_atoms=MAX_ATOMS, cond_dim=64,
        hidden_dim=64, num_layers=2, num_heads=4,
    ).to(device)
    ddpm = ConditionalDDPM(denoiser, timesteps=50).to(device)

    coords = torch.randn(1, MAX_ATOMS, 3, device=device) * 0.1
    types = torch.randint(0, NUM_ATOM_TYPES, (1, MAX_ATOMS), device=device)
    mask = torch.ones(1, MAX_ATOMS, device=device)
    mask[0, 30:] = 0  # simulate 30-atom molecule

    # V15+: compute_loss takes (x_0, c_global, c_patches, atom_types, mask)
    losses = ddpm.compute_loss(coords, c_global, c_patches, types, mask)
    print(f"  Loss: {losses['loss'].item():.4f}")
    print(f"  Coord loss: {losses['coord_loss'].item():.4f}")
    print(f"  Type loss: {losses['type_loss'].item():.4f}")

    # Backward
    losses["loss"].backward()
    print("  Backward pass: OK")

    # --- Test Prediction Heads ---
    print("\n[4] Testing Prediction Heads...")
    count_head = AtomCountHead(embed_dim=64, max_count=MAX_ATOMS).to(device)
    cls_logits, reg_val = count_head(c_global)
    print(f"  Count cls logits: {cls_logits.shape}, reg value: {reg_val.shape}")
    predicted_n = count_head.predict(c_global)
    print(f"  Predicted atom count: {predicted_n.item()}")

    n_atoms_gt = torch.tensor([30], device=device)
    count_loss = count_head.compute_loss(c_global, n_atoms_gt)
    print(f"  Count loss: {count_loss['count_loss'].item():.4f}")

    retrieval_head = MoleculeRetrievalHead(embed_dim=64, proj_dim=32).to(device)
    proj = retrieval_head(c_global)
    print(f"  Retrieval projection: {proj.shape}, norm: {proj.norm(dim=-1).item():.4f}")

    mol_emb = torch.nn.Embedding(100, 32).to(device)
    ret_loss = retrieval_head.compute_loss(c_global, torch.tensor([5], device=device), mol_emb)
    print(f"  Retrieval loss: {ret_loss['retrieval_loss'].item():.4f}")

    scores, indices = retrieval_head.retrieve(c_global, mol_emb, top_k=5)
    print(f"  Top-5 retrieved: indices={indices[0].tolist()}")

    # --- Test Ring Detection ---
    print("\n[5] Testing Ring Detection...")
    # Create a fake benzene ring in normalized space
    radius = 1.40 / 12.0
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    benzene_coords = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros(6),
    ], axis=-1).astype(np.float32)
    # Add a few hydrogens
    h_coords = benzene_coords * 1.7  # roughly attached to ring
    all_coords = np.concatenate([benzene_coords, h_coords], axis=0)
    elements = ["C"] * 6 + ["H"] * 6

    ring_info = detect_rings(all_coords, elements, normalized=True)
    print(f"  Detected {ring_info['n_rings']} rings")
    if ring_info['n_rings'] > 0:
        print(f"  Ring types: {ring_info['ring_types']}")
        print(f"  Ring centers: {ring_info['ring_centers']}")

    padded = pad_ring_info(ring_info)
    print(f"  Padded ring_centers shape: {padded['ring_centers'].shape}")
    print(f"  Ring templates available: {list(RING_TEMPLATES.keys())}")

    # --- Test Constraints ---
    print("\n[6] Testing Physical Constraints...")
    test_coords = torch.randn(2, MAX_ATOMS, 3) * 0.1
    test_types = torch.randint(0, NUM_ATOM_TYPES, (2, MAX_ATOMS))
    test_mask = torch.ones(2, MAX_ATOMS)
    test_mask[:, 40:] = 0

    bl_loss = bond_length_penalty(test_coords, test_types, test_mask)
    print(f"  Bond length penalty: {bl_loss.item():.6f}")

    all_constraints = compute_all_constraints(test_coords, test_types, test_mask)
    for k, v in all_constraints.items():
        print(f"  {k}: {v.item():.6f}")

    # --- Test Baseline ---
    print("\n[7] Testing ResNet3D Baseline...")
    resnet = ResNet3DRegression(
        img_size=64, num_frames=10, max_atoms=MAX_ATOMS,
        num_atom_types=NUM_ATOM_TYPES, base_ch=8,
    ).to(device)
    afm = torch.randn(1, 10, 64, 64, device=device)
    res_losses = resnet.compute_loss(afm, coords, types, mask)
    print(f"  Loss: {res_losses['loss'].item():.4f}")
    res_losses["loss"].backward()
    print("  Backward pass: OK")

    # --- Test Metrics ---
    print("\n[8] Testing Metrics...")
    pred_c = torch.randn(2, MAX_ATOMS, 3)
    gt_c = torch.randn(2, MAX_ATOMS, 3)
    m = torch.ones(2, MAX_ATOMS)
    m[:, 40:] = 0

    rmsd = compute_rmsd(pred_c, gt_c, m)
    print(f"  RMSD: {rmsd.tolist()}")

    pred_t = torch.randint(0, NUM_ATOM_TYPES, (2, MAX_ATOMS))
    gt_t = torch.randint(0, NUM_ATOM_TYPES, (2, MAX_ATOMS))
    recall = compute_bottom_atom_recall(pred_c, gt_c, pred_t, gt_t, m)
    print(f"  Bottom Recall: {recall.tolist()}")

    bottom_rmsd = compute_bottom_atom_rmsd(pred_c, gt_c, m)
    print(f"  Bottom RMSD: {bottom_rmsd.tolist()}")

    bond_valid = compute_bond_validity(pred_c, pred_t, m)
    print(f"  Bond Validity: {bond_valid.tolist()}")

    count_acc = compute_atom_count_accuracy(
        torch.tensor([30, 40]), torch.tensor([30, 42]),
    )
    print(f"  Count Accuracy: exact={count_acc['exact_match']:.2f}, mae={count_acc['mae']:.2f}")

    composite = compute_composite_score(
        rmsd=0.5, bottom_atom_score=0.6, bond_validity=0.8,
        ring_preservation=0.7, atom_count_accuracy=0.9, structure_similarity=0.5,
    )
    print(f"  Composite Score: {composite:.4f}")

    # --- Test RDKit Postprocess ---
    print("\n[9] Testing RDKit Postprocess...")
    try:
        from src.models.postprocess import rdkit_relaxation, RDKIT_AVAILABLE
        print(f"  RDKit available: {RDKIT_AVAILABLE}")
        relaxed = rdkit_relaxation(pred_c[:1], pred_t[:1], m[:1])
        print(f"  Relaxation output shape: {relaxed.shape}")
    except Exception as e:
        print(f"  Postprocess test: {e}")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    main()
