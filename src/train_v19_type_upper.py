"""
V19.3: GT-coordinate atom-type upper-bound experiment.

Purpose:
    Estimate how much atom-type information is present in the AFM stack when
    atom coordinates are already correct. This isolates the "type from AFM"
    problem from the much harder "joint coordinate + type reconstruction"
    problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.dataset import ATOM_TYPES, create_dataloaders
from src.models.gnn_type_classifier import GNNTypeClassifier
from src.models.video_vit import VideoViTEncoder


H_IDX = 0
C_IDX = 1
HETERO_CLASSES = {2, 3, 4, 5, 6, 7, 8, 9}


def _macro_f1_from_lists(gt_labels, pred_labels, n_classes=10):
    f1s = []
    for cls in range(n_classes):
        tp = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gt_labels, pred_labels) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gt_labels, pred_labels) if g == cls and p != cls)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(0.0 if prec + rec == 0 else 2.0 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(encoder, classifier, loader, device):
    encoder.eval()
    classifier.eval()

    total_loss = 0.0
    n_batches = 0
    pred_all = []
    gt_all = []
    pred_heavy = []
    gt_heavy = []
    pred_hetero = []
    gt_hetero = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)

            _, c_patches = encoder(afm)
            loss = classifier.compute_loss(coords, c_patches, atom_types, mask, afm_stack=afm)
            logits = classifier(coords, c_patches, mask, afm_stack=afm)
            pred = logits.argmax(dim=-1)

            total_loss += float(loss.item())
            n_batches += 1

            valid = (mask > 0) & (atom_types >= 0)
            heavy = valid & (atom_types != H_IDX)
            hetero = valid & (~torch.isin(atom_types, torch.tensor([H_IDX, C_IDX], device=device)))

            pred_all.extend(pred[valid].detach().cpu().tolist())
            gt_all.extend(atom_types[valid].detach().cpu().tolist())
            pred_heavy.extend(pred[heavy].detach().cpu().tolist())
            gt_heavy.extend(atom_types[heavy].detach().cpu().tolist())
            pred_hetero.extend((pred[hetero].detach().cpu().numpy() != C_IDX).astype(np.int64).tolist())
            gt_hetero.extend((atom_types[hetero].detach().cpu().numpy() != C_IDX).astype(np.int64).tolist())

    val_loss = total_loss / max(n_batches, 1)

    all_arr = np.asarray(pred_all, dtype=np.int64)
    gt_arr = np.asarray(gt_all, dtype=np.int64)
    heavy_arr = np.asarray(pred_heavy, dtype=np.int64)
    gt_heavy_arr = np.asarray(gt_heavy, dtype=np.int64)

    type_acc_micro = float((all_arr == gt_arr).mean()) if len(gt_arr) else 0.0
    heavy_type_acc = float((heavy_arr == gt_heavy_arr).mean()) if len(gt_heavy_arr) else 0.0
    macro_f1 = _macro_f1_from_lists(gt_all, pred_all, n_classes=len(ATOM_TYPES))

    pred_ch_fraction = float(np.isin(all_arr, [H_IDX, C_IDX]).mean()) if len(all_arr) else 0.0
    gt_ch_fraction = float(np.isin(gt_arr, [H_IDX, C_IDX]).mean()) if len(gt_arr) else 0.0
    ch_collapse_rate = float(np.clip((pred_ch_fraction - gt_ch_fraction) / max(1.0 - gt_ch_fraction, 1e-6), 0.0, 1.0))

    pred_het_labels = np.asarray(pred_hetero, dtype=np.int64)
    gt_het_labels = np.asarray(gt_hetero, dtype=np.int64)
    tp = int(((pred_het_labels == 1) & (gt_het_labels == 1)).sum())
    fp = int(((pred_het_labels == 1) & (gt_het_labels == 0)).sum())
    fn = int(((pred_het_labels == 0) & (gt_het_labels == 1)).sum())
    hetero_precision = tp / max(tp + fp, 1)
    hetero_recall = tp / max(tp + fn, 1)
    hetero_f1 = 0.0 if hetero_precision + hetero_recall == 0 else 2.0 * hetero_precision * hetero_recall / (hetero_precision + hetero_recall)

    return {
        "val_loss": float(val_loss),
        "type_acc_micro": float(type_acc_micro),
        "heavy_type_acc": float(heavy_type_acc),
        "type_macro_f1": float(macro_f1),
        "hetero_precision": float(hetero_precision),
        "hetero_recall": float(hetero_recall),
        "hetero_f1": float(hetero_f1),
        "ch_collapse_rate": float(ch_collapse_rate),
    }


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, _ = create_dataloaders(
        data_root=config["data_root"],
        param_key=config.get("param_key", "K-1"),
        img_size=config["img_size"],
        min_corrugation=config.get("min_corrugation", 0.0),
        augment_rotation=config.get("augment_rotation", True),
        require_ring=config.get("require_ring", False),
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
        max_samples=config.get("max_samples", 0),
        val_size=config.get("val_size", 0),
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    encoder = VideoViTEncoder(
        img_size=config["img_size"],
        num_frames=config.get("num_frames", 10),
        patch_size=config.get("patch_size", 16),
        temporal_patch_size=config.get("temporal_patch_size", 2),
        embed_dim=config.get("embed_dim", 256),
        depth=config.get("encoder_depth", 4),
        num_heads=config.get("num_heads", 4),
        drop_rate=config.get("drop_rate", 0.1),
    ).to(device)

    classifier = GNNTypeClassifier(
        cond_dim=config.get("embed_dim", 256),
        hidden_dim=config.get("hidden_dim", 128),
        num_gnn_layers=config.get("num_gnn_layers", 4),
        num_types=len(ATOM_TYPES),
        num_heads=config.get("num_heads", 4),
        bond_threshold=config.get("bond_threshold", 0.20),
    ).to(device)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=config.get("lr", 1e-4), weight_decay=config.get("weight_decay", 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config.get("epochs", 1)), 1), eta_min=config.get("min_lr", 1e-5))

    save_dir = Path(config.get("save_dir", "experiments/v19_type_upper/checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_key = None
    epochs = int(config.get("epochs", 2))

    for epoch in range(1, epochs + 1):
        encoder.train()
        classifier.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"V19.3-type-upper [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)

            _, c_patches = encoder(afm)
            loss = classifier.compute_loss(coords, c_patches, atom_types, mask, afm_stack=afm)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(train_losses[-10:]):.4f}")

        scheduler.step()

        val_metrics = evaluate(encoder, classifier, val_loader, device)
        train_metrics = {
            "loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss {train_metrics['loss']:.4f} | "
            f"val_loss {val_metrics['val_loss']:.4f} | "
            f"type_acc {val_metrics['type_acc_micro']:.4f} | "
            f"heavy_acc {val_metrics['heavy_type_acc']:.4f} | "
            f"macro_f1 {val_metrics['type_macro_f1']:.4f} | "
            f"hetero_f1 {val_metrics['hetero_f1']:.4f} | "
            f"collapse {val_metrics['ch_collapse_rate']:.4f}"
        )

        current_key = (
            float(val_metrics["type_macro_f1"]),
            float(val_metrics["heavy_type_acc"]),
            float(val_metrics["hetero_f1"]),
            float(val_metrics["type_acc_micro"]),
            -float(val_metrics["ch_collapse_rate"]),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            torch.save(
                {
                    "epoch": epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                save_dir / "best_v19_type_upper.pt",
            )

    with open(save_dir / "history_v19_type_upper.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": epochs,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "config": config,
        },
        save_dir / "last_v19_type_upper.pt",
    )
    print(f"Saved checkpoints to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    base_dir = Path(__file__).resolve().parents[1]
    if config.get("data_root") == "auto":
        config["data_root"] = str(base_dir / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = str(base_dir / "experiments" / "v19_type_upper_debug" / "checkpoints")

    train(config)
