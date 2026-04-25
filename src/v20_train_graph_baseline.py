"""
SUP-02:
Lightweight graph reconstruction baseline.

Route:
AFM stack -> proposal maps/count/z -> predicted node proposals -> legacy graph type head + edge head

This baseline intentionally avoids the V20 center-conditioned typed-graph
curriculum. It keeps the proposal extractor, but learns graph reconstruction
only on predicted proposal sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MAX_ATOMS, create_dataloaders
from src.models.v19_center_edge_head import CenterConditionedEdgeHead
from src.models.v19_joint_model import V19JointUNet
from src.models.v20_ablation_heads import LegacyGNNTypeHeadAdapter
from src.train_v19_object_joint import (
    TYPE_CLASS_WEIGHTS,
    build_edge_labels,
    build_predicted_edge_training_labels,
    build_predicted_type_training_batch,
    build_targets,
    evaluate,
    maybe_neutralize_z_map,
    maybe_strip_z_coords,
    save_preview,
)


def resolve_config(config: dict) -> dict:
    config = dict(config)
    if config.get("data_root") == "auto":
        config["data_root"] = str(ROOT / "dataverse_files" / "SUBMIT_QUAM-AFM" / "QUAM")
    if config.get("save_dir") == "auto":
        config["save_dir"] = str(ROOT / "experiments" / "v20_graph_baseline_medium10" / "checkpoints")
    return config


def build_graph_baseline_components(config: dict, device: torch.device):
    model = V19JointUNet(in_channels=10, base_ch=int(config.get("base_ch", 64))).to(device)
    type_head = LegacyGNNTypeHeadAdapter(
        shared_feat_dim=int(config.get("base_ch", 64)),
        hidden_dim=int(config.get("type_hidden_dim", 192)),
        num_types=10,
        num_gnn_layers=int(config.get("legacy_type_num_gnn_layers", 4)),
        num_heads=int(config.get("legacy_type_num_heads", 4)),
        bond_threshold=float(config.get("legacy_type_bond_threshold", 0.20)),
        token_grid_size=int(config.get("legacy_type_token_grid_size", 16)),
        label_smoothing=float(config.get("type_label_smoothing", 0.0)),
    ).to(device)
    edge_head = CenterConditionedEdgeHead(shared_feat_dim=int(config.get("base_ch", 64))).to(device)
    return model, type_head, edge_head


def load_graph_baseline_checkpoint(checkpoint_path: Path, config_path: Path | None, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in state and state["config"]:
        config = dict(state["config"])
    elif config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        raise FileNotFoundError("No config found in checkpoint and no config_path provided.")
    config = resolve_config(config)

    model, type_head, edge_head = build_graph_baseline_components(config, device)
    model.load_state_dict(state["model"], strict=False)
    type_head.load_state_dict(state["type_head"], strict=False)
    edge_head.load_state_dict(state["edge_head"], strict=False)
    model.eval()
    type_head.eval()
    edge_head.eval()
    return model, type_head, edge_head, config, state


def train(config: dict):
    config = resolve_config(config)
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

    model, type_head, edge_head = build_graph_baseline_components(config, device)
    params = list(model.parameters()) + list(type_head.parameters()) + list(edge_head.parameters())
    optimizer = optim.AdamW(
        params,
        lr=float(config.get("lr", 2e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(config.get("epochs", 1)), 1),
        eta_min=float(config.get("min_lr", 1e-5)),
    )

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_key = None
    start_epoch = 1

    resume_path_str = config.get("resume_from_checkpoint", "") or ""
    if resume_path_str:
        resume_path = Path(resume_path_str)
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            type_head.load_state_dict(state["type_head"], strict=False)
            edge_head.load_state_dict(state["edge_head"], strict=False)
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            history = state.get("history", history)
            saved_best = state.get("best_key")
            if saved_best is not None:
                best_key = tuple(saved_best)
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"Resumed from {resume_path} at epoch {start_epoch}")

    epochs = int(config.get("epochs", 10))
    if start_epoch > epochs:
        print(f"resume checkpoint already reached target epochs ({epochs}); nothing to do.")
        return

    type_class_weights = TYPE_CLASS_WEIGHTS.to(device)
    lambda_center = float(config.get("lambda_center", 20.0))
    lambda_atom_aux = float(config.get("lambda_atom_aux", 2.0))
    lambda_z = float(config.get("lambda_z", 6.0))
    lambda_object_count = float(config.get("lambda_object_count", 1.0))
    lambda_object_count_mae = float(config.get("lambda_object_count_mae", 0.15))
    lambda_type_obj_pred = float(config.get("lambda_type_obj_pred", 2.0))
    lambda_edge_obj_pred = float(config.get("lambda_edge_obj_pred", 1.0))
    pred_train_match_radius_px = float(config.get("pred_train_match_radius_px", 4.0))
    disable_z_for_object_heads = bool(config.get("disable_z_for_object_heads", False))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        type_head.train()
        edge_head.train()
        losses = []
        count_mae_values = []
        matched_rate_values = []

        pbar = tqdm(train_loader, desc=f"V20-graph [{epoch}/{epochs}]", leave=False)
        for batch in pbar:
            afm = batch["afm_stack"].to(device)
            coords = batch["coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            targets = build_targets(batch, config["img_size"], device)
            edge_labels = build_edge_labels(batch, device)

            pred, features = model.forward_with_features(afm)
            pred_01 = (pred + 1.0) * 0.5
            pred_01_heads = maybe_neutralize_z_map(pred_01, disable_z_for_object_heads)
            center_logits = features["center_logits"]
            center_map_01 = torch.sigmoid(center_logits)
            count_logits = features["count_logits"]

            target_center = targets[:, 0:1]
            count_targets = mask.sum(dim=1).long().clamp(min=0, max=MAX_ATOMS)
            pos = float(target_center.sum().item())
            neg = float(target_center.numel() - pos)
            center_pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0), device=device)
            center_bce = F.binary_cross_entropy_with_logits(center_logits, target_center, pos_weight=center_pos_weight)
            center_l1 = F.l1_loss(center_map_01, target_center)
            center_loss = (0.75 * center_bce + 0.25 * center_l1) * lambda_center

            count_ce = F.cross_entropy(count_logits, count_targets)
            count_pred = count_logits.argmax(dim=-1)
            count_mae = (count_pred.float() - count_targets.float()).abs().mean()
            count_loss = lambda_object_count * count_ce + lambda_object_count_mae * count_mae

            atom_map_aux = F.l1_loss(pred_01[:, 0:1], target_center) * lambda_atom_aux
            z_mask = targets[:, 0:1]
            z_loss = ((pred_01[:, 12:13] - targets[:, 12:13]).abs() * z_mask).sum() / z_mask.sum().clamp(min=1.0)
            z_loss = z_loss * (0.0 if disable_z_for_object_heads else lambda_z)

            coords_obj = maybe_strip_z_coords(coords, disable_z_for_object_heads)
            pred_train_coords, pred_train_types, pred_train_mask, pred_train_gt_index = build_predicted_type_training_batch(
                center_map_01,
                pred_01_heads,
                count_logits,
                coords_obj,
                atom_types,
                mask,
                config["img_size"],
                max_objects=MAX_ATOMS,
                match_radius_px=pred_train_match_radius_px,
            )
            pred_type_obj_loss, _ = type_head.compute_loss(
                pred_train_coords,
                features["enc1"],
                afm,
                pred_train_types,
                pred_train_mask,
                class_weight=type_class_weights,
                center_map=None,
            )
            pred_type_obj_loss = pred_type_obj_loss * lambda_type_obj_pred

            pred_edge_labels = build_predicted_edge_training_labels(pred_train_gt_index, pred_train_mask, edge_labels)
            pred_edge_obj_loss, _ = edge_head.compute_loss(
                pred_train_coords,
                features["enc1"],
                afm,
                pred_train_mask,
                pred_edge_labels,
            )
            pred_edge_obj_loss = pred_edge_obj_loss * lambda_edge_obj_pred

            loss = center_loss + count_loss + atom_map_aux + z_loss + pred_type_obj_loss + pred_edge_obj_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            count_mae_values.append(float(count_mae.item()))
            matched_valid = ((pred_train_mask > 0) & (pred_train_gt_index >= 0)).float()
            matched_rate_values.append(float(matched_valid.sum().item() / max(mask.sum().item(), 1.0)))
            pbar.set_postfix(
                loss=f"{np.mean(losses[-10:]):.3f}",
                cnt=f"{np.mean(count_mae_values[-10:]):.2f}",
                match=f"{np.mean(matched_rate_values[-10:]):.2f}",
            )

        scheduler.step()
        train_metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "count_mae_head": float(np.mean(count_mae_values)) if count_mae_values else 0.0,
            "matched_rate": float(np.mean(matched_rate_values)) if matched_rate_values else 0.0,
        }
        val_metrics = evaluate(model, type_head, edge_head, val_loader, config, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss {train_metrics['loss']:.4f} | "
            f"pred_score {val_metrics.get('pred_object_score', 0.0):.4f} | "
            f"pred_type_acc {val_metrics.get('pred_object_type_acc', 0.0):.4f} | "
            f"pred_macro_f1 {val_metrics.get('pred_object_macro_f1', 0.0):.4f} | "
            f"pred_edge_f1 {val_metrics.get('pred_object_edge_f1', 0.0):.4f} | "
            f"robust_edge_f1 {val_metrics.get('pred_object_edge_f1_robust', 0.0):.4f} | "
            f"pred_count_mae {val_metrics.get('pred_object_count_mae', 0.0):.4f} | "
            f"pred_z_mae {val_metrics.get('pred_object_z_mae', 0.0):.4f}"
        )

        current_key = (
            float(val_metrics.get("pred_object_score", 0.0)),
            float(val_metrics.get("pred_object_type_acc", 0.0)),
            float(val_metrics.get("pred_object_macro_f1", 0.0)),
            float(val_metrics.get("pred_object_edge_f1_robust", 0.0)),
            float(val_metrics.get("pred_object_edge_f1", 0.0)),
            -float(val_metrics.get("pred_object_count_mae", 999.0)),
            -float(val_metrics.get("pred_object_z_mae", 999.0)),
        )
        latest_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "type_head": type_head.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key) if best_key is not None else None,
            "val_metrics": val_metrics,
            "config": config,
        }
        torch.save(latest_state, save_dir / "latest_v20_graph_baseline.pt")

        if best_key is None or current_key > best_key:
            best_key = current_key
            latest_state["best_key"] = list(best_key)
            torch.save(latest_state, save_dir / "best_v20_graph_baseline.pt")
            save_preview(model, val_loader, config, device, save_dir / "best_preview.png")

        with open(save_dir / "history_v20_graph_baseline.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "type_head": type_head.state_dict(),
            "edge_head": edge_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key) if best_key is not None else None,
            "config": config,
        },
        save_dir / "last_v20_graph_baseline.pt",
    )
    print(f"Saved checkpoints to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    if args.resume_checkpoint:
        config["resume_from_checkpoint"] = args.resume_checkpoint
    train(config)
