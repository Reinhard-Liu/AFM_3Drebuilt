"""
Training pipeline for the Video ViT + Conditional Diffusion model,
and the baseline models.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import (
    QUAMAFMDataset, create_dataloaders,
    MAX_ATOMS, NUM_ATOM_TYPES, ATOM_TYPES,
)
from src.models.video_vit import VideoViTEncoder
from src.models.diffusion import SE3EquivariantDenoiser, ConditionalDDPM, compute_shape_descriptors
from src.models.baselines import ResNet3DRegression
from src.models.prediction_heads import (
    AtomCountHead, MoleculeRetrievalHead, RingDetectionHead, ScaffoldRelationHead,
    ScaffoldStructureCountHead, AtomSemanticHead, SiteGraphParserHead,
)
from src.models.scaffold_tokens import (
    GTScaffoldTokenEncoder,
    GTSlotSemanticConditionEncoder,
    GTSlotConnectionGraphEncoder,
)
from src.utils.metrics import (
    compute_rmsd, compute_bottom_atom_recall,
    compute_atom_count_accuracy, compute_bond_validity,
    compute_bottom_atom_rmsd, compute_composite_score,
    compute_structure_similarity, compute_formula_similarity,
    compute_structure_fidelity,
)
from src.models.constraints import compute_all_constraints
from src.eval_phase1 import compute_ring_preservation


class Logger:
    """Logger that writes to both console and file."""
    def __init__(self, log_file, mode='a'):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.mode = mode
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, self.mode) as f:
            f.write(message)
        if self.mode == 'w':
            self.mode = 'a'

    def flush(self):
        self.terminal.flush()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Main model: Video ViT + DDPM
# ============================================================

class AFM3DReconModel(nn.Module):
    """Full model: Video ViT encoder + Conditional DDPM + prediction heads."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.encoder = VideoViTEncoder(
            img_size=config["img_size"],
            num_frames=config["num_frames"],
            patch_size=config["patch_size"],
            temporal_patch_size=config["temporal_patch_size"],
            embed_dim=config["embed_dim"],
            depth=config["encoder_depth"],
            num_heads=config["num_heads"],
            drop_rate=config["drop_rate"],
        )
        denoiser = SE3EquivariantDenoiser(
            max_atoms=MAX_ATOMS,
            coord_dim=3,
            num_atom_types=NUM_ATOM_TYPES,
            cond_dim=config["embed_dim"],
            hidden_dim=config["denoiser_hidden_dim"],
            num_layers=config["denoiser_depth"],
            num_heads=config["num_heads"],
            scaffold_attn_every=config.get("v17_scaffold_cross_attn_every", 2),
            scaffold_attn_scale=config.get("v17_scaffold_cross_attn_scale", 0.1),
            slot_graph_scale=config.get("v18_slot_graph_scale", 0.15),
        )
        self.ddpm = ConditionalDDPM(
            denoiser=denoiser,
            timesteps=config["diffusion_steps"],
        )

        # Atom count prediction head
        self.count_head = AtomCountHead(
            embed_dim=config["embed_dim"],
            max_count=MAX_ATOMS,
        )

        # V9: Shape prediction head
        self.shape_head = nn.Sequential(
            nn.Linear(config["embed_dim"], 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.type_predictor = None  # V16b: keep type prediction on the main denoiser path by default

        # Molecule retrieval head + embedding database
        num_cids = config.get("num_cids", 0)
        self.retrieval_head = MoleculeRetrievalHead(
            embed_dim=config["embed_dim"],
            proj_dim=128,
            temperature=0.07,
        )
        if num_cids > 0:
            self.mol_embeddings = nn.Embedding(num_cids, 128)
        else:
            self.mol_embeddings = None

        # V16: Ring detection head
        self.ring_head = RingDetectionHead(
            embed_dim=config["embed_dim"],
        )

        # V17-Bridge: GT scaffold token encoder.
        # This does not affect the current training or sampling path yet; it
        # only prepares a clean token interface for the upcoming cross-attention
        # bridge experiment.
        self.scaffold_token_encoder = GTScaffoldTokenEncoder(
            embed_dim=config["embed_dim"],
            hidden_dim=config.get("denoiser_hidden_dim", 256),
            num_atom_types=NUM_ATOM_TYPES,
            max_attachment_tokens=config.get("v17_max_attachment_tokens", 24),
            max_sidechain_tokens=config.get("v17_max_sidechain_tokens", 48),
            include_sidechain_tokens=config.get("v17_include_sidechain_tokens", True),
            include_site_tokens=config.get("v18_include_gt_site_tokens", False),
            max_site_tokens=config.get("v18_max_site_tokens", 48),
            max_graph_depth=config.get("v17_max_graph_depth", 8),
        )
        self.scaffold_relation_head = ScaffoldRelationHead(
            embed_dim=config["embed_dim"],
            hidden_dim=config.get("denoiser_hidden_dim", 256),
            max_attachment_tokens=config.get("v17_max_attachment_tokens", 24),
            max_sidechain_tokens=config.get("v17_max_sidechain_tokens", 48),
            num_atom_types=NUM_ATOM_TYPES,
            max_graph_depth=config.get("v17_max_graph_depth", 8),
        )
        self.scaffold_count_head = ScaffoldStructureCountHead(
            embed_dim=config["embed_dim"],
            hidden_dim=config.get("denoiser_hidden_dim", 256),
        )
        self.slot_semantic_condition_encoder = GTSlotSemanticConditionEncoder(
            embed_dim=config["embed_dim"],
            hidden_dim=config.get("denoiser_hidden_dim", 256),
            max_atoms=MAX_ATOMS,
            num_atom_types=NUM_ATOM_TYPES,
        )
        self.slot_connection_graph_encoder = GTSlotConnectionGraphEncoder(
            max_atoms=MAX_ATOMS,
        )
        self.atom_semantic_head = AtomSemanticHead(
            embed_dim=config["embed_dim"],
            num_atom_types=NUM_ATOM_TYPES,
            hidden_dim=config.get("denoiser_hidden_dim", 256),
            max_atoms=MAX_ATOMS,
        )
        self.site_graph_parser_head = SiteGraphParserHead(
            embed_dim=config["embed_dim"],
            num_atom_types=NUM_ATOM_TYPES,
            hidden_dim=config.get("denoiser_hidden_dim", 256),
            max_site_tokens=config.get("v18_max_site_queries", 48),
        )

    def encode_scaffold_tokens(
        self,
        batch: dict,
        c_patches: torch.Tensor = None,
        predicted_relations: dict | None = None,
    ) -> tuple[dict | None, dict | None]:
        """Build scaffold conditioning tokens.

        The GT ring-system backbone is always used. When enabled, local
        attachment/sidechain relation tokens come from the prediction head
        instead of GT labels, so relation prediction can actually affect the
        denoiser path.
        """
        gt_cond = self.encode_gt_scaffold_tokens(batch)
        if gt_cond is None:
            return None, predicted_relations

        if not self.config.get("use_v17_bridge_predicted_relation_tokens", False):
            return gt_cond, predicted_relations

        teacher_force_prob = self.config.get("v17_relation_token_teacher_force_prob", 0.0)
        use_gt_relations = (
            self.training and teacher_force_prob > 0.0 and torch.rand(1).item() < teacher_force_prob
        )
        if use_gt_relations:
            return gt_cond, predicted_relations

        if predicted_relations is None:
            if c_patches is None:
                return gt_cond, predicted_relations
            predicted_relations = self.scaffold_relation_head(c_patches)

        pred_cond = self.scaffold_token_encoder.forward_from_predicted_relations(
            batch, predicted_relations
        )
        return pred_cond, predicted_relations

    def encode_gt_scaffold_tokens(self, batch: dict) -> dict | None:
        """Encode GT scaffold labels into token sequences for V17-Bridge.

        Returns None if the batch does not carry the optional V17 scaffold
        labels. This keeps existing V16 training/eval flows unchanged.
        """
        required = [
            "scaffold_n_ring_systems",
            "scaffold_system_objectness",
            "scaffold_relation_edges",
            "scaffold_atom_is_attachment_anchor",
            "scaffold_atom_to_ring_system_ids",
        ]
        if not all(k in batch for k in required):
            return None

        return self.scaffold_token_encoder(
            batch,
            coords=batch["coords"],
            atom_types=batch["atom_types"],
            atom_mask=batch["atom_mask"],
        )

    def encode_gt_slot_conditions(self, batch: dict) -> torch.Tensor | None:
        required = [
            "scaffold_atom_role",
            "scaffold_atom_parent_system_target",
            "scaffold_atom_canonical_site_index",
            "scaffold_atom_attachment_target_site",
            "scaffold_atom_sidechain_root_site",
            "scaffold_atom_hetero_target_class",
            "atom_mask",
        ]
        if not all(k in batch for k in required):
            return None
        return self.slot_semantic_condition_encoder(batch)

    def encode_gt_slot_edge_types(self, batch: dict) -> torch.Tensor | None:
        required = [
            "scaffold_local_edges",
            "scaffold_n_local_edges",
            "scaffold_sidechain_edges",
            "scaffold_n_sidechain_edges",
            "atom_mask",
        ]
        if not all(k in batch for k in required):
            return None
        return self.slot_connection_graph_encoder(batch)

    def compensate_atom_count(
        self,
        batch: dict,
        n_atoms: torch.Tensor,
        predicted_relations: dict | None = None,
        predicted_structure_counts: dict | None = None,
    ) -> torch.Tensor:
        """Scaffold-aware lower-bound compensation for atom count.

        The count head is frozen in bridge-only runs, so when scaffold-driven
        conditioning improves geometry but the count head lags behind, we allow
        scaffold evidence to raise obviously under-predicted counts.
        """
        if "scaffold_atom_is_scaffold" not in batch:
            return n_atoms

        scaffold_atoms = batch["scaffold_atom_is_scaffold"].sum(dim=1).float()
        source = self.config.get("v17_count_comp_source", "predicted")
        mode = self.config.get("v17_count_comp_mode", "lower_bound")
        blend_alpha = self.config.get("v17_count_comp_blend_alpha", 0.0)
        hybrid_alpha = self.config.get("v17_count_comp_hybrid_alpha", 0.5)
        side_ratio = self.config.get("v17_count_comp_sidechain_ratio", 0.95)
        pred_non_scaffold_ratio = self.config.get("v17_count_comp_pred_non_scaffold_ratio", 1.0)
        pred_edge_ratio = self.config.get("v17_count_comp_pred_edge_ratio", 1.0)
        relation_ratio = self.config.get("v17_count_comp_relation_ratio", 1.0)
        side_bias = self.config.get("v17_count_comp_bias", 0.0)
        min_extra = self.config.get("v17_count_comp_min_extra", 0)
        threshold = self.config.get("v17_count_comp_valid_threshold", 0.5)
        external_mode = self.config.get("v17_count_comp_external_mode", "max")
        use_relation_signal = self.config.get("v17_count_comp_use_relation_signal", True)

        predicted_structure_est = None
        if source in {"predicted_structure", "auto", "hybrid"} and predicted_structure_counts is not None:
            external_candidates = [
                pred_non_scaffold_ratio * predicted_structure_counts["non_scaffold_atom_count"].float(),
                pred_edge_ratio * predicted_structure_counts["sidechain_edge_count"].float(),
            ]
            if use_relation_signal and predicted_relations is not None:
                relation_probs = torch.sigmoid(predicted_relations["sidechain_valid_logits"].float())
                if threshold <= 0.0:
                    relation_est = relation_probs.sum(dim=-1)
                else:
                    relation_est = (relation_probs >= threshold).float().sum(dim=-1)
                external_candidates.append(relation_ratio * relation_est)

            external_stack = torch.stack(external_candidates, dim=0)
            if external_mode == "mean":
                external_est = external_stack.mean(dim=0)
            else:
                external_est = external_stack.max(dim=0).values

            predicted_structure_est = scaffold_atoms + external_est + side_bias
            if min_extra > 0:
                predicted_structure_est = torch.maximum(
                    predicted_structure_est,
                    scaffold_atoms + float(min_extra),
                )

        side_est = None
        if source in {"predicted", "auto"} and predicted_relations is not None:
            side_probs = torch.sigmoid(predicted_relations["sidechain_valid_logits"].float())
            if threshold <= 0.0:
                side_est = side_probs.sum(dim=-1)
            else:
                side_est = (side_probs >= threshold).float().sum(dim=-1)
        gt_edge_est = None
        if source in {"gt_edges", "auto", "hybrid"} and "scaffold_n_sidechain_edges" in batch:
            gt_edge_est = scaffold_atoms + side_ratio * batch["scaffold_n_sidechain_edges"].float() + side_bias
            if min_extra > 0:
                gt_edge_est = torch.maximum(gt_edge_est, scaffold_atoms + float(min_extra))

        est_count = None
        if source == "predicted_structure":
            est_count = predicted_structure_est
        elif source == "hybrid":
            if predicted_structure_est is not None and gt_edge_est is not None:
                est_count = hybrid_alpha * predicted_structure_est + (1.0 - hybrid_alpha) * gt_edge_est
            else:
                est_count = predicted_structure_est if predicted_structure_est is not None else gt_edge_est
        elif side_est is not None:
            est_count = scaffold_atoms + side_ratio * side_est + side_bias
            if min_extra > 0:
                est_count = torch.maximum(est_count, scaffold_atoms + float(min_extra))
        elif gt_edge_est is not None:
            est_count = gt_edge_est
        elif predicted_structure_est is not None:
            est_count = predicted_structure_est

        if est_count is None:
            return n_atoms

        if mode == "blend":
            compensated = ((1.0 - blend_alpha) * n_atoms.float() + blend_alpha * est_count).round()
        else:
            compensated = torch.maximum(n_atoms.float(), est_count.round())
        return compensated.long().clamp(1, MAX_ATOMS)

    def forward(self, batch: dict, enable_constraints: bool = False,
                epoch_ratio: float = 0.0) -> dict:
        """Training forward: encode AFM + compute all losses."""
        afm = batch["afm_stack"]       # (B, D, H, W)
        coords = batch["coords"]       # (B, N, 3)
        types = batch["atom_types"]    # (B, N)
        mask = batch["atom_mask"]      # (B, N)

        c_global, c_patches = self.encoder(afm)

        # V9: compute GT shape descriptors
        with torch.no_grad():
            gt_shape = compute_shape_descriptors(coords, mask)

        # V13: Build ring_info from batch for ring prediction loss
        ring_info = None
        if "ring_atom_indices" in batch and "ring_valid" in batch and "n_rings" in batch:
            ring_info = {
                "ring_atom_indices": batch["ring_atom_indices"],
                "ring_valid": batch["ring_valid"],
                "n_rings": batch["n_rings"],
            }

        scaffold_cond = None
        scaffold_relation_pred = None
        scaffold_count_pred = None
        slot_conditions = None
        slot_edge_types = None
        semantic_aux_weight = self.config.get("v18_semantic_aux_weight", 0.0)
        site_graph_aux_weight = self.config.get("v18_site_graph_aux_weight", 0.0)
        need_scaffold_rel_pred = (
            self.config.get("v17_scaffold_aux_weight", 0.0) > 0.0
            or (
                self.config.get("use_v17_bridge_gt_scaffold_tokens", False)
                and self.config.get("use_v17_bridge_predicted_relation_tokens", False)
            )
        )
        if need_scaffold_rel_pred:
            scaffold_relation_pred = self.scaffold_relation_head(c_patches)

        structure_count_aux_weight = self.config.get("v17_structure_count_aux_weight", 0.0)
        if structure_count_aux_weight > 0.0:
            scaffold_count_pred = self.scaffold_count_head(c_global)

        if self.config.get("use_v17_bridge_gt_scaffold_tokens", False):
            scaffold_cond, scaffold_relation_pred = self.encode_scaffold_tokens(
                batch,
                c_patches=c_patches,
                predicted_relations=scaffold_relation_pred,
            )
        if self.config.get("use_v18_gt_slot_conditions", False):
            slot_conditions = self.encode_gt_slot_conditions(batch)
        if self.config.get("use_v18_gt_slot_edge_types", False):
            slot_edge_types = self.encode_gt_slot_edge_types(batch)

        # Diffusion loss (V14: coord + type + shape + ring)
        losses = self.ddpm.compute_loss(coords, c_global, c_patches, types, mask,
                                        shape_desc=gt_shape, ring_info=ring_info,
                                        epoch_ratio=epoch_ratio,
                                        scaffold_tokens=None if scaffold_cond is None else scaffold_cond["tokens"],
                                        scaffold_token_mask=None if scaffold_cond is None else scaffold_cond["token_mask"],
                                        slot_conditions=slot_conditions,
                                        slot_edge_types=slot_edge_types)

        # V9: shape prediction loss (train shape_head to predict shape from AFM)
        pred_shape = self.shape_head(c_global.float())
        shape_pred_loss = F.mse_loss(pred_shape, gt_shape.float())
        losses["shape_pred_loss"] = shape_pred_loss

        # Atom count loss: track exact-count classification and MAE calibration separately
        count_losses = self.count_head.compute_loss(c_global, batch["n_atoms"])
        losses["count_ce_loss"] = count_losses["cls_loss"]
        losses["count_mae_loss"] = count_losses["reg_loss"]
        losses["count_loss"] = count_losses["count_loss"]

        # V16: Ring detection loss (uses c_global + c_patches)
        if "ring_centers" in batch and "ring_types" in batch:
            ring_det_losses = self.ring_head.compute_loss(
                c_global, c_patches,
                batch["ring_centers"], batch["ring_types"],
                batch["ring_valid"], batch["n_rings"],
            )
            losses["ring_det_loss"] = ring_det_losses["ring_total_loss"]
        else:
            losses["ring_det_loss"] = torch.tensor(0.0, device=c_global.device)

        scaffold_aux_weight = self.config.get("v17_scaffold_aux_weight", 0.0)
        if scaffold_aux_weight > 0.0 and scaffold_cond is not None:
            scaffold_targets = self.scaffold_token_encoder.build_aux_targets(
                batch, coords=batch["coords"], atom_types=batch["atom_types"], atom_mask=batch["atom_mask"]
            )
            if scaffold_relation_pred is None:
                scaffold_relation_pred = self.scaffold_relation_head(c_patches)
            scaffold_aux_losses = self.scaffold_relation_head.compute_loss_from_pred(
                scaffold_relation_pred, scaffold_targets
            )
            losses["scaffold_aux_loss"] = scaffold_aux_losses["scaffold_aux_loss"]
            losses["attachment_aux_loss"] = scaffold_aux_losses["attachment_aux_loss"]
            losses["sidechain_aux_loss"] = scaffold_aux_losses["sidechain_aux_loss"]
        else:
            losses["scaffold_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["attachment_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["sidechain_aux_loss"] = torch.tensor(0.0, device=c_global.device)

        if structure_count_aux_weight > 0.0 and "scaffold_total_scaffold_atoms" in batch:
            if scaffold_count_pred is None:
                scaffold_count_pred = self.scaffold_count_head(c_global)
            structure_count_targets = {
                "ring_system_count": batch["scaffold_n_ring_systems"].float(),
                "scaffold_atom_count": batch["scaffold_total_scaffold_atoms"].float(),
                "non_scaffold_atom_count": batch["scaffold_total_non_scaffold_atoms"].float(),
                "attachment_anchor_count": batch["scaffold_total_attachment_anchors"].float(),
                "sidechain_edge_count": batch["scaffold_total_sidechain_edges"].float(),
            }
            structure_count_losses = self.scaffold_count_head.compute_loss_from_pred(
                scaffold_count_pred, structure_count_targets
            )
            losses["ring_system_count_loss"] = structure_count_losses["ring_system_count_loss"]
            losses["scaffold_atom_count_loss"] = structure_count_losses["scaffold_atom_count_loss"]
            losses["non_scaffold_atom_count_loss"] = structure_count_losses["non_scaffold_atom_count_loss"]
            losses["attachment_anchor_count_loss"] = structure_count_losses["attachment_anchor_count_loss"]
            losses["sidechain_edge_count_loss"] = structure_count_losses["sidechain_edge_count_loss"]
            losses["structure_count_aux_loss"] = (
                self.config.get("v17_structure_ring_system_loss_weight", 1.0) * losses["ring_system_count_loss"]
                + self.config.get("v17_structure_scaffold_atom_loss_weight", 1.0) * losses["scaffold_atom_count_loss"]
                + self.config.get("v17_structure_non_scaffold_loss_weight", 2.0) * losses["non_scaffold_atom_count_loss"]
                + self.config.get("v17_structure_anchor_loss_weight", 1.0) * losses["attachment_anchor_count_loss"]
                + self.config.get("v17_structure_sidechain_edge_loss_weight", 2.0) * losses["sidechain_edge_count_loss"]
            )
        else:
            losses["structure_count_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["ring_system_count_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["scaffold_atom_count_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["non_scaffold_atom_count_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["attachment_anchor_count_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["sidechain_edge_count_loss"] = torch.tensor(0.0, device=c_global.device)

        if (
            semantic_aux_weight > 0.0
            and "x0_pred" in losses
            and "pred_type_logits" in losses
            and "scaffold_atom_role" in batch
        ):
            semantic_losses = self.atom_semantic_head.compute_loss(
                coords_pred=losses["x0_pred"],
                type_logits=losses["pred_type_logits"],
                c_global=c_global,
                gt_coords=batch["coords"],
                gt_types=batch["atom_types"],
                gt_mask=batch["atom_mask"],
                scaffold_labels=batch,
            )
            losses.update(semantic_losses)
        else:
            losses["semantic_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["semantic_valid_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["atom_role_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["attachment_site_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["hetero_site_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["semantic_parent_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["canonical_site_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["attachment_target_site_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["sidechain_root_site_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["hetero_target_class_aux_loss"] = torch.tensor(0.0, device=c_global.device)

        if site_graph_aux_weight > 0.0 and "scaffold_atom_canonical_site_index" in batch:
            site_graph_losses = self.site_graph_parser_head.compute_loss(
                c_patches=c_patches,
                gt_coords=batch["coords"],
                gt_mask=batch["atom_mask"],
                scaffold_labels=batch,
            )
            losses.update(site_graph_losses)
        else:
            losses["site_graph_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_object_valid_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_coord_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_parent_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_element_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_attachment_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_hetero_aux_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["site_edge_aux_loss"] = torch.tensor(0.0, device=c_global.device)

        # Retrieval stays as an optional analysis branch and is disabled on the V16b mainline
        losses["retrieval_loss"] = torch.tensor(0.0, device=c_global.device)

        # V16c: aligned with get_training_stage fractional boundaries
        stage1_end = 0.40
        stage2_end = 0.73
        if epoch_ratio <= stage1_end:
            count_ce_weight, count_mae_weight = 0.6, 0.2
        elif epoch_ratio <= stage2_end:
            count_ce_weight, count_mae_weight = 0.7, 0.3
        else:
            count_ce_weight, count_mae_weight = 0.8, 0.4

        # Physics constraints must act on predicted structures, not on GT inputs
        if enable_constraints and "x0_pred" in losses and "pred_type_logits" in losses:
            ring_atom_indices = batch.get("ring_atom_indices", None)
            ring_valid = batch.get("ring_valid", None)
            pred_coords = losses["x0_pred"]
            pred_types = losses["pred_type_logits"].argmax(dim=-1)
            constraint_losses = compute_all_constraints(
                pred_coords, pred_types, mask, ring_atom_indices, ring_valid
            )
            # V16c Phase 2B: store sub-items explicitly for logging
            losses["bond_length_loss"] = constraint_losses["bond_length_loss"]
            losses["bond_angle_loss"] = constraint_losses["bond_angle_loss"]
            losses["planarity_loss"] = constraint_losses.get("planarity_loss",
                torch.tensor(0.0, device=c_global.device))
            losses["constraint_loss"] = constraint_losses["total_constraint_loss"]
        else:
            losses["constraint_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["bond_length_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["bond_angle_loss"] = torch.tensor(0.0, device=c_global.device)
            losses["planarity_loss"] = torch.tensor(0.0, device=c_global.device)

        # V16d.1: stage-dependent constraint weighting and explicit
        # bond-length emphasis from config.
        #   Stage 1 (0-40%): no constraints yet (model learning basic coord/type structure)
        #   Stage 2 (40-73%): moderate constraint weight
        #   Stage 3 (73-100%): stronger refinement weight
        cw_s1 = self.config.get("constraint_weight_s1", 0.0)
        cw_s2 = self.config.get("constraint_weight_s2", 0.3)
        cw_s3 = self.config.get("constraint_weight_s3", 0.5)
        bond_lw = self.config.get("bond_length_weight", 0.0)

        if epoch_ratio <= 0.40:
            constraint_weight = cw_s1
        elif epoch_ratio <= 0.73:
            constraint_weight = cw_s2
        else:
            constraint_weight = cw_s3

        losses["loss"] = (
            losses["loss"]
            + count_ce_weight * losses["count_ce_loss"]
            + count_mae_weight * losses["count_mae_loss"]
            + 0.5 * losses["shape_pred_loss"]
            + constraint_weight * losses["constraint_loss"]
            + bond_lw * losses["bond_length_loss"]
            + 0.5 * losses["ring_det_loss"]
            + scaffold_aux_weight * losses["scaffold_aux_loss"]
            + structure_count_aux_weight * losses["structure_count_aux_loss"]
            + semantic_aux_weight * losses["semantic_aux_loss"]
            + site_graph_aux_weight * losses["site_graph_aux_loss"]
        )

        return losses

    @torch.no_grad()
    def generate(self, batch: dict, use_gt_count: bool = False,
                 use_ddim: bool = False, ddim_steps: int = 100,
                 use_gt_ring_info: bool = False,
                 use_gt_scaffold_tokens: bool = False,
                 use_predicted_relation_tokens: bool = False,
                 use_gt_scaffold_soft_constraint: bool = False,
                 scaffold_constraint_time_threshold: int = 200,
                 scaffold_constraint_scale: float = 0.12,
                 scaffold_plane_scale: float = 0.08,
                 scaffold_edge_scale: float = 0.0,
                 scaffold_sidechain_edge_scale: float = 0.0,
                 scaffold_post_guidance_scale: float = 0.0,
                 disable_guidance: bool = False,
                 disable_ring_snap: bool = False,
                 sampler: str = "ddim",
                 guidance_step_size: float = 0.002,
                 guidance_time_threshold: int = 500) -> dict:
        """V16c: Generate molecular structure from AFM stack.

        Args:
            use_gt_count: use GT atom count instead of predicted
            use_ddim: use DDIM sampling (deprecated, use sampler='ddim')
            ddim_steps: number of DDIM steps
            use_gt_ring_info: use GT ring info for ring constraints
            disable_guidance: disable all physics guidance
            disable_ring_snap: disable all ring snapping
            sampler: 'ddim' or 'ddpm'
        """
        afm = batch["afm_stack"]
        c_global, c_patches = self.encoder(afm)
        scaffold_relation_pred = None
        scaffold_count_pred = None

        if use_gt_count and "n_atoms" in batch:
            n_atoms = batch["n_atoms"]
        else:
            n_atoms = self.count_head.predict(c_global)

        # Ring info (GT or predicted)
        ring_info = None
        if not disable_ring_snap and use_gt_ring_info and "ring_atom_indices" in batch and "ring_templates" in batch and "ring_valid" in batch:
            ring_info = {
                "ring_atom_indices": batch["ring_atom_indices"],
                "ring_templates": batch["ring_templates"],
                "ring_valid": batch["ring_valid"],
            }

        # Predicted rings — DEAD CODE (V16c confirmed):
        # ring_head.predict() returns {n_rings, ring_centers, ring_types, ring_valid}.
        # _project_ring_constraints() needs {ring_atom_indices, ring_templates}.
        # predicted_rings lacks atom indices, so it CANNOT drive ring snapping.
        # Only GT ring_info (ring_atom_indices + ring_templates) works.
        # This variable is kept for future extension only.
        predicted_rings = None
        if not disable_ring_snap:
            predicted_rings = self.ring_head.predict(c_global, c_patches)

        target_shape = self.shape_head(c_global) if self.config.get("use_shape_guidance", False) else None

        use_physics = self.config.get("physics_guidance", True) and not disable_guidance

        # Resolve sampler type
        is_ddim = (sampler == "ddim") or use_ddim

        scaffold_cond = None
        slot_conditions = None
        slot_edge_types = None
        if use_gt_scaffold_tokens:
            if use_predicted_relation_tokens:
                scaffold_relation_pred = self.scaffold_relation_head(c_patches)
                scaffold_cond = self.scaffold_token_encoder.forward_from_predicted_relations(
                    batch, scaffold_relation_pred
                )
            else:
                scaffold_cond = self.encode_gt_scaffold_tokens(batch)
        if self.config.get("use_v18_gt_slot_conditions", False):
            slot_conditions = self.encode_gt_slot_conditions(batch)
        if self.config.get("use_v18_gt_slot_edge_types", False):
            slot_edge_types = self.encode_gt_slot_edge_types(batch)

        if self.config.get("v17_scaffold_count_compensation", False):
            source = self.config.get("v17_count_comp_source", "predicted")
            if self.config.get("v17_count_comp_use_relation_signal", True) and scaffold_relation_pred is None:
                scaffold_relation_pred = self.scaffold_relation_head(c_patches)
            if source in {"predicted_structure", "auto", "hybrid"}:
                scaffold_count_pred = self.scaffold_count_head(c_global)
            n_atoms = self.compensate_atom_count(
                batch,
                n_atoms,
                predicted_relations=scaffold_relation_pred,
                predicted_structure_counts=scaffold_count_pred,
            )

        scaffold_constraint = None
        if use_gt_scaffold_soft_constraint and "scaffold_atom_is_scaffold" in batch:
            scaffold_constraint = {
                "target_coords": batch["coords"],
                "scaffold_mask": batch["scaffold_atom_is_scaffold"],
                "system_objectness": batch.get("scaffold_system_objectness"),
                "system_centers": batch.get("scaffold_system_center"),
                "system_normals": batch.get("scaffold_system_normal"),
                "system_atom_indices": batch.get("scaffold_system_atom_indices"),
                "local_edges": batch.get("scaffold_local_edges"),
                "local_edge_lengths": batch.get("scaffold_local_edge_lengths"),
                "n_local_edges": batch.get("scaffold_n_local_edges"),
                "sidechain_edges": batch.get("scaffold_sidechain_edges"),
                "sidechain_edge_lengths": batch.get("scaffold_sidechain_edge_lengths"),
                "n_sidechain_edges": batch.get("scaffold_n_sidechain_edges"),
            }

        coords, type_logits = self.ddpm.sample(
            c_global, c_patches, n_atoms, max_atoms=MAX_ATOMS,
            ring_info=ring_info,
            predicted_rings=predicted_rings,
            use_ddim=is_ddim, ddim_steps=ddim_steps,
            use_physics_guidance=use_physics,
            target_shape=target_shape,
            disable_guidance=disable_guidance,
            disable_ring_snap=disable_ring_snap,
            scaffold_tokens=None if scaffold_cond is None else scaffold_cond["tokens"],
            scaffold_token_mask=None if scaffold_cond is None else scaffold_cond["token_mask"],
            slot_conditions=slot_conditions,
            slot_edge_types=slot_edge_types,
            scaffold_constraint=scaffold_constraint,
            scaffold_constraint_time_threshold=scaffold_constraint_time_threshold,
            scaffold_constraint_scale=scaffold_constraint_scale,
            scaffold_plane_scale=scaffold_plane_scale,
            scaffold_edge_scale=scaffold_edge_scale,
            scaffold_sidechain_edge_scale=scaffold_sidechain_edge_scale,
            scaffold_post_guidance_scale=scaffold_post_guidance_scale,
            guidance_step_size=guidance_step_size,
            guidance_time_threshold=guidance_time_threshold,
        )

        result = {
            "coords": coords,
            "type_logits": type_logits,
            "n_atoms_pred": n_atoms,
        }
        if scaffold_count_pred is not None:
            result["structure_count_pred"] = {
                k: v.detach() for k, v in scaffold_count_pred.items()
            }

        if self.mol_embeddings is not None:
            scores, indices = self.retrieval_head.retrieve(
                c_global, self.mol_embeddings, top_k=5,
            )
            result["retrieval_scores"] = scores
            result["retrieval_indices"] = indices

        return result


# ============================================================
# Training functions
# ============================================================

def _batch_to_device(batch, device):
    """Move batch tensors to device, skipping non-tensor values."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def get_training_stage(epoch: int, total_epochs: int = 30) -> int:
    """V16c: Determine training stage from epoch as fraction of total.

    Stage 1 (0-40%): base training
    Stage 2 (40-73%): constraint training (bond length/angle/planarity)
    Stage 3 (73-100%): fine-tuning with full loss

    Uses fractional boundaries so debug runs with fewer epochs still
    enter all stages.
    """
    ratio = epoch / max(total_epochs, 1)
    if ratio <= 0.40:
        return 1
    elif ratio <= 0.73:
        return 2
    else:
        return 3


def train_epoch(model, loader, optimizer, device, model_type="diffusion", epoch=None, total_epochs=None, scaler=None):
    model.train()
    totals = {"loss": 0.0, "coord_loss": 0.0, "type_loss": 0.0,
              "count_loss": 0.0, "count_ce_loss": 0.0, "count_mae_loss": 0.0,
              "constraint_loss": 0.0, "bond_length_loss": 0.0,
              "bond_angle_loss": 0.0, "planarity_loss": 0.0,
              "shape_loss": 0.0, "ring_det_loss": 0.0,
              "scaffold_aux_loss": 0.0, "attachment_aux_loss": 0.0,
              "sidechain_aux_loss": 0.0,
              "structure_count_aux_loss": 0.0,
              "ring_system_count_loss": 0.0,
              "scaffold_atom_count_loss": 0.0,
              "non_scaffold_atom_count_loss": 0.0,
              "attachment_anchor_count_loss": 0.0,
              "sidechain_edge_count_loss": 0.0,
              "semantic_aux_loss": 0.0,
              "semantic_valid_loss": 0.0,
              "atom_role_aux_loss": 0.0,
              "attachment_site_aux_loss": 0.0,
              "hetero_site_aux_loss": 0.0,
              "semantic_parent_aux_loss": 0.0,
              "canonical_site_aux_loss": 0.0,
              "attachment_target_site_aux_loss": 0.0,
              "sidechain_root_site_aux_loss": 0.0,
              "hetero_target_class_aux_loss": 0.0,
              "site_graph_aux_loss": 0.0,
              "site_object_valid_loss": 0.0,
              "site_coord_aux_loss": 0.0,
              "site_parent_aux_loss": 0.0,
              "site_element_aux_loss": 0.0,
              "site_attachment_aux_loss": 0.0,
              "site_hetero_aux_loss": 0.0,
              "site_edge_aux_loss": 0.0,
              "geom_aux_loss": 0.0, "bottom_z_loss": 0.0}
    n_batches = 0

    # Determine training stage features
    stage = get_training_stage(epoch, total_epochs) if (epoch and total_epochs) else 1
    enable_constraints = (stage >= 2)  # Stage 2+: physics constraints

    use_amp = scaler is not None

    desc = "Train"
    if epoch is not None and total_epochs is not None:
        desc = f"Train [{epoch}/{total_epochs}] S{stage}"
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad()

        if model_type == "diffusion":
            epoch_ratio = (epoch / total_epochs) if (epoch and total_epochs) else 0.0
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    losses = model(batch,
                                  enable_constraints=enable_constraints,
                                  epoch_ratio=epoch_ratio)
            else:
                losses = model(batch,
                              enable_constraints=enable_constraints,
                              epoch_ratio=epoch_ratio)
        else:
            losses = model.compute_loss(
                batch["afm_stack"],
                batch["coords"],
                batch["atom_types"],
                batch["atom_mask"],
            )

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
        n_batches += 1

        pbar.set_postfix(loss=f"{totals['loss'] / n_batches:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, device, model_type="diffusion",
             epoch=None, total_epochs=None):
    """V16c: validate with correct stage/epoch_ratio."""
    model.eval()
    totals = {"loss": 0.0, "coord_loss": 0.0, "type_loss": 0.0, "count_loss": 0.0,
              "count_ce_loss": 0.0, "count_mae_loss": 0.0, "constraint_loss": 0.0,
              "bond_length_loss": 0.0, "bond_angle_loss": 0.0, "planarity_loss": 0.0,
              "ring_det_loss": 0.0, "scaffold_aux_loss": 0.0,
              "attachment_aux_loss": 0.0, "sidechain_aux_loss": 0.0,
              "structure_count_aux_loss": 0.0,
              "ring_system_count_loss": 0.0,
              "scaffold_atom_count_loss": 0.0,
              "non_scaffold_atom_count_loss": 0.0,
              "attachment_anchor_count_loss": 0.0,
              "sidechain_edge_count_loss": 0.0,
              "semantic_aux_loss": 0.0,
              "semantic_valid_loss": 0.0,
              "atom_role_aux_loss": 0.0,
              "attachment_site_aux_loss": 0.0,
              "hetero_site_aux_loss": 0.0,
              "semantic_parent_aux_loss": 0.0,
              "canonical_site_aux_loss": 0.0,
              "attachment_target_site_aux_loss": 0.0,
              "sidechain_root_site_aux_loss": 0.0,
              "hetero_target_class_aux_loss": 0.0,
              "site_graph_aux_loss": 0.0,
              "site_object_valid_loss": 0.0,
              "site_coord_aux_loss": 0.0,
              "site_parent_aux_loss": 0.0,
              "site_element_aux_loss": 0.0,
              "site_attachment_aux_loss": 0.0,
              "site_hetero_aux_loss": 0.0,
              "site_edge_aux_loss": 0.0,
              "geom_aux_loss": 0.0, "bottom_z_loss": 0.0}
    n_batches = 0

    # V16c: pass correct epoch_ratio and enable_constraints to match training
    epoch_ratio = (epoch / total_epochs) if (epoch and total_epochs) else 0.0
    stage = get_training_stage(epoch, total_epochs) if (epoch and total_epochs) else 1
    enable_constraints = (stage >= 2)

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        batch = _batch_to_device(batch, device)

        if model_type == "diffusion":
            losses = model(batch, enable_constraints=enable_constraints,
                          epoch_ratio=epoch_ratio)
        else:
            losses = model.compute_loss(
                batch["afm_stack"],
                batch["coords"],
                batch["atom_types"],
                batch["atom_mask"],
            )

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item() if isinstance(losses[k], torch.Tensor) else losses[k]
        n_batches += 1

        pbar.set_postfix(loss=f"{totals['loss'] / n_batches:.4f}")

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_generation(model, loader, device, num_samples: int = 50,
                        use_ddim: bool = True, ddim_steps: int = 100,
                        use_gt_count: bool = False,
                        use_gt_scaffold_tokens: bool = False,
                        use_predicted_relation_tokens: bool = False,
                        use_gt_scaffold_soft_constraint: bool = False,
                        scaffold_constraint_time_threshold: int = 200,
                        scaffold_constraint_scale: float = 0.12,
                        scaffold_plane_scale: float = 0.08,
                        scaffold_edge_scale: float = 0.0,
                        scaffold_sidechain_edge_scale: float = 0.0,
                        scaffold_post_guidance_scale: float = 0.0,
                        disable_guidance: bool = False,
                        disable_ring_snap: bool = False,
                        sampler: str = "ddim",
                        guidance_step_size: float = 0.002,
                        guidance_time_threshold: int = 500):
    """V16c: Evaluate generation quality with configurable settings.

    V16c change: bond validity is computed with TWO masks:
    - GT mask: uses batch["atom_mask"] (ground truth atom count)
    - Pred mask: uses n_atoms_pred (predicted atom count)
    This allows diagnosing bond validity separately for each configuration.
    """
    model.eval()
    all_rmsd = []
    all_recall = []
    all_bottom_rmsd = []
    all_bond_valid_gt = []
    all_bond_valid_pred = []
    all_count_exact = []
    all_count_mae = []
    all_struct_sim = []
    all_formula_sim = []
    all_ring_pres = []
    all_fidelity = []
    count = 0

    sampler_desc = f"Eval(DDIM-{ddim_steps})" if use_ddim else "Eval(DDPM)"
    pbar = tqdm(loader, desc=sampler_desc, leave=False)
    for batch in pbar:
        if count >= num_samples:
            break
        batch = _batch_to_device(batch, device)

        gen_result = model.generate(batch, use_gt_count=use_gt_count,
                                    use_ddim=use_ddim, ddim_steps=ddim_steps,
                                    use_gt_scaffold_tokens=use_gt_scaffold_tokens,
                                    use_predicted_relation_tokens=use_predicted_relation_tokens,
                                    use_gt_scaffold_soft_constraint=use_gt_scaffold_soft_constraint,
                                    scaffold_constraint_time_threshold=scaffold_constraint_time_threshold,
                                    scaffold_constraint_scale=scaffold_constraint_scale,
                                    scaffold_plane_scale=scaffold_plane_scale,
                                    scaffold_edge_scale=scaffold_edge_scale,
                                    scaffold_sidechain_edge_scale=scaffold_sidechain_edge_scale,
                                    scaffold_post_guidance_scale=scaffold_post_guidance_scale,
                                    disable_guidance=disable_guidance,
                                    disable_ring_snap=disable_ring_snap,
                                    sampler=sampler,
                                    guidance_step_size=guidance_step_size,
                                    guidance_time_threshold=guidance_time_threshold)
        coords_pred = gen_result["coords"]
        type_logits = gen_result["type_logits"]
        pred_types = type_logits.argmax(dim=-1)
        n_pred = gen_result["n_atoms_pred"]
        B_batch = coords_pred.shape[0]
        N = coords_pred.shape[1]

        rmsd = compute_rmsd(coords_pred, batch["coords"], batch["atom_mask"],
                            n_atoms_pred=n_pred)
        recall = compute_bottom_atom_recall(
            coords_pred, batch["coords"],
            pred_types, batch["atom_types"],
            batch["atom_mask"],
        )
        bottom_rmsd = compute_bottom_atom_rmsd(
            coords_pred, batch["coords"], batch["atom_mask"],
        )

        # Bond validity with GT mask (baseline)
        bond_valid_gt = compute_bond_validity(
            coords_pred, pred_types, batch["atom_mask"],
        )

        # Bond validity with PRED mask (uses predicted atom count)
        # pred_mask[b, i] = 1 iff i < n_pred[b]
        idx = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
        pred_mask_float = (idx < n_pred.float().unsqueeze(1)).float()
        bond_valid_pred = compute_bond_validity(
            coords_pred, pred_types, pred_mask_float,
        )

        # Atom count accuracy
        count_acc = compute_atom_count_accuracy(n_pred, batch["n_atoms"])

        # 3D structure similarity (replaces CID retrieval)
        struct_sim = compute_structure_similarity(
            coords_pred, batch["coords"],
            pred_types, batch["atom_types"],
            batch["atom_mask"],
            n_atoms_pred=n_pred,
        )

        # V7: Formula similarity (molecule-level)
        formula_sim = compute_formula_similarity(
            pred_types, batch["atom_types"], batch["atom_mask"],
            n_atoms_pred=n_pred,
        )

        # V14: Ring preservation for accurate Composite
        ring_pres = compute_ring_preservation(
            coords_pred, batch["coords"], pred_types, batch["atom_types"],
            batch["atom_mask"], n_atoms_pred=n_pred,
        )
        fidelity = compute_structure_fidelity(
            coords_pred, batch["coords"],
            pred_types, batch["atom_types"],
            batch["atom_mask"],
            n_atoms_pred=n_pred,
            pred_mask=pred_mask_float,
            bond_validity_pred=bond_valid_pred,
            scaffold_labels=batch if "scaffold_n_ring_systems" in batch else None,
        )

        all_rmsd.append(rmsd)
        all_recall.append(recall)
        all_bottom_rmsd.append(bottom_rmsd)
        all_bond_valid_gt.append(bond_valid_gt)
        all_bond_valid_pred.append(bond_valid_pred)
        all_count_exact.append(count_acc["exact_match"])
        all_count_mae.append(count_acc["mae"])
        all_struct_sim.append(struct_sim)
        all_formula_sim.append(formula_sim)
        all_ring_pres.append(ring_pres["ring_preservation_mean"])
        all_fidelity.append(fidelity)
        count += batch["afm_stack"].shape[0]

    all_rmsd = torch.cat(all_rmsd)
    all_recall = torch.cat(all_recall)
    all_bottom_rmsd = torch.cat(all_bottom_rmsd)
    all_bond_valid_gt = torch.cat(all_bond_valid_gt)
    all_bond_valid_pred = torch.cat(all_bond_valid_pred)

    rmsd_mean = all_rmsd.mean().item()
    bottom_recall_mean = all_recall.mean().item()
    bond_valid_gt_mean = all_bond_valid_gt.mean().item()
    bond_valid_pred_mean = all_bond_valid_pred.mean().item()
    count_exact_mean = np.mean(all_count_exact) if all_count_exact else 0.0

    # V7: Formula similarity aggregation
    formula_sim_mean = float(np.mean([s["formula_similarity_mean"] for s in all_formula_sim])) if all_formula_sim else 0.0
    type_dist_mean = float(np.mean([s["type_distribution_match_mean"] for s in all_formula_sim])) if all_formula_sim else 0.0

    # Aggregate structure similarity scores
    struct_sim_mean = float(np.mean([s["overall_similarity_mean"] for s in all_struct_sim])) if all_struct_sim else 0.0
    type_match_mean = float(np.mean([s["type_match_rate_mean"] for s in all_struct_sim])) if all_struct_sim else 0.0
    coulomb_mean = float(np.mean([s["coulomb_similarity_mean"] for s in all_struct_sim])) if all_struct_sim else 0.0
    valence_mean = float(np.mean([s["valence_validity_mean"] for s in all_struct_sim])) if all_struct_sim else 0.0
    formula_sim_mean = float(np.mean([s["formula_similarity_mean"] for s in all_struct_sim])) if all_struct_sim else 0.0

    ring_pres_mean = float(np.mean(all_ring_pres)) if all_ring_pres else 0.0
    struct_fidelity_pass_rate = float(np.mean([s["struct_fidelity_pass_rate"] for s in all_fidelity])) if all_fidelity else 0.0
    soft_recon_score = float(np.mean([s["soft_recon_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    struct_fidelity_score = float(np.mean([s["struct_fidelity_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_count_exact_v18 = float(np.mean([s["atom_count_exact_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_count_abs_error = float(np.mean([s["atom_count_abs_error_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    matched_atom_rmsd = float(np.mean([s["matched_atom_rmsd_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    matched_heavy_atom_rmsd = float(np.mean([s["matched_heavy_atom_rmsd_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    matched_atom_mae = float(np.mean([s["matched_atom_mae_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_position_score = float(np.mean([s["atom_position_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heavy_atom_hit_rate_tight = float(np.mean([s["heavy_atom_hit_rate_tight_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heavy_atom_hit_rate_medium = float(np.mean([s["heavy_atom_hit_rate_medium_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heavy_atom_hit_rate_loose = float(np.mean([s["heavy_atom_hit_rate_loose_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    typed_heavy_atom_hit_rate_tight = float(np.mean([s["typed_heavy_atom_hit_rate_tight_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    typed_heavy_atom_hit_rate_medium = float(np.mean([s["typed_heavy_atom_hit_rate_medium_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    typed_heavy_atom_hit_rate_loose = float(np.mean([s["typed_heavy_atom_hit_rate_loose_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heteroatom_hit_rate_medium = float(np.mean([s["heteroatom_hit_rate_medium_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    macro_type_f1 = float(np.mean([s["macro_type_f1_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    ch_collapse_rate = float(np.mean([s["ch_collapse_rate_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_type_acc = float(np.mean([s["atom_type_acc_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heteroatom_precision = float(np.mean([s["heteroatom_precision_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heteroatom_recall = float(np.mean([s["heteroatom_recall_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    heteroatom_f1 = float(np.mean([s["heteroatom_f1_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_semantic_score = float(np.mean([s["atom_semantic_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    ring_count_exact = float(np.mean([s["ring_count_exact_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    ring_complete_rate = float(np.mean([s["ring_complete_rate_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    ring_integrity_score = float(np.mean([s["ring_integrity_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    ring_site_coverage_medium = float(np.mean([s["ring_site_coverage_medium_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    attachment_edge_f1 = float(np.mean([s["attachment_edge_f1_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    soft_attachment_site_acc_medium = float(np.mean([s["soft_attachment_site_acc_medium_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    scaffold_relation_f1 = float(np.mean([s["scaffold_relation_f1_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    connectivity_score = float(np.mean([s["connectivity_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    local_chem_score = float(np.mean([s["local_chem_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    global_shape_aux_score = float(np.mean([s["global_shape_aux_score_mean"] for s in all_fidelity])) if all_fidelity else 0.0
    atom_match_coverage = float(np.mean([s["atom_match_coverage_mean"] for s in all_fidelity])) if all_fidelity else 0.0

    composite = compute_composite_score(
        rmsd=rmsd_mean,
        bottom_atom_score=bottom_recall_mean,
        bond_validity=bond_valid_gt_mean,
        ring_preservation=ring_pres_mean,
        atom_count_accuracy=count_exact_mean,
        structure_similarity=struct_sim_mean,
    )

    return {
        "rmsd_mean": rmsd_mean,
        "rmsd_std": all_rmsd.std().item(),
        "bottom_recall_mean": bottom_recall_mean,
        "bottom_recall_std": all_recall.std().item(),
        "bottom_rmsd_mean": all_bottom_rmsd.mean().item(),
        "bond_validity_gt_masked": bond_valid_gt_mean,
        "bond_validity_pred_masked": bond_valid_pred_mean,
        # For backward compat, keep old name pointing to GT-masked
        "bond_validity_mean": bond_valid_gt_mean,
        "count_exact_match": count_exact_mean,
        "count_mae": np.mean(all_count_mae) if all_count_mae else 0.0,
        "structure_similarity": struct_sim_mean,
        "type_match_rate": type_match_mean,
        "coulomb_similarity": coulomb_mean,
        "valence_validity": valence_mean,
        "formula_similarity": formula_sim_mean,
        "type_distribution_match": type_dist_mean,
        "ring_preservation": ring_pres_mean,
        "composite_score": composite,
        # V18 early-direction soft metrics
        "soft_recon_score": soft_recon_score,
        "heavy_atom_hit_rate_tight": heavy_atom_hit_rate_tight,
        "heavy_atom_hit_rate_medium": heavy_atom_hit_rate_medium,
        "heavy_atom_hit_rate_loose": heavy_atom_hit_rate_loose,
        "typed_heavy_atom_hit_rate_tight": typed_heavy_atom_hit_rate_tight,
        "typed_heavy_atom_hit_rate_medium": typed_heavy_atom_hit_rate_medium,
        "typed_heavy_atom_hit_rate_loose": typed_heavy_atom_hit_rate_loose,
        "heteroatom_hit_rate_medium": heteroatom_hit_rate_medium,
        "macro_type_f1": macro_type_f1,
        "ch_collapse_rate": ch_collapse_rate,
        "ring_site_coverage_medium": ring_site_coverage_medium,
        "soft_attachment_site_acc_medium": soft_attachment_site_acc_medium,
        # V18 Phase-1 atom-level structure fidelity
        "struct_fidelity_pass_rate": struct_fidelity_pass_rate,
        "struct_fidelity_score": struct_fidelity_score,
        "atom_count_exact": atom_count_exact_v18,
        "atom_count_abs_error": atom_count_abs_error,
        "matched_atom_rmsd": matched_atom_rmsd,
        "matched_heavy_atom_rmsd": matched_heavy_atom_rmsd,
        "matched_atom_mae": matched_atom_mae,
        "atom_position_score": atom_position_score,
        "atom_type_acc": atom_type_acc,
        "heteroatom_precision": heteroatom_precision,
        "heteroatom_recall": heteroatom_recall,
        "heteroatom_f1": heteroatom_f1,
        "atom_semantic_score": atom_semantic_score,
        "ring_count_exact": ring_count_exact,
        "ring_complete_rate": ring_complete_rate,
        "ring_integrity_score": ring_integrity_score,
        "attachment_edge_f1": attachment_edge_f1,
        "scaffold_relation_f1": scaffold_relation_f1,
        "connectivity_score": connectivity_score,
        "local_chem_score": local_chem_score,
        "global_shape_aux_score": global_shape_aux_score,
        "atom_match_coverage": atom_match_coverage,
    }


@torch.no_grad()
def run_sanity_evals(model, loader, device, num_samples: int = 200,
                     ddim_steps: int = 50, epoch: int = 0,
                     use_gt_scaffold_tokens: bool = False,
                     use_predicted_relation_tokens: bool = False,
                     use_gt_scaffold_soft_constraint: bool = False,
                     scaffold_constraint_time_threshold: int = 200,
                     scaffold_constraint_scale: float = 0.12,
                     scaffold_plane_scale: float = 0.08,
                     scaffold_edge_scale: float = 0.0,
                     scaffold_sidechain_edge_scale: float = 0.0,
                     scaffold_post_guidance_scale: float = 0.0,
                     guidance_step_size: float = 0.002,
                     guidance_time_threshold: int = 500):
    """V16c Phase 2B: Run three sanity eval configurations and return results dict.

    Config A: GT count + no guidance + no ring snap (purest test of sampler)
    Config B: pred count + no guidance + no ring snap (test count head)
    Config C: pred count + physics-guided (NOT full mainline — ring snap is dead code)

    V16c: Bond validity is reported with TWO masks:
    - gt_masked: uses ground truth atom count (valid for all configs)
    - pred_masked: uses predicted atom count (clean for configs B/C)
    """
    configs = {
        "A_gt_noguidance": dict(use_gt_count=True, disable_guidance=True,
                                 disable_ring_snap=True, sampler="ddim"),
        "B_pred_noguidance": dict(use_gt_count=False, disable_guidance=True,
                                   disable_ring_snap=True, sampler="ddim"),
        # V16c Phase 2B: rename C to be accurate — this is NOT full mainline
        # (predicted_rings is dead code, ring snap is not yet functional).
        # C = physics-guided prediction only (no GT ring info).
        "C_pred_guided": dict(use_gt_count=False, disable_guidance=False,
                                disable_ring_snap=False, sampler="ddim"),
    }
    results = {}
    for name, kwargs in configs.items():
        r = evaluate_generation(model, loader, device, num_samples=num_samples,
                                use_ddim=True, ddim_steps=ddim_steps,
                                use_gt_scaffold_tokens=use_gt_scaffold_tokens,
                                use_predicted_relation_tokens=use_predicted_relation_tokens,
                                use_gt_scaffold_soft_constraint=use_gt_scaffold_soft_constraint,
                                scaffold_constraint_time_threshold=scaffold_constraint_time_threshold,
                                scaffold_constraint_scale=scaffold_constraint_scale,
                                scaffold_plane_scale=scaffold_plane_scale,
                                scaffold_edge_scale=scaffold_edge_scale,
                                scaffold_sidechain_edge_scale=scaffold_sidechain_edge_scale,
                                scaffold_post_guidance_scale=scaffold_post_guidance_scale,
                                guidance_step_size=guidance_step_size,
                                guidance_time_threshold=guidance_time_threshold,
                                **kwargs)
        results[name] = r
        print(f"  [{name}] Soft={r['soft_recon_score']:.4f} "
              f"Hit@M={r['heavy_atom_hit_rate_medium']:.4f} "
              f"TypedHit@M={r['typed_heavy_atom_hit_rate_medium']:.4f} "
              f"HeteroHit@M={r['heteroatom_hit_rate_medium']:.4f} "
              f"SoftAttach={r['soft_attachment_site_acc_medium']:.4f} "
              f"Pass={r['struct_fidelity_pass_rate']:.4f} "
              f"Fidelity={r['struct_fidelity_score']:.4f}")
    return results


@torch.no_grad()
def save_predictions(model, loader, device, save_path, num_samples: int = 100):
    """Save model predictions with structure similarity scores to JSON file.

    Required fields:
    1. coords: 3D atomic coordinates
    2. atom_types: predicted atom types
    3. n_atoms_pred: predicted number of atoms
    4. structure_similarity: 3D structure similarity score vs ground truth
    """
    model.eval()
    predictions = []
    count = 0

    print(f"Saving predictions to: {save_path}")

    pbar = tqdm(loader, desc="Generating predictions", leave=False)
    for batch in pbar:
        if count >= num_samples:
            break
        batch = _batch_to_device(batch, device)

        gen_result = model.generate(batch, use_gt_count=False)

        # Compute structure similarity for this batch
        pred_types_batch = gen_result["type_logits"].argmax(dim=-1)
        struct_sim = compute_structure_similarity(
            gen_result["coords"], batch["coords"],
            pred_types_batch, batch["atom_types"],
            batch["atom_mask"],
            n_atoms_pred=gen_result["n_atoms_pred"],
        )

        # Extract predictions for each sample in batch
        batch_size = gen_result["coords"].shape[0]
        for i in range(batch_size):
            if count >= num_samples:
                break

            pred_coords = gen_result["coords"][i].cpu().numpy()  # (max_atoms, 3)
            pred_type_logits = gen_result["type_logits"][i].cpu().numpy()  # (max_atoms, num_types)
            pred_types = pred_type_logits.argmax(axis=-1)  # (max_atoms,)
            n_atoms = gen_result["n_atoms_pred"][i].item()

            # Build prediction record with structure similarity
            record = {
                "sample_id": count,
                "coords": pred_coords[:n_atoms].tolist(),
                "atom_types": pred_types[:n_atoms].tolist(),
                "n_atoms_pred": int(n_atoms),
                "structure_similarity": {
                    "overall": float(struct_sim["overall_similarity"][i]),
                    "type_match_rate": float(struct_sim["type_match_rate"][i]),
                    "coulomb_similarity": float(struct_sim["coulomb_similarity"][i]),
                    "valence_validity": float(struct_sim["valence_validity"][i]),
                    "count_similarity": float(struct_sim["count_similarity"][i]),
                },
            }

            # Ground truth info for reference
            gt_n = int(batch["atom_mask"][i].sum().item())
            record["n_atoms_gt"] = gt_n

            predictions.append(record)
            count += 1

    # Save to JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            "num_samples": len(predictions),
            "fields": [
                "coords (3D atomic coordinates)",
                "atom_types (predicted atom types)",
                "n_atoms_pred (predicted number of atoms)",
                "structure_similarity (3D structure similarity vs ground truth)"
            ],
            "predictions": predictions
        }, f, indent=2)

    print(f"Saved {len(predictions)} predictions with structure similarity scores")
    return save_path


def get_default_config():
    return {
        # Data
        "data_root": os.path.join(os.path.dirname(__file__), "..", "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM"),
        "param_key": "K-1",
        "img_size": 128,
        "num_frames": 10,
        "min_corrugation": 0.0,
        "augment_rotation": True,
        "batch_size": 32,
        "num_workers": 0,
        # Video ViT
        "patch_size": 16,
        "temporal_patch_size": 2,
        "embed_dim": 512,
        "encoder_depth": 8,
        "num_heads": 8,
        "drop_rate": 0.1,
        # Diffusion
        "denoiser_hidden_dim": 256,
        "denoiser_depth": 6,
        "diffusion_steps": 1000,
        # Training
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "save_dir": os.path.join(os.path.dirname(__file__), "..", "checkpoints"),
        "log_interval": 10,
        # Model type: "diffusion" or "resnet3d"
        "model_type": "diffusion",
        "max_samples": 0,
        "val_size": 0,
        # Retrieval
        "num_cids": 0,  # set dynamically from dataset
        # V17-Bridge
        "use_v17_bridge_gt_scaffold_tokens": False,
        "use_v18_gt_slot_conditions": False,
        "use_v18_gt_slot_edge_types": False,
        "v17_return_bridge_labels": False,
        "v17_scaffold_cross_attn_every": 2,
        "v17_scaffold_cross_attn_scale": 0.1,
        "v17_max_attachment_tokens": 24,
        "v17_max_sidechain_tokens": 48,
        "v17_include_sidechain_tokens": True,
        "v18_include_gt_site_tokens": False,
        "v18_max_site_tokens": 48,
        "v17_max_graph_depth": 8,
        "v17_scaffold_aux_weight": 0.0,
        "v17_structure_count_aux_weight": 0.0,
        "use_v17_bridge_predicted_relation_tokens": False,
        "v17_relation_token_teacher_force_prob": 0.0,
        "v17_scaffold_count_compensation": False,
        "v17_count_comp_source": "predicted",
        "v17_count_comp_mode": "lower_bound",
        "v17_count_comp_blend_alpha": 0.0,
        "v17_count_comp_hybrid_alpha": 0.5,
        "v17_count_comp_sidechain_ratio": 0.95,
        "v17_count_comp_pred_non_scaffold_ratio": 1.0,
        "v17_count_comp_pred_edge_ratio": 1.0,
        "v17_count_comp_relation_ratio": 1.0,
        "v17_count_comp_bias": 0.0,
        "v17_count_comp_min_extra": 0,
        "v17_count_comp_valid_threshold": 0.5,
        "v17_count_comp_external_mode": "max",
        "v17_count_comp_use_relation_signal": True,
        "v17_structure_ring_system_loss_weight": 1.0,
        "v17_structure_scaffold_atom_loss_weight": 1.0,
        "v17_structure_non_scaffold_loss_weight": 2.0,
        "v17_structure_anchor_loss_weight": 1.0,
        "v17_structure_sidechain_edge_loss_weight": 2.0,
        "v18_semantic_aux_weight": 0.0,
        "v18_slot_graph_scale": 0.15,
        "use_type_class_weights": False,
        "constraint_weight_s1": 0.0,
        "constraint_weight_s2": 0.3,
        "constraint_weight_s3": 0.5,
        "bond_length_weight": 0.0,
        "guidance_step_size": 0.002,
        "guidance_time_threshold": 500,
        "bridge_train_mode": "full",
        "bridge_lr": 3e-4,
        "bridge_eval_use_gt_scaffold_tokens": False,
        "bridge_eval_use_predicted_relation_tokens": False,
        "bridge_eval_use_gt_scaffold_soft_constraint": False,
        "bridge_eval_scaffold_constraint_time_threshold": 200,
        "bridge_eval_scaffold_constraint_scale": 0.12,
        "bridge_eval_scaffold_plane_scale": 0.08,
        "bridge_eval_scaffold_edge_scale": 0.0,
        "bridge_eval_scaffold_sidechain_edge_scale": 0.0,
        "bridge_eval_scaffold_post_guidance_scale": 0.0,
        "v18_eval_use_structure_labels": True,
    }


def load_config(config_path: str) -> dict:
    """Load config from JSON file, with 'auto' path resolution."""
    with open(config_path, "r") as f:
        cfg = json.load(f)

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    if cfg.get("data_root") == "auto":
        cfg["data_root"] = os.path.join(base_dir, "dataverse_files", "SUBMIT_QUAM-AFM", "QUAM")
    if cfg.get("save_dir") == "auto":
        cfg["save_dir"] = os.path.join(base_dir, "checkpoints")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="AFM 3D Molecular Reconstruction")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "config.json"),
                        help="Path to config JSON file")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = get_default_config()
    config.update(load_config(args.config))
    config["eval_only"] = args.eval_only
    config["checkpoint"] = args.checkpoint

    device = get_device()

    # Setup logging to both console and file
    log_dir = config.get("save_dir", "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    log_mode = 'a' if (args.eval_only or config["checkpoint"]) else 'w'
    sys.stdout = Logger(log_file, mode=log_mode)

    print(f"Device: {device}")
    print(f"Model: {config['model_type']}")

    # Data
    need_bridge_labels = (
        config.get("v17_return_bridge_labels", False)
        or config.get("v17_structure_count_aux_weight", 0.0) > 0.0
        or config.get("v17_scaffold_count_compensation", False)
        or config.get("use_v17_bridge_gt_scaffold_tokens", False)
        or config.get("use_v18_gt_slot_conditions", False)
        or config.get("use_v18_gt_slot_edge_types", False)
        or config.get("use_v17_bridge_predicted_relation_tokens", False)
        or config.get("bridge_eval_use_gt_scaffold_tokens", False)
        or config.get("bridge_eval_use_predicted_relation_tokens", False)
        or config.get("v18_eval_use_structure_labels", True)
    )

    train_loader, val_loader, test_loader, num_cids = create_dataloaders(
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
        return_v17_bridge_labels=need_bridge_labels,
    )
    config["num_cids"] = num_cids
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}, CIDs: {num_cids}")

    # Model
    if config["model_type"] == "diffusion":
        model = AFM3DReconModel(config).to(device)
    else:
        model = ResNet3DRegression(
            img_size=config["img_size"],
            num_frames=config["num_frames"],
            max_atoms=MAX_ATOMS,
            num_atom_types=NUM_ATOM_TYPES,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    if config["model_type"] == "diffusion" and config.get("use_type_class_weights", False):
        print("Computing inverse-frequency type class weights from training data...")
        type_counts = torch.zeros(NUM_ATOM_TYPES, device=device, dtype=torch.float)
        for batch in tqdm(train_loader, desc="Type counts", leave=False):
            types = batch["atom_types"].to(device)
            mask = batch["atom_mask"].to(device)
            valid = (types >= 0).bool()
            active = valid & (mask > 0)
            flat = types[active].long()
            if flat.numel() > 0:
                type_counts.scatter_add_(
                    0, flat,
                    torch.ones(flat.shape, device=device, dtype=torch.float),
                )
        counts_safe = type_counts.clamp(min=1.0)
        weights = 1.0 / counts_safe
        weights = weights / weights.sum() * NUM_ATOM_TYPES
        weights = torch.sqrt(weights)
        weights = weights / weights.sum() * NUM_ATOM_TYPES
        model.ddpm.set_type_class_weights(weights)
        print(f"  Type class weights (sqrt-inverse-freq): {weights.cpu().tolist()}")
        for i, (name, cnt) in enumerate(zip(ATOM_TYPES, type_counts.cpu().tolist())):
            print(f"    {name}: count={int(cnt)}, weight={weights[i].item():.3f}")

    if config.get("bridge_train_mode", "full") == "bridge_only":
        allowed_prefixes = [
            "scaffold_token_encoder.",
            "slot_semantic_condition_encoder.",
            "slot_connection_graph_encoder.",
            "ddpm.denoiser.scaffold_proj.",
            "ddpm.denoiser.scaffold_cross_attn.",
            "ddpm.denoiser.scaffold_cross_norm.",
            "ddpm.denoiser.slot_condition_proj.",
            "ddpm.denoiser.slot_local_graph_proj.",
            "ddpm.denoiser.slot_sidechain_graph_proj.",
            "scaffold_relation_head.",
            "scaffold_count_head.",
            "atom_semantic_head.",
            "site_graph_parser_head.",
        ]
        for param in model.parameters():
            param.requires_grad = False
        trainable_names = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in allowed_prefixes):
                param.requires_grad = True
                trainable_names.append(name)
        print(f"Bridge-only training enabled: {len(trainable_names)} parameter tensors trainable")
        for name in trainable_names:
            print(f"  trainable: {name}")

    # Optimizer and scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.get("bridge_lr", config["lr"]) if config.get("bridge_train_mode", "full") == "bridge_only" else config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # Load checkpoint if provided (supports resume from latest.pt)
    resume_epoch = 0
    if config["checkpoint"]:
        state = torch.load(config["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state and "scheduler" in state and "epoch" in state:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            resume_epoch = state["epoch"]
            print(f"Loaded checkpoint: {config['checkpoint']} (resuming from epoch {resume_epoch})")
        else:
            print(f"Loaded checkpoint: {config['checkpoint']}")
    elif config.get("warm_start"):
        warm_start_path = config["warm_start"]
        state = torch.load(warm_start_path, map_location=device, weights_only=False)
        current_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        for key, value in state["model"].items():
            if key not in current_state:
                continue
            if current_state[key].shape != value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(current_state[key].shape)))
                continue
            filtered_state[key] = value
        load_result = model.load_state_dict(filtered_state, strict=False)
        print(f"Warm-started from {warm_start_path}")
        print(f"  Missing keys initialized randomly: {len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            print(f"  Unexpected keys ignored: {len(load_result.unexpected_keys)}")
        if skipped_mismatch:
            print(f"  Shape-mismatched keys skipped: {len(skipped_mismatch)}")
            for key, old_shape, new_shape in skipped_mismatch[:8]:
                print(f"    {key}: ckpt{old_shape} != current{new_shape}")

    # Wire model/config for graceful shutdown signal handler
    _training_model_ref[0] = model
    _training_cfg_ref[0] = config

    if config["eval_only"]:
        print("\n" + "="*60)
        print("Evaluation Only Mode - Test Set Results")
        print("="*60)
        results = evaluate_generation(
            model, test_loader, device, num_samples=len(test_loader.dataset),
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
        print(f"Soft Recon Score:  {results['soft_recon_score']:.4f}")
        print(f"Heavy Hit@Medium:  {results['heavy_atom_hit_rate_medium']:.4f}")
        print(f"Typed Hit@Medium:  {results['typed_heavy_atom_hit_rate_medium']:.4f}")
        print(f"Hetero Hit@Medium: {results['heteroatom_hit_rate_medium']:.4f}")
        print(f"Ring Site@Medium:  {results['ring_site_coverage_medium']:.4f}")
        print(f"Soft Attach@Med:   {results['soft_attachment_site_acc_medium']:.4f}")
        print(f"Macro Type F1:     {results['macro_type_f1']:.4f}")
        print(f"CH Collapse Rate:  {results['ch_collapse_rate']:.4f}")
        print(f"Struct Pass Rate:  {results['struct_fidelity_pass_rate']:.4f}")
        print(f"Struct Fidelity:   {results['struct_fidelity_score']:.4f}")
        print(f"Atom Count Exact:  {results['atom_count_exact']:.4f} (AbsErr: {results['atom_count_abs_error']:.4f})")
        print(f"Matched RMSD:      {results['matched_atom_rmsd']:.4f}")
        print(f"Heavy-Atom RMSD:   {results['matched_heavy_atom_rmsd']:.4f}")
        print(f"Atom Type Acc:     {results['atom_type_acc']:.4f}")
        print(f"Heteroatom F1:     {results['heteroatom_f1']:.4f}")
        print(f"Ring Complete:     {results['ring_complete_rate']:.4f}")
        print(f"Attach Edge F1:    {results['attachment_edge_f1']:.4f}")
        print(f"Bond(pred_mask):   {results['bond_validity_pred_masked']:.4f}")
        print(f"Local Chem Score:  {results['local_chem_score']:.4f}")
        print(f"Legacy Composite:  {results['composite_score']:.4f}")
        print("="*60)
        return

    # Training loop
    os.makedirs(config["save_dir"], exist_ok=True)
    history = {"train": [], "val": []}
    rmsd_history = []
    best_val_loss = float("inf")
    best_gen_key = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -float("inf"))
    # soft_recon, typed_hit_m, ring_site_cov_m, soft_attach_m, hetero_hit_m,
    # macro_type_f1, -ch_collapse_rate, -heavy_rmsd

    # V3: BF16 mixed precision
    use_amp = config.get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using BF16 mixed precision training")

    # V3: evaluation acceleration settings
    eval_ddim_steps = config.get("eval_ddim_steps", 100)
    eval_samples_per_epoch = config.get("eval_samples_per_epoch", 200)
    eval_full_interval = config.get("eval_full_interval", 5)  # full eval every N epochs

    early_stop = False
    start_epoch = resume_epoch + 1

    for epoch in range(start_epoch, config["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            model_type=config["model_type"],
            epoch=epoch, total_epochs=config["epochs"],
            scaler=scaler,
        )
        val_metrics = validate(
            model, val_loader, device,
            model_type=config["model_type"],
            epoch=epoch, total_epochs=config["epochs"],
        )
        scheduler.step()

        dt = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if epoch % config["log_interval"] == 0 or epoch == 1:
            log_stage = get_training_stage(epoch, config["epochs"])
            ring_det_str = ""
            if train_metrics.get("ring_det_loss", 0) > 0:
                ring_det_str = f", ring_det: {train_metrics['ring_det_loss']:.4f}"
            scaffold_aux_str = ""
            if train_metrics.get("scaffold_aux_loss", 0) > 1e-8:
                att = train_metrics.get("attachment_aux_loss", 0.0)
                side = train_metrics.get("sidechain_aux_loss", 0.0)
                scaffold_aux_str = f", scaffold_aux: {train_metrics['scaffold_aux_loss']:.4f}(att:{att:.3f}, side:{side:.3f})"
            structure_aux_str = ""
            if train_metrics.get("structure_count_aux_loss", 0) > 1e-8:
                rs = train_metrics.get("ring_system_count_loss", 0.0)
                sc = train_metrics.get("scaffold_atom_count_loss", 0.0)
                ext = train_metrics.get("non_scaffold_atom_count_loss", 0.0)
                structure_aux_str = (
                    f", structure_aux: {train_metrics['structure_count_aux_loss']:.4f}"
                    f"(rs:{rs:.3f}, sc:{sc:.3f}, ext:{ext:.3f})"
                )
            semantic_aux_str = ""
            if train_metrics.get("semantic_aux_loss", 0) > 1e-8:
                valid = train_metrics.get("semantic_valid_loss", 0.0)
                role = train_metrics.get("atom_role_aux_loss", 0.0)
                hetero = train_metrics.get("hetero_site_aux_loss", 0.0)
                attach = train_metrics.get("attachment_site_aux_loss", 0.0)
                site = train_metrics.get("canonical_site_aux_loss", 0.0)
                attach_site = train_metrics.get("attachment_target_site_aux_loss", 0.0)
                hetero_cls = train_metrics.get("hetero_target_class_aux_loss", 0.0)
                semantic_aux_str = (
                    f", semantic_aux: {train_metrics['semantic_aux_loss']:.4f}"
                    f"(v:{valid:.3f}, role:{role:.3f}, het:{hetero:.3f}, att:{attach:.3f}, "
                    f"site:{site:.3f}, att_site:{attach_site:.3f}, het_cls:{hetero_cls:.3f})"
                )
            site_graph_aux_str = ""
            if train_metrics.get("site_graph_aux_loss", 0) > 1e-8:
                sv = train_metrics.get("site_object_valid_loss", 0.0)
                sc = train_metrics.get("site_coord_aux_loss", 0.0)
                sp = train_metrics.get("site_parent_aux_loss", 0.0)
                se = train_metrics.get("site_element_aux_loss", 0.0)
                sa = train_metrics.get("site_attachment_aux_loss", 0.0)
                sh = train_metrics.get("site_hetero_aux_loss", 0.0)
                sedge = train_metrics.get("site_edge_aux_loss", 0.0)
                site_graph_aux_str = (
                    f", site_graph: {train_metrics['site_graph_aux_loss']:.4f}"
                    f"(v:{sv:.3f}, coord:{sc:.3f}, parent:{sp:.3f}, elem:{se:.3f}, "
                    f"att:{sa:.3f}, het:{sh:.3f}, edge:{sedge:.3f})"
                )
            constraint_str = ""
            if train_metrics.get("constraint_loss", 0) > 1e-8:
                bl = train_metrics.get("bond_length_loss", 0.0)
                ba = train_metrics.get("bond_angle_loss", 0.0)
                pl = train_metrics.get("planarity_loss", 0.0)
                constraint_str = (f", constraint: {train_metrics['constraint_loss']:.4f}"
                                  f"(bl:{bl:.3f}, ba:{ba:.3f}, pl:{pl:.3f})")
            geom_aux_str = ""
            if train_metrics.get("geom_aux_loss", 0) > 1e-8:
                geom_aux_str = f", geom: {train_metrics['geom_aux_loss']:.4f}"
            bottom_z_str = ""
            if train_metrics.get("bottom_z_loss", 0) > 1e-8:
                bottom_z_str = f", bottom_z: {train_metrics['bottom_z_loss']:.4f}"
            enable_c = "ON" if (log_stage >= 2) else "OFF"
            print(
                f"Epoch {epoch:3d}/{config['epochs']} S{log_stage} [constraints={enable_c}] | "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(coord: {train_metrics['coord_loss']:.4f}, type: {train_metrics['type_loss']:.4f}"
                f", count_ce: {train_metrics.get('count_ce_loss', 0.0):.4f}, count_mae: {train_metrics.get('count_mae_loss', 0.0):.4f}"
                f"{ring_det_str}{scaffold_aux_str}{structure_aux_str}{semantic_aux_str}{site_graph_aux_str}{constraint_str}{geom_aux_str}{bottom_z_str}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Time: {dt:.1f}s"
            )

        # V16c: Save best_proxy (by val_loss)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_path = os.path.join(config["save_dir"], f"best_proxy.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
            }, save_path)

        # Always save latest.pt after every epoch (overwrites, always recoverable)
        latest_path = os.path.join(config["save_dir"], f"latest_{config['model_type']}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_metrics["loss"],
            "config": config,
        }, latest_path)

        # Save epoch checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(config["save_dir"], f"epoch_{epoch}_{config['model_type']}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_metrics["loss"],
                "config": config,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # V16c: Generation evaluation with pred count (mainline config)
        is_full_eval = (epoch % eval_full_interval == 0) or (epoch == config["epochs"])
        n_eval = len(val_loader.dataset) if is_full_eval else eval_samples_per_epoch
        eval_label = "FULL" if is_full_eval else "quick"
        print(f"\n[Epoch {epoch}] Eval ({eval_label}, DDIM-{eval_ddim_steps}, {n_eval} samples)...")
        rmsd_results = evaluate_generation(
            model, val_loader, device, num_samples=n_eval,
            use_ddim=True, ddim_steps=eval_ddim_steps,
            use_gt_count=False, disable_guidance=False, disable_ring_snap=False,
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
        rmsd_mean = rmsd_results["rmsd_mean"]
        print(f"[Epoch {epoch}] Soft Recon Score: {rmsd_results['soft_recon_score']:.4f}")
        print(f"           Heavy Hit@Medium: {rmsd_results['heavy_atom_hit_rate_medium']:.4f}")
        print(f"           Typed Hit@Medium: {rmsd_results['typed_heavy_atom_hit_rate_medium']:.4f}")
        print(f"           Hetero Hit@Medium: {rmsd_results['heteroatom_hit_rate_medium']:.4f}")
        print(f"           Soft Attach@Med: {rmsd_results['soft_attachment_site_acc_medium']:.4f}")
        print(f"           Struct Pass Rate: {rmsd_results['struct_fidelity_pass_rate']:.4f}")
        print(f"           Struct Fidelity: {rmsd_results['struct_fidelity_score']:.4f}")
        print(f"           Atom Count Exact: {rmsd_results['atom_count_exact']:.4f} (AbsErr: {rmsd_results['atom_count_abs_error']:.4f})")
        print(f"           Matched RMSD: {rmsd_results['matched_atom_rmsd']:.4f}")
        print(f"           Heavy-Atom RMSD: {rmsd_results['matched_heavy_atom_rmsd']:.4f}")
        print(f"           Atom Type Acc: {rmsd_results['atom_type_acc']:.4f}")
        print(f"           Heteroatom F1: {rmsd_results['heteroatom_f1']:.4f}")
        print(f"           Ring Complete: {rmsd_results['ring_complete_rate']:.4f}")
        print(f"           Attach Edge F1: {rmsd_results['attachment_edge_f1']:.4f}")
        print(f"           Bond(pred_mask): {rmsd_results['bond_validity_pred_masked']:.4f}")
        print(f"           Legacy Composite: {rmsd_results['composite_score']:.4f}")

        # V18: Early training chooses best_gen by tolerance-based soft metrics.
        gen_key = (
            rmsd_results["soft_recon_score"],
            rmsd_results["typed_heavy_atom_hit_rate_medium"],
            rmsd_results["ring_site_coverage_medium"],
            rmsd_results["soft_attachment_site_acc_medium"],
            rmsd_results["heteroatom_hit_rate_medium"],
            rmsd_results["macro_type_f1"],
            -rmsd_results["ch_collapse_rate"],
            -rmsd_results["matched_heavy_atom_rmsd"],
        )
        if gen_key > best_gen_key:
            best_gen_key = gen_key
            gen_save_path = os.path.join(config["save_dir"], "best_gen.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "soft_recon_score": rmsd_results["soft_recon_score"],
                "typed_heavy_atom_hit_rate_medium": rmsd_results["typed_heavy_atom_hit_rate_medium"],
                "heavy_atom_hit_rate_medium": rmsd_results["heavy_atom_hit_rate_medium"],
                "ring_site_coverage_medium": rmsd_results["ring_site_coverage_medium"],
                "soft_attachment_site_acc_medium": rmsd_results["soft_attachment_site_acc_medium"],
                "heteroatom_hit_rate_medium": rmsd_results["heteroatom_hit_rate_medium"],
                "macro_type_f1": rmsd_results["macro_type_f1"],
                "ch_collapse_rate": rmsd_results["ch_collapse_rate"],
                "struct_fidelity_pass_rate": rmsd_results["struct_fidelity_pass_rate"],
                "struct_fidelity_score": rmsd_results["struct_fidelity_score"],
                "matched_heavy_atom_rmsd": rmsd_results["matched_heavy_atom_rmsd"],
                "composite_score": rmsd_results["composite_score"],
                "rmsd": rmsd_mean,
                "bond_validity_gt_masked": rmsd_results["bond_validity_gt_masked"],
                "bond_validity_pred_masked": rmsd_results["bond_validity_pred_masked"],
                "config": config,
            }, gen_save_path)
            print(
                "  -> New best_gen.pt "
                f"(soft={rmsd_results['soft_recon_score']:.4f}, "
                f"typed_hit_m={rmsd_results['typed_heavy_atom_hit_rate_medium']:.4f}, "
                f"ring_site_m={rmsd_results['ring_site_coverage_medium']:.4f}, "
                f"soft_attach_m={rmsd_results['soft_attachment_site_acc_medium']:.4f}, "
                f"hetero_hit_m={rmsd_results['heteroatom_hit_rate_medium']:.4f}, "
                f"heavy_rmsd={rmsd_results['matched_heavy_atom_rmsd']:.4f})"
            )

        rmsd_history.append({
            "epoch": epoch,
            "rmsd_mean": rmsd_results["rmsd_mean"],
            "rmsd_std": rmsd_results["rmsd_std"],
            "bottom_recall_mean": rmsd_results["bottom_recall_mean"],
            "bond_validity_gt_masked": rmsd_results["bond_validity_gt_masked"],
            "bond_validity_pred_masked": rmsd_results["bond_validity_pred_masked"],
            "count_exact_match": rmsd_results["count_exact_match"],
            "count_mae": rmsd_results["count_mae"],
            "type_match_rate": rmsd_results["type_match_rate"],
            "ring_preservation": rmsd_results.get("ring_preservation", 0.0),
            "composite_score": rmsd_results["composite_score"],
            "soft_recon_score": rmsd_results["soft_recon_score"],
            "heavy_atom_hit_rate_medium": rmsd_results["heavy_atom_hit_rate_medium"],
            "typed_heavy_atom_hit_rate_medium": rmsd_results["typed_heavy_atom_hit_rate_medium"],
            "heteroatom_hit_rate_medium": rmsd_results["heteroatom_hit_rate_medium"],
            "soft_attachment_site_acc_medium": rmsd_results["soft_attachment_site_acc_medium"],
            "ring_site_coverage_medium": rmsd_results["ring_site_coverage_medium"],
            "macro_type_f1": rmsd_results["macro_type_f1"],
            "ch_collapse_rate": rmsd_results["ch_collapse_rate"],
            "struct_fidelity_pass_rate": rmsd_results["struct_fidelity_pass_rate"],
            "struct_fidelity_score": rmsd_results["struct_fidelity_score"],
            "atom_count_exact": rmsd_results["atom_count_exact"],
            "atom_count_abs_error": rmsd_results["atom_count_abs_error"],
            "matched_atom_rmsd": rmsd_results["matched_atom_rmsd"],
            "matched_heavy_atom_rmsd": rmsd_results["matched_heavy_atom_rmsd"],
            "atom_type_acc": rmsd_results["atom_type_acc"],
            "heteroatom_f1": rmsd_results["heteroatom_f1"],
            "ring_complete_rate": rmsd_results["ring_complete_rate"],
            "attachment_edge_f1": rmsd_results["attachment_edge_f1"],
            "local_chem_score": rmsd_results["local_chem_score"],
        })

        # V16c: Three-config sanity eval every eval_full_interval epochs
        if is_full_eval:
            print(f"\n[Epoch {epoch}] Sanity Eval (3 configs, {min(n_eval, 200)} samples)...")
            sanity = run_sanity_evals(model, val_loader, device,
                                      num_samples=min(n_eval, 200),
                                      ddim_steps=eval_ddim_steps, epoch=epoch,
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
                                      guidance_time_threshold=config.get("guidance_time_threshold", 500))

    # Save evaluation metrics history
    if rmsd_history:
        metrics_path = os.path.join(config["save_dir"], f"metrics_{config['model_type']}.json")
        with open(metrics_path, "w") as f:
            json.dump(rmsd_history, f, indent=2)
        print(f"Evaluation metrics saved to: {metrics_path}")

    # Save training history
    history_path = os.path.join(config["save_dir"], f"history_{config['model_type']}.json")
    # Convert tensors in history
    for split in history:
        for i, m in enumerate(history[split]):
            history[split][i] = {k: float(v) for k, v in m.items()}
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {config['save_dir']}")

    # Final evaluation on test set
    print("\n" + "="*60)
    print("Final Evaluation on Test Set")
    print("="*60)
    results = evaluate_generation(
        model, test_loader, device, num_samples=len(test_loader.dataset),
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
    print(f"Soft Recon Score:  {results['soft_recon_score']:.4f}")
    print(f"Heavy Hit@Medium:  {results['heavy_atom_hit_rate_medium']:.4f}")
    print(f"Typed Hit@Medium:  {results['typed_heavy_atom_hit_rate_medium']:.4f}")
    print(f"Hetero Hit@Medium: {results['heteroatom_hit_rate_medium']:.4f}")
    print(f"Ring Site@Medium:  {results['ring_site_coverage_medium']:.4f}")
    print(f"Soft Attach@Med:   {results['soft_attachment_site_acc_medium']:.4f}")
    print(f"Macro Type F1:     {results['macro_type_f1']:.4f}")
    print(f"CH Collapse Rate:  {results['ch_collapse_rate']:.4f}")
    print(f"Struct Pass Rate:  {results['struct_fidelity_pass_rate']:.4f}")
    print(f"Struct Fidelity:   {results['struct_fidelity_score']:.4f}")
    print(f"Atom Count Exact:  {results['atom_count_exact']:.4f} (AbsErr: {results['atom_count_abs_error']:.4f})")
    print(f"Matched RMSD:      {results['matched_atom_rmsd']:.4f}")
    print(f"Heavy-Atom RMSD:   {results['matched_heavy_atom_rmsd']:.4f}")
    print(f"Atom Type Acc:     {results['atom_type_acc']:.4f}")
    print(f"Heteroatom F1:     {results['heteroatom_f1']:.4f}")
    print(f"Ring Complete:     {results['ring_complete_rate']:.4f}")
    print(f"Attach Edge F1:    {results['attachment_edge_f1']:.4f}")
    print(f"Bond(pred_mask):   {results['bond_validity_pred_masked']:.4f}")
    print(f"Local Chem Score:  {results['local_chem_score']:.4f}")
    print(f"Legacy Composite:  {results['composite_score']:.4f}")
    print("="*60)

    # Save model predictions with structure similarity
    print("\nSaving model predictions...")
    pred_path = os.path.join(config["save_dir"], f"predictions_{config['model_type']}.json")
    save_predictions(model, test_loader, device, pred_path, num_samples=100)
    print(f"Predictions saved to: {pred_path}")
    print(f"  Contains: coords, atom_types, n_atoms_pred, structure_similarity")



if __name__ == "__main__":
    import signal
    import sys

    # Graceful shutdown: save latest checkpoint on SIGTERM/SIGINT
    _training_model_ref = [None]
    _training_cfg_ref = [None]

    def _signal_handler(signum, frame):
        print(f"\n[Signal {signum}] Saving checkpoint before exit...")
        if _training_model_ref[0] is not None and _training_cfg_ref[0] is not None:
            import os as _os
            save_dir = _training_cfg_ref[0].get("save_dir", "checkpoints")
            _os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "model": _training_model_ref[0].state_dict(),
                "config": _training_cfg_ref[0],
            }, _os.path.join(save_dir, "emergency_latest.pt"))
            print(f"[Signal] Emergency checkpoint saved to {save_dir}/emergency_latest.pt")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    main()
