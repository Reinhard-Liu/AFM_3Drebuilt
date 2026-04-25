"""
Lightweight ablation-only head variants for V20 EXP-07.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn_type_classifier import GNNTypeClassifier


class LegacyGNNTypeHeadAdapter(nn.Module):
    """Legacy-style graph type head with a V19-compatible interface."""

    def __init__(
        self,
        shared_feat_dim: int = 64,
        hidden_dim: int = 192,
        num_types: int = 10,
        num_gnn_layers: int = 4,
        num_heads: int = 4,
        bond_threshold: float = 0.20,
        token_grid_size: int = 16,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_types = int(num_types)
        self.label_smoothing = float(label_smoothing)
        self.token_grid_size = int(max(token_grid_size, 4))
        self.classifier = GNNTypeClassifier(
            cond_dim=shared_feat_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_types=num_types,
            num_heads=num_heads,
            bond_threshold=bond_threshold,
        )

    def _shared_tokens(self, shared_feat: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(shared_feat, (self.token_grid_size, self.token_grid_size))
        return pooled.flatten(2).transpose(1, 2)

    def forward(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
        center_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del center_map
        tokens = self._shared_tokens(shared_feat)
        logits = self.classifier(coords, tokens, mask, afm_stack=afm_stack)
        coarse = torch.zeros(
            logits.shape[0],
            logits.shape[1],
            3,
            device=logits.device,
            dtype=logits.dtype,
        )
        hetero = torch.zeros(
            logits.shape[0],
            logits.shape[1],
            device=logits.device,
            dtype=logits.dtype,
        )
        return logits, coarse, hetero

    def compute_loss(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        class_weight: torch.Tensor | None = None,
        center_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, _, _ = self.forward(coords, shared_feat, afm_stack, mask, center_map=center_map)
        logits_flat = logits.reshape(-1, self.num_types)
        types_flat = atom_types.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = (mask_flat > 0) & (types_flat >= 0)
        if valid.sum() == 0:
            loss = torch.tensor(0.0, device=coords.device)
        else:
            loss = F.cross_entropy(
                logits_flat[valid],
                types_flat[valid],
                weight=class_weight,
                label_smoothing=self.label_smoothing,
            )
        return loss, logits


class ZeroEdgeHead(nn.Module):
    """Edge head ablation that always predicts no edges."""

    def forward(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del shared_feat, afm_stack, mask
        bsz, n_atoms, _ = coords.shape
        return torch.zeros(bsz, n_atoms, n_atoms, device=coords.device, dtype=coords.dtype)

    def compute_loss(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_labels
        logits = self.forward(coords, shared_feat, afm_stack, mask)
        return torch.tensor(0.0, device=coords.device), logits
