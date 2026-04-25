"""
Center-conditioned edge head for V19 object-level training.

This head predicts whether a pair of atom centers should be connected,
using shared AFM encoder features, local AFM traces, and geometric context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterConditionedEdgeHead(nn.Module):
    def __init__(
        self,
        shared_feat_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        node_in_dim = shared_feat_dim + 10 + 3 + 5
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        pair_in_dim = hidden_dim * 2 + 7
        self.edge_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # V20: a lightweight graph refinement branch that keeps warm-start
        # compatibility. Base pair logits are preserved; the refinement branch
        # starts from near-zero contribution and learns to use neighborhood
        # context to stabilize edge prediction on predicted centers.
        self.msg_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.refine_gate = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.node_refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.refine_edge_mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.msg_mlp[-1].weight)
        nn.init.zeros_(self.msg_mlp[-1].bias)
        nn.init.zeros_(self.node_refine[-1].weight)
        nn.init.zeros_(self.node_refine[-1].bias)
        nn.init.zeros_(self.refine_edge_mlp[-1].weight)
        nn.init.zeros_(self.refine_edge_mlp[-1].bias)

    @staticmethod
    def _sample_map_channels(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        grid = coords[..., :2].unsqueeze(2)
        sampled = F.grid_sample(feat, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).transpose(1, 2)

    @staticmethod
    def _compute_env_features(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n_atoms, _ = coords.shape
        device = coords.device

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye = torch.eye(n_atoms, device=device).unsqueeze(0)
        pair_mask = pair_mask * (1 - eye)

        neighbor_thresh = 0.20
        is_neighbor = (dist < neighbor_thresh) & (pair_mask > 0)
        n_neighbors = is_neighbor.float().sum(dim=-1)
        dist_masked = torch.where(is_neighbor, dist, torch.zeros_like(dist))
        mean_dist = dist_masked.sum(dim=-1) / n_neighbors.clamp(min=1.0)

        inf = torch.full_like(dist, 1e6)
        min_dist = torch.where(is_neighbor, dist, inf).min(dim=-1).values
        min_dist = torch.where(n_neighbors > 0, min_dist, torch.zeros_like(min_dist))

        ninf = torch.full_like(dist, -1e6)
        max_dist = torch.where(is_neighbor, dist, ninf).max(dim=-1).values
        max_dist = torch.where(n_neighbors > 0, max_dist, torch.zeros_like(max_dist))

        var = (((dist_masked - mean_dist.unsqueeze(-1)) ** 2) * is_neighbor.float()).sum(dim=-1) / n_neighbors.clamp(min=1.0)

        return torch.stack([
            n_neighbors / 6.0,
            mean_dist,
            min_dist,
            max_dist,
            var,
        ], dim=-1)

    def forward(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        shared_local = self._sample_map_channels(shared_feat, coords)
        afm_local = self._sample_map_channels(afm_stack, coords)
        env = self._compute_env_features(coords, mask)
        node_feat = torch.cat([shared_local, afm_local, coords, env], dim=-1)
        node_embed = self.node_mlp(node_feat)

        ci = coords.unsqueeze(2)
        cj = coords.unsqueeze(1)
        delta = ci - cj
        dist = delta.norm(dim=-1, keepdim=True)
        pair_geom = torch.cat([delta, delta.abs(), dist], dim=-1)

        hi = node_embed.unsqueeze(2).expand(-1, -1, coords.shape[1], -1)
        hj = node_embed.unsqueeze(1).expand(-1, coords.shape[1], -1, -1)
        pair_feat = torch.cat([hi, hj, pair_geom], dim=-1)
        base_logits = self.edge_mlp(pair_feat).squeeze(-1)

        n_atoms = coords.shape[1]
        pair_mask = (mask.unsqueeze(2) > 0) & (mask.unsqueeze(1) > 0)
        eye = torch.eye(n_atoms, device=coords.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & (~eye)

        base_prob = torch.sigmoid(base_logits).unsqueeze(-1)
        gate = torch.sigmoid(self.refine_gate(pair_feat))
        msg = self.msg_mlp(pair_feat)
        weighted_msg = msg * gate * base_prob * pair_mask.unsqueeze(-1).float()
        msg_sum = weighted_msg.sum(dim=2)
        msg_norm = pair_mask.sum(dim=2, keepdim=False).clamp(min=1).unsqueeze(-1).float()
        msg_mean = msg_sum / msg_norm

        refine_input = torch.cat([node_embed, msg_mean], dim=-1)
        node_embed_refined = node_embed + self.node_refine(refine_input)

        hi_ref = node_embed_refined.unsqueeze(2).expand(-1, -1, coords.shape[1], -1)
        hj_ref = node_embed_refined.unsqueeze(1).expand(-1, coords.shape[1], -1, -1)
        pair_feat_ref = torch.cat([hi_ref, hj_ref, pair_geom], dim=-1)
        refine_logits = self.refine_edge_mlp(pair_feat_ref).squeeze(-1)
        logits = base_logits + refine_logits
        return logits

    def compute_loss(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(coords, shared_feat, afm_stack, mask)

        valid_pair = (mask.unsqueeze(2) > 0) & (mask.unsqueeze(1) > 0)
        eye = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid_pair & (~eye)

        labels = edge_labels.float()
        if valid_pair.sum() == 0:
            return torch.tensor(0.0, device=coords.device), logits

        pos = labels[valid_pair].sum().item()
        neg = valid_pair.sum().item() - pos
        pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0), device=coords.device)
        loss = F.binary_cross_entropy_with_logits(
            logits[valid_pair],
            labels[valid_pair],
            pos_weight=pos_weight,
        )
        return loss, logits
