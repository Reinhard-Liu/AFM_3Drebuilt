"""
TypeNet: Decoupled atom type predictor.

Predicts atom types from three information sources:
1. Atom coordinates (position encoding)
2. Local coordination environment (neighbor statistics)
3. AFM patch features via cross-attention (spatial context from encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeNet(nn.Module):
    """Decoupled atom type predictor using coordinates + environment + AFM patches."""

    def __init__(
        self,
        cond_dim: int = 512,
        hidden_dim: int = 256,
        num_types: int = 10,
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_types = num_types

        # Source 1: coordinate encoding
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Source 2: coordination environment features
        # [n_neighbors, mean_dist, min_dist, max_dist, dist_var]
        self.env_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Source 3: AFM spatial info via cross-attention
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Fusion of 3 sources (3 * hidden_dim -> hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
        )

        # Transformer layers for inter-atom reasoning
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        # Type prediction head
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_types),
        )

        # Typical bond lengths (normalized by /12.0) for valence consistency
        # H, C, N, O, F, S, P, Cl, Br, I
        self.register_buffer(
            "typical_bond_lengths",
            torch.tensor([
                1.09, 1.54, 1.47, 1.43, 1.35, 1.81, 1.84, 1.77, 1.94, 2.14
            ]) / 12.0,
        )
        # Max valence per element type
        self.register_buffer(
            "max_valence",
            torch.tensor([1, 4, 3, 2, 1, 6, 5, 1, 1, 1], dtype=torch.float32),
        )

    def compute_coordination_features(
        self, coords: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute local coordination environment features.

        Args:
            coords: (B, N, 3) atom coordinates
            mask: (B, N) atom mask

        Returns:
            env_features: (B, N, 5) [n_neighbors, mean_dist, min_dist, max_dist, dist_var]
        """
        B, N, _ = coords.shape
        device = coords.device

        # Pairwise distances: (B, N, N)
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        dist = diff.norm(dim=-1)  # (B, N, N)

        # Mask: valid pairs only
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)
        # Exclude self
        eye = torch.eye(N, device=device).unsqueeze(0)
        pair_mask = pair_mask * (1 - eye)

        # Neighbor threshold: ~2.0 Angstrom / 12.0 = 0.167 in normalized space
        neighbor_thresh = 0.20
        is_neighbor = (dist < neighbor_thresh) & (pair_mask > 0)

        # Count neighbors
        n_neighbors = is_neighbor.float().sum(dim=-1)  # (B, N)

        # Distance stats (only among neighbors)
        large_val = 1e6
        dist_masked = dist + (1 - is_neighbor.float()) * large_val

        min_dist = dist_masked.min(dim=-1).values  # (B, N)
        min_dist = torch.where(n_neighbors > 0, min_dist, torch.zeros_like(min_dist))

        # For mean/max/var, use neighbor distances
        dist_neighbor = dist * is_neighbor.float()
        sum_dist = dist_neighbor.sum(dim=-1)
        mean_dist = sum_dist / n_neighbors.clamp(min=1)

        dist_masked_neg = dist - (1 - is_neighbor.float()) * large_val
        max_dist = dist_masked_neg.max(dim=-1).values
        max_dist = torch.where(n_neighbors > 0, max_dist, torch.zeros_like(max_dist))

        # Variance
        sq_diff = (dist_neighbor - mean_dist.unsqueeze(-1)) ** 2 * is_neighbor.float()
        dist_var = sq_diff.sum(dim=-1) / n_neighbors.clamp(min=1)

        # Normalize n_neighbors to reasonable range
        n_neighbors = n_neighbors / 6.0  # typical max neighbors

        env_features = torch.stack([
            n_neighbors, mean_dist, min_dist, max_dist, dist_var,
        ], dim=-1)  # (B, N, 5)

        return env_features

    def forward(
        self,
        coords: torch.Tensor,
        c_global: torch.Tensor,
        c_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) atom coordinates
            c_global: (B, cond_dim) global condition (unused here, kept for interface)
            c_patches: (B, P, cond_dim) patch-level features from ViT
            mask: (B, N) atom mask

        Returns:
            type_logits: (B, N, num_types)
        """
        # Source 1: coordinate features
        h_coord = self.coord_encoder(coords)  # (B, N, hidden_dim)

        # Source 2: environment features
        env_feat = self.compute_coordination_features(coords, mask)  # (B, N, 5)
        h_env = self.env_encoder(env_feat)  # (B, N, hidden_dim)

        # Source 3: cross-attention to AFM patches
        c_proj = self.cond_proj(c_patches)  # (B, P, hidden_dim)
        h_cross = self.cross_norm(h_coord)  # query from coord features
        h_cross, _ = self.cross_attn(h_cross, c_proj, c_proj)  # (B, N, hidden_dim)

        # Fuse all three sources
        h = self.fusion(torch.cat([h_coord, h_env, h_cross], dim=-1))  # (B, N, hidden_dim)

        # Transformer layers with padding mask
        src_key_padding_mask = (mask == 0)  # True = ignore
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        # Predict types
        type_logits = self.type_head(h)  # (B, N, num_types)

        return type_logits

    def valence_consistency_loss(
        self,
        type_logits: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize predicted types that are inconsistent with local bonding geometry.

        If an atom has many close neighbors, it shouldn't be predicted as H (valence 1).
        If an atom has few neighbors, it shouldn't be predicted as C (valence 4).

        Args:
            type_logits: (B, N, num_types)
            coords: (B, N, 3)
            mask: (B, N)

        Returns:
            loss: scalar
        """
        B, N, _ = type_logits.shape
        device = type_logits.device

        # Soft type probabilities
        type_probs = F.softmax(type_logits, dim=-1)  # (B, N, num_types)

        # Count neighbors within bonding distance
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)  # (B, N, N)

        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye = torch.eye(N, device=device).unsqueeze(0)
        pair_mask = pair_mask * (1 - eye)

        # Soft neighbor count using sigmoid around bond threshold
        bond_thresh = 0.18  # ~2.16 Angstrom normalized
        neighbor_score = torch.sigmoid((bond_thresh - dist) * 30.0) * pair_mask
        n_neighbors = neighbor_score.sum(dim=-1)  # (B, N)

        # Expected max valence from predicted type distribution
        expected_valence = (type_probs * self.max_valence.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, N)

        # Penalty: neighbor count exceeds expected valence
        excess = F.relu(n_neighbors - expected_valence)
        loss = (excess * mask).sum() / mask.sum().clamp(min=1)

        return loss
