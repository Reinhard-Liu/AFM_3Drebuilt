"""
GNN TypeClassifier: Post-hoc atom type prediction from generated coordinates.

Key design principles (learned from V1-V11 failures):
1. Completely independent from denoiser — no gradient competition
2. Trained on DDIM-generated coordinates — no exposure bias
3. Exploits precise chemical geometry (bond lengths, coordination numbers, angles)
4. Uses AFM patch features via cross-attention for element-specific info

References:
- EGNN (Satorras et al., ICML 2021): E(n) equivariant message passing
- SchNet (Schütt et al., NeurIPS 2017): continuous-filter convolution
- DimeNet++ (Gasteiger et al., ICLR 2020): directional message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Covalent radii in normalized space (Angstrom / 12.0)
COVALENT_RADII = torch.tensor([
    0.0258, 0.0642, 0.0608, 0.0550, 0.0533,
    0.0867, 0.0892, 0.0825, 0.0950, 0.1108,
])  # H, C, N, O, F, S, P, Cl, Br, I


class GNNTypeClassifier(nn.Module):
    """Predicts atom types from a molecular graph built on generated coordinates.

    Unlike denoiser's type_head (which shares transformer features with coord_head
    and suffers from gradient competition), this GNN:
    - Has its own parameters (no competition)
    - Is trained on generated coords (no exposure bias)
    - Exploits precise bond lengths and coordination numbers
    """

    def __init__(
        self,
        node_feat_dim: int = 64,
        edge_feat_dim: int = 32,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        num_types: int = 10,
        cond_dim: int = 512,
        num_heads: int = 4,
        bond_threshold: float = 0.20,  # max bond distance in normalized space
    ):
        super().__init__()
        self.num_types = num_types
        self.bond_threshold = bond_threshold
        self.hidden_dim = hidden_dim

        # Node feature encoder:
        # [coord(3) + n_neighbors(1) + mean_bond_len(1) + coord_std(3) + afm_local(10)] = 18
        self.node_encoder = nn.Sequential(
            nn.Linear(18, node_feat_dim),
            nn.GELU(),
            nn.Linear(node_feat_dim, hidden_dim),
        )

        # Edge feature encoder: [distance(1) + relative_pos(3)] = 4
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, edge_feat_dim),
            nn.GELU(),
            nn.Linear(edge_feat_dim, hidden_dim),
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # AFM patch cross-attention (element-specific spatial info)
        self.patch_proj = nn.Linear(cond_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Type prediction head
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_types),
        )

    @staticmethod
    def sample_afm_local(coords, afm_stack, mask):
        """Sample AFM pixel intensity at each atom's XY position.

        Each atom gets a 10-dim vector (one per AFM depth slice),
        providing element-specific local density information.

        Args:
            coords: (B, N, 3) in normalized [-1, 1] space
            afm_stack: (B, 10, H, W)
            mask: (B, N)

        Returns:
            afm_local: (B, N, 10)
        """
        B, N, _ = coords.shape
        D = afm_stack.shape[1]
        H = afm_stack.shape[2]

        # Map coords XY from [-1, 1] to [0, H-1]
        px = ((coords[:, :, 0] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)  # (B, N)
        py = ((coords[:, :, 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)

        # Sample: afm_stack[b, d, py, px] for each atom
        afm_local = torch.zeros(B, N, D, device=coords.device)
        for b in range(B):
            for i in range(N):
                if mask[b, i] > 0:
                    afm_local[b, i] = afm_stack[b, :, py[b, i], px[b, i]]

        return afm_local

    def build_graph(self, coords, mask, afm_stack=None):
        """Build molecular graph from coordinates using distance threshold.

        Returns:
            node_features: (B, N, 18) per-atom geometric + AFM features
            adjacency: (B, N, N)
            diff: (B, N, N, 3)
            dist: (B, N, N)
        """
        B, N, _ = coords.shape
        device = coords.device

        # Pairwise distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        dist = diff.norm(dim=-1)  # (B, N, N)

        # Adjacency: bond if distance < threshold
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye = torch.eye(N, device=device).unsqueeze(0)
        adj = ((dist < self.bond_threshold) & (pair_mask > 0)) & ~eye.bool()

        # Node features: [coord, n_neighbors, mean_bond_len, coord_relative_std]
        n_neighbors = adj.float().sum(dim=-1)  # (B, N)

        # Mean bond length per atom
        bond_dist = dist * adj.float() + (1 - adj.float()) * 999
        bond_dist_masked = torch.where(adj, dist, torch.zeros_like(dist))
        mean_bond_len = bond_dist_masked.sum(dim=-1) / n_neighbors.clamp(min=1)

        # Coordinate relative to center
        center = (coords * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        coord_rel = coords - center

        # AFM local pixel sampling (10 dim per atom)
        if afm_stack is not None:
            afm_local = self.sample_afm_local(coords, afm_stack, mask)
        else:
            afm_local = torch.zeros(B, N, 10, device=device)

        node_features = torch.cat([
            coord_rel,                          # (B, N, 3)
            n_neighbors.unsqueeze(-1) / 4.0,    # (B, N, 1) normalized
            mean_bond_len.unsqueeze(-1),         # (B, N, 1)
            coord_rel.abs(),                     # (B, N, 3) anisotropy
            afm_local,                           # (B, N, 10) AFM local density
        ], dim=-1)  # (B, N, 18)

        return node_features, adj, diff, dist

    def forward(self, coords, c_patches, mask, afm_stack=None):
        """
        Args:
            coords: (B, N, 3) generated atom coordinates
            c_patches: (B, P, cond_dim) AFM patch features from ViT
            mask: (B, N) atom mask
            afm_stack: (B, 10, H, W) optional AFM images for local sampling

        Returns:
            type_logits: (B, N, num_types)
        """
        B, N, _ = coords.shape

        # Build graph with AFM local features
        node_feat, adj, diff, dist = self.build_graph(coords, mask, afm_stack=afm_stack)

        # Encode nodes
        h = self.node_encoder(node_feat)  # (B, N, hidden_dim)

        # Encode edges (for message passing)
        # Edge features: [distance, relative_position(3)]
        dist_norm = dist.unsqueeze(-1) / 0.2  # normalize by threshold
        edge_feat_raw = torch.cat([dist_norm, diff / (dist.unsqueeze(-1) + 1e-8)], dim=-1)  # (B, N, N, 4)
        edge_feat = self.edge_encoder(edge_feat_raw)  # (B, N, N, hidden_dim)

        # Message passing
        attn_mask = ~adj  # True = no edge (ignore in message passing)
        for layer in self.gnn_layers:
            h = layer(h, edge_feat, adj, mask)

        # AFM cross-attention
        p = self.patch_proj(c_patches)
        h_norm = self.cross_norm(h)
        cross_out, _ = self.cross_attn(h_norm, p, p)
        h = h + 0.3 * cross_out

        # Predict types
        type_logits = self.type_head(h)
        return type_logits

    def compute_loss(self, coords, c_patches, atom_types, mask, afm_stack=None):
        """Compute classification loss."""
        type_logits = self.forward(coords, c_patches, mask, afm_stack=afm_stack)

        logits_flat = type_logits.reshape(-1, self.num_types)
        types_flat = atom_types.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = (mask_flat > 0) & (types_flat >= 0)

        if valid.sum() > 0:
            valid_types = types_flat[valid]
            counts = torch.bincount(valid_types, minlength=self.num_types).float().clamp(min=1.0)
            inv_freq = valid_types.numel() / (self.num_types * counts)
            class_weight = torch.sqrt(inv_freq).clamp(max=3.0).to(coords.device)
            loss = F.cross_entropy(logits_flat[valid], valid_types, weight=class_weight)
        else:
            loss = torch.tensor(0.0, device=coords.device)

        return loss


class MessagePassingLayer(nn.Module):
    """Simple message passing with edge features."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_feat, adj, mask):
        """
        h: (B, N, D)
        edge_feat: (B, N, N, D)
        adj: (B, N, N) bool
        mask: (B, N)
        """
        B, N, D = h.shape

        # Messages: concat(h_i, h_j, edge_ij) for each edge
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        messages = self.message_mlp(torch.cat([h_i, h_j, edge_feat], dim=-1))  # (B, N, N, D)

        # Mask messages by adjacency
        messages = messages * adj.unsqueeze(-1).float()

        # Aggregate (sum)
        agg = messages.sum(dim=2)  # (B, N, D)
        n_neighbors = adj.float().sum(dim=2, keepdim=True).clamp(min=1)
        agg = agg / n_neighbors  # mean aggregation

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new) * mask.unsqueeze(-1)

        return h
