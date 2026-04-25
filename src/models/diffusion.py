"""
Conditional Diffusion Model (DDPM) for 3D molecular structure generation.

V9: V8 base + shape-conditioned denoiser + shape guidance during sampling.
    Solves the structure compression (dense ball) problem.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from src.models.constraints import (
    IDEAL_BOND_LENGTHS, MAX_BOND_DIST, BOND_VALIDITY_TOLERANCE,
    VDW_RADII, BOND_TOLERANCE,
)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def compute_shape_descriptors(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute molecular shape descriptors from gyration tensor eigenvalues.

    Returns (B, 3): [asphericity, acylindricity, relative_shape_anisotropy]
    - asphericity: 0=sphere, large=anisotropic
    - acylindricity: 0=cylindrical, large=non-cylindrical
    - rel_shape_anisotropy: 0=sphere, 1=linear
    """
    B = coords.shape[0]
    descs = []
    with torch.amp.autocast('cuda', enabled=False):
      for b in range(B):
        m = mask[b].bool()
        n = m.sum().item()
        if n < 3:
            descs.append(torch.zeros(3, device=coords.device))
            continue
        c = coords[b, m].float()
        c = c - c.mean(0, keepdim=True)
        S = c.T @ c / n
        eig = torch.linalg.eigvalsh(S)
        l3, l2, l1 = eig[0], eig[1], eig[2]
        trace = (l1 + l2 + l3).clamp(min=1e-8)
        asphericity = l1 - 0.5 * (l2 + l3)
        acylindricity = l2 - l3
        anisotropy = 1.0 - 3.0 * (l1*l2 + l1*l3 + l2*l3) / (trace ** 2)
        descs.append(torch.stack([asphericity, acylindricity, anisotropy]))
    return torch.stack(descs)  # (B, 3)


class SE3EquivariantDenoiser(nn.Module):
    """V15 SpatialDenoiser: coord_head + type_head both with AFM patch cross-attention.

    V15 changes: removed SE(3) equivariance assumption (AFM has fixed coordinate frame).
    coord_head now receives c_patches spatial features via cross-attention.
    """

    def __init__(
        self,
        max_atoms: int = 85,
        coord_dim: int = 3,
        num_atom_types: int = 10,
        cond_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        scaffold_attn_every: int = 2,
        scaffold_attn_scale: float = 0.1,
        slot_graph_scale: float = 0.15,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.scaffold_attn_every = max(1, scaffold_attn_every)
        self.scaffold_attn_scale = scaffold_attn_scale
        self.slot_graph_scale = slot_graph_scale

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Global condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # V9: Shape conditioning projection
        self.shape_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input embedding
        self.coord_embed = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            DenoiserBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # V17-Bridge: token-level scaffold conditioning. This is kept optional
        # and only activates when GT scaffold tokens are explicitly passed in.
        self.scaffold_proj = nn.Linear(cond_dim, hidden_dim)
        self.scaffold_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.scaffold_cross_norm = nn.LayerNorm(hidden_dim)
        self.slot_condition_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_local_graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_sidechain_graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate noise prediction head
        self.coord_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, coord_dim),
        )

        # V15: AFM patch cross-attention for COORDINATE prediction
        self.coord_patch_proj = nn.Linear(cond_dim, hidden_dim)
        self.coord_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.coord_cross_norm = nn.LayerNorm(hidden_dim)

        # V8: AFM patch cross-attention for type prediction
        self.patch_proj = nn.Linear(cond_dim, hidden_dim)
        self.local_type_proj = nn.Linear(cond_dim, hidden_dim)
        self.type_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.type_cross_norm = nn.LayerNorm(hidden_dim)

        # V14: Type adapter — separate feature path to reduce coord/type competition
        self.type_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Atom type prediction head (with AFM spatial info)
        self.type_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

    def _sample_local_patch_features(self, coords: torch.Tensor, c_patches: torch.Tensor) -> torch.Tensor:
        """Sample local AFM patch features at predicted atom XY positions."""
        B, N, _ = coords.shape
        _, P, D = c_patches.shape
        if P % 64 != 0:
            pooled = c_patches.mean(dim=1, keepdim=True).expand(-1, N, -1)
            return self.local_type_proj(pooled)

        n_temporal = P // 64
        spatial = c_patches.view(B, n_temporal, 64, D).mean(dim=1)
        grid_feat = spatial.view(B, 8, 8, D).permute(0, 3, 1, 2).contiguous()
        # V16c: removed /3.0 that compressed sampling region to center 1/9
        grid = coords[..., :2].clamp(-1.0, 1.0).unsqueeze(2)
        sampled = F.grid_sample(
            grid_feat, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()
        return self.local_type_proj(sampled)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c_global: torch.Tensor,
        c_patches: torch.Tensor,
        mask: torch.Tensor,
        shape_desc: torch.Tensor = None,
        diffusion_params: dict = None,
        scaffold_tokens: torch.Tensor = None,
        scaffold_token_mask: torch.Tensor = None,
        slot_conditions: torch.Tensor = None,
        slot_edge_types: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            x_t: (B, N, 3) noisy coordinates
            t: (B,) timestep indices
            c_global: (B, cond_dim) global condition from ViT CLS
            c_patches: (B, P, cond_dim) patch-level features from ViT
            mask: (B, N) atom mask
            shape_desc: (B, 3) shape descriptors
            diffusion_params: dict with sqrt_alpha, sqrt_one_minus for x_0 reconstruction (EDM*+γ)
            scaffold_tokens: (B, T, cond_dim) optional GT scaffold tokens
            scaffold_token_mask: (B, T) bool mask for valid scaffold tokens

        Returns:
            eps_pred: (B, N, 3) predicted noise
            type_logits: (B, N, num_atom_types) atom type logits
        """
        B, N, _ = x_t.shape

        t_emb = self.time_embed(t)
        c_emb = self.cond_proj(c_global)
        h = self.coord_embed(x_t)
        if slot_conditions is not None:
            h = h + self.slot_condition_proj(slot_conditions.float()) * mask.unsqueeze(-1)

        # V9: inject shape conditioning
        global_bias = t_emb + c_emb
        if shape_desc is not None:
            shape_emb = self.shape_proj(shape_desc.to(global_bias.dtype))
            global_bias = global_bias + shape_emb
        h = h + global_bias.unsqueeze(1)

        attn_mask = (mask == 0)

        scaffold_memory = None
        scaffold_padding_mask = None
        use_scaffold = scaffold_tokens is not None and scaffold_token_mask is not None
        if use_scaffold:
            scaffold_memory = self.scaffold_proj(scaffold_tokens)
            scaffold_padding_mask = ~scaffold_token_mask.bool()

        for layer_idx, layer in enumerate(self.layers, start=1):
            if slot_edge_types is not None:
                h = self._apply_slot_graph_condition(h, slot_edge_types, mask)
            h = layer(h, attn_mask)
            if use_scaffold and (layer_idx % self.scaffold_attn_every == 0):
                h_norm = self.scaffold_cross_norm(h)
                scaffold_out, _ = self.scaffold_cross_attn(
                    h_norm, scaffold_memory, scaffold_memory,
                    key_padding_mask=scaffold_padding_mask,
                )
                h = h + self.scaffold_attn_scale * scaffold_out

        # V15: Coordinate head with AFM patch cross-attention
        p_coord = self.coord_patch_proj(c_patches)  # (B, P, hidden_dim)
        h_coord_norm = self.coord_cross_norm(h)
        coord_cross_out, _ = self.coord_cross_attn(h_coord_norm, p_coord, p_coord)
        h_for_coord = h + 0.1 * coord_cross_out  # small residual
        eps_pred = self.coord_head(h_for_coord)
        eps_pred = eps_pred * mask.unsqueeze(-1)

        # V14 EDM*+γ: Type head operates on RECONSTRUCTED clean coordinates
        # Instead of using features from noisy x_t, reconstruct x_0_pred and
        # re-embed clean coords. This gives type_head near-zero effective noise.
        if diffusion_params is not None:
            sqrt_alpha = diffusion_params['sqrt_alpha']  # (B, 1, 1)
            sqrt_one_minus = diffusion_params['sqrt_one_minus']  # (B, 1, 1)
            x_0_pred = (x_t - sqrt_one_minus * eps_pred.detach()) / sqrt_alpha.clamp(min=1e-8)
            x_0_pred = x_0_pred.clamp(-3.0, 3.0) * mask.unsqueeze(-1)
            h_clean = self.coord_embed(x_0_pred)
            h_clean = h_clean + global_bias.unsqueeze(1)
            local_afm = self._sample_local_patch_features(x_0_pred, c_patches)
            h_for_type = h + h_clean + 0.2 * local_afm
        else:
            h_for_type = h

        # Type head: cross-attention + adapter
        p = self.patch_proj(c_patches)  # (B, P, hidden_dim)
        h_norm = self.type_cross_norm(h_for_type)
        cross_out, _ = self.type_cross_attn(h_norm, p, p)
        h_for_type = h_for_type + 0.1 * cross_out  # small residual
        h_for_type = h_for_type + self.type_adapter(h_for_type)  # V14: adapter

        type_logits = self.type_head(h_for_type)

        return eps_pred, type_logits

    def _apply_slot_graph_condition(
        self,
        h: torch.Tensor,
        slot_edge_types: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_pair = (mask > 0.5).float()
        valid_pair = valid_pair.unsqueeze(1) * valid_pair.unsqueeze(2)

        local_adj = (slot_edge_types == 1).float() * valid_pair
        side_adj = (slot_edge_types == 2).float() * valid_pair

        local_deg = local_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        side_deg = side_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)

        local_msg = torch.bmm(local_adj, h) / local_deg
        side_msg = torch.bmm(side_adj, h) / side_deg
        graph_update = self.slot_local_graph_proj(local_msg) + self.slot_sidechain_graph_proj(side_msg)
        return h + self.slot_graph_scale * graph_update * mask.unsqueeze(-1)


# V15: Rename for clarity (not truly SE(3)-equivariant)
SpatialDenoiser = SE3EquivariantDenoiser


class DenoiserBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, key_padding_mask):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x



class ConditionalDDPM(nn.Module):
    """V15: Conditional DDPM with denoiser, diffusion schedule, and sampling."""

    def __init__(self, denoiser: SE3EquivariantDenoiser, timesteps: int = 1000):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus = torch.sqrt(1.0 - alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus)

        # V16: Posterior buffers for compatibility with previous V16 checkpoints
        # posterior_mean_coef1: for x_0-predicting denoiser posterior mean coefficient
        posterior_mean_coef1 = torch.sqrt(1.0 - alphas_cumprod_prev) / torch.sqrt(1.0 - alphas_cumprod).clamp(min=1e-8)
        # posterior_mean_coef2: noise coefficient in posterior mean
        posterior_mean_coef2 = (torch.sqrt(1.0 - alphas_cumprod_prev) * torch.sqrt(1.0 - betas) /
                               (torch.sqrt(1.0 - alphas_cumprod).clamp(min=1e-8) * alphas_cumprod.clamp(min=1e-8).sqrt()))
        # posterior_variance: DDPM posterior variance
        posterior_variance = alphas_cumprod_prev * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod_prev).clamp(min=1e-8)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('posterior_variance', posterior_variance)

        # V13 / V16c: VDW radii from constraints.py (unified).
        # Used ONLY for repulsion/connectivity guidance, NOT for training constraints.
        self.register_buffer('vdw_radii', VDW_RADII.clone())
        # V16d.1: optional global class weights for type CE.
        self._type_class_weights = None

    def set_type_class_weights(self, weights: torch.Tensor):
        """Set global inverse-frequency class weights for type CE."""
        self._type_class_weights = weights

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def compute_loss(
        self, x_0, c_global, c_patches, atom_types, mask,
        shape_desc=None, ring_info=None,
        epoch_ratio=0.0,
        scaffold_tokens=None,
        scaffold_token_mask=None,
        slot_conditions=None,
        slot_edge_types=None,
    ):
        """Training loss with EDM*+γ."""
        B = x_0.shape[0]
        device = x_0.device
        mask_bool = mask.bool()  # dataset may return float32 mask

        if shape_desc is None:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                shape_desc = compute_shape_descriptors(x_0.float(), mask)

        t_max = min(self.timesteps - 10, 990)
        t = torch.randint(0, t_max + 1, (B,), device=device)

        x_t, noise = self.q_sample(x_0, t)

        diffusion_params = {
            'sqrt_alpha': self.sqrt_alphas_cumprod[t].view(B, 1, 1),
            'sqrt_one_minus': self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1),
        }
        eps_pred, type_logits = self.denoiser(x_t, t, c_global, c_patches, mask,
                                               shape_desc=shape_desc,
                                               diffusion_params=diffusion_params,
                                               scaffold_tokens=scaffold_tokens,
                                               scaffold_token_mask=scaffold_token_mask,
                                               slot_conditions=slot_conditions,
                                               slot_edge_types=slot_edge_types)

        # Coordinate loss (unweighted MSE on noise prediction)
        # V16 design: no z_dim_weighting, no z_axis bias
        noise_diff_sq = (eps_pred - noise) ** 2 * mask.unsqueeze(-1)
        coord_loss = noise_diff_sq.sum() / mask.sum() / 3

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        x_0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha.clamp(min=1e-8)
        x_0_pred = x_0_pred.clamp(-3.0, 3.0) * mask.unsqueeze(-1)

        # Type loss
        type_loss = torch.tensor(0.0, device=device)
        valid = mask_bool.reshape(-1)
        if valid.sum() > 0:
            atom_types_flat = atom_types.reshape(-1)[valid]
            type_logits_flat = type_logits.reshape(-1, type_logits.shape[-1])[valid]
            # Prefer global inverse-frequency weights when provided by the
            # training script; otherwise fall back to the original per-batch
            # weighting used in earlier V16 runs.
            num_types = type_logits.shape[-1]
            if self._type_class_weights is not None:
                class_weight = self._type_class_weights.to(device)
            else:
                counts = torch.bincount(atom_types_flat.long(), minlength=num_types).float()
                weights = 1.0 / (counts + 1.0)
                weights = weights / weights.sum() * num_types
                class_weight = weights.to(device)

            alpha_bar = self.alphas_cumprod[t]
            snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-6)
            snr_weight = torch.sigmoid(torch.log(snr + 1e-8))
            snr_per_atom = snr_weight.unsqueeze(1).expand_as(atom_types).reshape(-1)
            snr_valid = snr_per_atom[valid]

            per_atom_loss = F.cross_entropy(
                type_logits_flat, atom_types_flat,
                weight=class_weight, reduction='none',
            )
            type_loss = (per_atom_loss * snr_valid).mean()

        # Shape loss
        with torch.amp.autocast('cuda', enabled=False):
            shape_loss = self._compute_shape_loss(x_0.float(), eps_pred.float(), t, mask.float())

        # Ring prediction loss
        ring_loss = torch.tensor(0.0, device=device)
        if ring_info is not None:
            ring_atom_indices = ring_info["ring_atom_indices"]
            ring_valid = ring_info["ring_valid"]
            n_rings_gt = ring_info["n_rings"]

            low_t_mask = t < 500
            if low_t_mask.any():
                bond_losses = []
                count_diffs = []
                for b in range(B):
                    if not low_t_mask[b]:
                        continue
                    for r in range(ring_atom_indices.shape[1]):
                        if ring_valid[b, r] < 0.5:
                            continue
                        idx = ring_atom_indices[b, r]
                        valid_mask = idx >= 0
                        valid_idx = idx[valid_mask].long()
                        if len(valid_idx) < 5:
                            continue
                        ring_coords = x_0_pred[b, valid_idx]
                        for k in range(len(valid_idx)):
                            k_next = (k + 1) % len(valid_idx)
                            dist = (ring_coords[k] - ring_coords[k_next]).norm()
                            bond_losses.append(F.relu(dist - 0.15))
                    n_pred = self._count_rings_fast(x_0_pred[b], int(mask[b].sum().item()))
                    count_diffs.append(abs(n_pred - n_rings_gt[b].item()))

                if bond_losses:
                    ring_loss = torch.stack(bond_losses).mean()
                if count_diffs:
                    ring_loss = ring_loss + 0.1 * (sum(count_diffs) / len(count_diffs))

        # V16c: aligned with get_training_stage fractional boundaries
        stage1_end = 0.40
        stage2_end = 0.73
        if epoch_ratio <= stage1_end:
            geom_aux_weight = 0.0
            bottom_z_weight = 0.0
        elif epoch_ratio <= stage2_end:
            geom_aux_weight = 0.05
            bottom_z_weight = 0.03  # V16c: enable bottom_z in Stage 2
        else:
            geom_aux_weight = 0.08
            bottom_z_weight = 0.05

        geom_aux_loss = self._compute_geom_aux_loss(x_0_pred, x_0, mask)
        if bottom_z_weight > 0.0:
            bottom_z_loss = self._compute_bottom_z_loss(x_0_pred, x_0, mask)
        else:
            bottom_z_loss = torch.tensor(0.0, device=device)

        loss = (
            coord_loss
            + 1.0 * type_loss
            + 0.5 * shape_loss
            + 0.3 * ring_loss
            + geom_aux_weight * geom_aux_loss
            + bottom_z_weight * bottom_z_loss
        )

        return {
            "loss": loss,
            "coord_loss": coord_loss,
            "type_loss": type_loss,
            "shape_loss": shape_loss,
            "ring_loss": ring_loss,
            "geom_aux_loss": geom_aux_loss,
            "bottom_z_loss": bottom_z_loss,
            "x0_pred": x_0_pred,
            "pred_type_logits": type_logits,
        }

    def _count_rings_fast(self, coords_single, n_atoms):
        """Count 5/6-membered rings in a single molecule's coords."""
        if n_atoms < 5:
            return 0
        coords = coords_single[:n_atoms].detach()
        dists = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        adj = (dists < 0.18) & (dists > 0.01)
        adj_list = [[] for _ in range(n_atoms)]
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if adj[i, j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        seen = set()
        count = 0
        for start in range(n_atoms):
            if len(adj_list[start]) < 2:
                continue
            for nb in adj_list[start]:
                if nb <= start:
                    continue
                for target_size in [6, 5]:
                    ring = ConditionalDDPM._find_ring_dfs(adj_list, [start, nb], start, target_size, set())
                    if ring is not None:
                        canonical = tuple(sorted(ring))
                        if canonical not in seen:
                            seen.add(canonical)
                            count += 1
        return count

    def _compute_bottom_z_loss(self, pred_coords, gt_coords, mask, z_threshold_ratio=0.3):
        """Apply a weak auxiliary loss on GT-defined bottom atoms during late training."""
        losses = []
        B = pred_coords.shape[0]
        device = pred_coords.device
        for b in range(B):
            m = mask[b].bool()
            n_valid = int(m.sum().item())
            if n_valid < 2:
                continue
            pred = pred_coords[b, :n_valid]
            gt = gt_coords[b, :n_valid]
            z_min = gt[:, 2].min()
            z_max = gt[:, 2].max()
            z_range = z_max - z_min
            if z_range < 1e-6:
                continue
            z_cutoff = z_min + z_range * z_threshold_ratio
            cost = torch.cdist(pred, gt)
            row_ind, col_ind = hungarian(cost.detach().cpu().numpy())
            matched_pred = pred[row_ind]
            matched_gt = gt[col_ind]
            bottom_mask = matched_gt[:, 2] <= z_cutoff
            if bottom_mask.sum() == 0:
                continue
            losses.append(F.smooth_l1_loss(matched_pred[bottom_mask, 2], matched_gt[bottom_mask, 2]))
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _compute_geom_aux_loss(self, pred_coords, gt_coords, mask, neighbor_threshold=0.22):
        """Encourage local pairwise distances to match GT neighborhood geometry."""
        losses = []
        B = pred_coords.shape[0]
        device = pred_coords.device
        for b in range(B):
            m = mask[b].bool()
            n_valid = int(m.sum().item())
            if n_valid < 3:
                continue
            pred = pred_coords[b, :n_valid]
            gt = gt_coords[b, :n_valid]
            cost = torch.cdist(pred, gt)
            row_ind, col_ind = hungarian(cost.detach().cpu().numpy())
            matched_pred = pred[row_ind]
            matched_gt = gt[col_ind]
            gt_dists = torch.cdist(matched_gt.unsqueeze(0), matched_gt.unsqueeze(0)).squeeze(0)
            pred_dists = torch.cdist(matched_pred.unsqueeze(0), matched_pred.unsqueeze(0)).squeeze(0)
            local_mask = (gt_dists < neighbor_threshold) & (gt_dists > 0.01)
            if local_mask.sum() == 0:
                continue
            losses.append(F.smooth_l1_loss(pred_dists[local_mask], gt_dists[local_mask]))
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _compute_shape_loss(self, x_0, eps_pred, t, mask):
        B, N, _ = x_0.shape
        device = x_0.device
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1).float()
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1).float()
        noise_ratio = sqrt_one_minus_alpha / sqrt_alpha.clamp(min=1e-4)
        weight_per_sample = 1.0 / (1.0 + noise_ratio.squeeze(-1).squeeze(-1) ** 2)

        losses, weights = [], []
        for b in range(B):
            m = mask[b].bool()
            n_valid = m.sum().item()
            if n_valid < 4:
                continue
            gt = x_0[b, m]
            gt_c = gt - gt.mean(0)
            gt_cov = gt_c.T @ gt_c / n_valid
            gt_eig = torch.linalg.eigvalsh(gt_cov)
            eps_err = eps_pred[b, m]
            pred = gt - noise_ratio[b].squeeze() * eps_err
            pred_c = pred - pred.mean(0)
            pred_cov = pred_c.T @ pred_c / n_valid
            pred_eig = torch.linalg.eigvalsh(pred_cov)
            losses.append(F.mse_loss(pred_eig, gt_eig))
            weights.append(weight_per_sample[b])
        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        losses = torch.stack(losses)
        weights = torch.stack(weights)
        return (losses * weights).sum() / weights.sum().clamp(min=1e-8)

    def _apply_physics_guidance(self, x_0, type_logits, mask, guidance_step_size=0.002):
        """Three-in-one physics constraint projection during sampling.

        Uses analytical gradients instead of torch.autograd.grad() to work
        correctly in no_grad context (e.g., during model.generate/eval).

        V16c fixes:
        - Bond enforcement now uses IDEAL_BOND_LENGTHS (not VDW-based threshold)
          as the target. This fixes the "dead gradient" bug where bonded pairs
          had zero gradient because deviation was clamped to 0.
        - Bond gradient: grad = (d - L) / d * (x_i - x_j), applied to ALL
          candidate bond pairs (not just "too-long" pairs).
        """
        B, N, _ = x_0.shape
        device = x_0.device
        x_0 = x_0.clone()
        types = type_logits.argmax(dim=-1)

        # pair_mask: (B, N, N) bool, True where both i and j are valid atoms
        pair_mask = (mask.unsqueeze(2) * mask.unsqueeze(1)).bool()
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        # Pairwise distances
        dists = torch.cdist(x_0, x_0)  # (B, N, N)

        # (1) Repulsive force: analytical gradient of overlap penalty
        # VDW-based: push atoms apart when closer than sum of VDW radii.
        r_i = self.vdw_radii[types].unsqueeze(2)  # (B, N, 1)
        r_j = self.vdw_radii[types].unsqueeze(1)   # (B, 1, N)
        min_dist = r_i + r_j  # (B, N, N)
        overlap = torch.clamp_min(min_dist - dists, min=0)  # (B, N, N)
        # Analytical: d(overlap^2)/d(x_i) = sum_j 2*overlap[i,j]*(x_i-x_j)/dists[i,j]
        overlap_mask = overlap > 0  # (B, N, N) bool
        diff = x_0.unsqueeze(2) - x_0.unsqueeze(1)  # (B, N, N, 3)
        grad_repulsion = (2.0 * overlap.unsqueeze(-1) * overlap_mask.unsqueeze(-1).float() *
                          diff / dists.unsqueeze(-1).clamp(min=1e-6))
        grad_repulsion = grad_repulsion.sum(dim=2)  # (B, N, 3)
        x_0 = x_0 - 0.01 * grad_repulsion * mask.unsqueeze(-1)

        # (2) Connectivity: snap isolated atoms to nearest neighbor
        for _conn_iter in range(2):
            dists_valid = torch.cdist(x_0, x_0).clone()
            dists_valid[~pair_mask] = 1e6
            min_nn_dist, min_nn_idx = dists_valid.min(dim=-1)
            isolated = (min_nn_dist > 0.22) & mask.bool()
            if not isolated.any():
                break
            for b in range(B):
                iso = isolated[b].nonzero(as_tuple=True)[0]
                for a in iso:
                    nn = min_nn_idx[b, a]
                    direction = x_0[b, nn] - x_0[b, a]
                    dist = direction.norm() + 1e-8
                    x_0[b, a] = x_0[b, nn] - direction / dist * 0.22

        # (3) Covalent bond enforcement: pull/push ALL candidate bond pairs to ideal length
        #
        # CORRECT FORMULA (V16c fix):
        #   E_bond = sum_{(i,j) in candidates} (d_ij - L_ij)^2
        #   dE/dx_i = sum_j 2 * (d_ij - L_ij) / d_ij * (x_i - x_j)
        #
        # OLD BUG:
        #   Used VDW-based threshold instead of IDEAL_BOND_LENGTHS.
        #   Also used clamp(min=0) on deviation, making gradient 0 for bonded pairs.
        #   Grad = (d - threshold).clamp(min=0) / d * (x_i - x_j)  → always 0
        #
        # NEW:
        #   Grad = (d - ideal) / d * (x_i - x_j)  → non-zero for ALL candidate pairs

        # Move shared constants to device
        ideal_bonds = IDEAL_BOND_LENGTHS.to(device)  # (10, 10)
        max_bonds = MAX_BOND_DIST.to(device)         # (10, 10)

        # Look up ideal bond length for every (i, j) pair
        safe_types = types.clamp(min=0, max=9)  # (B, N)
        ti = safe_types.unsqueeze(2).expand(B, N, N)  # (B, N, N)
        tj = safe_types.unsqueeze(1).expand(B, N, N)  # (B, N, N)
        ideal_len = ideal_bonds[ti, tj]    # (B, N, N) ideal bond length
        max_bond = max_bonds[ti, tj]       # (B, N, N) max candidate distance

        # Candidate bonds: distance < max_bond AND ideal > 0 AND both atoms valid
        candidate_mask = (dists < max_bond) & (ideal_len > 0) & pair_mask  # (B, N, N)
        if candidate_mask.any():
            diff_vectors = x_0.unsqueeze(2) - x_0.unsqueeze(1)  # (B, N, N, 3)
            # Bond gradient: (d - L) / d * unit_vector  [per atom i, summed over j]
            bond_deviation = dists - ideal_len  # (B, N, N) — can be negative (too short) or positive (too long)
            # Avoid div-by-zero
            safe_dists = dists.clamp(min=1e-6)  # (B, N, N)
            grad_per_atom = (bond_deviation.unsqueeze(-1) * candidate_mask.unsqueeze(-1).float() *
                             diff_vectors / safe_dists.unsqueeze(-1))
            grad_bond = grad_per_atom.sum(dim=2)  # (B, N, 3)
            x_0 = x_0 - guidance_step_size * grad_bond * mask.unsqueeze(-1)

        return x_0 * mask.unsqueeze(-1)

    @staticmethod
    def _project_ring_constraints(x, ring_atom_indices, ring_templates):
        """Project atoms to standard ring templates using Procrustes alignment."""
        B, N, _ = x.shape
        x = x.clone()
        for b in range(B):
            for ri in range(ring_atom_indices.shape[1]):
                idx = ring_atom_indices[b, ri].long()
                valid = idx >= 0
                valid_idx = idx[valid]
                n = valid.sum().item()
                if n < 5 or n > 6:
                    continue
                ring_coords = x[b, valid_idx]
                ring_coords_c = ring_coords - ring_coords.mean(0)
                num_templates = ring_templates.shape[0]
                best_aligned = None
                best_dist = float('inf')
                for ti in range(num_templates):
                    # Only use templates that match ring size
                    tmpl = ring_templates[ti].to(x.device)  # (max_ring_size, 3)
                    # Count valid template atoms (non-zero rows)
                    tmpl_valid = (tmpl.abs().sum(dim=-1) > 1e-8)
                    tmpl_n = tmpl_valid.sum().item()
                    if tmpl_n != n:
                        continue  # skip size-mismatched templates
                    template = tmpl[tmpl_valid]  # (n, 3)
                    template_c = template - template.mean(0)
                    H = ring_coords_c.T @ template_c  # (3, 3)
                    U, S, Vt = torch.linalg.svd(H)
                    R = Vt.T @ U.T
                    if torch.det(R) < 0:
                        Vt[-1] *= -1
                        R = Vt.T @ U.T
                    aligned = ring_coords_c @ R.T + ring_coords.mean(0)
                    dist = ((aligned - template) ** 2).sum()
                    if dist < best_dist:
                        best_dist = dist
                        best_aligned = aligned
                if best_aligned is not None:
                    x[b, valid_idx] = best_aligned
        return x

    def _auto_detect_and_project_rings(self, x_0, types, mask):
        """Detect rings from predicted coords + project to templates."""
        B, N, _ = x_0.shape
        device = x_0.device
        x_0 = x_0.clone()
        dists = torch.cdist(x_0, x_0)
        bond_thresh = 0.18
        adj = (dists < bond_thresh) & (dists > 0.01) & mask.unsqueeze(2) & mask.unsqueeze(1)
        for b in range(B):
            m = mask[b].bool()
            n = m.sum().item()
            if n < 5:
                continue
            adj_b = adj[b, :n, :n]
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_b[i, j]:
                        adj_list[i].append(j)
                        adj_list[j].append(i)
            seen = set()
            for start in range(n):
                if len(adj_list[start]) < 2:
                    continue
                for nb in adj_list[start]:
                    if nb <= start:
                        continue
                    for target_size in [6, 5]:
                        ring = ConditionalDDPM._find_ring_dfs(adj_list, [start, nb], start, target_size, set())
                        if ring is not None:
                            canonical = tuple(sorted(ring))
                            if canonical not in seen:
                                seen.add(canonical)
                                ring_indices = torch.tensor(ring, device=device)
                                x_0[b, ring_indices] = x_0[b, ring_indices]
        return x_0

    @staticmethod
    def _find_ring_dfs(adj_list, path, target, max_len, visited_set):
        """DFS to find ring of exactly max_len atoms."""
        if len(path) == max_len:
            if target in adj_list[path[-1]]:
                return list(path)
            return None
        current = path[-1]
        for nb in adj_list[current]:
            if nb == target and len(path) >= 3:
                continue
            if nb in path[1:]:
                continue
            path.append(nb)
            result = ConditionalDDPM._find_ring_dfs(adj_list, path, target, max_len, visited_set)
            if result is not None:
                return result
            path.pop()
        return None

    def _break_illegal_rings(self, x_0, mask):
        """Detect and break 3/4-membered rings."""
        B, N, _ = x_0.shape
        x_0 = x_0.clone()
        dists = torch.cdist(x_0, x_0) + (1 - mask.unsqueeze(2)) * 1e6 + (1 - mask.unsqueeze(1)) * 1e6
        adj = (dists < 0.18) & (dists > 0.01)
        adj_list = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if adj[0, i, j]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        to_fix = set()
        for start in range(N):
            if len(adj_list[start]) == 2:
                a, b = adj_list[start]
                if a != b and b not in adj_list[a]:
                    to_fix.add(start)
                if a not in adj_list[b]:
                    to_fix.add(start)
        for i in to_fix:
            if mask[0, i] > 0:
                neighbors = adj_list[i]
                if len(neighbors) >= 2:
                    n1, n2 = neighbors[0], neighbors[1]
                    direction = x_0[0, n2] - x_0[0, n1]
                    dist = direction.norm() + 1e-8
                    x_0[0, i] = x_0[0, n1] + direction / dist * 0.15
        return x_0

    def _apply_shape_guidance(self, x_0_pred, target_shape, mask, strength=0.1):
        """Pull predicted coords toward target molecular shape descriptors."""
        B = x_0_pred.shape[0]
        device = x_0_pred.device
        with torch.amp.autocast('cuda', enabled=False):
            shape_grads = []
            for b in range(B):
                coords = x_0_pred[b, mask[b].bool()].clone().detach().requires_grad_(True)
                with torch.enable_grad():
                    pred_shape_b = compute_shape_descriptors(coords.unsqueeze(0), mask[b:b+1].bool().float())
                    diff_b = (pred_shape_b - target_shape[b:b+1].detach()).sum()
                    grad = torch.autograd.grad([diff_b], [coords])[0]
                grad = grad - grad.mean(0)
                shape_grads.append(grad)
            grad_tensor = torch.zeros_like(x_0_pred)
            for b in range(B):
                valid = mask[b].bool()
                grad_tensor[b, valid] = shape_grads[b]
            x_0_pred = x_0_pred - strength * grad_tensor * mask.unsqueeze(-1)
        return x_0_pred

    def _snap_to_ring_templates(self, x_0_pred, predicted_rings, mask, blend=0.7):
        """Project detected rings to canonical ring templates.

        Args:
            x_0_pred: (B, N, 3) predicted coordinates in normalized space
            predicted_rings: dict from RingDetectionHead.predict()
                - n_rings: (B,) int tensor
                - ring_centers: (B, 10, 3) XYZ coordinates
                - ring_types: (B, 10) type indices (0-8)
                - ring_valid: (B, 10) float mask
            mask: (B, N) atom validity mask
            blend: fraction of template position in blend (0=keep all, 1=fully snap)
        Returns:
            x_0_pred with ring atoms snapped toward template positions
        """
        from src.models.ring_detection import RING_TEMPLATES, RING_TYPE_TO_IDX

        # Reverse map: type index → ring type name
        IDX_TO_RING_TYPE = {v: k for k, v in RING_TYPE_TO_IDX.items()}

        B, N, _ = x_0_pred.shape
        device = x_0_pred.device
        x_0 = x_0_pred.clone()

        # Track which atoms have been assigned to a ring
        assigned = torch.zeros(B, N, device=device, dtype=torch.bool)

        n_rings_pred = predicted_rings.get("n_rings", None)
        centers = predicted_rings["ring_centers"]      # (B, 10, 3)
        types = predicted_rings["ring_types"]          # (B, 10)
        valid = predicted_rings["ring_valid"]          # (B, 10)

        if n_rings_pred is None:
            return x_0

        SNAP_DIST = 0.25  # max distance (normalized) to consider atom assignment

        for b in range(B):
            n_rings_b = int(n_rings_pred[b].item())
            for ri in range(n_rings_b):
                if valid[b, ri] < 0.5:
                    continue

                ring_center = centers[b, ri].cpu().numpy()
                type_idx = int(types[b, ri].item())
                ring_type_name = IDX_TO_RING_TYPE.get(type_idx, None)
                if ring_type_name is None:
                    continue

                template = RING_TEMPLATES.get(ring_type_name)
                if template is None:
                    continue

                # Translate template to predicted ring center
                n_template = template.shape[0]
                template_world = template + ring_center  # (n, 3) in model space

                # Get valid (unassigned) atoms near this ring
                n_valid = int(mask[b].sum().item())
                if n_valid == 0:
                    continue

                atom_coords = x_0[b, :n_valid].cpu().numpy()

                # Skip atoms that are already assigned to another ring
                valid_mask = (~assigned[b, :n_valid]).detach().cpu().numpy()
                if not valid_mask.any():
                    continue

                # Compute distances from each atom to each template vertex
                diff = atom_coords[valid_mask][:, None, :] - template_world[None, :, :]  # (n_free, n_tmpl, 3)
                dists = np.sqrt((diff ** 2).sum(axis=-1))  # (n_free, n_tmpl)

                # Hungarian matching over all free atoms, then map back to global indices
                free_atom_indices = np.flatnonzero(valid_mask)
                cost = dists
                row_ind, col_ind = hungarian(cost)

                # Build assignment map: template vertex j → atom i
                atom_to_vertex = {}
                for atom_i, vertex_j in zip(row_ind, col_ind):
                    if cost[atom_i, vertex_j] < SNAP_DIST:
                        atom_to_vertex[atom_i] = vertex_j

                # Blend each assigned atom toward its template vertex position
                for atom_i, vertex_j in atom_to_vertex.items():
                    atom_global_idx = int(free_atom_indices[atom_i])
                    if vertex_j >= n_template:
                        continue
                    target = template_world[vertex_j]
                    x_0[b, atom_global_idx] = (
                        (1 - blend) * x_0[b, atom_global_idx]
                        + blend * torch.tensor(target, device=device, dtype=x_0.dtype)
                    )
                    assigned[b, atom_global_idx] = True

        return x_0


    # ================================================================
    # V16c: Corrected sampling helpers
    # ================================================================

    @staticmethod
    def _build_node_mask(n_atoms: 'torch.Tensor', max_atoms: int, device) -> 'torch.Tensor':
        """Build per-sample node mask from predicted/GT atom counts."""
        B = n_atoms.shape[0]
        node_mask = torch.zeros(B, max_atoms, device=device)
        for i in range(B):
            n = min(int(n_atoms[i].item()), max_atoms)
            if n > 0:
                node_mask[i, :n] = 1.0
        return node_mask

    @staticmethod
    def _remove_mean_with_mask(x: 'torch.Tensor', node_mask: 'torch.Tensor') -> 'torch.Tensor':
        """Remove center of mass for valid atoms, keep padding at zero."""
        mask_3d = node_mask.unsqueeze(-1)  # (B, N, 1)
        N_valid = node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # (B, 1, 1)
        mean = (x * mask_3d).sum(dim=1, keepdim=True) / N_valid  # (B, 1, 3)
        return (x - mean) * mask_3d

    def _get_ddim_time_pairs(self, ddim_steps: int):
        """Construct (t, t_prev) pairs for DDIM sampling.

        Following lucidrains: linspace from -1 to T-1, reversed, paired.
        Returns list of (t, t_prev) where t_prev = -1 means final step.
        """
        times = torch.linspace(-1, self.timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        return time_pairs

    def _ddim_step(self, x_t, eps_pred, t, t_prev, eta=0.0):
        """Single DDIM reverse step: x_t -> x_{t-1}.

        Implements Song et al. 2020 Eq. 12:
          x_{t-1} = sqrt(alpha_{t-1}) * x0_pred
                  + sqrt(1 - alpha_{t-1} - sigma^2) * eps_pred
                  + sigma * noise
        """
        B = x_t.shape[0]
        device = x_t.device

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        # Predict x_0 from x_t (using current timestep t)
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alpha_t.clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-3.0, 3.0)

        # If t_prev < 0, we are at the last step: return x0_pred directly
        if t_prev < 0:
            return x0_pred, x0_pred

        # Get alpha at PREVIOUS timestep (this is the critical fix over V16b)
        alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1)
        alpha_bar_prev = self.alphas_cumprod[t_prev]  # scalar
        sqrt_alpha_prev = alpha_bar_prev.sqrt()
        sqrt_one_minus_prev = (1.0 - alpha_bar_prev).sqrt()

        # DDIM sigma (Song et al. 2020 Eq. 16)
        if eta > 0:
            sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) *
                           (1 - alpha_bar_t / alpha_bar_prev)).sqrt().view(B, 1, 1)
        else:
            sigma = 0.0

        # Direction pointing to x_t
        if isinstance(sigma, float):
            c = sqrt_one_minus_prev
        else:
            c = (1.0 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt().view(B, 1, 1)

        # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + c * eps_pred + sigma * noise
        x_prev = sqrt_alpha_prev * x0_pred + c * eps_pred
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev, x0_pred

    def _ddpm_step(self, x_t, eps_pred, t):
        """Single DDPM reverse step: x_t -> x_{t-1}.

        Implements Ho et al. 2020 standard posterior:
          mu = coef1 * x0_pred + coef2 * x_t
          x_{t-1} = mu + sqrt(posterior_var) * noise  (noise=0 when t=0)
        """
        B = x_t.shape[0]

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)

        # Predict x_0
        x0_pred = (x_t - sqrt_one_minus_t * eps_pred) / sqrt_alpha_t.clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-3.0, 3.0)

        # Standard DDPM posterior mean coefficients (Ho et al. 2020)
        alpha_bar_t = self.alphas_cumprod[t].view(B, 1, 1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].view(B, 1, 1)
        beta_t = self.betas[t].view(B, 1, 1)

        coef1 = alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8)
        coef2 = (1 - beta_t).sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)
        mu = coef1 * x0_pred + coef2 * x_t

        # Posterior variance
        posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t).clamp(min=1e-8)

        # Add noise (except at t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float().view(B, 1, 1)
        x_prev = mu + nonzero_mask * posterior_var.sqrt() * noise

        return x_prev, x0_pred

    @torch.no_grad()
    def sample(
        self,
        c_global: 'torch.Tensor',
        c_patches: 'torch.Tensor',
        n_atoms: 'torch.Tensor',
        max_atoms: int = 85,
        ring_info: dict = None,
        predicted_rings: list = None,
        use_ddim: bool = False,
        ddim_steps: int = 50,
        use_physics_guidance: bool = False,
        target_shape: 'torch.Tensor' = None,
        disable_guidance: bool = False,
        disable_ring_snap: bool = False,
        scaffold_tokens: 'torch.Tensor' = None,
        scaffold_token_mask: 'torch.Tensor' = None,
        slot_conditions: 'torch.Tensor' = None,
        slot_edge_types: 'torch.Tensor' = None,
        scaffold_constraint: dict = None,
        scaffold_constraint_time_threshold: int = 200,
        scaffold_constraint_scale: float = 0.12,
        scaffold_plane_scale: float = 0.08,
        scaffold_edge_scale: float = 0.0,
        scaffold_sidechain_edge_scale: float = 0.0,
        scaffold_post_guidance_scale: float = 0.0,
        guidance_step_size: float = 0.002,
        guidance_time_threshold: int = 500,
    ):
        """V16c: Corrected sampling with proper DDIM/DDPM formulas and node masking.

        Key fixes over V16b:
        1. DDIM uses alpha_{t-1} (not alpha_t) for reconstruction step
        2. DDPM uses standard posterior mean/variance
        3. Per-sample node_mask from n_atoms, consistent with training
        4. Initial noise masked + CoM removal every step

        Note: predicted_rings is UNUSED (V16c confirmed dead code).
        Ring snapping ONLY works with ring_info (GT ring_atom_indices + ring_templates).
        The ring_head.predict() output lacks atom indices, so it cannot drive
        _project_ring_constraints. Only Config A with use_gt_ring_info=True enables
        real ring snapping.
        """
        B = c_global.shape[0]
        device = c_global.device

        # Build per-sample node mask from predicted/GT atom counts
        node_mask = self._build_node_mask(n_atoms, max_atoms, device)  # (B, max_atoms)

        # Shape descriptor (optional)
        shape_desc = target_shape if target_shape is not None else None

        # Initialize noise, masked to valid atoms only
        x_t = torch.randn(B, max_atoms, 3, device=device)
        x_t = x_t * node_mask.unsqueeze(-1)  # zero out padding
        x_t = self._remove_mean_with_mask(x_t, node_mask)  # CoM = 0

        # Override guidance flags
        if disable_guidance:
            use_physics_guidance = False
        if disable_ring_snap:
            ring_info = None
            predicted_rings = None

        # Helper: build diffusion_params for denoiser internal x0 reconstruction
        def _make_diffusion_params(t_idx):
            return {
                'sqrt_alpha': self.sqrt_alphas_cumprod[t_idx].view(B, 1, 1),
                'sqrt_one_minus': self.sqrt_one_minus_alphas_cumprod[t_idx].view(B, 1, 1),
            }

        type_logits = None

        if use_ddim:
            # DDIM sampling with correct time pairs
            time_pairs = self._get_ddim_time_pairs(ddim_steps)

            for t_cur, t_prev in time_pairs:
                t = torch.full((B,), t_cur, device=device, dtype=torch.long)

                # Denoiser forward
                eps_pred, type_logits = self.denoiser(
                    x_t, t, c_global, c_patches,
                    mask=node_mask.bool(),
                    shape_desc=shape_desc,
                    diffusion_params=_make_diffusion_params(t),
                    scaffold_tokens=scaffold_tokens,
                    scaffold_token_mask=scaffold_token_mask,
                    slot_conditions=slot_conditions,
                    slot_edge_types=slot_edge_types)

                eps_pred = eps_pred * node_mask.unsqueeze(-1)

                # DDIM step with correct alpha_{t-1}
                x_t, x0_pred = self._ddim_step(x_t, eps_pred, t, t_prev, eta=0.0)

                # Optional physics guidance (Phase 1: disabled by default)
                if use_physics_guidance and t_cur < guidance_time_threshold:
                    x0_guided = self._apply_physics_guidance(
                        x0_pred, type_logits, node_mask,
                        guidance_step_size=guidance_step_size,
                    )
                    if t_prev >= 0:
                        alpha_prev = self.alphas_cumprod[t_prev]
                        x_t = alpha_prev.sqrt() * x0_guided + (1 - alpha_prev).sqrt() * eps_pred
                    else:
                        x_t = x0_guided

                # Optional ring constraints
                if ring_info is not None and t_cur > 200:
                    x0_snap = self._project_ring_constraints(
                        x0_pred, ring_info["ring_atom_indices"], ring_info["ring_templates"])
                    if t_prev >= 0:
                        alpha_prev = self.alphas_cumprod[t_prev]
                        x_t = alpha_prev.sqrt() * x0_snap + (1 - alpha_prev).sqrt() * eps_pred
                    else:
                        x_t = x0_snap

                # V17-Bridge: low-noise soft scaffold constraint using GT scaffold
                # labels. This is intentionally weaker than ring template snap:
                # no Procrustes alignment, only a small pull toward GT scaffold
                # atom coordinates and system planes in late sampling steps.
                if scaffold_constraint is not None and t_cur <= scaffold_constraint_time_threshold:
                    x0_soft = self._apply_soft_scaffold_constraint(
                        x0_pred, scaffold_constraint,
                        node_mask=node_mask,
                        position_scale=scaffold_constraint_scale,
                        plane_scale=scaffold_plane_scale,
                        edge_scale=scaffold_edge_scale,
                        sidechain_edge_scale=scaffold_sidechain_edge_scale,
                    )
                    if scaffold_post_guidance_scale > 0.0:
                        x0_soft = self._apply_physics_guidance(
                            x0_soft, type_logits, node_mask,
                            guidance_step_size=guidance_step_size * scaffold_post_guidance_scale,
                        )
                    if t_prev >= 0:
                        alpha_prev = self.alphas_cumprod[t_prev]
                        x_t = alpha_prev.sqrt() * x0_soft + (1 - alpha_prev).sqrt() * eps_pred
                    else:
                        x_t = x0_soft

                # Enforce mask + CoM every step
                x_t = x_t * node_mask.unsqueeze(-1)
                x_t = self._remove_mean_with_mask(x_t, node_mask)

        else:
            # DDPM sampling with correct posterior
            for t_val in reversed(range(self.timesteps)):
                t = torch.full((B,), t_val, device=device, dtype=torch.long)

                eps_pred, type_logits = self.denoiser(
                    x_t, t, c_global, c_patches,
                    mask=node_mask.bool(),
                    shape_desc=shape_desc,
                    diffusion_params=_make_diffusion_params(t),
                    scaffold_tokens=scaffold_tokens,
                    scaffold_token_mask=scaffold_token_mask,
                    slot_conditions=slot_conditions,
                    slot_edge_types=slot_edge_types)

                eps_pred = eps_pred * node_mask.unsqueeze(-1)

                x_t, x0_pred = self._ddpm_step(x_t, eps_pred, t)

                # Enforce mask + CoM every step
                x_t = x_t * node_mask.unsqueeze(-1)
                x_t = self._remove_mean_with_mask(x_t, node_mask)

        coords = x_t

        # Ensure type_logits covers max_atoms
        if type_logits is not None and type_logits.shape[1] < max_atoms:
            pad = torch.zeros(B, max_atoms - type_logits.shape[1],
                            type_logits.shape[-1], device=device)
            type_logits = torch.cat([type_logits, pad], dim=1)

        return coords, type_logits

    @staticmethod
    def _apply_soft_scaffold_constraint(
        x_0: torch.Tensor,
        scaffold_constraint: dict,
        node_mask: torch.Tensor = None,
        position_scale: float = 0.12,
        plane_scale: float = 0.08,
        edge_scale: float = 0.0,
        sidechain_edge_scale: float = 0.0,
    ) -> torch.Tensor:
        """Apply a weak GT scaffold constraint in low-noise sampling.

        This bridge constraint is deliberately softer than ring snap:
        - Pull scaffold atoms partway toward GT scaffold coordinates
        - Gently project ring-system atoms toward GT system planes
        - Restore local scaffold bond lengths on GT scaffold edges
        """
        x = x_0.clone()
        target_coords = scaffold_constraint["target_coords"].to(x.device)
        scaffold_mask = scaffold_constraint["scaffold_mask"].to(x.device).float()
        x = x + position_scale * scaffold_mask.unsqueeze(-1) * (target_coords - x)

        system_objectness = scaffold_constraint.get("system_objectness")
        system_centers = scaffold_constraint.get("system_centers")
        system_normals = scaffold_constraint.get("system_normals")
        system_atom_indices = scaffold_constraint.get("system_atom_indices")
        local_edges = scaffold_constraint.get("local_edges")
        local_edge_lengths = scaffold_constraint.get("local_edge_lengths")
        n_local_edges = scaffold_constraint.get("n_local_edges")
        sidechain_edges = scaffold_constraint.get("sidechain_edges")
        sidechain_edge_lengths = scaffold_constraint.get("sidechain_edge_lengths")
        n_sidechain_edges = scaffold_constraint.get("n_sidechain_edges")
        if (
            system_objectness is None or system_centers is None or
            system_normals is None or system_atom_indices is None
        ):
            system_objectness = None

        if node_mask is not None:
            node_mask = node_mask.to(x.device).bool()

        B = x.shape[0]
        if system_objectness is not None:
            system_objectness = system_objectness.to(x.device)
            system_centers = system_centers.to(x.device)
            system_normals = system_normals.to(x.device)
            system_atom_indices = system_atom_indices.to(x.device)

            n_systems = system_objectness.shape[1]
            for b in range(B):
                for si in range(n_systems):
                    if system_objectness[b, si] < 0.5:
                        continue
                    idx = system_atom_indices[b, si].long()
                    valid = idx >= 0
                    valid_idx = idx[valid]
                    if node_mask is not None:
                        valid_idx = valid_idx[node_mask[b, valid_idx]]
                    if valid_idx.numel() < 3:
                        continue
                    center = system_centers[b, si]
                    normal = system_normals[b, si]
                    normal_norm = normal.norm()
                    if normal_norm < 1e-6:
                        continue
                    normal = normal / normal_norm
                    coords = x[b, valid_idx]
                    projections = ((coords - center) * normal).sum(-1, keepdim=True)
                    x[b, valid_idx] = coords - plane_scale * projections * normal

        if (
            edge_scale > 0.0 and local_edges is not None and
            local_edge_lengths is not None and n_local_edges is not None
        ):
            local_edges = local_edges.to(x.device)
            local_edge_lengths = local_edge_lengths.to(x.device)
            n_local_edges = n_local_edges.to(x.device)
            for b in range(B):
                n_edges = int(n_local_edges[b].item())
                if n_edges <= 0:
                    continue
                edges_b = local_edges[b, :n_edges].long()
                lengths_b = local_edge_lengths[b, :n_edges].float()
                for ei in range(n_edges):
                    i = int(edges_b[ei, 0].item())
                    j = int(edges_b[ei, 1].item())
                    if i < 0 or j < 0:
                        continue
                    if scaffold_mask[b, i] < 0.5 or scaffold_mask[b, j] < 0.5:
                        continue
                    if node_mask is not None and (not node_mask[b, i] or not node_mask[b, j]):
                        continue
                    target_len = float(lengths_b[ei].item())
                    if target_len <= 1e-6:
                        continue
                    diff = x[b, j] - x[b, i]
                    dist = diff.norm()
                    if dist <= 1e-6:
                        continue
                    correction = 0.5 * edge_scale * (dist - target_len) / dist * diff
                    x[b, i] = x[b, i] + correction
                    x[b, j] = x[b, j] - correction

        if (
            sidechain_edge_scale > 0.0 and sidechain_edges is not None and
            sidechain_edge_lengths is not None and n_sidechain_edges is not None
        ):
            sidechain_edges = sidechain_edges.to(x.device)
            sidechain_edge_lengths = sidechain_edge_lengths.to(x.device)
            n_sidechain_edges = n_sidechain_edges.to(x.device)
            for b in range(B):
                n_edges = int(n_sidechain_edges[b].item())
                if n_edges <= 0:
                    continue
                edges_b = sidechain_edges[b, :n_edges].long()
                lengths_b = sidechain_edge_lengths[b, :n_edges].float()
                for ei in range(n_edges):
                    i = int(edges_b[ei, 0].item())
                    j = int(edges_b[ei, 1].item())
                    if i < 0 or j < 0:
                        continue
                    if node_mask is not None and (not node_mask[b, i] or not node_mask[b, j]):
                        continue
                    target_len = float(lengths_b[ei].item())
                    if target_len <= 1e-6:
                        continue
                    diff = x[b, j] - x[b, i]
                    dist = diff.norm()
                    if dist <= 1e-6:
                        continue
                    correction = 0.5 * sidechain_edge_scale * (dist - target_len) / dist * diff
                    x[b, i] = x[b, i] + correction
                    x[b, j] = x[b, j] - correction

        return x
