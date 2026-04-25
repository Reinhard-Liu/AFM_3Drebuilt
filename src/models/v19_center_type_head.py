"""
Center-conditioned atom type head for V19.

Uses:
- shared local AFM encoder features
- raw AFM height trace at the candidate atom center
- GT/predicted coordinate and local coordination context

This is the bridge from "2D center first" to "type from local evidence".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterConditionedTypeHead(nn.Module):
    def __init__(
        self,
        shared_feat_dim: int = 64,
        hidden_dim: int = 192,
        num_types: int = 10,
        coarse_lambda: float = 0.35,
        hetero_lambda: float = 0.25,
        focal_gamma: float = 1.5,
        label_smoothing: float = 0.02,
        afm_radius_px: float = 2.0,
        feat_radius_px: float = 1.0,
        center_radius_px: float = 2.0,
        afm_patch_radius_px: float = 2.0,
        afm_patch_grid_size: int = 5,
    ):
        super().__init__()
        self.num_types = num_types
        self.coarse_lambda = float(coarse_lambda)
        self.hetero_lambda = float(hetero_lambda)
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = float(label_smoothing)
        self.afm_radius_px = float(afm_radius_px)
        self.feat_radius_px = float(feat_radius_px)
        self.center_radius_px = float(center_radius_px)
        self.afm_patch_radius_px = float(afm_patch_radius_px)
        self.afm_patch_grid_size = int(max(afm_patch_grid_size, 3))

        # 0: C/H, 1: N/O/S/P, 2: halogens (F/Cl/Br/I)
        coarse_map = torch.tensor([0, 0, 1, 1, 2, 1, 1, 2, 2, 2], dtype=torch.long)
        self.register_buffer("coarse_group_map", coarse_map, persistent=False)

        # [shared(center/mean/max) + afm(center/mean/max) + center(center/mean/max) + coords + env]
        in_dim = shared_feat_dim * 3 + 10 * 3 + 3 + 3 + 5
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.coarse_head = nn.Linear(hidden_dim, 3)
        self.hetero_head = nn.Linear(hidden_dim, 1)
        patch_flat_dim = 10 * self.afm_patch_grid_size * self.afm_patch_grid_size
        self.afm_patch_mlp = nn.Sequential(
            nn.Linear(patch_flat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fine_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_types),
        )
        nn.init.zeros_(self.afm_patch_mlp[-1].weight)
        nn.init.zeros_(self.afm_patch_mlp[-1].bias)

    @staticmethod
    def _sample_map_channels(feat: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Sample (B,C,H,W) feature map at normalized coords (B,N,3) -> (B,N,C)."""
        grid = coords[..., :2].unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(feat, grid, align_corners=True, mode="bilinear")
        return sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)

    @staticmethod
    def _sample_local_stats(
        feat: torch.Tensor,
        coords: torch.Tensor,
        radius_px: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample local center/mean/max features near candidate coordinates."""
        _, _, h, w = feat.shape
        dx = 2.0 * float(radius_px) / float(max(w - 1, 1))
        dy = 2.0 * float(radius_px) / float(max(h - 1, 1))
        offsets = [
            (0.0, 0.0),
            (-dx, 0.0),
            (dx, 0.0),
            (0.0, -dy),
            (0.0, dy),
            (-dx, -dy),
            (-dx, dy),
            (dx, -dy),
            (dx, dy),
        ]

        samples = []
        for ox, oy in offsets:
            shifted = coords[..., :2].clone()
            shifted[..., 0] = shifted[..., 0] + ox
            shifted[..., 1] = shifted[..., 1] + oy
            shifted = shifted.clamp(-1.0, 1.0)
            sampled = F.grid_sample(
                feat,
                shifted.unsqueeze(2),
                align_corners=True,
                mode="bilinear",
            ).squeeze(-1).transpose(1, 2)
            samples.append(sampled)

        stack = torch.stack(samples, dim=2)  # (B, N, K, C)
        center = stack[:, :, 0, :]
        mean = stack.mean(dim=2)
        maxv = stack.max(dim=2).values
        return center, mean, maxv

    @staticmethod
    def _sample_local_patch(
        feat: torch.Tensor,
        coords: torch.Tensor,
        radius_px: float,
        grid_size: int,
    ) -> torch.Tensor:
        """Sample a dense local patch around candidate coordinates.

        Returns: (B, N, C * grid_size * grid_size)
        """
        _, _, h, w = feat.shape
        dx = 2.0 * float(radius_px) / float(max(w - 1, 1))
        dy = 2.0 * float(radius_px) / float(max(h - 1, 1))
        offsets_x = torch.linspace(-dx, dx, steps=grid_size, device=coords.device, dtype=coords.dtype)
        offsets_y = torch.linspace(-dy, dy, steps=grid_size, device=coords.device, dtype=coords.dtype)

        patches = []
        for oy in offsets_y:
            for ox in offsets_x:
                shifted = coords[..., :2].clone()
                shifted[..., 0] = shifted[..., 0] + ox
                shifted[..., 1] = shifted[..., 1] + oy
                shifted = shifted.clamp(-1.0, 1.0)
                sampled = F.grid_sample(
                    feat,
                    shifted.unsqueeze(2),
                    align_corners=True,
                    mode="bilinear",
                ).squeeze(-1).transpose(1, 2)  # (B,N,C)
                patches.append(sampled)
        patch_stack = torch.stack(patches, dim=2)  # (B,N,K,C)
        return patch_stack.reshape(patch_stack.shape[0], patch_stack.shape[1], -1)

    @staticmethod
    def _compute_env_features(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape
        device = coords.device

        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = diff.norm(dim=-1)
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        eye = torch.eye(N, device=device).unsqueeze(0)
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

    def _focal_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(
            log_probs,
            targets,
            weight=class_weight,
            reduction="none",
        )
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        focal = (1.0 - pt).pow(self.focal_gamma)
        return (focal * ce).mean()

    def forward(
        self,
        coords: torch.Tensor,
        shared_feat: torch.Tensor,
        afm_stack: torch.Tensor,
        mask: torch.Tensor,
        center_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_center, shared_mean, shared_max = self._sample_local_stats(shared_feat, coords, self.feat_radius_px)
        afm_center, afm_mean, afm_max = self._sample_local_stats(afm_stack, coords, self.afm_radius_px)
        afm_patch = self._sample_local_patch(afm_stack, coords, self.afm_patch_radius_px, self.afm_patch_grid_size)
        patch_embed = self.afm_patch_mlp(afm_patch)
        env = self._compute_env_features(coords, mask)
        if center_map is None:
            center_stats = torch.zeros(coords.shape[0], coords.shape[1], 3, device=coords.device, dtype=coords.dtype)
        else:
            center_center, center_mean, center_max = self._sample_local_stats(center_map, coords, self.center_radius_px)
            center_stats = torch.cat([center_center, center_mean, center_max], dim=-1)

        feat = torch.cat(
            [
                shared_center,
                shared_mean,
                shared_max,
                afm_center,
                afm_mean,
                afm_max,
                center_stats,
                coords,
                env,
            ],
            dim=-1,
        )
        trunk = self.trunk(feat) + patch_embed
        coarse_logits = self.coarse_head(trunk)
        hetero_logits = self.hetero_head(trunk)
        coarse_prob = F.softmax(coarse_logits, dim=-1)
        hetero_prob = torch.sigmoid(hetero_logits)
        fine_feat = torch.cat([trunk, coarse_prob, hetero_prob], dim=-1)
        fine_logits = self.fine_head(fine_feat)
        return fine_logits, coarse_logits, hetero_logits.squeeze(-1)

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
        logits, coarse_logits, hetero_logits = self.forward(
            coords,
            shared_feat,
            afm_stack,
            mask,
            center_map=center_map,
        )

        logits_flat = logits.reshape(-1, self.num_types)
        coarse_logits_flat = coarse_logits.reshape(-1, 3)
        hetero_logits_flat = hetero_logits.reshape(-1)
        types_flat = atom_types.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = (mask_flat > 0) & (types_flat >= 0)

        if valid.sum() == 0:
            loss = torch.tensor(0.0, device=coords.device)
        else:
            valid_types = types_flat[valid]
            fine_loss = self._focal_cross_entropy(logits_flat[valid], valid_types, class_weight=class_weight)

            coarse_targets = self.coarse_group_map[valid_types]
            coarse_loss = F.cross_entropy(coarse_logits_flat[valid], coarse_targets)

            hetero_targets = (~torch.isin(valid_types, torch.tensor([0, 1], device=valid_types.device))).float()
            pos = hetero_targets.sum().item()
            neg = hetero_targets.numel() - pos
            pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0), device=coords.device)
            hetero_loss = F.binary_cross_entropy_with_logits(
                hetero_logits_flat[valid],
                hetero_targets,
                pos_weight=pos_weight,
            )

            loss = fine_loss + self.coarse_lambda * coarse_loss + self.hetero_lambda * hetero_loss
        return loss, logits
