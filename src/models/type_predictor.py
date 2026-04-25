"""
TypePredictor: Independent atom type prediction with noise-robust training.

Key difference from V6 TypeNet:
- Trains on coords + random noise (simulating inference-time RMSD error)
- This eliminates the exposure bias that killed V6 TypeNet
- Uses AFM patch cross-attention for element-specific spatial info (from V8)
- Independent parameters, does not share transformer with denoiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypePredictor(nn.Module):
    """Predicts atom types from (possibly noisy) coordinates + AFM patches.

    Trained with random noise injection to match inference-time coord quality.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        hidden_dim: int = 256,
        num_types: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        noise_std_range: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_types = num_types
        self.noise_std_range = noise_std_range  # max noise std during training

        # Coordinate encoder
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # AFM patch cross-attention (element-specific spatial info)
        self.patch_proj = nn.Linear(cond_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Independent transformer layers (NOT shared with denoiser)
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

    def forward(
        self,
        coords: torch.Tensor,
        c_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) atom coordinates (clean or noisy)
            c_patches: (B, P, cond_dim) AFM patch features from ViT
            mask: (B, N) atom mask

        Returns:
            type_logits: (B, N, num_types)
        """
        # Encode coordinates
        h = self.coord_encoder(coords)  # (B, N, hidden_dim)

        # Cross-attention to AFM patches
        p = self.patch_proj(c_patches)  # (B, P, hidden_dim)
        h_norm = self.cross_norm(h)
        cross_out, _ = self.cross_attn(h_norm, p, p)
        h = h + 0.3 * cross_out  # stronger residual than denoiser's 0.1

        # Transformer layers
        src_key_padding_mask = (mask == 0)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)

        # Predict types
        type_logits = self.type_head(h)
        return type_logits

    def compute_loss(
        self,
        gt_coords: torch.Tensor,
        c_patches: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Compute type prediction loss with noise-robust training.

        During training, adds random noise to GT coords to simulate
        the RMSD error present in inference-time generated coordinates.
        This eliminates the exposure bias that killed V6 TypeNet.

        Args:
            gt_coords: (B, N, 3) ground truth coordinates
            c_patches: (B, P, cond_dim) AFM patch features
            atom_types: (B, N) ground truth types
            mask: (B, N) atom mask
            training: if True, add noise to coords

        Returns:
            type_loss: scalar
        """
        B, N, _ = gt_coords.shape
        device = gt_coords.device

        # Add random noise during training to simulate inference coord error
        if training and self.noise_std_range > 0:
            # Random noise std per sample: uniform in [0, noise_std_range]
            noise_std = torch.rand(B, 1, 1, device=device) * self.noise_std_range
            noise = torch.randn_like(gt_coords) * noise_std
            coords = gt_coords + noise * mask.unsqueeze(-1)
        else:
            coords = gt_coords

        # Predict types
        type_logits = self.forward(coords, c_patches, mask)

        # CE loss with sqrt(inv_freq) class weights
        logits_flat = type_logits.reshape(-1, self.num_types)
        types_flat = atom_types.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = (mask_flat > 0) & (types_flat >= 0)

        if valid.sum() > 0:
            valid_types = types_flat[valid]
            counts = torch.bincount(valid_types, minlength=self.num_types).float().clamp(min=1.0)
            inv_freq = valid_types.numel() / (self.num_types * counts)
            class_weight = torch.sqrt(inv_freq).clamp(max=3.0).to(device)
            type_loss = F.cross_entropy(logits_flat[valid], valid_types, weight=class_weight)
        else:
            type_loss = torch.tensor(0.0, device=device)

        return type_loss
