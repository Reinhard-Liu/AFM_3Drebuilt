"""
Video Vision Transformer (Video ViT) Encoder for AFM image stacks.

Treats the D=10 AFM slices as video frames and extracts a global
condition vector c that encodes 3D structural information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbedding3D(nn.Module):
    """Embed AFM stack (D, H, W) into a sequence of patch tokens.

    Spatial patches: (patch_size, patch_size)
    Temporal patches: (temporal_patch_size,)
    """

    def __init__(
        self,
        img_size: int = 128,
        num_frames: int = 10,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 1,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch_size
        self.num_patches = self.num_spatial_patches * self.num_temporal_patches

        # 3D convolution for tubelet embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            stride=(temporal_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W) AFM stack

        Returns:
            tokens: (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        # Add channel dim: (B, 1, D, H, W)
        x = x.unsqueeze(1)
        # Tubelet embedding: (B, embed_dim, T', H', W')
        x = self.proj(x)
        # Flatten spatial and temporal: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class VideoViTEncoder(nn.Module):
    """Video Vision Transformer encoder for AFM image stacks.

    Outputs a condition vector c of shape (B, embed_dim).
    """

    def __init__(
        self,
        img_size: int = 128,
        num_frames: int = 10,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=1,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, D=10, H, W) AFM image stack

        Returns:
            c_global: (B, embed_dim) CLS token condition vector
            c_patches: (B, num_patches, embed_dim) patch-level features
        """
        B = x.shape[0]

        # Patch embedding
        tokens = self.patch_embed(x)  # (B, N, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, embed_dim)

        # Add positional embedding
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # Transformer blocks
        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)

        c_global = tokens[:, 0]    # (B, embed_dim)
        c_patches = tokens[:, 1:]  # (B, num_patches, embed_dim)
        return c_global, c_patches


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x
