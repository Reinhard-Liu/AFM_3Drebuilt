"""
Baseline models for comparison:
1. 3D-ResNet Regression: directly regresses atom coordinates from AFM stacks
2. Image-to-Image cGAN: generates 2D projected molecular images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Baseline 1: 3D-ResNet Regression
# ============================================================

class ResBlock3D(nn.Module):
    """3D residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet3DRegression(nn.Module):
    """3D-ResNet that directly regresses atom coordinates from AFM stack.

    Input: (B, D=10, H, W) AFM image stack
    Output: (B, max_atoms, 3) predicted coordinates
            (B, max_atoms, num_atom_types) atom type logits
    """

    def __init__(
        self,
        img_size: int = 128,
        num_frames: int = 10,
        max_atoms: int = 85,
        num_atom_types: int = 10,
        base_ch: int = 32,
    ):
        super().__init__()
        self.max_atoms = max_atoms

        self.encoder = nn.Sequential(
            # (B, 1, D=10, H, W)
            nn.Conv3d(1, base_ch, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            ResBlock3D(base_ch, base_ch * 2, stride=2),
            ResBlock3D(base_ch * 2, base_ch * 2),
            ResBlock3D(base_ch * 2, base_ch * 4, stride=2),
            ResBlock3D(base_ch * 4, base_ch * 4),
            ResBlock3D(base_ch * 4, base_ch * 8, stride=2),
            ResBlock3D(base_ch * 8, base_ch * 8),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        feat_dim = base_ch * 8
        self.coord_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, max_atoms * 3),
        )
        self.type_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, max_atoms * num_atom_types),
        )
        self.num_atom_types = num_atom_types

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, D=10, H, W) AFM stack

        Returns:
            coords: (B, max_atoms, 3)
            type_logits: (B, max_atoms, num_atom_types)
        """
        B = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, D, H, W)
        feat = self.encoder(x)
        feat = self.pool(feat).flatten(1)  # (B, feat_dim)

        coords = self.coord_head(feat).reshape(B, self.max_atoms, 3)
        type_logits = self.type_head(feat).reshape(B, self.max_atoms, self.num_atom_types)

        return coords, type_logits

    def compute_loss(
        self,
        x: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        coords_pred, type_logits = self.forward(x)

        # Coord MSE loss (masked)
        coord_loss = F.mse_loss(
            coords_pred * mask.unsqueeze(-1),
            gt_coords * mask.unsqueeze(-1),
        )

        # Type CE loss (masked)
        type_flat = type_logits.reshape(-1, self.num_atom_types)
        gt_flat = gt_types.reshape(-1)
        mask_flat = mask.reshape(-1)
        valid = (mask_flat > 0) & (gt_flat >= 0)

        if valid.sum() > 0:
            type_loss = F.cross_entropy(type_flat[valid], gt_flat[valid])
        else:
            type_loss = torch.tensor(0.0, device=x.device)

        loss = coord_loss + 0.1 * type_loss
        return {"loss": loss, "coord_loss": coord_loss, "type_loss": type_loss}

    @torch.no_grad()
    def generate(self, batch: dict, use_gt_count: bool = False,
                 use_ddim: bool = False, ddim_steps: int = 100) -> dict:
        """Generate molecular structure from AFM stack (for evaluation).

        Args:
            batch: dict with 'afm_stack' key
            use_gt_count: ignored (for API compatibility with diffusion model)

        Returns:
            dict with keys:
                coords: (B, max_atoms, 3)
                type_logits: (B, max_atoms, num_atom_types)
                n_atoms_pred: (B,) tensor of predicted atom counts (uses ground truth if available)
        """
        coords, type_logits = self.forward(batch["afm_stack"])

        # For ResNet3D, we don't predict atom count separately
        # Use ground truth if available, otherwise infer from type logits
        if use_gt_count and "n_atoms" in batch:
            n_atoms_pred = batch["n_atoms"]
        else:
            # Infer atom count from type predictions (count non-background predictions)
            # This is a heuristic: count atoms with predicted type != background
            pred_types = type_logits.argmax(dim=-1)
            # Assume background is indicated by low confidence across all types
            max_probs = type_logits.softmax(dim=-1).max(dim=-1)[0]
            n_atoms_pred = (max_probs > 0.5).sum(dim=-1)

        return {
            "coords": coords,
            "type_logits": type_logits,
            "n_atoms_pred": n_atoms_pred,
        }


# ============================================================
# Baseline 2: cGAN (simplified pix2pix-style)
# ============================================================

class UNetGenerator(nn.Module):
    """Simplified U-Net generator for image-to-image translation.

    Takes a multi-channel (D=10) AFM input and generates a 2D molecular
    projection image (ball-and-stick style).
    """

    def __init__(self, in_channels: int = 10, out_channels: int = 3, base_ch: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = self._block(in_channels, base_ch, normalize=False)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.enc4 = self._block(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.bottleneck = self._block(base_ch * 8, base_ch * 8)

        # Decoder: upsample -> concatenate skip -> fuse
        self.up4 = self._up_block(base_ch * 8, base_ch * 4)
        self.fuse4 = self._fuse_block(base_ch * 12, base_ch * 4)

        self.up3 = self._up_block(base_ch * 4, base_ch * 2)
        self.fuse3 = self._fuse_block(base_ch * 6, base_ch * 2)

        self.up2 = self._up_block(base_ch * 2, base_ch)
        self.fuse2 = self._fuse_block(base_ch * 3, base_ch)

        self.up1 = self._up_block(base_ch, base_ch // 2)
        self.fuse1 = self._fuse_block(base_ch + base_ch // 2, base_ch // 2)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_ch // 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_ch, out_ch, normalize=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _fuse_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d4 = self.fuse4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.fuse3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.fuse2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.fuse1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator."""

    def __init__(self, in_channels: int = 13, base_ch: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, cond], dim=1))
