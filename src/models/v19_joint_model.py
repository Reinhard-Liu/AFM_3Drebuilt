"""
V19 joint model:
shared AFM encoder -> dedicated center decoder + 2D structure decoder + z decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.dataset import MAX_ATOMS


class V19JointUNet(nn.Module):
    """Joint AFM-to-2D+z model with dedicated center, structure, and z decoders.

    The goal is to keep 2D type/structure learning from being dragged down by
    the z objective too early. The center branch is also separated so object-
    level heads no longer depend only on the dense occupancy map.
    """

    def __init__(self, in_channels: int = 10, base_ch: int = 64, max_objects: int = MAX_ATOMS):
        super().__init__()
        self.in_channels = in_channels
        self.base_ch = base_ch
        self.max_objects = max_objects

        # Shared encoder
        self.enc1 = self._block(in_channels, base_ch, normalize=False)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.enc4 = self._block(base_ch * 4, base_ch * 8)
        self.bottleneck = self._block(base_ch * 8, base_ch * 8)

        # Dedicated atom-center decoder
        self.up4_center = self._up_block(base_ch * 8, base_ch * 4)
        self.fuse4_center = self._fuse_block(base_ch * 12, base_ch * 4)
        self.up3_center = self._up_block(base_ch * 4, base_ch * 2)
        self.fuse3_center = self._fuse_block(base_ch * 6, base_ch * 2)
        self.up2_center = self._up_block(base_ch * 2, base_ch)
        self.fuse2_center = self._fuse_block(base_ch * 3, base_ch)
        self.up1_center = self._up_block(base_ch, base_ch // 2)
        self.fuse1_center = self._fuse_block(base_ch + base_ch // 2, base_ch // 2)
        self.final_center = nn.ConvTranspose2d(base_ch // 2, 1, 4, 2, 1)

        # 2D structure decoder
        self.up4_2d = self._up_block(base_ch * 8, base_ch * 4)
        self.fuse4_2d = self._fuse_block(base_ch * 12, base_ch * 4)
        self.up3_2d = self._up_block(base_ch * 4, base_ch * 2)
        self.fuse3_2d = self._fuse_block(base_ch * 6, base_ch * 2)
        self.up2_2d = self._up_block(base_ch * 2, base_ch)
        self.fuse2_2d = self._fuse_block(base_ch * 3, base_ch)
        self.up1_2d = self._up_block(base_ch, base_ch // 2)
        self.fuse1_2d = self._fuse_block(base_ch + base_ch // 2, base_ch // 2)
        self.final_2d = nn.Sequential(
            nn.ConvTranspose2d(base_ch // 2, 12, 4, 2, 1),
            nn.Tanh(),
        )

        # Dedicated z decoder
        self.up4_z = self._up_block(base_ch * 8, base_ch * 4)
        self.fuse4_z = self._fuse_block(base_ch * 12, base_ch * 4)
        self.up3_z = self._up_block(base_ch * 4, base_ch * 2)
        self.fuse3_z = self._fuse_block(base_ch * 6, base_ch * 2)
        self.up2_z = self._up_block(base_ch * 2, base_ch)
        self.fuse2_z = self._fuse_block(base_ch * 3, base_ch)
        self.up1_z = self._up_block(base_ch, base_ch // 2)
        self.fuse1_z = self._fuse_block(base_ch + base_ch // 2, base_ch // 2)
        self.final_z = nn.Sequential(
            nn.ConvTranspose2d(base_ch // 2, 1, 4, 2, 1),
            nn.Tanh(),
        )

        # Count head for count-conditioned object proposal.
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.GELU(),
            nn.LayerNorm(base_ch * 4),
            nn.Linear(base_ch * 4, max_objects + 1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, normalize: bool = True) -> nn.Sequential:
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _fuse_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _decode_center(self, b: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor) -> torch.Tensor:
        d4 = self.fuse4_center(torch.cat([self.up4_center(b), e4], dim=1))
        d3 = self.fuse3_center(torch.cat([self.up3_center(d4), e3], dim=1))
        d2 = self.fuse2_center(torch.cat([self.up2_center(d3), e2], dim=1))
        d1 = self.fuse1_center(torch.cat([self.up1_center(d2), e1], dim=1))
        return self.final_center(d1)

    def _decode_2d(self, b: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor) -> torch.Tensor:
        d4 = self.fuse4_2d(torch.cat([self.up4_2d(b), e4], dim=1))
        d3 = self.fuse3_2d(torch.cat([self.up3_2d(d4), e3], dim=1))
        d2 = self.fuse2_2d(torch.cat([self.up2_2d(d3), e2], dim=1))
        d1 = self.fuse1_2d(torch.cat([self.up1_2d(d2), e1], dim=1))
        return self.final_2d(d1)

    def _decode_z(self, b: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor) -> torch.Tensor:
        d4 = self.fuse4_z(torch.cat([self.up4_z(b), e4], dim=1))
        d3 = self.fuse3_z(torch.cat([self.up3_z(d4), e3], dim=1))
        d2 = self.fuse2_z(torch.cat([self.up2_z(d3), e2], dim=1))
        d1 = self.fuse1_z(torch.cat([self.up1_z(d2), e1], dim=1))
        return self.final_z(d1)

    def forward_with_features(self, afm_stack: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        e1 = self.enc1(afm_stack)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        center_logits = self._decode_center(b, e1, e2, e3, e4)
        pred_2d = self._decode_2d(b, e1, e2, e3, e4)
        pred_z = self._decode_z(b, e1, e2, e3, e4)
        count_logits = self.count_head(b)
        pred = torch.cat([pred_2d, pred_z], dim=1)
        features = {
            "enc1": e1,
            "enc2": e2,
            "enc3": e3,
            "enc4": e4,
            "bottleneck": b,
            "center_logits": center_logits,
            "count_logits": count_logits,
        }
        return pred, features

    def forward(self, afm_stack: torch.Tensor) -> torch.Tensor:
        pred, _ = self.forward_with_features(afm_stack)
        return pred

    @staticmethod
    def split_outputs(pred_01: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "atom_map": pred_01[:, 0:1],
            "bond_map": pred_01[:, 1:2],
            "type_maps": pred_01[:, 2:12],
            "z_map": pred_01[:, 12:13],
        }
