"""
预测头模块：原子数预测 + 分子对比检索 + 环检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from src.models.ring_detection import (
    MAX_RING_SYSTEMS,
    MAX_SYSTEM_SIDECHAIN_EDGES,
    ATOM_ROLE_TO_IDX,
)


class AtomCountHead(nn.Module):
    """原子数预测头：共享特征层 + 残差连接 + 分类/回归双分支

    架构改进：
    - 共享特征提取层 + 残差跳接（改善梯度流）
    - 分类分支有效 4 层深（shared 1 + cls 3）
    - Label Smoothing 防止相邻原子数过度自信
    - 推理时融合分类 argmax 和回归值
    """

    def __init__(self, embed_dim: int = 512, max_count: int = 85):
        super().__init__()
        self.max_count = max_count

        # 共享特征提取 + 残差跳接
        self.shared_mlp = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.res_proj = nn.Linear(embed_dim, 256)  # 维度对齐的跳接
        self.shared_norm = nn.LayerNorm(256)

        # 分类分支：3 层（共享层之后），总深度 4
        self.cls_branch = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, max_count),
        )

        # 回归分支：2 层（共享层之后），总深度 3
        self.reg_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def _shared_features(self, c: torch.Tensor) -> torch.Tensor:
        """提取共享特征，带残差连接。"""
        h = self.shared_mlp(c) + self.res_proj(c)
        return self.shared_norm(h)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        Args:
            c: ViT CLS token, shape (B, embed_dim)
        Returns:
            cls_logits: (B, max_count) 分类 logits
            reg_value: (B,) 回归预测值
        """
        h = self._shared_features(c)
        cls_logits = self.cls_branch(h)
        reg_value = self.reg_branch(h).squeeze(-1)
        return cls_logits, reg_value

    def predict(self, c: torch.Tensor) -> torch.Tensor:
        """推理时预测原子数：融合分类和回归结果"""
        cls_logits, reg_value = self.forward(c)
        cls_pred = cls_logits.argmax(dim=-1) + 1  # 1-indexed
        # 加权融合：分类结果为主，回归值做微调
        fused = 0.7 * cls_pred.float() + 0.3 * reg_value.clamp(1, self.max_count)
        return fused.round().long().clamp(1, self.max_count)

    def compute_loss(self, c: torch.Tensor, n_atoms: torch.Tensor) -> dict:
        """
        Args:
            c: (B, embed_dim)
            n_atoms: (B,) ground truth 原子数, 1-indexed
        Returns:
            dict with 'count_loss', 'cls_loss', 'reg_loss'
        """
        cls_logits, reg_value = self.forward(c)

        # 分类损失：label_smoothing=0.1 防止相邻原子数过度自信
        cls_target = (n_atoms - 1).long().clamp(0, self.max_count - 1)
        cls_loss = F.cross_entropy(cls_logits, cls_target, label_smoothing=0.1)

        # 回归损失：MSE比smooth_l1对大偏差更敏感
        reg_loss = F.mse_loss(reg_value, n_atoms.float())

        count_loss = cls_loss + 1.0 * reg_loss

        return {
            "count_loss": count_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
        }


class ScaffoldStructureCountHead(nn.Module):
    """结构规模预测头。

    目标不是直接生成坐标，而是先从全局特征中预测：
    - 有多少个环系骨架
    - 骨架里有多少原子
    - 骨架外还有多少原子
    - 有多少个骨架连接点
    - 有多少条骨架向外延伸的局部连接
    """

    TARGET_SPECS = {
        "ring_system_count": MAX_RING_SYSTEMS,
        "scaffold_atom_count": 85,
        "non_scaffold_atom_count": 85,
        "attachment_anchor_count": 85,
        "sidechain_edge_count": MAX_SYSTEM_SIDECHAIN_EDGES,
    }

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.res_proj = nn.Linear(embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, len(self.TARGET_SPECS))

    def _shared_features(self, c: torch.Tensor) -> torch.Tensor:
        h = self.shared(c) + self.res_proj(c)
        return self.norm(h)

    def forward(self, c: torch.Tensor) -> dict:
        h = self._shared_features(c)
        raw = F.softplus(self.out(h))
        outputs = {}
        for idx, (name, max_value) in enumerate(self.TARGET_SPECS.items()):
            outputs[name] = raw[:, idx].clamp(0.0, float(max_value))
        return outputs

    def predict(self, c: torch.Tensor) -> dict:
        pred = self.forward(c)
        out = {}
        for name, max_value in self.TARGET_SPECS.items():
            out[name] = pred[name].round().long().clamp(0, max_value)
        return out

    def compute_loss_from_pred(self, pred: dict, targets: dict) -> dict:
        losses = {}
        total = None
        for name in self.TARGET_SPECS:
            target = targets[name].float()
            loss = F.smooth_l1_loss(pred[name], target)
            losses[f"{name}_loss"] = loss
            total = loss if total is None else total + loss
        losses["structure_count_aux_loss"] = total
        return losses

    def compute_loss(self, c: torch.Tensor, targets: dict) -> dict:
        pred = self.forward(c)
        return self.compute_loss_from_pred(pred, targets)


class MoleculeRetrievalHead(nn.Module):
    """分子对比检索头：InfoNCE 对比学习 + 余弦相似度检索"""

    def __init__(self, embed_dim: int = 512, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        # 投影网络：将 ViT 特征映射到检索空间
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, proj_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: ViT CLS token, shape (B, embed_dim)
        Returns:
            proj: L2 归一化的投影向量, shape (B, proj_dim)
        """
        proj = self.projector(c)
        proj = F.normalize(proj, p=2, dim=-1)
        return proj

    def compute_loss(
        self,
        c: torch.Tensor,
        cid_idx: torch.Tensor,
        mol_embeddings: nn.Embedding,
    ) -> dict:
        """
        InfoNCE 对比损失

        Args:
            c: (B, embed_dim)
            cid_idx: (B,) ground truth CID 索引
            mol_embeddings: nn.Embedding(num_cids, proj_dim) 分子嵌入库
        Returns:
            dict with 'retrieval_loss'
        """
        query = self.forward(c)  # (B, proj_dim)

        # 归一化分子嵌入
        keys = F.normalize(mol_embeddings.weight, p=2, dim=-1)  # (num_cids, proj_dim)

        # 余弦相似度 / temperature
        sim = torch.mm(query, keys.t()) / self.temperature  # (B, num_cids)

        retrieval_loss = F.cross_entropy(sim, cid_idx)

        return {"retrieval_loss": retrieval_loss}

    @torch.no_grad()
    def retrieve(
        self,
        c: torch.Tensor,
        mol_embeddings: nn.Embedding,
        top_k: int = 5,
    ) -> tuple:
        """
        推理时检索 Top-K 最相似分子

        Args:
            c: (B, embed_dim)
            mol_embeddings: 分子嵌入库
            top_k: 返回的候选数
        Returns:
            scores: (B, top_k) 相似度分数
            indices: (B, top_k) CID 索引
        """
        query = self.forward(c)
        keys = F.normalize(mol_embeddings.weight, p=2, dim=-1)
        sim = torch.mm(query, keys.t())  # (B, num_cids)
        scores, indices = sim.topk(top_k, dim=-1)
        return scores, indices


class RingDetectionHead(nn.Module):
    """V16: 从 AFM 特征预测环数量、中心位置、类型。

    三分支架构：
    - 环数量：c_global → 分类(0-10) + 回归
    - 环中心：learned queries cross-attend to c_patches 空间特征 → XY 回归
    - 环类型：复用 query 特征 → 9 类分类
    """

    MAX_RINGS = 10
    NUM_RING_TYPES = 9  # benzene, pyridine, ..., other_6

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256,
                 max_rings: int = 10, num_ring_types: int = 9):
        super().__init__()
        self.max_rings = max_rings
        self.num_ring_types = num_ring_types

        # --- 环数量分支（从 c_global）---
        self.count_shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.count_res = nn.Linear(embed_dim, hidden_dim)
        self.count_norm = nn.LayerNorm(hidden_dim)
        self.count_cls = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, max_rings + 1))
        self.count_reg = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))

        # --- 环中心+类型分支（从 c_patches 空间特征）---
        self.patch_proj = nn.Linear(embed_dim, hidden_dim)  # 投影 c_patches
        self.ring_queries = nn.Parameter(torch.randn(max_rings, hidden_dim) * 0.02)
        self.center_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.center_norm = nn.LayerNorm(hidden_dim)
        # XY 回归（Z 固定为 0）
        self.center_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 2))
        # 类型分类
        self.type_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, num_ring_types))

    def _pool_spatial(self, c_patches):
        """将 (B, 320, D) 的时空 patch 池化为 (B, 64, D) 的空间特征。
        320 = 64 spatial × 5 temporal, 对 temporal 求均值。"""
        B, P, D = c_patches.shape
        # reshape: (B, 5, 64, D) → mean over temporal → (B, 64, D)
        spatial = c_patches.view(B, 5, 64, D).mean(dim=1)
        return spatial

    def forward(self, c_global, c_patches):
        """
        Args:
            c_global: (B, embed_dim)
            c_patches: (B, 320, embed_dim)
        Returns:
            count_logits: (B, max_rings+1)
            count_reg: (B,)
            centers: (B, max_rings, 2)  XY coordinates
            type_logits: (B, max_rings, num_ring_types)
        """
        B = c_global.shape[0]

        # 环数量
        h_count = self.count_shared(c_global) + self.count_res(c_global)
        h_count = self.count_norm(h_count)
        count_logits = self.count_cls(h_count)
        count_reg = self.count_reg(h_count).squeeze(-1)

        # 环中心 + 类型：queries cross-attend to spatial patches
        spatial = self._pool_spatial(c_patches)  # (B, 64, embed_dim)
        kv = self.patch_proj(spatial)  # (B, 64, hidden_dim)
        queries = self.ring_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 10, hidden_dim)
        q_norm = self.center_norm(queries)
        attended, _ = self.center_cross_attn(q_norm, kv, kv)  # (B, 10, hidden_dim)
        ring_features = queries + attended  # residual

        centers = self.center_mlp(ring_features)  # (B, 10, 2)
        type_logits = self.type_mlp(ring_features)  # (B, 10, 9)

        return count_logits, count_reg, centers, type_logits

    def compute_loss(self, c_global, c_patches,
                     gt_centers, gt_types, gt_valid, gt_n_rings):
        """
        Args:
            gt_centers: (B, 10, 3) — 使用前 2 维 XY
            gt_types: (B, 10) int
            gt_valid: (B, 10) float
            gt_n_rings: (B,) int
        """
        count_logits, count_reg, pred_centers, type_logits = self.forward(c_global, c_patches)
        device = c_global.device

        # 1. 环数量 loss
        count_cls_loss = F.cross_entropy(count_logits, gt_n_rings.long().clamp(0, self.max_rings))
        count_reg_loss = F.mse_loss(count_reg, gt_n_rings.float())
        count_loss = count_cls_loss + count_reg_loss

        # 2. 环中心 + 类型 loss（需要匈牙利匹配）
        gt_xy = gt_centers[:, :, :2]  # (B, 10, 2)
        center_losses = []
        type_losses = []

        for b in range(c_global.shape[0]):
            n_valid = int(gt_valid[b].sum().item())
            if n_valid == 0:
                continue

            # 取有效的 GT 环
            valid_mask = gt_valid[b] > 0.5
            gt_xy_b = gt_xy[b][valid_mask]  # (n_valid, 2)
            gt_types_b = gt_types[b][valid_mask].long()  # (n_valid,)

            # 匈牙利匹配：pred (10, 2) vs gt (n_valid, 2)
            with torch.no_grad():
                cost = torch.cdist(pred_centers[b], gt_xy_b)  # (10, n_valid)
                row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

            # 中心 MSE（只对匹配的 pair）
            matched_pred = pred_centers[b][row_ind]  # (n_match, 2)
            matched_gt = gt_xy_b[col_ind]  # (n_match, 2)
            center_losses.append(F.mse_loss(matched_pred, matched_gt))

            # 类型 CE（匹配后）
            matched_type_logits = type_logits[b][row_ind]  # (n_match, 9)
            matched_type_gt = gt_types_b[col_ind]  # (n_match,)
            type_losses.append(F.cross_entropy(matched_type_logits, matched_type_gt))

        center_loss = torch.stack(center_losses).mean() if center_losses else torch.tensor(0.0, device=device)
        type_loss = torch.stack(type_losses).mean() if type_losses else torch.tensor(0.0, device=device)

        ring_total_loss = count_loss + 5.0 * center_loss + type_loss

        return {
            "ring_total_loss": ring_total_loss,
            "ring_count_loss": count_loss,
            "ring_center_loss": center_loss,
            "ring_type_loss": type_loss,
        }

    @torch.no_grad()
    def predict(self, c_global, c_patches):
        """推理时预测环信息。"""
        count_logits, count_reg, pred_centers, type_logits = self.forward(c_global, c_patches)
        B = c_global.shape[0]
        device = c_global.device

        # 融合分类 + 回归得环数量
        count_cls = count_logits.argmax(dim=-1)  # (B,)
        n_rings = (0.7 * count_cls.float() + 0.3 * count_reg.clamp(0, self.max_rings)).round().long()
        n_rings = n_rings.clamp(0, self.max_rings)

        # 构建 3D 中心（Z=0）
        centers_3d = torch.zeros(B, self.max_rings, 3, device=device)
        centers_3d[:, :, :2] = pred_centers

        # 有效性 mask + 类型
        valid = torch.zeros(B, self.max_rings, device=device)
        for b in range(B):
            valid[b, :n_rings[b]] = 1.0
        types = type_logits.argmax(dim=-1)  # (B, 10)

        return {
            "n_rings": n_rings,
            "ring_centers": centers_3d,
            "ring_types": types,
            "ring_valid": valid,
        }


class ScaffoldRelationHead(nn.Module):
    """Auxiliary head for attachment/sidechain scaffold relation supervision.

    This is a minimal direct-supervision bridge: it predicts deterministic
    attachment and sidechain relation slots from AFM encoder features, without
    changing the main generation path.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        max_attachment_tokens: int = 24,
        max_sidechain_tokens: int = 48,
        num_atom_types: int = 10,
        max_graph_depth: int = 8,
    ):
        super().__init__()
        self.max_attachment_tokens = max_attachment_tokens
        self.max_sidechain_tokens = max_sidechain_tokens
        self.num_atom_types = num_atom_types
        self.max_graph_depth = max_graph_depth

        self.patch_proj = nn.Linear(embed_dim, hidden_dim)

        self.attachment_queries = nn.Parameter(torch.randn(max_attachment_tokens, hidden_dim) * 0.02)
        self.attachment_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attachment_norm = nn.LayerNorm(hidden_dim)
        self.attachment_valid = nn.Linear(hidden_dim, 1)
        self.attachment_parent = nn.Linear(hidden_dim, MAX_RING_SYSTEMS + 1)
        self.attachment_type = nn.Linear(hidden_dim, num_atom_types)
        self.attachment_rel = nn.Linear(hidden_dim, 3)

        self.sidechain_queries = nn.Parameter(torch.randn(max_sidechain_tokens, hidden_dim) * 0.02)
        self.sidechain_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.sidechain_norm = nn.LayerNorm(hidden_dim)
        self.sidechain_valid = nn.Linear(hidden_dim, 1)
        self.sidechain_parent = nn.Linear(hidden_dim, MAX_RING_SYSTEMS + 1)
        self.sidechain_type = nn.Linear(hidden_dim, num_atom_types)
        self.sidechain_kind = nn.Linear(hidden_dim, 3)
        self.sidechain_depth = nn.Linear(hidden_dim, max_graph_depth + 1)
        self.sidechain_rel = nn.Linear(hidden_dim, 3)

    def _pool_spatial(self, c_patches):
        B, P, D = c_patches.shape
        return c_patches.view(B, 5, 64, D).mean(dim=1)

    def forward(self, c_patches):
        B = c_patches.shape[0]
        spatial = self._pool_spatial(c_patches)
        kv = self.patch_proj(spatial)

        att_q = self.attachment_queries.unsqueeze(0).expand(B, -1, -1)
        att_feat, _ = self.attachment_cross_attn(self.attachment_norm(att_q), kv, kv)
        att_feat = att_q + att_feat

        side_q = self.sidechain_queries.unsqueeze(0).expand(B, -1, -1)
        side_feat, _ = self.sidechain_cross_attn(self.sidechain_norm(side_q), kv, kv)
        side_feat = side_q + side_feat

        return {
            "attachment_valid_logits": self.attachment_valid(att_feat).squeeze(-1),
            "attachment_parent_logits": self.attachment_parent(att_feat),
            "attachment_type_logits": self.attachment_type(att_feat),
            "attachment_rel": self.attachment_rel(att_feat),
            "sidechain_valid_logits": self.sidechain_valid(side_feat).squeeze(-1),
            "sidechain_parent_logits": self.sidechain_parent(side_feat),
            "sidechain_type_logits": self.sidechain_type(side_feat),
            "sidechain_kind_logits": self.sidechain_kind(side_feat),
            "sidechain_depth_logits": self.sidechain_depth(side_feat),
            "sidechain_rel": self.sidechain_rel(side_feat),
        }

    def compute_loss_from_pred(self, pred: dict, targets: dict):
        device = pred["attachment_valid_logits"].device

        att_valid_loss = F.binary_cross_entropy_with_logits(
            pred["attachment_valid_logits"], targets["attachment_valid"]
        )
        side_valid_loss = F.binary_cross_entropy_with_logits(
            pred["sidechain_valid_logits"], targets["sidechain_valid"]
        )

        att_mask = targets["attachment_valid"] > 0.5
        if att_mask.any():
            att_parent_loss = F.cross_entropy(
                pred["attachment_parent_logits"][att_mask], targets["attachment_parent"][att_mask]
            )
            att_type_loss = F.cross_entropy(
                pred["attachment_type_logits"][att_mask], targets["attachment_type"][att_mask]
            )
            att_rel_loss = F.mse_loss(
                pred["attachment_rel"][att_mask], targets["attachment_rel"][att_mask]
            )
        else:
            att_parent_loss = torch.tensor(0.0, device=device)
            att_type_loss = torch.tensor(0.0, device=device)
            att_rel_loss = torch.tensor(0.0, device=device)

        side_mask = targets["sidechain_valid"] > 0.5
        if side_mask.any():
            side_parent_loss = F.cross_entropy(
                pred["sidechain_parent_logits"][side_mask], targets["sidechain_parent"][side_mask]
            )
            side_type_loss = F.cross_entropy(
                pred["sidechain_type_logits"][side_mask], targets["sidechain_type"][side_mask]
            )
            side_kind_loss = F.cross_entropy(
                pred["sidechain_kind_logits"][side_mask], targets["sidechain_kind"][side_mask]
            )
            side_depth_loss = F.cross_entropy(
                pred["sidechain_depth_logits"][side_mask], targets["sidechain_depth"][side_mask]
            )
            side_rel_loss = F.mse_loss(
                pred["sidechain_rel"][side_mask], targets["sidechain_rel"][side_mask]
            )
        else:
            side_parent_loss = torch.tensor(0.0, device=device)
            side_type_loss = torch.tensor(0.0, device=device)
            side_kind_loss = torch.tensor(0.0, device=device)
            side_depth_loss = torch.tensor(0.0, device=device)
            side_rel_loss = torch.tensor(0.0, device=device)

        attachment_loss = att_valid_loss + att_parent_loss + att_type_loss + 2.0 * att_rel_loss
        sidechain_loss = (
            side_valid_loss + side_parent_loss + side_type_loss + side_kind_loss + side_depth_loss
            + 2.0 * side_rel_loss
        )
        scaffold_aux_loss = attachment_loss + sidechain_loss

        return {
            "scaffold_aux_loss": scaffold_aux_loss,
            "attachment_aux_loss": attachment_loss,
            "sidechain_aux_loss": sidechain_loss,
            "attachment_valid_loss": att_valid_loss,
            "attachment_parent_loss": att_parent_loss,
            "attachment_type_loss": att_type_loss,
            "attachment_rel_loss": att_rel_loss,
            "sidechain_valid_loss": side_valid_loss,
            "sidechain_parent_loss": side_parent_loss,
            "sidechain_type_loss": side_type_loss,
            "sidechain_kind_loss": side_kind_loss,
            "sidechain_depth_loss": side_depth_loss,
            "sidechain_rel_loss": side_rel_loss,
        }

    def compute_loss(self, c_patches, targets: dict):
        pred = self.forward(c_patches)
        return self.compute_loss_from_pred(pred, targets)


class AtomSemanticHead(nn.Module):
    """V18.1: predicted-atom semantic supervision head.

    This head reads the current reconstructed atom set and predicts three
    structure-critical semantics directly on atom slots:
    - atom role: scaffold core / attachment anchor / sidechain
    - attachment site flag
    - hetero-site flag
    - parent ring-system assignment

    The supervision is matched to GT atoms with a Hungarian assignment on the
    reconstructed coordinates, so the loss follows atom-level reconstruction
    quality instead of raw XYZ file order.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_atom_types: int = 10,
        hidden_dim: int = 256,
        max_atoms: int = 85,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.num_roles = len(ATOM_ROLE_TO_IDX)

        coord_dim = hidden_dim // 4
        type_dim = hidden_dim // 4
        index_dim = hidden_dim // 8
        global_dim = hidden_dim - coord_dim - type_dim - index_dim

        self.coord_proj = nn.Linear(3, coord_dim)
        self.type_proj = nn.Linear(num_atom_types, type_dim)
        self.index_embed = nn.Embedding(max_atoms, index_dim)
        self.global_proj = nn.Linear(embed_dim, global_dim)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self.semantic_valid = nn.Linear(hidden_dim, 1)
        self.atom_role = nn.Linear(hidden_dim, self.num_roles)
        self.attachment_site = nn.Linear(hidden_dim, 1)
        self.hetero_site = nn.Linear(hidden_dim, 1)
        self.parent_system = nn.Linear(hidden_dim, MAX_RING_SYSTEMS + 1)
        self.canonical_site = nn.Linear(hidden_dim, max_atoms + 1)
        self.attachment_target_site = nn.Linear(hidden_dim, max_atoms + 1)
        self.sidechain_root_site = nn.Linear(hidden_dim, max_atoms + 1)
        self.hetero_target_class = nn.Linear(hidden_dim, num_atom_types)

    def _build_features(self, coords_pred: torch.Tensor, type_logits: torch.Tensor, c_global: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords_pred.shape
        device = coords_pred.device
        coord_feat = self.coord_proj(coords_pred.float())
        type_feat = self.type_proj(F.softmax(type_logits.float(), dim=-1))
        index_ids = torch.arange(N, device=device, dtype=torch.long)
        index_feat = self.index_embed(index_ids).unsqueeze(0).expand(B, -1, -1)
        global_feat = self.global_proj(c_global.float()).unsqueeze(1).expand(B, N, -1)
        h = torch.cat([coord_feat, type_feat, index_feat, global_feat], dim=-1)
        h = self.fuse(h)
        return self.norm(h)

    def forward(self, coords_pred: torch.Tensor, type_logits: torch.Tensor, c_global: torch.Tensor) -> dict:
        h = self._build_features(coords_pred, type_logits, c_global)
        return {
            "semantic_valid_logits": self.semantic_valid(h).squeeze(-1),
            "atom_role_logits": self.atom_role(h),
            "attachment_site_logits": self.attachment_site(h).squeeze(-1),
            "hetero_site_logits": self.hetero_site(h).squeeze(-1),
            "parent_system_logits": self.parent_system(h),
            "canonical_site_logits": self.canonical_site(h),
            "attachment_target_site_logits": self.attachment_target_site(h),
            "sidechain_root_site_logits": self.sidechain_root_site(h),
            "hetero_target_class_logits": self.hetero_target_class(h),
        }

    def compute_loss_from_pred(
        self,
        pred: dict,
        coords_pred: torch.Tensor,
        type_logits: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor,
        gt_mask: torch.Tensor,
        scaffold_labels: dict,
    ) -> dict:
        device = coords_pred.device
        B, N, _ = coords_pred.shape

        valid_target = torch.zeros(B, N, device=device)
        role_target = torch.full((B, N), -1, device=device, dtype=torch.long)
        attachment_target = torch.zeros(B, N, device=device)
        hetero_target = torch.zeros(B, N, device=device)
        parent_target = torch.full((B, N), MAX_RING_SYSTEMS, device=device, dtype=torch.long)
        canonical_site_target = torch.full((B, N), self.max_atoms, device=device, dtype=torch.long)
        attachment_target_site = torch.full((B, N), self.max_atoms, device=device, dtype=torch.long)
        sidechain_root_site = torch.full((B, N), self.max_atoms, device=device, dtype=torch.long)
        hetero_target_class = torch.zeros(B, N, device=device, dtype=torch.long)

        pred_type_probs = F.softmax(type_logits.float(), dim=-1)

        for b in range(B):
            n_gt = int(gt_mask[b].sum().item())
            if n_gt <= 0:
                continue

            gt_coords_b = gt_coords[b, :n_gt].float()
            gt_types_b = gt_types[b, :n_gt].clamp(min=0).long()
            cost = torch.cdist(coords_pred[b].float(), gt_coords_b)
            type_cost = 1.0 - pred_type_probs[b][:, gt_types_b]
            cost = cost + 0.05 * type_cost

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            valid_target[b, row_ind] = 1.0
            role_target[b, row_ind] = scaffold_labels["scaffold_atom_role"][b, col_ind].long()
            attachment_target[b, row_ind] = scaffold_labels["scaffold_atom_attachment_site_target"][b, col_ind].float()
            hetero_target[b, row_ind] = scaffold_labels["scaffold_atom_hetero_site_target"][b, col_ind].float()

            parent = scaffold_labels["scaffold_atom_parent_system_target"][b, col_ind].long()
            parent = torch.where(parent >= 0, parent, torch.full_like(parent, MAX_RING_SYSTEMS))
            parent_target[b, row_ind] = parent
            canonical = scaffold_labels["scaffold_atom_canonical_site_index"][b, col_ind].long()
            canonical = torch.where(canonical >= 0, canonical, torch.full_like(canonical, self.max_atoms))
            canonical_site_target[b, row_ind] = canonical
            attach_site = scaffold_labels["scaffold_atom_attachment_target_site"][b, col_ind].long()
            attach_site = torch.where(attach_site >= 0, attach_site, torch.full_like(attach_site, self.max_atoms))
            attachment_target_site[b, row_ind] = attach_site
            side_root = scaffold_labels["scaffold_atom_sidechain_root_site"][b, col_ind].long()
            side_root = torch.where(side_root >= 0, side_root, torch.full_like(side_root, self.max_atoms))
            sidechain_root_site[b, row_ind] = side_root
            hetero_cls = scaffold_labels["scaffold_atom_hetero_target_class"][b, col_ind].long()
            hetero_target_class[b, row_ind] = hetero_cls.clamp(min=0, max=type_logits.shape[-1] - 1)

        valid_loss = F.binary_cross_entropy_with_logits(pred["semantic_valid_logits"], valid_target)
        matched_mask = valid_target > 0.5
        if matched_mask.any():
            role_valid_mask = matched_mask & (role_target >= 0) & (role_target < self.num_roles)
            parent_valid_mask = matched_mask & (parent_target >= 0) & (parent_target <= MAX_RING_SYSTEMS)

            if role_valid_mask.any():
                role_loss = F.cross_entropy(
                    pred["atom_role_logits"][role_valid_mask], role_target[role_valid_mask]
                )
            else:
                role_loss = torch.tensor(0.0, device=device)

            attachment_targets = attachment_target[matched_mask].clamp(0.0, 1.0)
            hetero_targets = hetero_target[matched_mask].clamp(0.0, 1.0)
            attachment_loss = F.binary_cross_entropy_with_logits(
                pred["attachment_site_logits"][matched_mask], attachment_targets
            )
            hetero_loss = F.binary_cross_entropy_with_logits(
                pred["hetero_site_logits"][matched_mask], hetero_targets
            )

            if parent_valid_mask.any():
                parent_loss = F.cross_entropy(
                    pred["parent_system_logits"][parent_valid_mask], parent_target[parent_valid_mask]
                )
            else:
                parent_loss = torch.tensor(0.0, device=device)

            site_valid_mask = matched_mask & (canonical_site_target >= 0) & (canonical_site_target <= self.max_atoms)
            if site_valid_mask.any():
                canonical_site_loss = F.cross_entropy(
                    pred["canonical_site_logits"][site_valid_mask], canonical_site_target[site_valid_mask]
                )
            else:
                canonical_site_loss = torch.tensor(0.0, device=device)

            attach_target_valid_mask = matched_mask & (attachment_target_site >= 0) & (attachment_target_site <= self.max_atoms)
            if attach_target_valid_mask.any():
                attachment_target_site_loss = F.cross_entropy(
                    pred["attachment_target_site_logits"][attach_target_valid_mask],
                    attachment_target_site[attach_target_valid_mask],
                )
            else:
                attachment_target_site_loss = torch.tensor(0.0, device=device)

            side_root_valid_mask = matched_mask & (sidechain_root_site >= 0) & (sidechain_root_site <= self.max_atoms)
            if side_root_valid_mask.any():
                sidechain_root_site_loss = F.cross_entropy(
                    pred["sidechain_root_site_logits"][side_root_valid_mask],
                    sidechain_root_site[side_root_valid_mask],
                )
            else:
                sidechain_root_site_loss = torch.tensor(0.0, device=device)

            hetero_class_loss = F.cross_entropy(
                pred["hetero_target_class_logits"][matched_mask],
                hetero_target_class[matched_mask],
            )
        else:
            role_loss = torch.tensor(0.0, device=device)
            attachment_loss = torch.tensor(0.0, device=device)
            hetero_loss = torch.tensor(0.0, device=device)
            parent_loss = torch.tensor(0.0, device=device)
            canonical_site_loss = torch.tensor(0.0, device=device)
            attachment_target_site_loss = torch.tensor(0.0, device=device)
            sidechain_root_site_loss = torch.tensor(0.0, device=device)
            hetero_class_loss = torch.tensor(0.0, device=device)

        semantic_aux_loss = (
            valid_loss + role_loss + attachment_loss + hetero_loss + parent_loss
            + canonical_site_loss + attachment_target_site_loss + sidechain_root_site_loss
            + hetero_class_loss
        )
        return {
            "semantic_aux_loss": semantic_aux_loss,
            "semantic_valid_loss": valid_loss,
            "atom_role_aux_loss": role_loss,
            "attachment_site_aux_loss": attachment_loss,
            "hetero_site_aux_loss": hetero_loss,
            "semantic_parent_aux_loss": parent_loss,
            "canonical_site_aux_loss": canonical_site_loss,
            "attachment_target_site_aux_loss": attachment_target_site_loss,
            "sidechain_root_site_aux_loss": sidechain_root_site_loss,
            "hetero_target_class_aux_loss": hetero_class_loss,
        }

    def compute_loss(
        self,
        coords_pred: torch.Tensor,
        type_logits: torch.Tensor,
        c_global: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor,
        gt_mask: torch.Tensor,
        scaffold_labels: dict,
    ) -> dict:
        pred = self.forward(coords_pred, type_logits, c_global)
        return self.compute_loss_from_pred(
            pred, coords_pred, type_logits, gt_coords, gt_types, gt_mask, scaffold_labels
        )


class SiteGraphParserHead(nn.Module):
    """V18.5: parse scaffold site objects and their local graph from AFM patches."""

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        max_site_tokens: int = 48,
        num_atom_types: int = 10,
    ):
        super().__init__()
        self.max_site_tokens = max_site_tokens

        self.patch_proj = nn.Linear(embed_dim, hidden_dim)
        self.site_queries = nn.Parameter(torch.randn(max_site_tokens, hidden_dim) * 0.02)
        self.site_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.site_norm = nn.LayerNorm(hidden_dim)

        self.site_valid = nn.Linear(hidden_dim, 1)
        self.site_coord = nn.Linear(hidden_dim, 3)
        self.site_parent = nn.Linear(hidden_dim, MAX_RING_SYSTEMS + 1)
        self.site_element = nn.Linear(hidden_dim, num_atom_types)
        self.site_attachment = nn.Linear(hidden_dim, 1)
        self.site_hetero = nn.Linear(hidden_dim, 1)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _pool_spatial(self, c_patches: torch.Tensor) -> torch.Tensor:
        bsz, n_patches, dim = c_patches.shape
        if n_patches % 64 != 0:
            return c_patches
        return c_patches.view(bsz, n_patches // 64, 64, dim).mean(dim=1)

    def forward(self, c_patches: torch.Tensor) -> dict:
        bsz = c_patches.shape[0]
        spatial = self._pool_spatial(c_patches)
        kv = self.patch_proj(spatial)

        queries = self.site_queries.unsqueeze(0).expand(bsz, -1, -1)
        feat, _ = self.site_cross_attn(self.site_norm(queries), kv, kv)
        feat = queries + feat

        q_i = feat.unsqueeze(2).expand(-1, -1, self.max_site_tokens, -1)
        q_j = feat.unsqueeze(1).expand(-1, self.max_site_tokens, -1, -1)
        pair_logits = self.pair_mlp(torch.cat([q_i, q_j], dim=-1)).squeeze(-1)

        return {
            "site_valid_logits": self.site_valid(feat).squeeze(-1),
            "site_coord": self.site_coord(feat),
            "site_parent_logits": self.site_parent(feat),
            "site_element_logits": self.site_element(feat),
            "site_attachment_logits": self.site_attachment(feat).squeeze(-1),
            "site_hetero_logits": self.site_hetero(feat).squeeze(-1),
            "site_pair_logits": pair_logits,
        }

    def _build_targets_from_batch(
        self,
        gt_coords: torch.Tensor,
        gt_mask: torch.Tensor,
        scaffold_labels: dict,
    ) -> dict:
        device = gt_coords.device
        bsz, _, _ = gt_coords.shape
        q = self.max_site_tokens

        valid = torch.zeros(bsz, q, device=device)
        coord = torch.zeros(bsz, q, 3, device=device)
        parent = torch.full((bsz, q), MAX_RING_SYSTEMS, device=device, dtype=torch.long)
        element = torch.zeros(bsz, q, device=device, dtype=torch.long)
        attachment = torch.zeros(bsz, q, device=device)
        hetero = torch.zeros(bsz, q, device=device)
        pair_adj = torch.zeros(bsz, q, q, device=device)

        for b in range(bsz):
            n_gt = int(gt_mask[b].sum().item())
            if n_gt <= 0:
                continue

            atom_mask = gt_mask[b, :n_gt] > 0.5
            scaffold_mask = scaffold_labels["scaffold_atom_is_scaffold"][b, :n_gt] > 0.5
            site_index = scaffold_labels["scaffold_atom_canonical_site_index"][b, :n_gt].long()
            parent_system = scaffold_labels["scaffold_atom_parent_system_target"][b, :n_gt].long()
            element_cls = scaffold_labels["scaffold_atom_hetero_target_class"][b, :n_gt].long()
            anchor_flag = scaffold_labels["scaffold_atom_is_attachment_anchor"][b, :n_gt] > 0.5

            group_to_atoms = {}
            atom_to_slot = {}
            for atom_idx in range(n_gt):
                if not atom_mask[atom_idx]:
                    continue
                if not scaffold_mask[atom_idx]:
                    continue
                site = int(site_index[atom_idx].item())
                parent_id = int(parent_system[atom_idx].item())
                if site < 0 or parent_id < 0:
                    continue
                key = (parent_id, site)
                group_to_atoms.setdefault(key, []).append(atom_idx)

            if not group_to_atoms:
                continue

            ordered_keys = sorted(group_to_atoms.keys())[:q]
            for slot, key in enumerate(ordered_keys):
                atoms = group_to_atoms[key]
                valid[b, slot] = 1.0
                coord[b, slot] = gt_coords[b, atoms].float().mean(dim=0)
                parent[b, slot] = key[0]

                cls_vals = element_cls[atoms]
                cls_vals = cls_vals[cls_vals >= 0]
                if cls_vals.numel() > 0:
                    uniq, counts = torch.unique(cls_vals, return_counts=True)
                    element[b, slot] = uniq[counts.argmax()]
                attachment[b, slot] = 1.0 if bool(anchor_flag[atoms].any().item()) else 0.0
                hetero[b, slot] = 1.0 if int(element[b, slot].item()) not in (0, 1) else 0.0
                for atom_idx in atoms:
                    atom_to_slot[atom_idx] = slot

            n_edges = int(scaffold_labels["scaffold_n_local_edges"][b].item())
            edges = scaffold_labels["scaffold_local_edges"][b, :n_edges].long()
            for edge in edges:
                u = int(edge[0].item())
                v = int(edge[1].item())
                if u not in atom_to_slot or v not in atom_to_slot:
                    continue
                su = atom_to_slot[u]
                sv = atom_to_slot[v]
                if su == sv:
                    continue
                pair_adj[b, su, sv] = 1.0
                pair_adj[b, sv, su] = 1.0

        return {
            "site_valid": valid,
            "site_coord": coord,
            "site_parent": parent,
            "site_element": element,
            "site_attachment": attachment,
            "site_hetero": hetero,
            "site_pair_adj": pair_adj,
        }

    def compute_loss_from_pred(self, pred: dict, targets: dict) -> dict:
        device = pred["site_valid_logits"].device
        bsz, q = pred["site_valid_logits"].shape

        valid_target = torch.zeros(bsz, q, device=device)
        coord_target = torch.zeros(bsz, q, 3, device=device)
        parent_target = torch.full((bsz, q), MAX_RING_SYSTEMS, device=device, dtype=torch.long)
        element_target = torch.zeros(bsz, q, device=device, dtype=torch.long)
        attachment_target = torch.zeros(bsz, q, device=device)
        hetero_target = torch.zeros(bsz, q, device=device)
        pair_target = torch.zeros(bsz, q, q, device=device)

        for b in range(bsz):
            gt_valid = targets["site_valid"][b] > 0.5
            n_gt = int(gt_valid.sum().item())
            if n_gt <= 0:
                continue

            gt_coord = targets["site_coord"][b, gt_valid]
            gt_parent = targets["site_parent"][b, gt_valid]
            gt_element = targets["site_element"][b, gt_valid]
            gt_attachment = targets["site_attachment"][b, gt_valid]
            gt_hetero = targets["site_hetero"][b, gt_valid]
            gt_pair = targets["site_pair_adj"][b][gt_valid][:, gt_valid]

            pred_coord = pred["site_coord"][b]
            pred_parent_prob = F.softmax(pred["site_parent_logits"][b], dim=-1)
            pred_element_prob = F.softmax(pred["site_element_logits"][b], dim=-1)
            cost = torch.cdist(pred_coord.float(), gt_coord.float())
            cost = cost + 0.05 * (1.0 - pred_parent_prob[:, gt_parent])
            cost = cost + 0.05 * (1.0 - pred_element_prob[:, gt_element])

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)

            valid_target[b, row_ind] = 1.0
            coord_target[b, row_ind] = gt_coord[col_ind]
            parent_target[b, row_ind] = gt_parent[col_ind]
            element_target[b, row_ind] = gt_element[col_ind]
            attachment_target[b, row_ind] = gt_attachment[col_ind]
            hetero_target[b, row_ind] = gt_hetero[col_ind]

            row_to_col = {int(r.item()): int(c.item()) for r, c in zip(row_ind, col_ind)}
            for qi in row_ind.tolist():
                for qj in row_ind.tolist():
                    if qi == qj:
                        continue
                    pair_target[b, qi, qj] = gt_pair[row_to_col[qi], row_to_col[qj]]

        valid_loss = F.binary_cross_entropy_with_logits(pred["site_valid_logits"], valid_target)
        matched_mask = valid_target > 0.5
        if matched_mask.any():
            coord_loss = F.smooth_l1_loss(pred["site_coord"][matched_mask], coord_target[matched_mask])
            parent_loss = F.cross_entropy(pred["site_parent_logits"][matched_mask], parent_target[matched_mask])
            element_loss = F.cross_entropy(pred["site_element_logits"][matched_mask], element_target[matched_mask])
            attachment_loss = F.binary_cross_entropy_with_logits(
                pred["site_attachment_logits"][matched_mask],
                attachment_target[matched_mask].clamp(0.0, 1.0),
            )
            hetero_loss = F.binary_cross_entropy_with_logits(
                pred["site_hetero_logits"][matched_mask],
                hetero_target[matched_mask].clamp(0.0, 1.0),
            )
        else:
            coord_loss = torch.tensor(0.0, device=device)
            parent_loss = torch.tensor(0.0, device=device)
            element_loss = torch.tensor(0.0, device=device)
            attachment_loss = torch.tensor(0.0, device=device)
            hetero_loss = torch.tensor(0.0, device=device)

        pair_mask = (valid_target.unsqueeze(1) > 0.5) & (valid_target.unsqueeze(2) > 0.5)
        upper = torch.triu(torch.ones(q, q, device=device, dtype=torch.bool), diagonal=1)
        pair_mask = pair_mask & upper.unsqueeze(0)
        if pair_mask.any():
            pair_loss = F.binary_cross_entropy_with_logits(
                pred["site_pair_logits"][pair_mask],
                pair_target[pair_mask].clamp(0.0, 1.0),
            )
        else:
            pair_loss = torch.tensor(0.0, device=device)

        total = valid_loss + coord_loss + parent_loss + element_loss + attachment_loss + hetero_loss + pair_loss
        return {
            "site_graph_aux_loss": total,
            "site_object_valid_loss": valid_loss,
            "site_coord_aux_loss": coord_loss,
            "site_parent_aux_loss": parent_loss,
            "site_element_aux_loss": element_loss,
            "site_attachment_aux_loss": attachment_loss,
            "site_hetero_aux_loss": hetero_loss,
            "site_edge_aux_loss": pair_loss,
        }

    def compute_loss(
        self,
        c_patches: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_mask: torch.Tensor,
        scaffold_labels: dict,
    ) -> dict:
        pred = self.forward(c_patches)
        targets = self._build_targets_from_batch(gt_coords, gt_mask, scaffold_labels)
        return self.compute_loss_from_pred(pred, targets)
