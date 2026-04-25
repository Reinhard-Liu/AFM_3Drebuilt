"""
V17-Bridge scaffold token encoder.

This module converts GT scaffold labels into token sequences that can later be
fed into the denoiser via cross-attention. The current bridge step is strictly
GT-conditioned: it does not predict scaffold tokens yet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ring_detection import (
    MAX_RING_SYSTEMS,
    MAX_SYSTEM_RELATIONS,
    MAX_SYSTEM_SIDECHAIN_EDGES,
    RELATION_TYPE_TO_IDX,
)


class GTScaffoldTokenEncoder(nn.Module):
    """Encode GT scaffold labels into token sequences.

    Token categories:
    - Ring-system tokens: one per valid ring system
    - Relation tokens: one per valid system-system relation
    - Attachment-anchor tokens: one per valid anchor atom (capped)
    - Sidechain-edge tokens: one per valid scaffold/non-scaffold or
      non-scaffold/non-scaffold local bond (capped)

    The encoder deliberately avoids canonical site tokens for now because the
    current site labels are not stable enough. A `site_labels_valid` scalar is
    still passed through the data path so the next stage can gate future site
    conditioning when the labels are upgraded.
    """

    TOKEN_RING_SYSTEM = 0
    TOKEN_RELATION = 1
    TOKEN_ATTACHMENT = 2
    TOKEN_SIDECHAIN = 3
    TOKEN_SITE = 4

    SIDECHAIN_KIND_ATTACH = 0
    SIDECHAIN_KIND_BRANCH = 1
    SIDECHAIN_KIND_MISC = 2

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_atom_types: int = 10,
        max_attachment_tokens: int = 24,
        max_sidechain_tokens: int = 48,
        include_sidechain_tokens: bool = True,
        include_site_tokens: bool = False,
        max_site_tokens: int = 48,
        max_graph_depth: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.max_attachment_tokens = max_attachment_tokens
        self.max_sidechain_tokens = max_sidechain_tokens
        self.include_sidechain_tokens = include_sidechain_tokens
        self.include_site_tokens = include_site_tokens
        self.max_site_tokens = max_site_tokens
        self.max_graph_depth = max_graph_depth

        rel_embed_dim = hidden_dim // 4
        aux_embed_dim = hidden_dim // 8

        self.token_type_embed = nn.Embedding(5, embed_dim)
        self.relation_type_embed = nn.Embedding(len(RELATION_TYPE_TO_IDX), rel_embed_dim)
        self.atom_type_embed = nn.Embedding(num_atom_types, rel_embed_dim)
        self.sidechain_kind_embed = nn.Embedding(3, aux_embed_dim)
        self.depth_embed = nn.Embedding(max_graph_depth + 2, aux_embed_dim)
        self.anchor_flag_embed = nn.Embedding(2, aux_embed_dim)
        self.site_index_embed = nn.Embedding(86, aux_embed_dim)

        # num_rings, num_atoms, aromaticity, has_heteroatom, center(3), normal(3)
        self.system_mlp = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # src_center(3), dst_center(3), rel_vec(3), relation_embed
        self.relation_mlp = nn.Sequential(
            nn.Linear(9 + rel_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # anchor_coord(3), parent_center(3), rel_vec(3), atom_type_embed
        self.attachment_mlp = nn.Sequential(
            nn.Linear(9 + rel_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # src_coord(3), dst_coord(3), rel_vec(3), parent_center(3), dst_atom,
        # edge kind, src/dst depth, src/dst anchor flags
        self.sidechain_mlp = nn.Sequential(
            nn.Linear(12 + rel_embed_dim + 5 * aux_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.site_mlp = nn.Sequential(
            nn.Linear(12 + rel_embed_dim + 2 * aux_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.output_norm = nn.LayerNorm(embed_dim)

    def _encode_system_tokens(self, batch: dict, b: int, device: torch.device):
        valid = batch["scaffold_system_objectness"][b] > 0.5
        n_systems = int(valid.sum().item())
        if n_systems == 0:
            return []

        num_rings = batch["scaffold_system_num_rings"][b, :n_systems].float().unsqueeze(-1)
        num_atoms = batch["scaffold_system_num_atoms"][b, :n_systems].float().unsqueeze(-1)
        aromaticity = batch["scaffold_system_aromaticity"][b, :n_systems].float().unsqueeze(-1)
        has_hetero = batch["scaffold_system_has_heteroatom"][b, :n_systems].float().unsqueeze(-1)
        center = batch["scaffold_system_center"][b, :n_systems].float()
        normal = batch["scaffold_system_normal"][b, :n_systems].float()

        features = torch.cat(
            [num_rings, num_atoms, aromaticity, has_hetero, center, normal], dim=-1
        ).to(device)
        tokens = self.system_mlp(features)
        tokens = tokens + self.token_type_embed(
            torch.full((n_systems,), self.TOKEN_RING_SYSTEM, device=device, dtype=torch.long)
        )
        return [tok for tok in tokens]

    def _collect_attachment_targets(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> dict | None:
        anchor_mask = (
            (batch["scaffold_atom_is_attachment_anchor"][b] > 0.5)
            & (atom_mask[b] > 0.5)
        )
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
        if anchor_idx.numel() == 0:
            return None

        anchor_idx = anchor_idx[: self.max_attachment_tokens]
        parent_system = batch["scaffold_atom_to_ring_system_ids"][b, anchor_idx, 0].long().to(device)
        valid_parent = parent_system >= 0
        if valid_parent.sum().item() == 0:
            return None

        anchor_idx = anchor_idx[valid_parent]
        parent_system = parent_system[valid_parent]
        anchor_coord = coords[b, anchor_idx].float().to(device)
        parent_center = batch["scaffold_system_center"][b, parent_system].float().to(device)
        rel_vec = anchor_coord - parent_center
        safe_atom_types = atom_types[b, anchor_idx].clamp(min=0).long().to(device)
        return {
            "anchor_idx": anchor_idx.to(device),
            "parent_system": parent_system,
            "anchor_coord": anchor_coord,
            "parent_center": parent_center,
            "rel_vec": rel_vec,
            "atom_types": safe_atom_types,
        }

    def _encode_sidechain_tokens(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        b: int,
        device: torch.device,
    ):
        if "scaffold_n_sidechain_edges" not in batch or "scaffold_sidechain_edges" not in batch:
            return []

        n_edges = int(batch["scaffold_n_sidechain_edges"][b].item())
        if n_edges <= 0:
            return []

        edges = batch["scaffold_sidechain_edges"][b, :n_edges].long()
        depths = batch.get("scaffold_atom_graph_depth")
        root_system_ids = batch.get("scaffold_atom_root_system_id")
        scaffold_mask = batch.get("scaffold_atom_is_scaffold")
        anchor_mask = batch.get("scaffold_atom_is_attachment_anchor")
        if depths is not None:
            depths = depths[b].long()
        if root_system_ids is not None:
            root_system_ids = root_system_ids[b].long()
        if scaffold_mask is not None:
            scaffold_mask = scaffold_mask[b] > 0.5
        if anchor_mask is not None:
            anchor_mask = anchor_mask[b] > 0.5

        valid_edges = []
        for edge in edges:
            a = int(edge[0].item())
            c = int(edge[1].item())
            if a < 0 or c < 0 or a >= atom_mask.shape[1] or c >= atom_mask.shape[1]:
                continue
            if atom_mask[b, a] <= 0.5 or atom_mask[b, c] <= 0.5:
                continue
            if depths is not None:
                da = int(depths[a].item())
                dc = int(depths[c].item())
            else:
                da = dc = -1

            if da >= 0 and dc >= 0 and da != dc:
                src, dst = (a, c) if da < dc else (c, a)
            elif da == 0 and dc < 0:
                src, dst = a, c
            elif dc == 0 and da < 0:
                src, dst = c, a
            else:
                src, dst = (a, c) if a <= c else (c, a)
            valid_edges.append((src, dst))

        if not valid_edges:
            return []

        valid_edges = valid_edges[: min(self.max_sidechain_tokens, MAX_SYSTEM_SIDECHAIN_EDGES)]
        src_idx = torch.tensor([src for src, _ in valid_edges], device=device, dtype=torch.long)
        dst_idx = torch.tensor([dst for _, dst in valid_edges], device=device, dtype=torch.long)

        src_coord = coords[b, src_idx].float().to(device)
        dst_coord = coords[b, dst_idx].float().to(device)
        rel_vec = dst_coord - src_coord
        if root_system_ids is not None:
            src_root = root_system_ids[src_idx].clone()
            dst_root = root_system_ids[dst_idx].clone()
            parent_system = torch.where(src_root >= 0, src_root, dst_root)
            valid_parent = parent_system >= 0
            if valid_parent.sum().item() == 0:
                return []
            src_idx = src_idx[valid_parent]
            dst_idx = dst_idx[valid_parent]
            src_coord = src_coord[valid_parent]
            dst_coord = dst_coord[valid_parent]
            rel_vec = rel_vec[valid_parent]
            src_root = src_root[valid_parent]
            dst_root = dst_root[valid_parent]
            parent_system = parent_system[valid_parent]
            safe_parent = parent_system.clamp(min=0)
            parent_center = batch["scaffold_system_center"][b, safe_parent].float().to(device)
            parent_center = parent_center * (parent_system >= 0).float().unsqueeze(-1)
        else:
            parent_system = torch.full_like(src_idx, -1)
            parent_center = torch.zeros_like(src_coord)

        src_is_scaffold = scaffold_mask[src_idx].long().to(device) if scaffold_mask is not None else torch.zeros_like(src_idx)
        dst_is_scaffold = scaffold_mask[dst_idx].long().to(device) if scaffold_mask is not None else torch.zeros_like(dst_idx)
        edge_kind = torch.full_like(src_idx, self.SIDECHAIN_KIND_MISC)
        edge_kind[(src_is_scaffold == 1) & (dst_is_scaffold == 0)] = self.SIDECHAIN_KIND_ATTACH
        edge_kind[(src_is_scaffold == 0) & (dst_is_scaffold == 0)] = self.SIDECHAIN_KIND_BRANCH

        dst_types = atom_types[b, dst_idx].clamp(min=0).long().to(device)
        atom_embed = self.atom_type_embed(dst_types)
        kind_embed = self.sidechain_kind_embed(edge_kind)

        def _safe_depth_embed(indices: torch.Tensor):
            if depths is None:
                return torch.zeros(indices.shape[0], self.depth_embed.embedding_dim, device=device)
            vals = depths[indices].clamp(min=-1, max=self.max_graph_depth).to(device)
            vals = vals + 1  # unknown=-1 -> bucket 0
            return self.depth_embed(vals)

        src_depth_embed = _safe_depth_embed(src_idx)
        dst_depth_embed = _safe_depth_embed(dst_idx)

        if anchor_mask is not None:
            src_anchor = self.anchor_flag_embed(anchor_mask[src_idx].long().to(device))
            dst_anchor = self.anchor_flag_embed(anchor_mask[dst_idx].long().to(device))
        else:
            src_anchor = torch.zeros(src_idx.shape[0], self.anchor_flag_embed.embedding_dim, device=device)
            dst_anchor = torch.zeros(dst_idx.shape[0], self.anchor_flag_embed.embedding_dim, device=device)

        features = torch.cat(
            [
                src_coord, dst_coord, rel_vec, parent_center,
                atom_embed, kind_embed, src_depth_embed, dst_depth_embed,
                src_anchor, dst_anchor,
            ],
            dim=-1,
        )

        tokens = self.sidechain_mlp(features)
        tokens = tokens + self.token_type_embed(
            torch.full((tokens.shape[0],), self.TOKEN_SIDECHAIN, device=device, dtype=torch.long)
        )
        return [tok for tok in tokens]

    def _encode_relation_tokens(self, batch: dict, b: int, device: torch.device):
        n_rel = int(batch["scaffold_n_relations"][b].item())
        if n_rel <= 0:
            return []

        edges = batch["scaffold_relation_edges"][b, :n_rel].long().to(device)
        rel_types = batch["scaffold_relation_types"][b, :n_rel].long().to(device)
        centers = batch["scaffold_system_center"][b].float().to(device)

        src_center = centers[edges[:, 0]]
        dst_center = centers[edges[:, 1]]
        rel_vec = dst_center - src_center
        rel_embed = self.relation_type_embed(rel_types)
        features = torch.cat([src_center, dst_center, rel_vec, rel_embed], dim=-1)

        tokens = self.relation_mlp(features)
        tokens = tokens + self.token_type_embed(
            torch.full((n_rel,), self.TOKEN_RELATION, device=device, dtype=torch.long)
        )
        return [tok for tok in tokens]

    def _soft_embed(self, probs: torch.Tensor, table: nn.Embedding, offset: int = 0) -> torch.Tensor:
        weight = table.weight[offset:]
        return probs @ weight

    def _predict_parent_center(
        self,
        batch: dict,
        parent_logits: torch.Tensor,
        b: int,
        device: torch.device,
    ) -> torch.Tensor:
        system_mask = batch["scaffold_system_objectness"][b].float().to(device)
        system_centers = batch["scaffold_system_center"][b].float().to(device)
        parent_probs = F.softmax(parent_logits, dim=-1)[..., :MAX_RING_SYSTEMS]
        parent_probs = parent_probs * system_mask.unsqueeze(0)
        denom = parent_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        parent_probs = parent_probs / denom
        return parent_probs @ system_centers

    def _encode_predicted_attachment_tokens(
        self,
        batch: dict,
        predicted_relations: dict,
        b: int,
        device: torch.device,
    ):
        valid_prob = torch.sigmoid(
            predicted_relations["attachment_valid_logits"][b].float().to(device)
        ).unsqueeze(-1)
        parent_center = self._predict_parent_center(
            batch,
            predicted_relations["attachment_parent_logits"][b].float().to(device),
            b,
            device,
        )
        atom_probs = F.softmax(
            predicted_relations["attachment_type_logits"][b].float().to(device), dim=-1
        )
        atom_embed = self._soft_embed(atom_probs, self.atom_type_embed)
        rel_vec = predicted_relations["attachment_rel"][b].float().to(device)
        anchor_coord = parent_center + rel_vec
        features = torch.cat([anchor_coord, parent_center, rel_vec, atom_embed], dim=-1)
        tokens = self.attachment_mlp(features)
        token_type = self.token_type_embed(
            torch.full((tokens.shape[0],), self.TOKEN_ATTACHMENT, device=device, dtype=torch.long)
        )
        tokens = valid_prob * (tokens + token_type)
        return [tok for tok in tokens]

    def _encode_predicted_sidechain_tokens(
        self,
        batch: dict,
        predicted_relations: dict,
        b: int,
        device: torch.device,
    ):
        valid_prob = torch.sigmoid(
            predicted_relations["sidechain_valid_logits"][b].float().to(device)
        ).unsqueeze(-1)
        parent_center = self._predict_parent_center(
            batch,
            predicted_relations["sidechain_parent_logits"][b].float().to(device),
            b,
            device,
        )
        rel_vec = predicted_relations["sidechain_rel"][b].float().to(device)
        src_coord = parent_center
        dst_coord = parent_center + rel_vec

        atom_probs = F.softmax(
            predicted_relations["sidechain_type_logits"][b].float().to(device), dim=-1
        )
        atom_embed = self._soft_embed(atom_probs, self.atom_type_embed)

        kind_probs = F.softmax(
            predicted_relations["sidechain_kind_logits"][b].float().to(device), dim=-1
        )
        kind_embed = self._soft_embed(kind_probs, self.sidechain_kind_embed)

        depth_probs = F.softmax(
            predicted_relations["sidechain_depth_logits"][b].float().to(device), dim=-1
        )
        depth_weight = self.depth_embed.weight[1:self.max_graph_depth + 2]
        dst_depth_embed = depth_probs @ depth_weight
        src_depth_embed = self.depth_embed.weight[1].unsqueeze(0).expand_as(dst_depth_embed)

        attach_prob = kind_probs[:, self.SIDECHAIN_KIND_ATTACH:self.SIDECHAIN_KIND_ATTACH + 1]
        src_anchor_embed = (
            (1.0 - attach_prob) * self.anchor_flag_embed.weight[0].unsqueeze(0)
            + attach_prob * self.anchor_flag_embed.weight[1].unsqueeze(0)
        )
        dst_anchor_embed = self.anchor_flag_embed.weight[0].unsqueeze(0).expand_as(src_anchor_embed)

        features = torch.cat(
            [
                src_coord, dst_coord, rel_vec, parent_center,
                atom_embed, kind_embed, src_depth_embed, dst_depth_embed,
                src_anchor_embed, dst_anchor_embed,
            ],
            dim=-1,
        )
        tokens = self.sidechain_mlp(features)
        token_type = self.token_type_embed(
            torch.full((tokens.shape[0],), self.TOKEN_SIDECHAIN, device=device, dtype=torch.long)
        )
        tokens = valid_prob * (tokens + token_type)
        return [tok for tok in tokens]

    def _encode_attachment_tokens(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        b: int,
        device: torch.device,
    ):
        targets = self._collect_attachment_targets(batch, coords, atom_types, atom_mask, b, device)
        if targets is None:
            return []

        atom_embed = self.atom_type_embed(targets["atom_types"])
        features = torch.cat(
            [targets["anchor_coord"], targets["parent_center"], targets["rel_vec"], atom_embed], dim=-1
        )

        tokens = self.attachment_mlp(features)
        tokens = tokens + self.token_type_embed(
            torch.full((tokens.shape[0],), self.TOKEN_ATTACHMENT, device=device, dtype=torch.long)
        )
        return [tok for tok in tokens]

    def _encode_site_tokens(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        b: int,
        device: torch.device,
    ):
        if not self.include_site_tokens:
            return []
        required = [
            "scaffold_atom_is_scaffold",
            "scaffold_atom_canonical_site_index",
            "scaffold_atom_hetero_target_class",
            "scaffold_atom_to_ring_system_ids",
            "scaffold_system_center",
            "scaffold_system_normal",
        ]
        if not all(k in batch for k in required):
            return []

        valid_mask = (
            (batch["scaffold_atom_is_scaffold"][b] > 0.5)
            & (atom_mask[b] > 0.5)
            & (batch["scaffold_atom_canonical_site_index"][b] >= 0)
        )
        site_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if site_idx.numel() == 0:
            return []

        site_order = batch["scaffold_atom_canonical_site_index"][b, site_idx].long()
        parent_system = batch["scaffold_atom_to_ring_system_ids"][b, site_idx, 0].long()
        valid_parent = parent_system >= 0
        if valid_parent.sum().item() == 0:
            return []

        site_idx = site_idx[valid_parent]
        site_order = site_order[valid_parent]
        parent_system = parent_system[valid_parent]
        sort_key = parent_system * 100 + site_order.clamp(min=0)
        sort_perm = torch.argsort(sort_key)
        site_idx = site_idx[sort_perm][: self.max_site_tokens]
        site_order = site_order[sort_perm][: self.max_site_tokens]
        parent_system = parent_system[sort_perm][: self.max_site_tokens]

        site_coord = coords[b, site_idx].float().to(device)
        parent_center = batch["scaffold_system_center"][b, parent_system].float().to(device)
        parent_normal = batch["scaffold_system_normal"][b, parent_system].float().to(device)
        rel_vec = site_coord - parent_center
        site_types = batch["scaffold_atom_hetero_target_class"][b, site_idx].long().clamp(min=0).to(device)
        atom_embed = self.atom_type_embed(site_types)
        site_embed = self.site_index_embed(site_order.clamp(min=0, max=85).to(device))
        anchor_flag = (batch["scaffold_atom_is_attachment_anchor"][b, site_idx] > 0.5).long().to(device)
        anchor_embed = self.anchor_flag_embed(anchor_flag)

        features = torch.cat(
            [site_coord, parent_center, rel_vec, parent_normal, atom_embed, site_embed, anchor_embed],
            dim=-1,
        )
        tokens = self.site_mlp(features)
        tokens = tokens + self.token_type_embed(
            torch.full((tokens.shape[0],), self.TOKEN_SITE, device=device, dtype=torch.long)
        )
        return [tok for tok in tokens]

    def build_aux_targets(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> dict:
        """Build deterministic scaffold supervision targets for auxiliary heads."""
        device = coords.device
        B = coords.shape[0]

        att_valid = torch.zeros(B, self.max_attachment_tokens, device=device)
        att_parent = torch.full((B, self.max_attachment_tokens), MAX_RING_SYSTEMS, device=device, dtype=torch.long)
        att_type = torch.zeros(B, self.max_attachment_tokens, device=device, dtype=torch.long)
        att_rel = torch.zeros(B, self.max_attachment_tokens, 3, device=device)

        side_valid = torch.zeros(B, self.max_sidechain_tokens, device=device)
        side_parent = torch.full((B, self.max_sidechain_tokens), MAX_RING_SYSTEMS, device=device, dtype=torch.long)
        side_type = torch.zeros(B, self.max_sidechain_tokens, device=device, dtype=torch.long)
        side_kind = torch.zeros(B, self.max_sidechain_tokens, device=device, dtype=torch.long)
        side_depth = torch.zeros(B, self.max_sidechain_tokens, device=device, dtype=torch.long)
        side_rel = torch.zeros(B, self.max_sidechain_tokens, 3, device=device)

        for b in range(B):
            att_targets = self._collect_attachment_targets(batch, coords, atom_types, atom_mask, b, device)
            if att_targets is not None:
                n_att = min(att_targets["parent_system"].shape[0], self.max_attachment_tokens)
                att_valid[b, :n_att] = 1.0
                att_parent[b, :n_att] = att_targets["parent_system"][:n_att]
                att_type[b, :n_att] = att_targets["atom_types"][:n_att]
                att_rel[b, :n_att] = att_targets["rel_vec"][:n_att]

            n_edges = int(batch.get("scaffold_n_sidechain_edges", torch.zeros(B, device=device, dtype=torch.long))[b].item()) if "scaffold_n_sidechain_edges" in batch else 0
            if n_edges <= 0:
                continue
            edges = batch["scaffold_sidechain_edges"][b, :n_edges].long()
            depths = batch["scaffold_atom_graph_depth"][b].long()
            roots = batch["scaffold_atom_root_system_id"][b].long()
            scaffold_mask = batch["scaffold_atom_is_scaffold"][b] > 0.5
            valid_edges = []
            for edge in edges:
                a = int(edge[0].item())
                c = int(edge[1].item())
                if a < 0 or c < 0 or a >= atom_mask.shape[1] or c >= atom_mask.shape[1]:
                    continue
                if atom_mask[b, a] <= 0.5 or atom_mask[b, c] <= 0.5:
                    continue
                da = int(depths[a].item())
                dc = int(depths[c].item())
                if da >= 0 and dc >= 0 and da != dc:
                    src, dst = (a, c) if da < dc else (c, a)
                elif da == 0 and dc < 0:
                    src, dst = a, c
                elif dc == 0 and da < 0:
                    src, dst = c, a
                else:
                    src, dst = (a, c) if a <= c else (c, a)
                parent = int(roots[src].item())
                if parent < 0:
                    parent = int(roots[dst].item())
                if parent < 0:
                    continue
                src_is_scaffold = bool(scaffold_mask[src].item())
                dst_is_scaffold = bool(scaffold_mask[dst].item())
                if src_is_scaffold and not dst_is_scaffold:
                    kind = self.SIDECHAIN_KIND_ATTACH
                elif (not src_is_scaffold) and (not dst_is_scaffold):
                    kind = self.SIDECHAIN_KIND_BRANCH
                else:
                    kind = self.SIDECHAIN_KIND_MISC
                depth = max(0, min(int(depths[dst].item()), self.max_graph_depth))
                valid_edges.append((src, dst, parent, kind, depth))

            valid_edges = valid_edges[: self.max_sidechain_tokens]
            if not valid_edges:
                continue
            src_idx = torch.tensor([x[0] for x in valid_edges], device=device, dtype=torch.long)
            dst_idx = torch.tensor([x[1] for x in valid_edges], device=device, dtype=torch.long)
            parent_ids = torch.tensor([x[2] for x in valid_edges], device=device, dtype=torch.long)
            kind_ids = torch.tensor([x[3] for x in valid_edges], device=device, dtype=torch.long)
            depth_ids = torch.tensor([x[4] for x in valid_edges], device=device, dtype=torch.long)
            n_side = len(valid_edges)
            side_valid[b, :n_side] = 1.0
            side_parent[b, :n_side] = parent_ids
            side_type[b, :n_side] = atom_types[b, dst_idx].clamp(min=0).long()
            side_kind[b, :n_side] = kind_ids
            side_depth[b, :n_side] = depth_ids
            parent_center = batch["scaffold_system_center"][b, parent_ids].float().to(device)
            side_rel[b, :n_side] = coords[b, dst_idx].float() - parent_center

        return {
            "attachment_valid": att_valid,
            "attachment_parent": att_parent,
            "attachment_type": att_type,
            "attachment_rel": att_rel,
            "sidechain_valid": side_valid,
            "sidechain_parent": side_parent,
            "sidechain_type": side_type,
            "sidechain_kind": side_kind,
            "sidechain_depth": side_depth,
            "sidechain_rel": side_rel,
        }

    def forward(
        self,
        batch: dict,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> dict:
        """Encode GT scaffold labels into padded token sequences."""
        device = coords.device
        batch_tokens = []
        system_counts = []
        relation_counts = []
        attachment_counts = []
        sidechain_counts = []
        site_counts = []

        for b in range(coords.shape[0]):
            tokens = []
            system_tokens = self._encode_system_tokens(batch, b, device)
            relation_tokens = self._encode_relation_tokens(batch, b, device)
            attachment_tokens = self._encode_attachment_tokens(
                batch, coords, atom_types, atom_mask, b, device
            )
            sidechain_tokens = []
            if self.include_sidechain_tokens:
                sidechain_tokens = self._encode_sidechain_tokens(
                    batch, coords, atom_types, atom_mask, b, device
                )
            site_tokens = self._encode_site_tokens(
                batch, coords, atom_types, atom_mask, b, device
            )

            tokens.extend(system_tokens)
            tokens.extend(relation_tokens)
            tokens.extend(attachment_tokens)
            tokens.extend(sidechain_tokens)
            tokens.extend(site_tokens)

            system_counts.append(len(system_tokens))
            relation_counts.append(len(relation_tokens))
            attachment_counts.append(len(attachment_tokens))
            sidechain_counts.append(len(sidechain_tokens))
            site_counts.append(len(site_tokens))

            if not tokens:
                tokens = [torch.zeros(self.embed_dim, device=device)]

            batch_tokens.append(torch.stack(tokens, dim=0))

        max_len = max(t.shape[0] for t in batch_tokens)
        padded = torch.zeros(coords.shape[0], max_len, self.embed_dim, device=device)
        token_mask = torch.zeros(coords.shape[0], max_len, dtype=torch.bool, device=device)

        for b, tokens in enumerate(batch_tokens):
            n_tok = tokens.shape[0]
            padded[b, :n_tok] = tokens
            token_mask[b, :n_tok] = True

        padded = self.output_norm(padded)
        return {
            "tokens": padded,
            "token_mask": token_mask,
            "system_token_count": torch.tensor(system_counts, device=device, dtype=torch.long),
            "relation_token_count": torch.tensor(relation_counts, device=device, dtype=torch.long),
            "attachment_token_count": torch.tensor(attachment_counts, device=device, dtype=torch.long),
            "sidechain_token_count": torch.tensor(sidechain_counts, device=device, dtype=torch.long),
            "site_token_count": torch.tensor(site_counts, device=device, dtype=torch.long),
        }

    def forward_from_predicted_relations(
        self,
        batch: dict,
        predicted_relations: dict,
    ) -> dict:
        """Build scaffold tokens with GT ring-system backbone and predicted local relations."""
        device = predicted_relations["attachment_valid_logits"].device
        batch_tokens = []
        system_counts = []
        relation_counts = []
        attachment_counts = []
        sidechain_counts = []
        site_counts = []

        for b in range(predicted_relations["attachment_valid_logits"].shape[0]):
            tokens = []
            system_tokens = self._encode_system_tokens(batch, b, device)
            relation_tokens = self._encode_relation_tokens(batch, b, device)
            attachment_tokens = self._encode_predicted_attachment_tokens(
                batch, predicted_relations, b, device
            )
            sidechain_tokens = []
            if self.include_sidechain_tokens:
                sidechain_tokens = self._encode_predicted_sidechain_tokens(
                    batch, predicted_relations, b, device
                )
            site_tokens = self._encode_site_tokens(
                batch,
                coords=batch["coords"],
                atom_types=batch["atom_types"],
                atom_mask=batch["atom_mask"],
                b=b,
                device=device,
            )

            tokens.extend(system_tokens)
            tokens.extend(relation_tokens)
            tokens.extend(attachment_tokens)
            tokens.extend(sidechain_tokens)
            tokens.extend(site_tokens)

            system_counts.append(len(system_tokens))
            relation_counts.append(len(relation_tokens))
            attachment_counts.append(len(attachment_tokens))
            sidechain_counts.append(len(sidechain_tokens))
            site_counts.append(len(site_tokens))

            if not tokens:
                tokens = [torch.zeros(self.embed_dim, device=device)]

            batch_tokens.append(torch.stack(tokens, dim=0))

        max_len = max(t.shape[0] for t in batch_tokens)
        padded = torch.zeros(len(batch_tokens), max_len, self.embed_dim, device=device)
        token_mask = torch.zeros(len(batch_tokens), max_len, dtype=torch.bool, device=device)

        for b, tokens in enumerate(batch_tokens):
            n_tok = tokens.shape[0]
            padded[b, :n_tok] = tokens
            token_mask[b, :n_tok] = True

        padded = self.output_norm(padded)
        return {
            "tokens": padded,
            "token_mask": token_mask,
            "system_token_count": torch.tensor(system_counts, device=device, dtype=torch.long),
            "relation_token_count": torch.tensor(relation_counts, device=device, dtype=torch.long),
            "attachment_token_count": torch.tensor(attachment_counts, device=device, dtype=torch.long),
            "sidechain_token_count": torch.tensor(sidechain_counts, device=device, dtype=torch.long),
            "site_token_count": torch.tensor(site_counts, device=device, dtype=torch.long),
        }


class GTSlotSemanticConditionEncoder(nn.Module):
    """Encode GT atom-slot semantic labels into per-slot conditioning vectors.

    This is the first V18.3 "harder" conditioning path: each atom slot receives
    a semantic bias derived from its GT structural role instead of only sharing
    a global scaffold token bank via cross-attention.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        max_atoms: int = 85,
        num_atom_types: int = 10,
    ):
        super().__init__()
        self.max_atoms = max_atoms

        aux_dim = hidden_dim // 6
        self.role_embed = nn.Embedding(4, aux_dim)  # invalid + 3 roles
        self.parent_system_embed = nn.Embedding(MAX_RING_SYSTEMS + 2, aux_dim)
        self.site_index_embed = nn.Embedding(max_atoms + 2, aux_dim)
        self.attachment_target_embed = nn.Embedding(max_atoms + 2, aux_dim)
        self.sidechain_root_embed = nn.Embedding(max_atoms + 2, aux_dim)
        self.element_class_embed = nn.Embedding(num_atom_types + 1, aux_dim)

        self.proj = nn.Sequential(
            nn.Linear(aux_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _shift_with_invalid_bucket(x: torch.Tensor, max_valid: int) -> torch.Tensor:
        out = x.long().clamp(min=-1, max=max_valid) + 1
        return out

    def forward(self, batch: dict) -> torch.Tensor:
        mask = batch["atom_mask"] > 0.5
        device = batch["atom_mask"].device

        role = self._shift_with_invalid_bucket(batch["scaffold_atom_role"], 2)
        parent = self._shift_with_invalid_bucket(batch["scaffold_atom_parent_system_target"], MAX_RING_SYSTEMS)
        site_index = self._shift_with_invalid_bucket(batch["scaffold_atom_canonical_site_index"], self.max_atoms)
        attach_site = self._shift_with_invalid_bucket(batch["scaffold_atom_attachment_target_site"], self.max_atoms)
        side_root = self._shift_with_invalid_bucket(batch["scaffold_atom_sidechain_root_site"], self.max_atoms)
        element_cls = self._shift_with_invalid_bucket(batch["scaffold_atom_hetero_target_class"], num_atom_types := (self.element_class_embed.num_embeddings - 2))

        h = torch.cat(
            [
                self.role_embed(role),
                self.parent_system_embed(parent),
                self.site_index_embed(site_index),
                self.attachment_target_embed(attach_site),
                self.sidechain_root_embed(side_root),
                self.element_class_embed(element_cls),
            ],
            dim=-1,
        )
        cond = self.norm(self.proj(h))
        cond = cond * mask.unsqueeze(-1).float().to(device)
        return cond


class GTSlotConnectionGraphEncoder(nn.Module):
    """Build GT slot-level connection graph conditions.

    Edge types:
    - 0: no edge
    - 1: scaffold-local edge
    - 2: scaffold-to-sidechain / sidechain-local edge
    """

    EDGE_NONE = 0
    EDGE_LOCAL = 1
    EDGE_SIDECHAIN = 2

    def __init__(self, max_atoms: int = 85):
        super().__init__()
        self.max_atoms = max_atoms

    def _fill_edges(
        self,
        edge_types: torch.Tensor,
        batch: dict,
        edge_key: str,
        count_key: str,
        edge_type: int,
    ) -> None:
        if edge_key not in batch or count_key not in batch:
            return

        B, N, _ = edge_types.shape
        atom_mask = batch["atom_mask"] > 0.5
        for b in range(B):
            n_edges = int(batch[count_key][b].item())
            if n_edges <= 0:
                continue
            edges = batch[edge_key][b, :n_edges].long()
            for edge in edges:
                a = int(edge[0].item())
                c = int(edge[1].item())
                if a < 0 or c < 0 or a >= N or c >= N:
                    continue
                if not (atom_mask[b, a] and atom_mask[b, c]):
                    continue
                edge_types[b, a, c] = edge_type
                edge_types[b, c, a] = edge_type

    def forward(self, batch: dict) -> torch.Tensor:
        atom_mask = batch["atom_mask"]
        B, N = atom_mask.shape
        device = atom_mask.device
        edge_types = torch.zeros(B, N, N, device=device, dtype=torch.long)
        self._fill_edges(edge_types, batch, "scaffold_local_edges", "scaffold_n_local_edges", self.EDGE_LOCAL)
        self._fill_edges(edge_types, batch, "scaffold_sidechain_edges", "scaffold_n_sidechain_edges", self.EDGE_SIDECHAIN)
        return edge_types
