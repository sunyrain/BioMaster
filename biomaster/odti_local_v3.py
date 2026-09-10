"""Atom--sequence-region interaction backbone, with persistent pair states.

Sequence regions are coverage-preserving windows, not inferred binding pockets.
No ligand--protein spatial distance or pose is assumed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LocalInteractionConfigV3:
    atom_input_dim: int = 40
    residue_input_dim: int = 1280
    position_input_dim: int = 3
    width: int = 192
    hidden_dim: int = 192
    pair_dim: int = 32
    blocks: int = 2
    dropout: float = 0.10


def masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Normalized weights, including exact zero for entirely missing groups."""
    mask = mask.bool()
    weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


class _BondMessage(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.node = nn.Linear(width, width, bias=False)
        self.bond = nn.Embedding(6, width, padding_idx=0)
        self.stereo = nn.Embedding(7, width, padding_idx=0)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atoms: Tensor, bond_type: Tensor, bond_stereo: Tensor, mask: Tensor) -> Tensor:
        adjacency = bond_type.ne(0) & mask[:, :, None] & mask[:, None, :]
        weights = adjacency.to(atoms.dtype)
        messages = torch.bmm(weights, self.node(atoms))
        # Count edge categories before embedding. Materializing [B,A,A,width]
        # would waste hundreds of MB for dynamically padded large molecules.
        for category in range(1, self.bond.num_embeddings):
            counts = (bond_type.eq(category) & adjacency).sum(dim=2).to(atoms.dtype)
            messages = messages + counts.unsqueeze(-1) * self.bond.weight[category]
        for category in range(1, self.stereo.num_embeddings):
            counts = (bond_stereo.eq(category) & adjacency).sum(dim=2).to(atoms.dtype)
            messages = messages + counts.unsqueeze(-1) * self.stereo.weight[category]
        messages = messages / weights.sum(dim=2, keepdim=True).clamp_min(1)
        result = self.norm(atoms + self.dropout(torch.nn.functional.gelu(messages)))
        return result * mask.unsqueeze(-1)


class _PairBlock(nn.Module):
    def __init__(self, width: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_score = nn.Linear(pair_dim, 1)
        self.atom_update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.residue_update = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.atom_norm = nn.LayerNorm(width)
        self.residue_norm = nn.LayerNorm(width)
        self.atom_pair = nn.Linear(width, pair_dim)
        self.residue_pair = nn.Linear(width, pair_dim)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim, pair_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(pair_dim * 2, pair_dim))
        self.pair_norm = nn.LayerNorm(pair_dim)

    def forward(self, atoms: Tensor, residues: Tensor, pair: Tensor, atom_mask: Tensor, residue_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pair_mask = atom_mask[:, :, None] & residue_mask[:, None, :]
        scores = self.edge_score(pair).squeeze(-1)
        atom_context = torch.bmm(masked_softmax(scores, pair_mask, -1), residues)
        residue_context = torch.bmm(masked_softmax(scores, pair_mask, -2).transpose(1, 2), atoms)
        atoms = self.atom_norm(atoms + self.atom_update(torch.cat((atoms, atom_context), -1))) * atom_mask.unsqueeze(-1)
        residues = self.residue_norm(residues + self.residue_update(torch.cat((residues, residue_context), -1))) * residue_mask.unsqueeze(-1)
        # Keep the previous pair state; do not collapse to two pooled vectors.
        interaction = self.atom_pair(atoms).unsqueeze(2) * self.residue_pair(residues).unsqueeze(1)
        pair = self.pair_norm(pair + self.pair_update(pair + interaction)) * pair_mask.unsqueeze(-1)
        return atoms, residues, pair


class LocalInteractionBackboneV3(nn.Module):
    def __init__(self, config: LocalInteractionConfigV3 | None = None) -> None:
        super().__init__()
        self.config = config or LocalInteractionConfigV3()
        c = self.config
        if min(c.width, c.hidden_dim, c.pair_dim, c.blocks) < 1:
            raise ValueError("local dimensions and blocks must be positive")
        self.atom_input = nn.Sequential(nn.Linear(c.atom_input_dim, c.width), nn.LayerNorm(c.width), nn.GELU())
        self.residue_input = nn.Sequential(nn.Linear(c.residue_input_dim + c.position_input_dim, c.width), nn.LayerNorm(c.width), nn.GELU())
        self.bond_messages = nn.ModuleList([_BondMessage(c.width, c.dropout) for _ in range(2)])
        self.atom_pair = nn.Linear(c.width, c.pair_dim)
        self.residue_pair = nn.Linear(c.width, c.pair_dim)
        self.pair_norm = nn.LayerNorm(c.pair_dim)
        self.blocks = nn.ModuleList([_PairBlock(c.width, c.pair_dim, c.dropout) for _ in range(c.blocks)])
        self.pair_score = nn.Linear(c.pair_dim, 1)
        self.region_output = nn.Sequential(nn.Linear(c.width * 2 + c.pair_dim, c.hidden_dim), nn.LayerNorm(c.hidden_dim), nn.GELU())
        self.region_score = nn.Linear(c.hidden_dim, 1)

    def forward(
        self, atom_features: Tensor, bond_type: Tensor, bond_stereo: Tensor,
        atom_mask: Tensor, residue_features: Tensor, residue_positions: Tensor,
        residue_mask: Tensor, region_mask: Tensor, *, return_pair_states: bool = False,
    ) -> dict[str, Tensor]:
        """Inputs: atoms [B,A,F], residues [B,K,R,E], corresponding masks.

        Outputs are ``local_hidden[B,H]`` and ``local_available[B]``. Entirely
        missing rows have exactly zero hidden states, including during training.
        """
        if atom_features.ndim != 3 or residue_features.ndim != 4:
            raise ValueError("expected atoms [B,A,F] and residues [B,K,R,E]")
        b, a, _ = atom_features.shape
        rb, k, r, _ = residue_features.shape
        if b != rb or min(a, k, r) < 1:
            raise ValueError("batch dimensions must agree and padded dimensions must be nonempty")
        if atom_mask.shape != (b, a) or residue_mask.shape != (b, k, r) or region_mask.shape != (b, k):
            raise ValueError("local feature mask shape mismatch")
        if bond_type.shape != (b, a, a) or bond_stereo.shape != (b, a, a):
            raise ValueError("bond tensors must be [B,A,A]")
        am = atom_mask.bool()
        rm = residue_mask.bool() & region_mask.bool().unsqueeze(-1)
        atoms = self.atom_input(atom_features.masked_fill(~am.unsqueeze(-1), 0)) * am.unsqueeze(-1)
        bt = bond_type.long().masked_fill(~(am[:, :, None] & am[:, None, :]), 0)
        bs = bond_stereo.long().masked_fill(bt.eq(0), 0)
        for message in self.bond_messages:
            atoms = message(atoms, bt, bs, am)
        residues = torch.cat((residue_features, residue_positions), -1).masked_fill(~rm.unsqueeze(-1), 0)
        residues = self.residue_input(residues).reshape(b * k, r, -1)
        residues = residues * rm.reshape(b * k, r, 1)
        atoms = atoms[:, None].expand(-1, k, -1, -1).reshape(b * k, a, -1)
        am = am[:, None].expand(-1, k, -1).reshape(b * k, a)
        rm_flat = rm.reshape(b * k, r)
        ap = self.atom_pair(atoms).unsqueeze(2)
        rp = self.residue_pair(residues).unsqueeze(1)
        pm = am[:, :, None] & rm_flat[:, None, :]
        pair = self.pair_norm(ap + rp + ap * rp) * pm.unsqueeze(-1)
        for block in self.blocks:
            atoms, residues, pair = block(atoms, residues, pair, am, rm_flat)
        pair_weights = masked_softmax(self.pair_score(pair).squeeze(-1).flatten(1), pm.flatten(1), -1).reshape(b * k, a, r)
        atom_pool = (atoms * pair_weights.sum(-1).unsqueeze(-1)).sum(1)
        residue_pool = (residues * pair_weights.sum(-2).unsqueeze(-1)).sum(1)
        pair_pool = (pair * pair_weights.unsqueeze(-1)).sum((1, 2))
        regions = self.region_output(torch.cat((atom_pool, residue_pool, pair_pool), -1)).reshape(b, k, -1)
        available_regions = rm.any(-1) & atom_mask.bool().any(-1, keepdim=True)
        region_weights = masked_softmax(self.region_score(regions).squeeze(-1), available_regions, -1)
        local_available = available_regions.any(-1)
        local_hidden = (regions * region_weights.unsqueeze(-1)).sum(1) * local_available.unsqueeze(-1)
        result = {"local_hidden": local_hidden, "local_available": local_available, "region_weights": region_weights}
        if return_pair_states:
            result["pair_states"] = pair.reshape(b, k, a, r, -1)
        return result
