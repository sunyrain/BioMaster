"""Independent-pocket interaction trunk with geometry-biased pair reasoning.

The ligand and receptor are in independent frames. Only intra-molecular
geometry is consumed; cross distances are predicted, never fabricated from
unrelated coordinates. This is a development architecture, not a validated
replacement for the selected ranker.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class PocketPrecisionConfig:
    width: int = 384
    pair_width: int = 128
    heads: int = 8
    blocks: int = 6
    outer_width: int = 16
    dropout: float = 0.1
    distance_bins: int = 64
    distance_max: float = 32.0
    parent_width: int = 192
    checkpoint_blocks: bool = True

    def __post_init__(self):
        if self.width % self.heads or self.pair_width % self.heads:
            raise ValueError('node and pair widths must be divisible by heads')
        if min(self.blocks, self.heads, self.outer_width) < 1:
            raise ValueError('positive architecture dimensions required')

    def to_dict(self):
        return asdict(self)


def mean_masked(x, mask, dim):
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1)


def safe_softmax(x, mask, dim=-1):
    p = torch.softmax(x.float().masked_fill(~mask, -1e9), dim) * mask
    return (p / p.sum(dim, keepdim=True).clamp_min(1e-8)).to(x.dtype)


class Transition(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))

    def forward(self, x):
        return x + self.net(x)


class DistanceBias(nn.Module):
    def __init__(self, heads, extra=0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0, 32, 32))
        self.net = nn.Sequential(nn.Linear(32 + extra, 128), nn.SiLU(), nn.Linear(128, heads))

    def forward(self, distance, extra=None):
        rbf = torch.exp(-0.5 * (distance.float().unsqueeze(-1) - self.centers).square())
        if extra is not None:
            rbf = torch.cat([rbf, extra.float()], -1)
        return self.net(rbf).permute(0, 3, 1, 2)


class BiasedAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.depth = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.gate = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, x, mask, bias):
        b, n, w = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, self.depth).permute(2, 0, 3, 1, 4)
        # Every masked row still has a finite attention row; output masking below
        # also makes wholly missing pockets safe, without NaN softmax gradients.
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.depth) + bias
        attn = safe_softmax(logits, mask[:, None, None, :])
        attn = F.dropout(attn, self.dropout, self.training)
        out = (attn @ v).transpose(1, 2).reshape(b, n, w)
        return (x + self.out(out * torch.sigmoid(self.gate(h)))) * mask[..., None]


class PairBlock(nn.Module):
    """Outer-product update, two geometric triangle paths and axial attention."""
    def __init__(self, cfg):
        super().__init__()
        w, p, h, o = cfg.width, cfg.pair_width, cfg.heads, cfg.outer_width
        self.heads, self.depth = h, w // h
        self.atom_self = BiasedAttention(w, h, cfg.dropout)
        self.residue_self = BiasedAttention(w, h, cfg.dropout)
        self.atom_outer = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, o))
        self.residue_outer = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, o))
        self.outer_out = nn.Linear(o * o, p)
        self.pair_norm = nn.LayerNorm(p)
        self.triangle_value = nn.Linear(p, p)
        self.triangle_gate = nn.Linear(p, p)
        self.triangle_out = nn.Linear(2 * p, p)
        self.row_attention = BiasedAttention(p, h, cfg.dropout)
        self.column_attention = BiasedAttention(p, h, cfg.dropout)
        self.pair_transition = Transition(p, cfg.dropout)
        self.atom_q = nn.Linear(w, w, bias=False)
        self.residue_k = nn.Linear(w, w, bias=False)
        self.atom_value = nn.Linear(w, w)
        self.residue_value = nn.Linear(w, w)
        self.pair_bias = nn.Linear(p, h, bias=False)
        self.atom_out = nn.Linear(w, w)
        self.residue_out = nn.Linear(w, w)
        self.atom_transition = Transition(w, cfg.dropout)
        self.residue_transition = Transition(w, cfg.dropout)

    def forward(self, a, r, z, am, rm, ab, rb):
        a = self.atom_self(a, am, ab)
        r = self.residue_self(r, rm, rb)
        pm = am[:, :, None] & rm[:, None, :]
        outer = torch.einsum('bio,bjq->bijoq', self.atom_outer(a), self.residue_outer(r))
        z = (z + self.outer_out(outer.flatten(-2))) * pm[..., None]
        value = self.triangle_value(self.pair_norm(z)) * pm[..., None]
        # The A-A-R and A-R-R paths use the two valid internal geometries.
        aw = safe_softmax(ab.mean(1), am[:, None, :])
        rw = safe_softmax(rb.mean(1), rm[:, None, :])
        left = torch.einsum('bik,bkjc->bijc', aw, value)
        right = torch.einsum('bjk,bikc->bijc', rw, value)
        z = z + torch.sigmoid(self.triangle_gate(z)) * self.triangle_out(torch.cat([left, right], -1))
        b, na, nr, p = z.shape
        rows = z.reshape(b * na, nr, p)
        row_mask = pm.reshape(b * na, nr)
        rows = self.row_attention(rows, row_mask,
                                  rb[:, None].expand(-1, na, -1, -1, -1).reshape(b * na, self.heads, nr, nr))
        z = rows.reshape(b, na, nr, p).transpose(1, 2)
        cols = self.column_attention(z.reshape(b * nr, na, p), pm.transpose(1, 2).reshape(b * nr, na),
                                    ab[:, None].expand(-1, nr, -1, -1, -1).reshape(b * nr, self.heads, na, na))
        z = self.pair_transition(cols.reshape(b, nr, na, p).transpose(1, 2)) * pm[..., None]
        aq = self.atom_q(a).reshape(b, na, self.heads, self.depth).transpose(1, 2)
        rk = self.residue_k(r).reshape(b, nr, self.heads, self.depth).transpose(1, 2)
        logits = aq @ rk.transpose(-1, -2) / math.sqrt(self.depth) + self.pair_bias(z).permute(0, 3, 1, 2)
        av = self.atom_value(a).reshape(b, na, self.heads, self.depth).transpose(1, 2)
        rv = self.residue_value(r).reshape(b, nr, self.heads, self.depth).transpose(1, 2)
        ac = (safe_softmax(logits, pm[:, None]) @ rv).transpose(1, 2).reshape(b, na, -1)
        rc = (safe_softmax(logits.transpose(-1, -2), pm[:, None].transpose(-1, -2)) @ av).transpose(1, 2).reshape(b, nr, -1)
        a = self.atom_transition(a + self.atom_out(ac)) * am[..., None]
        r = self.residue_transition(r + self.residue_out(rc)) * rm[..., None]
        return a, r, z


class PocketPrecision(nn.Module):
    """Process every pocket separately, then learn drug-conditioned set pooling.

    Batch dimension enumerates (relation, pocket) records; `owner` maps them
    back to parent representations. No target or pocket identifier is embedded.
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or PocketPrecisionConfig()
        c = self.cfg
        self.atom_input = nn.Sequential(nn.LayerNorm(552), nn.Linear(552, c.width))
        self.residue_input = nn.Sequential(nn.LayerNorm(1792), nn.Linear(1792, c.width))
        self.atom_bias = DistanceBias(c.heads, extra=13)  # bond type / stereo
        self.residue_bias = DistanceBias(c.heads, extra=17)  # heavy-atom minimum and residue frames
        self.atom_pair = nn.Linear(c.width, c.pair_width)
        self.residue_pair = nn.Linear(c.width, c.pair_width)
        self.blocks = nn.ModuleList([PairBlock(c) for _ in range(c.blocks)])
        self.pool_attention = nn.Linear(c.pair_width, 1)
        self.local_output = nn.Sequential(nn.LayerNorm(2 * c.width + c.pair_width + 256),
                                          nn.Linear(2 * c.width + c.pair_width + 256, c.width),
                                          nn.GELU(), Transition(c.width, c.dropout))
        self.pocket_gate = nn.Sequential(nn.Linear(c.width + 2, 128), nn.GELU(), nn.Linear(128, 1))
        self.refine = nn.Sequential(nn.LayerNorm(c.width), nn.Linear(c.width, c.parent_width), nn.GELU(),
                                    nn.Linear(c.parent_width, c.parent_width))
        # Small, nonzero initialization allows first-step gradients throughout
        # the local trunk. The frozen parent itself is never optimizer-updated.
        nn.init.normal_(self.refine[-1].weight, std=0.001)
        nn.init.zeros_(self.refine[-1].bias)
        self.distance_head = nn.Sequential(nn.LayerNorm(c.pair_width), nn.Linear(c.pair_width, c.distance_bins))
        self.contact_head = nn.Sequential(nn.LayerNorm(c.pair_width), nn.Linear(c.pair_width, 1))

    def forward(self, batch, parent_state, return_aux=False):
        am, rm = batch['atom_mask'], batch['residue_mask']
        pm = am[:, :, None] & rm[:, None, :]
        a = self.atom_input(torch.cat([batch['atom_tokens'], batch['atom_chemistry']], -1)) * am[..., None]
        r = self.residue_input(torch.cat([batch['residue_tokens'], batch['pocket_residue_tokens']], -1)) * rm[..., None]
        ab = self.atom_bias(batch['atom_distance'], batch['atom_edges'])
        rb = self.residue_bias(batch['residue_distance'], batch['residue_geometry'])
        z = (self.atom_pair(a)[:, :, None] + self.residue_pair(r)[:, None]) * pm[..., None]
        for block in self.blocks:
            if self.training and self.cfg.checkpoint_blocks:
                a, r, z = checkpoint(block, a, r, z, am, rm, ab, rb, use_reentrant=False)
            else:
                a, r, z = block(a, r, z, am, rm, ab, rb)
        pw = safe_softmax(self.pool_attention(z).squeeze(-1).flatten(1), pm.flatten(1))
        pooled = (pw[..., None] * z.flatten(1, 2)).sum(1)
        local = self.local_output(torch.cat([mean_masked(a, am, 1), mean_masked(r, rm, 1), pooled,
                                            batch['drug_aligned'], batch['pocket_aligned']], -1))
        cosine = (batch['drug_aligned'] * batch['pocket_aligned']).sum(-1)
        gate = self.pocket_gate(torch.cat([local, batch['pocket_metadata']], -1)).squeeze(-1) + cosine / 0.07
        owner = batch['owner']
        available = am.any(1) & rm.any(1)
        pooled_relation = local.new_zeros((len(parent_state), self.cfg.width))
        weights = gate.new_zeros(gate.shape)
        relation_available = torch.zeros(len(parent_state), dtype=torch.bool, device=local.device)
        for i in range(len(parent_state)):
            take = owner == i
            w = safe_softmax(gate[take], available[take])
            pooled_relation[i] = (w[:, None] * local[take]).sum(0)
            weights[take] = w
            relation_available[i] = available[take].any()
        delta = self.refine(pooled_relation) * relation_available[:, None]
        state = parent_state + delta
        if not return_aux:
            return state
        return state, dict(delta=delta, pocket_weights=weights, pocket_gate=gate,
                           distance_logits=self.distance_head(z), contact_logits=self.contact_head(z).squeeze(-1),
                           pair_mask=pm, relation_available=relation_available)


def structural_loss(aux, distance, mask, cfg, contact_cutoff=4.5):
    """Masked atom-to-residue minimum-heavy-atom distance/contact supervision.

    `distance` must come from a mapped experimental complex, in one coordinate
    frame. Unknown pairs contribute exactly zero and are not treated as negatives.
    """
    valid = mask & aux['pair_mask']
    if not valid.any():
        return aux['distance_logits'].sum() * 0 + aux['contact_logits'].sum() * 0
    truth = distance[valid]
    if not torch.isfinite(truth).all() or (truth < 0).any():
        raise ValueError('invalid observed complex distances')
    bins = (truth / cfg.distance_max * (cfg.distance_bins - 1)).long().clamp_max(cfg.distance_bins - 1)
    dloss = F.cross_entropy(aux['distance_logits'][valid].float(), bins)
    closs = F.binary_cross_entropy_with_logits(aux['contact_logits'][valid].float(), (truth < contact_cutoff).float())
    return dloss + closs
