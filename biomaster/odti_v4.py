"""V4 R1 global pair model and an explicitly scoped signed-evidence ablation.

This module does not implement the planned DrugCLIP geometry, residue-token
interaction, stereo adapter, endpoint heads, or backbone fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GlobalConfigV4:
    drug_dim: int = 768
    target_dim: int = 1280
    hidden: int = 256
    dropout: float = 0.1
    evidence_cap: float = 2.0


class GlobalPairV4(nn.Module):
    def __init__(self, config: GlobalConfigV4):
        super().__init__()
        self.config = config
        h = config.hidden
        self.drug = nn.Sequential(nn.LayerNorm(config.drug_dim), nn.Linear(config.drug_dim, h), nn.GELU(), nn.LayerNorm(h))
        self.target = nn.Sequential(nn.LayerNorm(config.target_dim), nn.Linear(config.target_dim, h), nn.GELU(), nn.LayerNorm(h))
        self.pair = nn.Sequential(nn.Linear(4*h, 2*h), nn.GELU(), nn.Dropout(config.dropout),
                                  nn.Linear(2*h, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, 1))
        self.drug_bilinear = nn.Linear(h, 64, bias=False)
        self.target_bilinear = nn.Linear(h, 64, bias=False)

    def potential(self, drug_state, target_state):
        pair = torch.cat([drug_state, target_state, drug_state * target_state, (drug_state-target_state).abs()], dim=-1)
        return self.pair(pair).squeeze(-1) + (self.drug_bilinear(drug_state) * self.target_bilinear(target_state)).sum(-1) / 8

    def forward(self, drug, target):
        return self.potential(self.drug(drug), self.target(target))


class SignedEvidenceV4(nn.Module):
    """Independent positive bonus and negative penalty, each bounded by cap.

    Negative inputs never enter the base or positive branch. Reference deltas
    are differences of one deterministic shared pair potential, not independent
    pairwise regressors. They are binary-task evidence, not affinity differences.
    """
    def __init__(self, base: GlobalPairV4):
        super().__init__()
        self.base = base
        h = base.config.hidden
        self.project = nn.Linear(h, 32)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(32*4 + 2, 64), nn.GELU(), nn.Linear(64, 2)) for _ in range(2)
        ])
        for branch in self.branches:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self.register_buffer('ramp', torch.tensor(1.))

    def forward(self, drug, target, references, similarities, mask):
        if references.ndim != 4 or references.shape[1] != 2 or mask.shape != similarities.shape or mask.shape != references.shape[:3]:
            raise ValueError('references [B,negative/positive,K,D], similarities/mask [B,2,K] required')
        # Deterministic shared potentials preserve exact reference cycles even
        # during evidence training. Gradients still flow in joint adaptation.
        self.base.eval()
        q, t = self.base.drug(drug), self.base.target(target)
        valid = mask.bool()
        raw = torch.where(valid[..., None], references, torch.zeros_like(references))
        r = self.base.drug(raw)
        return self.forward_states(q, t, r, similarities, mask)

    def forward_states(self, q, t, references, similarities, mask, reference_potential=None):
        """Equivalent evaluation path using freshly computed frozen states."""
        base = self.base.potential(q, t)
        effects, deltas = [], []
        for label in range(2):
            valid = mask[:, label].bool()
            # Sanitize padding before any projection: arbitrary/NaN padding
            # cannot affect a valid score or contaminate gradient computation.
            r = torch.where(valid[..., None], references[:, label], torch.zeros_like(references[:, label]))
            tt = t[:, None].expand_as(r)
            refscore = self.base.potential(r, tt) if reference_potential is None else reference_potential[:, label]
            delta = base[:, None] - torch.where(valid, refscore, torch.zeros_like(refscore))
            sim = torch.where(valid, similarities[:, label], torch.zeros_like(similarities[:, label])).clamp(0, 1)
            qr, rr, tr = self.project(q)[:, None].expand(-1, r.shape[1], -1), self.project(r), self.project(t)[:, None].expand(-1, r.shape[1], -1)
            state = torch.cat([qr-rr, qr*rr, tr, rr*tr, sim[..., None], delta[..., None]], dim=-1)
            outputs = self.branches[label](state)
            logits = (outputs[..., 0] + sim * 5).masked_fill(~valid, -1e4)
            weight = torch.softmax(logits, dim=-1) * valid
            weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
            effect = self.base.config.evidence_cap * (weight * sim * torch.sigmoid(outputs[..., 1])).sum(-1)
            effects.append(effect * self.ramp)
            deltas.append(torch.where(valid, delta, torch.zeros_like(delta)))
        negative, positive = effects
        return {'logit': base + positive - negative, 'base_logit': base,
                'positive_bonus': positive, 'negative_penalty': negative,
                'reference_delta': torch.stack(deltas, dim=1)}


__all__ = ['GlobalConfigV4', 'GlobalPairV4', 'SignedEvidenceV4']
