#!/usr/bin/env python3
"""Core model and losses for BioMaster-BiRoute V2.

The model keeps target-to-drug and drug-to-target retrieval asymmetric.  A
label-free route vector mixes those directional scores with a pair-interaction
expert.  There are deliberately no entity-ID embeddings, so an unseen entity
is represented only by its frozen molecular or protein foundation embedding.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualProjection(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class BioMasterBiRouteV2(nn.Module):
    """Asymmetric bidirectional retrieval with cold-regime route conditioning."""

    def __init__(
        self,
        drug_input_dim: int = 768,
        target_input_dim: int = 1280,
        latent_dim: int = 256,
        hidden_dim: int = 384,
        route_dim: int = 5,
        dropout: float = 0.15,
        route_conditioning: bool = True,
        symmetric_retrieval: bool = False,
    ) -> None:
        super().__init__()
        self.route_conditioning = route_conditioning
        self.symmetric_retrieval = symmetric_retrieval
        self.drug_encoder = ResidualProjection(drug_input_dim, latent_dim, dropout)
        self.target_encoder = ResidualProjection(target_input_dim, latent_dim, dropout)

        # Separate query/key maps are the central asymmetric design.  The
        # symmetric ablation reuses the target-to-drug maps in forward().
        self.target_query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.drug_key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.drug_query = nn.Linear(latent_dim, latent_dim, bias=False)
        self.target_key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.target_to_drug_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.drug_to_target_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

        pair_width = latent_dim * 4
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_width, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pair_head = nn.Linear(hidden_dim, 1)
        self.affinity_head = nn.Linear(hidden_dim, 1)
        gate_input = latent_dim * 2 + (route_dim if route_conditioning else 0)
        self.route_gate = nn.Sequential(
            nn.Linear(gate_input, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 3),
        )
        self.final_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _cosine_score(left: torch.Tensor, right: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
        left = torch.nn.functional.normalize(left, dim=1)
        right = torch.nn.functional.normalize(right, dim=1)
        scale = log_scale.clamp(math.log(1.0), math.log(100.0)).exp()
        return scale * (left * right).sum(dim=1)

    def forward(
        self,
        drug_features: torch.Tensor,
        target_features: torch.Tensor,
        route_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        drug = self.drug_encoder(drug_features)
        target = self.target_encoder(target_features)
        target_to_drug = self._cosine_score(
            self.target_query(target), self.drug_key(drug), self.target_to_drug_log_scale
        )
        if self.symmetric_retrieval:
            drug_to_target = target_to_drug
        else:
            drug_to_target = self._cosine_score(
                self.drug_query(drug), self.target_key(target), self.drug_to_target_log_scale
            )
        pair = torch.cat([drug, target, drug * target, torch.abs(drug - target)], dim=1)
        hidden = self.pair_trunk(pair)
        pair_score = self.pair_head(hidden).squeeze(1)
        gate_parts = [drug, target]
        if self.route_conditioning:
            gate_parts.append(route_features)
        route_weights = torch.softmax(self.route_gate(torch.cat(gate_parts, dim=1)), dim=1)
        expert_scores = torch.stack([pair_score, target_to_drug, drug_to_target], dim=1)
        final_logit = (route_weights * expert_scores).sum(dim=1) + self.final_bias
        return {
            "final_logit": final_logit,
            "pair_logit": pair_score,
            "target_to_drug_logit": target_to_drug,
            "drug_to_target_logit": drug_to_target,
            "affinity": self.affinity_head(hidden).squeeze(1),
            "route_weights": route_weights,
        }


def balanced_group_weights(group: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Inverse count within (query group, class), normalized to mean one."""
    weights = torch.empty_like(label, dtype=torch.float32)
    for value in torch.unique(group):
        selected = group.eq(value)
        for binary in (0, 1):
            part = selected & label.eq(binary)
            count = int(part.sum())
            if count:
                weights[part] = 1.0 / count
    return weights / weights.mean().clamp_min(1e-8)


def within_group_pairwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    maximum_pairs_per_group: int = 64,
) -> torch.Tensor:
    """Softplus pairwise loss over labelled positive/negative candidates.

    Capping the Cartesian product avoids allowing a large assay group to
    dominate an update.  The first pairs after a per-batch random permutation
    are used, so the cap samples different comparisons across epochs.
    """
    losses: list[torch.Tensor] = []
    unique_groups, counts = torch.unique(groups, return_counts=True)
    # Singleton queries cannot define a positive-negative ranking pair.  This
    # filter is also important for the old-drug direction, where most ChEMBL
    # compounds occur only once in a minibatch.
    for group in unique_groups[counts.gt(1)]:
        selected = groups.eq(group)
        positive = logits[selected & labels.gt(0.5)]
        negative = logits[selected & labels.lt(0.5)]
        if not positive.numel() or not negative.numel():
            continue
        differences = negative[:, None] - positive[None, :]
        if differences.numel() > maximum_pairs_per_group:
            differences = differences.flatten()[:maximum_pairs_per_group]
        losses.append(torch.nn.functional.softplus(differences).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def route_load_balance_loss(route_weights: torch.Tensor) -> torch.Tensor:
    """Small anti-collapse penalty; zero when mean route use is uniform."""
    mean_route = route_weights.mean(dim=0)
    uniform = torch.full_like(mean_route, 1.0 / mean_route.numel())
    return torch.mean((mean_route - uniform) ** 2)
