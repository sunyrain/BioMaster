"""Bounded, initially zero upgrades to a frozen V3 pair-ranking backbone."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class IncrementalConfig:
    hidden_dim: int = 384
    width: int = 96
    residual_bound: float = 4.0
    evidence_bound: float = 6.0
    evidence: bool = False
    pretrained: bool = False


class IncrementalV3(nn.Module):
    """Two direction-specific residuals retain an explicit frozen-score path.

    Evidence inputs follow support_summary_features: negative/positive maxima,
    means, availability and differences. Missing evidence contributes zero.
    Direct positive/negative evidence is monotonic with other inputs fixed.
    The learned neural correction is bounded and initialized to zero.
    """
    def __init__(self, config: IncrementalConfig):
        super().__init__()
        self.config = config
        self.adapter = nn.Sequential(nn.LayerNorm(config.hidden_dim + 1),
            nn.Linear(config.hidden_dim + 1, config.width), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(config.width, 2))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        if config.evidence:
            # clamp at zero has a usable subgradient, unlike a squared zero.
            self.positive_weight = nn.Parameter(torch.zeros(2, 2))
            self.negative_weight = nn.Parameter(torch.zeros(2, 2))
        if config.pretrained:
            self.drug = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 48), nn.GELU())
            self.target = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, 48), nn.GELU())
            self.interaction = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 2))
            nn.init.zeros_(self.interaction[-1].weight)
            nn.init.zeros_(self.interaction[-1].bias)

    def forward(self, base, hidden, support, drug=None, target=None):
        residual = self.adapter(torch.cat([hidden, base[:, None] / 5], dim=-1))
        if self.config.pretrained:
            if drug is None or target is None:
                raise ValueError('pretrained branch requires both aligned feature banks')
            d, t = self.drug(drug), self.target(target)
            residual = residual + self.interaction(torch.cat([d, t, d*t, (d-t).abs()], -1))
        delta = self.config.residual_bound * torch.tanh(residual / self.config.residual_bound)
        if self.config.evidence:
            positive = support[:, 1] * support[:, 5]
            negative = support[:, 0] * support[:, 4]
            pos = torch.stack([positive, positive.square()], -1)
            neg = torch.stack([negative, negative.square()], -1)
            evidence = 8 * (pos @ self.positive_weight.clamp(0, 2).T - neg @ self.negative_weight.clamp(0, 2).T)
            delta = delta + self.config.evidence_bound * torch.tanh(evidence / self.config.evidence_bound)
        return base[:, None] + delta

    @torch.no_grad()
    def constrain(self):
        if self.config.evidence:
            self.positive_weight.clamp_(0, 2)
            self.negative_weight.clamp_(0, 2)


def measured_query_loss(scores, labels, mask, rank_weight):
    """Equal-query BCE and all measured P/N pairs; padding is never labelled.

    Each input is [queries, max_items]. At least one valid item per query is
    required; a single-class query contributes BCE but no pairwise term.
    """
    clean_scores = torch.where(mask, scores, torch.zeros_like(scores))
    clean_labels = torch.where(mask, labels, torch.zeros_like(labels))
    bce = F.binary_cross_entropy_with_logits(clean_scores, clean_labels, reduction='none')
    bce = ((bce * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    positive = mask & (clean_labels == 1)
    negative = mask & (clean_labels == 0)
    valid_pairs = positive[:, :, None] & negative[:, None, :]
    # axis 1 holds positive candidates, axis 2 holds negative candidates.
    loss = F.softplus(clean_scores[:, None, :] - clean_scores[:, :, None])
    count = valid_pairs.sum((1, 2))
    valid_queries = count > 0
    ranks = (loss * valid_pairs).sum((1, 2)) / count.clamp_min(1)
    rank = ranks[valid_queries].mean() if valid_queries.any() else clean_scores.sum() * 0
    return bce + rank_weight * rank
