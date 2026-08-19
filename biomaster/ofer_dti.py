"""Observation-aware first-event DTI model components.

The OFER-DTI branch models two different processes:

* ``q_active(d, t)``: biochemical activity conditional on an observation;
* ``h_observation(d, t, year)``: the probability that the pair receives its
  first recorded binding observation in a historical year.

Rows before a first event are therefore survival rows, not biochemical
inactive labels.  This module is deliberately independent of the large local
risk-set artifacts so that forward and loss behavior can be unit-tested on
small synthetic tensors before the Phase-A trainer is opened.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OFERDTIConfig:
    drug_input_dim: int = 2048
    target_input_dim: int = 1024
    latent_dim: int = 256
    bilinear_rank: int = 32
    hidden_dim: int = 128
    dropout: float = 0.12
    observation_weight: float = 1.0
    active_weight: float = 1.0
    drug_rank_weight: float = 0.25
    target_rank_weight: float = 0.10
    pair_age_default: float = 0.0


class OFERDTIModel(nn.Module):
    """Dual-process first-event DTI model with low-rank pair interaction."""

    def __init__(self, config: OFERDTIConfig | None = None) -> None:
        super().__init__()
        self.config = config or OFERDTIConfig()
        cfg = self.config
        if cfg.latent_dim < 1 or cfg.bilinear_rank < 1:
            raise ValueError("latent_dim and bilinear_rank must be positive")

        self.drug_encoder = nn.Sequential(
            nn.Linear(cfg.drug_input_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(cfg.target_input_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.drug_bilinear = nn.Linear(cfg.latent_dim, cfg.bilinear_rank, bias=False)
        self.target_bilinear = nn.Linear(cfg.latent_dim, cfg.bilinear_rank, bias=False)
        pair_width = cfg.latent_dim * 4 + cfg.bilinear_rank
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_width, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.active_head = nn.Linear(cfg.hidden_dim, 1)
        self.observation_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 4, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )
        self.direct_active_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim + 4, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    @staticmethod
    def time_basis(calendar_year: Tensor, pair_age: Tensor | None = None) -> Tensor:
        year = calendar_year.float().reshape(-1)
        if pair_age is None:
            age = torch.zeros_like(year)
        else:
            age = pair_age.float().reshape(-1)
            if age.shape != year.shape:
                raise ValueError("pair_age must match calendar_year")
        normalized_year = (year - 2000.0) / 25.0
        normalized_age = torch.log1p(age.clamp_min(0.0)) / 4.0
        return torch.stack(
            [normalized_year, normalized_year.square(), normalized_age, normalized_age.square()],
            dim=1,
        )

    def encode_pair(self, drug: Tensor, target: Tensor) -> dict[str, Tensor]:
        if drug.ndim != 2 or target.ndim != 2 or drug.shape[0] != target.shape[0]:
            raise ValueError("drug and target must be rank-2 tensors with equal batch size")
        drug_embedding = self.drug_encoder(drug)
        target_embedding = self.target_encoder(target)
        pair = torch.cat(
            [
                drug_embedding,
                target_embedding,
                drug_embedding * target_embedding,
                torch.abs(drug_embedding - target_embedding),
                self.drug_bilinear(drug_embedding) * self.target_bilinear(target_embedding),
            ],
            dim=1,
        )
        hidden = self.pair_trunk(pair)
        return {
            "drug_embedding": drug_embedding,
            "target_embedding": target_embedding,
            "pair_hidden": hidden,
            "active_logit": self.active_head(hidden).squeeze(1),
        }

    def forward(
        self,
        drug: Tensor,
        target: Tensor,
        calendar_year: Tensor,
        pair_age: Tensor | None = None,
    ) -> dict[str, Tensor]:
        encoded = self.encode_pair(drug, target)
        time = self.time_basis(calendar_year, pair_age)
        time_pair = torch.cat([encoded["pair_hidden"], time], dim=1)
        encoded["time_basis"] = time
        encoded["observation_logit"] = self.observation_head(time_pair).squeeze(1)
        encoded["direct_active_logit"] = self.direct_active_head(time_pair).squeeze(1)
        return encoded


def _weighted_bce(logits: Tensor, labels: Tensor, weights: Tensor | None = None) -> Tensor:
    value = F.binary_cross_entropy_with_logits(
        logits.reshape(-1), labels.float().reshape(-1), reduction="none"
    )
    if weights is None:
        return value.mean()
    weights = weights.float().reshape(-1)
    if weights.shape != value.shape:
        raise ValueError("weights must match labels")
    return (value * weights).sum() / weights.sum().clamp_min(1e-8)


def _weighted_mean(value: Tensor, weights: Tensor | None = None) -> Tensor:
    value = value.reshape(-1)
    if weights is None:
        return value.mean()
    weights = weights.float().reshape(-1)
    if weights.shape != value.shape:
        raise ValueError("weights must match value")
    return (value * weights).sum() / weights.sum().clamp_min(1e-8)


def _within_group_rank_loss(
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    losses: list[Tensor] = []
    for group in torch.unique(groups.reshape(-1)):
        selected = groups.reshape(-1).eq(group)
        positive = logits[selected & labels.reshape(-1).gt(0.5)]
        negative = logits[selected & labels.reshape(-1).lt(0.5)]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        pair_loss = F.softplus(negative.reshape(1, -1) - positive.reshape(-1, 1))
        if weights is not None:
            pair_weights = weights.reshape(-1)[selected & labels.reshape(-1).gt(0.5)]
            pair_weights = pair_weights.reshape(-1, 1).expand_as(pair_loss)
            pair_loss = pair_loss * pair_weights
        losses.append(pair_loss.mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _within_group_listwise_loss(
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    weights: Tensor | None = None,
) -> Tensor:
    """Softmax mass assigned to active events within a query-year group."""

    values: list[Tensor] = []
    labels = labels.reshape(-1)
    groups = groups.reshape(-1)
    weights_flat = weights.reshape(-1) if weights is not None else None
    for group in torch.unique(groups):
        selected = groups.eq(group)
        group_logits = logits.reshape(-1)[selected]
        group_labels = labels[selected]
        positive = group_labels.gt(0.5)
        negative = group_labels.lt(0.5)
        if not positive.any() or not negative.any():
            continue
        denominator = torch.logsumexp(group_logits, dim=0)
        numerator = torch.logsumexp(group_logits[positive], dim=0)
        value = -(numerator - denominator)
        if weights_flat is not None:
            value = value * weights_flat[selected].mean()
        values.append(value)
    if not values:
        return logits.sum() * 0.0
    return torch.stack(values).mean()


def ofer_phase_a_loss(
    outputs: dict[str, Tensor],
    observation_event: Tensor,
    active_given_observed: Tensor,
    drug_group: Tensor,
    target_group: Tensor,
    sample_weight: Tensor | None = None,
    sampling_role: Tensor | None = None,
    drug_query_group: Tensor | None = None,
    target_query_group: Tensor | None = None,
    variant: str = "OFER_FULL",
    config: OFERDTIConfig | None = None,
) -> dict[str, Tensor]:
    """Compute the frozen Phase-A dual-process objective.

    ``active_given_observed`` must be ``-1`` for survival/unobserved rows;
    those rows contribute to the observation hazard but not to conditional
    activity BCE or ranking losses.  ``variant`` follows the frozen Phase-A
    controls: ``OFER_FULL``, ``STATIC_OBSERVED_ONLY_BCE``,
    ``STATIC_FNML_STYLE``, ``DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD`` and
    ``TIMESTAMP_SHUFFLED_OFER``.
    """

    cfg = config or OFERDTIConfig()
    allowed = {
        "OFER_FULL",
        "STATIC_OBSERVED_ONLY_BCE",
        "STATIC_FNML_STYLE",
        "DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD",
        "TIMESTAMP_SHUFFLED_OFER",
    }
    if variant not in allowed:
        raise ValueError(f"unknown OFER variant: {variant}")
    observed = observation_event.reshape(-1).gt(0.5)
    active = active_given_observed.reshape(-1)
    activity_mask = observed & active.ge(0.0)
    if activity_mask.any():
        active_loss = _weighted_bce(
            outputs["active_logit"][activity_mask],
            active[activity_mask],
            sample_weight[activity_mask] if sample_weight is not None else None,
        )
        drug_rank = _within_group_listwise_loss(
            outputs["active_logit"][activity_mask],
            active[activity_mask],
            (drug_query_group if drug_query_group is not None else drug_group)[activity_mask],
            sample_weight[activity_mask] if sample_weight is not None else None,
        )
        target_rank = _within_group_listwise_loss(
            outputs["active_logit"][activity_mask],
            active[activity_mask],
            (target_query_group if target_query_group is not None else target_group)[activity_mask],
            sample_weight[activity_mask] if sample_weight is not None else None,
        )
    else:
        active_loss = outputs["active_logit"].sum() * 0.0
        drug_rank = active_loss
        target_rank = active_loss
    observation_loss = _weighted_bce(outputs["observation_logit"], observation_event, sample_weight)
    direct_active_labels = (observed & active.eq(1.0)).float()
    direct_active_loss = _weighted_bce(
        outputs["direct_active_logit"], direct_active_labels, sample_weight
    )

    # Exact first-event competing-risk likelihood.  Observed active rows use
    # h_obs*q_active, observed weak/inactive rows use h_obs*(1-q_active), and
    # pre-event rows use the right-censored survival term (1-h_obs).
    hazard = outputs["observation_logit"].sigmoid().clamp(1e-6, 1.0 - 1e-6)
    q_active = outputs["active_logit"].sigmoid().clamp(1e-6, 1.0 - 1e-6)
    valid_event = (~observed) | activity_mask
    event_nll = torch.zeros_like(hazard)
    event_nll[~observed] = -torch.log1p(-hazard[~observed])
    active_observed = observed & active.eq(1.0)
    inactive_observed = observed & active.eq(0.0)
    event_nll[active_observed] = -torch.log(hazard[active_observed] * q_active[active_observed])
    event_nll[inactive_observed] = -torch.log(
        hazard[inactive_observed] * (1.0 - q_active[inactive_observed])
    )
    competing_event = _weighted_mean(
        event_nll[valid_event],
        sample_weight.reshape(-1)[valid_event] if sample_weight is not None else None,
    )

    static_observation = torch.ones_like(observed, dtype=torch.bool)
    if sampling_role is not None:
        static_observation = sampling_role.reshape(-1).ne(1)
    static_fnml_observation = _weighted_bce(
        outputs["observation_logit"][static_observation],
        observation_event.reshape(-1)[static_observation],
        sample_weight.reshape(-1)[static_observation] if sample_weight is not None else None,
    ) if static_observation.any() else outputs["observation_logit"].sum() * 0.0

    if variant in {"OFER_FULL", "TIMESTAMP_SHUFFLED_OFER"}:
        observation_component = competing_event
        total = cfg.observation_weight * competing_event
        total = total + cfg.drug_rank_weight * drug_rank + cfg.target_rank_weight * target_rank
    elif variant == "STATIC_OBSERVED_ONLY_BCE":
        observation_component = outputs["observation_logit"].sum() * 0.0
        total = cfg.active_weight * active_loss
        total = total + cfg.drug_rank_weight * drug_rank + cfg.target_rank_weight * target_rank
    elif variant == "STATIC_FNML_STYLE":
        observation_component = static_fnml_observation
        total = (
            cfg.active_weight * active_loss
            + cfg.observation_weight * static_fnml_observation
            + cfg.drug_rank_weight * drug_rank
            + cfg.target_rank_weight * target_rank
        )
    else:  # DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD
        observation_component = outputs["observation_logit"].sum() * 0.0
        total = cfg.active_weight * direct_active_loss
    return {
        "total": total,
        "observation": observation_loss,
        "active": active_loss,
        "competing_event": competing_event,
        "direct_active": direct_active_loss,
        "static_fnml_observation": static_fnml_observation,
        "drug_rank": drug_rank,
        "target_rank": target_rank,
    }


def ofer_discovery_score(
    active_logit: Tensor,
    observation_hazards: Tensor,
) -> Tensor:
    """Compute ``q_active × (1 - cumulative survival)`` over year steps."""

    if active_logit.ndim != 1 or observation_hazards.ndim != 2:
        raise ValueError("active_logit must be [pairs], hazards must be [pairs, years]")
    if observation_hazards.shape[0] != active_logit.shape[0]:
        raise ValueError("hazard and active batch dimensions must match")
    hazard = observation_hazards.sigmoid().clamp(0.0, 1.0)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    return active_logit.sigmoid() * (1.0 - survival[:, -1])


__all__ = [
    "OFERDTIConfig",
    "OFERDTIModel",
    "ofer_phase_a_loss",
    "ofer_discovery_score",
]
