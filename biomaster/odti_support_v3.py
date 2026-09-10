"""Trainable interaction ranking with separate active/inactive support sets.

The support bank and all entity/fold exclusions belong to the data loader.
This module receives explicit masks and never samples evidence or invents a
negative label.  Channel 0 is inactive support and channel 1 is active support.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2


@dataclass(frozen=True)
class SupportV3Config:
    hidden_dim: int = 192
    support_enabled: bool = True
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least two")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _projection(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class SupportInteractionRankerV3(nn.Module):
    """A freshly initialized V2 topology plus a learned evidence interaction.

    ``local_hidden`` has width ``support_config.hidden_dim``.  Where
    ``local_available`` is true, a new local head *replaces* the global score;
    it is not an old-score residual.  Support evidence then augments the
    selected primary path.  With no support, ``final_logit == base_logit``
    exactly, including rows with one missing local modality.

    All arguments not specific to V3 are forwarded to the V2 model, including
    ``target_aux``, ``target_aux_mask`` and optional structure inputs.  This
    class never loads a checkpoint or freezes its base.
    """

    def __init__(
        self,
        base_config: ODTIV2Config,
        family_count: int,
        support_config: SupportV3Config | None = None,
    ) -> None:
        super().__init__()
        self.base_config = base_config
        self.support_config = support_config or SupportV3Config()
        self.base = RoutedInteractionRankerV2(
            family_count=family_count, config=base_config, use_conplex=False
        )
        width = self.support_config.hidden_dim
        embedding = base_config.embedding_dim
        dropout = self.support_config.dropout
        self.query_projection = _projection(
            embedding * 5 + base_config.family_dim, width, dropout
        )
        self.support_projection = _projection(embedding * 3 + 1, width, dropout)
        self.support_attention_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1)
                )
                for _ in range(2)
            ]
        )
        # Six quality features: availability, mean similarity and maximum
        # similarity for each class.  No raw target identity/degree feature.
        self.support_gate_head = nn.Sequential(
            nn.Linear(width + 6, width), nn.GELU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.support_gate_head[-1].weight)
        nn.init.zeros_(self.support_gate_head[-1].bias)
        self.support_delta_head = nn.Sequential(
            _projection(width * 4 + 6, width, dropout), nn.Linear(width, 1)
        )
        self.local_head = nn.Linear(width, 1)
        self.local_affinity_head = nn.Linear(width, 1)
        self.local_affinity_log_variance_head = nn.Linear(width, 1)

    def _query_hidden(self, output: dict[str, Tensor], family: Tensor) -> Tensor:
        drug = output["interaction_drug_embedding"]
        target = output["interaction_target_embedding"]
        return self.query_projection(
            torch.cat(
                [
                    drug,
                    target,
                    drug * target,
                    (drug - target).abs(),
                    self.base.bilinear(drug, target),
                    self.base.family_embedding(family),
                ],
                dim=-1,
            )
        )

    def forward(
        self,
        drug: Tensor,
        target: Tensor,
        family: Tensor,
        *,
        support_drug_features: Tensor | None = None,
        support_similarity: Tensor | None = None,
        support_mask: Tensor | None = None,
        support_keep_mask: Tensor | None = None,
        local_hidden: Tensor | None = None,
        local_available: Tensor | None = None,
        **base_kwargs: Tensor | None,
    ) -> dict[str, Tensor]:
        """Score a batch; evidence dropout is exclusively caller-controlled.

        Supports are ``[B, 2, K, drug_input_dim]``.  Their similarities and
        masks are ``[B, 2, K]``.  Optional ``support_keep_mask`` can be ``[B]``,
        ``[B, 2]`` or ``[B, 2, K]``.  Masked feature/similarity padding is
        ignored even if it contains NaN.  Eval scores do not depend on support
        ordering, candidate ordering, padding length or batch partitioning.
        """
        output = self.base(drug, target, family, **base_kwargs)
        global_logit = output["final_logit"]
        hidden = self._query_hidden(output, family)
        batch_size = drug.shape[0]
        local_mask = torch.zeros(batch_size, dtype=torch.bool, device=drug.device)
        base_logit = global_logit
        if local_hidden is not None:
            if local_hidden.shape != (batch_size, self.support_config.hidden_dim):
                raise ValueError("local_hidden must be [B, support_config.hidden_dim]")
            if local_available is None:
                local_mask = torch.ones_like(local_mask)
            else:
                if local_available.shape != (batch_size,):
                    raise ValueError("local_available must be [B]")
                local_mask = local_available.to(device=drug.device).gt(0.5)
            clean_local = torch.where(
                local_mask[:, None], local_hidden.to(hidden), torch.zeros_like(hidden)
            )
            if not torch.isfinite(clean_local).all():
                raise ValueError("available local_hidden entries must be finite")
            hidden = torch.where(local_mask[:, None], clean_local, hidden)
            base_logit = torch.where(
                local_mask, self.local_head(clean_local).squeeze(-1), global_logit
            )
            output["affinity"] = torch.where(
                local_mask,
                self.local_affinity_head(clean_local).squeeze(-1),
                output["affinity"],
            )
            local_variance = self.local_affinity_log_variance_head(clean_local)
            output["affinity_log_variance"] = torch.where(
                local_mask,
                local_variance.squeeze(-1).clamp(
                    self.base_config.affinity_min_log_variance,
                    self.base_config.affinity_max_log_variance,
                ),
                output["affinity_log_variance"],
            )
        elif local_available is not None:
            if local_available.shape != (batch_size,):
                raise ValueError("local_available must be [B]")
            if local_available.gt(0.5).any():
                raise ValueError("available local rows require local_hidden")

        support_gate = torch.zeros_like(base_logit)
        support_delta = torch.zeros_like(base_logit)
        attention = hidden.new_zeros((batch_size, 2, 0))
        evidence_mask = torch.zeros((batch_size, 2, 0), dtype=torch.bool, device=drug.device)
        class_available = torch.zeros((batch_size, 2), dtype=torch.bool, device=drug.device)
        any_argument = any(
            value is not None
            for value in (support_drug_features, support_similarity, support_mask)
        )
        if self.support_config.support_enabled and any_argument:
            if any(
                value is None
                for value in (support_drug_features, support_similarity, support_mask)
            ):
                raise ValueError("support features, similarity and mask must be supplied together")
            if (
                support_drug_features.ndim != 4
                or support_drug_features.shape[:2] != (batch_size, 2)
                or support_drug_features.shape[-1] != self.base_config.drug_input_dim
            ):
                raise ValueError("support_drug_features must be [B, 2, K, drug_input_dim]")
            shape = support_drug_features.shape[:3]
            if support_similarity.shape != shape or support_mask.shape != shape:
                raise ValueError("support similarity and mask must be [B, 2, K]")
            evidence_mask = support_mask.to(device=drug.device).gt(0.5)
            if support_keep_mask is not None:
                keep = support_keep_mask.to(device=drug.device).gt(0.5)
                if keep.shape == (batch_size,):
                    keep = keep[:, None, None]
                elif keep.shape == (batch_size, 2):
                    keep = keep[:, :, None]
                elif keep.shape != shape:
                    raise ValueError("support_keep_mask must be [B], [B, 2] or [B, 2, K]")
                evidence_mask = evidence_mask & keep
            attention = hidden.new_zeros(shape)
            class_available = evidence_mask.any(dim=-1)
            if evidence_mask.any():
                features = support_drug_features.to(drug)
                similarity = torch.where(
                    evidence_mask, support_similarity.to(drug), torch.zeros_like(attention)
                )
                valid_features = features[evidence_mask]
                if not torch.isfinite(valid_features).all() or not torch.isfinite(similarity).all():
                    raise ValueError("valid support features and similarities must be finite")
                # Only encode real records; padding cannot consume encoder
                # compute or introduce NaNs into linear-layer gradients.
                encoded = self.base.drug_encoder(valid_features)
                support_encoded = encoded.new_zeros((*shape, encoded.shape[-1]))
                support_encoded[evidence_mask] = encoded
                query = output["drug_base_embedding"][:, None, None, :]
                values = self.support_projection(
                    torch.cat(
                        [
                            support_encoded,
                            support_encoded - query,
                            support_encoded * query,
                            similarity[..., None],
                        ],
                        dim=-1,
                    )
                )
                contexts = []
                for label, head in enumerate(self.support_attention_heads):
                    class_values = values[:, label]
                    query_context = hidden[:, None, :].expand_as(class_values)
                    scores = head(torch.cat([query_context, class_values], dim=-1)).squeeze(-1)
                    mask = evidence_mask[:, label]
                    # Finite sentinel plus renormalization makes all-masked
                    # classes exactly zero without an all-minus-inf softmax.
                    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                    weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                        torch.finfo(weights.dtype).tiny
                    )
                    attention[:, label] = weights
                    contexts.append((weights[..., None] * class_values).sum(dim=1))
                count = evidence_mask.sum(dim=-1).clamp_min(1)
                mean_similarity = similarity.sum(dim=-1) / count
                max_similarity = similarity.masked_fill(~evidence_mask, -torch.inf).amax(dim=-1)
                max_similarity = torch.where(class_available, max_similarity, torch.zeros_like(max_similarity))
                quality = torch.cat(
                    [class_available.to(hidden.dtype), mean_similarity, max_similarity], dim=-1
                )
                available = class_available.any(dim=-1)
                raw_gate = torch.sigmoid(
                    self.support_gate_head(torch.cat([hidden, quality], dim=-1)).squeeze(-1)
                )
                raw_delta = self.support_delta_head(
                    torch.cat(
                        [hidden, contexts[1], contexts[0], contexts[1] - contexts[0], quality],
                        dim=-1,
                    )
                ).squeeze(-1)
                support_gate = torch.where(available, raw_gate, torch.zeros_like(raw_gate))
                support_delta = torch.where(available, raw_delta, torch.zeros_like(raw_delta))
        elif self.support_config.support_enabled and support_keep_mask is not None:
            raise ValueError("support_keep_mask requires support inputs")

        final_logit = base_logit + support_gate * support_delta
        output.update(
            {
                "global_base_logit": global_logit,
                "base_logit": base_logit,
                "final_logit": final_logit,
                "drug_to_target_logit": final_logit,
                "target_to_drug_logit": final_logit,
                "drug_to_target_residual": torch.zeros_like(final_logit),
                "target_to_drug_residual": torch.zeros_like(final_logit),
                "support_gate": support_gate,
                "support_delta": support_delta,
                "support_attention": attention,
                "support_effective_mask": evidence_mask,
                "support_class_available": class_available,
                "interaction_hidden": hidden,
                "local_available": local_mask,
            }
        )
        return output


def drug_macro_pairwise_loss(
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    binary_observed: Tensor,
    *,
    mode: str = "pairwise",
    temperature: float = 1.0,
    margin: float = 0.0,
) -> Tensor:
    """Drug-equal measured-PN ranking; unknown rows never enter competition.

    ``pairwise`` averages softplus over every positive/negative combination,
    then over eligible drugs. ``hard`` averages each positive's logsumexp of
    measured negatives (including a zero reference) before the drug average.
    They agree for a single negative.  Neither mode assigns labels to unknown
    candidates or claims to optimize recall over a fully unmeasured panel.
    """
    if mode not in {"pairwise", "hard"}:
        raise ValueError("mode must be pairwise or hard")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(margin):
        raise ValueError("margin must be finite")
    logits = logits.reshape(-1)
    labels = labels.to(device=logits.device).reshape(-1)
    groups = groups.to(device=logits.device).reshape(-1)
    observed = binary_observed.to(device=logits.device).reshape(-1).gt(0.5)
    if not (logits.shape == labels.shape == groups.shape == observed.shape):
        raise ValueError("logits, labels, groups and binary_observed must have equal lengths")
    scores, target_labels, query_ids = logits[observed], labels[observed], groups[observed]
    if not torch.isfinite(scores).all():
        raise ValueError("observed logits must be finite")
    if not ((target_labels == 0) | (target_labels == 1)).all():
        raise ValueError("observed ranking labels must be explicitly binary")
    if not torch.isfinite(query_ids).all():
        raise ValueError("observed query IDs must be finite")
    terms = []
    for query in torch.unique(query_ids):
        selected = query_ids == query
        positive = scores[selected & (target_labels == 1)]
        negative = scores[selected & (target_labels == 0)]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        differences = (negative[None, :] - positive[:, None] + margin) / temperature
        if mode == "pairwise":
            terms.append(F.softplus(differences).mean())
        else:
            zero = differences.new_zeros((positive.numel(), 1))
            terms.append(torch.logsumexp(torch.cat([zero, differences], dim=1), dim=1).mean())
    return torch.stack(terms).mean() if terms else scores.sum() * 0.0


__all__ = ["SupportV3Config", "SupportInteractionRankerV3", "drug_macro_pairwise_loss"]
