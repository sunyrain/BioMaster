"""BioMaster-ODTI V2 model components.

The V2 implementation keeps the audited V1 sequence/chemistry base and adds
three explicitly optional components:

* a bilinear drug--target interaction term;
* a pocket/structure residual branch with an exact missing-modality fallback;
* multi-task outputs for measured activity, observation propensity and
  auxiliary affinity.

The structure branch is deliberately residual.  If ``structure_mask`` is
zero, the returned ``final_logit`` is exactly the base logit.  Missing
structure is therefore not encoded as a negative interaction and cannot
silently change the universal sequence/chemistry score.

This module is intentionally independent from the large local data products.
It can be used by the production scripts and by small synthetic smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ODTIV2Config:
    """Architecture and loss defaults for the strengthened model."""

    drug_input_dim: int = 2048
    drug_aux_input_dim: int = 0
    drug_aux_gate_init_bias: float | None = None
    target_input_dim: int = 1024
    target_aux_input_dim: int = 0
    target_aux_gate_init_bias: float | None = None
    target_extra_input_dim: int = 0
    target_extra_gate_init_bias: float | None = -4.0
    target_token_input_dim: int = 0
    target_token_heads: int = 4
    target_token_max_len: int = 1022
    target_token_gate_init_bias: float | None = -4.0
    structure_input_dim: int = 0
    # ``legacy_full`` preserves the audited V2 checkpoint topology.  The new
    # ``low_rank_film`` mode removes the ~embedding_dim^3 full bilinear tensor
    # and conditions each entity representation on the other before routing.
    interaction_mode: str = "legacy_full"
    interaction_rank: int = 48
    film_scale: float = 0.10
    # Optional semantic slices of a flat structure vector.  V2 pocket-context
    # stores use quality/chemistry/geometry/consensus groups; old 19-D stores
    # leave this empty and retain the legacy single encoder.
    structure_group_dims: tuple[int, ...] = ()
    enhanced_structure_interaction: bool = False
    structure_gate_init_bias: float | None = None
    local_pair_atom_input_dim: int = 0
    local_pair_pocket_input_dim: int = 0
    local_pair_pocket_aux_dim: int = 0
    local_pair_hidden_dim: int = 96
    local_pair_layers: int = 2
    local_pair_heads: int = 4
    local_pair_gate_init_bias: float | None = -4.0
    embedding_dim: int = 192
    hidden_dim: int = 256
    family_dim: int = 24
    expert_count: int = 6
    dropout: float = 0.12
    rank_weight: float = 0.12
    # Optional reverse-query ranking pressure.  The audited V2 default keeps
    # this off for backwards compatibility; enabling a small value targets
    # old-drug retrieval without changing the binary activity objective.
    drug_rank_weight: float = 0.0
    expert_balance_weight: float = 0.0
    listwise_weight: float = 0.0
    affinity_weight: float = 0.06
    observation_weight: float = 0.10
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.10
    rank_max_pairs: int = 4096
    affinity_min_log_variance: float = -6.0
    affinity_max_log_variance: float = 4.0


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value)


class _LowRankBilinear(nn.Module):
    """Factorized vector-valued bilinear interaction.

    The legacy ``nn.Bilinear(E, E, E)`` uses E^3 weights (7.1M at E=192).
    This factorization uses two E->R projections and one R->E projection,
    retaining multiplicative interaction with far lower variance and memory.
    """

    def __init__(self, embedding_dim: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("interaction_rank must be positive")
        self.drug = nn.Linear(embedding_dim, rank, bias=False)
        self.target = nn.Linear(embedding_dim, rank, bias=False)
        self.output = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, drug: Tensor, target: Tensor) -> Tensor:
        return self.output(self.drug(drug) * self.target(target))


class _GroupedStructureEncoder(nn.Module):
    """Encode semantic pocket feature groups and pool them conditionally."""

    def __init__(
        self,
        group_dims: tuple[int, ...],
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if not group_dims or any(dimension < 1 for dimension in group_dims):
            raise ValueError("structure_group_dims must contain positive dimensions")
        self.group_dims = tuple(int(value) for value in group_dims)
        self.encoders = nn.ModuleList(
            [
                _Encoder(
                    dimension,
                    max(32, min(128, dimension * 8)),
                    embedding_dim,
                    dropout,
                )
                for dimension in self.group_dims
            ]
        )
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.group_bias = nn.Parameter(torch.zeros(len(self.group_dims)))
        self.scale = embedding_dim**-0.5

    def forward(self, value: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        pieces = torch.split(value, self.group_dims, dim=1)
        groups = torch.stack(
            [encoder(piece) for encoder, piece in zip(self.encoders, pieces)],
            dim=1,
        )
        query = self.query(target).unsqueeze(1)
        logits = (groups * query).sum(dim=2) * self.scale + self.group_bias
        weights = torch.softmax(logits, dim=1)
        return (groups * weights.unsqueeze(2)).sum(dim=1), weights


class _DenseEdgeGraphLayer(nn.Module):
    """Small edge-typed message-passing layer for padded local graphs."""

    def __init__(self, width: int, edge_types: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.edge_projections = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(edge_types)]
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor, edge_type: Tensor, mask: Tensor) -> Tensor:
        message = torch.zeros_like(value)
        degree = torch.zeros(
            value.shape[:2], device=value.device, dtype=value.dtype
        )
        for index, projection in enumerate(self.edge_projections, start=1):
            adjacency = edge_type.eq(index).to(dtype=value.dtype)
            message = message + torch.bmm(adjacency, projection(value))
            degree = degree + adjacency.sum(dim=2)
        message = message / degree.clamp_min(1.0).unsqueeze(2)
        update = F.gelu(self.self_projection(value) + message)
        output = self.norm(value + self.dropout(update))
        return output * mask.unsqueeze(2).to(dtype=output.dtype)


class _DenseEdgeGraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        edge_types: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("local_pair_layers must be positive")
        self.input = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()
        )
        self.layers = nn.ModuleList(
            [_DenseEdgeGraphLayer(width, edge_types, dropout) for _ in range(layers)]
        )

    def forward(self, value: Tensor, edge_type: Tensor, mask: Tensor) -> Tensor:
        output = self.input(value) * mask.unsqueeze(2).to(dtype=value.dtype)
        for layer in self.layers:
            output = layer(output, edge_type, mask)
        return output


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    weight = mask.unsqueeze(2).to(dtype=value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class RoutedInteractionRankerV2(nn.Module):
    """Sequence/chemistry ranker with an optional pocket residual.

    Inputs are dense feature matrices.  ``structure`` can be any audited
    pocket-level vector representation (for example a pooled residue graph or
    structure encoder output); it is not assumed to be available for every
    pair.  ``structure_mask`` is a float/bool vector with one entry per pair.
    """

    def __init__(
        self,
        family_count: int,
        config: ODTIV2Config | None = None,
        use_conplex: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or ODTIV2Config()
        cfg = self.config
        if family_count < 1:
            raise ValueError("family_count must be positive")
        if cfg.structure_input_dim < 0:
            raise ValueError("structure_input_dim must be non-negative")
        if cfg.drug_aux_input_dim < 0:
            raise ValueError("drug_aux_input_dim must be non-negative")
        if cfg.target_aux_input_dim < 0:
            raise ValueError("target_aux_input_dim must be non-negative")
        if cfg.target_extra_input_dim < 0:
            raise ValueError("target_extra_input_dim must be non-negative")
        if cfg.target_token_input_dim < 0:
            raise ValueError("target_token_input_dim must be non-negative")
        if cfg.target_token_heads < 1:
            raise ValueError("target_token_heads must be positive")
        if cfg.target_token_input_dim > 0 and cfg.embedding_dim % cfg.target_token_heads:
            raise ValueError("embedding_dim must be divisible by target_token_heads")
        if cfg.interaction_mode not in {"legacy_full", "low_rank_film"}:
            raise ValueError("interaction_mode must be legacy_full or low_rank_film")
        if cfg.interaction_rank < 1:
            raise ValueError("interaction_rank must be positive")
        if cfg.film_scale < 0:
            raise ValueError("film_scale must be non-negative")
        if cfg.structure_group_dims:
            if sum(cfg.structure_group_dims) != cfg.structure_input_dim:
                raise ValueError(
                    "structure_group_dims must sum to structure_input_dim"
                )
            if any(value < 1 for value in cfg.structure_group_dims):
                raise ValueError("structure group dimensions must be positive")
        local_dims = (
            cfg.local_pair_atom_input_dim,
            cfg.local_pair_pocket_input_dim,
            cfg.local_pair_pocket_aux_dim,
        )
        if any(value < 0 for value in local_dims):
            raise ValueError("local graph input dimensions must be non-negative")
        local_enabled = all(value > 0 for value in local_dims[:2])
        if local_enabled and cfg.local_pair_hidden_dim < 8:
            raise ValueError("local_pair_hidden_dim must be at least 8")
        if local_enabled and cfg.local_pair_heads < 1:
            raise ValueError("local_pair_heads must be positive")
        if local_enabled and cfg.local_pair_hidden_dim % cfg.local_pair_heads:
            raise ValueError("local_pair_hidden_dim must divide local_pair_heads")
        self.use_conplex = bool(use_conplex)
        self.has_structure = cfg.structure_input_dim > 0
        self.has_drug_aux = cfg.drug_aux_input_dim > 0
        self.has_target_aux = cfg.target_aux_input_dim > 0
        self.has_target_extra = cfg.target_extra_input_dim > 0
        self.has_target_tokens = cfg.target_token_input_dim > 0
        self.has_local_pair = (
            cfg.local_pair_atom_input_dim > 0
            and cfg.local_pair_pocket_input_dim > 0
        )

        self.drug_encoder = _Encoder(
            cfg.drug_input_dim, 512, cfg.embedding_dim, cfg.dropout
        )
        if self.has_drug_aux:
            self.drug_aux_encoder = _Encoder(
                cfg.drug_aux_input_dim, 512, cfg.embedding_dim, cfg.dropout
            )
            self.drug_aux_gate = nn.Sequential(
                nn.Linear(cfg.embedding_dim * 2, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            if cfg.drug_aux_gate_init_bias is not None:
                nn.init.constant_(
                    self.drug_aux_gate[-1].bias,
                    float(cfg.drug_aux_gate_init_bias),
                )
        self.target_encoder = _Encoder(
            cfg.target_input_dim, 512, cfg.embedding_dim, cfg.dropout
        )
        if self.has_target_aux:
            self.target_aux_encoder = _Encoder(
                cfg.target_aux_input_dim, 512, cfg.embedding_dim, cfg.dropout
            )
            self.target_aux_gate = nn.Sequential(
                nn.Linear(cfg.embedding_dim * 2, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            if cfg.target_aux_gate_init_bias is not None:
                # Optional residual-style cold start.  It is not the audited
                # V2 default because changing initialization requires a new
                # paired training suite; old checkpoints remain reproducible.
                nn.init.constant_(
                    self.target_aux_gate[-1].bias,
                    float(cfg.target_aux_gate_init_bias),
                )
        if self.has_target_extra:
            self.target_extra_encoder = _Encoder(
                cfg.target_extra_input_dim, 512, cfg.embedding_dim, cfg.dropout
            )
            self.target_extra_gate = nn.Sequential(
                nn.Linear(cfg.embedding_dim * 2, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            if cfg.target_extra_gate_init_bias is not None:
                nn.init.constant_(
                    self.target_extra_gate[-1].bias,
                    float(cfg.target_extra_gate_init_bias),
                )
        if self.has_target_tokens:
            self.target_token_encoder = _Encoder(
                cfg.target_token_input_dim, 512, cfg.embedding_dim, cfg.dropout
            )
            self.target_token_cross_attention = nn.MultiheadAttention(
                cfg.embedding_dim,
                cfg.target_token_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.target_token_gate = nn.Sequential(
                nn.Linear(cfg.embedding_dim * 3, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            # Start as a near-zero residual by default so the token branch
            # cannot erase the audited pooled representation at epoch 1.  A
            # paired screen may deliberately relax this initialization to
            # test whether the conservative gate is simply too slow to learn.
            if cfg.target_token_gate_init_bias is not None:
                nn.init.constant_(
                    self.target_token_gate[-1].bias,
                    float(cfg.target_token_gate_init_bias),
                )
        self.family_embedding = nn.Embedding(family_count, cfg.family_dim)

        # Keep legacy topology loadable, but offer a substantially smaller
        # factorized interaction for entity-cold generalization screens.
        if cfg.interaction_mode == "legacy_full":
            self.bilinear = nn.Bilinear(
                cfg.embedding_dim, cfg.embedding_dim, cfg.embedding_dim, bias=False
            )
        else:
            self.bilinear = _LowRankBilinear(
                cfg.embedding_dim, cfg.interaction_rank
            )
            self.drug_conditioned_by_target = nn.Linear(
                cfg.embedding_dim, cfg.embedding_dim * 2
            )
            self.target_conditioned_by_drug = nn.Linear(
                cfg.embedding_dim, cfg.embedding_dim * 2
            )
            # Exact identity initialization makes the first forward pass use
            # the pooled base embeddings while allowing gradients to learn
            # pair-conditioned scale/shift immediately.
            nn.init.zeros_(self.drug_conditioned_by_target.weight)
            nn.init.zeros_(self.drug_conditioned_by_target.bias)
            nn.init.zeros_(self.target_conditioned_by_drug.weight)
            nn.init.zeros_(self.target_conditioned_by_drug.bias)
        pair_width = cfg.embedding_dim * 5 + cfg.family_dim
        self.pair_trunk = nn.Sequential(
            nn.Linear(pair_width, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
        )
        self.shared_head = nn.Linear(cfg.hidden_dim, 1)
        self.expert_heads = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, 1) for _ in range(cfg.expert_count)]
        )
        # Route on both entities.  The V1/V2 draft used target-only routing,
        # which makes every molecule seen by the same target share the same
        # expert mixture.  Pair-conditioned routing lets the experts specialize
        # in chemistry/target interaction regimes without changing the output
        # contract.
        self.gate = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2 + cfg.family_dim, 96),
            nn.GELU(),
            nn.Linear(96, cfg.expert_count),
        )
        self.affinity_head = nn.Linear(cfg.hidden_dim, 1)
        self.observation_head = nn.Linear(cfg.hidden_dim, 1)
        self.affinity_log_variance_head = nn.Linear(cfg.hidden_dim, 1)
        self.dot_scale = nn.Parameter(torch.tensor(2.3025851, dtype=torch.float32))

        if self.has_structure:
            if cfg.structure_group_dims:
                self.structure_encoder = _GroupedStructureEncoder(
                    cfg.structure_group_dims, cfg.embedding_dim, cfg.dropout
                )
            else:
                self.structure_encoder = _Encoder(
                    cfg.structure_input_dim, 256, cfg.embedding_dim, cfg.dropout
                )
            structure_width = cfg.embedding_dim * (
                9 if cfg.enhanced_structure_interaction else 5
            ) + cfg.family_dim
            self.structure_trunk = nn.Sequential(
                nn.Linear(structure_width, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
            )
            self.structure_gate = nn.Sequential(
                nn.Linear(
                    cfg.embedding_dim * (
                        3 if cfg.enhanced_structure_interaction else 2
                    ) + cfg.family_dim,
                    96,
                ),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            if cfg.structure_gate_init_bias is not None:
                nn.init.constant_(
                    self.structure_gate[-1].bias,
                    float(cfg.structure_gate_init_bias),
                )
            self.structure_head = nn.Linear(cfg.hidden_dim, 1)

        if self.has_local_pair:
            self.local_atom_encoder = _DenseEdgeGraphEncoder(
                cfg.local_pair_atom_input_dim,
                cfg.local_pair_hidden_dim,
                edge_types=5,
                layers=cfg.local_pair_layers,
                dropout=cfg.dropout,
            )
            self.local_pocket_encoder = _DenseEdgeGraphEncoder(
                cfg.local_pair_pocket_input_dim + cfg.local_pair_pocket_aux_dim,
                cfg.local_pair_hidden_dim,
                edge_types=4,
                layers=cfg.local_pair_layers,
                dropout=cfg.dropout,
            )
            self.local_atom_to_pocket = nn.MultiheadAttention(
                cfg.local_pair_hidden_dim,
                cfg.local_pair_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.local_pocket_to_atom = nn.MultiheadAttention(
                cfg.local_pair_hidden_dim,
                cfg.local_pair_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            local_width = cfg.local_pair_hidden_dim * 8
            self.local_pair_trunk = nn.Sequential(
                nn.Linear(local_width + cfg.embedding_dim * 2 + cfg.family_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
            )
            self.local_pair_gate = nn.Sequential(
                nn.Linear(cfg.embedding_dim * 2 + cfg.local_pair_hidden_dim * 2 + cfg.family_dim, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            if cfg.local_pair_gate_init_bias is not None:
                nn.init.constant_(
                    self.local_pair_gate[-1].bias,
                    float(cfg.local_pair_gate_init_bias),
                )
            self.local_pair_head = nn.Linear(cfg.hidden_dim, 1)

        if self.use_conplex:
            self.conplex_weight = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
            self.conplex_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @staticmethod
    def _mask(value: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if value is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        mask = value.to(device=device, dtype=torch.float32).reshape(-1)
        if mask.numel() != batch_size:
            raise ValueError("structure_mask must have one value per input row")
        return mask.clamp(0.0, 1.0)

    def forward(
        self,
        drug: Tensor,
        target: Tensor,
        family: Tensor,
        conplex: Tensor | None = None,
        structure: Tensor | None = None,
        structure_mask: Tensor | None = None,
        target_aux: Tensor | None = None,
        target_aux_mask: Tensor | None = None,
        target_extra: Tensor | None = None,
        target_extra_mask: Tensor | None = None,
        target_tokens: Tensor | None = None,
        target_token_mask: Tensor | None = None,
        drug_aux: Tensor | None = None,
        drug_aux_mask: Tensor | None = None,
        ligand_atom_features: Tensor | None = None,
        ligand_atom_mask: Tensor | None = None,
        ligand_bond_type: Tensor | None = None,
        pocket_residue_features: Tensor | None = None,
        pocket_residue_aux: Tensor | None = None,
        pocket_residue_mask: Tensor | None = None,
        pocket_distance_bin: Tensor | None = None,
        local_pair_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if drug.ndim != 2 or target.ndim != 2:
            raise ValueError("drug and target features must be rank-2 tensors")
        if drug.shape[0] != target.shape[0] or family.shape[0] != drug.shape[0]:
            raise ValueError("drug, target and family batch sizes must match")
        batch_size = drug.shape[0]
        cfg = self.config
        drug_base_encoded = self.drug_encoder(drug)
        drug_encoded = drug_base_encoded
        drug_aux_embedding = torch.zeros_like(drug_base_encoded)
        drug_aux_gate = torch.zeros(batch_size, device=drug.device, dtype=drug.dtype)
        if self.has_drug_aux:
            drug_aux_provided = drug_aux is not None
            if drug_aux is None:
                drug_aux = torch.zeros(
                    batch_size,
                    cfg.drug_aux_input_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            if drug_aux.ndim != 2 or drug_aux.shape != (batch_size, cfg.drug_aux_input_dim):
                raise ValueError("drug_aux features must be [batch, drug_aux_input_dim]")
            drug_aux_embedding = self.drug_aux_encoder(drug_aux)
            if drug_aux_mask is None:
                aux_mask = torch.full(
                    (batch_size,),
                    1.0 if drug_aux_provided else 0.0,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            else:
                aux_mask = self._mask(drug_aux_mask, batch_size, drug.device)
            drug_aux_gate = torch.sigmoid(
                self.drug_aux_gate(
                    torch.cat([drug_base_encoded, drug_aux_embedding], dim=1)
                ).squeeze(1)
            ) * aux_mask
            drug_encoded = drug_base_encoded + drug_aux_embedding * drug_aux_gate.unsqueeze(1)
        target_base_encoded = self.target_encoder(target)
        target_encoded = target_base_encoded
        target_aux_embedding = torch.zeros_like(target_base_encoded)
        target_aux_gate = torch.zeros(batch_size, device=drug.device, dtype=drug.dtype)
        if self.has_target_aux:
            target_aux_provided = target_aux is not None
            if target_aux is None:
                target_aux = torch.zeros(
                    batch_size,
                    cfg.target_aux_input_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            if target_aux.ndim != 2 or target_aux.shape != (batch_size, cfg.target_aux_input_dim):
                raise ValueError("target_aux features must be [batch, target_aux_input_dim]")
            target_aux_embedding = self.target_aux_encoder(target_aux)
            if target_aux_mask is None:
                aux_mask = torch.full(
                    (batch_size,),
                    1.0 if target_aux_provided else 0.0,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            else:
                aux_mask = self._mask(target_aux_mask, batch_size, drug.device)
            target_aux_gate = torch.sigmoid(
                self.target_aux_gate(
                    torch.cat([target_base_encoded, target_aux_embedding], dim=1)
                ).squeeze(1)
            ) * aux_mask
            target_encoded = target_base_encoded + target_aux_embedding * target_aux_gate.unsqueeze(1)
        target_extra_embedding = torch.zeros_like(target_base_encoded)
        target_extra_gate = torch.zeros(batch_size, device=drug.device, dtype=drug.dtype)
        if self.has_target_extra:
            target_extra_provided = target_extra is not None
            if target_extra is None:
                target_extra = torch.zeros(
                    batch_size,
                    cfg.target_extra_input_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            if target_extra.ndim != 2 or target_extra.shape != (
                batch_size,
                cfg.target_extra_input_dim,
            ):
                raise ValueError(
                    "target_extra features must be [batch, target_extra_input_dim]"
                )
            target_extra_embedding = self.target_extra_encoder(target_extra)
            if target_extra_mask is None:
                extra_mask = torch.full(
                    (batch_size,),
                    1.0 if target_extra_provided else 0.0,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            else:
                extra_mask = self._mask(target_extra_mask, batch_size, drug.device)
            target_extra_gate = torch.sigmoid(
                self.target_extra_gate(
                    torch.cat([target_encoded, target_extra_embedding], dim=1)
                ).squeeze(1)
            ) * extra_mask
            target_encoded = (
                target_encoded
                + target_extra_embedding * target_extra_gate.unsqueeze(1)
            )
        target_token_context = torch.zeros_like(target_encoded)
        target_token_gate = torch.zeros(batch_size, device=drug.device, dtype=drug.dtype)
        if self.has_target_tokens:
            token_provided = target_tokens is not None
            if target_tokens is None:
                target_tokens = torch.zeros(
                    batch_size,
                    1,
                    cfg.target_token_input_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            if target_tokens.ndim != 3 or target_tokens.shape[0] != batch_size:
                raise ValueError("target_tokens must be [batch, length, target_token_input_dim]")
            if target_tokens.shape[2] != cfg.target_token_input_dim:
                raise ValueError("target token feature width does not match config")
            token_length = target_tokens.shape[1]
            if token_length < 1:
                raise ValueError("target_tokens must contain at least one token")
            if target_token_mask is None:
                token_mask = torch.full(
                    (batch_size, token_length),
                    1.0 if token_provided else 0.0,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            else:
                token_mask = target_token_mask.to(device=drug.device, dtype=drug.dtype)
                if token_mask.ndim != 2 or token_mask.shape != (batch_size, token_length):
                    raise ValueError("target_token_mask must match [batch, token_length]")
                token_mask = token_mask.clamp(0.0, 1.0)
            token_available = token_mask.any(dim=1).to(dtype=drug.dtype)
            encoded_tokens = self.target_token_encoder(target_tokens)
            # MultiheadAttention cannot receive an all-masked row. Add a
            # harmless zero key for missing rows, then gate the context out.
            safe_mask = token_mask > 0.5
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask = safe_mask.clone()
                safe_mask[empty_rows, 0] = True
                encoded_tokens = encoded_tokens.clone()
                encoded_tokens[empty_rows, 0] = 0.0
            query = (drug_encoded + target_encoded).unsqueeze(1)
            attended, _ = self.target_token_cross_attention(
                query,
                encoded_tokens,
                encoded_tokens,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            target_token_context = attended.squeeze(1)
            target_token_gate = torch.sigmoid(
                self.target_token_gate(
                    torch.cat([drug_encoded, target_encoded, target_token_context], dim=1)
                ).squeeze(1)
            ) * token_available
            target_encoded = target_encoded + target_token_context * target_token_gate.unsqueeze(1)
        family_encoded = self.family_embedding(family.long())
        interaction_drug = drug_encoded
        interaction_target = target_encoded
        drug_film_scale = torch.zeros_like(drug_encoded)
        drug_film_shift = torch.zeros_like(drug_encoded)
        target_film_scale = torch.zeros_like(target_encoded)
        target_film_shift = torch.zeros_like(target_encoded)
        if cfg.interaction_mode == "low_rank_film":
            drug_film_scale, drug_film_shift = self.drug_conditioned_by_target(
                target_encoded
            ).chunk(2, dim=1)
            target_film_scale, target_film_shift = self.target_conditioned_by_drug(
                drug_encoded
            ).chunk(2, dim=1)
            interaction_drug = (
                drug_encoded
                * (1.0 + cfg.film_scale * torch.tanh(drug_film_scale))
                + cfg.film_scale * drug_film_shift
            )
            interaction_target = (
                target_encoded
                * (1.0 + cfg.film_scale * torch.tanh(target_film_scale))
                + cfg.film_scale * target_film_shift
            )
        bilinear = self.bilinear(interaction_drug, interaction_target)
        pair = torch.cat(
            [
                interaction_drug,
                interaction_target,
                interaction_drug * interaction_target,
                torch.abs(interaction_drug - interaction_target),
                bilinear,
                family_encoded,
            ],
            dim=1,
        )
        hidden = self.pair_trunk(pair)
        experts = torch.cat([head(hidden) for head in self.expert_heads], dim=1)
        gate = torch.softmax(
            self.gate(
                torch.cat(
                    [interaction_drug, interaction_target, family_encoded], dim=1
                )
            ),
            dim=1,
        )
        routed = (gate * experts).sum(dim=1)
        normalized_drug = F.normalize(interaction_drug, dim=1)
        normalized_target = F.normalize(interaction_target, dim=1)
        dot = (normalized_drug * normalized_target).sum(dim=1)
        base_logit = self.shared_head(hidden).squeeze(1) + routed + self.dot_scale.exp() * dot
        if self.use_conplex:
            if conplex is None:
                raise ValueError("conplex features are required for conplex_augmented")
            base_logit = base_logit + self.conplex_weight * conplex.reshape(-1) + self.conplex_bias

        structure_gate = torch.zeros(batch_size, device=drug.device, dtype=drug.dtype)
        residual_logit = torch.zeros_like(base_logit)
        fused_hidden = hidden
        structure_embedding = torch.zeros_like(target_encoded)
        structure_group_weights = torch.zeros(
            batch_size,
            len(cfg.structure_group_dims) if cfg.structure_group_dims else 1,
            device=drug.device,
            dtype=drug.dtype,
        )
        if self.has_structure:
            if structure is None:
                structure = torch.zeros(
                    batch_size,
                    cfg.structure_input_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            if structure.ndim != 2 or structure.shape[0] != batch_size:
                raise ValueError("structure features must be [batch, structure_input_dim]")
            if structure.shape[1] != cfg.structure_input_dim:
                raise ValueError("structure feature width does not match config")
            if cfg.structure_group_dims:
                structure_embedding, structure_group_weights = self.structure_encoder(
                    structure, interaction_target
                )
            else:
                structure_embedding = self.structure_encoder(structure)
                structure_group_weights = torch.ones(
                    batch_size, 1, device=drug.device, dtype=drug.dtype
                )
            if cfg.enhanced_structure_interaction:
                drug_target_product = interaction_drug * interaction_target
                structure_pair = torch.cat(
                    [
                        interaction_drug,
                        interaction_target,
                        structure_embedding,
                        interaction_drug * structure_embedding,
                        interaction_target * structure_embedding,
                        drug_target_product,
                        drug_target_product * structure_embedding,
                        torch.abs(interaction_drug - structure_embedding),
                        torch.abs(interaction_target - structure_embedding),
                        family_encoded,
                    ],
                    dim=1,
                )
                gate_input = torch.cat(
                    [
                        interaction_drug,
                        interaction_target,
                        structure_embedding,
                        family_encoded,
                    ],
                    dim=1,
                )
            else:
                structure_pair = torch.cat(
                    [
                        interaction_drug,
                        interaction_target,
                        structure_embedding,
                        interaction_drug * structure_embedding,
                        torch.abs(interaction_target - structure_embedding),
                        family_encoded,
                    ],
                    dim=1,
                )
                gate_input = torch.cat(
                    [interaction_target, structure_embedding, family_encoded],
                    dim=1,
                )
            residual_hidden = self.structure_trunk(structure_pair)
            raw_gate = torch.sigmoid(
                self.structure_gate(gate_input).squeeze(1)
            )
            structure_mask_tensor = self._mask(structure_mask, batch_size, drug.device)
            structure_gate = raw_gate * structure_mask_tensor
            structure_group_weights = (
                structure_group_weights * structure_mask_tensor.unsqueeze(1)
            )
            residual_logit = self.structure_head(residual_hidden).squeeze(1) * structure_gate
            fused_hidden = hidden + residual_hidden * structure_gate.unsqueeze(1)

        local_pair_gate = torch.zeros(
            batch_size, device=drug.device, dtype=drug.dtype
        )
        local_pair_residual_logit = torch.zeros_like(base_logit)
        local_atom_pool = torch.zeros(
            batch_size,
            cfg.local_pair_hidden_dim,
            device=drug.device,
            dtype=drug.dtype,
        )
        local_pocket_pool = torch.zeros_like(local_atom_pool)
        if self.has_local_pair:
            required_local = {
                "ligand_atom_features": ligand_atom_features,
                "ligand_atom_mask": ligand_atom_mask,
                "ligand_bond_type": ligand_bond_type,
                "pocket_residue_features": pocket_residue_features,
                "pocket_residue_mask": pocket_residue_mask,
                "pocket_distance_bin": pocket_distance_bin,
            }
            missing_local = [name for name, value in required_local.items() if value is None]
            if missing_local:
                # A configured branch may receive an entirely absent optional
                # modality at deployment; construct a single safe zero node so
                # the exact mask-gated fallback remains defined.
                ligand_atom_features = torch.zeros(
                    batch_size, 1, cfg.local_pair_atom_input_dim,
                    device=drug.device, dtype=drug.dtype,
                )
                ligand_atom_mask = torch.zeros(
                    batch_size, 1, device=drug.device, dtype=drug.dtype
                )
                ligand_bond_type = torch.zeros(
                    batch_size, 1, 1, device=drug.device, dtype=torch.long
                )
                pocket_residue_features = torch.zeros(
                    batch_size, 1, cfg.local_pair_pocket_input_dim,
                    device=drug.device, dtype=drug.dtype,
                )
                pocket_residue_aux = torch.zeros(
                    batch_size, 1, cfg.local_pair_pocket_aux_dim,
                    device=drug.device, dtype=drug.dtype,
                )
                pocket_residue_mask = torch.zeros(
                    batch_size, 1, device=drug.device, dtype=drug.dtype
                )
                pocket_distance_bin = torch.zeros(
                    batch_size, 1, 1, device=drug.device, dtype=torch.long
                )
            if pocket_residue_aux is None:
                pocket_residue_aux = torch.zeros(
                    batch_size,
                    pocket_residue_features.shape[1],
                    cfg.local_pair_pocket_aux_dim,
                    device=drug.device,
                    dtype=drug.dtype,
                )
            atom_mask_bool = ligand_atom_mask.to(device=drug.device).gt(0.5)
            pocket_mask_bool = pocket_residue_mask.to(device=drug.device).gt(0.5)
            available = atom_mask_bool.any(dim=1) & pocket_mask_bool.any(dim=1)
            if local_pair_mask is not None:
                available = available & local_pair_mask.to(device=drug.device).reshape(-1).gt(0.5)
            safe_atom_mask = atom_mask_bool.clone()
            safe_pocket_mask = pocket_mask_bool.clone()
            empty_atom = ~safe_atom_mask.any(dim=1)
            empty_pocket = ~safe_pocket_mask.any(dim=1)
            safe_atom_mask[empty_atom, 0] = True
            safe_pocket_mask[empty_pocket, 0] = True
            ligand_atom_features = ligand_atom_features.to(
                device=drug.device, dtype=drug.dtype
            )
            pocket_residue_features = pocket_residue_features.to(
                device=drug.device, dtype=drug.dtype
            )
            pocket_residue_aux = pocket_residue_aux.to(
                device=drug.device, dtype=drug.dtype
            )
            atom_tokens = self.local_atom_encoder(
                ligand_atom_features,
                ligand_bond_type.to(device=drug.device, dtype=torch.long),
                atom_mask_bool,
            )
            pocket_tokens = self.local_pocket_encoder(
                torch.cat([pocket_residue_features, pocket_residue_aux], dim=2),
                pocket_distance_bin.to(device=drug.device, dtype=torch.long),
                pocket_mask_bool,
            )
            if empty_atom.any():
                atom_tokens = atom_tokens.clone()
                atom_tokens[empty_atom, 0] = 0.0
            if empty_pocket.any():
                pocket_tokens = pocket_tokens.clone()
                pocket_tokens[empty_pocket, 0] = 0.0
            atom_attended, _ = self.local_atom_to_pocket(
                atom_tokens,
                pocket_tokens,
                pocket_tokens,
                key_padding_mask=~safe_pocket_mask,
                need_weights=False,
            )
            pocket_attended, _ = self.local_pocket_to_atom(
                pocket_tokens,
                atom_tokens,
                atom_tokens,
                key_padding_mask=~safe_atom_mask,
                need_weights=False,
            )
            local_atom_pool = _masked_mean(atom_tokens, atom_mask_bool)
            local_pocket_pool = _masked_mean(pocket_tokens, pocket_mask_bool)
            atom_cross_pool = _masked_mean(atom_attended, atom_mask_bool)
            pocket_cross_pool = _masked_mean(pocket_attended, pocket_mask_bool)
            local_features = torch.cat(
                [
                    local_atom_pool,
                    local_pocket_pool,
                    atom_cross_pool,
                    pocket_cross_pool,
                    local_atom_pool * local_pocket_pool,
                    atom_cross_pool * pocket_cross_pool,
                    torch.abs(local_atom_pool - local_pocket_pool),
                    torch.abs(atom_cross_pool - pocket_cross_pool),
                    interaction_drug,
                    interaction_target,
                    family_encoded,
                ],
                dim=1,
            )
            local_hidden = self.local_pair_trunk(local_features)
            raw_local_gate = torch.sigmoid(
                self.local_pair_gate(
                    torch.cat(
                        [
                            interaction_drug,
                            interaction_target,
                            local_atom_pool,
                            local_pocket_pool,
                            family_encoded,
                        ],
                        dim=1,
                    )
                ).squeeze(1)
            )
            local_pair_gate = raw_local_gate * available.to(dtype=drug.dtype)
            local_pair_residual_logit = (
                self.local_pair_head(local_hidden).squeeze(1) * local_pair_gate
            )
            fused_hidden = fused_hidden + local_hidden * local_pair_gate.unsqueeze(1)

        final_logit = base_logit + residual_logit + local_pair_residual_logit
        return {
            "base_logit": base_logit,
            "final_logit": final_logit,
            "affinity": self.affinity_head(fused_hidden).squeeze(1),
            "observation_logit": self.observation_head(fused_hidden).squeeze(1),
            "affinity_log_variance": self.affinity_log_variance_head(fused_hidden)
            .squeeze(1)
            .clamp(cfg.affinity_min_log_variance, cfg.affinity_max_log_variance),
            "gate": gate,
            "structure_gate": structure_gate,
            "drug_embedding": drug_encoded,
            "interaction_drug_embedding": interaction_drug,
            "drug_base_embedding": drug_base_encoded,
            "drug_aux_embedding": drug_aux_embedding,
            "drug_aux_gate": drug_aux_gate,
            "target_embedding": target_encoded,
            "interaction_target_embedding": interaction_target,
            "target_base_embedding": target_base_encoded,
            "target_aux_embedding": target_aux_embedding,
            "target_aux_gate": target_aux_gate,
            "target_extra_embedding": target_extra_embedding,
            "target_extra_gate": target_extra_gate,
            "target_token_context": target_token_context,
            "target_token_gate": target_token_gate,
            "structure_embedding": structure_embedding,
            "structure_group_weights": structure_group_weights,
            "local_pair_gate": local_pair_gate,
            "local_pair_residual_logit": local_pair_residual_logit,
            "local_atom_pool": local_atom_pool,
            "local_pocket_pool": local_pocket_pool,
            "drug_film_scale": drug_film_scale,
            "drug_film_shift": drug_film_shift,
            "target_film_scale": target_film_scale,
            "target_film_shift": target_film_shift,
        }


def target_balanced_weights(labels: Tensor, target_group: Tensor) -> Tensor:
    """Return inverse target/class frequency weights with mean one."""

    labels = labels.long().reshape(-1)
    target_group = target_group.long().reshape(-1)
    if labels.numel() != target_group.numel():
        raise ValueError("labels and target_group must have equal length")
    weights = torch.ones_like(labels, dtype=torch.float32)
    for target in torch.unique(target_group):
        selected = target_group.eq(target)
        for label in (0, 1):
            part = selected & labels.eq(label)
            count = int(part.sum())
            if count:
                weights[part] = 1.0 / count
    return weights / weights.mean().clamp_min(1e-8)


def within_group_rank_loss(
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    max_pairs: int = 4096,
) -> Tensor:
    """Within-group pairwise ranking loss, safe for one-class groups.

    The original implementation compared only the mean positive and mean
    negative logit.  Pairwise comparisons provide a much closer surrogate for
    target-level retrieval metrics.  Very large groups are deterministically
    subsampled to keep the loss quadratic term bounded.
    """

    losses: list[Tensor] = []
    for group in torch.unique(groups):
        selected = groups.eq(group)
        positive = logits[selected & labels.gt(0.5)]
        negative = logits[selected & labels.lt(0.5)]
        if positive.numel() and negative.numel():
            if max_pairs <= 0:
                raise ValueError("max_pairs must be positive")
            # Keep the batch order (which is seeded by the trainer) so the
            # subsampling remains reproducible without a second RNG stream.
            max_positive = max(1, min(positive.numel(), int(max_pairs**0.5)))
            max_negative = max(1, min(negative.numel(), max_pairs // max_positive))
            positive = positive[:max_positive]
            negative = negative[:max_negative]
            pairwise = negative.reshape(1, -1) - positive.reshape(-1, 1)
            losses.append(F.softplus(pairwise).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def within_group_listwise_loss(
    logits: Tensor,
    labels: Tensor,
    groups: Tensor,
    max_items: int = 256,
) -> Tensor:
    """Smooth target-wise listwise loss for retrieval-oriented training.

    For each group containing both classes, the objective maximizes the total
    softmax mass assigned to positives. Unlike a single positive/negative
    pair, this approximates the ranking pressure induced by Recall@K and NDCG
    while remaining well-defined for multiple positives and imbalanced groups.
    Rows are kept in deterministic batch order when a group is truncated.
    """

    if max_items <= 0:
        raise ValueError("max_items must be positive")
    losses: list[Tensor] = []
    for group in torch.unique(groups):
        selected = groups.eq(group)
        group_logits = logits[selected]
        group_labels = labels[selected]
        positive = group_labels.gt(0.5)
        negative = ~positive
        if not positive.any() or not negative.any():
            continue
        if group_logits.numel() > max_items:
            group_logits = group_logits[:max_items]
            positive = positive[:max_items]
            negative = negative[:max_items]
            if not positive.any() or not negative.any():
                continue
        log_denominator = torch.logsumexp(group_logits, dim=0)
        log_positive_mass = torch.logsumexp(group_logits[positive], dim=0)
        losses.append(-(log_positive_mass - log_denominator))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _multi_positive_info_nce(similarity: Tensor, positive_mask: Tensor, temperature: float) -> Tensor:
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return similarity.sum() * 0.0
    logits = similarity / max(float(temperature), 1e-4)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
    log_numerator = torch.logsumexp(positive_logits, dim=1)
    return -(log_numerator[valid] - log_denominator[valid]).mean()


def expert_balance_loss(gate: Tensor) -> Tensor:
    """Penalize collapse of the routed expert mixture.

    ``gate`` is already a row-normalized softmax matrix.  Matching its mean
    usage to a uniform prior is a weak, architecture-agnostic regularizer;
    it does not force any individual pair to use all experts.  The default
    loss weight is zero so the audited V2 behavior remains unchanged until a
    paired ablation supports enabling it.
    """

    if gate.ndim != 2 or gate.shape[1] < 1:
        raise ValueError("gate must be [batch, expert_count]")
    mean_gate = gate.mean(dim=0)
    uniform = torch.full_like(mean_gate, 1.0 / mean_gate.numel())
    return (mean_gate - uniform).square().mean()


def bidirectional_contrastive_loss(
    drug_embedding: Tensor,
    target_embedding: Tensor,
    labels: Tensor,
    drug_group: Tensor,
    target_group: Tensor,
    temperature: float = 0.10,
) -> Tensor:
    """Multi-positive drug↔target contrastive loss.

    A positive matrix is built from repeated drug/target entities and the
    measured binary labels.  The loss becomes zero if a minibatch contains no
    positive pair or no repeated entity with a positive counterpart.
    """

    d = F.normalize(drug_embedding, dim=1)
    t = F.normalize(target_embedding, dim=1)
    similarity = d @ t.T
    labels = labels.reshape(-1).gt(0.5)
    same_drug = drug_group.reshape(-1, 1).eq(drug_group.reshape(1, -1))
    same_target = target_group.reshape(-1, 1).eq(target_group.reshape(1, -1))
    positive_dt = same_drug & labels.reshape(1, -1)
    positive_td = same_target & labels.reshape(1, -1)
    return 0.5 * (
        _multi_positive_info_nce(similarity, positive_dt, temperature)
        + _multi_positive_info_nce(similarity.T, positive_td, temperature)
    )


def censored_affinity_loss(
    prediction: Tensor,
    lower: Tensor,
    upper: Tensor,
    observed: Tensor | None = None,
    log_variance: Tensor | None = None,
) -> Tensor:
    """Heteroscedastic interval-censored Gaussian negative log likelihood.

    ``lower`` and ``upper`` may be equal for exact measurements.  Infinite
    bounds are supported and reduce to one-sided penalties.  ``observed`` can
    mask rows that have no affinity measurement.
    """

    prediction = prediction.reshape(-1)
    lower = lower.reshape(-1)
    upper = upper.reshape(-1)
    if not (prediction.shape == lower.shape == upper.shape):
        raise ValueError("prediction/lower/upper shapes must match")
    if log_variance is None:
        log_variance = torch.zeros_like(prediction)
    else:
        log_variance = log_variance.reshape(-1)
        if log_variance.shape != prediction.shape:
            raise ValueError("log_variance must match prediction shape")
    if observed is None:
        observed_mask = torch.isfinite(lower) | torch.isfinite(upper)
    else:
        observed_mask = observed.reshape(-1).bool()
    if not observed_mask.any():
        return prediction.sum() * 0.0
    value = prediction[observed_mask]
    lo = lower[observed_mask]
    hi = upper[observed_mask]
    variance = log_variance[observed_mask].clamp(-12.0, 8.0)
    scale = torch.exp(0.5 * variance).clamp_min(1e-4)
    exact = torch.isfinite(lo) & torch.isfinite(hi) & torch.isclose(lo, hi)
    losses = []
    if exact.any():
        residual = (value[exact] - lo[exact]) / scale[exact]
        # The additive constant is omitted; it has no effect on optimization.
        losses.append(0.5 * (residual.square() + variance[exact]))
    lower_only = torch.isfinite(lo) & ~torch.isfinite(hi)
    if lower_only.any():
        z = (lo[lower_only] - value[lower_only]) / scale[lower_only]
        losses.append(-torch.special.log_ndtr(-z))
    upper_only = ~torch.isfinite(lo) & torch.isfinite(hi)
    if upper_only.any():
        z = (hi[upper_only] - value[upper_only]) / scale[upper_only]
        losses.append(-torch.special.log_ndtr(z))
    interval = torch.isfinite(lo) & torch.isfinite(hi) & ~exact
    if interval.any():
        z_lo = (lo[interval] - value[interval]) / scale[interval]
        z_hi = (hi[interval] - value[interval]) / scale[interval]
        probability = torch.special.ndtr(z_hi) - torch.special.ndtr(z_lo)
        losses.append(-torch.log(probability.clamp_min(1e-8)))
    if not losses:
        return prediction.sum() * 0.0
    return torch.cat([part.reshape(-1) for part in losses]).mean()


def odti_v2_loss(
    outputs: dict[str, Tensor],
    labels: Tensor,
    target_group: Tensor,
    drug_group: Tensor,
    affinity_lower: Tensor | None = None,
    affinity_upper: Tensor | None = None,
    affinity_observed: Tensor | None = None,
    observed_labels: Tensor | None = None,
    binary_observed: Tensor | None = None,
    config: ODTIV2Config | None = None,
) -> dict[str, Tensor]:
    """Compute the strengthened multi-task loss and return all components."""

    cfg = config or ODTIV2Config()
    labels = labels.float().reshape(-1)
    # External affinity-only rows carry no trustworthy binary label.  They are
    # retained in the forward pass so the affinity head can learn from them,
    # but must be removed from every binary-derived objective.  Keeping this
    # mask inside the loss makes the data contract explicit and prevents a
    # caller from accidentally encoding unknown rows as BCE negatives.
    if binary_observed is None:
        binary_mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        binary_mask = binary_observed.to(device=labels.device).reshape(-1).bool()
        if binary_mask.numel() != labels.numel():
            raise ValueError("binary_observed must have one value per row")
    if binary_mask.any():
        binary_labels = labels[binary_mask]
        binary_logits = outputs["final_logit"][binary_mask]
        binary_targets = target_group[binary_mask]
        binary_drugs = drug_group[binary_mask]
        weights = target_balanced_weights(binary_labels, binary_targets).to(labels.device)
        bce_row = F.binary_cross_entropy_with_logits(
            binary_logits, binary_labels, reduction="none"
        )
        bce = (bce_row * weights).mean()
        rank = within_group_rank_loss(
            binary_logits, binary_labels, binary_targets, cfg.rank_max_pairs
        )
        drug_rank = within_group_rank_loss(
            binary_logits, binary_labels, binary_drugs, cfg.rank_max_pairs
        )
        listwise = within_group_listwise_loss(
            binary_logits, binary_labels, binary_targets
        )
        contrastive = bidirectional_contrastive_loss(
            outputs["drug_embedding"][binary_mask],
            outputs["target_embedding"][binary_mask],
            binary_labels,
            binary_drugs,
            binary_targets,
            cfg.contrastive_temperature,
        )
    else:
        zero = outputs["final_logit"].sum() * 0.0
        bce = zero
        rank = zero
        drug_rank = zero
        listwise = zero
        contrastive = zero
    expert_balance = expert_balance_loss(outputs["gate"])
    if affinity_lower is not None and affinity_upper is not None:
        affinity = censored_affinity_loss(
            outputs["affinity"],
            affinity_lower,
            affinity_upper,
            affinity_observed,
            outputs.get("affinity_log_variance"),
        )
    else:
        affinity = outputs["affinity"].sum() * 0.0
    if observed_labels is None:
        observation = outputs["observation_logit"].sum() * 0.0
    else:
        observation = F.binary_cross_entropy_with_logits(
            outputs["observation_logit"], observed_labels.float().reshape(-1)
        )
    total = (
        bce
        + cfg.rank_weight * rank
        + cfg.drug_rank_weight * drug_rank
        + cfg.expert_balance_weight * expert_balance
        + cfg.listwise_weight * listwise
        + cfg.affinity_weight * affinity
        + cfg.observation_weight * observation
        + cfg.contrastive_weight * contrastive
    )
    return {
        "total": total,
        "bce": bce,
        "rank": rank,
        "drug_rank": drug_rank,
        "expert_balance": expert_balance,
        "listwise": listwise,
        "affinity": affinity,
        "observation": observation,
        "contrastive": contrastive,
    }


def ensemble_summary(probabilities: Tensor) -> dict[str, Tensor]:
    """Summarize independently trained model predictions for active learning."""

    if probabilities.ndim != 2 or probabilities.shape[0] < 1:
        raise ValueError("probabilities must be [n_models, n_pairs]")
    return {
        "mean": probabilities.mean(dim=0),
        "std": probabilities.std(dim=0, unbiased=probabilities.shape[0] > 1),
        "lower": probabilities.min(dim=0).values,
        "upper": probabilities.max(dim=0).values,
    }


__all__ = [
    "ODTIV2Config",
    "RoutedInteractionRankerV2",
    "target_balanced_weights",
    "within_group_rank_loss",
    "within_group_listwise_loss",
    "expert_balance_loss",
    "bidirectional_contrastive_loss",
    "censored_affinity_loss",
    "odti_v2_loss",
    "ensemble_summary",
]
