"""Balanced sampling and bidirectional retrieval utilities.

The comprehensive ChEMBL relation table is deliberately left uncapped.  This
module changes only how rows contribute gradients: targets are temperature
sampled, binary labels are balanced within each selected target, and chemical
scaffolds are sampled before individual rows.  Affinity-only rows receive an
explicit batch quota so their influence is not diluted by ChEMBL table size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BalancedSamplingConfig:
    """Configuration for one deterministic balanced training epoch."""

    batch_size: int = 1024
    steps_per_epoch: int = 128
    rows_per_target: int = 16
    target_frequency_power: float = 0.5
    affinity_chunk_fraction: float = 0.09375
    inactive_only_negative_fraction: float = 0.25

    def validate(self) -> None:
        if self.batch_size < 1 or self.steps_per_epoch < 1:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if self.rows_per_target < 2 or self.batch_size % self.rows_per_target:
            raise ValueError("rows_per_target must divide batch_size and be at least two")
        if not 0.0 <= self.target_frequency_power <= 1.0:
            raise ValueError("target_frequency_power must be in [0, 1]")
        if not 0.0 <= self.affinity_chunk_fraction < 1.0:
            raise ValueError("affinity_chunk_fraction must be in [0, 1)")
        if not 0.0 <= self.inactive_only_negative_fraction <= 1.0:
            raise ValueError("inactive_only_negative_fraction must be in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class QuerySamplingConfig:
    """Configuration for explicit query-first binary retrieval batches."""

    batch_size: int = 1024
    steps_per_epoch: int = 64
    rows_per_query: int = 16
    query_frequency_power: float = 0.0

    def validate(self) -> None:
        if self.batch_size < 1 or self.steps_per_epoch < 1:
            raise ValueError("batch_size and steps_per_epoch must be positive")
        if self.rows_per_query < 2 or self.rows_per_query % 2:
            raise ValueError("rows_per_query must be an even integer of at least two")
        if self.batch_size % self.rows_per_query:
            raise ValueError("rows_per_query must divide batch_size")
        if not 0.0 <= self.query_frequency_power <= 1.0:
            raise ValueError("query_frequency_power must be in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _safe_scaffold(value: object, row_index: int) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text if text else f"NO_SCAFFOLD::{row_index}"


class _ScaffoldPool:
    """Draw scaffolds uniformly and rows uniformly within a scaffold."""

    def __init__(self, indices: np.ndarray, scaffolds: np.ndarray):
        if len(indices) < 1:
            raise ValueError("a scaffold pool cannot be empty")
        grouped: dict[str, list[int]] = {}
        for index, scaffold in zip(indices, scaffolds, strict=True):
            grouped.setdefault(_safe_scaffold(scaffold, int(index)), []).append(int(index))
        self.groups = tuple(np.asarray(values, dtype=np.int64) for values in grouped.values())
        self.row_count = int(len(indices))
        self.scaffold_count = int(len(self.groups))

    def draw(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if count < 1:
            return np.empty(0, dtype=np.int64)
        replace_scaffolds = len(self.groups) < count
        selected = rng.choice(len(self.groups), size=count, replace=replace_scaffolds)
        rows = [int(rng.choice(self.groups[int(group)])) for group in selected]
        return np.asarray(rows, dtype=np.int64)


@dataclass
class _BinaryTargetPool:
    positive: _ScaffoldPool | None
    negative_numeric: _ScaffoldPool | None
    negative_inactive_only: _ScaffoldPool | None
    row_count: int

    @property
    def labels_available(self) -> tuple[int, ...]:
        labels = []
        if self.negative_numeric is not None or self.negative_inactive_only is not None:
            labels.append(0)
        if self.positive is not None:
            labels.append(1)
        return tuple(labels)


def _make_pool(frame: pd.DataFrame, indices: np.ndarray) -> _ScaffoldPool | None:
    if len(indices) < 1:
        return None
    scaffolds = frame.loc[indices, "murcko_scaffold"].to_numpy(dtype=object)
    return _ScaffoldPool(indices, scaffolds)


def _binary_target_pools(
    data: pd.DataFrame,
    positions: np.ndarray,
) -> dict[int, _BinaryTargetPool]:
    part = data.iloc[positions]
    pools: dict[int, _BinaryTargetPool] = {}
    for target, local in part.groupby("target_feature_index", sort=True):
        absolute = local.index.to_numpy(dtype=np.int64)
        labels = pd.to_numeric(local["binary_label"], errors="raise").to_numpy(dtype=np.int8)
        affinity = pd.to_numeric(local["mean_pchembl"], errors="coerce").to_numpy(dtype=float)
        inactive = pd.to_numeric(
            local.get("any_explicit_inactive", pd.Series(0, index=local.index)),
            errors="coerce",
        ).fillna(0).to_numpy(dtype=float) > 0
        inactive_only = (labels == 0) & ~np.isfinite(affinity) & inactive
        pools[int(target)] = _BinaryTargetPool(
            positive=_make_pool(data, absolute[labels == 1]),
            negative_numeric=_make_pool(data, absolute[(labels == 0) & ~inactive_only]),
            negative_inactive_only=_make_pool(data, absolute[inactive_only]),
            row_count=int(len(local)),
        )
    return pools


def _affinity_target_pools(
    data: pd.DataFrame,
    positions: np.ndarray,
) -> dict[int, _ScaffoldPool]:
    part = data.iloc[positions]
    result: dict[int, _ScaffoldPool] = {}
    for target, local in part.groupby("target_feature_index", sort=True):
        pool = _make_pool(data, local.index.to_numpy(dtype=np.int64))
        if pool is not None:
            result[int(target)] = pool
    return result


def _draw_negative(
    pool: _BinaryTargetPool,
    count: int,
    inactive_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    numeric = pool.negative_numeric
    inactive = pool.negative_inactive_only
    if numeric is None and inactive is None:
        return np.empty(0, dtype=np.int64)
    if numeric is None:
        return inactive.draw(count, rng)  # type: ignore[union-attr]
    if inactive is None:
        return numeric.draw(count, rng)
    inactive_count = int(round(count * inactive_fraction))
    inactive_count = min(max(inactive_count, 0), count)
    return np.concatenate(
        [numeric.draw(count - inactive_count, rng), inactive.draw(inactive_count, rng)]
    )


def _draw_binary_chunk(
    pool: _BinaryTargetPool,
    count: int,
    inactive_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    has_positive = pool.positive is not None
    has_negative = pool.negative_numeric is not None or pool.negative_inactive_only is not None
    if has_positive and has_negative:
        positive_count = count // 2
        negative_count = count - positive_count
        return np.concatenate(
            [
                pool.positive.draw(positive_count, rng),  # type: ignore[union-attr]
                _draw_negative(pool, negative_count, inactive_fraction, rng),
            ]
        )
    if has_positive:
        return pool.positive.draw(count, rng)  # type: ignore[union-attr]
    return _draw_negative(pool, count, inactive_fraction, rng)


def _sample_targets(
    targets: np.ndarray,
    row_counts: np.ndarray,
    count: int,
    power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = np.power(row_counts.astype(np.float64), power)
    probabilities /= probabilities.sum()
    return rng.choice(
        targets,
        size=count,
        replace=count > len(targets),
        p=probabilities,
    ).astype(np.int64)


def balanced_training_batches(
    positions: np.ndarray,
    data: pd.DataFrame,
    seed: int,
    config: BalancedSamplingConfig,
) -> list[np.ndarray]:
    """Return deterministic target/label/scaffold-balanced batches.

    Every returned batch has exactly ``batch_size`` rows.  Binary targets are
    selected with probability proportional to ``n**target_frequency_power``;
    setting the power to zero is target-uniform and one recovers raw target
    frequency.  A selected two-class target contributes an exactly balanced
    positive/negative chunk.
    """

    config.validate()
    required = {
        "target_feature_index",
        "binary_label",
        "binary_observed",
        "mean_pchembl",
        "murcko_scaffold",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"balanced sampler is missing columns: {sorted(missing)}")
    positions = np.asarray(positions, dtype=np.int64)
    if len(positions) < 1 or positions.min() < 0 or positions.max() >= len(data):
        raise ValueError("positions are empty or out of bounds")
    observed = pd.to_numeric(
        data.iloc[positions]["binary_observed"], errors="raise"
    ).to_numpy(dtype=np.int8)
    binary_positions = positions[observed == 1]
    affinity_positions = positions[observed == 0]
    binary_pools = _binary_target_pools(data, binary_positions)
    affinity_pools = _affinity_target_pools(data, affinity_positions)
    if not binary_pools:
        raise ValueError("balanced sampler requires binary-observed rows")

    chunks_per_batch = config.batch_size // config.rows_per_target
    affinity_chunks = int(round(chunks_per_batch * config.affinity_chunk_fraction))
    if not affinity_pools:
        affinity_chunks = 0
    affinity_chunks = min(max(affinity_chunks, 0), chunks_per_batch - 1)
    binary_chunks = chunks_per_batch - affinity_chunks
    binary_targets = np.asarray(sorted(binary_pools), dtype=np.int64)
    binary_counts = np.asarray(
        [binary_pools[int(target)].row_count for target in binary_targets], dtype=np.float64
    )
    affinity_targets = np.asarray(sorted(affinity_pools), dtype=np.int64)
    affinity_counts = np.asarray(
        [affinity_pools[int(target)].row_count for target in affinity_targets], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    for _ in range(config.steps_per_epoch):
        chunks = []
        chosen_binary = _sample_targets(
            binary_targets,
            binary_counts,
            binary_chunks,
            config.target_frequency_power,
            rng,
        )
        for target in chosen_binary:
            chunks.append(
                _draw_binary_chunk(
                    binary_pools[int(target)],
                    config.rows_per_target,
                    config.inactive_only_negative_fraction,
                    rng,
                )
            )
        if affinity_chunks:
            chosen_affinity = _sample_targets(
                affinity_targets,
                affinity_counts,
                affinity_chunks,
                config.target_frequency_power,
                rng,
            )
            for target in chosen_affinity:
                chunks.append(affinity_pools[int(target)].draw(config.rows_per_target, rng))
        batch = np.concatenate(chunks).astype(np.int64, copy=False)
        if len(batch) != config.batch_size:
            raise RuntimeError("balanced sampler produced an incomplete batch")
        rng.shuffle(batch)
        batches.append(batch)
    return batches


class QueryFirstBatchSampler:
    """Precomputed query pools reusable across epochs."""

    def __init__(
        self,
        positions: np.ndarray,
        data: pd.DataFrame,
        query_column: str,
        config: QuerySamplingConfig,
    ) -> None:
        config.validate()
        required = {
            query_column,
            "binary_label",
            "binary_observed",
            "murcko_scaffold",
        }
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"query-first sampler is missing columns: {sorted(missing)}")
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) < 1 or positions.min() < 0 or positions.max() >= len(data):
            raise ValueError("positions are empty or out of bounds")
        frame = data.iloc[positions]
        observed = pd.to_numeric(
            frame["binary_observed"], errors="raise"
        ).to_numpy(dtype=np.int8)
        frame = frame.loc[observed == 1].copy()
        frame["__binary_label"] = pd.to_numeric(
            frame["binary_label"], errors="raise"
        ).to_numpy(dtype=np.int8)
        # Vectorized prefiltering is essential for D->T: the comprehensive
        # table contains hundreds of thousands of singleton drugs but only a
        # small subset can define a positive-vs-negative query order.
        two_class = (
            frame.groupby(query_column, sort=False)["__binary_label"]
            .transform("nunique")
            .eq(2)
        )
        frame = frame.loc[two_class]
        pools: dict[int, tuple[_ScaffoldPool, _ScaffoldPool, int]] = {}
        for query, group in frame.groupby(query_column, sort=True):
            labels = group["__binary_label"].to_numpy(dtype=np.int8)
            absolute = group.index.to_numpy(dtype=np.int64)
            positive = _make_pool(data, absolute[labels == 1])
            negative = _make_pool(data, absolute[labels == 0])
            if positive is not None and negative is not None:
                pools[int(query)] = (positive, negative, int(len(group)))
        if not pools:
            raise ValueError("query-first sampler requires at least one two-class query")
        self.config = config
        self.pools = pools
        self.query_ids = np.asarray(sorted(pools), dtype=np.int64)
        self.row_counts = np.asarray(
            [pools[int(query)][2] for query in self.query_ids], dtype=float
        )

    @property
    def query_count(self) -> int:
        return int(len(self.query_ids))

    def batches(self, seed: int) -> list[np.ndarray]:
        config = self.config
        chunks_per_batch = config.batch_size // config.rows_per_query
        per_class = config.rows_per_query // 2
        rng = np.random.default_rng(seed)
        batches: list[np.ndarray] = []
        for _ in range(config.steps_per_epoch):
            selected = _sample_targets(
                self.query_ids,
                self.row_counts,
                chunks_per_batch,
                config.query_frequency_power,
                rng,
            )
            chunks = []
            for query in selected:
                positive, negative, _ = self.pools[int(query)]
                chunks.append(
                    np.concatenate(
                        [positive.draw(per_class, rng), negative.draw(per_class, rng)]
                    )
                )
            batch = np.concatenate(chunks).astype(np.int64, copy=False)
            if len(batch) != config.batch_size:
                raise RuntimeError("query-first sampler produced an incomplete batch")
            rng.shuffle(batch)
            batches.append(batch)
        return batches


def sampling_audit(batches: list[np.ndarray], data: pd.DataFrame) -> dict[str, object]:
    """Summarize realized row, label, target and scaffold exposure."""

    if not batches:
        raise ValueError("sampling audit requires at least one batch")
    emitted = np.concatenate(batches)
    part = data.iloc[emitted].copy()
    observed = pd.to_numeric(part["binary_observed"], errors="coerce").fillna(0).eq(1)
    binary = part.loc[observed]
    target_counts = binary.groupby("target_feature_index", sort=False).size().to_numpy(float)
    probability = target_counts / max(target_counts.sum(), 1.0)
    effective_targets = float(1.0 / np.square(probability).sum()) if len(probability) else 0.0
    sorted_counts = np.sort(target_counts)
    if len(sorted_counts) and sorted_counts.sum() > 0:
        index = np.arange(1, len(sorted_counts) + 1)
        gini = float(
            np.sum((2 * index - len(sorted_counts) - 1) * sorted_counts)
            / (len(sorted_counts) * sorted_counts.sum())
        )
    else:
        gini = 0.0
    one_class_rows = 0
    binary_rows = 0
    for batch in batches:
        frame = data.iloc[batch]
        frame = frame.loc[
            pd.to_numeric(frame["binary_observed"], errors="coerce").fillna(0).eq(1)
        ]
        for _, group in frame.groupby("target_feature_index", sort=False):
            binary_rows += len(group)
            if group["binary_label"].nunique() < 2:
                one_class_rows += len(group)
    labels = pd.to_numeric(binary["binary_label"], errors="coerce").to_numpy(dtype=float)
    return {
        "batches": int(len(batches)),
        "rows_emitted": int(len(emitted)),
        "unique_rows_emitted": int(np.unique(emitted).size),
        "unique_targets_emitted": int(part["target_feature_index"].nunique()),
        "binary_rows": int(len(binary)),
        "affinity_only_rows": int((~observed).sum()),
        "binary_positive_fraction": float(labels.mean()) if len(labels) else None,
        "binary_one_class_target_batch_row_fraction": (
            float(one_class_rows / binary_rows) if binary_rows else None
        ),
        "binary_target_gini": gini,
        "binary_effective_targets": effective_targets,
        "unique_scaffolds_emitted": int(part["murcko_scaffold"].fillna("").nunique()),
    }


def target_to_drug_metrics(
    frame: pd.DataFrame,
    binary_score: np.ndarray,
    affinity_score: np.ndarray,
    affinity_weight: float = 0.45,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate one held relation against a fixed drug library per query."""

    required = {"query_id", "binary_label", "target_chembl_id", "parent_standard_inchi_key"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"target-to-drug frame is missing columns: {sorted(missing)}")
    work = frame[[
        "query_id", "binary_label", "target_chembl_id", "parent_standard_inchi_key"
    ]].copy()
    binary_score = np.asarray(binary_score, dtype=float).reshape(-1)
    affinity_score = np.asarray(affinity_score, dtype=float).reshape(-1)
    if len(work) != len(binary_score) or len(work) != len(affinity_score):
        raise ValueError("score lengths do not match target-to-drug frame")
    work["binary_score"] = binary_score
    work["affinity_score"] = affinity_score
    binary_rank_unit = work.groupby("query_id", sort=False)["binary_score"].rank(
        method="average", pct=True
    )
    affinity_rank_unit = work.groupby("query_id", sort=False)["affinity_score"].rank(
        method="average", pct=True
    )
    work["fusion_score"] = (
        (1.0 - affinity_weight) * binary_rank_unit + affinity_weight * affinity_rank_unit
    )
    work["rank"] = work.groupby("query_id", sort=False)["fusion_score"].rank(
        method="min", ascending=False
    ).astype(int)
    sizes = work.groupby("query_id", sort=False)["rank"].transform("size")
    work["top_percentile"] = 1.0 - (work["rank"] - 1) / sizes.clip(lower=1)
    known = work.loc[work["binary_label"].eq(1)].copy()
    positive_counts = known.groupby("query_id").size()
    if len(positive_counts) < 1 or not positive_counts.eq(1).all():
        raise ValueError("each target-to-drug query must have exactly one held positive")
    candidate_counts = work.groupby("query_id").size()
    universe = int(candidate_counts.iloc[0])
    if not candidate_counts.eq(universe).all():
        raise ValueError("target-to-drug candidate universes are inconsistent")
    log_reciprocal = 1.0 / np.log2(known["rank"].to_numpy(dtype=float) + 1.0)
    metrics: dict[str, float | int] = {
        "queries": int(len(known)),
        "candidate_drugs": universe,
        "mean_rank": float(known["rank"].mean()),
        "median_rank": float(known["rank"].median()),
        "mrr": float((1.0 / known["rank"]).mean()),
        "mean_reciprocal_log_rank": float(log_reciprocal.mean()),
        "mean_top_percentile": float(known["top_percentile"].mean()),
        "hit_at_10": float(known["rank"].le(10).mean()),
        "hit_at_36": float(known["rank"].le(36).mean()),
        "hit_at_72": float(known["rank"].le(72).mean()),
    }
    return metrics, known


def fuse_percentile_scores(
    binary: np.ndarray,
    affinity: np.ndarray,
    strategy: str = "linear",
    affinity_weight: float = 0.45,
) -> np.ndarray:
    """Fuse target-wise rank percentiles with optional binary shortlisting.

    ``top20_refine`` and ``top10_refine`` make the binary head the eligibility
    gate.  The affinity head can reorder only candidates already admitted to
    that target's binary shortlist, so it cannot rescue a low-binary candidate.
    """

    binary = np.asarray(binary, dtype=float)
    affinity = np.asarray(affinity, dtype=float)
    if binary.shape != affinity.shape:
        raise ValueError("binary and affinity percentile arrays must have the same shape")
    weight = float(affinity_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("affinity_weight must be in [0, 1]")
    if strategy == "linear":
        return (1.0 - weight) * binary + weight * affinity
    if strategy == "product":
        return binary * ((1.0 - weight) + weight * affinity)
    if strategy == "geometric":
        epsilon = 1e-6
        return np.power(binary + epsilon, 1.0 - weight) * np.power(
            affinity + epsilon, weight
        )
    if strategy in {"top20_refine", "top10_refine"}:
        threshold = 0.80 if strategy == "top20_refine" else 0.90
        shortlist = binary >= threshold
        score = binary.copy()
        score[shortlist] = (
            1.0
            + (1.0 - weight) * binary[shortlist]
            + weight * affinity[shortlist]
        )
        return score
    raise ValueError(f"unknown fusion strategy: {strategy}")


__all__ = [
    "BalancedSamplingConfig",
    "QuerySamplingConfig",
    "QueryFirstBatchSampler",
    "balanced_training_batches",
    "sampling_audit",
    "target_to_drug_metrics",
    "fuse_percentile_scores",
]
