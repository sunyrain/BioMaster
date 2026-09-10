"""Data contracts, deterministic query sampling, and metrics for V3 experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SPLIT_VERSION = "COMPREHENSIVE_DRUG_ENTITY_HOLDOUT_V3_20260905"


def entity_hash(key: str, salt: str = SPLIT_VERSION) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8], "big")


def prepare_relations(frame: pd.DataFrame, max_entities: int = 0) -> tuple[pd.DataFrame, dict]:
    """Keep explicit binary observations and split whole chemical entities.

    This is an entity holdout, not a historical-time simulation. Conflicting
    duplicate labels are excluded rather than arbitrarily keeping one label.
    """
    required = {"drug_feature_index", "target_feature_index", "binary_label",
                "parent_standard_inchi_key", "target_assay_family"}
    if not required.issubset(frame):
        raise ValueError(f"missing relation columns: {sorted(required - set(frame))}")
    data = frame.copy()
    data["source_row"] = np.arange(len(data), dtype=np.int64)
    label = pd.to_numeric(data.binary_label, errors="coerce")
    observed = pd.to_numeric(data.get("binary_observed", pd.Series(1, index=data.index)), errors="coerce")
    data = data.loc[label.isin([0, 1]) & observed.eq(1)].copy()
    data["binary_label"] = data.binary_label.astype(np.int8)
    data["entity_key"] = data.parent_standard_inchi_key.fillna("").astype(str).str.strip()
    if data.entity_key.isin(["", "nan", "None"]).any():
        raise ValueError("every observed relation needs a nonempty chemical entity key")
    for col in ["drug_feature_index", "target_feature_index"]:
        values = pd.to_numeric(data[col], errors="raise")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"invalid {col}")
        data[col] = values.astype(np.int64)
    # Standardization can merge salts/protonation forms with different source
    # InChIs. Split connected components of source entity <-> feature identity,
    # so neither aliases nor identical model inputs can cross a holdout.
    pairs = data[["drug_feature_index", "entity_key"]].drop_duplicates()
    ambiguous = pairs.drug_feature_index.duplicated(keep=False)
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for _, part in pairs.loc[ambiguous].groupby("drug_feature_index", sort=True):
        first = root(part.entity_key.iloc[0])
        for key in part.entity_key.iloc[1:]:
            other = root(key)
            representative = min(first, other)
            parents[first] = parents[other] = representative
            first = representative
    component_map = {key: root(key) for key in list(parents)}
    data["entity_key"] = data.entity_key.map(lambda key: component_map.get(key, key))
    original_feature_count = data.drug_feature_index.nunique()
    data["drug_feature_index"] = data.groupby("entity_key").drug_feature_index.transform("min")
    collapsed_features = original_feature_count - data.drug_feature_index.nunique()
    duplicate_keys = ["entity_key", "target_feature_index"]
    conflicts = data.groupby(duplicate_keys).binary_label.transform("nunique").gt(1)
    conflict_rows = int(conflicts.sum())
    data = data.loc[~conflicts].drop_duplicates(duplicate_keys).copy()
    hashes = {key: entity_hash(key) for key in sorted(data.entity_key.unique())}
    if max_entities > 0:
        chosen = set(sorted(hashes, key=lambda key: (entity_hash(key, "V3_SMOKE_ENTITY_SUBSET"), key))[:max_entities])
        data = data.loc[data.entity_key.isin(chosen)].copy()
    data["split"] = data.entity_key.map(
        lambda key: "train" if hashes[key] % 100 < 80 else ("validation" if hashes[key] % 100 < 90 else "test")
    )
    data["target_assay_family"] = data.target_assay_family.fillna("unknown").astype(str)
    families = sorted(data.target_assay_family.unique())
    data["family_index"] = data.target_assay_family.map({s: i for i, s in enumerate(families)}).astype(np.int64)
    data = data.sort_values(["drug_feature_index", "target_feature_index"]).reset_index(drop=True)
    data["v3_row"] = np.arange(len(data), dtype=np.int64)
    audit = {"protocol": SPLIT_VERSION, "raw_rows": len(frame), "binary_unique_rows": len(data),
             "conflicting_observed_rows_excluded": conflict_rows, "max_entities": max_entities,
             "aliased_feature_indices_collapsed": int(collapsed_features),
             "source_feature_entity_conflicts_merged": int(pairs.loc[ambiguous].drug_feature_index.nunique()),
             "families": families, "split_method": "SHA256(InChI/standardized-feature connected component), buckets 0..79/80..89/90..99",
             "temporal_claim": False, "affinity_only_training": False,
             "label_scope": "existing explicit binary activity labels; quantitative and functional-specialist heads not trained in this run"}
    sets = {}
    for split, part in data.groupby("split"):
        sets[split] = set(part.entity_key)
        audit[split] = {"rows": len(part), "drugs": part.entity_key.nunique(),
                        "targets": part.target_feature_index.nunique(),
                        "positive_rows": int(part.binary_label.sum()),
                        "two_class_drugs": int(part.groupby("entity_key").binary_label.nunique().eq(2).sum())}
    if set(sets) != {"train", "validation", "test"}:
        raise ValueError("entity subset must contain train, validation, and test")
    if any(sets[a] & sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise AssertionError("chemical entity crossed split boundary")
    audit["entity_overlap"] = 0
    audit["assignment_sha256"] = hashlib.sha256(
        data[["entity_key", "target_feature_index", "binary_label", "split"]].to_csv(index=False).encode()
    ).hexdigest()
    return data, audit


@dataclass
class QueryBatchStream:
    """Equal-query cycles; no repeated rows used to pad a short query."""

    drug_indices: np.ndarray
    labels: np.ndarray
    train_positions: np.ndarray
    queries_per_batch: int = 64
    max_items: int = 16

    def __post_init__(self):
        if self.queries_per_batch < 1 or self.max_items < 2:
            raise ValueError("positive query count and max_items >= 2 required")
        pools = {}
        for pos in self.train_positions:
            pools.setdefault(int(self.drug_indices[pos]), [[], []])[int(self.labels[pos])].append(int(pos))
        self.pools = {k: (np.asarray(v[0], dtype=np.int64), np.asarray(v[1], dtype=np.int64))
                      for k, v in pools.items() if v[0] and v[1]}
        self.keys = np.asarray(sorted(self.pools), dtype=np.int64)

    def batches(self, seed: int):
        if not len(self.keys):
            raise ValueError("no training drugs with observed positive and negative targets")
        rng = np.random.default_rng(seed)
        while True:
            order = rng.permutation(self.keys)
            for start in range(0, len(order), self.queries_per_batch):
                result = []
                for key in order[start:start + self.queries_per_batch]:
                    negative, positive = self.pools[int(key)]
                    p_count = min(len(positive), max(1, self.max_items // 2))
                    n_count = min(len(negative), self.max_items - p_count)
                    p_count = min(len(positive), self.max_items - n_count)
                    result.extend(rng.choice(positive, p_count, replace=False))
                    result.extend(rng.choice(negative, n_count, replace=False))
                yield np.asarray(result, dtype=np.int64)


def query_microbatches(positions: np.ndarray, drug_indices: np.ndarray, max_rows: int):
    """Keep a whole query together, even when it exceeds the requested size."""
    pending = []
    size = 0
    # First-occurrence ordering is preserved without assuming numeric group IDs.
    for group in dict.fromkeys(drug_indices[positions].tolist()):
        part = positions[drug_indices[positions] == group]
        if pending and size + len(part) > max_rows:
            yield np.concatenate(pending)
            pending, size = [], 0
        pending.append(part)
        size += len(part)
    if pending:
        yield np.concatenate(pending)


def observed_metrics(labels: np.ndarray, scores: np.ndarray, drugs: np.ndarray, targets: np.ndarray) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if not len(labels) or not np.isfinite(scores).all():
        raise ValueError("evaluation requires nonempty finite predictions")
    result = {"rows": len(labels), "positive_rows": int(labels.sum()),
              "candidate_scope": "explicitly measured positive/negative relations only",
              "micro_ap": float(average_precision_score(labels, scores)),
              "micro_auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None}
    frame = pd.DataFrame({"label": labels, "score": scores, "drug": drugs, "target": targets})
    for axis in ["drug", "target"]:
        ap, recalls5, recalls20, ndcg20, lengths = [], [], [], [], []
        for _, part in frame.groupby(axis, sort=False):
            if part.label.nunique() < 2:
                continue
            y = part.label.to_numpy()
            s = part.score.to_numpy()
            ap.append(float(average_precision_score(y, s)))
            # Stable deterministic tie break is independent of model batches.
            key = part.target.to_numpy() if axis == "drug" else part.drug.to_numpy()
            ranked = y[np.lexsort((key, -s))]
            recalls5.append(float(ranked[:5].sum() / y.sum()))
            recalls20.append(float(ranked[:20].sum() / y.sum()))
            weights = 1 / np.log2(np.arange(min(20, len(y))) + 2)
            ideal = weights[:min(int(y.sum()), len(weights))].sum()
            ndcg20.append(float((ranked[:20] * weights).sum() / ideal))
            lengths.append(len(y))
        result[f"{axis}_two_class_queries"] = len(ap)
        result[f"{axis}_macro_ap"] = float(np.mean(ap)) if ap else None
        result[f"{axis}_macro_recall_at_5"] = float(np.mean(recalls5)) if ap else None
        result[f"{axis}_macro_recall_at_20"] = float(np.mean(recalls20)) if ap else None
        result[f"{axis}_macro_ndcg_at_20"] = float(np.mean(ndcg20)) if ap else None
        result[f"{axis}_queries_with_more_than_20_candidates"] = int(np.sum(np.array(lengths) > 20))
    return result


def support_summary_features(similarities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A matched-input simple baseline; axis 0 negative, axis 1 positive."""
    mask = np.asarray(mask, dtype=bool)
    values = np.where(mask, similarities, 0).astype(np.float32)
    count = mask.sum(-1)
    maximum = values.max(-1)
    mean = values.sum(-1) / np.maximum(count, 1)
    return np.column_stack([maximum, mean, count > 0,
                            maximum[:, 1] - maximum[:, 0], mean[:, 1] - mean[:, 0]]).astype(np.float32)
