"""Leakage-safe observed positive/negative support retrieval for BioMaster V3.

A store is an immutable training-fold snapshot.  Retrieval excludes the query's
chemical entity, independently of its batch companions.  The label axis is
always ``0 = measured negative, 1 = measured positive``.  Unknowns and entities
with contradictory training labels for a target are absent from both pools.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


STORE_VERSION = "biomaster_support_store_v3.1"


def _array_hash(array: np.ndarray) -> str:
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    if array.dtype.kind == "O":
        raise TypeError("Hash object arrays only after conversion to fixed-width strings")
    for start in range(0, len(array), 4096):
        digest.update(np.ascontiguousarray(array[start:start + 4096]).tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _keys_and_codes(keys: Sequence | None, size: int, *, missing_unique: bool) -> tuple[np.ndarray, np.ndarray]:
    if keys is None:
        values = np.asarray([f"__feature_{index}" for index in range(size)]) if missing_unique else np.full(size, "")
    else:
        raw = np.asarray(keys)
        if raw.ndim != 1 or len(raw) != size:
            raise ValueError("Entity/scaffold keys must align with every feature row")
        # Missing entity IDs must never collapse distinct molecules into one entity.
        values = np.asarray([
            (f"__feature_{index}" if missing_unique else "")
            if value is None or str(value).lower() in {"", "nan", "none", "<na>"}
            else str(value)
            for index, value in enumerate(raw)
        ])
    _, codes = np.unique(values, return_inverse=True)
    if not missing_unique:
        codes = codes.astype(np.int64)
        codes[values == ""] = -1
    return values, codes.astype(np.int64)


def _positions(positions: Sequence, relation_count: int) -> np.ndarray:
    array = np.asarray(positions)
    if array.dtype.kind == "b":
        if array.ndim != 1 or len(array) != relation_count:
            raise ValueError("Boolean train_positions must align with relations")
        return np.flatnonzero(array)
    if array.ndim != 1 or (array.size and array.dtype.kind not in "iu"):
        raise ValueError("train_positions must be a one-dimensional integer index or boolean mask")
    array = array.astype(np.int64)
    if array.size and (array.min() < 0 or array.max() >= relation_count):
        raise IndexError("train_positions out of bounds")
    return np.unique(array)


@dataclass(frozen=True)
class SupportBatch:
    """Feature-row indices, exact Tanimoto similarities and valid support mask."""

    indices: np.ndarray
    similarities: np.ndarray
    mask: np.ndarray

    def __iter__(self):
        yield self.indices
        yield self.similarities
        yield self.mask

    def __len__(self) -> int:
        return len(self.indices)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"indices": self.indices, "similarities": self.similarities, "mask": self.mask}


class SupportStore:
    """Observed support pool from an explicit training subset.

    ``features`` is a binary array [molecules, Morgan bits]. ``entity_keys`` and
    optional ``scaffold_keys`` align with feature rows, not relation rows. Labels
    other than numeric 0/1 are ignored. ``observed`` can additionally exclude
    pseudo/unknown labels even if an upstream table assigns them 0 or 1.

    RDKit's packed-bit exact Tanimoto is the default backend. A bounded-memory
    NumPy packed-bit backend is available when RDKit is unavailable. Stores hold
    only feature references; ``save`` does not copy the large feature matrix.
    """

    def __init__(
        self,
        features: np.ndarray,
        drug_indices: Sequence,
        target_indices: Sequence,
        labels: Sequence,
        train_positions: Sequence,
        entity_keys: Sequence | None = None,
        scaffold_keys: Sequence | None = None,
        *,
        observed: Sequence | None = None,
        backend: str = "rdkit",
        feature_path: str | Path | None = None,
    ):
        self.features = np.asanyarray(features)
        if self.features.ndim != 2 or self.features.shape[1] == 0:
            raise ValueError("features must be [molecules, fingerprint bits]")
        for start in range(0, len(self.features), 4096):
            block = self.features[start:start + 4096]
            if not np.isin(block, (0, 1)).all():
                raise ValueError("features must contain binary 0/1 Morgan bits")
        drugs = np.asarray(drug_indices, dtype=np.int64)
        targets = np.asarray(target_indices, dtype=np.int64)
        values = np.asarray(labels, dtype=np.float64)
        if drugs.ndim != 1 or targets.shape != drugs.shape or values.shape != drugs.shape:
            raise ValueError("drug_indices, target_indices and labels must be aligned vectors")
        positions = _positions(train_positions, len(drugs))
        valid = np.isin(values[positions], [0, 1])
        if observed is not None:
            observation_mask = np.asarray(observed)
            if observation_mask.shape != drugs.shape:
                raise ValueError("observed must align with relations")
            valid &= observation_mask[positions] == 1
        eligible = positions[valid]
        train_drugs, train_targets = drugs[eligible], targets[eligible]
        if eligible.size and ((train_drugs < 0).any() or (train_drugs >= len(features)).any() or (train_targets < 0).any()):
            raise IndexError("Training support drug/target index out of bounds")
        entities, self.entity_codes = _keys_and_codes(entity_keys, len(features), missing_unique=True)
        scaffolds, self.scaffold_codes = _keys_and_codes(scaffold_keys, len(features), missing_unique=False)
        self.has_scaffolds = scaffold_keys is not None
        train_labels = values[eligible].astype(np.int8)
        train_entities = self.entity_codes[train_drugs]
        ordering = np.lexsort((train_drugs, train_labels, train_entities, train_targets))
        d, t, y, e = (array[ordering] for array in (train_drugs, train_targets, train_labels, train_entities))
        if len(d):
            starts = np.r_[0, np.flatnonzero((t[1:] != t[:-1]) | (e[1:] != e[:-1])) + 1]
            label_min = np.minimum.reduceat(y, starts)
            label_max = np.maximum.reduceat(y, starts)
            nonconflict = label_min == label_max
            representative_drugs = np.minimum.reduceat(d, starts)[nonconflict]
            representative_targets = t[starts][nonconflict]
            representative_labels = label_min[nonconflict]
            conflict_count = int((~nonconflict).sum())
            group_count = len(starts)
        else:
            representative_drugs = np.empty(0, dtype=np.int64)
            representative_targets = np.empty(0, dtype=np.int64)
            representative_labels = np.empty(0, dtype=np.int8)
            conflict_count = group_count = 0
        ordering = np.lexsort((representative_drugs, representative_labels, representative_targets))
        self.pool_drug_indices = representative_drugs[ordering]
        self.pool_target_indices = representative_targets[ordering]
        self.pool_labels = representative_labels[ordering]
        self.feature_path = str(Path(feature_path).resolve()) if feature_path else None
        if self.feature_path is None and isinstance(features, np.memmap) and features.filename:
            self.feature_path = str(Path(features.filename).resolve())
        self.feature_hash = _array_hash(self.features)
        self.metadata = {
            "version": STORE_VERSION,
            "feature_hash": self.feature_hash,
            "feature_shape": list(self.features.shape),
            "training_positions_hash": _array_hash(positions),
            "selected_relations_hash": _json_hash({
                "drugs": _array_hash(train_drugs), "targets": _array_hash(train_targets),
                "labels": _array_hash(train_labels),
            }),
            "entity_keys_hash": _array_hash(entities),
            "scaffold_keys_hash": _array_hash(scaffolds),
            "training_rows": len(positions),
            "observed_binary_training_rows": len(eligible),
            "unique_target_entities_before_conflict_filter": group_count,
            "conflicting_target_entities_excluded": conflict_count,
            "support_rows": len(self.pool_drug_indices),
            "positive_support_rows": int((self.pool_labels == 1).sum()),
            "negative_support_rows": int((self.pool_labels == 0).sum()),
            "has_scaffolds": self.has_scaffolds,
            "label_axis": ["negative", "positive"],
            "self_exclusion": "all records with the query chemical entity",
            "conflict_policy": "exclude conflicting target/entity from both support classes",
        }
        self.store_hash = _json_hash(self.metadata)
        self._initialize_backend(backend)

    def _initialize_backend(self, backend: str) -> None:
        if backend not in {"rdkit", "numpy", "auto"}:
            raise ValueError("backend must be rdkit, numpy or auto")
        self.backend = backend
        self._rdkit = None
        if backend in {"rdkit", "auto"}:
            try:
                from rdkit import DataStructs
                self._rdkit = DataStructs
                self.backend = "rdkit"
            except ImportError:
                if backend == "rdkit":
                    raise
                self.backend = "numpy"
        self._packed = np.packbits(self.features, axis=1, bitorder="little")
        self._bit_counts = self.features.sum(axis=1, dtype=np.int32)
        self._popcount = np.asarray([index.bit_count() for index in range(256)], dtype=np.uint8)
        self._fingerprints: dict[int, object] = {}
        self._pools: dict[tuple[int, int], np.ndarray] = {}
        self._pool_fingerprints: dict[tuple[int, int], list] = {}
        if len(self.pool_drug_indices):
            changes = (self.pool_target_indices[1:] != self.pool_target_indices[:-1]) | (self.pool_labels[1:] != self.pool_labels[:-1])
            starts = np.r_[0, np.flatnonzero(changes) + 1]
            ends = np.r_[starts[1:], len(self.pool_drug_indices)]
            for start, end in zip(starts, ends):
                key = (int(self.pool_target_indices[start]), int(self.pool_labels[start]))
                self._pools[key] = self.pool_drug_indices[start:end]

    def _fingerprint(self, index: int):
        if index not in self._fingerprints:
            self._fingerprints[index] = self._rdkit.CreateFromBinaryText(self._packed[index].tobytes())
        return self._fingerprints[index]

    def _similarities(self, query: int, key: tuple[int, int], candidates: np.ndarray) -> np.ndarray:
        if self.backend == "rdkit":
            if key not in self._pool_fingerprints:
                self._pool_fingerprints[key] = [self._fingerprint(int(index)) for index in candidates]
            return np.asarray(self._rdkit.BulkTanimotoSimilarity(self._fingerprint(query), self._pool_fingerprints[key]), dtype=np.float32)
        intersections = np.empty(len(candidates), dtype=np.int32)
        # Bound the [support, packed bytes] scratch space regardless of target size.
        for start in range(0, len(candidates), 4096):
            subset = candidates[start:start + 4096]
            overlap = np.bitwise_and(self._packed[subset], self._packed[query])
            intersections[start:start + len(subset)] = self._popcount[overlap].sum(axis=1, dtype=np.int32)
        unions = self._bit_counts[candidates] + self._bit_counts[query] - intersections
        # RDKit defines the similarity of two empty bit vectors as zero.
        return np.divide(intersections, unions, out=np.zeros(len(candidates), dtype=np.float32), where=unions != 0)

    @staticmethod
    def _stable_topk(candidates: np.ndarray, similarities: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
        allowed = np.flatnonzero(valid)
        if len(allowed) > k:
            values = similarities[allowed]
            threshold = np.partition(values, len(values) - k)[len(values) - k]
            allowed = allowed[values >= threshold]
        ordering = np.lexsort((candidates[allowed], -similarities[allowed]))
        return allowed[ordering[:k]]

    def retrieve(
        self,
        drug_indices: Sequence,
        target_indices: Sequence,
        k: int = 16,
        exclude_scaffold: bool = False,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> SupportBatch:
        drugs, targets = np.asarray(drug_indices, dtype=np.int64), np.asarray(target_indices, dtype=np.int64)
        if drugs.ndim != 1 or targets.shape != drugs.shape:
            raise ValueError("query drug_indices and target_indices must be aligned vectors")
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        k = int(k)
        if len(drugs) and ((drugs < 0).any() or (drugs >= len(self.features)).any() or (targets < 0).any()):
            raise IndexError("Query drug/target index out of bounds")
        if exclude_scaffold and not self.has_scaffolds:
            raise ValueError("exclude_scaffold requires feature-aligned scaffold_keys")
        indices = np.full((len(drugs), 2, k), -1, dtype=np.int32)
        similarities = np.zeros(indices.shape, dtype=np.float32)
        mask = np.zeros(indices.shape, dtype=bool)
        # Grouping improves target-pool locality without changing query semantics.
        order = np.argsort(targets, kind="stable")
        completed = 0
        last_target = None
        for row in order:
            drug, target = int(drugs[row]), int(targets[row])
            if last_target is not None and target != last_target and progress is not None:
                progress(completed, len(drugs))
            last_target = target
            for label in (0, 1):
                key = (target, label)
                candidates = self._pools.get(key)
                if candidates is None:
                    continue
                valid = self.entity_codes[candidates] != self.entity_codes[drug]
                scaffold = self.scaffold_codes[drug]
                if exclude_scaffold and scaffold >= 0:
                    valid &= self.scaffold_codes[candidates] != scaffold
                if not valid.any():
                    continue
                values = self._similarities(drug, key, candidates)
                selected = self._stable_topk(candidates, values, valid, k)
                count = len(selected)
                indices[row, label, :count] = candidates[selected]
                similarities[row, label, :count] = values[selected]
                mask[row, label, :count] = True
            completed += 1
        if progress is not None:
            progress(completed, len(drugs))
        return SupportBatch(indices, similarities, mask)

    def cache_retrieve(
        self,
        out_dir: str | Path,
        drug_indices: Sequence,
        target_indices: Sequence,
        k: int = 16,
        exclude_scaffold: bool = False,
        *,
        progress: Callable[[int, int], None] | None = None,
        force: bool = False,
    ) -> SupportBatch:
        """Retrieve once, persist arrays, then return read-only mmap-backed results.

        The identity includes all feature bytes, training support labels/entities,
        ordered queries and exclusion settings. Each identity gets its own folder.
        A manifest is written last so interrupted cache writes are never reused.
        """
        drugs, targets = np.asarray(drug_indices, dtype=np.int64), np.asarray(target_indices, dtype=np.int64)
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer")
        if drugs.ndim != 1 or targets.shape != drugs.shape:
            raise ValueError("query drug_indices and target_indices must be aligned vectors")
        identity = {
            "store_hash": self.store_hash, "query_drugs_hash": _array_hash(drugs),
            "query_targets_hash": _array_hash(targets), "rows": len(drugs),
            "k": int(k), "exclude_scaffold": bool(exclude_scaffold),
            "label_axis": ["negative", "positive"],
        }
        cache_hash = _json_hash(identity)
        directory = Path(out_dir) / cache_hash
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        names = ("indices", "similarities", "mask")
        paths = {name: directory / f"{name}.npy" for name in names}
        valid_cache = False
        if not force and manifest_path.exists() and all(path.exists() for path in paths.values()):
            try:
                manifest = json.loads(manifest_path.read_text())
                valid_cache = manifest.get("identity") == identity and all(
                    _file_hash(paths[name]) == manifest["array_sha256"][name] for name in names
                )
            except (OSError, KeyError, ValueError):
                valid_cache = False
        if not valid_cache:
            # Validate k/query vectors in retrieve before trusting their cache identity.
            batch = self.retrieve(drugs, targets, k=k, exclude_scaffold=exclude_scaffold, progress=progress)
            manifest_path.unlink(missing_ok=True)
            for name, array in batch.as_dict().items():
                temporary = directory / f"{name}.pending.npy"
                np.save(temporary, array, allow_pickle=False)
                temporary.replace(paths[name])
            manifest = {"identity": identity, "cache_hash": cache_hash,
                        "array_sha256": {name: _file_hash(paths[name]) for name in names}}
            temporary_manifest = directory / "manifest.pending.json"
            temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary_manifest.replace(manifest_path)
        return SupportBatch(*(np.load(paths[name], mmap_mode="r", allow_pickle=False) for name in names))

    def save(self, out_dir: str | Path, *, feature_path: str | Path | None = None) -> Path:
        """Save only pool indices and metadata; the existing feature file is referenced."""
        feature_source = str(Path(feature_path).resolve()) if feature_path else self.feature_path
        if feature_source is None:
            raise ValueError("save requires an existing feature_path; the store never duplicates large features")
        feature_file = Path(feature_source)
        if not feature_file.is_file():
            raise FileNotFoundError(feature_file)
        referenced_features = np.load(feature_file, mmap_mode="r", allow_pickle=False)
        if _array_hash(referenced_features) != self.feature_hash:
            raise ValueError("Referenced feature_path differs from the store feature matrix")
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "support_index.npz"
        temporary_index = directory / "support_index.pending.npz"
        np.savez_compressed(temporary_index, drug_indices=self.pool_drug_indices,
                            target_indices=self.pool_target_indices, labels=self.pool_labels,
                            entity_codes=self.entity_codes, scaffold_codes=self.scaffold_codes)
        temporary_index.replace(index_path)
        manifest = {"metadata": self.metadata, "store_hash": self.store_hash,
                    "feature_path": feature_source, "index_file": index_path.name,
                    "index_sha256": _file_hash(index_path)}
        path = directory / "SUPPORT_STORE_MANIFEST_V3.json"
        temporary_manifest = directory / "SUPPORT_STORE_MANIFEST_V3.pending.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, backend: str = "rdkit", features: np.ndarray | None = None) -> "SupportStore":
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path /= "SUPPORT_STORE_MANIFEST_V3.json"
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest["metadata"]
        if metadata["version"] != STORE_VERSION or _json_hash(metadata) != manifest["store_hash"]:
            raise ValueError("Invalid or unsupported support store manifest")
        store = cls.__new__(cls)
        store.feature_path = manifest["feature_path"]
        store.features = np.asanyarray(features) if features is not None else np.load(store.feature_path, mmap_mode="r", allow_pickle=False)
        store.feature_hash = _array_hash(store.features)
        if store.feature_hash != metadata["feature_hash"]:
            raise ValueError("Feature content hash does not match the support snapshot")
        index_path = manifest_path.parent / manifest["index_file"]
        if _file_hash(index_path) != manifest["index_sha256"]:
            raise ValueError("Support index checksum does not match its manifest")
        with np.load(index_path, allow_pickle=False) as arrays:
            store.pool_drug_indices = arrays["drug_indices"]
            store.pool_target_indices = arrays["target_indices"]
            store.pool_labels = arrays["labels"]
            store.entity_codes = arrays["entity_codes"]
            store.scaffold_codes = arrays["scaffold_codes"]
        store.has_scaffolds = bool(metadata["has_scaffolds"])
        store.metadata, store.store_hash = metadata, manifest["store_hash"]
        store._initialize_backend(backend)
        return store
