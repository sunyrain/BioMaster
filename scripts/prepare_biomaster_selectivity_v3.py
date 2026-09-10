#!/usr/bin/env python3
"""Prepare the V3 entity-held-out experiment and its exact support cache."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.odti_support_data_v3 import SupportBatch, SupportStore
from biomaster.selectivity_training_v3 import SPLIT_VERSION, entity_hash, prepare_relations

DEFAULT_PACKAGE = ROOT / "outputs/retrain_20260901/comprehensive_training_v1"
DEFAULT_RELATIONS = DEFAULT_PACKAGE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz"
DEFAULT_FEATURES = DEFAULT_PACKAGE / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy"
DEFAULT_OUT = ROOT / "outputs/biomaster_v3_20260905/data"
SUPPORT_K = 16
RETAIN_COLUMNS = ["source_row", "v3_row", "drug_feature_index", "target_feature_index", "binary_label",
                  "entity_key", "split", "family_index", "target_assay_family", "murcko_scaffold",
                  "calibration_pair_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment_hash(data: pd.DataFrame) -> str:
    return hashlib.sha256(data[["entity_key", "target_feature_index", "binary_label", "split"]]
                          .to_csv(index=False).encode()).hexdigest()


@contextmanager
def preparation_lock(path: Path):
    with path.open("a") as handle:
        print(json.dumps({"event": "waiting_for_data_preparation_lock", "path": str(path)}), flush=True)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prepare_training_data(
    relations_path: str | Path,
    features_path: str | Path,
    out_dir: str | Path,
    max_entities: int = 0,
    k: int = SUPPORT_K,
) -> tuple[pd.DataFrame, dict, SupportStore, SupportBatch]:
    """Create or validate/reuse a complete V3 data and support snapshot.

    A mismatched completed manifest raises explicitly; select a new output
    directory to build a different experiment. Partial runs without a completed
    data manifest can restart and reuse matching low-level support caches.
    """
    if isinstance(k, bool) or int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    relations_path, features_path = Path(relations_path).resolve(), Path(features_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with preparation_lock(out_dir / ".prepare.lock"):
        return _prepare_locked(relations_path, features_path, out_dir, max_entities, int(k))


def _prepare_locked(relations_path: Path, features_path: Path, out_dir: Path, max_entities: int, k: int):
    started = time.monotonic()
    log_path = out_dir / "PREPARE_V3_LOG.jsonl"
    def report(event: str, **details) -> None:
        payload = {"event": event, "elapsed_seconds": round(time.monotonic() - started, 2), **details}
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")
    report("hashing_source_inputs")
    identity = {
        "protocol": SPLIT_VERSION,
        "relations_path": str(relations_path), "relations_sha256": sha256_file(relations_path),
        "features_path": str(features_path), "features_sha256": sha256_file(features_path),
        "max_entities": int(max_entities), "support_k": k,
        "prepare_relations_code_sha256": hashlib.sha256(
            (inspect.getsource(prepare_relations) + inspect.getsource(entity_hash)).encode()).hexdigest(),
    }
    manifest_path = out_dir / "DATA_MANIFEST_V3.json"
    prepared_path = out_dir / "RELATIONS_V3.csv.gz"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            expected = manifest.get("identity", {})
            differences = {key: {"stored": expected.get(key), "requested": value}
                           for key, value in identity.items() if expected.get(key) != value}
            raise ValueError(f"V3 data manifest differs from requested experiment: {differences}. Use a new out_dir.")
        if sha256_file(prepared_path) != manifest["prepared_relations_sha256"]:
            raise ValueError("Prepared V3 relation table checksum changed")
        data = pd.read_csv(prepared_path, low_memory=False)
        audit = manifest["audit"]
        if assignment_hash(data) != manifest["assignment_sha256"]:
            raise ValueError("V3 entity/target/label/split assignments differ from saved manifest")
        store = SupportStore.load(out_dir / "support_store")
        if store.store_hash != manifest["store_hash"]:
            raise ValueError("Support snapshot does not match the V3 data manifest")
        report("reusing_verified_data_snapshot", rows=len(data), store_hash=store.store_hash)
    else:
        report("reading_and_splitting_relations")
        raw = pd.read_csv(relations_path, low_memory=False)
        data, audit = prepare_relations(raw, max_entities=max_entities)
        del raw
        data = data[[column for column in RETAIN_COLUMNS if column in data]].copy()
        if "murcko_scaffold" not in data:
            data["murcko_scaffold"] = ""
        if "calibration_pair_id" not in data:
            data["calibration_pair_id"] = ""
        features = np.load(features_path, mmap_mode="r", allow_pickle=False)
        if data.drug_feature_index.max() >= len(features):
            raise ValueError("Prepared drug index exceeds source feature matrix")
        # Aliased molecule indices were already merged by prepare_relations.
        # Unused feature rows retain distinct placeholder identities and no scaffold.
        entities = np.asarray([f"__unused_feature_{index}" for index in range(len(features))], dtype=object)
        scaffolds = np.full(len(features), "", dtype=object)
        molecule_rows = data.drop_duplicates("drug_feature_index")
        entities[molecule_rows.drug_feature_index.to_numpy()] = molecule_rows.entity_key.to_numpy()
        known_scaffolds = data.loc[data.murcko_scaffold.notna() & data.murcko_scaffold.astype(str).ne("")]
        scaffold_rows = known_scaffolds.drop_duplicates("drug_feature_index")
        scaffolds[scaffold_rows.drug_feature_index.to_numpy()] = scaffold_rows.murcko_scaffold.to_numpy()
        audit["features_shape"] = list(features.shape)
        audit["scaffold_conflicting_feature_rows"] = int(
            known_scaffolds.groupby("drug_feature_index").murcko_scaffold.nunique().gt(1).sum())
        report("building_support_snapshot", rows=len(data), train_rows=int(data.split.eq("train").sum()),
               audit=audit)
        store = SupportStore(features, data.drug_feature_index.to_numpy(), data.target_feature_index.to_numpy(),
                             data.binary_label.to_numpy(), np.flatnonzero(data.split.eq("train").to_numpy()),
                             entity_keys=entities, scaffold_keys=scaffolds, feature_path=features_path)
        store.save(out_dir / "support_store")
        temporary_path = out_dir / "RELATIONS_V3.pending.csv.gz"
        data.to_csv(temporary_path, index=False, compression="gzip")
        temporary_path.replace(prepared_path)
        report("support_snapshot_saved", metadata=store.metadata)
    last_report = [time.monotonic()]
    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last_report[0] >= 20 or done == total:
            report("support_cache_progress", done=done, total=total)
            last_report[0] = now
    batch = store.cache_retrieve(out_dir / "support_normal", data.drug_feature_index.to_numpy(),
                                 data.target_feature_index.to_numpy(), k=k, progress=progress)
    if not manifest_path.exists():
        manifest = {
            "identity": identity, "audit": audit, "assignment_sha256": assignment_hash(data),
            "prepared_relations_path": str(prepared_path), "prepared_relations_sha256": sha256_file(prepared_path),
            "store_hash": store.store_hash,
            "support_indices_path": str(batch.indices.filename),
            "support_similarities_path": str(batch.similarities.filename),
            "support_mask_path": str(batch.mask.filename),
            "support_shape": list(batch.indices.shape), "label_axis": ["negative", "positive"],
            "status": "COMPLETE",
        }
        temporary_manifest = out_dir / "DATA_MANIFEST_V3.pending.json"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(manifest_path)
    report("v3_data_complete", rows=len(data), support_shape=list(batch.indices.shape),
           valid_supports=int(batch.mask.sum()), manifest=str(manifest_path))
    return data, audit, store, batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-entities", type=int, default=0)
    parser.add_argument("--support-k", "--k", dest="support_k", type=int, default=SUPPORT_K)
    args = parser.parse_args()
    prepare_training_data(args.relations, args.features, args.out, max_entities=args.max_entities, k=args.support_k)


if __name__ == "__main__":
    main()
