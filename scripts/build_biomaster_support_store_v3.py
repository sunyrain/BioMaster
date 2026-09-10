#!/usr/bin/env python3
"""Build an immutable V3 support snapshot from explicit training positions.

Example:
  python scripts/build_biomaster_support_store_v3.py \
    --relations RELATIONS.csv.gz --features MORGAN.npy \
    --feature-index DRUG_FEATURE_INDEX.csv.gz --train-positions train.npy \
    --out outputs/support_v3 --cache-all-relations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.odti_support_data_v3 import SupportStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-positions", "--train-split", dest="train_positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path)
    parser.add_argument("--drug-column", default="drug_feature_index")
    parser.add_argument("--target-column", default="target_feature_index")
    parser.add_argument("--label-column", default="binary_label")
    parser.add_argument("--observed-column", default="binary_observed")
    parser.add_argument("--entity-column", default="model_ligand_smiles")
    parser.add_argument("--scaffold-column", default="murcko_scaffold")
    parser.add_argument("--backend", choices=["rdkit", "numpy", "auto"], default="rdkit")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--exclude-scaffold", action="store_true")
    parser.add_argument("--cache-all-relations", action="store_true")
    parser.add_argument("--cache-queries", type=Path, help="Optional CSV with query drug/target columns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    relations = pd.read_csv(args.relations, low_memory=False)
    features = np.load(args.features, mmap_mode="r", allow_pickle=False)
    positions = np.load(args.train_positions, allow_pickle=False)
    feature_index_path = args.feature_index
    if feature_index_path is None:
        candidate = args.features.parent / "DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz"
        feature_index_path = candidate if candidate.exists() else None
    entities = scaffolds = None
    if feature_index_path is not None:
        index = pd.read_csv(feature_index_path, low_memory=False)
        if index[args.drug_column].duplicated().any():
            raise ValueError("Feature index must have unique drug_feature_index rows")
        index = index.set_index(args.drug_column).reindex(np.arange(len(features)))
        if args.entity_column not in index:
            raise ValueError(f"Entity column {args.entity_column!r} absent from feature index")
        if index[args.entity_column].isna().any():
            raise ValueError("Every feature row must have a chemical entity identity in --feature-index")
        entities = index[args.entity_column].to_numpy()
        if args.scaffold_column in index:
            scaffolds = index[args.scaffold_column].to_numpy()
    else:
        # A chemical entity key is essential when feature rows contain duplicate molecules.
        raise ValueError("Provide --feature-index with feature-aligned chemical entity identifiers")
    store = SupportStore(
        features, relations[args.drug_column].to_numpy(), relations[args.target_column].to_numpy(),
        relations[args.label_column].to_numpy(), positions, entity_keys=entities, scaffold_keys=scaffolds,
        observed=relations[args.observed_column].to_numpy() if args.observed_column in relations else None,
        backend=args.backend, feature_path=args.features,
    )
    manifest_path = store.save(args.out)
    print(json.dumps({"event": "support_store_saved", "manifest": str(manifest_path),
                      "seconds": round(time.monotonic() - started, 2), **store.metadata}), flush=True)
    if args.cache_all_relations and args.cache_queries:
        raise ValueError("Choose either --cache-all-relations or --cache-queries")
    queries = relations if args.cache_all_relations else (
        pd.read_csv(args.cache_queries, low_memory=False) if args.cache_queries else None
    )
    if queries is not None:
        last_report = [0.0]
        def progress(done: int, total: int) -> None:
            elapsed = time.monotonic() - started
            if elapsed - last_report[0] >= 20 or done == total:
                print(json.dumps({"event": "support_retrieval", "done": done,
                                  "total": total, "seconds": round(elapsed, 2)}), flush=True)
                last_report[0] = elapsed
        batch = store.cache_retrieve(args.out / "retrieval_cache", queries[args.drug_column].to_numpy(),
                                     queries[args.target_column].to_numpy(), k=args.k,
                                     exclude_scaffold=args.exclude_scaffold, progress=progress)
        print(json.dumps({"event": "support_cache_complete", "rows": len(batch),
                          "valid_supports": int(batch.mask.sum()),
                          "seconds": round(time.monotonic() - started, 2)}), flush=True)


if __name__ == "__main__":
    main()
