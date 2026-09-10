#!/usr/bin/env python3
"""Audit V3 sequence-region coverage and optionally cache lazy molecular graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomaster.odti_local_features_v3 import DEFAULT_DRUG_INDEX, DEFAULT_TARGET_INDICES, DEFAULT_RESIDUE_ARRAY, DEFAULT_RESIDUE_INDEX, LocalFeatureStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-index", default=str(DEFAULT_DRUG_INDEX))
    parser.add_argument("--target-index", action="append", help="Repeat for base, BindingDB and augmentation indices; defaults to all three.")
    parser.add_argument("--residue-array", default=str(DEFAULT_RESIDUE_ARRAY))
    parser.add_argument("--residue-index", default=str(DEFAULT_RESIDUE_INDEX))
    parser.add_argument("--pocket-store", type=Path, default=None, help="Optional frozen P2Rank regions; omitted uses sequence regions only.")
    parser.add_argument("--max-regions", type=int, default=4)
    parser.add_argument("--tokens-per-region", type=int, default=32)
    parser.add_argument("--max-atoms", type=int, default=128)
    parser.add_argument("--cache-drugs", type=int, default=0, help="Number of drugs to precompute; 0 keeps all drug graphs lazy.")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs/biomaster_local_features_v3"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = LocalFeatureStore(args.drug_index, args.target_index or DEFAULT_TARGET_INDICES, args.residue_array, args.residue_index, max_regions=args.max_regions, tokens_per_region=args.tokens_per_region, max_atoms=args.max_atoms, cache_dir=out / "drug_graphs", pocket_store=args.pocket_store)
    for index in sorted(store.smiles)[:max(0, args.cache_drugs)]:
        store.drug(index)
    report = store.coverage_report()
    report["inputs"] = {"drug_index": args.drug_index, "target_indices": [str(p) for p in (args.target_index or DEFAULT_TARGET_INDICES)], "residue_array": args.residue_array, "residue_index": args.residue_index}
    path = out / "LOCAL_FEATURES_V3_MANIFEST.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
