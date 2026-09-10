#!/usr/bin/env python3
"""Freeze label-free P2Rank candidates for the V3 local interaction model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomaster.odti_local_features_v3 import DEFAULT_TARGET_INDICES
from biomaster.odti_pockets_v3 import DEFAULT_POCKET_MASTER, DEFAULT_POCKET_STORE, build_pocket_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_POCKET_MASTER)
    parser.add_argument("--target-index", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=DEFAULT_POCKET_STORE)
    parser.add_argument("--min-probability", type=float, default=0.2)
    parser.add_argument("--max-pockets", type=int, default=3)
    parser.add_argument("--jaccard-threshold", type=float, default=0.5)
    parser.add_argument("--min-residues", type=int, default=5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("frozen pocket cache already exists; select a new output path")
    payload = build_pocket_store(args.master, args.target_index or list(DEFAULT_TARGET_INDICES), min_probability=args.min_probability, max_pockets=args.max_pockets, jaccard_threshold=args.jaccard_threshold, min_residues=args.min_residues)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.pending")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), **payload["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
