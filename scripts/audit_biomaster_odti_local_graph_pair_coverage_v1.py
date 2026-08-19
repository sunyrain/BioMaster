#!/usr/bin/env python3
"""Audit local graph coverage and residual activation by frozen split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_biomaster_odti_baselines_v1 import split_masks  # noqa: E402
from train_biomaster_odti_v2 import load_local_graph_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"))
    parser.add_argument("--store", default=str(ROOT / "outputs/biomaster_odti_local_graph_features_v1"))
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.pairs, low_memory=False)
    store = load_local_graph_features(args.store, int(data["drug_feature_index"].max()) + 1, int(data["target_feature_index"].max()) + 1)
    drug_available = np.asarray(store["ligand_available"])[data["drug_feature_index"].to_numpy(dtype=np.int64)]
    target_available = np.asarray(store["pocket_available"])[data["target_feature_index"].to_numpy(dtype=np.int64)]
    pair_available = drug_available & target_available
    protocols = [
        "S1_SCAFFOLD_COLD_DRUG", "S2_HOMOLOGY_COLD_TARGET", "S3_STRICT_DOUBLE_COLD",
        "S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD",
    ]
    rows=[]
    for protocol in protocols:
        folds = [-1] if protocol in {"S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"} else list(range(5))
        for fold in folds:
            masks=split_masks(data,protocol,fold)
            for role in ["train","valid","test"]:
                positions=np.flatnonzero(masks[role])
                if not len(positions): continue
                rows.append({"protocol":protocol,"fold":fold,"role":role,"rows":len(positions),
                             "ligand_available":float(drug_available[positions].mean()),
                             "pocket_available":float(target_available[positions].mean()),
                             "local_pair_available":float(pair_available[positions].mean())})
    payload={"status":"PASS","pair_count":len(data),"ligand_entity_count":int(drug_available.sum()),
             "pocket_entity_count":int(target_available.sum()),"rows":rows}
    if args.predictions:
        activations=[]
        for path in args.predictions:
            frame=pd.read_csv(path)
            column="v2_local_pair_gate"
            if column in frame:
                activations.append({"path":path,"rows":len(frame),"mean_gate":float(frame[column].mean()),
                                    "active_fraction":float((frame[column]>0).mean()),
                                    "max_gate":float(frame[column].max())})
        payload["prediction_activation"] = activations
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
