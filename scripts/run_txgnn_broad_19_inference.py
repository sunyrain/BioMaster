#!/usr/bin/env python3
"""Run TxGNN drug-disease indication inference for all broad directions.

The single-direction script reloads the full graph and model for each disease.
This batch runner loads TxData/full_graph and the pretrained model once, then
scores every direction listed in txgnn_broad_proxy_catalog.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_txgnn_cancer_inference import build_name_crosswalk, normalize_numeric_identifier


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def read_catalog(path: Path, directions: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "direction",
        "direction_label_zh",
        "preferred_txgnn_proxy_id",
        "preferred_txgnn_proxy_name",
        "preferred_proxy_present_in_txgnn_node_table",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TxGNN proxy catalog is missing columns: {sorted(missing)}")
    df["direction"] = df["direction"].map(clean)
    df["preferred_txgnn_proxy_id"] = df["preferred_txgnn_proxy_id"].map(clean)
    df["preferred_txgnn_proxy_name"] = df["preferred_txgnn_proxy_name"].map(clean)
    if directions:
        df = df[df["direction"].isin(directions)].copy()
    df = df[df["preferred_txgnn_proxy_id"].ne("")].copy()
    if df.empty:
        raise ValueError("No cataloged TxGNN directions to run.")
    return df


def id_lookup(mapping: dict[Any, Any]) -> tuple[dict[str, int], dict[int, str]]:
    id_to_idx: dict[str, int] = {}
    idx_to_id: dict[int, str] = {}
    for idx_raw, node_id_raw in mapping.items():
        node_id = normalize_numeric_identifier(clean(node_id_raw))
        try:
            idx = int(float(idx_raw))
        except (TypeError, ValueError):
            continue
        id_to_idx[node_id] = idx
        idx_to_id[idx] = node_id
    return id_to_idx, idx_to_id


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_direction(
    tx_model: Any,
    mapped_drugs: list[dict[str, Any]],
    id2idx_drug: dict[str, int],
    disease_idx: int,
    direction_row: pd.Series,
) -> list[dict[str, Any]]:
    pred_df = pd.DataFrame(
        {
            "x_idx": [id2idx_drug[row["txgnn_drugbank_id"]] for row in mapped_drugs],
            "relation": ["indication"] * len(mapped_drugs),
            "y_idx": [disease_idx] * len(mapped_drugs),
        }
    )
    with torch.no_grad():
        pred_score = tx_model.predict(pred_df)
        logits = pred_score[("drug", "indication", "disease")].reshape(-1).detach().cpu()
        probs = torch.sigmoid(logits).numpy()

    rows: list[dict[str, Any]] = []
    for item, logit, prob in zip(mapped_drugs, logits.numpy(), probs):
        rows.append(
            {
                "direction": clean(direction_row["direction"]),
                "direction_label_zh": clean(direction_row["direction_label_zh"]),
                "drug_id": item["drug_id"],
                "drug_name": item["drug_name"],
                "txgnn_drugbank_id": item["txgnn_drugbank_id"],
                "txgnn_drug_name": item["txgnn_drug_name"],
                "disease_id": clean(direction_row.get("direction_disease_id", "")),
                "txgnn_proxy_id": clean(direction_row["preferred_txgnn_proxy_id"]),
                "txgnn_proxy_name": clean(direction_row["preferred_txgnn_proxy_name"]),
                "txgnn_disease_idx": disease_idx,
                "relation": "indication",
                "txgnn_indication_logit": float(logit),
                "txgnn_indication_score": float(prob),
                "mapping_status": item["match_status"],
                "mapping_field": item["match_field"],
                "mapping_rule": "exact_normalized_name_to_txgnn_drugbank_node",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch TxGNN inference for broad disease directions.")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--catalog", default="outputs/broad_mechanism_layer_v2/txgnn_broad_proxy_catalog.csv")
    parser.add_argument("--direction-catalog", default="outputs/broad_mechanism_layer_v2/direction_catalog.csv")
    parser.add_argument("--txgnn-data-dir", default="data/raw/txgnn")
    parser.add_argument("--model-dir", default="data/raw/txgnn/TxGNNExplorer")
    parser.add_argument("--crosswalk-out", default="outputs/broad_mechanism_layer_v2/txgnn_direction_runs/txgnn_drug_name_crosswalk.csv")
    parser.add_argument("--outdir", default="outputs/broad_mechanism_layer_v2/txgnn_direction_runs")
    parser.add_argument("--merged-out", default="outputs/broad_mechanism_layer_v2/txgnn_broad_19_direction_scores.csv")
    parser.add_argument("--metadata-out", default="outputs/broad_mechanism_layer_v2/txgnn_broad_19_direction_scores.metadata.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--directions", nargs="*", default=None, help="Optional subset of direction names.")
    args = parser.parse_args()

    started = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    directions = set(args.directions) if args.directions else None
    catalog = read_catalog(Path(args.catalog), directions=directions)
    if Path(args.direction_catalog).exists():
        direction_catalog = pd.read_csv(args.direction_catalog)
        catalog = catalog.merge(
            direction_catalog[["direction", "disease_id"]].rename(columns={"disease_id": "direction_disease_id"}),
            on="direction",
            how="left",
        )
    else:
        catalog["direction_disease_id"] = ""

    crosswalk, crosswalk_metadata = build_name_crosswalk(
        drugs_csv=Path(args.drugs),
        name_mapping_path=Path(args.model_dir) / "name_mapping.pkl",
        out_csv=Path(args.crosswalk_out),
    )

    from txgnn import TxData, TxGNN

    print("[TxGNN] loading TxData/full_graph", flush=True)
    tx_data = TxData(data_folder_path=str(args.txgnn_data_dir))
    tx_data.prepare_split(split="full_graph", seed=42)
    mappings = tx_data.retrieve_id_mapping()

    id2idx_drug, _idx2id_drug = id_lookup(mappings["idx2id_drug"])
    id2idx_disease, _idx2id_disease = id_lookup(mappings["idx2id_disease"])
    mapped = [row for row in crosswalk if clean(row.get("txgnn_drugbank_id")) in id2idx_drug]
    if not mapped:
        raise ValueError("No BioMaster drugs mapped to TxGNN drug nodes.")

    print(f"[TxGNN] loading pretrained model from {args.model_dir}", flush=True)
    tx_model = TxGNN(data=tx_data, weight_bias_track=False, device=args.device)
    tx_model.load_pretrained(str(args.model_dir))

    fieldnames = [
        "direction",
        "direction_label_zh",
        "drug_id",
        "drug_name",
        "txgnn_drugbank_id",
        "txgnn_drug_name",
        "disease_id",
        "txgnn_proxy_id",
        "txgnn_proxy_name",
        "txgnn_disease_idx",
        "relation",
        "txgnn_indication_logit",
        "txgnn_indication_score",
        "mapping_status",
        "mapping_field",
        "mapping_rule",
    ]

    all_rows: list[dict[str, Any]] = []
    direction_meta: list[dict[str, Any]] = []
    for item in catalog.itertuples(index=False):
        row = pd.Series(item._asdict())
        direction = clean(row["direction"])
        proxy_id = normalize_numeric_identifier(clean(row["preferred_txgnn_proxy_id"]))
        disease_idx = id2idx_disease.get(proxy_id)
        status = "scored"
        error = ""
        rows: list[dict[str, Any]] = []
        if disease_idx is None:
            status = "missing_txgnn_disease_node"
            error = f"TxGNN disease node not found for proxy_id={proxy_id}"
            print(f"[TxGNN] {direction}: {error}", flush=True)
        else:
            print(f"[TxGNN] scoring {direction}: {proxy_id} / {clean(row['preferred_txgnn_proxy_name'])}", flush=True)
            rows = score_direction(tx_model, mapped, id2idx_drug, disease_idx, row)
            write_csv(outdir / f"txgnn_{direction}_scores.csv", rows, fieldnames)
            all_rows.extend(rows)
        direction_meta.append(
            {
                "direction": direction,
                "direction_label_zh": clean(row.get("direction_label_zh", "")),
                "txgnn_proxy_id": proxy_id,
                "txgnn_proxy_name": clean(row.get("preferred_txgnn_proxy_name", "")),
                "txgnn_disease_idx": disease_idx if disease_idx is not None else "",
                "status": status,
                "error": error,
                "prediction_rows": len(rows),
                "mapped_drugs_present_in_txgnn_graph": len(mapped),
            }
        )

    write_csv(Path(args.merged_out), all_rows, fieldnames)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "input_drugs": args.drugs,
        "catalog": args.catalog,
        "txgnn_data_dir": args.txgnn_data_dir,
        "model_dir": args.model_dir,
        "crosswalk_csv": args.crosswalk_out,
        "merged_output_csv": args.merged_out,
        "directions_requested": int(len(catalog)),
        "directions_scored": int(sum(1 for row in direction_meta if row["status"] == "scored")),
        "prediction_rows": int(len(all_rows)),
        "unique_drugs_scored": int(pd.DataFrame(all_rows)["drug_id"].nunique()) if all_rows else 0,
        "device": args.device,
        "crosswalk": crosswalk_metadata,
        "direction_runs": direction_meta,
        "score_note": "txgnn_indication_score is sigmoid(txgnn_indication_logit). TxGNN is drug-disease evidence, not drug-target binding evidence.",
    }
    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_out).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
