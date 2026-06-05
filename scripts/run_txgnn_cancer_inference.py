from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def normalize_name(value: str) -> str:
    value = (value or "").lower()
    value = re.sub(
        r"\b(usp|hydrochloride|hcl|dihydrochloride|mesylate|tosylate|ditosylate|dimesylate|dimeglumine|sodium|potassium|calcium|phosphate|sulfate|sulphate|acetate|maleate|dimaleate|fumarate|tartrate|bitartrate|malate|succinate|citrate|esylate|besylate|adipate|pamoate|lactate|gluconate|choline|meglumine|tromethamine|bromide|chloride|dichloride|nitrate|hydrate|monohydrate|dihydrate|anhydrous|injection|tablets?|capsules?)\b",
        " ",
        value,
    )
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    value = re.sub(r"\b[drsl]\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_numeric_identifier(value: Any) -> str:
    text = str(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_name_crosswalk(
    drugs_csv: Path,
    name_mapping_path: Path,
    out_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, drugs = read_rows(drugs_csv)
    with name_mapping_path.open("rb") as handle:
        mapping = pickle.load(handle)

    id2name_drug = mapping["id2name_drug"]
    txgnn_by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for drugbank_id, txgnn_name in id2name_drug.items():
        normalized = normalize_name(str(txgnn_name))
        if normalized:
            txgnn_by_name[normalized].append((str(drugbank_id), str(txgnn_name)))

    crosswalk: list[dict[str, Any]] = []
    for row in drugs:
        matches: list[dict[str, Any]] = []
        for field in ["drug_name", "brand_name", "chembl_pref_name"]:
            normalized = normalize_name(row.get(field, ""))
            if normalized and normalized in txgnn_by_name:
                for drugbank_id, txgnn_name in txgnn_by_name[normalized]:
                    matches.append(
                        {
                            "match_field": field,
                            "normalized_match_name": normalized,
                            "txgnn_drugbank_id": drugbank_id,
                            "txgnn_drug_name": txgnn_name,
                        }
                    )
                break

        if matches:
            primary = sorted(matches, key=lambda item: item["txgnn_drugbank_id"])[0]
            status = "mapped_exact_normalized_name"
            candidate_count = len(matches)
        else:
            primary = {
                "match_field": "",
                "normalized_match_name": "",
                "txgnn_drugbank_id": "",
                "txgnn_drug_name": "",
            }
            status = "unmapped"
            candidate_count = 0

        crosswalk.append(
            {
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "brand_name": row.get("brand_name", ""),
                "chembl_id": row.get("chembl_id", ""),
                "chembl_pref_name": row.get("chembl_pref_name", ""),
                "pubchem_cid": row.get("pubchem_cid", ""),
                "inchikey": row.get("inchikey", ""),
                "txgnn_drugbank_id": primary["txgnn_drugbank_id"],
                "txgnn_drug_name": primary["txgnn_drug_name"],
                "match_field": primary["match_field"],
                "normalized_match_name": primary["normalized_match_name"],
                "match_status": status,
                "candidate_count": candidate_count,
            }
        )

    fields = [
        "drug_id",
        "drug_name",
        "brand_name",
        "chembl_id",
        "chembl_pref_name",
        "pubchem_cid",
        "inchikey",
        "txgnn_drugbank_id",
        "txgnn_drug_name",
        "match_field",
        "normalized_match_name",
        "match_status",
        "candidate_count",
    ]
    write_rows(out_csv, fields, crosswalk)
    metadata = {
        "drug_rows": len(drugs),
        "txgnn_drug_nodes": len(id2name_drug),
        "mapped_rows": sum(1 for row in crosswalk if row["match_status"] != "unmapped"),
        "unmapped_rows": sum(1 for row in crosswalk if row["match_status"] == "unmapped"),
        "mapping_rule": "exact match after conservative name normalization against TxGNN id2name_drug; no fuzzy matching is used",
    }
    return crosswalk, metadata


def run_txgnn_predictions(
    txgnn_data_dir: Path,
    model_dir: Path,
    crosswalk: list[dict[str, Any]],
    disease_mondo_numeric_id: str,
    disease_id: str,
    disease_name: str,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from txgnn import TxData, TxGNN

    tx_data = TxData(data_folder_path=str(txgnn_data_dir))
    tx_data.prepare_split(split="full_graph", seed=42)

    mappings = tx_data.retrieve_id_mapping()
    idx2id_drug = {float(k): str(v) for k, v in mappings["idx2id_drug"].items()}
    idx2id_disease = {float(k): str(v) for k, v in mappings["idx2id_disease"].items()}
    id2idx_drug = {drugbank_id: idx for idx, drugbank_id in idx2id_drug.items()}

    disease_idx = None
    requested_disease_id = normalize_numeric_identifier(disease_mondo_numeric_id)
    for idx, disease_id in idx2id_disease.items():
        if normalize_numeric_identifier(disease_id) == requested_disease_id:
            disease_idx = idx
            break
    if disease_idx is None:
        raise ValueError(f"Could not find TxGNN disease id {disease_mondo_numeric_id}")

    tx_model = TxGNN(data=tx_data, weight_bias_track=False, device=device)
    tx_model.load_pretrained(str(model_dir))

    mapped = [row for row in crosswalk if row.get("txgnn_drugbank_id") in id2idx_drug]
    if not mapped:
        raise ValueError("No BioMaster drugs mapped to TxGNN drug nodes")

    pred_df = pd.DataFrame(
        {
            "x_idx": [id2idx_drug[row["txgnn_drugbank_id"]] for row in mapped],
            "relation": ["indication"] * len(mapped),
            "y_idx": [disease_idx] * len(mapped),
        }
    )
    with torch.no_grad():
        pred_score = tx_model.predict(pred_df)
        logits = pred_score[("drug", "indication", "disease")].reshape(-1).detach().cpu()
        probs = torch.sigmoid(logits).numpy()

    rows: list[dict[str, Any]] = []
    for item, logit, prob in zip(mapped, logits.numpy(), probs):
        rows.append(
            {
                "drug_id": item["drug_id"],
                "drug_name": item["drug_name"],
                "txgnn_drugbank_id": item["txgnn_drugbank_id"],
                "txgnn_drug_name": item["txgnn_drug_name"],
                "disease_id": disease_id,
                "disease_name": disease_name,
                "txgnn_disease_numeric_id": disease_mondo_numeric_id,
                "txgnn_disease_idx": int(disease_idx),
                "relation": "indication",
                "txgnn_indication_logit": float(logit),
                "txgnn_indication_score": float(prob),
                "mapping_status": item["match_status"],
                "mapping_field": item["match_field"],
                "mapping_rule": "exact_normalized_name_to_txgnn_drugbank_node",
            }
        )

    metadata = {
        "txgnn_split": "full_graph",
        "disease_id": disease_id,
        "disease_name": disease_name,
        "txgnn_disease_numeric_id": disease_mondo_numeric_id,
        "txgnn_disease_idx": int(disease_idx),
        "mapped_drugs_in_crosswalk": len([row for row in crosswalk if row.get("txgnn_drugbank_id")]),
        "mapped_drugs_present_in_txgnn_graph": len(mapped),
        "prediction_rows": len(rows),
        "device": device,
        "score_note": "txgnn_indication_score is sigmoid(txgnn_indication_logit).",
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TxGNN indication inference for a selected disease and BioMaster mapped drugs.")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--txgnn-data-dir", default="data/raw/txgnn")
    parser.add_argument("--model-dir", default="data/raw/txgnn/TxGNNExplorer")
    parser.add_argument("--crosswalk-out", default="data/processed/txgnn_drug_name_crosswalk.csv")
    parser.add_argument("--out", default="data/processed/txgnn_drug_disease_scores.csv")
    parser.add_argument("--metadata-out", default="data/processed/txgnn_drug_disease_scores.metadata.json")
    parser.add_argument("--disease-mondo-numeric-id", default="4992")
    parser.add_argument("--disease-id", default="MONDO_0004992")
    parser.add_argument("--disease-name", default="cancer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--crosswalk-only", action="store_true")
    args = parser.parse_args()

    crosswalk, crosswalk_metadata = build_name_crosswalk(
        drugs_csv=Path(args.drugs),
        name_mapping_path=Path(args.model_dir) / "name_mapping.pkl",
        out_csv=Path(args.crosswalk_out),
    )

    metadata: dict[str, Any] = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_drugs": args.drugs,
        "txgnn_data_dir": args.txgnn_data_dir,
        "model_dir": args.model_dir,
        "crosswalk_csv": args.crosswalk_out,
        "output_csv": args.out,
        "crosswalk": crosswalk_metadata,
    }

    if not args.crosswalk_only:
        rows, prediction_metadata = run_txgnn_predictions(
            txgnn_data_dir=Path(args.txgnn_data_dir),
            model_dir=Path(args.model_dir),
            crosswalk=crosswalk,
            disease_mondo_numeric_id=args.disease_mondo_numeric_id,
            disease_id=args.disease_id,
            disease_name=args.disease_name,
            device=args.device,
        )
        fields = [
            "drug_id",
            "drug_name",
            "txgnn_drugbank_id",
            "txgnn_drug_name",
            "disease_id",
            "disease_name",
            "txgnn_disease_numeric_id",
            "txgnn_disease_idx",
            "relation",
            "txgnn_indication_logit",
            "txgnn_indication_score",
            "mapping_status",
            "mapping_field",
            "mapping_rule",
        ]
        write_rows(Path(args.out), fields, rows)
        metadata["prediction"] = prediction_metadata

    Path(args.metadata_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_out).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
