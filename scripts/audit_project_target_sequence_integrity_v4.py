#!/usr/bin/env python3
"""Audit frozen ConPLEx target sequences against Boltz/P2Rank receptor templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_boltz2_complex_input_package import parse_pdb_sequence  # noqa: E402


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def afdb_entries(accession: str, retries: int = 4) -> list[dict[str, Any]]:
    last_error = ""
    for attempt in range(retries):
        try:
            response = requests.get(
                f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=60
            )
            if response.status_code == 404:
                return []
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"AlphaFold DB query failed for {accession}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="configs/project_targets_v4.csv")
    parser.add_argument(
        "--sequences",
        default="outputs/full_conplex_active_moiety_v4/protein_sequence_representatives.csv",
    )
    parser.add_argument(
        "--top3000",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "pre_boltz_top3000_v4_fully_audited.csv"
        ),
    )
    parser.add_argument(
        "--pocket-source",
        default=(
            "outputs/chembl_moa_enhanced_information_package_v1/"
            "candidate_pool_106k_enhanced_scored.csv"
        ),
    )
    parser.add_argument("--target-extension", default="configs/target_scope_extension_v4.csv")
    parser.add_argument("--integrity-manifest", default="configs/project_targets_v4_integrity.csv")
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/current_production_package_v2/full_untruncated_universe_v4/"
            "target_sequence_integrity_v4"
        ),
    )
    parser.add_argument("--skip-alphafold-resolution", action="store_true")
    args = parser.parse_args()

    targets = pd.read_csv(args.targets, low_memory=False).fillna("")
    sequences = pd.read_csv(args.sequences, low_memory=False).fillna("")
    top = pd.read_csv(args.top3000, low_memory=False).fillna("")
    pocket_source = pd.read_csv(args.pocket_source, low_memory=False).fillna("")
    target_extension = pd.read_csv(args.target_extension, low_memory=False).fillna("")
    pocket_rows = pd.concat(
        [
            pocket_source[["sequence_key", "top_pocket_residue_ids"]],
            target_extension[["sequence_key", "top_pocket_residue_ids"]],
        ],
        ignore_index=True,
    ).drop_duplicates("sequence_key", keep="last")
    pocket_by_sequence = pocket_rows.set_index("sequence_key")["top_pocket_residue_ids"].astype(str)
    sequence_meta = sequences.drop_duplicates("sequence_key").set_index("sequence_key")
    rows: list[dict[str, Any]] = []
    for _, target in targets.iterrows():
        sequence_key = clean(target.get("sequence_key"))
        if sequence_key not in sequence_meta.index:
            raise ValueError(f"Frozen target lacks a ConPLEx sequence: {sequence_key}")
        meta = sequence_meta.loc[sequence_key]
        expected = clean(meta.get("sequence"))
        pdb_path = Path(clean(target.get("pdb_path")))
        if not pdb_path.is_absolute():
            pdb_path = ROOT / pdb_path
        parsed = parse_pdb_sequence(pdb_path)
        template = clean(parsed.get("sequence"))
        status = "exact_match" if expected == template else "sequence_mismatch"
        if not template:
            status = "template_sequence_missing"
        pocket_definition = clean(pocket_by_sequence.get(sequence_key, ""))
        pocket_definition_status = "present" if pocket_definition else "missing"
        pocket_definition_payload = pocket_definition or "NO_POCKET_DEFINITION"
        rows.append(
            {
                **target.to_dict(),
                "expected_representative_protein_id": clean(meta.get("representative_protein_id")),
                "expected_sequence_length": len(expected),
                "expected_sequence_sha256": sha(expected),
                "template_sequence_length": len(template),
                "template_sequence_sha256": sha(template) if template else "",
                "receptor_pdb_sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest()
                if pdb_path.is_file()
                else "",
                "pocket_definition_sha256": sha(pocket_definition_payload),
                "pocket_definition_status": pocket_definition_status,
                "sequence_match_status": status,
                "template_chain": clean(parsed.get("chain")),
                "template_parse_error": clean(parsed.get("error")),
                "alphafold_exact_sequence_entry": "",
                "alphafold_exact_sequence_pdb_url": "",
                "alphafold_resolution_status": "not_queried",
            }
        )

    audit = pd.DataFrame(rows)
    mismatch = audit["sequence_match_status"].ne("exact_match")
    if mismatch.any() and not args.skip_alphafold_resolution:
        for idx, row in audit[mismatch].iterrows():
            expected = clean(sequence_meta.loc[row["sequence_key"]].get("sequence"))
            accessions = []
            for value in [row.get("representative_protein_id"), row.get("expected_representative_protein_id")]:
                accession = re.sub(r"-\d+$", "", clean(value))
                if accession and accession not in accessions:
                    accessions.append(accession)
            exact: list[dict[str, Any]] = []
            query_error = ""
            for accession in accessions:
                try:
                    exact.extend(entry for entry in afdb_entries(accession) if clean(entry.get("sequence")) == expected)
                except Exception as exc:  # noqa: BLE001
                    query_error = str(exc)
            deduplicated = {clean(entry.get("entryId")): entry for entry in exact if clean(entry.get("entryId"))}
            if deduplicated:
                entry = sorted(deduplicated.values(), key=lambda item: clean(item.get("entryId")))[0]
                audit.loc[idx, "alphafold_exact_sequence_entry"] = clean(entry.get("entryId"))
                audit.loc[idx, "alphafold_exact_sequence_pdb_url"] = clean(entry.get("pdbUrl"))
                audit.loc[idx, "alphafold_resolution_status"] = "exact_sequence_model_available"
            elif query_error:
                audit.loc[idx, "alphafold_resolution_status"] = f"query_failed:{query_error}"
            else:
                audit.loc[idx, "alphafold_resolution_status"] = "no_exact_sequence_model_available"

    status_by_sequence = audit.set_index("sequence_key")["sequence_match_status"].to_dict()
    pair_audit = top[["pair_id", "drug_chembl_id", "drug_names", "sequence_key", "primary_gene"]].copy()
    pair_audit["sequence_match_status"] = pair_audit["sequence_key"].map(status_by_sequence)
    pair_audit["structure_sequence_mismatch_v4"] = pair_audit["sequence_match_status"].ne("exact_match")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "project_targets_sequence_integrity_v4.csv"
    pair_path = out_dir / "top3000_sequence_integrity_v4.csv"
    audit.to_csv(target_path, index=False)
    pair_audit.to_csv(pair_path, index=False)
    integrity_columns = [
        "sequence_key",
        "primary_gene",
        "representative_protein_id",
        "pdb_path",
        "expected_representative_protein_id",
        "expected_sequence_length",
        "expected_sequence_sha256",
        "template_sequence_length",
        "template_sequence_sha256",
        "receptor_pdb_sha256",
        "pocket_definition_sha256",
        "pocket_definition_status",
        "sequence_match_status",
    ]
    integrity_path = Path(args.integrity_manifest)
    integrity_path.parent.mkdir(parents=True, exist_ok=True)
    audit[integrity_columns].to_csv(integrity_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_rows": int(len(audit)),
        "exact_match_targets": int(audit["sequence_match_status"].eq("exact_match").sum()),
        "mismatch_targets": int(mismatch.sum()),
        "top3000_rows": int(len(pair_audit)),
        "top3000_mismatch_rows": int(pair_audit["structure_sequence_mismatch_v4"].sum()),
        "alphafold_resolution_counts": audit.loc[mismatch, "alphafold_resolution_status"].value_counts().to_dict(),
        "policy": "Sequence-mismatched rows are excluded from final1000/final384 unless rebuilt and rerun with an exact template.",
        "files": {
            target_path.name: hashlib.sha256(target_path.read_bytes()).hexdigest(),
            pair_path.name: hashlib.sha256(pair_path.read_bytes()).hexdigest(),
            str(integrity_path): hashlib.sha256(integrity_path.read_bytes()).hexdigest(),
        },
    }
    (out_dir / "target_sequence_integrity_v4.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
