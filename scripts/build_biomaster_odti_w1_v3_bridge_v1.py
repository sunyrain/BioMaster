#!/usr/bin/env python3
"""Audit and materialize the V17 W1 -> ODTI V3 identity bridge.

The operator-facing primary file remains blinded.  This bridge is a data-team
manifest built from the controlled unblinding key and the frozen candidate
identity table.  It verifies whether each W1 candidate already exists in the
ChEMBL37 training store; unseen prospective pairs are explicitly retained as
unseen rather than forced into a calibration_pair_id or a negative label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "w1_result_ingestion_v17"
TEMPLATES = V17 / "input_templates_v17"
IDENTITY = RUN / "final_evidence_routing_v18/W1_CANDIDATE_CHEMBL37_IDENTITY_8_V15.csv"
PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
CONTRACT = ROOT / "outputs/biomaster_odti_v3_data_contract_v1/BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json"
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_w1_v3_bridge_v1"

PRIMARY = TEMPLATES / "W1_BLINDED_PRIMARY_RUN_RESULT_INPUT_16_V17.csv"
BLIND_KEY = TEMPLATES / "W1_CONTROLLED_UNBLINDING_KEY_16_V17.csv"
PROCUREMENT = TEMPLATES / "W1_PROCUREMENT_AND_RECEIVED_LOT_QC_INPUT_8_V17.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_bridge(out_dir: Path) -> dict[str, Any]:
    required = [PRIMARY, BLIND_KEY, PROCUREMENT, IDENTITY, PAIRS, CONTRACT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    primary = pd.read_csv(PRIMARY, low_memory=False)
    blind = pd.read_csv(BLIND_KEY, low_memory=False)
    procurement = pd.read_csv(PROCUREMENT, low_memory=False)
    identity = pd.read_csv(IDENTITY, low_memory=False)
    pairs = pd.read_csv(
        PAIRS,
        low_memory=False,
        usecols=[
            "calibration_pair_id",
            "parent_molecule_chembl_id",
            "parent_standard_inchi_key",
            "target_chembl_id",
        ],
    )
    contract = json.loads(CONTRACT.read_text())

    checks: dict[str, bool] = {
        "v3_contract_pass": contract.get("status") == "PASS",
        "primary_has_16_rows": len(primary) == 16,
        "blind_key_has_16_rows": len(blind) == 16,
        "procurement_has_8_rows": len(procurement) == 8,
        "identity_has_8_rows": len(identity) == 8,
        "identity_candidate_ranks_unique": identity["w1_candidate_rank"].is_unique,
        "blind_key_has_two_runs_per_candidate": (
            blind.groupby("w1_candidate_rank").size().eq(2).all()
            and set(blind["w1_candidate_rank"]) == set(range(1, 9))
        ),
        "primary_operator_file_has_no_drug_identity": not any(
            column in primary.columns
            for column in ["molecule_name", "molecule_inchikey", "w1_candidate_drug"]
        ),
        "primary_rows_locked_before_unblinding": primary["unblinding_status"].eq("LOCKED_NOT_UNBLINDED").all(),
        "primary_and_blind_row_hashes_match": set(primary["frozen_row_key_sha256"]) == set(blind["frozen_row_key_sha256"]),
        "procurement_and_blind_candidate_ranks_match": set(procurement["w1_candidate_rank"]) == set(range(1, 9)),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))

    identity_for_join = identity[
        [
            "w1_candidate_rank",
            "molecule_name",
            "bundle_molecule_chembl_id",
            "molecule_inchikey",
            "molecule_smiles",
        ]
    ].rename(
        columns={
            "molecule_name": "identity_molecule_name",
            "molecule_inchikey": "identity_molecule_inchikey",
        }
    )
    blind_identity = blind.merge(
        identity_for_join,
        on="w1_candidate_rank",
        how="left",
        validate="many_to_one",
    )
    checks["blind_key_identity_matches_candidate_identity"] = bool(
        blind_identity["molecule_inchikey"].map(clean).eq(
            blind_identity["identity_molecule_inchikey"].map(clean)
        ).all()
    )
    if not checks["blind_key_identity_matches_candidate_identity"]:
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    blind_identity["target_entity_key"] = blind_identity["target_chembl_id"].map(clean)
    blind_identity["drug_entity_key"] = blind_identity["molecule_inchikey"].map(clean)
    blind_identity["prospective_pair_id"] = blind_identity.apply(
        lambda row: f"W1_V17_R{int(row['w1_candidate_rank']):02d}_{row['target_chembl_id']}",
        axis=1,
    )

    pair_by_inchi_target = pairs.assign(
        parent_standard_inchi_key=pairs["parent_standard_inchi_key"].map(clean),
        target_chembl_id=pairs["target_chembl_id"].map(clean),
    )
    pair_key = set(zip(pair_by_inchi_target["parent_standard_inchi_key"], pair_by_inchi_target["target_chembl_id"]))
    blind_identity["training_store_overlap_status"] = blind_identity.apply(
        lambda row: (
            "EXACT_FROZEN_PAIR_PRESENT"
            if (row["drug_entity_key"], row["target_entity_key"]) in pair_key
            else "UNSEEN_LIGAND_OR_PAIR"
        ),
        axis=1,
    )
    blind_identity["calibration_pair_id"] = ""
    exact_lookup = {
        (clean(row.parent_standard_inchi_key), clean(row.target_chembl_id)): clean(row.calibration_pair_id)
        for row in pair_by_inchi_target.itertuples()
    }
    for index, row in blind_identity.iterrows():
        key = (row["drug_entity_key"], row["target_entity_key"])
        if key in exact_lookup:
            blind_identity.at[index, "calibration_pair_id"] = exact_lookup[key]

    manifest = blind_identity[
        [
            "frozen_row_key_sha256",
            "w1_candidate_rank",
            "prospective_pair_id",
            "calibration_pair_id",
            "training_store_overlap_status",
            "drug_entity_key",
            "target_entity_key",
            "target_chembl_id",
            "gene_symbol",
            "assay_lane",
            "plate_id",
            "independent_run",
            "blinded_sample_code",
        ]
    ].copy()
    manifest.insert(0, "w1_candidate_id", manifest["w1_candidate_rank"].map(lambda value: f"W1_RANK_{int(value):02d}"))
    manifest["activity_class"] = "PENDING"
    manifest["assay_status"] = "PENDING"
    manifest["unblinding_status"] = "LOCKED_NOT_UNBLINDED"
    manifest["claim_boundary"] = "W1 bridge identity/provenance only; no activity or binding claim"

    checks.update(
        {
            "all_bridge_rows_have_prospective_pair_id": manifest["prospective_pair_id"].ne("").all(),
            "all_bridge_rows_have_exact_entity_keys": manifest["drug_entity_key"].ne("").all()
            & manifest["target_entity_key"].ne("").all(),
            "no_unknown_pairs_encoded_as_negative": not manifest["activity_class"].isin(["inactive", "negative"]).any(),
            "all_current_w1_candidates_are_unseen_in_frozen_store": manifest["training_store_overlap_status"].eq("UNSEEN_LIGAND_OR_PAIR").all(),
        }
    )
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_csv = out_dir / "W1_V17_TO_ODTI_V3_BRIDGE_MANIFEST_V1.csv"
    summary_json = out_dir / "W1_V17_TO_ODTI_V3_BRIDGE_SUMMARY_V1.json"
    summary_md = out_dir / "W1_V17_TO_ODTI_V3_BRIDGE_SUMMARY_V1.md"
    manifest.to_csv(bridge_csv, index=False)

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "bridge_name": "BIOMASTER_ODTI_W1_V17_TO_V3_BRIDGE_V1",
        "claim_status": "IDENTITY_AND_PROVENANCE_BRIDGE_ONLY; NO W1 READOUT AVAILABLE",
        "rows": int(len(manifest)),
        "candidates": int(manifest["w1_candidate_rank"].nunique()),
        "independent_runs": int(manifest["independent_run"].nunique()),
        "training_store_overlap_status_counts": {
            str(key): int(value)
            for key, value in manifest["training_store_overlap_status"].value_counts(dropna=False).items()
        },
        "checks": checks,
        "required_inputs": {
            "primary": relative_or_absolute(PRIMARY),
            "blind_key": relative_or_absolute(BLIND_KEY),
            "procurement": relative_or_absolute(PROCUREMENT),
            "candidate_identity": relative_or_absolute(IDENTITY),
            "frozen_pairs": relative_or_absolute(PAIRS),
            "v3_contract": relative_or_absolute(CONTRACT),
        },
        "input_sha256": {relative_or_absolute(path): sha256(path) for path in required},
        "outputs": {"bridge_manifest": relative_or_absolute(bridge_csv)},
        "rules": {
            "operator_primary_file_remains_blinded": True,
            "calibration_pair_id_nullable_for_unseen_w1": True,
            "prospective_pair_id_required": True,
            "unknown_or_pending_not_negative": True,
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# W1 V17 → ODTI V3 bridge summary",
        "",
        f"状态：`{summary['status']}`；{summary['claim_status']}。",
        "",
        f"- rows：{summary['rows']}（{summary['candidates']} candidates × {summary['independent_runs']} independent runs）",
        f"- frozen-store overlap：{summary['training_store_overlap_status_counts']}",
        "- 当前 8 个 W1 候选均未在冻结 ChEMBL37 pair store 中找到 exact InChIKey–target pair，因此 `calibration_pair_id` 留空，使用 namespaced `prospective_pair_id`。",
        "- operator-facing primary file 仍不含 drug identity；identity 只通过受控 unblinding key 加入 data-team bridge。",
        "",
        "## 关键不变量",
        "",
        "```text",
        "prospective_pair_id required",
        "calibration_pair_id nullable for unseen W1 pairs",
        "unknown/pending != negative",
        "operator primary file remains blinded",
        "```",
    ]
    summary_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "rows": summary["rows"],
        "candidates": summary["candidates"],
        "overlap": summary["training_store_overlap_status_counts"],
        "manifest": str(bridge_csv),
        "summary": str(summary_json),
    }, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    build_bridge(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
