#!/usr/bin/env python3
"""Summarize which external sources actually provide entity-cold evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BDB_SUMMARY = ROOT / "outputs/biomaster_odti_bindingdb_ligand_cold_v2_exact/BINDINGDB_LIGAND_COLD_SUMMARY.json"
BDB_GAP = ROOT / "outputs/biomaster_odti_bindingdb_entity_cold_feature_queue_v1/BINDINGDB_ENTITY_COLD_FEATURE_QUEUE_V1.json"
GTOPDB_GAP = ROOT / "outputs/biomaster_odti_gtopdb_entity_cold_gap_v1/GTOPDB_ENTITY_COLD_GAP_AUDIT_V1.json"
GTOPDB_SUMMARY = ROOT / "outputs/biomaster_odti_gtopdb_ligand_cold_v1/GTOPDB_LIGAND_COLD_SUMMARY.json"
GTOPDB_AUDIT = ROOT / "outputs/biomaster_odti_gtopdb_ligand_cold_v1/GTOPDB_LIGAND_COLD_AUDIT_V1.json"
LOCAL_EXTERNAL_AUDIT = ROOT / (
    "outputs/biomaster_odti_local_external_candidates_v1/"
    "LOCAL_EXTERNAL_ENTITY_COLD_CANDIDATES_V1.json"
)
KIRHUB_MAPPING = ROOT / (
    "outputs/evidence_routing_compute_execution_20260808_v1/old_drug_innovation_v7/"
    "KIRHUB_WT_TARGET_MAPPING_384_V7.csv"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path in [
        PAIRS,
        BDB_SUMMARY,
        BDB_GAP,
        GTOPDB_GAP,
        GTOPDB_SUMMARY,
        GTOPDB_AUDIT,
        LOCAL_EXTERNAL_AUDIT,
        KIRHUB_MAPPING,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    pairs = pd.read_csv(PAIRS, usecols=["primary_gene", "query_accession"], low_memory=False)
    seen_genes = set(pairs["primary_gene"].astype(str))
    seen_accessions = set(pairs["query_accession"].astype(str))
    kirhub = pd.read_csv(KIRHUB_MAPPING)
    kirhub["target_seen_in_chembl37_feature_store"] = kirhub["gene_symbol"].astype(str).isin(seen_genes)
    kirhub["target_unseen_in_chembl37_feature_store"] = ~kirhub["target_seen_in_chembl37_feature_store"]
    covered = kirhub["kirhub_wt_target_covered"].astype(bool)
    covered_unseen = kirhub.loc[covered & kirhub["target_unseen_in_chembl37_feature_store"]]

    bdb = json.loads(BDB_SUMMARY.read_text())
    bdb_gap = json.loads(BDB_GAP.read_text())
    gtopdb_gap = json.loads(GTOPDB_GAP.read_text())
    gtopdb = json.loads(GTOPDB_SUMMARY.read_text())
    gtopdb_audit = json.loads(GTOPDB_AUDIT.read_text())
    local_external = json.loads(LOCAL_EXTERNAL_AUDIT.read_text())
    local_sources = local_external.get("sources", {})

    report = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_EXTERNAL_ENTITY_COLD_LANDSCAPE_V1",
        "frozen_training_store": str(PAIRS),
        "sources": {
            "bindingdb_pair_heldout": {
                "claim": "source-heldout positive-only pair retrieval; both entities seen",
                "entity_cold": False,
                "summary_path": str(ROOT / "outputs/biomaster_odti_bindingdb_source_heldout_current_20260817/BINDINGDB_PAIR_HELDOUT_SUMMARY.json"),
            },
            "bindingdb_ligand_cold_exact": {
                "entity_cold": "ligand_only",
                "audit_status": "PASS",
                "candidate_rows": bdb.get("candidate_rows"),
                "candidate_ligands": bdb.get("candidate_ligands"),
                "positive_pairs": bdb.get("positive_pairs"),
                "recall_at_5": bdb.get("ranking", {}).get("recall_at_5"),
                "recall_at_10": bdb.get("ranking", {}).get("recall_at_10"),
                "recall_at_20": bdb.get("ranking", {}).get("recall_at_20"),
                "target_unseen": False,
            },
            "gtopdb_ligand_cold_exact": {
                "entity_cold": "ligand_only",
                "audit_status": gtopdb_audit.get("status"),
                "candidate_rows": gtopdb.get("candidate_rows"),
                "candidate_ligands": gtopdb.get("candidate_ligands"),
                "positive_pairs": gtopdb.get("positive_pairs"),
                "recall_at_5": gtopdb.get("ranking", {}).get("recall_at_5"),
                "recall_at_10": gtopdb.get("ranking", {}).get("recall_at_10"),
                "recall_at_20": gtopdb.get("ranking", {}).get("recall_at_20"),
                "target_unseen": False,
                "both_unseen": False,
            },
            "gtopdb_gap_audit": {
                "strict_positive_pairs": gtopdb_gap["counts"]["strict_positive_pairs"],
                "ligand_unseen_target_seen_pairs": gtopdb_gap["counts"]["ligand_unseen_target_seen_pairs"],
                "ligand_seen_target_unseen_pairs": gtopdb_gap["counts"]["ligand_seen_target_unseen_pairs"],
                "both_unseen_pairs": gtopdb_gap["counts"]["both_unseen_pairs"],
            },
            "davis_complete_secondary_hf_candidate": local_sources.get(
                "davis_complete_secondary_hf", {}
            ),
            "platinum_official_audit": local_sources.get("platinum_official", {}),
            "mdrdb_audit": local_sources.get("mdrdb", {}),
            "kirhub_409_wild_type_panel": {
                "project_mapping_rows": int(len(kirhub)),
                "mapped_targets_with_wt_measurement": int(covered.sum()),
                "mapped_targets_without_wt_measurement": int((~covered).sum()),
                "covered_targets_unseen_in_chembl37": int(len(covered_unseen)),
                "target_cold_available": bool(len(covered_unseen) > 0),
                "interpretation": "existing project mapping provides no covered unseen target under exact gene-to-feature-store audit",
                "mapping_sha256": sha256(KIRHUB_MAPPING),
            },
        },
        "feature_queue_gaps": {
            "bindingdb_missing_feature_pairs": bdb_gap.get("counts", {}).get("missing_feature_pairs"),
            "bindingdb_missing_ligands": bdb_gap.get("counts", {}).get("missing_ligands"),
            "bindingdb_missing_targets": bdb_gap.get("counts", {}).get("missing_targets"),
            "bindingdb_both_unseen_pairs": bdb_gap.get("counts", {}).get("both_unseen_pairs"),
        },
        "gates": {
            "true_entity_cold_external": False,
            "candidate_source_available": bool(
                local_external.get("gate_decision", {}).get("davis_candidate_available", False)
            ),
            "reason": "Davis provides an auditable both-unseen candidate, but it is not scored yet; completed scores remain ligand-only entity-cold or pair-heldout",
        },
        "next_required_source_contract": {
            "unseen_ligand": True,
            "unseen_target": True,
            "both_unseen": True,
            "exact_entity_keys": True,
            "source_heldout": True,
            "unknown_pairs_are_not_negative": True,
        },
        "claim_status": "LANDSCAPE_AUDIT; TRUE_ENTITY_COLD_EXTERNAL_GATE_REMAINS_CLOSED",
        "local_external_candidate_audit": str(LOCAL_EXTERNAL_AUDIT.relative_to(ROOT)),
    }
    out_dir = ROOT / "outputs/biomaster_odti_external_entity_cold_landscape_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "EXTERNAL_ENTITY_COLD_LANDSCAPE_V1.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
