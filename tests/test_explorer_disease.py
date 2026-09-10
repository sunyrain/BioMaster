"""Small source fixtures exercise ordering, exclusion and indexed provenance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biomaster.explorer_disease import (
    DiseaseEvidenceStore, DIRECTION_LABELS, SNAPSHOT, REQUIRED, LOGITS, MASK, OT, EC, PAIR, NODES,
)


def csv_file(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def json_file(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def refresh_manifest(path: Path) -> None:
    artifacts = {f"{SNAPSHOT}/{name}": hashlib.sha256((path / name).read_bytes()).hexdigest()
                 for name in REQUIRED if name != "SOURCE_AND_ARTIFACT_MANIFEST.json"}
    json_file(path / "SOURCE_AND_ARTIFACT_MANIFEST.json", {"no_training": True, "artifact_sha256": artifacts})


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / SNAPSHOT; path.mkdir(parents=True)
    drugs = {key: {"name": name} for key, name in [("D1", "Drug One"), ("D2", "Drug Two"), ("HOLD", "Held Drug"), ("UNMAPPED", "Unmapped Drug")]}
    # A legacy value must never be used as disease inference for an unmapped drug.
    drugs["UNMAPPED"]["txgnn_diseases"] = [{"id": "cancer", "score": 99}]
    targets = {"T1": {"name": "Target One"}, "T2": {"name": "Target Two"}}
    cov = []
    for identifier, entity in drugs.items():
        cov.append({"ligand_inchikey": identifier, "drug_names": entity["name"],
                    "graph_drug_id": "" if identifier == "UNMAPPED" else "DB_" + identifier,
                    "graph_drug_name": entity["name"], "mapping_rule": "EXACT_NAME",
                    "mapping_status": "MAPPED_NAME_STRUCTURE_NOT_VERIFIED_IN_GRAPH" if identifier != "UNMAPPED" else "UNMAPPED",
                    "candidate_ids": "", "identity_hold": False,
                    "inference_eligible": identifier != "UNMAPPED",
                    "identity_concordance": "STRUCTURE_AND_GRAPH_IDENTIFIER_CONFLICT_HOLD" if identifier == "HOLD" else "SAME_NORMALIZED_EC_NODE_NOT_STEREO_PROOF",
                    "interpretation_eligible": identifier in {"D1", "D2"}, "ec_node_count": 1,
                    "ec_disease_endpoint_count": 1 if identifier == "D1" else 0,
                    "ec_non_text_asserted_disease_endpoint_count": 1 if identifier == "D1" else 0,
                    "disease_evidence_status": "TXGNN_QUERY_AND_SOURCE_EVIDENCE_REVIEW" if identifier in {"D1", "D2"} else "SOURCE_EVIDENCE_ONLY_NO_USABLE_TXGNN_SCORE" if identifier == "HOLD" else "IDENTIFIED_ENTITY_NO_DISEASE_EVIDENCE_IN_THIS_RUN"})
    csv_file(path / "DRUG_COVERAGE_FINAL_720.csv", cov)
    csv_file(path / "TARGET_COVERAGE_888.csv", [{"target_chembl_id": key, "gene_symbol": entity["name"], "old_gene_ids": "123", "old_gene_mapping_status": "MAPPED", "ec_node_count": 1, "ot_status": "COMPLETE" if key == "T1" else "MISSING_ENSEMBL"} for key, entity in targets.items()])
    csv_file(path / "OT_TARGET_COVERAGE_888.csv", [{"target_chembl_id": key, "gene_symbol": entity["name"], "ot_id": "ENSG1" if key == "T1" else "", "status": "COMPLETE" if key == "T1" else "MISSING_ENSEMBL", "count": 2 if key == "T1" else 0} for key, entity in targets.items()])
    # Physical order differs both from graph indices and the catalog order.
    csv_file(path / "TXGNN_SCORE_DRUG_ORDER.csv", [dict(next(row for row in cov if row["ligand_inchikey"] == key), drug_idx=graph_index) for key, graph_index in [("D2", 99), ("D1", 0), ("HOLD", 42)]])
    diseases = [{"id": key, "node_name": name, "idx": index, "node_source": source} for index, (key, name, source) in enumerate([("1", "Neural disease", "MONDO"), ("2", "Excluded neoplasm", "MONDO"), ("3", "Unclassified disease", "MONDO"), ("4_5", "Merged neoplasm node", "MONDO_grouped")])]
    csv_file(path / "TXGNN_SCORE_DISEASE_ORDER.csv", diseases)
    csv_file(path / "TXGNN_ALL_DISEASE_DIRECTIONS.csv", [dict(row, direction_labels=direction) for row, direction in zip(diseases, ["neurological", "neoplasm", "UNCLASSIFIED_BY_EC_ONTOLOGY", "neoplasm"])])
    csv_file(path / "DISEASE_ID_CROSSWALK.csv", [{"disease_id": f"MONDO:{member:07d}", "txgnn_disease_id": diseases[index]["id"], "score_column": index, "mapping_scope": "MEMBER_OF_MERGED_NODE" if index == 3 else "EXACT_SINGLE_NODE", "merged_member_count": 2 if index == 3 else 1} for index, members in enumerate([[1], [2], [3], [4, 5]]) for member in members])
    np.save(path / LOGITS, np.array([[1, 9, 8, 7], [10, 20, 30, 40], [100, 90, 80, 70]], dtype=np.float32))
    np.save(path / MASK, np.array([[0, 1, 0, 0], [1, 0, 2, 0], [4, 4, 4, 4]], dtype=np.uint8))
    pq.write_table(pa.Table.from_pylist([
        {"target_chembl_id": "T1", "gene_symbol": "Target One", "ot_id": "ENSG1", "disease_id": "MONDO_0000004", "disease_name": "Merged member disease", "overall_score": .9, "therapeutic_areas": "cancer", "therapeutic_area_ids": "MONDO_0004992", "txgnn_score_column": 3, "txgnn_mapping_scope": "MEMBER_OF_MERGED_NODE", "datatype_scores_json": '{"genetic_association": 0.8}', "source_ensembl_id": "ENSG1", "source": "OpenTargets_26.06", "treatment_direction_established": False},
        {"target_chembl_id": "T1", "gene_symbol": "Target One", "ot_id": "ENSG1", "disease_id": "EFO_123", "disease_name": "Unmapped source trait", "overall_score": .2, "therapeutic_areas": "trait", "therapeutic_area_ids": "EFO_1", "txgnn_score_column": -1, "txgnn_mapping_scope": "UNMAPPED", "datatype_scores_json": '{}', "source_ensembl_id": "ENSG1", "source": "OpenTargets_26.06", "treatment_direction_established": False},
    ]), path / OT)
    ec = []
    for project, other, relation, predicate, evidence in [("drug:D1", "target:T1", "DRUG_TARGET", "biolink:affects", "TEXT_MINING_OR_PREDICTION"), ("drug:D1", "MONDO:0000001", "DRUG_DISEASE", "biolink:in_clinical_trials_for", "NON_TEXT_SOURCE_ASSERTION"), ("target:T1", "MONDO:0000002", "TARGET_DISEASE", "biolink:associated_with", "NON_TEXT_SOURCE_ASSERTION")]:
        ec.append({"project_key": project, "other_id": other, "relation_class": relation, "ec_subject": "EC:drug", "ec_object": "EC:target", "predicate": predicate, "qualified_predicate": None, "object_direction_qualifier": "decreased", "primary_sources": "infores:source", "knowledge_level": "prediction", "agent_type": "text_mining_agent", "evidence_class": evidence, "mapping_ambiguous": True, "increment_class": "NEW_ENDPOINT_PAIR_ON_COMPARABLE_NODES", "publications": "PMID:123"})
    pq.write_table(pa.Table.from_pylist(ec), path / EC)
    pq.write_table(pa.Table.from_pylist([{"ligand_inchikey": drug, "target_chembl_id": target, "txgnn_status": "RAW_SCORED" if drug != "UNMAPPED" else "UNMAPPED", "ot_status": "COMPLETE" if target == "T1" else "MISSING_ENSEMBL", "top50_graph_novel_diseases_with_ot_evidence": 1 if drug == "D1" and target == "T1" else 0, "eckg_drug_target_relation_present": drug == "D1" and target == "T1", "historical_dti_scope": "UNCHANGED", "identity_review_status": "CONFLICT_HOLD" if drug == "HOLD" else "REVIEWED"} for drug in drugs for target in targets]), path / PAIR)
    pq.write_table(pa.Table.from_pylist([{"id": "MONDO:0000001", "name": "Named neural disease"}, {"id": "MONDO:0000002", "name": "Named neoplasm"}]), path / NODES)
    json_file(path / "VALIDATION.json", {"all_pass": True, "checks": {"fixture_verified": True}, "no_training": True})
    json_file(path / "TXGNN_RUN_COMPLETE.json", {"drugs": 3, "diseases": 4, "scores": 12, "relation": "indication", "training": False})
    json_file(path / "FINAL_SUMMARY.json", {"project_drugs": 4, "project_targets": 2, "raw_scored_drugs": 3, "scored_diseases": 4, "score_count": 12, "interpretation_eligible_drugs": 2, "total_masked_query_cells": 7, "identity_conflict_hold_drugs": 1, "ot_evidence_rows": 2, "all_pair_bookkeeping_rows": 8, "direction_disease_node_counts": {"speciality_neoplasm": 2, "speciality_neurological": 1}})
    refresh_manifest(path)
    return tmp_path, path, drugs, targets


def store(snapshot):
    root, _, drugs, targets = snapshot
    return DiseaseEvidenceStore(root, drugs, targets)


def test_optional_missing_and_invalid_snapshot(tmp_path):
    missing = DiseaseEvidenceStore(tmp_path, {}, {})
    assert not missing.enabled and missing.summary["status"] == "snapshot_missing"
    assert missing.query("drug", "missing", "txgnn_diseases")["items"] == []
    (tmp_path / SNAPSHOT).mkdir(parents=True)
    with pytest.raises(ValueError, match="Missing required artifacts"):
        DiseaseEvidenceStore(tmp_path, {}, {})


def test_physical_drug_order_masked_rank_and_merged_node(snapshot):
    s = store(snapshot)
    assert isinstance(s.logits, np.memmap) and isinstance(s.mask, np.memmap)
    first = s.query("drug", "D1", "txgnn_diseases")["items"]
    assert [(r["id"], r["score"], r["rank"]) for r in first] == [("4_5", 40, 1), ("2", 20, 2)]
    assert first[0]["member_disease_ids"] == ["MONDO:0000004", "MONDO:0000005"]
    assert first[0]["mapping_scope"] == "MEMBER_OF_MERGED_NODE"
    assert first[0]["physical_drug_row"] == 1 and first[0]["raw_denominator"] == 4
    filtered = s.query("drug", "D2", "txgnn_diseases", search="Merged", direction="neoplasm")
    assert filtered["total"] == 1 and filtered["all_total"] == 3
    assert filtered["items"][0]["rank"] == 2 and filtered["items"][0]["raw_rank"] == 3
    assert filtered["items"][0]["denominator"] == 3
    page = s.query("drug", "D2", "txgnn_diseases", page=2, page_size=1)
    assert page["items"][0]["rank"] == 2


def test_raw_trace_hold_and_no_legacy_backfill(snapshot):
    s = store(snapshot)
    raw = s.query("drug", "D1", "txgnn_diseases", scope="all")
    assert raw["total"] == 4 and raw["denominator"] == 4
    assert [r["rank"] for r in raw["items"]] == [1, 2, 3, 4]
    excluded = next(r for r in raw["items"] if r["id"] == "3")
    assert excluded["exclusion_mask"] == 2 and excluded["eligible_rank"] is None
    assert excluded["exclusion_reasons"] and not excluded["interpretation_eligible"]
    held = s.metadata("drug", "HOLD")["txgnn"]
    assert held["identity_hold"] and not held["legacy_identity_hold_field"]
    assert held["raw_available"] and not held["interpretation_eligible"]
    assert s.total("drug", "HOLD", "txgnn_diseases") == 0
    assert s.query("drug", "HOLD", "txgnn_diseases")["items"] == []
    assert all(r["exclusion_mask"] == 4 for r in s.query("drug", "HOLD", "txgnn_diseases", scope="all")["items"])
    assert s.preview("drug", "UNMAPPED", "txgnn_diseases") == []
    assert s.metadata("drug", "UNMAPPED")["txgnn"]["status"] == "unavailable"


def test_ot_ec_and_pair_preserve_source_semantics(snapshot):
    s = store(snapshot)
    ot = s.query("target", "T1", "target_diseases")
    assert ot["total"] == 2 and ot["items"][0]["score"] == .9
    assert ot["items"][0]["datatype_scores"] == {"genetic_association": .8}
    assert ot["items"][0]["txgnn_mapping_scope"] == "MEMBER_OF_MERGED_NODE"
    assert "/ENSG1/MONDO_0000004" in ot["items"][0]["url"]
    filtered = s.query("target", "T1", "target_diseases", search="Unmapped source")
    assert filtered["items"][0]["rank"] == 2 and filtered["all_total"] == 2
    assert s.metadata("target", "T2")["disease_coverage"]["status"] == "MISSING_ENSEMBL"
    # A drug→target audit row is visible from the target, without becoming binding truth.
    ec = s.query("target", "T1", "eckg")
    assert ec["total"] == 2
    source = next(r for r in ec["items"] if r["relation_class"] == "DRUG_TARGET")
    assert source["name"] == "Drug One" and source["predicate"] == "biolink:affects"
    assert source["mapping_ambiguous"] and source["evidence_class"] == "TEXT_MINING_OR_PREDICTION"
    assert source["publications"] == "PMID:123" and "已知结合" in source["interpretation"]
    pair = s.pair("D1", "T1")
    assert pair["coverage"]["top50_graph_novel_diseases_with_ot_evidence"] == 1
    assert pair["eckg_total"] == 1 and len(pair["eckg_records"]) == 1
    assert s.counts["eckg_relations"] == 3 and s.counts["disease_pair_coverage"] == 8
    assert len(s.summary["directions"]) == 23 and s.summary["unclassified_count"] == 1


def test_manifest_integrity_and_changed_source_fingerprint(snapshot):
    s = store(snapshot)
    root, path, drugs, targets = snapshot
    old_cache = s.cache_path
    values = np.load(path / LOGITS).copy(); values[1, 3] = 41
    np.save(path / LOGITS, values)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        DiseaseEvidenceStore(root, drugs, targets)
    refresh_manifest(path)
    revised = store(snapshot)
    assert revised.cache_path != old_cache and old_cache.exists()
    assert revised.preview("drug", "D1", "txgnn_diseases", 1)[0]["score"] == 41


def test_inconsistent_hold_and_invalid_inputs_fail_closed(snapshot):
    root, path, drugs, targets = snapshot
    masks = np.load(path / MASK).copy(); masks[2, 0] = 0
    np.save(path / MASK, masks); refresh_manifest(path)
    with pytest.raises(ValueError, match="Identity-held row"):
        DiseaseEvidenceStore(root, drugs, targets)
    masks[2, 0] = 4; np.save(path / MASK, masks); refresh_manifest(path)
    s = store(snapshot)
    with pytest.raises(ValueError, match="Unknown disease direction"):
        s.query("drug", "D1", "txgnn_diseases", direction="invented")
    with pytest.raises(ValueError, match="scope"):
        s.query("drug", "D1", "txgnn_diseases", scope="legacy")
    with pytest.raises(ValueError, match="Unsupported disease section"):
        s.query("target", "T1", "txgnn_diseases")
    with pytest.raises(KeyError):
        s.metadata("drug", "not-in-catalog")
