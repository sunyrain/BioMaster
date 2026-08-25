#!/usr/bin/env python3
"""Build an activity-audited, old-drug-centric view of the V5 720 x 384 matrix."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_v5_l2_pair_md_audit_v1 import clean, related_molecules


ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v5"
    / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V5.csv.gz"
)
SPECIES = (
    ROOT
    / "outputs/strict_affinity_main_queue_2005_2026_v2"
    / "STRICT_STANDARD_MODEL_LIGAND_SPECIES_MAP.csv"
)
CHEMBL = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OUT = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1"
    / "drug_centric_cross_target_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def chembl_annotation(pairs: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{CHEMBL}?mode=ro", uri=True)
    try:
        related = related_molecules(connection, species)
        connection.execute(
            "CREATE TEMP TABLE drug_related (ligand_inchikey TEXT, molregno INTEGER, PRIMARY KEY(ligand_inchikey, molregno))"
        )
        connection.executemany(
            "INSERT INTO drug_related VALUES (?, ?)",
            [
                (inchikey, molregno)
                for inchikey, molregnos in related.items()
                for molregno in sorted(molregnos)
            ],
        )
        targets = sorted(set(pairs["target_chembl_id"]))
        connection.execute("CREATE TEMP TABLE active_targets (target_chembl_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO active_targets VALUES (?)", [(value,) for value in targets])
        activity = pd.read_sql_query(
            """
            SELECT dr.ligand_inchikey, td.chembl_id AS target_chembl_id,
                   COUNT(DISTINCT a.activity_id) AS any_activity_rows,
                   COUNT(DISTINCT CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN a.activity_id END) AS strict_numeric_rows,
                   COUNT(DISTINCT CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN a.assay_id END) AS strict_assay_count,
                   COUNT(DISTINCT CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN COALESCE(a.doc_id, ass.doc_id) END) AS strict_document_count,
                   MIN(CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN a.pchembl_value END) AS strict_pchembl_min,
                   MAX(CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN a.pchembl_value END) AS strict_pchembl_max,
                   AVG(CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND a.standard_type IN ('Ki', 'Kd', 'IC50')
                       AND a.standard_relation = '=' AND a.pchembl_value IS NOT NULL
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       THEN a.pchembl_value END) AS strict_pchembl_mean,
                   MAX(CASE WHEN
                       ass.assay_type = 'B' AND ass.confidence_score >= 9
                       AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
                       AND COALESCE(a.potential_duplicate, 0) = 0
                       AND LOWER(COALESCE(a.activity_comment, '') || ' ' ||
                                 COALESCE(a.standard_text_value, '') || ' ' ||
                                 COALESCE(a.text_value, '')) GLOB '*inactive*'
                       THEN 1 ELSE 0 END) AS strict_explicit_inactive
            FROM drug_related dr
            JOIN activities a ON a.molregno = dr.molregno
            JOIN assays ass ON ass.assay_id = a.assay_id
            JOIN target_dictionary td ON td.tid = ass.tid
            JOIN active_targets t ON t.target_chembl_id = td.chembl_id
            GROUP BY dr.ligand_inchikey, td.chembl_id
            """,
            connection,
        )
        mechanism = pd.read_sql_query(
            """
            SELECT dr.ligand_inchikey, td.chembl_id AS target_chembl_id,
                   COUNT(DISTINCT dm.mec_id) AS mechanism_rows,
                   GROUP_CONCAT(DISTINCT dm.action_type) AS mechanism_action_types
            FROM drug_related dr
            JOIN drug_mechanism dm ON dm.molregno = dr.molregno
            JOIN target_dictionary td ON td.tid = dm.tid
            JOIN active_targets t ON t.target_chembl_id = td.chembl_id
            GROUP BY dr.ligand_inchikey, td.chembl_id
            """,
            connection,
        )
    finally:
        connection.close()
    annotation = activity.merge(
        mechanism, on=["ligand_inchikey", "target_chembl_id"], how="outer",
        validate="one_to_one",
    )
    for column in [
        "any_activity_rows", "strict_numeric_rows", "strict_assay_count",
        "strict_document_count", "strict_explicit_inactive", "mechanism_rows",
    ]:
        annotation[column] = pd.to_numeric(annotation[column], errors="coerce").fillna(0).astype(int)
    annotation["mechanism_action_types"] = annotation["mechanism_action_types"].fillna("")
    minimum = pd.to_numeric(annotation["strict_pchembl_min"], errors="coerce")
    maximum = pd.to_numeric(annotation["strict_pchembl_max"], errors="coerce")
    mean = pd.to_numeric(annotation["strict_pchembl_mean"], errors="coerce")
    explicit = annotation["strict_explicit_inactive"].gt(0)
    mechanism_present = annotation["mechanism_rows"].gt(0)
    any_activity = annotation["any_activity_rows"].gt(0)
    conflict = (minimum.le(5.0) & maximum.ge(6.0)) | (explicit & maximum.ge(6.0))
    annotation["chembl37_pair_record_class"] = "N0_NO_CHEMBL37_ACTIVITY_OR_MOA"
    annotation.loc[any_activity, "chembl37_pair_record_class"] = "R1_NONSTRICT_ACTIVITY_ONLY"
    annotation.loc[mean.notna() | explicit, "chembl37_pair_record_class"] = (
        "K3_STRICT_GREY_NEGATIVE_OR_INACTIVE"
    )
    annotation.loc[mean.ge(6.0), "chembl37_pair_record_class"] = "K1_STRICT_BINDING_POSITIVE"
    annotation.loc[conflict, "chembl37_pair_record_class"] = "K2_STRICT_BINDING_CONFLICTING"
    annotation.loc[mechanism_present, "chembl37_pair_record_class"] = "K0_CHEMBL37_MOA_RELATIONSHIP"
    return annotation


def coalesce_support(data: pd.DataFrame) -> pd.Series:
    boltz = data["main_boltz_boltz_evidence_tier"].fillna("").isin(
        {"BOLTZ_STRUCTURE_A", "BOLTZ_STRUCTURE_B", "BOLTZ_STRUCTURE_C"}
    ) | data["increment_boltz_boltz_evidence_tier"].fillna("").isin(
        {"BOLTZ_STRUCTURE_A", "BOLTZ_STRUCTURE_B", "BOLTZ_STRUCTURE_C"}
    )
    gnina = data["main_gnina_gnina_structure_evidence_tier"].fillna("").astype(str).str.contains(
        "GNINA", regex=False
    ) | data["increment_gnina_gnina_remote_support"].pipe(bool_series)
    return boltz | gnina


def main() -> None:
    usecols = [
        "ligand_inchikey", "ligand_smiles", "drug_names", "project_entity_ids",
        "target_chembl_id", "gene_symbol", "target_route_family",
        "unified_target_route", "target_gate_pass_for_discovery", "pairId",
        "conplex_rank_within_ligand_384", "conplex_percentile_within_ligand_384",
        "drugclip_rank_within_ligand_382", "drugclip_percentile_within_ligand_382",
        "dta_bidirectional_top10pct_concordant_384", "is_any_frozen_known_relationship",
        "main_boltz_boltz_evidence_tier", "main_gnina_gnina_structure_evidence_tier",
        "increment_boltz_boltz_evidence_tier", "increment_gnina_gnina_remote_support",
        "any_physical_pair_calculation_completed_v5", "final_pair_evidence_layer_v5",
        "final_pair_evidence_layer_zh_v5", "pair_next_action_v5",
    ]
    pairs = pd.read_csv(MATRIX, usecols=usecols, low_memory=False)
    species = pd.read_csv(SPECIES, low_memory=False)
    if len(pairs) != 720 * 384 or pairs["pairId"].duplicated().any():
        raise RuntimeError("V5 pair universe is not exactly 720 x 384")
    annotation = chembl_annotation(pairs, species)
    data = pairs.merge(
        annotation, on=["ligand_inchikey", "target_chembl_id"], how="left",
        validate="one_to_one",
    )
    integer_columns = [
        "any_activity_rows", "strict_numeric_rows", "strict_assay_count",
        "strict_document_count", "strict_explicit_inactive", "mechanism_rows",
    ]
    for column in integer_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    data["mechanism_action_types"] = data["mechanism_action_types"].fillna("")
    data["chembl37_pair_record_class"] = data["chembl37_pair_record_class"].fillna(
        "N0_NO_CHEMBL37_ACTIVITY_OR_MOA"
    )
    data["local_chembl37_unreported_pair"] = data["chembl37_pair_record_class"].eq(
        "N0_NO_CHEMBL37_ACTIVITY_OR_MOA"
    )
    data["target_gate_pass_for_discovery"] = bool_series(data["target_gate_pass_for_discovery"])
    data["any_physical_pair_calculation_completed_v5"] = bool_series(
        data["any_physical_pair_calculation_completed_v5"]
    )
    data["dta_bidirectional_top10pct_concordant_384"] = bool_series(
        data["dta_bidirectional_top10pct_concordant_384"]
    )
    stable = data["final_pair_evidence_layer_v5"].isin(
        {"L1_MULTI_METHOD_STATE_STABLE", "L2_BOLTZ_REPRODUCED", "L2_GNINA_RECEPTOR_STATE_STABLE"}
    )
    primary_support = coalesce_support(data)
    data["drug_centric_evidence_stage"] = "D4_DTA_ONLY_OR_UNSUPPORTED"
    data.loc[data["any_physical_pair_calculation_completed_v5"], "drug_centric_evidence_stage"] = (
        "D3_PHYSICAL_COMPUTED_NOT_SUPPORTED"
    )
    data.loc[primary_support, "drug_centric_evidence_stage"] = "D2_PRIMARY_PHYSICAL_SUPPORT"
    data.loc[stable, "drug_centric_evidence_stage"] = "D1_STATE_OR_SEED_STABLE"
    stage_priority = {
        "D1_STATE_OR_SEED_STABLE": 0,
        "D2_PRIMARY_PHYSICAL_SUPPORT": 1,
        "D3_PHYSICAL_COMPUTED_NOT_SUPPORTED": 2,
        "D4_DTA_ONLY_OR_UNSUPPORTED": 3,
    }
    data["drug_centric_stage_priority"] = data["drug_centric_evidence_stage"].map(stage_priority)
    data["dta_cross_target_consensus_score"] = (
        pd.to_numeric(data["conplex_percentile_within_ligand_384"], errors="coerce").fillna(0)
        + pd.to_numeric(data["drugclip_percentile_within_ligand_382"], errors="coerce").fillna(0)
    ) / 2.0
    eligible = data[
        data["target_gate_pass_for_discovery"] & data["local_chembl37_unreported_pair"]
    ].copy()
    eligible = eligible.sort_values(
        [
            "ligand_inchikey", "drug_centric_stage_priority",
            "dta_bidirectional_top10pct_concordant_384", "dta_cross_target_consensus_score",
            "conplex_rank_within_ligand_384", "target_chembl_id",
        ],
        ascending=[True, True, False, False, True, True],
        kind="mergesort",
    )
    eligible["local_unreported_rank_within_drug"] = eligible.groupby(
        "ligand_inchikey"
    ).cumcount() + 1
    top5 = eligible[eligible["local_unreported_rank_within_drug"].le(5)].copy()
    physical_candidates = eligible[
        eligible["drug_centric_evidence_stage"].isin(
            {"D1_STATE_OR_SEED_STABLE", "D2_PRIMARY_PHYSICAL_SUPPORT"}
        )
    ].copy()

    grouped = data.groupby(
        ["ligand_inchikey", "drug_names", "project_entity_ids"], dropna=False
    )
    summary = grouped.agg(
        active_target_count=("target_chembl_id", "size"),
        experimental_pocket_targets=(
            "target_route_family", lambda values: int((values == "EXPERIMENTAL_POCKET_MAINLINE").sum())
        ),
        recovered_no_experimental_pocket_targets=(
            "target_route_family", lambda values: int((values == "PREDICTED_POCKET_RECOVERY").sum())
        ),
        chembl37_moa_pairs=(
            "chembl37_pair_record_class", lambda values: int((values == "K0_CHEMBL37_MOA_RELATIONSHIP").sum())
        ),
        strict_positive_pairs=(
            "chembl37_pair_record_class", lambda values: int((values == "K1_STRICT_BINDING_POSITIVE").sum())
        ),
        conflicting_pairs=(
            "chembl37_pair_record_class", lambda values: int((values == "K2_STRICT_BINDING_CONFLICTING").sum())
        ),
        grey_negative_inactive_pairs=(
            "chembl37_pair_record_class", lambda values: int((values == "K3_STRICT_GREY_NEGATIVE_OR_INACTIVE").sum())
        ),
        nonstrict_activity_only_pairs=(
            "chembl37_pair_record_class", lambda values: int((values == "R1_NONSTRICT_ACTIVITY_ONLY").sum())
        ),
        local_unreported_pairs=("local_chembl37_unreported_pair", "sum"),
        physical_pairs_completed=("any_physical_pair_calculation_completed_v5", "sum"),
        stable_physical_pairs=(
            "drug_centric_evidence_stage", lambda values: int((values == "D1_STATE_OR_SEED_STABLE").sum())
        ),
    ).reset_index()
    physical_unreported_counts = physical_candidates.groupby("ligand_inchikey").size()
    stable_unreported_counts = physical_candidates[
        physical_candidates["drug_centric_evidence_stage"].eq("D1_STATE_OR_SEED_STABLE")
    ].groupby("ligand_inchikey").size()
    summary["local_unreported_physical_support_pairs"] = (
        summary["ligand_inchikey"].map(physical_unreported_counts).fillna(0).astype(int)
    )
    summary["local_unreported_stable_pairs"] = (
        summary["ligand_inchikey"].map(stable_unreported_counts).fillna(0).astype(int)
    )
    best = top5[top5["local_unreported_rank_within_drug"].eq(1)].set_index("ligand_inchikey")
    summary["top_local_unreported_target"] = summary["ligand_inchikey"].map(best["gene_symbol"])
    summary["top_local_unreported_target_chembl_id"] = summary["ligand_inchikey"].map(
        best["target_chembl_id"]
    )
    summary["top_local_unreported_evidence_stage"] = summary["ligand_inchikey"].map(
        best["drug_centric_evidence_stage"]
    )
    summary = summary.sort_values(
        ["local_unreported_stable_pairs", "local_unreported_physical_support_pairs", "drug_names"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    summary["drug_centric_program_rank"] = np.arange(1, len(summary) + 1)

    OUT.mkdir(parents=True, exist_ok=True)
    pair_audit_path = OUT / "PAIR_CHEMBL37_ACTIVITY_AUDIT_720_X_384_V1.csv.gz"
    drug_summary_path = OUT / "DRUG_CENTRIC_720_CROSS_TARGET_SUMMARY_V1.csv"
    top5_path = OUT / "DRUG_CENTRIC_TOP5_LOCAL_UNREPORTED_PER_DRUG_V1.csv"
    physical_path = OUT / "DRUG_CENTRIC_LOCAL_UNREPORTED_PHYSICAL_CANDIDATES_V1.csv"
    pair_columns = [
        "pairId", "ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol",
        "target_route_family", "target_gate_pass_for_discovery",
        "chembl37_pair_record_class", "any_activity_rows", "strict_numeric_rows",
        "strict_pchembl_min", "strict_pchembl_max", "strict_pchembl_mean",
        "strict_explicit_inactive", "mechanism_rows", "mechanism_action_types",
        "local_chembl37_unreported_pair", "drug_centric_evidence_stage",
        "dta_cross_target_consensus_score", "dta_bidirectional_top10pct_concordant_384",
        "final_pair_evidence_layer_v5",
    ]
    data[pair_columns].to_csv(pair_audit_path, index=False, compression="gzip")
    summary.to_csv(drug_summary_path, index=False)
    top5.to_csv(top5_path, index=False)
    physical_candidates.to_csv(physical_path, index=False)
    checks = {
        "full_pair_universe": len(data) == 720 * 384,
        "all_720_drugs_summarized": len(summary) == 720,
        "exact_384_targets_per_drug": summary["active_target_count"].eq(384).all(),
        "known_unreported_partition_complete": data["chembl37_pair_record_class"].notna().all(),
        "top5_cap_respected": top5.groupby("ligand_inchikey").size().le(5).all(),
        "unreported_candidates_have_no_local_record": physical_candidates[
            "local_chembl37_unreported_pair"
        ].all(),
        "hard_target_gate_respected": physical_candidates["target_gate_pass_for_discovery"].all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pair_rows": int(len(data)),
        "drugs": int(data["ligand_inchikey"].nunique()),
        "targets": int(data["target_chembl_id"].nunique()),
        "chembl37_pair_record_class_counts": {
            str(key): int(value) for key, value in data["chembl37_pair_record_class"].value_counts().items()
        },
        "local_unreported_gate_pass_pairs": int(len(eligible)),
        "local_unreported_physical_candidate_pairs": int(len(physical_candidates)),
        "local_unreported_stable_pairs": int(
            physical_candidates["drug_centric_evidence_stage"].eq("D1_STATE_OR_SEED_STABLE").sum()
        ),
        "scope_boundary": (
            "Local-unreported means no activity or mechanism row for a related approved active "
            "species and the exact ChEMBL 37 target. It is not a claim of literature novelty, "
            "binding, efficacy, or mutant-specific activity. Boltz scores are never compared "
            "across targets."
        ),
        "inputs": {
            str(MATRIX.relative_to(ROOT)): sha256(MATRIX),
            str(SPECIES.relative_to(ROOT)): sha256(SPECIES),
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [pair_audit_path, drug_summary_path, top5_path, physical_path]
        },
    }
    summary_path = OUT / "DRUG_CENTRIC_CROSS_TARGET_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
