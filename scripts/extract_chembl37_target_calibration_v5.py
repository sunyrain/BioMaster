#!/usr/bin/env python3
"""Build target-level positive/negative calibration coverage from ChEMBL 37."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/calibrated_pipeline_v5.yaml"
TARGETS = ROOT / "configs/project_targets_v4.csv"
ANCHORS = ROOT / "outputs/chembl_moa_enhanced_information_package_v1/chembl_moa_anchor_gene_table_v2.csv"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/chembl37_target_calibration_v5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def split_ids(value: Any) -> list[str]:
    return [token for token in re.split(r"[;,|\s]+", clean(value)) if token]


def locate_db(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted((ROOT / "downloads/chembl_37").glob("**/*.db"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one ChEMBL 37 .db, found {candidates}")
    return candidates[0]


def load_project_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGETS, dtype=str).fillna("")
    anchors = pd.read_csv(ANCHORS, low_memory=False).fillna("")
    anchor_columns = [
        "gene",
        "canonical_uniprot",
        "chembl_moa_target_chembl_ids",
        "project_assay_family",
        "target_class_l1",
    ]
    anchors = anchors[anchor_columns].drop_duplicates("gene")
    out = targets.merge(anchors, left_on="primary_gene", right_on="gene", how="left", validate="many_to_one")
    # ChEMBL target components are normally keyed to canonical UniProt accessions,
    # whereas the project sequence may be an isoform.  Labels are mapped at the
    # target-gene level and evaluated on the exact project sequence separately.
    canonical = out["canonical_uniprot"].map(clean)
    representative = out["representative_protein_id"].map(clean)
    out["query_accession"] = canonical.where(canonical.ne(""), representative)
    out["project_sequence_accession"] = representative
    out["anchor_accession_match"] = representative.eq(canonical)
    return out


def validate_schema(connection: sqlite3.Connection) -> None:
    required = {
        "activities": {
            "activity_id",
            "assay_id",
            "doc_id",
            "molregno",
            "standard_relation",
            "standard_type",
            "standard_units",
            "standard_value",
            "pchembl_value",
            "activity_comment",
            "standard_text_value",
            "text_value",
            "data_validity_comment",
            "potential_duplicate",
        },
        "assays": {"assay_id", "tid", "assay_type", "confidence_score", "relationship_type", "doc_id"},
        "target_dictionary": {"tid", "target_type", "organism", "tax_id", "chembl_id"},
        "target_components": {"tid", "component_id", "homologue"},
        "component_sequences": {"component_id", "accession", "sequence_md5sum"},
        "molecule_hierarchy": {"molregno", "parent_molregno", "active_molregno"},
        "molecule_dictionary": {"molregno", "chembl_id", "pref_name"},
        "compound_structures": {"molregno", "canonical_smiles", "standard_inchi_key"},
        "docs": {"doc_id", "chembl_id", "year", "pubmed_id", "doi"},
    }
    existing = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    for table, columns in required.items():
        if table not in existing:
            raise RuntimeError(f"Missing table {table}")
        present = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        missing = columns - present
        if missing:
            raise RuntimeError(f"Missing columns in {table}: {sorted(missing)}")


def target_map(connection: sqlite3.Connection, project: pd.DataFrame) -> pd.DataFrame:
    accessions = sorted(set(project["query_accession"]) - {""})
    placeholders = ",".join("?" for _ in accessions)
    query = f"""
        SELECT
            cs.accession AS query_accession,
            cs.sequence_md5sum AS chembl_sequence_md5,
            td.tid,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS chembl_target_name,
            td.target_type,
            td.organism,
            td.tax_id,
            tc.homologue
        FROM component_sequences cs
        JOIN target_components tc ON tc.component_id = cs.component_id
        JOIN target_dictionary td ON td.tid = tc.tid
        WHERE cs.accession IN ({placeholders})
          AND td.target_type = 'SINGLE PROTEIN'
          AND td.tax_id = 9606
    """
    mapped = pd.read_sql_query(query, connection, params=accessions)
    mapped = mapped.sort_values(["query_accession", "homologue", "target_chembl_id"], kind="mergesort")
    mapped["is_direct_component"] = pd.to_numeric(mapped["homologue"], errors="coerce").fillna(1).eq(0)
    merged = project.merge(mapped, on="query_accession", how="left", validate="one_to_many")
    merged["anchor_target_id_matches"] = [
        clean(target_id) in set(split_ids(anchor_ids))
        for target_id, anchor_ids in zip(
            merged["target_chembl_id"].fillna(""), merged["chembl_moa_target_chembl_ids"].fillna("")
        )
    ]
    return merged


def create_target_temp_table(connection: sqlite3.Connection, mapped: pd.DataFrame) -> None:
    direct = mapped[
        mapped["target_chembl_id"].notna()
        & mapped["is_direct_component"].fillna(False)
    ][["sequence_key", "primary_gene", "target_assay_family", "query_accession", "tid", "target_chembl_id"]].drop_duplicates()
    connection.execute("DROP TABLE IF EXISTS temp.project_targets_v5")
    connection.execute(
        """
        CREATE TEMP TABLE project_targets_v5 (
            sequence_key TEXT,
            primary_gene TEXT,
            target_assay_family TEXT,
            query_accession TEXT,
            tid INTEGER,
            target_chembl_id TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO project_targets_v5 VALUES (?, ?, ?, ?, ?, ?)",
        [tuple(row) for row in direct.itertuples(index=False, name=None)],
    )
    connection.execute("CREATE INDEX temp.idx_project_targets_v5_tid ON project_targets_v5(tid)")


def pair_query(config: dict[str, Any]) -> str:
    calibration = config["chembl_calibration"]
    standard_types = ",".join(f"'{item}'" for item in calibration["numeric_standard_types"])
    inactive_terms = " OR ".join(
        f"LOWER(COALESCE(a.activity_comment, '') || ' ' || COALESCE(a.standard_text_value, '') || ' ' || COALESCE(a.text_value, '')) LIKE '%{term.lower()}%'"
        for term in calibration["explicit_inactive_patterns"]
    )
    return f"""
        WITH strict_activity AS (
            SELECT
                pt.sequence_key,
                pt.primary_gene,
                pt.target_assay_family,
                pt.query_accession,
                pt.target_chembl_id,
                a.activity_id,
                a.assay_id,
                COALESCE(a.doc_id, ass.doc_id) AS doc_id,
                a.standard_type,
                a.standard_relation,
                a.standard_value,
                a.standard_units,
                a.pchembl_value,
                a.activity_comment,
                a.standard_text_value,
                a.text_value,
                ass.relationship_type,
                ass.confidence_score,
                d.year AS document_year,
                COALESCE(mh.parent_molregno, a.molregno) AS parent_molregno,
                CASE WHEN ({inactive_terms}) THEN 1 ELSE 0 END AS explicit_inactive
            FROM project_targets_v5 pt
            JOIN assays ass ON ass.tid = pt.tid
            JOIN activities a ON a.assay_id = ass.assay_id
            LEFT JOIN molecule_hierarchy mh ON mh.molregno = a.molregno
            LEFT JOIN docs d ON d.doc_id = COALESCE(a.doc_id, ass.doc_id)
            WHERE ass.assay_type = 'B'
              AND ass.confidence_score >= {int(calibration['minimum_assay_confidence'])}
              AND COALESCE(a.potential_duplicate, 0) = 0
              AND COALESCE(a.data_validity_comment, '') IN ('', 'Manually validated')
              AND (
                    (a.pchembl_value IS NOT NULL
                     AND a.standard_type IN ({standard_types})
                     AND a.standard_relation = '=')
                    OR ({inactive_terms})
                  )
        ),
        pair_aggregate AS (
            SELECT
                sequence_key,
                primary_gene,
                target_assay_family,
                query_accession,
                target_chembl_id,
                parent_molregno,
                COUNT(*) AS activity_rows,
                COUNT(DISTINCT assay_id) AS assay_count,
                COUNT(DISTINCT doc_id) AS document_count,
                MIN(pchembl_value) AS min_pchembl,
                MAX(pchembl_value) AS max_pchembl,
                AVG(pchembl_value) AS mean_pchembl,
                SUM(CASE WHEN pchembl_value IS NOT NULL THEN 1 ELSE 0 END) AS numeric_rows,
                MAX(explicit_inactive) AS any_explicit_inactive,
                MIN(document_year) AS min_document_year,
                MAX(document_year) AS max_document_year,
                GROUP_CONCAT(DISTINCT standard_type) AS standard_types,
                GROUP_CONCAT(DISTINCT relationship_type) AS relationship_types,
                GROUP_CONCAT(DISTINCT assay_id) AS assay_ids,
                GROUP_CONCAT(DISTINCT doc_id) AS doc_ids
            FROM strict_activity
            GROUP BY sequence_key, primary_gene, target_assay_family, query_accession, target_chembl_id, parent_molregno
        )
        SELECT
            pa.*,
            md.chembl_id AS parent_molecule_chembl_id,
            md.pref_name AS parent_molecule_name,
            cs.canonical_smiles AS parent_canonical_smiles,
            cs.standard_inchi_key AS parent_standard_inchi_key
        FROM pair_aggregate pa
        LEFT JOIN molecule_dictionary md ON md.molregno = pa.parent_molregno
        LEFT JOIN compound_structures cs ON cs.molregno = pa.parent_molregno
    """


def classify_pairs(pairs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = pairs.copy()
    positive = float(config["chembl_calibration"]["positive_pchembl_min"])
    negative = float(config["chembl_calibration"]["negative_pchembl_max"])
    minimum = pd.to_numeric(out["min_pchembl"], errors="coerce")
    maximum = pd.to_numeric(out["max_pchembl"], errors="coerce")
    mean = pd.to_numeric(out["mean_pchembl"], errors="coerce")
    explicit = pd.to_numeric(out["any_explicit_inactive"], errors="coerce").fillna(0).gt(0)
    out["numeric_positive_negative_conflict"] = minimum.le(negative) & maximum.ge(positive)
    out["explicit_inactive_positive_conflict"] = explicit & maximum.ge(positive)
    conflict = out["numeric_positive_negative_conflict"] | out["explicit_inactive_positive_conflict"]
    out["calibration_label"] = "grey_or_unresolved"
    out.loc[mean.ge(positive) & ~conflict, "calibration_label"] = "positive"
    out.loc[(mean.le(negative) | explicit) & ~conflict, "calibration_label"] = "negative_or_inactive"
    out.loc[conflict, "calibration_label"] = "conflicting_exclude"
    return out


def coverage_table(project: pd.DataFrame, mapped: pd.DataFrame, pairs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    def unique_ids(values: pd.Series) -> int:
        identifiers: set[str] = set()
        for value in values:
            identifiers.update(token for token in clean(value).split(",") if token)
        return len(identifiers)

    base = project[
        ["sequence_key", "primary_gene", "target_assay_family", "query_accession", "structure_bin"]
    ].drop_duplicates("sequence_key")
    map_summary = mapped.groupby("sequence_key", dropna=False).agg(
        chembl_single_protein_target_count=("target_chembl_id", lambda values: values.dropna().nunique()),
        direct_component_target_count=("is_direct_component", lambda values: int(values.fillna(False).sum())),
        anchor_target_id_any_match=("anchor_target_id_matches", "max"),
    ).reset_index()
    pair_summary = pairs.groupby("sequence_key", dropna=False).agg(
        strict_compound_pairs=("parent_molregno", "nunique"),
        positive_compounds=("calibration_label", lambda values: int((values == "positive").sum())),
        negative_compounds=("calibration_label", lambda values: int((values == "negative_or_inactive").sum())),
        grey_compounds=("calibration_label", lambda values: int((values == "grey_or_unresolved").sum())),
        conflicting_compounds=("calibration_label", lambda values: int((values == "conflicting_exclude").sum())),
        strict_activity_rows=("activity_rows", "sum"),
        strict_assay_count=("assay_ids", unique_ids),
        strict_document_count=("doc_ids", unique_ids),
    ).reset_index()
    out = base.merge(map_summary, on="sequence_key", how="left").merge(pair_summary, on="sequence_key", how="left")
    count_columns = [
        "chembl_single_protein_target_count",
        "direct_component_target_count",
        "strict_compound_pairs",
        "positive_compounds",
        "negative_compounds",
        "grey_compounds",
        "conflicting_compounds",
        "strict_activity_rows",
        "strict_assay_count",
        "strict_document_count",
    ]
    for column in count_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    out["anchor_target_id_any_match"] = out["anchor_target_id_any_match"].fillna(False).astype(bool)

    tiers = config["chembl_calibration"]["target_tiers"]
    out["calibration_tier_v5"] = "T4_sparse"
    t3 = tiers["T3_positive_only"]
    mask = out["positive_compounds"].ge(int(t3["min_positive_compounds"]))
    out.loc[mask, "calibration_tier_v5"] = "T3_positive_only"
    t2 = tiers["T2_target_calibrated_limited"]
    mask = (
        out["positive_compounds"].ge(int(t2["min_positive_compounds"]))
        & out["negative_compounds"].ge(int(t2["min_negative_compounds"]))
        & out["strict_document_count"].ge(int(t2["min_documents"]))
    )
    out.loc[mask, "calibration_tier_v5"] = "T2_target_calibrated_limited"
    t1 = tiers["T1_target_calibrated"]
    mask = (
        out["positive_compounds"].ge(int(t1["min_positive_compounds"]))
        & out["negative_compounds"].ge(int(t1["min_negative_compounds"]))
        & out["strict_document_count"].ge(int(t1["min_documents"]))
    )
    out.loc[mask, "calibration_tier_v5"] = "T1_target_calibrated"
    return out.sort_values(
        ["calibration_tier_v5", "positive_compounds", "negative_compounds", "primary_gene"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    db = locate_db(args.db)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    project = load_project_targets()
    if len(project) != 463:
        raise RuntimeError(f"Expected 463 project targets, got {len(project)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA temp_store = MEMORY")
        validate_schema(connection)
        mapped = target_map(connection, project)
        create_target_temp_table(connection, mapped)
        pairs = pd.read_sql_query(pair_query(config), connection)
    finally:
        connection.close()

    pairs = classify_pairs(pairs, config)
    coverage = coverage_table(project, mapped, pairs, config)
    mapped.to_csv(args.output_dir / "PROJECT463_CHEMBL37_TARGET_MAPPING_V5.csv", index=False)
    pairs.to_csv(
        args.output_dir / "PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz",
        index=False,
        compression="gzip",
    )
    coverage.to_csv(args.output_dir / "PROJECT463_CALIBRATION_COVERAGE_V5.csv", index=False)

    tier_counts = coverage["calibration_tier_v5"].value_counts().to_dict()
    summary = {
        "status": "passed",
        "created_utc": now(),
        "chembl_db": str(db),
        "chembl_db_sha256": sha256(db),
        "project_targets": int(len(project)),
        "mapped_targets": int(coverage["chembl_single_protein_target_count"].gt(0).sum()),
        "strict_pair_rows": int(len(pairs)),
        "positive_pair_rows": int(pairs["calibration_label"].eq("positive").sum()),
        "negative_pair_rows": int(pairs["calibration_label"].eq("negative_or_inactive").sum()),
        "grey_pair_rows": int(pairs["calibration_label"].eq("grey_or_unresolved").sum()),
        "conflicting_pair_rows": int(pairs["calibration_label"].eq("conflicting_exclude").sum()),
        "target_tiers": {str(key): int(value) for key, value in tier_counts.items()},
        "inputs": {
            str(CONFIG.relative_to(ROOT)): sha256(CONFIG),
            str(TARGETS.relative_to(ROOT)): sha256(TARGETS),
            str(ANCHORS.relative_to(ROOT)): sha256(ANCHORS),
        },
    }
    (args.output_dir / "CHEMBL37_CALIBRATION_SUMMARY_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
