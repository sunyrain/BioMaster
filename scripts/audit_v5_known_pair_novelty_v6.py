#!/usr/bin/env python3
"""Audit V5 pair novelty against ChEMBL 37 mechanisms and exact parent-compound activity."""

from __future__ import annotations

import glob
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi


ROOT = Path(__file__).resolve().parents[1]
V5 = (
    ROOT
    / "outputs/current_production_package_v2/calibrated_portfolio_v5"
    / "FINAL1000_EVIDENCE_STRATIFIED_V5.csv"
)
MECHANISMS = ROOT / "outputs/target_catalog_quality_audit_v1/chembl37_mechanisms_with_human_target_map.csv"
ACTIVITY = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_v5"
    / "PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
)
OUT = ROOT / "outputs/current_production_package_v2/evidence_reuse_action_v6/known_pair_novelty_audit"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_db() -> Path:
    paths = [Path(path) for path in glob.glob(str(ROOT / "downloads/chembl_37/**/*.db"), recursive=True)]
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one ChEMBL 37 database, got {paths}")
    return paths[0]


def molecule_key(smiles: Any) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    return inchi.MolToInchiKey(molecule) if molecule is not None else ""


def plain_chembl_id(value: Any) -> str:
    text = str(value or "").split("__", 1)[0]
    return text if text.startswith("CHEMBL") and text[6:].isdigit() else ""


def phase4_structures(db: Path) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        data = pd.read_sql_query(
            """
            SELECT md.chembl_id, md.pref_name, md.max_phase, cs.standard_inchi_key
            FROM compound_structures cs
            JOIN molecule_dictionary md ON md.molregno = cs.molregno
            WHERE md.max_phase = 4 AND cs.standard_inchi_key IS NOT NULL
            """,
            connection,
        )
    finally:
        connection.close()
    data["connectivity_block"] = data["standard_inchi_key"].str.split("-").str[0]
    return data


def identity_map(portfolio: pd.DataFrame, phase4: pd.DataFrame) -> pd.DataFrame:
    drugs = portfolio[
        ["drug_chembl_id", "base_chembl_id", "drug_name", "model_ligand_smiles"]
    ].drop_duplicates("drug_chembl_id").copy()
    drugs["project_standard_inchi_key_v6"] = drugs["model_ligand_smiles"].map(molecule_key)
    drugs["project_connectivity_block_v6"] = drugs["project_standard_inchi_key_v6"].str.split("-").str[0]
    exact = phase4.groupby("standard_inchi_key")["chembl_id"].agg(lambda values: sorted(set(values))).to_dict()
    block = phase4.groupby("connectivity_block")["chembl_id"].agg(lambda values: sorted(set(values))).to_dict()
    pref = phase4.set_index("chembl_id")["pref_name"].fillna("").to_dict()
    rows: list[dict[str, Any]] = []
    for row in drugs.itertuples(index=False):
        project_id = plain_chembl_id(row.drug_chembl_id)
        base_id = plain_chembl_id(row.base_chembl_id)
        exact_ids = exact.get(row.project_standard_inchi_key_v6, [])
        block_ids = block.get(row.project_connectivity_block_v6, [])
        candidate_ids: list[str]
        status: str
        if exact_ids:
            candidate_ids = exact_ids
            status = "exact_stereo_phase4_inchikey"
        elif base_id and base_id in block_ids:
            candidate_ids = [base_id]
            status = "base_chembl_id_in_phase4_connectivity"
        elif project_id and project_id in block_ids:
            candidate_ids = [project_id]
            status = "project_chembl_id_in_phase4_connectivity"
        elif len(block_ids) == 1:
            candidate_ids = block_ids
            status = "unique_phase4_connectivity"
        elif len(block_ids) > 1:
            candidate_ids = block_ids
            status = "ambiguous_multiple_phase4_stereoisomers"
        else:
            candidate_ids = sorted({item for item in [base_id, project_id] if item})
            status = "no_phase4_structure_match_use_project_ids" if candidate_ids else "unmapped"
        rows.append(
            {
                "drug_chembl_id": row.drug_chembl_id,
                "drug_name": row.drug_name,
                "project_standard_inchi_key_v6": row.project_standard_inchi_key_v6,
                "project_connectivity_block_v6": row.project_connectivity_block_v6,
                "official_active_moiety_mapping_status_v6": status,
                "official_active_moiety_chembl_ids_v6": ";".join(candidate_ids),
                "official_active_moiety_names_v6": ";".join(pref.get(item, "") for item in candidate_ids),
                "official_active_moiety_candidate_n_v6": len(candidate_ids),
            }
        )
    return pd.DataFrame(rows)


def mechanism_index() -> dict[tuple[str, str], list[dict[str, Any]]]:
    data = pd.read_csv(MECHANISMS, low_memory=False)
    data = data[
        data["organism"].eq("Homo sapiens")
        & pd.to_numeric(data["direct_interaction"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(data["molecular_mechanism"], errors="coerce").fillna(0).eq(1)
    ].copy()
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in data.itertuples(index=False):
        genes = [gene.strip() for gene in str(row.component_gene_symbols or "").split(";") if gene.strip()]
        identifiers = {str(row.molecule_chembl_id), str(row.parent_molecule_chembl_id)} - {"", "nan", "None"}
        record = {
            "action_type": str(row.action_type or ""),
            "mechanism_of_action": str(row.mechanism_of_action or ""),
            "target_type": str(row.target_type or ""),
            "max_phase": row.max_phase,
        }
        for identifier in identifiers:
            for gene in genes:
                index.setdefault((identifier, gene), []).append(record)
    return index


def activity_index() -> dict[tuple[str, str], set[str]]:
    columns = ["sequence_key", "parent_molecule_chembl_id", "calibration_label"]
    data = pd.read_csv(ACTIVITY, usecols=columns, low_memory=False)
    index: dict[tuple[str, str], set[str]] = {}
    for sequence, molecule, label in data.itertuples(index=False, name=None):
        index.setdefault((str(sequence), str(molecule)), set()).add(str(label))
    return index


def classify(labels: set[str], mechanisms: list[dict[str, Any]], mapping_status: str) -> str:
    if mechanisms:
        return "C1_known_chembl_moa_component"
    if "positive" in labels:
        return "C2_known_quantitative_binding_positive"
    if "negative_or_inactive" in labels and len(labels) == 1:
        return "N_known_quantitative_negative"
    if labels:
        return "R_quantitative_grey_or_conflicting"
    if mapping_status in {
        "ambiguous_multiple_phase4_stereoisomers",
        "no_phase4_structure_match_use_project_ids",
        "unmapped",
    }:
        return "R_active_moiety_mapping_ambiguous"
    return "D_unreported_in_local_chembl37"


def main() -> None:
    db = locate_db()
    for path in [V5, MECHANISMS, ACTIVITY, db]:
        if not path.is_file():
            raise FileNotFoundError(path)
    portfolio = pd.read_csv(V5, low_memory=False)
    phase4 = phase4_structures(db)
    identities = identity_map(portfolio, phase4)
    mechanisms = mechanism_index()
    activities = activity_index()
    data = portfolio.merge(identities, on=["drug_chembl_id", "drug_name"], how="left", validate="many_to_one")
    audit_rows: list[dict[str, Any]] = []
    for row in data.itertuples(index=False):
        identifiers = [item for item in str(row.official_active_moiety_chembl_ids_v6 or "").split(";") if item]
        mechanism_rows: list[dict[str, Any]] = []
        labels: set[str] = set()
        activity_ids: list[str] = []
        for identifier in identifiers:
            mechanism_rows.extend(mechanisms.get((identifier, str(row.primary_gene)), []))
            found = activities.get((str(row.sequence_key), identifier), set())
            if found:
                activity_ids.append(identifier)
                labels.update(found)
        pair_class = classify(labels, mechanism_rows, str(row.official_active_moiety_mapping_status_v6))
        audit_rows.append(
            {
                "drug_chembl_id": row.drug_chembl_id,
                "sequence_key": row.sequence_key,
                "primary_gene": row.primary_gene,
                "known_pair_class_v6": pair_class,
                "known_chembl_moa_component_v6": bool(mechanism_rows),
                "known_chembl_moa_action_types_v6": ";".join(
                    sorted({record["action_type"] for record in mechanism_rows if record["action_type"]})
                ),
                "known_chembl_moa_descriptions_v6": ";".join(
                    sorted({record["mechanism_of_action"] for record in mechanism_rows if record["mechanism_of_action"]})
                ),
                "known_chembl_moa_target_types_v6": ";".join(
                    sorted({record["target_type"] for record in mechanism_rows if record["target_type"]})
                ),
                "exact_chembl_activity_labels_v6": ";".join(sorted(labels)),
                "exact_chembl_activity_molecule_ids_v6": ";".join(sorted(set(activity_ids))),
                "discovery_eligible_after_known_pair_audit_v6": pair_class == "D_unreported_in_local_chembl37",
                "previous_known_pair_flag_v5": bool(row.is_known_fda_target_pair),
            }
        )
    audit = data.merge(
        pd.DataFrame(audit_rows),
        on=["drug_chembl_id", "sequence_key", "primary_gene"],
        how="left",
        validate="one_to_one",
    )
    audit["newly_caught_known_or_conflicting_pair_v6"] = (
        ~audit["known_pair_class_v6"].eq("D_unreported_in_local_chembl37")
        & ~audit["previous_known_pair_flag_v5"]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUT / "FDA_ACTIVE_MOIETY_CHEMBL37_IDENTITY_MAP_V6.csv"
    audit_path = OUT / "FINAL1000_KNOWN_PAIR_NOVELTY_AUDIT_V6.csv"
    rediscovery_path = OUT / "FINAL1000_REDISCOVERY_CONTROL_AND_CONTRADICTION_V6.csv"
    discovery_path = OUT / "FINAL1000_LOCAL_CHEMBL37_UNREPORTED_V6.csv"
    identities.to_csv(identity_path, index=False)
    audit.to_csv(audit_path, index=False)
    audit[~audit["discovery_eligible_after_known_pair_audit_v6"]].to_csv(rediscovery_path, index=False)
    audit[audit["discovery_eligible_after_known_pair_audit_v6"]].to_csv(discovery_path, index=False)
    class_counts = audit["known_pair_class_v6"].value_counts().to_dict()
    lane_class = pd.crosstab(audit["portfolio_lane_v5"], audit["known_pair_class_v6"])
    lane_path = OUT / "KNOWN_PAIR_CLASS_BY_PORTFOLIO_LANE_V6.csv"
    lane_class.to_csv(lane_path)
    checks = {
        "rows_1000": len(audit) == 1000,
        "pairs_unique": not audit[["drug_chembl_id", "sequence_key"]].duplicated().any(),
        "identity_rows_match_unique_drugs": len(identities) == portfolio["drug_chembl_id"].nunique(),
        "classes_complete": audit["known_pair_class_v6"].notna().all(),
        "discovery_and_non_discovery_partition": int(audit["discovery_eligible_after_known_pair_audit_v6"].sum())
        + int((~audit["discovery_eligible_after_known_pair_audit_v6"]).sum())
        == 1000,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failures = sorted(key for key, value in checks.items() if not value)
    summary = {
        "status": "passed" if not failures else "failed",
        "created_utc": now(),
        "checks": checks,
        "failures": failures,
        "portfolio_rows": int(len(audit)),
        "unique_drugs": int(portfolio["drug_chembl_id"].nunique()),
        "known_pair_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "discovery_eligible_rows": int(audit["discovery_eligible_after_known_pair_audit_v6"].sum()),
        "newly_caught_rows": int(audit["newly_caught_known_or_conflicting_pair_v6"].sum()),
        "inputs": {
            str(V5.relative_to(ROOT)): sha256(V5),
            str(MECHANISMS.relative_to(ROOT)): sha256(MECHANISMS),
            str(ACTIVITY.relative_to(ROOT)): sha256(ACTIVITY),
            str(db): sha256(db),
        },
    }
    summary["active_moiety_phase4_mapped_drugs"] = int(
        (~identities["official_active_moiety_mapping_status_v6"].isin(
            ["unmapped", "no_phase4_structure_match_use_project_ids"]
        )).sum()
    )
    output_paths = [identity_path, audit_path, rediscovery_path, discovery_path, lane_path]
    summary["outputs"] = {str(path.relative_to(ROOT)): sha256(path) for path in output_paths}
    summary_path = OUT / "FINAL1000_KNOWN_PAIR_NOVELTY_AUDIT_SUMMARY_V6.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
