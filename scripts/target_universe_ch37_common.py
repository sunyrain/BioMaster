#!/usr/bin/env python3
"""Shared extraction utilities for the ChEMBL 37 target universe."""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BOOL_OT_COLUMNS = [
    "l1_enzyme", "l1_ion_channel", "l1_transporter", "l1_epigenetic_regulator",
    "l1_transcription_factor", "l1_other_nuclear_protein", "l1_membrane_receptor",
    "l1_secreted_protein", "l1_surface_antigen", "l1_adhesion", "l1_structural_protein",
    "l1_unclassified_protein", "l1_other_cytosolic_protein", "class_contains_kinase",
    "class_contains_gpcr", "sm_approved_drug", "sm_advanced_clinical", "sm_phase1_clinical",
    "sm_structure_with_ligand", "sm_high_quality_ligand", "sm_high_quality_pocket",
    "sm_med_quality_pocket", "sm_druggable_family", "ab_approved_drug", "pr_approved_drug",
    "protein_coding", "sm_clinical_evidence", "sm_structure_or_pocket_evidence",
    "sm_any_evidence", "sm_direct_nonfamily_evidence", "sm_high_conf_evidence",
    "sm_family_only", "project_target_engagement_class", "excluded_membrane_receptor_gpcr",
    "excluded_secreted_surface_adhesion_structural", "project_standard_any_sm",
    "project_standard_direct_sm", "project_standard_high_conf_sm",
]

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "UNK": "X",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def join_unique(values: Iterable[Any], separator: str = ";") -> str:
    output: list[str] = []
    for value in values:
        for token in re.split(r"[;|]", clean(value)):
            token = token.strip()
            if token and token not in output:
                output.append(token)
    return separator.join(sorted(output))


def load_official_targets(connection: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT DISTINCT
            td.tid,
            td.chembl_id AS target_chembl_id,
            td.pref_name AS target_name,
            td.organism,
            td.target_type,
            tc.component_id,
            tc.homologue,
            cs.accession AS uniprot_accession,
            cs.description AS protein_description,
            cs.sequence,
            cs.sequence_md5sum,
            cs.db_source AS sequence_db_source,
            cs.db_version AS sequence_db_version,
            (
                SELECT group_concat(component_synonym, ';')
                FROM component_synonyms syn
                WHERE syn.component_id = cs.component_id AND syn.syn_type = 'GENE_SYMBOL'
            ) AS gene_symbol
        FROM drug_mechanism dm
        JOIN target_dictionary td ON td.tid = dm.tid
        JOIN target_components tc ON tc.tid = td.tid
        JOIN component_sequences cs ON cs.component_id = tc.component_id
        WHERE td.organism = 'Homo sapiens' AND td.target_type = 'SINGLE PROTEIN'
        ORDER BY td.chembl_id
    """
    output = pd.read_sql_query(query, connection)
    if output["target_chembl_id"].duplicated().any():
        raise RuntimeError("A human SINGLE PROTEIN target ID maps to more than one component")
    output["gene_symbol"] = output["gene_symbol"].map(lambda value: clean(value).split(";")[0])
    output["sequence"] = output["sequence"].map(clean)
    output["sequence_length"] = output["sequence"].str.len()
    output["sequence_sha256"] = output["sequence"].map(sha256_text)
    output["target_key"] = "CH37_" + output["target_chembl_id"].astype(str)
    output["sequence_key"] = output["sequence_sha256"].str[:16].map(lambda value: f"SEQ37_{value}")
    return output


def load_mechanism_summary(connection: sqlite3.Connection) -> pd.DataFrame:
    query = """
        SELECT td.chembl_id AS target_chembl_id, dm.mec_id,
               md.chembl_id AS molecule_chembl_id, md.pref_name AS molecule_name,
               md.molecule_type, md.max_phase, dm.action_type, dm.direct_interaction,
               dm.molecular_mechanism, dm.disease_efficacy, dm.mechanism_of_action
        FROM drug_mechanism dm
        JOIN target_dictionary td ON td.tid = dm.tid
        JOIN molecule_dictionary md ON md.molregno = dm.molregno
        WHERE td.organism = 'Homo sapiens' AND td.target_type = 'SINGLE PROTEIN'
    """
    rows = pd.read_sql_query(query, connection)
    rows["is_small_molecule"] = rows["molecule_type"].eq("Small molecule")
    rows["is_approved"] = pd.to_numeric(rows["max_phase"], errors="coerce").eq(4)
    rows["is_sm_approved"] = rows["is_small_molecule"] & rows["is_approved"]
    rows["is_sm_direct"] = rows["is_small_molecule"] & pd.to_numeric(
        rows["direct_interaction"], errors="coerce"
    ).eq(1)
    output: list[dict[str, Any]] = []
    for target_id, group in rows.groupby("target_chembl_id", sort=False):
        sm = group[group["is_small_molecule"]]
        molecule_types = sorted(set(clean(value) for value in group["molecule_type"] if clean(value)))
        record: dict[str, Any] = {
            "target_chembl_id": target_id,
            "moa_record_count": int(len(group)),
            "moa_molecule_count": int(group["molecule_chembl_id"].nunique()),
            "moa_max_phase": float(pd.to_numeric(group["max_phase"], errors="coerce").max()),
            "moa_molecule_types": ";".join(molecule_types),
            "moa_action_types": join_unique(group["action_type"]),
            "small_molecule_moa": bool(len(sm)),
            "small_molecule_direct_moa": bool(group["is_sm_direct"].any()),
            "small_molecule_moa_record_count": int(len(sm)),
            "small_molecule_count": int(sm["molecule_chembl_id"].nunique()),
            "approved_small_molecule_moa": bool(group["is_sm_approved"].any()),
            "approved_small_molecule_count": int(
                group.loc[group["is_sm_approved"], "molecule_chembl_id"].nunique()
            ),
            "small_molecule_max_phase": float(pd.to_numeric(sm["max_phase"], errors="coerce").max()) if len(sm) else math.nan,
            "small_molecule_action_types": join_unique(sm["action_type"]),
        }
        for molecule_type in [
            "Small molecule", "Antibody", "Protein", "Antibody drug conjugate", "Oligonucleotide",
            "Oligosaccharide", "Gene", "Enzyme", "Cell", "Vaccine component", "Unknown",
        ]:
            key = re.sub(r"[^a-z0-9]+", "_", molecule_type.lower()).strip("_")
            record[f"moa_has_{key}"] = molecule_type in molecule_types
        output.append(record)
    return pd.DataFrame(output)


def load_classification(connection: sqlite3.Connection, component_ids: set[int]) -> pd.DataFrame:
    classes = pd.read_sql_query(
        "SELECT protein_class_id, parent_id, pref_name, short_name, class_level FROM protein_classification",
        connection,
    )
    class_by_id = classes.set_index("protein_class_id").to_dict(orient="index")
    placeholders = ",".join("?" for _ in component_ids)
    leaves = pd.read_sql_query(
        f"SELECT component_id, protein_class_id FROM component_class WHERE component_id IN ({placeholders})",
        connection,
        params=sorted(component_ids),
    )
    leaves_by_component = leaves.groupby("component_id")["protein_class_id"].apply(list).to_dict()
    output: list[dict[str, Any]] = []
    for component_id in sorted(component_ids):
        ancestors: dict[int, dict[str, Any]] = {}
        for leaf_id in leaves_by_component.get(component_id, []):
            current = int(leaf_id)
            visited: set[int] = set()
            while current and current not in visited and current in class_by_id:
                visited.add(current)
                item = class_by_id[current]
                ancestors[current] = item
                parent = item.get("parent_id")
                current = int(parent) if pd.notna(parent) else 0
        names = [clean(item.get("pref_name")) for item in ancestors.values()]
        shorts = [clean(item.get("short_name")) for item in ancestors.values()]
        by_level: dict[int, list[str]] = defaultdict(list)
        for item in ancestors.values():
            level = int(item["class_level"])
            name = clean(item["pref_name"])
            if name and name not in by_level[level]:
                by_level[level].append(name)
        text = " ".join(names + shorts).lower()
        output.append({
            "component_id": component_id,
            "target_class_l1": ";".join(sorted(by_level[1])),
            "target_class_l2": ";".join(sorted(by_level[2])),
            "target_class_l3": ";".join(sorted(by_level[3])),
            "target_class_leaf": ";".join(sorted(
                clean(class_by_id[int(value)]["pref_name"])
                for value in leaves_by_component.get(component_id, []) if int(value) in class_by_id
            )),
            "target_class_all": ";".join(sorted(set(names))),
            "chembl_gpcr": bool(
                "g protein-coupled receptor" in text or any(short.upper().startswith("7TM") for short in shorts)
            ),
            "chembl_kinase": bool("kinase" in text),
        })
    return pd.DataFrame(output)


def aggregate_opentargets(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    for column in BOOL_OT_COLUMNS:
        source[column] = source.get(column, False)
        source[column] = source[column].fillna(False).astype(bool)
    text_columns = ["id", "approvedName", "biotype", "target_class_labels", "target_class_l1", "project_assay_family"]
    output: list[dict[str, Any]] = []
    for symbol, group in source.dropna(subset=["approvedSymbol"]).groupby("approvedSymbol", sort=False):
        row: dict[str, Any] = {"gene_symbol": clean(symbol), "ot_record_count": int(len(group))}
        for column in text_columns:
            row[f"ot_{column}"] = join_unique(group[column])
        for column in BOOL_OT_COLUMNS:
            row[f"ot_{column}"] = bool(group[column].any())
        output.append(row)
    return pd.DataFrame(output)


def classify_assay_lane(row: pd.Series) -> str:
    l1 = set(clean(row.get("target_class_l1")).split(";")) - {""}
    if bool(row.get("is_gpcr")):
        return "GPCR_MEMBRANE_ASSAY"
    if bool(row.get("chembl_kinase")):
        return "KINASE_BIOCHEMICAL"
    if "Enzyme" in l1:
        return "ENZYME_BIOCHEMICAL"
    if l1 & {"Transcription factor", "Epigenetic regulator", "Other nuclear protein"}:
        return "NUCLEAR_EPIGENETIC_DOMAIN"
    if "Ion channel" in l1:
        return "ION_CHANNEL_FUNCTIONAL"
    if "Transporter" in l1 or "Auxiliary transport protein" in l1:
        return "TRANSPORTER_MEMBRANE_FUNCTIONAL"
    if "Membrane receptor" in l1 or "Other membrane protein" in l1:
        return "NON_GPCR_MEMBRANE_SPECIAL"
    if l1 & {"Secreted protein", "Surface antigen", "Adhesion"}:
        return "EXTRACELLULAR_SPECIAL"
    return "NONCANONICAL_REVIEW"


def pdb_sequence_and_plddt(path: Path) -> tuple[str, float | None, float | None]:
    sequence: list[str] = []
    plddt: list[float] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            key = (line[21], line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            sequence.append(AA3_TO_1.get(line[17:20].strip(), "X"))
            try:
                plddt.append(float(line[60:66]))
            except ValueError:
                pass
    mean = float(sum(plddt) / len(plddt)) if plddt else None
    low_pct = float(sum(value < 50 for value in plddt) / len(plddt) * 100) if plddt else None
    return "".join(sequence), mean, low_pct


def index_local_alphafold(directory: Path) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for path in sorted(directory.glob("AF-*-F1-model_v*.pdb")):
        sequence, mean_plddt, low_pct = pdb_sequence_and_plddt(path)
        match = re.match(r"AF-(.+)-F1-model_v(\d+)\.pdb$", path.name)
        output.append({
            "af_accession": match.group(1) if match else "",
            "af_version": int(match.group(2)) if match else 0,
            "af_pdb_path": str(path.resolve()),
            "af_sequence_length": len(sequence),
            "sequence_sha256": sha256_text(sequence),
            "af_mean_plddt": mean_plddt,
            "af_low_plddt_pct": low_pct,
        })
    return pd.DataFrame(output)


def select_exact_structures(master: pd.DataFrame, af_index: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in af_index.to_dict(orient="records"):
        grouped[row["sequence_sha256"]].append(row)
    output: list[dict[str, Any]] = []
    for target in master.to_dict(orient="records"):
        candidates = sorted(
            grouped.get(target["sequence_sha256"], []),
            key=lambda row: (
                row["af_accession"] == target["uniprot_accession"], row["af_version"],
                row["af_mean_plddt"] if row["af_mean_plddt"] is not None else -1, row["af_accession"],
            ),
            reverse=True,
        )
        selected = candidates[0] if candidates else {}
        output.append({
            "target_chembl_id": target["target_chembl_id"],
            "af_exact_sequence_model": bool(candidates),
            "af_exact_model_count": len(candidates),
            "af_selected_accession": selected.get("af_accession", ""),
            "af_selected_version": selected.get("af_version", math.nan),
            "af_pdb_path": selected.get("af_pdb_path", ""),
            "af_mean_plddt": selected.get("af_mean_plddt", math.nan),
            "af_low_plddt_pct": selected.get("af_low_plddt_pct", math.nan),
        })
    return pd.DataFrame(output)


def pdb_ca_plddt_by_residue(path: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "_"
            residue = line[22:26].strip() + line[26].strip()
            try:
                output[f"{chain}_{residue}"] = float(line[60:66])
            except ValueError:
                continue
    return output


def read_p2rank(path: Path, receptor_path: Path | None = None) -> dict[str, Any]:
    pred = pd.read_csv(path)
    pred.columns = [clean(column) for column in pred.columns]
    if pred.empty:
        return {"p2rank_status": "completed_no_pocket", "p2rank_tier": "D_NO_POCKET"}
    for column in ["rank", "score", "probability", "sas_points", "surf_atoms", "center_x", "center_y", "center_z"]:
        if column in pred:
            pred[column] = pd.to_numeric(pred[column], errors="coerce")
    top = pred.sort_values("rank").iloc[0]
    probability = float(top.get("probability")) if pd.notna(top.get("probability")) else math.nan
    score = float(top.get("score")) if pd.notna(top.get("score")) else math.nan
    sas = float(top.get("sas_points")) if pd.notna(top.get("sas_points")) else math.nan
    if probability >= 0.50 and score >= 5 and sas >= 20:
        tier = "A_HIGH_CONFIDENCE"
    elif probability >= 0.20 and score >= 3 and sas >= 10:
        tier = "B_MODERATE_CONFIDENCE"
    elif probability >= 0.05 and score >= 1:
        tier = "C_WEAK_REVIEW"
    else:
        tier = "D_LOW_CONFIDENCE"
    result = {
        "p2rank_status": "completed", "p2rank_tier": tier,
        "p2rank_top_rank": top.get("rank"), "p2rank_top_score": score,
        "p2rank_top_probability": probability, "p2rank_top_sas_points": sas,
        "p2rank_top_surf_atoms": top.get("surf_atoms"), "p2rank_center_x": top.get("center_x"),
        "p2rank_center_y": top.get("center_y"), "p2rank_center_z": top.get("center_z"),
        "p2rank_residue_ids": clean(top.get("residue_ids")), "p2rank_file": str(path.resolve()),
    }
    if receptor_path is not None and receptor_path.is_file():
        confidence = pdb_ca_plddt_by_residue(receptor_path)
        residue_ids = clean(top.get("residue_ids")).split()
        values = [confidence[value] for value in residue_ids if value in confidence]
        if values:
            result.update({
                "pocket_residue_count": len(residue_ids),
                "pocket_plddt_mapped_count": len(values),
                "pocket_mean_plddt": float(sum(values) / len(values)),
                "pocket_median_plddt": float(statistics.median(values)),
                "pocket_min_plddt": float(min(values)),
                "pocket_residues_plddt_ge70_pct": float(sum(value >= 70 for value in values) / len(values) * 100),
            })
    return result


def attach_p2rank(master: pd.DataFrame, directories: list[Path]) -> pd.DataFrame:
    by_basename: dict[str, Path] = {}
    for directory in directories:
        if directory.is_dir():
            for path in directory.glob("*_predictions.csv"):
                by_basename[path.name[: -len("_predictions.csv")]] = path
    output: list[dict[str, Any]] = []
    for row in master.to_dict(orient="records"):
        basename = Path(clean(row.get("af_pdb_path"))).name
        result = {
            "target_chembl_id": row["target_chembl_id"],
            "p2rank_status": "not_run_no_exact_structure" if not basename else "not_run_exact_structure",
            "p2rank_tier": "NOT_RUN",
            "p2rank_file": "",
        }
        if basename in by_basename:
            receptor_path = Path(clean(row.get("af_pdb_path")))
            result.update(read_p2rank(by_basename[basename], receptor_path))
        output.append(result)
    return master.merge(pd.DataFrame(output), on="target_chembl_id", how="left", validate="one_to_one")
