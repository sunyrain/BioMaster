#!/usr/bin/env python3
"""Build the unified drug-centric 720 x 384 target-pair matrix.

The strict 338-target matrix and recovered 46-target matrix were originally
ranked independently.  Concatenating them would leave within-drug target ranks
incompatible.  This builder freezes one disjoint 384-target universe, recomputes
all target-within-drug ranks, retains pocket-source-stratified ranks, and locks
known ChEMBL 37 mechanism relationships so they cannot be presented as novel.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
MAIN_MATRIX = ROOT / "outputs/strict_dta_720x338_v1/STRICT_DTA_720_X_338_EVIDENCE_MATRIX.csv.gz"
RECOVERED_MATRIX = ROOT / "outputs/recovered_dta_720x46_v1/RECOVERED_DTA_720_X_46_EVIDENCE_MATRIX_V1.csv.gz"
MAIN_TARGETS = ROOT / "outputs/final_target_package_ch37/FINAL_FROZEN_CHEMBL_SM_TARGETS_WITH_DRUGLIKE_HOLO_338.csv"
RECOVERED_TARGETS = ROOT / "outputs/recovered_target_program_integrated_v1/RECOVERED_46_INTEGRATED_TARGET_OUTCOMES_V1.csv"
ACTIVE_TARGETS = ROOT / "outputs/recovered_target_program_integrated_v1/ACTIVE_TARGET_BRANCHES_384_V1.csv"
LIGANDS = ROOT / "outputs/strict_affinity_main_queue_2005_2026_v2/STRICT_UNIQUE_STANDARD_MODEL_LIGAND_STRUCTURES.csv"
LIGAND_SPECIES = ROOT / "outputs/strict_affinity_main_queue_2005_2026_v2/STRICT_STANDARD_MODEL_LIGAND_SPECIES_MAP.csv"
CHEMBL = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OUTDIR = ROOT / "outputs/unified_pair_program_720x384_v1"

N_LIGANDS = 720
N_MAIN = 338
N_RECOVERED = 46
N_TARGETS = 384
N_DRUGCLIP = 382
EXPECTED_ROWS = N_LIGANDS * N_TARGETS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def rank_desc(values: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(
        rankdata(-values, axis=axis, method="average", nan_policy="omit"),
        dtype=np.float32,
    )


def percentile(ranks: np.ndarray, total: int) -> np.ndarray:
    return 1.0 - (ranks.astype(np.float32) - 1.0) / max(1, total - 1)


def join_unique(values: list[Any] | pd.Series) -> str:
    return ";".join(sorted({clean(v) for v in values if clean(v)}))


def build_target_master() -> pd.DataFrame:
    active = pd.read_csv(ACTIVE_TARGETS, dtype=str).fillna("")
    main = pd.read_csv(MAIN_TARGETS, dtype=str).fillna("")
    recovered = pd.read_csv(RECOVERED_TARGETS, dtype=str).fillna("")
    if len(active) != N_TARGETS or active["target_chembl_id"].nunique() != N_TARGETS:
        raise RuntimeError("Active target manifest is not exactly 384 unique targets")

    main_metadata = main[[
        "target_chembl_id", "target_class_zh", "assay_lane",
        "experimental_pocket_evidence_category", "experimental_pocket_evidence_zh",
        "representative_pdb_id", "representative_chain_id", "representative_ligand_id",
        "positive_compounds", "negative_compounds", "target_calibration_tier",
    ]].copy()
    main_metadata["mechanistic_branch"] = main_metadata["assay_lane"]
    main_metadata["mechanistic_branch_zh"] = main_metadata["target_class_zh"]
    main_metadata["mechanistic_subclass"] = "LEGACY_MAINLINE_CLASSIFICATION"
    main_metadata["pocket_evidence_tier"] = main_metadata["experimental_pocket_evidence_category"]
    main_metadata["pocket_evidence_source"] = "PREFERRED_DRUGLIKE_HOLO_EXPERIMENTAL_POCKET"
    main_metadata["drugclip_formal_eligible"] = True

    recovered_metadata = recovered[[
        "target_chembl_id", "target_compute_class_zh", "assay_lane", "mechanistic_branch",
        "mechanistic_branch_zh", "mechanistic_subclass", "computed_pocket_evidence",
        "positive_compounds", "negative_compounds", "calibration_status",
    ]].copy().rename(columns={
        "target_compute_class_zh": "target_class_zh",
        "computed_pocket_evidence": "pocket_evidence_tier",
        "calibration_status": "target_calibration_tier",
    })
    recovered_metadata["experimental_pocket_evidence_category"] = ""
    recovered_metadata["experimental_pocket_evidence_zh"] = ""
    recovered_metadata["representative_pdb_id"] = ""
    recovered_metadata["representative_chain_id"] = ""
    recovered_metadata["representative_ligand_id"] = ""
    recovered_metadata["pocket_evidence_source"] = "COMPUTATIONAL_POCKET_PREDICTION"
    recovered_metadata["drugclip_formal_eligible"] = ~recovered_metadata["target_chembl_id"].isin(
        {"CHEMBL2019", "CHEMBL1846"}  # DIO1 and RYR1: low-confidence fragment models.
    )

    metadata = pd.concat([main_metadata, recovered_metadata], ignore_index=True)
    if len(metadata) != N_TARGETS or metadata["target_chembl_id"].nunique() != N_TARGETS:
        raise RuntimeError("Mainline/recovered target metadata are not a disjoint 338+46 union")
    target_master = active.merge(metadata, on=["target_chembl_id", "assay_lane"], how="left", validate="one_to_one")
    if target_master["pocket_evidence_source"].isna().any():
        raise RuntimeError("Target metadata merge left missing rows")
    target_master = target_master.rename(columns={
        "pocket_evidence_source": "pair_pocket_evidence_source",
        "binding_site_evidence_source": "active_manifest_pocket_evidence_source",
    })
    target_master["drugclip_formal_eligible"] = bool_series(target_master["drugclip_formal_eligible"])
    target_master = target_master.sort_values("target_chembl_id", kind="mergesort").reset_index(drop=True)
    return target_master


def related_molecule_map(connection: sqlite3.Connection, species: pd.DataFrame) -> tuple[dict[str, set[int]], pd.DataFrame]:
    chembl_ids = sorted({clean(v) for v in species["ligand_species_chembl_id"] if clean(v)})
    placeholders = ",".join("?" for _ in chembl_ids)
    molecules = pd.read_sql_query(
        f"SELECT molregno, chembl_id, pref_name, molecule_type, max_phase FROM molecule_dictionary WHERE chembl_id IN ({placeholders})",
        connection,
        params=chembl_ids,
    )
    if len(molecules) != len(chembl_ids):
        missing = sorted(set(chembl_ids) - set(molecules["chembl_id"]))
        raise RuntimeError(f"ChEMBL molecule IDs missing for ligand species: {missing}")
    chembl_to_molregno = molecules.set_index("chembl_id")["molregno"].astype(int).to_dict()
    seed_molregnos = sorted(set(chembl_to_molregno.values()))

    connection.execute("DROP TABLE IF EXISTS temp.unified_seed_molecules")
    connection.execute("CREATE TEMP TABLE unified_seed_molecules (molregno INTEGER PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO unified_seed_molecules(molregno) VALUES (?)",
        [(value,) for value in seed_molregnos],
    )
    seed_hierarchy = pd.read_sql_query(
        """
        SELECT h.molregno, h.parent_molregno, h.active_molregno
        FROM molecule_hierarchy h
        JOIN unified_seed_molecules s ON s.molregno = h.molregno
        """,
        connection,
    )
    hierarchy_by_seed = seed_hierarchy.set_index("molregno").to_dict(orient="index")
    roots = set(seed_molregnos)
    for row in hierarchy_by_seed.values():
        for field in ("parent_molregno", "active_molregno"):
            if pd.notna(row.get(field)):
                roots.add(int(row[field]))
    connection.execute("DROP TABLE IF EXISTS temp.unified_root_molecules")
    connection.execute("CREATE TEMP TABLE unified_root_molecules (molregno INTEGER PRIMARY KEY)")
    connection.executemany(
        "INSERT INTO unified_root_molecules(molregno) VALUES (?)",
        [(value,) for value in sorted(roots)],
    )
    related_hierarchy = pd.read_sql_query(
        """
        SELECT DISTINCT h.molregno, h.parent_molregno, h.active_molregno
        FROM molecule_hierarchy h
        LEFT JOIN unified_root_molecules m ON m.molregno = h.molregno
        LEFT JOIN unified_root_molecules p ON p.molregno = h.parent_molregno
        LEFT JOIN unified_root_molecules a ON a.molregno = h.active_molregno
        WHERE m.molregno IS NOT NULL OR p.molregno IS NOT NULL OR a.molregno IS NOT NULL
        """,
        connection,
    )

    rows_by_parent: dict[int, set[int]] = defaultdict(set)
    rows_by_active: dict[int, set[int]] = defaultdict(set)
    for row in related_hierarchy.itertuples(index=False):
        molregno = int(row.molregno)
        if pd.notna(row.parent_molregno):
            rows_by_parent[int(row.parent_molregno)].add(molregno)
        if pd.notna(row.active_molregno):
            rows_by_active[int(row.active_molregno)].add(molregno)

    inchikey_to_related: dict[str, set[int]] = defaultdict(set)
    for row in species.itertuples(index=False):
        inchikey = clean(row.ligand_inchikey)
        chembl_id = clean(row.ligand_species_chembl_id)
        seed = int(chembl_to_molregno[chembl_id])
        hierarchy = hierarchy_by_seed.get(seed, {})
        parent = int(hierarchy["parent_molregno"]) if pd.notna(hierarchy.get("parent_molregno")) else seed
        active = int(hierarchy["active_molregno"]) if pd.notna(hierarchy.get("active_molregno")) else seed
        related = {seed, parent, active}
        related.update(rows_by_parent.get(parent, set()))
        related.update(rows_by_active.get(active, set()))
        inchikey_to_related[inchikey].update(related)
    return inchikey_to_related, molecules


def build_chembl_mechanism_pairs(ligands: pd.DataFrame, active_target_ids: set[str]) -> pd.DataFrame:
    species = pd.read_csv(LIGAND_SPECIES, dtype=str).fillna("")
    connection = sqlite3.connect(CHEMBL)
    try:
        inchikey_to_related, seed_molecules = related_molecule_map(connection, species)
        all_related = sorted(set().union(*inchikey_to_related.values()))
        connection.execute("DROP TABLE IF EXISTS temp.unified_related_molecules")
        connection.execute("CREATE TEMP TABLE unified_related_molecules (molregno INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO unified_related_molecules(molregno) VALUES (?)",
            [(value,) for value in all_related],
        )
        mechanisms = pd.read_sql_query(
            """
            SELECT dm.molregno, md.chembl_id AS mechanism_molecule_chembl_id,
                   md.molecule_type, td.chembl_id AS target_chembl_id,
                   dm.action_type, dm.direct_interaction, dm.molecular_mechanism,
                   dm.mechanism_of_action
            FROM drug_mechanism dm
            JOIN unified_related_molecules r ON r.molregno = dm.molregno
            JOIN molecule_dictionary md ON md.molregno = dm.molregno
            JOIN target_dictionary td ON td.tid = dm.tid
            WHERE td.organism = 'Homo sapiens'
              AND td.target_type = 'SINGLE PROTEIN'
              AND md.molecule_type = 'Small molecule'
            """,
            connection,
        )
    finally:
        connection.close()

    mechanisms = mechanisms[mechanisms["target_chembl_id"].isin(active_target_ids)].copy()
    mechanism_by_molregno: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in mechanisms.to_dict(orient="records"):
        mechanism_by_molregno[int(row["molregno"])].append(row)
    species_by_key = species.groupby("ligand_inchikey")["ligand_species_chembl_id"].apply(
        lambda values: {clean(v) for v in values if clean(v)}
    ).to_dict()
    seed_chembl_to_molregno = seed_molecules.set_index("chembl_id")["molregno"].astype(int).to_dict()

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for inchikey, related in inchikey_to_related.items():
        exact_molregnos = {seed_chembl_to_molregno[c] for c in species_by_key.get(inchikey, set())}
        for molregno in related:
            tier = "EXACT_MODEL_SPECIES" if molregno in exact_molregnos else "RELATED_PARENT_ACTIVE_OR_FORM"
            for row in mechanism_by_molregno.get(molregno, []):
                key = (inchikey, clean(row["target_chembl_id"]))
                record = records.setdefault(key, {
                    "ligand_inchikey": inchikey,
                    "target_chembl_id": key[1],
                    "is_chembl37_mechanism_relationship": True,
                    "chembl37_mechanism_match_tiers": set(),
                    "chembl37_mechanism_molecule_ids": set(),
                    "chembl37_action_types": set(),
                    "chembl37_mechanism_of_action": set(),
                    "chembl37_direct_interaction_any": False,
                })
                record["chembl37_mechanism_match_tiers"].add(tier)
                record["chembl37_mechanism_molecule_ids"].add(clean(row["mechanism_molecule_chembl_id"]))
                record["chembl37_action_types"].add(clean(row["action_type"]))
                record["chembl37_mechanism_of_action"].add(clean(row["mechanism_of_action"]))
                record["chembl37_direct_interaction_any"] = (
                    record["chembl37_direct_interaction_any"]
                    or clean(row["direct_interaction"]) in {"1", "1.0", "True", "true"}
                )
    output_rows: list[dict[str, Any]] = []
    for record in records.values():
        for column in [
            "chembl37_mechanism_match_tiers", "chembl37_mechanism_molecule_ids",
            "chembl37_action_types", "chembl37_mechanism_of_action",
        ]:
            record[column] = ";".join(sorted(value for value in record[column] if value))
        output_rows.append(record)
    output = pd.DataFrame(output_rows)
    if output.empty:
        output = pd.DataFrame(columns=[
            "ligand_inchikey", "target_chembl_id", "is_chembl37_mechanism_relationship",
            "chembl37_mechanism_match_tiers", "chembl37_mechanism_molecule_ids",
            "chembl37_action_types", "chembl37_mechanism_of_action",
            "chembl37_direct_interaction_any",
        ])
    return output.sort_values(["ligand_inchikey", "target_chembl_id"], kind="mergesort")


def prepare_main_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["active_target_branch"] = "STRICT_EXPERIMENTAL_POCKET_MAINLINE_338"
    output["pair_pocket_evidence_source"] = "PREFERRED_DRUGLIKE_HOLO_EXPERIMENTAL_POCKET"
    output["pocket_evidence_tier"] = output["target_chembl_id"].map(lambda _: "EXPERIMENTAL_HOLO")
    output["drugclip_formal_eligible"] = True
    output["conplex_rank_within_ligand_original_branch"] = output["conplex_rank_within_ligand"]
    output["conplex_percentile_within_ligand_original_branch"] = output["conplex_percentile_within_ligand"]
    output["drugclip_rank_within_ligand_original_branch"] = output["drugclip_rank_within_ligand"]
    output["drugclip_percentile_within_ligand_original_branch"] = output["drugclip_percentile_within_ligand"]
    output["drugclip_evidence_scope"] = "EXPERIMENTAL_HOLO_338"
    return output


def prepare_recovered_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().rename(columns={
        "drugclip_predicted_pocket_cosine_mean": "drugclip_cosine_mean",
        "drugclip_predicted_pocket_sixfold_std": "drugclip_sixfold_std",
        "conplex_rank_within_ligand_46": "conplex_rank_within_ligand_original_branch",
        "conplex_percentile_within_ligand_46": "conplex_percentile_within_ligand_original_branch",
        "drugclip_rank_within_ligand_44": "drugclip_rank_within_ligand_original_branch",
        "drugclip_percentile_within_ligand_44": "drugclip_percentile_within_ligand_original_branch",
        "computed_pocket_evidence": "pocket_evidence_tier",
    })
    output["active_target_branch"] = "RECOVERED_NO_EXPERIMENTAL_POCKET_46"
    output["pair_pocket_evidence_source"] = "COMPUTATIONAL_POCKET_PREDICTION"
    output["drugclip_evidence_scope"] = np.where(
        bool_series(output["drugclip_formal_eligible"]),
        "PREDICTED_POCKET_44",
        "NO_FORMAL_3D_FRAGMENT_STRUCTURE",
    )
    for column in [
        "is_known_relationship_control", "known_relationship_class", "receptor_protocol_status",
        "redock_status", "redock_best_symmetry_rmsd_A", "docking_context_included",
        "reference_ligand_completeness_final",
    ]:
        if column not in output:
            output[column] = False if column in {"is_known_relationship_control", "docking_context_included"} else ""
    return output


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ligands = pd.read_csv(LIGANDS, dtype=str).fillna("").sort_values("ligand_inchikey", kind="mergesort")
    target_master = build_target_master()
    main = prepare_main_matrix(pd.read_csv(MAIN_MATRIX, low_memory=False))
    recovered = prepare_recovered_matrix(pd.read_csv(RECOVERED_MATRIX, low_memory=False))
    if len(main) != N_LIGANDS * N_MAIN or len(recovered) != N_LIGANDS * N_RECOVERED:
        raise RuntimeError("Input DTA matrices do not have the frozen dimensions")
    if set(main["ligand_inchikey"]) != set(recovered["ligand_inchikey"]):
        raise RuntimeError("The 338 and 46 matrices use different ligand universes")
    if set(main["target_chembl_id"]) & set(recovered["target_chembl_id"]):
        raise RuntimeError("The 338 and 46 target branches overlap")

    canonical_columns = [
        "ligand_inchikey", "ligand_smiles", "drug_names", "project_entity_ids",
        "target_chembl_id", "gene_symbol", "assay_lane", "active_target_branch",
        "pair_pocket_evidence_source", "pocket_evidence_tier", "drugclip_evidence_scope",
        "drugclip_formal_eligible", "conplex_score", "conplex_rank_within_target",
        "conplex_rank_within_ligand_original_branch", "conplex_percentile_within_target",
        "conplex_percentile_within_ligand_original_branch", "drugclip_cosine_mean",
        "drugclip_sixfold_std", "drugclip_rank_within_target",
        "drugclip_rank_within_ligand_original_branch", "drugclip_percentile_within_target",
        "drugclip_percentile_within_ligand_original_branch", "is_known_relationship_control",
        "known_relationship_class", "receptor_protocol_status", "redock_status",
        "redock_best_symmetry_rmsd_A", "docking_context_included",
        "reference_ligand_completeness_final",
    ]
    matrix = pd.concat([main[canonical_columns], recovered[canonical_columns]], ignore_index=True)
    matrix["drugclip_formal_eligible"] = bool_series(matrix["drugclip_formal_eligible"])
    matrix["is_known_relationship_control"] = bool_series(matrix["is_known_relationship_control"])
    matrix = matrix.sort_values(["ligand_inchikey", "target_chembl_id"], kind="mergesort").reset_index(drop=True)
    if len(matrix) != EXPECTED_ROWS or matrix[["ligand_inchikey", "target_chembl_id"]].duplicated().any():
        raise RuntimeError("Unified pair matrix is not exactly 720 x 384 unique pairs")

    ligand_order = ligands["ligand_inchikey"].tolist()
    target_order = target_master["target_chembl_id"].tolist()
    conplex = matrix.pivot(index="ligand_inchikey", columns="target_chembl_id", values="conplex_score").reindex(index=ligand_order, columns=target_order)
    drugclip = matrix.pivot(index="ligand_inchikey", columns="target_chembl_id", values="drugclip_cosine_mean").reindex(index=ligand_order, columns=target_order)
    if conplex.isna().any().any() or drugclip.notna().sum().sum() != N_LIGANDS * N_DRUGCLIP:
        raise RuntimeError("Unified score matrices contain unexpected missing values")
    conplex_values = conplex.to_numpy(dtype=np.float32)
    drugclip_values = drugclip.to_numpy(dtype=np.float32)

    conplex_target_rank = rank_desc(conplex_values, axis=0)
    conplex_ligand_rank = rank_desc(conplex_values, axis=1)
    drugclip_target_rank = rank_desc(drugclip_values, axis=0)
    drugclip_ligand_rank = rank_desc(drugclip_values, axis=1)
    conplex_target_pct = percentile(conplex_target_rank, N_LIGANDS)
    conplex_ligand_pct = percentile(conplex_ligand_rank, N_TARGETS)
    drugclip_target_pct = percentile(drugclip_target_rank, N_LIGANDS)
    drugclip_ligand_pct = percentile(drugclip_ligand_rank, N_DRUGCLIP)

    matrix_index = pd.MultiIndex.from_frame(matrix[["ligand_inchikey", "target_chembl_id"]])
    grid_index = pd.MultiIndex.from_product([ligand_order, target_order], names=["ligand_inchikey", "target_chembl_id"])
    if not matrix_index.equals(grid_index):
        raise RuntimeError("Unified pair row order differs from the frozen ligand-target grid")
    matrix["conplex_rank_within_target"] = conplex_target_rank.reshape(-1)
    matrix["conplex_percentile_within_target"] = conplex_target_pct.reshape(-1)
    matrix["conplex_rank_within_ligand_384"] = conplex_ligand_rank.reshape(-1)
    matrix["conplex_percentile_within_ligand_384"] = conplex_ligand_pct.reshape(-1)
    matrix["drugclip_rank_within_target"] = drugclip_target_rank.reshape(-1)
    matrix["drugclip_percentile_within_target"] = drugclip_target_pct.reshape(-1)
    matrix["drugclip_rank_within_ligand_382"] = drugclip_ligand_rank.reshape(-1)
    matrix["drugclip_percentile_within_ligand_382"] = drugclip_ligand_pct.reshape(-1)

    # Source-stratified ranks prevent predicted-pocket scores from silently
    # changing the formal experimental-holo ranking and vice versa.
    matrix["conplex_rank_within_ligand_same_pocket_source"] = np.nan
    matrix["conplex_percentile_within_ligand_same_pocket_source"] = np.nan
    matrix["drugclip_rank_within_ligand_same_pocket_source"] = np.nan
    matrix["drugclip_percentile_within_ligand_same_pocket_source"] = np.nan
    source_sizes = {
        "PREFERRED_DRUGLIKE_HOLO_EXPERIMENTAL_POCKET": N_MAIN,
        "COMPUTATIONAL_POCKET_PREDICTION": N_RECOVERED,
    }
    for source, total in source_sizes.items():
        target_ids = target_master.loc[target_master["pair_pocket_evidence_source"].eq(source), "target_chembl_id"].tolist()
        eligible_ids = target_master.loc[
            target_master["pair_pocket_evidence_source"].eq(source) & target_master["drugclip_formal_eligible"],
            "target_chembl_id",
        ].tolist()
        source_conplex = conplex[target_ids].to_numpy(dtype=np.float32)
        source_conplex_rank = rank_desc(source_conplex, axis=1)
        source_conplex_pct = percentile(source_conplex_rank, len(target_ids))
        source_drugclip = drugclip[eligible_ids].to_numpy(dtype=np.float32)
        source_drugclip_rank = rank_desc(source_drugclip, axis=1)
        source_drugclip_pct = percentile(source_drugclip_rank, len(eligible_ids))
        conplex_rank_map = pd.DataFrame(source_conplex_rank, index=ligand_order, columns=target_ids).stack()
        conplex_pct_map = pd.DataFrame(source_conplex_pct, index=ligand_order, columns=target_ids).stack()
        drugclip_rank_map = pd.DataFrame(source_drugclip_rank, index=ligand_order, columns=eligible_ids).stack()
        drugclip_pct_map = pd.DataFrame(source_drugclip_pct, index=ligand_order, columns=eligible_ids).stack()
        mask = matrix["pair_pocket_evidence_source"].eq(source)
        keys = pd.MultiIndex.from_frame(matrix.loc[mask, ["ligand_inchikey", "target_chembl_id"]])
        matrix.loc[mask, "conplex_rank_within_ligand_same_pocket_source"] = conplex_rank_map.reindex(keys).to_numpy()
        matrix.loc[mask, "conplex_percentile_within_ligand_same_pocket_source"] = conplex_pct_map.reindex(keys).to_numpy()
        matrix.loc[mask, "drugclip_rank_within_ligand_same_pocket_source"] = drugclip_rank_map.reindex(keys).to_numpy()
        matrix.loc[mask, "drugclip_percentile_within_ligand_same_pocket_source"] = drugclip_pct_map.reindex(keys).to_numpy()

    matrix["dta_target_percentile_disagreement"] = (
        matrix["conplex_percentile_within_target"] - matrix["drugclip_percentile_within_target"]
    ).abs()
    matrix["dta_ligand_percentile_disagreement_384_382"] = (
        matrix["conplex_percentile_within_ligand_384"] - matrix["drugclip_percentile_within_ligand_382"]
    ).abs()
    matrix["dta_target_top10pct_concordant"] = (
        matrix["drugclip_formal_eligible"]
        & matrix["conplex_percentile_within_target"].ge(0.9)
        & matrix["drugclip_percentile_within_target"].ge(0.9)
    )
    matrix["dta_drug_centric_top10pct_concordant_384"] = (
        matrix["drugclip_formal_eligible"]
        & matrix["conplex_percentile_within_ligand_384"].ge(0.9)
        & matrix["drugclip_percentile_within_ligand_382"].ge(0.9)
    )
    matrix["dta_bidirectional_top10pct_concordant_384"] = (
        matrix["dta_target_top10pct_concordant"]
        & matrix["dta_drug_centric_top10pct_concordant_384"]
    )
    matrix["sequence_only_fragment_exploratory_top5pct"] = (
        ~matrix["drugclip_formal_eligible"]
        & matrix["conplex_percentile_within_target"].ge(0.95)
    )

    chembl_known = build_chembl_mechanism_pairs(ligands, set(target_order))
    matrix = matrix.merge(
        chembl_known,
        on=["ligand_inchikey", "target_chembl_id"],
        how="left",
        validate="one_to_one",
    )
    matrix["is_chembl37_mechanism_relationship"] = bool_series(matrix["is_chembl37_mechanism_relationship"])
    matrix["chembl37_direct_interaction_any"] = bool_series(matrix["chembl37_direct_interaction_any"])
    for column in [
        "chembl37_mechanism_match_tiers", "chembl37_mechanism_molecule_ids",
        "chembl37_action_types", "chembl37_mechanism_of_action",
    ]:
        matrix[column] = matrix[column].fillna("")
    matrix["is_any_frozen_known_relationship"] = (
        matrix["is_known_relationship_control"] | matrix["is_chembl37_mechanism_relationship"]
    )
    matrix["pair_novelty_class_384"] = np.select(
        [
            matrix["is_known_relationship_control"],
            matrix["is_chembl37_mechanism_relationship"],
        ],
        [
            "K0_KNOWN_PROJECT_RELATIONSHIP",
            "K1_KNOWN_CHEMBL37_MOA_RELATIONSHIP",
        ],
        default="N1_UNRECORDED_IN_FROZEN_PROJECT_AND_CHEMBL37_MOA",
    )
    matrix["cross_pocket_source_drugclip_rank_policy"] = np.where(
        matrix["pair_pocket_evidence_source"].eq("COMPUTATIONAL_POCKET_PREDICTION"),
        "EXPLORATORY_ONLY_PREDICTED_POCKET_NOT_CALIBRATED_TO_EXPERIMENTAL_HOLO",
        "FORMAL_EXPERIMENTAL_HOLO_SOURCE_STRATIFIED_RANK_AVAILABLE",
    )

    target_master_path = OUTDIR / "UNIFIED_ACTIVE_TARGET_MASTER_384_V1.csv"
    known_path = OUTDIR / "UNIFIED_CHEMBL37_KNOWN_MOA_PAIRS_720_X_384_V1.csv"
    matrix_path = OUTDIR / "UNIFIED_DTA_720_X_384_PAIR_MATRIX_V1.csv.gz"
    shortlist_path = OUTDIR / "UNIFIED_DTA_PAIR_SHORTLIST_720_X_384_V1.csv.gz"
    target_master.to_csv(target_master_path, index=False)
    chembl_known.to_csv(known_path, index=False)
    matrix.to_csv(matrix_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    shortlist = matrix[
        matrix["is_any_frozen_known_relationship"]
        | matrix["dta_target_top10pct_concordant"]
        | matrix["dta_drug_centric_top10pct_concordant_384"]
        | matrix["sequence_only_fragment_exploratory_top5pct"]
    ].copy()
    shortlist["shortlist_reasons"] = shortlist.apply(
        lambda row: ";".join(filter(None, [
            "KNOWN_RELATIONSHIP" if row["is_any_frozen_known_relationship"] else "",
            "TARGET_TOP10_TWO_MODEL" if row["dta_target_top10pct_concordant"] else "",
            "DRUG_CENTRIC_TOP10_TWO_MODEL" if row["dta_drug_centric_top10pct_concordant_384"] else "",
            "BIDIRECTIONAL_TOP10_TWO_MODEL" if row["dta_bidirectional_top10pct_concordant_384"] else "",
            "SEQUENCE_ONLY_FRAGMENT_TOP5" if row["sequence_only_fragment_exploratory_top5pct"] else "",
        ])),
        axis=1,
    )
    shortlist = shortlist.sort_values(
        ["is_any_frozen_known_relationship", "dta_bidirectional_top10pct_concordant_384",
         "dta_target_top10pct_concordant", "dta_drug_centric_top10pct_concordant_384",
         "ligand_inchikey", "target_chembl_id"],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    )
    shortlist.to_csv(shortlist_path, index=False, compression={"method": "gzip", "compresslevel": 5})

    rank_invariant = matrix["conplex_rank_within_target"].notna().all() and (
        matrix.loc[matrix["drugclip_formal_eligible"], "drugclip_rank_within_target"].notna().all()
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if rank_invariant else "FAIL",
        "rows": len(matrix),
        "ligands": matrix["ligand_inchikey"].nunique(),
        "targets": matrix["target_chembl_id"].nunique(),
        "target_branches": matrix.drop_duplicates("target_chembl_id")["active_target_branch"].value_counts().to_dict(),
        "conplex_targets": int(matrix.groupby("target_chembl_id")["conplex_score"].count().eq(N_LIGANDS).sum()),
        "drugclip_targets": int(matrix.groupby("target_chembl_id")["drugclip_cosine_mean"].count().eq(N_LIGANDS).sum()),
        "known_project_pairs": int(matrix["is_known_relationship_control"].sum()),
        "known_chembl37_moa_pairs": int(matrix["is_chembl37_mechanism_relationship"].sum()),
        "known_union_pairs": int(matrix["is_any_frozen_known_relationship"].sum()),
        "target_top10_two_model_pairs": int(matrix["dta_target_top10pct_concordant"].sum()),
        "drug_centric_top10_two_model_pairs": int(matrix["dta_drug_centric_top10pct_concordant_384"].sum()),
        "bidirectional_top10_two_model_pairs": int(matrix["dta_bidirectional_top10pct_concordant_384"].sum()),
        "sequence_only_fragment_top5_pairs": int(matrix["sequence_only_fragment_exploratory_top5pct"].sum()),
        "shortlist_pairs": len(shortlist),
        "novelty_counts": matrix["pair_novelty_class_384"].value_counts().to_dict(),
        "rank_policy": {
            "conplex_drug_centric": "Rank each drug across all 384 active targets",
            "drugclip_drug_centric": "Rank each drug across 382 structurally eligible targets; cross-source rank is exploratory",
            "source_stratified": "Experimental-holo 338 and predicted-pocket 44 ranks are retained separately",
            "no_cross_target_probability": True,
        },
        "novelty_boundary": (
            "N1 means absent only from the frozen project relationship controls and ChEMBL 37 small-molecule MoA mapping; "
            "it is not a claim of absence from all databases or literature."
        ),
        "outputs": {
            "target_master": str(target_master_path),
            "known_moa_pairs": str(known_path),
            "matrix": str(matrix_path),
            "shortlist": str(shortlist_path),
        },
        "sha256": {
            "matrix": sha256(matrix_path),
            "shortlist": sha256(shortlist_path),
        },
    }
    summary_path = OUTDIR / "UNIFIED_DTA_720_X_384_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
