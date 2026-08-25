#!/usr/bin/env python3
"""Audit the 15 V5 L2 pairs and freeze a capped short-MD authorization manifest."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V5_RESULTS = EXEC / "final_evidence_routing_v5/INCREMENTAL_REMOTE_N2_N3_PAIR_RESULTS_57_V1.csv"
V5_TARGETS = EXEC / "final_evidence_routing_v5/TARGET_EVIDENCE_LAYER_ROUTING_384_V5.csv"
MULTISEED = (
    ROOT
    / "outputs/unified_pair_compute_increment_384_v1/execution_v1/boltz_multiseed"
    / "stability_evaluation/UNIFIED_REMOTE_INCREMENT_BOLTZ_MULTI_SEED_STABILITY_V1.csv.gz"
)
SPECIES = (
    ROOT
    / "outputs/strict_affinity_main_queue_2005_2026_v2"
    / "STRICT_STANDARD_MODEL_LIGAND_SPECIES_MAP.csv"
)
CHEMBL = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OUT = EXEC / "v5_l2_pair_deepening_v1"

PASS_DECISION = "PASS_SCORE_AND_CONDITIONAL_POSE_STABILITY"
STANDARD_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def join_unique(values: Any) -> str:
    return ";".join(sorted({clean(value) for value in values if clean(value)}))


def boltz_pocket_residue_ids(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    positions: set[int] = set()
    for constraint in payload.get("constraints", []) or []:
        pocket = constraint.get("pocket") if isinstance(constraint, dict) else None
        if not pocket:
            continue
        for contact in pocket.get("contacts", []) or []:
            if isinstance(contact, (list, tuple)) and len(contact) >= 2:
                positions.add(int(contact[1]))
    if len(positions) < 3:
        raise ValueError(f"Fewer than three pocket contacts: {path}")
    return ";".join(str(value) for value in sorted(positions))


def ligand_features(smiles: str) -> dict[str, Any]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid ligand SMILES: {smiles}")
    elements = sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})
    specialized = sorted(set(elements) - STANDARD_ELEMENTS)
    acrylamide = Chem.MolFromSmarts("C=CC(=O)N")
    has_acrylamide = bool(acrylamide is not None and molecule.HasSubstructMatch(acrylamide))
    has_boron = "B" in elements
    if has_boron:
        warhead = "BORON_REVERSIBLE_COVALENT_WARHEAD"
    elif has_acrylamide:
        warhead = "ACRYLAMIDE_ELECTROPHILE_WARHEAD"
    else:
        warhead = "NONE_DETECTED_BY_FROZEN_RULES"
    return {
        "ligand_formal_charge": int(Chem.GetFormalCharge(molecule)),
        "ligand_elements": ";".join(elements),
        "specialized_ligand_elements": ";".join(specialized),
        "reactive_warhead_class": warhead,
    }


def related_molecules(connection: sqlite3.Connection, species: pd.DataFrame) -> dict[str, set[int]]:
    identifiers = sorted(set(species["ligand_species_chembl_id"].map(clean)) - {""})
    placeholders = ",".join("?" for _ in identifiers)
    molecules = pd.read_sql_query(
        f"SELECT molregno, chembl_id FROM molecule_dictionary WHERE chembl_id IN ({placeholders})",
        connection,
        params=identifiers,
    )
    identifier_to_molregno = molecules.set_index("chembl_id")["molregno"].astype(int).to_dict()
    missing = sorted(set(identifiers) - set(identifier_to_molregno))
    if missing:
        raise RuntimeError(f"ChEMBL species identifiers are missing: {missing}")
    seeds = sorted(set(identifier_to_molregno.values()))
    connection.execute("CREATE TEMP TABLE audited_seed_molecules (molregno INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO audited_seed_molecules VALUES (?)", [(value,) for value in seeds])
    hierarchy = pd.read_sql_query(
        """
        SELECT h.molregno, h.parent_molregno, h.active_molregno
        FROM molecule_hierarchy h JOIN audited_seed_molecules s ON s.molregno = h.molregno
        """,
        connection,
    )
    hierarchy_by_seed = hierarchy.set_index("molregno").to_dict(orient="index")
    roots = set(seeds)
    for record in hierarchy_by_seed.values():
        for column in ("parent_molregno", "active_molregno"):
            if pd.notna(record.get(column)):
                roots.add(int(record[column]))
    connection.execute("CREATE TEMP TABLE audited_root_molecules (molregno INTEGER PRIMARY KEY)")
    connection.executemany("INSERT INTO audited_root_molecules VALUES (?)", [(value,) for value in sorted(roots)])
    related = pd.read_sql_query(
        """
        SELECT DISTINCT h.molregno, h.parent_molregno, h.active_molregno
        FROM molecule_hierarchy h
        LEFT JOIN audited_root_molecules m ON m.molregno = h.molregno
        LEFT JOIN audited_root_molecules p ON p.molregno = h.parent_molregno
        LEFT JOIN audited_root_molecules a ON a.molregno = h.active_molregno
        WHERE m.molregno IS NOT NULL OR p.molregno IS NOT NULL OR a.molregno IS NOT NULL
        """,
        connection,
    )
    by_parent: dict[int, set[int]] = defaultdict(set)
    by_active: dict[int, set[int]] = defaultdict(set)
    for row in related.itertuples(index=False):
        if pd.notna(row.parent_molregno):
            by_parent[int(row.parent_molregno)].add(int(row.molregno))
        if pd.notna(row.active_molregno):
            by_active[int(row.active_molregno)].add(int(row.molregno))
    output: dict[str, set[int]] = defaultdict(set)
    for row in species.itertuples(index=False):
        seed = identifier_to_molregno[clean(row.ligand_species_chembl_id)]
        record = hierarchy_by_seed.get(seed, {})
        parent = int(record["parent_molregno"]) if pd.notna(record.get("parent_molregno")) else seed
        active = int(record["active_molregno"]) if pd.notna(record.get("active_molregno")) else seed
        output[clean(row.ligand_inchikey)].update({seed, parent, active})
        output[clean(row.ligand_inchikey)].update(by_parent.get(parent, set()))
        output[clean(row.ligand_inchikey)].update(by_active.get(active, set()))
    return output


def chembl_pair_audit(pairs: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    connection = sqlite3.connect(f"file:{CHEMBL}?mode=ro", uri=True)
    try:
        mapping = related_molecules(connection, species)
        connection.execute(
            "CREATE TEMP TABLE audited_pairs (pair_id TEXT, target_chembl_id TEXT, molregno INTEGER)"
        )
        rows = []
        for row in pairs.itertuples(index=False):
            for molregno in sorted(mapping[clean(row.ligand_inchikey)]):
                rows.append((clean(row.pairId), clean(row.target_chembl_id), molregno))
        connection.executemany("INSERT INTO audited_pairs VALUES (?, ?, ?)", rows)
        connection.execute("CREATE INDEX temp.idx_audited_pairs ON audited_pairs(target_chembl_id, molregno)")
        activities = pd.read_sql_query(
            """
            SELECT p.pair_id, md.chembl_id AS activity_molecule_chembl_id,
                   a.activity_id, a.assay_id, COALESCE(a.doc_id, ass.doc_id) AS doc_id,
                   ass.assay_type, ass.confidence_score, ass.relationship_type,
                   a.standard_type, a.standard_relation, a.standard_value,
                   a.standard_units, a.pchembl_value, a.activity_comment,
                   a.standard_text_value, a.text_value, a.data_validity_comment,
                   COALESCE(a.potential_duplicate, 0) AS potential_duplicate
            FROM audited_pairs p
            JOIN target_dictionary td ON td.chembl_id = p.target_chembl_id
            JOIN assays ass ON ass.tid = td.tid
            JOIN activities a ON a.assay_id = ass.assay_id AND a.molregno = p.molregno
            JOIN molecule_dictionary md ON md.molregno = a.molregno
            """,
            connection,
        )
        mechanisms = pd.read_sql_query(
            """
            SELECT p.pair_id, md.chembl_id AS mechanism_molecule_chembl_id,
                   dm.action_type, dm.direct_interaction, dm.molecular_mechanism,
                   dm.mechanism_of_action, dm.binding_site_comment
            FROM audited_pairs p
            JOIN target_dictionary td ON td.chembl_id = p.target_chembl_id
            JOIN drug_mechanism dm ON dm.tid = td.tid AND dm.molregno = p.molregno
            JOIN molecule_dictionary md ON md.molregno = dm.molregno
            """,
            connection,
        )
    finally:
        connection.close()

    if not activities.empty:
        text = (
            activities["activity_comment"].fillna("").astype(str)
            + " " + activities["standard_text_value"].fillna("").astype(str)
            + " " + activities["text_value"].fillna("").astype(str)
        ).str.lower()
        activities["explicit_inactive"] = text.str.contains(
            r"inactive|not active|no activity", regex=True
        )
        activities["strict_numeric"] = (
            activities["assay_type"].eq("B")
            & pd.to_numeric(activities["confidence_score"], errors="coerce").ge(9)
            & activities["standard_type"].isin(["Ki", "Kd", "IC50"])
            & activities["standard_relation"].eq("=")
            & pd.to_numeric(activities["pchembl_value"], errors="coerce").notna()
            & activities["data_validity_comment"].fillna("").isin(["", "Manually validated"])
            & pd.to_numeric(activities["potential_duplicate"], errors="coerce").fillna(0).eq(0)
        )
        activities["strict_explicit_inactive"] = (
            activities["assay_type"].eq("B")
            & pd.to_numeric(activities["confidence_score"], errors="coerce").ge(9)
            & activities["explicit_inactive"]
            & activities["data_validity_comment"].fillna("").isin(["", "Manually validated"])
            & pd.to_numeric(activities["potential_duplicate"], errors="coerce").fillna(0).eq(0)
        )

    records = []
    for pair_id in pairs["pairId"]:
        subset = activities[activities["pair_id"].eq(pair_id)].copy()
        mechanism = mechanisms[mechanisms["pair_id"].eq(pair_id)].copy()
        strict = subset[subset.get("strict_numeric", False)].copy() if not subset.empty else subset
        pchembl = pd.to_numeric(strict.get("pchembl_value", pd.Series(dtype=float)), errors="coerce").dropna()
        mean = float(pchembl.mean()) if len(pchembl) else None
        minimum = float(pchembl.min()) if len(pchembl) else None
        maximum = float(pchembl.max()) if len(pchembl) else None
        explicit = bool(subset.get("strict_explicit_inactive", pd.Series(dtype=bool)).any())
        conflict = bool(
            (minimum is not None and maximum is not None and minimum <= 5.0 and maximum >= 6.0)
            or (explicit and maximum is not None and maximum >= 6.0)
        )
        if not mechanism.empty:
            pair_class = "K0_CHEMBL37_MOA_RELATIONSHIP"
        elif conflict:
            pair_class = "K2_STRICT_BINDING_CONFLICTING"
        elif mean is not None and mean >= 6.0:
            pair_class = "K1_STRICT_BINDING_POSITIVE"
        elif mean is not None and mean <= 5.0:
            pair_class = "K4_STRICT_BINDING_NEGATIVE"
        elif mean is not None or explicit:
            pair_class = "K3_STRICT_BINDING_GREY_OR_INACTIVE"
        elif not subset.empty:
            pair_class = "R1_CHEMBL37_NONSTRICT_ACTIVITY_ONLY"
        else:
            pair_class = "N0_NO_CHEMBL37_PAIR_RECORD"
        records.append(
            {
                "pairId": pair_id,
                "chembl37_known_pair_class": pair_class,
                "chembl37_any_activity_rows": int(len(subset)),
                "chembl37_strict_numeric_rows": int(len(strict)),
                "chembl37_strict_assay_count": int(strict["assay_id"].nunique()) if len(strict) else 0,
                "chembl37_strict_document_count": int(strict["doc_id"].nunique()) if len(strict) else 0,
                "chembl37_strict_pchembl_min": minimum,
                "chembl37_strict_pchembl_max": maximum,
                "chembl37_strict_pchembl_mean": mean,
                "chembl37_explicit_inactive": explicit,
                "chembl37_mechanism_rows": int(len(mechanism)),
                "chembl37_mechanism_action_types": join_unique(mechanism.get("action_type", [])),
                "chembl37_mechanism_descriptions": join_unique(
                    mechanism.get("mechanism_of_action", [])
                ),
                "chembl37_activity_molecule_ids": join_unique(
                    subset.get("activity_molecule_chembl_id", [])
                ),
                "chembl37_mechanism_molecule_ids": join_unique(
                    mechanism.get("mechanism_molecule_chembl_id", [])
                ),
                "local_database_novel_pair": pair_class == "N0_NO_CHEMBL37_PAIR_RECORD",
            }
        )
    return pd.DataFrame(records)


def novelty_priority(pair_class: str) -> int:
    order = {
        "N0_NO_CHEMBL37_PAIR_RECORD": 0,
        "K4_STRICT_BINDING_NEGATIVE": 1,
        "R1_CHEMBL37_NONSTRICT_ACTIVITY_ONLY": 2,
        "K3_STRICT_BINDING_GREY_OR_INACTIVE": 3,
        "K2_STRICT_BINDING_CONFLICTING": 4,
        "K1_STRICT_BINDING_POSITIVE": 5,
        "K0_CHEMBL37_MOA_RELATIONSHIP": 6,
    }
    return order[pair_class]


def main() -> None:
    inputs = [V5_RESULTS, V5_TARGETS, MULTISEED, SPECIES, CHEMBL]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    results = pd.read_csv(V5_RESULTS, low_memory=False)
    stable_ids = set(
        results.loc[
            results["increment_multiseed_multiseed_final_decision"].eq(PASS_DECISION), "pairId"
        ]
    )
    multiseed = pd.read_csv(MULTISEED, low_memory=False)
    pairs = multiseed[multiseed["pairId"].isin(stable_ids)].copy()
    if len(pairs) != 15 or pairs["pairId"].duplicated().any():
        raise RuntimeError(f"Expected 15 unique stable V5 pairs, got {len(pairs)}")
    targets = pd.read_csv(V5_TARGETS, low_memory=False)[
        [
            "target_chembl_id", "target_class_zh", "representative_pdb_id",
            "representative_chain_id", "experimental_pocket_evidence_category",
            "positive_compounds", "negative_compounds", "target_calibration_tier",
        ]
    ]
    pairs = pairs.merge(targets, on="target_chembl_id", how="left", validate="many_to_one")
    species_all = pd.read_csv(SPECIES, low_memory=False)
    species = species_all[species_all["ligand_inchikey"].isin(pairs["ligand_inchikey"])].copy()
    if set(species["ligand_inchikey"]) != set(pairs["ligand_inchikey"]):
        raise RuntimeError("One or more stable-pair ligand species are not mapped")
    known = chembl_pair_audit(pairs, species)
    pairs = pairs.merge(known, on="pairId", how="left", validate="one_to_one")

    rows = []
    for row in pairs.itertuples(index=False):
        target = clean(row.target_chembl_id)
        protocol_path = ROOT / "outputs/strict_receptor_protocol_338_v1/targets" / target / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        yaml_path = Path(clean(row.yamlPath)).resolve()
        cif_path = Path(clean(row.boltzCifPath)).resolve()
        receptor = ROOT / "outputs/strict_receptor_protocol_338_v1/targets" / target / "receptor_protein_prepared.pdb"
        if not yaml_path.is_file() or not cif_path.is_file() or not receptor.is_file():
            raise FileNotFoundError(f"Missing structural input for {row.pairId}")
        context = protocol.get("docking_context_included_items", []) or []
        context_names = sorted(
            {clean(item.get("residue_name", "")).upper() for item in context if item.get("residue_name")}
        )
        features = ligand_features(clean(row.ligand_smiles))
        warhead = features["reactive_warhead_class"]
        if warhead == "BORON_REVERSIBLE_COVALENT_WARHEAD":
            route = "S2_BORON_COVALENT_PARAMETERIZATION_REQUIRED"
            status = "BLOCKED_GENERIC_MD"
            reason = "Boronic-acid pharmacophore requires bespoke covalent/QM-aware parameterization."
        elif warhead == "ACRYLAMIDE_ELECTROPHILE_WARHEAD":
            route = "S9_TARGET_SITE_COVALENT_REVIEW_REQUIRED"
            status = "BLOCKED_GENERIC_MD"
            reason = "Electrophile requires target-pocket nucleophile and covalent-state review before MD."
        elif context_names or features["specialized_ligand_elements"]:
            route = "S8_SPECIALIZED_CONTEXT_OR_ELEMENT_PARAMETERIZATION"
            status = "BLOCKED_GENERIC_MD"
            reason = "Nonstandard context or ligand element is outside the frozen generic protocol."
        else:
            route = "A_STANDARD_NONCOVALENT_AFTER_POSE_GATE"
            status = "PROVISIONAL_STANDARD"
            reason = "Experimental pocket and standard ligand chemistry; pose mapping gate remains mandatory."
        rows.append(
            {
                "incremental_queue_rank": int(row.incremental_queue_rank),
                "pair_id": clean(row.pairId),
                "fda_generic_name": clean(row.drug_names),
                "ligand_inchikey": clean(row.ligand_inchikey),
                "primary_gene": clean(row.gene_symbol),
                "target_chembl_id": target,
                "target_class_zh": clean(row.target_class_zh),
                "novelty_lane": clean(row.novelty_lane),
                "max_tanimoto_to_target_measured_positive": row.max_tanimoto_to_target_measured_positive,
                "dta_priority_score_384": row.dta_priority_score_384,
                "boltz_evidence_tier": clean(row.boltz_evidence_tier),
                "boltz_affinity_probability_binary": row.boltzAffinityProbabilityBinary,
                "boltz_ligand_iptm": row.boltzLigandIptm,
                "boltz_support_count_3seed": int(row.boltz_support_count_3seed),
                "pose_max_centroid_distance_A": row.pose_max_centroid_distance_A,
                "pose_min_interface_jaccard": row.pose_min_interface_jaccard,
                "chembl37_known_pair_class": clean(row.chembl37_known_pair_class),
                "chembl37_any_activity_rows": int(row.chembl37_any_activity_rows),
                "chembl37_strict_numeric_rows": int(row.chembl37_strict_numeric_rows),
                "chembl37_strict_assay_count": int(row.chembl37_strict_assay_count),
                "chembl37_strict_document_count": int(row.chembl37_strict_document_count),
                "chembl37_strict_pchembl_min": row.chembl37_strict_pchembl_min,
                "chembl37_strict_pchembl_max": row.chembl37_strict_pchembl_max,
                "chembl37_strict_pchembl_mean": row.chembl37_strict_pchembl_mean,
                "chembl37_explicit_inactive": bool(row.chembl37_explicit_inactive),
                "chembl37_mechanism_rows": int(row.chembl37_mechanism_rows),
                "chembl37_mechanism_action_types": clean(row.chembl37_mechanism_action_types),
                "chembl37_mechanism_descriptions": clean(row.chembl37_mechanism_descriptions),
                "chembl37_activity_molecule_ids": clean(row.chembl37_activity_molecule_ids),
                "chembl37_mechanism_molecule_ids": clean(row.chembl37_mechanism_molecule_ids),
                "local_database_novel_pair": bool(row.local_database_novel_pair),
                "pdb_id": clean(protocol.get("pdb_id")),
                "experimental_chain": clean(protocol.get("target_chain_id")),
                "reference_ligand_id": clean(protocol.get("reference_ligand_id")),
                "context_heterogen_count": len(context),
                "required_context_residues": ";".join(context_names),
                "reactive_warhead_class": warhead,
                **features,
                "md_parameterization_route": route,
                "md_pre_pose_status": status,
                "md_pre_pose_reason": reason,
                "active_moiety_smiles": clean(row.ligand_smiles),
                "pdb_path": str(receptor.resolve()),
                "boltz_cif_path_refined": str(cif_path),
                "boltz_yaml_path": str(yaml_path),
                "boltz_pocket_residue_ids": boltz_pocket_residue_ids(yaml_path),
                "canonical_pocket_residue_ids": clean(protocol.get("canonical_pocket_residues")),
                "alignment_residue_id_source": "BOLTZ_YAML_CONSTRAINT_1_BASED_SEQUENCE_POSITION",
                "protocol_json": str(protocol_path.resolve()),
                "source_pair_evidence_layer": "L2_BOLTZ_REPRODUCED",
                "source_pair_evidence_layer_zh": "Boltz多seed和条件pose稳定",
                "md_interpretation": "conditional-pose stress test; not affinity or binding validation",
            }
        )
    audit = pd.DataFrame(rows)
    audit["novelty_priority"] = audit["chembl37_known_pair_class"].map(novelty_priority)
    tier_order = {"BOLTZ_STRUCTURE_A": 0, "BOLTZ_STRUCTURE_B": 1, "BOLTZ_STRUCTURE_C": 2}
    audit["boltz_tier_priority"] = audit["boltz_evidence_tier"].map(tier_order)
    audit = audit.sort_values(
        [
            "primary_gene", "novelty_priority", "boltz_tier_priority",
            "boltz_affinity_probability_binary", "incremental_queue_rank",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    audit["target_md_priority"] = audit.groupby("target_chembl_id").cumcount() + 1
    standard = audit["md_pre_pose_status"].eq("PROVISIONAL_STANDARD")
    within_cap = audit["target_md_priority"].le(2)
    audit["md_authorization_pre_pose"] = "NOT_AUTHORIZED"
    audit.loc[standard & within_cap, "md_authorization_pre_pose"] = "AUTHORIZED_PENDING_POSE_GATE"
    audit.loc[standard & ~within_cap, "md_authorization_pre_pose"] = "HELD_BY_MAX2_PER_TARGET_CAP"
    audit.loc[~standard, "md_authorization_pre_pose"] = "BLOCKED_SPECIALIZED_PARAMETERIZATION"
    audit["md_screen_rank"] = pd.NA
    authorized_index = audit.index[audit["md_authorization_pre_pose"].eq("AUTHORIZED_PENDING_POSE_GATE")]
    audit.loc[authorized_index, "md_screen_rank"] = range(1, len(authorized_index) + 1)
    audit["final384_rank"] = audit["incremental_queue_rank"]
    audit["md_context_tier"] = audit["md_parameterization_route"]
    audit = audit.sort_values("incremental_queue_rank", kind="mergesort").reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "V5_L2_PAIR_KNOWN_RELATION_AND_CONTEXT_AUDIT_V1.csv"
    authorized_path = OUT / "V5_L2_MD_AUTHORIZED_PENDING_POSE_GATE_V1.csv"
    blocked_path = OUT / "V5_L2_MD_SPECIALIZED_OR_CAPPED_V1.csv"
    audit.to_csv(audit_path, index=False)
    authorized = audit[audit["md_authorization_pre_pose"].eq("AUTHORIZED_PENDING_POSE_GATE")].copy()
    authorized["md_screen_rank"] = authorized["md_screen_rank"].astype(int)
    authorized.to_csv(authorized_path, index=False)
    audit[~audit["md_authorization_pre_pose"].eq("AUTHORIZED_PENDING_POSE_GATE")].to_csv(
        blocked_path, index=False
    )
    checks = {
        "exactly_15_stable_pairs": len(audit) == 15,
        "pair_ids_unique": not audit["pair_id"].duplicated().any(),
        "all_structural_inputs_present": all(
            Path(path).is_file()
            for column in ["pdb_path", "boltz_cif_path_refined", "boltz_yaml_path", "protocol_json"]
            for path in audit[column]
        ),
        "max_two_authorized_per_target": bool(
            authorized.groupby("target_chembl_id").size().le(2).all()
        ),
        "known_pair_classes_complete": audit["chembl37_known_pair_class"].notna().all(),
        "generic_md_excludes_reactive_warheads": not authorized[
            "reactive_warhead_class"
        ].ne("NONE_DETECTED_BY_FROZEN_RULES").any(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "stable_pair_count": int(len(audit)),
        "unique_targets": int(audit["target_chembl_id"].nunique()),
        "known_pair_class_counts": {
            str(key): int(value) for key, value in audit["chembl37_known_pair_class"].value_counts().items()
        },
        "local_database_novel_pairs": int(audit["local_database_novel_pair"].sum()),
        "authorized_pending_pose_gate": int(len(authorized)),
        "blocked_specialized_parameterization": int(
            audit["md_authorization_pre_pose"].eq("BLOCKED_SPECIALIZED_PARAMETERIZATION").sum()
        ),
        "held_by_target_cap": int(
            audit["md_authorization_pre_pose"].eq("HELD_BY_MAX2_PER_TARGET_CAP").sum()
        ),
        "md_scope": (
            "Single-seed capped 5 ns pose-retention stress test after a frozen mapping gate; "
            "not affinity, binding, efficacy, or target validation."
        ),
        "novelty_scope": (
            "N0 means no exact related-active-species activity or mechanism row for the same "
            "ChEMBL 37 target in the local database; it does not establish literature novelty."
        ),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs[:-1]},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [audit_path, authorized_path, blocked_path]
        },
    }
    summary_path = OUT / "V5_L2_PAIR_MD_AUDIT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
