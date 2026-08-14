#!/usr/bin/env python3
"""Prepare the exact recovered 720 x 46 DTA program and 44 predicted pockets."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parents[1]
RECOVERED_DIR = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1"
RECOVERED = RECOVERED_DIR / "RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
LIGAND_DIR = ROOT / "outputs/strict_affinity_main_queue_2005_2026_v2"
LIGANDS = LIGAND_DIR / "STRICT_UNIQUE_STANDARD_MODEL_LIGAND_STRUCTURES.csv"
STRICT_DTA_INPUTS = ROOT / "outputs/strict_dta_720x338_v1/inputs"
P2RANK_ATLAS = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/P2RANK_ALL_PREDICTED_POCKETS_875.csv.gz"
CONSENSUS = ROOT / "outputs/no_experimental_pocket_prediction_ch37_v1/NO_EXPERIMENTAL_POCKET_CONSENSUS_TARGETS_151_V1.csv"
FPOCKET_CANDIDATES = ROOT / "outputs/no_experimental_pocket_prediction_ch37_v1/FPOCKET_POCKET_CANDIDATES_NO_EXPERIMENTAL_146_V1.csv.gz"
OUT = ROOT / "outputs/recovered_dta_720x46_v1"
INPUTS = OUT / "inputs"
EXPECTED_LIGANDS = 720
EXPECTED_TARGETS = 46
EXPECTED_EXACT_POCKETS = 44
EXPECTED_PAIRS = EXPECTED_LIGANDS * EXPECTED_TARGETS
EXCLUSION = "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_lmdb(records: list[dict[str, Any]], path: Path) -> None:
    path.unlink(missing_ok=True)
    environment = lmdb.open(
        str(path), subdir=False, readonly=False, lock=True, readahead=False,
        meminit=False, map_size=max(1024**3, len(records) * 8 * 1024**2),
    )
    with environment.begin(write=True) as transaction:
        for index, record in enumerate(records):
            transaction.put(str(index).encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
    environment.sync()
    environment.close()


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        destination.unlink()
    destination.symlink_to(source.resolve())


def element_for_atom(atom: Any) -> str:
    element = str(getattr(atom, "element", "")).strip().upper()
    if element:
        return element
    name = str(atom.get_name()).strip().upper()
    return "CL" if name.startswith("CL") else name[:1]


def p2rank_record(row: pd.Series, positions: set[int], matched_rank: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pdb_path = Path(row["af_pdb_path"])
    structure = PDBParser(QUIET=True).get_structure(row["target_chembl_id"], str(pdb_path))
    atoms: list[str] = []
    coordinates: list[np.ndarray] = []
    residue_ids: list[str] = []
    b_factors: list[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if int(residue.id[1]) not in positions:
                    continue
                residue_ids.append(f"{chain.id}:{residue.resname}:{residue.id[1]}")
                for atom in residue.get_atoms():
                    element = element_for_atom(atom)
                    if not element or element == "H":
                        continue
                    atoms.append(element)
                    coordinates.append(np.asarray(atom.coord, dtype=np.float32))
                    b_factors.append(float(atom.bfactor))
        break
    audit = {
        "pocket_source_method": f"P2RANK_MATCHED_RANK_{matched_rank}_DUAL_METHOD_CONFIRMED",
        "pocket_source_file": str(pdb_path),
        "pocket_definition": f"all heavy atoms of AlphaFold residues assigned to P2Rank rank-{matched_rank} pocket; this exact site is independently supported by fpocket",
        "pocket_residue_count": len(set(residue_ids)),
        "pocket_atom_count": len(atoms),
        "pocket_residue_ids": ";".join(residue_ids),
        "pocket_mean_plddt": float(np.mean(b_factors)) if b_factors else np.nan,
    }
    return {
        "pocket": row["target_chembl_id"],
        "pocket_atoms": atoms,
        "pocket_coordinates": coordinates,
    }, audit


def fpocket_record(row: pd.Series, atom_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    structure = PDBParser(QUIET=True).get_structure(row["target_chembl_id"], str(atom_file))
    atoms: list[str] = []
    coordinates: list[np.ndarray] = []
    residue_ids: list[str] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_ids.append(f"{chain.id}:{residue.resname}:{residue.id[1]}")
                for atom in residue.get_atoms():
                    element = element_for_atom(atom)
                    if not element or element == "H":
                        continue
                    atoms.append(element)
                    coordinates.append(np.asarray(atom.coord, dtype=np.float32))
        break
    audit = {
        "pocket_source_method": "FPOCKET_RANK1_GEOMETRIC_RESCUE",
        "pocket_source_file": str(atom_file),
        "pocket_definition": "heavy protein atoms contacting fpocket rank-1 alpha spheres on exact-sequence AlphaFold model",
        "pocket_residue_count": len(set(residue_ids)),
        "pocket_atom_count": len(atoms),
        "pocket_residue_ids": ";".join(sorted(set(residue_ids))),
        "pocket_mean_plddt": np.nan,
    }
    return {
        "pocket": row["target_chembl_id"],
        "pocket_atoms": atoms,
        "pocket_coordinates": coordinates,
    }, audit


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    ligands = pd.read_csv(LIGANDS, dtype=str).fillna("").sort_values("ligand_inchikey", kind="mergesort").reset_index(drop=True)
    recovered = pd.read_csv(RECOVERED, dtype=str).fillna("")
    universe = pd.read_csv(UNIVERSE, dtype=str).fillna("")
    targets = recovered.merge(
        universe[["target_chembl_id", "sequence", "sequence_sha256"]].rename(columns={"sequence_sha256": "universe_sequence_sha256"}),
        on="target_chembl_id", how="left", validate="one_to_one",
    )
    if "best_method_match_p2rank_rank" not in targets.columns:
        consensus = pd.read_csv(CONSENSUS, dtype=str).fillna("")[
            ["target_chembl_id", "best_method_match_p2rank_rank"]
        ]
        targets = targets.merge(consensus, on="target_chembl_id", how="left", validate="one_to_one")
    targets = targets.sort_values("target_chembl_id", kind="mergesort").reset_index(drop=True)

    scope_checks = {
        "exactly_46_targets": len(targets) == EXPECTED_TARGETS and targets["target_chembl_id"].nunique() == EXPECTED_TARGETS,
        "only_no_experimental_pocket_exclusion": targets["first_exclusion_reason"].eq(EXCLUSION).all(),
        "all_earlier_hard_gates_pass": targets[["passes_non_gpcr", "passes_chembl_small_molecule_moa", "passes_supported_target_class"]].eq("True").all().all(),
        "all_experimental_pocket_gate_fail": targets["passes_any_experimental_pocket"].eq("False").all(),
        "sequence_hashes_match": targets["sequence_sha256"].eq(targets["universe_sequence_sha256"]).all(),
        "exactly_720_ligands": len(ligands) == EXPECTED_LIGANDS and ligands["ligand_inchikey"].nunique() == EXPECTED_LIGANDS,
    }
    scope_checks = {key: bool(value) for key, value in scope_checks.items()}
    if not all(scope_checks.values()):
        raise ValueError(f"Recovered-scope contract failed: {scope_checks}")

    targets.to_csv(INPUTS / "RECOVERED_TARGETS_FROZEN_46_V1.csv", index=False)
    conplex_input = INPUTS / "CONPLEX_720_X_46_INPUT.tsv"
    with conplex_input.open("w", encoding="utf-8") as handle:
        for ligand in ligands.itertuples(index=False):
            for target in targets.itertuples(index=False):
                handle.write(f"{target.target_chembl_id}\t{ligand.ligand_inchikey}\t{target.sequence}\t{ligand.ligand_smiles}\n")

    p2rank = pd.read_csv(P2RANK_ATLAS, dtype=str).fillna("")
    p2rank["_rank"] = pd.to_numeric(p2rank["p2rank_rank"], errors="raise").astype(int)
    p2rank = p2rank.set_index(["uniprot_accession", "_rank"])
    fpocket = pd.read_csv(FPOCKET_CANDIDATES, dtype=str).fillna("")
    fpocket = fpocket[pd.to_numeric(fpocket["fpocket_rank"], errors="coerce").eq(1)].set_index("target_chembl_id")

    pocket_records: list[dict[str, Any]] = []
    pocket_audits: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        audit: dict[str, Any] = {
            "target_chembl_id": row["target_chembl_id"],
            "gene_symbol": row["gene_symbol"],
            "uniprot_accession": row["uniprot_accession"],
            "computed_pocket_evidence": row["computed_pocket_evidence"],
            "structure_strategy": row["structure_strategy"],
            "status": "EXCLUDED_LOW_CONFIDENCE_FRAGMENT_MODEL",
            "exclusion_reason": "DrugCLIP formal pocket scoring requires an exact-sequence full structure; fragment pocket hypotheses remain exploratory only",
            "lmdb_index": "",
        }
        if row["af_exact_sequence_model"] != "True":
            pocket_audits.append(audit)
            continue
        if row["computed_pocket_evidence"] == "P1_P2RANK_FPOCKET_SAME_SITE":
            matched_rank = int(float(row["best_method_match_p2rank_rank"]))
            matched = p2rank.loc[(row["uniprot_accession"], matched_rank)]
            if isinstance(matched, pd.DataFrame):
                raise ValueError(f"Duplicate P2Rank rank-{matched_rank} records for {row['uniprot_accession']}")
            positions = {int(value) for value in matched["p2rank_residue_positions"].split(";") if value}
            record, method_audit = p2rank_record(row, positions, matched_rank)
        elif row["computed_pocket_evidence"] == "P2_FPOCKET_GEOMETRIC_RESCUE":
            rank1 = fpocket.loc[row["target_chembl_id"]]
            if isinstance(rank1, pd.DataFrame):
                raise ValueError(f"Duplicate fpocket rank-1 records for {row['target_chembl_id']}")
            record, method_audit = fpocket_record(row, Path(rank1["fpocket_atom_file"]))
        else:
            raise ValueError(f"Unexpected exact-structure evidence for {row['target_chembl_id']}: {row['computed_pocket_evidence']}")
        audit.update(method_audit)
        if int(audit["pocket_atom_count"]) < 20:
            audit.update(status="FAILED", exclusion_reason=f"POCKET_TOO_SMALL_{audit['pocket_atom_count']}_ATOMS")
        else:
            audit.update(status="OK", exclusion_reason="", lmdb_index=len(pocket_records))
            pocket_records.append(record)
        pocket_audits.append(audit)

    pocket_lmdb = INPUTS / "recovered44_predicted_pockets.lmdb"
    write_lmdb(pocket_records, pocket_lmdb)
    pocket_manifest = INPUTS / "RECOVERED44_DRUGCLIP_POCKET_MANIFEST.csv"
    pd.DataFrame(pocket_audits).to_csv(pocket_manifest, index=False)

    ligand_lmdb_source = STRICT_DTA_INPUTS / "strict720_ligands.lmdb"
    ligand_manifest_source = STRICT_DTA_INPUTS / "STRICT720_DRUGCLIP_LIGAND_MANIFEST.csv"
    ligand_lmdb = INPUTS / "strict720_ligands.lmdb"
    ligand_manifest = INPUTS / "STRICT720_DRUGCLIP_LIGAND_MANIFEST.csv"
    safe_symlink(ligand_lmdb_source, ligand_lmdb)
    safe_symlink(ligand_manifest_source, ligand_manifest)

    summary = {
        "created_utc": now(),
        "status": "PASS" if len(pocket_records) == EXPECTED_EXACT_POCKETS else "INCOMPLETE",
        "scope_policy": "Only the 46 targets whose first exclusion reason is lack of an eligible experimental pocket are recovered; no other hard-gate exclusions are restored.",
        "scope_checks": scope_checks,
        "ligands": len(ligands),
        "targets": len(targets),
        "conplex_pairs": EXPECTED_PAIRS,
        "drugclip_predicted_pockets": len(pocket_records),
        "drugclip_target_ligand_pairs": len(pocket_records) * len(ligands),
        "fragment_targets_excluded_from_formal_drugclip": int((targets["af_exact_sequence_model"] != "True").sum()),
        "pocket_methods": {
            str(key): int(value) for key, value in
            pd.DataFrame(pocket_audits)["pocket_source_method"].fillna("FRAGMENT_EXCLUDED").value_counts().items()
        },
        "interpretation": {
            "ConPLEx": "sequence-ligand ranking evidence for all 46 recovered targets; not Kd or a calibrated binder probability",
            "DrugCLIP": "predicted-pocket ligand retrieval evidence for 44 exact-sequence AlphaFold structures; strictly distinguished from experimental holo-pocket evidence",
            "fragment_policy": "DIO1 and RYR1 remain sequence-only in formal DTA because low-confidence fragment models are not reliable enough for precise 3D retrieval ranking",
        },
        "sha256": {
            "recovered_scope": sha256(RECOVERED),
            "ligands_source": sha256(LIGANDS),
            "conplex_input": sha256(conplex_input),
            "predicted_pocket_lmdb": sha256(pocket_lmdb),
            "reused_ligand_lmdb": sha256(ligand_lmdb_source),
        },
    }
    (OUT / "RECOVERED_DTA_INPUT_PREPARATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Recovered DTA input contract is incomplete")


if __name__ == "__main__":
    main()
