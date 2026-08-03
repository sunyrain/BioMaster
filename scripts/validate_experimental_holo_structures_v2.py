#!/usr/bin/env python3
"""Validate experimental target-chain/ligand contacts and prepare docking receptors."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from Bio import Align
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, is_nucleic, is_aa
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/affinity_first_remote_discovery_v1"
DEFAULT_ENTRIES = BASE / "experimental_structure_atlas_v1/TARGET_EXPERIMENTAL_ENTRY_CANDIDATES_V1.csv"
DEFAULT_LIGANDS = BASE / "experimental_structure_atlas_v1/EXPERIMENTAL_ENTRY_LIGANDS_V1.csv"
DEFAULT_READINESS = BASE / "target_docking_calibration_v1/TARGET_DOCKING_CALIBRATION_READINESS_463_V1.csv"
DEFAULT_OUT = BASE / "experimental_holo_validation_v2"
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
METALS = {"CA", "CD", "CO", "CU", "FE", "K", "MG", "MN", "NA", "NI", "ZN"}
COFACTORS = {
    "ADP", "AMP", "ATP", "COA", "FAD", "FMN", "GDP", "GTP", "HEM", "NAD", "NAP",
    "NDP", "PLP", "PMP", "SAM", "SAH", "UDP",
}


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def download_cif(pdb_id: str, cache_dir: Path, max_bytes: int, retries: int = 4) -> dict[str, Any]:
    target = cache_dir / f"{pdb_id.lower()}.cif.gz"
    if target.exists() and target.stat().st_size > 100:
        return {"pdb_id": pdb_id, "status": "cached", "path": str(target), "bytes": target.stat().st_size}
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(RCSB_DOWNLOAD.format(pdb_id=pdb_id.upper()), timeout=90, stream=True) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0) or 0)
                if expected > max_bytes:
                    return {"pdb_id": pdb_id, "status": "too_large", "path": "", "bytes": expected}
                total = 0
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise RuntimeError("download exceeded per-entry size limit")
                        handle.write(chunk)
            partial.replace(target)
            return {"pdb_id": pdb_id, "status": "downloaded", "path": str(target), "bytes": total}
        except Exception as exc:
            partial.unlink(missing_ok=True)
            if attempt + 1 == retries:
                return {"pdb_id": pdb_id, "status": f"failed:{type(exc).__name__}", "path": "", "bytes": 0}
            time.sleep(1.5 * (attempt + 1))
    return {"pdb_id": pdb_id, "status": "failed", "path": "", "bytes": 0}


def atom_coords(residue: Any) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        element = clean(getattr(atom, "element", "")).upper()
        if element not in {"H", "D"}:
            coords.append(np.asarray(atom.coord, dtype=float))
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)


def protein_chain_data(chain: Any) -> tuple[list[Any], str, np.ndarray]:
    residues = []
    letters = []
    heavy = []
    for residue in chain.get_residues():
        if not is_aa(residue, standard=False):
            continue
        residues.append(residue)
        try:
            letters.append(seq1(residue.get_resname(), custom_map={"MSE": "M"}))
        except Exception:
            letters.append("X")
        coords = atom_coords(residue)
        if len(coords):
            heavy.extend(coords)
    return residues, "".join(letters), np.asarray(heavy, dtype=float) if heavy else np.empty((0, 3))


def ca_atoms(residues: list[Any]) -> list[Any | None]:
    return [residue["CA"] if "CA" in residue else None for residue in residues]


def align_experimental_to_af(
    experimental_residues: list[Any], experimental_sequence: str, af_residues: list[Any], af_sequence: str
) -> tuple[np.ndarray | None, np.ndarray | None, int, float, float]:
    if len(experimental_sequence) < 15 or len(af_sequence) < 15:
        return None, None, 0, math.nan, math.nan
    aligner = Align.PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -7.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(af_sequence, experimental_sequence)[0]
    af_ca = ca_atoms(af_residues)
    exp_ca = ca_atoms(experimental_residues)
    fixed = []
    moving = []
    identical = 0
    aligned_count = 0
    for (af_start, af_end), (exp_start, exp_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        span = min(af_end - af_start, exp_end - exp_start)
        for offset in range(span):
            ai = int(af_start + offset)
            ei = int(exp_start + offset)
            aligned_count += 1
            if af_sequence[ai] == experimental_sequence[ei]:
                identical += 1
            if af_ca[ai] is not None and exp_ca[ei] is not None and af_sequence[ai] == experimental_sequence[ei]:
                fixed.append(af_ca[ai])
                moving.append(exp_ca[ei])
    if len(fixed) < 15:
        return None, None, len(fixed), math.nan, identical / max(aligned_count, 1)
    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    rotation, translation = superimposer.rotran
    return rotation, translation, len(fixed), float(superimposer.rms), identical / max(aligned_count, 1)


def transform_point(point: np.ndarray, rotation: np.ndarray | None, translation: np.ndarray | None) -> np.ndarray | None:
    if rotation is None or translation is None:
        return None
    return np.dot(point, rotation) + translation


def load_af_chains(readiness: pd.DataFrame) -> dict[str, dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    result: dict[str, dict[str, Any]] = {}
    for _, row in readiness.iterrows():
        sequence_key = clean(row["sequence_key"])
        path = Path(clean(row.get("pdb_path")))
        if not path.exists():
            continue
        try:
            structure = parser.get_structure(sequence_key, str(path))
            chains = list(structure[0].get_chains())
            if not chains:
                continue
            residues, sequence, _ = protein_chain_data(max(chains, key=lambda c: sum(is_aa(r, False) for r in c)))
            result[sequence_key] = {"residues": residues, "sequence": sequence}
        except Exception:
            continue
    return result


def locate_chain(model: Any, requested: str) -> Any | None:
    chain_by_id = {clean(chain.id): chain for chain in model.get_chains()}
    if requested in chain_by_id:
        return chain_by_id[requested]
    if requested.upper() in chain_by_id:
        return chain_by_id[requested.upper()]
    return None


def ligand_instances(model: Any, ligand_rows: pd.DataFrame) -> list[dict[str, Any]]:
    wanted = {clean(value).upper() for value in ligand_rows["chem_comp_id"]}
    metadata = {}
    for _, row in ligand_rows.iterrows():
        key = (
            clean(row.get("chem_comp_id")).upper(),
            clean(row.get("chain_id")),
            int(finite_float(row.get("author_residue_number"), -999999)),
        )
        metadata[key] = row
    instances = []
    for chain in model.get_chains():
        for residue in chain.get_residues():
            code = clean(residue.get_resname()).upper()
            if code not in wanted or is_aa(residue, standard=False) or is_nucleic(residue, standard=False):
                continue
            coords = atom_coords(residue)
            if len(coords) < 3:
                continue
            residue_number = int(residue.id[1])
            row = metadata.get((code, clean(chain.id), residue_number))
            if row is None:
                matching = ligand_rows[ligand_rows["chem_comp_id"].astype(str).str.upper().eq(code)]
                row = matching.iloc[0] if len(matching) else pd.Series(dtype=object)
            instances.append(
                {
                    "chem_comp_id": code,
                    "chem_comp_name": clean(row.get("chem_comp_name")),
                    "weight": finite_float(row.get("weight")),
                    "ligand_chain_id": clean(chain.id),
                    "ligand_residue_number": residue_number,
                    "ligand_insertion_code": clean(residue.id[2]),
                    "coords": coords,
                    "heavy_atoms": len(coords),
                }
            )
    return instances


def evaluate_target_entry(
    target_row: pd.Series,
    model: Any,
    ligand_rows: pd.DataFrame,
    af_data: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    target_chain_id = clean(target_row.get("chain_id"))
    target_chain = locate_chain(model, target_chain_id)
    base = {
        "sequence_key": clean(target_row.get("sequence_key")),
        "primary_gene": clean(target_row.get("primary_gene")),
        "pdb_id": clean(target_row.get("pdb_id")).lower(),
        "target_chain_id": target_chain_id,
        "coverage": finite_float(target_row.get("coverage")),
        "resolution": finite_float(target_row.get("resolution")),
        "experimental_method": clean(target_row.get("experimental_method")),
    }
    if target_chain is None:
        return [{**base, "validation_status": "target_chain_not_found"}]
    residues, sequence, target_coords = protein_chain_data(target_chain)
    if len(target_coords) < 30:
        return [{**base, "validation_status": "target_chain_too_small"}]
    rotation = translation = None
    alignment_atoms = 0
    alignment_rmsd = alignment_identity = math.nan
    if af_data:
        rotation, translation, alignment_atoms, alignment_rmsd, alignment_identity = align_experimental_to_af(
            residues, sequence, af_data["residues"], af_data["sequence"]
        )
    target_tree = cKDTree(target_coords)
    all_chain_coords = {}
    for chain in model.get_chains():
        _, _, coords = protein_chain_data(chain)
        if len(coords):
            all_chain_coords[clean(chain.id)] = coords
    rows = []
    for ligand in ligand_instances(model, ligand_rows):
        ligand_coords = ligand.pop("coords")
        distances, _ = target_tree.query(ligand_coords, k=1)
        min_distance = float(np.min(distances))
        contact_residues = 0
        for residue in residues:
            coords = atom_coords(residue)
            if len(coords) and float(np.min(cKDTree(coords).query(ligand_coords, k=1)[0])) <= 5.0:
                contact_residues += 1
        interface_chains = []
        for chain_id, coords in all_chain_coords.items():
            if float(np.min(cKDTree(coords).query(ligand_coords, k=1)[0])) <= 5.0:
                interface_chains.append(chain_id)
        center = ligand_coords.mean(axis=0)
        transformed = transform_point(center, rotation, translation)
        af_center = np.asarray(
            [
                finite_float(target_row.get("top_pocket_center_x")),
                finite_float(target_row.get("top_pocket_center_y")),
                finite_float(target_row.get("top_pocket_center_z")),
            ]
        )
        pocket_distance = (
            float(np.linalg.norm(transformed - af_center))
            if transformed is not None and np.all(np.isfinite(af_center))
            else math.nan
        )
        if math.isfinite(pocket_distance) and pocket_distance <= 8.0:
            pocket_status = "confirmed_af_pocket_overlap"
        elif math.isfinite(pocket_distance) and pocket_distance <= 12.0:
            pocket_status = "near_af_pocket"
        elif math.isfinite(pocket_distance):
            pocket_status = "divergent_from_af_top_pocket"
        else:
            pocket_status = "af_alignment_unavailable"
        contact_valid = min_distance <= 5.0 and contact_residues >= 3 and ligand["heavy_atoms"] >= 6
        rows.append(
            {
                **base,
                **ligand,
                "ligand_center_x": float(center[0]),
                "ligand_center_y": float(center[1]),
                "ligand_center_z": float(center[2]),
                "ligand_extent_x": float(np.ptp(ligand_coords[:, 0])),
                "ligand_extent_y": float(np.ptp(ligand_coords[:, 1])),
                "ligand_extent_z": float(np.ptp(ligand_coords[:, 2])),
                "min_ligand_target_distance": min_distance,
                "target_contact_residues_5a": contact_residues,
                "contacting_protein_chains_5a": ";".join(sorted(interface_chains)),
                "interface_ligand": len(interface_chains) > 1,
                "possible_covalent_contact": min_distance < 1.75,
                "reference_ligand_role": "cofactor_or_nucleotide" if ligand["chem_comp_id"] in COFACTORS else "small_molecule",
                "alignment_ca_atoms": alignment_atoms,
                "alignment_identity": alignment_identity,
                "alignment_rmsd": alignment_rmsd,
                "ligand_center_distance_to_af_top_pocket": pocket_distance,
                "project_pocket_status": pocket_status,
                "validation_status": "chain_contact_validated" if contact_valid else "no_valid_target_chain_contact",
            }
        )
    return rows or [{**base, "validation_status": "candidate_ligand_not_found_in_coordinates"}]


def write_receptor_pdb(model: Any, target_chains: set[str], ligand_center: np.ndarray, output: Path) -> int:
    chain_map = {}
    available = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    lines = []
    serial = 1
    for chain in model.get_chains():
        chain_id = clean(chain.id)
        if chain_id not in target_chains:
            continue
        if chain_id not in chain_map:
            chain_map[chain_id] = available[len(chain_map) % len(available)]
        out_chain = chain_map[chain_id]
        for residue in chain.get_residues():
            if not is_aa(residue, standard=False):
                continue
            resname = clean(residue.get_resname())[:3].upper() or "UNK"
            resseq = int(residue.id[1])
            icode = clean(residue.id[2])[:1] or " "
            for atom in residue.get_atoms():
                if atom.is_disordered() and clean(atom.get_altloc()) not in {"", "A"}:
                    continue
                element = clean(getattr(atom, "element", "")).upper()[:2]
                if element in {"H", "D"}:
                    continue
                x, y, z = atom.coord
                name = atom.get_name()[:4]
                lines.append(
                    f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {out_chain}{resseq:4d}{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{float(atom.occupancy or 1.0):6.2f}{float(atom.bfactor or 0.0):6.2f}          {element:>2s}\n"
                )
                serial += 1
    # Retain pocket-proximal metal ions because they can be essential for enzyme docking.
    for chain in model.get_chains():
        for residue in chain.get_residues():
            code = clean(residue.get_resname()).upper()
            if code not in METALS:
                continue
            coords = atom_coords(residue)
            if not len(coords) or float(np.min(np.linalg.norm(coords - ligand_center, axis=1))) > 6.0:
                continue
            for atom in residue.get_atoms():
                x, y, z = atom.coord
                element = clean(getattr(atom, "element", code)).upper()[:2]
                lines.append(
                    f"HETATM{serial:5d} {atom.get_name()[:4]:>4s} {code[:3]:>3s} Z{int(residue.id[1]):4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{float(atom.occupancy or 1.0):6.2f}{float(atom.bfactor or 0.0):6.2f}          {element:>2s}\n"
                )
                serial += 1
    lines.append("END\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="ascii")
    return serial - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", default=str(DEFAULT_ENTRIES))
    parser.add_argument("--ligands", default=str(DEFAULT_LIGANDS))
    parser.add_argument("--readiness", default=str(DEFAULT_READINESS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max-entries-per-target", type=int, default=5)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-entry-mb", type=int, default=250)
    parser.add_argument("--limit-targets", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cache_dir = out_dir / "mmcif_cache"
    receptor_dir = out_dir / "validated_receptors"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    receptor_dir.mkdir(parents=True, exist_ok=True)

    entries = pd.read_csv(args.entries, low_memory=False).fillna("")
    ligands = pd.read_csv(args.ligands, low_memory=False).fillna("")
    readiness = pd.read_csv(args.readiness, low_memory=False).fillna("")
    if args.limit_targets > 0:
        keep_keys = set(readiness["sequence_key"].astype(str).head(args.limit_targets))
        readiness = readiness[readiness["sequence_key"].isin(keep_keys)].copy()
        entries = entries[entries["sequence_key"].isin(keep_keys)].copy()
    candidates = entries[entries["entry_has_candidate_small_molecule"].astype(bool)].copy()
    candidates["coverage_sort"] = pd.to_numeric(candidates["coverage"], errors="coerce").fillna(-1)
    candidates["resolution_sort"] = pd.to_numeric(candidates["resolution"], errors="coerce").fillna(99)
    candidates = candidates.sort_values(
        ["sequence_key", "coverage_sort", "resolution_sort"], ascending=[True, False, True], kind="mergesort"
    ).groupby("sequence_key", as_index=False).head(args.max_entries_per_target)
    pdb_ids = sorted(set(candidates["pdb_id"].astype(str).str.lower()))

    download_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_cif, pdb_id, cache_dir, args.max_entry_mb * 1024 * 1024): pdb_id
            for pdb_id in pdb_ids
        }
        for index, future in enumerate(as_completed(futures), start=1):
            download_rows.append(future.result())
            if index % 100 == 0:
                print(f"downloaded_or_checked={index}/{len(futures)}", flush=True)
    downloads = pd.DataFrame(download_rows)
    downloads.to_csv(out_dir / "EXPERIMENTAL_MMCIF_DOWNLOAD_AUDIT_V2.csv", index=False)
    path_by_pdb = {row["pdb_id"]: row["path"] for row in download_rows if clean(row.get("path"))}

    af_chains = load_af_chains(readiness)
    readiness_index = readiness.set_index("sequence_key", drop=False)
    ligand_groups = {key: group.copy() for key, group in ligands[ligands["candidate_small_molecule"].astype(bool)].groupby("pdb_id")}
    entry_groups = {key: group.copy() for key, group in candidates.groupby("pdb_id")}
    parser_mmcif = MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    audit_rows = []
    for index, (pdb_id, group) in enumerate(entry_groups.items(), start=1):
        path = clean(path_by_pdb.get(pdb_id))
        if not path:
            for _, target_row in group.iterrows():
                audit_rows.append({"sequence_key": target_row["sequence_key"], "primary_gene": target_row["primary_gene"], "pdb_id": pdb_id, "validation_status": "mmcif_unavailable"})
            continue
        unpacked = cache_dir / f"{pdb_id}.cif"
        try:
            if not unpacked.exists() or unpacked.stat().st_size == 0:
                with gzip.open(path, "rb") as source, unpacked.open("wb") as target:
                    shutil.copyfileobj(source, target)
            structure = parser_mmcif.get_structure(pdb_id, str(unpacked))
            model = structure[0]
            entry_ligands = ligand_groups.get(pdb_id, pd.DataFrame(columns=ligands.columns))
            for _, target_row in group.iterrows():
                sequence_key = clean(target_row["sequence_key"])
                merged = target_row.copy()
                if sequence_key in readiness_index.index:
                    for column in ["pdb_path", "top_pocket_center_x", "top_pocket_center_y", "top_pocket_center_z"]:
                        merged[column] = readiness_index.loc[sequence_key, column]
                audit_rows.extend(evaluate_target_entry(merged, model, entry_ligands, af_chains.get(sequence_key)))
        except Exception as exc:
            for _, target_row in group.iterrows():
                audit_rows.append({"sequence_key": target_row["sequence_key"], "primary_gene": target_row["primary_gene"], "pdb_id": pdb_id, "validation_status": f"parse_failed:{type(exc).__name__}"})
        finally:
            unpacked.unlink(missing_ok=True)
        if index % 100 == 0:
            print(f"validated_entries={index}/{len(entry_groups)}", flush=True)

    audit = pd.DataFrame(audit_rows)
    audit_path = out_dir / "EXPERIMENTAL_HOLO_CHAIN_LIGAND_POCKET_AUDIT_V2.csv.gz"
    audit.to_csv(audit_path, index=False, compression={"method": "gzip", "compresslevel": 5})

    validated = audit[audit["validation_status"].eq("chain_contact_validated")].copy()
    if len(validated):
        validated["pocket_rank"] = validated["project_pocket_status"].map(
            {"confirmed_af_pocket_overlap": 0, "near_af_pocket": 1, "af_alignment_unavailable": 2, "divergent_from_af_top_pocket": 3}
        ).fillna(4)
        validated["role_rank"] = validated["reference_ligand_role"].ne("small_molecule").astype(int)
        validated["resolution_rank"] = pd.to_numeric(validated["resolution"], errors="coerce").fillna(99)
        validated["coverage_rank"] = -pd.to_numeric(validated["coverage"], errors="coerce").fillna(0)
        validated = validated.sort_values(
            ["sequence_key", "pocket_rank", "role_rank", "coverage_rank", "resolution_rank", "target_contact_residues_5a"],
            ascending=[True, True, True, True, True, False], kind="mergesort",
        )
        selected = validated.groupby("sequence_key", as_index=False).head(1).copy()
    else:
        selected = validated

    selection_by_target = {row["sequence_key"]: row for _, row in selected.iterrows()}
    target_rows = []
    for _, target in readiness.iterrows():
        sequence_key = clean(target["sequence_key"])
        row = selection_by_target.get(sequence_key)
        base = {
            "sequence_key": sequence_key,
            "primary_gene": clean(target.get("primary_gene")),
            "target_assay_family": clean(target.get("target_assay_family")),
            "structure_bin": clean(target.get("structure_bin")),
            "calibration_ready_8x8": bool(target.get("calibration_ready_8x8")),
            "calibration_ready_12x12": bool(target.get("calibration_ready_12x12")),
            "af_receptor_path": clean(target.get("pdb_path")),
            "af_center_x": finite_float(target.get("top_pocket_center_x")),
            "af_center_y": finite_float(target.get("top_pocket_center_y")),
            "af_center_z": finite_float(target.get("top_pocket_center_z")),
        }
        if row is None:
            target_rows.append({**base, "validated_experimental_holo": False, "docking_receptor_source": "alphafold_p2rank", "docking_receptor_path": base["af_receptor_path"], "box_center_x": base["af_center_x"], "box_center_y": base["af_center_y"], "box_center_z": base["af_center_z"], "box_size_x": 22.0, "box_size_y": 22.0, "box_size_z": 22.0, "selection_status": "no_validated_experimental_holo_af_fallback"})
            continue
        af_center_available = all(
            math.isfinite(base[column]) for column in ["af_center_x", "af_center_y", "af_center_z"]
        )
        pocket_status = clean(row["project_pocket_status"])
        use_experimental = pocket_status in {"confirmed_af_pocket_overlap", "near_af_pocket"} or (
            not af_center_available and pocket_status == "af_alignment_unavailable"
        )
        receptor_path = ""
        if use_experimental:
            pdb_id = clean(row["pdb_id"])
            packed = path_by_pdb.get(pdb_id)
            unpacked = cache_dir / f"{pdb_id}.selected.cif"
            try:
                with gzip.open(packed, "rb") as source, unpacked.open("wb") as target_handle:
                    shutil.copyfileobj(source, target_handle)
                structure = parser_mmcif.get_structure(f"selected_{pdb_id}", str(unpacked))
                chains = set(clean(row["contacting_protein_chains_5a"]).split(";"))
                chains.discard("")
                chains.add(clean(row["target_chain_id"]))
                receptor = receptor_dir / f"{sequence_key}_{pdb_id}.pdb"
                center = np.asarray([row["ligand_center_x"], row["ligand_center_y"], row["ligand_center_z"]], dtype=float)
                atoms_written = write_receptor_pdb(structure[0], chains, center, receptor)
                if atoms_written >= 30:
                    receptor_path = str(receptor)
                else:
                    use_experimental = False
            except Exception:
                use_experimental = False
            finally:
                unpacked.unlink(missing_ok=True)
        if use_experimental and receptor_path:
            sizes = [max(22.0, min(30.0, finite_float(row[f"ligand_extent_{axis}"], 12.0) + 10.0)) for axis in "xyz"]
            target_rows.append({**base, "validated_experimental_holo": True, "selected_pdb_id": row["pdb_id"], "selected_target_chain": row["target_chain_id"], "selected_reference_ligand": row["chem_comp_id"], "project_pocket_status": row["project_pocket_status"], "ligand_center_distance_to_af_top_pocket": row["ligand_center_distance_to_af_top_pocket"], "alignment_rmsd": row["alignment_rmsd"], "docking_receptor_source": "experimental_holo", "docking_receptor_path": receptor_path, "box_center_x": row["ligand_center_x"], "box_center_y": row["ligand_center_y"], "box_center_z": row["ligand_center_z"], "box_size_x": sizes[0], "box_size_y": sizes[1], "box_size_z": sizes[2], "selection_status": ("validated_holo_selected" if af_center_available else "validated_holo_selected_no_af_pocket_available")})
        else:
            target_rows.append({**base, "validated_experimental_holo": True, "selected_pdb_id": row["pdb_id"], "selected_target_chain": row["target_chain_id"], "selected_reference_ligand": row["chem_comp_id"], "project_pocket_status": row["project_pocket_status"], "ligand_center_distance_to_af_top_pocket": row["ligand_center_distance_to_af_top_pocket"], "alignment_rmsd": row["alignment_rmsd"], "docking_receptor_source": "alphafold_p2rank", "docking_receptor_path": base["af_receptor_path"], "box_center_x": base["af_center_x"], "box_center_y": base["af_center_y"], "box_center_z": base["af_center_z"], "box_size_x": 22.0, "box_size_y": 22.0, "box_size_z": 22.0, "selection_status": "validated_contact_but_pocket_divergent_af_fallback"})

    targets = pd.DataFrame(target_rows)
    target_path = out_dir / "TARGET_DOCKING_RECEPTOR_SELECTION_463_V2.csv"
    targets.to_csv(target_path, index=False)
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "targets": int(len(targets)),
        "candidate_pdb_entries_requested": len(pdb_ids),
        "candidate_pdb_entries_available": int(downloads["path"].astype(str).str.len().gt(0).sum()),
        "chain_ligand_instances_audited": int(len(audit)),
        "targets_with_validated_chain_contact": int(audit[audit["validation_status"].eq("chain_contact_validated")]["sequence_key"].nunique()),
        "targets_with_experimental_holo_selected": int(targets["docking_receptor_source"].eq("experimental_holo").sum()),
        "targets_using_alphafold_fallback": int(targets["docking_receptor_source"].eq("alphafold_p2rank").sum()),
        "ready_8x8_with_experimental_holo": int((targets["calibration_ready_8x8"] & targets["docking_receptor_source"].eq("experimental_holo")).sum()),
        "policy": "Experimental receptors require atom-level target-chain contact and agreement within 12 A of the aligned AlphaFold/P2Rank project pocket. When no AlphaFold/P2Rank box exists, a validated experimental holo ligand defines the box directly. Remaining targets use the pre-audited AlphaFold pocket fallback.",
        "outputs": {"audit": str(audit_path.relative_to(ROOT)), "target_receptor_selection": str(target_path.relative_to(ROOT))},
    }
    (out_dir / "EXPERIMENTAL_HOLO_VALIDATION_V2_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
