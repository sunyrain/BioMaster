"""Read-only structural evidence for the BioMaster explorer.

Canonical residue indices are displayed on exact-sequence AlphaFold models.
Prepared experimental receptors retain their own numbering and are deliberately
not highlighted with canonical residue indices without an explicit alignment.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
from pathlib import Path
from urllib.parse import quote


def _rows(path: Path):
    if not path.is_file():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _residues(value):
    return sorted({int(part) for part in re.split(r"[; ,]+", str(value or "")) if part.isdigit()})


def _center(row, prefix):
    values = [_number(row.get(prefix + axis)) for axis in ("x", "y", "z")]
    return values if all(value is not None for value in values) else None


def load_structures(root: Path, targets: dict) -> dict:
    """Return serializable evidence and a private, validated file allowlist."""
    root = root.resolve()
    af_directory = root / "data/processed/alphafold_receptors_v6"
    protocol_directory = root / "outputs/strict_receptor_protocol_338_v1"
    allowed = [af_directory.resolve(), protocol_directory.resolve()]
    files = {}
    result = {key: {"pockets": [], "structure": None} for key in targets}
    accession_targets = {}
    for key, target in targets.items():
        identifiers = target.get("identifiers", {})
        accession = identifiers.get("uniprot_id") or identifiers.get("uniprot_accession")
        if accession:
            accession_targets.setdefault(accession, []).append(key)

    def safe_path(raw):
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = root / path
        resolved = path.resolve()
        if not any(resolved.is_relative_to(directory) for directory in allowed):
            return None
        return resolved if resolved.is_file() else None

    def url(key, pocket=""):
        base = f"/api/structure/{quote(str(key), safe='')}"
        return base + (f"?pocket={quote(pocket, safe='')}" if pocket else "")

    sources = []

    def source(relative, name):
        path = root / relative
        if path.is_file():
            sources.append({"id": path.stem, "name": name, "path": relative, "source": name})
        return path

    universe = source(
        "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv",
        "AlphaFold exact-sequence structures / ChEMBL target universe",
    )
    for row in _rows(universe):
        keys = accession_targets.get(row.get("uniprot_accession"), [])
        if row.get("target_chembl_id") in result:
            keys = list(set(keys + [row["target_chembl_id"]]))
        path = safe_path(row.get("af_pdb_path"))
        # Exact sequence is essential for canonical residue highlighting.
        exact = str(row.get("af_exact_sequence_model", "")).lower() == "true"
        if not path or not exact:
            continue
        for key in keys:
            files[(key, "")] = path
            result[key]["structure"] = {
                "url": url(key), "format": "pdb", "source": "AlphaFold DB",
                "type": "predicted", "coordinate_system": "uniprot_canonical",
                "accession": row.get("uniprot_accession"),
                "mean_plddt": _number(row.get("af_mean_plddt")),
                "pocket_mean_plddt": _number(row.get("pocket_mean_plddt")),
                "note": "精确序列 AlphaFold 预测模型；实验位点按 UniProt 残基编号映射。",
            }

    atlas_dir = "outputs/chembl37_known_pocket_atlas/final_atlas/"
    representatives = {}
    for row in _rows(root / (atlas_dir + "TARGET_REPRESENTATIVE_KNOWN_POCKET_737.csv")):
        representatives[row.get("uniprot_accession")] = row
    known = source(atlas_dir + "KNOWN_POCKET_CANONICAL_RESIDUE_SETS.csv.gz", "Curated experimental pocket atlas")
    for row in _rows(known):
        accession = row.get("uniprot_accession")
        keys = accession_targets.get(accession, [])
        if row.get("target_chembl_id") in result:
            keys = list(set(keys + [row["target_chembl_id"]]))
        representative = representatives.get(accession, {})
        pocket_id = row.get("known_pocket_id", "")
        is_representative = pocket_id == representative.get("known_pocket_id")
        for key in keys:
            path = files.get((key, ""))
            if path:
                files[(key, pocket_id)] = path
            residues = _residues(row.get("residue_set_key"))
            result[key]["pockets"].append({
                "id": pocket_id,
                "name": f"{str(row.get('representative_pdb_id', '')).upper()} · {row.get('representative_ligand_id', '')}",
                "type": "experimental", "source": row.get("evidence_sources"),
                "grade": row.get("known_pocket_grade"), "representative": is_representative,
                "residues": residues if path else [], "canonical_residues": residues,
                "residue_count": len(residues),
                "center": _center(representative, "known_centroid_") if is_representative and path else None,
                "pdb_id": row.get("representative_pdb_id"), "chain": row.get("representative_chain_id"),
                "ligand": row.get("representative_ligand_id"), "ligand_name": row.get("representative_ligand_name"),
                "method": row.get("representative_method"), "resolution": _number(row.get("representative_resolution")),
                "structure_url": url(key, pocket_id) if path else None, "structure_format": "pdb",
                "structure_source": "AlphaFold DB · 实验位点映射" if path else None,
                "coordinate_system": "uniprot_canonical" if path else None,
                "external_url": f"https://www.rcsb.org/structure/{row.get('representative_pdb_id', '')}",
            })

    predictions = source(atlas_dir + "P2RANK_ALL_PREDICTED_POCKETS_875.csv.gz", "P2Rank predicted pockets")
    for row in _rows(predictions):
        for key in accession_targets.get(row.get("uniprot_accession"), []):
            rank = int(_number(row.get("p2rank_rank")) or 0)
            pocket_id = f"p2rank-{rank}"
            path = files.get((key, ""))
            if path:
                files[(key, pocket_id)] = path
            residues = _residues(row.get("p2rank_residue_positions"))
            result[key]["pockets"].append({
                "id": pocket_id, "name": f"P2Rank · Pocket {rank}", "type": "predicted",
                "source": "P2Rank", "rank": rank, "residues": residues if path else [],
                "canonical_residues": residues, "residue_count": len(residues),
                "center": _center(row, "p2rank_center_") if path else None,
                "probability": _number(row.get("p2rank_probability")), "score": _number(row.get("p2rank_score")),
                "structure_url": url(key, pocket_id) if path else None, "structure_format": "pdb",
                "structure_source": "AlphaFold DB", "coordinate_system": "uniprot_canonical" if path else None,
            })

    protocols = source(
        "outputs/strict_receptor_protocol_338_v1/FINAL_RECEPTOR_PROTOCOL_AUDIT_338.csv",
        "Frozen experimental receptor protocols",
    )
    for row in _rows(protocols):
        key = row.get("target_chembl_id")
        if key not in result:
            continue
        protocol_path = safe_path(row.get("protocol_json"))
        if not protocol_path:
            continue
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        receptor = safe_path(protocol.get("files", {}).get("receptor_prepared_with_context_pdb"))
        ligand = safe_path(protocol.get("files", {}).get("reference_ligand_crystal_sdf"))
        if not receptor:
            continue
        files[(key, "holo")] = receptor
        if ligand:
            files[(key, "holo-ligand")] = ligand
        result[key]["pockets"].insert(0, {
            "id": "holo", "name": f"{str(row.get('pdb_id', '')).upper()} · 实验受体与共晶配体",
            "source": "PDB · frozen receptor protocol", "type": "experimental",
            "residues": [], "canonical_residues": _residues(protocol.get("canonical_pocket_residues")),
            "center": _center(row, "box_center_"), "pdb_id": row.get("pdb_id"),
            "chain": row.get("target_chain_id"), "ligand": row.get("reference_ligand_id"),
            "structure_url": url(key, "holo"), "structure_format": "pdb", "structure_source": "PDB · prepared experimental receptor",
            "ligand_url": url(key, "holo-ligand") if ligand else None, "ligand_format": "sdf",
            "coordinate_system": "experimental_structure", "protocol_status": row.get("final_audit_status"),
            "redocking_rmsd": _number(row.get("redock_best_symmetry_rmsd_A_final")),
            "note": "实验受体经项目流程准备；共晶配体使用同一坐标系。未将 UniProt 编号直接套用于实验结构。",
            "external_url": f"https://www.rcsb.org/structure/{row.get('pdb_id', '')}",
        })
        if result[key]["structure"] is None:
            files[(key, "")] = receptor
            result[key]["structure"] = {
                "url": url(key), "format": "pdb", "source": "PDB · prepared experimental receptor",
                "type": "experimental", "coordinate_system": "experimental_structure", "pdb_id": row.get("pdb_id"),
            }

    for item in result.values():
        item["pockets"].sort(key=lambda pocket: (
            0 if pocket["id"] == "holo" else 1 if pocket.get("representative") else 2 if pocket["type"] == "predicted" else 3,
            pocket.get("rank", 0),
        ))
    return {
        "targets": result, "files": files, "sources": sources,
        "counts": {"pockets": sum(len(item["pockets"]) for item in result.values()),
                   "structures": sum(item["structure"] is not None for item in result.values())},
    }
