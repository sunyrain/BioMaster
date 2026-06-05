from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree


NOVEL_CLASSES = {
    "disease_context_supported_new_pair",
    "model_priority_without_txgnn_kg_path",
}


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def int_value(value: Any, default: int = 0) -> int:
    parsed = number(value)
    return default if parsed is None else int(parsed)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100.0 if denominator else 0.0


def pct_str(value: float | int | None) -> str:
    return "NA" if value is None else f"{float(value):.2f}%"


def fmt(value: float | int | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def round_float(value: Any, digits: int = 4) -> float | str:
    parsed = number(value)
    return "" if parsed is None else round(parsed, digits)


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def abs_or_root(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "FE", "ZN", "MG", "MN", "CA", "NA"}:
        return letters[:2].upper()
    return letters[0].upper()


def heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def expected_bond_limits(atom_a: Chem.Atom, atom_b: Chem.Atom, bond: Chem.Bond) -> tuple[float, float]:
    symbols = {atom_a.GetSymbol().upper(), atom_b.GetSymbol().upper()}
    min_len = 0.85
    if "I" in symbols:
        max_len = 2.45
    elif "BR" in symbols:
        max_len = 2.25
    elif "CL" in symbols:
        max_len = 2.10
    elif symbols & {"S", "P"}:
        max_len = 2.05
    elif bond.GetBondType() == Chem.BondType.TRIPLE:
        max_len = 1.65
    elif bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.AROMATIC}:
        max_len = 1.80
    else:
        max_len = 1.95
    return min_len, max_len


def read_ligand(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "ligand_sdf_missing"}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        return {"ok": False, "error": "ligand_sdf_unreadable"}
    if mol.GetNumConformers() == 0:
        return {"ok": False, "error": "ligand_has_no_conformer"}
    sanitize_status = "ok"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - record but keep geometry auditing.
        sanitize_status = f"failed:{type(exc).__name__}"

    conf = mol.GetConformer()
    coords_by_atom: dict[int, np.ndarray] = {}
    heavy_coords: list[np.ndarray] = []
    for atom_idx in heavy_atom_indices(mol):
        pos = conf.GetAtomPosition(atom_idx)
        coord = np.asarray([float(pos.x), float(pos.y), float(pos.z)], dtype=float)
        coords_by_atom[atom_idx] = coord
        heavy_coords.append(coord)
    if not heavy_coords:
        return {"ok": False, "error": "ligand_has_no_heavy_atoms"}
    coord_arr = np.vstack(heavy_coords)
    if not np.isfinite(coord_arr).all():
        return {"ok": False, "error": "ligand_coordinates_invalid"}

    bond_lengths: list[float] = []
    bond_warning_count = 0
    bond_severe_count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        distance = float(np.linalg.norm(coords_by_atom[begin.GetIdx()] - coords_by_atom[end.GetIdx()]))
        bond_lengths.append(distance)
        min_len, max_len = expected_bond_limits(begin, end, bond)
        if distance < 0.65 or distance > 3.0:
            bond_severe_count += 1
        elif distance < min_len or distance > max_len:
            bond_warning_count += 1

    topo = Chem.GetDistanceMatrix(mol)
    internal_warning_count = 0
    internal_severe_count = 0
    heavy_indices = heavy_atom_indices(mol)
    for i, atom_i in enumerate(heavy_indices):
        for atom_j in heavy_indices[i + 1 :]:
            if topo[atom_i, atom_j] <= 2:
                continue
            distance = float(np.linalg.norm(coords_by_atom[atom_i] - coords_by_atom[atom_j]))
            if distance < 0.70:
                internal_severe_count += 1
            elif distance < 0.95:
                internal_warning_count += 1

    bbox = coord_arr.max(axis=0) - coord_arr.min(axis=0)
    return {
        "ok": True,
        "mol": mol,
        "coords": coord_arr,
        "heavyAtomCount": int(coord_arr.shape[0]),
        "atomCount": int(mol.GetNumAtoms()),
        "formalCharge": int(Chem.GetFormalCharge(mol)),
        "sanitizeStatus": sanitize_status,
        "bondCount": int(len(bond_lengths)),
        "minBondLength": float(min(bond_lengths)) if bond_lengths else None,
        "maxBondLength": float(max(bond_lengths)) if bond_lengths else None,
        "meanBondLength": float(np.mean(bond_lengths)) if bond_lengths else None,
        "bondLengthWarningCount": int(bond_warning_count),
        "bondLengthSevereCount": int(bond_severe_count),
        "internalClashWarningCount": int(internal_warning_count),
        "internalClashSevereCount": int(internal_severe_count),
        "bboxMax": float(bbox.max()),
        "centroid": coord_arr.mean(axis=0),
        "error": "",
    }


def read_receptor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "receptor_pdb_missing"}
    coords: list[list[float]] = []
    atom_residue_indices: list[int] = []
    residue_index: dict[tuple[str, str, str, str], int] = {}
    residue_plddt_values: list[list[float]] = []
    residue_rows: list[dict[str, Any]] = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            element = element or element_from_atom_name(atom_name)
            if element == "H" or atom_name.upper().startswith("H"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                plddt = float(line[60:66])
            except ValueError:
                continue
            chain = line[21:22].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26:27].strip()
            resname = line[17:20].strip().upper()
            key = (chain, resseq, icode, resname)
            if key not in residue_index:
                residue_index[key] = len(residue_rows)
                residue_rows.append(
                    {
                        "residueName": resname,
                        "residueNumber": resseq,
                        "insertionCode": icode,
                        "chain": chain,
                        "residueLabel": f"{resname}{resseq}{icode}:{chain}",
                    }
                )
                residue_plddt_values.append([])
            ridx = residue_index[key]
            atom_residue_indices.append(ridx)
            residue_plddt_values[ridx].append(plddt)
            coords.append([x, y, z])
    if not coords:
        return {"ok": False, "error": "receptor_has_no_heavy_atoms"}
    coord_arr = np.asarray(coords, dtype=float)
    if not np.isfinite(coord_arr).all():
        return {"ok": False, "error": "receptor_coordinates_invalid"}
    residue_plddt = np.asarray([float(np.mean(values)) for values in residue_plddt_values], dtype=float)
    for idx, row in enumerate(residue_rows):
        row["residuePlddt"] = float(residue_plddt[idx])
    return {
        "ok": True,
        "coords": coord_arr,
        "tree": cKDTree(coord_arr),
        "atomResidueIndices": np.asarray(atom_residue_indices, dtype=int),
        "residues": residue_rows,
        "residuePlddt": residue_plddt,
        "atomCount": int(coord_arr.shape[0]),
        "residueCount": int(len(residue_rows)),
        "globalMeanPlddt": float(np.mean(residue_plddt)),
        "centroid": coord_arr.mean(axis=0),
        "error": "",
    }


def ligand_receptor_metrics(ligand: dict[str, Any], receptor: dict[str, Any]) -> dict[str, Any]:
    ligand_coords = ligand["coords"]
    receptor_tree: cKDTree = receptor["tree"]
    nearest = receptor_tree.query(ligand_coords, k=1)[0]
    groups_075 = receptor_tree.query_ball_point(ligand_coords, r=0.75)
    groups_10 = receptor_tree.query_ball_point(ligand_coords, r=1.00)
    groups_20 = receptor_tree.query_ball_point(ligand_coords, r=2.00)
    groups_40 = receptor_tree.query_ball_point(ligand_coords, r=4.00)
    groups_50 = receptor_tree.query_ball_point(ligand_coords, r=5.00)
    atom_residue_indices = receptor["atomResidueIndices"]
    residue_indices_5a = sorted({int(atom_residue_indices[idx]) for group in groups_50 for idx in group})
    plddt_values = [float(receptor["residues"][idx]["residuePlddt"]) for idx in residue_indices_5a]
    severe_pairs = sum(len(group) for group in groups_075)
    warning_pairs = sum(len(group) for group in groups_10)
    atoms_contact_4 = sum(1 for group in groups_40 if group)
    atoms_contact_5 = sum(1 for group in groups_50 if group)
    return {
        "minLigandReceptorDistance": float(np.min(nearest)),
        "medianLigandReceptorNearestDistance": float(np.median(nearest)),
        "ligandAtomsWithContact2A": int(sum(1 for group in groups_20 if group)),
        "ligandAtomsWithContact4A": int(atoms_contact_4),
        "ligandAtomsWithContact5A": int(atoms_contact_5),
        "ligandContactCoverage4APct": pct(atoms_contact_4, ligand["heavyAtomCount"]),
        "ligandContactCoverage5APct": pct(atoms_contact_5, ligand["heavyAtomCount"]),
        "severeLigandReceptorClashPairs075A": int(severe_pairs),
        "warningLigandReceptorClashPairs1A": int(warning_pairs),
        "pocketResidues5A": int(len(residue_indices_5a)),
        "pocketMeanPlddt5A": float(np.mean(plddt_values)) if plddt_values else None,
        "pocketLowPlddtResiduePct5A": pct(sum(1 for value in plddt_values if value < 50.0), len(plddt_values))
        if plddt_values
        else None,
        "centroidDistanceToReceptor": float(np.linalg.norm(ligand["centroid"] - receptor["centroid"])),
    }


def quality_flags(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    if row.get("candidateStatus") != "completed":
        hard.append("candidate_not_completed")
    if row.get("ligandReadStatus") != "ok":
        hard.append(str(row.get("ligandError") or "ligand_read_failed"))
    if row.get("receptorReadStatus") != "ok":
        hard.append(str(row.get("receptorError") or "receptor_read_failed"))
    if int_value(row.get("ligandHeavyAtomCount")) < 3 and row.get("ligandReadStatus") == "ok":
        hard.append("ligand_too_small")
    if int_value(row.get("receptorHeavyAtomCount")) < 100 and row.get("receptorReadStatus") == "ok":
        hard.append("receptor_too_small")
    if int_value(row.get("bondLengthSevereCount")) > 0:
        hard.append("severe_ligand_bond_geometry")
    if int_value(row.get("internalClashSevereCount")) > 0:
        hard.append("severe_ligand_internal_clash")
    if (number(row.get("minLigandReceptorDistance")) or 999.0) < 0.50:
        hard.append("extreme_ligand_receptor_overlap")
    if int_value(row.get("severeLigandReceptorClashPairs075A")) >= 5:
        hard.append("many_severe_ligand_receptor_clashes")
    if int_value(row.get("pocketResidues5A")) == 0 and row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok":
        hard.append("no_receptor_contact_within_5A")

    sanitize = str(row.get("ligandSanitizeStatus") or "")
    if sanitize and sanitize != "ok":
        soft.append("ligand_sanitize_warning")
    if int_value(row.get("bondLengthWarningCount")) > 0:
        soft.append("ligand_bond_length_warning")
    if int_value(row.get("internalClashWarningCount")) > 0:
        soft.append("ligand_internal_clash_warning")
    if 0 < int_value(row.get("severeLigandReceptorClashPairs075A")) < 5:
        soft.append("local_ligand_receptor_clash")
    if int_value(row.get("warningLigandReceptorClashPairs1A")) > int_value(row.get("severeLigandReceptorClashPairs075A")):
        soft.append("possible_ligand_receptor_clash")
    if int_value(row.get("ligandAtomsWithContact4A")) == 0 and row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok":
        soft.append("no_contact_within_4A")
    contact_coverage = number(row.get("ligandContactCoverage4APct")) or 0.0
    if 0 < contact_coverage < 20.0:
        soft.append("low_ligand_contact_coverage")
    pocket_mean = number(row.get("pocketMeanPlddt5A"))
    if pocket_mean is not None and pocket_mean < 70.0:
        soft.append("low_or_moderate_pocket_plddt")
    low_plddt = number(row.get("pocketLowPlddtResiduePct5A"))
    if low_plddt is not None and low_plddt > 30.0:
        soft.append("many_low_plddt_pocket_residues")
    bbox = number(row.get("ligandBboxMax"))
    if bbox is not None and bbox > 60.0:
        soft.append("large_ligand_coordinate_span")
    return hard, soft


def classify_quality(row: dict[str, Any]) -> tuple[float, str, str, str]:
    hard, soft = quality_flags(row)
    score = 100.0
    score -= 28.0 * len(hard)
    score -= 6.0 * len(soft)
    contact_coverage = number(row.get("ligandContactCoverage4APct")) or 0.0
    pocket_mean = number(row.get("pocketMeanPlddt5A")) or 0.0
    if contact_coverage < 20.0:
        score -= 10.0
    elif contact_coverage >= 50.0:
        score += 4.0
    if pocket_mean < 50.0:
        score -= 18.0
    elif pocket_mean < 70.0:
        score -= 8.0
    elif pocket_mean >= 80.0:
        score += 4.0
    score = max(0.0, min(100.0, score))

    if hard:
        tier = "D_pose_quality_fail"
        action = "structure_resolution_required"
    elif soft:
        tier = "C_pose_quality_review"
        action = "pose_quality_review"
    elif contact_coverage >= 50.0 and pocket_mean >= 80.0 and int_value(row.get("pocketResidues5A")) >= 5:
        tier = "A_pose_quality_supported"
        action = "structure_quality_supported"
    else:
        tier = "B_pose_quality_acceptable"
        action = "structure_quality_acceptable"
    reason_bits = hard or soft or ["geometry_and_contact_quality_supported"]
    return round(score, 4), tier, action, "; ".join(reason_bits)


def is_novel(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or "").lower()
    return truthy(row.get("strictNovelPairFlag")) or str(row.get("noveltyClass") or "") in NOVEL_CLASSES or "new_pair" in novelty


def is_known(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


def audit_row(root: Path, row: pd.Series, ligand_cache: dict[str, dict[str, Any]], receptor_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ligand_path = abs_or_root(root, row.get("confidenceSdfPath"))
    receptor_path = abs_or_root(root, row.get("receptorPdbPath"))
    base = {
        "sotaMlAdmetRankGlobal": row.get("sotaMlAdmetRankGlobal", ""),
        "sotaContextRankGlobal": row.get("sotaContextRankGlobal", ""),
        "sotaReadyRankGlobal": row.get("sotaReadyRankGlobal", ""),
        "finalRankGlobal": row.get("finalRankGlobal", ""),
        "direction": row.get("direction", ""),
        "directionLabelZhFinal": row.get("directionLabelZhFinal", ""),
        "pairId": row.get("pairId", ""),
        "drugId": row.get("drugId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "strictNovelPairFlag": row.get("strictNovelPairFlag", ""),
        "noveltyClass": row.get("auditNoveltyClass") or row.get("noveltyClass", ""),
        "candidateStatus": row.get("status", ""),
        "diffdock": row.get("diffdock", ""),
        "structureConfidenceTier": row.get("structureConfidenceTier", ""),
        "poseInterpretabilityTier": row.get("poseInterpretabilityTier", ""),
        "poseAuditStatus": row.get("poseAuditStatus", ""),
        "poseAuditReason": row.get("poseAuditReason", ""),
        "confidenceSdfPath": str(ligand_path) if ligand_path else "",
        "receptorPdbPath": str(receptor_path) if receptor_path else "",
    }
    if base["candidateStatus"] != "completed":
        base.update(
            {
                "ligandReadStatus": "not_read",
                "receptorReadStatus": "not_read",
                "ligandError": "",
                "receptorError": "",
            }
        )
        score, tier, action, reason = classify_quality(base)
        base.update(
            {
                "poseQualityScore": score,
                "poseQualityTier": tier,
                "poseQualityAction": action,
                "poseQualityReason": reason,
                "poseQualityHardFlags": reason,
                "poseQualitySoftFlags": "",
            }
        )
        return base

    if not ligand_path:
        ligand = {"ok": False, "error": "ligand_path_missing"}
    else:
        ligand = ligand_cache.get(str(ligand_path))
        if ligand is None:
            ligand = read_ligand(ligand_path)
            ligand_cache[str(ligand_path)] = ligand
    if not receptor_path:
        receptor = {"ok": False, "error": "receptor_path_missing"}
    else:
        receptor = receptor_cache.get(str(receptor_path))
        if receptor is None:
            receptor = read_receptor(receptor_path)
            receptor_cache[str(receptor_path)] = receptor

    base.update(
        {
            "ligandReadStatus": "ok" if ligand.get("ok") else "failed",
            "ligandError": ligand.get("error", ""),
            "receptorReadStatus": "ok" if receptor.get("ok") else "failed",
            "receptorError": receptor.get("error", ""),
        }
    )
    if ligand.get("ok"):
        base.update(
            {
                "ligandAtomCount": ligand.get("atomCount"),
                "ligandHeavyAtomCount": ligand.get("heavyAtomCount"),
                "ligandFormalCharge": ligand.get("formalCharge"),
                "ligandSanitizeStatus": ligand.get("sanitizeStatus"),
                "ligandBondCount": ligand.get("bondCount"),
                "minLigandBondLength": round_float(ligand.get("minBondLength")),
                "maxLigandBondLength": round_float(ligand.get("maxBondLength")),
                "meanLigandBondLength": round_float(ligand.get("meanBondLength")),
                "bondLengthWarningCount": ligand.get("bondLengthWarningCount"),
                "bondLengthSevereCount": ligand.get("bondLengthSevereCount"),
                "internalClashWarningCount": ligand.get("internalClashWarningCount"),
                "internalClashSevereCount": ligand.get("internalClashSevereCount"),
                "ligandBboxMax": round_float(ligand.get("bboxMax")),
            }
        )
    if receptor.get("ok"):
        base.update(
            {
                "receptorHeavyAtomCount": receptor.get("atomCount"),
                "receptorResidueCount": receptor.get("residueCount"),
                "receptorGlobalMeanPlddt": round_float(receptor.get("globalMeanPlddt")),
            }
        )
    if ligand.get("ok") and receptor.get("ok"):
        for key, value in ligand_receptor_metrics(ligand, receptor).items():
            base[key] = round_float(value)

    hard, soft = quality_flags(base)
    score, tier, action, reason = classify_quality(base)
    base.update(
        {
            "poseQualityScore": score,
            "poseQualityTier": tier,
            "poseQualityAction": action,
            "poseQualityReason": reason,
            "poseQualityHardFlags": "; ".join(hard) if hard else "none",
            "poseQualitySoftFlags": "; ".join(soft) if soft else "none",
        }
    )
    return base


def rank_column(df: pd.DataFrame) -> str:
    for column in ["sotaMlAdmetRankGlobal", "sotaContextRankGlobal", "sotaReadyRankGlobal", "finalRankGlobal"]:
        if column in df.columns:
            return column
    return df.columns[0]


def build_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = root / args.source
    final_df = pd.read_csv(source_path).fillna("")
    sort_col = rank_column(final_df)
    final_df["_rankSort"] = pd.to_numeric(final_df[sort_col], errors="coerce").fillna(999999999)
    selected = final_df.sort_values("_rankSort").copy()
    if args.top_n > 0:
        selected = selected.head(args.top_n).copy()

    ligand_cache: dict[str, dict[str, Any]] = {}
    receptor_cache: dict[str, dict[str, Any]] = {}
    rows = [audit_row(root, row, ligand_cache, receptor_cache) for _, row in selected.iterrows()]
    audit_df = pd.DataFrame(rows).fillna("")

    merge_cols = [
        "direction",
        "pairId",
        "poseQualityScore",
        "poseQualityTier",
        "poseQualityAction",
        "poseQualityReason",
        "poseQualityHardFlags",
        "poseQualitySoftFlags",
        "ligandReadStatus",
        "receptorReadStatus",
        "ligandSanitizeStatus",
        "bondLengthWarningCount",
        "bondLengthSevereCount",
        "internalClashWarningCount",
        "internalClashSevereCount",
        "minLigandReceptorDistance",
        "ligandContactCoverage4APct",
        "pocketResidues5A",
        "pocketMeanPlddt5A",
        "pocketLowPlddtResiduePct5A",
        "severeLigandReceptorClashPairs075A",
        "warningLigandReceptorClashPairs1A",
    ]
    existing_merge_cols = [column for column in merge_cols if column in audit_df.columns]
    augmented = final_df.drop(columns=["_rankSort"], errors="ignore").merge(
        audit_df[existing_merge_cols],
        on=["direction", "pairId"],
        how="left",
        suffixes=("", "_poseQuality"),
    )
    base_score = pd.to_numeric(augmented.get("sotaMlAdmetScore", augmented.get("sotaContextScore", 0)), errors="coerce").fillna(0)
    quality_score = pd.to_numeric(augmented.get("poseQualityScore"), errors="coerce").fillna(40)
    augmented["sotaPoseQualityScore"] = (0.82 * base_score + 0.18 * quality_score).round(4)
    augmented["sotaPoseQualityTier"] = augmented.get("poseQualityTier", "")
    augmented["sotaPoseQualityAction"] = augmented.get("poseQualityAction", "")
    augmented = augmented.sort_values(["sotaPoseQualityScore"], ascending=False).reset_index(drop=True)
    augmented.insert(0, "sotaPoseQualityRankGlobal", np.arange(1, len(augmented) + 1))

    out_dir = root / args.out_dir
    final_dir = root / args.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "pose_quality_candidate_audit.csv"
    direction_path = out_dir / "pose_quality_direction_summary.csv"
    summary_path = out_dir / "pose_quality_summary.json"
    md_path = out_dir / "POSE_QUALITY_AUDIT.md"
    final_matrix_path = final_dir / "final_priority_pose_quality_matrix.csv"
    shortlist_path = final_dir / "final_priority_pose_quality_top300_expert_shortlist.csv"
    review_path = final_dir / "final_priority_pose_quality_review_queue.csv"
    final_md_path = final_dir / "FINAL_PRIORITY_POSE_QUALITY_AUDIT.md"

    audit_df.to_csv(audit_path, index=False)
    direction_df = direction_summary(audit_df)
    direction_df.to_csv(direction_path, index=False)
    summary = summarize(audit_df, direction_df, len(final_df), len(selected), len(ligand_cache), len(receptor_cache), args.source)
    write_json(summary_path, summary)
    md_text = markdown(summary, direction_df, audit_df)
    md_path.write_text(md_text, encoding="utf-8")

    augmented.to_csv(final_matrix_path, index=False)
    supported = augmented[augmented["poseQualityTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    supported.head(300).to_csv(shortlist_path, index=False)
    review = augmented[~augmented["poseQualityTier"].astype(str).str.startswith(("A_", "B_"))].copy()
    review.head(300).to_csv(review_path, index=False)
    final_md_path.write_text(md_text, encoding="utf-8")
    return {
        "summary": summary,
        "audit_path": str(audit_path.relative_to(root)),
        "direction_path": str(direction_path.relative_to(root)),
        "final_matrix_path": str(final_matrix_path.relative_to(root)),
        "shortlist_path": str(shortlist_path.relative_to(root)),
        "review_path": str(review_path.relative_to(root)),
        "md_path": str(md_path.relative_to(root)),
        "final_md_path": str(final_md_path.relative_to(root)),
    }


def direction_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for direction, group in audit_df.groupby("direction", dropna=False):
        tiers = Counter(group["poseQualityTier"].astype(str))
        supported = int(sum(group["poseQualityTier"].astype(str).str.startswith(("A_", "B_"))))
        readable = int(sum((group["ligandReadStatus"].astype(str) == "ok") & (group["receptorReadStatus"].astype(str) == "ok")))
        rows.append(
            {
                "direction": direction,
                "candidateRows": int(len(group)),
                "readableRows": readable,
                "qualitySupportedRows": supported,
                "qualitySupportedPct": round(pct(supported, len(group)), 4),
                "knownRows": int(sum(group.apply(lambda row: is_known(row.to_dict()), axis=1))),
                "novelRows": int(sum(group.apply(lambda row: is_novel(row.to_dict()), axis=1))),
                "tierCounts": json.dumps(dict(tiers), ensure_ascii=False, sort_keys=True),
                "medianPoseQualityScore": round_float(pd.to_numeric(group["poseQualityScore"], errors="coerce").median()),
                "medianMinLigandReceptorDistance": round_float(
                    pd.to_numeric(group.get("minLigandReceptorDistance"), errors="coerce").median()
                ),
                "medianContactCoverage4APct": round_float(pd.to_numeric(group.get("ligandContactCoverage4APct"), errors="coerce").median()),
                "medianPocketMeanPlddt5A": round_float(pd.to_numeric(group.get("pocketMeanPlddt5A"), errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values("direction")


def summarize(
    audit_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    source_rows: int,
    audited_rows: int,
    unique_ligands: int,
    unique_receptors: int,
    source: str,
) -> dict[str, Any]:
    tiers = Counter(audit_df["poseQualityTier"].astype(str))
    reasons = Counter()
    for value in audit_df["poseQualityReason"].astype(str):
        for part in [item.strip() for item in value.split(";") if item.strip()]:
            reasons[part] += 1
    supported_mask = audit_df["poseQualityTier"].astype(str).str.startswith(("A_", "B_"))
    readable_mask = (audit_df["ligandReadStatus"].astype(str) == "ok") & (audit_df["receptorReadStatus"].astype(str) == "ok")
    rank_series = pd.to_numeric(
        audit_df.get("sotaMlAdmetRankGlobal", audit_df.get("sotaContextRankGlobal", audit_df.get("sotaReadyRankGlobal"))),
        errors="coerce",
    ).fillna(999999999)
    top100 = audit_df[rank_series <= 100].copy()
    top300 = audit_df[rank_series <= 300].copy()
    known_rows = int(sum(audit_df.apply(lambda row: is_known(row.to_dict()), axis=1)))
    novel_rows = int(sum(audit_df.apply(lambda row: is_novel(row.to_dict()), axis=1)))
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": (
            f"RDKit-based pose-quality audit over {audited_rows} of {source_rows} final candidates. "
            "This is a local PoseBusters-like quality gate and does not replace standard PoseBusters/ProLIF/GNINA validation."
        ),
        "source": source,
        "sourceRows": int(source_rows),
        "candidateRows": int(audited_rows),
        "uniqueLigandsRead": int(unique_ligands),
        "uniqueReceptorsRead": int(unique_receptors),
        "readableRows": int(readable_mask.sum()),
        "readablePct": round(pct(int(readable_mask.sum()), audited_rows), 4),
        "qualitySupportedRows": int(supported_mask.sum()),
        "qualitySupportedPct": round(pct(int(supported_mask.sum()), audited_rows), 4),
        "knownRows": known_rows,
        "novelRows": novel_rows,
        "poseQualityTierCounts": dict(tiers),
        "poseQualityReasonCounts": dict(reasons),
        "medianPoseQualityScore": round_float(pd.to_numeric(audit_df["poseQualityScore"], errors="coerce").median()),
        "medianMinLigandReceptorDistance": round_float(pd.to_numeric(audit_df.get("minLigandReceptorDistance"), errors="coerce").median()),
        "medianContactCoverage4APct": round_float(pd.to_numeric(audit_df.get("ligandContactCoverage4APct"), errors="coerce").median()),
        "medianPocketMeanPlddt5A": round_float(pd.to_numeric(audit_df.get("pocketMeanPlddt5A"), errors="coerce").median()),
        "top100": {
            "rows": int(len(top100)),
            "qualitySupportedRows": int(sum(top100["poseQualityTier"].astype(str).str.startswith(("A_", "B_")))),
            "tierCounts": dict(Counter(top100["poseQualityTier"].astype(str))),
            "knownRows": int(sum(top100.apply(lambda row: is_known(row.to_dict()), axis=1))) if len(top100) else 0,
            "novelRows": int(sum(top100.apply(lambda row: is_novel(row.to_dict()), axis=1))) if len(top100) else 0,
        },
        "top300": {
            "rows": int(len(top300)),
            "qualitySupportedRows": int(sum(top300["poseQualityTier"].astype(str).str.startswith(("A_", "B_")))),
            "tierCounts": dict(Counter(top300["poseQualityTier"].astype(str))),
        },
        "directionRows": json_safe(direction_df.to_dict(orient="records")),
        "methodNote": (
            "Ligand SDF conformers are parsed with RDKit; heavy-atom bond geometry, intramolecular nonbonded clashes, "
            "ligand-receptor severe overlaps, contact coverage, receptor contact residues, and AlphaFold pLDDT from PDB "
            "B-factors are used to assign transparent A/B/C/D quality tiers."
        ),
    }


def markdown(summary: dict[str, Any], direction_df: pd.DataFrame, audit_df: pd.DataFrame) -> str:
    examples = audit_df.sort_values("poseQualityScore", ascending=False).head(10)
    lines = [
        "# RDKit Pose-Quality Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows audited: {summary['candidateRows']}",
        f"- Readable ligand/receptor rows: {summary['readableRows']} ({summary['readablePct']:.2f}%)",
        f"- A/B pose-quality supported rows: {summary['qualitySupportedRows']} ({summary['qualitySupportedPct']:.2f}%)",
        f"- Known rows: {summary['knownRows']}; novel rows: {summary['novelRows']}",
        f"- Tier counts: {summary['poseQualityTierCounts']}",
        f"- Median pose-quality score: {summary['medianPoseQualityScore']}",
        f"- Median ligand-receptor minimum distance: {summary['medianMinLigandReceptorDistance']}",
        f"- Median 4 A ligand contact coverage: {summary['medianContactCoverage4APct']}",
        f"- Median pocket mean pLDDT: {summary['medianPocketMeanPlddt5A']}",
        f"- Top100 quality-supported rows: {summary['top100']['qualitySupportedRows']}/{summary['top100']['rows']}",
        "",
        "## Direction Summary",
        "",
    ]
    for _, row in direction_df.iterrows():
        lines.append(
            f"- {row['direction']}: {row['qualitySupportedRows']}/{row['candidateRows']} A/B supported "
            f"({row['qualitySupportedPct']:.2f}%); median score {row['medianPoseQualityScore']}; "
            f"median pocket pLDDT {row['medianPocketMeanPlddt5A']}."
        )
    lines.extend(["", "## Representative High-Quality Poses", ""])
    for _, row in examples.iterrows():
        lines.append(
            f"- {row.get('drug')} - {row.get('target')} ({row.get('direction')}): "
            f"{row.get('poseQualityTier')}, score {row.get('poseQualityScore')}, "
            f"contact coverage {row.get('ligandContactCoverage4APct')}%, pocket pLDDT {row.get('pocketMeanPlddt5A')}."
        )
    lines.extend(["", "## Method Note", "", summary["methodNote"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local RDKit pose-quality audit for final candidates.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/final_prioritization/final_priority_ml_admet_matrix.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/pose_quality")
    parser.add_argument("--final-dir", default="outputs/sota_validation/final_prioritization")
    parser.add_argument("--top-n", type=int, default=0, help="0 means audit all source rows.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build_audit(root, args)
    print(json.dumps(json_safe(result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
