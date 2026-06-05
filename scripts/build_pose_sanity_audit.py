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


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def abs_or_root(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


def read_candidates(path: Path, top_n_per_direction: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "rank" not in df:
        raise ValueError(f"{path} does not contain a rank column")
    df["rankNumeric"] = pd.to_numeric(df["rank"], errors="coerce").fillna(999999999).astype(int)
    if top_n_per_direction:
        df = df[df["rankNumeric"] <= top_n_per_direction].copy()
    return df


def read_ligand_coords(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "ligand_sdf_missing"}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        return {"ok": False, "error": "ligand_sdf_unreadable"}
    if mol.GetNumConformers() == 0:
        return {"ok": False, "error": "ligand_has_no_conformer"}
    conf = mol.GetConformer()
    coords: list[list[float]] = []
    heavy_coords: list[list[float]] = []
    atomic_nums: list[int] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz = [float(pos.x), float(pos.y), float(pos.z)]
        coords.append(xyz)
        atomic_nums.append(atom.GetAtomicNum())
        if atom.GetAtomicNum() > 1:
            heavy_coords.append(xyz)
    arr = np.asarray(heavy_coords or coords, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return {"ok": False, "error": "ligand_coordinates_invalid"}
    bbox = arr.max(axis=0) - arr.min(axis=0)
    return {
        "ok": True,
        "coords": arr,
        "atomCount": int(len(coords)),
        "heavyAtomCount": int(len(heavy_coords)),
        "formalCharge": int(Chem.GetFormalCharge(mol)),
        "bboxX": float(bbox[0]),
        "bboxY": float(bbox[1]),
        "bboxZ": float(bbox[2]),
        "bboxMax": float(bbox.max()),
        "centroid": arr.mean(axis=0),
        "error": "",
    }


def read_receptor_coords(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "receptor_pdb_missing"}
    coords: list[list[float]] = []
    residues: set[tuple[str, str, str]] = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            atom_name = line[12:16].strip().upper()
            if element == "H" or atom_name.startswith("H"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
            residues.add((line[21:22].strip(), line[22:26].strip(), line[17:20].strip()))
    if not coords:
        return {"ok": False, "error": "receptor_has_no_heavy_atoms"}
    arr = np.asarray(coords, dtype=float)
    if not np.isfinite(arr).all():
        return {"ok": False, "error": "receptor_coordinates_invalid"}
    bbox = arr.max(axis=0) - arr.min(axis=0)
    return {
        "ok": True,
        "coords": arr,
        "atomCount": int(arr.shape[0]),
        "residueCount": int(len(residues)),
        "bboxX": float(bbox[0]),
        "bboxY": float(bbox[1]),
        "bboxZ": float(bbox[2]),
        "bboxMax": float(bbox.max()),
        "centroid": arr.mean(axis=0),
        "error": "",
    }


def receptor_from_cache(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path)
    if key not in cache:
        cache[key] = read_receptor_coords(path)
    return cache[key]


def contact_metrics(ligand: dict[str, Any], receptor: dict[str, Any]) -> dict[str, Any]:
    ligand_coords = ligand["coords"]
    receptor_coords = receptor["coords"]
    tree = cKDTree(receptor_coords)
    min_distances = tree.query(ligand_coords, k=1)[0]
    contacts_2 = tree.query_ball_point(ligand_coords, r=2.0)
    contacts_4 = tree.query_ball_point(ligand_coords, r=4.0)
    contacts_5 = tree.query_ball_point(ligand_coords, r=5.0)
    near_atoms_4 = sorted({idx for group in contacts_4 for idx in group})
    near_atoms_5 = sorted({idx for group in contacts_5 for idx in group})
    centroid_distance = float(np.linalg.norm(ligand["centroid"] - receptor["centroid"]))
    return {
        "minHeavyAtomDistance": float(np.min(min_distances)),
        "medianNearestDistance": float(np.median(min_distances)),
        "ligandAtomsWithContact2A": int(sum(1 for group in contacts_2 if group)),
        "ligandAtomsWithContact4A": int(sum(1 for group in contacts_4 if group)),
        "ligandAtomsWithContact5A": int(sum(1 for group in contacts_5 if group)),
        "receptorAtomsWithin4A": int(len(near_atoms_4)),
        "receptorAtomsWithin5A": int(len(near_atoms_5)),
        "centroidDistanceToReceptor": centroid_distance,
    }


def classify_pose(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("status") != "completed":
        return "not_applicable", "candidate_not_completed"
    if row.get("ligandReadStatus") != "ok":
        return "fail", row.get("ligandError") or "ligand_read_failed"
    if row.get("receptorReadStatus") != "ok":
        return "fail", row.get("receptorError") or "receptor_read_failed"
    heavy_atoms = int(row.get("ligandHeavyAtomCount") or 0)
    receptor_atoms = int(row.get("receptorHeavyAtomCount") or 0)
    min_distance = number(row.get("minHeavyAtomDistance"))
    contact4 = int(row.get("ligandAtomsWithContact4A") or 0)
    bbox_max = number(row.get("ligandBboxMax"))
    if heavy_atoms < 3:
        return "fail", "ligand_too_small_for_pose_audit"
    if receptor_atoms < 100:
        return "fail", "receptor_too_small_for_pose_audit"
    if bbox_max is not None and bbox_max > 80:
        return "warning", "ligand_coordinate_span_unusually_large"
    if min_distance is not None and min_distance < 0.65:
        return "warning", "severe_ligand_receptor_clash"
    if min_distance is not None and min_distance < 1.0:
        return "warning", "possible_ligand_receptor_clash"
    if contact4 == 0:
        return "warning", "no_receptor_contact_within_4A"
    if min_distance is not None and min_distance > 8.0:
        return "warning", "ligand_far_from_receptor_surface"
    return "pass", "basic_geometry_ok"


def audit_rows(root: Path, candidates: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    receptor_cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for _, item in candidates.sort_values(["direction", "rankNumeric"]).iterrows():
        ligand_path = abs_or_root(root, item.get("confidenceSdfPath") or item.get("rank1SdfPath"))
        receptor_path = abs_or_root(root, item.get("receptorPdbPath") or item.get("diffdock_receptor_pdb_path"))
        base = {
            "direction": item.get("direction", ""),
            "labelZh": DIRECTION_LABELS.get(item.get("direction", ""), item.get("direction", "")),
            "rank": int(item.get("rankNumeric") or 999999999),
            "pairId": item.get("pairId", item.get("pair_id", "")),
            "drugId": item.get("drugId", item.get("drug_id", "")),
            "drug": item.get("drug", item.get("drug_name", "")),
            "target": item.get("target", item.get("gene_name", "")),
            "protein": item.get("protein", item.get("protein_id", "")),
            "directionScore": number(item.get("directionScore")),
            "affinityScore": number(item.get("affinityScore", item.get("affinity_score"))),
            "diffdock": number(item.get("diffdock", item.get("diffdock_confidence"))),
            "status": item.get("status", item.get("structural_status", "")),
            "ligandSdfPath": str(ligand_path) if ligand_path else "",
            "receptorPdbPath": str(receptor_path) if receptor_path else "",
        }
        if base["status"] != "completed":
            base.update(
                {
                    "ligandReadStatus": "not_read",
                    "receptorReadStatus": "not_read",
                    "poseAuditStatus": "not_applicable",
                    "poseAuditReason": "candidate_not_completed",
                }
            )
            rows.append(base)
            continue
        ligand = read_ligand_coords(ligand_path) if ligand_path else {"ok": False, "error": "ligand_path_missing"}
        receptor = receptor_from_cache(receptor_path, receptor_cache) if receptor_path else {"ok": False, "error": "receptor_path_missing"}
        base.update(
            {
                "ligandReadStatus": "ok" if ligand.get("ok") else "failed",
                "ligandError": ligand.get("error", ""),
                "ligandAtomCount": ligand.get("atomCount", ""),
                "ligandHeavyAtomCount": ligand.get("heavyAtomCount", ""),
                "ligandFormalCharge": ligand.get("formalCharge", ""),
                "ligandBboxX": round(float(ligand.get("bboxX", 0.0)), 3) if ligand.get("ok") else "",
                "ligandBboxY": round(float(ligand.get("bboxY", 0.0)), 3) if ligand.get("ok") else "",
                "ligandBboxZ": round(float(ligand.get("bboxZ", 0.0)), 3) if ligand.get("ok") else "",
                "ligandBboxMax": round(float(ligand.get("bboxMax", 0.0)), 3) if ligand.get("ok") else "",
                "receptorReadStatus": "ok" if receptor.get("ok") else "failed",
                "receptorError": receptor.get("error", ""),
                "receptorHeavyAtomCount": receptor.get("atomCount", ""),
                "receptorResidueCount": receptor.get("residueCount", ""),
                "receptorBboxMax": round(float(receptor.get("bboxMax", 0.0)), 3) if receptor.get("ok") else "",
            }
        )
        if ligand.get("ok") and receptor.get("ok"):
            metrics = contact_metrics(ligand, receptor)
            for key, value in metrics.items():
                base[key] = round(value, 4) if isinstance(value, float) else value
        pose_status, reason = classify_pose(base)
        base["poseAuditStatus"] = pose_status
        base["poseAuditReason"] = reason
        rows.append(base)
    return rows, receptor_cache


def summarize(rows: list[dict[str, Any]], receptor_cache: dict[str, dict[str, Any]], top_n_per_direction: int | None) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    readable = [row for row in completed if row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok"]
    by_direction: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("direction", "")].append(row)
    for direction, group in sorted(grouped.items()):
        group_completed = [row for row in group if row.get("status") == "completed"]
        group_readable = [
            row
            for row in group_completed
            if row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok"
        ]
        by_direction[direction] = {
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "rows": len(group),
            "completedRows": len(group_completed),
            "readablePoseRows": len(group_readable),
            "readablePosePct": pct(len(group_readable), len(group_completed)),
            "poseAuditStatusCounts": dict(Counter(row.get("poseAuditStatus", "NA") for row in group)),
            "poseAuditReasonCounts": dict(Counter(row.get("poseAuditReason", "NA") for row in group)),
            "medianMinHeavyAtomDistance": float(pd.Series([row["minHeavyAtomDistance"] for row in group_readable if "minHeavyAtomDistance" in row]).median())
            if group_readable
            else None,
            "medianLigandAtomsWithContact4A": float(pd.Series([row["ligandAtomsWithContact4A"] for row in group_readable if "ligandAtomsWithContact4A" in row]).median())
            if group_readable
            else None,
        }
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topNPerDirection": top_n_per_direction,
        "candidateRows": len(rows),
        "completedRows": len(completed),
        "readablePoseRows": len(readable),
        "readablePosePct": pct(len(readable), len(completed)),
        "uniqueReceptorsRead": len(receptor_cache),
        "poseAuditStatusCounts": dict(Counter(row.get("poseAuditStatus", "NA") for row in rows)),
        "poseAuditReasonCounts": dict(Counter(row.get("poseAuditReason", "NA") for row in rows)),
        "byDirection": by_direction,
        "methodNote": (
            "Lightweight structural sanity audit using RDKit SDF parsing and receptor PDB heavy-atom contacts. "
            "This checks whether DiffDock poses are readable and geometrically plausible; it is not a docking-score rescoring model."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit basic geometry and read-status of completed DiffDock poses.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidates", default="outputs/disease_directions/disease_direction_integrated_candidates.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/pose_sanity")
    parser.add_argument("--top-n-per-direction", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    candidates = read_candidates(root / args.candidates, args.top_n_per_direction)
    rows, receptor_cache = audit_rows(root, candidates)
    summary = summarize(rows, receptor_cache, args.top_n_per_direction)
    summary["inputs"] = {
        "candidates": args.candidates,
    }
    summary["outputs"] = {
        "poseAudit": str((out_dir / "candidate_pose_sanity_audit.csv").resolve()),
        "summary": str((out_dir / "pose_sanity_summary.json").resolve()),
    }
    fields = [
        "direction",
        "labelZh",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "directionScore",
        "affinityScore",
        "diffdock",
        "status",
        "poseAuditStatus",
        "poseAuditReason",
        "ligandReadStatus",
        "ligandError",
        "ligandAtomCount",
        "ligandHeavyAtomCount",
        "ligandFormalCharge",
        "ligandBboxX",
        "ligandBboxY",
        "ligandBboxZ",
        "ligandBboxMax",
        "receptorReadStatus",
        "receptorError",
        "receptorHeavyAtomCount",
        "receptorResidueCount",
        "receptorBboxMax",
        "minHeavyAtomDistance",
        "medianNearestDistance",
        "ligandAtomsWithContact2A",
        "ligandAtomsWithContact4A",
        "ligandAtomsWithContact5A",
        "receptorAtomsWithin4A",
        "receptorAtomsWithin5A",
        "centroidDistanceToReceptor",
        "ligandSdfPath",
        "receptorPdbPath",
    ]
    write_csv(out_dir / "candidate_pose_sanity_audit.csv", fields, rows)
    write_json(out_dir / "pose_sanity_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
