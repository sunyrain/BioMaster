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
from sklearn.metrics import average_precision_score, roc_auc_score


CUTOFFS = [20, 50, 100, 200, 500, 1000]
NOVEL_STRUCTURE_CLASSES = {
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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def round_float(value: Any, digits: int = 4) -> float | str:
    parsed = number(value)
    return "" if parsed is None else round(parsed, digits)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
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
    return path if path.is_absolute() else root / path


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
    heavy_coords: list[list[float]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        heavy_coords.append([float(pos.x), float(pos.y), float(pos.z)])
    if not heavy_coords:
        return {"ok": False, "error": "ligand_has_no_heavy_atoms"}
    coords = np.asarray(heavy_coords, dtype=float)
    if not np.isfinite(coords).all():
        return {"ok": False, "error": "ligand_coordinates_invalid"}
    return {"ok": True, "coords": coords, "heavyAtomCount": int(coords.shape[0]), "error": ""}


def read_receptor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "receptor_pdb_missing"}

    coords: list[list[float]] = []
    atom_residue_indices: list[int] = []
    atom_plddt: list[float] = []
    residue_index: dict[tuple[str, str, str, str, str], int] = {}
    residue_values: list[list[float]] = []
    residue_labels: list[str] = []

    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            atom_name = line[12:16].strip().upper()
            if element == "H" or atom_name.startswith("H"):
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
            resname = line[17:20].strip()
            key = (chain, resseq, icode, resname, line[16:17].strip())
            if key not in residue_index:
                residue_index[key] = len(residue_values)
                residue_values.append([])
                residue_labels.append(f"{resname}{resseq}{icode}:{chain}")
            ridx = residue_index[key]
            coords.append([x, y, z])
            atom_residue_indices.append(ridx)
            atom_plddt.append(plddt)
            residue_values[ridx].append(plddt)

    if not coords:
        return {"ok": False, "error": "receptor_has_no_heavy_atoms"}

    coord_arr = np.asarray(coords, dtype=float)
    plddt_arr = np.asarray(atom_plddt, dtype=float)
    residue_plddt = np.asarray([float(np.mean(values)) for values in residue_values], dtype=float)
    if not np.isfinite(coord_arr).all() or not np.isfinite(plddt_arr).all() or not np.isfinite(residue_plddt).all():
        return {"ok": False, "error": "receptor_coordinates_or_plddt_invalid"}

    return {
        "ok": True,
        "coords": coord_arr,
        "tree": cKDTree(coord_arr),
        "atomResidueIndices": np.asarray(atom_residue_indices, dtype=int),
        "atomPlddt": plddt_arr,
        "residuePlddt": residue_plddt,
        "residueLabels": residue_labels,
        "atomCount": int(coord_arr.shape[0]),
        "residueCount": int(residue_plddt.shape[0]),
        "globalMeanPlddt": float(np.mean(residue_plddt)),
        "globalMedianPlddt": float(np.median(residue_plddt)),
        "globalLowPlddtResiduePct": pct(int(np.sum(residue_plddt < 50.0)), int(residue_plddt.shape[0])),
        "error": "",
    }


def residue_stats(receptor: dict[str, Any], residue_indices: set[int]) -> dict[str, Any]:
    if not residue_indices:
        return {
            "residueCount": 0,
            "meanPlddt": None,
            "medianPlddt": None,
            "minPlddt": None,
            "lowPlddtPct": None,
            "confidentPlddtPct": None,
            "veryHighPlddtPct": None,
            "residueLabels": "",
        }
    ordered = sorted(residue_indices)
    values = receptor["residuePlddt"][ordered]
    labels = [receptor["residueLabels"][idx] for idx in ordered[:30]]
    return {
        "residueCount": int(values.shape[0]),
        "meanPlddt": float(np.mean(values)),
        "medianPlddt": float(np.median(values)),
        "minPlddt": float(np.min(values)),
        "lowPlddtPct": pct(int(np.sum(values < 50.0)), int(values.shape[0])),
        "confidentPlddtPct": pct(int(np.sum(values >= 70.0)), int(values.shape[0])),
        "veryHighPlddtPct": pct(int(np.sum(values >= 90.0)), int(values.shape[0])),
        "residueLabels": "; ".join(labels),
    }


def pocket_metrics(ligand: dict[str, Any], receptor: dict[str, Any]) -> dict[str, Any]:
    ligand_coords = ligand["coords"]
    tree: cKDTree = receptor["tree"]
    nearest = tree.query(ligand_coords, k=1)[0]
    atom_residue_indices = receptor["atomResidueIndices"]

    contact_atoms_by_radius: dict[int, set[int]] = {}
    contact_residues_by_radius: dict[int, set[int]] = {}
    ligand_contacts_by_radius: dict[int, int] = {}
    for radius in [4, 5, 8]:
        groups = tree.query_ball_point(ligand_coords, r=float(radius))
        atom_indices = {idx for group in groups for idx in group}
        residue_indices = {int(atom_residue_indices[idx]) for idx in atom_indices}
        contact_atoms_by_radius[radius] = atom_indices
        contact_residues_by_radius[radius] = residue_indices
        ligand_contacts_by_radius[radius] = sum(1 for group in groups if group)

    stats5 = residue_stats(receptor, contact_residues_by_radius[5])
    stats8 = residue_stats(receptor, contact_residues_by_radius[8])
    return {
        "minHeavyAtomDistance": float(np.min(nearest)),
        "medianNearestDistance": float(np.median(nearest)),
        "ligandHeavyAtomCount": ligand["heavyAtomCount"],
        "ligandAtomsWithContact4A": ligand_contacts_by_radius[4],
        "ligandAtomsWithContact5A": ligand_contacts_by_radius[5],
        "ligandAtomsWithContact8A": ligand_contacts_by_radius[8],
        "receptorAtomsWithin4A": len(contact_atoms_by_radius[4]),
        "receptorAtomsWithin5A": len(contact_atoms_by_radius[5]),
        "receptorAtomsWithin8A": len(contact_atoms_by_radius[8]),
        "pocketResiduesWithin4A": len(contact_residues_by_radius[4]),
        "pocketResiduesWithin5A": stats5["residueCount"],
        "pocketResiduesWithin8A": stats8["residueCount"],
        "pocketMeanPlddt5A": stats5["meanPlddt"],
        "pocketMedianPlddt5A": stats5["medianPlddt"],
        "pocketMinPlddt5A": stats5["minPlddt"],
        "pocketLowPlddtResiduePct5A": stats5["lowPlddtPct"],
        "pocketConfidentResiduePct5A": stats5["confidentPlddtPct"],
        "pocketVeryHighResiduePct5A": stats5["veryHighPlddtPct"],
        "pocketMeanPlddt8A": stats8["meanPlddt"],
        "pocketResidueLabels5A": stats5["residueLabels"],
        "globalMeanPlddt": receptor["globalMeanPlddt"],
        "globalMedianPlddt": receptor["globalMedianPlddt"],
        "globalLowPlddtResiduePct": receptor["globalLowPlddtResiduePct"],
        "receptorAtomCount": receptor["atomCount"],
        "receptorResidueCount": receptor["residueCount"],
    }


def classify_structure(row: dict[str, Any]) -> tuple[str, str, float]:
    if row.get("status") != "completed":
        return "not_applicable", "candidate_not_completed", 0.0
    if row.get("ligandReadStatus") != "ok":
        return "fail", row.get("ligandError") or "ligand_read_failed", 0.0
    if row.get("receptorReadStatus") != "ok":
        return "fail", row.get("receptorError") or "receptor_read_failed", 0.0

    contact_residues = int(row.get("pocketResiduesWithin5A") or 0)
    mean_plddt = number(row.get("pocketMeanPlddt5A"))
    low_pct = number(row.get("pocketLowPlddtResiduePct5A")) or 0.0
    min_distance = number(row.get("minHeavyAtomDistance"))
    if contact_residues == 0 or mean_plddt is None:
        return "D_no_contact_pocket", "no_receptor_residue_within_5A", 20.0

    contact_factor = min(1.0, contact_residues / 8.0)
    score = mean_plddt * (0.70 + 0.30 * contact_factor) - 0.20 * low_pct
    if contact_residues < 3:
        score = min(score, 55.0)
    if min_distance is not None and min_distance < 0.65:
        score = min(score, 70.0)
    score = max(0.0, min(100.0, score))

    if mean_plddt >= 80.0 and low_pct <= 10.0 and contact_residues >= 5:
        return "A_high_confidence_pocket", "high_plddt_contact_pocket", score
    if mean_plddt >= 70.0 and contact_residues >= 3:
        return "B_moderate_confidence_pocket", "moderate_plddt_contact_pocket", score
    if mean_plddt >= 50.0:
        return "C_low_confidence_or_sparse_pocket", "low_plddt_or_sparse_contact_pocket", score
    return "D_unreliable_pocket", "very_low_plddt_contact_pocket", score


def base_row(item: pd.Series) -> dict[str, Any]:
    fields = [
        "direction",
        "directionLabelZhFinal",
        "labelZh",
        "finalRankGlobal",
        "finalRankWithinDirection",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "directionScore",
        "affinityScore",
        "diffdock",
        "status",
        "poseAuditStatus",
        "poseAuditReason",
        "ligandSdfPath",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    row = {field: item.get(field, "") for field in fields if field in item.index}
    if "ligandSdfPath" not in row and "confidenceSdfPath" in row:
        row["ligandSdfPath"] = row.get("confidenceSdfPath", "")
    return row


def audit_pose_rows(root: Path, pose_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = pose_df.fillna("").groupby("receptorPdbPath", sort=False, dropna=False)
    for receptor_value, group in grouped:
        receptor_path = abs_or_root(root, receptor_value)
        receptor = read_receptor(receptor_path) if receptor_path else {"ok": False, "error": "receptor_path_missing"}
        for _, item in group.iterrows():
            row = base_row(item)
            row["ligandSdfPath"] = str(abs_or_root(root, row.get("ligandSdfPath")) or "")
            row["receptorPdbPath"] = str(receptor_path or "")
            row["receptorReadStatus"] = "ok" if receptor.get("ok") else "failed"
            row["receptorError"] = receptor.get("error", "")
            if row.get("status") != "completed":
                row["ligandReadStatus"] = "not_read"
                row["structureConfidenceTier"] = "not_applicable"
                row["structureConfidenceReason"] = "candidate_not_completed"
                row["structureConfidenceScore"] = 0.0
                rows.append(row)
                continue
            ligand_path = abs_or_root(root, row.get("ligandSdfPath"))
            ligand = read_ligand_coords(ligand_path) if ligand_path else {"ok": False, "error": "ligand_path_missing"}
            row["ligandReadStatus"] = "ok" if ligand.get("ok") else "failed"
            row["ligandError"] = ligand.get("error", "")
            if ligand.get("ok") and receptor.get("ok"):
                row.update(pocket_metrics(ligand, receptor))
                if row.get("pocketMeanPlddt5A") not in (None, ""):
                    row["pocketVsGlobalPlddtDelta"] = float(row["pocketMeanPlddt5A"]) - float(row["globalMeanPlddt"])
            tier, reason, score = classify_structure(row)
            row["structureConfidenceTier"] = tier
            row["structureConfidenceReason"] = reason
            row["structureConfidenceScore"] = round(score, 4)
            for key in [
                "minHeavyAtomDistance",
                "medianNearestDistance",
                "pocketMeanPlddt5A",
                "pocketMedianPlddt5A",
                "pocketMinPlddt5A",
                "pocketLowPlddtResiduePct5A",
                "pocketConfidentResiduePct5A",
                "pocketVeryHighResiduePct5A",
                "pocketMeanPlddt8A",
                "globalMeanPlddt",
                "globalMedianPlddt",
                "globalLowPlddtResiduePct",
                "pocketVsGlobalPlddtDelta",
            ]:
                if key in row:
                    row[key] = round_float(row[key], 4)
            rows.append(row)
    return rows


def summarize_audit(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completed = [row for row in rows if row.get("status") == "completed"]
    audited = [row for row in completed if row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok"]
    pocket_values = [number(row.get("pocketMeanPlddt5A")) for row in audited]
    pocket_values = [value for value in pocket_values if value is not None]
    tier_counts = Counter(row.get("structureConfidenceTier", "") for row in rows)
    direction_summary: list[dict[str, Any]] = []
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[str(row.get("direction", ""))].append(row)
    for direction, group in sorted(by_direction.items()):
        group_completed = [row for row in group if row.get("status") == "completed"]
        group_audited = [row for row in group_completed if row.get("ligandReadStatus") == "ok" and row.get("receptorReadStatus") == "ok"]
        group_pockets = [number(row.get("pocketMeanPlddt5A")) for row in group_audited]
        group_pockets = [value for value in group_pockets if value is not None]
        counts = Counter(row.get("structureConfidenceTier", "") for row in group)
        direction_summary.append(
            {
                "direction": direction,
                "candidateRows": len(group),
                "completedRows": len(group_completed),
                "auditedCompletedRows": len(group_audited),
                "tierA": counts.get("A_high_confidence_pocket", 0),
                "tierB": counts.get("B_moderate_confidence_pocket", 0),
                "tierC": counts.get("C_low_confidence_or_sparse_pocket", 0),
                "tierD": counts.get("D_unreliable_pocket", 0) + counts.get("D_no_contact_pocket", 0),
                "failRows": counts.get("fail", 0),
                "medianPocketMeanPlddt5A": round(float(np.median(group_pockets)), 4) if group_pockets else None,
                "highConfidencePctOfAudited": round(
                    pct(counts.get("A_high_confidence_pocket", 0), len(group_audited)), 4
                )
                if group_audited
                else 0.0,
            }
        )
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": len(rows),
        "completedRows": len(completed),
        "auditedCompletedRows": len(audited),
        "tierCounts": dict(tier_counts),
        "medianPocketMeanPlddt5A": round(float(np.median(pocket_values)), 4) if pocket_values else None,
        "meanPocketMeanPlddt5A": round(float(np.mean(pocket_values)), 4) if pocket_values else None,
        "highConfidencePctOfAudited": round(
            pct(tier_counts.get("A_high_confidence_pocket", 0), len(audited)), 4
        )
        if audited
        else 0.0,
        "moderateOrHighConfidencePctOfAudited": round(
            pct(
                tier_counts.get("A_high_confidence_pocket", 0)
                + tier_counts.get("B_moderate_confidence_pocket", 0),
                len(audited),
            ),
            4,
        )
        if audited
        else 0.0,
        "methodNote": "AlphaFold pLDDT is read from the PDB B-factor field. Pocket confidence is computed over residues with at least one heavy atom within 5 A of the docked ligand.",
    }
    return summary, direction_summary


def tier_penalty(tier: Any) -> float:
    text = str(tier or "")
    if text.startswith("A_"):
        return 0.0
    if text.startswith("B_"):
        return 2.5
    if text.startswith("C_"):
        return 7.5
    if text.startswith("D_"):
        return 15.0
    if text in {"fail", "not_applicable"}:
        return 20.0
    return 10.0


def safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    y_true = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(scores, errors="coerce").fillna(0)
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def safe_ap(labels: pd.Series, scores: pd.Series) -> float | None:
    y_true = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    y_score = pd.to_numeric(scores, errors="coerce").fillna(0)
    if int(y_true.sum()) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def topk_metrics(df: pd.DataFrame, score_col: str, label_col: str) -> list[dict[str, Any]]:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    total = len(ranked)
    positives = int(pd.to_numeric(ranked[label_col], errors="coerce").fillna(0).astype(int).sum())
    base_rate = positives / total if total else 0.0
    rows: list[dict[str, Any]] = []
    for cutoff in CUTOFFS:
        if cutoff > total:
            continue
        top = ranked.head(cutoff)
        hits = int(pd.to_numeric(top[label_col], errors="coerce").fillna(0).astype(int).sum())
        expected = cutoff * base_rate
        rows.append(
            {
                "scoreColumn": score_col,
                "cutoff": cutoff,
                "hits": hits,
                "positives": positives,
                "precisionPct": round(pct(hits, cutoff), 4),
                "recallPct": round(pct(hits, positives), 4),
                "randomExpectedHits": round(expected, 4),
                "enrichmentVsRandom": round(hits / expected, 4) if expected else None,
            }
        )
    return rows


def integrate_final_priority(root: Path, final_table: Path, audit_rows_path: Path, out_dir: Path) -> dict[str, Any]:
    final = pd.read_csv(final_table).fillna("")
    audit = pd.read_csv(audit_rows_path).fillna("")
    key_cols = ["direction", "pairId"]
    keep_cols = [
        "direction",
        "pairId",
        "structureConfidenceTier",
        "structureConfidenceReason",
        "structureConfidenceScore",
        "pocketResiduesWithin5A",
        "pocketMeanPlddt5A",
        "pocketMedianPlddt5A",
        "pocketMinPlddt5A",
        "pocketLowPlddtResiduePct5A",
        "pocketConfidentResiduePct5A",
        "pocketVeryHighResiduePct5A",
        "globalMeanPlddt",
        "globalLowPlddtResiduePct",
        "pocketVsGlobalPlddtDelta",
        "pocketResidueLabels5A",
    ]
    audit_small = audit[[col for col in keep_cols if col in audit.columns]].drop_duplicates(key_cols)
    merged = final.merge(audit_small, how="left", on=key_cols)
    merged["knownDrugTargetPair"] = pd.to_numeric(merged.get("knownDrugTargetPair", 0), errors="coerce").fillna(0).astype(int)
    merged["finalPriorityScore"] = pd.to_numeric(merged["finalPriorityScore"], errors="coerce").fillna(0.0)
    merged["structureConfidenceScore"] = pd.to_numeric(
        merged.get("structureConfidenceScore", 0), errors="coerce"
    ).fillna(0.0)
    merged["structureConfidencePenalty"] = merged["structureConfidenceTier"].map(tier_penalty)
    merged["structureAdjustedPriorityScore"] = (
        merged["finalPriorityScore"] - merged["structureConfidencePenalty"]
    ).round(4)
    merged = merged.sort_values("structureAdjustedPriorityScore", ascending=False).reset_index(drop=True)
    merged["structureAdjustedRankGlobal"] = np.arange(1, len(merged) + 1)
    merged["structureAdjustedRankWithinDirection"] = (
        merged.groupby("direction")["structureAdjustedPriorityScore"].rank(method="first", ascending=False).astype(int)
    )

    cols_first = [
        "structureAdjustedRankGlobal",
        "structureAdjustedRankWithinDirection",
        "finalRankGlobal",
        "finalRankWithinDirection",
        "direction",
        "pairId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "finalPriorityScore",
        "structureAdjustedPriorityScore",
        "structureConfidenceTier",
        "structureConfidenceScore",
        "structureConfidencePenalty",
        "pocketResiduesWithin5A",
        "pocketMeanPlddt5A",
        "pocketLowPlddtResiduePct5A",
        "reviewTrack",
        "noveltyClass",
        "knownDrugTargetPair",
    ]
    ordered_cols = [col for col in cols_first if col in merged.columns] + [
        col for col in merged.columns if col not in cols_first
    ]
    adjusted_path = out_dir / "final_priority_structure_adjusted_table.csv"
    merged[ordered_cols].to_csv(adjusted_path, index=False)

    audit_path = out_dir / "final_priority_structure_confidence_audit.csv"
    audit_cols = [
        "finalRankGlobal",
        "structureAdjustedRankGlobal",
        "direction",
        "pairId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "finalPriorityScore",
        "structureAdjustedPriorityScore",
        "structureConfidenceTier",
        "structureConfidenceScore",
        "structureConfidencePenalty",
        "pocketResiduesWithin5A",
        "pocketMeanPlddt5A",
        "pocketLowPlddtResiduePct5A",
        "pocketResidueLabels5A",
        "poseAuditStatus",
        "poseAuditReason",
        "finalPriorityTier",
        "reviewTrack",
        "noveltyClass",
        "knownDrugTargetPair",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    merged[[col for col in audit_cols if col in merged.columns]].to_csv(audit_path, index=False)

    low_conf = merged[
        (pd.to_numeric(merged.get("finalRankGlobal", 999999), errors="coerce").fillna(999999) <= 500)
        & (~merged["structureConfidenceTier"].astype(str).str.startswith(("A_", "B_")))
    ].copy()
    low_conf.head(200).to_csv(out_dir / "final_priority_structure_low_confidence_review.csv", index=False)

    novel_mask = merged.get("noveltyClass", "").astype(str).isin(NOVEL_STRUCTURE_CLASSES)
    high_conf_novel = merged[
        merged["structureConfidenceTier"].astype(str).str.startswith(("A_", "B_")) & novel_mask
    ].head(200)
    high_conf_novel.to_csv(out_dir / "final_priority_structure_supported_novel_shortlist.csv", index=False)

    validation_rows = topk_metrics(merged, "finalPriorityScore", "knownDrugTargetPair")
    validation_rows.extend(topk_metrics(merged, "structureAdjustedPriorityScore", "knownDrugTargetPair"))
    write_csv(out_dir / "final_priority_structure_confidence_topk_validation.csv", validation_rows)

    labels = merged["knownDrugTargetPair"]
    original_scores = merged["finalPriorityScore"]
    adjusted_scores = merged["structureAdjustedPriorityScore"]
    tier_counts = dict(Counter(merged["structureConfidenceTier"].astype(str)))
    top100_adjusted = merged.sort_values("structureAdjustedPriorityScore", ascending=False).head(100)
    top100_original = merged.sort_values("finalPriorityScore", ascending=False).head(100)
    original_metrics = {row["cutoff"]: row for row in topk_metrics(merged, "finalPriorityScore", "knownDrugTargetPair")}
    adjusted_metrics = {
        row["cutoff"]: row for row in topk_metrics(merged, "structureAdjustedPriorityScore", "knownDrugTargetPair")
    }

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": int(len(merged)),
        "joinedStructureRows": int(merged["structureConfidenceTier"].astype(str).ne("").sum()),
        "structureConfidenceTierCounts": tier_counts,
        "originalAuroc": safe_auc(labels, original_scores),
        "adjustedAuroc": safe_auc(labels, adjusted_scores),
        "originalAveragePrecision": safe_ap(labels, original_scores),
        "adjustedAveragePrecision": safe_ap(labels, adjusted_scores),
        "originalRecallAt100Pct": original_metrics.get(100, {}).get("recallPct"),
        "adjustedRecallAt100Pct": adjusted_metrics.get(100, {}).get("recallPct"),
        "originalPrecisionAt100Pct": original_metrics.get(100, {}).get("precisionPct"),
        "adjustedPrecisionAt100Pct": adjusted_metrics.get(100, {}).get("precisionPct"),
        "originalEnrichmentAt100": original_metrics.get(100, {}).get("enrichmentVsRandom"),
        "adjustedEnrichmentAt100": adjusted_metrics.get(100, {}).get("enrichmentVsRandom"),
        "top100OriginalTierCounts": dict(Counter(top100_original["structureConfidenceTier"].astype(str))),
        "top100AdjustedTierCounts": dict(Counter(top100_adjusted["structureConfidenceTier"].astype(str))),
        "top500OriginalLowConfidenceRows": int(len(low_conf)),
        "structureSupportedNovelRows": int(len(high_conf_novel)),
        "structureSupportedNovelEligibleRows": int(novel_mask.sum()),
        "structureSupportedNovelClassCounts": dict(Counter(high_conf_novel["noveltyClass"].astype(str))),
        "methodNote": "A conservative structure-adjusted score subtracts a pLDDT/pocket-confidence penalty from the previous final priority score. The original priority table is preserved.",
    }
    write_json(out_dir / "final_priority_structure_confidence_summary.json", summary)
    return summary


def markdown(audit_summary: dict[str, Any], final_summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Final Priority Structure Confidence Audit",
            "",
            f"Generated: {final_summary.get('created_utc')}",
            "",
            "## Method",
            "",
            "AlphaFold pLDDT values were read from receptor PDB B-factor fields. For each completed DiffDock pose, residues with at least one heavy atom within 5 A of the ligand were treated as the local docking pocket. The audit reports pocket pLDDT, pocket contact density, and a conservative reliability tier.",
            "",
            "## Whole Structural Candidate Set",
            "",
            f"- Candidate rows: {audit_summary.get('candidateRows')}",
            f"- Completed rows: {audit_summary.get('completedRows')}",
            f"- Audited completed rows: {audit_summary.get('auditedCompletedRows')}",
            f"- Tier counts: {audit_summary.get('tierCounts')}",
            f"- Median pocket mean pLDDT: {audit_summary.get('medianPocketMeanPlddt5A')}",
            f"- A-tier pocket confidence among audited completed rows: {audit_summary.get('highConfidencePctOfAudited')}%",
            f"- A/B-tier pocket confidence among audited completed rows: {audit_summary.get('moderateOrHighConfidencePctOfAudited')}%",
            "",
            "## Final Priority Table",
            "",
            f"- Final candidates joined with structure confidence: {final_summary.get('joinedStructureRows')} / {final_summary.get('candidateRows')}",
            f"- Structure tier counts: {final_summary.get('structureConfidenceTierCounts')}",
            f"- Original AP vs structure-adjusted AP: {final_summary.get('originalAveragePrecision')} / {final_summary.get('adjustedAveragePrecision')}",
            f"- Original Recall@100 vs structure-adjusted Recall@100: {final_summary.get('originalRecallAt100Pct')}% / {final_summary.get('adjustedRecallAt100Pct')}%",
            f"- Original Top100 structure tiers: {final_summary.get('top100OriginalTierCounts')}",
            f"- Structure-adjusted Top100 structure tiers: {final_summary.get('top100AdjustedTierCounts')}",
            f"- High-priority low-confidence review rows: {final_summary.get('top500OriginalLowConfidenceRows')}",
            f"- Structure-supported new-candidate eligible rows: {final_summary.get('structureSupportedNovelEligibleRows')}",
            f"- Structure-supported new-candidate shortlist rows: {final_summary.get('structureSupportedNovelRows')}",
            f"- Structure-supported new-candidate classes: {final_summary.get('structureSupportedNovelClassCounts')}",
            "",
            "## Outputs",
            "",
            "- Whole-set audit: `outputs/sota_validation/structure_confidence_top10000/candidate_structure_confidence_audit.csv`",
            "- Final-priority audit: `outputs/sota_validation/final_prioritization/final_priority_structure_confidence_audit.csv`",
            "- Structure-adjusted table: `outputs/sota_validation/final_prioritization/final_priority_structure_adjusted_table.csv`",
            "- Low-confidence review: `outputs/sota_validation/final_prioritization/final_priority_structure_low_confidence_review.csv`",
            "- Structure-supported novel shortlist: `outputs/sota_validation/final_prioritization/final_priority_structure_supported_novel_shortlist.csv`",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AlphaFold pocket pLDDT confidence for DiffDock candidate poses.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--pose-audit",
        default="outputs/sota_validation/pose_sanity_top10000/candidate_pose_sanity_audit.csv",
    )
    parser.add_argument(
        "--final-table",
        default="outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/structure_confidence_top10000")
    parser.add_argument("--final-out-dir", default="outputs/sota_validation/final_prioritization")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    final_out_dir = root / args.final_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out_dir.mkdir(parents=True, exist_ok=True)

    pose_df = pd.read_csv(root / args.pose_audit).fillna("")
    rows = audit_pose_rows(root, pose_df)
    audit_path = out_dir / "candidate_structure_confidence_audit.csv"
    write_csv(audit_path, rows)
    summary, direction_summary = summarize_audit(rows)
    write_json(out_dir / "structure_confidence_summary.json", summary)
    write_csv(out_dir / "structure_confidence_direction_summary.csv", direction_summary)

    review_rows = sorted(
        (
            row
            for row in rows
            if row.get("status") == "completed"
            and not str(row.get("structureConfidenceTier", "")).startswith(("A_", "B_"))
            and number(row.get("rank")) is not None
        ),
        key=lambda row: (str(row.get("direction", "")), int(float(row.get("rank") or 999999))),
    )[:500]
    write_csv(out_dir / "structure_confidence_low_confidence_top_review.csv", review_rows)

    final_summary = integrate_final_priority(root, root / args.final_table, audit_path, final_out_dir)
    (final_out_dir / "FINAL_PRIORITY_STRUCTURE_CONFIDENCE_AUDIT.md").write_text(
        markdown(summary, final_summary), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "audit_rows": len(rows),
                "summary": summary,
                "final_summary": final_summary,
                "out_dir": args.out_dir,
                "final_out_dir": args.final_out_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
