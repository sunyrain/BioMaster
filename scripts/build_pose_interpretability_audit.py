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


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"}
AROMATIC_RESIDUES = {"PHE", "TRP", "TYR", "HIS"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
POLAR_RESIDUES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS"}
POLAR_ELEMENTS = {"N", "O", "S", "P"}
NOVEL_CLASS_MARKERS = ("new_pair", "model_priority", "sparse_external_context")


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


def is_novel_candidate(row: dict[str, Any]) -> bool:
    novelty = str(row.get("noveltyClass") or "").lower()
    action = str(row.get("sotaReadyAction") or "").lower()
    return (
        truthy(row.get("strictNovelPairFlag"))
        or action == "novel_pair_expert_review"
        or any(marker in novelty for marker in NOVEL_CLASS_MARKERS)
    )


def is_known_candidate(row: dict[str, Any]) -> bool:
    return truthy(row.get("knownDrugTargetPair"))


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
    if not fieldnames:
        fieldnames = ["empty"]
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


def element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "FE", "ZN", "MG", "MN", "CA", "NA"}:
        return letters[:2].upper()
    return letters[0].upper()


def read_ligand(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "ligand_sdf_missing"}
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        return {"ok": False, "error": "ligand_sdf_unreadable"}
    if mol.GetNumConformers() == 0:
        return {"ok": False, "error": "ligand_has_no_conformer"}
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    conf = mol.GetConformer()
    coords: list[list[float]] = []
    atom_rows: list[dict[str, Any]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([float(pos.x), float(pos.y), float(pos.z)])
        atom_rows.append(
            {
                "ligandAtomIndex": atom.GetIdx(),
                "ligandAtomSymbol": atom.GetSymbol(),
                "ligandAtomFormalCharge": atom.GetFormalCharge(),
                "ligandAtomAromatic": bool(atom.GetIsAromatic()),
            }
        )
    if not coords:
        return {"ok": False, "error": "ligand_has_no_heavy_atoms"}
    coord_arr = np.asarray(coords, dtype=float)
    if not np.isfinite(coord_arr).all():
        return {"ok": False, "error": "ligand_coordinates_invalid"}
    return {"ok": True, "coords": coord_arr, "atoms": atom_rows, "heavyAtomCount": int(coord_arr.shape[0]), "error": ""}


def read_receptor(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "error": "receptor_pdb_missing"}

    coords: list[list[float]] = []
    atom_rows: list[dict[str, Any]] = []
    atom_residue_indices: list[int] = []
    residue_index: dict[tuple[str, str, str, str, str], int] = {}
    residue_values: list[list[float]] = []
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
            altloc = line[16:17].strip()
            key = (chain, resseq, icode, resname, altloc)
            if key not in residue_index:
                residue_index[key] = len(residue_values)
                residue_values.append([])
                residue_rows.append(
                    {
                        "residueName": resname,
                        "residueNumber": resseq,
                        "insertionCode": icode,
                        "chain": chain,
                        "residueLabel": f"{resname}{resseq}{icode}:{chain}",
                    }
                )
            ridx = residue_index[key]
            atom_idx = len(coords)
            coords.append([x, y, z])
            atom_residue_indices.append(ridx)
            residue_values[ridx].append(plddt)
            atom_rows.append(
                {
                    "receptorAtomArrayIndex": atom_idx,
                    "receptorAtomName": atom_name,
                    "receptorElement": element,
                    "residueIndex": ridx,
                    "atomPlddt": plddt,
                }
            )

    if not coords:
        return {"ok": False, "error": "receptor_has_no_heavy_atoms"}
    coord_arr = np.asarray(coords, dtype=float)
    residue_plddt = np.asarray([float(np.mean(values)) for values in residue_values], dtype=float)
    if not np.isfinite(coord_arr).all() or not np.isfinite(residue_plddt).all():
        return {"ok": False, "error": "receptor_coordinates_or_plddt_invalid"}
    for idx, row in enumerate(residue_rows):
        row["residuePlddt"] = float(residue_plddt[idx])
    return {
        "ok": True,
        "coords": coord_arr,
        "tree": cKDTree(coord_arr),
        "atoms": atom_rows,
        "atomResidueIndices": np.asarray(atom_residue_indices, dtype=int),
        "residues": residue_rows,
        "residuePlddt": residue_plddt,
        "residueCount": int(residue_plddt.shape[0]),
        "atomCount": int(coord_arr.shape[0]),
        "globalMeanPlddt": float(np.mean(residue_plddt)),
        "globalMedianPlddt": float(np.median(residue_plddt)),
        "error": "",
    }


def residue_charge(residue_name: str) -> int:
    if residue_name in POSITIVE_RESIDUES:
        return 1
    if residue_name in NEGATIVE_RESIDUES:
        return -1
    return 0


def classify_contact(
    residue_name: str,
    receptor_element: str,
    ligand_atom: dict[str, Any],
    distance: float,
) -> str:
    ligand_symbol = str(ligand_atom.get("ligandAtomSymbol") or "").upper()
    ligand_charge = int_value(ligand_atom.get("ligandAtomFormalCharge"))
    receptor_charge = residue_charge(residue_name)
    receptor_element = receptor_element.upper()
    if distance < 1.2:
        return "steric_clash_review"
    if receptor_charge and ligand_charge and receptor_charge * ligand_charge < 0 and distance <= 4.0:
        return "salt_bridge_candidate"
    if ligand_symbol in POLAR_ELEMENTS and receptor_element in POLAR_ELEMENTS and distance <= 3.6:
        return "polar_hbond_candidate"
    if residue_name in AROMATIC_RESIDUES and (ligand_atom.get("ligandAtomAromatic") or ligand_symbol == "C") and distance <= 5.0:
        return "aromatic_or_pi_contact"
    if residue_name in HYDROPHOBIC_RESIDUES and ligand_symbol == "C" and distance <= 4.6:
        return "hydrophobic_contact"
    if residue_name in POLAR_RESIDUES and (ligand_symbol in POLAR_ELEMENTS or receptor_element in POLAR_ELEMENTS):
        return "polar_vdw_contact"
    return "van_der_waals_contact"


def contact_priority(contact_class: str) -> int:
    order = {
        "steric_clash_review": 0,
        "salt_bridge_candidate": 1,
        "polar_hbond_candidate": 2,
        "aromatic_or_pi_contact": 3,
        "hydrophobic_contact": 4,
        "polar_vdw_contact": 5,
        "van_der_waals_contact": 6,
    }
    return order.get(contact_class, 9)


def candidate_contacts(row: pd.Series, ligand: dict[str, Any], receptor: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ligand_coords = ligand["coords"]
    receptor_coords = receptor["coords"]
    tree: cKDTree = receptor["tree"]
    atom_residue_indices = receptor["atomResidueIndices"]

    nearest_distance = tree.query(ligand_coords, k=1)[0]
    residue_contacts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ligand_atoms_with_4a = 0
    ligand_atoms_with_5a = 0
    receptor_atoms_with_4a: set[int] = set()
    receptor_atoms_with_5a: set[int] = set()

    groups_5a = tree.query_ball_point(ligand_coords, r=5.0)
    groups_4a = tree.query_ball_point(ligand_coords, r=4.0)
    for ligand_pos, atom_indices in enumerate(groups_4a):
        if atom_indices:
            ligand_atoms_with_4a += 1
            receptor_atoms_with_4a.update(atom_indices)
    for ligand_pos, atom_indices in enumerate(groups_5a):
        if atom_indices:
            ligand_atoms_with_5a += 1
            receptor_atoms_with_5a.update(atom_indices)
        ligand_atom = ligand["atoms"][ligand_pos]
        ligand_coord = ligand_coords[ligand_pos]
        for receptor_atom_index in atom_indices:
            receptor_atom = receptor["atoms"][receptor_atom_index]
            residue_index = int(atom_residue_indices[receptor_atom_index])
            residue = receptor["residues"][residue_index]
            distance = float(np.linalg.norm(ligand_coord - receptor_coords[receptor_atom_index]))
            contact_class = classify_contact(
                residue["residueName"],
                receptor_atom["receptorElement"],
                ligand_atom,
                distance,
            )
            residue_contacts[residue_index].append(
                {
                    "distance": distance,
                    "contactClass": contact_class,
                    "receptorAtomName": receptor_atom["receptorAtomName"],
                    "receptorElement": receptor_atom["receptorElement"],
                    "ligandAtomIndex": ligand_atom["ligandAtomIndex"],
                    "ligandAtomSymbol": ligand_atom["ligandAtomSymbol"],
                    "ligandAtomFormalCharge": ligand_atom["ligandAtomFormalCharge"],
                    "ligandAtomAromatic": ligand_atom["ligandAtomAromatic"],
                }
            )

    contact_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    low_confidence_residues: list[str] = []
    high_confidence_residues: list[str] = []
    top_labels: list[str] = []

    base = {
        "sotaReadyRankGlobal": row.get("sotaReadyRankGlobal", ""),
        "finalRankGlobal": row.get("finalRankGlobal", ""),
        "direction": row.get("direction", ""),
        "directionLabelZhFinal": row.get("directionLabelZhFinal", ""),
        "pairId": row.get("pairId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "sotaReadyTier": row.get("sotaReadyTier", ""),
        "sotaReadyAction": row.get("sotaReadyAction", ""),
        "noveltyClass": row.get("auditNoveltyClass") or row.get("noveltyClass", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "strictNovelPairFlag": row.get("strictNovelPairFlag", ""),
        "diffdock": row.get("diffdock", ""),
        "poseAuditStatus": row.get("poseAuditStatus", ""),
        "poseAuditReason": row.get("poseAuditReason", ""),
        "structureConfidenceTier": row.get("structureConfidenceTier", ""),
        "pocketMeanPlddt5A": row.get("pocketMeanPlddt5A", ""),
    }

    for residue_index, contacts in residue_contacts.items():
        residue = receptor["residues"][residue_index]
        best = sorted(contacts, key=lambda item: (contact_priority(item["contactClass"]), item["distance"]))[0]
        min_distance = min(item["distance"] for item in contacts)
        contact_class = best["contactClass"]
        class_counts[contact_class] += 1
        label = residue["residueLabel"]
        top_labels.append(label)
        residue_plddt = float(residue["residuePlddt"])
        if residue_plddt >= 70.0:
            high_confidence_residues.append(label)
        if residue_plddt < 50.0:
            low_confidence_residues.append(label)
        contact_rows.append(
            {
                **base,
                "residueLabel": label,
                "residueName": residue["residueName"],
                "residueNumber": residue["residueNumber"],
                "chain": residue["chain"],
                "residuePlddt": round_float(residue_plddt),
                "minContactDistance": round_float(min_distance),
                "contactClass": contact_class,
                "contactsWithin5A": len(contacts),
                "contactsWithin4A": sum(1 for item in contacts if item["distance"] <= 4.0),
                **{key: best[key] for key in best if key != "distance"},
            }
        )

    pocket_residue_count = len(residue_contacts)
    pocket_plddt_values = [float(receptor["residues"][idx]["residuePlddt"]) for idx in residue_contacts]
    computed_pocket_mean = float(np.mean(pocket_plddt_values)) if pocket_plddt_values else None
    low_pct = pct(sum(1 for value in pocket_plddt_values if value < 50.0), len(pocket_plddt_values)) if pocket_plddt_values else None
    summary = {
        **base,
        "candidateStatus": row.get("status", ""),
        "ligandHeavyAtomCount": ligand["heavyAtomCount"],
        "receptorResidueCount": receptor["residueCount"],
        "globalMeanPlddt": round_float(receptor["globalMeanPlddt"]),
        "computedMinHeavyAtomDistance": round_float(float(np.min(nearest_distance))),
        "computedMedianNearestDistance": round_float(float(np.median(nearest_distance))),
        "ligandAtomsWithContact4A": ligand_atoms_with_4a,
        "ligandAtomsWithContact5A": ligand_atoms_with_5a,
        "receptorAtomsWithin4A": len(receptor_atoms_with_4a),
        "receptorAtomsWithin5A": len(receptor_atoms_with_5a),
        "computedPocketResidues5A": pocket_residue_count,
        "computedPocketMeanPlddt5A": round_float(computed_pocket_mean),
        "computedPocketLowPlddtResiduePct5A": round_float(low_pct),
        "contactClassCounts": "; ".join(f"{key}:{class_counts[key]}" for key in sorted(class_counts)),
        "hydrophobicContactResidues": class_counts.get("hydrophobic_contact", 0),
        "polarOrHbondContactResidues": class_counts.get("polar_hbond_candidate", 0) + class_counts.get("polar_vdw_contact", 0),
        "saltBridgeCandidateResidues": class_counts.get("salt_bridge_candidate", 0),
        "aromaticContactResidues": class_counts.get("aromatic_or_pi_contact", 0),
        "clashReviewResidues": class_counts.get("steric_clash_review", 0),
        "topContactResidues": "; ".join(top_labels[:20]),
        "highConfidenceContactResidues": "; ".join(high_confidence_residues[:20]),
        "lowConfidenceContactResidues": "; ".join(low_confidence_residues[:20]),
    }
    tier, reason = classify_interpretability(summary)
    summary["poseInterpretabilityTier"] = tier
    summary["poseInterpretabilityReason"] = reason
    summary["discussionSummary"] = discussion_summary(summary)
    return summary, contact_rows


def classify_interpretability(summary: dict[str, Any]) -> tuple[str, str]:
    if summary.get("candidateStatus") != "completed":
        return "D_not_interpretable_pose", "candidate_not_completed"
    if summary.get("computedPocketResidues5A", 0) == 0:
        return "D_not_interpretable_pose", "no_residue_contact_within_5A"
    min_distance = number(summary.get("computedMinHeavyAtomDistance"))
    if min_distance is not None and min_distance < 0.65:
        return "D_not_interpretable_pose", "severe_steric_clash_geometry"
    if int_value(summary.get("clashReviewResidues")) > 0 or summary.get("poseAuditStatus") == "warning":
        return "C_geometry_or_context_review_pose", "geometry_warning_or_clash_review"
    pocket_mean = number(summary.get("computedPocketMeanPlddt5A")) or 0.0
    residue_count = int_value(summary.get("computedPocketResidues5A"))
    structure_tier = str(summary.get("structureConfidenceTier") or "")
    if structure_tier.startswith("A_") and pocket_mean >= 80.0 and residue_count >= 5:
        return "A_mechanistically_discussable_pose", "high_confidence_pocket_with_residue_contacts"
    if structure_tier.startswith(("A_", "B_")) and pocket_mean >= 70.0 and residue_count >= 3:
        return "B_structurally_supported_pose", "moderate_or_high_confidence_pocket_contacts"
    return "C_geometry_or_context_review_pose", "low_confidence_or_sparse_pocket_contacts"


def discussion_summary(summary: dict[str, Any]) -> str:
    contact_bits = []
    for label, key in [
        ("hydrophobic", "hydrophobicContactResidues"),
        ("polar/H-bond-like", "polarOrHbondContactResidues"),
        ("aromatic/pi-like", "aromaticContactResidues"),
        ("salt-bridge-like", "saltBridgeCandidateResidues"),
    ]:
        value = int_value(summary.get(key))
        if value:
            contact_bits.append(f"{value} {label} residue contacts")
    contact_text = ", ".join(contact_bits) if contact_bits else "mainly van der Waals residue contacts"
    residues = summary.get("topContactResidues") or "no named residues"
    return (
        f"{summary.get('drug')} - {summary.get('target')} has {summary.get('computedPocketResidues5A')} residues within 5 A "
        f"(mean pocket pLDDT {summary.get('computedPocketMeanPlddt5A')}); {contact_text}. "
        f"Representative residues: {residues}."
    )


def failed_summary(row: pd.Series, ligand_error: str = "", receptor_error: str = "") -> dict[str, Any]:
    tier = "D_not_interpretable_pose"
    reason = ligand_error or receptor_error or "missing_or_unreadable_structure"
    return {
        "sotaReadyRankGlobal": row.get("sotaReadyRankGlobal", ""),
        "finalRankGlobal": row.get("finalRankGlobal", ""),
        "direction": row.get("direction", ""),
        "directionLabelZhFinal": row.get("directionLabelZhFinal", ""),
        "pairId": row.get("pairId", ""),
        "drug": row.get("drug", ""),
        "target": row.get("target", ""),
        "protein": row.get("protein", ""),
        "proteinName": row.get("proteinName", ""),
        "sotaReadyTier": row.get("sotaReadyTier", ""),
        "sotaReadyAction": row.get("sotaReadyAction", ""),
        "noveltyClass": row.get("auditNoveltyClass") or row.get("noveltyClass", ""),
        "knownDrugTargetPair": row.get("knownDrugTargetPair", ""),
        "strictNovelPairFlag": row.get("strictNovelPairFlag", ""),
        "diffdock": row.get("diffdock", ""),
        "poseAuditStatus": row.get("poseAuditStatus", ""),
        "poseAuditReason": row.get("poseAuditReason", ""),
        "structureConfidenceTier": row.get("structureConfidenceTier", ""),
        "pocketMeanPlddt5A": row.get("pocketMeanPlddt5A", ""),
        "candidateStatus": row.get("status", ""),
        "computedPocketResidues5A": 0,
        "poseInterpretabilityTier": tier,
        "poseInterpretabilityReason": reason,
        "discussionSummary": f"{row.get('drug')} - {row.get('target')} cannot be interpreted at residue-contact level: {reason}.",
    }


def build_audit(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source_path = root / args.source
    final_df = pd.read_csv(source_path).fillna("")
    final_df["sotaReadyRankGlobalNumeric"] = pd.to_numeric(final_df.get("sotaReadyRankGlobal"), errors="coerce").fillna(999999999)
    selected = final_df.sort_values("sotaReadyRankGlobalNumeric").head(args.top_n).copy()

    receptor_cache: dict[str, dict[str, Any]] = {}
    ligand_cache: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []

    for _, row in selected.iterrows():
        ligand_path = abs_or_root(root, row.get("confidenceSdfPath"))
        receptor_path = abs_or_root(root, row.get("receptorPdbPath"))
        if not ligand_path:
            summary_rows.append(failed_summary(row, ligand_error="ligand_path_missing"))
            continue
        if not receptor_path:
            summary_rows.append(failed_summary(row, receptor_error="receptor_path_missing"))
            continue

        ligand_key = str(ligand_path)
        receptor_key = str(receptor_path)
        ligand = ligand_cache.get(ligand_key)
        if ligand is None:
            ligand = read_ligand(ligand_path)
            ligand_cache[ligand_key] = ligand
        receptor = receptor_cache.get(receptor_key)
        if receptor is None:
            receptor = read_receptor(receptor_path)
            receptor_cache[receptor_key] = receptor
        if not ligand.get("ok") or not receptor.get("ok"):
            summary_rows.append(
                failed_summary(
                    row,
                    ligand_error="" if ligand.get("ok") else ligand.get("error", "ligand_read_failed"),
                    receptor_error="" if receptor.get("ok") else receptor.get("error", "receptor_read_failed"),
                )
            )
            continue
        summary, contacts = candidate_contacts(row, ligand, receptor)
        summary_rows.append(summary)
        contact_rows.extend(contacts)

    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    audit_path = out_dir / "pose_interpretability_top_candidate_audit.csv"
    contact_path = out_dir / "pose_interpretability_residue_contacts.csv"
    summary_path = out_dir / "pose_interpretability_summary.json"
    direction_path = out_dir / "pose_interpretability_direction_summary.csv"
    augmented_path = final_dir / "final_priority_pose_interpretability_augmented_table.csv"
    shortlist_path = final_dir / "final_priority_pose_interpretability_expert_shortlist.csv"
    review_path = final_dir / "final_priority_pose_interpretability_review_queue.csv"
    md_path = final_dir / "FINAL_PRIORITY_POSE_INTERPRETABILITY_AUDIT.md"

    write_csv(audit_path, summary_rows)
    write_csv(contact_path, contact_rows)

    summary_by_pair = pd.DataFrame(summary_rows)
    merge_keys = ["direction", "pairId"]
    augmented = final_df.drop(columns=["sotaReadyRankGlobalNumeric"], errors="ignore").merge(
        summary_by_pair[
            [
                "direction",
                "pairId",
                "poseInterpretabilityTier",
                "poseInterpretabilityReason",
                "computedPocketResidues5A",
                "computedPocketMeanPlddt5A",
                "computedMinHeavyAtomDistance",
                "contactClassCounts",
                "topContactResidues",
                "highConfidenceContactResidues",
                "lowConfidenceContactResidues",
                "discussionSummary",
            ]
        ],
        on=merge_keys,
        how="left",
    )
    augmented.to_csv(augmented_path, index=False)

    interpretable = summary_by_pair[
        summary_by_pair["poseInterpretabilityTier"].astype(str).str.startswith(("A_", "B_"))
    ].copy()
    if not interpretable.empty:
        interpretable["rankSort"] = pd.to_numeric(interpretable["sotaReadyRankGlobal"], errors="coerce").fillna(999999999)
        interpretable = interpretable.sort_values(["rankSort", "pairId"])
    shortlist = interpretable.head(args.shortlist_n).drop(columns=["rankSort"], errors="ignore")
    shortlist.to_csv(shortlist_path, index=False)

    review = summary_by_pair[
        ~summary_by_pair["poseInterpretabilityTier"].astype(str).str.startswith(("A_", "B_"))
    ].copy()
    if not review.empty:
        review["rankSort"] = pd.to_numeric(review["sotaReadyRankGlobal"], errors="coerce").fillna(999999999)
        review = review.sort_values(["rankSort", "pairId"])
    review.head(args.review_n).drop(columns=["rankSort"], errors="ignore").to_csv(review_path, index=False)

    direction_summary = summarize_by_direction(summary_rows)
    write_csv(direction_path, direction_summary)

    summary = summarize(summary_rows, contact_rows, len(final_df), args.top_n, args.shortlist_n)
    write_json(summary_path, summary)
    md_path.write_text(markdown(summary, summary_rows, direction_summary), encoding="utf-8")
    return {
        "summary": summary,
        "audit_path": str(audit_path.relative_to(root)),
        "contact_path": str(contact_path.relative_to(root)),
        "augmented_path": str(augmented_path.relative_to(root)),
        "shortlist_path": str(shortlist_path.relative_to(root)),
        "review_path": str(review_path.relative_to(root)),
        "md_path": str(md_path.relative_to(root)),
    }


def summarize_by_direction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[str(row.get("direction") or "")].append(row)
    result = []
    for direction, group in sorted(by_direction.items()):
        tiers = Counter(row.get("poseInterpretabilityTier", "") for row in group)
        known = sum(1 for row in group if is_known_candidate(row))
        novel = sum(1 for row in group if is_novel_candidate(row))
        result.append(
            {
                "direction": direction,
                "rows": len(group),
                "tierA": tiers.get("A_mechanistically_discussable_pose", 0),
                "tierB": tiers.get("B_structurally_supported_pose", 0),
                "tierC": tiers.get("C_geometry_or_context_review_pose", 0),
                "tierD": tiers.get("D_not_interpretable_pose", 0),
                "knownDrugTargetRows": known,
                "novelRows": novel,
                "medianPocketResidues5A": median_number(row.get("computedPocketResidues5A") for row in group),
                "medianPocketMeanPlddt5A": median_number(row.get("computedPocketMeanPlddt5A") for row in group),
            }
        )
    return result


def median_number(values: Any) -> float | None:
    parsed = [number(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return round(float(np.median(parsed)), 4) if parsed else None


def summarize(
    rows: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    source_rows: int,
    top_n: int,
    shortlist_n: int,
) -> dict[str, Any]:
    tiers = Counter(row.get("poseInterpretabilityTier", "") for row in rows)
    reasons = Counter(row.get("poseInterpretabilityReason", "") for row in rows)
    classes = Counter(row.get("contactClass", "") for row in contacts)
    completed = [row for row in rows if row.get("candidateStatus") == "completed"]
    interpretable = [
        row
        for row in rows
        if str(row.get("poseInterpretabilityTier")).startswith(("A_", "B_"))
    ]
    top100 = [
        row
        for row in rows
        if (number(row.get("sotaReadyRankGlobal")) or 999999999) <= 100
    ]
    novel_interpretable = [row for row in interpretable if is_novel_candidate(row)]
    known_interpretable = [row for row in interpretable if is_known_candidate(row)]
    if len(rows) >= source_rows:
        scope = f"Residue-contact interpretability audit over all {source_rows} SOTA-ready final candidates."
    else:
        scope = f"Residue-contact interpretability audit over SOTA-ready Top{top_n} of {source_rows} final candidates."
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": scope,
        "candidateRows": len(rows),
        "completedRows": len(completed),
        "contactResidueRows": len(contacts),
        "poseInterpretabilityTierCounts": dict(tiers),
        "poseInterpretabilityReasonCounts": dict(reasons),
        "contactClassCounts": dict(classes),
        "interpretableRows": len(interpretable),
        "interpretablePct": round(pct(len(interpretable), len(rows)), 4),
        "novelInterpretableRows": len(novel_interpretable),
        "knownInterpretableRows": len(known_interpretable),
        "top100InterpretableRows": sum(
            1 for row in top100 if str(row.get("poseInterpretabilityTier")).startswith(("A_", "B_"))
        ),
        "top100TierCounts": dict(Counter(row.get("poseInterpretabilityTier", "") for row in top100)),
        "medianPocketResidues5A": median_number(row.get("computedPocketResidues5A") for row in rows),
        "medianPocketMeanPlddt5A": median_number(row.get("computedPocketMeanPlddt5A") for row in rows),
        "shortlistRows": min(shortlist_n, len(interpretable)),
        "methodNote": (
            "Contacts are computed between docked ligand heavy atoms and receptor heavy atoms within 5 A. "
            "AlphaFold pLDDT is read from the receptor PDB B-factor field. Contact classes are rule-based "
            "and intended for expert triage, not as definitive interaction annotation."
        ),
    }


def markdown(summary: dict[str, Any], rows: list[dict[str, Any]], direction_summary: list[dict[str, Any]]) -> str:
    examples = sorted(rows, key=lambda row: number(row.get("sotaReadyRankGlobal")) or 999999999)[:12]
    lines = [
        "# Final-Priority Pose Interpretability Audit",
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
        f"- Completed pose rows: {summary['completedRows']}",
        f"- Residue-contact rows: {summary['contactResidueRows']}",
        f"- Interpretable A/B pose rows: {summary['interpretableRows']} ({summary['interpretablePct']:.2f}%)",
        f"- Novel interpretable rows: {summary['novelInterpretableRows']}",
        f"- Known-target interpretable rows: {summary['knownInterpretableRows']}",
        f"- Top100 interpretable rows: {summary['top100InterpretableRows']}",
        f"- Median 5 A pocket residues: {summary['medianPocketResidues5A']}",
        f"- Median 5 A pocket mean pLDDT: {summary['medianPocketMeanPlddt5A']}",
        f"- Tier counts: {summary['poseInterpretabilityTierCounts']}",
        f"- Contact class counts: {summary['contactClassCounts']}",
        "",
        "## Direction Summary",
        "",
    ]
    for item in direction_summary:
        lines.append(
            f"- {item['direction']}: rows {item['rows']}; A/B {item['tierA'] + item['tierB']}; "
            f"C {item['tierC']}; D {item['tierD']}; median pocket pLDDT {item['medianPocketMeanPlddt5A']}."
        )
    lines.extend(["", "## Representative Discussion Examples", ""])
    for row in examples:
        lines.append(
            f"- Rank {row.get('sotaReadyRankGlobal')}: {row.get('drug')} - {row.get('target')} "
            f"({row.get('direction')}), {row.get('poseInterpretabilityTier')}. {row.get('discussionSummary')}"
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            summary["methodNote"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build residue-contact interpretability audit for high-priority poses.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_sota_ready_matrix.csv",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/pose_interpretability")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--shortlist-n", type=int, default=200)
    parser.add_argument("--review-n", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    result = build_audit(root, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
