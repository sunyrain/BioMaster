#!/usr/bin/env python3
"""Evaluate the two LOXL2 Boltz-2 candidate ensembles and freeze the MD gate.

The Boltz runs intentionally used the predicted P2Rank rank-3 site.  This
evaluator does not treat a generated pose as evidence by itself: it combines
Boltz confidence, pose reproducibility after pocket alignment, and contact-set
reproducibility.  MD is authorized only when every explicit frozen gate passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1"
MANIFEST = OUTDIR / "RECOVERED_BOLTZ2_LOXL2_INPUT_MANIFEST_V1.csv"

POCKET_RESIDUES = (328, 329, 330, 331, 332, 333, 471, 474, 478, 484, 512, 513, 514, 718, 722, 723, 724, 725)
CONTACT_CUTOFF_A = 5.0

MD_THRESHOLDS = {
    "median_complex_iplddt_min": 0.50,
    "median_ligand_iptm_min": 0.55,
    "median_pocket_aligned_ligand_rmsd_A_max": 3.0,
    "median_contact_jaccard_min": 0.50,
}


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return row-vector rotation/translation mapping moving onto fixed."""
    if moving.shape != fixed.shape or moving.shape[0] < 3:
        raise ValueError(f"Invalid Kabsch arrays: moving={moving.shape}, fixed={fixed.shape}")
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    moving_zero = moving - moving_center
    fixed_zero = fixed - fixed_center
    u, _, vt = np.linalg.svd(moving_zero.T @ fixed_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = fixed_center - moving_center @ rotation
    aligned = moving @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - fixed) ** 2, axis=1))))
    return rotation, translation, rmsd


def structure_arrays(path: Path) -> dict[str, Any]:
    structure = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    model = next(structure.get_models())
    protein = model["A"]
    ligand = model["B"]

    pocket_ca: dict[int, np.ndarray] = {}
    protein_atoms: list[np.ndarray] = []
    protein_atom_residue: list[int] = []
    for residue in protein:
        residue_number = int(residue.id[1])
        if residue_number in POCKET_RESIDUES and "CA" in residue:
            pocket_ca[residue_number] = residue["CA"].coord.astype(float)
        for atom in residue:
            if atom.element.upper() == "H":
                continue
            protein_atoms.append(atom.coord.astype(float))
            protein_atom_residue.append(residue_number)

    ligand_atoms: dict[str, np.ndarray] = {}
    for residue in ligand:
        for atom in residue:
            if atom.element.upper() == "H":
                continue
            ligand_atoms[atom.name.strip()] = atom.coord.astype(float)

    if set(pocket_ca) != set(POCKET_RESIDUES):
        missing = sorted(set(POCKET_RESIDUES) - set(pocket_ca))
        raise ValueError(f"{path}: missing pocket CA residues {missing}")
    if not ligand_atoms:
        raise ValueError(f"{path}: no ligand heavy atoms")

    protein_xyz = np.asarray(protein_atoms)
    ligand_xyz = np.asarray(list(ligand_atoms.values()))
    distances = np.linalg.norm(protein_xyz[:, None, :] - ligand_xyz[None, :, :], axis=2)
    contacting_indices = np.where(distances.min(axis=1) <= CONTACT_CUTOFF_A)[0]
    contacts = {protein_atom_residue[i] for i in contacting_indices}
    return {
        "pocket_ca": pocket_ca,
        "ligand_atoms": ligand_atoms,
        "contact_residues": contacts,
        "ligand_centroid": ligand_xyz.mean(axis=0),
        "pocket_centroid": np.asarray(list(pocket_ca.values())).mean(axis=0),
    }


def find_prediction_dir(pair: pd.Series) -> Path:
    drug = str(pair["drug_names"])
    run_root = OUTDIR / "runs" / drug
    candidates = sorted(run_root.glob("boltz_results_*/predictions/*"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one prediction directory for {drug}, found {candidates}")
    return candidates[0]


def evaluate_pair(
    pair: pd.Series,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prediction_dir = find_prediction_dir(pair)
    confidence_files = sorted(prediction_dir.glob("confidence_*_model_*.json"))
    model_files = sorted(prediction_dir.glob("*_model_*.pdb"))
    affinity_files = sorted(prediction_dir.glob("affinity_*.json"))
    if len(confidence_files) != 5 or len(model_files) != 5 or len(affinity_files) != 1:
        raise RuntimeError(
            f"Incomplete Boltz ensemble for {pair['pair_id']}: "
            f"confidence={len(confidence_files)}, models={len(model_files)}, affinity={len(affinity_files)}"
        )

    structures = [structure_arrays(path) for path in model_files]
    reference = structures[0]
    reference_pocket = np.asarray([reference["pocket_ca"][r] for r in POCKET_RESIDUES])
    reference_ligand_names = sorted(reference["ligand_atoms"])
    reference_ligand = np.asarray([reference["ligand_atoms"][name] for name in reference_ligand_names])
    reference_contacts = reference["contact_residues"]
    affinity = read_json(affinity_files[0])

    rows: list[dict[str, Any]] = []
    for model_index, (model_file, confidence_file, arrays) in enumerate(
        zip(model_files, confidence_files, structures, strict=True)
    ):
        confidence = read_json(confidence_file)
        ligand_names = sorted(arrays["ligand_atoms"])
        if ligand_names != reference_ligand_names:
            raise ValueError(f"Ligand atom-name mismatch in {model_file}")
        moving_pocket = np.asarray([arrays["pocket_ca"][r] for r in POCKET_RESIDUES])
        rotation, translation, pocket_alignment_rmsd = kabsch(moving_pocket, reference_pocket)
        moving_ligand = np.asarray([arrays["ligand_atoms"][name] for name in ligand_names])
        aligned_ligand = moving_ligand @ rotation + translation
        ligand_rmsd = float(
            np.sqrt(np.mean(np.sum((aligned_ligand - reference_ligand) ** 2, axis=1)))
        )
        contacts = arrays["contact_residues"]
        union = contacts | reference_contacts
        contact_jaccard = float(len(contacts & reference_contacts) / len(union)) if union else 1.0
        pocket_contacts = contacts & set(POCKET_RESIDUES)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "target_chembl_id": pair["target_chembl_id"],
                "gene_symbol": pair["gene_symbol"],
                "drug_names": pair["drug_names"],
                "ligand_inchikey": pair["ligand_inchikey"],
                "model_index": model_index,
                "confidence_score": confidence["confidence_score"],
                "ptm": confidence["ptm"],
                "ligand_iptm": confidence["ligand_iptm"],
                "complex_plddt": confidence["complex_plddt"],
                "complex_iplddt": confidence["complex_iplddt"],
                "complex_ipde": confidence["complex_ipde"],
                "pocket_alignment_ca_rmsd_A": pocket_alignment_rmsd,
                "pocket_aligned_ligand_rmsd_to_model0_A": ligand_rmsd,
                "contact_residue_count_5A": len(contacts),
                "pocket_residue_contact_count_5A": len(pocket_contacts),
                "contact_jaccard_to_model0": contact_jaccard,
                "ligand_to_pocket_centroid_distance_A": float(
                    np.linalg.norm(arrays["ligand_centroid"] - arrays["pocket_centroid"])
                ),
                "model_path": str(model_file),
            }
        )

    frame = pd.DataFrame(rows)
    pairwise_rows: list[dict[str, Any]] = []
    for fixed_index in range(len(structures)):
        fixed = structures[fixed_index]
        fixed_pocket = np.asarray([fixed["pocket_ca"][r] for r in POCKET_RESIDUES])
        fixed_ligand_names = sorted(fixed["ligand_atoms"])
        fixed_ligand = np.asarray([fixed["ligand_atoms"][name] for name in fixed_ligand_names])
        for moving_index in range(fixed_index + 1, len(structures)):
            moving = structures[moving_index]
            moving_ligand_names = sorted(moving["ligand_atoms"])
            if moving_ligand_names != fixed_ligand_names:
                raise ValueError("Ligand atom-name mismatch across pairwise comparison")
            moving_pocket = np.asarray(
                [moving["pocket_ca"][r] for r in POCKET_RESIDUES]
            )
            rotation, translation, pocket_rmsd = kabsch(moving_pocket, fixed_pocket)
            moving_ligand = np.asarray(
                [moving["ligand_atoms"][name] for name in moving_ligand_names]
            )
            aligned_ligand = moving_ligand @ rotation + translation
            ligand_rmsd = float(
                np.sqrt(np.mean(np.sum((aligned_ligand - fixed_ligand) ** 2, axis=1)))
            )
            fixed_contacts = fixed["contact_residues"]
            moving_contacts = moving["contact_residues"]
            union = fixed_contacts | moving_contacts
            contact_jaccard = (
                float(len(fixed_contacts & moving_contacts) / len(union)) if union else 1.0
            )
            pairwise_rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "target_chembl_id": pair["target_chembl_id"],
                    "gene_symbol": pair["gene_symbol"],
                    "drug_names": pair["drug_names"],
                    "fixed_model_index": fixed_index,
                    "moving_model_index": moving_index,
                    "pocket_alignment_ca_rmsd_A": pocket_rmsd,
                    "pocket_aligned_ligand_rmsd_A": ligand_rmsd,
                    "contact_jaccard": contact_jaccard,
                }
            )
    pairwise_frame = pd.DataFrame(pairwise_rows)
    if len(pairwise_frame) != 10:
        raise RuntimeError("Expected all 10 non-self pairwise comparisons for five models")
    median_iplddt = float(frame["complex_iplddt"].median())
    median_iptm = float(frame["ligand_iptm"].median())
    median_ligand_rmsd = float(pairwise_frame["pocket_aligned_ligand_rmsd_A"].median())
    median_jaccard = float(pairwise_frame["contact_jaccard"].median())
    gates = {
        "passes_complex_iplddt": median_iplddt >= MD_THRESHOLDS["median_complex_iplddt_min"],
        "passes_ligand_iptm": median_iptm >= MD_THRESHOLDS["median_ligand_iptm_min"],
        "passes_pose_rmsd": median_ligand_rmsd <= MD_THRESHOLDS["median_pocket_aligned_ligand_rmsd_A_max"],
        "passes_contact_jaccard": median_jaccard >= MD_THRESHOLDS["median_contact_jaccard_min"],
    }
    md_authorized = all(gates.values())
    if md_authorized:
        interpretation = "BOLTZ_MD_GATE_PASS"
        next_action = "COFACTOR_COMPLETE_SYSTEM_PREPARATION_AND_RESTRAINED_MD"
    elif median_iptm >= MD_THRESHOLDS["median_ligand_iptm_min"]:
        interpretation = "MODERATE_INTERFACE_SIGNAL_BUT_LOW_COMPLEX_CONFIDENCE"
        next_action = "EXPERIMENT_ONLY_NO_MD;REBUILD_WITH_CU_AND_LTQ_IF_STRUCTURAL_FOLLOWUP_NEEDED"
    else:
        interpretation = "WEAK_OR_UNSTABLE_INTERFACE_AND_LOW_COMPLEX_CONFIDENCE"
        next_action = "EXPERIMENTAL_DEPRIORITIZATION;NO_MD"

    summary = {
        "pair_id": pair["pair_id"],
        "target_chembl_id": pair["target_chembl_id"],
        "gene_symbol": pair["gene_symbol"],
        "drug_names": pair["drug_names"],
        "ligand_inchikey": pair["ligand_inchikey"],
        "models_completed": len(frame),
        "median_ligand_iptm": median_iptm,
        "min_ligand_iptm": float(frame["ligand_iptm"].min()),
        "max_ligand_iptm": float(frame["ligand_iptm"].max()),
        "median_complex_iplddt": median_iplddt,
        "min_complex_iplddt": float(frame["complex_iplddt"].min()),
        "max_complex_iplddt": float(frame["complex_iplddt"].max()),
        "median_pocket_aligned_ligand_rmsd_A": median_ligand_rmsd,
        "max_pocket_aligned_ligand_rmsd_A": float(pairwise_frame["pocket_aligned_ligand_rmsd_A"].max()),
        "median_contact_jaccard": median_jaccard,
        "min_contact_jaccard": float(pairwise_frame["contact_jaccard"].min()),
        "median_pocket_residue_contact_count_5A": float(
            frame["pocket_residue_contact_count_5A"].median()
        ),
        "affinity_pred_value": affinity["affinity_pred_value"],
        "affinity_probability_binary": affinity["affinity_probability_binary"],
        **gates,
        "cofactor_complete": False,
        "missing_structural_context": "CU;LTQ_COVALENT_COFACTOR;EXPERIMENTAL_OR_VALIDATED_HOLO_TEMPLATE",
        "md_authorized": md_authorized,
        "boltz_interpretation": interpretation,
        "next_action": next_action,
    }
    return rows, pairwise_rows, summary


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    if len(manifest) != 2 or manifest["pair_id"].nunique() != 2:
        raise RuntimeError("Expected exactly two frozen LOXL2 candidates")
    all_samples: list[dict[str, Any]] = []
    all_pairwise: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for _, pair in manifest.iterrows():
        samples, pairwise, summary = evaluate_pair(pair)
        all_samples.extend(samples)
        all_pairwise.extend(pairwise)
        summaries.append(summary)

    sample_frame = pd.DataFrame(all_samples).sort_values(["drug_names", "model_index"])
    pairwise_frame = pd.DataFrame(all_pairwise).sort_values(
        ["drug_names", "fixed_model_index", "moving_model_index"]
    )
    summary_frame = pd.DataFrame(summaries).sort_values("drug_names")
    if len(sample_frame) != 10 or len(pairwise_frame) != 20 or summary_frame["md_authorized"].any():
        raise RuntimeError("Unexpected Boltz sample count or MD gate result")

    sample_path = OUTDIR / "RECOVERED_BOLTZ2_LOXL2_SAMPLE_METRICS_V1.csv"
    pairwise_path = OUTDIR / "RECOVERED_BOLTZ2_LOXL2_PAIRWISE_POSE_CONSISTENCY_V1.csv"
    summary_path = OUTDIR / "RECOVERED_BOLTZ2_LOXL2_PAIR_SUMMARY_V1.csv"
    json_path = OUTDIR / "RECOVERED_BOLTZ2_LOXL2_EVALUATION_SUMMARY_V1.json"
    sample_frame.to_csv(sample_path, index=False)
    pairwise_frame.to_csv(pairwise_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    payload = {
        "scope": "Two LOXL2 candidates that passed GNINA control-calibrated computational triage",
        "target_count": 1,
        "pair_count": 2,
        "boltz_models_per_pair": 5,
        "pocket_source": "LOXL2 matched P2Rank rank 3; 18 pocket residues",
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "md_gate_thresholds": MD_THRESHOLDS,
        "md_authorized_pairs": int(summary_frame["md_authorized"].sum()),
        "md_blocked_pairs": int((~summary_frame["md_authorized"]).sum()),
        "global_md_decision": "NO_MD_LOW_COMPLEX_CONFIDENCE_AND_MISSING_CU_LTQ_CONTEXT",
        "scientific_boundary": (
            "Boltz confidence and pose reproducibility are orthogonal computational triage only; "
            "neither pair is binding evidence until experimentally measured."
        ),
        "outputs": {
            "sample_metrics": str(sample_path),
            "pairwise_pose_consistency": str(pairwise_path),
            "pair_summary": str(summary_path),
        },
        "pairs": summary_frame.to_dict(orient="records"),
    }
    json_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
