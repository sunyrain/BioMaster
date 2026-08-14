#!/usr/bin/env python3
"""Prepare GNINA candidate docking only for strongly qualified recovered pockets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
DTA = ROOT / "outputs/recovered_dta_720x46_v1/RECOVERED_DTA_720_X_46_EVIDENCE_MATRIX_V1.csv.gz"
QUALIFICATION = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_PREDICTED_POCKET_METRICS_23_V1.csv"
CONTROL_JOBS = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_POCKET_VALIDATION_JOBS_23_V1.csv"
OUT = ROOT / "outputs/recovered_gnina_candidate_docking_v1"
GNINA = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
RDLogger.DisableLog("rdApp.*")


def stable_seed(text: str) -> int:
    return 1 + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def make_3d(row: pd.Series) -> tuple[Chem.Mol | None, str]:
    molecule = Chem.MolFromSmiles(str(row["ligand_smiles"]))
    if molecule is None:
        return None, "RDKIT_PARSE_FAILED"
    molecule = Chem.AddHs(molecule)
    base_seed = stable_seed(str(row["candidate_pair_id"]))
    status = -1
    for attempt in range(5):
        params = AllChem.ETKDGv3()
        params.randomSeed = 1 + (base_seed + attempt * 104729) % 2_000_000_000
        params.maxIterations = 2000
        params.useRandomCoords = attempt > 0
        params.clearConfs = True
        status = AllChem.EmbedMolecule(molecule, params)
        if status == 0:
            break
    if status:
        return None, "ETKDG_FAILED"
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=300)
            optimization = "MMFF94"
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=300)
            optimization = "UFF"
    except Exception:
        optimization = "OPTIMIZATION_FAILED_GEOMETRY_RETAINED"
    molecule.SetProp("_Name", str(row["candidate_pair_id"]))
    for key in [
        "target_chembl_id", "gene_symbol", "ligand_inchikey", "drug_names", "candidate_route",
        "conplex_rank_within_target", "drugclip_rank_within_target",
    ]:
        molecule.SetProp(key, str(row[key]))
    molecule.SetProp("conformer_method", "RDKIT_ETKDGV3")
    molecule.SetProp("conformer_optimization", optimization)
    return molecule, "OK"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dta = pd.read_csv(DTA)
    qualification = pd.read_csv(QUALIFICATION)
    qualified = qualification[
        qualification["predicted_pocket_qualification"].eq(
            "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION"
        )
    ]["target_chembl_id"].tolist()
    if len(qualified) != 3:
        raise ValueError(f"Expected 3 strongly qualified pockets, found {qualified}")
    dta = dta[dta["target_chembl_id"].isin(qualified)].copy()

    selected_groups = []
    for target, group in dta.groupby("target_chembl_id", sort=True):
        concordant = group[group["dta_target_top10pct_concordant"]].copy()
        if len(concordant):
            selected = concordant.copy()
            selected["candidate_route"] = "TWO_MODEL_TARGET_TOP10PCT_CONCORDANT"
            selected["candidate_selection_note"] = "Eligible for control-calibrated candidate docking"
            selected["_minimax_rank"] = selected[["conplex_rank_within_target", "drugclip_rank_within_target"]].max(axis=1)
            selected = selected.sort_values(
                ["_minimax_rank", "dta_target_percentile_disagreement", "conplex_rank_within_target"],
                kind="mergesort",
            )
        else:
            conplex_top = group.nsmallest(10, "conplex_rank_within_target").copy()
            drugclip_top = group.nsmallest(10, "drugclip_rank_within_target").copy()
            selected = pd.concat([conplex_top, drugclip_top], ignore_index=True).drop_duplicates("ligand_inchikey")
            selected["candidate_route"] = "MODEL_DISAGREEMENT_UNION_TOP10_EACH"
            selected["candidate_selection_note"] = "Diagnostic docking only; not eligible for formal promotion without DTA concordance"
            selected["_conplex_top10"] = selected["conplex_rank_within_target"].le(10)
            selected["_drugclip_top10"] = selected["drugclip_rank_within_target"].le(10)
            selected = selected.sort_values(
                ["_conplex_top10", "_drugclip_top10", "conplex_rank_within_target", "drugclip_rank_within_target"],
                ascending=[False, False, True, True], kind="mergesort",
            )
        selected["candidate_rank_within_target_route"] = range(1, len(selected) + 1)
        selected_groups.append(selected)
    candidates = pd.concat(selected_groups, ignore_index=True)
    candidates["candidate_pair_id"] = (
        "RECOVERED_CANDIDATE_" + candidates["target_chembl_id"] + "_" + candidates["ligand_inchikey"]
    )
    candidates_path = OUT / "RECOVERED_GNINA_CANDIDATE_PANEL_V1.csv"
    candidates.to_csv(candidates_path, index=False)

    controls = pd.read_csv(CONTROL_JOBS, dtype=str).fillna("").set_index("target_chembl_id")
    jobs = []
    failures = []
    for target, group in candidates.groupby("target_chembl_id", sort=True):
        target_dir = OUT / "targets" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        ligand_sdf = target_dir / "candidates.sdf"
        writer = Chem.SDWriter(str(ligand_sdf))
        prepared = 0
        for _, row in group.iterrows():
            molecule, status = make_3d(row)
            if molecule is None:
                failures.append({"target_chembl_id": target, "candidate_pair_id": row["candidate_pair_id"], "failure": status})
                continue
            writer.write(molecule)
            prepared += 1
        writer.close()
        protocol = controls.loc[target]
        output = target_dir / "docked_candidates.sdf"
        log = target_dir / "gnina.log"
        command = [
            str(GNINA), "-r", protocol["receptor_pdbqt"], "-l", str(ligand_sdf), "-o", str(output), "--log", str(log),
            "--center_x", protocol["box_center_x"], "--center_y", protocol["box_center_y"], "--center_z", protocol["box_center_z"],
            "--size_x", protocol["box_size_x"], "--size_y", protocol["box_size_y"], "--size_z", protocol["box_size_z"],
            "--exhaustiveness", "16", "--num_modes", "5", "--cnn_scoring", "rescore", "--seed", str(stable_seed(target + "candidate")),
        ]
        jobs.append({
            "target_chembl_id": target, "gene_symbol": group.iloc[0]["gene_symbol"],
            "candidate_route": group.iloc[0]["candidate_route"],
            "requested_candidates": len(group), "prepared_controls": prepared,
            "receptor_pdbqt": protocol["receptor_pdbqt"], "ligand_sdf": str(ligand_sdf),
            "output_sdf": str(output), "gnina_log": str(log),
            "job_status": "READY" if prepared == len(group) else "BLOCKED_PREPARATION",
            "command_json": json.dumps(command, ensure_ascii=False),
        })
    jobs_frame = pd.DataFrame(jobs)
    jobs_path = OUT / "RECOVERED_GNINA_CANDIDATE_JOBS_V1.csv"
    jobs_frame.to_csv(jobs_path, index=False)
    pd.DataFrame(failures).to_csv(OUT / "RECOVERED_GNINA_CANDIDATE_CONFORMER_FAILURES_V1.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS" if jobs_frame["job_status"].eq("READY").all() else "INCOMPLETE",
        "strongly_qualified_targets": len(qualified), "candidate_pairs": len(candidates),
        "two_model_concordant_candidate_pairs": int(candidates["candidate_route"].eq("TWO_MODEL_TARGET_TOP10PCT_CONCORDANT").sum()),
        "model_disagreement_diagnostic_pairs": int(candidates["candidate_route"].eq("MODEL_DISAGREEMENT_UNION_TOP10_EACH").sum()),
        "targets_by_route": candidates.groupby("candidate_route")["target_chembl_id"].nunique().to_dict(),
        "conformer_failures": len(failures), "jobs": str(jobs_path.relative_to(ROOT)),
    }
    (OUT / "RECOVERED_GNINA_CANDIDATE_PREPARATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Candidate docking preparation incomplete")


if __name__ == "__main__":
    main()
