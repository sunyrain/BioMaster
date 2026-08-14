#!/usr/bin/env python3
"""Prepare 12x12 GNINA controls for recovered predicted pockets with adequate history."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1/RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
LABELS = ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5_official/CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
P2RANK_ATLAS = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/P2RANK_ALL_PREDICTED_POCKETS_875.csv.gz"
CONSENSUS = ROOT / "outputs/no_experimental_pocket_prediction_ch37_v1/NO_EXPERIMENTAL_POCKET_CONSENSUS_TARGETS_151_V1.csv"
FPOCKET = ROOT / "outputs/no_experimental_pocket_prediction_ch37_v1/FPOCKET_POCKET_CANDIDATES_NO_EXPERIMENTAL_146_V1.csv.gz"
OUT = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1"
TARGET_DIR = OUT / "targets"
MEEKO = ROOT / ".conda_envs/md_openmm/bin/mk_prepare_receptor.py"
OBABEL = Path("/root/miniconda3/bin/obabel")
GNINA = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
EXPECTED_TARGETS = 23
PER_CLASS = 12
RDLogger.DisableLog("rdApp.*")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return 1 + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(core, canonical=True) if core.GetNumAtoms() else "ACYCLIC"


def docking_domain(smiles: str) -> tuple[bool, dict[str, Any]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return False, {"docking_domain_reason": "RDKIT_PARSE_FAILED"}
    allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    properties = {
        "control_mw": float(Descriptors.MolWt(molecule)),
        "control_heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "control_rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "control_formal_charge": int(Chem.GetFormalCharge(molecule)),
        "control_unusual_atoms": int(sum(atom.GetAtomicNum() not in allowed for atom in molecule.GetAtoms())),
        "control_fragment_count": int(len(Chem.GetMolFrags(molecule))),
    }
    checks = {
        "MW_100_900": 100.0 <= properties["control_mw"] <= 900.0,
        "HEAVY_ATOMS_7_65": 7 <= properties["control_heavy_atoms"] <= 65,
        "ROTATABLE_BONDS_LE_20": properties["control_rotatable_bonds"] <= 20,
        "ABS_CHARGE_LE_2": abs(properties["control_formal_charge"]) <= 2,
        "ALLOWED_ELEMENTS": properties["control_unusual_atoms"] == 0,
        "SINGLE_FRAGMENT": properties["control_fragment_count"] == 1,
    }
    passed = all(checks.values())
    properties["docking_domain_reason"] = "PASS" if passed else ";".join(key for key, value in checks.items() if not value)
    return passed, properties


def select_diverse(group: pd.DataFrame, positive: bool) -> pd.DataFrame:
    work = group.copy()
    work["_potency"] = pd.to_numeric(work["max_pchembl"], errors="coerce")
    work["_documents"] = pd.to_numeric(work["document_count"], errors="coerce").fillna(0)
    work["_rows"] = pd.to_numeric(work["activity_rows"], errors="coerce").fillna(0)
    if positive:
        work = work.sort_values(["_potency", "_documents", "_rows"], ascending=[False, False, False], kind="mergesort")
    else:
        work["_inactive"] = pd.to_numeric(work["any_explicit_inactive"], errors="coerce").fillna(0)
        work = work.sort_values(["_inactive", "_potency", "_documents", "_rows"], ascending=[False, True, False, False], kind="mergesort")
    candidates: list[tuple[int, str]] = []
    selected: list[int] = []
    seen: set[str] = set()
    for index, row in work.iterrows():
        core = scaffold(clean(row["model_ligand_smiles"]))
        if not core:
            continue
        candidates.append((index, core))
        if core not in seen:
            selected.append(index)
            seen.add(core)
        if len(selected) == PER_CLASS:
            break
    if len(selected) < PER_CLASS:
        for index, _ in candidates:
            if index not in selected:
                selected.append(index)
            if len(selected) == PER_CLASS:
                break
    return work.loc[selected].copy()


def make_3d(row: pd.Series) -> tuple[Chem.Mol | None, str]:
    molecule = Chem.MolFromSmiles(clean(row["model_ligand_smiles"]))
    if molecule is None:
        return None, "RDKIT_PARSE_FAILED"
    molecule = Chem.AddHs(molecule)
    base_seed = stable_seed(clean(row["control_pair_id"]))
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
    molecule.SetProp("_Name", clean(row["control_pair_id"]))
    for key in ["target_chembl_id", "primary_gene", "parent_molecule_chembl_id", "binary_label", "calibration_label"]:
        molecule.SetProp(key, clean(row[key]))
    molecule.SetProp("conformer_method", "RDKIT_ETKDGV3")
    molecule.SetProp("conformer_optimization", optimization)
    return molecule, "OK"


def prepare_protein(source: Path, output: Path) -> dict[str, Any]:
    fixer = PDBFixer(filename=str(source))
    fixer.findMissingResidues()
    missing_residue_segments = len(fixer.missingResidues)
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    nonstandard = [f"{res.chain.id}:{res.id}:{replacement}" for res, replacement in fixer.nonstandardResidues]
    if fixer.nonstandardResidues:
        fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    with output.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    return {
        "missing_residue_segments_not_built": missing_residue_segments,
        "nonstandard_residues_replaced": ";".join(nonstandard),
        "missing_heavy_atoms_added": missing_atoms,
        "hydrogenation_ph": 7.4,
    }


def prepare_pdbqt(prepared_pdb: Path, pdbqt: Path, target_dir: Path) -> tuple[str, str]:
    env = os.environ.copy()
    env["NUMEXPR_MAX_THREADS"] = "64"
    env["NUMEXPR_NUM_THREADS"] = "64"
    with (target_dir / "meeko.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(MEEKO), "--read_pdb", str(prepared_pdb), "-o", str(pdbqt.with_suffix("")),
             "-p", str(pdbqt), "-j", str(target_dir / "receptor_parameterized.json"),
             "--charge_model", "gasteiger"],
            stdout=log, stderr=subprocess.STDOUT, check=False, env=env,
        )
    if completed.returncode == 0 and pdbqt.is_file() and pdbqt.stat().st_size > 500:
        return "MEEKO_0.7.1_GASTEIGER", ""
    with (target_dir / "openbabel_fallback.log").open("w", encoding="utf-8") as log:
        fallback = subprocess.run(
            [str(OBABEL), str(prepared_pdb), "-O", str(pdbqt), "-xr"],
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if fallback.returncode == 0 and pdbqt.is_file() and pdbqt.stat().st_size > 500:
        return "OPENBABEL_3_X_RIGID_FALLBACK", f"MEEKO_RETURN_CODE_{completed.returncode}"
    raise RuntimeError(f"PDBQT preparation failed: Meeko {completed.returncode}, OpenBabel {fallback.returncode}")


def pocket_box(row: pd.Series, p2rank: pd.DataFrame, consensus: pd.DataFrame, fpocket: pd.DataFrame) -> dict[str, float]:
    if row["computed_pocket_evidence"] == "P1_P2RANK_FPOCKET_SAME_SITE":
        matched_rank = int(float(consensus.loc[row["target_chembl_id"], "best_method_match_p2rank_rank"]))
        source = p2rank.loc[(row["uniprot_accession"], matched_rank)]
        center = np.asarray([float(source[f"p2rank_center_{axis}"]) for axis in "xyz"], dtype=float)
        positions = {int(value) for value in re.findall(r"\d+", clean(source["p2rank_residue_positions"]))}
        coordinates = []
        for line in Path(row["af_pdb_path"]).read_text(encoding="utf-8").splitlines():
            if not line.startswith("ATOM"):
                continue
            try:
                residue_position = int(line[22:26])
            except ValueError:
                continue
            if residue_position in positions:
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        method = f"P2RANK_MATCHED_RANK_{matched_rank}_DUAL_METHOD_CONFIRMED"
    else:
        source = fpocket.loc[row["target_chembl_id"]]
        center = np.asarray([float(source[f"fpocket_center_{axis}"]) for axis in "xyz"], dtype=float)
        coordinates = []
        for line in Path(source["fpocket_atom_file"]).read_text(encoding="utf-8").splitlines():
            if line.startswith("ATOM"):
                coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        method = "FPOCKET_RANK1_GEOMETRIC_RESCUE"
    xyz = np.asarray(coordinates, dtype=float)
    if xyz.shape[0] < 4:
        raise ValueError(f"Insufficient pocket coordinates for {row['target_chembl_id']}")
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    size = np.clip(extent + 10.0, 20.0, 30.0)
    return {
        "box_center_x": round(float(center[0]), 4), "box_center_y": round(float(center[1]), 4), "box_center_z": round(float(center[2]), 4),
        "box_size_x": round(float(size[0]), 4), "box_size_y": round(float(size[1]), 4), "box_size_z": round(float(size[2]), 4),
        "pocket_box_source": method,
    }


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    recovered = pd.read_csv(RECOVERED, dtype=str).fillna("")
    labels = pd.read_csv(LABELS, dtype=str).fillna("")
    labels = labels[labels["target_chembl_id"].isin(recovered["target_chembl_id"]) & labels["binary_label"].isin(["0", "1"])].copy()
    domain_rows = []
    for index, row in labels.iterrows():
        passed, properties = docking_domain(clean(row["model_ligand_smiles"]))
        domain_rows.append({"_index": index, "docking_domain_ok": passed, **properties})
    domain = pd.DataFrame(domain_rows).set_index("_index")
    labels = labels.join(domain)
    domain_exclusions = labels[~labels["docking_domain_ok"]].copy()
    labels = labels[labels["docking_domain_ok"]].copy()
    domain_exclusions.to_csv(OUT / "RECOVERED_GNINA_CONTROL_DOCKING_DOMAIN_EXCLUSIONS_V1.csv.gz", index=False, compression="gzip")
    counts = labels.groupby("target_chembl_id")["binary_label"].agg(
        positive_pool=lambda values: int((values == "1").sum()),
        negative_pool=lambda values: int((values == "0").sum()),
    ).reset_index()
    ready = counts[(counts["positive_pool"] >= PER_CLASS) & (counts["negative_pool"] >= PER_CLASS)]["target_chembl_id"]
    targets = recovered[
        recovered["target_chembl_id"].isin(ready) & recovered["af_exact_sequence_model"].eq("True")
    ].sort_values("target_chembl_id", kind="mergesort").copy()
    if len(targets) != EXPECTED_TARGETS:
        raise ValueError(f"Expected {EXPECTED_TARGETS} 12x12 exact-structure targets, found {len(targets)}")
    labels = labels[labels["target_chembl_id"].isin(targets["target_chembl_id"])].copy()
    p2rank = pd.read_csv(P2RANK_ATLAS, dtype=str).fillna("")
    p2rank["_rank"] = pd.to_numeric(p2rank["p2rank_rank"], errors="raise").astype(int)
    p2rank = p2rank.set_index(["uniprot_accession", "_rank"])
    consensus = pd.read_csv(CONSENSUS, dtype=str).fillna("").set_index("target_chembl_id")
    fpocket = pd.read_csv(FPOCKET, dtype=str).fillna("")
    fpocket = fpocket[pd.to_numeric(fpocket["fpocket_rank"], errors="coerce").eq(1)].set_index("target_chembl_id")

    selected_groups = []
    for target_id, group in labels.groupby("target_chembl_id", sort=True):
        positive = select_diverse(group[group["binary_label"].eq("1")], True)
        negative = select_diverse(group[group["binary_label"].eq("0")], False)
        selected = pd.concat([positive, negative], ignore_index=True)
        if len(positive) != PER_CLASS or len(negative) != PER_CLASS:
            raise ValueError(f"Control selection contract failed for {target_id}")
        selected["control_pair_id"] = (
            "RECOVERED_" + selected["target_chembl_id"] + "_" + selected["parent_molecule_chembl_id"]
        )
        selected_groups.append(selected)
    controls = pd.concat(selected_groups, ignore_index=True)
    controls.to_csv(OUT / "RECOVERED_GNINA_CONTROL_PANEL_23_X_24_V1.csv.gz", index=False, compression="gzip")

    jobs = []
    failures = []
    for _, target in targets.iterrows():
        target_id = target["target_chembl_id"]
        target_dir = TARGET_DIR / target_id
        target_dir.mkdir(parents=True, exist_ok=True)
        box = pocket_box(target, p2rank, consensus, fpocket)
        prior_job_path = target_dir / "job.json"
        prior_job = json.loads(prior_job_path.read_text(encoding="utf-8")) if prior_job_path.is_file() else {}
        prior_center = np.asarray([
            float(prior_job.get(f"box_center_{axis}", math.nan)) for axis in "xyz"
        ], dtype=float)
        new_center = np.asarray([box[f"box_center_{axis}"] for axis in "xyz"], dtype=float)
        pocket_protocol_changed = bool(
            prior_job and (not np.isfinite(prior_center).all() or np.linalg.norm(prior_center - new_center) > 0.01)
        )
        group = controls[controls["target_chembl_id"].eq(target_id)].copy()
        ligands_path = target_dir / "controls.sdf"
        old_control_ids: set[str] = set()
        if ligands_path.is_file():
            for molecule in Chem.SDMolSupplier(str(ligands_path), removeHs=False, sanitize=False):
                if molecule is not None and molecule.HasProp("_Name"):
                    old_control_ids.add(molecule.GetProp("_Name"))
        new_control_ids = set(group["control_pair_id"])
        control_panel_changed = bool(prior_job and old_control_ids != new_control_ids)
        if pocket_protocol_changed or control_panel_changed:
            archive_name = "archive_out_of_docking_domain_panel_v1" if control_panel_changed else "archive_top1_site_mismatch_v1"
            archive = target_dir / archive_name
            archive.mkdir(parents=True, exist_ok=True)
            for name in ["docked_controls.sdf", "run_status.json", "gnina.log", "gnina.stdout"]:
                source = target_dir / name
                destination = archive / name
                if source.exists() and not destination.exists():
                    shutil.move(str(source), str(destination))
        writer = Chem.SDWriter(str(ligands_path))
        prepared_controls = 0
        for _, control in group.iterrows():
            molecule, status = make_3d(control)
            if molecule is None:
                failures.append({"target_chembl_id": target_id, "control_pair_id": control["control_pair_id"], "failure": status})
                continue
            writer.write(molecule)
            prepared_controls += 1
        writer.close()
        source_pdb = Path(target["af_pdb_path"])
        prepared_pdb = target_dir / "receptor_protein_prepared.pdb"
        pdbqt = target_dir / "receptor.pdbqt"
        if prepared_pdb.is_file() and pdbqt.is_file() and pdbqt.stat().st_size > 500 and prior_job:
            prep_audit = {
                key: prior_job.get(key, "") for key in [
                    "missing_residue_segments_not_built", "nonstandard_residues_replaced",
                    "missing_heavy_atoms_added", "hydrogenation_ph",
                ]
            }
            receptor_method = prior_job.get("receptor_preparation_method", "REUSED_EXISTING")
            fallback_reason = prior_job.get("receptor_preparation_fallback_reason", "")
        else:
            prep_audit = prepare_protein(source_pdb, prepared_pdb)
            receptor_method, fallback_reason = prepare_pdbqt(prepared_pdb, pdbqt, target_dir)
        output_sdf = target_dir / "docked_controls.sdf"
        log_path = target_dir / "gnina.log"
        command = [
            str(GNINA), "-r", str(pdbqt), "-l", str(ligands_path), "-o", str(output_sdf), "--log", str(log_path),
            "--center_x", str(box["box_center_x"]), "--center_y", str(box["box_center_y"]), "--center_z", str(box["box_center_z"]),
            "--size_x", str(box["box_size_x"]), "--size_y", str(box["box_size_y"]), "--size_z", str(box["box_size_z"]),
            "--exhaustiveness", "16", "--num_modes", "5", "--cnn_scoring", "rescore", "--seed", str(stable_seed(target_id)),
        ]
        job_status = "READY" if prepared_controls == 2 * PER_CLASS and pdbqt.stat().st_size > 500 else "BLOCKED_PREPARATION"
        job = {
            "target_chembl_id": target_id, "gene_symbol": target["gene_symbol"],
            "computed_pocket_evidence": target["computed_pocket_evidence"],
            "requested_controls": len(group), "prepared_controls": prepared_controls,
            "positive_controls": int(group["binary_label"].eq("1").sum()), "negative_controls": int(group["binary_label"].eq("0").sum()),
            "receptor_source_pdb": str(source_pdb), "receptor_pdbqt": str(pdbqt),
            "receptor_pdbqt_sha256": sha256(pdbqt), "receptor_preparation_method": receptor_method,
            "receptor_preparation_fallback_reason": fallback_reason, "ligand_sdf": str(ligands_path),
            "output_sdf": str(output_sdf), "gnina_log": str(log_path), **box,
            **prep_audit, "pocket_protocol_changed_from_prior": pocket_protocol_changed,
            "control_panel_changed_from_prior": control_panel_changed,
            "job_status": job_status, "command_json": json.dumps(command, ensure_ascii=False),
        }
        jobs.append(job)
        (target_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"prepared {target_id} {target['gene_symbol']} controls={prepared_controls} receptor={receptor_method}", flush=True)

    jobs_frame = pd.DataFrame(jobs).sort_values("target_chembl_id", kind="mergesort")
    jobs_path = OUT / "RECOVERED_GNINA_POCKET_VALIDATION_JOBS_23_V1.csv"
    jobs_frame.to_csv(jobs_path, index=False)
    pd.DataFrame(failures).to_csv(OUT / "RECOVERED_GNINA_CONTROL_PREPARATION_FAILURES_V1.csv", index=False)
    summary = {
        "created_utc": now(), "status": "PASS" if jobs_frame["job_status"].eq("READY").all() else "INCOMPLETE",
        "scope": "23 of the exact 46 recovered targets with at least 12 historical positive and 12 historical negative ChEMBL controls",
        "targets": len(jobs_frame), "ready_targets": int(jobs_frame["job_status"].eq("READY").sum()),
        "selected_controls": len(controls), "prepared_controls": int(jobs_frame["prepared_controls"].sum()),
        "control_conformer_failures": len(failures),
        "docking_domain_excluded_labeled_pairs": len(domain_exclusions),
        "targets_with_control_panel_replacement": int(jobs_frame["control_panel_changed_from_prior"].sum()),
        "receptor_preparation_methods": {str(k): int(v) for k, v in jobs_frame["receptor_preparation_method"].value_counts().items()},
        "limitations": [
            "Predicted-pocket docking validates ranking utility only; it does not convert the site into experimental pocket evidence.",
            "AlphaFold monomer receptors omit membrane, oligomer, cofactor, and induced-fit context unless independently modeled later.",
            "Targets without a 12x12 control panel are intentionally excluded from this qualification stage.",
        ],
        "jobs": str(jobs_path.relative_to(ROOT)), "jobs_sha256": sha256(jobs_path),
    }
    (OUT / "RECOVERED_GNINA_POCKET_VALIDATION_PREPARATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError("Recovered GNINA preparation is incomplete")


if __name__ == "__main__":
    main()
