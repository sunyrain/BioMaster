#!/usr/bin/env python3
"""Prepare signed GNINA and Boltz inputs for the 384-rerank N2/N3 increment."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger

from prepare_boltz2_calibration_338_v1 import (
    ROOT,
    payload_sha256,
    safe_token,
    sha256,
    target_template,
    yaml_payload,
)


BASE = ROOT / "outputs/unified_pair_compute_increment_384_v1"
QUEUE = BASE / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENTAL_PAIR_QUEUE_V1.csv"
TARGETS = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v2"
    / "TARGET_EVIDENCE_LAYER_ROUTING_338_V2.csv"
)
PROTOCOLS = ROOT / "outputs/strict_receptor_protocol_338_v1/FINAL_RECEPTOR_PROTOCOL_AUDIT_338.csv"
CONFORMERS = ROOT / "outputs/gnina_discovery_7511_v1/execution_inputs/FDA720_CONFORMER_LIBRARY_V1.sdf"
GNINA_CONFIG = ROOT / "configs/gnina_calibration_338_v1.yaml"
BOLTZ_THRESHOLDS = (
    ROOT
    / "outputs/boltz2_calibration_338_v1/evaluation"
    / "BOLTZ2_CALIBRATION_DEVELOPMENT_THRESHOLDS_V1.csv"
)
OUT = BASE / "execution_v1"
RDLogger.DisableLog("rdApp.*")


def clean(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str) -> int:
    return 1 + int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 2_000_000_000


def prepare_gnina(queue: pd.DataFrame) -> tuple[Path, dict]:
    selected = queue[queue["gnina_increment_required"]].copy()
    output_dir = OUT / "gnina_inputs"
    target_dir = output_dir / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    protocols = pd.read_csv(PROTOCOLS, low_memory=False).set_index("target_chembl_id")
    config = yaml.safe_load(GNINA_CONFIG.read_text(encoding="utf-8"))
    conformers = {
        molecule.GetProp("_Name"): molecule
        for molecule in Chem.SDMolSupplier(str(CONFORMERS), removeHs=False)
        if molecule is not None and molecule.HasProp("_Name")
    }
    missing = sorted(set(selected["ligand_inchikey"].astype(str)) - set(conformers))
    if missing:
        raise RuntimeError(f"Frozen conformers missing: {missing}")

    properties = [
        "pairId",
        "ligand_inchikey",
        "drug_names",
        "project_entity_ids",
        "target_chembl_id",
        "gene_symbol",
        "assay_lane",
        "novelty_lane",
        "max_tanimoto_to_target_measured_positive",
        "dta_priority_score_384",
        "incremental_queue_rank",
        "incremental_target_rank",
        "unified_target_route",
    ]
    jobs = []
    for target, group in selected.groupby("target_chembl_id", sort=True):
        protocol = protocols.loc[str(target)]
        directory = target_dir / str(target)
        directory.mkdir(parents=True, exist_ok=True)
        ligand_path = directory / "remote_increment_candidates.sdf"
        writer = Chem.SDWriter(str(ligand_path))
        for row in group.sort_values("incremental_queue_rank").itertuples(index=False):
            molecule = Chem.Mol(conformers[str(row.ligand_inchikey)])
            molecule.SetProp("_Name", str(row.pairId))
            molecule.SetProp("row_type", "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENT")
            for key in properties:
                molecule.SetProp(key, clean(getattr(row, key)))
            writer.write(molecule)
        writer.close()

        receptor = Path(str(protocol.receptor_pdbqt))
        expected_hash = str(protocol.receptor_pdbqt_sha256_final)
        observed_hash = file_sha256(receptor) if receptor.exists() else ""
        box = [
            float(getattr(protocol, column))
            for column in [
                "box_center_x",
                "box_center_y",
                "box_center_z",
                "box_size_x",
                "box_size_y",
                "box_size_z",
            ]
        ]
        output = directory / "docked.sdf"
        log = directory / "gnina.log"
        command = [
            config["execution"]["gnina_binary"],
            "-r",
            str(receptor),
            "-l",
            str(ligand_path),
            "-o",
            str(output),
            "--log",
            str(log),
            "--center_x",
            str(box[0]),
            "--center_y",
            str(box[1]),
            "--center_z",
            str(box[2]),
            "--size_x",
            str(box[3]),
            "--size_y",
            str(box[4]),
            "--size_z",
            str(box[5]),
            "--exhaustiveness",
            str(config["execution"]["exhaustiveness"]),
            "--num_modes",
            str(config["execution"]["num_modes"]),
            "--cnn_scoring",
            str(config["execution"]["cnn_scoring"]),
            "--seed",
            str(stable_seed("UNIFIED_REMOTE_INCREMENT__" + str(target))),
        ]
        ready = (
            receptor.exists()
            and expected_hash == observed_hash
            and all(math.isfinite(value) for value in box)
        )
        jobs.append(
            {
                "target_chembl_id": target,
                "gene_symbol": group["gene_symbol"].iloc[0],
                "pdb_id": protocol.pdb_id,
                "requested_candidates": len(group),
                "prepared_controls": len(group),
                "receptor_pdbqt": str(receptor),
                "receptor_sha256_expected": expected_hash,
                "receptor_sha256_observed": observed_hash,
                "ligand_sdf": str(ligand_path),
                "ligand_sdf_sha256": file_sha256(ligand_path),
                "output_sdf": str(output),
                "gnina_log": str(log),
                "box_center_x": box[0],
                "box_center_y": box[1],
                "box_center_z": box[2],
                "box_size_x": box[3],
                "box_size_y": box[4],
                "box_size_z": box[5],
                "job_status": "READY" if ready else "BLOCKED_PREPARATION",
                "command_json": json.dumps(command, ensure_ascii=False),
            }
        )
    jobs_frame = pd.DataFrame(jobs).sort_values("target_chembl_id", kind="mergesort")
    path = output_dir / "GNINA_UNIFIED_REMOTE_INCREMENT_JOBS_V1.csv"
    jobs_frame.to_csv(path, index=False)
    summary = {
        "pairs": len(selected),
        "targets": int(selected["target_chembl_id"].nunique()),
        "ready_targets": int(jobs_frame["job_status"].eq("READY").sum()),
        "blocked_targets": int(jobs_frame["job_status"].ne("READY").sum()),
    }
    return path, summary


def prepare_boltz(queue: pd.DataFrame, targets: pd.DataFrame) -> tuple[Path, dict]:
    selected = queue[queue["boltz_increment_required"]].copy()
    output_dir = OUT / "boltz_inputs"
    input_dir = output_dir / "inputs"
    template_dir = output_dir / "target_templates"
    input_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    target_columns = [
        "target_chembl_id",
        "current_target_route",
        "current_target_state",
        "current_target_qualification_strength",
        "boltz_target_qualification",
    ]
    selected = selected.drop(columns=["boltz_target_qualification"], errors="ignore").merge(
        targets[target_columns], on="target_chembl_id", how="left", validate="many_to_one"
    )
    threshold_columns = [
        "target_chembl_id",
        "development_positive_q25",
        "development_positive_median",
        "development_negative_q90",
        "development_negative_q95",
        "development_negative_q99",
    ]
    thresholds = pd.read_csv(BOLTZ_THRESHOLDS, low_memory=False)[threshold_columns]
    selected = selected.merge(
        thresholds, on="target_chembl_id", how="left", validate="many_to_one"
    )
    if not selected["boltz_target_qualification"].eq("BOLTZ_REMOTE_QUALIFIED").all():
        raise RuntimeError("Boltz queue contains a target without remote qualification")
    if selected[threshold_columns[1:]].isna().any().any():
        raise RuntimeError("Boltz thresholds are incomplete")
    selected["novelty_class"] = selected["novelty_lane"]
    selected["boltz_discovery_route"] = "UNIFIED_384_RERANK_REMOTE_N2_N3"
    selected["boltz_use_policy"] = "USE_FOR_N2_N3_TARGET_INTERNAL_RANKING"
    selected["in_current_gnina_7511"] = selected["mainline_gnina_completed"]
    selected["externalQueueRank"] = selected["incremental_queue_rank"]

    templates = {}
    for target in sorted(selected["target_chembl_id"].astype(str).unique()):
        protocol = (
            ROOT
            / "outputs/strict_receptor_protocol_338_v1/targets"
            / target
            / "protocol.json"
        )
        templates[target] = target_template(protocol, 6.0, 48, template_dir)

    rows = []
    for row in selected.sort_values("externalQueueRank").itertuples(index=False):
        data = row._asdict()
        target = str(row.target_chembl_id)
        template = templates[target]
        molecule = Chem.MolFromSmiles(str(row.ligand_smiles))
        if molecule is None:
            raise ValueError(f"Invalid ligand SMILES: {row.pairId}")
        smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        payload = yaml_payload(template, smiles, 6.0)
        yaml_name = (
            f"unified_increment_{int(row.externalQueueRank):05d}_"
            f"{safe_token(str(row.pairId))}.yaml"
        )
        yaml_path = input_dir / yaml_name
        yaml_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8"
        )
        yaml_hash = sha256(yaml_path)
        rows.append(
            {
                **data,
                "drugId": row.project_entity_ids,
                "drug": row.drug_names,
                "target": row.gene_symbol,
                "canonicalSmiles": smiles,
                "knownDrugTargetPair": False,
                "yamlFile": yaml_name,
                "yamlPath": str(yaml_path),
                "yamlSha256": yaml_hash,
                "inputSignatureSha256": payload_sha256(
                    {
                        "yamlSha256": yaml_hash,
                        "pairId": row.pairId,
                        "route": row.boltz_discovery_route,
                        "targetQualification": row.boltz_target_qualification,
                        "templateStructureSha256": template["template_structure_sha256"],
                    }
                ),
                "inputSignatureVersion": "boltz2_unified_384_rerank_increment_v1",
                "templateStructurePath": str(template["template_structure_path"]),
                "templateStructureSha256": template["template_structure_sha256"],
                "pocketContactCount": len(template["contacts"]),
                "proteinTotalResidues": sum(len(sequence) for sequence in template["sequences"].values()),
                "sequenceCropApplied": template["sequence_crop_applied"],
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest["pairId"].duplicated().any():
        raise RuntimeError("Duplicate pairId in Boltz manifest")
    path = output_dir / "BOLTZ2_UNIFIED_REMOTE_INCREMENT_INPUT_MANIFEST_V1.csv"
    manifest.to_csv(path, index=False)
    summary = {
        "pairs": len(manifest),
        "targets": int(manifest["target_chembl_id"].nunique()),
        "n2": int(manifest["novelty_class"].eq("N2_SCAFFOLD_HOP").sum()),
        "n3": int(manifest["novelty_class"].eq("N3_REMOTE").sum()),
    }
    return path, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE, low_memory=False)
    targets = pd.read_csv(TARGETS, low_memory=False)
    if len(queue) == 0 or queue["pairId"].duplicated().any():
        raise RuntimeError("Frozen incremental queue is empty or duplicated")
    gnina_path, gnina_summary = prepare_gnina(queue)
    boltz_path, boltz_summary = prepare_boltz(queue, targets)
    summary = {
        "package_name": "UNIFIED_RERANK_INCREMENT_COMPUTE_PREPARATION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS"
        if gnina_summary["blocked_targets"] == 0
        and gnina_summary["pairs"] == int(queue["gnina_increment_required"].sum())
        and boltz_summary["pairs"] == int(queue["boltz_increment_required"].sum())
        else "FAIL",
        "frozen_pairs": len(queue),
        "gnina": {**gnina_summary, "jobs": str(gnina_path)},
        "boltz": {**boltz_summary, "manifest": str(boltz_path)},
    }
    path = OUT / "UNIFIED_RERANK_INCREMENT_COMPUTE_PREPARATION_SUMMARY_V1.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise RuntimeError(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
