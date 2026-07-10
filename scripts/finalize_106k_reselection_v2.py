#!/usr/bin/env python3
"""Merge reused and delta Boltz results, then finalize the v2 packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import (  # noqa: E402
    add_evidence_tier_v2,
    add_priority_score_v2,
    assert_unique_pairs,
    bool_series,
    build_agent_review_pool,
    file_sha256,
    select_formal_packages,
)
from scripts.build_current_production_package_v2 import assay_summary, build_teacher_table  # noqa: E402
from scripts.finalize_boltz_refined_3000_package import collect_result_files  # noqa: E402
from scripts.audit_boltz_pose_stability import audit_pair, pocket_residue_ids_from_yaml  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "current_pipeline_v2.yaml"
STAGE_DIR = ROOT / "outputs" / "current_production_package_v2" / "stage1_106k_reselection"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def code_state() -> dict[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
        ).splitlines()
        formal_files = [
            ROOT / "biomaster/production.py",
            ROOT / "configs/current_pipeline_v4.yaml",
            ROOT / "configs/target_scope_extension_v4.csv",
            ROOT / "configs/project_drugs_v4.csv",
            ROOT / "configs/project_targets_v4.csv",
            ROOT / "configs/project_targets_v4_integrity.csv",
            ROOT / "scripts/freeze_project_scope_v4.py",
            ROOT / "scripts/build_full_project_universe_v3.py",
            ROOT / "scripts/build_full_project_universe_v4.py",
            ROOT / "scripts/build_boltz2_complex_input_package.py",
            ROOT / "scripts/run_boltz2_batched_queue.py",
            ROOT / "scripts/rebuild_boltz_output_provenance_v4.py",
            ROOT / "scripts/finalize_boltz_refined_3000_package.py",
            ROOT / "scripts/audit_boltz_pose_stability.py",
            ROOT / "scripts/annotate_sequence_homology_risk.py",
            ROOT / "scripts/annotate_compound_assay_liability.py",
            ROOT / "scripts/finalize_106k_reselection_v2.py",
            ROOT / "scripts/finalize_full_universe_v4.py",
        ]
        return {
            "git_commit": commit,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_files_sha256": hashlib.sha256("\n".join(sorted(untracked)).encode("utf-8")).hexdigest(),
            "formal_source_bundle_sha256": hashlib.sha256(
                "".join(file_sha256(path) for path in formal_files if path.exists()).encode("utf-8")
            ).hexdigest(),
        }
    except Exception:  # noqa: BLE001
        return {
            "git_commit": "",
            "tracked_diff_sha256": "",
            "untracked_files_sha256": "",
            "formal_source_bundle_sha256": "",
        }


def choose_series(df: pd.DataFrame, delta_col: str, old_col: str, default: Any = "") -> pd.Series:
    delta = df.get(delta_col, pd.Series(default, index=df.index))
    old = df.get(old_col, pd.Series(default, index=df.index))
    present = delta.notna() & delta.astype(str).ne("")
    return delta.where(present, old)


def build(
    config_path: Path,
    allow_partial: bool,
    *,
    stage_dir: Path = STAGE_DIR,
    selected_filename: str = "pre_boltz_top3000_v2.csv",
    input_package_name: str = "boltz_delta_input_package",
    run_name: str = "boltz_delta_full_run",
    output_subdir: str = "formal_after_106k_reselection",
    version_label: str = "v2_complete",
    conplex_reference_path: Path | None = None,
    result_source_label: str = "v2_delta_refined",
    audit_pose_stability: bool = False,
    max_incomplete: int = 3,
    reuse_old_results: bool = True,
    require_input_signatures: bool = False,
    max_pose_incomplete: int = 0,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected_path = stage_dir / selected_filename
    manifest_path = stage_dir / input_package_name / "boltz2_input_manifest.csv"
    full_run_dir = stage_dir / run_name / "batch_runs"
    old_path = ROOT / config["inputs"]["refined_top3000"]
    all_pairs_path = ROOT / config["inputs"]["all_direct_pairs"]
    required_paths = [selected_path, manifest_path, all_pairs_path]
    if reuse_old_results:
        required_paths.append(old_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    selected = pd.read_csv(selected_path, low_memory=False).fillna("")
    manifest = pd.read_csv(manifest_path, low_memory=False).fillna("")
    old = pd.read_csv(old_path, low_memory=False).fillna("") if reuse_old_results else pd.DataFrame()
    reference_path = conplex_reference_path or all_pairs_path
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    reference = pd.read_csv(reference_path, usecols=["conplex_score"], low_memory=False)["conplex_score"]
    assert_unique_pairs(selected, "v2 selected Top3000")
    if require_input_signatures:
        sequence_contract_path = ROOT / config["inputs"]["protein_sequence_representatives"]
        integrity_manifest_path = ROOT / config["inputs"]["project_target_integrity_manifest"]
        if file_sha256(sequence_contract_path) != str(
            config["scope"]["protein_sequence_representatives_sha256"]
        ):
            raise RuntimeError("Frozen protein sequence table SHA-256 mismatch")
        if file_sha256(integrity_manifest_path) != str(
            config["scope"]["project_target_integrity_manifest_sha256"]
        ):
            raise RuntimeError("Frozen target integrity manifest SHA-256 mismatch")
        integrity_manifest = pd.read_csv(integrity_manifest_path, low_memory=False).fillna("")
        if len(integrity_manifest) != 463 or integrity_manifest["sequence_key"].duplicated().any():
            raise RuntimeError("Frozen target integrity manifest row contract failed")
        if len(selected) != int(config["contracts"]["refined_top3000_rows"]):
            raise RuntimeError(f"Selected Top3000 row contract failed: {len(selected)}")
        expected_reference_rows = int(
            config["contracts"].get(
                "project_physical_pair_rows", config["contracts"]["project_cartesian_rows"]
            )
        )
        if len(reference) != expected_reference_rows:
            raise RuntimeError(f"ConPLEx project reference row contract failed: {len(reference)}")
        required_signature_columns = {
            "pairId",
            "yamlFile",
            "yamlSha256",
            "modelLigandSmiles",
            "sourceRowSha256",
            "proteinSequenceSha256",
            "receptorPdbPath",
            "receptorPdbSha256",
            "pocketConstraintSha256",
            "inputSignatureSha256",
            "inputSignatureVersion",
        }
        missing_signature_columns = required_signature_columns - set(manifest.columns)
        if missing_signature_columns:
            raise RuntimeError(f"Boltz manifest lacks signature columns: {sorted(missing_signature_columns)}")
        if set(manifest["pairId"].astype(str)) != set(selected["pair_id"].astype(str)):
            raise RuntimeError("Signed Boltz manifest does not exactly cover selected Top3000")
        if manifest["pairId"].duplicated().any() or manifest["inputSignatureSha256"].duplicated().any():
            raise RuntimeError("Signed Boltz manifest contains duplicate pair or input signatures")
        selected_ligands = selected.set_index("pair_id")["model_ligand_smiles"].astype(str).to_dict()
        for _, row in manifest.iterrows():
            yaml_path = stage_dir / input_package_name / "inputs" / str(row["yamlFile"])
            if not yaml_path.exists() or file_sha256(yaml_path) != str(row["yamlSha256"]):
                raise RuntimeError(f"Boltz YAML signature mismatch: {yaml_path}")
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            sequences = payload.get("sequences") or []
            protein_sequence = str(((sequences[0] if sequences else {}).get("protein") or {}).get("sequence") or "")
            ligand_smiles = str(((sequences[1] if len(sequences) > 1 else {}).get("ligand") or {}).get("smiles") or "")
            protein_sha = hashlib.sha256(protein_sequence.encode("utf-8")).hexdigest()
            receptor_path = Path(str(row["receptorPdbPath"]))
            if not receptor_path.is_absolute():
                receptor_path = ROOT / receptor_path
            receptor_sha = file_sha256(receptor_path) if receptor_path.is_file() else ""
            pocket_sha = payload_sha256(payload.get("constraints", []))
            source_sha = payload_sha256(
                {
                    "externalQueueRank": int(float(row["externalQueueRank"])),
                    "pairId": str(row["pairId"]),
                    "modelLigandSmiles": ligand_smiles,
                    "proteinSequenceSha256": protein_sha,
                    "receptorPdbPath": str(receptor_path),
                    "receptorPdbSha256": receptor_sha,
                    "pocketConstraintSha256": pocket_sha,
                }
            )
            input_sha = payload_sha256(
                {
                    "yamlSha256": str(row["yamlSha256"]),
                    "sourceRowSha256": source_sha,
                    "proteinSequenceSha256": protein_sha,
                    "receptorPdbSha256": receptor_sha,
                    "pocketConstraintSha256": pocket_sha,
                    "modelLigandSmiles": ligand_smiles,
                }
            )
            expected_values = {
                "modelLigandSmiles": ligand_smiles,
                "sourceRowSha256": source_sha,
                "proteinSequenceSha256": protein_sha,
                "receptorPdbSha256": receptor_sha,
                "pocketConstraintSha256": pocket_sha,
                "inputSignatureSha256": input_sha,
                "inputSignatureVersion": "boltz_complete_input_sha256_v2",
            }
            for column, expected_value in expected_values.items():
                if str(row[column]) != str(expected_value):
                    raise RuntimeError(f"Boltz input signature field mismatch for {row['pairId']}: {column}")
            if selected_ligands[str(row["pairId"])] != ligand_smiles:
                raise RuntimeError(f"Selected/model ligand mismatch for {row['pairId']}")
        run_plan_path = stage_dir / run_name / "run_plan.json"
        if not run_plan_path.exists():
            raise FileNotFoundError(run_plan_path)
        run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
        expected_parameters = config.get("boltz_contract", {})
        if run_plan.get("runParameters") != expected_parameters:
            raise RuntimeError(
                f"Boltz run parameters do not match contract: {run_plan.get('runParameters')} != {expected_parameters}"
            )
        manifest_bundle = [
            {
                "pairId": str(row["pairId"]),
                "yamlSha256": str(row["yamlSha256"]),
                "inputSignatureSha256": str(row["inputSignatureSha256"]),
            }
            for _, row in manifest.assign(
                _rank=pd.to_numeric(manifest["externalQueueRank"], errors="coerce")
            ).sort_values("_rank").iterrows()
        ]
        if run_plan.get("inputManifestSha256") != file_sha256(manifest_path):
            raise RuntimeError("Boltz run plan manifest SHA-256 mismatch")
        if run_plan.get("inputBundleSha256") != payload_sha256(manifest_bundle):
            raise RuntimeError("Boltz run plan input bundle SHA-256 mismatch")
        batch_plan_path = stage_dir / run_name / "batch_plan.csv"
        batch_status_path = stage_dir / run_name / "batch_status.csv"
        if not batch_plan_path.exists() or not batch_status_path.exists():
            raise FileNotFoundError("Boltz batch plan/status provenance is incomplete")
        if run_plan.get("batchPlanSha256") != file_sha256(batch_plan_path):
            raise RuntimeError("Boltz batch plan SHA-256 mismatch")
        batch_plan = pd.read_csv(batch_plan_path, low_memory=False).fillna("")
        batch_status = pd.read_csv(batch_status_path, low_memory=False).fillna("")
        batch_count_ok = len(batch_status) <= len(batch_plan) if allow_partial else len(batch_status) == len(batch_plan)
        if not batch_count_ok or not batch_status["status"].eq("success").all():
            raise RuntimeError("Boltz batch completion contract failed")
        plan_signatures = batch_plan.set_index("batch")["batchInputSignature"].astype(str).to_dict()
        if any(
            str(row["batchInputSignature"]) != plan_signatures.get(str(row["batch"]), "")
            for _, row in batch_status.iterrows()
        ):
            raise RuntimeError("Boltz batch status signatures do not match the batch plan")
        model_environment = run_plan.get("modelEnvironment") or {}
        if not model_environment.get("boltz") or not model_environment.get("modelArtifacts"):
            raise RuntimeError("Boltz model environment/checkpoint provenance is incomplete")
        provenance_path = stage_dir / run_name / "result_provenance.csv"
        if not provenance_path.exists():
            raise FileNotFoundError(provenance_path)
        provenance = pd.read_csv(provenance_path, low_memory=False).fillna("")
        provenance_count_ok = len(provenance) <= len(manifest) if allow_partial else len(provenance) == len(manifest)
        if not provenance_count_ok or provenance["pairId"].duplicated().any():
            raise RuntimeError("Boltz result provenance row contract failed")
        provenance_pairs = set(provenance["pairId"].astype(str))
        manifest_pairs = set(manifest["pairId"].astype(str))
        if (allow_partial and not provenance_pairs.issubset(manifest_pairs)) or (
            not allow_partial and provenance_pairs != manifest_pairs
        ):
            raise RuntimeError("Boltz result provenance does not match the signed manifest")
        completed_provenance = provenance["resultCompletedVerified"].astype(str).str.lower().isin(
            {"true", "1", "1.0"}
        )
        if not completed_provenance.all():
            raise RuntimeError("Boltz result provenance includes incomplete rows")
        manifest_signatures = manifest.set_index("pairId")["inputSignatureSha256"].astype(str).to_dict()
        if any(
            str(row["inputSignatureSha256"]) != manifest_signatures[str(row["pairId"])]
            for _, row in provenance.iterrows()
        ):
            raise RuntimeError("Boltz result provenance input signatures do not match the manifest")
        if not provenance["runParameterSignature"].astype(str).eq(
            str(run_plan["runParameterSignature"])
        ).all():
            raise RuntimeError("Boltz result provenance run parameter signatures are inconsistent")
        scope_summary_path = stage_dir / "full_untruncated_universe_v4_summary.json"
        scope_summary = json.loads(scope_summary_path.read_text(encoding="utf-8"))
        scope_checks = {
            "project_drugs": 750,
            "project_targets": 463,
            "project_cartesian_rows": int(config["contracts"]["project_cartesian_rows"]),
            "selected_rows": int(config["contracts"]["refined_top3000_rows"]),
            "delta_rows_requiring_boltz": int(config["contracts"]["refined_top3000_rows"]),
            "active_moiety_target_duplicates_selected": 0,
        }
        for key, expected_value in scope_checks.items():
            if scope_summary.get(key) != expected_value:
                raise RuntimeError(f"Upstream scope contract mismatch: {key}")
        if scope_summary.get("legacy_recall_gate_applied") or not scope_summary.get("boltz_reuse_disabled"):
            raise RuntimeError("Upstream scope gate/reuse contract mismatch")
    else:
        run_plan_path = stage_dir / run_name / "run_plan.json"

    delta_results = collect_result_files([(result_source_label, full_run_dir)])
    if delta_results.empty and not allow_partial:
        raise RuntimeError("No v2 delta Boltz results found")
    if require_input_signatures and not allow_partial:
        extended_provenance_path = stage_dir / run_name / "result_provenance_with_output_hashes.csv"
        extended_summary_path = stage_dir / run_name / "result_provenance_with_output_hashes.summary.json"
        if not extended_provenance_path.exists() or not extended_summary_path.exists():
            raise FileNotFoundError(
                "Output-hash provenance must be rebuilt before formal finalization"
            )
        extended = pd.read_csv(extended_provenance_path, low_memory=False).fillna("")
        if len(extended) != len(manifest) or extended["pairId"].duplicated().any():
            raise RuntimeError("Output-hash provenance row contract failed")
        extended_summary = json.loads(extended_summary_path.read_text(encoding="utf-8"))
        if extended_summary.get("output_provenance_sha256") != file_sha256(extended_provenance_path):
            raise RuntimeError("Output-hash provenance summary SHA-256 mismatch")
        if set(extended["pairId"].astype(str)) != set(manifest["pairId"].astype(str)):
            raise RuntimeError("Output-hash provenance does not cover the signed manifest")
        expected_input_signatures = manifest.set_index("pairId")["inputSignatureSha256"].astype(str)
        if any(
            str(row["inputSignatureSha256"])
            != expected_input_signatures.get(str(row["pairId"]), "")
            for _, row in extended.iterrows()
        ):
            raise RuntimeError("Output-hash provenance input signatures do not match")
        hash_columns = [
            "boltz_confidence_sha256_refined",
            "boltz_affinity_sha256_refined",
            "boltz_cif_model0_sha256_refined",
            "boltz_cif_model1_sha256_refined",
        ]
        observed_hashes = extended.set_index("boltz_stem")[hash_columns].astype(str)
        recomputed_hashes = delta_results.set_index("boltz_stem")[hash_columns].astype(str)
        if not observed_hashes.sort_index().equals(recomputed_hashes.sort_index()):
            raise RuntimeError("Recomputed Boltz output hashes differ from formal provenance")
    manifest["boltz_stem"] = manifest["yamlFile"].astype(str).map(lambda value: Path(value).stem)
    if delta_results.empty:
        manifest_columns = [
            column
            for column in ["pairId", "boltz_stem", "yamlSha256", "modelLigandSmiles", "inputSignatureVersion"]
            if column in manifest.columns
        ]
        delta_map = manifest[manifest_columns].copy()
    else:
        if delta_results["boltz_stem"].duplicated().any():
            raise RuntimeError("Duplicate Boltz result stems")
        manifest_columns = [
            column
            for column in ["pairId", "boltz_stem", "yamlSha256", "modelLigandSmiles", "inputSignatureVersion"]
            if column in manifest.columns
        ]
        delta_map = manifest[manifest_columns].merge(
            delta_results, on="boltz_stem", how="left", validate="one_to_one"
        )
    delta_map = delta_map.rename(columns={"pairId": "pair_id"})

    old_columns = [
        "drug_chembl_id",
        "sequence_key",
        "boltz_completed_refined",
        "boltz_confidence_score_refined",
        "boltz_ligand_iptm_refined",
        "boltz_complex_iplddt_refined",
        "boltz_affinity_pred_value_refined",
        "boltz_affinity_probability_refined",
        "boltz_support_tier_refined",
        "boltz_support_reason_refined",
        "boltz_composite_score_refined",
        "boltz_run_source",
        "boltz_confidence_path_refined",
        "boltz_affinity_path_refined",
        "boltz_cif_path_refined",
        "boltz_cif_model1_path_refined",
        "boltz_output_integrity_reason",
        "boltz_confidence_sha256_refined",
        "boltz_affinity_sha256_refined",
        "boltz_cif_model0_sha256_refined",
        "boltz_cif_model1_sha256_refined",
    ]
    if reuse_old_results:
        old_columns = [column for column in old_columns if column in old.columns]
        old_for_merge = old[old_columns].drop_duplicates(["drug_chembl_id", "sequence_key"])
    else:
        old_for_merge = pd.DataFrame(columns=["drug_chembl_id", "sequence_key"])
    combined = selected.merge(
        old_for_merge,
        on=["drug_chembl_id", "sequence_key"],
        how="left",
        suffixes=("", "_old"),
        validate="one_to_one",
    ).merge(delta_map, on="pair_id", how="left", suffixes=("", "_delta"), validate="one_to_one")

    result_columns = [
        "boltz_completed_refined",
        "boltz_confidence_score_refined",
        "boltz_ligand_iptm_refined",
        "boltz_complex_iplddt_refined",
        "boltz_affinity_pred_value_refined",
        "boltz_affinity_probability_refined",
        "boltz_support_tier_refined",
        "boltz_support_reason_refined",
        "boltz_composite_score_refined",
        "boltz_run_source",
        "boltz_confidence_path_refined",
        "boltz_affinity_path_refined",
        "boltz_cif_path_refined",
        "boltz_cif_model1_path_refined",
        "boltz_output_integrity_reason",
        "boltz_confidence_sha256_refined",
        "boltz_affinity_sha256_refined",
        "boltz_cif_model0_sha256_refined",
        "boltz_cif_model1_sha256_refined",
    ]
    for column in result_columns:
        delta_name = f"{column}_delta" if f"{column}_delta" in combined.columns else column
        old_name = column if column in combined.columns else f"{column}_old"
        combined[column] = choose_series(combined, delta_name, old_name)
    combined["boltz_completed_refined"] = bool_series(combined, "boltz_completed_refined")

    sequence_path = ROOT / config["inputs"]["protein_sequence_representatives"]
    if not sequence_path.exists():
        raise FileNotFoundError(sequence_path)
    sequence_table = pd.read_csv(sequence_path, low_memory=False).fillna("")
    expected_sequence_sha = sequence_table.set_index("sequence_key")["sequence"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    used_sequence_sha = manifest.set_index("pairId")["proteinSequenceSha256"].astype(str)
    combined["expected_protein_sequence_sha256_v4"] = combined["sequence_key"].map(expected_sequence_sha)
    combined["boltz_protein_sequence_sha256_v4"] = combined["pair_id"].map(used_sequence_sha)
    combined["structure_sequence_mismatch_v4"] = (
        combined["expected_protein_sequence_sha256_v4"].astype(str)
        != combined["boltz_protein_sequence_sha256_v4"].astype(str)
    )
    if combined["expected_protein_sequence_sha256_v4"].astype(str).eq("").any():
        raise RuntimeError("One or more Top3000 targets lack a frozen expected protein sequence")
    if require_input_signatures:
        frozen_status = integrity_manifest.set_index("sequence_key")["sequence_match_status"].astype(str)
        declared_mismatch = combined["sequence_key"].map(frozen_status).ne("exact_match")
        if not declared_mismatch.equals(combined["structure_sequence_mismatch_v4"]):
            raise RuntimeError("Computed sequence mismatch differs from frozen target integrity manifest")
    missing = ~combined["boltz_completed_refined"]
    combined.loc[missing, "boltz_support_tier_refined"] = "U_boltz_not_completed"
    combined.loc[missing, "boltz_support_reason_refined"] = "missing_reused_or_delta_refined_output"
    completed_rows = int((~missing).sum())
    if not allow_partial and completed_rows < len(combined) - max_incomplete:
        raise RuntimeError(f"Only {completed_rows}/{len(combined)} Top3000 rows have refined Boltz results")

    if audit_pose_stability:
        pose_rows: list[dict[str, Any]] = []
        pose_cache: dict[str, dict[str, Any]] = {}
        yaml_by_pair = manifest.set_index("pairId")["yamlFile"].astype(str).to_dict()
        ligand_by_pair = combined.set_index("pair_id").get(
            "model_ligand_smiles", pd.Series(dtype=str)
        ).astype(str).to_dict()
        pose_inputs = zip(
            combined.get("pair_id", pd.Series("", index=combined.index)).astype(str),
            combined.get("boltz_cif_path_refined", pd.Series("", index=combined.index)).astype(str),
            strict=True,
        )
        for pair_id, value in pose_inputs:
            if not value:
                pose_rows.append({"pose_stability_completed": False, "pose_stability_reason": "missing_model_0_path"})
                continue
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            yaml_path = stage_dir / input_package_name / "inputs" / yaml_by_pair.get(pair_id, "")
            ligand_smiles = ligand_by_pair.get(pair_id, "")
            key = f"{path}__{yaml_path}__{ligand_smiles}"
            if key not in pose_cache:
                pocket_residue_ids = pocket_residue_ids_from_yaml(yaml_path)
                pose_cache[key] = audit_pair(
                    path, pocket_residue_ids, ligand_smiles=ligand_smiles
                )
            pose_rows.append(pose_cache[key])
        pose_annotations = pd.DataFrame(pose_rows, index=combined.index)
        for column in pose_annotations.columns:
            combined[column] = pose_annotations[column]
        pose_incomplete = int((~bool_series(combined, "pose_stability_completed")).sum())
        if not allow_partial and pose_incomplete > max_pose_incomplete:
            raise RuntimeError(
                f"Pose stability incomplete for {pose_incomplete}/{len(combined)} rows; "
                f"contract allows {max_pose_incomplete}"
            )

    if "sequence_homology_extension_risk" in combined.columns:
        homology_risk = bool_series(combined, "sequence_homology_extension_risk")
        combined["family_or_rediscovery_risk_v2"] = (
            bool_series(combined, "family_or_rediscovery_risk_v2") | homology_risk
        )
        existing_notes = combined.get("risk_notes_v2", pd.Series("", index=combined.index)).fillna("").astype(str)
        combined["risk_notes_v2"] = existing_notes
        needs_note = homology_risk & ~existing_notes.str.contains("sequence_homology_extension", regex=False)
        combined.loc[needs_note, "risk_notes_v2"] = (
            existing_notes.loc[needs_note].str.rstrip(";")
            + existing_notes.loc[needs_note].ne("").map({True: ";", False: ""})
            + "sequence_homology_extension"
        )
        combined.loc[homology_risk, "candidate_role_v2"] = "family_extension_or_rediscovery_control"

    if version_label.startswith("v4"):
        predictions = pd.read_csv(
            ROOT / config["inputs"]["full_conplex_predictions"],
            sep="\t",
            header=None,
            names=["drug_chembl_id", "sequence_key", "conplex_score"],
        )
        drug_manifest = pd.read_csv(
            ROOT / config["inputs"]["project_drug_manifest"], low_memory=False
        ).fillna("")
        target_manifest = pd.read_csv(
            ROOT / config["inputs"]["project_target_manifest"], low_memory=False
        ).fillna("")
        active_by_drug = drug_manifest.set_index("drug_chembl_id")["model_ligand_smiles"].astype(str)
        physical = predictions[
            predictions["drug_chembl_id"].isin(active_by_drug.index)
            & predictions["sequence_key"].isin(target_manifest["sequence_key"])
        ].copy()
        physical["_active_key"] = physical["drug_chembl_id"].map(active_by_drug)
        spread = physical.groupby(["_active_key", "sequence_key"])["conplex_score"].agg(
            lambda values: float(values.max()) - float(values.min())
        )
        if (spread > 1e-8).any():
            raise RuntimeError("Equivalent model-ligand structures have inconsistent ConPLEx scores")
        physical = physical.drop_duplicates(["_active_key", "sequence_key"]).copy()
        physical["target_rank_active_collapsed_v4"] = physical.groupby("sequence_key")[
            "conplex_score"
        ].rank(method="average", ascending=False)
        expected_physical = int(config["contracts"]["project_physical_pair_rows"])
        if len(physical) != expected_physical:
            raise RuntimeError(
                f"Active-collapsed physical space mismatch: {len(physical)} != {expected_physical}"
            )
        rank_lookup = physical.set_index(["_active_key", "sequence_key"])[
            "target_rank_active_collapsed_v4"
        ]
        combined["target_rank_id_weighted_v4"] = combined.get("target_rank", "")
        pair_index = pd.MultiIndex.from_arrays(
            [combined["model_ligand_smiles"].astype(str), combined["sequence_key"].astype(str)]
        )
        combined["target_rank_active_collapsed_v4"] = rank_lookup.reindex(pair_index).to_numpy()
        if combined["target_rank_active_collapsed_v4"].isna().any():
            raise RuntimeError("Missing active-collapsed target rank for one or more Top3000 rows")
        combined["target_rank"] = combined["target_rank_active_collapsed_v4"]
        combined["target_pair_count_in_project_space"] = int(active_by_drug.nunique())

    combined = add_priority_score_v2(combined, reference)
    combined = add_evidence_tier_v2(combined)
    combined["candidate_action_status"] = "unknown_requires_functional_assay"
    combined["pair_evidence_limit"] = (
        "computational_binding_priority_not_validated_affinity_or_disease_efficacy"
    )
    combined["default_assay_strategy"] = combined["target_assay_family_v2"].map(
        {
            "kinase": "biochemical kinase assay + orthogonal binding/engagement counterscreen",
            "enzyme": "purified enzyme activity + orthogonal binding/engagement assay",
            "transporter": "substrate transport assay + uptake/efflux counterscreen",
            "nuclear_epigenetic": "cofactor/reporter assay + CETSA or biochemical engagement",
            "ion_channel": "electrophysiology/flux assay + membrane liability counterscreen",
            "other_assayable": "target-specific biochemical engagement + orthogonal counterscreen",
        }
    ).fillna("target-specific engagement and orthogonal counterscreen")
    if version_label.startswith("v4"):
        combined["priority_score_v4"] = combined["priority_score_v2"]
        combined["pair_specific_evidence_score_v4"] = combined["pair_specific_evidence_score_v2"]
        combined["evidence_tier_v4"] = combined["evidence_tier_v2"]
        combined["selection_contract_version"] = "active_moiety_physics_first_v4"
    out_dir = ROOT / config["outputs"]["directory"] / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    if allow_partial:
        checkpoint = out_dir / f"refined_top3000_checkpoint_{version_label}.csv"
        combined.to_csv(checkpoint, index=False)
        summary = {
            "created_utc": now_utc(),
            "status": "checkpoint_not_formal",
            "top3000_rows": int(len(combined)),
            "refined_completed_rows": completed_rows,
            "refined_incomplete_rows": int(missing.sum()),
            "delta_result_rows_found": int(len(delta_results)),
            "pose_stability_audited": bool(audit_pose_stability),
            "formal_outputs_written": False,
            "checkpoint_sha256": file_sha256(checkpoint),
            "version_label": version_label,
            "code_state": code_state(),
        }
        (out_dir / f"checkpoint_summary_{version_label}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    final1000, final384 = select_formal_packages(combined, config["selection"])
    review_pool = (
        build_agent_review_pool(final1000, config["selection"]["review_pool"])
        if "review_pool" in config["selection"]
        else final384.copy()
    )
    if len(final1000) != 1000 or len(final384) != 384:
        raise RuntimeError(f"Incomplete formal package: final1000={len(final1000)}, final384={len(final384)}")
    expected_review_pool = int(config.get("contracts", {}).get("review_pool_rows", len(review_pool)))
    if len(review_pool) != expected_review_pool:
        raise RuntimeError(
            f"Incomplete review pool: rows={len(review_pool)}, expected={expected_review_pool}"
        )
    assert_unique_pairs(final1000, "final1000 after delta")
    assert_unique_pairs(final384, "final384 after delta")
    assert_unique_pairs(review_pool, "agent review pool after delta")
    if not set(final384["pair_id"]).issubset(set(final1000["pair_id"])):
        raise RuntimeError("final384 is not nested in final1000")
    if not set(review_pool["pair_id"]).issubset(set(final1000["pair_id"])):
        raise RuntimeError("agent review pool is not nested in final1000")
    for label, frame in [
        ("final1000", final1000),
        ("final384", final384),
        ("agent review pool", review_pool),
    ]:
        if bool_series(frame, "exact_known_target_v2").any():
            raise RuntimeError(f"{label} contains exact known target pairs")
        if bool_series(frame, "family_or_rediscovery_risk_v2").any():
            raise RuntimeError(f"{label} contains family-extension or rediscovery risks")
        if bool_series(frame, "severe_compound_liability").any():
            raise RuntimeError(f"{label} contains severe compound liabilities")
        if bool_series(frame, "structure_sequence_mismatch_v4").any():
            raise RuntimeError(f"{label} contains a protein/template sequence mismatch")
        compound = frame.get("active_moiety_smiles", pd.Series("", index=frame.index)).astype(str)
        compound = compound.where(
            compound.ne(""), frame.get("canonical_smiles_rdkit", pd.Series("", index=frame.index)).astype(str)
        )
        compound = compound.where(compound.ne(""), frame["drug_chembl_id"].astype(str))
        if pd.DataFrame({"compound": compound, "target": frame["sequence_key"]}).duplicated().any():
            raise RuntimeError(f"{label} contains duplicate active-moiety x target rows")
    if audit_pose_stability and not final384["pose_stability_tier"].isin(
        ["A_stable_conditional_pose", "B_moderate_conditional_pose"]
    ).all():
        raise RuntimeError("final384 contains unsupported conditional poses")
    if audit_pose_stability and not review_pool["pose_stability_tier"].isin(
        ["A_stable_conditional_pose", "B_moderate_conditional_pose"]
    ).all():
        raise RuntimeError("agent review pool contains unsupported conditional poses")

    top3000_output = out_dir / f"refined_top3000_{version_label}.csv"
    final1000_output = out_dir / f"final1000_candidates_{version_label}.csv"
    final384_output = out_dir / f"final384_nomination_{version_label}.csv"
    review_pool_output = out_dir / f"agent_review_pool_{version_label}.csv"
    combined.to_csv(top3000_output, index=False)
    final1000.to_csv(final1000_output, index=False)
    final384.to_csv(final384_output, index=False)
    review_pool.to_csv(review_pool_output, index=False)
    build_teacher_table(final1000).to_csv(out_dir / f"final1000_teacher_readable_zh_{version_label}.csv", index=False)
    build_teacher_table(final384).to_csv(out_dir / f"final384_teacher_readable_zh_{version_label}.csv", index=False)
    build_teacher_table(review_pool).to_csv(
        out_dir / f"agent_review_pool_teacher_readable_zh_{version_label}.csv", index=False
    )
    assay = assay_summary(final384)
    assay.to_csv(out_dir / f"final384_assay_group_summary_{version_label}.csv", index=False)
    with pd.ExcelWriter(out_dir / f"FORMAL_PRODUCTION_PACKAGE_{version_label.upper()}.xlsx", engine="openpyxl") as writer:
        build_teacher_table(final384).to_excel(writer, index=False, sheet_name="384_nomination")
        build_teacher_table(final1000).to_excel(writer, index=False, sheet_name="final1000")
        build_teacher_table(review_pool).to_excel(writer, index=False, sheet_name="review_pool")
        assay.to_excel(writer, index=False, sheet_name="assay_groups")

    summary = {
        "created_utc": now_utc(),
        "top3000_rows": int(len(combined)),
        "refined_completed_rows": completed_rows,
        "refined_incomplete_rows": int(missing.sum()),
        "delta_result_rows_found": int(len(delta_results)),
        "sequence_homology_extension_risk_rows": int(
            bool_series(combined, "sequence_homology_extension_risk").sum()
        ),
        "structure_sequence_mismatch_rows": int(
            bool_series(combined, "structure_sequence_mismatch_v4").sum()
        ),
        "pose_stability_audited": bool(audit_pose_stability),
        "max_pose_incomplete_contract": int(max_pose_incomplete),
        "pose_stability_completed_rows": int(bool_series(combined, "pose_stability_completed").sum()),
        "pose_stability_tiers": combined.get(
            "pose_stability_tier", pd.Series("", index=combined.index)
        ).value_counts().to_dict(),
        "pose_ligand_rmsd_methods": combined.get(
            "pose_ligand_rmsd_method", pd.Series("", index=combined.index)
        ).value_counts().to_dict(),
        "final1000_rows": int(len(final1000)),
        "final1000_boltz_ab": int(final1000["boltz_support_tier_refined"].astype(str).str.startswith(("A_", "B_")).sum()),
        "final384_rows": int(len(final384)),
        "agent_review_pool_rows": int(len(review_pool)),
        "agent_review_pool_unique_drugs": int(review_pool["drug_chembl_id"].nunique()),
        "agent_review_pool_unique_targets": int(review_pool["primary_gene"].nunique()),
        "final384_unique_drugs": int(final384["drug_chembl_id"].nunique()),
        "final384_unique_targets": int(final384["primary_gene"].nunique()),
        "final384_tiers": final384["evidence_tier_v2"].value_counts().to_dict(),
        "final1000_unique_drugs": int(final1000["drug_chembl_id"].nunique()),
        "final1000_unique_targets": int(final1000["primary_gene"].nunique()),
        "final1000_assay_families": final1000["target_assay_family_v2"].value_counts().to_dict(),
        "final384_assay_families": final384["target_assay_family_v2"].value_counts().to_dict(),
        "final384_boltz_review_classes": final384["boltz_review_class_v3"].value_counts().to_dict(),
        "final384_assay_interference_review_rows": int(
            bool_series(final384, "assay_interference_review").sum()
        ),
        "allow_partial": bool(allow_partial),
        "max_incomplete_contract": int(max_incomplete),
        "legacy_boltz_reuse_enabled": bool(reuse_old_results),
        "input_signatures_required": bool(require_input_signatures),
        "conplex_reference_path": str(reference_path),
        "conplex_reference_rows": int(len(reference)),
        "source_sha256": {
            "selected_top3000": file_sha256(selected_path),
            "boltz_input_manifest": file_sha256(manifest_path),
            "reused_refined_top3000": file_sha256(old_path) if reuse_old_results else "",
            "conplex_reference": file_sha256(reference_path),
            "boltz_run_plan": file_sha256(run_plan_path) if run_plan_path.exists() else "",
        },
        "output_sha256": {
            "refined_top3000": file_sha256(top3000_output),
            "final1000": file_sha256(final1000_output),
            "final384": file_sha256(final384_output),
            "agent_review_pool": file_sha256(review_pool_output),
        },
    }
    summary["version_label"] = version_label
    summary["code_state"] = code_state()
    (out_dir / f"formal_completion_summary_{version_label}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize v2 after the 106k delta Boltz run.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary = build(Path(args.config).resolve(), args.allow_partial)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
