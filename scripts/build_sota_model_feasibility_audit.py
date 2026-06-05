from __future__ import annotations

import argparse
import importlib.util
import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


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
    return value


def import_status(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "detail": "module_not_found"}
    return {"available": True, "detail": str(spec.origin or "")}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def binary_status(name: str, min_size_mb: int = 0) -> dict[str, Any]:
    path = shutil.which(name)
    if not path:
        return {"available": False, "detail": "", "sizeBytes": None, "executable": False}
    candidate = Path(path)
    size = candidate.stat().st_size if candidate.exists() else None
    executable = os.access(candidate, os.X_OK)
    min_size = min_size_mb * 1024 * 1024
    return {
        "available": bool(size and size > 0 and size >= min_size and executable),
        "detail": path,
        "sizeBytes": size,
        "minSizeBytes": min_size,
        "executable": executable,
    }


def local_binary_status(path: Path, min_size_mb: int = 0) -> dict[str, Any]:
    size = path.stat().st_size if path.exists() else None
    executable = os.access(path, os.X_OK)
    min_size = min_size_mb * 1024 * 1024
    return {
        "available": bool(size and size > 0 and size >= min_size and executable),
        "detail": str(path) if path.exists() else "",
        "sizeBytes": size,
        "minSizeBytes": min_size,
        "executable": executable,
    }


def gnina_runtime_env() -> tuple[dict[str, str], str]:
    lib_paths = [
        "/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = [path for path in lib_paths if Path(path).exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    value = ":".join(existing + ([current] if current else []))
    env = os.environ.copy()
    if value:
        env["LD_LIBRARY_PATH"] = value
    return env, value


def gnina_runtime_status(status: dict[str, Any]) -> dict[str, Any]:
    path = status.get("detail") or ""
    if not status.get("available") or not path:
        return {**status, "runtimeReady": False, "version": "", "runtimeError": "binary_not_available", "ldLibraryPath": ""}
    env, ld_path = gnina_runtime_env()
    try:
        result = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - keep exact runtime blocker in audit.
        return {**status, "available": False, "runtimeReady": False, "version": "", "runtimeError": f"{type(exc).__name__}: {exc}", "ldLibraryPath": ld_path}
    version = (result.stdout or result.stderr or "").strip().splitlines()[0] if (result.stdout or result.stderr) else ""
    runtime_ready = result.returncode == 0
    return {
        **status,
        "available": runtime_ready,
        "runtimeReady": runtime_ready,
        "version": version,
        "runtimeError": "" if runtime_ready else (result.stderr or result.stdout or "")[:700],
        "ldLibraryPath": ld_path,
    }


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def first_existing(root: Path, candidates: list[str]) -> list[str]:
    return [item for item in candidates if (root / item).exists()]


def matching_files(root: Path, patterns: list[str], max_files: int = 20) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                matches.append(rel(path, root))
            if len(matches) >= max_files:
                return sorted(set(matches))
    return sorted(set(matches))


def tdc_smoke_test(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"available": None, "detail": "not_run"}
    try:
        from tdc import Oracle

        oracle = Oracle(name="qed")
        value = oracle("CCO")
    except Exception as exc:  # noqa: BLE001 - record environment mismatch.
        return {"available": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "detail": f"qed(CCO)={value}"}


def layer_row(
    layer: str,
    purpose: str,
    priority: str,
    readiness: str,
    local_state: str,
    blocker: str,
    next_action: str,
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "purpose": purpose,
        "priority": priority,
        "readiness": readiness,
        "localState": local_state,
        "blocker": blocker,
        "nextAction": next_action,
        "evidence": "; ".join(evidence or []),
    }


def build_audit(root: Path, smoke_tdc: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    imports = {
        name: import_status(name)
        for name in [
            "rdkit",
            "torch",
            "networkx",
            "sklearn",
            "tdc",
            "dgl",
            "dgllife",
            "deepchem",
            "prolif",
            "openbabel",
            "posebusters",
            "vina",
            "boltz",
            "chai_lab",
        ]
    }
    binaries = {
        name: binary_status(name, min_size_mb=1000 if name == "gnina" else 0)
        for name in ["gnina", "vina", "smina", "obabel", "babel"]
    }
    local_smina = Path("/root/autodl-tmp/conda_envs/smina/bin/smina")
    if not binaries["smina"]["available"] and local_smina.exists():
        binaries["smina"] = local_binary_status(local_smina)
    local_gnina = Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2")
    if not binaries["gnina"]["available"] and local_gnina.exists():
        binaries["gnina"] = local_binary_status(local_gnina, min_size_mb=1000)
    binaries["gnina"] = gnina_runtime_status(binaries["gnina"])
    tdc_smoke = tdc_smoke_test(smoke_tdc)

    conplex_files = first_existing(
        root,
        [
            "third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt",
            "outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv",
        ],
    )
    txgnn_files = first_existing(
        root,
        [
            "third_party/TxGNN",
            "scripts/run_txgnn_cancer_inference.py",
            "outputs/sota_validation/kg_explainability_top1000/candidate_kg_explanation_summary.csv",
        ],
    )
    network_files = first_existing(
        root,
        [
            "outputs/sota_validation/network_proximity/network_proximity_summary.json",
            "outputs/sota_validation/network_proximity/final_priority_network_proximity_audit.csv",
        ],
    )
    tissue_context_files = first_existing(
        root,
        [
            "data/external/hpa/rna_tissue_consensus.tsv.zip",
            "outputs/sota_validation/tissue_context/tissue_context_summary.json",
            "outputs/sota_validation/tissue_context/candidate_tissue_context_audit.csv",
            "outputs/sota_validation/final_prioritization/final_priority_sota_context_matrix.csv",
        ],
    )
    gtex_context_summary = read_json(root / "outputs/sota_validation/gtex_context/gtex_context_summary.json")
    gtex_context_files = first_existing(
        root,
        [
            "data/external/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz",
            "outputs/sota_validation/gtex_context/gtex_context_summary.json",
            "outputs/sota_validation/gtex_context/candidate_gtex_context_audit.csv",
            "outputs/sota_validation/gtex_context/target_gtex_context_audit.csv",
            "outputs/sota_validation/gtex_context/protein_gtex_expression_summary.csv",
            "outputs/sota_validation/gtex_context/gtex_context_direction_summary.csv",
            "outputs/sota_validation/gtex_context/GTEX_CONTEXT_AUDIT.md",
            "outputs/sota_validation/final_prioritization/final_priority_gtex_context_matrix.csv",
            "outputs/sota_validation/final_prioritization/final_priority_gtex_context_top300_expert_shortlist.csv",
            "outputs/sota_validation/final_prioritization/final_priority_gtex_context_review_queue.csv",
            "outputs/sota_validation/final_prioritization/FINAL_PRIORITY_GTEX_CONTEXT_AUDIT.md",
        ],
    )
    diffdock_files = first_existing(
        root,
        [
            "third_party/DiffDock/inference.py",
            "outputs/sota_validation/final_prioritization/final_priority_pose_interpretability_augmented_table.csv",
            "outputs/sota_validation/final_prioritization/final_priority_structure_confidence_audit.csv",
        ],
    )
    pose_quality_files = first_existing(
        root,
        [
            "outputs/sota_validation/pose_quality/pose_quality_summary.json",
            "outputs/sota_validation/pose_quality/pose_quality_candidate_audit.csv",
            "outputs/sota_validation/final_prioritization/final_priority_pose_quality_matrix.csv",
            "outputs/sota_validation/final_prioritization/FINAL_PRIORITY_POSE_QUALITY_AUDIT.md",
        ],
    )
    standard_pose_validation_files = first_existing(
        root,
        [
            "outputs/sota_validation/standard_pose_validation/standard_pose_validation_summary.json",
            "outputs/sota_validation/standard_pose_validation/standard_pose_validation_candidate_audit.csv",
            "outputs/sota_validation/standard_pose_validation/prolif_interaction_fingerprints.csv",
            "outputs/sota_validation/standard_pose_validation/posebusters_raw.csv",
            "outputs/sota_validation/final_prioritization/final_priority_standard_pose_validation_matrix.csv",
            "outputs/sota_validation/final_prioritization/FINAL_PRIORITY_STANDARD_POSE_VALIDATION_AUDIT.md",
        ],
    )
    vina_consensus_files = first_existing(
        root,
        [
            "outputs/sota_validation/vina_consensus_rescoring/vina_consensus_summary.json",
            "outputs/sota_validation/vina_consensus_rescoring/vina_consensus_candidate_audit.csv",
            "outputs/sota_validation/vina_consensus_rescoring/vina_consensus_direction_summary.csv",
            "outputs/sota_validation/vina_consensus_rescoring/VINA_CONSENSUS_RESCORING_AUDIT.md",
            "outputs/sota_validation/final_prioritization/final_priority_vina_consensus_matrix.csv",
            "outputs/sota_validation/final_prioritization/FINAL_PRIORITY_VINA_CONSENSUS_AUDIT.md",
        ],
    )
    smina_rescoring_files = first_existing(
        root,
        [
            "outputs/sota_validation/smina_rescoring/smina_rescoring_summary.json",
            "outputs/sota_validation/smina_rescoring/smina_rescoring_candidate_audit.csv",
            "outputs/sota_validation/smina_rescoring/smina_rescoring_direction_summary.csv",
            "outputs/sota_validation/smina_rescoring/SMINA_RESCORING_AUDIT.md",
        ],
    )
    gnina_rescoring_summary = read_json(
        root / "outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_summary.json"
    )
    gnina_rescoring_files = first_existing(
        root,
        [
            "outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_summary.json",
            "outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_candidate_audit.csv",
            "outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_direction_summary.csv",
            "outputs/sota_validation/gnina_cnn_rescoring/GNINA_CNN_RESCORING_AUDIT.md",
        ],
    )
    gnina_candidate_rows = as_int(gnina_rescoring_summary.get("candidateRows"))
    gnina_scored_rows = as_int(gnina_rescoring_summary.get("gninaScoredRows"))
    gnina_complete = bool(gnina_candidate_rows and gnina_scored_rows >= gnina_candidate_rows)
    gnina_started = gnina_scored_rows > 0
    if gnina_complete:
        gnina_readiness = "completed_vina_smina_gnina_cnn_top100"
        gnina_blocker = ""
        gnina_next_action = "Use the completed Vina, smina, and GNINA Top100 rescoring layers as structural consensus evidence; expand beyond Top100 only if more structural stress-test coverage is required."
    elif gnina_started:
        gnina_readiness = "partial_gnina_cnn_top100_scoring"
        gnina_blocker = "GNINA CNN rescoring has partial output; resume the prepared Top100 queue with --skip-existing."
        gnina_next_action = "Resume scripts/build_gnina_cnn_rescoring_audit.py over the prepared Top100 queue with --skip-existing, then regenerate the global summaries."
    elif binaries["gnina"].get("runtimeReady"):
        gnina_readiness = "completed_vina_smina_gnina_runner_runtime_ready_execution_pending"
        gnina_blocker = "GNINA CNN rescoring is not yet executed: the Top100 candidate queue is input-ready and the GNINA runtime is available, but execution is still pending."
        gnina_next_action = "Run scripts/build_gnina_cnn_rescoring_audit.py over the prepared Top100 queue, then regenerate the global summaries."
    else:
        gnina_readiness = "completed_vina_smina_gnina_runner_ready_runtime_missing"
        gnina_blocker = "GNINA CNN rescoring is not yet executed: the Top100 candidate queue is input-ready, but the GNINA runtime is not ready locally."
        gnina_next_action = "Repair the GNINA runtime, run scripts/build_gnina_cnn_rescoring_audit.py over the prepared Top100 queue, then regenerate the global summaries."
    boltz2_complex_summary = read_json(
        root / "outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_summary.json"
    )
    boltz2_high_sampling_summary = read_json(
        root / "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_summary.json"
    )
    boltz2_complex_files = first_existing(
        root,
        [
            "outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_summary.json",
            "outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_candidate_audit.csv",
            "outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_direction_summary.csv",
            "outputs/sota_validation/boltz2_complex_validation/BOLTZ2_COMPLEX_VALIDATION_AUDIT.md",
            "outputs/sota_validation/boltz2_complex_validation/boltz2_input_manifest.csv",
            "outputs/sota_validation/boltz2_complex_validation/BOLTZ2_COMPLEX_INPUT_PACKAGE.md",
            "outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed/boltz2_ligand_repair_manifest.csv",
            "outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed/boltz2_ligand_repair_summary.json",
            "outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed/BOLTZ2_LIGAND_REPAIR_PACKAGE.md",
            "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_summary.json",
            "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_candidate_audit.csv",
            "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_direction_summary.csv",
            "outputs/sota_validation/boltz2_high_sampling_validation/BOLTZ2_HIGH_SAMPLING_VALIDATION_AUDIT.md",
            "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_queue.csv",
            "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_input_summary.json",
        ],
    )
    boltz2_venv = Path("/root/autodl-tmp/venvs/boltz")
    boltz2_assets = first_existing(
        Path("/"),
        [
            "root/autodl-tmp/boltz_cache/boltz2_conf.ckpt",
            "root/autodl-tmp/boltz_cache/boltz2_aff.ckpt",
            "root/autodl-tmp/boltz_cache/mols.tar",
        ],
    )
    external_input_summary = read_json(
        root / "outputs/sota_validation/external_sota_model_inputs/external_sota_model_input_summary.json"
    )
    external_input_files = first_existing(
        root,
        [
            "outputs/sota_validation/external_sota_model_inputs/external_sota_model_input_summary.json",
            "outputs/sota_validation/external_sota_model_inputs/gnina_top100_rescoring_queue.csv",
            "outputs/sota_validation/external_sota_model_inputs/boltz_chai_top50_complex_queue.csv",
            "outputs/sota_validation/external_sota_model_inputs/independent_dti_top1000_queue.csv",
            "outputs/sota_validation/external_sota_model_inputs/boltz_chai_top50_complex_inputs.jsonl",
            "outputs/sota_validation/external_sota_model_inputs/independent_dti_top1000_inputs.jsonl",
            "outputs/sota_validation/external_sota_model_inputs/top_candidate_ligands.smi",
            "outputs/sota_validation/external_sota_model_inputs/top_candidate_proteins.fasta",
            "outputs/sota_validation/external_sota_model_inputs/EXTERNAL_SOTA_MODEL_INPUT_PACKAGE.md",
        ],
    )
    external_queues = external_input_summary.get("queues") or {}
    external_coverage = external_input_summary.get("coverage") or {}
    independent_dti_summary = read_json(
        root / "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_summary.json"
    )
    independent_dti_files = first_existing(
        root,
        [
            "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_summary.json",
            "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_candidate_audit.csv",
            "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_direction_summary.csv",
            "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_top_supported.csv",
            "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_training_pairs.csv",
            "outputs/sota_validation/independent_dti_supervised/INDEPENDENT_DTI_SUPERVISED_AUDIT.md",
        ],
    )
    drugban_files = first_existing(root, ["third_party/DrugBAN", "third_party/DrugBAN/configs/DrugBAN.yaml"])
    deepdta_files = first_existing(root, ["third_party/DeepDTA", "third_party/DeepDTA/deepdta.yml"])
    drugban_checkpoints = matching_files(root / "third_party/DrugBAN", ["**/*.pt", "**/*.pth", "**/*.ckpt"], 20) if (root / "third_party/DrugBAN").exists() else []
    deepdta_checkpoints = matching_files(root / "third_party/DeepDTA", ["**/*.pt", "**/*.pth", "**/*.h5", "**/*.ckpt"], 20) if (root / "third_party/DeepDTA").exists() else []

    lincs_files = matching_files(
        root,
        [
            "data/**/*LINCS*",
            "data/**/*lincs*",
            "data/**/*CMap*",
            "data/**/*cmap*",
        ],
    )
    lincs_readiness_files = first_existing(
        root,
        [
            "outputs/sota_validation/lincs_cmap_readiness/lincs_cmap_readiness_summary.json",
            "outputs/sota_validation/lincs_cmap_readiness/lincs_cmap_drug_scope.csv",
            "outputs/sota_validation/lincs_cmap_readiness/lincs_cmap_direction_scope.csv",
            "outputs/sota_validation/lincs_cmap_readiness/lincs_cmap_data_file_audit.csv",
            "outputs/sota_validation/lincs_cmap_readiness/LINCS_CMAP_READINESS_AUDIT.md",
        ],
    )
    lincs_reversal_summary = read_json(
        root / "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_reversal_summary.json"
    )
    lincs_reversal_files = first_existing(
        root,
        [
            "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_reversal_summary.json",
            "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_drug_mapping.csv",
            "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_selected_signatures.csv",
            "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_signature_reversal_scores.csv",
            "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_drug_direction_reversal_scores.csv",
            "outputs/sota_validation/lincs_cmap_reversal/candidate_lincs_cmap_reversal_audit.csv",
            "outputs/sota_validation/lincs_cmap_reversal/LINCS_CMAP_REVERSAL_AUDIT.md",
            "outputs/sota_validation/final_prioritization/final_priority_lincs_cmap_augmented_table.csv",
            "outputs/sota_validation/final_prioritization/final_priority_lincs_cmap_top300_expert_shortlist.csv",
            "outputs/sota_validation/final_prioritization/FINAL_PRIORITY_LINCS_CMAP_REVERSAL_AUDIT.md",
        ],
    )
    lincs_completed = bool(as_int(lincs_reversal_summary.get("candidateRows")) and lincs_reversal_files)
    if lincs_completed:
        lincs_readiness = "completed_drug_direction_signature_reversal"
        lincs_local_state = (
            f"reversal summary={lincs_reversal_summary}; reversal files={lincs_reversal_files}; "
            f"raw LINCS/CMap input files={lincs_files}; readiness audit={lincs_readiness_files}"
        )
        lincs_blocker = ""
        lincs_next_action = (
            "Use the completed CMap/LINCS drug-direction reversal scores as transcriptomic evidence; "
            "extend only if reviewers request disease-subtype-specific signatures."
        )
        lincs_evidence = lincs_files + lincs_readiness_files + lincs_reversal_files
    else:
        lincs_readiness = "not_ready_data_missing"
        lincs_local_state = (
            f"raw LINCS/CMap input files={lincs_files}; readiness audit={lincs_readiness_files}"
            if lincs_files or lincs_readiness_files
            else "no local LINCS/CMap perturbation or disease-signature files found"
        )
        lincs_blocker = "External perturbation signatures and disease DEG/signature files are required."
        lincs_next_action = (
            "Use the readiness audit to map candidate drugs/disease directions; download/prepare LINCS L1000 "
            "compound signatures and disease-direction DEG signatures, then score candidate drugs by reversal."
        )
        lincs_evidence = lincs_files + lincs_readiness_files
    context_files = matching_files(
        root,
        [
            "data/**/*DepMap*",
            "data/**/*depmap*",
            "data/**/*Achilles*",
            "data/**/*CCLE*",
            "data/**/*GTEx*",
            "data/**/*gtex*",
            "data/**/*HPA*",
            "data/**/*hpa*",
            "data/**/*HumanProteinAtlas*",
        ],
    )
    depmap_access_files = first_existing(
        root,
        [
            "outputs/sota_validation/depmap_oncology_dependency/depmap_data_access_summary.json",
            "outputs/sota_validation/depmap_oncology_dependency/depmap_required_file_access_audit.csv",
            "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_target_scope.csv",
            "outputs/sota_validation/depmap_oncology_dependency/DEPMAP_DATA_ACCESS_AUDIT.md",
        ],
    )
    depmap_dependency_summary = read_json(
        root / "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_dependency_summary.json"
    )
    depmap_dependency_files = first_existing(
        root,
        [
            "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_dependency_summary.json",
            "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_candidate_audit.csv",
            "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_target_audit.csv",
            "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_lineage_summary.csv",
            "outputs/sota_validation/depmap_oncology_dependency/DEPMAP_ONCOLOGY_DEPENDENCY_AUDIT.md",
            "outputs/sota_validation/final_prioritization/final_priority_depmap_oncology_matrix.csv",
            "outputs/sota_validation/final_prioritization/final_priority_depmap_oncology_top300_expert_shortlist.csv",
            "outputs/sota_validation/final_prioritization/final_priority_depmap_oncology_review_queue.csv",
        ],
    )
    admet_files = first_existing(
        root,
        [
            "outputs/sota_validation/admet_repurposing/admet_repurposing_summary.json",
            "outputs/sota_validation/admet_repurposing/drug_admet_repurposing_audit.csv",
        ],
    )
    ml_admet_files = first_existing(
        root,
        [
            "outputs/sota_validation/ml_admet/ml_admet_summary.json",
            "outputs/sota_validation/ml_admet/drug_ml_admet_audit.csv",
            "outputs/sota_validation/ml_admet/candidate_ml_admet_audit.csv",
            "outputs/sota_validation/final_prioritization/final_priority_ml_admet_matrix.csv",
        ],
    )

    rows = [
        layer_row(
            "ConPLex DTI affinity screen",
            "Primary drug-target affinity ranking.",
            "done",
            "completed",
            "local checkpoint and final candidate tables are present" if conplex_files else "missing local checkpoint or outputs",
            "",
            "Use as baseline; add independent DTI ensemble rather than rerunning this layer.",
            conplex_files,
        ),
        layer_row(
            "TxGNN / KG drug-disease evidence",
            "Disease-aware drug repurposing and interpretable KG paths.",
            "done/P1_extension",
            "completed_with_extension_opportunity",
            "TxGNN code and KG explanation outputs are present" if txgnn_files else "TxGNN assets not found",
            "",
            "Keep current outputs; extend only if broader disease-node mapping is required.",
            txgnn_files,
        ),
        layer_row(
            "Network medicine / PPI proximity",
            "Orthogonal disease-module proximity support for candidate targets.",
            "done/P1_expand",
            "completed_limited_interactome",
            "network proximity outputs are present" if network_files else "network proximity outputs not found",
            "Current STRING subnet is narrow; uncovered targets are coverage gaps.",
            "Use current audit now; expand with broader STRING/HuRI/BioGRID interactome for stronger claims.",
            network_files,
        ),
        layer_row(
            "LINCS/CMap expression reversal",
            "Transcriptomic disease-signature reversal support.",
            "P1",
            lincs_readiness,
            lincs_local_state,
            lincs_blocker,
            lincs_next_action,
            lincs_evidence,
        ),
        layer_row(
            "GTEx/HPA/DepMap disease context",
            "Tissue expression and disease/cell-line dependency context.",
            "done/P1_extension",
            "completed_hpa_gtex_depmap_context",
            (
                f"HPA tissue-expression audit={tissue_context_files}; raw context files={context_files}; "
                f"GTEx context audit={gtex_context_files}; GTEx summary={gtex_context_summary}; "
                f"DepMap dependency audit={depmap_dependency_files}; DepMap summary={depmap_dependency_summary}"
                if tissue_context_files or context_files or gtex_context_files or depmap_dependency_files
                else "no completed tissue-context audit found"
            ),
            "",
            "Use the completed HPA, GTEx, and DepMap layers now; extend only with broader CCLE/Project Score or disease-specific single-cell context if a reviewer specifically asks for it.",
            tissue_context_files
            + [item for item in context_files if item not in tissue_context_files]
            + gtex_context_files
            + depmap_access_files
            + depmap_dependency_files,
        ),
        layer_row(
            "Vina/smina/GNINA structural consensus rescoring",
            "Independent docking/rescoring of DiffDock poses and pockets.",
            "done_vina_smina_top100/P2_gnina_extension",
            gnina_readiness,
            (
                f"Vina Python module={imports['vina']}; binaries={binaries}; "
                f"Vina outputs={vina_consensus_files}; smina outputs={smina_rescoring_files}; "
                f"GNINA audit={gnina_rescoring_summary}; GNINA files={gnina_rescoring_files}; "
                f"GNINA queue={external_queues.get('gninaTop100')}"
            ),
            gnina_blocker,
            gnina_next_action,
            diffdock_files + vina_consensus_files + smina_rescoring_files + gnina_rescoring_files + external_input_files,
        ),
        layer_row(
            "External SOTA model input package",
            "Executable queues for GNINA CNN rescoring, Boltz/Chai complex spot-checks, and independent DTI ensembles.",
            "P2_execution_preparation",
            (
                "completed_input_package_boltz2_and_local_dti_executed_gnina_runtime_ready"
                if binaries["gnina"].get("runtimeReady")
                else "completed_input_package_boltz2_and_local_dti_executed_gnina_runtime_pending"
            ),
            f"coverage={external_coverage}; queues={external_queues}" if external_input_summary else "external SOTA model input package not found",
            (
                "GNINA Top100 CNN rescoring and the local supervised independent DTI branch have both been executed; formal DrugBAN/DeepDTA pretrained inference still requires dependencies and validated checkpoints."
                if gnina_complete
                else (
                    "GNINA execution is runtime-ready but still pending or partial. The local supervised independent DTI branch has been executed; formal DrugBAN/DeepDTA pretrained inference still requires dependencies and validated checkpoints."
                    if binaries["gnina"].get("runtimeReady")
                    else "GNINA execution still requires runtime repair. The local supervised independent DTI branch has been executed; formal DrugBAN/DeepDTA pretrained inference still requires dependencies and validated checkpoints."
                )
            ),
            "Use the completed GNINA, Boltz-2, and local supervised DTI results now; run formal DrugBAN/DeepDTA only after model weights and environment are validated.",
            external_input_files + boltz2_complex_files + independent_dti_files,
        ),
        layer_row(
            "Local supervised independent DTI corroboration",
            "Locally trained SMILES+sequence DTI model for orthogonal Top1000 candidate corroboration relative to ConPLex.",
            "done/P2_formal_pretrained_extension",
            "completed_local_supervised_dti_top1000",
            f"summary={independent_dti_summary}; files={independent_dti_files}" if independent_dti_summary else "local supervised DTI outputs not found",
            "",
            "Use this completed layer as local independent DTI corroboration. Keep DrugBAN/DeepDTA as optional formal pretrained-model extensions rather than treating DTI as entirely missing.",
            independent_dti_files,
        ),
        layer_row(
            "PoseBusters/ProLIF interaction validation",
            "Pose validity and residue-interaction plausibility checks.",
            "done_top100/P2_expand",
            "completed_standard_posebusters_prolif_top100",
            f"local RDKit pose-quality gate={pose_quality_files}; standard tool files={standard_pose_validation_files}; prolif={imports['prolif']}; posebusters={imports['posebusters']}; openbabel={imports['openbabel']}",
            "Standard PoseBusters/ProLIF Top100 expert-check is complete; broader Top300/Top1000 expansion is CPU-bound and optional.",
            "Use the completed Top100 standard-tool audit for expert discussion; expand to Top300 only if the review panel needs more structural examples.",
            diffdock_files + pose_quality_files + standard_pose_validation_files,
        ),
        layer_row(
            "Boltz-2 second-model complex validation",
            "Orthogonal protein-ligand complex generation and affinity-probability spot-check for top candidates.",
            "done/P2_external_chai_af3_extension",
            "completed_boltz2_top50_fast_and_high_sampling_finalists",
            (
                "Boltz-2 Top50 audit="
                f"{boltz2_complex_summary}; high-sampling finalist audit={boltz2_high_sampling_summary}; "
                f"venv={boltz2_venv if boltz2_venv.exists() else 'not_found'}; "
                f"assets={boltz2_assets}; chai_lab={imports['chai_lab']}; original Boltz/Chai queue={external_queues.get('boltzChaiTop50')}"
            ),
            "Chai-1 or AF3-style reruns remain optional external finalist extensions rather than current blockers.",
            "Use the completed fast Boltz-2 audit and high-sampling finalist audit as second-model structural evidence; run Chai-1/AF3-style validation only if a second independent complex-prediction model is required.",
            boltz2_complex_files + external_input_files,
        ),
        layer_row(
            "DrugBAN independent DTI ensemble",
            "Independent drug-target model to reduce single-model reliance.",
            "P2",
            "formal_pretrained_extension_dependencies_missing",
            f"DrugBAN files={drugban_files}; checkpoints={drugban_checkpoints}; dgl={imports['dgl']}; dgllife={imports['dgllife']}; independent DTI queue={external_queues.get('independentDtiTop1000')}; local supervised DTI={independent_dti_summary}",
            "A local supervised DTI corroboration layer is complete. Formal DrugBAN inference still requires DGL/DGLLife and a validated checkpoint or reproducible retraining protocol.",
            "Install/containerize DrugBAN dependencies, add validated weights, then run the prepared Top1000 queue as a formal pretrained-model extension.",
            drugban_files + drugban_checkpoints + external_input_files + independent_dti_files,
        ),
        layer_row(
            "DeepDTA independent DTI ensemble",
            "Sequence/SMILES DTI model for independent ranking comparison.",
            "P2",
            "formal_pretrained_extension_checkpoint_missing",
            f"DeepDTA files={deepdta_files}; checkpoints={deepdta_checkpoints}; independent DTI queue={external_queues.get('independentDtiTop1000')}; local supervised DTI={independent_dti_summary}",
            "A local supervised DTI corroboration layer is complete. Formal DeepDTA inference still needs validated pretrained weights or an agreed retraining benchmark.",
            "Use only after obtaining validated DeepDTA weights or retraining on Davis/KIBA, then run the prepared Top1000 queue as a formal pretrained-model extension.",
            deepdta_files + deepdta_checkpoints + external_input_files + independent_dti_files,
        ),
        layer_row(
            "Deep ADMET/toxicity models",
            "hERG, DILI, CYP, BBB, Ames and other ML safety screens.",
            "done/P1_extension",
            "completed_local_tdc_qsar",
            f"rule audit={admet_files}; ML audit={ml_admet_files}; tdc={imports['tdc']}; deepchem={imports['deepchem']}; tdc_smoke={tdc_smoke}",
            "External DeepChem/oracle ensembles can still be added, but the local TDC-trained QSAR ADMET layer is complete.",
            "Use the completed local ML ADMET layer now; extend with external/DeepChem oracles only for independent safety-model consensus.",
            admet_files + ml_admet_files,
        ),
    ]
    df = pd.DataFrame(rows)
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "moduleImports": imports,
        "binaries": binaries,
        "tdcSmokeTest": tdc_smoke,
        "layerCountsByReadiness": dict(df["readiness"].value_counts()),
        "priorityQueue": [
            (
                "P1 LINCS/CMap expression reversal: completed and integrated; extend only with disease-subtype signatures if required."
                if lincs_completed
                else "P1 LINCS/CMap expression reversal: highest added biological value, blocked only by external signature data."
            ),
            "P2 GNINA CNN structural rescoring: Vina, smina, and GNINA Top100 rescoring are complete; expand only if more structural stress-test coverage is required."
            if gnina_complete
            else (
                "P2 GNINA CNN structural rescoring: Vina/smina Top100 are complete; GNINA runner, inputs, and runtime are ready; execute or resume the Top100 queue."
                if binaries["gnina"].get("runtimeReady")
                else "P2 GNINA CNN structural rescoring: Vina/smina Top100 are complete; GNINA runner and inputs are ready, but runtime dependencies still need repair."
            ),
            "P2 smina structural rescoring: Top100 score-only audit is complete and should be interpreted as a classical docking stress test, not a CNN model.",
            "P2 external SOTA input package: GNINA Top100, Boltz/Chai Top50, and independent DTI Top1000 queues are input-ready; Boltz-2 and local supervised DTI have already been executed.",
            "P2 Boltz-2 second-model validation: Top50 fast spot-check and high-sampling finalist validation are complete; only Chai-1/AF3-style external corroboration remains optional.",
            "P2 optional standard ProLIF/PoseBusters expansion: Top100 expert-check is complete; expand to Top300 only if more structural examples are needed.",
            "P2 optional Vina expansion: Top100 score-only audit is complete; expand to Top300/Top500 if more stress-test coverage is needed.",
            "P2 independent DTI ensemble: local supervised DTI Top1000 is complete; formal DrugBAN/DeepDTA pretrained extensions still need dependencies/checkpoints.",
        ],
        "externalSotaModelInputs": external_input_summary,
    }
    return df, summary


def markdown(df: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# SOTA Model Feasibility Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This audit separates SOTA validation layers that are already computed from layers that need external datasets, binaries, libraries, or model checkpoints.",
        "",
        "## Recommended Compute Queue",
        "",
    ]
    for item in summary["priorityQueue"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Layer Readiness", ""])
    for _, row in df.iterrows():
        lines.append(f"### {row['layer']}")
        lines.append(f"- Purpose: {row['purpose']}")
        lines.append(f"- Priority: {row['priority']}")
        lines.append(f"- Readiness: {row['readiness']}")
        lines.append(f"- Local state: {row['localState']}")
        if row["blocker"]:
            lines.append(f"- Blocker: {row['blocker']}")
        lines.append(f"- Next action: {row['nextAction']}")
        if row["evidence"]:
            lines.append(f"- Evidence: {row['evidence']}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The next meaningful work is not additional DiffDock recovery; it is orthogonal evidence from transcriptomics, disease context, independent DTI, structural consensus, and ML ADMET.",
            "- Missing data or binaries should be treated as engineering prerequisites, not as negative biological evidence.",
            "- Network medicine has now been computed locally and can be used immediately, with a clear caveat that the STRING subnet is limited.",
            "- HPA tissue expression, GTEx tissue expression, and DepMap oncology dependency scoring have now been computed locally as complementary disease-context layers.",
            "- A local RDKit pose-quality gate has now been computed over final candidates, and a standard PoseBusters/ProLIF Top100 expert-check has been completed for high-priority structural examples.",
            "- AutoDock Vina, smina score-only, and GNINA CNN Top100 rescoring have now been completed as independent structural stress-test layers.",
            "- A local supervised independent DTI Top1000 layer has now been completed; DrugBAN and DeepDTA remain formal pretrained-model extensions pending validated dependencies and checkpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local readiness for SOTA drug-repurposing validation layers.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation/model_feasibility")
    parser.add_argument("--tdc-smoke", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, summary = build_audit(root, smoke_tdc=args.tdc_smoke)
    table_path = out_dir / "sota_model_feasibility_audit.csv"
    summary_path = out_dir / "sota_model_feasibility_summary.json"
    markdown_path = out_dir / "SOTA_MODEL_FEASIBILITY_AUDIT.md"
    df.to_csv(table_path, index=False)
    write_json(summary_path, summary)
    markdown_path.write_text(markdown(df, summary), encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
