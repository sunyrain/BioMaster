from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def pct(value: float | int | None) -> str:
    return "NA" if value is None else f"{float(value):.2f}%"


def fmt(value: float | int | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    item: Any = payload
    for key in keys:
        if not isinstance(item, dict):
            return default
        item = item.get(key)
    return default if item is None else item


def last_json_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def file_count(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def glob_file_count(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob(pattern) if item.is_file())


def process_matches(patterns: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    matches: list[str] = []
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "build_sota_compute_closure_summary.py" in lowered:
            continue
        if all(pattern in lowered for pattern in lowered_patterns):
            matches.append(line.strip())
    return matches


def vina_shard_progress(root: Path) -> dict[str, Any]:
    logs = sorted(root.glob("logs/vina_final3921_shard_*.log"))
    completed_audits = sorted(
        root.glob("outputs/sota_validation/vina_consensus_rescoring_final3921_shards/shard_*/vina_consensus_candidate_audit.csv")
    )
    processed = 0
    latest: dict[str, Any] = {}
    for log in logs:
        progress = last_json_progress(log)
        if progress:
            processed += int(number(progress.get("processed")) or 0)
            if not latest or log.stat().st_mtime >= latest.get("_mtime", 0):
                latest = {**progress, "_mtime": log.stat().st_mtime, "logPath": str(log.relative_to(root))}
    cache_root = root / "outputs/sota_validation/vina_consensus_rescoring_final3921_shards"
    return {
        "logCount": len(logs),
        "completedShardAuditCount": len(completed_audits),
        "processedRowsFromLogs": processed,
        "ligandPdbqtCacheFiles": glob_file_count(cache_root, "pdbqt_cache_shard_*/ligands/*.pdbqt"),
        "receptorPdbqtCacheFiles": glob_file_count(cache_root, "pdbqt_cache_shard_*/receptors/*.pdbqt"),
        "latestPairId": latest.get("pairId"),
        "latestVinaStatus": latest.get("vinaStatus"),
        "latestLogPath": latest.get("logPath"),
    }


def build_payload(root: Path) -> dict[str, Any]:
    diffdock = read_json(root / "outputs/sota_validation/final_diffdock_completion_after_round4.json")
    diffdock_full_progress = read_json(
        root / "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json"
    )
    diffdock_ligand_failure_audit = read_json(
        root
        / "outputs/report_scale/diffdock_full_run/ligand_failure_audit/diffdock_full_ligand_failure_summary.json"
    )
    diffdock_ligand_rescue = read_json(
        root
        / "outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent/diffdock_ligand_rescue_summary.json"
    )
    diffdock_ligand_rescue_watcher = last_json_progress(root / "logs/diffdock_ligand_rescue_after_full.log")
    diffdock_multi_ligand_rescue = read_json(
        root
        / "outputs/report_scale/diffdock_full_multi_ligand_rescue/aggregate/diffdock_multi_ligand_rescue_summary.json"
    )
    diffdock_multi_ligand_rescue_watcher = last_json_progress(
        root / "logs/diffdock_multi_ligand_rescue_after_single.log"
    )
    diffdock_finalizer_status = read_json(
        root / "outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json"
    )
    diffdock_post_finalization_status = read_json(
        root / "outputs/report_scale/full_diffdock_final/post_finalization_status.json"
    )
    p0 = read_json(root / "outputs/sota_validation/sota_p0_validation_summary.json")
    admet = read_json(root / "outputs/sota_validation/admet_repurposing/admet_repurposing_summary.json")
    ml_admet = read_json(root / "outputs/sota_validation/ml_admet/ml_admet_summary.json")
    kg_short = read_json(root / "outputs/sota_validation/kg_explainability/kg_explainability_summary.json")
    kg_top = read_json(root / "outputs/sota_validation/kg_explainability_top1000/kg_explainability_summary.json")
    stratified = read_json(root / "outputs/sota_validation/known_target_stratified/known_target_stratified_summary.json")
    pose_top = read_json(root / "outputs/sota_validation/pose_sanity_top10000/pose_sanity_summary.json")
    pose_interpretability = read_json(root / "outputs/sota_validation/pose_interpretability/pose_interpretability_summary.json")
    pose_quality = read_json(root / "outputs/sota_validation/pose_quality/pose_quality_summary.json")
    standard_pose_validation = read_json(
        root / "outputs/sota_validation/standard_pose_validation/standard_pose_validation_summary.json"
    )
    standard_pose_validation_top300 = read_json(
        root / "outputs/sota_validation/standard_pose_validation_top300/standard_pose_validation_summary.json"
    )
    standard_pose_validation_full3921 = read_json(
        root / "outputs/sota_validation/standard_pose_validation_full3921/standard_pose_validation_summary.json"
    )
    vina_consensus = read_json(root / "outputs/sota_validation/vina_consensus_rescoring/vina_consensus_summary.json")
    vina_consensus_top300 = read_json(
        root / "outputs/sota_validation/vina_consensus_rescoring_top300_ext/vina_consensus_summary.json"
    )
    vina_consensus_final3921_single = read_json(
        root / "outputs/sota_validation/vina_consensus_rescoring_final3921/vina_consensus_summary.json"
    )
    vina_consensus_final3921_merged = read_json(
        root / "outputs/sota_validation/vina_consensus_rescoring_final3921_merged/vina_consensus_summary.json"
    )
    vina_consensus_final3921 = vina_consensus_final3921_merged or vina_consensus_final3921_single
    vina_consensus_final3921_source = (
        "merged_shards"
        if vina_consensus_final3921_merged
        else ("single_run" if vina_consensus_final3921_single else "")
    )
    smina_rescoring = read_json(root / "outputs/sota_validation/smina_rescoring/smina_rescoring_summary.json")
    smina_rescoring_top300 = read_json(
        root / "outputs/sota_validation/smina_rescoring_top300/smina_rescoring_summary.json"
    )
    smina_rescoring_final3921 = read_json(
        root / "outputs/sota_validation/smina_rescoring_final3921/smina_rescoring_summary.json"
    )
    gnina_rescoring = read_json(
        root / "outputs/sota_validation/gnina_cnn_rescoring/gnina_cnn_rescoring_summary.json"
    )
    gnina_rescoring_top300 = read_json(
        root / "outputs/sota_validation/gnina_cnn_rescoring_top300_cpu/gnina_cnn_rescoring_summary.json"
    )
    gnina_rescoring_full3921 = read_json(
        root / "outputs/sota_validation/gnina_cnn_rescoring_full3921_cpu/gnina_cnn_rescoring_summary.json"
    )
    boltz2_complex = read_json(
        root / "outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation_summary.json"
    )
    boltz2_high_sampling = read_json(
        root / "outputs/sota_validation/boltz2_high_sampling_validation/boltz2_high_sampling_summary.json"
    )
    structure_confidence = read_json(root / "outputs/sota_validation/structure_confidence_top10000/structure_confidence_summary.json")
    stage6_top10000 = read_json(root / "outputs/report_scale/STAGE6_TOP10000_STRUCTURE_AUDIT_SUMMARY.json")
    final_priority = read_json(root / "outputs/sota_validation/final_prioritization/final_candidate_priority_summary.json")
    final_priority_validation = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_validation_summary.json")
    final_structure_confidence = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_structure_confidence_summary.json")
    target_druggability = read_json(root / "outputs/sota_validation/target_druggability/target_druggability_summary.json")
    chemotype_diversity = read_json(root / "outputs/sota_validation/chemotype_diversity/chemotype_diversity_summary.json")
    sota_ready = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_sota_ready_summary.json")
    network_proximity_expanded = read_json(
        root / "outputs/sota_validation/network_proximity_expanded/network_proximity_summary.json"
    )
    network_proximity_legacy = read_json(root / "outputs/sota_validation/network_proximity/network_proximity_summary.json")
    network_proximity = network_proximity_expanded or network_proximity_legacy
    network_proximity_source = "expanded_string_huri" if network_proximity_expanded else "legacy_filtered_string"
    sota_network = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_sota_network_summary.json")
    tissue_context = read_json(root / "outputs/sota_validation/tissue_context/tissue_context_summary.json")
    gtex_context = read_json(root / "outputs/sota_validation/gtex_context/gtex_context_summary.json")
    depmap_access = read_json(root / "outputs/sota_validation/depmap_oncology_dependency/depmap_data_access_summary.json")
    depmap_dependency = read_json(
        root / "outputs/sota_validation/depmap_oncology_dependency/depmap_oncology_dependency_summary.json"
    )
    lincs_readiness = read_json(root / "outputs/sota_validation/lincs_cmap_readiness/lincs_cmap_readiness_summary.json")
    lincs_reversal = read_json(root / "outputs/sota_validation/lincs_cmap_reversal/lincs_cmap_reversal_summary.json")
    expert_review = read_json(
        root / "outputs/sota_validation/expert_review_panel/integrated_expert_review_panel_summary.json"
    )
    external_inputs = read_json(root / "outputs/sota_validation/external_sota_model_inputs/external_sota_model_input_summary.json")
    external_inputs_gnina_full3921 = read_json(
        root / "outputs/sota_validation/external_sota_model_inputs_gnina_full3921/external_sota_model_input_summary.json"
    )
    independent_dti = read_json(
        root / "outputs/sota_validation/independent_dti_supervised/independent_dti_supervised_summary.json"
    )
    independent_dti_full3921 = read_json(
        root / "outputs/sota_validation/independent_dti_supervised_full3921/independent_dti_supervised_summary.json"
    )
    model_feasibility = read_json(root / "outputs/sota_validation/model_feasibility/sota_model_feasibility_summary.json")
    rank_stability = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_rank_stability_summary.json")
    experimental_validation = read_json(root / "outputs/sota_validation/experimental_validation/experimental_validation_panel_summary.json")
    wetlab_validation = read_json(root / "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json")
    final_pre_experiment_gate = read_json(
        root / "outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate_summary.json"
    )
    experiment_closure_audit = read_json(
        root / "outputs/sota_validation/experiment_closure_audit/experiment_closure_audit.json"
    )
    validation_panel_benchmark = read_json(root / "outputs/sota_validation/validation_panel_benchmark/validation_panel_benchmark_summary.json")
    validation_panel_diversity = read_json(root / "outputs/sota_validation/validation_panel_diversity/validation_panel_diversity_summary.json")
    fda_label_mechanism = read_json(root / "outputs/sota_validation/fda_label_mechanism/fda_label_mechanism_summary.json")
    fda_label_temporal = read_json(root / "outputs/sota_validation/fda_label_temporal/fda_label_temporal_summary.json")
    final_priority_ablation = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_ablation_summary.json")
    novelty = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_novelty_leakage_summary.json")
    specificity = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_direction_specificity_summary.json")
    significance = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_topk_significance_summary.json")
    concordance = read_json(root / "outputs/sota_validation/final_prioritization/final_priority_evidence_concordance_summary.json")
    diversity = read_json(root / "outputs/sota_validation/final_prioritization/final_candidate_diversity_summary.json")
    manifest = read_json(root / "outputs/sota_validation/sota_artifact_manifest.json")
    txgnn = read_json(root / "outputs/sota_validation/txgnn_multi_direction_scores.metadata.json")
    opentargets = read_json(root / "outputs/sota_validation/opentargets_direction_scores_top_pages.metadata.json")

    benchmark = p0.get("known_target_benchmark") or {}
    enrichment = {int(row["cutoff"]): row for row in benchmark.get("enrichment", []) if row.get("cutoff") is not None}
    independent_dti_topk = {
        int(row["cutoff"]): row
        for row in metric(independent_dti, "candidateKnownBenchmark", "topK", default=[])
        if row.get("cutoff") is not None
    }
    independent_dti_full3921_topk = {
        int(row["cutoff"]): row
        for row in metric(independent_dti_full3921, "candidateKnownBenchmark", "topK", default=[])
        if row.get("cutoff") is not None
    }
    independent_dti_validation = {
        str(row["split"]): row for row in independent_dti.get("validationMetrics", []) if row.get("split")
    }
    independent_dti_full3921_validation = {
        str(row["split"]): row for row in independent_dti_full3921.get("validationMetrics", []) if row.get("split")
    }
    gnina_runtime = metric(model_feasibility, "binaries", "gnina", default={}) or {}
    gnina_binary_available = gnina_rescoring.get("gninaBinaryAvailable")
    if gnina_binary_available is False and gnina_runtime.get("available") is True:
        gnina_binary_available = True
    gnina_runtime_ready = gnina_rescoring.get("gninaRuntimeReady")
    if gnina_runtime_ready is None:
        gnina_runtime_ready = gnina_runtime.get("runtimeReady")
    gnina_status = (
        "completed_gnina_cnn_top100"
        if number(gnina_rescoring.get("gninaScoredRows")) and number(gnina_rescoring.get("gninaScoredRows")) > 0
        else (
            "runner_ready_input_ready_runtime_ready_execution_pending"
            if gnina_runtime_ready
            else "runner_ready_input_ready_runtime_missing"
        )
    )
    vina_final3921_progress = last_json_progress(root / "logs/vina_final3921.log")
    vina_final3921_ligand_cache_count = file_count(
        root / "outputs/sota_validation/vina_consensus_rescoring_final3921/pdbqt_cache/ligands",
        "*.pdbqt",
    )
    vina_final3921_receptor_cache_count = file_count(
        root / "outputs/sota_validation/vina_consensus_rescoring_final3921/pdbqt_cache/receptors",
        "*.pdbqt",
    )
    vina_final3921_shards = vina_shard_progress(root)
    gnina_full3921_progress = last_json_progress(root / "logs/gnina_full3921_cpu.log")
    gnina_full3921_candidate_rows = metric(external_inputs_gnina_full3921, "queues", "gninaTop100", "rows")
    gnina_full3921_output_count = file_count(
        root / "outputs/sota_validation/gnina_cnn_rescoring_full3921_cpu/gnina_outputs",
        "*.sdf",
    )

    completed_modules = [
        {
            "module": "ConPLex affinity screen and disease-direction ranking",
            "status": "completed",
            "evidence": "Existing ranked candidate matrices were used as the base layer for all downstream validation.",
        },
        {
            "module": "DiffDock disease-direction structural docking",
            "status": "completed_with_stable_residual_failures",
            "candidateRows": diffdock.get("integrated_rows"),
            "completedRows": diffdock.get("completed"),
            "missingRows": diffdock.get("missing"),
            "completionPct": diffdock.get("completion_pct"),
            "evidence": "SDF and SMILES rescue rounds were completed; the remaining missing outputs are stable graph/sampling boundary cases.",
        },
        {
            "module": "Open Targets multi-disease target evidence",
            "status": "completed",
            "rows": opentargets.get("rows"),
            "accessions": opentargets.get("unique_protein_ids") or opentargets.get("unique_accessions"),
            "genes": opentargets.get("unique_gene_names") or opentargets.get("unique_genes"),
            "evidence": "Expanded disease-direction evidence was integrated into the P0 validation coverage audit.",
        },
        {
            "module": "TxGNN multi-direction drug-disease evidence",
            "status": "completed",
            "rows": txgnn.get("rows"),
            "directions": txgnn.get("directions"),
            "evidence": "Representative disease directions were inferred for oncology, cardiovascular, infectious, neurology/psychiatry, and immunology/inflammation.",
        },
        {
            "module": "Known-target benchmark and enrichment",
            "status": "completed",
            "positivePairs": benchmark.get("positivePairsFromKnownTargets"),
            "auroc": benchmark.get("auroc"),
            "averagePrecision": benchmark.get("averagePrecision"),
            "recallAt10000Pct": metric(enrichment.get(10000, {}), "recallPct"),
            "enrichmentAt10000": metric(enrichment.get(10000, {}), "enrichmentVsRandom"),
            "recallAt100000Pct": metric(enrichment.get(100000, {}), "recallPct"),
            "enrichmentAt100000": metric(enrichment.get(100000, {}), "enrichmentVsRandom"),
            "evidence": "Pair-level AUROC/AUPRC, Recall@K, random enrichment, and positive-pair ranks were generated.",
        },
        {
            "module": "Known-target stratified validation",
            "status": "completed",
            "targetRecords": stratified.get("targetRecords"),
            "recordRecallAt100000Pct": metric(stratified, "overall", "recordRecallAt100000Pct"),
            "recordRecallAt1000000Pct": metric(stratified, "overall", "recordRecallAt1000000Pct"),
            "evidence": "Record-level known-target recall was stratified by approval year, therapeutic area, and route.",
        },
        {
            "module": "ADMET and repurposability filtering",
            "status": "completed_rule_based",
            "drugRows": admet.get("drugRows"),
            "candidateRows": admet.get("candidateRows"),
            "drugAdmetTierCounts": admet.get("drugAdmetTierCounts"),
            "evidence": "Transparent RDKit descriptors, PAINS/Brenk alerts, route feasibility, and label-text flags were computed.",
        },
        {
            "module": "ML ADMET/toxicity endpoint audit",
            "status": "completed_local_tdc_qsar",
            "drugRows": ml_admet.get("drugRows"),
            "candidateRows": ml_admet.get("candidateRows"),
            "trainedEndpointCount": ml_admet.get("trainedEndpointCount"),
            "trainedEndpoints": ml_admet.get("trainedEndpoints"),
            "lowMlAdmetRiskCandidatePct": ml_admet.get("lowMlAdmetRiskCandidatePct"),
            "highMlAdmetRiskCandidatePct": ml_admet.get("highMlAdmetRiskCandidatePct"),
            "top100": ml_admet.get("top100"),
            "evidence": "Public TDC ADME/Tox endpoint data were used to train local molecular-fingerprint QSAR models for hERG, DILI, Ames, CYP inhibition, P-gp, BBB, and HIA, then score the FDA drug library and final candidates.",
        },
        {
            "module": "TxGNN KG explainability paths",
            "status": "completed",
            "shortlistCandidateRows": kg_short.get("candidateRows"),
            "shortlistAnyKgPathPct": kg_short.get("anyKgPathPct"),
            "top1000CandidateRows": kg_top.get("candidateRows"),
            "top1000AnyKgPathPct": kg_top.get("anyKgPathPct"),
            "top1000PathRows": kg_top.get("pathRows"),
            "evidence": "Direct drug-target, drug-disease, target-disease, PPI bridge, and known-target disease bridge paths were generated.",
        },
        {
            "module": "DiffDock pose sanity audit",
            "status": "completed_lightweight_geometry",
            "candidateRows": pose_top.get("candidateRows"),
            "completedRows": pose_top.get("completedRows"),
            "readablePoseRows": pose_top.get("readablePoseRows"),
            "readablePosePct": pose_top.get("readablePosePct"),
            "poseAuditStatusCounts": pose_top.get("poseAuditStatusCounts"),
            "evidence": "RDKit/PDB heavy-atom contact checks were run over the disease-direction structural candidate set.",
        },
        {
            "module": "AlphaFold pocket-confidence structural audit",
            "status": "completed_plddt_pocket_qc",
            "candidateRows": structure_confidence.get("candidateRows"),
            "auditedCompletedRows": structure_confidence.get("auditedCompletedRows"),
            "medianPocketMeanPlddt5A": structure_confidence.get("medianPocketMeanPlddt5A"),
            "moderateOrHighConfidencePctOfAudited": structure_confidence.get("moderateOrHighConfidencePctOfAudited"),
            "finalJoinedRows": final_structure_confidence.get("joinedStructureRows"),
            "structureSupportedNovelRows": final_structure_confidence.get("structureSupportedNovelRows"),
            "evidence": "Docked ligand pockets were audited against AlphaFold pLDDT values from receptor PDB B-factor fields, producing structure-adjusted ranking and low-confidence review tables.",
        },
        {
            "module": "Stage6 Top10000 oncology structural audit",
            "status": "completed_pose_and_pocket_qc",
            "candidateRows": stage6_top10000.get("candidateRows"),
            "uniqueDrugs": stage6_top10000.get("uniqueDrugs"),
            "uniqueProteins": stage6_top10000.get("uniqueProteins"),
            "diffdockCompletedRows": stage6_top10000.get("diffdockCompletedRows"),
            "diffdockCompletedPct": stage6_top10000.get("diffdockCompletedPct"),
            "diffdockMissingRows": stage6_top10000.get("diffdockMissingRows"),
            "top100CompletedRows": stage6_top10000.get("top100CompletedRows"),
            "top100MissingRows": stage6_top10000.get("top100MissingRows"),
            "top300CompletedRows": stage6_top10000.get("top300CompletedRows"),
            "top300MissingRows": stage6_top10000.get("top300MissingRows"),
            "poseAuditStatusCounts": metric(stage6_top10000, "poseSanity", "poseAuditStatusCounts"),
            "moderateOrHighConfidencePctOfAudited": metric(
                stage6_top10000, "structureConfidence", "moderateOrHighConfidencePctOfAudited"
            ),
            "top100OriginalTierCounts": metric(stage6_top10000, "structureConfidence", "top100OriginalTierCounts"),
            "top100AdjustedTierCounts": metric(stage6_top10000, "structureConfidence", "top100AdjustedTierCounts"),
            "evidence": "The completed Stage6 Top10000 oncology consensus table was audited with pose-sanity and AlphaFold pocket-confidence checks; missing DiffDock rows are retained as technical missing outputs, not biological negatives.",
        },
        {
            "module": "Residue-contact pose interpretability audit",
            "status": "completed_final_candidate_contact_qc",
            "candidateRows": pose_interpretability.get("candidateRows"),
            "completedRows": pose_interpretability.get("completedRows"),
            "contactResidueRows": pose_interpretability.get("contactResidueRows"),
            "interpretableRows": pose_interpretability.get("interpretableRows"),
            "novelInterpretableRows": pose_interpretability.get("novelInterpretableRows"),
            "top100InterpretableRows": pose_interpretability.get("top100InterpretableRows"),
            "evidence": "Final-candidate docked poses were converted into residue-contact, contact-class, pocket pLDDT, and expert discussion tables.",
        },
        {
            "module": "RDKit pose-quality gate",
            "status": "completed_local_posebusters_like_qc",
            "candidateRows": pose_quality.get("candidateRows"),
            "readableRows": pose_quality.get("readableRows"),
            "qualitySupportedRows": pose_quality.get("qualitySupportedRows"),
            "qualitySupportedPct": pose_quality.get("qualitySupportedPct"),
            "top100": pose_quality.get("top100"),
            "evidence": "Final-candidate docked poses were audited for ligand conformer readability, bond geometry, intramolecular clashes, ligand-receptor severe overlaps, contact coverage, and pocket pLDDT as a local PoseBusters-like quality gate.",
        },
        {
            "module": "Standard PoseBusters/ProLIF Top100 structural validation",
            "status": "completed_standard_tool_expert_check",
            "candidateRows": standard_pose_validation.get("candidateRows"),
            "structureInputReadyRows": standard_pose_validation.get("structureInputReadyRows"),
            "posebustersPassRows": standard_pose_validation.get("posebustersPassRows"),
            "posebustersPassPct": standard_pose_validation.get("posebustersPassPct"),
            "prolifOkRows": standard_pose_validation.get("prolifOkRows"),
            "prolifInteractionRows": standard_pose_validation.get("prolifInteractionRows"),
            "standardSupportedRows": standard_pose_validation.get("standardSupportedRows"),
            "standardSupportedPct": standard_pose_validation.get("standardSupportedPct"),
            "tierCounts": standard_pose_validation.get("standardPoseValidationTierCounts"),
            "evidence": "High-priority Top100 docked poses were checked with standard PoseBusters and ProLIF tooling, producing pose-validity calls plus residue-level ligand-protein interaction fingerprints for expert review.",
        },
        {
            "module": "Standard PoseBusters/ProLIF Top300 structural validation extension",
            "status": "completed_standard_tool_extension_top300",
            "candidateRows": standard_pose_validation_top300.get("candidateRows"),
            "structureInputReadyRows": standard_pose_validation_top300.get("structureInputReadyRows"),
            "posebustersPassRows": standard_pose_validation_top300.get("posebustersPassRows"),
            "posebustersPassPct": standard_pose_validation_top300.get("posebustersPassPct"),
            "prolifOkRows": standard_pose_validation_top300.get("prolifOkRows"),
            "prolifInteractionRows": standard_pose_validation_top300.get("prolifInteractionRows"),
            "standardSupportedRows": standard_pose_validation_top300.get("standardSupportedRows"),
            "standardSupportedPct": standard_pose_validation_top300.get("standardSupportedPct"),
            "tierCounts": standard_pose_validation_top300.get("standardPoseValidationTierCounts"),
            "evidence": "The completed Top300 extension broadens the standard PoseBusters/ProLIF stress test beyond the original Top100 while preserving the Top100 result as the formal high-priority expert-check layer.",
        },
        {
            "module": "Standard PoseBusters/ProLIF full final-candidate structural validation",
            "status": "completed_standard_tool_extension_full3921"
            if standard_pose_validation_full3921
            else "running_standard_tool_extension_full3921",
            "candidateRows": standard_pose_validation_full3921.get("candidateRows"),
            "structureInputReadyRows": standard_pose_validation_full3921.get("structureInputReadyRows"),
            "posebustersPassRows": standard_pose_validation_full3921.get("posebustersPassRows"),
            "posebustersPassPct": standard_pose_validation_full3921.get("posebustersPassPct"),
            "prolifOkRows": standard_pose_validation_full3921.get("prolifOkRows"),
            "prolifInteractionRows": standard_pose_validation_full3921.get("prolifInteractionRows"),
            "standardSupportedRows": standard_pose_validation_full3921.get("standardSupportedRows"),
            "standardSupportedPct": standard_pose_validation_full3921.get("standardSupportedPct"),
            "tierCounts": standard_pose_validation_full3921.get("standardPoseValidationTierCounts"),
            "evidence": "The full final-priority candidate set is being checked with the same standard PoseBusters/ProLIF tooling used for the Top100 and Top300 expert-check layers.",
        },
        {
            "module": "AutoDock Vina Top100 structural consensus rescoring",
            "status": "completed_vina_python_score_only_top100",
            "candidateRows": vina_consensus.get("candidateRows"),
            "structureInputReadyRows": vina_consensus.get("structureInputReadyRows"),
            "pdbqtReadyRows": vina_consensus.get("pdbqtReadyRows"),
            "vinaScoredRows": vina_consensus.get("vinaScoredRows"),
            "vinaConsensusSupportedRows": vina_consensus.get("vinaConsensusSupportedRows"),
            "vinaConsensusSupportedPct": vina_consensus.get("vinaConsensusSupportedPct"),
            "medianVinaScoreKcalMol": vina_consensus.get("medianVinaScoreKcalMol"),
            "medianVinaOptimizedScoreKcalMol": vina_consensus.get("medianVinaOptimizedScoreKcalMol"),
            "tierCounts": vina_consensus.get("vinaConsensusTierCounts"),
            "evidence": "Top100 high-priority DiffDock poses were converted to PDBQT and independently stress-tested with AutoDock Vina score-only plus local optimization. The result is a conservative structural consensus layer, not a new affinity model.",
        },
        {
            "module": "smina Top100 structural rescoring",
            "status": "completed_smina_score_only_top100",
            "candidateRows": smina_rescoring.get("candidateRows"),
            "inputReadyRows": smina_rescoring.get("inputReadyRows"),
            "sminaScoredRows": smina_rescoring.get("sminaScoredRows"),
            "sminaSupportedRows": smina_rescoring.get("sminaSupportedRows"),
            "sminaSupportedPct": smina_rescoring.get("sminaSupportedPct"),
            "medianSminaAffinityKcalMol": smina_rescoring.get("medianSminaAffinityKcalMol"),
            "tierCounts": smina_rescoring.get("tierCounts"),
            "evidence": "The prepared Top100 structural queue was independently rescored with smina score-only mode, including receptor PDBQT conversion for missing receptor caches. This is a second classical docking/scoring stress test and not GNINA CNN scoring.",
        },
        {
            "module": "GNINA CNN Top100 rescoring execution audit",
            "status": gnina_status,
            "candidateRows": gnina_rescoring.get("candidateRows"),
            "inputReadyRows": gnina_rescoring.get("inputReadyRows"),
            "gninaBinaryAvailable": gnina_binary_available,
            "gninaRuntimeReady": gnina_runtime_ready,
            "gninaVersion": gnina_rescoring.get("gninaVersion") or gnina_runtime.get("version"),
            "gninaLdLibraryPath": gnina_rescoring.get("gninaLdLibraryPath") or gnina_runtime.get("ldLibraryPath"),
            "gninaScoredRows": gnina_rescoring.get("gninaScoredRows"),
            "gninaSupportedRows": gnina_rescoring.get("gninaSupportedRows"),
            "statusCounts": gnina_rescoring.get("statusCounts"),
            "tierCounts": gnina_rescoring.get("tierCounts"),
            "evidence": (
                "GNINA CNN rescoring has been executed over the prepared Top100 structural queue; completed rows now carry parsed CNNscore/CNNaffinity and consensus tiers."
                if gnina_status == "completed_gnina_cnn_top100"
                else "A reproducible GNINA CNN rescoring runner and audit layer were created for the prepared Top100 structural queue. GNINA is runtime-ready locally; current rows remain unscored until execution resumes, so no CNN score is claimed yet."
            ),
        },
        {
            "module": "Boltz-2 Top50 second-model complex validation",
            "status": "completed_boltz2_fast_spotcheck_top50",
            "candidateRows": boltz2_complex.get("candidateRows"),
            "completedRows": boltz2_complex.get("completedRows"),
            "completedPct": boltz2_complex.get("completedPct"),
            "originalLigandRows": boltz2_complex.get("initialOriginalLigandRows"),
            "repairedParentLigandRows": boltz2_complex.get("repairedParentLigandRows"),
            "abSupportedRows": boltz2_complex.get("abSupportedRows"),
            "abSupportedPct": boltz2_complex.get("abSupportedPct"),
            "medianConfidenceScore": boltz2_complex.get("medianConfidenceScore"),
            "medianLigandIptm": boltz2_complex.get("medianLigandIptm"),
            "medianAffinityProbabilityBinary": boltz2_complex.get("medianAffinityProbabilityBinary"),
            "tierCounts": boltz2_complex.get("tierCounts"),
            "evidence": "Boltz-2 was executed locally on the selected Top50 complex spot-check set as an orthogonal structure-generation and affinity-probability layer. Salt/solvate/counter-ion failures were rerun as largest organic parent fragments and are explicitly flagged.",
        },
        {
            "module": "Boltz-2 high-sampling finalist validation",
            "status": "completed_boltz2_high_sampling_finalists",
            "candidateRows": boltz2_high_sampling.get("candidateRows"),
            "completedRows": boltz2_high_sampling.get("completedRows"),
            "completedPct": boltz2_high_sampling.get("completedPct"),
            "knownRows": boltz2_high_sampling.get("knownRows"),
            "novelOrExtensionRows": boltz2_high_sampling.get("novelOrExtensionRows"),
            "abSupportedRows": boltz2_high_sampling.get("abSupportedRows"),
            "abSupportedPct": boltz2_high_sampling.get("abSupportedPct"),
            "medianHighConfidenceScore": boltz2_high_sampling.get("medianHighConfidenceScore"),
            "medianHighLigandIptm": boltz2_high_sampling.get("medianHighLigandIptm"),
            "medianHighAffinityProbabilityBinary": boltz2_high_sampling.get("medianHighAffinityProbabilityBinary"),
            "medianCompositeDeltaVsFast": boltz2_high_sampling.get("medianCompositeDeltaVsFast"),
            "tierCounts": boltz2_high_sampling.get("tierCounts"),
            "evidence": "Selected original-ligand Boltz-2 finalists were rerun with higher recycling, structure sampling, and affinity sampling. This is a stronger second-model structural corroboration layer for finalists, not a DiffDock rerun.",
        },
        {
            "module": "ChEMBL target druggability and clinical-stage audit",
            "status": "completed_uniprot_exact_match",
            "candidateRows": target_druggability.get("candidateRows"),
            "uniqueTargets": target_druggability.get("uniqueTargets"),
            "exactUniProtMatchPct": target_druggability.get("exactUniProtMatchPct"),
            "phase3Or4Pct": target_druggability.get("phase3Or4Pct"),
            "smallMoleculeModalityPct": target_druggability.get("smallMoleculeModalityPct"),
            "directionDiseaseFitPct": target_druggability.get("directionDiseaseFitPct"),
            "targetAdjustedAveragePrecision": target_druggability.get("targetAdjustedAveragePrecision"),
            "targetAdjustedRecallAt100Pct": target_druggability.get("targetAdjustedRecallAt100Pct"),
            "evidence": "Final candidates were mapped by exact UniProt accession to the ChEMBL druggable-proteome table, adding clinical phase, druggable modality, target class, and disease-direction context.",
        },
        {
            "module": "Drug chemotype scaffold and similarity-diversity audit",
            "status": "completed_rdkit_scaffold_cluster_qc",
            "candidateRows": chemotype_diversity.get("candidateRows"),
            "uniqueDrugs": chemotype_diversity.get("uniqueDrugs"),
            "structureMappedUniqueDrugPct": chemotype_diversity.get("structureMappedUniqueDrugPct"),
            "uniqueMurckoScaffolds": chemotype_diversity.get("uniqueMurckoScaffolds"),
            "uniqueChemotypeClusters": chemotype_diversity.get("uniqueChemotypeClusters"),
            "top100UniqueScaffolds": chemotype_diversity.get("top100UniqueScaffolds"),
            "top100TopScaffoldPct": chemotype_diversity.get("top100TopScaffoldPct"),
            "diverseShortlistRows": chemotype_diversity.get("diverseShortlistRows"),
            "evidence": "FDA small-molecule SMILES were mapped to candidate drugs, then audited by Murcko scaffolds and Morgan-fingerprint Butina clusters to quantify chemotype concentration and generate a scaffold-capped shortlist.",
        },
        {
            "module": "SOTA-ready multi-evidence decision matrix",
            "status": "completed_integrated_expert_triage",
            "candidateRows": sota_ready.get("candidateRows"),
            "tierCounts": sota_ready.get("tierCounts"),
            "actionCounts": sota_ready.get("actionCounts"),
            "sotaReadyTop100KnownRows": sota_ready.get("sotaReadyTop100KnownRows"),
            "sotaReadyTop100NovelRows": sota_ready.get("sotaReadyTop100NovelRows"),
            "diverseShortlistRows": sota_ready.get("diverseShortlistRows"),
            "evidence": "Final candidates were re-integrated across model score, KG/disease evidence, structure confidence, target druggability, chemotype diversity, direction specificity, and risk flags into expert action queues.",
        },
        {
            "module": "Network medicine / PPI proximity audit",
            "status": "completed_expanded_string_huri_interactome"
            if network_proximity_source == "expanded_string_huri"
            else "completed_limited_interactome",
            "source": network_proximity_source,
            "candidateRows": network_proximity.get("candidateRows"),
            "finalPriorityRows": network_proximity.get("finalPriorityRows"),
            "stringGraphNodes": network_proximity.get("stringGraphNodes"),
            "stringGraphEdges": network_proximity.get("stringGraphEdges"),
            "candidateRowsStringCoveredPct": network_proximity.get("candidateRowsStringCoveredPct"),
            "uniqueTargetsStringCoveredPct": network_proximity.get("uniqueTargetsStringCoveredPct"),
            "uniqueTargetsStringCovered": network_proximity.get("uniqueTargetsStringCovered"),
            "uniqueTargetProteinPairs": network_proximity.get("uniqueTargetProteinPairs"),
            "finalPriorityTierCounts": network_proximity.get("finalPriorityTierCounts"),
            "evidence": "Open Targets disease-module seeds and the current STRING/HuRI PPI graph were used to add coverage-aware target proximity evidence; uncovered targets are labeled as data gaps rather than negative biology.",
        },
        {
            "module": "SOTA-network candidate reprioritization",
            "status": "completed_network_medicine_integration",
            "candidateRows": sota_network.get("candidateRows"),
            "networkPositiveRows": sota_network.get("networkPositiveRows"),
            "networkDirectRows": sota_network.get("networkDirectRows"),
            "networkCoverageGapRows": sota_network.get("networkCoverageGapRows"),
            "oldSotaReadyAveragePrecision": sota_network.get("oldSotaReadyAveragePrecision"),
            "sotaNetworkAveragePrecision": sota_network.get("sotaNetworkAveragePrecision"),
            "top100": sota_network.get("top100"),
            "evidence": "The network-medicine layer was merged into the existing SOTA-ready matrix without overwriting the original ranking, producing a network-adjusted matrix, shortlists, TopK metrics, and rank-shift review.",
        },
        {
            "module": "HPA tissue-expression context audit",
            "status": "completed_hpa_tissue_plausibility_qc",
            "candidateRows": tissue_context.get("candidateRows"),
            "uniqueTargets": tissue_context.get("uniqueTargets"),
            "targetDirectionRows": tissue_context.get("targetDirectionRows"),
            "candidateHpaMatchedPct": tissue_context.get("candidateHpaMatchedPct"),
            "candidateTissuePositivePct": tissue_context.get("candidateTissuePositivePct"),
            "targetDirectionTissuePositivePct": tissue_context.get("targetDirectionTissuePositivePct"),
            "top100": tissue_context.get("top100"),
            "evidence": "Human Protein Atlas consensus RNA expression was mapped to disease-direction tissue panels, adding tissue-plausibility support and mismatch review queues without replacing disease causality evidence.",
        },
        {
            "module": "GTEx tissue-expression context audit",
            "status": "completed_gtex_tissue_plausibility_qc",
            "candidateRows": gtex_context.get("candidateRows"),
            "uniqueTargets": gtex_context.get("uniqueTargets"),
            "targetDirectionRows": gtex_context.get("targetDirectionRows"),
            "gtexRelease": gtex_context.get("gtexRelease"),
            "candidateGtexMatchedPct": gtex_context.get("candidateGtexMatchedPct"),
            "candidateGtexPositivePct": gtex_context.get("candidateGtexPositivePct"),
            "targetDirectionGtexPositivePct": gtex_context.get("targetDirectionGtexPositivePct"),
            "top100": gtex_context.get("top100"),
            "evidence": "GTEx V10 median-TPM tissue expression was mapped to disease-direction tissue panels as an independent bulk-tissue corroboration layer for target-context plausibility.",
        },
        {
            "module": "DepMap oncology dependency scoring",
            "status": "completed_depmap_crispr_dependency_qc",
            "oncologyCandidateRows": depmap_dependency.get("oncologyCandidateRows"),
            "uniqueOncologyTargets": depmap_dependency.get("uniqueOncologyTargets"),
            "depmapRelease": depmap_dependency.get("depmapRelease"),
            "depmapReleaseDate": depmap_dependency.get("depmapReleaseDate"),
            "matchedTargetsPct": depmap_dependency.get("depmapMatchedTargetsPct"),
            "dependencyPositiveCandidateRows": depmap_dependency.get("depmapDependencyPositiveCandidateRows"),
            "dependencyPositiveCandidateRowsPct": depmap_dependency.get("depmapDependencyPositiveCandidateRowsPct"),
            "dependencyPositiveTargets": depmap_dependency.get("depmapDependencyPositiveTargets"),
            "dependencyPositiveTargetsPct": depmap_dependency.get("depmapDependencyPositiveTargetsPct"),
            "top100Oncology": depmap_dependency.get("top100Oncology"),
            "evidence": "DepMap Public CRISPR dependency probabilities were mapped to oncology candidate targets to add a cancer-cell vulnerability layer; this supports target context, not direct drug binding or clinical efficacy.",
        },
        {
            "module": "LINCS/CMap disease-signature reversal scoring",
            "status": "completed_drug_direction_signature_reversal",
            "candidateRows": lincs_reversal.get("candidateRows"),
            "candidateDrugs": lincs_reversal.get("candidateDrugs"),
            "mappedCandidateDrugs": lincs_reversal.get("mappedCandidateDrugs"),
            "mappedCandidateRows": lincs_reversal.get("mappedCandidateRows"),
            "allQcTrtCpSignatureRows": lincs_reversal.get("allQcTrtCpSignatureRows"),
            "selectedSignatureCount": lincs_reversal.get("selectedSignatureCount"),
            "drugDirectionScoreRows": lincs_reversal.get("drugDirectionScoreRows"),
            "candidatePositiveReversalRows": lincs_reversal.get("candidatePositiveReversalRows"),
            "top100PositiveReversalRows": lincs_reversal.get("top100PositiveReversalRows"),
            "candidateTierCounts": lincs_reversal.get("candidateTierCounts"),
            "evidence": "LINCS/CMap Level 5 compound perturbation signatures were scored against direction-level disease signatures. The score is a drug-disease transcriptomic reversal layer, not direct target-binding proof.",
        },
        {
            "module": "Integrated expert review panel",
            "status": "completed_multi_evidence_expert_shortlist",
            "candidateRows": expert_review.get("candidateRows"),
            "panelRows": expert_review.get("panelRows"),
            "wave1Rows": expert_review.get("wave1Rows"),
            "panelUniqueDrugs": expert_review.get("panelUniqueDrugs"),
            "panelUniqueTargets": expert_review.get("panelUniqueTargets"),
            "panelUniqueScaffolds": expert_review.get("panelUniqueScaffolds"),
            "panelDirectionCounts": expert_review.get("panelDirectionCounts"),
            "panelNoveltyGroupCounts": expert_review.get("panelNoveltyGroupCounts"),
            "panelCmapTierCounts": expert_review.get("panelCmapTierCounts"),
            "panelStructureTierCounts": expert_review.get("panelStructureTierCounts"),
            "panelAdmetTierCounts": expert_review.get("panelAdmetTierCounts"),
            "evidence": "Affinity, disease evidence, pathway/CREEDS, LINCS/CMap, GTEx/HPA, DepMap, ADMET, contraindication flags, structure audits, and known-target/novelty labels were merged into a diversity-constrained Top50 and Wave1 panel.",
        },
        {
            "module": "External SOTA model input package",
            "status": "completed_input_package_with_local_dti_execution",
            "candidateRows": metric(external_inputs, "coverage", "candidateRows"),
            "uniquePairs": metric(external_inputs, "coverage", "uniquePairs"),
            "uniqueDrugs": metric(external_inputs, "coverage", "uniqueDrugs"),
            "uniqueProteins": metric(external_inputs, "coverage", "uniqueProteins"),
            "gninaRows": metric(external_inputs, "queues", "gninaTop100", "rows"),
            "gninaInputReadyRows": metric(external_inputs, "queues", "gninaTop100", "inputReadyRows"),
            "boltzChaiRows": metric(external_inputs, "queues", "boltzChaiTop50", "rows"),
            "boltzChaiInputReadyRows": metric(external_inputs, "queues", "boltzChaiTop50", "inputReadyRows"),
            "independentDtiRows": metric(external_inputs, "queues", "independentDtiTop1000", "rows"),
            "independentDtiInputReadyRows": metric(external_inputs, "queues", "independentDtiTop1000", "inputReadyRows"),
            "ligandAssetCount": metric(external_inputs, "sharedInputAssets", "topCandidateLigandCount"),
            "proteinAssetCount": metric(external_inputs, "sharedInputAssets", "topCandidateProteinCount"),
            "evidence": "Executable input queues were generated for GNINA CNN rescoring, Boltz/Chai-style complex spot-checks, and independent DTI ensemble inference. The local supervised DTI branch has now been executed; formal DrugBAN/DeepDTA pretrained inference still requires validated weights and dependencies.",
        },
        {
            "module": "Local supervised independent DTI corroboration",
            "status": "completed_local_supervised_dti_top1000",
            "candidateRows": independent_dti.get("candidateRows"),
            "scoredRows": independent_dti.get("scoredRows"),
            "uniqueCandidateDrugs": independent_dti.get("uniqueCandidateDrugs"),
            "uniqueCandidateProteins": independent_dti.get("uniqueCandidateProteins"),
            "knownRows": independent_dti.get("knownRows"),
            "novelRows": independent_dti.get("novelRows"),
            "abSupportedRows": independent_dti.get("abSupportedRows"),
            "abSupportedPct": independent_dti.get("abSupportedPct"),
            "tierCounts": independent_dti.get("tierCounts"),
            "candidateKnownBenchmark": independent_dti.get("candidateKnownBenchmark"),
            "validationMetrics": independent_dti.get("validationMetrics"),
            "evidence": "A reproducible local ExtraTrees DTI model was trained from FDA label drug-UniProt positives and sampled unlabeled negatives, excluding exact FDA-positive candidate pairs before scoring the Top1000 SMILES+sequence queue. This provides an orthogonal DTI corroboration layer relative to ConPLex, while not being claimed as DrugBAN/DeepDTA pretrained inference.",
        },
        {
            "module": "Local supervised independent DTI full final-candidate extension",
            "status": "completed_local_supervised_dti_full3921"
            if independent_dti_full3921
            else "not_started_local_supervised_dti_full3921",
            "candidateRows": independent_dti_full3921.get("candidateRows"),
            "scoredRows": independent_dti_full3921.get("scoredRows"),
            "uniqueCandidateDrugs": independent_dti_full3921.get("uniqueCandidateDrugs"),
            "uniqueCandidateProteins": independent_dti_full3921.get("uniqueCandidateProteins"),
            "knownRows": independent_dti_full3921.get("knownRows"),
            "novelRows": independent_dti_full3921.get("novelRows"),
            "abSupportedRows": independent_dti_full3921.get("abSupportedRows"),
            "abSupportedPct": independent_dti_full3921.get("abSupportedPct"),
            "tierCounts": independent_dti_full3921.get("tierCounts"),
            "candidateKnownBenchmark": independent_dti_full3921.get("candidateKnownBenchmark"),
            "validationMetrics": independent_dti_full3921.get("validationMetrics"),
            "evidence": "The same local supervised DTI protocol was expanded from the prepared Top1000 queue to the full 3921 final-priority candidate matrix, preserving exact FDA-positive candidate-pair exclusion from training.",
        },
        {
            "module": "SOTA model feasibility audit",
            "status": "completed_engineering_readiness_audit",
            "layerCountsByReadiness": model_feasibility.get("layerCountsByReadiness"),
            "priorityQueue": model_feasibility.get("priorityQueue"),
            "evidence": "Local availability of SOTA-adjacent validation layers was audited across transcriptomics, tissue context, structural rescoring, independent DTI, complex prediction, and ML ADMET prerequisites.",
        },
        {
            "module": "Rank stability and consensus-priority audit",
            "status": "completed_multi_ranking_stability_qc",
            "candidateRows": rank_stability.get("candidateRows"),
            "rankingMethods": rank_stability.get("rankingMethods"),
            "consensusTop100Rows3PlusMethods": rank_stability.get("consensusTop100Rows3PlusMethods"),
            "allMethodTop100IntersectionRows": rank_stability.get("allMethodTop100IntersectionRows"),
            "finalVsSotaReadyTop100OverlapRows": rank_stability.get("finalVsSotaReadyTop100OverlapRows"),
            "finalVsSotaReadySpearman": rank_stability.get("finalVsSotaReadySpearman"),
            "evidence": "Final-priority, structure-adjusted, target-adjusted, and SOTA-ready rankings were compared by TopK overlap, rank correlation, consensus candidates, and large rank-delta review.",
        },
        {
            "module": "Experimental validation planning panel",
            "status": "completed_assay_planning_triage",
            "candidateRows": experimental_validation.get("candidateRows"),
            "experimentReadyRows": experimental_validation.get("experimentReadyRows"),
            "reviewReadyRows": experimental_validation.get("reviewReadyRows"),
            "novelExperimentOrReviewReadyRows": experimental_validation.get("novelExperimentOrReviewReadyRows"),
            "balancedPanelRows": experimental_validation.get("balancedPanelRows"),
            "balancedPanelUniqueDrugs": experimental_validation.get("balancedPanelUniqueDrugs"),
            "balancedPanelUniqueTargets": experimental_validation.get("balancedPanelUniqueTargets"),
            "balancedPanelUniqueScaffolds": experimental_validation.get("balancedPanelUniqueScaffolds"),
            "evidence": "Final candidates were converted into assay-planning queues with transparent validation score, disease-balanced shortlist, novelty shortlist, positive controls, risk-review queue, and assay modality labels.",
        },
        {
            "module": "Wet-lab pre-experiment focusing package",
            "status": "completed_decision_grade_wetlab_focus",
            "candidateRows": wetlab_validation.get("candidateRows"),
            "experimentReadyRows": wetlab_validation.get("experimentReadyRows"),
            "reviewReadyRows": wetlab_validation.get("reviewReadyRows"),
            "expertTop50Rows": wetlab_validation.get("expertTop50Rows"),
            "wave1Rows": wetlab_validation.get("wave1Rows"),
            "focusedTop12Rows": wetlab_validation.get("focusedTop12Rows"),
            "focusedCore6Rows": wetlab_validation.get("focusedCore6Rows"),
            "purchaseAndAssayQueueRows": wetlab_validation.get("purchaseAndAssayQueueRows"),
            "preExperimentDecisionRows": wetlab_validation.get("preExperimentDecisionRows"),
            "preExperimentDecisionCounts": wetlab_validation.get("preExperimentDecisionCounts"),
            "firstExperimentPanelRows": wetlab_validation.get("firstExperimentPanelRows"),
            "firstExperimentPanelRoleCounts": wetlab_validation.get("firstExperimentPanelRoleCounts"),
            "firstExperimentPanelDirectionCounts": wetlab_validation.get("firstExperimentPanelDirectionCounts"),
            "prePurchaseFocusRows": wetlab_validation.get("prePurchaseFocusRows"),
            "prePurchaseTierCounts": wetlab_validation.get("prePurchaseTierCounts"),
            "prePurchaseActionCounts": wetlab_validation.get("prePurchaseActionCounts"),
            "experimentExecutionProtocolRows": wetlab_validation.get("experimentExecutionProtocolRows"),
            "experimentExecutionProtocolCoreRows": wetlab_validation.get("experimentExecutionProtocolCoreRows"),
            "experimentExecutionProtocolExtensionRows": wetlab_validation.get("experimentExecutionProtocolExtensionRows"),
            "experimentExecutionProtocolAssayCounts": wetlab_validation.get("experimentExecutionProtocolAssayCounts"),
            "procurementPlatformChecklistRows": wetlab_validation.get("procurementPlatformChecklistRows"),
            "procurementPlatformChecklistCoreRows": wetlab_validation.get("procurementPlatformChecklistCoreRows"),
            "procurementPlatformChecklistExtensionRows": wetlab_validation.get("procurementPlatformChecklistExtensionRows"),
            "procurementPlatformChecklistStatusCounts": wetlab_validation.get("procurementPlatformChecklistStatusCounts"),
            "finalPreExperimentGateRows": final_pre_experiment_gate.get("inputRows"),
            "finalPreExperimentGateTierCounts": final_pre_experiment_gate.get("finalGateTierCounts"),
            "finalPreExperimentGateGoCandidateRows": final_pre_experiment_gate.get("goCandidateRows"),
            "finalPreExperimentGateBackupRows": final_pre_experiment_gate.get("backupRows"),
            "finalPreExperimentGateExpertReviewRows": final_pre_experiment_gate.get("expertReviewRows"),
            "finalPreExperimentGateHoldRows": final_pre_experiment_gate.get("holdRows"),
            "finalPreExperimentGateManualGates": final_pre_experiment_gate.get("manualGates"),
            "finalPreExperimentGateOutputs": final_pre_experiment_gate.get("outputs"),
            "wave1RoleCounts": wetlab_validation.get("wave1RoleCounts"),
            "wave1DirectionCounts": wetlab_validation.get("wave1DirectionCounts"),
            "outputs": wetlab_validation.get("outputs"),
            "evidence": "The expert Top50 and Wave1 panels were converted into a stricter wet-lab execution package with core 6, focused 12, backup 24, disease-direction Top10, purchase/assay queue, primary assay, orthogonal assay, counterscreen, execution protocol, vendor/platform checklist, pre-experiment go/hold decision matrix, and a final pre-purchase manual-gate checklist.",
        },
        {
            "module": "Validation panel benchmark and calibration",
            "status": "completed_known_target_calibration_qc",
            "candidateRows": validation_panel_benchmark.get("candidateRows"),
            "knownDrugTargetRows": validation_panel_benchmark.get("knownDrugTargetRows"),
            "validationAuroc": validation_panel_benchmark.get("validationAuroc"),
            "validationAveragePrecision": validation_panel_benchmark.get("validationAveragePrecision"),
            "top100KnownRows": validation_panel_benchmark.get("top100KnownRows"),
            "top100KnownEnrichment": validation_panel_benchmark.get("top100KnownEnrichment"),
            "balancedKnownRows": validation_panel_benchmark.get("balancedKnownRows"),
            "balancedNovelRows": validation_panel_benchmark.get("balancedNovelRows"),
            "positiveControlKnownRows": validation_panel_benchmark.get("positiveControlKnownRows"),
            "novelShortlistInterpretableRows": validation_panel_benchmark.get("novelShortlistInterpretableRows"),
            "evidence": "Experimental validation queues were audited against known drug-target positives, TopK enrichment, group calibration, and queue-level control/novel/interpretability composition.",
        },
        {
            "module": "Validation panel diversity and wave-1 coverage audit",
            "status": "completed_panel_design_qc",
            "candidateRows": validation_panel_diversity.get("candidateRows"),
            "abInterpretableEligibleRows": validation_panel_diversity.get("abInterpretableEligibleRows"),
            "top100UniqueDrugs": validation_panel_diversity.get("top100UniqueDrugs"),
            "top100UniqueTargets": validation_panel_diversity.get("top100UniqueTargets"),
            "top100UniqueScaffolds": validation_panel_diversity.get("top100UniqueScaffolds"),
            "balancedUniqueDrugs": validation_panel_diversity.get("balancedUniqueDrugs"),
            "balancedUniqueTargets": validation_panel_diversity.get("balancedUniqueTargets"),
            "balancedUniqueScaffolds": validation_panel_diversity.get("balancedUniqueScaffolds"),
            "wave1Rows": validation_panel_diversity.get("wave1Rows"),
            "wave1UniqueDrugs": validation_panel_diversity.get("wave1UniqueDrugs"),
            "wave1UniqueTargets": validation_panel_diversity.get("wave1UniqueTargets"),
            "wave1UniqueScaffolds": validation_panel_diversity.get("wave1UniqueScaffolds"),
            "wave1GateCounts": validation_panel_diversity.get("wave1GateCounts"),
            "evidence": "The assay-planning outputs were audited for concentration across disease direction, drug, target, scaffold, mechanism gate, and assay modality; a capped wave-1 validation panel was generated for practical follow-up design.",
        },
        {
            "module": "FDA label mechanism and action-type audit",
            "status": "completed_label_mechanism_qc",
            "fdaRows": fda_label_mechanism.get("fdaRows"),
            "fdaRowsWithTarget": fda_label_mechanism.get("fdaRowsWithTarget"),
            "expandedFdaDrugUniprotPairs": fda_label_mechanism.get("expandedFdaDrugUniprotPairs"),
            "candidateRows": fda_label_mechanism.get("candidateRows"),
            "fdaLabelTargetMatchRows": fda_label_mechanism.get("fdaLabelTargetMatchRows"),
            "fdaLabelMatchedDrugs": fda_label_mechanism.get("fdaLabelMatchedDrugs"),
            "fdaLabelMatchedTargets": fda_label_mechanism.get("fdaLabelMatchedTargets"),
            "candidateTargetFdaLabeledByAnyDrugRows": fda_label_mechanism.get("candidateTargetFdaLabeledByAnyDrugRows"),
            "top100LabelTargetRows": fda_label_mechanism.get("top100LabelTargetRows"),
            "balancedLabelTargetRows": fda_label_mechanism.get("balancedLabelTargetRows"),
            "wave1LabelTargetRows": fda_label_mechanism.get("wave1LabelTargetRows"),
            "evidence": "FDA Target ChEMBL IDs were expanded through the local target-component cache to UniProt accessions, then matched to candidate drug-protein pairs to separate label-target recalls, same-drug new-target extensions, and clinically labeled targets paired with different approved drugs.",
        },
        {
            "module": "FDA label temporal generalization audit",
            "status": "completed_retrospective_time_sliced_qc",
            "candidateRows": fda_label_temporal.get("candidateRows"),
            "exactLabelRows": fda_label_temporal.get("exactLabelRows"),
            "exactLabel2016PlusRows": fda_label_temporal.get("exactLabel2016PlusRows"),
            "exactLabel2021PlusRows": fda_label_temporal.get("exactLabel2021PlusRows"),
            "targetContext2021PlusRows": fda_label_temporal.get("targetContext2021PlusRows"),
            "top100Exact2016PlusRows": fda_label_temporal.get("top100Exact2016PlusRows"),
            "top100Exact2021PlusRows": fda_label_temporal.get("top100Exact2021PlusRows"),
            "top300Exact2016PlusRows": fda_label_temporal.get("top300Exact2016PlusRows"),
            "top300Exact2021PlusRows": fda_label_temporal.get("top300Exact2021PlusRows"),
            "balancedExact2021PlusRows": fda_label_temporal.get("balancedExact2021PlusRows"),
            "wave1Exact2021PlusRows": fda_label_temporal.get("wave1Exact2021PlusRows"),
            "evidence": "FDA label-target rows were stratified by approval-year era to stress-test whether the validation ranking and assay queues recover modern and recent label mechanisms; this is a retrospective time-sliced audit, not a true prospective deployment test.",
        },
        {
            "module": "Final multi-evidence candidate prioritization",
            "status": "completed",
            "candidateRows": final_priority.get("candidateRows"),
            "tierCounts": final_priority.get("tierCounts"),
            "reviewTrackCounts": final_priority.get("reviewTrackCounts"),
            "evidence": "Model score, disease evidence, KG path, ADMET, known-target recall, and pose sanity were integrated into final expert triage tables.",
        },
        {
            "module": "Final-priority known-target validation",
            "status": "completed",
            "candidateRows": final_priority_validation.get("candidateRows"),
            "knownDrugTargetRows": final_priority_validation.get("knownDrugTargetRows"),
            "recallAt100Pct": final_priority_validation.get("recallAt100Pct"),
            "precisionAt100Pct": final_priority_validation.get("precisionAt100Pct"),
            "enrichmentAt100": final_priority_validation.get("enrichmentAt100"),
            "evidence": "Known drug-target positives inside the final triage table were used to test whether the integrated priority score still enriches recoverable biology.",
        },
        {
            "module": "Final-priority ablation and robustness audit",
            "status": "completed",
            "candidateRows": final_priority_ablation.get("candidateRows"),
            "knownDrugTargetRows": final_priority_ablation.get("knownDrugTargetRows"),
            "finalRecallAt100Pct": metric(final_priority_ablation, "fullVariant", "recallAt100Pct"),
            "modelOnlyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "model_only", "recallAt100Pct"),
            "kgOnlyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "kg_only", "recallAt100Pct"),
            "withoutKgRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "without_kg_component", "recallAt100Pct"),
            "withoutRiskPenaltyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "without_risk_penalty", "recallAt100Pct"),
            "evidence": "Single-layer and leave-one-component-out rankings were compared against known-target positives, with TopK enrichment and bootstrap confidence intervals.",
        },
        {
            "module": "Novelty and known-mechanism leakage audit",
            "status": "completed",
            "candidateRows": novelty.get("candidateRows"),
            "directKnownMechanismRows": novelty.get("directKnownMechanismRows"),
            "strictNovelRows": metric(novelty, "strictNovel", "rows"),
            "strictNovelAbRows": metric(novelty, "strictNovel", "abRows"),
            "top100StrictNovelPct": metric(novelty, "topK", "100", "strictNovelPairPct"),
            "top100KnownMechanismPct": metric(novelty, "topK", "100", "directKnownMechanismPct"),
            "evidence": "Known benchmark pairs, direct KG drug-target edges, known disease-use hypotheses, safety context, sparse evidence, and strict novel pairs were separated for interpretation.",
        },
        {
            "module": "Cross-disease direction specificity audit",
            "status": "completed",
            "uniquePairs": specificity.get("uniquePairs"),
            "multiDirectionPairIds": specificity.get("multiDirectionPairIds"),
            "broadGeneralistPairIds": specificity.get("broadGeneralistPairIds"),
            "top100PairTopDirectionPct": metric(specificity, "topK", "100", "pairTopDirectionPct"),
            "top100DirectionSpecificPct": metric(specificity, "topK", "100", "directionSpecificPct"),
            "evidence": "The same drug-target pair was compared across disease directions to distinguish disease-focused candidates from broad multi-direction generalists.",
        },
        {
            "module": "TopK significance and stratified random baseline audit",
            "status": "completed",
            "iterations": significance.get("iterations"),
            "top100ObservedHits": metric(significance, "topK", "100", "observedHits"),
            "top100GlobalExpectedHits": metric(significance, "topK", "100", "globalExpectedHits"),
            "top100GlobalP": metric(significance, "topK", "100", "globalHypergeomPGe"),
            "top100StratifiedP": metric(significance, "topK", "100", "stratifiedPermutationPGe"),
            "top500ObservedHits": metric(significance, "topK", "500", "observedHits"),
            "top500GlobalExpectedHits": metric(significance, "topK", "500", "globalExpectedHits"),
            "evidence": "Observed known-target hits in final TopK were compared with global random and disease-direction-stratified random baselines.",
        },
        {
            "module": "Multi-evidence concordance audit",
            "status": "completed",
            "candidateRows": concordance.get("candidateRows"),
            "multiEvidenceRows": concordance.get("multiEvidenceRows"),
            "highConcordanceRows": concordance.get("highConcordanceRows"),
            "singleEvidenceDominatedRows": concordance.get("singleEvidenceDominatedRows"),
            "top100HighConcordancePct": metric(concordance, "topK", "100", "highConcordancePct"),
            "top500HighConcordancePct": metric(concordance, "topK", "500", "highConcordancePct"),
            "evidence": "Model, disease, KG, ADMET, structure, and direction-label evidence layers were counted to separate mature multi-evidence candidates from sparse-evidence priorities.",
        },
        {
            "module": "Candidate diversity and concentration audit",
            "status": "completed",
            "diverseRows": diversity.get("diverseRows"),
            "overallTop20UniqueDrugsMean": diversity.get("overallTop20UniqueDrugsMean"),
            "overallTop20UniqueFamiliesMean": diversity.get("overallTop20UniqueFamiliesMean"),
            "evidence": "Top candidate concentration was quantified by drug, target, and target family, and a diversity-capped expert shortlist was generated.",
        },
        {
            "module": "Computation artifact manifest and reproducibility inventory",
            "status": "completed",
            "artifactCount": manifest.get("artifact_count"),
            "totalSizeBytes": manifest.get("total_size_bytes"),
            "sourceScriptCount": len(manifest.get("source_scripts") or []),
            "latestArtifactMtimeUtc": manifest.get("latest_artifact_mtime_utc"),
            "evidence": "Core SOTA computation outputs were indexed with size, timestamp, row/column counts where available, SHA256 hashes, and source-script attribution.",
        },
    ]
    if vina_consensus_top300:
        completed_modules.append(
            {
                "module": "AutoDock Vina Top300 structural consensus rescoring extension",
                "status": "completed_vina_python_score_only_top300",
                "candidateRows": vina_consensus_top300.get("candidateRows"),
                "structureInputReadyRows": vina_consensus_top300.get("structureInputReadyRows"),
                "pdbqtReadyRows": vina_consensus_top300.get("pdbqtReadyRows"),
                "vinaScoredRows": vina_consensus_top300.get("vinaScoredRows"),
                "vinaConsensusSupportedRows": vina_consensus_top300.get("vinaConsensusSupportedRows"),
                "vinaConsensusSupportedPct": vina_consensus_top300.get("vinaConsensusSupportedPct"),
                "medianVinaScoreKcalMol": vina_consensus_top300.get("medianVinaScoreKcalMol"),
                "medianVinaOptimizedScoreKcalMol": vina_consensus_top300.get("medianVinaOptimizedScoreKcalMol"),
                "tierCounts": vina_consensus_top300.get("vinaConsensusTierCounts"),
                "evidence": "The AutoDock Vina score-only/local-optimization stress test has been extended from Top100 to Top300 final-priority candidates.",
            }
        )
    if vina_consensus_final3921:
        merge_status = vina_consensus_final3921.get("mergeStatus")
        missing_rows = vina_consensus_final3921.get("missingFinalRows")
        final3921_status = (
            "completed_vina_python_score_only_final3921"
            if not missing_rows
            else "partial_vina_python_score_only_final3921_merged"
        )
        completed_modules.append(
            {
                "module": "AutoDock Vina final3921 structural consensus rescoring",
                "status": final3921_status,
                "source": vina_consensus_final3921_source,
                "mergeStatus": merge_status,
                "mergedAuditRows": vina_consensus_final3921.get("mergedAuditRows"),
                "coveredFinalRows": vina_consensus_final3921.get("coveredFinalRows"),
                "missingFinalRows": missing_rows,
                "candidateRows": vina_consensus_final3921.get("candidateRows"),
                "structureInputReadyRows": vina_consensus_final3921.get("structureInputReadyRows"),
                "pdbqtReadyRows": vina_consensus_final3921.get("pdbqtReadyRows"),
                "vinaScoredRows": vina_consensus_final3921.get("vinaScoredRows"),
                "vinaConsensusSupportedRows": vina_consensus_final3921.get("vinaConsensusSupportedRows"),
                "vinaConsensusSupportedPct": vina_consensus_final3921.get("vinaConsensusSupportedPct"),
                "medianVinaScoreKcalMol": vina_consensus_final3921.get("medianVinaScoreKcalMol"),
                "medianVinaOptimizedScoreKcalMol": vina_consensus_final3921.get("medianVinaOptimizedScoreKcalMol"),
                "tierCounts": vina_consensus_final3921.get("vinaConsensusTierCounts"),
                "evidence": "Final-priority rows with available structure inputs are independently stress-tested with AutoDock Vina score-only plus local optimization; merged shard output is used when available.",
            }
        )
    if smina_rescoring_top300:
        completed_modules.append(
            {
                "module": "smina Top300 structural rescoring extension",
                "status": "completed_smina_score_only_top300",
                "candidateRows": smina_rescoring_top300.get("candidateRows"),
                "inputReadyRows": smina_rescoring_top300.get("inputReadyRows"),
                "sminaScoredRows": smina_rescoring_top300.get("sminaScoredRows"),
                "sminaSupportedRows": smina_rescoring_top300.get("sminaSupportedRows"),
                "sminaSupportedPct": smina_rescoring_top300.get("sminaSupportedPct"),
                "medianSminaAffinityKcalMol": smina_rescoring_top300.get("medianSminaAffinityKcalMol"),
                "tierCounts": smina_rescoring_top300.get("tierCounts"),
                "evidence": "The smina score-only stress test has been extended from Top100 to Top300 final-priority candidates.",
            }
        )
    if smina_rescoring_final3921:
        completed_modules.append(
            {
                "module": "smina final3921 structural rescoring",
                "status": "completed_smina_score_only_final3921",
                "candidateRows": smina_rescoring_final3921.get("candidateRows"),
                "inputReadyRows": smina_rescoring_final3921.get("inputReadyRows"),
                "sminaScoredRows": smina_rescoring_final3921.get("sminaScoredRows"),
                "sminaSupportedRows": smina_rescoring_final3921.get("sminaSupportedRows"),
                "sminaSupportedPct": smina_rescoring_final3921.get("sminaSupportedPct"),
                "medianSminaAffinityKcalMol": smina_rescoring_final3921.get("medianSminaAffinityKcalMol"),
                "tierCounts": smina_rescoring_final3921.get("tierCounts"),
                "evidence": "All final-priority rows with available structure inputs were independently rescored with smina score-only mode.",
            }
        )
    if gnina_rescoring_top300:
        completed_modules.append(
            {
                "module": "GNINA CNN Top300 structural rescoring extension",
                "status": "completed_gnina_cnn_top300_cpu",
                "candidateRows": gnina_rescoring_top300.get("candidateRows"),
                "inputReadyRows": gnina_rescoring_top300.get("inputReadyRows"),
                "gninaScoredRows": gnina_rescoring_top300.get("gninaScoredRows"),
                "gninaScoredPct": gnina_rescoring_top300.get("gninaScoredPct"),
                "gninaSupportedRows": gnina_rescoring_top300.get("gninaSupportedRows"),
                "gninaSupportedPct": gnina_rescoring_top300.get("gninaSupportedPct"),
                "medianGninaCnnScore": gnina_rescoring_top300.get("medianGninaCnnScore"),
                "medianGninaCnnAffinity": gnina_rescoring_top300.get("medianGninaCnnAffinity"),
                "statusCounts": gnina_rescoring_top300.get("statusCounts"),
                "tierCounts": gnina_rescoring_top300.get("tierCounts"),
                "evidence": "GNINA CNN rescoring has been extended from Top100 to Top300 in CPU-only mode, preserving GPU capacity for the full DiffDock expansion.",
            }
        )
    if gnina_rescoring_full3921:
        completed_modules.append(
            {
                "module": "GNINA CNN full final-candidate structural rescoring",
                "status": "completed_gnina_cnn_full3921_cpu",
                "candidateRows": gnina_rescoring_full3921.get("candidateRows"),
                "inputReadyRows": gnina_rescoring_full3921.get("inputReadyRows"),
                "gninaScoredRows": gnina_rescoring_full3921.get("gninaScoredRows"),
                "gninaScoredPct": gnina_rescoring_full3921.get("gninaScoredPct"),
                "gninaSupportedRows": gnina_rescoring_full3921.get("gninaSupportedRows"),
                "gninaSupportedPct": gnina_rescoring_full3921.get("gninaSupportedPct"),
                "medianGninaCnnScore": gnina_rescoring_full3921.get("medianGninaCnnScore"),
                "medianGninaCnnAffinity": gnina_rescoring_full3921.get("medianGninaCnnAffinity"),
                "statusCounts": gnina_rescoring_full3921.get("statusCounts"),
                "tierCounts": gnina_rescoring_full3921.get("tierCounts"),
                "evidence": "The GNINA CNN structural consensus stress test has been expanded to the full final-candidate structural queue in CPU-only mode.",
            }
        )

    local_blockers = [
        {
            "module": "Independent DTI ensemble with DrugBAN/DeepDTA",
            "status": "local_supervised_dti_complete_formal_pretrained_models_pending",
            "reason": "A local supervised independent DTI model has scored the Top1000 SMILES+sequence queue. Formal DrugBAN/DeepDTA pretrained inference remains pending because DrugBAN lacks DGL/DGLLife and DeepDTA has no validated pretrained checkpoint in the current workspace.",
            "neededInput": "Validated DrugBAN/DeepDTA model weights plus the required graph-learning dependencies or a prebuilt inference container if formal pretrained-model corroboration is required.",
        },
        {
            "module": "Optional Chai-1/AF3 external second-model complex extension",
            "status": "optional_external_extension_after_boltz2_top50_and_high_sampling",
            "reason": "Boltz-2 Top50 local second-model validation and high-sampling finalist validation are complete. Chai-1 or AF3-style reruns would be an optional external corroboration layer for final candidates rather than a blocker for the current SOTA package.",
            "neededInput": "A Chai-1 or AF3-compatible environment/API plus agreed finalist subset if an additional complex-prediction model is required.",
        },
    ]
    if not lincs_reversal:
        local_blockers.insert(
            0,
            {
                "module": "LINCS/CMap disease signature reversal",
                "status": "readiness_audit_complete_scoring_data_missing",
                "reason": "Candidate drug and disease-direction readiness has been audited, but scored LINCS/CMap reversal outputs are not present locally.",
                "neededInput": "Disease expression signatures plus LINCS/CMap drug perturbation profiles.",
            },
        )
    if gnina_status != "completed_gnina_cnn_top100":
        local_blockers.insert(
            1,
            {
                "module": "GNINA CNN structural consensus rescoring",
                "status": gnina_status,
                "reason": "AutoDock Vina and smina Top100 score-only rescoring are complete and the GNINA Top100 queue is input-ready, but GNINA CNN rescoring has not yet completed.",
                "neededInput": "Runtime repair or execution resume for the prepared GNINA Top100 queue.",
            },
        )

    active_computation = []
    if diffdock_full_progress:
        diffdock_full_processes = process_matches(["run_diffdock_full_job.py"])
        diffdock_full_running = bool(diffdock_full_processes)
        diffdock_full_status = "running" if diffdock_full_running else "paused_ready_to_resume"
        diffdock_full_evidence = (
            "This is the 913k-row full DiffDock expansion over the druggable-proteome candidate queue. "
            "A matching run_diffdock_full_job.py process is active."
            if diffdock_full_running
            else "This is the 913k-row full DiffDock expansion over the druggable-proteome candidate queue. "
            "No matching run_diffdock_full_job.py process is active, so the queue is paused and restartable from completed chunks."
        )
        active_computation.append(
            {
                "module": "Full druggable-proteome DiffDock queue",
                "status": diffdock_full_status,
                "completedJobs": diffdock_full_progress.get("completedJobs"),
                "totalJobs": diffdock_full_progress.get("totalJobs"),
                "completedJobPct": diffdock_full_progress.get("completedJobPct"),
                "scoredRows": diffdock_full_progress.get("scoredRows"),
                "totalRows": diffdock_full_progress.get("totalRows"),
                "scoredRowPct": diffdock_full_progress.get("scoredRowPct"),
                "completedOutputs": diffdock_full_progress.get("completedOutputs"),
                "missingOutputsInScoredJobs": diffdock_full_progress.get("missingOutputsInScoredJobs"),
                "activeLocks": diffdock_full_progress.get("activeLocks"),
                "busyGpuCount": metric(diffdock_full_progress, "gpu", "busyGpuCount"),
                "gpuCount": metric(diffdock_full_progress, "gpu", "gpuCount"),
                "etaHours": diffdock_full_progress.get("etaHours"),
                "etaDays": diffdock_full_progress.get("etaDays"),
                "estimatedFinishUtc": diffdock_full_progress.get("estimatedFinishUtc"),
                "processCount": len(diffdock_full_processes),
                "evidence": diffdock_full_evidence + " Missing outputs in finalized chunks are retained as technical missing outputs, not biological negatives.",
            }
        )
    if not vina_consensus_final3921 and (vina_final3921_progress or (root / "logs/vina_final3921.log").exists()):
        active_computation.append(
            {
                "module": "AutoDock Vina final3921 CPU-only structural consensus rescoring",
                "status": "running",
                "candidateRows": 3921,
                "processedRows": vina_final3921_progress.get("processed") or vina_final3921_ligand_cache_count,
                "progressBasis": "json_progress" if vina_final3921_progress.get("processed") else "ligand_pdbqt_cache",
                "ligandPdbqtCacheFiles": vina_final3921_ligand_cache_count,
                "receptorPdbqtCacheFiles": vina_final3921_receptor_cache_count,
                "latestPairId": vina_final3921_progress.get("pairId"),
                "latestVinaStatus": vina_final3921_progress.get("vinaStatus"),
                "logPath": "logs/vina_final3921.log",
                "evidence": "This CPU-only final-candidate Vina rescoring job was launched under low priority on cores 80-87 and does not use GPU resources reserved for the full DiffDock queue.",
            }
        )
    if vina_final3921_shards.get("logCount") and (
        not vina_consensus_final3921 or vina_consensus_final3921.get("missingFinalRows")
    ):
        active_computation.append(
            {
                "module": "AutoDock Vina final3921 CPU shard expansion",
                "status": "running",
                "candidateRows": 3921,
                "processedRows": vina_final3921_shards.get("processedRowsFromLogs"),
                "progressBasis": "sum_of_last_json_progress_by_shard_log",
                "completedShardAuditCount": vina_final3921_shards.get("completedShardAuditCount"),
                "shardLogCount": vina_final3921_shards.get("logCount"),
                "ligandPdbqtCacheFiles": vina_final3921_shards.get("ligandPdbqtCacheFiles"),
                "receptorPdbqtCacheFiles": vina_final3921_shards.get("receptorPdbqtCacheFiles"),
                "latestPairId": vina_final3921_shards.get("latestPairId"),
                "latestVinaStatus": vina_final3921_shards.get("latestVinaStatus"),
                "logPath": vina_final3921_shards.get("latestLogPath"),
                "evidence": "The final3921 Vina workload is also split into independent CPU-only shards so completed shard audits can be merged before the original single-process backup finishes.",
            }
        )
    if not gnina_rescoring_full3921 and (root / "logs/gnina_full3921_cpu.log").exists():
        active_computation.append(
            {
                "module": "GNINA CNN full final-candidate CPU-only rescoring",
                "status": "running",
                "candidateRows": gnina_full3921_candidate_rows,
                "processedRows": gnina_full3921_progress.get("processed") or gnina_full3921_output_count,
                "progressBasis": "json_progress" if gnina_full3921_progress.get("processed") else "output_sdf_count",
                "latestPairId": "",
                "latestGninaStatus": "",
                "logPath": "logs/gnina_full3921_cpu.log",
                "evidence": "The GNINA CNN structural consensus layer is being expanded over the full final-candidate structural queue in CPU-only mode, leaving GPUs available for DiffDock.",
            }
        )
    if diffdock_ligand_rescue and not diffdock_ligand_rescue.get("completedRows"):
        watcher_event = diffdock_ligand_rescue_watcher.get("event")
        watcher_waiting = watcher_event == "watcher_status" and not diffdock_ligand_rescue_watcher.get("ready")
        rescue_status = (
            "queued_for_auto_start_after_main_queue"
            if watcher_waiting
            else "prepared_not_started_gpu_busy"
        )
        rescue_evidence = (
            "CHEMBL3039504 triggered clustered DiffDock technical failures with the original multi-fragment/salt SDF. "
            "A largest-organic-parent rescue queue has been prepared and should be run only after a real GPU slot is free."
        )
        if watcher_waiting:
            rescue_evidence += (
                " A background watcher is active and will start this rescue queue after the main full DiffDock queue completes "
                "and GPUs are confirmed idle."
            )
        active_computation.append(
            {
                "module": "Full DiffDock ligand-specific technical rescue queue",
                "status": rescue_status,
                "ligandId": diffdock_ligand_rescue.get("ligandId"),
                "queuedRows": diffdock_ligand_rescue.get("queuedRows"),
                "jobs": diffdock_ligand_rescue.get("jobs"),
                "chunkSize": diffdock_ligand_rescue.get("chunkSize"),
                "jobIndex": diffdock_ligand_rescue.get("jobIndex"),
                "parentSdf": diffdock_ligand_rescue.get("parentSdf"),
                "watcherLog": "logs/diffdock_ligand_rescue_after_full.log"
                if diffdock_ligand_rescue_watcher
                else "",
                "watcherEvent": watcher_event,
                "watcherReady": diffdock_ligand_rescue_watcher.get("ready"),
                "watcherMainQueue": diffdock_ligand_rescue_watcher.get("mainQueue"),
                "evidence": rescue_evidence,
            }
        )
    if diffdock_multi_ligand_rescue:
        multi_queues = metric(diffdock_finalizer_status, "queues", default={}) or {}
        multi_status = metric(multi_queues, "multiRescue", default={}) or {}
        multi_completed = multi_status.get("completedJobs")
        multi_total = multi_status.get("totalJobs") or diffdock_multi_ligand_rescue.get("jobs")
        multi_all_complete = bool(multi_status.get("allComplete"))
        if not multi_all_complete:
            watcher_event = diffdock_multi_ligand_rescue_watcher.get("event")
            watcher_waiting = watcher_event == "watcher_status" and not diffdock_multi_ligand_rescue_watcher.get("ready")
            multi_rescue_status = (
                "queued_for_auto_start_after_single_ligand_rescue"
                if watcher_waiting
                else "prepared_not_started_gpu_busy"
            )
            active_computation.append(
                {
                    "module": "Full DiffDock multi-ligand technical rescue queue",
                    "status": multi_rescue_status,
                    "candidateLigands": diffdock_multi_ligand_rescue.get("candidateLigands"),
                    "queuedLigands": diffdock_multi_ligand_rescue.get("queuedLigands"),
                    "queuedRows": diffdock_multi_ligand_rescue.get("queuedRows"),
                    "jobs": multi_total,
                    "completedJobs": multi_completed,
                    "chunkSize": diffdock_multi_ligand_rescue.get("chunkSize"),
                    "latestAuditScoredLigands": diffdock_ligand_failure_audit.get("scoredLigands"),
                    "latestAuditScoredRows": diffdock_ligand_failure_audit.get("scoredRows"),
                    "latestAuditMissingRows": diffdock_ligand_failure_audit.get("missingRows"),
                    "latestAuditMissingPct": diffdock_ligand_failure_audit.get("missingPct"),
                    "latestAuditRescueRecommendedLigands": diffdock_ligand_failure_audit.get(
                        "rescueRecommendedLigands"
                    ),
                    "latestAuditRescueRecommendedRows": diffdock_ligand_failure_audit.get("rescueRecommendedRows"),
                    "latestAuditZeroCompletedChunks": diffdock_ligand_failure_audit.get("zeroCompletedChunks"),
                    "latestAuditMaskRotateZeroCompletedChunks": diffdock_ligand_failure_audit.get(
                        "maskRotateZeroCompletedChunks"
                    ),
                    "jobIndex": diffdock_multi_ligand_rescue.get("aggregateJobIndex"),
                    "watcherLog": "logs/diffdock_multi_ligand_rescue_after_single.log"
                    if diffdock_multi_ligand_rescue_watcher
                    else "",
                    "watcherEvent": watcher_event,
                    "watcherReady": diffdock_multi_ligand_rescue_watcher.get("ready"),
                    "watcherMainQueue": diffdock_multi_ligand_rescue_watcher.get("mainQueue"),
                    "watcherPrerequisiteQueues": diffdock_multi_ligand_rescue_watcher.get("prerequisiteQueues"),
                    "evidence": (
                        "This aggregate parent-ligand rescue queue covers ligands with clustered technical DiffDock "
                        "missing outputs after the full queue. A background watcher will run it only after the main "
                        "queue and the single-ligand rescue are complete and GPU slots are genuinely free."
                    ),
                }
            )
    if diffdock_finalizer_status:
        finalizer_processes = process_matches(["watch_and_finalize_diffdock_full_after_rescues.py"])
        finalizer_phase = diffdock_finalizer_status.get("phase") or "unknown"
        finalizer_ready = bool(diffdock_finalizer_status.get("readyToFinalize"))
        finalizer_completed = finalizer_phase == "completed"
        if not finalizer_completed:
            active_computation.append(
                {
                    "module": "Full DiffDock post-rescue finalization watcher",
                    "status": "running" if finalizer_processes else "waiting_status_present_process_not_detected",
                    "phase": finalizer_phase,
                    "readyToFinalize": finalizer_ready,
                    "queueStatuses": diffdock_finalizer_status.get("queues"),
                    "activeLockCount": len(diffdock_finalizer_status.get("activeLocks") or []),
                    "processCount": len(finalizer_processes),
                    "statusJson": "outputs/report_scale/diffdock_full_final_scores/finalize_after_rescues_status.json",
                    "logPath": "logs/diffdock_full_finalize_after_rescues.log",
                    "mergedScoresOut": "outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv",
                    "scoreCompatibleOut": "outputs/report_scale/diffdock_full_final_scores/scores/diffdock_full_final_merged.scores.csv",
                    "mergeSummaryJson": "outputs/report_scale/diffdock_full_final_scores/diffdock_full_final_merged_summary.json",
                    "evidence": (
                        "This watcher waits for the main full DiffDock queue plus single- and multi-ligand rescue "
                        "queues, then collects and merges score CSVs. Rescue-completed rows override prior "
                        "technical missing outputs; unresolved rows remain technical missing outputs, not "
                        "biological negatives."
                    ),
                }
            )
    if diffdock_post_finalization_status:
        post_processes = process_matches(["watch_and_run_full_diffdock_post_finalization.py"])
        post_phase = diffdock_post_finalization_status.get("phase") or "unknown"
        post_completed = post_phase == "completed"
        if not post_completed:
            active_computation.append(
                {
                    "module": "Full DiffDock post-finalization merge and audit watcher",
                    "status": "running" if post_processes else "waiting_status_present_process_not_detected",
                    "phase": post_phase,
                    "readyToRun": diffdock_post_finalization_status.get("readyToRun"),
                    "alreadyCompleted": diffdock_post_finalization_status.get("alreadyCompleted"),
                    "finalizerPhase": diffdock_post_finalization_status.get("finalizerPhase"),
                    "mergedScoresExists": diffdock_post_finalization_status.get("mergedScoresExists"),
                    "mergedScoreRows": diffdock_post_finalization_status.get("mergedScoreRows"),
                    "minMergedRows": diffdock_post_finalization_status.get("minMergedRows"),
                    "processCount": len(post_processes),
                    "statusJson": "outputs/report_scale/full_diffdock_final/post_finalization_status.json",
                    "logPath": "outputs/report_scale/full_diffdock_final/logs/post_finalization.log",
                    "expectedOutputs": {
                        "mergedStage5All": "outputs/report_scale/full_diffdock_final/full_diffdock_merged_stage5_all.csv",
                        "mergedStage5Top10000": "outputs/report_scale/full_diffdock_final/full_diffdock_merged_stage5_top10000.csv",
                        "mergedStage5Top100000": "outputs/report_scale/full_diffdock_final/full_diffdock_merged_stage5_top100000.csv",
                        "poseSanityTop10000": "outputs/sota_validation/pose_sanity_full_diffdock_top10000/candidate_pose_sanity_audit.csv",
                        "structureConfidenceTop10000": "outputs/sota_validation/structure_confidence_full_diffdock_top10000/candidate_structure_confidence_audit.csv",
                    },
                    "evidence": (
                        "This watcher waits for the final post-rescue full DiffDock score merge, then rebuilds "
                        "the Stage5-plus-structure candidate tables and refreshes the Top10000 pose sanity, "
                        "pocket-confidence, closure-summary, artifact-manifest, and dashboard-status outputs."
                    ),
                }
            )

    if experiment_closure_audit and experiment_closure_audit.get("overallStatus") != "complete":
        active_computation.append(
            {
                "module": "Strict experiment closure audit",
                "status": experiment_closure_audit.get("overallStatus"),
                "passedRequiredChecks": experiment_closure_audit.get("passedRequiredChecks"),
                "totalRequiredChecks": experiment_closure_audit.get("totalRequiredChecks"),
                "failedRequiredChecks": experiment_closure_audit.get("failedRequiredChecks"),
                "auditJson": "outputs/sota_validation/experiment_closure_audit/experiment_closure_audit.json",
                "auditMarkdown": "outputs/sota_validation/experiment_closure_audit/EXPERIMENT_CLOSURE_AUDIT.md",
                "evidence": (
                    "Machine-readable completion audit is present. The experiment remains open until all required "
                    "main DiffDock, rescue, final merge, post-finalization, website/status, and wet-lab package "
                    "checks pass."
                ),
            }
        )

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Computation-first SOTA closure summary; web/dashboard updates intentionally deferred.",
        "completedModules": completed_modules,
        "activeComputation": active_computation,
        "localBlockers": local_blockers,
        "experimentClosureAudit": {
            "overallStatus": experiment_closure_audit.get("overallStatus"),
            "passedRequiredChecks": experiment_closure_audit.get("passedRequiredChecks"),
            "totalRequiredChecks": experiment_closure_audit.get("totalRequiredChecks"),
            "failedRequiredChecks": experiment_closure_audit.get("failedRequiredChecks"),
            "createdUtc": experiment_closure_audit.get("createdUtc"),
        },
        "headline": {
            "diffdockCompletionPct": diffdock.get("completion_pct"),
            "diffdockCompletedRows": diffdock.get("completed"),
            "diffdockMissingRows": diffdock.get("missing"),
            "fullDiffdockCompletedJobs": diffdock_full_progress.get("completedJobs"),
            "fullDiffdockTotalJobs": diffdock_full_progress.get("totalJobs"),
            "fullDiffdockCompletedJobPct": diffdock_full_progress.get("completedJobPct"),
            "fullDiffdockScoredRows": diffdock_full_progress.get("scoredRows"),
            "fullDiffdockTotalRows": diffdock_full_progress.get("totalRows"),
            "fullDiffdockScoredRowPct": diffdock_full_progress.get("scoredRowPct"),
            "fullDiffdockCompletedOutputs": diffdock_full_progress.get("completedOutputs"),
            "fullDiffdockMissingOutputsInScoredJobs": diffdock_full_progress.get("missingOutputsInScoredJobs"),
            "fullDiffdockActiveLocks": diffdock_full_progress.get("activeLocks"),
            "fullDiffdockBusyGpuCount": metric(diffdock_full_progress, "gpu", "busyGpuCount"),
            "fullDiffdockGpuCount": metric(diffdock_full_progress, "gpu", "gpuCount"),
            "fullDiffdockEtaHours": diffdock_full_progress.get("etaHours"),
            "fullDiffdockEtaDays": diffdock_full_progress.get("etaDays"),
            "fullDiffdockEstimatedFinishUtc": diffdock_full_progress.get("estimatedFinishUtc"),
            "fullDiffdockLigandRescueLigandId": diffdock_ligand_rescue.get("ligandId"),
            "fullDiffdockLigandRescueQueuedRows": diffdock_ligand_rescue.get("queuedRows"),
            "fullDiffdockLigandRescueJobs": diffdock_ligand_rescue.get("jobs"),
            "fullDiffdockLigandRescueParentSdf": diffdock_ligand_rescue.get("parentSdf"),
            "fullDiffdockLigandRescueWatcherEvent": diffdock_ligand_rescue_watcher.get("event"),
            "fullDiffdockLigandRescueWatcherReady": diffdock_ligand_rescue_watcher.get("ready"),
            "fullDiffdockLigandRescueWatcherLog": "logs/diffdock_ligand_rescue_after_full.log"
            if diffdock_ligand_rescue_watcher
            else "",
            "fullDiffdockMultiLigandRescueCandidateLigands": diffdock_multi_ligand_rescue.get("candidateLigands"),
            "fullDiffdockMultiLigandRescueQueuedLigands": diffdock_multi_ligand_rescue.get("queuedLigands"),
            "fullDiffdockMultiLigandRescueQueuedRows": diffdock_multi_ligand_rescue.get("queuedRows"),
            "fullDiffdockMultiLigandRescueJobs": diffdock_multi_ligand_rescue.get("jobs"),
            "fullDiffdockLatestFailureAuditScoredLigands": diffdock_ligand_failure_audit.get("scoredLigands"),
            "fullDiffdockLatestFailureAuditScoredRows": diffdock_ligand_failure_audit.get("scoredRows"),
            "fullDiffdockLatestFailureAuditMissingRows": diffdock_ligand_failure_audit.get("missingRows"),
            "fullDiffdockLatestFailureAuditMissingPct": diffdock_ligand_failure_audit.get("missingPct"),
            "fullDiffdockLatestFailureAuditRescueRecommendedLigands": diffdock_ligand_failure_audit.get(
                "rescueRecommendedLigands"
            ),
            "fullDiffdockLatestFailureAuditRescueRecommendedRows": diffdock_ligand_failure_audit.get(
                "rescueRecommendedRows"
            ),
            "fullDiffdockLatestFailureAuditZeroCompletedChunks": diffdock_ligand_failure_audit.get(
                "zeroCompletedChunks"
            ),
            "fullDiffdockLatestFailureAuditMaskRotateZeroCompletedChunks": diffdock_ligand_failure_audit.get(
                "maskRotateZeroCompletedChunks"
            ),
            "fullDiffdockMultiLigandRescueWatcherEvent": diffdock_multi_ligand_rescue_watcher.get("event"),
            "fullDiffdockMultiLigandRescueWatcherReady": diffdock_multi_ligand_rescue_watcher.get("ready"),
            "fullDiffdockMultiLigandRescueWatcherLog": "logs/diffdock_multi_ligand_rescue_after_single.log"
            if diffdock_multi_ligand_rescue_watcher
            else "",
            "fullDiffdockFinalizerPhase": diffdock_finalizer_status.get("phase"),
            "fullDiffdockFinalizerReadyToFinalize": diffdock_finalizer_status.get("readyToFinalize"),
            "fullDiffdockFinalizerMainQueue": metric(diffdock_finalizer_status, "queues", "main"),
            "fullDiffdockFinalizerSingleRescueQueue": metric(diffdock_finalizer_status, "queues", "singleRescue"),
            "fullDiffdockFinalizerMultiRescueQueue": metric(diffdock_finalizer_status, "queues", "multiRescue"),
            "fullDiffdockFinalizerActiveLockCount": len(diffdock_finalizer_status.get("activeLocks") or []),
            "fullDiffdockFinalizerMergeCompletedRows": metric(
                diffdock_finalizer_status, "mergeSummary", "completedRows"
            ),
            "fullDiffdockFinalizerMergeMissingRows": metric(diffdock_finalizer_status, "mergeSummary", "missingRows"),
            "fullDiffdockFinalizerMergeCompletedPct": metric(
                diffdock_finalizer_status, "mergeSummary", "completedPct"
            ),
            "fullDiffdockFinalMergedScores": "outputs/report_scale/diffdock_scores_full_913170_with_rescues.csv",
            "fullDiffdockFinalScoreCompatibleCsv": "outputs/report_scale/diffdock_full_final_scores/scores/diffdock_full_final_merged.scores.csv",
            "fullDiffdockPostFinalizePhase": diffdock_post_finalization_status.get("phase"),
            "fullDiffdockPostFinalizeReadyToRun": diffdock_post_finalization_status.get("readyToRun"),
            "fullDiffdockPostFinalizeAlreadyCompleted": diffdock_post_finalization_status.get("alreadyCompleted"),
            "fullDiffdockPostFinalizeMergedScoreRows": diffdock_post_finalization_status.get("mergedScoreRows"),
            "fullDiffdockPostFinalizeMinMergedRows": diffdock_post_finalization_status.get("minMergedRows"),
            "top1000KgPathPct": kg_top.get("anyKgPathPct"),
            "top1000KgPathRows": kg_top.get("pathRows"),
            "poseReadablePct": pose_top.get("readablePosePct"),
            "posePassRows": metric(pose_top, "poseAuditStatusCounts", "pass"),
            "poseWarningRows": metric(pose_top, "poseAuditStatusCounts", "warning"),
            "structureConfidenceAuditedRows": structure_confidence.get("auditedCompletedRows"),
            "structureConfidenceMedianPocketMeanPlddt5A": structure_confidence.get("medianPocketMeanPlddt5A"),
            "structureConfidenceHighPct": structure_confidence.get("highConfidencePctOfAudited"),
            "structureConfidenceModerateOrHighPct": structure_confidence.get("moderateOrHighConfidencePctOfAudited"),
            "finalStructureJoinedRows": final_structure_confidence.get("joinedStructureRows"),
            "finalStructureTierCounts": final_structure_confidence.get("structureConfidenceTierCounts"),
            "finalStructureOriginalAveragePrecision": final_structure_confidence.get("originalAveragePrecision"),
            "finalStructureAdjustedAveragePrecision": final_structure_confidence.get("adjustedAveragePrecision"),
            "finalStructureOriginalRecallAt100Pct": final_structure_confidence.get("originalRecallAt100Pct"),
            "finalStructureAdjustedRecallAt100Pct": final_structure_confidence.get("adjustedRecallAt100Pct"),
            "finalStructureTop100OriginalTierCounts": final_structure_confidence.get("top100OriginalTierCounts"),
            "finalStructureTop100AdjustedTierCounts": final_structure_confidence.get("top100AdjustedTierCounts"),
            "finalStructureTop500OriginalLowConfidenceRows": final_structure_confidence.get("top500OriginalLowConfidenceRows"),
            "finalStructureSupportedNovelRows": final_structure_confidence.get("structureSupportedNovelRows"),
            "finalStructureSupportedNovelEligibleRows": final_structure_confidence.get("structureSupportedNovelEligibleRows"),
            "stage6Top10000CandidateRows": stage6_top10000.get("candidateRows"),
            "stage6Top10000UniqueDrugs": stage6_top10000.get("uniqueDrugs"),
            "stage6Top10000UniqueProteins": stage6_top10000.get("uniqueProteins"),
            "stage6Top10000DiffdockCompletedRows": stage6_top10000.get("diffdockCompletedRows"),
            "stage6Top10000DiffdockCompletedPct": stage6_top10000.get("diffdockCompletedPct"),
            "stage6Top10000DiffdockMissingRows": stage6_top10000.get("diffdockMissingRows"),
            "stage6Top10000Top100CompletedRows": stage6_top10000.get("top100CompletedRows"),
            "stage6Top10000Top100MissingRows": stage6_top10000.get("top100MissingRows"),
            "stage6Top10000Top300CompletedRows": stage6_top10000.get("top300CompletedRows"),
            "stage6Top10000Top300MissingRows": stage6_top10000.get("top300MissingRows"),
            "stage6Top10000PoseReadablePct": metric(stage6_top10000, "poseSanity", "readablePosePct"),
            "stage6Top10000PosePassRows": metric(stage6_top10000, "poseSanity", "poseAuditStatusCounts", "pass"),
            "stage6Top10000PoseWarningRows": metric(
                stage6_top10000, "poseSanity", "poseAuditStatusCounts", "warning"
            ),
            "stage6Top10000PocketAuditedRows": metric(
                stage6_top10000, "structureConfidence", "auditedCompletedRows"
            ),
            "stage6Top10000PocketMedianPlddt5A": metric(
                stage6_top10000, "structureConfidence", "medianPocketMeanPlddt5A"
            ),
            "stage6Top10000PocketModerateOrHighPct": metric(
                stage6_top10000, "structureConfidence", "moderateOrHighConfidencePctOfAudited"
            ),
            "stage6Top10000Top100OriginalTierCounts": metric(
                stage6_top10000, "structureConfidence", "top100OriginalTierCounts"
            ),
            "stage6Top10000Top100AdjustedTierCounts": metric(
                stage6_top10000, "structureConfidence", "top100AdjustedTierCounts"
            ),
            "poseInterpretabilityCandidateRows": pose_interpretability.get("candidateRows"),
            "poseInterpretabilityCompletedRows": pose_interpretability.get("completedRows"),
            "poseInterpretabilityContactResidueRows": pose_interpretability.get("contactResidueRows"),
            "poseInterpretabilityTierCounts": pose_interpretability.get("poseInterpretabilityTierCounts"),
            "poseInterpretabilityInterpretableRows": pose_interpretability.get("interpretableRows"),
            "poseInterpretabilityInterpretablePct": pose_interpretability.get("interpretablePct"),
            "poseInterpretabilityNovelRows": pose_interpretability.get("novelInterpretableRows"),
            "poseInterpretabilityKnownRows": pose_interpretability.get("knownInterpretableRows"),
            "poseInterpretabilityTop100Rows": pose_interpretability.get("top100InterpretableRows"),
            "poseInterpretabilityMedianPocketResidues5A": pose_interpretability.get("medianPocketResidues5A"),
            "poseInterpretabilityMedianPocketMeanPlddt5A": pose_interpretability.get("medianPocketMeanPlddt5A"),
            "poseQualityCandidateRows": pose_quality.get("candidateRows"),
            "poseQualityReadableRows": pose_quality.get("readableRows"),
            "poseQualityReadablePct": pose_quality.get("readablePct"),
            "poseQualitySupportedRows": pose_quality.get("qualitySupportedRows"),
            "poseQualitySupportedPct": pose_quality.get("qualitySupportedPct"),
            "poseQualityTierCounts": pose_quality.get("poseQualityTierCounts"),
            "poseQualityMedianScore": pose_quality.get("medianPoseQualityScore"),
            "poseQualityMedianContactCoverage4APct": pose_quality.get("medianContactCoverage4APct"),
            "poseQualityMedianPocketMeanPlddt5A": pose_quality.get("medianPocketMeanPlddt5A"),
            "poseQualityTop100SupportedRows": metric(pose_quality, "top100", "qualitySupportedRows"),
            "poseQualityTop100Rows": metric(pose_quality, "top100", "rows"),
            "poseQualityTop100TierCounts": metric(pose_quality, "top100", "tierCounts"),
            "poseQualityTop100KnownRows": metric(pose_quality, "top100", "knownRows"),
            "poseQualityTop100NovelRows": metric(pose_quality, "top100", "novelRows"),
            "standardPoseValidationCandidateRows": standard_pose_validation.get("candidateRows"),
            "standardPoseValidationInputReadyRows": standard_pose_validation.get("structureInputReadyRows"),
            "standardPoseValidationPoseBustersPassRows": standard_pose_validation.get("posebustersPassRows"),
            "standardPoseValidationPoseBustersPassPct": standard_pose_validation.get("posebustersPassPct"),
            "standardPoseValidationProlifOkRows": standard_pose_validation.get("prolifOkRows"),
            "standardPoseValidationProlifOkPct": standard_pose_validation.get("prolifOkPct"),
            "standardPoseValidationInteractionRows": standard_pose_validation.get("prolifInteractionRows"),
            "standardPoseValidationSupportedRows": standard_pose_validation.get("standardSupportedRows"),
            "standardPoseValidationSupportedPct": standard_pose_validation.get("standardSupportedPct"),
            "standardPoseValidationTierCounts": standard_pose_validation.get("standardPoseValidationTierCounts"),
            "standardPoseValidationMedianInteractions": standard_pose_validation.get("medianProlifInteractionCount"),
            "standardPoseValidationTop100KnownRows": metric(standard_pose_validation, "top100", "knownRows"),
            "standardPoseValidationTop100NovelRows": metric(standard_pose_validation, "top100", "novelRows"),
            "standardPoseValidationTop300CandidateRows": standard_pose_validation_top300.get("candidateRows"),
            "standardPoseValidationTop300InputReadyRows": standard_pose_validation_top300.get("structureInputReadyRows"),
            "standardPoseValidationTop300PoseBustersPassRows": standard_pose_validation_top300.get("posebustersPassRows"),
            "standardPoseValidationTop300PoseBustersPassPct": standard_pose_validation_top300.get("posebustersPassPct"),
            "standardPoseValidationTop300ProlifOkRows": standard_pose_validation_top300.get("prolifOkRows"),
            "standardPoseValidationTop300ProlifOkPct": standard_pose_validation_top300.get("prolifOkPct"),
            "standardPoseValidationTop300InteractionRows": standard_pose_validation_top300.get("prolifInteractionRows"),
            "standardPoseValidationTop300SupportedRows": standard_pose_validation_top300.get("standardSupportedRows"),
            "standardPoseValidationTop300SupportedPct": standard_pose_validation_top300.get("standardSupportedPct"),
            "standardPoseValidationTop300TierCounts": standard_pose_validation_top300.get("standardPoseValidationTierCounts"),
            "standardPoseValidationTop300MedianInteractions": standard_pose_validation_top300.get("medianProlifInteractionCount"),
            "standardPoseValidationFull3921CandidateRows": standard_pose_validation_full3921.get("candidateRows"),
            "standardPoseValidationFull3921InputReadyRows": standard_pose_validation_full3921.get(
                "structureInputReadyRows"
            ),
            "standardPoseValidationFull3921PoseBustersPassRows": standard_pose_validation_full3921.get(
                "posebustersPassRows"
            ),
            "standardPoseValidationFull3921PoseBustersPassPct": standard_pose_validation_full3921.get(
                "posebustersPassPct"
            ),
            "standardPoseValidationFull3921ProlifOkRows": standard_pose_validation_full3921.get("prolifOkRows"),
            "standardPoseValidationFull3921ProlifOkPct": standard_pose_validation_full3921.get("prolifOkPct"),
            "standardPoseValidationFull3921InteractionRows": standard_pose_validation_full3921.get(
                "prolifInteractionRows"
            ),
            "standardPoseValidationFull3921SupportedRows": standard_pose_validation_full3921.get(
                "standardSupportedRows"
            ),
            "standardPoseValidationFull3921SupportedPct": standard_pose_validation_full3921.get(
                "standardSupportedPct"
            ),
            "standardPoseValidationFull3921TierCounts": standard_pose_validation_full3921.get(
                "standardPoseValidationTierCounts"
            ),
            "standardPoseValidationFull3921MedianInteractions": standard_pose_validation_full3921.get(
                "medianProlifInteractionCount"
            ),
            "vinaConsensusCandidateRows": vina_consensus.get("candidateRows"),
            "vinaConsensusInputReadyRows": vina_consensus.get("structureInputReadyRows"),
            "vinaConsensusPdbqtReadyRows": vina_consensus.get("pdbqtReadyRows"),
            "vinaConsensusScoredRows": vina_consensus.get("vinaScoredRows"),
            "vinaConsensusScoredPct": vina_consensus.get("vinaScoredPct"),
            "vinaConsensusSupportedRows": vina_consensus.get("vinaConsensusSupportedRows"),
            "vinaConsensusSupportedPct": vina_consensus.get("vinaConsensusSupportedPct"),
            "vinaConsensusTierCounts": vina_consensus.get("vinaConsensusTierCounts"),
            "vinaConsensusMedianScoreKcalMol": vina_consensus.get("medianVinaScoreKcalMol"),
            "vinaConsensusMedianOptimizedScoreKcalMol": vina_consensus.get("medianVinaOptimizedScoreKcalMol"),
            "vinaConsensusTop100KnownRows": metric(vina_consensus, "top100", "knownRows"),
            "vinaConsensusTop100NovelRows": metric(vina_consensus, "top100", "novelRows"),
            "vinaConsensusTop300CandidateRows": vina_consensus_top300.get("candidateRows"),
            "vinaConsensusTop300InputReadyRows": vina_consensus_top300.get("structureInputReadyRows"),
            "vinaConsensusTop300PdbqtReadyRows": vina_consensus_top300.get("pdbqtReadyRows"),
            "vinaConsensusTop300ScoredRows": vina_consensus_top300.get("vinaScoredRows"),
            "vinaConsensusTop300ScoredPct": vina_consensus_top300.get("vinaScoredPct"),
            "vinaConsensusTop300SupportedRows": vina_consensus_top300.get("vinaConsensusSupportedRows"),
            "vinaConsensusTop300SupportedPct": vina_consensus_top300.get("vinaConsensusSupportedPct"),
            "vinaConsensusTop300TierCounts": vina_consensus_top300.get("vinaConsensusTierCounts"),
            "vinaConsensusTop300MedianScoreKcalMol": vina_consensus_top300.get("medianVinaScoreKcalMol"),
            "vinaConsensusTop300MedianOptimizedScoreKcalMol": vina_consensus_top300.get(
                "medianVinaOptimizedScoreKcalMol"
            ),
            "vinaConsensusFinal3921CandidateRows": vina_consensus_final3921.get("candidateRows"),
            "vinaConsensusFinal3921Source": vina_consensus_final3921_source,
            "vinaConsensusFinal3921MergeStatus": vina_consensus_final3921.get("mergeStatus"),
            "vinaConsensusFinal3921MergedAuditRows": vina_consensus_final3921.get("mergedAuditRows"),
            "vinaConsensusFinal3921CoveredFinalRows": vina_consensus_final3921.get("coveredFinalRows"),
            "vinaConsensusFinal3921MissingFinalRows": vina_consensus_final3921.get("missingFinalRows"),
            "vinaConsensusFinal3921InputReadyRows": vina_consensus_final3921.get("structureInputReadyRows"),
            "vinaConsensusFinal3921PdbqtReadyRows": vina_consensus_final3921.get("pdbqtReadyRows"),
            "vinaConsensusFinal3921ScoredRows": vina_consensus_final3921.get("vinaScoredRows"),
            "vinaConsensusFinal3921ScoredPct": vina_consensus_final3921.get("vinaScoredPct"),
            "vinaConsensusFinal3921SupportedRows": vina_consensus_final3921.get("vinaConsensusSupportedRows"),
            "vinaConsensusFinal3921SupportedPct": vina_consensus_final3921.get("vinaConsensusSupportedPct"),
            "vinaConsensusFinal3921TierCounts": vina_consensus_final3921.get("vinaConsensusTierCounts"),
            "vinaConsensusFinal3921MedianScoreKcalMol": vina_consensus_final3921.get("medianVinaScoreKcalMol"),
            "vinaConsensusFinal3921MedianOptimizedScoreKcalMol": vina_consensus_final3921.get(
                "medianVinaOptimizedScoreKcalMol"
            ),
            "vinaConsensusFinal3921ProcessedRows": vina_final3921_progress.get("processed")
            or vina_final3921_ligand_cache_count,
            "vinaConsensusFinal3921ProgressBasis": "json_progress"
            if vina_final3921_progress.get("processed")
            else "ligand_pdbqt_cache",
            "vinaConsensusFinal3921LigandPdbqtCacheFiles": vina_final3921_ligand_cache_count,
            "vinaConsensusFinal3921ReceptorPdbqtCacheFiles": vina_final3921_receptor_cache_count,
            "vinaConsensusFinal3921LatestPairId": vina_final3921_progress.get("pairId"),
            "vinaConsensusFinal3921LatestStatus": vina_final3921_progress.get("vinaStatus"),
            "vinaConsensusFinal3921ShardLogCount": vina_final3921_shards.get("logCount"),
            "vinaConsensusFinal3921CompletedShardAuditCount": vina_final3921_shards.get("completedShardAuditCount"),
            "vinaConsensusFinal3921ShardProcessedRows": vina_final3921_shards.get("processedRowsFromLogs"),
            "vinaConsensusFinal3921ShardLigandPdbqtCacheFiles": vina_final3921_shards.get("ligandPdbqtCacheFiles"),
            "vinaConsensusFinal3921ShardReceptorPdbqtCacheFiles": vina_final3921_shards.get("receptorPdbqtCacheFiles"),
            "vinaConsensusFinal3921ShardLatestPairId": vina_final3921_shards.get("latestPairId"),
            "vinaConsensusFinal3921ShardLatestStatus": vina_final3921_shards.get("latestVinaStatus"),
            "sminaCandidateRows": smina_rescoring.get("candidateRows"),
            "sminaInputReadyRows": smina_rescoring.get("inputReadyRows"),
            "sminaScoredRows": smina_rescoring.get("sminaScoredRows"),
            "sminaScoredPct": smina_rescoring.get("sminaScoredPct"),
            "sminaSupportedRows": smina_rescoring.get("sminaSupportedRows"),
            "sminaSupportedPct": smina_rescoring.get("sminaSupportedPct"),
            "sminaMedianAffinityKcalMol": smina_rescoring.get("medianSminaAffinityKcalMol"),
            "sminaTierCounts": smina_rescoring.get("tierCounts"),
            "sminaTop300CandidateRows": smina_rescoring_top300.get("candidateRows"),
            "sminaTop300InputReadyRows": smina_rescoring_top300.get("inputReadyRows"),
            "sminaTop300ScoredRows": smina_rescoring_top300.get("sminaScoredRows"),
            "sminaTop300ScoredPct": smina_rescoring_top300.get("sminaScoredPct"),
            "sminaTop300SupportedRows": smina_rescoring_top300.get("sminaSupportedRows"),
            "sminaTop300SupportedPct": smina_rescoring_top300.get("sminaSupportedPct"),
            "sminaTop300MedianAffinityKcalMol": smina_rescoring_top300.get("medianSminaAffinityKcalMol"),
            "sminaTop300TierCounts": smina_rescoring_top300.get("tierCounts"),
            "sminaFinal3921CandidateRows": smina_rescoring_final3921.get("candidateRows"),
            "sminaFinal3921InputReadyRows": smina_rescoring_final3921.get("inputReadyRows"),
            "sminaFinal3921ScoredRows": smina_rescoring_final3921.get("sminaScoredRows"),
            "sminaFinal3921ScoredPct": smina_rescoring_final3921.get("sminaScoredPct"),
            "sminaFinal3921SupportedRows": smina_rescoring_final3921.get("sminaSupportedRows"),
            "sminaFinal3921SupportedPct": smina_rescoring_final3921.get("sminaSupportedPct"),
            "sminaFinal3921MedianAffinityKcalMol": smina_rescoring_final3921.get("medianSminaAffinityKcalMol"),
            "sminaFinal3921TierCounts": smina_rescoring_final3921.get("tierCounts"),
            "gninaCandidateRows": gnina_rescoring.get("candidateRows"),
            "gninaInputReadyRows": gnina_rescoring.get("inputReadyRows"),
            "gninaInputReadyPct": gnina_rescoring.get("inputReadyPct"),
            "gninaBinaryAvailable": gnina_binary_available,
            "gninaRuntimeReady": gnina_runtime_ready,
            "gninaVersion": gnina_rescoring.get("gninaVersion") or gnina_runtime.get("version"),
            "gninaLdLibraryPath": gnina_rescoring.get("gninaLdLibraryPath") or gnina_runtime.get("ldLibraryPath"),
            "gninaScoredRows": gnina_rescoring.get("gninaScoredRows"),
            "gninaScoredPct": gnina_rescoring.get("gninaScoredPct"),
            "gninaSupportedRows": gnina_rescoring.get("gninaSupportedRows"),
            "gninaSupportedPct": gnina_rescoring.get("gninaSupportedPct"),
            "gninaStatusCounts": gnina_rescoring.get("statusCounts"),
            "gninaTierCounts": gnina_rescoring.get("tierCounts"),
            "gninaTop300CandidateRows": gnina_rescoring_top300.get("candidateRows"),
            "gninaTop300InputReadyRows": gnina_rescoring_top300.get("inputReadyRows"),
            "gninaTop300ScoredRows": gnina_rescoring_top300.get("gninaScoredRows"),
            "gninaTop300ScoredPct": gnina_rescoring_top300.get("gninaScoredPct"),
            "gninaTop300SupportedRows": gnina_rescoring_top300.get("gninaSupportedRows"),
            "gninaTop300SupportedPct": gnina_rescoring_top300.get("gninaSupportedPct"),
            "gninaTop300MedianCnnScore": gnina_rescoring_top300.get("medianGninaCnnScore"),
            "gninaTop300MedianCnnAffinity": gnina_rescoring_top300.get("medianGninaCnnAffinity"),
            "gninaTop300StatusCounts": gnina_rescoring_top300.get("statusCounts"),
            "gninaTop300TierCounts": gnina_rescoring_top300.get("tierCounts"),
            "gninaFull3921CandidateRows": gnina_rescoring_full3921.get("candidateRows"),
            "gninaFull3921InputReadyRows": gnina_rescoring_full3921.get("inputReadyRows"),
            "gninaFull3921ScoredRows": gnina_rescoring_full3921.get("gninaScoredRows"),
            "gninaFull3921ScoredPct": gnina_rescoring_full3921.get("gninaScoredPct"),
            "gninaFull3921SupportedRows": gnina_rescoring_full3921.get("gninaSupportedRows"),
            "gninaFull3921SupportedPct": gnina_rescoring_full3921.get("gninaSupportedPct"),
            "gninaFull3921MedianCnnScore": gnina_rescoring_full3921.get("medianGninaCnnScore"),
            "gninaFull3921MedianCnnAffinity": gnina_rescoring_full3921.get("medianGninaCnnAffinity"),
            "gninaFull3921StatusCounts": gnina_rescoring_full3921.get("statusCounts"),
            "gninaFull3921TierCounts": gnina_rescoring_full3921.get("tierCounts"),
            "gninaFull3921PreparedRows": gnina_full3921_candidate_rows,
            "gninaFull3921ProcessedRows": gnina_full3921_progress.get("processed") or gnina_full3921_output_count,
            "boltz2CandidateRows": boltz2_complex.get("candidateRows"),
            "boltz2CompletedRows": boltz2_complex.get("completedRows"),
            "boltz2CompletedPct": boltz2_complex.get("completedPct"),
            "boltz2OriginalLigandRows": boltz2_complex.get("initialOriginalLigandRows"),
            "boltz2RepairedParentLigandRows": boltz2_complex.get("repairedParentLigandRows"),
            "boltz2KnownCompletedRows": boltz2_complex.get("knownCompletedRows"),
            "boltz2NovelOrExtensionCompletedRows": boltz2_complex.get("novelOrExtensionCompletedRows"),
            "boltz2AbSupportedRows": boltz2_complex.get("abSupportedRows"),
            "boltz2AbSupportedPct": boltz2_complex.get("abSupportedPct"),
            "boltz2MedianConfidenceScore": boltz2_complex.get("medianConfidenceScore"),
            "boltz2MedianLigandIptm": boltz2_complex.get("medianLigandIptm"),
            "boltz2MedianAffinityProbabilityBinary": boltz2_complex.get("medianAffinityProbabilityBinary"),
            "boltz2TierCounts": boltz2_complex.get("tierCounts"),
            "boltz2HighCandidateRows": boltz2_high_sampling.get("candidateRows"),
            "boltz2HighCompletedRows": boltz2_high_sampling.get("completedRows"),
            "boltz2HighCompletedPct": boltz2_high_sampling.get("completedPct"),
            "boltz2HighKnownRows": boltz2_high_sampling.get("knownRows"),
            "boltz2HighNovelOrExtensionRows": boltz2_high_sampling.get("novelOrExtensionRows"),
            "boltz2HighAbSupportedRows": boltz2_high_sampling.get("abSupportedRows"),
            "boltz2HighAbSupportedPct": boltz2_high_sampling.get("abSupportedPct"),
            "boltz2HighMedianConfidenceScore": boltz2_high_sampling.get("medianHighConfidenceScore"),
            "boltz2HighMedianLigandIptm": boltz2_high_sampling.get("medianHighLigandIptm"),
            "boltz2HighMedianAffinityProbabilityBinary": boltz2_high_sampling.get("medianHighAffinityProbabilityBinary"),
            "boltz2HighMedianCompositeDeltaVsFast": boltz2_high_sampling.get("medianCompositeDeltaVsFast"),
            "boltz2HighTierCounts": boltz2_high_sampling.get("tierCounts"),
            "targetDruggabilityCandidateRows": target_druggability.get("candidateRows"),
            "targetDruggabilityUniqueTargets": target_druggability.get("uniqueTargets"),
            "targetDruggabilityExactUniProtMatchPct": target_druggability.get("exactUniProtMatchPct"),
            "targetDruggabilityPhase4Pct": target_druggability.get("phase4Pct"),
            "targetDruggabilityPhase3Or4Pct": target_druggability.get("phase3Or4Pct"),
            "targetDruggabilitySmallMoleculePct": target_druggability.get("smallMoleculeModalityPct"),
            "targetDruggabilityDirectionFitPct": target_druggability.get("directionDiseaseFitPct"),
            "targetDruggabilityTierCounts": target_druggability.get("targetDruggabilityTierCounts"),
            "targetDruggabilityOriginalAveragePrecision": target_druggability.get("originalAveragePrecision"),
            "targetDruggabilityAdjustedAveragePrecision": target_druggability.get("targetAdjustedAveragePrecision"),
            "targetDruggabilityOriginalRecallAt100Pct": target_druggability.get("originalRecallAt100Pct"),
            "targetDruggabilityAdjustedRecallAt100Pct": target_druggability.get("targetAdjustedRecallAt100Pct"),
            "targetDruggabilityHighTranslatabilityShortlistRows": target_druggability.get("highTranslatabilityShortlistRows"),
            "targetDruggabilityLowPhaseOrContextReviewRows": target_druggability.get("lowPhaseOrContextReviewRows"),
            "chemotypeCandidateRows": chemotype_diversity.get("candidateRows"),
            "chemotypeUniqueDrugs": chemotype_diversity.get("uniqueDrugs"),
            "chemotypeStructureMappedUniqueDrugPct": chemotype_diversity.get("structureMappedUniqueDrugPct"),
            "chemotypeValidStructureCandidatePct": chemotype_diversity.get("validStructureCandidatePct"),
            "chemotypeUniqueMurckoScaffolds": chemotype_diversity.get("uniqueMurckoScaffolds"),
            "chemotypeUniqueClusters": chemotype_diversity.get("uniqueChemotypeClusters"),
            "chemotypeTop100UniqueDrugs": chemotype_diversity.get("top100UniqueDrugs"),
            "chemotypeTop100UniqueScaffolds": chemotype_diversity.get("top100UniqueScaffolds"),
            "chemotypeTop100UniqueClusters": chemotype_diversity.get("top100UniqueChemotypeClusters"),
            "chemotypeTop100TopScaffoldPct": chemotype_diversity.get("top100TopScaffoldPct"),
            "chemotypeTop100TopClusterPct": chemotype_diversity.get("top100TopChemotypeClusterPct"),
            "chemotypeTop100ScaffoldHhi": chemotype_diversity.get("top100ScaffoldHHI"),
            "chemotypeTop500UniqueScaffolds": chemotype_diversity.get("top500UniqueScaffolds"),
            "chemotypeDiverseShortlistRows": chemotype_diversity.get("diverseShortlistRows"),
            "chemotypeDiverseShortlistUniqueDrugs": chemotype_diversity.get("diverseShortlistUniqueDrugs"),
            "chemotypeDiverseShortlistUniqueScaffolds": chemotype_diversity.get("diverseShortlistUniqueScaffolds"),
            "sotaReadyCandidateRows": sota_ready.get("candidateRows"),
            "sotaReadyUniqueDrugs": sota_ready.get("uniqueDrugs"),
            "sotaReadyUniqueTargets": sota_ready.get("uniqueTargets"),
            "sotaReadyTierCounts": sota_ready.get("tierCounts"),
            "sotaReadyActionCounts": sota_ready.get("actionCounts"),
            "sotaReadyOriginalAveragePrecision": sota_ready.get("originalAveragePrecision"),
            "sotaReadyAveragePrecision": sota_ready.get("sotaReadyAveragePrecision"),
            "sotaReadyOriginalTop100KnownRows": sota_ready.get("originalTop100KnownRows"),
            "sotaReadyTop100KnownRows": sota_ready.get("sotaReadyTop100KnownRows"),
            "sotaReadyTop100KnownPct": sota_ready.get("sotaReadyTop100KnownPct"),
            "sotaReadyTop100NovelRows": sota_ready.get("sotaReadyTop100NovelRows"),
            "sotaReadyTop100StructureABRows": sota_ready.get("sotaReadyTop100StructureABRows"),
            "sotaReadyTop100TargetABRows": sota_ready.get("sotaReadyTop100TargetABRows"),
            "sotaReadyTop100UniqueDrugs": sota_ready.get("sotaReadyTop100UniqueDrugs"),
            "sotaReadyTop100UniqueTargets": sota_ready.get("sotaReadyTop100UniqueTargets"),
            "sotaReadyExpertShortlistRows": sota_ready.get("expertShortlistRows"),
            "sotaReadyNovelShortlistRows": sota_ready.get("novelShortlistRows"),
            "sotaReadyDiverseShortlistRows": sota_ready.get("diverseShortlistRows"),
            "sotaReadyDiverseShortlistUniqueDrugs": sota_ready.get("diverseShortlistUniqueDrugs"),
            "sotaReadyDiverseShortlistUniqueTargets": sota_ready.get("diverseShortlistUniqueTargets"),
            "sotaReadyDiverseShortlistUniqueScaffolds": sota_ready.get("diverseShortlistUniqueScaffolds"),
            "networkProximityCandidateRows": network_proximity.get("candidateRows"),
            "networkProximityFinalPriorityRows": network_proximity.get("finalPriorityRows"),
            "networkProximitySource": network_proximity_source,
            "networkProximityStringGraphNodes": network_proximity.get("stringGraphNodes"),
            "networkProximityStringGraphEdges": network_proximity.get("stringGraphEdges"),
            "networkProximityCandidateRowsStringCoveredPct": network_proximity.get("candidateRowsStringCoveredPct"),
            "networkProximityUniqueTargetsStringCoveredPct": network_proximity.get("uniqueTargetsStringCoveredPct"),
            "networkProximityUniqueTargetsStringCovered": network_proximity.get("uniqueTargetsStringCovered"),
            "networkProximityUniqueTargetProteinPairs": network_proximity.get("uniqueTargetProteinPairs"),
            "networkProximityFinalPriorityTierCounts": network_proximity.get("finalPriorityTierCounts"),
            "sotaNetworkCandidateRows": sota_network.get("candidateRows"),
            "sotaNetworkTierCounts": sota_network.get("sotaNetworkTierCounts"),
            "sotaNetworkActionCounts": sota_network.get("sotaNetworkActionCounts"),
            "sotaNetworkPositiveRows": sota_network.get("networkPositiveRows"),
            "sotaNetworkDirectRows": sota_network.get("networkDirectRows"),
            "sotaNetworkCoverageGapRows": sota_network.get("networkCoverageGapRows"),
            "sotaNetworkPositivePct": sota_network.get("networkPositivePct"),
            "sotaNetworkOldAveragePrecision": sota_network.get("oldSotaReadyAveragePrecision"),
            "sotaNetworkAveragePrecision": sota_network.get("sotaNetworkAveragePrecision"),
            "sotaNetworkTop100KnownRows": metric(sota_network, "top100", "knownDrugTargetRows"),
            "sotaNetworkTop100NetworkPositiveRows": metric(sota_network, "top100", "networkPositiveRows"),
            "sotaNetworkTop100NetworkDirectRows": metric(sota_network, "top100", "networkDirectRows"),
            "sotaNetworkTop100NovelRows": metric(sota_network, "top100", "novelRows"),
            "sotaNetworkTop100TierCounts": metric(sota_network, "top100", "tierCounts"),
            "sotaNetworkTop100NetworkTierCounts": metric(sota_network, "top100", "networkTierCounts"),
            "tissueContextCandidateRows": tissue_context.get("candidateRows"),
            "tissueContextUniqueTargets": tissue_context.get("uniqueTargets"),
            "tissueContextTargetDirectionRows": tissue_context.get("targetDirectionRows"),
            "tissueContextCandidateHpaMatchedPct": tissue_context.get("candidateHpaMatchedPct"),
            "tissueContextUniqueTargetHpaMatchedPct": tissue_context.get("uniqueTargetHpaMatchedPct"),
            "tissueContextCandidatePositiveRows": tissue_context.get("candidateTissuePositiveRows"),
            "tissueContextCandidatePositivePct": tissue_context.get("candidateTissuePositivePct"),
            "tissueContextTargetDirectionPositiveRows": tissue_context.get("targetDirectionTissuePositiveRows"),
            "tissueContextTargetDirectionPositivePct": tissue_context.get("targetDirectionTissuePositivePct"),
            "tissueContextTierCounts": tissue_context.get("candidateTissueTierCounts"),
            "sotaContextTierCounts": tissue_context.get("sotaContextTierCounts"),
            "sotaContextOldAveragePrecision": tissue_context.get("oldSotaNetworkAveragePrecision"),
            "sotaContextAveragePrecision": tissue_context.get("sotaContextAveragePrecision"),
            "sotaContextTop100KnownRows": metric(tissue_context, "top100", "knownDrugTargetRows"),
            "sotaContextTop100NovelRows": metric(tissue_context, "top100", "novelRows"),
            "sotaContextTop100TissuePositiveRows": metric(tissue_context, "top100", "tissuePositiveRows"),
            "sotaContextTop100HpaMatchedRows": metric(tissue_context, "top100", "hpaMatchedRows"),
            "sotaContextTop100NetworkPositiveRows": metric(tissue_context, "top100", "networkPositiveRows"),
            "sotaContextTop100TissueTierCounts": metric(tissue_context, "top100", "tissueTierCounts"),
            "gtexContextCandidateRows": gtex_context.get("candidateRows"),
            "gtexContextUniqueTargets": gtex_context.get("uniqueTargets"),
            "gtexContextTargetDirectionRows": gtex_context.get("targetDirectionRows"),
            "gtexContextRelease": gtex_context.get("gtexRelease"),
            "gtexContextCandidateMatchedRows": gtex_context.get("candidateGtexMatchedRows"),
            "gtexContextCandidateMatchedPct": gtex_context.get("candidateGtexMatchedPct"),
            "gtexContextCandidatePositiveRows": gtex_context.get("candidateGtexPositiveRows"),
            "gtexContextCandidatePositivePct": gtex_context.get("candidateGtexPositivePct"),
            "gtexContextTargetDirectionPositiveRows": gtex_context.get("targetDirectionGtexPositiveRows"),
            "gtexContextTargetDirectionPositivePct": gtex_context.get("targetDirectionGtexPositivePct"),
            "gtexContextTierCounts": gtex_context.get("candidateGtexTierCounts"),
            "sotaGtexContextTierCounts": gtex_context.get("sotaGtexContextTierCounts"),
            "sotaGtexContextOldAveragePrecision": gtex_context.get("oldSotaContextAveragePrecision"),
            "sotaGtexContextAveragePrecision": gtex_context.get("sotaGtexContextAveragePrecision"),
            "sotaGtexContextTop100KnownRows": metric(gtex_context, "top100", "knownDrugTargetRows"),
            "sotaGtexContextTop100NovelRows": metric(gtex_context, "top100", "novelRows"),
            "sotaGtexContextTop100GtexPositiveRows": metric(gtex_context, "top100", "gtexPositiveRows"),
            "sotaGtexContextTop100GtexMatchedRows": metric(gtex_context, "top100", "gtexMatchedRows"),
            "sotaGtexContextTop100HpaPositiveRows": metric(gtex_context, "top100", "hpaPositiveRows"),
            "sotaGtexContextTop100NetworkPositiveRows": metric(gtex_context, "top100", "networkPositiveRows"),
            "sotaGtexContextTop100GtexTierCounts": metric(gtex_context, "top100", "gtexTierCounts"),
            "depmapOncologyRows": depmap_dependency.get("oncologyCandidateRows"),
            "depmapUniqueOncologyTargets": depmap_dependency.get("uniqueOncologyTargets"),
            "depmapRelease": depmap_dependency.get("depmapRelease"),
            "depmapReleaseDate": depmap_dependency.get("depmapReleaseDate"),
            "depmapMatchedTargetsPct": depmap_dependency.get("depmapMatchedTargetsPct"),
            "depmapPositiveCandidateRows": depmap_dependency.get("depmapDependencyPositiveCandidateRows"),
            "depmapPositiveCandidatePct": depmap_dependency.get("depmapDependencyPositiveCandidateRowsPct"),
            "depmapPositiveTargets": depmap_dependency.get("depmapDependencyPositiveTargets"),
            "depmapPositiveTargetsPct": depmap_dependency.get("depmapDependencyPositiveTargetsPct"),
            "depmapCandidateTierCounts": depmap_dependency.get("candidateTierCounts"),
            "depmapTop100Oncology": depmap_dependency.get("top100Oncology"),
            "lincsReadinessCandidateRows": lincs_readiness.get("sourceRows"),
            "lincsReadinessUniqueDrugs": lincs_readiness.get("uniqueCandidateDrugs"),
            "lincsReadinessUniqueTargets": lincs_readiness.get("uniqueCandidateTargets"),
            "lincsReadinessStructureMappedDrugs": lincs_readiness.get("structureMappedUniqueDrugs"),
            "lincsReadinessStructureMappedDrugPct": lincs_readiness.get("structureMappedUniqueDrugsPct"),
            "lincsReadinessPerturbationFiles": lincs_readiness.get("perturbationSignatureCandidateFiles"),
            "lincsReadinessDiseaseSignatureFiles": lincs_readiness.get("diseaseSignatureCandidateFiles"),
            "lincsReadinessCalculationReadiness": (
                "superseded_by_completed_lincs_cmap_reversal"
                if lincs_reversal
                else lincs_readiness.get("calculationReadiness")
            ),
            "lincsReadinessSupersededByReversal": bool(lincs_reversal),
            "lincsReversalCandidateRows": lincs_reversal.get("candidateRows"),
            "lincsReversalCandidateDrugs": lincs_reversal.get("candidateDrugs"),
            "lincsReversalMappedCandidateDrugs": lincs_reversal.get("mappedCandidateDrugs"),
            "lincsReversalMappedCandidateRows": lincs_reversal.get("mappedCandidateRows"),
            "lincsReversalQcTrtCpSignatureRows": lincs_reversal.get("allQcTrtCpSignatureRows"),
            "lincsReversalSelectedSignatureCount": lincs_reversal.get("selectedSignatureCount"),
            "lincsReversalDrugDirectionScoreRows": lincs_reversal.get("drugDirectionScoreRows"),
            "lincsReversalPositiveRows": lincs_reversal.get("candidatePositiveReversalRows"),
            "lincsReversalTop100PositiveRows": lincs_reversal.get("top100PositiveReversalRows"),
            "lincsReversalTierCounts": lincs_reversal.get("candidateTierCounts"),
            "expertReviewCandidateRows": expert_review.get("candidateRows"),
            "expertReviewPanelRows": expert_review.get("panelRows"),
            "expertReviewWave1Rows": expert_review.get("wave1Rows"),
            "expertReviewPanelUniqueDrugs": expert_review.get("panelUniqueDrugs"),
            "expertReviewPanelUniqueTargets": expert_review.get("panelUniqueTargets"),
            "expertReviewPanelUniqueScaffolds": expert_review.get("panelUniqueScaffolds"),
            "expertReviewPanelDirectionCounts": expert_review.get("panelDirectionCounts"),
            "expertReviewNoveltyGroupCounts": expert_review.get("panelNoveltyGroupCounts"),
            "expertReviewCmapTierCounts": expert_review.get("panelCmapTierCounts"),
            "expertReviewStructureTierCounts": expert_review.get("panelStructureTierCounts"),
            "expertReviewAdmetTierCounts": expert_review.get("panelAdmetTierCounts"),
            "externalInputCandidateRows": metric(external_inputs, "coverage", "candidateRows"),
            "externalInputUniquePairs": metric(external_inputs, "coverage", "uniquePairs"),
            "externalInputUniqueDrugs": metric(external_inputs, "coverage", "uniqueDrugs"),
            "externalInputUniqueProteins": metric(external_inputs, "coverage", "uniqueProteins"),
            "externalInputSmilesRows": metric(external_inputs, "coverage", "canonicalSmilesRows"),
            "externalInputSequenceRows": metric(external_inputs, "coverage", "sequenceMappedRows"),
            "externalInputSdfRows": metric(external_inputs, "coverage", "confidenceSdfExistingRows"),
            "externalInputReceptorPdbRows": metric(external_inputs, "coverage", "receptorPdbExistingRows"),
            "externalInputGninaRows": metric(external_inputs, "queues", "gninaTop100", "rows"),
            "externalInputGninaReadyRows": metric(external_inputs, "queues", "gninaTop100", "inputReadyRows"),
            "externalInputGninaUniqueDrugs": metric(external_inputs, "queues", "gninaTop100", "uniqueDrugs"),
            "externalInputGninaUniqueProteins": metric(external_inputs, "queues", "gninaTop100", "uniqueProteins"),
            "externalInputBoltzRows": metric(external_inputs, "queues", "boltzChaiTop50", "rows"),
            "externalInputBoltzReadyRows": metric(external_inputs, "queues", "boltzChaiTop50", "inputReadyRows"),
            "externalInputBoltzUniqueDrugs": metric(external_inputs, "queues", "boltzChaiTop50", "uniqueDrugs"),
            "externalInputBoltzUniqueProteins": metric(external_inputs, "queues", "boltzChaiTop50", "uniqueProteins"),
            "externalInputDtiRows": metric(external_inputs, "queues", "independentDtiTop1000", "rows"),
            "externalInputDtiReadyRows": metric(external_inputs, "queues", "independentDtiTop1000", "inputReadyRows"),
            "externalInputDtiUniqueDrugs": metric(external_inputs, "queues", "independentDtiTop1000", "uniqueDrugs"),
            "externalInputDtiUniqueProteins": metric(external_inputs, "queues", "independentDtiTop1000", "uniqueProteins"),
            "externalInputLigandAssetCount": metric(external_inputs, "sharedInputAssets", "topCandidateLigandCount"),
            "externalInputProteinAssetCount": metric(external_inputs, "sharedInputAssets", "topCandidateProteinCount"),
            "independentDtiCandidateRows": independent_dti.get("candidateRows"),
            "independentDtiScoredRows": independent_dti.get("scoredRows"),
            "independentDtiUniqueDrugs": independent_dti.get("uniqueCandidateDrugs"),
            "independentDtiUniqueProteins": independent_dti.get("uniqueCandidateProteins"),
            "independentDtiKnownRows": independent_dti.get("knownRows"),
            "independentDtiNovelRows": independent_dti.get("novelRows"),
            "independentDtiAbSupportedRows": independent_dti.get("abSupportedRows"),
            "independentDtiAbSupportedPct": independent_dti.get("abSupportedPct"),
            "independentDtiTierCounts": independent_dti.get("tierCounts"),
            "independentDtiKnownBenchmarkAuroc": metric(independent_dti, "candidateKnownBenchmark", "auroc"),
            "independentDtiKnownBenchmarkAveragePrecision": metric(
                independent_dti, "candidateKnownBenchmark", "averagePrecision"
            ),
            "independentDtiTop50KnownHits": metric(independent_dti_topk.get(50, {}), "knownHits"),
            "independentDtiTop50RecallPct": metric(independent_dti_topk.get(50, {}), "recallPct"),
            "independentDtiTop50Enrichment": metric(independent_dti_topk.get(50, {}), "enrichmentVsQueueBaseline"),
            "independentDtiTop100KnownHits": metric(independent_dti_topk.get(100, {}), "knownHits"),
            "independentDtiTop100RecallPct": metric(independent_dti_topk.get(100, {}), "recallPct"),
            "independentDtiTop100Enrichment": metric(independent_dti_topk.get(100, {}), "enrichmentVsQueueBaseline"),
            "independentDtiTop300KnownHits": metric(independent_dti_topk.get(300, {}), "knownHits"),
            "independentDtiTop300RecallPct": metric(independent_dti_topk.get(300, {}), "recallPct"),
            "independentDtiTop300Enrichment": metric(independent_dti_topk.get(300, {}), "enrichmentVsQueueBaseline"),
            "independentDtiPairHoldoutAuroc": metric(independent_dti_validation.get("pair_stratified_holdout", {}), "auroc"),
            "independentDtiDrugHoldoutAuroc": metric(independent_dti_validation.get("drug_group_holdout", {}), "auroc"),
            "independentDtiTargetHoldoutAuroc": metric(
                independent_dti_validation.get("target_group_holdout", {}), "auroc"
            ),
            "independentDtiFull3921CandidateRows": independent_dti_full3921.get("candidateRows"),
            "independentDtiFull3921ScoredRows": independent_dti_full3921.get("scoredRows"),
            "independentDtiFull3921UniqueDrugs": independent_dti_full3921.get("uniqueCandidateDrugs"),
            "independentDtiFull3921UniqueProteins": independent_dti_full3921.get("uniqueCandidateProteins"),
            "independentDtiFull3921KnownRows": independent_dti_full3921.get("knownRows"),
            "independentDtiFull3921NovelRows": independent_dti_full3921.get("novelRows"),
            "independentDtiFull3921AbSupportedRows": independent_dti_full3921.get("abSupportedRows"),
            "independentDtiFull3921AbSupportedPct": independent_dti_full3921.get("abSupportedPct"),
            "independentDtiFull3921TierCounts": independent_dti_full3921.get("tierCounts"),
            "independentDtiFull3921KnownBenchmarkAuroc": metric(
                independent_dti_full3921, "candidateKnownBenchmark", "auroc"
            ),
            "independentDtiFull3921KnownBenchmarkAveragePrecision": metric(
                independent_dti_full3921, "candidateKnownBenchmark", "averagePrecision"
            ),
            "independentDtiFull3921Top50KnownHits": metric(
                independent_dti_full3921_topk.get(50, {}), "knownHits"
            ),
            "independentDtiFull3921Top50RecallPct": metric(
                independent_dti_full3921_topk.get(50, {}), "recallPct"
            ),
            "independentDtiFull3921Top50Enrichment": metric(
                independent_dti_full3921_topk.get(50, {}), "enrichmentVsQueueBaseline"
            ),
            "independentDtiFull3921Top100KnownHits": metric(
                independent_dti_full3921_topk.get(100, {}), "knownHits"
            ),
            "independentDtiFull3921Top100RecallPct": metric(
                independent_dti_full3921_topk.get(100, {}), "recallPct"
            ),
            "independentDtiFull3921Top100Enrichment": metric(
                independent_dti_full3921_topk.get(100, {}), "enrichmentVsQueueBaseline"
            ),
            "independentDtiFull3921Top300KnownHits": metric(
                independent_dti_full3921_topk.get(300, {}), "knownHits"
            ),
            "independentDtiFull3921Top300RecallPct": metric(
                independent_dti_full3921_topk.get(300, {}), "recallPct"
            ),
            "independentDtiFull3921Top300Enrichment": metric(
                independent_dti_full3921_topk.get(300, {}), "enrichmentVsQueueBaseline"
            ),
            "independentDtiFull3921PairHoldoutAuroc": metric(
                independent_dti_full3921_validation.get("pair_stratified_holdout", {}), "auroc"
            ),
            "independentDtiFull3921DrugHoldoutAuroc": metric(
                independent_dti_full3921_validation.get("drug_group_holdout", {}), "auroc"
            ),
            "independentDtiFull3921TargetHoldoutAuroc": metric(
                independent_dti_full3921_validation.get("target_group_holdout", {}), "auroc"
            ),
            "mlAdmetDrugRows": ml_admet.get("drugRows"),
            "mlAdmetCandidateRows": ml_admet.get("candidateRows"),
            "mlAdmetTrainedEndpointCount": ml_admet.get("trainedEndpointCount"),
            "mlAdmetTrainedEndpoints": ml_admet.get("trainedEndpoints"),
            "mlAdmetLowRiskCandidateRows": ml_admet.get("lowMlAdmetRiskCandidateRows"),
            "mlAdmetLowRiskCandidatePct": ml_admet.get("lowMlAdmetRiskCandidatePct"),
            "mlAdmetHighRiskCandidateRows": ml_admet.get("highMlAdmetRiskCandidateRows"),
            "mlAdmetHighRiskCandidatePct": ml_admet.get("highMlAdmetRiskCandidatePct"),
            "mlAdmetCandidateSafetyTierCounts": ml_admet.get("candidateMlAdmetSafetyTierCounts"),
            "sotaMlAdmetTierCounts": ml_admet.get("sotaMlAdmetTierCounts"),
            "mlAdmetTop100LowRiskRows": metric(ml_admet, "top100", "lowMlAdmetRiskRows"),
            "mlAdmetTop100HighRiskRows": metric(ml_admet, "top100", "highMlAdmetRiskRows"),
            "mlAdmetTop100KnownRows": metric(ml_admet, "top100", "knownDrugTargetRows"),
            "mlAdmetTop100NovelRows": metric(ml_admet, "top100", "novelRows"),
            "mlAdmetTop100SafetyTierCounts": metric(ml_admet, "top100", "mlAdmetSafetyTierCounts"),
            "modelFeasibilityReadinessCounts": model_feasibility.get("layerCountsByReadiness"),
            "modelFeasibilityPriorityQueue": model_feasibility.get("priorityQueue"),
            "rankStabilityConsensusTop100Rows": rank_stability.get("consensusTop100Rows3PlusMethods"),
            "rankStabilityConsensusTop500Rows": rank_stability.get("consensusTop500Rows3PlusMethods"),
            "rankStabilityAllMethodTop100IntersectionRows": rank_stability.get("allMethodTop100IntersectionRows"),
            "rankStabilityAllMethodTop100IntersectionPct": rank_stability.get("allMethodTop100IntersectionPct"),
            "rankStabilityFinalVsSotaTop100OverlapRows": rank_stability.get("finalVsSotaReadyTop100OverlapRows"),
            "rankStabilityFinalVsSotaTop100Jaccard": rank_stability.get("finalVsSotaReadyTop100Jaccard"),
            "rankStabilityFinalVsSotaSpearman": rank_stability.get("finalVsSotaReadySpearman"),
            "rankStabilityConsensusTop100KnownRows": rank_stability.get("consensusTop100KnownRows"),
            "rankStabilityConsensusTop100NovelRows": rank_stability.get("consensusTop100NovelRows"),
            "rankStabilityLargeDeltaRows": rank_stability.get("largeRankDeltaRowsAbs1000"),
            "experimentalValidationCandidateRows": experimental_validation.get("candidateRows"),
            "experimentalValidationTierCounts": experimental_validation.get("validationTierCounts"),
            "experimentalValidationGateCounts": experimental_validation.get("validationGateCounts"),
            "experimentalValidationAssayCounts": experimental_validation.get("assayModalityCounts"),
            "experimentalValidationExperimentReadyRows": experimental_validation.get("experimentReadyRows"),
            "experimentalValidationReviewReadyRows": experimental_validation.get("reviewReadyRows"),
            "experimentalValidationExperimentOrReviewReadyRows": experimental_validation.get("experimentOrReviewReadyRows"),
            "experimentalValidationNovelReadyRows": experimental_validation.get("novelExperimentOrReviewReadyRows"),
            "experimentalValidationPositiveControlReadyRows": experimental_validation.get("positiveControlReadyRows"),
            "experimentalValidationBalancedPanelRows": experimental_validation.get("balancedPanelRows"),
            "experimentalValidationBalancedDirections": experimental_validation.get("balancedPanelDirections"),
            "experimentalValidationBalancedUniqueDrugs": experimental_validation.get("balancedPanelUniqueDrugs"),
            "experimentalValidationBalancedUniqueTargets": experimental_validation.get("balancedPanelUniqueTargets"),
            "experimentalValidationBalancedUniqueScaffolds": experimental_validation.get("balancedPanelUniqueScaffolds"),
            "experimentalValidationNovelPanelRows": experimental_validation.get("novelValidationPanelRows"),
            "experimentalValidationTop100ReadyRows": experimental_validation.get("top100ExperimentOrReviewReadyRows"),
            "experimentalValidationTop100NovelRows": experimental_validation.get("top100NovelRows"),
            "wetlabFocusedTop12Rows": wetlab_validation.get("focusedTop12Rows"),
            "wetlabFocusedCore6Rows": wetlab_validation.get("focusedCore6Rows"),
            "wetlabWave1Rows": wetlab_validation.get("wave1Rows"),
            "wetlabExpertTop50Rows": wetlab_validation.get("expertTop50Rows"),
            "wetlabPurchaseAndAssayQueueRows": wetlab_validation.get("purchaseAndAssayQueueRows"),
            "wetlabPreExperimentDecisionRows": wetlab_validation.get("preExperimentDecisionRows"),
            "wetlabPreExperimentDecisionCounts": wetlab_validation.get("preExperimentDecisionCounts"),
            "wetlabFirstExperimentPanelRows": wetlab_validation.get("firstExperimentPanelRows"),
            "wetlabFirstExperimentPanelRoleCounts": wetlab_validation.get("firstExperimentPanelRoleCounts"),
            "wetlabFirstExperimentPanelDirectionCounts": wetlab_validation.get("firstExperimentPanelDirectionCounts"),
            "wetlabPrePurchaseFocusRows": wetlab_validation.get("prePurchaseFocusRows"),
            "wetlabPrePurchaseTierCounts": wetlab_validation.get("prePurchaseTierCounts"),
            "wetlabPrePurchaseActionCounts": wetlab_validation.get("prePurchaseActionCounts"),
            "wetlabExperimentExecutionProtocolRows": wetlab_validation.get("experimentExecutionProtocolRows"),
            "wetlabExperimentExecutionProtocolCoreRows": wetlab_validation.get("experimentExecutionProtocolCoreRows"),
            "wetlabExperimentExecutionProtocolExtensionRows": wetlab_validation.get("experimentExecutionProtocolExtensionRows"),
            "wetlabExperimentExecutionProtocolAssayCounts": wetlab_validation.get("experimentExecutionProtocolAssayCounts"),
            "wetlabProcurementPlatformChecklistRows": wetlab_validation.get("procurementPlatformChecklistRows"),
            "wetlabProcurementPlatformChecklistCoreRows": wetlab_validation.get("procurementPlatformChecklistCoreRows"),
            "wetlabProcurementPlatformChecklistExtensionRows": wetlab_validation.get("procurementPlatformChecklistExtensionRows"),
            "wetlabProcurementPlatformChecklistStatusCounts": wetlab_validation.get("procurementPlatformChecklistStatusCounts"),
            "wetlabFinalPreExperimentGateRows": final_pre_experiment_gate.get("inputRows"),
            "wetlabFinalPreExperimentGateTierCounts": final_pre_experiment_gate.get("finalGateTierCounts"),
            "wetlabFinalPreExperimentGateGoCandidateRows": final_pre_experiment_gate.get("goCandidateRows"),
            "wetlabFinalPreExperimentGateBackupRows": final_pre_experiment_gate.get("backupRows"),
            "wetlabFinalPreExperimentGateExpertReviewRows": final_pre_experiment_gate.get("expertReviewRows"),
            "wetlabFinalPreExperimentGateHoldRows": final_pre_experiment_gate.get("holdRows"),
            "wetlabFinalPreExperimentGateManualGateCount": final_pre_experiment_gate.get("manualGateCount"),
            "wetlabWave1RoleCounts": wetlab_validation.get("wave1RoleCounts"),
            "wetlabWave1DirectionCounts": wetlab_validation.get("wave1DirectionCounts"),
            "experimentClosureOverallStatus": experiment_closure_audit.get("overallStatus"),
            "experimentClosurePassedRequiredChecks": experiment_closure_audit.get("passedRequiredChecks"),
            "experimentClosureTotalRequiredChecks": experiment_closure_audit.get("totalRequiredChecks"),
            "experimentClosureFailedRequiredChecks": experiment_closure_audit.get("failedRequiredChecks"),
            "validationPanelBenchmarkKnownRows": validation_panel_benchmark.get("knownDrugTargetRows"),
            "validationPanelBenchmarkAuroc": validation_panel_benchmark.get("validationAuroc"),
            "validationPanelBenchmarkAveragePrecision": validation_panel_benchmark.get("validationAveragePrecision"),
            "validationPanelBenchmarkTop100KnownRows": validation_panel_benchmark.get("top100KnownRows"),
            "validationPanelBenchmarkTop100Enrichment": validation_panel_benchmark.get("top100KnownEnrichment"),
            "validationPanelBenchmarkTop300KnownRows": validation_panel_benchmark.get("top300KnownRows"),
            "validationPanelBenchmarkTop300Enrichment": validation_panel_benchmark.get("top300KnownEnrichment"),
            "validationPanelBenchmarkBalancedKnownRows": validation_panel_benchmark.get("balancedKnownRows"),
            "validationPanelBenchmarkBalancedNovelRows": validation_panel_benchmark.get("balancedNovelRows"),
            "validationPanelBenchmarkBalancedInterpretableRows": validation_panel_benchmark.get("balancedInterpretableRows"),
            "validationPanelBenchmarkPositiveControlKnownRows": validation_panel_benchmark.get("positiveControlKnownRows"),
            "validationPanelBenchmarkNovelInterpretableRows": validation_panel_benchmark.get("novelShortlistInterpretableRows"),
            "validationPanelDiversityEligibleRows": validation_panel_diversity.get("abInterpretableEligibleRows"),
            "validationPanelDiversityTop100UniqueDrugs": validation_panel_diversity.get("top100UniqueDrugs"),
            "validationPanelDiversityTop100UniqueTargets": validation_panel_diversity.get("top100UniqueTargets"),
            "validationPanelDiversityTop100UniqueScaffolds": validation_panel_diversity.get("top100UniqueScaffolds"),
            "validationPanelDiversityTop100TopDrugPct": validation_panel_diversity.get("top100TopDrugPct"),
            "validationPanelDiversityTop100TopScaffoldPct": validation_panel_diversity.get("top100TopScaffoldPct"),
            "validationPanelDiversityBalancedUniqueDrugs": validation_panel_diversity.get("balancedUniqueDrugs"),
            "validationPanelDiversityBalancedUniqueTargets": validation_panel_diversity.get("balancedUniqueTargets"),
            "validationPanelDiversityBalancedUniqueScaffolds": validation_panel_diversity.get("balancedUniqueScaffolds"),
            "validationPanelDiversityBalancedTopDrugPct": validation_panel_diversity.get("balancedTopDrugPct"),
            "validationPanelDiversityBalancedTopScaffoldPct": validation_panel_diversity.get("balancedTopScaffoldPct"),
            "validationPanelDiversityWave1Rows": validation_panel_diversity.get("wave1Rows"),
            "validationPanelDiversityWave1DirectionCount": validation_panel_diversity.get("wave1DirectionCount"),
            "validationPanelDiversityWave1UniqueDrugs": validation_panel_diversity.get("wave1UniqueDrugs"),
            "validationPanelDiversityWave1UniqueTargets": validation_panel_diversity.get("wave1UniqueTargets"),
            "validationPanelDiversityWave1UniqueScaffolds": validation_panel_diversity.get("wave1UniqueScaffolds"),
            "validationPanelDiversityWave1TopDrugPct": validation_panel_diversity.get("wave1TopDrugPct"),
            "validationPanelDiversityWave1TopTargetPct": validation_panel_diversity.get("wave1TopTargetPct"),
            "validationPanelDiversityWave1TopScaffoldPct": validation_panel_diversity.get("wave1TopScaffoldPct"),
            "validationPanelDiversityWave1KnownRows": validation_panel_diversity.get("wave1KnownRows"),
            "validationPanelDiversityWave1NovelRows": validation_panel_diversity.get("wave1NovelRows"),
            "validationPanelDiversityWave1InterpretableRows": validation_panel_diversity.get("wave1InterpretableRows"),
            "validationPanelDiversityWave1GateCounts": validation_panel_diversity.get("wave1GateCounts"),
            "validationPanelDiversityWave1AssayCounts": validation_panel_diversity.get("wave1AssayCounts"),
            "fdaLabelRows": fda_label_mechanism.get("fdaRows"),
            "fdaLabelRowsWithTarget": fda_label_mechanism.get("fdaRowsWithTarget"),
            "fdaLabelUniqueDrugs": fda_label_mechanism.get("uniqueFdaDrugs"),
            "fdaLabelUniqueTargetChemblIds": fda_label_mechanism.get("uniqueFdaTargetChemblIds"),
            "fdaLabelExpandedDrugUniprotPairs": fda_label_mechanism.get("expandedFdaDrugUniprotPairs"),
            "fdaLabelExpandedRows": fda_label_mechanism.get("expandedFdaLabelRows"),
            "fdaLabelCandidateRows": fda_label_mechanism.get("candidateRows"),
            "fdaLabelCandidateDrugMappedPct": fda_label_mechanism.get("fdaDrugMappedPct"),
            "fdaLabelTargetMatchRows": fda_label_mechanism.get("fdaLabelTargetMatchRows"),
            "fdaLabelTargetMatchPct": fda_label_mechanism.get("fdaLabelTargetMatchPct"),
            "fdaLabelMatchedDrugs": fda_label_mechanism.get("fdaLabelMatchedDrugs"),
            "fdaLabelMatchedTargets": fda_label_mechanism.get("fdaLabelMatchedTargets"),
            "fdaLabelCandidateTargetAnyDrugRows": fda_label_mechanism.get("candidateTargetFdaLabeledByAnyDrugRows"),
            "fdaLabelCandidateTargetAnyDrugPct": fda_label_mechanism.get("candidateTargetFdaLabeledByAnyDrugPct"),
            "fdaLabelMechanismClassCounts": fda_label_mechanism.get("mechanismClassCounts"),
            "fdaLabelTop100Rows": fda_label_mechanism.get("top100LabelTargetRows"),
            "fdaLabelTop100PrecisionPct": fda_label_mechanism.get("top100LabelTargetPrecisionPct"),
            "fdaLabelTop100RecallPct": fda_label_mechanism.get("top100LabelTargetRecallPct"),
            "fdaLabelTop100Enrichment": fda_label_mechanism.get("top100LabelTargetEnrichment"),
            "fdaLabelTop300Rows": fda_label_mechanism.get("top300LabelTargetRows"),
            "fdaLabelTop300PrecisionPct": fda_label_mechanism.get("top300LabelTargetPrecisionPct"),
            "fdaLabelTop300RecallPct": fda_label_mechanism.get("top300LabelTargetRecallPct"),
            "fdaLabelTop300Enrichment": fda_label_mechanism.get("top300LabelTargetEnrichment"),
            "fdaLabelBalancedRows": fda_label_mechanism.get("balancedLabelTargetRows"),
            "fdaLabelBalancedClinicallyLabeledTargetNewDrugRows": fda_label_mechanism.get("balancedClinicallyLabeledTargetNewDrugRows"),
            "fdaLabelWave1Rows": fda_label_mechanism.get("wave1LabelTargetRows"),
            "fdaLabelWave1ClinicallyLabeledTargetNewDrugRows": fda_label_mechanism.get("wave1ClinicallyLabeledTargetNewDrugRows"),
            "fdaLabelPositiveControlRows": fda_label_mechanism.get("positiveControlLabelTargetRows"),
            "fdaTemporalCandidateRows": fda_label_temporal.get("candidateRows"),
            "fdaTemporalExactLabelRows": fda_label_temporal.get("exactLabelRows"),
            "fdaTemporalExact2016PlusRows": fda_label_temporal.get("exactLabel2016PlusRows"),
            "fdaTemporalExact2021PlusRows": fda_label_temporal.get("exactLabel2021PlusRows"),
            "fdaTemporalExact2016PlusPctOfExact": fda_label_temporal.get("exactLabel2016PlusPctOfExact"),
            "fdaTemporalExact2021PlusPctOfExact": fda_label_temporal.get("exactLabel2021PlusPctOfExact"),
            "fdaTemporalTargetContext2021PlusRows": fda_label_temporal.get("targetContext2021PlusRows"),
            "fdaTemporalTargetContext2021PlusPct": fda_label_temporal.get("targetContext2021PlusPct"),
            "fdaTemporalExactLabelEraCounts": fda_label_temporal.get("exactLabelEraCounts"),
            "fdaTemporalClassCounts": fda_label_temporal.get("temporalMechanismClassCounts"),
            "fdaTemporalTop100Exact2016PlusRows": fda_label_temporal.get("top100Exact2016PlusRows"),
            "fdaTemporalTop100Exact2016PlusRecallPct": fda_label_temporal.get("top100Exact2016PlusRecallPct"),
            "fdaTemporalTop100Exact2016PlusEnrichment": fda_label_temporal.get("top100Exact2016PlusEnrichment"),
            "fdaTemporalTop100Exact2021PlusRows": fda_label_temporal.get("top100Exact2021PlusRows"),
            "fdaTemporalTop100Exact2021PlusRecallPct": fda_label_temporal.get("top100Exact2021PlusRecallPct"),
            "fdaTemporalTop100Exact2021PlusEnrichment": fda_label_temporal.get("top100Exact2021PlusEnrichment"),
            "fdaTemporalTop300Exact2016PlusRows": fda_label_temporal.get("top300Exact2016PlusRows"),
            "fdaTemporalTop300Exact2016PlusRecallPct": fda_label_temporal.get("top300Exact2016PlusRecallPct"),
            "fdaTemporalTop300Exact2021PlusRows": fda_label_temporal.get("top300Exact2021PlusRows"),
            "fdaTemporalTop300Exact2021PlusRecallPct": fda_label_temporal.get("top300Exact2021PlusRecallPct"),
            "fdaTemporalSplit2015FutureTop300Rows": fda_label_temporal.get("split2015FutureLabelTop300Rows"),
            "fdaTemporalSplit2015FutureTop300RecallPct": fda_label_temporal.get("split2015FutureLabelTop300RecallPct"),
            "fdaTemporalSplit2015FutureTop300Enrichment": fda_label_temporal.get("split2015FutureLabelTop300Enrichment"),
            "fdaTemporalSplit2020FutureRows": fda_label_temporal.get("split2020FutureLabelRows"),
            "fdaTemporalSplit2020FutureTop100Rows": fda_label_temporal.get("split2020FutureLabelTop100Rows"),
            "fdaTemporalSplit2020FutureTop300Rows": fda_label_temporal.get("split2020FutureLabelTop300Rows"),
            "fdaTemporalBalancedExact2016PlusRows": fda_label_temporal.get("balancedExact2016PlusRows"),
            "fdaTemporalBalancedExact2021PlusRows": fda_label_temporal.get("balancedExact2021PlusRows"),
            "fdaTemporalBalancedTargetContext2021PlusRows": fda_label_temporal.get("balancedTargetContext2021PlusRows"),
            "fdaTemporalWave1Exact2016PlusRows": fda_label_temporal.get("wave1Exact2016PlusRows"),
            "fdaTemporalWave1Exact2021PlusRows": fda_label_temporal.get("wave1Exact2021PlusRows"),
            "fdaTemporalWave1TargetContext2021PlusRows": fda_label_temporal.get("wave1TargetContext2021PlusRows"),
            "fdaTemporalPositiveControlExact2016PlusRows": fda_label_temporal.get("positiveControlExact2016PlusRows"),
            "fdaTemporalPositiveControlExact2021PlusRows": fda_label_temporal.get("positiveControlExact2021PlusRows"),
            "admetTierA": metric(admet, "drugAdmetTierCounts", "A"),
            "admetTierB": metric(admet, "drugAdmetTierCounts", "B"),
            "knownPairRecallAt100000Pct": metric(enrichment.get(100000, {}), "recallPct"),
            "knownPairEnrichmentAt100000": metric(enrichment.get(100000, {}), "enrichmentVsRandom"),
            "knownRecordRecallAt100000Pct": metric(stratified, "overall", "recordRecallAt100000Pct"),
            "finalPriorityCandidateRows": final_priority.get("candidateRows"),
            "finalPriorityTierCounts": final_priority.get("tierCounts"),
            "finalPriorityReviewTrackCounts": final_priority.get("reviewTrackCounts"),
            "finalPriorityValidationKnownRows": final_priority_validation.get("knownDrugTargetRows"),
            "finalPriorityValidationAuroc": final_priority_validation.get("auroc"),
            "finalPriorityValidationAveragePrecision": final_priority_validation.get("averagePrecision"),
            "finalPriorityValidationRecallAt100Pct": final_priority_validation.get("recallAt100Pct"),
            "finalPriorityValidationPrecisionAt100Pct": final_priority_validation.get("precisionAt100Pct"),
            "finalPriorityValidationEnrichmentAt100": final_priority_validation.get("enrichmentAt100"),
            "ablationFinalAp": metric(final_priority_ablation, "fullVariant", "averagePrecision"),
            "ablationModelOnlyAp": metric(final_priority_ablation, "selectedVariants", "model_only", "averagePrecision"),
            "ablationKgOnlyAp": metric(final_priority_ablation, "selectedVariants", "kg_only", "averagePrecision"),
            "ablationWithoutKgAp": metric(final_priority_ablation, "selectedVariants", "without_kg_component", "averagePrecision"),
            "ablationWithoutRiskPenaltyAp": metric(final_priority_ablation, "selectedVariants", "without_risk_penalty", "averagePrecision"),
            "ablationModelOnlyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "model_only", "recallAt100Pct"),
            "ablationKgOnlyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "kg_only", "recallAt100Pct"),
            "ablationWithoutKgRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "without_kg_component", "recallAt100Pct"),
            "ablationWithoutRiskPenaltyRecallAt100Pct": metric(final_priority_ablation, "selectedVariants", "without_risk_penalty", "recallAt100Pct"),
            "noveltyDirectKnownMechanismRows": novelty.get("directKnownMechanismRows"),
            "noveltyDirectKnownMechanismPct": novelty.get("directKnownMechanismPct"),
            "noveltyStrictNovelRows": metric(novelty, "strictNovel", "rows"),
            "noveltyStrictNovelPct": metric(novelty, "strictNovel", "pct"),
            "noveltyStrictNovelAbRows": metric(novelty, "strictNovel", "abRows"),
            "noveltyTop100KnownMechanismPct": metric(novelty, "topK", "100", "directKnownMechanismPct"),
            "noveltyTop100StrictNovelPct": metric(novelty, "topK", "100", "strictNovelPairPct"),
            "noveltyTop100KnownDiseaseUseNewTargetPct": metric(novelty, "topK", "100", "knownDiseaseUseNewTargetPct"),
            "noveltyTop100SafetyPct": metric(novelty, "topK", "100", "safetyOrContraindicationPct"),
            "specificityUniquePairs": specificity.get("uniquePairs"),
            "specificityMultiDirectionPairIds": specificity.get("multiDirectionPairIds"),
            "specificityMultiDirectionPairIdPct": specificity.get("multiDirectionPairIdPct"),
            "specificityBroadGeneralistPairIds": specificity.get("broadGeneralistPairIds"),
            "specificityBroadGeneralistPairIdPct": specificity.get("broadGeneralistPairIdPct"),
            "specificityTop100MultiDirectionPct": metric(specificity, "topK", "100", "multiDirectionPairPct"),
            "specificityTop100BroadGeneralistPct": metric(specificity, "topK", "100", "broadGeneralistPct"),
            "specificityTop100DirectionSpecificPct": metric(specificity, "topK", "100", "directionSpecificPct"),
            "specificityTop100PairTopDirectionPct": metric(specificity, "topK", "100", "pairTopDirectionPct"),
            "specificityFocusedShortlistRows": specificity.get("focusedDirectionSpecificShortlistRows"),
            "significanceIterations": significance.get("iterations"),
            "significanceTop100ObservedHits": metric(significance, "topK", "100", "observedHits"),
            "significanceTop100GlobalExpectedHits": metric(significance, "topK", "100", "globalExpectedHits"),
            "significanceTop100StratifiedExpectedHits": metric(significance, "topK", "100", "stratifiedExpectedHits"),
            "significanceTop100GlobalEnrichment": metric(significance, "topK", "100", "globalEnrichmentVsRandom"),
            "significanceTop100StratifiedEnrichment": metric(significance, "topK", "100", "stratifiedEnrichmentVsRandom"),
            "significanceTop100GlobalP": metric(significance, "topK", "100", "globalHypergeomPGe"),
            "significanceTop100PermutationP": metric(significance, "topK", "100", "globalPermutationPGe"),
            "significanceTop100StratifiedP": metric(significance, "topK", "100", "stratifiedPermutationPGe"),
            "significanceTop500ObservedHits": metric(significance, "topK", "500", "observedHits"),
            "significanceTop500GlobalExpectedHits": metric(significance, "topK", "500", "globalExpectedHits"),
            "significanceTop500StratifiedExpectedHits": metric(significance, "topK", "500", "stratifiedExpectedHits"),
            "significanceTop500GlobalEnrichment": metric(significance, "topK", "500", "globalEnrichmentVsRandom"),
            "significanceTop500StratifiedEnrichment": metric(significance, "topK", "500", "stratifiedEnrichmentVsRandom"),
            "significanceTop500GlobalP": metric(significance, "topK", "500", "globalHypergeomPGe"),
            "significanceTop500PermutationP": metric(significance, "topK", "500", "globalPermutationPGe"),
            "significanceTop500StratifiedP": metric(significance, "topK", "500", "stratifiedPermutationPGe"),
            "concordanceMultiEvidenceRows": concordance.get("multiEvidenceRows"),
            "concordanceMultiEvidencePct": concordance.get("multiEvidencePct"),
            "concordanceHighRows": concordance.get("highConcordanceRows"),
            "concordanceHighPct": concordance.get("highConcordancePct"),
            "concordanceSingleEvidenceRows": concordance.get("singleEvidenceDominatedRows"),
            "concordanceSingleEvidencePct": concordance.get("singleEvidenceDominatedPct"),
            "concordanceMeanEvidenceSupportCount": concordance.get("meanEvidenceSupportCount"),
            "concordanceMeanStrongEvidenceCount": concordance.get("meanStrongEvidenceCount"),
            "concordanceTop100MultiEvidencePct": metric(concordance, "topK", "100", "multiEvidencePct"),
            "concordanceTop100HighPct": metric(concordance, "topK", "100", "highConcordancePct"),
            "concordanceTop100SingleEvidencePct": metric(concordance, "topK", "100", "singleEvidenceDominatedPct"),
            "concordanceTop500MultiEvidencePct": metric(concordance, "topK", "500", "multiEvidencePct"),
            "concordanceTop500HighPct": metric(concordance, "topK", "500", "highConcordancePct"),
            "concordanceTop500SingleEvidencePct": metric(concordance, "topK", "500", "singleEvidenceDominatedPct"),
            "diverseShortlistRows": diversity.get("diverseRows"),
            "overallTop20UniqueDrugsMean": diversity.get("overallTop20UniqueDrugsMean"),
            "overallTop20UniqueFamiliesMean": diversity.get("overallTop20UniqueFamiliesMean"),
            "artifactManifestCount": manifest.get("artifact_count"),
            "artifactManifestTotalSizeBytes": manifest.get("total_size_bytes"),
            "artifactManifestSourceScriptCount": len(manifest.get("source_scripts") or []),
            "artifactManifestLatestMtimeUtc": manifest.get("latest_artifact_mtime_utc"),
        },
    }


def markdown(payload: dict[str, Any]) -> str:
    h = payload["headline"]
    lines = [
        "# BioMaster SOTA Compute Closure Summary",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "This summary records computation-first progress. Web/dashboard updates are intentionally deferred.",
        "",
        "## Headline Metrics",
        "",
        f"- DiffDock structural completion: {pct(h.get('diffdockCompletionPct'))} "
        f"({h.get('diffdockCompletedRows')} completed, {h.get('diffdockMissingRows')} missing).",
        f"- Full DiffDock active expansion: status "
        f"{next((item.get('status') for item in payload.get('activeComputation', []) if item.get('module') == 'Full druggable-proteome DiffDock queue'), 'not_present')}; "
        f"jobs {h.get('fullDiffdockCompletedJobs')}/"
        f"{h.get('fullDiffdockTotalJobs')} ({pct(h.get('fullDiffdockCompletedJobPct'))}); scored rows "
        f"{h.get('fullDiffdockScoredRows')}/{h.get('fullDiffdockTotalRows')} "
        f"({pct(h.get('fullDiffdockScoredRowPct'))}); completed outputs "
        f"{h.get('fullDiffdockCompletedOutputs')}; missing outputs in scored chunks "
        f"{h.get('fullDiffdockMissingOutputsInScoredJobs')}; active jobs "
        f"{h.get('fullDiffdockActiveLocks')}; GPUs busy "
        f"{h.get('fullDiffdockBusyGpuCount')}/{h.get('fullDiffdockGpuCount')}; ETA "
        f"{fmt(h.get('fullDiffdockEtaHours'), 2)} hours "
        f"({fmt(h.get('fullDiffdockEtaDays'), 2)} days), finish UTC "
        f"{h.get('fullDiffdockEstimatedFinishUtc')}.",
        f"- Full DiffDock ligand technical rescue prepared: ligand {h.get('fullDiffdockLigandRescueLigandId')}; "
        f"queued rows {h.get('fullDiffdockLigandRescueQueuedRows')}; jobs {h.get('fullDiffdockLigandRescueJobs')}; "
        f"parent SDF `{h.get('fullDiffdockLigandRescueParentSdf')}`.",
        f"- Full DiffDock multi-ligand technical rescue prepared: candidate ligands "
        f"{h.get('fullDiffdockMultiLigandRescueCandidateLigands')}; queued ligands "
        f"{h.get('fullDiffdockMultiLigandRescueQueuedLigands')}; queued rows "
        f"{h.get('fullDiffdockMultiLigandRescueQueuedRows')}; jobs "
        f"{h.get('fullDiffdockMultiLigandRescueJobs')}; latest audit recommends "
        f"{h.get('fullDiffdockLatestFailureAuditRescueRecommendedLigands')} ligands / "
        f"{h.get('fullDiffdockLatestFailureAuditRescueRecommendedRows')} rows for rescue.",
        f"- Full DiffDock post-rescue finalization watcher: phase "
        f"{h.get('fullDiffdockFinalizerPhase')}; ready to finalize "
        f"{h.get('fullDiffdockFinalizerReadyToFinalize')}; active locks "
        f"{h.get('fullDiffdockFinalizerActiveLockCount')}; final merged CSV "
        f"`{h.get('fullDiffdockFinalMergedScores')}`.",
        f"- Full DiffDock post-finalization merge/audit watcher: phase "
        f"{h.get('fullDiffdockPostFinalizePhase')}; ready to run "
        f"{h.get('fullDiffdockPostFinalizeReadyToRun')}; merged score rows "
        f"{h.get('fullDiffdockPostFinalizeMergedScoreRows')}/"
        f"{h.get('fullDiffdockPostFinalizeMinMergedRows')}; already completed "
        f"{h.get('fullDiffdockPostFinalizeAlreadyCompleted')}.",
        f"- Top1000 KG explainability coverage: {pct(h.get('top1000KgPathPct'))}; path rows: {h.get('top1000KgPathRows')}.",
        f"- Pose readability after geometry audit: {pct(h.get('poseReadablePct'))}; pass rows: {h.get('posePassRows')}, warning rows: {h.get('poseWarningRows')}.",
        f"- AlphaFold pocket-confidence audit: {h.get('structureConfidenceAuditedRows')} completed poses audited; "
        f"median 5A pocket mean pLDDT {fmt(h.get('structureConfidenceMedianPocketMeanPlddt5A'), 2)}; "
        f"A/B pocket confidence {pct(h.get('structureConfidenceModerateOrHighPct'))}.",
        f"- Structure-adjusted final ranking: joined rows {h.get('finalStructureJoinedRows')}; "
        f"Top100 A/B pocket confidence shifted from {h.get('finalStructureTop100OriginalTierCounts')} "
        f"to {h.get('finalStructureTop100AdjustedTierCounts')}; high-priority low-confidence review rows {h.get('finalStructureTop500OriginalLowConfidenceRows')}.",
        f"- Structure-supported new-candidate shortlist: {h.get('finalStructureSupportedNovelRows')} rows selected from "
        f"{h.get('finalStructureSupportedNovelEligibleRows')} eligible new-pair/model-priority rows.",
        f"- Stage6 Top10000 oncology structural audit: {h.get('stage6Top10000CandidateRows')} rows, "
        f"{h.get('stage6Top10000UniqueDrugs')} drugs, {h.get('stage6Top10000UniqueProteins')} proteins; "
        f"DiffDock completed {h.get('stage6Top10000DiffdockCompletedRows')} "
        f"({pct(h.get('stage6Top10000DiffdockCompletedPct'))}), missing "
        f"{h.get('stage6Top10000DiffdockMissingRows')}; Top100 completed/missing "
        f"{h.get('stage6Top10000Top100CompletedRows')}/{h.get('stage6Top10000Top100MissingRows')}; "
        f"Top300 completed/missing {h.get('stage6Top10000Top300CompletedRows')}/"
        f"{h.get('stage6Top10000Top300MissingRows')}; readable poses "
        f"{pct(h.get('stage6Top10000PoseReadablePct'))}; A/B pocket confidence "
        f"{pct(h.get('stage6Top10000PocketModerateOrHighPct'))}; structure-adjusted Top100 tiers "
        f"{h.get('stage6Top10000Top100AdjustedTierCounts')}.",
        f"- Residue-contact pose interpretability: {h.get('poseInterpretabilityCandidateRows')} candidates audited; "
        f"{h.get('poseInterpretabilityInterpretableRows')} A/B interpretable poses "
        f"({pct(h.get('poseInterpretabilityInterpretablePct'))}); Top100 interpretable poses "
        f"{h.get('poseInterpretabilityTop100Rows')}; contact-residue rows {h.get('poseInterpretabilityContactResidueRows')}.",
        f"- Pose interpretation content: novel interpretable rows {h.get('poseInterpretabilityNovelRows')}, "
        f"known-target interpretable rows {h.get('poseInterpretabilityKnownRows')}; median 5A pocket residues "
        f"{h.get('poseInterpretabilityMedianPocketResidues5A')}, median 5A pocket mean pLDDT "
        f"{fmt(h.get('poseInterpretabilityMedianPocketMeanPlddt5A'), 2)}.",
        f"- RDKit pose-quality gate: {h.get('poseQualityCandidateRows')} final candidates audited; "
        f"readable ligand/receptor rows {h.get('poseQualityReadableRows')} "
        f"({pct(h.get('poseQualityReadablePct'))}); A/B pose-quality supported rows "
        f"{h.get('poseQualitySupportedRows')} ({pct(h.get('poseQualitySupportedPct'))}); tiers "
        f"{h.get('poseQualityTierCounts')}.",
        f"- Pose-quality Top100: A/B supported rows {h.get('poseQualityTop100SupportedRows')}/"
        f"{h.get('poseQualityTop100Rows')}, known rows {h.get('poseQualityTop100KnownRows')}, "
        f"novel rows {h.get('poseQualityTop100NovelRows')}; tier counts "
        f"{h.get('poseQualityTop100TierCounts')}.",
        f"- Standard PoseBusters/ProLIF Top100 check: structure-ready rows "
        f"{h.get('standardPoseValidationInputReadyRows')}/{h.get('standardPoseValidationCandidateRows')}; "
        f"PoseBusters pass {h.get('standardPoseValidationPoseBustersPassRows')} "
        f"({pct(h.get('standardPoseValidationPoseBustersPassPct'))}); ProLIF OK "
        f"{h.get('standardPoseValidationProlifOkRows')} ({pct(h.get('standardPoseValidationProlifOkPct'))}); "
        f"A/B standard-supported rows {h.get('standardPoseValidationSupportedRows')} "
        f"({pct(h.get('standardPoseValidationSupportedPct'))}); interaction fingerprint rows "
        f"{h.get('standardPoseValidationInteractionRows')}; tiers "
        f"{h.get('standardPoseValidationTierCounts')}.",
        f"- Standard PoseBusters/ProLIF Top300 extension: structure-ready rows "
        f"{h.get('standardPoseValidationTop300InputReadyRows')}/"
        f"{h.get('standardPoseValidationTop300CandidateRows')}; PoseBusters pass "
        f"{h.get('standardPoseValidationTop300PoseBustersPassRows')} "
        f"({pct(h.get('standardPoseValidationTop300PoseBustersPassPct'))}); ProLIF OK "
        f"{h.get('standardPoseValidationTop300ProlifOkRows')} "
        f"({pct(h.get('standardPoseValidationTop300ProlifOkPct'))}); A/B standard-supported rows "
        f"{h.get('standardPoseValidationTop300SupportedRows')} "
        f"({pct(h.get('standardPoseValidationTop300SupportedPct'))}); interaction fingerprint rows "
        f"{h.get('standardPoseValidationTop300InteractionRows')}; tiers "
        f"{h.get('standardPoseValidationTop300TierCounts')}.",
        f"- Standard PoseBusters/ProLIF full final-candidate extension: structure-ready rows "
        f"{h.get('standardPoseValidationFull3921InputReadyRows')}/"
        f"{h.get('standardPoseValidationFull3921CandidateRows')}; PoseBusters pass "
        f"{h.get('standardPoseValidationFull3921PoseBustersPassRows')} "
        f"({pct(h.get('standardPoseValidationFull3921PoseBustersPassPct'))}); ProLIF OK "
        f"{h.get('standardPoseValidationFull3921ProlifOkRows')} "
        f"({pct(h.get('standardPoseValidationFull3921ProlifOkPct'))}); A/B standard-supported rows "
        f"{h.get('standardPoseValidationFull3921SupportedRows')} "
        f"({pct(h.get('standardPoseValidationFull3921SupportedPct'))}); interaction fingerprint rows "
        f"{h.get('standardPoseValidationFull3921InteractionRows')}; tiers "
        f"{h.get('standardPoseValidationFull3921TierCounts')}.",
        f"- AutoDock Vina Top100 structural consensus rescoring: input-ready rows "
        f"{h.get('vinaConsensusInputReadyRows')}/{h.get('vinaConsensusCandidateRows')}; "
        f"PDBQT-ready rows {h.get('vinaConsensusPdbqtReadyRows')}; Vina-scored rows "
        f"{h.get('vinaConsensusScoredRows')} ({pct(h.get('vinaConsensusScoredPct'))}); "
        f"A/B consensus-supported rows {h.get('vinaConsensusSupportedRows')} "
        f"({pct(h.get('vinaConsensusSupportedPct'))}); median score-only "
        f"{fmt(h.get('vinaConsensusMedianScoreKcalMol'), 2)} kcal/mol; median optimized "
        f"{fmt(h.get('vinaConsensusMedianOptimizedScoreKcalMol'), 2)} kcal/mol; tiers "
        f"{h.get('vinaConsensusTierCounts')}.",
        f"- smina Top100 structural rescoring: input-ready rows "
        f"{h.get('sminaInputReadyRows')}/{h.get('sminaCandidateRows')}; scored rows "
        f"{h.get('sminaScoredRows')} ({pct(h.get('sminaScoredPct'))}); A/B supported rows "
        f"{h.get('sminaSupportedRows')} ({pct(h.get('sminaSupportedPct'))}); median affinity "
        f"{fmt(h.get('sminaMedianAffinityKcalMol'), 2)} kcal/mol; tiers "
        f"{h.get('sminaTierCounts')}.",
        f"- GNINA CNN Top100 execution audit: input-ready rows "
        f"{h.get('gninaInputReadyRows')}/{h.get('gninaCandidateRows')}; binary available "
        f"{h.get('gninaBinaryAvailable')}; runtime ready {h.get('gninaRuntimeReady')}; GNINA-scored rows "
        f"{h.get('gninaScoredRows')} ({pct(h.get('gninaScoredPct'))}); A/B supported rows "
        f"{h.get('gninaSupportedRows')} ({pct(h.get('gninaSupportedPct'))}); status counts "
        f"{h.get('gninaStatusCounts')}; tiers {h.get('gninaTierCounts')}.",
        f"- GNINA CNN Top300 extension: input-ready rows "
        f"{h.get('gninaTop300InputReadyRows')}/{h.get('gninaTop300CandidateRows')}; scored rows "
        f"{h.get('gninaTop300ScoredRows')} ({pct(h.get('gninaTop300ScoredPct'))}); A/B supported rows "
        f"{h.get('gninaTop300SupportedRows')} ({pct(h.get('gninaTop300SupportedPct'))}); median CNNscore "
        f"{fmt(h.get('gninaTop300MedianCnnScore'), 4)}; median CNNaffinity "
        f"{fmt(h.get('gninaTop300MedianCnnAffinity'), 4)}; tiers {h.get('gninaTop300TierCounts')}.",
        f"- GNINA CNN full final-candidate extension: prepared rows "
        f"{h.get('gninaFull3921PreparedRows')}; processed rows {h.get('gninaFull3921ProcessedRows')}; "
        f"completed scored rows {h.get('gninaFull3921ScoredRows')} "
        f"({pct(h.get('gninaFull3921ScoredPct'))}); A/B supported rows "
        f"{h.get('gninaFull3921SupportedRows')} ({pct(h.get('gninaFull3921SupportedPct'))}); tiers "
        f"{h.get('gninaFull3921TierCounts')}.",
        f"- Boltz-2 Top50 second-model complex validation: completed rows "
        f"{h.get('boltz2CompletedRows')}/{h.get('boltz2CandidateRows')} "
        f"({pct(h.get('boltz2CompletedPct'))}); original-ligand rows "
        f"{h.get('boltz2OriginalLigandRows')}, parent-fragment repaired rows "
        f"{h.get('boltz2RepairedParentLigandRows')}; A/B second-model supported rows "
        f"{h.get('boltz2AbSupportedRows')} ({pct(h.get('boltz2AbSupportedPct'))}); median confidence "
        f"{fmt(h.get('boltz2MedianConfidenceScore'), 4)}, median ligand iPTM "
        f"{fmt(h.get('boltz2MedianLigandIptm'), 4)}, median affinity probability "
        f"{fmt(h.get('boltz2MedianAffinityProbabilityBinary'), 4)}; tiers "
        f"{h.get('boltz2TierCounts')}.",
        f"- Boltz-2 high-sampling finalist validation: completed rows "
        f"{h.get('boltz2HighCompletedRows')}/{h.get('boltz2HighCandidateRows')} "
        f"({pct(h.get('boltz2HighCompletedPct'))}); known rows {h.get('boltz2HighKnownRows')}, "
        f"novel/extension rows {h.get('boltz2HighNovelOrExtensionRows')}; A/B high-sampling supported rows "
        f"{h.get('boltz2HighAbSupportedRows')} ({pct(h.get('boltz2HighAbSupportedPct'))}); "
        f"median confidence {fmt(h.get('boltz2HighMedianConfidenceScore'), 4)}, median ligand iPTM "
        f"{fmt(h.get('boltz2HighMedianLigandIptm'), 4)}, median affinity probability "
        f"{fmt(h.get('boltz2HighMedianAffinityProbabilityBinary'), 4)}; median composite delta vs fast run "
        f"{fmt(h.get('boltz2HighMedianCompositeDeltaVsFast'), 4)}; tiers {h.get('boltz2HighTierCounts')}.",
        f"- Target druggability audit: {h.get('targetDruggabilityCandidateRows')} rows, "
        f"{h.get('targetDruggabilityUniqueTargets')} unique targets, exact UniProt match "
        f"{pct(h.get('targetDruggabilityExactUniProtMatchPct'))}; Phase 3/4 rows "
        f"{pct(h.get('targetDruggabilityPhase3Or4Pct'))}; small-molecule modality "
        f"{pct(h.get('targetDruggabilitySmallMoleculePct'))}.",
        f"- Target-adjusted ranking: AP {fmt(h.get('targetDruggabilityOriginalAveragePrecision'), 4)} -> "
        f"{fmt(h.get('targetDruggabilityAdjustedAveragePrecision'), 4)}; Recall@100 "
        f"{pct(h.get('targetDruggabilityOriginalRecallAt100Pct'))} -> "
        f"{pct(h.get('targetDruggabilityAdjustedRecallAt100Pct'))}; high-translatability shortlist "
        f"{h.get('targetDruggabilityHighTranslatabilityShortlistRows')} rows.",
        f"- Chemotype diversity audit: {h.get('chemotypeUniqueDrugs')} unique drugs mapped to structures "
        f"({pct(h.get('chemotypeStructureMappedUniqueDrugPct'))}); {h.get('chemotypeUniqueMurckoScaffolds')} Murcko scaffolds "
        f"and {h.get('chemotypeUniqueClusters')} chemotype clusters.",
        f"- Top100 chemotype concentration: {h.get('chemotypeTop100UniqueDrugs')} unique drugs, "
        f"{h.get('chemotypeTop100UniqueScaffolds')} scaffolds, {h.get('chemotypeTop100UniqueClusters')} clusters; "
        f"top scaffold {pct(h.get('chemotypeTop100TopScaffoldPct'))}; top cluster {pct(h.get('chemotypeTop100TopClusterPct'))}.",
        f"- Chemotype-diverse shortlist: {h.get('chemotypeDiverseShortlistRows')} rows with "
        f"{h.get('chemotypeDiverseShortlistUniqueDrugs')} drugs and {h.get('chemotypeDiverseShortlistUniqueScaffolds')} scaffolds.",
        f"- SOTA-ready decision matrix: {h.get('sotaReadyCandidateRows')} rows; tiers {h.get('sotaReadyTierCounts')}; "
        f"actions {h.get('sotaReadyActionCounts')}.",
        f"- SOTA-ready Top100: known rows {h.get('sotaReadyTop100KnownRows')} "
        f"({pct(h.get('sotaReadyTop100KnownPct'))}), novel rows {h.get('sotaReadyTop100NovelRows')}, "
        f"structure A/B rows {h.get('sotaReadyTop100StructureABRows')}, target A/B rows {h.get('sotaReadyTop100TargetABRows')}.",
        f"- SOTA-ready review artifacts: expert shortlist {h.get('sotaReadyExpertShortlistRows')} rows, "
        f"novel shortlist {h.get('sotaReadyNovelShortlistRows')} rows, diverse shortlist "
        f"{h.get('sotaReadyDiverseShortlistRows')} rows with {h.get('sotaReadyDiverseShortlistUniqueDrugs')} drugs.",
        f"- Network medicine / PPI proximity audit: {h.get('networkProximityCandidateRows')} candidate rows and "
        f"{h.get('networkProximityFinalPriorityRows')} final-priority rows audited on a STRING/HuRI PPI graph with "
        f"{h.get('networkProximityStringGraphNodes')} nodes and {h.get('networkProximityStringGraphEdges')} edges; "
        f"candidate-row coverage {pct(h.get('networkProximityCandidateRowsStringCoveredPct'))}; "
        f"unique target-protein coverage {h.get('networkProximityUniqueTargetsStringCovered')}/"
        f"{h.get('networkProximityUniqueTargetProteinPairs')} "
        f"({pct(h.get('networkProximityUniqueTargetsStringCoveredPct'))}); "
        f"final-priority tiers {h.get('networkProximityFinalPriorityTierCounts')}.",
        f"- SOTA-network reprioritization: {h.get('sotaNetworkCandidateRows')} rows; network-positive rows "
        f"{h.get('sotaNetworkPositiveRows')} ({pct(h.get('sotaNetworkPositivePct'))}), direct disease-module rows "
        f"{h.get('sotaNetworkDirectRows')}, coverage-gap rows {h.get('sotaNetworkCoverageGapRows')}; AP "
        f"{fmt(h.get('sotaNetworkOldAveragePrecision'), 4)} -> {fmt(h.get('sotaNetworkAveragePrecision'), 4)}.",
        f"- SOTA-network Top100: network-positive rows {h.get('sotaNetworkTop100NetworkPositiveRows')}, "
        f"direct disease-module rows {h.get('sotaNetworkTop100NetworkDirectRows')}, known rows "
        f"{h.get('sotaNetworkTop100KnownRows')}, novel rows {h.get('sotaNetworkTop100NovelRows')}; "
        f"network tiers {h.get('sotaNetworkTop100NetworkTierCounts')}.",
        f"- HPA tissue-context audit: {h.get('tissueContextCandidateRows')} candidates, "
        f"{h.get('tissueContextUniqueTargets')} unique targets, and {h.get('tissueContextTargetDirectionRows')} "
        f"target-direction rows audited; HPA gene-symbol match {pct(h.get('tissueContextCandidateHpaMatchedPct'))}; "
        f"A/B relevant-tissue support {h.get('tissueContextCandidatePositiveRows')} rows "
        f"({pct(h.get('tissueContextCandidatePositivePct'))}).",
        f"- SOTA-context Top100: tissue-positive rows {h.get('sotaContextTop100TissuePositiveRows')}, "
        f"HPA-matched rows {h.get('sotaContextTop100HpaMatchedRows')}, network-positive rows "
        f"{h.get('sotaContextTop100NetworkPositiveRows')}, known rows {h.get('sotaContextTop100KnownRows')}, "
        f"novel rows {h.get('sotaContextTop100NovelRows')}; tissue tiers "
        f"{h.get('sotaContextTop100TissueTierCounts')}.",
        f"- GTEx tissue-context audit: {h.get('gtexContextCandidateRows')} candidates, "
        f"{h.get('gtexContextUniqueTargets')} unique targets, and {h.get('gtexContextTargetDirectionRows')} "
        f"target-direction rows audited using {h.get('gtexContextRelease')}; GTEx gene-symbol match "
        f"{h.get('gtexContextCandidateMatchedRows')} rows ({pct(h.get('gtexContextCandidateMatchedPct'))}); "
        f"A/B relevant-tissue support {h.get('gtexContextCandidatePositiveRows')} rows "
        f"({pct(h.get('gtexContextCandidatePositivePct'))}).",
        f"- SOTA-GTEx-context Top100: GTEx-positive rows "
        f"{h.get('sotaGtexContextTop100GtexPositiveRows')}, GTEx-matched rows "
        f"{h.get('sotaGtexContextTop100GtexMatchedRows')}, HPA-positive rows "
        f"{h.get('sotaGtexContextTop100HpaPositiveRows')}, network-positive rows "
        f"{h.get('sotaGtexContextTop100NetworkPositiveRows')}, known rows "
        f"{h.get('sotaGtexContextTop100KnownRows')}, novel rows "
        f"{h.get('sotaGtexContextTop100NovelRows')}; GTEx tiers "
        f"{h.get('sotaGtexContextTop100GtexTierCounts')}.",
        f"- DepMap oncology dependency scoring: oncology rows {h.get('depmapOncologyRows')}, "
        f"unique oncology targets {h.get('depmapUniqueOncologyTargets')}, release "
        f"{h.get('depmapRelease')} ({h.get('depmapReleaseDate')}); target match "
        f"{pct(h.get('depmapMatchedTargetsPct'))}; dependency-positive candidates "
        f"{h.get('depmapPositiveCandidateRows')} ({pct(h.get('depmapPositiveCandidatePct'))}); "
        f"dependency-positive targets {h.get('depmapPositiveTargets')} "
        f"({pct(h.get('depmapPositiveTargetsPct'))}); Top100 oncology dependency-positive rows "
        f"{metric(h, 'depmapTop100Oncology', 'positiveRows')}.",
        f"- LINCS/CMap disease-signature reversal scoring: {h.get('lincsReversalCandidateRows')} candidates, "
        f"{h.get('lincsReversalMappedCandidateDrugs')}/{h.get('lincsReversalCandidateDrugs')} drugs mapped to CMap, "
        f"{h.get('lincsReversalMappedCandidateRows')} candidate rows mapped; QC trt_cp signatures "
        f"{h.get('lincsReversalQcTrtCpSignatureRows')}; selected signatures "
        f"{h.get('lincsReversalSelectedSignatureCount')}; positive reversal rows "
        f"{h.get('lincsReversalPositiveRows')}; Top100 positive reversal rows "
        f"{h.get('lincsReversalTop100PositiveRows')}; tiers {h.get('lincsReversalTierCounts')}.",
        f"- Integrated expert review panel: Top{h.get('expertReviewPanelRows')} panel and Wave1 "
        f"{h.get('expertReviewWave1Rows')} selected from {h.get('expertReviewCandidateRows')} candidates; "
        f"{h.get('expertReviewPanelUniqueDrugs')} drugs, {h.get('expertReviewPanelUniqueTargets')} targets, "
        f"{h.get('expertReviewPanelUniqueScaffolds')} scaffolds; directions "
        f"{h.get('expertReviewPanelDirectionCounts')}; novelty {h.get('expertReviewNoveltyGroupCounts')}; "
        f"CMap tiers {h.get('expertReviewCmapTierCounts')}.",
        f"- External SOTA model input package: source rows {h.get('externalInputCandidateRows')}, "
        f"unique pairs/drugs/proteins {h.get('externalInputUniquePairs')}/"
        f"{h.get('externalInputUniqueDrugs')}/{h.get('externalInputUniqueProteins')}; GNINA Top100 ready "
        f"{h.get('externalInputGninaReadyRows')}/{h.get('externalInputGninaRows')}, Boltz/Chai Top50 input-ready "
        f"{h.get('externalInputBoltzReadyRows')}/{h.get('externalInputBoltzRows')}, independent DTI Top1000 ready "
        f"{h.get('externalInputDtiReadyRows')}/{h.get('externalInputDtiRows')}; shared ligand/protein assets "
        f"{h.get('externalInputLigandAssetCount')}/{h.get('externalInputProteinAssetCount')}.",
        f"- Local supervised independent DTI corroboration: scored rows "
        f"{h.get('independentDtiScoredRows')}/{h.get('independentDtiCandidateRows')} across "
        f"{h.get('independentDtiUniqueDrugs')} drugs and {h.get('independentDtiUniqueProteins')} proteins; "
        f"A/B supported rows {h.get('independentDtiAbSupportedRows')} "
        f"({pct(h.get('independentDtiAbSupportedPct'))}); tier counts "
        f"{h.get('independentDtiTierCounts')}.",
        f"- Independent DTI known-pair benchmark: AUROC "
        f"{fmt(h.get('independentDtiKnownBenchmarkAuroc'), 4)}, AP "
        f"{fmt(h.get('independentDtiKnownBenchmarkAveragePrecision'), 4)}; Top50 known hits "
        f"{h.get('independentDtiTop50KnownHits')} with recall "
        f"{pct(h.get('independentDtiTop50RecallPct'))} and enrichment "
        f"{fmt(h.get('independentDtiTop50Enrichment'), 2)}x; Top100 known hits "
        f"{h.get('independentDtiTop100KnownHits')} with recall "
        f"{pct(h.get('independentDtiTop100RecallPct'))}.",
        f"- Independent DTI split validation: pair-holdout AUROC "
        f"{fmt(h.get('independentDtiPairHoldoutAuroc'), 4)}, drug-holdout AUROC "
        f"{fmt(h.get('independentDtiDrugHoldoutAuroc'), 4)}, target-holdout AUROC "
        f"{fmt(h.get('independentDtiTargetHoldoutAuroc'), 4)}.",
        f"- Local supervised independent DTI full final-candidate extension: scored rows "
        f"{h.get('independentDtiFull3921ScoredRows')}/{h.get('independentDtiFull3921CandidateRows')} across "
        f"{h.get('independentDtiFull3921UniqueDrugs')} drugs and "
        f"{h.get('independentDtiFull3921UniqueProteins')} proteins; A/B supported rows "
        f"{h.get('independentDtiFull3921AbSupportedRows')} "
        f"({pct(h.get('independentDtiFull3921AbSupportedPct'))}); tier counts "
        f"{h.get('independentDtiFull3921TierCounts')}.",
        f"- Independent DTI full3921 known-pair benchmark: AUROC "
        f"{fmt(h.get('independentDtiFull3921KnownBenchmarkAuroc'), 4)}, AP "
        f"{fmt(h.get('independentDtiFull3921KnownBenchmarkAveragePrecision'), 4)}; Top50 known hits "
        f"{h.get('independentDtiFull3921Top50KnownHits')} with recall "
        f"{pct(h.get('independentDtiFull3921Top50RecallPct'))} and enrichment "
        f"{fmt(h.get('independentDtiFull3921Top50Enrichment'), 2)}x; Top100 known hits "
        f"{h.get('independentDtiFull3921Top100KnownHits')} with recall "
        f"{pct(h.get('independentDtiFull3921Top100RecallPct'))}.",
        f"- Independent DTI full3921 split validation: pair-holdout AUROC "
        f"{fmt(h.get('independentDtiFull3921PairHoldoutAuroc'), 4)}, drug-holdout AUROC "
        f"{fmt(h.get('independentDtiFull3921DrugHoldoutAuroc'), 4)}, target-holdout AUROC "
        f"{fmt(h.get('independentDtiFull3921TargetHoldoutAuroc'), 4)}.",
        f"- ML ADMET endpoint audit: {h.get('mlAdmetTrainedEndpointCount')} TDC-trained endpoints scored "
        f"{h.get('mlAdmetDrugRows')} drugs and {h.get('mlAdmetCandidateRows')} candidate rows; "
        f"low/manageable-risk candidate rows {h.get('mlAdmetLowRiskCandidateRows')} "
        f"({pct(h.get('mlAdmetLowRiskCandidatePct'))}), high-risk review rows "
        f"{h.get('mlAdmetHighRiskCandidateRows')} ({pct(h.get('mlAdmetHighRiskCandidatePct'))}).",
        f"- SOTA-ML-ADMET Top100: low/manageable-risk rows {h.get('mlAdmetTop100LowRiskRows')}, "
        f"high-risk rows {h.get('mlAdmetTop100HighRiskRows')}, known rows {h.get('mlAdmetTop100KnownRows')}, "
        f"novel rows {h.get('mlAdmetTop100NovelRows')}; safety tiers "
        f"{h.get('mlAdmetTop100SafetyTierCounts')}.",
        f"- SOTA model feasibility audit: readiness counts {h.get('modelFeasibilityReadinessCounts')}; next queue "
        f"{h.get('modelFeasibilityPriorityQueue')}.",
        f"- Rank stability audit: consensus Top100 rows in at least 3 methods {h.get('rankStabilityConsensusTop100Rows')}; "
        f"all-method Top100 intersection {h.get('rankStabilityAllMethodTop100IntersectionRows')} "
        f"({pct(h.get('rankStabilityAllMethodTop100IntersectionPct'))}).",
        f"- Final vs SOTA-ready ranking stability: Top100 overlap {h.get('rankStabilityFinalVsSotaTop100OverlapRows')}; "
        f"Jaccard {fmt(h.get('rankStabilityFinalVsSotaTop100Jaccard'), 4)}; "
        f"Spearman {fmt(h.get('rankStabilityFinalVsSotaSpearman'), 4)}.",
        f"- Consensus-priority interpretation: Top100 known rows {h.get('rankStabilityConsensusTop100KnownRows')}, "
        f"novel rows {h.get('rankStabilityConsensusTop100NovelRows')}; large final-vs-SOTA rank-delta review rows "
        f"{h.get('rankStabilityLargeDeltaRows')}.",
        f"- Experimental validation panel: {h.get('experimentalValidationCandidateRows')} candidates triaged; "
        f"experiment-ready rows {h.get('experimentalValidationExperimentReadyRows')}, expert-review-ready rows "
        f"{h.get('experimentalValidationReviewReadyRows')}, novel experiment/review-ready rows "
        f"{h.get('experimentalValidationNovelReadyRows')}.",
        f"- Disease-balanced validation shortlist: {h.get('experimentalValidationBalancedPanelRows')} rows across "
        f"{h.get('experimentalValidationBalancedDirections')} directions, "
        f"{h.get('experimentalValidationBalancedUniqueDrugs')} drugs, "
        f"{h.get('experimentalValidationBalancedUniqueTargets')} targets, and "
        f"{h.get('experimentalValidationBalancedUniqueScaffolds')} scaffolds.",
        f"- Validation assay planning: novel validation panel {h.get('experimentalValidationNovelPanelRows')} rows; "
        f"positive-control ready rows {h.get('experimentalValidationPositiveControlReadyRows')}; "
        f"Top100 experiment/review-ready rows {h.get('experimentalValidationTop100ReadyRows')} with "
        f"{h.get('experimentalValidationTop100NovelRows')} novel rows.",
        f"- Validation panel calibration: known positives {h.get('validationPanelBenchmarkKnownRows')}; "
        f"validation-score AUROC/AP {fmt(h.get('validationPanelBenchmarkAuroc'), 4)}/"
        f"{fmt(h.get('validationPanelBenchmarkAveragePrecision'), 4)}; Top100 known rows "
        f"{h.get('validationPanelBenchmarkTop100KnownRows')}, enrichment "
        f"{fmt(h.get('validationPanelBenchmarkTop100Enrichment'), 2)}x; Top300 known rows "
        f"{h.get('validationPanelBenchmarkTop300KnownRows')}, enrichment "
        f"{fmt(h.get('validationPanelBenchmarkTop300Enrichment'), 2)}x.",
        f"- Queue calibration: balanced shortlist known {h.get('validationPanelBenchmarkBalancedKnownRows')}, "
        f"novel {h.get('validationPanelBenchmarkBalancedNovelRows')}, interpretable "
        f"{h.get('validationPanelBenchmarkBalancedInterpretableRows')}; positive-control queue known "
        f"{h.get('validationPanelBenchmarkPositiveControlKnownRows')}/120; novel shortlist interpretable "
        f"{h.get('validationPanelBenchmarkNovelInterpretableRows')}/300.",
        f"- Validation-panel diversity: A/B interpretable eligible rows {h.get('validationPanelDiversityEligibleRows')}; "
        f"raw Top100 has {h.get('validationPanelDiversityTop100UniqueDrugs')} drugs, "
        f"{h.get('validationPanelDiversityTop100UniqueTargets')} targets, and "
        f"{h.get('validationPanelDiversityTop100UniqueScaffolds')} scaffolds; top drug/scaffold concentration "
        f"{pct(h.get('validationPanelDiversityTop100TopDrugPct'))}/{pct(h.get('validationPanelDiversityTop100TopScaffoldPct'))}.",
        f"- Diversity-controlled validation design: balanced shortlist has "
        f"{h.get('validationPanelDiversityBalancedUniqueDrugs')} drugs, "
        f"{h.get('validationPanelDiversityBalancedUniqueTargets')} targets, and "
        f"{h.get('validationPanelDiversityBalancedUniqueScaffolds')} scaffolds; top drug/scaffold concentration "
        f"{pct(h.get('validationPanelDiversityBalancedTopDrugPct'))}/{pct(h.get('validationPanelDiversityBalancedTopScaffoldPct'))}.",
        f"- Wave-1 diverse validation panel: {h.get('validationPanelDiversityWave1Rows')} rows across "
        f"{h.get('validationPanelDiversityWave1DirectionCount')} directions; "
        f"{h.get('validationPanelDiversityWave1UniqueDrugs')} drugs, "
        f"{h.get('validationPanelDiversityWave1UniqueTargets')} targets, "
        f"{h.get('validationPanelDiversityWave1UniqueScaffolds')} scaffolds; known/novel/interpretable "
        f"{h.get('validationPanelDiversityWave1KnownRows')}/{h.get('validationPanelDiversityWave1NovelRows')}/"
        f"{h.get('validationPanelDiversityWave1InterpretableRows')}; gates "
        f"{h.get('validationPanelDiversityWave1GateCounts')}.",
        f"- FDA label mechanism audit: {h.get('fdaLabelRows')} FDA small-molecule rows, "
        f"{h.get('fdaLabelRowsWithTarget')} with target annotation; expanded to "
        f"{h.get('fdaLabelExpandedDrugUniprotPairs')} FDA drug-UniProt pairs and "
        f"{h.get('fdaLabelExpandedRows')} label-component rows.",
        f"- FDA label support in candidates: {h.get('fdaLabelCandidateRows')} candidate rows audited; FDA-drug mapping "
        f"{pct(h.get('fdaLabelCandidateDrugMappedPct'))}; label-target exact matches "
        f"{h.get('fdaLabelTargetMatchRows')} ({pct(h.get('fdaLabelTargetMatchPct'))}), covering "
        f"{h.get('fdaLabelMatchedDrugs')} drugs and {h.get('fdaLabelMatchedTargets')} targets.",
        f"- FDA mechanism classes: {h.get('fdaLabelMechanismClassCounts')}; candidate targets already FDA-labeled "
        f"for any approved drug: {h.get('fdaLabelCandidateTargetAnyDrugRows')} "
        f"({pct(h.get('fdaLabelCandidateTargetAnyDrugPct'))}).",
        f"- FDA label TopK and queues: Top100 label-target rows {h.get('fdaLabelTop100Rows')} "
        f"(precision {pct(h.get('fdaLabelTop100PrecisionPct'))}, recall {pct(h.get('fdaLabelTop100RecallPct'))}, "
        f"enrichment {fmt(h.get('fdaLabelTop100Enrichment'), 2)}x); Top300 rows "
        f"{h.get('fdaLabelTop300Rows')}; balanced/wave-1/positive-control label-target rows "
        f"{h.get('fdaLabelBalancedRows')}/{h.get('fdaLabelWave1Rows')}/{h.get('fdaLabelPositiveControlRows')}.",
        f"- FDA label temporal audit: exact label-target rows {h.get('fdaTemporalExactLabelRows')}; "
        f"2016+ rows {h.get('fdaTemporalExact2016PlusRows')} "
        f"({pct(h.get('fdaTemporalExact2016PlusPctOfExact'))} of exact labels), 2021+ rows "
        f"{h.get('fdaTemporalExact2021PlusRows')} "
        f"({pct(h.get('fdaTemporalExact2021PlusPctOfExact'))} of exact labels); 2021+ target-context rows "
        f"{h.get('fdaTemporalTargetContext2021PlusRows')} "
        f"({pct(h.get('fdaTemporalTargetContext2021PlusPct'))}).",
        f"- FDA temporal TopK stress test: Top100 2016+/2021+ exact label hits "
        f"{h.get('fdaTemporalTop100Exact2016PlusRows')}/{h.get('fdaTemporalTop100Exact2021PlusRows')}; "
        f"Top300 2016+/2021+ exact label hits "
        f"{h.get('fdaTemporalTop300Exact2016PlusRows')}/{h.get('fdaTemporalTop300Exact2021PlusRows')}; "
        f"split-2015 future-label Top300 hits {h.get('fdaTemporalSplit2015FutureTop300Rows')} "
        f"(recall {pct(h.get('fdaTemporalSplit2015FutureTop300RecallPct'))}, enrichment "
        f"{fmt(h.get('fdaTemporalSplit2015FutureTop300Enrichment'), 2)}x).",
        f"- FDA temporal queue coverage: balanced 2016+/2021+ exact labels "
        f"{h.get('fdaTemporalBalancedExact2016PlusRows')}/{h.get('fdaTemporalBalancedExact2021PlusRows')}; "
        f"wave-1 2016+/2021+ exact labels "
        f"{h.get('fdaTemporalWave1Exact2016PlusRows')}/{h.get('fdaTemporalWave1Exact2021PlusRows')}; "
        f"2021+ target-context rows balanced/wave-1 "
        f"{h.get('fdaTemporalBalancedTargetContext2021PlusRows')}/"
        f"{h.get('fdaTemporalWave1TargetContext2021PlusRows')}.",
        f"- ADMET A/B drug tiers: A={h.get('admetTierA')}, B={h.get('admetTierB')}.",
        f"- Known-target pair Recall@100000: {pct(h.get('knownPairRecallAt100000Pct'))}; enrichment {fmt(h.get('knownPairEnrichmentAt100000'))}x.",
        f"- Known-target record Recall@100000: {pct(h.get('knownRecordRecallAt100000Pct'))}.",
        f"- Final priority rows: {h.get('finalPriorityCandidateRows')}; tier counts: {h.get('finalPriorityTierCounts')}.",
        f"- Final-priority known-target validation: {h.get('finalPriorityValidationKnownRows')} known pairs; "
        f"Recall@100 {pct(h.get('finalPriorityValidationRecallAt100Pct'))}; "
        f"Precision@100 {pct(h.get('finalPriorityValidationPrecisionAt100Pct'))}; "
        f"enrichment {fmt(h.get('finalPriorityValidationEnrichmentAt100'))}x.",
        f"- Final-priority ablation: AP final/model-only/KG-only/without-KG = "
        f"{fmt(h.get('ablationFinalAp'), 4)}/{fmt(h.get('ablationModelOnlyAp'), 4)}/"
        f"{fmt(h.get('ablationKgOnlyAp'), 4)}/{fmt(h.get('ablationWithoutKgAp'), 4)}; "
        f"KG-only Recall@100 {pct(h.get('ablationKgOnlyRecallAt100Pct'))}.",
        f"- Risk-penalty sensitivity: without risk penalty AP {fmt(h.get('ablationWithoutRiskPenaltyAp'), 4)}, "
        f"Recall@100 {pct(h.get('ablationWithoutRiskPenaltyRecallAt100Pct'))}.",
        f"- Novelty audit: direct known-mechanism rows {h.get('noveltyDirectKnownMechanismRows')} "
        f"({pct(h.get('noveltyDirectKnownMechanismPct'))}); strict novel rows {h.get('noveltyStrictNovelRows')} "
        f"({pct(h.get('noveltyStrictNovelPct'))}), strict novel A/B rows {h.get('noveltyStrictNovelAbRows')}.",
        f"- Top100 novelty composition: known mechanism {pct(h.get('noveltyTop100KnownMechanismPct'))}, "
        f"known disease-use/new-target {pct(h.get('noveltyTop100KnownDiseaseUseNewTargetPct'))}, "
        f"strict novel {pct(h.get('noveltyTop100StrictNovelPct'))}, safety context {pct(h.get('noveltyTop100SafetyPct'))}.",
        f"- Direction specificity: unique pairs {h.get('specificityUniquePairs')}; multi-direction pairs "
        f"{h.get('specificityMultiDirectionPairIds')} ({pct(h.get('specificityMultiDirectionPairIdPct'))}); "
        f"broad-generalist pairs {h.get('specificityBroadGeneralistPairIds')} ({pct(h.get('specificityBroadGeneralistPairIdPct'))}).",
        f"- Top100 direction specificity: multi-direction rows {pct(h.get('specificityTop100MultiDirectionPct'))}, "
        f"broad-generalist rows {pct(h.get('specificityTop100BroadGeneralistPct'))}, "
        f"pair-top-direction rows {pct(h.get('specificityTop100PairTopDirectionPct'))}, "
        f"direction-specific rows {pct(h.get('specificityTop100DirectionSpecificPct'))}.",
        f"- TopK significance: Top100 observed known hits {h.get('significanceTop100ObservedHits')} vs global expected "
        f"{fmt(h.get('significanceTop100GlobalExpectedHits'), 2)} and stratified expected "
        f"{fmt(h.get('significanceTop100StratifiedExpectedHits'), 2)}; global p={h.get('significanceTop100GlobalP')}, "
        f"stratified permutation p={h.get('significanceTop100StratifiedP')}.",
        f"- Top500 significance: observed known hits {h.get('significanceTop500ObservedHits')} vs global expected "
        f"{fmt(h.get('significanceTop500GlobalExpectedHits'), 2)} and stratified expected "
        f"{fmt(h.get('significanceTop500StratifiedExpectedHits'), 2)}; global p={h.get('significanceTop500GlobalP')}, "
        f"stratified permutation p={h.get('significanceTop500StratifiedP')}.",
        f"- Evidence concordance: multi-evidence rows {h.get('concordanceMultiEvidenceRows')} "
        f"({pct(h.get('concordanceMultiEvidencePct'))}); high-concordance rows {h.get('concordanceHighRows')} "
        f"({pct(h.get('concordanceHighPct'))}); single/sparse-evidence rows {h.get('concordanceSingleEvidenceRows')} "
        f"({pct(h.get('concordanceSingleEvidencePct'))}).",
        f"- Top100 evidence maturity: multi-evidence {pct(h.get('concordanceTop100MultiEvidencePct'))}, "
        f"high-concordance {pct(h.get('concordanceTop100HighPct'))}, "
        f"single/sparse-evidence {pct(h.get('concordanceTop100SingleEvidencePct'))}.",
        f"- Diversity-capped expert shortlist rows: {h.get('diverseShortlistRows')}; mean unique drugs in direction Top20: {fmt(h.get('overallTop20UniqueDrugsMean'), 1)}.",
        f"- Reproducibility manifest: {h.get('artifactManifestCount')} core computation artifacts indexed; "
        f"{h.get('artifactManifestSourceScriptCount')} source scripts attributed; latest artifact mtime {h.get('artifactManifestLatestMtimeUtc')}.",
        "",
        "## Active Computation",
        "",
    ]
    for item in payload.get("activeComputation", []):
        status = item.get("status", "")
        if item.get("module") == "Full DiffDock post-rescue finalization watcher":
            queues = item.get("queueStatuses") or {}
            main_queue = queues.get("main") or {}
            single_queue = queues.get("singleRescue") or {}
            multi_queue = queues.get("multiRescue") or {}
            lines.append(
                f"- {item['module']}: {status}. Phase {item.get('phase')}; ready to finalize "
                f"{item.get('readyToFinalize')}; queue jobs main "
                f"{main_queue.get('completedJobs')}/{main_queue.get('totalJobs')}, single rescue "
                f"{single_queue.get('completedJobs')}/{single_queue.get('totalJobs')}, multi rescue "
                f"{multi_queue.get('completedJobs')}/{multi_queue.get('totalJobs')}; active locks "
                f"{item.get('activeLockCount')}; status `{item.get('statusJson')}`. "
                f"{item.get('evidence', '')}"
            )
        elif "queuedRows" in item:
            completed_jobs = item.get("completedJobs")
            total_jobs = item.get("jobs") or item.get("totalJobs")
            completed_clause = (
                f"; completed jobs {completed_jobs}/{total_jobs}"
                if completed_jobs is not None
                else f"; jobs {total_jobs}"
            )
            lines.append(
                f"- {item['module']}: {status}. Queued rows {item.get('queuedRows')}"
                f"{completed_clause}; chunk size {item.get('chunkSize')}; job index "
                f"`{item.get('jobIndex')}`. {item.get('evidence', '')}"
            )
        elif "completedJobs" in item:
            lines.append(
                f"- {item['module']}: {status}. Jobs {item.get('completedJobs')}/{item.get('totalJobs')} "
                f"({pct(item.get('completedJobPct'))}); scored rows {item.get('scoredRows')}/"
                f"{item.get('totalRows')} ({pct(item.get('scoredRowPct'))}); GPUs busy "
                f"{item.get('busyGpuCount')}/{item.get('gpuCount')}; ETA "
                f"{fmt(item.get('etaHours'), 2)} hours. {item.get('evidence', '')}"
            )
        else:
            processed = item.get("processedRows")
            candidate_rows = item.get("candidateRows")
            progress = pct(100.0 * processed / candidate_rows) if processed and candidate_rows else "NA"
            latest_status = item.get("latestVinaStatus") or item.get("latestGninaStatus")
            lines.append(
                f"- {item['module']}: {status}. Processed rows {processed}/{candidate_rows} "
                f"({progress}); latest pair {item.get('latestPairId')}; latest status "
                f"{latest_status}; log `{item.get('logPath')}`. "
                f"{item.get('evidence', '')}"
            )
    lines.extend([
        "",
        "## Completed Computation",
        "",
    ])
    for item in payload["completedModules"]:
        status = item.get("status", "")
        lines.append(f"- {item['module']}: {status}. {item.get('evidence', '')}")
    lines.extend(["", "## Remaining Local Blockers", ""])
    for item in payload["localBlockers"]:
        lines.append(f"- {item['module']}: {item['status']}. {item['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build computation-first SOTA closure summary.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-json", default="outputs/sota_validation/sota_compute_closure_summary.json")
    parser.add_argument("--out-md", default="outputs/sota_validation/SOTA_COMPUTE_CLOSURE_SUMMARY.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root)
    write_json(root / args.out_json, payload)
    (root / args.out_md).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "headline": payload["headline"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
