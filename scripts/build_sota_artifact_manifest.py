from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXCLUDED_OUTPUT_NAMES = {
    "sota_artifact_manifest.csv",
    "sota_artifact_manifest.json",
    "SOTA_ARTIFACT_MANIFEST.md",
    "sota_compute_closure_summary.json",
    "SOTA_COMPUTE_CLOSURE_SUMMARY.md",
}


SOURCE_RULES: list[tuple[str, str]] = [
    ("outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json", "scripts/summarize_diffdock_full_progress.py"),
    ("outputs/report_scale/diffdock_full_run/DIFFDOCK_FULL_PROGRESS_SUMMARY.md", "scripts/summarize_diffdock_full_progress.py"),
    ("outputs/report_scale/diffdock_full_ligand_rescue/**/*", "scripts/build_full_diffdock_ligand_rescue_queue.py"),
    ("outputs/report_scale/diffdock_full_multi_ligand_rescue/**/*", "scripts/build_diffdock_full_multi_ligand_rescue_queue.py"),
    ("outputs/report_scale/diffdock_full_final_scores/**/*", "scripts/watch_and_finalize_diffdock_full_after_rescues.py"),
    ("outputs/report_scale/diffdock_scores_full_913170*.csv", "scripts/watch_and_finalize_diffdock_full_after_rescues.py"),
    ("outputs/report_scale/full_diffdock_final/**/*", "scripts/watch_and_run_full_diffdock_post_finalization.py"),
    ("data/processed/ligands_sdf_chembl_parent/*", "scripts/build_full_diffdock_ligand_rescue_queue.py"),
    ("outputs/report_scale/diffdock_top10000_run/diffdock_scores_top10000.csv", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/diffdock_top10000_failure_audit.csv", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/stage6_top10000_consensus_candidates.csv", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/stage6_top10000_consensus_candidates.metadata.json", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/stage6_top10000_report_summary.json", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/stage6_top10000_structure_audit_input.csv", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/stage6_top10000_structure_audit_input.metadata.json", "scripts/finalize_diffdock_top10000_when_done.py"),
    ("outputs/report_scale/STAGE6_TOP10000_STRUCTURE_AUDIT_SUMMARY.json", "scripts/build_structure_confidence_audit.py"),
    ("outputs/report_scale/STAGE6_TOP10000_STRUCTURE_AUDIT_SUMMARY.md", "scripts/build_structure_confidence_audit.py"),
    ("outputs/sota_validation/final_diffdock_completion_after_*.json", "scripts/merge_druggable_diffdock_results.py"),
    ("outputs/sota_validation/opentargets_direction_scores_top_pages*", "scripts/download_opentargets_filtered.py"),
    ("outputs/sota_validation/txgnn_*", "scripts/run_txgnn_cancer_inference.py"),
    ("outputs/sota_validation/txgnn_direction_runs/*", "scripts/run_txgnn_cancer_inference.py"),
    ("outputs/sota_validation/sota_p0_validation_summary.json", "scripts/run_sota_p0_validation.py"),
    ("outputs/sota_validation/known_target_*.csv", "scripts/run_sota_p0_validation.py"),
    ("outputs/sota_validation/admet_repurposing/*", "scripts/build_admet_repurposing_audit.py"),
    ("outputs/sota_validation/kg_explainability*/*", "scripts/build_kg_explainability_audit.py"),
    ("outputs/sota_validation/known_target_stratified/*", "scripts/build_known_target_stratified_validation.py"),
    ("outputs/sota_validation/pose_sanity*/*", "scripts/build_pose_sanity_audit.py"),
    ("outputs/sota_validation/pose_interpretability/*", "scripts/build_pose_interpretability_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_pose_interpretability*", "scripts/build_pose_interpretability_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_POSE_INTERPRETABILITY_AUDIT.md", "scripts/build_pose_interpretability_audit.py"),
    ("outputs/sota_validation/pose_quality/*", "scripts/build_pose_quality_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_pose_quality*", "scripts/build_pose_quality_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_POSE_QUALITY_AUDIT.md", "scripts/build_pose_quality_audit.py"),
    ("outputs/sota_validation/standard_pose_validation/*", "scripts/build_standard_pose_validation_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_standard_pose_validation*", "scripts/build_standard_pose_validation_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_STANDARD_POSE_VALIDATION_AUDIT.md", "scripts/build_standard_pose_validation_audit.py"),
    ("outputs/sota_validation/vina_consensus_rescoring/*", "scripts/build_vina_consensus_rescoring_audit.py"),
    ("outputs/sota_validation/vina_consensus_rescoring/pdbqt_cache/*/*", "scripts/build_vina_consensus_rescoring_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_vina_consensus*", "scripts/build_vina_consensus_rescoring_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_VINA_CONSENSUS_AUDIT.md", "scripts/build_vina_consensus_rescoring_audit.py"),
    ("outputs/sota_validation/smina_rescoring/*", "scripts/build_smina_structural_rescoring_audit.py"),
    ("outputs/sota_validation/gnina_cnn_rescoring/*", "scripts/build_gnina_cnn_rescoring_audit.py"),
    ("outputs/sota_validation/boltz2_complex_validation/boltz2_complex_validation*", "scripts/build_boltz2_complex_validation_audit.py"),
    ("outputs/sota_validation/boltz2_complex_validation/BOLTZ2_COMPLEX_VALIDATION_AUDIT.md", "scripts/build_boltz2_complex_validation_audit.py"),
    ("outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed/*", "scripts/build_boltz2_ligand_repair_package.py"),
    ("outputs/sota_validation/boltz2_complex_validation/boltz2_input*", "scripts/build_boltz2_complex_input_package.py"),
    ("outputs/sota_validation/boltz2_complex_validation/inputs/*.yaml", "scripts/build_boltz2_complex_input_package.py"),
    ("outputs/sota_validation/boltz2_complex_validation/BOLTZ2_COMPLEX_INPUT_PACKAGE.md", "scripts/build_boltz2_complex_input_package.py"),
    ("outputs/sota_validation/boltz2_high_sampling_validation/*", "scripts/build_boltz2_high_sampling_validation.py"),
    ("outputs/sota_validation/boltz2_high_sampling_validation/inputs/*.yaml", "scripts/build_boltz2_high_sampling_validation.py"),
    ("outputs/sota_validation/boltz2_high_sampling_validation/shards/*/*.yaml", "scripts/build_boltz2_high_sampling_validation.py"),
    ("outputs/sota_validation/boltz2_high_sampling_validation/runs/**/*", "scripts/build_boltz2_high_sampling_validation.py"),
    ("outputs/sota_validation/experimental_validation/*", "scripts/build_experimental_validation_panel.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_experimental_validation*", "scripts/build_experimental_validation_panel.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_EXPERIMENTAL_VALIDATION_PANEL.md", "scripts/build_experimental_validation_panel.py"),
    ("outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate*", "scripts/build_final_pre_experiment_gate.py"),
    ("outputs/sota_validation/wetlab_validation_package/WETLAB_FINAL_PRE_EXPERIMENT_GATE.md", "scripts/build_final_pre_experiment_gate.py"),
    ("outputs/sota_validation/wetlab_validation_package/*", "scripts/build_wetlab_validation_package.py"),
    ("outputs/sota_validation/wetlab_candidate_detailed_review/*", "scripts/build_wetlab_candidate_detailed_review_pdf.py"),
    ("docs/assets/wetlab-candidate-detailed-review.pdf", "scripts/build_wetlab_candidate_detailed_review_pdf.py"),
    ("outputs/sota_validation/professor_candidate_detailed_review/*", "scripts/build_professor_candidate_detailed_review_pdf.py"),
    ("docs/assets/professor-candidate-detailed-review.pdf", "scripts/build_professor_candidate_detailed_review_pdf.py"),
    ("outputs/sota_validation/professor_candidate_gpt_review_brief/*", "scripts/build_professor_candidate_gpt_review_brief_pdf.py"),
    ("docs/assets/professor-candidate-gpt-review-brief.pdf", "scripts/build_professor_candidate_gpt_review_brief_pdf.py"),
    ("outputs/sota_validation/validation_panel_benchmark/*", "scripts/build_validation_panel_benchmark.py"),
    ("outputs/sota_validation/validation_panel_diversity/*", "scripts/build_validation_panel_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_validation_panel_*diversity*", "scripts/build_validation_panel_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_validation_panel_wave1*", "scripts/build_validation_panel_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_VALIDATION_PANEL_DIVERSITY_AUDIT.md", "scripts/build_validation_panel_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_validation_panel*", "scripts/build_validation_panel_benchmark.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_VALIDATION_PANEL_BENCHMARK.md", "scripts/build_validation_panel_benchmark.py"),
    ("outputs/sota_validation/fda_label_mechanism/*", "scripts/build_fda_label_mechanism_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_fda_label_mechanism*", "scripts/build_fda_label_mechanism_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_FDA_LABEL_MECHANISM_AUDIT.md", "scripts/build_fda_label_mechanism_audit.py"),
    ("outputs/sota_validation/fda_label_temporal/*", "scripts/build_fda_label_temporal_generalization_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_fda_label_temporal*", "scripts/build_fda_label_temporal_generalization_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_FDA_LABEL_TEMPORAL_GENERALIZATION_AUDIT.md", "scripts/build_fda_label_temporal_generalization_audit.py"),
    ("outputs/sota_validation/structure_confidence_top10000/*", "scripts/build_structure_confidence_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_structure*", "scripts/build_structure_confidence_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_STRUCTURE_CONFIDENCE_AUDIT.md", "scripts/build_structure_confidence_audit.py"),
    ("outputs/sota_validation/target_druggability/*", "scripts/build_target_druggability_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_target_druggability*", "scripts/build_target_druggability_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_TARGET_DRUGGABILITY_AUDIT.md", "scripts/build_target_druggability_audit.py"),
    ("outputs/sota_validation/chemotype_diversity/*", "scripts/build_chemotype_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_chemotype*", "scripts/build_chemotype_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_CHEMOTYPE_DIVERSITY_AUDIT.md", "scripts/build_chemotype_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_sota_ready*", "scripts/build_sota_ready_decision_matrix.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_SOTA_READY_DECISION_MATRIX.md", "scripts/build_sota_ready_decision_matrix.py"),
    ("outputs/sota_validation/network_proximity/*", "scripts/build_network_proximity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_sota_network*", "scripts/build_sota_network_priority.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_SOTA_NETWORK_MEDICINE_AUDIT.md", "scripts/build_sota_network_priority.py"),
    ("outputs/sota_validation/tissue_context/*", "scripts/build_tissue_context_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_sota_context*", "scripts/build_tissue_context_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_SOTA_TISSUE_CONTEXT_AUDIT.md", "scripts/build_tissue_context_audit.py"),
    ("outputs/sota_validation/gtex_context/*", "scripts/build_gtex_context_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_gtex_context*", "scripts/build_gtex_context_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_GTEX_CONTEXT_AUDIT.md", "scripts/build_gtex_context_audit.py"),
    ("outputs/sota_validation/depmap_oncology_dependency/*", "scripts/build_depmap_oncology_dependency_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_depmap_oncology*", "scripts/build_depmap_oncology_dependency_audit.py"),
    ("outputs/sota_validation/lincs_cmap_readiness/*", "scripts/build_lincs_cmap_readiness_audit.py"),
    ("outputs/sota_validation/external_sota_model_inputs/*", "scripts/build_external_sota_model_input_package.py"),
    ("outputs/sota_validation/independent_dti_supervised/*", "scripts/build_independent_dti_supervised_audit.py"),
    ("outputs/sota_validation/external_dependency_audit/*", "scripts/audit_sota_external_dependencies.py"),
    ("outputs/sota_validation/SOTA_EXTERNAL_DEPENDENCY_DOWNLOAD_PLAN.md", "scripts/audit_sota_external_dependencies.py"),
    ("outputs/sota_validation/ml_admet/*", "scripts/build_ml_admet_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_ml_admet*", "scripts/build_ml_admet_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_ML_ADMET_AUDIT.md", "scripts/build_ml_admet_audit.py"),
    ("outputs/sota_validation/model_feasibility/*", "scripts/build_sota_model_feasibility_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_rank_stability*", "scripts/build_rank_stability_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_RANK_STABILITY_AUDIT.md", "scripts/build_rank_stability_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_ablation*", "scripts/build_final_priority_ablation_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_ABLATION_AUDIT.md", "scripts/build_final_priority_ablation_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_validation*", "scripts/build_final_priority_validation.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_shortlist_known_target_audit.csv", "scripts/build_final_priority_validation.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_known_target_*.csv", "scripts/build_final_priority_validation.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_top_known_target_hits.csv", "scripts/build_final_priority_validation.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_novelty*", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_strict_novel_shortlist.csv", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_mechanism_extension_shortlist.csv", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_known_mechanism_positive_controls.csv", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_safety_context_review.csv", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_NOVELTY_LEAKAGE_AUDIT.md", "scripts/build_novelty_leakage_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_direction_specificity*", "scripts/build_direction_specificity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_direction_specific_shortlist.csv", "scripts/build_direction_specificity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_broad_generalist_review.csv", "scripts/build_direction_specificity_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_DIRECTION_SPECIFICITY_AUDIT.md", "scripts/build_direction_specificity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_topk_significance*", "scripts/build_topk_significance_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_direction_topk_significance.csv", "scripts/build_topk_significance_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_TOPK_SIGNIFICANCE_AUDIT.md", "scripts/build_topk_significance_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_evidence_concordance*", "scripts/build_evidence_concordance_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_high_concordance_shortlist.csv", "scripts/build_evidence_concordance_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_priority_single_evidence_review.csv", "scripts/build_evidence_concordance_audit.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_PRIORITY_EVIDENCE_CONCORDANCE_AUDIT.md", "scripts/build_evidence_concordance_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_candidate_divers*", "scripts/build_candidate_diversity_audit.py"),
    ("outputs/sota_validation/final_prioritization/final_candidate_priority*", "scripts/build_final_candidate_prioritization.py"),
    ("outputs/sota_validation/final_prioritization/FINAL_CANDIDATE_EXPERT_REVIEW.md", "scripts/build_final_candidate_prioritization.py"),
    ("outputs/sota_validation/top1000_*", "scripts/build_sota_readiness_audit.py"),
    ("outputs/sota_validation/sota_compute_closure_summary.json", "scripts/build_sota_compute_closure_summary.py"),
    ("outputs/sota_validation/SOTA_COMPUTE_CLOSURE_SUMMARY.md", "scripts/build_sota_compute_closure_summary.py"),
]


REPORT_SCALE_WHITELIST = [
    "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json",
    "outputs/report_scale/diffdock_full_run/DIFFDOCK_FULL_PROGRESS_SUMMARY.md",
    "outputs/report_scale/diffdock_full_ligand_rescue/**/*.json",
    "outputs/report_scale/diffdock_full_ligand_rescue/**/*.md",
    "outputs/report_scale/diffdock_full_ligand_rescue/**/*.csv",
    "outputs/report_scale/diffdock_full_multi_ligand_rescue/**/*.json",
    "outputs/report_scale/diffdock_full_multi_ligand_rescue/**/*.md",
    "outputs/report_scale/diffdock_full_multi_ligand_rescue/**/*.csv",
    "outputs/report_scale/diffdock_full_final_scores/**/*.json",
    "outputs/report_scale/diffdock_full_final_scores/**/*.md",
    "outputs/report_scale/diffdock_full_final_scores/**/*.csv",
    "outputs/report_scale/diffdock_scores_full_913170*.csv",
    "outputs/report_scale/full_diffdock_final/**/*.json",
    "outputs/report_scale/full_diffdock_final/**/*.md",
    "outputs/report_scale/full_diffdock_final/**/*.csv",
    "data/processed/ligands_sdf_chembl_parent/*.sdf",
    "outputs/report_scale/diffdock_top10000_run/diffdock_scores_top10000.csv",
    "outputs/report_scale/diffdock_top10000_failure_audit.csv",
    "outputs/report_scale/stage6_top10000_consensus_candidates.csv",
    "outputs/report_scale/stage6_top10000_consensus_candidates.metadata.json",
    "outputs/report_scale/stage6_top10000_report_summary.json",
    "outputs/report_scale/stage6_top10000_structure_audit_input.csv",
    "outputs/report_scale/stage6_top10000_structure_audit_input.metadata.json",
    "outputs/report_scale/STAGE6_TOP10000_STRUCTURE_AUDIT_SUMMARY.json",
    "outputs/report_scale/STAGE6_TOP10000_STRUCTURE_AUDIT_SUMMARY.md",
]

PUBLISHED_ASSET_WHITELIST = [
    "docs/assets/wetlab-candidate-detailed-review.pdf",
    "docs/assets/professor-candidate-detailed-review.pdf",
    "docs/assets/professor-candidate-gpt-review-brief.pdf",
]


def utc_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_script(rel_path: str) -> str:
    for pattern, script in SOURCE_RULES:
        if fnmatch.fnmatch(rel_path, pattern):
            return script
    return ""


def module_name(rel_path: str, script: str) -> str:
    if script:
        return Path(script).stem
    parts = Path(rel_path).parts
    if len(parts) >= 3:
        return parts[2]
    return "sota_validation"


def summarize_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return {"row_count": 0, "column_count": 0, "columns_sample": ""}
            row_count = sum(1 for _ in reader)
    except csv.Error:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = sum(1 for _ in handle)
        return {"row_count": max(lines - 1, 0), "column_count": None, "columns_sample": ""}
    return {
        "row_count": row_count,
        "column_count": len(header),
        "columns_sample": "|".join(header[:12]),
    }


def flatten_json(value: Any, prefix: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    if depth > 3:
        return []
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten_json(value[key], child_prefix, depth + 1))
        return items
    if isinstance(value, list):
        return [(f"{prefix}.length" if prefix else "length", len(value))]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > 90:
            value = value[:87] + "..."
        return [(prefix, value)]
    return []


def summarize_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"json_metric_summary": ""}
    metrics = []
    for key, value in flatten_json(payload):
        if key and value not in ("", None):
            metrics.append(f"{key}={value}")
        if len(metrics) >= 24:
            break
    return {"json_metric_summary": "; ".join(metrics)}


def summarize_markdown(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    heading_count = sum(1 for line in lines if line.lstrip().startswith("#"))
    return {"line_count": len(lines), "heading_count": heading_count}


def file_summary(path: Path, root: Path) -> dict[str, Any]:
    rel_path = path.relative_to(root).as_posix()
    stat = path.stat()
    script = source_script(rel_path)
    summary: dict[str, Any] = {
        "rel_path": rel_path,
        "module": module_name(rel_path, script),
        "source_script": script,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": utc_timestamp(stat.st_mtime),
        "sha256": sha256(path),
        "row_count": None,
        "column_count": None,
        "columns_sample": "",
        "line_count": None,
        "heading_count": None,
        "json_metric_summary": "",
    }
    if path.suffix.lower() == ".csv":
        summary.update(summarize_csv(path))
    elif path.suffix.lower() == ".json":
        summary.update(summarize_json(path))
    elif path.suffix.lower() == ".md":
        summary.update(summarize_markdown(path))
    return summary


def build_payload(root: Path) -> dict[str, Any]:
    output_root = root / "outputs/sota_validation"
    artifacts = []
    seen: set[str] = set()
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_OUTPUT_NAMES:
            continue
        item = file_summary(path, root)
        artifacts.append(item)
        seen.add(item["rel_path"])

    for rel_pattern in REPORT_SCALE_WHITELIST:
        if any(token in rel_pattern for token in ["*", "?", "["]):
            paths = sorted(root.glob(rel_pattern))
        else:
            paths = [root / rel_pattern]
        for path in paths:
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            if rel_path in seen:
                continue
            artifacts.append(file_summary(path, root))
            seen.add(rel_path)

    for rel_pattern in PUBLISHED_ASSET_WHITELIST:
        paths = sorted(root.glob(rel_pattern))
        for path in paths:
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            if rel_path in seen:
                continue
            artifacts.append(file_summary(path, root))
            seen.add(rel_path)

    suffix_counts = Counter(item["suffix"] or "no_suffix" for item in artifacts)
    module_counts = Counter(item["module"] for item in artifacts)
    latest_mtime = max((item["mtime_utc"] for item in artifacts), default="")
    total_size = sum(int(item["size_bytes"]) for item in artifacts)
    source_scripts = sorted({item["source_script"] for item in artifacts if item["source_script"]})

    return {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "File-level reproducibility manifest for computation outputs under outputs/sota_validation plus selected report_scale progress, Stage6 artifacts, and published static report assets.",
        "artifact_count": len(artifacts),
        "total_size_bytes": total_size,
        "latest_artifact_mtime_utc": latest_mtime,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "module_counts": dict(sorted(module_counts.items())),
        "source_scripts": source_scripts,
        "artifacts": artifacts,
    }


def write_csv(path: Path, artifacts: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rel_path",
        "module",
        "source_script",
        "suffix",
        "size_bytes",
        "mtime_utc",
        "sha256",
        "row_count",
        "column_count",
        "columns_sample",
        "line_count",
        "heading_count",
        "json_metric_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in artifacts:
            writer.writerow({key: item.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown(payload: dict[str, Any]) -> str:
    by_size = sorted(payload["artifacts"], key=lambda item: int(item["size_bytes"]), reverse=True)[:20]
    published_assets = [item for item in payload["artifacts"] if item["rel_path"].startswith("docs/assets/")]
    lines = [
        "# SOTA Computation Artifact Manifest",
        "",
        f"Generated: {payload['created_utc']}",
        "",
        "This manifest indexes computation outputs, selected report_scale progress and Stage6 artifacts, plus selected published static report assets.",
        "",
        "## Summary",
        "",
        f"- Artifact files indexed: {payload['artifact_count']}",
        f"- Total artifact size: {payload['total_size_bytes']} bytes",
        f"- Latest artifact mtime: {payload['latest_artifact_mtime_utc']}",
        f"- File types: {payload['suffix_counts']}",
        "",
        "## Source Script Coverage",
        "",
    ]
    for module, count in sorted(payload["module_counts"].items()):
        lines.append(f"- {module}: {count} files")

    if published_assets:
        lines.extend(
            [
                "",
                "## Published Static Assets",
                "",
                "| Artifact | Size bytes | Source script |",
                "| --- | ---: | --- |",
            ]
        )
        for item in sorted(published_assets, key=lambda row: row["rel_path"]):
            lines.append(f"| {item['rel_path']} | {item['size_bytes']} | {item.get('source_script', '')} |")

    lines.extend(["", "## Largest Artifacts", "", "| Artifact | Rows | Size bytes | Source script |", "| --- | ---: | ---: | --- |"])
    for item in by_size:
        rows = "" if item.get("row_count") is None else item.get("row_count")
        lines.append(
            f"| {item['rel_path']} | {rows} | {item['size_bytes']} | {item.get('source_script', '')} |"
        )

    lines.extend(
        [
            "",
            "## Machine-Readable Files",
            "",
            "- CSV manifest: `outputs/sota_validation/sota_artifact_manifest.csv`",
            "- JSON manifest: `outputs/sota_validation/sota_artifact_manifest.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a file-level manifest for BioMaster SOTA computation outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-csv", default="outputs/sota_validation/sota_artifact_manifest.csv")
    parser.add_argument("--out-json", default="outputs/sota_validation/sota_artifact_manifest.json")
    parser.add_argument("--out-md", default="outputs/sota_validation/SOTA_ARTIFACT_MANIFEST.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root)
    write_csv(root / args.out_csv, payload["artifacts"])
    write_json(root / args.out_json, payload)
    (root / args.out_md).write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_csv": args.out_csv,
                "out_json": args.out_json,
                "out_md": args.out_md,
                "artifact_count": payload["artifact_count"],
                "total_size_bytes": payload["total_size_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
