from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "sizeBytes": stat.st_size,
        "mtimeUtc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def pct(value: float | int | None, total: float | int | None) -> float | None:
    if value is None or total in (None, 0):
        return None
    return float(value) / float(total) * 100.0


def counts_to_rows(counter: dict[str, Any] | None, limit: int | None = None) -> list[dict[str, Any]]:
    if not counter:
        return []
    rows = [{"label": str(label), "value": value} for label, value in counter.items()]
    rows.sort(key=lambda row: row["value"], reverse=True)
    return rows[:limit] if limit else rows


def module_lookup(modules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row.get("module", ""): row for row in modules}


def build_payload(root: Path) -> dict[str, Any]:
    diffdock = load_json(root / "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json", {})
    closure = load_json(root / "outputs/sota_validation/sota_compute_closure_summary.json", {})
    feasibility = load_json(root / "outputs/sota_validation/model_feasibility/sota_model_feasibility_summary.json", {})
    sota_ready = load_json(
        root / "outputs/sota_validation/final_prioritization/final_priority_sota_ready_summary.json", {}
    )
    validation = load_json(root / "outputs/sota_validation/experimental_validation/experimental_validation_panel_summary.json", {})
    wetlab = load_json(root / "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json", {})
    final_pre_experiment_gate = load_json(
        root / "outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate_summary.json", {}
    )
    detailed_candidate_review = load_json(
        root / "outputs/sota_validation/wetlab_candidate_detailed_review/wetlab_candidate_detailed_review_summary.json",
        {},
    )
    final_priority = load_json(
        root / "outputs/sota_validation/final_prioritization/final_candidate_priority_summary.json", {}
    )
    experiment_closure = load_json(
        root / "outputs/sota_validation/experiment_closure_audit/experiment_closure_audit.json", {}
    )

    headline = closure.get("headline", {})
    modules = closure.get("completedModules", [])
    by_module = module_lookup(modules)

    standard_full_summary_path = (
        root / "outputs/sota_validation/standard_pose_validation_full3921/standard_pose_validation_summary.json"
    )
    standard_full = load_json(standard_full_summary_path, None)
    standard_log = root / "logs/standard_pose_validation_full3921.log"

    full_diffdock = {
        "status": "running",
        "completedJobs": diffdock.get("completedJobs") or headline.get("fullDiffdockCompletedJobs"),
        "totalJobs": diffdock.get("totalJobs") or headline.get("fullDiffdockTotalJobs"),
        "completedJobPct": diffdock.get("completedJobPct") or headline.get("fullDiffdockCompletedJobPct"),
        "scoredRows": diffdock.get("scoredRows") or headline.get("fullDiffdockScoredRows"),
        "totalRows": diffdock.get("totalRows") or headline.get("fullDiffdockTotalRows"),
        "scoredRowPct": diffdock.get("scoredRowPct") or headline.get("fullDiffdockScoredRowPct"),
        "completedOutputs": diffdock.get("completedOutputs") or headline.get("fullDiffdockCompletedOutputs"),
        "missingOutputsInScoredJobs": diffdock.get("missingOutputsInScoredJobs")
        or headline.get("fullDiffdockMissingOutputsInScoredJobs"),
        "activeLocks": diffdock.get("activeLocks") or headline.get("fullDiffdockActiveLocks") or [],
        "activeDetails": diffdock.get("activeDetails") or [],
        "etaHours": diffdock.get("etaHours") or headline.get("fullDiffdockEtaHours"),
        "etaDays": diffdock.get("etaDays") or headline.get("fullDiffdockEtaDays"),
        "estimatedFinishUtc": diffdock.get("estimatedFinishUtc") or headline.get("fullDiffdockEstimatedFinishUtc"),
        "busyGpuCount": (diffdock.get("gpu") or {}).get("busyGpuCount") or headline.get("fullDiffdockBusyGpuCount"),
        "gpuCount": (diffdock.get("gpu") or {}).get("gpuCount") or headline.get("fullDiffdockGpuCount"),
        "gpus": (diffdock.get("gpu") or {}).get("gpus") or [],
        "createdUtc": diffdock.get("createdUtc"),
        "interpretationZh": "全量 DiffDock 正在覆盖 druggable-proteome 候选队列。已完成 chunk 中的 missing output 是技术性缺失，不应解释为生物学阴性。",
    }

    multi_ligand_rescue = {
        "status": "queued_for_auto_start_after_single_ligand_rescue",
        "candidateLigands": headline.get("fullDiffdockMultiLigandRescueCandidateLigands"),
        "queuedLigands": headline.get("fullDiffdockMultiLigandRescueQueuedLigands"),
        "queuedRows": headline.get("fullDiffdockMultiLigandRescueQueuedRows"),
        "jobs": headline.get("fullDiffdockMultiLigandRescueJobs"),
        "watcherReady": headline.get("fullDiffdockMultiLigandRescueWatcherReady"),
        "watcherLog": headline.get("fullDiffdockMultiLigandRescueWatcherLog"),
        "latestAuditScoredLigands": headline.get("fullDiffdockLatestFailureAuditScoredLigands"),
        "latestAuditScoredRows": headline.get("fullDiffdockLatestFailureAuditScoredRows"),
        "latestAuditMissingRows": headline.get("fullDiffdockLatestFailureAuditMissingRows"),
        "latestAuditMissingPct": headline.get("fullDiffdockLatestFailureAuditMissingPct"),
        "latestAuditRescueRecommendedLigands": headline.get(
            "fullDiffdockLatestFailureAuditRescueRecommendedLigands"
        ),
        "latestAuditRescueRecommendedRows": headline.get("fullDiffdockLatestFailureAuditRescueRecommendedRows"),
        "latestAuditZeroCompletedChunks": headline.get("fullDiffdockLatestFailureAuditZeroCompletedChunks"),
        "latestAuditMaskRotateZeroCompletedChunks": headline.get(
            "fullDiffdockLatestFailureAuditMaskRotateZeroCompletedChunks"
        ),
        "interpretationZh": (
            "预构建 multi-ligand rescue 队列是当前可运行快照；最终 watcher 会在主队列和单 ligand rescue 完成后重跑 failure audit，"
            "再按最新技术缺失结果重建 rescue 队列。"
        ),
    }

    standard_full_status = {
        "status": "completed" if standard_full else "running_no_summary_yet",
        "summary": standard_full,
        "summaryFile": file_status(standard_full_summary_path),
        "logFile": file_status(standard_log),
        "interpretationZh": (
            "full3921 标准 PoseBusters/ProLIF 结构验证已完成。"
            if standard_full
            else "full3921 标准 PoseBusters/ProLIF 结构验证仍在运行，最终 summary 尚未生成。"
        ),
    }

    active = closure.get("activeComputation", [])
    rescue = next((row for row in active if "rescue" in row.get("module", "").lower()), {})

    evidence_layers = [
        {
            "label": "KG explainability",
            "value": headline.get("top1000KgPathPct"),
            "suffix": "%",
            "note": f"{headline.get('top1000KgPathRows', 'NA')} path rows",
        },
        {
            "label": "Structure A/B pocket",
            "value": headline.get("structureConfidenceModerateOrHighPct"),
            "suffix": "%",
            "note": f"{headline.get('structureConfidenceAuditedRows', 'NA')} audited poses",
        },
        {
            "label": "Pose interpretability",
            "value": headline.get("poseInterpretabilityInterpretablePct"),
            "suffix": "%",
            "note": f"{headline.get('poseInterpretabilityContactResidueRows', 'NA')} residue contacts",
        },
        {
            "label": "Local DTI support",
            "value": headline.get("independentDtiFull3921AbSupportedPct"),
            "suffix": "%",
            "note": f"{headline.get('independentDtiFull3921ScoredRows', 'NA')} scored rows",
        },
        {
            "label": "ML ADMET low/manageable",
            "value": headline.get("mlAdmetLowRiskCandidatePct"),
            "suffix": "%",
            "note": f"{headline.get('mlAdmetTrainedEndpointCount', 'NA')} endpoints",
        },
        {
            "label": "Tissue context HPA",
            "value": headline.get("tissueContextCandidatePositivePct"),
            "suffix": "%",
            "note": "relevant tissue expression",
        },
        {
            "label": "GTEx context",
            "value": headline.get("gtexContextCandidatePositivePct"),
            "suffix": "%",
            "note": headline.get("gtexContextRelease", "GTEx"),
        },
        {
            "label": "Network medicine",
            "value": headline.get("sotaNetworkPositivePct"),
            "suffix": "%",
            "note": "STRING subnet support",
        },
    ]

    structural_models = [
        {
            "label": "DiffDock disease-direction reps",
            "completed": headline.get("diffdockCompletedRows"),
            "total": (headline.get("diffdockCompletedRows") or 0) + (headline.get("diffdockMissingRows") or 0),
            "pct": headline.get("diffdockCompletionPct"),
            "note": "SDF/SMILES rescue complete",
        },
        {
            "label": "Vina final3921",
            "completed": headline.get("vinaConsensusFinal3921ScoredRows"),
            "total": headline.get("vinaConsensusFinal3921CandidateRows"),
            "pct": headline.get("vinaConsensusFinal3921ScoredPct"),
            "note": "classical score-only stress test",
        },
        {
            "label": "smina final3921",
            "completed": headline.get("sminaFinal3921ScoredRows"),
            "total": headline.get("sminaFinal3921CandidateRows"),
            "pct": headline.get("sminaFinal3921ScoredPct"),
            "note": "independent classical docking score",
        },
        {
            "label": "GNINA CNN full3921",
            "completed": headline.get("gninaFull3921ScoredRows"),
            "total": headline.get("gninaFull3921CandidateRows"),
            "pct": headline.get("gninaFull3921ScoredPct"),
            "note": "CPU execution, GPU preserved for DiffDock",
        },
        {
            "label": "Boltz-2 Top50",
            "completed": headline.get("boltz2CompletedRows"),
            "total": headline.get("boltz2CandidateRows"),
            "pct": headline.get("boltz2CompletedPct"),
            "note": "orthogonal complex model",
        },
        {
            "label": "Boltz-2 high-sampling finalists",
            "completed": headline.get("boltz2HighCompletedRows"),
            "total": headline.get("boltz2HighCandidateRows"),
            "pct": headline.get("boltz2HighCompletedPct"),
            "note": "higher confidence finalist rerun",
        },
    ]

    blockers = [
        {
            "label": "LINCS/CMap perturbation signatures",
            "status": "external_data_missing",
            "detailZh": "候选药物和疾病方向已完成 readiness audit，但本地没有扰动签名和疾病 DEG/signature 文件。",
        },
        {
            "label": "DrugBAN formal pretrained inference",
            "status": "dependency_missing",
            "detailZh": "local supervised DTI 已完成；正式 DrugBAN 分支仍缺 dgl 和 dgllife 环境。",
        },
        {
            "label": "DeepDTA/GraphDTA validated checkpoints",
            "status": "checkpoint_missing",
            "detailZh": "需要可信 pretrained weights 或重新训练配置，当前不把本地模型冒充外部 pretrained SOTA。",
        },
        {
            "label": "Chai-1 / AF3-style optional corroboration",
            "status": "optional_external_model",
            "detailZh": "Boltz-2 Top50 和 high-sampling finalists 已完成；Chai-1 或 AF3-style 可作为最终候选外部结构佐证。",
        },
    ]

    module_status_counts = Counter(row.get("status", "unknown") for row in modules)

    return {
        "updatedUtc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sourceUpdatedUtc": closure.get("created_utc") or diffdock.get("createdUtc"),
        "summary": {
            "completedModuleCount": len(modules),
            "activeComputationCount": len(active),
            "experimentClosureStatus": experiment_closure.get("overallStatus")
            or headline.get("experimentClosureOverallStatus"),
            "experimentClosurePassedRequiredChecks": experiment_closure.get("passedRequiredChecks")
            or headline.get("experimentClosurePassedRequiredChecks"),
            "experimentClosureTotalRequiredChecks": experiment_closure.get("totalRequiredChecks")
            or headline.get("experimentClosureTotalRequiredChecks"),
            "artifactManifestCount": headline.get("artifactManifestCount"),
            "artifactManifestTotalSizeBytes": headline.get("artifactManifestTotalSizeBytes"),
            "sourceScriptCount": headline.get("artifactManifestSourceScriptCount"),
            "candidateRows": headline.get("finalPriorityCandidateRows") or sota_ready.get("candidateRows"),
            "sotaReadyA": (sota_ready.get("tierCounts") or {}).get("A_sota_ready_expert_priority"),
            "sotaReadyB": (sota_ready.get("tierCounts") or {}).get("B_review_ready_priority"),
            "experimentReadyRows": validation.get("experimentReadyRows"),
            "reviewReadyRows": validation.get("reviewReadyRows"),
            "balancedPanelRows": validation.get("balancedPanelRows"),
            "wave1Rows": headline.get("validationPanelDiversityWave1Rows"),
        },
        "active": {
            "fullDiffdock": full_diffdock,
            "standardPoseFull3921": standard_full_status,
            "experimentClosure": {
                "overallStatus": experiment_closure.get("overallStatus")
                or headline.get("experimentClosureOverallStatus"),
                "passedRequiredChecks": experiment_closure.get("passedRequiredChecks")
                or headline.get("experimentClosurePassedRequiredChecks"),
                "totalRequiredChecks": experiment_closure.get("totalRequiredChecks")
                or headline.get("experimentClosureTotalRequiredChecks"),
                "failedRequiredChecks": experiment_closure.get("failedRequiredChecks")
                or headline.get("experimentClosureFailedRequiredChecks")
                or [],
                "createdUtc": experiment_closure.get("createdUtc"),
                "auditJson": "outputs/sota_validation/experiment_closure_audit/experiment_closure_audit.json",
                "auditMarkdown": "outputs/sota_validation/experiment_closure_audit/EXPERIMENT_CLOSURE_AUDIT.md",
                "interpretationZh": "全部 required checks 通过前，实验计算闭环不能标记完成。",
            },
            "ligandRescue": {
                "status": rescue.get("status", "queued_for_auto_start_after_main_queue"),
                "ligandId": rescue.get("ligandId") or headline.get("fullDiffdockLigandRescueLigandId"),
                "queuedRows": rescue.get("queuedRows") or headline.get("fullDiffdockLigandRescueQueuedRows"),
                "jobs": rescue.get("jobs") or headline.get("fullDiffdockLigandRescueJobs"),
                "watcherReady": rescue.get("watcherReady") or headline.get("fullDiffdockLigandRescueWatcherReady"),
                "watcherLog": rescue.get("watcherLog") or headline.get("fullDiffdockLigandRescueWatcherLog"),
                "interpretationZh": rescue.get("evidence")
                or "CHEMBL3039504 parent-ligand rescue queue has been prepared and waits for GPU idle after the main queue.",
            },
            "multiLigandRescue": multi_ligand_rescue,
        },
        "completedModules": [
            {
                "module": row.get("module"),
                "status": row.get("status"),
                "evidence": row.get("evidence"),
                "candidateRows": row.get("candidateRows"),
                "completedRows": row.get("completedRows") or row.get("scoredRows"),
                "completionPct": row.get("completionPct") or row.get("scoredPct") or row.get("supportedPct"),
            }
            for row in modules
        ],
        "moduleStatusCounts": counts_to_rows(dict(module_status_counts)),
        "evidenceLayers": evidence_layers,
        "structuralModels": structural_models,
        "sotaReady": {
            "candidateRows": sota_ready.get("candidateRows"),
            "uniqueDrugs": sota_ready.get("uniqueDrugs"),
            "uniqueTargets": sota_ready.get("uniqueTargets"),
            "tierCounts": counts_to_rows(sota_ready.get("tierCounts")),
            "actionCounts": counts_to_rows(sota_ready.get("actionCounts")),
            "top100KnownRows": sota_ready.get("sotaReadyTop100KnownRows"),
            "top100NovelRows": sota_ready.get("sotaReadyTop100NovelRows"),
            "top100StructureABRows": sota_ready.get("sotaReadyTop100StructureABRows"),
            "top100TargetABRows": sota_ready.get("sotaReadyTop100TargetABRows"),
            "expertShortlistRows": sota_ready.get("expertShortlistRows"),
            "novelShortlistRows": sota_ready.get("novelShortlistRows"),
            "diverseShortlistRows": sota_ready.get("diverseShortlistRows"),
        },
        "validationPanel": {
            "candidateRows": validation.get("candidateRows"),
            "experimentReadyRows": validation.get("experimentReadyRows"),
            "reviewReadyRows": validation.get("reviewReadyRows"),
            "novelReadyRows": validation.get("novelExperimentOrReviewReadyRows"),
            "positiveControlReadyRows": validation.get("positiveControlReadyRows"),
            "balancedPanelRows": validation.get("balancedPanelRows"),
            "balancedPanelDirections": validation.get("balancedPanelDirections"),
            "balancedUniqueDrugs": validation.get("balancedPanelUniqueDrugs"),
            "balancedUniqueTargets": validation.get("balancedPanelUniqueTargets"),
            "balancedUniqueScaffolds": validation.get("balancedPanelUniqueScaffolds"),
            "tierCounts": counts_to_rows(validation.get("validationTierCounts")),
            "gateCounts": counts_to_rows(validation.get("validationGateCounts")),
            "assayCounts": counts_to_rows(validation.get("assayModalityCounts")),
        },
        "wetlabFocus": {
            "candidateRows": wetlab.get("candidateRows"),
            "experimentReadyRows": wetlab.get("experimentReadyRows"),
            "reviewReadyRows": wetlab.get("reviewReadyRows"),
            "expertTop50Rows": wetlab.get("expertTop50Rows"),
            "wave1Rows": wetlab.get("wave1Rows"),
            "purchaseAndAssayQueueRows": wetlab.get("purchaseAndAssayQueueRows"),
            "firstExperimentPanelRows": wetlab.get("firstExperimentPanelRows"),
            "firstExperimentRoleCounts": counts_to_rows(wetlab.get("firstExperimentPanelRoleCounts")),
            "firstExperimentDirectionCounts": counts_to_rows(wetlab.get("firstExperimentPanelDirectionCounts")),
            "prePurchaseFocusRows": wetlab.get("prePurchaseFocusRows"),
            "prePurchaseTierCounts": counts_to_rows(wetlab.get("prePurchaseTierCounts")),
            "prePurchaseActionCounts": counts_to_rows(wetlab.get("prePurchaseActionCounts")),
            "experimentExecutionProtocolRows": wetlab.get("experimentExecutionProtocolRows"),
            "experimentExecutionProtocolCoreRows": wetlab.get("experimentExecutionProtocolCoreRows"),
            "experimentExecutionProtocolExtensionRows": wetlab.get("experimentExecutionProtocolExtensionRows"),
            "experimentExecutionProtocolAssayCounts": counts_to_rows(
                wetlab.get("experimentExecutionProtocolAssayCounts")
            ),
            "procurementPlatformChecklistRows": wetlab.get("procurementPlatformChecklistRows"),
            "procurementPlatformChecklistCoreRows": wetlab.get("procurementPlatformChecklistCoreRows"),
            "procurementPlatformChecklistExtensionRows": wetlab.get("procurementPlatformChecklistExtensionRows"),
            "procurementPlatformChecklistStatusCounts": counts_to_rows(
                wetlab.get("procurementPlatformChecklistStatusCounts")
            ),
            "finalPreExperimentGateRows": final_pre_experiment_gate.get("inputRows"),
            "finalPreExperimentGateTierCounts": counts_to_rows(
                final_pre_experiment_gate.get("finalGateTierCounts")
            ),
            "finalPreExperimentGateGoCandidateRows": final_pre_experiment_gate.get("goCandidateRows"),
            "finalPreExperimentGateBackupRows": final_pre_experiment_gate.get("backupRows"),
            "finalPreExperimentGateExpertReviewRows": final_pre_experiment_gate.get("expertReviewRows"),
            "finalPreExperimentGateHoldRows": final_pre_experiment_gate.get("holdRows"),
            "finalPreExperimentManualGateCount": final_pre_experiment_gate.get("manualGateCount"),
            "finalPreExperimentOutputs": final_pre_experiment_gate.get("outputs") or {},
            "detailedCandidateReviewRows": detailed_candidate_review.get("candidateRows"),
            "detailedCandidateReviewLiteratureAuditRows": detailed_candidate_review.get("literatureAuditRows"),
            "detailedCandidateReviewOutputs": detailed_candidate_review.get("outputs") or {},
            "outputs": wetlab.get("outputs") or {},
            "interpretationZh": (
                "湿实验前收敛口径为：从全候选进入采购/实验队列，再压缩为第一轮 12 个候选，"
                "首批优先核心 6 个；最终实验前门控会继续要求临床可达浓度、机制方向、模型表达、"
                "精确文献 novelty、采购质量和 assay interference 等人工核查后才能下单。"
            ),
        },
        "benchmarks": {
            "knownPairRecallAt100000Pct": headline.get("knownPairRecallAt100000Pct"),
            "knownPairEnrichmentAt100000": headline.get("knownPairEnrichmentAt100000"),
            "knownRecordRecallAt100000Pct": headline.get("knownRecordRecallAt100000Pct"),
            "finalPriorityValidationRecallAt100Pct": headline.get("finalPriorityValidationRecallAt100Pct"),
            "finalPriorityValidationPrecisionAt100Pct": headline.get("finalPriorityValidationPrecisionAt100Pct"),
            "finalPriorityValidationEnrichmentAt100": headline.get("finalPriorityValidationEnrichmentAt100"),
            "significanceTop100ObservedHits": headline.get("significanceTop100ObservedHits"),
            "significanceTop100GlobalExpectedHits": headline.get("significanceTop100GlobalExpectedHits"),
            "significanceTop100GlobalP": headline.get("significanceTop100GlobalP"),
            "concordanceMultiEvidencePct": headline.get("concordanceMultiEvidencePct"),
            "concordanceTop100MultiEvidencePct": headline.get("concordanceTop100MultiEvidencePct"),
        },
        "dependencies": {
            "moduleImports": {
                key: {"available": value.get("available"), "detail": value.get("detail")}
                for key, value in (feasibility.get("moduleImports") or {}).items()
            },
            "binaries": {
                key: {
                    "available": value.get("available"),
                    "runtimeReady": value.get("runtimeReady"),
                    "detail": value.get("detail"),
                }
                for key, value in (feasibility.get("binaries") or {}).items()
            },
            "priorityQueue": feasibility.get("priorityQueue") or [],
            "blockers": blockers,
        },
        "finalPriority": {
            "candidateRows": final_priority.get("candidateRows"),
            "tierCounts": counts_to_rows(final_priority.get("tierCounts")),
            "reviewTrackCounts": counts_to_rows(final_priority.get("reviewTrackCounts"), limit=10),
        },
        "notableModules": {
            key: by_module.get(key, {})
            for key in [
                "DiffDock disease-direction structural docking",
                "Standard PoseBusters/ProLIF Top300 structural validation extension",
                "GNINA CNN full final-candidate structural rescoring",
                "Boltz-2 high-sampling finalist validation",
                "Local supervised independent DTI full final-candidate extension",
                "Experimental validation planning panel",
                "FDA label temporal generalization audit",
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build website compute-status data from current local artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/assets/compute-status.js")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "window.BIOMASTER_COMPUTE_STATUS = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(out),
                "completedModuleCount": payload["summary"]["completedModuleCount"],
                "fullDiffdockPct": payload["active"]["fullDiffdock"]["completedJobPct"],
                "updatedUtc": payload["updatedUtc"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
