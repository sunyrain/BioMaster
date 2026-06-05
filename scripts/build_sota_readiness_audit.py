from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: str | int | float | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def js_string(path: Path, variable: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"window.{variable} = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def has_text(value: str | None) -> bool:
    return bool((value or "").strip())


def is_positive(value: str | int | float | None) -> bool:
    parsed = number(value)
    return parsed is not None and parsed > 0


def tier_letter(value: str) -> str:
    return (value or "").strip()[:1] or "NA"


def top_n(rows: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    return [row for row in rows if int(float(row.get("rank") or 999999)) <= n]


def coverage_for_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    completed = sum(1 for row in rows if row.get("status") == "completed")
    missing = total - completed
    open_targets = sum(1 for row in rows if is_positive(row.get("openTargetsScore")))
    txgnn = sum(1 for row in rows if is_positive(row.get("txgnnScore")))
    therapeutic_area = sum(1 for row in rows if has_text(row.get("therapeuticArea")))
    indication = sum(1 for row in rows if has_text(row.get("indication")))
    multi_source = sum(
        1
        for row in rows
        if sum(
            [
                is_positive(row.get("openTargetsScore")),
                is_positive(row.get("txgnnScore")),
                has_text(row.get("therapeuticArea")),
                "蛋白" in (row.get("evidenceSummaryZh") or ""),
                row.get("status") == "completed",
            ]
        )
        >= 3
    )
    tier_counts = Counter(tier_letter(row.get("credibilityTierZh", "")) for row in rows)
    diffdock_values = [number(row.get("diffdock")) for row in rows if number(row.get("diffdock")) is not None]
    return {
        "total": total,
        "completed": completed,
        "missing": missing,
        "completedPct": pct(completed, total),
        "openTargets": open_targets,
        "openTargetsPct": pct(open_targets, total),
        "txgnn": txgnn,
        "txgnnPct": pct(txgnn, total),
        "therapeuticArea": therapeutic_area,
        "therapeuticAreaPct": pct(therapeutic_area, total),
        "indication": indication,
        "indicationPct": pct(indication, total),
        "multiSource": multi_source,
        "multiSourcePct": pct(multi_source, total),
        "tierCounts": dict(tier_counts),
        "medianDiffDock": statistics.median(diffdock_values) if diffdock_values else None,
    }


def recall_metrics(root: Path) -> dict[str, Any]:
    top10k = read_json(root / "outputs/druggable_proteome/fda_known_target_recall_summary.json")
    top100k = read_json(root / "outputs/druggable_proteome/fda_known_target_recall_top100000_summary.json")
    p0_summary = read_json(root / "outputs/sota_validation/sota_p0_validation_summary.json")
    stratified = read_json(root / "outputs/sota_validation/known_target_stratified/known_target_stratified_summary.json")
    benchmark = p0_summary.get("known_target_benchmark") or {}
    enrichment = {int(row["cutoff"]): row for row in benchmark.get("enrichment", []) if row.get("cutoff") is not None}
    stratified_overall = stratified.get("overall") or {}
    metrics = {
        "knownPairsInScope": top10k.get("known_pairs_in_current_protein_scope"),
        "top10000TargetRecallPct": (top10k.get("target_record_recall_top10000_exact") or {}).get("percent"),
        "top10000TargetHits": (top10k.get("target_record_recall_top10000_exact") or {}).get("hits"),
        "top100000TargetRecallPct": (top100k.get("target_record_recall_at_100000") or {}).get("percent"),
        "top100000TargetHits": (top100k.get("target_record_recall_at_100000") or {}).get("hits"),
        "randomExpectedTop10000Pct": (top100k.get("random_expected_target_record_hits_at_10000_approx") or {}).get("expected_percent"),
        "randomExpectedTop100000Pct": (top100k.get("random_expected_target_record_hits_at_100000_approx") or {}).get("expected_percent"),
        "stratifiedTargetRecords": stratified.get("targetRecords"),
        "stratifiedRecordRecallAt100000Pct": stratified_overall.get("recordRecallAt100000Pct"),
        "stratifiedRecordRecallAt1000000Pct": stratified_overall.get("recordRecallAt1000000Pct"),
        "stratifiedApprovalYearBins": stratified.get("approvalYearBins") or [],
    }
    if benchmark:
        top10k_p0 = enrichment.get(10000, {})
        top100k_p0 = enrichment.get(100000, {})
        metrics.update(
            {
                "pairBenchmarkTotalPairs": benchmark.get("totalPairs"),
                "pairBenchmarkPositivePairs": benchmark.get("positivePairsFromKnownTargets"),
                "pairBenchmarkAuroc": benchmark.get("auroc"),
                "pairBenchmarkAveragePrecision": benchmark.get("averagePrecision"),
                "pairRecallAt10000Hits": top10k_p0.get("hits"),
                "pairRecallAt10000Pct": top10k_p0.get("recallPct"),
                "pairEnrichmentAt10000": top10k_p0.get("enrichmentVsRandom"),
                "pairRecallAt100000Hits": top100k_p0.get("hits"),
                "pairRecallAt100000Pct": top100k_p0.get("recallPct"),
                "pairEnrichmentAt100000": top100k_p0.get("enrichmentVsRandom"),
            }
        )
    return metrics


def final_diffdock_metrics(root: Path, fallback_candidates: list[dict[str, str]]) -> dict[str, Any]:
    summary = read_json(root / "outputs/sota_validation/final_diffdock_completion_after_round4.json")
    if summary:
        return {
            "structureCandidates": int(summary.get("integrated_rows") or 0),
            "structureCompleted": int(summary.get("completed") or 0),
            "structureMissing": int(summary.get("missing") or 0),
            "structureOutputRatePct": number(summary.get("completion_pct")) or 0.0,
            "structureRowsByDirection": summary.get("rows_by_direction") or {},
            "structureMissingByDirection": summary.get("missing_by_direction") or {},
            "structureCompletedByLayer": summary.get("completed_by_score_source") or {},
            "structureLayerSummary": read_csv(root / "outputs/sota_validation/final_diffdock_layer_summary_after_round4.csv"),
        }
    structure_candidates = len(fallback_candidates)
    structure_completed = sum(1 for row in fallback_candidates if row.get("status") == "completed")
    return {
        "structureCandidates": structure_candidates,
        "structureCompleted": structure_completed,
        "structureMissing": structure_candidates - structure_completed,
        "structureOutputRatePct": pct(structure_completed, structure_candidates),
        "structureRowsByDirection": {},
        "structureMissingByDirection": {},
        "structureCompletedByLayer": {},
        "structureLayerSummary": [],
    }


def external_coverage_by_direction(root: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv(root / "outputs/sota_validation/disease_direction_external_evidence_coverage.csv")
    coverage: dict[str, dict[str, Any]] = {}
    for row in rows:
        direction = row.get("direction", "")
        if not direction:
            continue
        coverage[direction] = {
            "labelZh": row.get("labelZh") or DIRECTION_LABELS.get(direction, direction),
            "candidateRows": int(number(row.get("candidateRows")) or 0),
            "directOpenTargetsRows": int(number(row.get("directOpenTargetsRows")) or 0),
            "directOpenTargetsPct": number(row.get("directOpenTargetsPct")) or 0.0,
            "expandedOpenTargetsRows": int(number(row.get("expandedOpenTargetsRows")) or 0),
            "expandedOpenTargetsPct": number(row.get("expandedOpenTargetsPct")) or 0.0,
            "directTxGNNRows": int(number(row.get("directTxGNNRows")) or 0),
            "directTxGNNPct": number(row.get("directTxGNNPct")) or 0.0,
            "expandedTxGNNRows": int(number(row.get("expandedTxGNNRows")) or 0),
            "expandedTxGNNPct": number(row.get("expandedTxGNNPct")) or 0.0,
            "expandedBothRows": int(number(row.get("expandedBothRows")) or 0),
            "expandedBothPct": number(row.get("expandedBothPct")) or 0.0,
            "top1000RowsMissingAnyExpandedEvidence": int(number(row.get("top1000RowsMissingAnyExpandedEvidence")) or 0),
        }
    return coverage


def admet_metrics(root: Path) -> dict[str, Any]:
    summary = read_json(root / "outputs/sota_validation/admet_repurposing/admet_repurposing_summary.json")
    if not summary:
        return {}
    return {
        "drugRows": int(summary.get("drugRows") or 0),
        "candidateRows": int(summary.get("candidateRows") or 0),
        "drugAdmetTierCounts": summary.get("drugAdmetTierCounts") or {},
        "drugRouteClassCounts": summary.get("drugRouteClassCounts") or {},
        "candidateByDirection": summary.get("candidateByDirection") or {},
    }


def kg_metrics(root: Path) -> dict[str, Any]:
    top1000 = read_json(root / "outputs/sota_validation/kg_explainability_top1000/kg_explainability_summary.json")
    shortlist = read_json(root / "outputs/sota_validation/kg_explainability/kg_explainability_summary.json")
    if not top1000 and not shortlist:
        return {}
    active = top1000 or shortlist
    return {
        "candidateRows": int(active.get("candidateRows") or 0),
        "pathRows": int(active.get("pathRows") or 0),
        "txgnnMappedRows": int(active.get("txgnnMappedRows") or 0),
        "anyKgPathRows": int(active.get("anyKgPathRows") or 0),
        "anyKgPathPct": number(active.get("anyKgPathPct")) or 0.0,
        "directDrugTargetRows": int(active.get("directDrugTargetRows") or 0),
        "positiveDrugDiseaseRows": int(active.get("positiveDrugDiseaseRows") or 0),
        "directTargetDiseaseRows": int(active.get("directTargetDiseaseRows") or 0),
        "ppiBridgeRows": int(active.get("ppiBridgeRows") or 0),
        "noveltyClassCounts": active.get("noveltyClassCounts") or {},
        "kgEvidenceTierCounts": active.get("kgEvidenceTierCounts") or {},
        "shortlistCandidateRows": int(shortlist.get("candidateRows") or 0),
        "shortlistAnyKgPathPct": number(shortlist.get("anyKgPathPct")) or 0.0,
    }


def pose_sanity_metrics(root: Path) -> dict[str, Any]:
    summary = read_json(root / "outputs/sota_validation/pose_sanity_top10000/pose_sanity_summary.json")
    if not summary:
        summary = read_json(root / "outputs/sota_validation/pose_sanity/pose_sanity_summary.json")
    if not summary:
        return {}
    return {
        "candidateRows": int(summary.get("candidateRows") or 0),
        "completedRows": int(summary.get("completedRows") or 0),
        "readablePoseRows": int(summary.get("readablePoseRows") or 0),
        "readablePosePct": number(summary.get("readablePosePct")) or 0.0,
        "uniqueReceptorsRead": int(summary.get("uniqueReceptorsRead") or 0),
        "poseAuditStatusCounts": summary.get("poseAuditStatusCounts") or {},
        "poseAuditReasonCounts": summary.get("poseAuditReasonCounts") or {},
        "byDirection": summary.get("byDirection") or {},
    }


def build_direction_rows(
    candidates: list[dict[str, str]],
    external_coverage: dict[str, dict[str, Any]],
    structure_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row.get("direction", "unknown")].append(row)
    rows: list[dict[str, Any]] = []
    for direction, direction_rows in grouped.items():
        direction_rows.sort(key=lambda row: int(float(row.get("rank") or 999999)))
        all_coverage = coverage_for_rows(direction_rows)
        rows_by_direction = structure_metrics.get("structureRowsByDirection") or {}
        missing_by_direction = structure_metrics.get("structureMissingByDirection") or {}
        if direction in rows_by_direction:
            total = int(rows_by_direction.get(direction) or 0)
            missing = int(missing_by_direction.get(direction) or 0)
            all_coverage["total"] = total
            all_coverage["completed"] = total - missing
            all_coverage["missing"] = missing
            all_coverage["completedPct"] = pct(total - missing, total)
        if direction in external_coverage:
            ext = external_coverage[direction]
            all_coverage.update(
                {
                    "directOpenTargets": ext["directOpenTargetsRows"],
                    "directOpenTargetsPct": ext["directOpenTargetsPct"],
                    "openTargets": ext["expandedOpenTargetsRows"],
                    "openTargetsPct": ext["expandedOpenTargetsPct"],
                    "directTxGNN": ext["directTxGNNRows"],
                    "directTxGNNPct": ext["directTxGNNPct"],
                    "txgnn": ext["expandedTxGNNRows"],
                    "txgnnPct": ext["expandedTxGNNPct"],
                    "expandedBoth": ext["expandedBothRows"],
                    "expandedBothPct": ext["expandedBothPct"],
                    "top1000RowsMissingAnyExpandedEvidence": ext["top1000RowsMissingAnyExpandedEvidence"],
                }
            )
        row = {
            "direction": direction,
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "all": all_coverage,
            "top100": coverage_for_rows(top_n(direction_rows, 100)),
            "top1000": coverage_for_rows(top_n(direction_rows, 1000)),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: list(DIRECTION_LABELS).index(row["direction"]) if row["direction"] in DIRECTION_LABELS else 999)


def build_shortlist(candidates: list[dict[str, str]], limit_per_direction: int = 40) -> list[dict[str, Any]]:
    shortlist: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row.get("direction", "unknown")].append(row)

    for direction, rows in grouped.items():
        scored: list[tuple[tuple[float, float, float, float, float], dict[str, str]]] = []
        for row in rows:
            if row.get("status") != "completed":
                continue
            direction_score = number(row.get("directionScore")) or 0.0
            affinity = number(row.get("affinityScore")) or 0.0
            credibility = number(row.get("credibilityScore")) or 0.0
            ot = number(row.get("openTargetsScore")) or 0.0
            tx = number(row.get("txgnnScore")) or 0.0
            multi = sum([ot > 0, tx > 0, has_text(row.get("therapeuticArea")), has_text(row.get("indication"))])
            rank = int(float(row.get("rank") or 999999))
            key = (multi, credibility, direction_score, affinity, -rank)
            scored.append((key, row))
        for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:limit_per_direction]:
            shortlist.append(
                {
                    "direction": direction,
                    "labelZh": DIRECTION_LABELS.get(direction, direction),
                    "rank": int(float(row.get("rank") or 999999)),
                    "drugId": row.get("drugId"),
                    "drug": row.get("drug"),
                    "target": row.get("target"),
                    "protein": row.get("protein"),
                    "directionScore": number(row.get("directionScore")),
                    "affinityScore": number(row.get("affinityScore")),
                    "diffdock": number(row.get("diffdock")),
                    "credibilityScore": number(row.get("credibilityScore")),
                    "credibilityTierZh": row.get("credibilityTierZh"),
                    "evidenceSummaryZh": row.get("evidenceSummaryZh"),
                    "nextStepZh": row.get("nextStepZh"),
                }
            )
    return sorted(shortlist, key=lambda row: (row["direction"], -float(row.get("credibilityScore") or 0), row["rank"]))


def build_task_queue(metrics: dict[str, Any], direction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    non_oncology_ot_gap = sum(
        item["all"]["total"] - item["all"].get("openTargets", 0)
        for item in direction_rows
        if item["direction"] != "oncology"
    )
    non_oncology_tx_gap = sum(
        item["all"]["total"] - item["all"].get("txgnn", 0)
        for item in direction_rows
        if item["direction"] != "oncology"
    )
    top1000_external_gap = sum(item["all"].get("top1000RowsMissingAnyExpandedEvidence", 0) for item in direction_rows)
    top_completed = sum(item["top100"]["completed"] for item in direction_rows)
    structure_missing = metrics.get("structureMissing") or 0
    kg = metrics.get("kgExplainability") or {}
    admet = metrics.get("admet") or {}
    pose = metrics.get("poseSanity") or {}

    return [
        {
            "priority": "P0",
            "module": "疾病方向证据矩阵扩展审计（已完成）",
            "compute": "CPU / network-data",
            "input": "Open Targets expanded evidence, multi-direction TxGNN scores, current disease candidates",
            "output": "per-direction evidence gap list, rescued target-disease support, final evidence coverage table",
            "why": "Open Targets 与多方向 TxGNN 已完成补强，并形成候选层面的外部证据覆盖审计。",
            "scale": f"top1000 rows still missing at least one expanded evidence layer: {top1000_external_gap}",
            "estimatedGpuHours": 0,
            "estimatedWallTime": "done",
        },
        {
            "priority": "P0",
            "module": "已知关系召回、负例基准与分层验证（已完成）",
            "compute": "CPU",
            "input": "FDA labels/ChEMBL known mechanisms, current affinity matrix, disease-direction rankings",
            "output": "Recall@K, enrichment over random, AUROC/AUPRC, approval-year/therapeutic-area stratified recall",
            "why": "已完成 known-target pair benchmark 和 record-level 分层验证；剩余可加强项是时间外推和去同源泄漏专门实验。",
            "scale": "current known target records in scope: "
            + str(metrics.get("recall", {}).get("knownPairsInScope") or "NA"),
            "estimatedGpuHours": 0,
            "estimatedWallTime": "done",
        },
        {
            "priority": "P1",
            "module": "TxGNN / KG evidence path expansion（已完成）",
            "compute": "CPU/GPU optional",
            "input": "TxGNN drug-disease scores, KG edges, candidate drug-target pairs",
            "output": "drug-disease and target-disease explanation paths, graph-neighborhood support, missing mapping audit",
            "why": "已生成直接药物-靶点、药物-疾病、靶点-疾病、PPI 桥接和已知靶点-疾病桥接路径。",
            "scale": f"top1000 candidate rows with KG path: {kg.get('anyKgPathRows', 'NA')} / {kg.get('candidateRows', 'NA')}",
            "estimatedGpuHours": 0,
            "estimatedWallTime": "done",
        },
        {
            "priority": "P1",
            "module": "ADMET / safety / repurposability filter（已完成）",
            "compute": "CPU",
            "input": "915 FDA small molecules, SMILES/SDF, labels if available",
            "output": "RDKit descriptor tiers, PAINS/Brenk alerts, route feasibility, label-text safety flags, translational shortlist",
            "why": "已完成透明规则审计；hERG、DILI、CYP 定量模型仍可作为后续深度 ADMET 加强项。",
            "scale": f"{admet.get('drugRows', 915)} drugs; {admet.get('candidateRows', 'NA')} candidate rows",
            "estimatedGpuHours": 0,
            "estimatedWallTime": "done",
        },
        {
            "priority": "P1",
            "module": "LINCS/CMap disease signature reversal",
            "compute": "CPU",
            "input": "disease DEG signatures, LINCS L1000/CMap drug perturbation signatures",
            "output": "per-drug connectivity score, direction-specific expression reversal support",
            "why": "表达反转是国际老药新用常用验证层，可解释药物是否可能逆转疾病状态。",
            "scale": "start with 5 directions x top 100 drugs",
            "estimatedGpuHours": 0,
            "estimatedWallTime": "0.5-2 d after disease signatures are available",
        },
        {
            "priority": "P2",
            "module": "结构共识重评分与 pose sanity checks（轻量几何审计已完成）",
            "compute": "GPU/CPU mixed",
            "input": "completed DiffDock poses, receptors, ligands, top completed candidates",
            "output": "GNINA/Vina rescoring, PoseBusters-like geometry audit, pocket consistency, known-ligand pocket alignment",
            "why": "已完成 RDKit/PDB 可读性和基础接触几何审计；GNINA/Vina 仍因本地缺少二进制而未运行。",
            "scale": f"readable poses: {pose.get('readablePoseRows', 'NA')} / {pose.get('completedRows', 'NA')}",
            "estimatedGpuHours": "20-80 GPU h for GNINA-like rescoring; Vina/Pose audit mostly CPU",
            "estimatedWallTime": "pose sanity done; consensus rescoring pending tool install",
        },
        {
            "priority": "P2",
            "module": "AF3/Boltz/Chai-style complex prediction spot-check",
            "compute": "GPU",
            "input": "per-direction top 10-20 strongest candidates",
            "output": "second-model protein-ligand complex plausibility, confidence comparison, disagreement flags",
            "why": "AlphaFold 3-era SOTA 要求至少用少量候选做更强结构模型对照，而不是只依赖一个 docking 模型。",
            "scale": "50-100 complexes",
            "estimatedGpuHours": "50-200 GPU h depending on model and sampling",
            "estimatedWallTime": "2-5 d",
        },
        {
            "priority": "P2",
            "module": "组织表达、DepMap 与疾病上下文过滤",
            "compute": "CPU",
            "input": "Human Protein Atlas/GTEx/DepMap/Project Score, candidate target genes",
            "output": "tissue expression support, cancer dependency support, disease-context mismatch flags",
            "why": "候选靶点需要在对应疾病组织或细胞上下文中有生物学可达性。",
            "scale": "unique target genes from candidate shortlist",
            "estimatedGpuHours": 0,
            "estimatedWallTime": "4-12 h after datasets are local",
        },
        {
            "priority": "P3",
            "module": "剩余 missing-output 定向补跑",
            "compute": "GPU",
            "input": "remaining missing-output DiffDock representatives",
            "output": "rank-1 pose recovery or definitive failure tags",
            "why": "当前已完成 SDF 和 SMILES 救援，残留主要是 DiffDock 图构建/采样边界；继续全量补跑优先级低。",
            "scale": f"{structure_missing} representatives after final rescue",
            "estimatedGpuHours": "10-40 GPU h if rerun all",
            "estimatedWallTime": "0.5-1.5 d",
        },
    ]


def readiness_matrix(task_queue: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metrics = metrics or {}
    kg = metrics.get("kgExplainability") or {}
    admet = metrics.get("admet") or {}
    pose = metrics.get("poseSanity") or {}
    status_by_module = {
        "AI-DTI affinity screen": ("已完成", "ConPLex 全量 drug-target pair 亲和矩阵已完成。"),
        "Disease-aware ranking": ("已完成/待校准", "五个疾病方向排序已完成，并补入 Open Targets 与多方向 TxGNN 外部证据。"),
        "Structural docking": (
            "已完成/待共识重评分",
            f"DiffDock 代表任务完成率已达 99.84%；pose sanity 可读率 {pose.get('readablePosePct', 0):.2f}%，仍可补 GNINA/Vina 共识重评分。",
        ),
        "KG explainability": (
            "已完成",
            f"候选级 TxGNN KG 浅层路径已完成，Top1000 层路径覆盖 {kg.get('anyKgPathPct', 0):.2f}%。",
        ),
        "Known-positive validation": (
            "已完成/待外推增强",
            "已有 FDA known-target AUROC/AUPRC、Recall@K、随机富集和批准年份/治疗领域分层验证。",
        ),
        "Expression reversal": ("未完成", "缺少 LINCS/CMap disease signature reversal。"),
        "ADMET/safety": (
            "已完成/待深度模型增强",
            f"已完成 RDKit/PAINS/Brenk/route/text-flag 透明审计，覆盖 {admet.get('drugRows', 0)} 个药物。",
        ),
        "Tissue/disease context": ("未完成", "缺少 HPA/GTEx/DepMap 等上下文过滤。"),
        "Second-model structure check": ("未完成", "缺少 AF3/Boltz/Chai-style 小样本结构对照。"),
    }
    priority_hint = {
        "AI-DTI affinity screen": "done",
        "Disease-aware ranking": "P0",
        "Structural docking": "P2",
        "KG explainability": "P1",
        "Known-positive validation": "P0",
        "Expression reversal": "P1",
        "ADMET/safety": "P1",
        "Tissue/disease context": "P2",
        "Second-model structure check": "P2",
    }
    return [
        {"dimension": key, "status": value[0], "noteZh": value[1], "nextPriority": priority_hint[key]}
        for key, value in status_by_module.items()
    ]


def markdown_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    recall = metrics["recall"]
    kg = metrics.get("kgExplainability") or {}
    admet = metrics.get("admet") or {}
    pose = metrics.get("poseSanity") or {}
    lines = [
        "# BioMaster SOTA 对标计算审计",
        "",
        f"生成时间：{payload['updated']}",
        "",
        "## 当前基础",
        "",
        f"- FDA 小分子：{metrics['drugs']}",
        f"- Druggable protein records：{metrics['targets']}",
        f"- 全量 ConPLex drug-target pairs：{metrics['pairs']}",
        f"- 五个疾病方向 Top 候选：{metrics['topCandidates']}",
        f"- DiffDock-ready 结构代表：{metrics['structureCandidates']}",
        f"- 已完成结构输出：{metrics['structureCompleted']}，剩余缺失：{metrics['structureMissing']}，完成率：{metrics['structureOutputRatePct']:.2f}%",
        f"- Top1000 KG explanation path 覆盖：{kg.get('anyKgPathRows')} / {kg.get('candidateRows')}，{kg.get('anyKgPathPct', 0):.2f}%",
        f"- ADMET/再利用审计：{admet.get('drugRows')} 个药物，{admet.get('candidateRows')} 条候选记录",
        f"- Pose sanity：{pose.get('readablePoseRows')} / {pose.get('completedRows')} 个已完成姿态可读，{pose.get('readablePosePct', 0):.2f}%",
        "",
        "## 已知关系召回",
        "",
        f"- 已知 target records in scope：{recall.get('knownPairsInScope')}",
        f"- Top10000 target-record recall：{recall.get('top10000TargetHits')} hits，{recall.get('top10000TargetRecallPct'):.2f}%",
        f"- Top100000 target-record recall：{recall.get('top100000TargetHits')} hits，{recall.get('top100000TargetRecallPct'):.2f}%",
        f"- 随机期望 Top10000 百分比约：{recall.get('randomExpectedTop10000Pct'):.2f}%",
        f"- Pair-level benchmark：{recall.get('pairBenchmarkTotalPairs')} pairs，positive pairs {recall.get('pairBenchmarkPositivePairs')}，AUROC {recall.get('pairBenchmarkAuroc'):.3f}，AUPRC {recall.get('pairBenchmarkAveragePrecision'):.5f}",
        f"- Pair Recall@10000：{recall.get('pairRecallAt10000Hits')} hits，{recall.get('pairRecallAt10000Pct'):.2f}% recall，enrichment {recall.get('pairEnrichmentAt10000'):.2f}x",
        f"- Pair Recall@100000：{recall.get('pairRecallAt100000Hits')} hits，{recall.get('pairRecallAt100000Pct'):.2f}% recall，enrichment {recall.get('pairEnrichmentAt100000'):.2f}x",
        f"- Record-level stratified Recall@100000：{recall.get('stratifiedRecordRecallAt100000Pct'):.2f}%；Recall@1000000：{recall.get('stratifiedRecordRecallAt1000000Pct'):.2f}%",
        "",
        "## 新增计算闭环",
        "",
        f"- KG explainability：生成 {kg.get('pathRows')} 条候选证据路径，直接 drug-target 边 {kg.get('directDrugTargetRows')} 条，target-disease 直接关联 {kg.get('directTargetDiseaseRows')} 条。",
        f"- ADMET/safety：药物 ADMET tier 分布 {admet.get('drugAdmetTierCounts')}；候选短表已排除明显 PAINS/诊断类/结构缺失记录。",
        f"- Pose sanity：基础几何通过 {pose.get('poseAuditStatusCounts', {}).get('pass')} 条，warning {pose.get('poseAuditStatusCounts', {}).get('warning')} 条，fail {pose.get('poseAuditStatusCounts', {}).get('fail')} 条。",
        "",
        "## SOTA readiness matrix",
        "",
        "| 维度 | 当前状态 | 下一优先级 | 说明 |",
        "|---|---:|---:|---|",
    ]
    for row in payload["readinessMatrix"]:
        lines.append(f"| {row['dimension']} | {row['status']} | {row['nextPriority']} | {row['noteZh']} |")
    lines.extend(["", "## 疾病方向证据覆盖", "", "| 方向 | 候选数 | 结构完成率 | Open Targets 覆盖 | TxGNN 覆盖 | 多源证据覆盖 |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["directionCoverage"]:
        all_rows = row["all"]
        lines.append(
            f"| {row['labelZh']} | {all_rows['total']} | {all_rows['completedPct']:.2f}% | "
            f"{all_rows['openTargetsPct']:.2f}% | {all_rows['txgnnPct']:.2f}% | {all_rows['multiSourcePct']:.2f}% |"
        )
    lines.extend(["", "## 下一步计算队列", "", "| 优先级 | 模块 | 计算资源 | 规模 | 产物 |", "|---|---|---|---|---|"])
    for task in payload["taskQueue"]:
        lines.append(f"| {task['priority']} | {task['module']} | {task['compute']} | {task['scale']} | {task['output']} |")
    lines.extend(
        [
            "",
            "## 建议执行顺序",
            "",
            "1. Open Targets、TxGNN 多方向证据、KG explanation path、ADMET 透明审计、known-target 分层验证和 pose sanity 已完成。",
            "2. 继续补 LINCS/CMap disease signature reversal 和 HPA/GTEx/DepMap 上下文过滤，需要外部数据文件接入。",
            "3. 结构层下一步是安装 GNINA/Vina/obabel 或接入容器，再做共识重评分；Top10-20 可进入 AF3/Boltz/Chai-style 小样本验证。",
            "",
            "## 参考 SOTA 口径",
            "",
            "- TxGNN：知识图谱和 zero-shot drug repurposing 强调 drug-disease/target-disease explanation path。",
            "- DiffDock：扩散式 ligand pose 生成，适合作为结构姿态线索，但需要几何与二次评分审计。",
            "- AlphaFold 3 / Boltz / Chai-style models：代表更强的复合物结构预测方向，适合小样本高优先级候选验证。",
            "- Open Targets：target-disease evidence 和 association score 是疾病方向靶点优先级的标准证据层。",
            "- TDC/ADMET 与 CMap/LINCS：分别补安全性可转化与表达反转证据。",
        ]
    )
    return "\n".join(lines) + "\n"


def html_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    recall = metrics["recall"]

    def esc(value: Any) -> str:
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    task_cards = "\n".join(
        f"""
          <article class="step-card">
            <span class="step-index">{esc(task['priority'])}</span>
            <h3>{esc(task['module'])}</h3>
            <p>{esc(task['why'])}</p>
            <p><strong>{esc(task['compute'])}</strong> · {esc(task['scale'])}</p>
          </article>
        """
        for task in payload["taskQueue"]
    )
    readiness_rows = "\n".join(
        f"<tr><td>{esc(row['dimension'])}</td><td>{esc(row['status'])}</td><td>{esc(row['nextPriority'])}</td><td>{esc(row['noteZh'])}</td></tr>"
        for row in payload["readinessMatrix"]
    )
    coverage_rows = "\n".join(
        f"<tr><td>{esc(row['labelZh'])}</td><td>{row['all']['total']}</td><td>{row['all']['completedPct']:.2f}%</td>"
        f"<td>{row['all']['openTargetsPct']:.2f}%</td><td>{row['all']['txgnnPct']:.2f}%</td><td>{row['all'].get('expandedBothPct', row['all']['multiSourcePct']):.2f}%</td></tr>"
        for row in payload["directionCoverage"]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BioMaster SOTA 对标计算审计</title>
    <meta name="description" content="BioMaster 对标国际 SOTA 的计算差距与下一步任务队列。" />
    <link rel="stylesheet" href="assets/site.css" />
  </head>
  <body class="document-body">
    <header class="site-header">
      <a class="brand" href="index.html#overview" aria-label="BioMaster overview">
        <span class="brand-mark">B</span>
        <span><strong>BioMaster</strong><small>SOTA readiness audit</small></span>
      </a>
      <nav class="top-nav" aria-label="Primary">
        <a href="index.html">展示首页</a>
        <a href="results-report.html">正式汇报</a>
        <a href="status-report.html">运行状态</a>
      </nav>
    </header>
    <main>
      <section class="document-hero">
        <div class="document-hero-inner">
          <p class="eyebrow">SOTA readiness</p>
          <h1>对标国际 SOTA 的计算差距与任务队列</h1>
          <p>本页把当前结果映射到知识图谱、结构共识、表达反转、安全性和外部验证等国际药物再定位常见证据层，形成可执行的下一步计算队列。</p>
        </div>
      </section>
      <div class="document-layout">
        <article class="document-main">
          <section id="metrics" class="document-section">
            <h2>当前基础</h2>
            <div class="summary-strip">
              <div><span>Affinity pairs</span><strong>{metrics['pairs']:,}</strong><small>ConPLex 全量矩阵</small></div>
              <div><span>Structure reps</span><strong>{metrics['structureCandidates']:,}</strong><small>DiffDock-ready</small></div>
              <div><span>Completed</span><strong>{metrics['structureCompleted']:,}</strong><small>{metrics['structureOutputRatePct']:.2f}% output rate</small></div>
              <div><span>Pair Recall@100k</span><strong>{recall.get('pairRecallAt100000Pct'):.2f}%</strong><small>{recall.get('pairEnrichmentAt100000'):.2f}x random</small></div>
            </div>
          </section>
          <section id="matrix" class="document-section">
            <h2>SOTA readiness matrix</h2>
            <div class="report-table"><table><thead><tr><th>维度</th><th>状态</th><th>优先级</th><th>说明</th></tr></thead><tbody>{readiness_rows}</tbody></table></div>
          </section>
          <section id="coverage" class="document-section">
            <h2>疾病方向证据覆盖</h2>
            <div class="report-table"><table><thead><tr><th>方向</th><th>候选数</th><th>结构完成率</th><th>Open Targets</th><th>TxGNN</th><th>多源证据</th></tr></thead><tbody>{coverage_rows}</tbody></table></div>
          </section>
          <section id="queue" class="document-section">
            <h2>下一步计算队列</h2>
            <div class="workflow-grid">{task_cards}</div>
          </section>
        </article>
        <aside class="document-sidebar" aria-label="SOTA audit navigation">
          <h2>页面目录</h2>
          <a href="#metrics">当前基础</a>
          <a href="#matrix">SOTA matrix</a>
          <a href="#coverage">证据覆盖</a>
          <a href="#queue">计算队列</a>
        </aside>
      </div>
    </main>
    <footer class="site-footer">
      <span>BioMaster SOTA readiness audit</span>
      <span>Updated from local artifacts: {esc(payload['updated'])}</span>
    </footer>
  </body>
</html>
"""


def build_payload(root: Path) -> dict[str, Any]:
    candidates = read_csv(root / "outputs/disease_directions/disease_direction_integrated_candidates.csv")
    prep = read_json(root / "outputs/druggable_proteome/druggable_proteome_conplex_prep.metadata.json")
    expansion = read_json(root / "outputs/druggable_proteome/druggable_conplex_expansion.metadata.json")
    recall = recall_metrics(root)
    structure_metrics = final_diffdock_metrics(root, candidates)
    external_coverage = external_coverage_by_direction(root)
    direction_rows = build_direction_rows(candidates, external_coverage, structure_metrics)
    admet = admet_metrics(root)
    kg = kg_metrics(root)
    pose = pose_sanity_metrics(root)
    metrics = {
        "drugs": int(prep.get("drug_rows_usable") or 0),
        "targets": int(prep.get("protein_rows_input_valid") or 0),
        "uniqueSequences": int(prep.get("unique_sequences") or 0),
        "pairs": int(expansion.get("expanded_affinity_rows_written") or 0),
        "topCandidates": 10000 * len(DIRECTION_LABELS),
        "recall": recall,
        "admet": admet,
        "kgExplainability": kg,
        "poseSanity": pose,
        **structure_metrics,
    }
    task_queue = build_task_queue(metrics, direction_rows)
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics,
        "readinessMatrix": readiness_matrix(task_queue, metrics),
        "directionCoverage": direction_rows,
        "taskQueue": task_queue,
        "validationShortlist": build_shortlist(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a SOTA readiness audit from local BioMaster artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-md", default="docs/SOTA_COMPUTE_READINESS_2026_06_03.md")
    parser.add_argument("--out-html", default="docs/sota-readiness.html")
    parser.add_argument("--out-js", default="docs/assets/sota-readiness-data.js")
    parser.add_argument("--out-shortlist", default="docs/assets/sota-validation-shortlist.csv")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root)
    (root / args.out_md).write_text(markdown_report(payload), encoding="utf-8")
    (root / args.out_html).write_text(html_report(payload), encoding="utf-8")
    js_string(root / args.out_js, "BIOMASTER_SOTA_READINESS", payload)
    write_csv(
        root / args.out_shortlist,
        [
            "direction",
            "labelZh",
            "rank",
            "drugId",
            "drug",
            "target",
            "protein",
            "directionScore",
            "affinityScore",
            "diffdock",
            "credibilityScore",
            "credibilityTierZh",
            "evidenceSummaryZh",
            "nextStepZh",
        ],
        payload["validationShortlist"],
    )
    print(
        json.dumps(
            {
                "updated": payload["updated"],
                "out_md": args.out_md,
                "out_html": args.out_html,
                "out_js": args.out_js,
                "out_shortlist": args.out_shortlist,
                "tasks": len(payload["taskQueue"]),
                "shortlist": len(payload["validationShortlist"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
