from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


DIRECTIONS = [
    {
        "key": "oncology",
        "label": "Oncology",
        "label_zh": "肿瘤",
        "summary_zh": "用于观察已批准药物与肿瘤相关靶点、癌症证据和结构可行性之间的交集。",
    },
    {
        "key": "infectious_disease",
        "label": "Infectious disease",
        "label_zh": "感染性疾病",
        "summary_zh": "用于筛选抗感染药物、免疫相关受体和感染疾病证据之间的潜在配对。",
    },
    {
        "key": "cardiovascular",
        "label": "Cardiovascular",
        "label_zh": "心血管",
        "summary_zh": "用于聚焦血压、血管张力、心血管调控和相关代谢/受体通路候选。",
    },
    {
        "key": "neurology_psychiatry",
        "label": "Neurology / psychiatry",
        "label_zh": "神经/精神",
        "summary_zh": "用于评估神经递质受体、阿片受体、胆碱能和精神神经疾病相关候选。",
    },
    {
        "key": "immunology_inflammation",
        "label": "Immunology / inflammation",
        "label_zh": "免疫/炎症",
        "summary_zh": "用于寻找抗炎、免疫调节、过敏和炎症疾病方向的药物-蛋白候选。",
    },
]

EVIDENCE_TRANSLATIONS = {
    "FDA therapeutic area match": "FDA 药物治疗领域与该疾病方向一致",
    "FDA indication/target text match": "FDA 适应症或靶点文本与该疾病方向存在直接匹配",
    "protein ICD-11 disease-class match": "蛋白的 ICD-11 疾病类别与该方向相关",
    "Open Targets cancer/neoplasm association": "Open Targets 支持该靶点与肿瘤/肿瘤相关疾病存在关联",
    "TxGNN cancer drug-disease signal": "TxGNN 药物-疾病图谱给出癌症方向信号",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: str | float | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def number_text(value: str | float | None, digits: int = 3) -> str:
    parsed = number(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.{digits}f}"


def load_scores(score_dir: Path, score_source: str = "primary_full_run") -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    for path in sorted(score_dir.glob("*.scores.csv")):
        for row in read_csv(path):
            row = dict(row)
            row["score_source"] = score_source
            row["score_file"] = str(path)
            scores[row["pair_id"]] = row
    return scores


def load_extra_score_layer(root: Path, score_dirs: list[str] | None) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    scores: dict[str, dict[str, str]] = {}
    raw_rows = 0
    loaded_dirs: list[str] = []
    for value in score_dirs or []:
        score_dir = Path(value)
        if not score_dir.is_absolute():
            score_dir = root / score_dir
        if not score_dir.exists():
            continue
        loaded_dirs.append(str(score_dir))
        for path in sorted(score_dir.glob("*.scores.csv")):
            for row in read_csv(path):
                raw_rows += 1
                row = dict(row)
                row["score_source"] = "priority_rerun"
                row["score_file"] = str(path)
                existing = scores.get(row["pair_id"])
                if not existing or existing.get("status") != "completed" or row.get("status") == "completed":
                    scores[row["pair_id"]] = row
    completed = sum(1 for row in scores.values() if row.get("status") == "completed")
    missing = sum(1 for row in scores.values() if row.get("status") != "completed")
    return scores, {
        "score_dirs": loaded_dirs,
        "raw_rows": raw_rows,
        "unique_pairs": len(scores),
        "completed": completed,
        "missing": missing,
    }


def merge_score_layers(
    primary_scores: dict[str, dict[str, str]],
    extra_scores: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], set[str]]:
    merged = {pair_id: dict(score) for pair_id, score in primary_scores.items()}
    recovered_pair_ids: set[str] = set()
    for pair_id, extra_score in extra_scores.items():
        if extra_score.get("status") != "completed":
            continue
        primary_status = primary_scores.get(pair_id, {}).get("status")
        if primary_status == "completed":
            continue
        merged[pair_id] = dict(extra_score)
        recovered_pair_ids.add(pair_id)
    return merged, recovered_pair_ids


def score_source_label(score_source: str | None) -> str:
    if score_source == "priority_rerun":
        return "missing-output priority rerun"
    return "primary full DiffDock run"


def score_source_label_zh(score_source: str | None) -> str:
    if score_source == "priority_rerun":
        return "缺失输出优先补跑"
    return "原始全量 DiffDock"


def translate_evidence(summary: str) -> list[str]:
    parts = [part.strip() for part in summary.split(";") if part.strip()]
    translated: list[str] = []
    for part in parts:
        translated.append(EVIDENCE_TRANSLATIONS.get(part, part))
    return translated


def classify_candidate(row: dict[str, str], status: str) -> tuple[str, str]:
    evidence = row.get("direction_evidence_summary", "")
    rank = int(float(row.get("direction_rank") or 999999))
    if status != "completed" and rank <= 50:
        return "rescue", "高优先级结构审计"
    if "FDA therapeutic area match" in evidence and ("Open Targets" in evidence or "TxGNN" in evidence):
        return "multi_source", "多源疾病证据一致"
    if "FDA therapeutic area match" in evidence:
        return "therapeutic_area", "治疗领域一致"
    if "protein ICD-11 disease-class match" in evidence:
        return "protein_context", "蛋白疾病类别支持"
    return "exploratory", "探索性候选"


def evidence_path(row: dict[str, str], score: dict[str, str], direction_zh: str) -> list[str]:
    evidence = row.get("direction_evidence_summary", "")
    path: list[str] = []
    if "FDA therapeutic area match" in evidence or row.get("therapeutic_area"):
        path.append("药物治疗领域/适应症")
    if "protein ICD-11 disease-class match" in evidence:
        path.append("蛋白疾病类别")
    if number(row.get("opentargets_direction_score")):
        path.append("Open Targets 靶点-疾病证据")
    if number(row.get("txgnn_direction_score")):
        path.append("TxGNN 药物-疾病图谱")
    if number(row.get("affinity_score")) is not None:
        path.append("ConPLex 药物-蛋白亲和预测")
    if score.get("status") == "completed":
        path.append("DiffDock 结构姿态")
    elif score.get("status") == "missing_output":
        path.append("DiffDock 缺失输出审计")
    return [row.get("drug_name", "Drug"), row.get("gene_name", "Target"), direction_zh, *path]


def credibility_profile(row: dict[str, str], score: dict[str, str], direction_zh: str) -> dict[str, Any]:
    status = score.get("status") or "not_yet_run"
    evidence = row.get("direction_evidence_summary", "")
    direction_score = number(row.get("direction_score")) or 0.0
    affinity_score = number(row.get("affinity_score")) or 0.0
    opentargets_score = number(row.get("opentargets_direction_score")) or 0.0
    txgnn_score = number(row.get("txgnn_direction_score")) or 0.0
    represented = int(float(row.get("represented_pair_count") or 1))

    evidence_points = 0
    if "FDA therapeutic area match" in evidence:
        evidence_points += 8
    if "protein ICD-11 disease-class match" in evidence:
        evidence_points += 6
    if opentargets_score >= 0.5:
        evidence_points += 8
    elif opentargets_score > 0:
        evidence_points += 4
    if txgnn_score >= 0.5:
        evidence_points += 8
    elif txgnn_score > 0:
        evidence_points += 4

    score_value = 0
    score_value += 25 if direction_score >= 0.95 else 20 if direction_score >= 0.90 else 14 if direction_score >= 0.80 else 8
    score_value += 25 if affinity_score >= 0.95 else 20 if affinity_score >= 0.90 else 14 if affinity_score >= 0.80 else 8
    score_value += min(evidence_points, 24)
    score_value += 16 if status == "completed" else 0
    score_value += 5 if represented >= 10 else 3 if represented >= 3 else 0
    score_value = min(score_value, 100)

    multi_source = ("FDA therapeutic area match" in evidence) and (opentargets_score >= 0.5 or txgnn_score >= 0.5)
    if status != "completed":
        tier = "D"
        tier_zh = "D｜结构补跑优先"
        posture_zh = "高分但结构证据缺失"
        next_step = "优先检查受体结构、配体 SDF 和 DiffDock 图构建日志，完成定向补跑后再进入机制判断。"
    elif multi_source and direction_score >= 0.93 and affinity_score >= 0.90:
        tier = "A"
        tier_zh = "A｜多源强证据"
        posture_zh = "老药新用强候选或已知正控召回"
        next_step = "优先核查 FDA label、DrugBank/ChEMBL 已知靶点、ClinicalTrials 和 PubMed，区分已知召回与可转化新用途。"
    elif direction_score >= 0.88 and affinity_score >= 0.88 and evidence_points >= 8:
        tier = "B"
        tier_zh = "B｜机制邻近优先"
        posture_zh = "疾病方向与药物-蛋白互作一致"
        next_step = "补充通路、组织表达、疾病亚型和二次结构评分，筛选 20-50 个专家审阅候选。"
    elif affinity_score >= 0.82 or direction_score >= 0.82:
        tier = "C"
        tier_zh = "C｜探索性再定位"
        posture_zh = "具备模型信号但疾病证据仍需补强"
        next_step = "先引入 CMap/LINCS 转录组反转、ADMET/安全性和反证检索，再决定是否进入实验短名单。"
    else:
        tier = "D"
        tier_zh = "D｜低优先级审阅"
        posture_zh = "当前证据不足"
        next_step = "保留为背景候选，除非后续文献、转录组或临床证据显著增强。"

    path = evidence_path(row, score, direction_zh)
    gates = [
        "已知适应症/已知靶点核查",
        "PubMed 与 ClinicalTrials 文献证据",
        "CMap/LINCS 疾病签名反转",
        "ADMET、禁忌证和药物相互作用审阅",
    ]
    if status == "completed":
        gates.append("二次 docking / pocket 审计")
    else:
        gates.insert(0, "结构缺失补跑")

    return {
        "credibilityScore": round(score_value, 1),
        "credibilityTier": tier,
        "credibilityTierZh": tier_zh,
        "repurposingPostureZh": posture_zh,
        "evidencePathZh": " → ".join(path),
        "nextStepZh": next_step,
        "validationGatesZh": "；".join(gates),
    }


def build_rationale(row: dict[str, str], score: dict[str, str], direction_zh: str) -> str:
    status = score.get("status") or "not_yet_run"
    confidence = number(score.get("diffdock_confidence"))
    source_zh = score_source_label_zh(score.get("score_source"))
    pieces = [
        (
            f"该候选来自{direction_zh}方向，疾病方向分数为 {number_text(row.get('direction_score'))}，"
            f"ConPLex 亲和预测为 {number_text(row.get('affinity_score'))}。"
            f"这表示模型在疾病证据和药物-蛋白相互作用两个层面都将 {row.get('drug_name', '')} - "
            f"{row.get('gene_name', '')} 排在较高优先级。"
        )
    ]

    translated = translate_evidence(row.get("direction_evidence_summary", ""))
    if translated:
        pieces.append("支持证据包括：" + "；".join(translated) + "。")

    therapeutic_area = row.get("therapeutic_area", "")
    indication = row.get("indication", "")
    if therapeutic_area or indication:
        context = "；".join(item for item in [therapeutic_area, indication] if item)
        pieces.append(f"药物记录中的治疗领域/适应症信息为：{context}。")

    if status == "completed" and confidence is not None:
        pieces.append(
            f"DiffDock 已通过{source_zh}产生可审阅的 rank-1 结合姿态，confidence 为 {confidence:.2f}；"
            "该值用于判断结构构象是否值得专家查看，不能直接等同于结合自由能或药效强度。"
        )
    elif status == "missing_output":
        pieces.append(
            "本轮 DiffDock 未产出可解析的 rank-1 confidence SDF，属于结构计算缺失输出；"
            "这不是药效否定证据，应作为受体/配体准备或参数补跑的优先审计对象。"
        )
    else:
        pieces.append("该候选尚未获得结构层面的可审阅输出，当前主要依据疾病证据和亲和预测排序。")

    represented = row.get("represented_pair_count", "")
    if represented and represented not in ("", "1"):
        pieces.append(f"该结构代表可回填到 {represented} 条相同药物和相同蛋白序列的 UniProt 记录。")

    return "".join(pieces)


def build_candidate(
    row: dict[str, str],
    score: dict[str, str],
    direction_key: str,
    direction_label: str,
    direction_zh: str,
) -> dict[str, Any]:
    status = score.get("status") or "not_yet_run"
    confidence = number(score.get("diffdock_confidence"))
    category, category_zh = classify_candidate(row, status)
    direction_rank = int(float(row.get("direction_rank") or 999999))
    credibility = credibility_profile(row, score, direction_zh)
    return {
        "rank": direction_rank,
        "direction": direction_key,
        "directionLabel": direction_label,
        "directionLabelZh": direction_zh,
        "pairId": row.get("pair_id", ""),
        "drugId": row.get("drug_id", ""),
        "drug": row.get("drug_name", ""),
        "target": row.get("gene_name", ""),
        "protein": row.get("protein_id", ""),
        "proteinName": row.get("protein_name", ""),
        "directionScore": number(row.get("direction_score")),
        "affinityScore": number(row.get("affinity_score")),
        "affinityComponent": number(row.get("affinity_component")),
        "openTargetsScore": number(row.get("opentargets_direction_score")),
        "txgnnScore": number(row.get("txgnn_direction_score")),
        "drugDirectionScore": number(row.get("drug_direction_score")),
        "proteinDirectionScore": number(row.get("protein_direction_score")),
        "representedPairCount": int(float(row.get("represented_pair_count") or 1)),
        "representedProteins": row.get("represented_protein_ids", ""),
        "diffdock": confidence,
        "status": status,
        "scoreSource": score.get("score_source", ""),
        "scoreSourceLabel": score_source_label(score.get("score_source")),
        "scoreSourceLabelZh": score_source_label_zh(score.get("score_source")),
        "scoreFile": score.get("score_file", ""),
        "category": category,
        "categoryZh": category_zh,
        **credibility,
        "evidenceSummary": row.get("direction_evidence_summary", ""),
        "evidenceSummaryZh": "；".join(translate_evidence(row.get("direction_evidence_summary", ""))),
        "rationaleZh": build_rationale(row, score, direction_zh),
        "therapeuticArea": row.get("therapeutic_area", ""),
        "indication": row.get("indication", ""),
        "receptorStatus": row.get("diffdock_receptor_status", ""),
        "selectionReason": row.get("representative_selection_reason", ""),
        "confidenceSdfPath": score.get("confidence_sdf_path", ""),
        "rank1SdfPath": score.get("rank1_sdf_path", ""),
        "receptorPdbPath": row.get("diffdock_receptor_pdb_path", ""),
    }


def score_sort_key(candidate: dict[str, Any]) -> tuple[int, float, float, float]:
    completed = 1 if candidate["status"] == "completed" else 0
    return (
        completed,
        float(candidate.get("directionScore") or 0),
        float(candidate.get("affinityScore") or 0),
        float(candidate.get("diffdock") or -9999),
    )


def write_integrated_outputs(
    root: Path,
    candidates: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    out_csv = root / "outputs/disease_directions/disease_direction_integrated_candidates.csv"
    out_summary = root / "outputs/disease_directions/disease_direction_summary.csv"
    out_json = root / "outputs/disease_directions/disease_direction_dashboard_summary.json"

    candidate_fields = [
        "direction",
        "directionLabel",
        "directionLabelZh",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "directionScore",
        "affinityScore",
        "affinityComponent",
        "openTargetsScore",
        "txgnnScore",
        "diffdock",
        "status",
        "scoreSource",
        "scoreSourceLabelZh",
        "scoreFile",
        "categoryZh",
        "credibilityScore",
        "credibilityTierZh",
        "repurposingPostureZh",
        "evidencePathZh",
        "nextStepZh",
        "validationGatesZh",
        "evidenceSummaryZh",
        "rationaleZh",
        "therapeuticArea",
        "indication",
        "representedPairCount",
        "receptorStatus",
        "confidenceSdfPath",
        "receptorPdbPath",
    ]
    summary_fields = [
        "direction",
        "label",
        "labelZh",
        "preparedPairs",
        "chunks",
        "scoreChunks",
        "scoredRows",
        "completed",
        "missing",
        "primaryCompleted",
        "primaryMissing",
        "rerunRecovered",
        "successRatePct",
        "medianDiffDock",
    ]
    write_csv(out_csv, candidate_fields, candidates)
    write_csv(out_summary, summary_fields, summaries)
    out_json.write_text(json.dumps({"metadata": metadata, "directions": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def counter_rows(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    return [{"label": label or "NA", "value": value} for label, value in counter.most_common(limit)]


def build_payload(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    all_candidates: list[dict[str, Any]] = []
    display_candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    top_targets: Counter[str] = Counter()
    top_drugs: Counter[str] = Counter()
    structural_status: Counter[str] = Counter()
    category_status: Counter[str] = Counter()
    credibility_status: Counter[str] = Counter()
    validation_status: Counter[str] = Counter()
    score_source_status: Counter[str] = Counter()
    extra_scores, extra_metadata = load_extra_score_layer(root, args.extra_score_dir)
    total_primary_completed = 0
    total_primary_missing = 0
    total_rerun_recovered = 0

    for config in DIRECTIONS:
        direction_root = root / "outputs/disease_directions" / config["key"]
        ready_path = direction_root / "top10000_diffdock_ready.csv"
        metadata = load_json(direction_root / "diffdock_run/diffdock_full_run.metadata.json")
        primary_scores = load_scores(direction_root / "diffdock_run/scores")
        scores, recovered_pair_ids = merge_score_layers(primary_scores, extra_scores)
        ready_rows = read_csv(ready_path)
        scored_pair_ids = set(primary_scores)
        scored_ready_rows = [row for row in ready_rows if row.get("pair_id") in scored_pair_ids]
        candidates = [
            build_candidate(row, scores.get(row["pair_id"], {}), config["key"], config["label"], config["label_zh"])
            for row in scored_ready_rows
        ]
        primary_completed = sum(1 for pair_id in scored_pair_ids if primary_scores.get(pair_id, {}).get("status") == "completed")
        primary_missing = len(scored_pair_ids) - primary_completed
        rerun_recovered = sum(1 for row in scored_ready_rows if row.get("pair_id") in recovered_pair_ids)
        confidences = [candidate["diffdock"] for candidate in candidates if candidate["status"] == "completed" and candidate["diffdock"] is not None]
        completed = sum(1 for candidate in candidates if candidate["status"] == "completed")
        missing = sum(1 for candidate in candidates if candidate["status"] != "completed")
        total_primary_completed += primary_completed
        total_primary_missing += primary_missing
        total_rerun_recovered += rerun_recovered
        summary = {
            "direction": config["key"],
            "label": config["label"],
            "labelZh": config["label_zh"],
            "summaryZh": config["summary_zh"],
            "preparedPairs": int(metadata.get("structure_ready_rows") or len(ready_rows)),
            "chunks": int(metadata.get("input_chunks") or 0),
            "scoreChunks": len(list((direction_root / "diffdock_run/scores").glob("*.scores.csv"))),
            "scoredRows": len(candidates),
            "completed": completed,
            "missing": missing,
            "primaryCompleted": primary_completed,
            "primaryMissing": primary_missing,
            "rerunRecovered": rerun_recovered,
            "successRatePct": completed / len(candidates) * 100 if candidates else 0.0,
            "medianDiffDock": statistics.median(confidences) if confidences else None,
            "topCompleted": sorted([c for c in candidates if c["status"] == "completed"], key=lambda c: c["rank"])[:5],
        }
        summaries.append(summary)
        all_candidates.extend(candidates)
        display_candidates.extend(sorted(candidates, key=lambda item: item["rank"])[: args.candidates_per_direction])
        top_targets.update(candidate["target"] for candidate in candidates[:200])
        top_drugs.update(candidate["drug"] for candidate in candidates[:200])
        structural_status.update(candidate["status"] for candidate in candidates)
        category_status.update(candidate["categoryZh"] for candidate in candidates[:200])
        credibility_status.update(candidate["credibilityTierZh"] for candidate in candidates)
        validation_status.update(candidate["repurposingPostureZh"] for candidate in candidates[:200])
        score_source_status.update(candidate["scoreSourceLabelZh"] for candidate in candidates if candidate["status"] == "completed")

    display_candidates.sort(key=lambda item: (item["direction"], item["rank"]))
    total_prepared = sum(item["preparedPairs"] for item in summaries)
    total_completed = sum(item["completed"] for item in summaries)
    total_missing = sum(item["missing"] for item in summaries)
    total_chunks = sum(item["chunks"] for item in summaries)
    total_score_chunks = sum(item["scoreChunks"] for item in summaries)
    prep = load_json(root / "outputs/druggable_proteome/druggable_proteome_conplex_prep.metadata.json")
    expansion = load_json(root / "outputs/druggable_proteome/druggable_conplex_expansion.metadata.json")

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_candidates": len(all_candidates),
        "display_candidates": len(display_candidates),
        "directions": len(DIRECTIONS),
        "extra_score_layer": extra_metadata,
        "rerun_recovered": total_rerun_recovered,
    }
    write_integrated_outputs(root, all_candidates, summaries, metadata)

    return {
        "updated": metadata["created_utc"],
        "mode": "disease_direction_druggable_proteome",
        "labels": {
            "primaryScore": "Disease score",
            "primaryScoreLong": "Disease-aware evidence score",
            "candidateScope": "Disease-direction Top10000 DiffDock-ready representatives",
        },
        "metrics": {
            "drugs": int(prep.get("drug_rows_usable") or 915),
            "targets": int(prep.get("protein_rows_input_valid") or 5306),
            "uniqueSequences": int(prep.get("unique_sequences") or 891),
            "pairs": int(expansion.get("expanded_affinity_rows_written") or 4854990),
            "topCandidates": 10000 * len(DIRECTIONS),
            "structureCandidates": total_prepared,
            "structureCompleted": total_completed,
            "structureMissing": total_missing,
            "top1000Completed": total_completed,
            "top1000Missing": total_missing,
            "fullScoreFiles": total_score_chunks,
            "fullJobsTotal": total_chunks,
            "fullRowsScored": total_prepared,
            "fullRowsTotal": total_prepared,
            "fullRowProgressPct": 100.0 if total_prepared else 0.0,
            "fullCompletedOutputs": total_completed,
            "fullMissingOutputs": total_missing,
            "fullOutputRatePct": total_completed / total_prepared * 100 if total_prepared else 0.0,
            "primaryCompletedOutputs": total_primary_completed,
            "primaryMissingOutputs": total_primary_missing,
            "rerunRecoveredOutputs": total_rerun_recovered,
            "rerunScoreRows": int(extra_metadata.get("raw_rows") or 0),
            "rerunUniquePairs": int(extra_metadata.get("unique_pairs") or 0),
            "zeroCompletedChunks": 0,
            "diseaseDirections": len(DIRECTIONS),
        },
        "diseaseDirections": summaries,
        "charts": {
            "evidenceCoverage": [
                {"label": "ConPLex screened", "value": int(expansion.get("expanded_affinity_rows_written") or 4854990)},
                {"label": "Disease Top10000 sets", "value": 10000 * len(DIRECTIONS)},
                {"label": "DiffDock-ready pairs", "value": total_prepared},
                {"label": "Completed docking outputs", "value": total_completed},
            ],
            "topTargets": counter_rows(top_targets, 10),
            "topDrugs": counter_rows(top_drugs, 10),
            "structuralStatus": counter_rows(structural_status),
            "scoreSources": counter_rows(score_source_status),
            "credibilityTiers": counter_rows(credibility_status),
            "validationPostures": counter_rows(validation_status, 8),
            "receptorStatus": [{"label": "disease directions", "value": len(DIRECTIONS)}, {"label": "DiffDock chunks", "value": total_chunks}],
            "txgnnStatus": counter_rows(category_status, 8),
        },
        "candidates": display_candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build disease-direction dashboard assets.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/assets/dashboard-data.js")
    parser.add_argument("--candidates-per-direction", type=int, default=80)
    parser.add_argument(
        "--extra-score-dir",
        action="append",
        default=[
            "outputs/disease_directions/missing_output_priority_rerun/scores",
            "outputs/disease_directions/missing_output_priority_rerun_round2/scores",
        ],
        help="Additional DiffDock score directory. Completed rows here recover primary-run missing outputs.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = build_payload(root, args)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("window.BIOMASTER_DATA = " + json.dumps(payload, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "directions": len(payload["diseaseDirections"]),
                "candidates": len(payload["candidates"]),
                "completed": payload["metrics"]["fullCompletedOutputs"],
                "missing": payload["metrics"]["fullMissingOutputs"],
                "primary_completed": payload["metrics"]["primaryCompletedOutputs"],
                "rerun_recovered": payload["metrics"]["rerunRecoveredOutputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
