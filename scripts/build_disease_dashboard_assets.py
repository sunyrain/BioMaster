from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any


DOCS_ASSET_DIR = Path("docs/assets")

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


def rel_to_docs(path: Path) -> str:
    parts = path.resolve().parts
    docs_index = parts.index("docs")
    return Path(*parts[docs_index + 1 :]).as_posix()


def copy_asset(source_value: str, destination: Path) -> str | None:
    if not source_value:
        return None
    source = Path(source_value)
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return rel_to_docs(destination)


def load_scores(score_dir: Path) -> dict[str, dict[str, str]]:
    scores: dict[str, dict[str, str]] = {}
    for path in sorted(score_dir.glob("*.scores.csv")):
        for row in read_csv(path):
            scores[row["pair_id"]] = row
    return scores


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


def build_rationale(row: dict[str, str], score: dict[str, str], direction_zh: str) -> str:
    status = score.get("status") or "not_yet_run"
    confidence = number(score.get("diffdock_confidence"))
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
            f"DiffDock 已产生可审阅的 rank-1 结合姿态，confidence 为 {confidence:.2f}；"
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
        "category": category,
        "categoryZh": category_zh,
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
        "categoryZh",
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
        "successRatePct",
        "medianDiffDock",
    ]
    write_csv(out_csv, candidate_fields, candidates)
    write_csv(out_summary, summary_fields, summaries)
    out_json.write_text(json.dumps({"metadata": metadata, "directions": summaries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_structure_samples(root: Path, candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    per_direction: Counter[str] = Counter()
    structure_dir = root / DOCS_ASSET_DIR / "structures"

    for candidate in sorted(candidates, key=lambda item: (item["direction"], item["rank"])):
        if candidate["status"] != "completed":
            continue
        if per_direction[candidate["direction"]] >= 2:
            continue
        ligand_source = candidate.get("confidenceSdfPath") or candidate.get("rank1SdfPath")
        receptor_source = candidate.get("receptorPdbPath")
        safe_pair = candidate["pairId"].replace("__", "_")
        ligand_url = copy_asset(str(ligand_source), structure_dir / f"{candidate['direction']}_{safe_pair}_ligand.sdf")
        receptor_url = copy_asset(str(receptor_source), structure_dir / f"{candidate['direction']}_{safe_pair}_receptor.pdb")
        if not ligand_url or not receptor_url:
            continue
        samples.append(
            {
                "pairId": candidate["pairId"],
                "rank": candidate["rank"],
                "direction": candidate["direction"],
                "directionLabelZh": candidate["directionLabelZh"],
                "drug": candidate["drug"],
                "target": candidate["target"],
                "protein": candidate["protein"],
                "confidence": candidate["diffdock"],
                "consensus": candidate["directionScore"],
                "receptorStatus": candidate["receptorStatus"],
                "rationaleZh": candidate["rationaleZh"],
                "ligandUrl": ligand_url,
                "receptorUrl": receptor_url,
            }
        )
        per_direction[candidate["direction"]] += 1
        if len(samples) >= limit:
            break
    return samples


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

    for config in DIRECTIONS:
        direction_root = root / "outputs/disease_directions" / config["key"]
        ready_path = direction_root / "top10000_diffdock_ready.csv"
        metadata = load_json(direction_root / "diffdock_run/diffdock_full_run.metadata.json")
        scores = load_scores(direction_root / "diffdock_run/scores")
        ready_rows = read_csv(ready_path)
        scored_pair_ids = set(scores)
        scored_ready_rows = [row for row in ready_rows if row.get("pair_id") in scored_pair_ids]
        candidates = [
            build_candidate(row, scores.get(row["pair_id"], {}), config["key"], config["label"], config["label_zh"])
            for row in scored_ready_rows
        ]
        confidences = [candidate["diffdock"] for candidate in candidates if candidate["status"] == "completed" and candidate["diffdock"] is not None]
        completed = sum(1 for candidate in candidates if candidate["status"] == "completed")
        missing = sum(1 for candidate in candidates if candidate["status"] != "completed")
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
    }
    write_integrated_outputs(root, all_candidates, summaries, metadata)
    structure_samples = build_structure_samples(root, display_candidates, args.structure_samples)

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
            "receptorStatus": [{"label": "disease directions", "value": len(DIRECTIONS)}, {"label": "DiffDock chunks", "value": total_chunks}],
            "txgnnStatus": counter_rows(category_status, 8),
        },
        "candidates": display_candidates,
        "structureSamples": structure_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build disease-direction dashboard assets.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/assets/dashboard-data.js")
    parser.add_argument("--candidates-per-direction", type=int, default=80)
    parser.add_argument("--structure-samples", type=int, default=10)
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
                "structure_samples": len(payload["structureSamples"]),
                "completed": payload["metrics"]["fullCompletedOutputs"],
                "missing": payload["metrics"]["fullMissingOutputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
