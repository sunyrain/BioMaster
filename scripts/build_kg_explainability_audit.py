from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DIRECTION_LABELS = {
    "oncology": "肿瘤",
    "infectious_disease": "感染性疾病",
    "cardiovascular": "心血管",
    "neurology_psychiatry": "神经/精神",
    "immunology_inflammation": "免疫/炎症",
}

DIRECTION_DISEASE_TERMS = {
    "oncology": [
        "cancer",
        "tumor",
        "tumour",
        "neoplasm",
        "carcinoma",
        "sarcoma",
        "leukemia",
        "lymphoma",
        "myeloma",
        "melanoma",
    ],
    "infectious_disease": [
        "infection",
        "infectious",
        "tuberculosis",
        "hiv",
        "viral",
        "virus",
        "bacterial",
        "hepatitis",
        "fungal",
        "sepsis",
    ],
    "cardiovascular": [
        "cardiovascular",
        "hypertension",
        "hypertensive",
        "heart",
        "cardiac",
        "coronary",
        "vascular",
        "arterial",
        "stroke",
        "thrombosis",
        "arrhythmia",
    ],
    "neurology_psychiatry": [
        "parkinson",
        "alzheimer",
        "dementia",
        "schizophrenia",
        "depression",
        "epilepsy",
        "migraine",
        "neuro",
        "psychi",
        "restless legs",
    ],
    "immunology_inflammation": [
        "asthma",
        "immune",
        "autoimmune",
        "inflammation",
        "inflammatory",
        "arthritis",
        "psoriasis",
        "lupus",
        "colitis",
        "crohn",
        "allergy",
        "allergic",
    ],
}

DRUG_DISEASE_RELATIONS = {"indication", "off-label use", "contraindication"}
KG_RELATIONS_OF_INTEREST = DRUG_DISEASE_RELATIONS | {
    "drug_protein",
    "disease_protein",
    "protein_protein",
}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def norm_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def norm_gene(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().upper())


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


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "nan":
            return text
    return ""


def relation_polarity(relation: str) -> str:
    if relation == "contraindication":
        return "caution"
    if relation in {"indication", "off-label use"}:
        return "positive_context"
    return "support"


def disease_matches_direction(
    direction: str,
    disease_id: str,
    disease_name: str,
    disease_ids_by_direction: dict[str, set[str]],
    disease_names_by_direction: dict[str, set[str]],
) -> bool:
    disease_id_norm = norm_id(disease_id)
    disease_name_norm = norm_text(disease_name)
    if disease_id_norm and disease_id_norm in disease_ids_by_direction.get(direction, set()):
        return True
    if disease_name_norm and disease_name_norm in disease_names_by_direction.get(direction, set()):
        return True
    return any(term in disease_name_norm for term in DIRECTION_DISEASE_TERMS.get(direction, []))


def matching_directions(
    disease_id: str,
    disease_name: str,
    disease_ids_by_direction: dict[str, set[str]],
    disease_names_by_direction: dict[str, set[str]],
) -> list[str]:
    return [
        direction
        for direction in DIRECTION_LABELS
        if disease_matches_direction(direction, disease_id, disease_name, disease_ids_by_direction, disease_names_by_direction)
    ]


def load_candidate_context(root: Path, candidates_path: Path, integrated_path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(candidates_path, dtype=str).fillna("")
    if integrated_path.exists():
        integrated_cols = [
            "direction",
            "pairId",
            "openTargetsScore",
            "txgnnScore",
            "scoreSource",
            "categoryZh",
            "credibilityTierZh",
            "therapeuticArea",
            "indication",
            "evidenceSummaryZh",
            "rationaleZh",
        ]
        integrated = pd.read_csv(integrated_path, dtype=str, usecols=lambda col: col in integrated_cols).fillna("")
        candidates = candidates.merge(
            integrated,
            on=["direction", "pairId"],
            how="left",
            suffixes=("", "_integrated"),
        )
    else:
        for col in ["openTargetsScore", "txgnnScore", "scoreSource", "categoryZh", "credibilityTierZh", "therapeuticArea", "indication"]:
            if col not in candidates:
                candidates[col] = ""
    candidates["targetGeneNorm"] = candidates["target"].map(norm_gene)
    candidates["candidateIndex"] = range(1, len(candidates) + 1)
    return candidates


def load_crosswalk(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        drug_id = row.get("drug_id", "")
        if not drug_id or drug_id in lookup:
            continue
        lookup[drug_id] = {
            "txgnnDrugbankId": row.get("txgnn_drugbank_id", ""),
            "txgnnDrugName": row.get("txgnn_drug_name", ""),
            "txgnnMatchStatus": row.get("match_status", ""),
            "txgnnMatchField": row.get("match_field", ""),
        }
    return lookup


def load_txgnn_direction_context(path: Path) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    if not path.exists():
        return {}, {}
    rows = read_csv(path)
    disease_by_direction: dict[str, dict[str, str]] = {}
    scores_by_direction_drugbank: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        direction = row.get("direction_proxy", "")
        if not direction:
            continue
        disease_by_direction.setdefault(
            direction,
            {
                "txgnnDiseaseId": norm_id(first_nonempty(row.get("txgnn_disease_numeric_id"), row.get("disease_id"))),
                "txgnnDiseaseName": row.get("disease_name", ""),
                "txgnnDiseaseIdx": row.get("txgnn_disease_idx", ""),
                "txgnnOriginalDiseaseId": row.get("disease_id", ""),
            },
        )
        drugbank = row.get("txgnn_drugbank_id", "")
        if drugbank:
            scores_by_direction_drugbank[(direction, drugbank)] = {
                "txgnnIndicationScore": number(row.get("txgnn_indication_score")),
                "txgnnIndicationLogit": number(row.get("txgnn_indication_logit")),
                "txgnnDiseaseName": row.get("disease_name", ""),
                "txgnnDiseaseId": norm_id(first_nonempty(row.get("txgnn_disease_numeric_id"), row.get("disease_id"))),
            }
    return disease_by_direction, scores_by_direction_drugbank


def build_disease_match_sets(
    disease_by_direction: dict[str, dict[str, str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    disease_ids: dict[str, set[str]] = {direction: set() for direction in DIRECTION_LABELS}
    disease_names: dict[str, set[str]] = {direction: set() for direction in DIRECTION_LABELS}
    for direction, item in disease_by_direction.items():
        for key in ["txgnnDiseaseId", "txgnnOriginalDiseaseId"]:
            value = norm_id(item.get(key))
            if value:
                disease_ids.setdefault(direction, set()).add(value)
        name = norm_text(item.get("txgnnDiseaseName"))
        if name:
            disease_names.setdefault(direction, set()).add(name)
    return disease_ids, disease_names


def extract_drug_gene(row: pd.Series) -> tuple[str, str, str, str] | None:
    x_type = row.get("x_type", "")
    y_type = row.get("y_type", "")
    if x_type == "drug" and y_type == "gene/protein":
        return norm_id(row.get("x_id")), row.get("x_name", ""), norm_gene(row.get("y_name")), row.get("y_name", "")
    if x_type == "gene/protein" and y_type == "drug":
        return norm_id(row.get("y_id")), row.get("y_name", ""), norm_gene(row.get("x_name")), row.get("x_name", "")
    return None


def extract_drug_disease(row: pd.Series) -> tuple[str, str, str, str] | None:
    x_type = row.get("x_type", "")
    y_type = row.get("y_type", "")
    if x_type == "drug" and y_type == "disease":
        return norm_id(row.get("x_id")), row.get("x_name", ""), norm_id(row.get("y_id")), row.get("y_name", "")
    if x_type == "disease" and y_type == "drug":
        return norm_id(row.get("y_id")), row.get("y_name", ""), norm_id(row.get("x_id")), row.get("x_name", "")
    return None


def extract_gene_disease(row: pd.Series) -> tuple[str, str, str, str] | None:
    x_type = row.get("x_type", "")
    y_type = row.get("y_type", "")
    if x_type == "gene/protein" and y_type == "disease":
        return norm_gene(row.get("x_name")), row.get("x_name", ""), norm_id(row.get("y_id")), row.get("y_name", "")
    if x_type == "disease" and y_type == "gene/protein":
        return norm_gene(row.get("y_name")), row.get("y_name", ""), norm_id(row.get("x_id")), row.get("x_name", "")
    return None


def extract_ppi(row: pd.Series) -> tuple[str, str] | None:
    if row.get("x_type") == "gene/protein" and row.get("y_type") == "gene/protein":
        left = norm_gene(row.get("x_name"))
        right = norm_gene(row.get("y_name"))
        if left and right and left != right:
            return left, right
    return None


def scan_kg(
    kg_path: Path,
    candidate_drugbank_ids: set[str],
    candidate_genes: set[str],
    disease_ids_by_direction: dict[str, set[str]],
    disease_names_by_direction: dict[str, set[str]],
    chunksize: int,
) -> dict[str, Any]:
    drug_target_edges: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    drug_known_targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    drug_disease_edges: dict[tuple[str, str], set[tuple[str, str, str, str]]] = defaultdict(set)
    target_disease_edges: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    disease_genes: dict[str, set[str]] = defaultdict(set)
    ppi_neighbors: dict[str, set[str]] = defaultdict(set)
    relation_rows = Counter()

    usecols = [
        "relation",
        "display_relation",
        "x_id",
        "x_type",
        "x_name",
        "y_id",
        "y_type",
        "y_name",
    ]
    for chunk in pd.read_csv(kg_path, dtype=str, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk.fillna("")
        chunk = chunk[chunk["relation"].isin(KG_RELATIONS_OF_INTEREST)]
        if chunk.empty:
            continue
        relation_rows.update(chunk["relation"].value_counts().to_dict())

        drug_protein = chunk[chunk["relation"].eq("drug_protein")]
        for _, row in drug_protein.iterrows():
            extracted = extract_drug_gene(row)
            if not extracted:
                continue
            drug_id, _drug_name, gene_norm, gene_name = extracted
            if drug_id in candidate_drugbank_ids:
                drug_known_targets[drug_id].add((gene_norm, first_nonempty(gene_name, gene_norm)))
                if gene_norm in candidate_genes:
                    drug_target_edges[(drug_id, gene_norm)].add((row.get("display_relation", ""), first_nonempty(gene_name, gene_norm)))

        drug_disease = chunk[chunk["relation"].isin(DRUG_DISEASE_RELATIONS)]
        for _, row in drug_disease.iterrows():
            extracted = extract_drug_disease(row)
            if not extracted:
                continue
            drug_id, _drug_name, disease_id, disease_name = extracted
            if drug_id not in candidate_drugbank_ids:
                continue
            for direction in matching_directions(disease_id, disease_name, disease_ids_by_direction, disease_names_by_direction):
                drug_disease_edges[(drug_id, direction)].add(
                    (
                        row.get("relation", ""),
                        row.get("display_relation", ""),
                        norm_id(disease_id),
                        disease_name,
                    )
                )

        disease_protein = chunk[chunk["relation"].eq("disease_protein")]
        for _, row in disease_protein.iterrows():
            extracted = extract_gene_disease(row)
            if not extracted:
                continue
            gene_norm, gene_name, disease_id, disease_name = extracted
            directions = matching_directions(disease_id, disease_name, disease_ids_by_direction, disease_names_by_direction)
            if not directions:
                continue
            for direction in directions:
                disease_genes[direction].add(gene_norm)
                if gene_norm in candidate_genes:
                    target_disease_edges[(gene_norm, direction)].add(
                        (
                            row.get("display_relation", ""),
                            norm_id(disease_id),
                            disease_name,
                        )
                    )

        ppi = chunk[chunk["relation"].eq("protein_protein")]
        for _, row in ppi.iterrows():
            extracted = extract_ppi(row)
            if not extracted:
                continue
            left, right = extracted
            if left in candidate_genes:
                ppi_neighbors[left].add(right)
            if right in candidate_genes:
                ppi_neighbors[right].add(left)

    return {
        "drugTargetEdges": drug_target_edges,
        "drugKnownTargets": drug_known_targets,
        "drugDiseaseEdges": drug_disease_edges,
        "targetDiseaseEdges": target_disease_edges,
        "diseaseGenes": disease_genes,
        "ppiNeighbors": ppi_neighbors,
        "relationRowsScanned": dict(relation_rows),
    }


def compact_edge_list(edges: set[tuple[Any, ...]], limit: int = 5) -> list[tuple[Any, ...]]:
    return sorted(edges, key=lambda item: tuple(str(part) for part in item))[:limit]


def kg_score(
    direct_drug_target: bool,
    positive_drug_disease: bool,
    contraindication_context: bool,
    direct_target_disease: bool,
    ppi_bridge_count: int,
    drug_target_disease_bridge_count: int,
    txgnn_score: float | None,
    open_targets_score: float | None,
) -> int:
    score = 0
    score += 25 if direct_drug_target else 0
    score += 20 if positive_drug_disease else 0
    score += 6 if contraindication_context else 0
    score += 20 if direct_target_disease else 0
    score += min(15, 5 * ppi_bridge_count)
    score += min(15, 5 * drug_target_disease_bridge_count)
    score += 10 if txgnn_score is not None and txgnn_score >= 0.5 else 0
    score += 10 if open_targets_score is not None and open_targets_score > 0 else 0
    return max(0, min(100, score))


def evidence_tier(score: int) -> str:
    if score >= 70:
        return "A"
    if score >= 45:
        return "B"
    if score >= 20:
        return "C"
    return "D"


def novelty_class(
    direct_drug_target: bool,
    positive_drug_disease: bool,
    contraindication_context: bool,
    direct_target_disease: bool,
    ppi_bridge_count: int,
    drug_target_disease_bridge_count: int,
) -> str:
    if direct_drug_target and positive_drug_disease:
        return "known_mechanism_or_known_disease_use"
    if contraindication_context and not positive_drug_disease:
        return "known_negative_or_safety_context"
    if direct_drug_target:
        return "known_drug_target_mechanism"
    if positive_drug_disease:
        return "known_disease_use_with_predicted_target"
    if direct_target_disease or ppi_bridge_count or drug_target_disease_bridge_count:
        return "disease_context_supported_new_pair"
    return "model_priority_without_txgnn_kg_path"


def explanation_zh(
    direct_drug_target: bool,
    positive_drug_disease: bool,
    contraindication_context: bool,
    direct_target_disease: bool,
    ppi_bridge_count: int,
    drug_target_disease_bridge_count: int,
    novelty: str,
) -> str:
    parts: list[str] = []
    if direct_drug_target:
        parts.append("TxGNN/DrugBank KG 中已有药物-候选靶点关系")
    if direct_target_disease:
        parts.append("候选靶点与该疾病方向存在 KG 疾病关联")
    if positive_drug_disease:
        parts.append("药物在该疾病方向存在 indication 或 off-label 疾病上下文")
    if contraindication_context:
        parts.append("药物在该疾病方向存在 contraindication 上下文，需要安全性审阅")
    if ppi_bridge_count:
        parts.append(f"候选靶点通过 {ppi_bridge_count} 个 PPI 邻居连接到疾病相关蛋白")
    if drug_target_disease_bridge_count:
        parts.append(f"药物的 {drug_target_disease_bridge_count} 个已知靶点连接到该疾病方向")
    if not parts:
        parts.append("TxGNN KG 中未找到浅层路径，当前主要依赖 AI 亲和、疾病排序和结构姿态，需要外部文献或实验验证")

    posture = {
        "known_mechanism_or_known_disease_use": "更像已知机制/已知用途召回或相邻适应症外拓。",
        "known_negative_or_safety_context": "更像安全性或禁忌相关候选，不应直接作为正向再利用结论。",
        "known_drug_target_mechanism": "药物-靶点机制较清楚，但疾病方向仍需验证。",
        "known_disease_use_with_predicted_target": "疾病用途较有上下文，候选靶点是需要验证的新机制。",
        "disease_context_supported_new_pair": "属于有疾病图谱支撑的新组合，适合进入机制审阅。",
        "model_priority_without_txgnn_kg_path": "属于模型优先级候选，KG 暂未提供浅层解释。",
    }.get(novelty, "")
    return "；".join(parts) + "。" + posture


def build_outputs(
    candidates: pd.DataFrame,
    crosswalk: dict[str, dict[str, str]],
    disease_by_direction: dict[str, dict[str, str]],
    txgnn_scores: dict[tuple[str, str], dict[str, Any]],
    kg: dict[str, Any],
    max_bridge_paths: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    combined_shortlist: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        direction = row.get("direction", "")
        drug_id = row.get("drugId", "")
        target_gene = norm_gene(row.get("target", ""))
        pair_id = row.get("pairId", "")
        cross = crosswalk.get(drug_id, {})
        drugbank = cross.get("txgnnDrugbankId", "")
        disease_info = disease_by_direction.get(direction, {})
        tx_info = txgnn_scores.get((direction, drugbank), {})
        tx_score = tx_info.get("txgnnIndicationScore")
        open_targets_score = number(row.get("openTargetsScore"))

        dt_edges = kg["drugTargetEdges"].get((drugbank, target_gene), set()) if drugbank else set()
        dd_edges = kg["drugDiseaseEdges"].get((drugbank, direction), set()) if drugbank else set()
        td_edges = kg["targetDiseaseEdges"].get((target_gene, direction), set())
        ppi_bridges = sorted(kg["ppiNeighbors"].get(target_gene, set()) & kg["diseaseGenes"].get(direction, set()))
        drug_target_disease_bridges = sorted(
            {
                gene
                for gene, _gene_name in kg["drugKnownTargets"].get(drugbank, set())
                if gene in kg["diseaseGenes"].get(direction, set())
            }
        )

        positive_dd = any(edge[0] in {"indication", "off-label use"} for edge in dd_edges)
        contraindication_dd = any(edge[0] == "contraindication" for edge in dd_edges)
        score = kg_score(
            direct_drug_target=bool(dt_edges),
            positive_drug_disease=positive_dd,
            contraindication_context=contraindication_dd,
            direct_target_disease=bool(td_edges),
            ppi_bridge_count=len(ppi_bridges),
            drug_target_disease_bridge_count=len(drug_target_disease_bridges),
            txgnn_score=tx_score,
            open_targets_score=open_targets_score,
        )
        tier = evidence_tier(score)
        novelty = novelty_class(
            direct_drug_target=bool(dt_edges),
            positive_drug_disease=positive_dd,
            contraindication_context=contraindication_dd,
            direct_target_disease=bool(td_edges),
            ppi_bridge_count=len(ppi_bridges),
            drug_target_disease_bridge_count=len(drug_target_disease_bridges),
        )

        base = {
            "direction": direction,
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "rank": int(number(row.get("rank")) or 999999),
            "pairId": pair_id,
            "drugId": drug_id,
            "drug": row.get("drug", ""),
            "target": row.get("target", ""),
            "protein": row.get("protein", ""),
            "txgnnDrugbankId": drugbank,
            "txgnnDrugName": cross.get("txgnnDrugName", ""),
            "txgnnMatchStatus": cross.get("txgnnMatchStatus", ""),
            "txgnnDiseaseId": disease_info.get("txgnnDiseaseId", tx_info.get("txgnnDiseaseId", "")),
            "txgnnDiseaseName": disease_info.get("txgnnDiseaseName", tx_info.get("txgnnDiseaseName", "")),
            "directionScore": number(row.get("directionScore")),
            "affinityScore": number(row.get("affinityScore")),
            "diffdock": number(row.get("diffdock")),
            "status": row.get("status", ""),
            "credibilityScore": number(row.get("credibilityScore")),
            "openTargetsScore": open_targets_score,
            "integratedTxgnnScore": number(row.get("txgnnScore")),
            "txgnnIndicationScore": tx_score,
            "admetScore": number(row.get("admetScore")),
            "admetTier": row.get("admetTier", ""),
            "translationalScore": number(row.get("translationalScore")),
        }

        def add_path(path_type: str, path_text: str, relation: str, polarity: str, evidence: str) -> None:
            path_rows.append(
                {
                    **base,
                    "pathType": path_type,
                    "pathText": path_text,
                    "relation": relation,
                    "polarity": polarity,
                    "evidence": evidence,
                }
            )

        for display_relation, gene_name in compact_edge_list(dt_edges, max_bridge_paths):
            add_path(
                "direct_drug_target",
                f"{base['txgnnDrugName'] or base['drug']} --{display_relation}--> {gene_name}",
                display_relation,
                "support",
                "TxGNN KG drug_protein edge",
            )

        for relation, display_relation, disease_id, disease_name in compact_edge_list(dd_edges, max_bridge_paths):
            add_path(
                "direct_drug_disease",
                f"{base['txgnnDrugName'] or base['drug']} --{display_relation or relation}--> {disease_name}",
                relation,
                relation_polarity(relation),
                f"TxGNN KG drug-disease edge; disease_id={disease_id}",
            )

        for display_relation, disease_id, disease_name in compact_edge_list(td_edges, max_bridge_paths):
            add_path(
                "direct_target_disease",
                f"{base['target']} --{display_relation}--> {disease_name}",
                display_relation,
                "support",
                f"TxGNN KG disease_protein edge; disease_id={disease_id}",
            )

        for bridge in ppi_bridges[:max_bridge_paths]:
            add_path(
                "ppi_target_disease_bridge",
                f"{base['target']} --ppi--> {bridge} --associated with--> {base['txgnnDiseaseName']}",
                "ppi+disease_protein",
                "support",
                "Candidate target has a PPI neighbor that is disease-associated in TxGNN KG",
            )

        for bridge in drug_target_disease_bridges[:max_bridge_paths]:
            add_path(
                "drug_known_target_disease_bridge",
                f"{base['txgnnDrugName'] or base['drug']} --known target--> {bridge} --associated with--> {base['txgnnDiseaseName']}",
                "drug_protein+disease_protein",
                "support",
                "The drug has another known target linked to this disease direction in TxGNN KG",
            )

        summary = {
            **base,
            "kgEvidenceScore": score,
            "kgEvidenceTier": tier,
            "noveltyClass": novelty,
            "hasTxgnnDrugMapping": int(bool(drugbank)),
            "hasDirectDrugTargetEdge": int(bool(dt_edges)),
            "hasPositiveDrugDiseaseEdge": int(positive_dd),
            "hasContraindicationDiseaseEdge": int(contraindication_dd),
            "hasDirectTargetDiseaseEdge": int(bool(td_edges)),
            "ppiDiseaseBridgeCount": len(ppi_bridges),
            "drugTargetDiseaseBridgeCount": len(drug_target_disease_bridges),
            "pathCount": sum(1 for item in path_rows if item["pairId"] == pair_id and item["direction"] == direction),
            "directDrugTargetRelations": "; ".join(sorted({edge[0] for edge in dt_edges})),
            "drugDiseaseRelations": "; ".join(sorted({edge[0] for edge in dd_edges})),
            "targetDiseaseExamples": "; ".join(sorted({edge[2] for edge in td_edges})[:5]),
            "ppiDiseaseBridgeGenes": "; ".join(ppi_bridges[:10]),
            "drugTargetDiseaseBridgeGenes": "; ".join(drug_target_disease_bridges[:10]),
            "kgExplanationZh": explanation_zh(
                direct_drug_target=bool(dt_edges),
                positive_drug_disease=positive_dd,
                contraindication_context=contraindication_dd,
                direct_target_disease=bool(td_edges),
                ppi_bridge_count=len(ppi_bridges),
                drug_target_disease_bridge_count=len(drug_target_disease_bridges),
                novelty=novelty,
            ),
        }
        summary_rows.append(summary)

        if (
            row.get("status", "") == "completed"
            and row.get("admetTier", "") in {"A", "B"}
            and tier in {"A", "B"}
            and not contraindication_dd
        ):
            combined_shortlist.append(summary)

    summary_rows.sort(key=lambda item: (item["direction"], -item["kgEvidenceScore"], item["rank"]))
    combined_shortlist.sort(key=lambda item: (item["direction"], -item["kgEvidenceScore"], -(item.get("translationalScore") or 0), item["rank"]))
    return summary_rows, path_rows, combined_shortlist


def summarize(summary_rows: list[dict[str, Any]], path_rows: list[dict[str, Any]], kg_scan: dict[str, Any]) -> dict[str, Any]:
    by_direction: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["direction"]].append(row)
    for direction, rows in sorted(grouped.items()):
        by_direction[direction] = {
            "labelZh": DIRECTION_LABELS.get(direction, direction),
            "candidateRows": len(rows),
            "txgnnMappedRows": sum(row["hasTxgnnDrugMapping"] for row in rows),
            "anyKgPathRows": sum(1 for row in rows if row["pathCount"] > 0),
            "anyKgPathPct": pct(sum(1 for row in rows if row["pathCount"] > 0), len(rows)),
            "directDrugTargetRows": sum(row["hasDirectDrugTargetEdge"] for row in rows),
            "positiveDrugDiseaseRows": sum(row["hasPositiveDrugDiseaseEdge"] for row in rows),
            "contraindicationRows": sum(row["hasContraindicationDiseaseEdge"] for row in rows),
            "directTargetDiseaseRows": sum(row["hasDirectTargetDiseaseEdge"] for row in rows),
            "ppiBridgeRows": sum(1 for row in rows if row["ppiDiseaseBridgeCount"] > 0),
            "drugTargetDiseaseBridgeRows": sum(1 for row in rows if row["drugTargetDiseaseBridgeCount"] > 0),
            "kgEvidenceTierCounts": dict(Counter(row["kgEvidenceTier"] for row in rows)),
            "noveltyClassCounts": dict(Counter(row["noveltyClass"] for row in rows)),
            "medianKgEvidenceScore": float(pd.Series([row["kgEvidenceScore"] for row in rows]).median()) if rows else None,
        }

    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": len(summary_rows),
        "pathRows": len(path_rows),
        "txgnnMappedRows": sum(row["hasTxgnnDrugMapping"] for row in summary_rows),
        "anyKgPathRows": sum(1 for row in summary_rows if row["pathCount"] > 0),
        "anyKgPathPct": pct(sum(1 for row in summary_rows if row["pathCount"] > 0), len(summary_rows)),
        "directDrugTargetRows": sum(row["hasDirectDrugTargetEdge"] for row in summary_rows),
        "positiveDrugDiseaseRows": sum(row["hasPositiveDrugDiseaseEdge"] for row in summary_rows),
        "contraindicationRows": sum(row["hasContraindicationDiseaseEdge"] for row in summary_rows),
        "directTargetDiseaseRows": sum(row["hasDirectTargetDiseaseEdge"] for row in summary_rows),
        "ppiBridgeRows": sum(1 for row in summary_rows if row["ppiDiseaseBridgeCount"] > 0),
        "drugTargetDiseaseBridgeRows": sum(1 for row in summary_rows if row["drugTargetDiseaseBridgeCount"] > 0),
        "kgEvidenceTierCounts": dict(Counter(row["kgEvidenceTier"] for row in summary_rows)),
        "noveltyClassCounts": dict(Counter(row["noveltyClass"] for row in summary_rows)),
        "pathTypeCounts": dict(Counter(row["pathType"] for row in path_rows)),
        "byDirection": by_direction,
        "relationRowsScanned": kg_scan.get("relationRowsScanned", {}),
        "methodNote": (
            "Deterministic shallow TxGNN KG audit over ADMET-filtered candidates. "
            "Paths cover direct drug-target, direct drug-disease, target-disease, "
            "target-PPI-disease, and drug-known-target-disease evidence. Absence of a "
            "path means no shallow path was found in this KG slice, not absence of biology."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build candidate-level TxGNN KG explainability audit.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--candidates",
        default="outputs/sota_validation/admet_repurposing/sota_admet_filtered_candidate_shortlist.csv",
    )
    parser.add_argument(
        "--integrated-candidates",
        default="outputs/disease_directions/disease_direction_integrated_candidates.csv",
    )
    parser.add_argument("--kg", default="data/raw/txgnn/kg.csv")
    parser.add_argument("--crosswalk", default="outputs/sota_validation/txgnn_direction_drug_name_crosswalk.csv")
    parser.add_argument("--txgnn-scores", default="outputs/sota_validation/txgnn_multi_direction_scores.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/kg_explainability")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--max-bridge-paths", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    candidates = load_candidate_context(root, root / args.candidates, root / args.integrated_candidates)
    crosswalk = load_crosswalk(root / args.crosswalk)
    disease_by_direction, txgnn_scores = load_txgnn_direction_context(root / args.txgnn_scores)
    disease_ids_by_direction, disease_names_by_direction = build_disease_match_sets(disease_by_direction)

    candidates["txgnnDrugbankId"] = candidates["drugId"].map(lambda drug_id: crosswalk.get(drug_id, {}).get("txgnnDrugbankId", ""))
    candidate_drugbank_ids = {norm_id(value) for value in candidates["txgnnDrugbankId"].tolist() if norm_id(value)}
    candidate_genes = {norm_gene(value) for value in candidates["target"].tolist() if norm_gene(value)}

    kg_scan = scan_kg(
        root / args.kg,
        candidate_drugbank_ids=candidate_drugbank_ids,
        candidate_genes=candidate_genes,
        disease_ids_by_direction=disease_ids_by_direction,
        disease_names_by_direction=disease_names_by_direction,
        chunksize=args.chunksize,
    )
    summary_rows, path_rows, combined_shortlist = build_outputs(
        candidates,
        crosswalk=crosswalk,
        disease_by_direction=disease_by_direction,
        txgnn_scores=txgnn_scores,
        kg=kg_scan,
        max_bridge_paths=args.max_bridge_paths,
    )
    summary = summarize(summary_rows, path_rows, kg_scan)
    summary["inputs"] = {
        "candidates": args.candidates,
        "integratedCandidates": args.integrated_candidates,
        "kg": args.kg,
        "crosswalk": args.crosswalk,
        "txgnnScores": args.txgnn_scores,
    }
    summary["outputs"] = {
        "candidateSummary": str((out_dir / "candidate_kg_explanation_summary.csv").resolve()),
        "pathTable": str((out_dir / "candidate_kg_explanation_paths.csv").resolve()),
        "combinedShortlist": str((out_dir / "sota_admet_kg_explainable_shortlist.csv").resolve()),
        "summary": str((out_dir / "kg_explainability_summary.json").resolve()),
    }

    summary_fields = [
        "direction",
        "labelZh",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "txgnnDrugbankId",
        "txgnnDrugName",
        "txgnnMatchStatus",
        "txgnnDiseaseId",
        "txgnnDiseaseName",
        "directionScore",
        "affinityScore",
        "diffdock",
        "status",
        "credibilityScore",
        "openTargetsScore",
        "integratedTxgnnScore",
        "txgnnIndicationScore",
        "admetScore",
        "admetTier",
        "translationalScore",
        "kgEvidenceScore",
        "kgEvidenceTier",
        "noveltyClass",
        "hasTxgnnDrugMapping",
        "hasDirectDrugTargetEdge",
        "hasPositiveDrugDiseaseEdge",
        "hasContraindicationDiseaseEdge",
        "hasDirectTargetDiseaseEdge",
        "ppiDiseaseBridgeCount",
        "drugTargetDiseaseBridgeCount",
        "pathCount",
        "directDrugTargetRelations",
        "drugDiseaseRelations",
        "targetDiseaseExamples",
        "ppiDiseaseBridgeGenes",
        "drugTargetDiseaseBridgeGenes",
        "kgExplanationZh",
    ]
    path_fields = [
        "direction",
        "labelZh",
        "rank",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "txgnnDrugbankId",
        "txgnnDrugName",
        "txgnnDiseaseId",
        "txgnnDiseaseName",
        "pathType",
        "pathText",
        "relation",
        "polarity",
        "evidence",
    ]
    write_csv(out_dir / "candidate_kg_explanation_summary.csv", summary_fields, summary_rows)
    write_csv(out_dir / "candidate_kg_explanation_paths.csv", path_fields, path_rows)
    write_csv(out_dir / "sota_admet_kg_explainable_shortlist.csv", summary_fields, combined_shortlist)
    write_json(out_dir / "kg_explainability_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
