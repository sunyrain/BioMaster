#!/usr/bin/env python3
"""Build a detailed professor-review PDF for disease-direction Top10 candidates."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from weasyprint import HTML


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("outputs/sota_validation/professor_candidate_detailed_review")
CANDIDATES = Path("outputs/disease_directions/disease_direction_integrated_candidates.csv")
EXPERT_SCORED = Path("outputs/sota_validation/expert_review_panel/integrated_expert_review_scored_candidates.csv")
FINAL_PRIORITY = Path("outputs/sota_validation/final_prioritization/final_candidate_priority_table.csv")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wetlab_candidate_detailed_review_pdf import (  # noqa: E402
    TARGET_SYNONYMS,
    base_drug_name,
    clinical_trials_query,
    clinical_trials_term,
    int_text,
    num,
    pubmed_query,
    pubmed_term,
    text,
)


DIRECTION_ORDER = [
    "oncology",
    "infectious_disease",
    "cardiovascular",
    "neurology_psychiatry",
    "immunology_inflammation",
]

TARGET_SYNONYMS.update(
    {
        "ADRA1A": ["ADRA1A", "alpha-1A adrenergic receptor", "alpha 1A adrenergic receptor"],
        "AVPR1A": ["AVPR1A", "vasopressin V1A receptor", "arginine vasopressin receptor 1A"],
        "AVPR1B": ["AVPR1B", "vasopressin V1B receptor", "arginine vasopressin receptor 1B"],
        "CA2": ["CA2", "carbonic anhydrase 2", "carbonic anhydrase II"],
        "CA9": ["CA9", "carbonic anhydrase 9", "carbonic anhydrase IX"],
        "CHRM2": ["CHRM2", "muscarinic acetylcholine receptor M2", "M2 muscarinic receptor"],
        "CXCR4": ["CXCR4", "C-X-C chemokine receptor type 4"],
        "DRD1": ["DRD1", "dopamine receptor D1"],
        "DRD2": ["DRD2", "dopamine receptor D2"],
        "DRD3": ["DRD3", "dopamine receptor D3"],
        "EDNRB": ["EDNRB", "endothelin receptor type B", "endothelin B receptor"],
        "FRK": ["FRK", "Fyn-related kinase", "PTK5"],
        "FSHR": ["FSHR", "follicle-stimulating hormone receptor"],
        "GCGR": ["GCGR", "glucagon receptor"],
        "GNRHR": ["GNRHR", "gonadotropin-releasing hormone receptor"],
        "HTR1A": ["HTR1A", "5-HT1A receptor", "serotonin receptor 1A"],
        "HTR1B": ["HTR1B", "5-HT1B receptor", "serotonin receptor 1B"],
        "HTR2A": ["HTR2A", "5-HT2A receptor", "serotonin receptor 2A"],
        "HTR3A": ["HTR3A", "5-HT3A receptor", "serotonin receptor 3A"],
        "HTR4": ["HTR4", "5-HT4 receptor", "serotonin receptor 4"],
        "LHCGR": ["LHCGR", "lutropin-choriogonadotropic hormone receptor"],
        "NPY5R": ["NPY5R", "neuropeptide Y receptor Y5"],
        "OPRK1": ["OPRK1", "kappa opioid receptor"],
        "OPRM1": ["OPRM1", "mu opioid receptor"],
        "PTGER4": ["PTGER4", "prostaglandin E2 receptor EP4"],
        "RXFP1": ["RXFP1", "relaxin family peptide receptor 1"],
        "SLC5A2": ["SLC5A2", "sodium-glucose cotransporter 2", "SGLT2"],
        "SLC6A4": ["SLC6A4", "serotonin transporter", "SERT"],
        "SSTR2": ["SSTR2", "somatostatin receptor 2"],
        "TSHR": ["TSHR", "thyrotropin receptor", "thyroid-stimulating hormone receptor"],
        "VDR": ["VDR", "vitamin D receptor"],
    }
)


def esc(value: Any) -> str:
    return html.escape(text(value, ""), quote=True)


def val(row: pd.Series, name: str, default: str = "-") -> str:
    return text(row.get(name), default)


def normalize_drug_name(value: Any) -> str:
    s = str(value or "").lower()
    s = re.sub(
        r"\b(hydrochloride|hydrochrloride|phosphate|bromide|succinate|mesylate|maleate|"
        r"dimaleate|dihydrochloride|acetate|sodium|potassium|calcium)\b",
        "",
        s,
    )
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def number_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def is_truthy_number(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
        return int(float(value)) == 1
    except Exception:
        return False


def load_candidate_pool(root: Path) -> pd.DataFrame:
    """Use final expert-prioritized rows when available, not raw direction Top10 rows."""
    expert_path = root / EXPERT_SCORED
    final_path = root / FINAL_PRIORITY
    raw_path = root / CANDIDATES

    if expert_path.exists():
        candidates = pd.read_csv(expert_path)
        if final_path.exists():
            final = pd.read_csv(final_path)
            final_cols = [
                "direction",
                "drugId",
                "protein",
                "rank",
                "directionScore",
                "status",
                "credibilityScore",
                "finalRankWithinDirection",
                "finalPriorityTier",
                "therapeuticArea",
                "indication",
                "targetDiseaseExamples",
                "validationGatesZh",
                "poseAuditReason",
            ]
            final_cols = [col for col in final_cols if col in final.columns]
            candidates = candidates.merge(
                final[final_cols],
                on=["direction", "drugId", "protein"],
                how="left",
                suffixes=("", "_final"),
            )
    else:
        candidates = pd.read_csv(raw_path)
        candidates = merge_expert_fields(root, candidates)

    if "directionLabelZh" not in candidates.columns and "directionLabelZhFinal" in candidates.columns:
        candidates["directionLabelZh"] = candidates["directionLabelZhFinal"]
    if "txgnnScore" not in candidates.columns and "integratedTxgnnScore" in candidates.columns:
        candidates["txgnnScore"] = candidates["integratedTxgnnScore"]
    if "rank" not in candidates.columns and "finalRankWithinDirection" in candidates.columns:
        candidates["rank"] = candidates["finalRankWithinDirection"]
    if "credibilityTierZh" not in candidates.columns:
        tier = candidates.get("finalPriorityTier", pd.Series([""] * len(candidates)))
        candidates["credibilityTierZh"] = tier.fillna("").map(
            lambda value: f"{value}｜最终多证据优先级" if str(value) else "专家审阅候选"
        )
    if "rationaleZh" not in candidates.columns:
        candidates["rationaleZh"] = candidates.get("expertRationaleZh", "")
    elif "expertRationaleZh" in candidates.columns:
        candidates["rationaleZh"] = candidates["rationaleZh"].fillna(candidates["expertRationaleZh"])
    if "evidencePathZh" not in candidates.columns:
        candidates["evidencePathZh"] = candidates.get("kgExplanationZh", "")
    if "status" not in candidates.columns:
        candidates["status"] = "completed"
    if "directionScore" not in candidates.columns:
        candidates["directionScore"] = candidates.get(
            "finalPriorityScore",
            candidates.get("expertReviewScore", pd.Series([0.0] * len(candidates))),
        )

    return candidates


def _candidate_sort(block: pd.DataFrame) -> pd.DataFrame:
    if "expertReviewScore" in block.columns:
        by = ["expertReviewScore"]
        ascending = [False]
        if "finalPriorityScore" in block.columns:
            by.append("finalPriorityScore")
            ascending.append(False)
        if "rank" in block.columns:
            by.append("rank")
            ascending.append(True)
        return block.sort_values(
            by,
            ascending=ascending,
            na_position="last",
        )
    if "finalPriorityScore" in block.columns:
        by = ["finalPriorityScore"]
        ascending = [False]
        if "rank" in block.columns:
            by.append("rank")
            ascending.append(True)
        return block.sort_values(
            by,
            ascending=ascending,
            na_position="last",
        )
    if "rank" in block.columns:
        return block.sort_values("rank", na_position="last")
    return block


def _diverse_top(block: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    selected: list[pd.Series] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for max_per_drug, max_per_target in [(1, 1), (1, 2), (2, 2), (3, 3), (999, 999)]:
        drug_counts: dict[str, int] = {}
        target_counts: dict[str, int] = {}
        for row in selected:
            drug_counts[str(row["_drug_norm"])] = drug_counts.get(str(row["_drug_norm"]), 0) + 1
            target_counts[str(row["target"])] = target_counts.get(str(row["target"]), 0) + 1
        for _, row in block.iterrows():
            pair = (str(row["_drug_norm"]), str(row.get("target", "")), str(row.get("protein", "")))
            if pair in seen_pairs:
                continue
            drug_norm = str(row["_drug_norm"])
            target = str(row.get("target", ""))
            if drug_counts.get(drug_norm, 0) >= max_per_drug:
                continue
            if target_counts.get(target, 0) >= max_per_target:
                continue
            selected.append(row)
            seen_pairs.add(pair)
            drug_counts[drug_norm] = drug_counts.get(drug_norm, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
            if len(selected) >= n:
                return pd.DataFrame(selected)
    return pd.DataFrame(selected)


def select_direction_top10(candidates: pd.DataFrame) -> pd.DataFrame:
    candidates = candidates.copy()
    if "directionLabelZh" not in candidates.columns and "directionLabelZhFinal" in candidates.columns:
        candidates["directionLabelZh"] = candidates["directionLabelZhFinal"]
    blocks: list[pd.DataFrame] = []
    for direction in DIRECTION_ORDER:
        block = _candidate_sort(candidates[candidates["direction"] == direction].copy())
        if block.empty:
            continue
        block["_drug_norm"] = block["drug"].map(normalize_drug_name)
        block = block.drop_duplicates(["_drug_norm", "target", "protein"], keep="first")
        block = _diverse_top(block, 10)
        block["reviewRankInDirection"] = range(1, len(block) + 1)
        blocks.append(block)
    selected = pd.concat(blocks, ignore_index=True)
    selected["reportRank"] = range(1, len(selected) + 1)
    return selected


def merge_expert_fields(root: Path, selected: pd.DataFrame) -> pd.DataFrame:
    expert_path = root / EXPERT_SCORED
    if not expert_path.exists():
        return selected
    expert = pd.read_csv(expert_path)
    wanted = [
        "direction",
        "drugId",
        "protein",
        "knownDrugTargetPair",
        "noveltyClass",
        "reviewTrack",
        "expertReviewScore",
        "expertReviewTier",
        "expertNoveltyGroup",
        "expertEvidenceSupportCount",
        "expertRiskPenalty",
        "sotaContextScore",
        "integratedTxgnnScore",
        "kgEvidenceScore",
        "expertModelScore",
        "expertDiseaseEvidenceScore",
        "expertPathwayScore",
        "expertCmapScore",
        "expertTissueScore",
        "expertStructureScore",
        "expertAdmetScore",
        "expertDepmapScore",
        "cmapReversalTier",
        "cmapBestRawReversal",
        "cmapBestCell",
        "cmapSignatureCount",
        "pathwayDiseaseContextTier",
        "reactomeTopPathways",
        "creedsTargetDirectionHit",
        "creedsMatchedDiseases",
        "gtexContextTier",
        "gtexTopRelevantTissuesByTpm",
        "tissueContextTier",
        "topRelevantTissuesByNtpm",
        "depmapDependencyTier",
        "depmapDependencyPositiveFlag",
        "admetTier",
        "sotaMlAdmetTier",
        "mlAdmetRiskFlags",
        "structureConfidenceTier",
        "standardPoseValidationTier",
        "standardPoseValidationReason",
        "prolifInteractionTypes",
        "prolifTopInteractions",
        "poseAuditStatus",
        "contraindicationFlag",
        "hasContraindicationDiseaseEdge",
        "assayModality",
        "assayRationale",
        "expertRationaleZh",
        "expertReviewGapsZh",
        "expertDiscussionNoteZh",
        "kgExplanationZh",
    ]
    existing = [c for c in wanted if c in expert.columns]
    merged = selected.merge(
        expert[existing],
        on=["direction", "drugId", "protein"],
        how="left",
        suffixes=("", "_expert"),
    )
    return merged


def build_literature_audit(panel: pd.DataFrame, out_dir: Path, refresh: bool) -> pd.DataFrame:
    cache_path = out_dir / "professor_candidate_detailed_review_literature_cache.json"
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cached = {}

    session = requests.Session()
    records: list[dict[str, Any]] = []
    changed = False
    for _, row in panel.iterrows():
        key = f"{val(row, 'direction')}__{val(row, 'drug')}__{val(row, 'target')}"
        if key in cached and not refresh:
            record = cached[key]
        else:
            direct_term = pubmed_term(row, include_disease=False)
            disease_term = pubmed_term(row, include_disease=True)
            ct_term = clinical_trials_term(row)
            direct = pubmed_query(session, direct_term)
            time.sleep(0.34)
            direct_disease = pubmed_query(session, disease_term)
            time.sleep(0.34)
            clinical = clinical_trials_query(session, ct_term)
            time.sleep(0.2)
            record = {
                "direction": val(row, "direction"),
                "directionLabelZh": val(row, "directionLabelZh"),
                "drug": val(row, "drug"),
                "target": val(row, "target"),
                "pubmedDirectPairQuery": direct_term,
                "pubmedDirectPairCount": direct.get("count"),
                "pubmedDirectPairPmids": ";".join(direct.get("ids", [])),
                "pubmedDirectPairUrl": direct.get("url"),
                "pubmedDiseasePairQuery": disease_term,
                "pubmedDiseasePairCount": direct_disease.get("count"),
                "pubmedDiseasePairPmids": ";".join(direct_disease.get("ids", [])),
                "pubmedDiseasePairUrl": direct_disease.get("url"),
                "clinicalTrialsDrugDiseaseQuery": ct_term,
                "clinicalTrialsDrugDiseaseCount": clinical.get("count"),
                "clinicalTrialsDrugDiseaseNcts": ";".join(clinical.get("ids", [])),
                "clinicalTrialsDrugDiseaseUrl": clinical.get("url"),
                "queriedUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            cached[key] = record
            changed = True
        decision = literature_decision(row, record)
        record = dict(record)
        record.update(decision)
        records.append(record)

    if changed or refresh or not cache_path.exists():
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = pd.DataFrame(records)
    audit.to_csv(out_dir / "professor_candidate_detailed_review_literature_audit.csv", index=False)
    return audit


def literature_decision(row: pd.Series, record: dict[str, Any]) -> dict[str, str]:
    pair_count = number_or_none(record.get("pubmedDirectPairCount"))
    disease_pair_count = number_or_none(record.get("pubmedDiseasePairCount"))
    novelty = val(row, "noveltyClass", "")
    group = val(row, "expertNoveltyGroup", "")
    known_pair = is_truthy_number(row.get("knownDrugTargetPair"))

    if known_pair or group == "positive_control_or_known_mechanism" or novelty in {
        "known_drug_target_mechanism",
        "known_mechanism_or_known_disease_use",
    }:
        return {
            "literatureStatusZh": "已知机制或阳性对照",
            "literatureActionZh": "不作为新发现推进；可保留为体系校准、召回验证或机制延展背景。",
            "reviewActionZh": "若预算有限，可忽略为新发现候选，只保留少数阳性对照。",
            "noveltyPriorityZh": "低新颖性，高校准价值。",
        }
    if group == "risk_review" or "known_negative_or_safety_context" in novelty:
        return {
            "literatureStatusZh": "风险/负向或安全上下文",
            "literatureActionZh": "不建议直接进入首轮验证；需专家确认疾病方向、药理方向和安全窗口后再考虑。",
            "reviewActionZh": "可作为排除项或二线审阅项，除非有明确机制假设。",
            "noveltyPriorityZh": "推进优先级低。",
        }
    if pair_count is None:
        return {
            "literatureStatusZh": "在线检索未完成",
            "literatureActionZh": "暂按本地 novelty 标签处理；购买前必须人工检索 PubMed、ClinicalTrials、专利和中文文献。",
            "reviewActionZh": "暂缓采购，先补人工排重。",
            "noveltyPriorityZh": "待确认。",
        }
    if pair_count >= 20 or (disease_pair_count is not None and disease_pair_count >= 10):
        return {
            "literatureStatusZh": "公开组合线索较多",
            "literatureActionZh": "优先做人工排重；若已有直接 drug-target-disease 机制验证，则应降级或忽略。",
            "reviewActionZh": "可作为机制延展，不宜包装为新用途/新靶点发现。",
            "noveltyPriorityZh": "中低新颖性。",
        }
    if pair_count >= 1 or (disease_pair_count is not None and disease_pair_count >= 1):
        return {
            "literatureStatusZh": "有少量文献线索",
            "literatureActionZh": "保留候选，但购买前需逐篇确认是否已有直接靶点验证，还是只属于背景共现。",
            "reviewActionZh": "适合作为中等新颖性候选，与更干净的新组合并行审阅。",
            "noveltyPriorityZh": "中等新颖性。",
        }
    if "known_disease_use_with_predicted_target" in novelty:
        return {
            "literatureStatusZh": "未见明显直接组合文献；属于已知用途背景下的新靶点假设",
            "literatureActionZh": "可作为机制延展推进，重点验证候选靶点是否真实参与该药在该方向的作用。",
            "reviewActionZh": "适合专家审阅；先做低成本靶点 engagement 和方向性 readout。",
            "noveltyPriorityZh": "中高新颖性。",
        }
    return {
        "literatureStatusZh": "未见明显直接组合文献",
        "literatureActionZh": "可作为新用途/新靶点候选推进；购买前仍需人工排重和专利检索。",
        "reviewActionZh": "优先进入专家讨论和可行性核查。",
        "noveltyPriorityZh": "较高新颖性。",
    }


def target_plan(row: pd.Series) -> dict[str, str]:
    target = val(row, "target")
    protein = val(row, "proteinName")
    drug = val(row, "drug")
    gpcr = {
        "ADRA1A",
        "AVPR1A",
        "AVPR1B",
        "CHRM2",
        "CHRM3",
        "CXCR4",
        "DRD1",
        "DRD2",
        "DRD3",
        "EDNRA",
        "EDNRB",
        "F2R",
        "FSHR",
        "GCGR",
        "GNRHR",
        "HTR1A",
        "HTR1B",
        "HTR2A",
        "HTR4",
        "LHCGR",
        "NPY5R",
        "OPRK1",
        "OPRM1",
        "OXTR",
        "PTAFR",
        "PTGER2",
        "PTGER4",
        "RXFP1",
        "SSTR2",
        "TSHR",
    }
    kinase = {"EGFR", "KIT", "FRK"}
    if target in kinase:
        return {
            "class": "激酶/酪氨酸激酶类靶点",
            "primary": f"纯化 {target} kinase 活性或结合实验，得到 IC50/Kd；细胞中检测相应 phospho-marker。",
            "orthogonal": "CETSA、NanoBRET、target pull-down 或下游 pERK/pAKT/pSTAT 等通路 readout。",
            "counter": "同家族 kinase panel、靶点低表达细胞、非相关 RTK/kinase counterscreen 和细胞毒性窗口。",
            "model": "优先选靶点高表达且通路可诱导的肿瘤或疾病相关细胞；必须配靶点低表达/阴性模型。",
            "go": "非毒性浓度下出现剂量依赖靶点抑制，并能在正交 readout 中复现；若只有泛毒性或只影响无关通路则停止。",
        }
    if target in {"CA2", "CA9"}:
        return {
            "class": "酶类靶点",
            "primary": f"{target} 酶活抑制实验，比较 CA 同工酶选择性。",
            "orthogonal": "热稳定性、结合实验或细胞内 pH/碳酸酐酶相关功能 readout。",
            "counter": "CA1/CA2/CA9/CA12 同工酶 panel、细胞毒性和非特异 pH 干扰。",
            "model": "选择表达相应 CA 同工酶且疾病方向相关的细胞；CA9 候选应考虑低氧诱导模型。",
            "go": "必须证明同工酶选择性和可达到浓度下的细胞 readout；若只显示非特异 pH 或毒性效应则降级。",
        }
    if target in {"SLC5A2", "SLC6A4"}:
        substrate = "葡萄糖/钠依赖转运" if target == "SLC5A2" else "5-HT uptake"
        return {
            "class": "转运体靶点",
            "primary": f"{target} 转运功能实验，检测 {substrate} 的抑制或调节。",
            "orthogonal": "放射性/荧光底物 uptake、转运体表达救援或 knockdown/overexpression 对照。",
            "counter": "同家族转运体、底物荧光/读数干扰、细胞毒性和膜完整性检查。",
            "model": "先用工程细胞或高表达模型确认功能，再进入疾病相关细胞。",
            "go": "只有在转运体依赖 readout、表达依赖性和非毒性窗口一致时推进。",
        }
    if target == "VDR":
        return {
            "class": "核受体靶点",
            "primary": "VDR response element reporter、CYP24A1/经典 VDR 靶基因表达和配体竞争实验。",
            "orthogonal": "VDR knockdown/antagonist rescue、转录组方向性验证和配体结合实验。",
            "counter": "RXR/其他核受体、维生素 D 通路背景、细胞毒性和分化效应干扰。",
            "model": "选择免疫/炎症相关细胞，先确认 VDR 表达和经典 VDR readout 可诱导。",
            "go": "需要靶点依赖转录 readout 与疾病相关表型同向；若只有广泛分化或毒性效应则停止。",
        }
    if target == "AR":
        return {
            "class": "核受体靶点",
            "primary": "AR response element reporter、经典 AR 靶基因表达和配体竞争实验。",
            "orthogonal": "AR knockdown/antagonist rescue、核转位 readout 和转录组方向性验证。",
            "counter": "其他核受体、激素背景、细胞毒性和广泛转录干扰。",
            "model": "优先用 AR 表达明确的工程细胞或疾病相关细胞，再讨论感染亚型模型。",
            "go": "必须看到 AR 依赖的剂量反应；若不能和激素背景或毒性区分则停止。",
        }
    if target == "PPIA":
        return {
            "class": "宿主因子/酶类靶点",
            "primary": "PPIA/cyclophilin A 结合或酶活实验，并用 Cyclosporine 作阳性机制对照。",
            "orthogonal": "CETSA、配体竞争、PPIA knockdown/rescue 和 calcineurin/NFAT 相关 readout。",
            "counter": "PPIA/PPIB 等 cyclophilin 家族、免疫抑制通路、细胞毒性和泛抗病毒效应。",
            "model": "先确认 PPIA target engagement，再进入病毒复制或感染宿主细胞模型。",
            "go": "需要 PPIA 依赖、非毒性窗口和感染模型 readout 同时成立；否则只作为机制正控。",
        }
    if target == "FZD2":
        return {
            "class": "Wnt/FZD 宿主通路靶点",
            "primary": "FZD2/Wnt reporter 或 beta-catenin 通路 readout，确认药物是否调节 Wnt 信号。",
            "orthogonal": "FZD2 knockdown/overexpression、Wnt ligand rescue、LRP6/DVL/beta-catenin 下游标志验证。",
            "counter": "其他 FZD 受体、荧光/报告基因干扰、CYP/转运体干扰和细胞毒性。",
            "model": "先用 Wnt reporter 工程细胞，再进入相关感染宿主细胞模型。",
            "go": "必须出现 FZD2 依赖、方向一致的 Wnt readout；若只有 assay interference 则停止。",
        }
    if target == "HTR3A":
        return {
            "class": "配体门控离子通道",
            "primary": "5-HT3A 电生理、膜电位或钙流 readout，区分拮抗、激动或变构调节。",
            "orthogonal": "配体竞争、受体表达依赖性、已知 5-HT3 antagonist 对照。",
            "counter": "其他 5-HT 受体、离子通道非特异干扰、读数淬灭和细胞毒性。",
            "model": "先用 HTR3A 工程细胞；疾病模型需确认受体表达。",
            "go": "必须看到受体依赖剂量反应和 subtype/counter-screen 特异性。",
        }
    if target in gpcr:
        second = "cAMP" if target in {"GCGR", "GNRHR", "SSTR2", "TSHR", "FSHR", "LHCGR", "RXFP1", "PTGER2", "PTGER4"} else "calcium/IP1/beta-arrestin"
        return {
            "class": "GPCR 受体靶点",
            "primary": f"{target} 细胞功能实验，检测 {second}，明确 agonist、antagonist 或间接调节模式。",
            "orthogonal": "配体竞争、受体内吞、knockdown/overexpression、已知拮抗剂/激动剂 rescue。",
            "counter": "同家族/同亚型受体 panel、该药已知主靶点 counterscreen、受体阴性细胞和细胞毒性。",
            "model": "先用工程细胞建立清晰剂量反应，再选择表达该受体的疾病相关细胞。",
            "go": "需要受体表达依赖、方向一致的功能 readout；若疾病表型不能被靶点干预 rescue，则不推进。",
        }
    return {
        "class": "需要按靶点类型定制的功能靶点",
        "primary": f"围绕 {target}/{protein} 设计直接结合或功能 readout。",
        "orthogonal": "至少一个正交 target engagement 实验和一个疾病相关功能 readout。",
        "counter": f"围绕 {drug} 的已知主药理、同家族靶点、细胞毒性和 assay interference 做 counterscreen。",
        "model": "先确认靶点表达，再进入疾病相关细胞或工程模型。",
        "go": "必须同时满足剂量反应、靶点依赖性和非毒性窗口。",
    }


def candidate_decision(row: pd.Series, lit: pd.Series) -> str:
    status = val(lit, "literatureStatusZh")
    action = val(lit, "reviewActionZh")
    if status == "已知机制或阳性对照":
        return "不列为新发现优先候选；可用于阳性对照、流程召回和实验体系校准。"
    if status == "风险/负向或安全上下文":
        return "不建议直接采购进入首轮；需要专家先确认机制方向、风险是否可接受以及是否存在更好替代。"
    if "公开组合线索较多" in status:
        return "先人工排重；若已有直接机制验证则忽略，若只是共现或相邻用途则作为机制延展保留。"
    if "未见明显直接组合文献" in status:
        return "可进入专家讨论；优先做低成本靶点 engagement 和 counterscreen，避免直接跳到复杂疾病模型。"
    return action


def compact(value: Any, limit: int = 420) -> str:
    s = " ".join(text(value, "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "..."


def row_lit_lookup(literature: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    return {
        (row["direction"], row["drug"], row["target"]): row
        for _, row in literature.iterrows()
    }


def build_summary_rows(panel: pd.DataFrame, literature: pd.DataFrame) -> str:
    lookup = row_lit_lookup(literature)
    rows: list[str] = []
    for _, row in panel.iterrows():
        lit = lookup[(row["direction"], row["drug"], row["target"])]
        rows.append(
            "<tr>"
            f"<td>{int(row['reportRank'])}</td>"
            f"<td>{esc(row['directionLabelZh'])}<br><span>Top {int(row['reviewRankInDirection'])}</span></td>"
            f"<td><strong>{esc(row['drug'])}</strong><br><span>{esc(row['drugId'])}</span></td>"
            f"<td><strong>{esc(row['target'])}</strong><br><span>{esc(row['protein'])}</span></td>"
            f"<td>{esc(row['credibilityTierZh'])}</td>"
            f"<td class='num'>{num(row['directionScore'], 3)}</td>"
            f"<td class='num'>{num(row['affinityScore'], 3)}</td>"
            f"<td>{esc(val(lit, 'literatureStatusZh'))}</td>"
            f"<td>{esc(candidate_decision(row, lit))}</td>"
            "</tr>"
        )
    return "".join(rows)


def build_candidate_cards(panel: pd.DataFrame, literature: pd.DataFrame) -> str:
    lookup = row_lit_lookup(literature)
    cards: list[str] = []
    for _, row in panel.iterrows():
        lit = lookup[(row["direction"], row["drug"], row["target"])]
        plan = target_plan(row)
        clinical_n = int_text(lit.get("clinicalTrialsDrugDiseaseCount"))
        pubmed_n = int_text(lit.get("pubmedDirectPairCount"))
        disease_pubmed_n = int_text(lit.get("pubmedDiseasePairCount"))
        original_rank = int(row["rank"]) if pd.notna(row.get("rank")) else int(row["reviewRankInDirection"])
        cards.append(
            f"""
            <section class="candidate">
              <div class="candidate-heading">
                <div>
                  <div class="kicker">{esc(row['directionLabelZh'])} Top {int(row['reviewRankInDirection'])} / website original rank {original_rank}</div>
                  <h2>{esc(row['drug'])} - {esc(row['target'])}</h2>
                  <p>{esc(row['protein'])} · {esc(row['proteinName'])}</p>
                </div>
                <div class="score-block">
                  <b>{num(row['directionScore'], 3)}</b>
                  <span>direction score</span>
                </div>
              </div>

              <div class="tag-row">
                <span>{esc(row['credibilityTierZh'])}</span>
                <span>{esc(val(row, 'expertNoveltyGroup'))}</span>
                <span>{esc(val(row, 'noveltyClass'))}</span>
                <span>known pair: {'yes' if is_truthy_number(row.get('knownDrugTargetPair')) else 'no'}</span>
              </div>

              <div class="grid two">
                <div class="box">
                  <h3>为什么入选</h3>
                  <p>{esc(compact(row.get('rationaleZh'), 620))}</p>
                  <ul>
                    <li>Affinity: {num(row.get('affinityScore'))}; Open Targets: {num(row.get('openTargetsScore'))}; TxGNN: {num(row.get('txgnnScore'))}; expert score: {num(row.get('expertReviewScore'), 1)}.</li>
                    <li>KG/疾病解释：{esc(compact(row.get('kgExplanationZh') or row.get('evidencePathZh'), 420))}</li>
                    <li>治疗领域/已知适应症：{esc(row.get('therapeuticArea'))}; {esc(compact(row.get('indication'), 260))}</li>
                  </ul>
                </div>
                <div class="box">
                  <h3>是否已有研究</h3>
                  <p><strong>{esc(val(lit, 'literatureStatusZh'))}</strong>：{esc(val(lit, 'literatureActionZh'))}</p>
                  <ul>
                    <li>PubMed drug-target 共现：{pubmed_n}; drug-target-disease 共现：{disease_pubmed_n}.</li>
                    <li>ClinicalTrials 药物-疾病方向：{clinical_n}; 该计数只说明药物在该疾病方向常见，不等于靶点机制已验证。</li>
                    <li>处理建议：{esc(candidate_decision(row, lit))}</li>
                  </ul>
                </div>
              </div>

              <div class="grid three">
                <div class="box">
                  <h3>通路与疾病上下文</h3>
                  <p>{esc(compact(row.get('evidenceSummaryZh'), 360))}</p>
                  <p>Reactome/CREEDS: {esc(val(row, 'pathwayDiseaseContextTier'))}; {esc(compact(row.get('reactomeTopPathways'), 280))}</p>
                </div>
                <div class="box">
                  <h3>签名、组织与模型</h3>
                  <p>CMap: {esc(val(row, 'cmapReversalTier'))}, raw {num(row.get('cmapBestRawReversal'))}, cell {esc(val(row, 'cmapBestCell'))}.</p>
                  <p>GTEx/HPA: {esc(val(row, 'gtexContextTier'))} / {esc(val(row, 'tissueContextTier'))}; DepMap: {esc(val(row, 'depmapDependencyTier'))}.</p>
                </div>
                <div class="box">
                  <h3>结构与 ADMET</h3>
                  <p>DiffDock/status: {num(row.get('diffdock'), 2)} / {esc(row.get('status'))}; structure: {esc(val(row, 'structureConfidenceTier'))}; pose: {esc(val(row, 'poseAuditStatus') or val(row, 'standardPoseValidationTier'))}.</p>
                  <p>ADMET: {esc(val(row, 'admetTier'))}; risk flags: {esc(compact(row.get('mlAdmetRiskFlags'), 220))}.</p>
                </div>
              </div>

              <div class="box feasibility">
                <h3>实验可行性与首轮验证设计</h3>
                <p><strong>靶点类别：</strong>{esc(plan['class'])}。<strong>推荐实验：</strong>{esc(row.get('assayModality') or plan['class'])}; {esc(row.get('assayRationale'))}</p>
                <ul>
                  <li>Primary readout：{esc(plan['primary'])}</li>
                  <li>正交验证：{esc(plan['orthogonal'])}</li>
                  <li>Counter-screen：{esc(plan['counter'])}</li>
                  <li>模型选择：{esc(plan['model'])}</li>
                  <li>Go/No-Go：{esc(plan['go'])}</li>
                </ul>
              </div>

              <div class="box risk">
                <h3>购买前必须核查</h3>
                <p>化合物可得性、盐型/母体形式、纯度、溶解度、临床游离暴露、目标蛋白在模型中的表达、调控方向、该药已知主靶点造成的 confounding、禁忌证和药物相互作用。</p>
                <p>专家讨论点：{esc(compact(row.get('expertDiscussionNoteZh') or row.get('nextStepZh'), 520))}</p>
              </div>
            </section>
            """
        )
    return "\n".join(cards)


def build_html(panel: pd.DataFrame, literature: pd.DataFrame, out_dir: Path) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    direction_counts = panel["directionLabelZh"].value_counts().to_dict()
    status_counts = literature["literatureStatusZh"].value_counts().to_dict()
    newish = int(literature["literatureStatusZh"].astype(str).str.contains("未见明显直接组合文献", na=False).sum())
    knownish = int(literature["literatureStatusZh"].astype(str).str.contains("已知机制|公开组合线索较多", regex=True, na=False).sum())
    riskish = int(literature["literatureStatusZh"].astype(str).str.contains("风险", na=False).sum())
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>BioMaster 候选逐项详细评审</title>
  <style>
    @font-face {{
      font-family: "Noto Sans CJK";
      src: url("file:///usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc");
    }}
    @font-face {{
      font-family: "Noto Sans CJK";
      src: url("file:///usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc");
      font-weight: 700;
    }}
    @page {{
      size: A4;
      margin: 13mm 12mm 14mm 12mm;
      @bottom-right {{ content: counter(page); color: #687385; font-size: 8pt; }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans CJK", sans-serif;
      color: #172033;
      font-size: 8.7pt;
      line-height: 1.45;
    }}
    h1, h2, h3, p {{ margin-top: 0; }}
    h1 {{ font-size: 26pt; line-height: 1.1; margin-bottom: 5mm; color: #163b5c; }}
    h2 {{ font-size: 14pt; margin-bottom: 1.5mm; color: #143b5f; }}
    h3 {{ font-size: 9.6pt; margin-bottom: 1.2mm; color: #244761; }}
    p {{ margin-bottom: 2mm; }}
    ul {{ margin: 0; padding-left: 4.5mm; }}
    li {{ margin-bottom: 1mm; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 7.2pt; line-height: 1.25; }}
    th {{ text-align: left; background: #edf4fa; color: #183d5d; padding: 1.6mm 1.2mm; border-bottom: 1px solid #b9c8d8; }}
    td {{ vertical-align: top; padding: 1.4mm 1.2mm; border-bottom: 1px solid #e2e8f0; }}
    td span, .muted {{ color: #667085; font-size: 7.2pt; }}
    .cover {{ min-height: 250mm; page-break-after: always; display: flex; flex-direction: column; justify-content: center; }}
    .kicker {{ text-transform: uppercase; letter-spacing: .06em; color: #0f6b7a; font-weight: 700; font-size: 7.4pt; margin-bottom: 1.2mm; }}
    .lead {{ font-size: 11.2pt; color: #344054; max-width: 165mm; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 3mm; margin: 7mm 0; }}
    .metric {{ background: #f5f8fb; border: 1px solid #d6e1eb; border-radius: 2mm; padding: 3mm; }}
    .metric b {{ display: block; font-size: 17pt; color: #0f5d93; line-height: 1.05; }}
    .metric span {{ color: #667085; font-size: 7.7pt; }}
    .section {{ page-break-after: always; }}
    .candidate {{ page-break-inside: avoid; break-inside: avoid; border-top: 2px solid #c6d7e8; padding-top: 3mm; margin-bottom: 5mm; }}
    .candidate-heading {{ display: grid; grid-template-columns: 1fr 27mm; gap: 3mm; align-items: start; }}
    .candidate-heading p {{ color: #667085; margin-bottom: 1.5mm; }}
    .score-block {{ text-align: center; background: #edf7f2; border: 1px solid #c8e2d4; border-radius: 2mm; padding: 2mm; }}
    .score-block b {{ display: block; font-size: 14pt; color: #12613c; }}
    .score-block span {{ color: #4b6356; font-size: 6.8pt; }}
    .tag-row {{ display: flex; flex-wrap: wrap; gap: 1.4mm; margin-bottom: 2.5mm; }}
    .tag-row span {{ border: 1px solid #d4dce7; background: #f8fafc; border-radius: 20px; padding: .8mm 1.8mm; font-size: 7pt; color: #344054; }}
    .grid {{ display: grid; gap: 2.3mm; margin-bottom: 2.3mm; }}
    .grid.two {{ grid-template-columns: 1fr 1fr; }}
    .grid.three {{ grid-template-columns: 1fr 1fr 1fr; }}
    .box {{ border: 1px solid #d8e0ea; background: #fff; border-radius: 2mm; padding: 2.5mm; }}
    .feasibility {{ background: #fbfcff; }}
    .risk {{ background: #fffaf3; border-color: #efd6ad; }}
    .num {{ text-align: right; white-space: nowrap; }}
    .note {{ background: #f7fbff; border-left: 4px solid #3478a6; padding: 3mm; margin: 4mm 0; }}
    .small {{ color: #667085; font-size: 7.4pt; }}
  </style>
</head>
<body>
  <section class="cover">
    <div class="kicker">BioMaster professor candidate review</div>
    <h1>疾病方向 Top10 候选逐项详细评审</h1>
    <p class="lead">本文件对当前展示网站中五个疾病方向各 Top10 候选逐条说明：为什么入选、是否已有研究、实验可行性、关键风险、购买前核查项和 Go/No-Go 原则。报告用于专家审阅和湿实验前聚焦，不构成生物学验证结论。</p>
    <div class="metric-grid">
      <div class="metric"><b>{len(panel)}</b><span>逐项候选</span></div>
      <div class="metric"><b>{len(direction_counts)}</b><span>疾病方向</span></div>
      <div class="metric"><b>{newish}</b><span>未见明显直接组合文献</span></div>
      <div class="metric"><b>{knownish + riskish}</b><span>建议标注/排重/降级</span></div>
    </div>
    <div class="note">
      <p><strong>阅读口径。</strong>“已知机制或阳性对照”不应作为新发现宣传；“公开组合线索较多”应先人工排重，若已有直接验证可忽略；“风险/负向或安全上下文”不建议直接进入首轮；“未见明显直接组合文献”仍需专利、中文文献、供应商和实验模型人工复核。</p>
      <p class="small">生成时间：{generated}。在线检索源：NCBI PubMed ESearch 与 ClinicalTrials.gov API v2；查询 URL 保存在同目录 literature audit CSV。</p>
    </div>
  </section>

  <section class="section">
    <div class="kicker">Scope and triage</div>
    <h2>候选总览</h2>
    <p>方向分布：{esc(json.dumps(direction_counts, ensure_ascii=False))}。已有研究/排重状态：{esc(json.dumps(status_counts, ensure_ascii=False))}。</p>
    <table>
      <thead>
        <tr>
          <th>#</th><th>方向</th><th>药物</th><th>靶点</th><th>可信度</th><th>Dir.</th><th>Affinity</th><th>已有研究状态</th><th>建议</th>
        </tr>
      </thead>
      <tbody>{build_summary_rows(panel, literature)}</tbody>
    </table>
  </section>

  {build_candidate_cards(panel, literature)}
</body>
</html>"""


def build_markdown(panel: pd.DataFrame, literature: pd.DataFrame) -> str:
    lookup = row_lit_lookup(literature)
    lines = [
        "# BioMaster 疾病方向 Top10 候选逐项详细评审",
        "",
        "本 Markdown 与 PDF 内容一致，用于后续人工编辑。",
        "",
        "## 候选总览",
        "",
        "| # | 方向 | 候选 | 可信度 | 方向分 | 亲和分 | 已有研究状态 | 建议 |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for _, row in panel.iterrows():
        lit = lookup[(row["direction"], row["drug"], row["target"])]
        lines.append(
            f"| {int(row['reportRank'])} | {row['directionLabelZh']} Top {int(row['reviewRankInDirection'])} | "
            f"{row['drug']} - {row['target']} | {text(row.get('credibilityTierZh'))} | "
            f"{num(row.get('directionScore'))} | {num(row.get('affinityScore'))} | "
            f"{lit['literatureStatusZh']} | {candidate_decision(row, lit)} |"
        )
    lines.append("\n## 逐项评审\n")
    for _, row in panel.iterrows():
        lit = lookup[(row["direction"], row["drug"], row["target"])]
        plan = target_plan(row)
        lines.extend(
            [
                f"### {int(row['reportRank'])}. {row['drug']} - {row['target']} ({row['directionLabelZh']} Top {int(row['reviewRankInDirection'])})",
                "",
                f"- 身份：{row['protein']}，{row['proteinName']}；网站原始 rank {int(row['rank'])}；known pair: {'yes' if is_truthy_number(row.get('knownDrugTargetPair')) else 'no'}；noveltyClass: {text(row.get('noveltyClass'))}。",
                f"- 入选理由：{text(row.get('rationaleZh'))}",
                f"- 模型证据：direction {num(row.get('directionScore'))}；affinity {num(row.get('affinityScore'))}；Open Targets {num(row.get('openTargetsScore'))}；TxGNN {num(row.get('txgnnScore'))}；expert score {num(row.get('expertReviewScore'), 1)}。",
                f"- 已有研究：PubMed drug-target {int_text(lit.get('pubmedDirectPairCount'))}；drug-target-disease {int_text(lit.get('pubmedDiseasePairCount'))}；ClinicalTrials 药物-疾病方向 {int_text(lit.get('clinicalTrialsDrugDiseaseCount'))}；结论：{lit['literatureStatusZh']}，{lit['literatureActionZh']}",
                f"- 通路/签名/组织：CMap {text(row.get('cmapReversalTier'))} raw {num(row.get('cmapBestRawReversal'))}；Reactome/CREEDS {text(row.get('pathwayDiseaseContextTier'))}；GTEx/HPA {text(row.get('gtexContextTier'))}/{text(row.get('tissueContextTier'))}；DepMap {text(row.get('depmapDependencyTier'))}。",
                f"- 结构/ADMET：DiffDock {num(row.get('diffdock'), 2)}，status {text(row.get('status'))}；structure {text(row.get('structureConfidenceTier'))}；pose {text(row.get('poseAuditStatus') or row.get('standardPoseValidationTier'))}；ADMET {text(row.get('admetTier'))}；risk flags {text(row.get('mlAdmetRiskFlags'))}。",
                f"- 实验可行性：{plan['class']}；primary readout：{plan['primary']}；正交验证：{plan['orthogonal']}；counter-screen：{plan['counter']}；模型：{plan['model']}；Go/No-Go：{plan['go']}",
                f"- 处理建议：{candidate_decision(row, lit)}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def publish(root: Path, pdf_path: Path) -> Path:
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    published = assets / "professor-candidate-detailed-review.pdf"
    shutil.copy2(pdf_path, published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--refresh-literature", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_pool(root)
    panel = select_direction_top10(candidates)
    literature = build_literature_audit(panel, out_dir, refresh=args.refresh_literature)

    html_text = build_html(panel, literature, out_dir)
    md_text = build_markdown(panel, literature)
    html_path = out_dir / "PROFESSOR_CANDIDATE_DETAILED_REVIEW.html"
    md_path = out_dir / "PROFESSOR_CANDIDATE_DETAILED_REVIEW.md"
    pdf_path = out_dir / "PROFESSOR_CANDIDATE_DETAILED_REVIEW.pdf"
    html_path.write_text(html_text, encoding="utf-8")
    md_path.write_text(md_text, encoding="utf-8")
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    published = publish(root, pdf_path)

    summary = {
        "createdUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidateRows": int(len(panel)),
        "directionCounts": {str(k): int(v) for k, v in panel["direction"].value_counts().to_dict().items()},
        "literatureStatusCounts": {
            str(k): int(v) for k, v in literature["literatureStatusZh"].value_counts().to_dict().items()
        },
        "outputs": {
            "html": str(html_path.relative_to(root)),
            "markdown": str(md_path.relative_to(root)),
            "pdf": str(pdf_path.relative_to(root)),
            "literatureAuditCsv": str(
                (out_dir / "professor_candidate_detailed_review_literature_audit.csv").relative_to(root)
            ),
            "publishedPdf": str(published.relative_to(root)),
        },
    }
    (out_dir / "professor_candidate_detailed_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
