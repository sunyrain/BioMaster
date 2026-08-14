#!/usr/bin/env python3
"""Build the new-only ChEMBL 37 human single-protein target universe."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from target_universe_ch37_common import (
    BOOL_OT_COLUMNS,
    aggregate_opentargets,
    attach_p2rank,
    classify_assay_lane,
    index_local_alphafold,
    load_classification,
    load_mechanism_summary,
    load_official_targets,
    select_exact_structures,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/target_universe_ch37_v2.yaml"
DB = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OT = ROOT / "outputs/target_catalog_quality_audit_v1/opentargets_26_06_project_standard_target_flags.csv"
AF_DIR = ROOT / "data/processed/alphafold_receptors_v6"
OUTDIR = ROOT / "outputs/target_universe_ch37_v2"
P2RANK_DIR = OUTDIR / "p2rank_all_exact_v2"
CALIBRATION = OUTDIR / "chembl37_calibration_all888_v2/TARGET_ALL888_CALIBRATION_COVERAGE_V2.csv"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def classify_master(master: pd.DataFrame) -> pd.DataFrame:
    output = master.copy()
    output["is_gpcr"] = output["chembl_gpcr"].fillna(False).astype(bool) | output[
        "ot_class_contains_gpcr"
    ].fillna(False).astype(bool)
    output["opentargets_matched"] = output["ot_id"].fillna("").astype(str).str.strip().ne("")
    output["assay_lane"] = output.apply(classify_assay_lane, axis=1)

    output["evidence_class"] = "NO_DIRECT_SM_EVIDENCE_NON_GPCR"
    output.loc[output["is_gpcr"] & output["small_molecule_moa"], "evidence_class"] = "SM_MOA_GPCR"
    output.loc[output["is_gpcr"] & ~output["small_molecule_moa"], "evidence_class"] = "GPCR_WITHOUT_SM_MOA"
    output.loc[~output["is_gpcr"] & output["small_molecule_moa"], "evidence_class"] = "SM_MOA_NON_GPCR"
    output.loc[
        ~output["is_gpcr"] & ~output["small_molecule_moa"] & output["ot_project_standard_direct_sm"],
        "evidence_class",
    ] = "OT_DIRECT_EXTENSION_NON_GPCR"

    output["set_all_official"] = True
    output["set_sequence_dta_all"] = output["sequence"].notna() & output["sequence_length"].gt(0)
    output["set_gpcr_all"] = output["is_gpcr"]
    output["set_non_gpcr_all"] = ~output["is_gpcr"]
    output["set_small_molecule_moa_all"] = output["small_molecule_moa"].astype(bool)
    output["set_small_molecule_moa_non_gpcr"] = output["small_molecule_moa"] & ~output["is_gpcr"]
    output["set_approved_sm_moa_all"] = output["approved_small_molecule_moa"].astype(bool)
    output["set_approved_sm_moa_non_gpcr"] = output["approved_small_molecule_moa"] & ~output["is_gpcr"]
    output["set_direct_sm_broad_non_gpcr"] = ~output["is_gpcr"] & (
        output["small_molecule_moa"] | output["ot_project_standard_direct_sm"]
    )
    output["set_standard_biochemical_class"] = output["assay_lane"].isin(
        ["KINASE_BIOCHEMICAL", "ENZYME_BIOCHEMICAL", "NUCLEAR_EPIGENETIC_DOMAIN"]
    )
    output["set_sm_moa_non_gpcr_standard_biochemical"] = (
        output["set_small_molecule_moa_non_gpcr"] & output["set_standard_biochemical_class"]
    )
    return output


def attach_new_calibration(master: pd.DataFrame) -> pd.DataFrame:
    if not CALIBRATION.is_file():
        master["calibration_status"] = "NOT_COMPUTED"
        master["calibration_8x8"] = False
        return master
    calibration = pd.read_csv(CALIBRATION, low_memory=False)
    calibration = calibration.drop(
        columns=[column for column in ["gene_symbol", "assay_lane"] if column in calibration],
        errors="ignore",
    )
    output = master.merge(calibration, on="target_chembl_id", how="left", validate="one_to_one")
    output["calibration_status"] = output["target_calibration_tier"].fillna("T5_SPARSE_OR_NONE")
    output["calibration_8x8"] = (
        pd.to_numeric(output["positive_compounds"], errors="coerce").fillna(0).ge(8)
        & pd.to_numeric(output["negative_compounds"], errors="coerce").fillna(0).ge(8)
    )
    return output


def finalize_sets(master: pd.DataFrame) -> pd.DataFrame:
    output = master.copy()
    output["pocket_local_confident"] = (
        pd.to_numeric(output.get("pocket_mean_plddt"), errors="coerce").ge(70)
        & pd.to_numeric(output.get("pocket_residues_plddt_ge70_pct"), errors="coerce").ge(70)
    )
    output["structure_ready_permissive"] = output["af_exact_sequence_model"].fillna(False).astype(bool) & output[
        "p2rank_tier"
    ].fillna("").isin(["A_HIGH_CONFIDENCE", "B_MODERATE_CONFIDENCE"])
    output["structure_ready_strict"] = (
        output["af_exact_sequence_model"].fillna(False).astype(bool)
        & output["p2rank_tier"].fillna("").eq("A_HIGH_CONFIDENCE")
        & output["pocket_local_confident"]
    )
    output["structure_ready"] = output["structure_ready_permissive"]
    output["set_exact_structure_all"] = output["af_exact_sequence_model"].fillna(False).astype(bool)
    output["set_structure_ready_all"] = output["structure_ready_permissive"]
    output["set_structure_strict_all"] = output["structure_ready_strict"]
    output["set_structure_ready_sm_moa_all"] = output["structure_ready_permissive"] & output["set_small_molecule_moa_all"]
    output["set_structure_strict_sm_moa_all"] = output["structure_ready_strict"] & output["set_small_molecule_moa_all"]
    output["set_structure_ready_sm_moa_non_gpcr"] = output["structure_ready_permissive"] & output[
        "set_small_molecule_moa_non_gpcr"
    ]
    output["set_structure_strict_sm_moa_non_gpcr"] = output["structure_ready_strict"] & output[
        "set_small_molecule_moa_non_gpcr"
    ]
    output["set_structure_ready_direct_sm_broad_non_gpcr"] = output["structure_ready_permissive"] & output[
        "set_direct_sm_broad_non_gpcr"
    ]
    output["set_calibration_8x8_all"] = output["calibration_8x8"].fillna(False).astype(bool)
    output["set_structure_and_calibration_all"] = output["structure_ready_permissive"] & output["calibration_8x8"]
    output["set_structure_and_calibration_sm_moa_non_gpcr"] = (
        output["structure_ready_permissive"] & output["calibration_8x8"] & output["set_small_molecule_moa_non_gpcr"]
    )
    output["set_structure_strict_and_calibration_sm_moa_non_gpcr"] = (
        output["structure_ready_strict"] & output["calibration_8x8"] & output["set_small_molecule_moa_non_gpcr"]
    )
    output["set_structure_strict_and_calibration_all"] = (
        output["structure_ready_strict"] & output["calibration_8x8"]
    )
    output["set_structure_strict_direct_sm_broad_non_gpcr"] = (
        output["structure_ready_strict"] & output["set_direct_sm_broad_non_gpcr"]
    )
    output["set_structure_strict_and_calibration_direct_sm_broad_non_gpcr"] = (
        output["structure_ready_strict"]
        & output["calibration_8x8"]
        & output["set_direct_sm_broad_non_gpcr"]
    )
    output["assay_confirmation_status"] = "NOT_LAB_OR_VENDOR_CONFIRMED"
    return output


def count_row(name: str, description: str, mask: pd.Series, total: int) -> dict[str, Any]:
    count = int(mask.fillna(False).astype(bool).sum())
    return {
        "集合": name,
        "定义": description,
        "数量": count,
        "占888比例_pct": round(count / total * 100, 2),
    }


def build_count_tables(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total = len(master)
    collection = pd.DataFrame([
        count_row("官方全集", "ChEMBL 37 人源 SINGLE PROTEIN 且存在 MoA", master["set_all_official"], total),
        count_row("序列DTA全集", "官方全集且有完整氨基酸序列", master["set_sequence_dta_all"], total),
        count_row("GPCR全集", "ChEMBL和Open Targets一致标记为GPCR", master["set_gpcr_all"], total),
        count_row("非GPCR全集", "官方全集排除GPCR", master["set_non_gpcr_all"], total),
        count_row("小分子MoA全集", "至少一条 ChEMBL Small molecule MoA", master["set_small_molecule_moa_all"], total),
        count_row("非GPCR小分子MoA", "小分子MoA且非GPCR", master["set_small_molecule_moa_non_gpcr"], total),
        count_row("获批小分子MoA全集", "至少一个Phase 4小分子具有ChEMBL MoA", master["set_approved_sm_moa_all"], total),
        count_row("非GPCR获批小分子MoA", "获批小分子MoA且非GPCR", master["set_approved_sm_moa_non_gpcr"], total),
        count_row("非GPCR广义直接小分子", "非GPCR且满足ChEMBL小分子MoA或OT direct-SM", master["set_direct_sm_broad_non_gpcr"], total),
        count_row("标准生化体系类", "激酶、酶、核内/表观遗传蛋白", master["set_standard_biochemical_class"], total),
        count_row("非GPCR小分子MoA标准生化体系", "非GPCR小分子MoA且属于标准生化体系类", master["set_sm_moa_non_gpcr_standard_biochemical"], total),
        count_row("精确序列结构全集", "AlphaFold模型与ChEMBL序列SHA-256完全一致", master["set_exact_structure_all"], total),
        count_row("全体宽松结构就绪", "精确序列结构且项目P2Rank分层A/B", master["set_structure_ready_all"], total),
        count_row("全体严格结构就绪", "精确序列结构、项目P2Rank分层A且口袋局部pLDDT达标", master["set_structure_strict_all"], total),
        count_row("小分子MoA宽松结构就绪", "小分子MoA全集与宽松结构就绪交集", master["set_structure_ready_sm_moa_all"], total),
        count_row("小分子MoA严格结构就绪", "小分子MoA全集与严格结构就绪交集", master["set_structure_strict_sm_moa_all"], total),
        count_row("非GPCR小分子MoA宽松结构就绪", "非GPCR小分子MoA与宽松结构就绪交集", master["set_structure_ready_sm_moa_non_gpcr"], total),
        count_row("非GPCR小分子MoA严格结构就绪", "非GPCR小分子MoA与严格结构就绪交集", master["set_structure_strict_sm_moa_non_gpcr"], total),
        count_row("非GPCR广义直接小分子宽松结构就绪", "非GPCR广义直接小分子与宽松结构就绪交集", master["set_structure_ready_direct_sm_broad_non_gpcr"], total),
        count_row("非GPCR广义直接小分子严格结构就绪", "非GPCR广义直接小分子与严格结构就绪交集", master["set_structure_strict_direct_sm_broad_non_gpcr"], total),
        count_row("历史正负至少8+8", "严格binding历史阳性和低活性/inactive均至少8个", master["set_calibration_8x8_all"], total),
        count_row("全体宽松结构且校准就绪", "宽松结构就绪与历史8+8交集", master["set_structure_and_calibration_all"], total),
        count_row("全体严格结构且校准就绪", "严格结构就绪与历史8+8交集", master["set_structure_strict_and_calibration_all"], total),
        count_row("非GPCR小分子MoA宽松结构且校准", "非GPCR小分子MoA、宽松结构就绪和历史8+8交集", master["set_structure_and_calibration_sm_moa_non_gpcr"], total),
        count_row("非GPCR小分子MoA严格结构且校准", "非GPCR小分子MoA、严格结构就绪和历史8+8交集", master["set_structure_strict_and_calibration_sm_moa_non_gpcr"], total),
        count_row("非GPCR广义直接小分子严格结构且校准", "非GPCR广义直接小分子、严格结构就绪和历史8+8交集", master["set_structure_strict_and_calibration_direct_sm_broad_non_gpcr"], total),
    ])

    def count_table(column: str, category_name: str) -> pd.DataFrame:
        frame = master[column].fillna("MISSING").value_counts().rename_axis(category_name).reset_index(name="数量")
        frame["占888比例_pct"] = (frame["数量"] / total * 100).round(2)
        return frame

    evidence_labels = {
        "SM_MOA_NON_GPCR": "ChEMBL小分子MoA-非GPCR",
        "SM_MOA_GPCR": "ChEMBL小分子MoA-GPCR",
        "OT_DIRECT_EXTENSION_NON_GPCR": "Open Targets直接小分子扩展-非GPCR",
        "GPCR_WITHOUT_SM_MOA": "无ChEMBL小分子MoA-GPCR",
        "NO_DIRECT_SM_EVIDENCE_NON_GPCR": "无直接小分子证据-非GPCR",
    }
    assay_labels = {
        "ENZYME_BIOCHEMICAL": "酶-生化体系",
        "KINASE_BIOCHEMICAL": "激酶-生化体系",
        "GPCR_MEMBRANE_ASSAY": "GPCR-膜蛋白体系",
        "ION_CHANNEL_FUNCTIONAL": "离子通道-功能体系",
        "TRANSPORTER_MEMBRANE_FUNCTIONAL": "转运体-膜蛋白功能体系",
        "NUCLEAR_EPIGENETIC_DOMAIN": "核内/表观遗传-结构域或功能体系",
        "NON_GPCR_MEMBRANE_SPECIAL": "非GPCR膜蛋白-专门体系",
        "EXTRACELLULAR_SPECIAL": "胞外/分泌/黏附-专门体系",
        "NONCANONICAL_REVIEW": "非常规类别-逐靶点复核",
    }
    calibration_labels = {
        "T1_RICH_POSITIVE_NEGATIVE": "T1-正负数据丰富",
        "T2_ADEQUATE_POSITIVE_NEGATIVE": "T2-正负数据较充分",
        "T3_MINIMUM_POSITIVE_NEGATIVE": "T3-达到最低8+8",
        "T4_POSITIVE_ONLY": "T4-阳性充分但阴性不足",
        "T5_SPARSE_OR_NONE": "T5-数据稀疏或无数据",
        "NOT_COMPUTED": "未计算",
    }
    pocket_labels = {
        "A_HIGH_CONFIDENCE": "A-项目高置信口袋",
        "B_MODERATE_CONFIDENCE": "B-项目中等置信口袋",
        "C_WEAK_REVIEW": "C-项目弱口袋需复核",
        "D_LOW_CONFIDENCE": "D-项目低置信口袋",
        "D_NO_POCKET": "D-未识别口袋",
        "NOT_RUN": "未运行-无精确结构",
    }

    modality_fields = {
        "小分子": "moa_has_small_molecule",
        "抗体": "moa_has_antibody",
        "蛋白药": "moa_has_protein",
        "抗体偶联药物": "moa_has_antibody_drug_conjugate",
        "寡核苷酸": "moa_has_oligonucleotide",
        "寡糖": "moa_has_oligosaccharide",
        "基因疗法": "moa_has_gene",
        "酶制剂": "moa_has_enzyme",
        "细胞疗法": "moa_has_cell",
        "疫苗组分": "moa_has_vaccine_component",
        "分子类型未知": "moa_has_unknown",
    }
    modality = pd.DataFrame([
        {
            "机制分子类型": name,
            "靶点数量_非互斥": int(master[column].fillna(False).astype(bool).sum()),
            "占888比例_pct": round(master[column].fillna(False).astype(bool).sum() / total * 100, 2),
        }
        for name, column in modality_fields.items()
    ]).sort_values("靶点数量_非互斥", ascending=False)

    structure = pd.DataFrame([
        count_row("序列完全一致AlphaFold", "本地模型序列SHA-256与ChEMBL序列一致", master["af_exact_sequence_model"], total),
        count_row("P2Rank A", "高置信口袋", master["p2rank_tier"].eq("A_HIGH_CONFIDENCE"), total),
        count_row("P2Rank B", "中等置信口袋", master["p2rank_tier"].eq("B_MODERATE_CONFIDENCE"), total),
        count_row("P2Rank C", "弱口袋，需复核", master["p2rank_tier"].eq("C_WEAK_REVIEW"), total),
        count_row("P2Rank D/无口袋", "低置信或未识别口袋", master["p2rank_tier"].astype(str).str.startswith("D_"), total),
        count_row("口袋局部pLDDT达标", "口袋平均pLDDT>=70且至少70%残基pLDDT>=70", master["pocket_local_confident"], total),
        count_row("宽松结构就绪", "序列一致AF且项目P2Rank分层A/B", master["structure_ready_permissive"], total),
        count_row("严格结构就绪", "序列一致AF、项目P2Rank分层A且口袋局部pLDDT达标", master["structure_ready_strict"], total),
    ])
    evidence_coverage = pd.DataFrame([
        count_row("Open Targets可映射", "按官方gene symbol匹配Open Targets 26.06", master["opentargets_matched"], total),
        count_row("ChEMBL小分子MoA", "至少一条Small molecule MoA", master["small_molecule_moa"], total),
        count_row("ChEMBL获批小分子MoA", "至少一个Phase 4小分子MoA", master["approved_small_molecule_moa"], total),
        count_row("Open Targets直接小分子", "Open Targets project_standard_direct_sm", master["ot_project_standard_direct_sm"], total),
        count_row("精确序列AlphaFold", "与ChEMBL序列SHA-256完全一致", master["af_exact_sequence_model"], total),
        count_row("存在严格binding记录", "ChEMBL严格binding pair数量大于0", master["strict_compound_pairs"].gt(0), total),
        count_row("存在严格阳性", "至少一个均值pChEMBL>=6且无冲突的化合物", master["positive_compounds"].gt(0), total),
        count_row("存在低活性或明确inactive", "至少一个均值pChEMBL<=5或明确inactive且无冲突的化合物", master["negative_compounds"].gt(0), total),
    ])
    protein_class_counts: dict[str, int] = {}
    for value in master["target_class_l1"].fillna(""):
        for token in str(value).split(";"):
            if token:
                protein_class_counts[token] = protein_class_counts.get(token, 0) + 1
    protein_class_labels = {
        "Enzyme": "酶", "Membrane receptor": "膜受体", "Secreted protein": "分泌蛋白",
        "Unclassified protein": "未分类蛋白", "Ion channel": "离子通道", "Transporter": "转运体",
        "Transcription factor": "转录因子", "Surface antigen": "表面抗原", "Adhesion": "黏附蛋白",
        "Epigenetic regulator": "表观遗传调控蛋白", "Other cytosolic protein": "其他胞质蛋白",
        "Structural protein": "结构蛋白", "Other membrane protein": "其他膜蛋白",
        "Auxiliary transport protein": "辅助转运蛋白", "Other nuclear protein": "其他核蛋白",
    }
    protein_class = pd.DataFrame([
        {
            "ChEMBL一级类别_非互斥": protein_class_labels.get(name, name),
            "数量": count,
            "占888比例_pct": round(count / total * 100, 2),
        }
        for name, count in protein_class_counts.items()
    ]).sort_values("数量", ascending=False, kind="mergesort")
    evidence_class = count_table("evidence_class", "证据分类")
    evidence_class["证据分类"] = evidence_class["证据分类"].map(evidence_labels).fillna(evidence_class["证据分类"])
    assay_class = count_table("assay_lane", "实验体系")
    assay_class["实验体系"] = assay_class["实验体系"].map(assay_labels).fillna(assay_class["实验体系"])
    calibration_class = count_table("calibration_status", "历史数据等级")
    calibration_class["历史数据等级"] = calibration_class["历史数据等级"].map(calibration_labels).fillna(calibration_class["历史数据等级"])
    p2rank_class = count_table("p2rank_tier", "项目口袋等级")
    p2rank_class["项目口袋等级"] = p2rank_class["项目口袋等级"].map(pocket_labels).fillna(p2rank_class["项目口袋等级"])
    return {
        "可用集合": collection,
        "证据覆盖": evidence_coverage,
        "证据互斥分类": evidence_class,
        "实验体系分类": assay_class,
        "ChEMBL一级类别": protein_class,
        "机制分子类型": modality,
        "结构分类": structure,
        "校准分类": calibration_class,
        "P2Rank分类": p2rank_class,
    }


def build_teacher_table(master: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "target_chembl_id": "ChEMBL靶点ID", "gene_symbol": "基因", "target_name": "靶点名称",
        "uniprot_accession": "UniProt编号", "sequence_length": "序列长度",
        "target_class_l1": "一级类别", "assay_lane": "实验体系分类", "is_gpcr": "是否GPCR",
        "small_molecule_moa": "ChEMBL小分子MoA", "approved_small_molecule_moa": "已有获批小分子MoA",
        "ot_project_standard_direct_sm": "OpenTargets直接小分子证据", "evidence_class": "证据互斥分类",
        "af_exact_sequence_model": "序列一致AlphaFold", "af_mean_plddt": "平均pLDDT",
        "p2rank_tier": "P2Rank等级", "structure_ready": "结构就绪",
        "pocket_mean_plddt": "口袋平均pLDDT", "pocket_residues_plddt_ge70_pct": "口袋残基pLDDT达标比例",
        "structure_ready_permissive": "宽松结构就绪", "structure_ready_strict": "严格结构就绪",
        "calibration_status": "历史数据等级", "positive_compounds": "历史阳性数",
        "negative_compounds": "历史低活性或inactive数", "calibration_8x8": "历史正负至少8+8",
        "assay_confirmation_status": "实验确认状态",
    }
    return master[[column for column in columns if column in master]].rename(columns=columns)


def write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column[:200]) + 2, 42)
                sheet.column_dimensions[column[0].column_letter].width = max(width, 10)


def write_summary_report(outdir: Path, tables: dict[str, pd.DataFrame], master: pd.DataFrame) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        columns = [str(column) for column in frame.columns]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
        return "\n".join(lines)

    text = f"""# ChEMBL 37 人源单蛋白靶点全集 V2

## 1. 数据口径

唯一权威起点为 ChEMBL 37 中物种为 Homo sapiens、类型为 SINGLE PROTEIN、且至少存在一条 drug mechanism/MoA 记录的靶点。该口径得到 888 个 target ID、888 个基因、888 个 UniProt 组件和 888 条唯一氨基酸序列。

本报告只汇总新 888 体系的数量与比例。不同“可用集合”服务于不同计算目的，不将小分子证据、GPCR、结构口袋和历史数据压成一个总分。

## 2. 不同口径的可用集合

{md_table(tables['可用集合'])}

## 3. 互斥证据分类

{md_table(tables['证据互斥分类'])}

五类互斥分类合计必须为 888。ChEMBL 小分子 MoA 是最严格的小分子机制锚点；Open Targets direct-SM 仅形成扩展集合；GPCR 单列而不删除。

## 4. 证据覆盖

{md_table(tables['证据覆盖'])}

## 5. 蛋白家族与实验体系

{md_table(tables['实验体系分类'])}

实验体系分类表示技术路线，不表示实验室已具备 construct、蛋白、对照或成熟 protocol。

ChEMBL一级类别为非互斥标签：

{md_table(tables['ChEMBL一级类别'])}

## 6. ChEMBL机制分子类型

{md_table(tables['机制分子类型'])}

该表为非互斥统计；同一靶点可同时存在小分子、抗体或蛋白机制记录，因此各行不能相加为 888。

## 7. 结构与口袋

{md_table(tables['结构分类'])}

P2Rank只输出口袋分数、概率和残基，A/B/C/D是本项目为组织结果而定义的等级，并非P2Rank官方分类。宽松结构口径要求精确序列模型和项目A/B；严格结构口径进一步要求项目A及口袋平均pLDDT不低于70、且至少70%的口袋残基pLDDT不低于70。二者都只表示靶点可进入结构计算，不证明任何具体药物与该靶点结合。

## 8. 历史binding数据充足度

{md_table(tables['校准分类'])}

历史阴性仅包括定量低活性或明确 inactive；未测和数据库未记录不作为阴性。该等级只衡量历史数据量，不代表 cold-scaffold、远程发现或前瞻命中率。

## 9. 正式使用

1. 全序列探索使用“序列 DTA 全集”。
2. 有小分子机制先验的全家族筛选使用“小分子 MoA 全集”。
3. 不开展 GPCR 的主筛使用“非 GPCR 小分子 MoA”。
4. 接受 Open Targets 扩展时使用“非 GPCR 广义直接小分子”。
5. 初筛Docking可使用“宽松结构就绪”，高置信结构计算使用“严格结构就绪”。
6. 需要回顾性局部校准时再叠加“历史正负至少 8+8”。
7. 任何集合均不等于实验室 assay-ready；实验启动仍需独立确认。
"""
    (outdir / "TARGET_UNIVERSE_SUMMARY_COUNTS_ZH_V2.md").write_text(text, encoding="utf-8")


def build(outdir: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        master = load_official_targets(connection)
        master = master.merge(load_mechanism_summary(connection), on="target_chembl_id", validate="one_to_one")
        master = master.merge(
            load_classification(connection, set(master["component_id"].astype(int))),
            on="component_id", validate="one_to_one",
        )
    finally:
        connection.close()
    master = master.merge(aggregate_opentargets(OT), on="gene_symbol", how="left", validate="many_to_one")
    for column in [f"ot_{value}" for value in BOOL_OT_COLUMNS]:
        master[column] = master[column].fillna(False).astype(bool)
    master = classify_master(master)

    af_index = index_local_alphafold(AF_DIR)
    master = master.merge(select_exact_structures(master, af_index), on="target_chembl_id", validate="one_to_one")
    master = attach_p2rank(master, [P2RANK_DIR])
    master = attach_new_calibration(master)
    master = finalize_sets(master)
    master = master.sort_values(["evidence_class", "assay_lane", "gene_symbol"], kind="mergesort").reset_index(drop=True)

    expected = config["authoritative_scope"]
    checks = {
        "targets_888": len(master) == int(expected["expected_targets"]),
        "genes_888": master["gene_symbol"].nunique() == int(expected["expected_genes"]),
        "accessions_888": master["uniprot_accession"].nunique() == int(expected["expected_accessions"]),
        "sequences_888": master["sequence_sha256"].nunique() == int(expected["expected_sequences"]),
        "target_ids_unique": master["target_chembl_id"].is_unique,
        "evidence_class_exhaustive": master["evidence_class"].notna().all(),
        "all_sequences_present": master["set_sequence_dta_all"].all(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Target universe checks failed: {checks}")

    tables = build_count_tables(master)
    teacher = build_teacher_table(master)
    master.to_csv(outdir / "TARGET_UNIVERSE_OFFICIAL_888_V2.csv", index=False)
    teacher.to_csv(outdir / "TARGET_UNIVERSE_REVIEW_ZH_888_V2.csv", index=False)

    set_outputs = {
        "TARGET_SET_SEQUENCE_DTA_ALL_V2.csv": "set_sequence_dta_all",
        "TARGET_SET_GPCR_ALL_V2.csv": "set_gpcr_all",
        "TARGET_SET_NON_GPCR_ALL_V2.csv": "set_non_gpcr_all",
        "TARGET_SET_SMALL_MOLECULE_MOA_ALL_V2.csv": "set_small_molecule_moa_all",
        "TARGET_SET_SMALL_MOLECULE_MOA_NON_GPCR_V2.csv": "set_small_molecule_moa_non_gpcr",
        "TARGET_SET_APPROVED_SM_MOA_ALL_V2.csv": "set_approved_sm_moa_all",
        "TARGET_SET_APPROVED_SM_MOA_NON_GPCR_V2.csv": "set_approved_sm_moa_non_gpcr",
        "TARGET_SET_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_direct_sm_broad_non_gpcr",
        "TARGET_SET_STANDARD_BIOCHEMICAL_CLASS_V2.csv": "set_standard_biochemical_class",
        "TARGET_SET_SM_MOA_NON_GPCR_STANDARD_BIOCHEMICAL_V2.csv": "set_sm_moa_non_gpcr_standard_biochemical",
        "TARGET_SET_EXACT_STRUCTURE_ALL_V2.csv": "set_exact_structure_all",
        "TARGET_SET_STRUCTURE_READY_ALL_V2.csv": "set_structure_ready_all",
        "TARGET_SET_STRUCTURE_STRICT_ALL_V2.csv": "set_structure_strict_all",
        "TARGET_SET_STRUCTURE_READY_SM_MOA_ALL_V2.csv": "set_structure_ready_sm_moa_all",
        "TARGET_SET_STRUCTURE_STRICT_SM_MOA_ALL_V2.csv": "set_structure_strict_sm_moa_all",
        "TARGET_SET_STRUCTURE_READY_SM_MOA_NON_GPCR_V2.csv": "set_structure_ready_sm_moa_non_gpcr",
        "TARGET_SET_STRUCTURE_STRICT_SM_MOA_NON_GPCR_V2.csv": "set_structure_strict_sm_moa_non_gpcr",
        "TARGET_SET_STRUCTURE_STRICT_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_structure_strict_direct_sm_broad_non_gpcr",
        "TARGET_SET_CALIBRATION_8X8_ALL_V2.csv": "set_calibration_8x8_all",
        "TARGET_SET_STRUCTURE_AND_CALIBRATION_SM_MOA_NON_GPCR_V2.csv": "set_structure_and_calibration_sm_moa_non_gpcr",
        "TARGET_SET_STRUCTURE_STRICT_AND_CALIBRATION_SM_MOA_NON_GPCR_V2.csv": "set_structure_strict_and_calibration_sm_moa_non_gpcr",
        "TARGET_SET_STRUCTURE_STRICT_AND_CALIBRATION_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_structure_strict_and_calibration_direct_sm_broad_non_gpcr",
    }
    for filename, flag in set_outputs.items():
        master[master[flag]].to_csv(outdir / filename, index=False)
    for name, table in tables.items():
        table.to_csv(outdir / f"COUNTS_{name}_V2.csv", index=False)

    write_excel(
        outdir / "CHEMBL37_TARGET_UNIVERSE_V2.xlsx",
        {"靶点全集888": teacher, **tables},
    )
    write_excel(
        outdir / "CHEMBL37_TARGET_UNIVERSE_COUNTS_ONLY_ZH_V2.xlsx",
        tables,
    )
    write_summary_report(outdir, tables, master)
    summary = {
        "package_name": config["package_name"],
        "package_version": config["package_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "collection_counts": tables["可用集合"].to_dict(orient="records"),
        "evidence_class_counts": tables["证据互斥分类"].to_dict(orient="records"),
        "assay_lane_counts": tables["实验体系分类"].to_dict(orient="records"),
    }
    (outdir / "TARGET_UNIVERSE_SUMMARY_V2.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
