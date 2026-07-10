#!/usr/bin/env python3
"""Audit drug/target scope and known-control attrition for the formal funnel."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import bool_series, file_sha256  # noqa: E402
from build_full_project_universe_v3 import DIRECT_ACTIONS, split_genes  # noqa: E402


DRUGS = ROOT / "data/processed/drug_library_active_moiety_v4.csv"
SEQUENCES = ROOT / "outputs/full_conplex_active_moiety_v4/protein_sequence_representatives.csv"
ANCHORS = ROOT / "outputs/chembl_moa_enhanced_information_package_v1/chembl_moa_anchor_gene_table_v2.csv"
PROJECT_POOL = ROOT / "outputs/chembl_moa_enhanced_information_package_v1/candidate_pool_106k_enhanced_scored.csv"
MECHANISMS = ROOT / "outputs/target_catalog_quality_audit_v1/chembl37_mechanisms_with_human_target_map.csv"
KNOWN_PROJECT = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v4/known_control_calibration_v4.csv"
DRUG_MANIFEST = ROOT / "configs/project_drugs_v4.csv"
TARGET_MANIFEST = ROOT / "configs/project_targets_v4.csv"
OUT_DIR = ROOT / "outputs/current_production_package_v2/universe_scope_audit_v4"
CONFIG = ROOT / "configs/current_pipeline_v4.yaml"


def inferred_exclusion(row: pd.Series) -> str:
    moa = str(row.get("mechanism_of_action") or "").lower()
    name = str(row.get("drug_name") or "").lower()
    mw = pd.to_numeric(pd.Series([row.get("molecular_weight")]), errors="coerce").iloc[0]
    reasons: list[str] = []
    if re.search(r"\bdna\b|cross.?link|alkylat|intercalat", moa):
        reasons.append("broad_DNA_or_reactive_mechanism")
    if re.search(r"peptidoglycan|beta-glucan|cell membrane|rna polymerase", moa):
        reasons.append("pathogen_or_nonhuman_primary_mechanism")
    if pd.notna(mw) and float(mw) > 900:
        reasons.append("very_high_molecular_weight_or_macrocycle")
    if re.search(r"glucagon|vancomycin|daptomycin|caspofungin|micafungin|anidulafungin", name):
        reasons.append("peptide_or_natural_product_like")
    return ";".join(dict.fromkeys(reasons)) or "current_rule_exclusion_requires_manual_review"


def chembl_known_pairs(
    drugs: pd.DataFrame, sequences: pd.DataFrame, mechanisms: pd.DataFrame
) -> pd.DataFrame:
    base_to_ids: dict[str, set[str]] = {}
    for drug in drugs["drug_id"].astype(str):
        base_to_ids.setdefault(drug.split("__")[0], set()).add(drug)
    gene_to_sequences: dict[str, set[str]] = {}
    for _, row in sequences.iterrows():
        for gene in split_genes(row.get("gene_names")):
            gene_to_sequences.setdefault(gene, set()).add(str(row["sequence_key"]))

    eligible = mechanisms[
        mechanisms["organism"].eq("Homo sapiens")
        & mechanisms["target_type"].eq("SINGLE PROTEIN")
        & mechanisms["action_type"].astype(str).str.upper().isin(DIRECT_ACTIONS)
    ]
    rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        drug_ids = set(base_to_ids.get(str(row.get("molecule_chembl_id")), set()))
        drug_ids.update(base_to_ids.get(str(row.get("parent_molecule_chembl_id")), set()))
        for drug_id in drug_ids:
            for gene in split_genes(row.get("component_gene_symbols")):
                for sequence_key in gene_to_sequences.get(gene, set()):
                    rows.append(
                        {
                            "drug_chembl_id": drug_id,
                            "sequence_key": sequence_key,
                            "gene": gene,
                            "action_type": row.get("action_type", ""),
                            "mechanism_of_action": row.get("mechanism_of_action", ""),
                        }
                    )
    if not rows:
        return pd.DataFrame(columns=["drug_chembl_id", "sequence_key", "gene"])
    return pd.DataFrame(rows).drop_duplicates(["drug_chembl_id", "sequence_key"])


def build() -> dict[str, Any]:
    for path in [
        DRUGS,
        SEQUENCES,
        ANCHORS,
        PROJECT_POOL,
        MECHANISMS,
        KNOWN_PROJECT,
        DRUG_MANIFEST,
        TARGET_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    drugs = pd.read_csv(DRUGS).fillna("")
    sequences = pd.read_csv(SEQUENCES).fillna("")
    anchors = pd.read_csv(ANCHORS, low_memory=False).fillna("")
    project = pd.read_csv(
        PROJECT_POOL,
        usecols=["drug_chembl_id", "sequence_key"],
        low_memory=False,
    ).fillna("")
    mechanisms = pd.read_csv(MECHANISMS, low_memory=False).fillna("")
    known_project = pd.read_csv(KNOWN_PROJECT, low_memory=False).fillna("")
    drug_manifest = pd.read_csv(DRUG_MANIFEST, low_memory=False).fillna("")
    target_manifest = pd.read_csv(TARGET_MANIFEST, low_memory=False).fillna("")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if file_sha256(DRUG_MANIFEST) != config["scope"]["project_drug_manifest_sha256"]:
        raise ValueError("Frozen drug manifest SHA-256 mismatch")
    if file_sha256(TARGET_MANIFEST) != config["scope"]["project_target_manifest_sha256"]:
        raise ValueError("Frozen target manifest SHA-256 mismatch")
    extra_drugs = set(config.get("scope", {}).get("extra_direct_action_drug_ids", []))
    project_drugs = set(drug_manifest["drug_chembl_id"].astype(str))
    target_extension_path = ROOT / config["inputs"]["target_scope_extension"]
    target_extension = pd.read_csv(target_extension_path).fillna("")
    project_targets = set(target_manifest["sequence_key"].astype(str))
    source_derived_drugs = set(project["drug_chembl_id"]) | extra_drugs
    source_derived_targets = set(project["sequence_key"]) | set(target_extension["sequence_key"].astype(str))
    if project_drugs != source_derived_drugs or project_targets != source_derived_targets:
        raise ValueError("Frozen v4 entity manifests do not match source-derived project scope")
    direct_mask = drugs["action_type"].astype(str).str.upper().isin(DIRECT_ACTIONS)
    excluded_direct_drugs = drugs[direct_mask & ~drugs["drug_id"].isin(project_drugs)].copy()
    excluded_direct_drugs["inferred_current_exclusion"] = excluded_direct_drugs.apply(inferred_exclusion, axis=1)

    outside_direct_targets = anchors[
        bool_series(anchors, "project_standard_direct_sm")
        & ~bool_series(anchors, "in_current_462_target_engagement")
        & ~anchors["gene"].isin(set(target_extension["primary_gene"]))
    ].copy()
    known_full = chembl_known_pairs(drugs, sequences, mechanisms)
    known_full["drug_in_project"] = known_full["drug_chembl_id"].isin(project_drugs)
    known_full["target_in_project"] = known_full["sequence_key"].isin(project_targets)
    known_full["pair_in_project_scope"] = known_full["drug_in_project"] & known_full["target_in_project"]

    structure_ok = known_project["structure_bin"].isin(
        ["A_strict_overlapping_pocket", "B_strict_supported_overlap", "C_manual_review_structure"]
    )
    direct_sm = bool_series(known_project, "anchor_project_standard_direct_sm")
    drug_rank = pd.to_numeric(known_project["rank_within_drug"], errors="coerce")
    target_rank = pd.to_numeric(known_project["target_rank"], errors="coerce")
    rank_union = (drug_rank <= 300) | (target_rank <= 300)
    score_gate = pd.to_numeric(known_project["conplex_score"], errors="coerce") >= 0.05
    recall_by_rank: dict[str, dict[str, float]] = {}
    for label, ranks, universe_size in [
        ("project463", drug_rank, len(project_targets)),
        (
            "full891",
            pd.to_numeric(known_project["rank_within_drug_full891"], errors="coerce"),
            int(sequences["sequence_key"].nunique()),
        ),
        ("target750", target_rank, len(project_drugs)),
    ]:
        recall_by_rank[label] = {}
        for cutoff in [10, 50, 100, 300]:
            observed = float((ranks <= cutoff).mean())
            random_expectation = min(1.0, cutoff / universe_size)
            recall_by_rank[label][f"top{cutoff}_recall"] = observed
            recall_by_rank[label][f"top{cutoff}_random_expectation"] = random_expectation
            recall_by_rank[label][f"top{cutoff}_enrichment"] = observed / random_expectation

    summary = {
        "drug_universe": {
            "all_fda_rows": int(len(drugs)),
            "direct_action_rows": int(direct_mask.sum()),
            "project_rows": int(len(project_drugs)),
            "explicit_scope_extension_rows": int(len(extra_drugs)),
            "explicit_scope_extension_drug_ids": sorted(extra_drugs),
            "excluded_direct_action_rows": int(len(excluded_direct_drugs)),
        },
        "target_universe": {
            "chembl_moa_genes": int(anchors["gene"].nunique()),
            "unique_sequences": int(sequences["sequence_key"].nunique()),
            "project_target_engagement_sequences": int(len(project_targets)),
            "explicit_target_scope_extensions": target_extension["primary_gene"].astype(str).tolist(),
            "opentargets_direct_sm_genes": int(bool_series(anchors, "project_standard_direct_sm").sum()),
            "direct_sm_genes_outside_current_462": int(len(outside_direct_targets)),
            "gpcr_excluded_genes": int(bool_series(anchors, "excluded_membrane_receptor_gpcr").sum()),
            "secreted_surface_structural_excluded_genes": int(
                bool_series(anchors, "excluded_secreted_surface_adhesion_structural").sum()
            ),
        },
        "pair_spaces": {
            "raw_915_x_891": int(drugs["drug_id"].nunique() * sequences["sequence_key"].nunique()),
            "project_drugs_x_targets": int(len(project_drugs) * len(project_targets)),
            "unique_model_ligands_x_targets": int(
                drug_manifest["model_ligand_smiles"].nunique() * len(project_targets)
            ),
            "legacy_top300_derived_discovery_pool": int(len(project)),
        },
        "chembl37_known_direct_moa": {
            "full_scope_pairs": int(len(known_full)),
            "project_scope_pairs": int(known_full["pair_in_project_scope"].sum()),
            "excluded_by_drug_scope": int((~known_full["drug_in_project"]).sum()),
            "excluded_by_target_scope": int((~known_full["target_in_project"]).sum()),
            "excluded_by_both": int((~known_full["drug_in_project"] & ~known_full["target_in_project"]).sum()),
        },
        "project_known_union_calibration": {
            "scope": "frozen 750x463 v4 project universe",
            "pairs": int(len(known_project)),
            "active_moiety_target_pairs": int(
                known_project.drop_duplicates(["knowledge_compound_key", "sequence_key"]).shape[0]
            ),
            "direct_sm_retained": int(direct_sm.sum()),
            "structure_retained": int(structure_ok.sum()),
            "legacy_rank_union_retained": int(rank_union.sum()),
            "legacy_absolute_score_retained": int(score_gate.sum()),
            "legacy_combined_gate_retained": int((direct_sm & structure_ok & rank_union & score_gate).sum()),
            "v4_structure_and_tractability_retained": int((direct_sm & structure_ok).sum()),
            "v4_structure_and_tractability_recall": float((direct_sm & structure_ok).mean()),
            "warning": "Known-label/ChEMBL overlap makes this calibration, not temporal generalization.",
        },
        "conplex_rank_calibration": recall_by_rank,
        "source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [
                DRUGS,
                SEQUENCES,
                ANCHORS,
                PROJECT_POOL,
                MECHANISMS,
                KNOWN_PROJECT,
                CONFIG,
                DRUG_MANIFEST,
                TARGET_MANIFEST,
                target_extension_path,
            ]
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded_direct_drugs.to_csv(OUT_DIR / "excluded_direct_action_drugs_audit.csv", index=False)
    outside_direct_targets.to_csv(OUT_DIR / "direct_sm_targets_outside_462.csv", index=False)
    known_full.to_csv(OUT_DIR / "chembl37_known_pairs_scope_audit.csv", index=False)
    (OUT_DIR / "universe_scope_audit_v4.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = f"""# FDA 老药新用筛选空间审计（v4）

## 药物与靶点口径

- FDA 结构库：{summary['drug_universe']['all_fda_rows']} 个条目。
- 具有直接作用类型标签：{summary['drug_universe']['direct_action_rows']} 个；当前项目药物：{summary['drug_universe']['project_rows']} 个，其中 {summary['drug_universe']['explicit_scope_extension_rows']} 个由边界审计显式恢复。
- ChEMBL-MoA：{summary['target_universe']['chembl_moa_genes']} 个基因、{summary['target_universe']['unique_sequences']} 条唯一序列。
- 当前非 GPCR、可做 target-engagement 的项目靶点：{summary['target_universe']['project_target_engagement_sequences']} 条唯一序列，其中恢复的口袋共识靶点为 {', '.join(summary['target_universe']['explicit_target_scope_extensions'])}。
- 原始计算空间：{summary['pair_spaces']['raw_915_x_891']:,} 对；ID 审计空间：{summary['pair_spaces']['project_drugs_x_targets']:,} 对；唯一模型结构物理空间：{summary['pair_spaces']['unique_model_ligands_x_targets']:,} 对。

## 已知对审计

- ChEMBL 37 人源单蛋白直接 MoA 在原始空间映射到 {summary['chembl37_known_direct_moa']['full_scope_pairs']} 对。
- 其中 {summary['chembl37_known_direct_moa']['project_scope_pairs']} 对处于冻结的 750×463 项目口径；主要损失来自主动排除的 GPCR、分泌/表面与非标准 target-engagement 靶点。
- 项目内 ChEMBL+FDA 联合阳性校准集为 {summary['project_known_union_calibration']['pairs']} 个 ID-pair，按活性母体折叠后为 {summary['project_known_union_calibration']['active_moiety_target_pairs']} 对。
- 旧 Top300+绝对分数+结构/可做性门槛保留 {summary['project_known_union_calibration']['legacy_combined_gate_retained']} 对。
- v4 取消 Top300 和绝对分数硬门槛后，结构/可做性层保留 {summary['project_known_union_calibration']['v4_structure_and_tractability_retained']} 对（{100*summary['project_known_union_calibration']['v4_structure_and_tractability_recall']:.2f}%）。

## 边界说明

- 上述召回是已知知识校准，不是无泄露的未来发现准确率。
- 已知对不进入 discovery 排名，仅用于检查流程是否系统性杀死可结合对。
- 仍未进入项目的直接作用药物和 3 个 OT 直接小分子靶点已分别输出人工复核清单。
"""
    (OUT_DIR / "UNIVERSE_SCOPE_AUDIT_V4_ZH.md").write_text(md, encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
