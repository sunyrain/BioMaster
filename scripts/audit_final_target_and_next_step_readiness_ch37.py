#!/usr/bin/env python3
"""Audit the frozen 338 targets and select one computation-ready holo site per target."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "outputs/final_target_package_ch37/FINAL_FROZEN_CHEMBL_SM_TARGETS_WITH_DRUGLIKE_HOLO_338.csv"
INSTANCES = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/KNOWN_POCKET_INSTANCES_ANNOTATED_FULL.csv.gz"
CORRECTIONS = ROOT / "configs/pdbe_experiment_metadata_corrections_ch37.csv"
OUTDIR = ROOT / "outputs/final_target_package_ch37"

PREFERRED_GRADES = {"K1_DRUG_MAPPED_EXPERIMENTAL", "K2_SPECIALIZED_CURATED_SITE"}
DRUGLIKE_CLASSES = {"DRUG_MAPPED", "DRUGLIKE_UNMAPPED"}


def clean_key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_usable(method: str, resolution: float | None) -> bool:
    method = str(method).lower()
    if "nmr" in method:
        return True
    return pd.notna(resolution) and float(resolution) <= 4.0


def is_preferred(method: str, resolution: float | None) -> bool:
    method = str(method).lower()
    if "nmr" in method:
        return True
    if pd.isna(resolution):
        return False
    if "electron" in method:
        return float(resolution) <= 3.5
    return float(resolution) <= 3.0


def main() -> None:
    targets = pd.read_csv(TARGETS, low_memory=False)
    instances = pd.read_csv(INSTANCES, low_memory=False)
    corrections = pd.read_csv(CORRECTIONS)
    if len(targets) != 338 or targets["target_chembl_id"].nunique() != 338:
        raise ValueError("The frozen target package must contain 338 unique targets")

    for column in ["pdb_id", "chain_id", "ligand_id", "uniprot_accession"]:
        instances[column] = clean_key(instances[column])
    corrections["pdb_id"] = clean_key(corrections["pdb_id"])
    correction_method = corrections.set_index("pdb_id")["experimental_method"].to_dict()
    correction_resolution = corrections.set_index("pdb_id")["resolution"].to_dict()

    candidates = instances[
        instances["target_chembl_id"].isin(targets["target_chembl_id"])
        & instances["known_pocket_grade"].isin(PREFERRED_GRADES)
    ].copy()
    missing_method = candidates["experimental_method"].fillna("").astype(str).str.strip().eq("")
    missing_resolution = pd.to_numeric(candidates["resolution"], errors="coerce").isna()
    candidates.loc[missing_method, "experimental_method"] = candidates.loc[missing_method, "pdb_id"].map(
        correction_method
    )
    candidates.loc[missing_resolution, "resolution"] = candidates.loc[missing_resolution, "pdb_id"].map(
        correction_resolution
    )
    candidates["resolution"] = pd.to_numeric(candidates["resolution"], errors="coerce")
    candidates["is_druglike_holo_ligand"] = candidates["ligand_class"].isin(DRUGLIKE_CLASSES)
    candidates["usable_structure_quality_rechecked"] = [
        is_usable(method, resolution)
        for method, resolution in zip(candidates["experimental_method"], candidates["resolution"])
    ]
    candidates["preferred_structure_quality_rechecked"] = [
        is_preferred(method, resolution)
        for method, resolution in zip(candidates["experimental_method"], candidates["resolution"])
    ]
    candidates["grade_order"] = candidates["known_pocket_grade"].map(
        {"K1_DRUG_MAPPED_EXPERIMENTAL": 1, "K2_SPECIALIZED_CURATED_SITE": 2}
    )
    candidates["source_count"] = candidates[
        ["biolip2_support", "scpdb_support", "klifs_structure_support", "klifs_ligand_support"]
    ].fillna(False).astype(bool).sum(axis=1)
    candidates = candidates.sort_values(
        [
            "target_chembl_id",
            "is_druglike_holo_ligand",
            "usable_structure_quality_rechecked",
            "preferred_structure_quality_rechecked",
            "grade_order",
            "source_count",
            "binding_residue_count",
            "resolution",
        ],
        ascending=[True, False, False, False, True, False, False, True],
    )
    selected = candidates.drop_duplicates("target_chembl_id").copy()
    if selected["target_chembl_id"].nunique() != 338:
        missing = sorted(set(targets["target_chembl_id"]) - set(selected["target_chembl_id"]))
        raise ValueError(f"No preferred pocket instance for final targets: {missing}")

    selected["compute_readiness"] = "REVIEW_STRUCTURE_OR_SITE"
    selected.loc[
        selected["is_druglike_holo_ligand"] & selected["usable_structure_quality_rechecked"],
        "compute_readiness",
    ] = "READY_CONVENTIONAL_DRUGLIKE_HOLO"
    selected.loc[
        ~selected["is_druglike_holo_ligand"] & selected["usable_structure_quality_rechecked"],
        "compute_readiness",
    ] = "READY_SPECIAL_COFACTOR_OR_FUNCTIONAL_SITE"
    selected["compute_protocol_zh"] = selected["compute_readiness"].map({
        "READY_CONVENTIONAL_DRUGLIKE_HOLO": "常规药物样holo口袋流程",
        "READY_SPECIAL_COFACTOR_OR_FUNCTIONAL_SITE": "辅因子/功能位点专项流程",
        "REVIEW_STRUCTURE_OR_SITE": "结构或位点人工复核",
    })
    selected["structure_coordinate_package_status"] = "NEW_PACKAGE_NOT_DOWNLOADED_OR_PREPARED"
    selected["metadata_correction_applied"] = selected["pdb_id"].isin(set(corrections["pdb_id"]))

    selected_columns = [
        "target_chembl_id",
        "gene_symbol",
        "uniprot_accession",
        "pdb_id",
        "chain_id",
        "ligand_id",
        "ligand_name",
        "ligand_class",
        "experimental_method",
        "resolution",
        "uniprot_residue_positions",
        "binding_residue_count",
        "is_druglike_holo_ligand",
        "usable_structure_quality_rechecked",
        "preferred_structure_quality_rechecked",
        "biolip2_support",
        "biolip2_affinity_support",
        "scpdb_support",
        "klifs_structure_support",
        "klifs_ligand_support",
        "compute_readiness",
        "compute_protocol_zh",
        "metadata_correction_applied",
        "structure_coordinate_package_status",
    ]
    selected = selected[selected_columns].merge(
        targets[[
            "target_chembl_id",
            "target_class_zh",
            "assay_lane",
            "sequence",
            "sequence_sha256",
            "small_molecule_direct_moa",
            "approved_small_molecule_moa",
            "positive_compounds",
            "negative_compounds",
            "target_calibration_tier",
            "calibration_8x8",
        ]],
        on="target_chembl_id",
        how="left",
        validate="one_to_one",
    )
    selected = selected.sort_values(["compute_readiness", "target_class_zh", "gene_symbol"])

    conventional = selected["compute_readiness"].eq("READY_CONVENTIONAL_DRUGLIKE_HOLO")
    special = selected["compute_readiness"].eq("READY_SPECIAL_COFACTOR_OR_FUNCTIONAL_SITE")
    blocked = ~(conventional | special)
    summary = {
        "final_target_anchors": 338,
        "unique_genes": int(targets["gene_symbol"].nunique()),
        "unique_accessions": int(targets["uniprot_accession"].nunique()),
        "unique_sequences": int(targets["sequence_sha256"].nunique()),
        "chembl_direct_small_molecule_moa": int(targets["small_molecule_direct_moa"].fillna(False).astype(bool).sum()),
        "conventional_druglike_holo_ready": int(conventional.sum()),
        "special_cofactor_or_functional_site_ready": int(special.sum()),
        "blocked_or_manual_review": int(blocked.sum()),
        "new_coordinate_packages_still_required": 338,
        "p2rank_top1_matches_experimental_pocket": int(
            targets["p2rank_top1_any_known_combined_match_8a"].fillna(False).astype(bool).sum()
        ),
        "p2rank_top3_matches_experimental_pocket": int(
            targets["p2rank_top3_any_known_combined_match_8a"].fillna(False).astype(bool).sum()
        ),
        "target_calibration_8x8": int(targets["calibration_8x8"].fillna(False).astype(bool).sum()),
        "special_protocol_targets": selected.loc[special, "gene_symbol"].tolist(),
    }
    gaps = pd.DataFrame([
        {"模块": "靶点锚点", "当前状态": "READY", "数量/范围": "338个唯一靶点/基因/accession/序列",
         "下一动作": "冻结为target anchor universe"},
        {"模块": "实验口袋", "当前状态": "READY", "数量/范围": "338个常规药物样holo口袋",
         "下一动作": "按选定PDB链和配体建立结构清单"},
        {"模块": "实验结构坐标", "当前状态": "NOT_BUILT", "数量/范围": "338套新结构包",
         "下一动作": "从PDBe下载mmCIF/PDB、生物组装并核对目标链和配体"},
        {"模块": "受体准备", "当前状态": "NOT_BUILT", "数量/范围": "338靶点",
         "下一动作": "补全缺失残基/原子、质子化、金属和辅因子策略，生成receptor-ready结构"},
        {"模块": "FDA药物实体", "当前状态": "REBUILD_REQUIRED", "数量/范围": "旧表915记录；旧项目750记录/723结构",
         "下一动作": "重新核对批准实体、盐/复方/前药/活性物、非治疗性排除和唯一模型结构"},
        {"模块": "正式pair空间", "当前状态": "NOT_FROZEN", "数量/范围": "338 × 待定唯一药物结构",
         "下一动作": "药物主表冻结后生成全量pair并标记exact-known排除/校准"},
        {"模块": "模型推理", "当前状态": "DO_NOT_START", "数量/范围": "未生成新正式pair",
         "下一动作": "完成结构包与药物包后，再部署DTA和结构级级联"},
    ])

    readiness_path = OUTDIR / "FINAL_TARGET_COMPUTE_READINESS_338.csv"
    gaps_path = OUTDIR / "NEXT_STEP_INPUT_GAPS_ZH.csv"
    summary_path = OUTDIR / "FINAL_TARGET_COMPUTE_READINESS_SUMMARY.json"
    selected.to_csv(readiness_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target_path = TARGETS
    funnel_path = OUTDIR / "FINAL_TARGET_FUNNEL_COUNTS_ZH.csv"
    exclusion_path = OUTDIR / "FINAL_TARGET_EXCLUSION_AUDIT_888.csv"
    manifest = {
        "status": "FROZEN",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "target_count": 338,
        "definition": "ChEMBL 37 human SINGLE PROTEIN small-molecule direct MoA; non-GPCR; supported target class; preferred druglike experimental holo pocket",
        "explicitly_removed_after_preferred_pocket_gate": ["DNMT3A", "ODC1", "TXN"],
        "authoritative_files": {
            str(target_path.relative_to(ROOT)): sha256(target_path),
            str(readiness_path.relative_to(ROOT)): sha256(readiness_path),
            str(funnel_path.relative_to(ROOT)): sha256(funnel_path),
            str(exclusion_path.relative_to(ROOT)): sha256(exclusion_path),
        },
        "upstream_source_hashes": {
            str(INSTANCES.relative_to(ROOT)): sha256(INSTANCES),
            str(CORRECTIONS.relative_to(ROOT)): sha256(CORRECTIONS),
        },
        "usage_rule": "New pair-space construction must read the frozen 338 target CSV; legacy 341/463 target manifests are not production inputs.",
    }
    (OUTDIR / "FROZEN_TARGET_UNIVERSE_338_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
