#!/usr/bin/env python3
"""Build the strict ChEMBL 37 small-molecule target set with preferred pockets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
POCKETS = ROOT / "outputs/chembl37_known_pocket_atlas/report/TARGET_POCKET_DECISION_PACKAGE_888.csv"
POCKET_INSTANCES = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/KNOWN_POCKET_INSTANCES_ANNOTATED_FULL.csv.gz"
OUTDIR = ROOT / "outputs/final_target_package_ch37"

ALLOWED_ASSAY_LANES = {
    "KINASE_BIOCHEMICAL",
    "ENZYME_BIOCHEMICAL",
    "NUCLEAR_EPIGENETIC_DOMAIN",
    "ION_CHANNEL_FUNCTIONAL",
    "TRANSPORTER_MEMBRANE_FUNCTIONAL",
}

LANE_ZH = {
    "KINASE_BIOCHEMICAL": "激酶",
    "ENZYME_BIOCHEMICAL": "酶（非激酶）",
    "NUCLEAR_EPIGENETIC_DOMAIN": "核内/表观遗传蛋白",
    "ION_CHANNEL_FUNCTIONAL": "离子通道",
    "TRANSPORTER_MEMBRANE_FUNCTIONAL": "转运体",
}

POCKET_EVIDENCE = {
    "K1_DRUG_MAPPED_EXPERIMENTAL": (
        "DRUG_MAPPED_HIGH_QUALITY_EXPERIMENTAL",
        "药物映射的高质量实验口袋",
        "优先",
    ),
    "K2_SPECIALIZED_CURATED_SITE": (
        "SPECIALIZED_CURATED_EXPERIMENTAL",
        "专业口袋库确认的实验位点",
        "优先",
    ),
    "K3_EXPERIMENTAL_DRUGLIKE_SITE": (
        "EXPERIMENTAL_DRUGLIKE_REVIEW",
        "实验药物样配体口袋（需复核）",
        "复核",
    ),
    "K4_FUNCTIONAL_OR_FRAGMENT_SITE": (
        "FUNCTIONAL_OR_FRAGMENT_REVIEW",
        "功能配体/片段口袋（需复核）",
        "复核",
    ),
}


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(MASTER, low_memory=False)
    pockets = pd.read_csv(POCKETS, low_memory=False)
    pocket_instances = pd.read_csv(POCKET_INSTANCES, low_memory=False)
    if len(master) != 888 or master["target_chembl_id"].nunique() != 888:
        raise ValueError("Official target master must contain exactly 888 unique ChEMBL targets")
    if len(pockets) != 888 or pockets["target_chembl_id"].nunique() != 888:
        raise ValueError("Pocket decision package must contain exactly 888 unique ChEMBL targets")

    pocket_columns = [
        "target_chembl_id",
        "known_unique_pocket_count",
        "known_k1_pocket_count",
        "known_k1_k2_pocket_count",
        "known_drug_mapped",
        "known_biolip2",
        "known_biolip2_affinity",
        "known_scpdb",
        "known_klifs",
        "known_gpcrdb",
        "representative_known_pocket_grade",
        "representative_known_pocket_sources",
        "representative_pdb_id",
        "representative_chain_id",
        "representative_ligand_id",
        "representative_known_pocket_residues",
        "representative_known_pocket_residue_count",
        "p2rank_top1_any_known_combined_match_8a",
        "p2rank_top3_any_known_combined_match_8a",
    ]
    data = master.merge(
        pockets[pocket_columns], on="target_chembl_id", how="left", validate="one_to_one"
    )

    non_gpcr = ~as_bool(data["is_gpcr"])
    chembl_small_molecule_moa = non_gpcr & as_bool(data["small_molecule_moa"])
    supported_class = data["assay_lane"].isin(ALLOWED_ASSAY_LANES)
    has_experimental_pocket = (
        pd.to_numeric(data["known_unique_pocket_count"], errors="coerce").fillna(0).gt(0)
    )
    has_preferred_experimental_pocket = (
        pd.to_numeric(data["known_k1_k2_pocket_count"], errors="coerce").fillna(0).gt(0)
    )
    preferred_druglike_target_ids = set(
        pocket_instances.loc[
            pocket_instances["known_pocket_grade"].isin(
                {"K1_DRUG_MAPPED_EXPERIMENTAL", "K2_SPECIALIZED_CURATED_SITE"}
            )
            & pocket_instances["ligand_class"].isin({"DRUG_MAPPED", "DRUGLIKE_UNMAPPED"}),
            "target_chembl_id",
        ]
    )
    has_preferred_druglike_experimental_pocket = data["target_chembl_id"].isin(
        preferred_druglike_target_ids
    )
    preferred_target_mask = (
        chembl_small_molecule_moa & supported_class & has_preferred_experimental_pocket
    )
    final_mask = preferred_target_mask & has_preferred_druglike_experimental_pocket

    audit = data[[
        "target_chembl_id",
        "gene_symbol",
        "uniprot_accession",
        "target_name",
        "assay_lane",
        "is_gpcr",
        "small_molecule_moa",
        "ot_project_standard_direct_sm",
        "known_unique_pocket_count",
    ]].copy()
    audit["passes_non_gpcr"] = non_gpcr
    audit["passes_chembl_small_molecule_moa"] = chembl_small_molecule_moa
    audit["passes_supported_target_class"] = supported_class
    audit["passes_any_experimental_pocket"] = has_experimental_pocket
    audit["passes_preferred_experimental_pocket"] = has_preferred_experimental_pocket
    audit["passes_preferred_druglike_experimental_pocket"] = (
        has_preferred_druglike_experimental_pocket
    )
    audit["in_final_target_set"] = final_mask
    audit["first_exclusion_reason"] = "INCLUDED"
    audit.loc[~non_gpcr, "first_exclusion_reason"] = "EXCLUDE_GPCR"
    audit.loc[non_gpcr & ~chembl_small_molecule_moa, "first_exclusion_reason"] = (
        "EXCLUDE_NO_CHEMBL_SMALL_MOLECULE_MOA"
    )
    audit.loc[chembl_small_molecule_moa & ~supported_class, "first_exclusion_reason"] = (
        "EXCLUDE_UNSUPPORTED_TARGET_CLASS"
    )
    audit.loc[
        chembl_small_molecule_moa & supported_class & ~has_experimental_pocket,
        "first_exclusion_reason",
    ] = (
        "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"
    )
    audit.loc[
        chembl_small_molecule_moa
        & supported_class
        & has_experimental_pocket
        & ~has_preferred_experimental_pocket,
        "first_exclusion_reason",
    ] = "EXCLUDE_NO_PREFERRED_EXPERIMENTAL_POCKET"
    audit.loc[
        preferred_target_mask & ~has_preferred_druglike_experimental_pocket,
        "first_exclusion_reason",
    ] = "EXCLUDE_NO_PREFERRED_DRUGLIKE_HOLO_POCKET"

    final = data.loc[final_mask].copy()
    final["target_class_zh"] = final["assay_lane"].map(LANE_ZH)
    pocket_details = final["representative_known_pocket_grade"].map(POCKET_EVIDENCE)
    final["experimental_pocket_evidence_category"] = pocket_details.map(lambda value: value[0])
    final["experimental_pocket_evidence_zh"] = pocket_details.map(lambda value: value[1])
    final["pocket_use_priority_zh"] = pocket_details.map(lambda value: value[2])
    final["membrane_assay_branch"] = final["assay_lane"].isin(
        {"ION_CHANNEL_FUNCTIONAL", "TRANSPORTER_MEMBRANE_FUNCTIONAL"}
    )
    final["preferred_experimental_pocket_count"] = pd.to_numeric(
        final["known_k1_k2_pocket_count"], errors="coerce"
    ).fillna(0).astype(int)
    final["final_target_definition"] = (
        "CHEMBL_SMALL_MOLECULE_MOA;NON_GPCR;SUPPORTED_TARGET_CLASS;PREFERRED_EXPERIMENTAL_POCKET"
    )

    output_columns = [
        "target_chembl_id",
        "gene_symbol",
        "uniprot_accession",
        "target_name",
        "protein_description",
        "target_class_zh",
        "assay_lane",
        "membrane_assay_branch",
        "sequence",
        "sequence_length",
        "sequence_sha256",
        "small_molecule_moa",
        "small_molecule_direct_moa",
        "approved_small_molecule_moa",
        "small_molecule_moa_record_count",
        "small_molecule_count",
        "small_molecule_max_phase",
        "ot_project_standard_direct_sm",
        "ot_sm_structure_with_ligand",
        "ot_sm_high_quality_pocket",
        "ot_sm_med_quality_pocket",
        "known_unique_pocket_count",
        "preferred_experimental_pocket_count",
        "experimental_pocket_evidence_category",
        "experimental_pocket_evidence_zh",
        "pocket_use_priority_zh",
        "representative_known_pocket_sources",
        "representative_pdb_id",
        "representative_chain_id",
        "representative_ligand_id",
        "representative_known_pocket_residues",
        "representative_known_pocket_residue_count",
        "af_exact_sequence_model",
        "af_selected_accession",
        "af_pdb_path",
        "p2rank_status",
        "p2rank_tier",
        "p2rank_top_score",
        "p2rank_top_probability",
        "p2rank_top1_any_known_combined_match_8a",
        "p2rank_top3_any_known_combined_match_8a",
        "positive_compounds",
        "negative_compounds",
        "target_calibration_tier",
        "calibration_8x8",
        "final_target_definition",
    ]
    final = final[output_columns].sort_values(
        ["membrane_assay_branch", "target_class_zh", "gene_symbol", "target_chembl_id"]
    )
    if len(final) != 338 or final["target_chembl_id"].nunique() != 338:
        raise ValueError(f"Expected 338 unique final targets, found {len(final)}")

    funnel = pd.DataFrame([
        {"步骤": 0, "集合": "ChEMBL 37人源单蛋白MoA全集", "保留数": 888, "本步去除数": 0,
         "规则": "Homo sapiens + SINGLE PROTEIN + 至少一条drug mechanism/MoA记录"},
        {"步骤": 1, "集合": "排除GPCR", "保留数": int(non_gpcr.sum()), "本步去除数": int((~non_gpcr).sum()),
         "规则": "排除ChEMBL或Open Targets标记为GPCR的靶点"},
        {"步骤": 2, "集合": "保留ChEMBL小分子MoA靶点", "保留数": int(chembl_small_molecule_moa.sum()),
         "本步去除数": int((non_gpcr & ~chembl_small_molecule_moa).sum()),
         "规则": "非GPCR，且ChEMBL 37 molecule_type=Small molecule的MoA记录至少一条"},
        {"步骤": 3, "集合": "保留五类可实验靶点", "保留数": int((chembl_small_molecule_moa & supported_class).sum()),
         "本步去除数": int((chembl_small_molecule_moa & ~supported_class).sum()),
         "规则": "激酶、酶、核内/表观遗传蛋白、离子通道、转运体"},
        {"步骤": 4, "集合": "要求已有合格实验口袋", "保留数": int((chembl_small_molecule_moa & supported_class & has_experimental_pocket).sum()),
         "本步去除数": int((chembl_small_molecule_moa & supported_class & ~has_experimental_pocket).sum()),
         "规则": "至少一个映射到canonical UniProt残基、接触残基>=3且已去除水/添加剂/离子/微小碎片的实验配体接触集合"},
        {"步骤": 5, "集合": "要求优先实验口袋", "保留数": int(preferred_target_mask.sum()),
         "本步去除数": int((chembl_small_molecule_moa & supported_class & has_experimental_pocket & ~has_preferred_experimental_pocket).sum()),
         "规则": "只保留药物映射高质量实验口袋或专业口袋数据库确认的实验位点"},
        {"步骤": 6, "集合": "要求药物样holo口袋", "保留数": int(final_mask.sum()),
         "本步去除数": int((preferred_target_mask & ~has_preferred_druglike_experimental_pocket).sum()),
         "规则": "优先实验口袋中至少存在DRUG_MAPPED或DRUGLIKE_UNMAPPED实验配体；排除仅有辅因子/功能片段位点"},
    ])

    summary = {
        "definition": "ChEMBL small-molecule MoA AND non-GPCR AND supported target class AND preferred druglike holo experimental pocket",
        "official_targets": 888,
        "after_gpcr_exclusion": int(non_gpcr.sum()),
        "after_chembl_small_molecule_moa": int(chembl_small_molecule_moa.sum()),
        "after_supported_target_class": int((chembl_small_molecule_moa & supported_class).sum()),
        "after_any_experimental_pocket": int(
            (chembl_small_molecule_moa & supported_class & has_experimental_pocket).sum()
        ),
        "final_targets": int(final_mask.sum()),
        "removed_for_no_experimental_pocket_at_last_step": int(
            (chembl_small_molecule_moa & supported_class & ~has_experimental_pocket).sum()
        ),
        "removed_for_nonpreferred_pocket_at_last_step": int(
            (
                chembl_small_molecule_moa
                & supported_class
                & has_experimental_pocket
                & ~has_preferred_experimental_pocket
            ).sum()
        ),
        "removed_for_no_preferred_druglike_holo_pocket": int(
            (preferred_target_mask & ~has_preferred_druglike_experimental_pocket).sum()
        ),
        "preferred_experimental_pocket_targets": int(final["pocket_use_priority_zh"].eq("优先").sum()),
        "review_pocket_targets": int(final["pocket_use_priority_zh"].eq("复核").sum()),
        "biochemical_or_nuclear_targets": int((~final["membrane_assay_branch"]).sum()),
        "ion_channel_or_transporter_targets": int(final["membrane_assay_branch"].sum()),
        "target_class_counts": final["target_class_zh"].value_counts().to_dict(),
        "pocket_evidence_counts": final["experimental_pocket_evidence_zh"].value_counts().to_dict(),
    }

    final.to_csv(OUTDIR / "FINAL_FROZEN_CHEMBL_SM_TARGETS_WITH_DRUGLIKE_HOLO_338.csv", index=False)
    audit.to_csv(OUTDIR / "FINAL_TARGET_EXCLUSION_AUDIT_888.csv", index=False)
    funnel.to_csv(OUTDIR / "FINAL_TARGET_FUNNEL_COUNTS_ZH.csv", index=False)
    with pd.ExcelWriter(OUTDIR / "FINAL_FROZEN_CHEMBL_SM_TARGET_PACKAGE_338.xlsx", engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="冻结338靶点", index=False)
        funnel.to_excel(writer, sheet_name="漏斗与规则", index=False)
        audit.to_excel(writer, sheet_name="888排除审计", index=False)
    (OUTDIR / "FINAL_TARGET_SET_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
