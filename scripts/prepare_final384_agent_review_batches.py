#!/usr/bin/env python3
"""Build disjoint, schema-checked review batches for a formal v4 candidate pool."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def merge_optional(base: pd.DataFrame, path: str, columns: list[str]) -> pd.DataFrame:
    if not path:
        return base
    frame = pd.read_csv(path, low_memory=False).fillna("")
    present = [column for column in columns if column in frame.columns]
    if "pair_id" in frame.columns:
        keep = ["pair_id", *present]
        frame = frame[keep]
        if frame["pair_id"].duplicated().any():
            raise ValueError(f"Duplicate pair_id in optional evidence: {path}")
        return base.merge(frame, on="pair_id", how="left", validate="one_to_one")

    if {"drug_chembl_id", "candidate_anchor_gene"}.issubset(frame.columns):
        frame = frame.rename(columns={"candidate_anchor_gene": "primary_gene"})
    keys = ["drug_chembl_id", "primary_gene"]
    if not set(keys).issubset(frame.columns):
        raise ValueError(f"Cannot build drug-target key for optional evidence: {path}")
    frame = frame[[*keys, *present]]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate drug-target key in optional evidence: {path}")
    return base.merge(frame, on=keys, how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--opentargets", default="")
    parser.add_argument("--chembl", default="")
    parser.add_argument("--pubmed", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=384)
    parser.add_argument("--rank-column", default="")
    parser.add_argument("--output-prefix", default="final384")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates, low_memory=False).fillna("")
    required = {"pair_id", "drug_chembl_id", "drug_names", "primary_gene", "protein_names"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")
    if candidates["pair_id"].duplicated().any():
        raise ValueError("Formal review input contains duplicate pair_id rows")
    if args.expected_rows > 0 and len(candidates) != args.expected_rows:
        raise ValueError(
            f"Formal review input has {len(candidates)} rows; expected {args.expected_rows}"
        )
    rank_column = args.rank_column.strip()
    if not rank_column:
        rank_column = next(
            (column for column in ["review_pool_rank", "final384_rank", "final1000_rank"] if column in candidates),
            "",
        )
    if not rank_column:
        raise ValueError("A rank column is required for deterministic review batching")
    if rank_column not in candidates.columns:
        raise ValueError(f"Rank column is missing: {rank_column}")

    merged = merge_optional(
        candidates,
        args.opentargets,
        [
            "ot_full_disease_count",
            "ot_full_top_diseases",
            "ot_full_top_genetic_diseases",
            "ot_full_top_clinical_diseases",
            "ot_full_top_literature_diseases",
        ],
    )
    merged = merge_optional(
        merged,
        args.chembl,
        [
            "chembl_exact_activity_status",
            "chembl_exact_activity_count",
            "chembl_exact_binding_count",
            "chembl_exact_binding_pchembl_ge_5_count",
            "chembl_exact_raw_binding_pchembl_ge_5_count",
            "chembl_exact_manual_binding_review_count",
            "chembl_exact_max_binding_pchembl",
            "chembl_exact_document_years",
            "chembl_exact_document_ids",
            "chembl_activity_query_ok",
            "chembl_activity_query_errors",
            "chembl_hierarchy_query_ok",
            "chembl_molecule_ids_queried",
            "chembl_assay_metadata_query_ok",
        ],
    )
    merged = merge_optional(
        merged,
        args.pubmed,
        [
            "pair_pubmed_count_2000_2026",
            "pair_pubmed_pmids_2000_2026",
            "pair_pubmed_url_2000_2026",
            "post_approval_pair_pubmed_count",
            "post_approval_pair_pubmed_pmids",
            "post_approval_pair_pubmed_url",
            "representative_pair_pmids",
            "representative_post_approval_pmids",
            "literature_class",
            "lit_ok",
            "pubmed_query_schema",
            "pubmed_query_sha256",
            "pair_pubmed_query_error",
            "post_approval_pubmed_query_error",
        ],
    )

    input_columns = [
        rank_column,
        "deep_review_stage_v6",
        "in_final500_v6",
        "top500_rank_v6",
        "pair_id",
        "drug_chembl_id",
        "drug_names",
        "fda_therapeutic_area",
        "fda_indication",
        "fda_moa",
        "fda_action_type",
        "fda_target_names",
        "primary_gene",
        "protein_names",
        "target_assay_family_v2",
        "target_assay_family",
        "v5_strength_tier",
        "v5_pair_physics_score",
        "top500_selection_score_v6",
        "experimental_execution_tier_v6",
        "priority_score_v2",
        "pair_specific_evidence_score_v2",
        "conplex_score",
        "rank_within_drug",
        "target_rank",
        "boltz_affinity_probability_refined",
        "boltz_ligand_iptm_refined",
        "boltz_confidence_score_refined",
        "pose_stability_tier",
        "pose_ligand_rmsd",
        "pose_interface_residue_jaccard",
        "structure_bin",
        "anchor_availability_tier",
        "compound_liability_notes",
        "assay_interference_review",
        "brenk_developability_review",
        "default_assay_strategy",
        "candidate_disease_v6",
        "candidate_disease_area_v6",
        "candidate_disease_basis_v6",
        "ot_primary_disease_v6",
        "ot_primary_disease_evidence_tier_v6",
        "ot_primary_disease_evidence_channels_v6",
        "ot_full_top_diseases",
        "ot_full_top_genetic_diseases",
        "ot_full_top_clinical_diseases",
        "ot_full_top_literature_diseases",
        "chembl_exact_activity_status",
        "chembl_exact_binding_count",
        "chembl_exact_raw_binding_pchembl_ge_5_count",
        "chembl_exact_manual_binding_review_count",
        "chembl_exact_max_binding_pchembl",
        "chembl_exact_document_years",
        "chembl_exact_document_ids",
        "chembl_activity_query_ok",
        "chembl_activity_query_errors",
        "chembl_hierarchy_query_ok",
        "chembl_molecule_ids_queried",
        "chembl_assay_metadata_query_ok",
        "pair_pubmed_count_2000_2026",
        "pair_pubmed_pmids_2000_2026",
        "pair_pubmed_url_2000_2026",
        "post_approval_pair_pubmed_count",
        "post_approval_pair_pubmed_pmids",
        "post_approval_pair_pubmed_url",
        "pubmed_screen_tier_v6",
        "pubmed_screen_titles_v6",
        "pubmed_screen_dois_v6",
        "literature_evidence_tier_v6",
        "literature_judgment_v6",
        "active_species_status_v6",
        "exposure_feasibility_v6",
        "assay_plan_v6",
        "key_risks_v6",
        "representative_pair_pmids",
        "representative_post_approval_pmids",
        "literature_class",
        "lit_ok",
        "pubmed_query_schema",
        "pubmed_query_sha256",
        "pair_pubmed_query_error",
        "post_approval_pubmed_query_error",
    ]
    review_input = merged[[column for column in input_columns if column in merged.columns]].copy()
    review_input = review_input.sort_values([rank_column, "pair_id"], kind="mergesort").reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_all_name = f"{args.output_prefix}_agent_review_input_all.csv"
    review_input.to_csv(out_dir / input_all_name, index=False)
    batch_count = max(1, min(args.batches, len(review_input)))
    batch_size = math.ceil(len(review_input) / batch_count)
    batch_rows = []
    for batch_index in range(batch_count):
        start = batch_index * batch_size
        stop = min(len(review_input), start + batch_size)
        if start >= stop:
            continue
        batch = review_input.iloc[start:stop].copy()
        name = f"batch_{batch_index + 1:02d}"
        input_path = out_dir / f"{name}_input.csv"
        output_path = out_dir / f"{name}_review.csv"
        batch.to_csv(input_path, index=False)
        template = pd.DataFrame(
            {
                "pair_id": batch["pair_id"],
                "agent_feasibility_grade": "",
                "agent_verdict": "",
                "agent_literature_class": "",
                "agent_primary_disease": "",
                "agent_repurposing_status": "",
                "agent_disease_evidence": "",
                "agent_mechanism_rationale": "",
                "agent_exposure_feasibility": "",
                "agent_active_species_status": "",
                "agent_assay_plan": "",
                "agent_key_risks": "",
                "agent_database_query_resolution": "",
                "agent_confidence": "",
                "agent_sources": "",
                "agent_reviewed_utc": "",
            }
        )
        template.to_csv(output_path, index=False)
        batch_rows.append(
            {
                "batch": name,
                "start_rank": int(batch[rank_column].min()),
                "end_rank": int(batch[rank_column].max()),
                "rows": len(batch),
                "input": str(input_path),
                "output": str(output_path),
            }
        )

    instructions = """# 候选池逐条智能体审阅合同

每一行必须检索并判断当前药物活性母体与候选靶点的精确关系，不得把同家族、同通路、单纯疾病共现当成直接结合证据。

- `agent_feasibility_grade`: A / B / C / D。A = 物理假说、人体/体外暴露和 assay 均可执行且无明显反证；B = 总体可测但有一项重要不确定性；C = 暴露、选择性或 assay 存在主要障碍；D = 明显矛盾、不可实现或非特异风险占主导。精确文献已验证本身不自动等于 A，而应进入对照/再发现队列。
- `agent_verdict`: 一句话说明保留、后置或剔除理由。
- `agent_literature_class`: exact_pair_validated / functional_only / indirect_or_family_only / no_exact_report_found / contradictory。
- ChEMBL 只有 `exact_binding_activity_pchembl_ge_5` 才是通过关系符、有效性、wild-type 语境和 assay confidence 过滤后的强精确对；`manual_exact_binding_review` 必须回看 assay/document，不能直接写 `exact_pair_validated`。
- `agent_primary_disease`: 最多给一个最可检验的新适应症；证据不足则写 `未指定`。
- `agent_repurposing_status`: `new_disease_area` / `new_indication_same_area` / `target_only_no_disease_claim` / `original_indication_or_not_repurposing`。必须将候选病种与 FDA 原适应症比较，不能把肿瘤内换一个靶点自动写成新领域。
- `agent_disease_evidence`: 区分靶点-疾病证据、药物-疾病证据和推断。
- `agent_mechanism_rationale`: 明确作用方向未知，不能仅凭结合模型声称激动/抑制。
- `agent_exposure_feasibility`: 评估已知人体暴露或可实现体外浓度，不知道时明确写未知。
- `agent_active_species_status`: `parent_drug_relevant` / `salt_normalization_adequate` / `active_species_uncertain` / `prodrug_active_metabolite_requires_rerun`。若上市药主要以活性代谢物起效且母体暴露不足，必须写最后一类，不能沿用母体 Boltz 结果。
- `agent_assay_plan`: 给出 primary assay、正对照、已知靶点反筛、细胞毒性/膜干扰门。
- `agent_key_risks`: 至少写一个主要风险。
- `agent_database_query_resolution`: `not_needed` / `resolved_manually` / `unresolved`。若 ChEMBL 或 PubMed 自动查询失败，只有给出可核实来源并人工补查后才可写 `resolved_manually`。
- `agent_confidence`: high / medium / low。
- `agent_sources`: DOI、PMID 或稳定网页链接，分号分隔；不得伪造来源。

完成文件必须保持输入 `pair_id` 原样且一行一对，所有字段非空。
"""
    (out_dir / "AGENT_REVIEW_INSTRUCTIONS_ZH.md").write_text(instructions, encoding="utf-8")
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": len(review_input),
        "unique_pairs": int(review_input["pair_id"].nunique()),
        "rank_column": rank_column,
        "input_all": str(out_dir / input_all_name),
        "batches": len(batch_rows),
        "batch_size_max": max(row["rows"] for row in batch_rows),
        "batch_manifest": batch_rows,
    }
    (out_dir / "agent_review_batch_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
