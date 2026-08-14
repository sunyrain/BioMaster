#!/usr/bin/env python3
"""Assemble the V18 full 30-candidate identity/readiness/robustness routing package."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
V17 = RUN / "final_evidence_routing_v17"
PORTFOLIO = RUN / "full_candidate_portfolio_v18"
OUT = RUN / "final_evidence_routing_v18"
PAIR_NAME = "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz"
EXPECTED_PAIR_SHA256 = "8e1c73a0b16122bee6e4d3d7ca5dd9dbc816759c63a2976ec57799cdf97b7a9a"
INHERITED_DIRS = ["pubchem_control_identity_raw_v14", "pubchem_candidate_identity_raw_v15", "input_templates_v17", "dry_run_evaluation_v17"]
V18_RAW = "pubchem_full28_identity_raw_v18"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    required = [
        V17 / "EVIDENCE_LAYER_ROUTING_SUMMARY_V17.json",
        V17 / "UNIFIED_V17_REPRODUCIBILITY_AUDIT.json",
        PORTFOLIO / "FULL_CANDIDATE_PORTFOLIO_PROTOCOL_FROZEN_V18.json",
        PORTFOLIO / "FULL30_COMPUTATIONAL_ROBUSTNESS_SUMMARY_V18.json",
        PORTFOLIO / "FULL30_IDENTITY_CHEMISTRY_READINESS_SUMMARY_V18.json",
        PORTFOLIO / "FULL30_PORTFOLIO_INDEPENDENT_AUDIT_V18.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    v17_summary=json.loads(required[0].read_text()); v17_audit=json.loads(required[1].read_text())
    protocol=json.loads(required[2].read_text()); robust=json.loads(required[3].read_text())
    identity=json.loads(required[4].read_text()); audit=json.loads(required[5].read_text())
    if not (v17_summary["status"]==v17_audit["status"]=="PASS_INFRASTRUCTURE_AWAITING_EXTERNAL_DATA" and robust["status"]==identity["status"]==audit["status"]=="PASS"):
        raise RuntimeError("Upstream package/audits not in required states")

    OUT.mkdir(parents=True,exist_ok=True); declared={}
    for source in sorted(path for path in V17.iterdir() if path.is_file()):
        dest=OUT/source.name; shutil.copy2(source,dest); declared[dest.relative_to(OUT).as_posix()]=dest
    for directory in INHERITED_DIRS:
        destdir=OUT/directory; destdir.mkdir(exist_ok=True)
        for source in sorted(path for path in (V17/directory).iterdir() if path.is_file()):
            dest=destdir/source.name; shutil.copy2(source,dest); declared[dest.relative_to(OUT).as_posix()]=dest
    for source in sorted(path for path in PORTFOLIO.iterdir() if path.is_file()):
        dest=OUT/source.name; shutil.copy2(source,dest); declared[dest.relative_to(OUT).as_posix()]=dest
    destdir=OUT/V18_RAW; destdir.mkdir(exist_ok=True)
    for source in sorted((PORTFOLIO/V18_RAW).glob("*.json")):
        dest=destdir/source.name; shutil.copy2(source,dest); declared[dest.relative_to(OUT).as_posix()]=dest
    pair_core=OUT/PAIR_NAME
    if sha256(pair_core)!=EXPECTED_PAIR_SHA256: raise RuntimeError("Pair core changed")

    target17=pd.read_csv(OUT/"TARGET_EVIDENCE_LAYER_ROUTING_384_V17.csv",low_memory=False)
    master17=pd.read_csv(OUT/"MASTER_VALIDATION_QUEUE_63_ROWS_V17.csv",low_memory=False)
    pairrisk=pd.read_csv(OUT/"FULL30_PAIR_IDENTITY_CHEMISTRY_RISK_V18.csv",low_memory=False)
    routes=pd.read_csv(OUT/"FULL30_ASSAY_ROUTE_AND_AUTHORIZATION_V18.csv",low_memory=False)
    ranksum=pd.read_csv(OUT/"FULL30_BIDIRECTIONAL_ROBUSTNESS_SUMMARY_30_V18.csv",low_memory=False)
    physical=pd.read_csv(OUT/"FULL30_PHYSICAL_MODEL_COVERAGE_AND_RANKS_30_V18.csv",low_memory=False)
    independence=pd.read_csv(OUT/"FULL30_SELECTION_BIAS_AND_EVIDENCE_INDEPENDENCE_30_V18.csv",low_memory=False)

    identity_cols=[
        "v10_integrated_case_rank","ligand_inchikey","target_chembl_id","drug_names","gene_symbol",
        "molecule_chembl_id","pref_name","chembl_standard_inchikey","project_vs_chembl_full_inchikey_status",
        "source_label_vs_chembl_preferred_name_class","identity_execution_status","prodrug","pubchem_cid",
        "pubchem_identity_status","pubchem_vendor_record_status","preassay_chemistry_flags",
        "experimental_handling_risk_tier","risk_tier_reasons","chembl_known_mechanism_count","chembl_known_mechanisms",
    ]
    review=pairrisk[identity_cols].rename(columns={"v10_integrated_case_rank":"candidate_rank","ligand_inchikey":"project_full_inchikey","drug_names":"drug_name","pref_name":"modeled_chembl_preferred_entity"})
    route_cols=[
        "candidate_rank","execution_wave","target_assay_class","primary_assay_route","orthogonal_confirmation_route",
        "required_counterassays","active_species_requirement","execution_authorization","success_gate",
    ]
    review=review.merge(routes[route_cols],on="candidate_rank",validate="one_to_one")
    robustness_cols=[
        "candidate_rank","deployment_branch","target_lane_evidence_status","known_target_component_count",
        "target_family_relation","top10_view_count_of_8","top20_view_count_of_8",
        "supportive_ge80pct_view_count_of_8","target_centered_supportive_count_of_4",
        "drug_centered_supportive_count_of_4","minimum_directional_percentile","maximum_directional_percentile",
        "directional_percentile_range","rank_discordance_flag","selection_bias_status","rank_rule",
    ]
    review=review.merge(ranksum[robustness_cols],on="candidate_rank",validate="one_to_one")
    physical_cols=[
        "candidate_rank","boltz_primary_completed","boltz_affinity_probability_binary","boltz_support_count_3seed",
        "physical_metadata_consistency_flag","boltz_rank_within_selectively_computed_target_subset",
        "boltz_target_subset_denominator","gnina_primary_completed","gnina_primary_cnn_affinity",
        "gnina_cnn_rank_within_selectively_computed_target_subset","gnina_target_subset_denominator",
        "receptor_ensemble_decision","comparison_bias",
    ]
    review=review.merge(physical[physical_cols],on="candidate_rank",validate="one_to_one")
    independent_cols=["candidate_rank","independent_exact_pair_validation_status","internal_selection_overlap"]
    review=review.merge(independence[independent_cols],on="candidate_rank",validate="one_to_one")
    review["portfolio_decision_state"]="COMPUTATIONAL_AUDIT_ONLY_NO_RERANK"
    review.loc[review.execution_wave.eq("W1_BLINDED_CANDIDATE_PILOT"),"portfolio_decision_state"]="W1_FROZEN_PENDING_REAL_PROCUREMENT_AND_EXPERIMENTAL_DATA"
    review.loc[review.execution_wave.eq("W2_CONTINGENT_ONLY"),"portfolio_decision_state"]="W2_CONTINGENT_NOT_PROCUREMENT_OR_ASSAY_AUTHORIZED"
    review.loc[review.execution_wave.eq("VETO_NOT_AUTHORIZED"),"portfolio_decision_state"]="VETO_NO_PROCUREMENT_OR_ASSAY_AUTHORIZED"
    review.loc[review.identity_execution_status.str.contains("HOLD"),"portfolio_decision_state"] += "|ENTITY_NAME_ADJUDICATION_HOLD"
    review["orthogonal_priority_reason"]="FROZEN_STANDARD_ORTHOGONAL_AND_COUNTERASSAY_ROUTE"
    review.loc[review.rank_discordance_flag.eq("HIGH_RANGE_GE_0.50"),"orthogonal_priority_reason"]="HIGH_MODEL_PERCENTILE_RANGE_GE_0.50_PRIORITY_ORTHOGONAL_DISCRIMINATION"
    review.loc[review.physical_metadata_consistency_flag.ne("CONSISTENT_MAIN_AND_MULTISEED_METADATA"),"orthogonal_priority_reason"] += "|MISSING_MAIN_PHYSICAL_RESULT_DO_NOT_IMPUTE"
    review["claim_boundary"]="Identity/readiness/internal sensitivity audit only; no binding, efficacy, rank change or execution authorization beyond frozen wave."
    review_path=OUT/"FULL30_CANDIDATE_PORTFOLIO_REVIEW_V18.csv"; review.to_csv(review_path,index=False)

    # annotate targets only by aggregate candidate counts; pair-specific data remains in master/review tables
    targetagg=review.groupby("target_chembl_id").agg(
        v18_candidate_pair_count=("candidate_rank","size"),
        v18_w1_pair_count=("execution_wave",lambda s:int((s=="W1_BLINDED_CANDIDATE_PILOT").sum())),
        v18_w2_pair_count=("execution_wave",lambda s:int((s=="W2_CONTINGENT_ONLY").sum())),
        v18_veto_pair_count=("execution_wave",lambda s:int((s=="VETO_NOT_AUTHORIZED").sum())),
        v18_high_discordance_pair_count=("rank_discordance_flag",lambda s:int((s=="HIGH_RANGE_GE_0.50").sum())),
        v18_identity_hold_pair_count=("identity_execution_status",lambda s:int(s.str.contains("HOLD").sum())),
        v18_boltz_completed_pair_count=("boltz_primary_completed",lambda s:int(s.astype(bool).sum())),
        v18_gnina_completed_pair_count=("gnina_primary_completed",lambda s:int(s.astype(bool).sum())),
    ).reset_index()
    target18=target17.merge(targetagg,on="target_chembl_id",how="left",validate="one_to_one")
    aggcols=[c for c in targetagg.columns if c!="target_chembl_id"]
    target18[aggcols]=target18[aggcols].fillna(0).astype(int)
    target18["v18_full_candidate_portfolio_scope"]="NO_FROZEN_TOP30_PAIR_ON_TARGET"
    target18.loc[target18.v18_candidate_pair_count.gt(0),"v18_full_candidate_portfolio_scope"]="FULL30_IDENTITY_READINESS_AND_INTERNAL_ROBUSTNESS_AUDITED"
    target_path=OUT/"TARGET_EVIDENCE_LAYER_ROUTING_384_V18.csv"; target18.to_csv(target_path,index=False)

    ann=review.rename(columns={
        "candidate_rank":"candidate_rank_context","project_full_inchikey":"molecule_inchikey",
        "modeled_chembl_preferred_entity":"v18_modeled_chembl_preferred_entity",
        "source_label_vs_chembl_preferred_name_class":"v18_source_label_entity_class",
        "identity_execution_status":"v18_identity_execution_status","experimental_handling_risk_tier":"v18_risk_tier",
        "rank_discordance_flag":"v18_rank_discordance_flag","supportive_ge80pct_view_count_of_8":"v18_supportive_view_count_of_8",
        "boltz_primary_completed":"v18_boltz_primary_completed","gnina_primary_completed":"v18_gnina_primary_completed",
        "independent_exact_pair_validation_status":"v18_external_pair_validation_status",
        "execution_authorization":"v18_execution_authorization","portfolio_decision_state":"v18_portfolio_decision_state",
    })[[
        "candidate_rank_context","molecule_inchikey","target_chembl_id","v18_modeled_chembl_preferred_entity",
        "v18_source_label_entity_class","v18_identity_execution_status","v18_risk_tier",
        "v18_rank_discordance_flag","v18_supportive_view_count_of_8","v18_boltz_primary_completed",
        "v18_gnina_primary_completed","v18_external_pair_validation_status","v18_execution_authorization",
        "v18_portfolio_decision_state",
    ]]
    master18=master17.merge(ann,on=["candidate_rank_context","molecule_inchikey","target_chembl_id"],how="left",validate="one_to_one")
    candidate_mask=master18.validation_layer.eq("L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS")
    master18["v18_full_candidate_portfolio_scope"]="NOT_IN_FROZEN_TOP30_PAIR_SCOPE"
    master18.loc[candidate_mask,"v18_full_candidate_portfolio_scope"]="FULL30_IDENTITY_READINESS_AND_INTERNAL_ROBUSTNESS_AUDITED"
    master_path=OUT/"MASTER_VALIDATION_QUEUE_63_ROWS_V18.csv"; master18.to_csv(master_path,index=False)

    checks={
        "upstream_v17_and_v18_audits_pass": v17_audit["checks_passed"]==v17_audit["checks_total"]==40 and audit["checks_passed"]==audit["checks_total"]==24,
        "immutable_pair_core_hash_preserved":sha256(pair_core)==EXPECTED_PAIR_SHA256,
        "universe_888_minus_480_minus_24_equals_384":888-480-24==len(target18)==384,
        "pocket_partition_338_plus_only46_recovered":target18.active_target_branch.value_counts().to_dict()=={"STRICT_EXPERIMENTAL_POCKET_MAINLINE_338":338,"RECOVERED_NO_EXPERIMENTAL_POCKET_46":46},
        "deployment_partition_185_plus199":target18.old_drug_target_deployment_branch_v10.value_counts().to_dict()=={"UNSEEDED_TARGET_DTA_199":199,"SEEDED_KNOWN_GRAPH_185":185},
        "all_v17_target_rows_preserved":target17.equals(target18[target17.columns]),
        "all_v17_master_rows_preserved":master17.equals(master18[master17.columns]),
        "master_still_63_layer_partition":len(master18)==63 and master18.validation_layer.value_counts().to_dict()=={"L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS":30,"L0_EXTERNAL_SOURCE_REDISCOVERY_CONTROL":17,"L1_W1_TARGET_MATCHED_POSITIVE_CONTROL":8,"L1_W1_TARGET_MATCHED_NUMERIC_WEAK_CONTROL":8},
        "review_exact_30_rank_order_and_wave_partition":len(review)==30 and review.candidate_rank.tolist()==list(range(1,31)) and review.execution_wave.value_counts().to_dict()=={"W2_CONTINGENT_ONLY":21,"W1_BLINDED_CANDIDATE_PILOT":8,"VETO_NOT_AUTHORIZED":1},
        "review_exact_28_entities_16_targets":review.project_full_inchikey.nunique()==28 and review.target_chembl_id.nunique()==16,
        "exact_one_entity_name_hold_rank21":review.identity_execution_status.str.contains("HOLD").sum()==1 and review.loc[review.identity_execution_status.str.contains("HOLD"),"candidate_rank"].tolist()==[21],
        "risk_partition_8_17_5":review.experimental_handling_risk_tier.value_counts().to_dict()=={"R2_MODERATE_COUNTERASSAY":17,"R3_HIGH_SPECIAL_HANDLING":8,"R1_STANDARD":5},
        "robustness_high_discordance_exact16":review.rank_discordance_flag.eq("HIGH_RANGE_GE_0.50").sum()==16,
        "physical_coverage_29_boltz_10_gnina":review.boltz_primary_completed.astype(bool).sum()==29 and review.gnina_primary_completed.astype(bool).sum()==10,
        "zero_external_exact_pair_validation":review.independent_exact_pair_validation_status.eq("NONE_IN_FROZEN_CHEMBL_BINDINGDB_AND_LITERATURE_AUDITS").all(),
        "w2_and_veto_not_execution_authorized":review.loc[review.execution_wave.ne("W1_BLINDED_CANDIDATE_PILOT"),"portfolio_decision_state"].str.contains("NOT_PROCUREMENT_OR_ASSAY_AUTHORIZED|NO_PROCUREMENT_OR_ASSAY_AUTHORIZED").all(),
        "v17_w1_real_data_state_still_zero":v17_summary["v17_result_ingestion"]["real_result_rows_received"]==0 and v17_summary["v17_result_ingestion"]["assay_released_candidates"]==0,
        "target_aggregate_sums_equal30_and_exact16_targets":target18.v18_candidate_pair_count.sum()==30 and target18.v18_candidate_pair_count.gt(0).sum()==16,
        "master_exact30_candidate_rows_annotated":master18.v18_full_candidate_portfolio_scope.eq("FULL30_IDENTITY_READINESS_AND_INTERNAL_ROBUSTNESS_AUDITED").sum()==30,
    }
    checks={k:bool(v) for k,v in checks.items()}
    summary={
        "package_name":"FULL30_IDENTITY_READINESS_AND_INTERNAL_ROBUSTNESS_AUDITED_ROUTING_V18",
        "created_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
        "universe":v17_summary["universe"],"routing_unchanged":v17_summary["routing_unchanged"],"validation_queue":v17_summary["validation_queue"],
        "v17_result_ingestion":v17_summary["v17_result_ingestion"],
        "v18_full_candidate_portfolio":{
            "candidate_pairs":30,"unique_entities":28,"unique_targets":16,"w1":8,"w2":21,"veto":1,
            "seeded":19,"unseeded":11,"chembl_full_key_exact":27,"chembl_representation_mismatch":1,
            "entity_name_adjudication_holds":1,"chembl_prodrugs":3,"pubchem_exact_full_key":28,
            "risk_tier_r3":8,"risk_tier_r2":17,"risk_tier_r1":5,"bidirectional_rank_rows":240,
            "target_model_concordance_rows":180,"high_rank_discordance_pairs":16,"boltz_primary_completed":29,
            "gnina_primary_completed":10,"external_exact_pair_validations":0,"w2_assay_authorized":0,"veto_assay_authorized":0,
        },
        "critical_findings":[
            "Rank 21 source label ketoconazole maps to frozen CHEMBL295698 and full InChIKey XMAYWYJOQHXEEK-ZEQKJWHPSA-N, the levoketoconazole stereochemical entity; it remains under entity-name adjudication hold and is not replaced by racemic ketoconazole.",
            "Rank 21 has no completed main Boltz probability and no GNINA result; multiseed metadata is not used to impute a main physical result.",
            "Romidepsin is a ChEMBL prodrug and receives a reduced-disulfide/dithiol-state handling route without inventing a separate ChEMBL entity.",
            "Sixteen of 30 pairs have a directional model-percentile range >=0.50 and are routed to orthogonal discrimination without reranking.",
        ],
        "claim_boundaries":[
            "No hard-gate target was recovered; active universe remains 384 = 888 - 480 - 24.",
            "Only the 46 targets removed solely for lacking experimental pockets remain recovered.",
            "V18 preserves all frozen scores, candidate ranks, waves, active-species rules and V14 W1 plates.",
            "The 30 pairs were selected using overlapping internal evidence; V18 is descriptive sensitivity analysis, not independent validation.",
            "PubChem vendor-record indicators are not current stock, purity, lead time, jurisdictional availability or endorsement.",
            "W2 and veto readiness artifacts do not authorize procurement, plate assignment or assay execution.",
            "No exact-pair external validation was found for any of the 30 frozen pairs in the inherited frozen audits.",
            "No computational score, identity match, rank or agreement establishes binding, mechanism, efficacy or repurposing.",
            "No real W1 experimental result has been received; assay-released candidates remain zero.",
        ],
        "inputs":{str(path.relative_to(ROOT)):sha256(path) for path in [*required,pair_core]},
    }
    summary_path=OUT/"EVIDENCE_LAYER_ROUTING_SUMMARY_V18.json"; summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    report_path=OUT/"CURRENT_PROGRESS_REPORT_V18_ZH.md"
    report_path.write_text(
        "# V18 当前进展\n\n"
        "靶点宇宙仍为 384 = 888 - 480 - 24；其中 338 个严格实验口袋靶点、46 个仅因缺少实验口袋而恢复。没有恢复任何硬门槛淘汰靶点。部署仍为 185 个 seeded 与 199 个 unseeded，276,480 对核心哈希未变。\n\n"
        "已将身份、化学风险、双向排名和选择偏倚审计从 W1 扩展到全部 30 对候选：28 个唯一全 InChIKey 实体、16 个靶点，W1=8、W2=21、veto=1；seeded=19、unseeded=11。完成 240 个双向排名和 180 个靶点内模型相关性。Boltz 主结果覆盖 29/30，GNINA 覆盖 10/30；16 对模型百分位跨度至少 0.50。\n\n"
        "28 个实体 PubChem 全键精确匹配 28/28；ChEMBL 全键为 27 个完全一致、1 个 serdexmethylphenidate 表示差异。发现 rank 21 的来源标签 ketoconazole 实际冻结结构为 CHEMBL295698/levoketoconazole 全键实体，已设置实体名称裁决 hold，未替换成外消旋 ketoconazole。该对同时缺少主 Boltz 与 GNINA，禁止用多种子元数据填补。\n\n"
        "化学处理分层为 R3=8、R2=17、R1=5。前药为 nitazoxanide、romidepsin、serdexmethylphenidate；romidepsin 新增二硫键还原/二硫醇状态检测。W2 与 veto 只完成计算及实验前设计，不授权采购、板位或湿实验。W1 仍为真实结果 0、实验放行 0。\n",
        encoding="utf-8")
    for path in [review_path,target_path,master_path,summary_path,report_path]:declared[path.relative_to(OUT).as_posix()]=path
    manifest_path=OUT/"EVIDENCE_LAYER_ROUTING_MANIFEST_V18.json"
    manifest={"package":summary["package_name"],"created_utc":datetime.now(timezone.utc).isoformat(),"status":summary["status"],"files":{rel:{"sha256":sha256(path),"bytes":path.stat().st_size} for rel,path in sorted(declared.items())}}
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":summary["status"],"checks_passed":sum(checks.values()),"checks_total":len(checks),"declared_files":len(manifest["files"]),"pair_core_sha256":sha256(pair_core),"summary_sha256":sha256(summary_path),"manifest_sha256":sha256(manifest_path)},ensure_ascii=False,indent=2))
    if not all(checks.values()):
        print(json.dumps({k:v for k,v in checks.items() if not v},indent=2));raise SystemExit(1)


if __name__=="__main__":main()
