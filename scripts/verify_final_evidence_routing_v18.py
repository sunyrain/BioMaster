#!/usr/bin/env python3
"""Independent end-to-end verifier for the final V18 full-candidate package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"outputs/evidence_routing_compute_execution_20260808_v1"
FINAL=RUN/"final_evidence_routing_v18"
SUMMARY=FINAL/"EVIDENCE_LAYER_ROUTING_SUMMARY_V18.json"
MANIFEST=FINAL/"EVIDENCE_LAYER_ROUTING_MANIFEST_V18.json"
AUDIT=FINAL/"UNIFIED_V18_REPRODUCIBILITY_AUDIT.json"
PAIR_NAME="PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V10.csv.gz"
PAIR_HASH="8e1c73a0b16122bee6e4d3d7ca5dd9dbc816759c63a2976ec57799cdf97b7a9a"
CACHE={}


def sha256(path:Path)->str:
    resolved=path.resolve()
    if resolved in CACHE:return CACHE[resolved]
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    CACHE[resolved]=h.hexdigest();return CACHE[resolved]


def main()->None:
    summary=json.loads(SUMMARY.read_text());manifest=json.loads(MANIFEST.read_text())
    v17audit=json.loads((FINAL/"UNIFIED_V17_REPRODUCIBILITY_AUDIT.json").read_text())
    portaudit=json.loads((FINAL/"FULL30_PORTFOLIO_INDEPENDENT_AUDIT_V18.json").read_text())
    robust=json.loads((FINAL/"FULL30_COMPUTATIONAL_ROBUSTNESS_SUMMARY_V18.json").read_text())
    identity=json.loads((FINAL/"FULL30_IDENTITY_CHEMISTRY_READINESS_SUMMARY_V18.json").read_text())
    target17=pd.read_csv(FINAL/"TARGET_EVIDENCE_LAYER_ROUTING_384_V17.csv",low_memory=False)
    target18=pd.read_csv(FINAL/"TARGET_EVIDENCE_LAYER_ROUTING_384_V18.csv",low_memory=False)
    master17=pd.read_csv(FINAL/"MASTER_VALIDATION_QUEUE_63_ROWS_V17.csv",low_memory=False)
    master18=pd.read_csv(FINAL/"MASTER_VALIDATION_QUEUE_63_ROWS_V18.csv",low_memory=False)
    review=pd.read_csv(FINAL/"FULL30_CANDIDATE_PORTFOLIO_REVIEW_V18.csv",low_memory=False)
    entities=pd.read_csv(FINAL/"FULL28_CHEMBL37_PUBCHEM_IDENTITY_V18.csv",low_memory=False)
    species=pd.read_csv(FINAL/"FULL28_PRODRUG_ACTIVE_SPECIES_HANDLING_3_V18.csv",low_memory=False)
    routes=pd.read_csv(FINAL/"FULL30_ASSAY_ROUTE_AND_AUTHORIZATION_V18.csv",low_memory=False)
    ranks=pd.read_csv(FINAL/"FULL30_BIDIRECTIONAL_MODEL_RANKS_240_V18.csv",low_memory=False)
    concord=pd.read_csv(FINAL/"FULL30_TARGET_MODEL_CONCORDANCE_180_V18.csv",low_memory=False)
    physical=pd.read_csv(FINAL/"FULL30_PHYSICAL_MODEL_COVERAGE_AND_RANKS_30_V18.csv",low_memory=False)
    independent=pd.read_csv(FINAL/"FULL30_SELECTION_BIAS_AND_EVIDENCE_INDEPENDENCE_30_V18.csv",low_memory=False)
    v17status=pd.read_csv(FINAL/"W1_RESULT_INGESTION_MASTER_STATUS_8_V17.csv",low_memory=False)
    pair=pd.read_csv(FINAL/PAIR_NAME,usecols=["ligand_inchikey","target_chembl_id","old_drug_target_deployment_branch_v10"],low_memory=False)
    missing=[rel for rel in manifest["files"] if not (FINAL/rel).is_file()]
    manifest_valid=not missing and all(sha256(FINAL/rel)==entry["sha256"] and (FINAL/rel).stat().st_size==entry["bytes"] for rel,entry in manifest["files"].items())
    input_valid=all((ROOT/rel).is_file() and sha256(ROOT/rel)==value for rel,value in summary["inputs"].items())
    dirs={d:len(list((FINAL/d).glob("*"))) for d in ["pubchem_control_identity_raw_v14","pubchem_candidate_identity_raw_v15","input_templates_v17","dry_run_evaluation_v17","pubchem_full28_identity_raw_v18"]}
    candidate_rows=master18[master18.validation_layer.eq("L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS")]
    noncandidate=master18[~master18.validation_layer.eq("L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS")]
    expected_drugs=["omaveloxolone","etonogestrel","nitazoxanide","acoramidis","serdexmethylphenidate","lorlatinib","elacestrant","ixazomib","indomethacin","ixazomib","maralixibat","cholecalciferol","repotrectinib","calcitriol","dronabinol","romidepsin","rucaparib","calcipotriene","calcitriol","ganaxolone","ketoconazole","brexanolone","olodaterol","vamorolone","ulipristal","pitavastatin","paroxetine","paricalcitol","levamlodipine","selinexor"]
    hold=review[review.identity_execution_status.str.contains("HOLD")]
    high=review[review.rank_discordance_flag.eq("HIGH_RANGE_GE_0.50")]
    checks={
        "summary_pass_exact19_builder_checks":summary["status"]=="PASS" and len(summary["checks"])==19 and all(summary["checks"].values()),
        "manifest_pass_exact252_declared_files":manifest["status"]=="PASS" and len(manifest["files"])==252,
        "manifest_excludes_self_and_posthoc_audit":MANIFEST.name not in manifest["files"] and AUDIT.name not in manifest["files"],
        "manifest_no_missing_files":not missing,
        "all_manifest_hashes_sizes_recomputed":manifest_valid,
        "all_summary_inputs_rehashed":input_valid,
        "inherited_v17_and_v18_audits_pass":v17audit["checks_passed"]==v17audit["checks_total"]==40 and portaudit["checks_passed"]==portaudit["checks_total"]==24,
        "both_v18_builder_summaries_pass20":robust["status"]==identity["status"]=="PASS" and len(robust["checks"])==len(identity["checks"])==20 and all(robust["checks"].values()) and all(identity["checks"].values()),
        "raw_and_input_directory_counts_exact":dirs=={"pubchem_control_identity_raw_v14":51,"pubchem_candidate_identity_raw_v15":9,"input_templates_v17":5,"dry_run_evaluation_v17":6,"pubchem_full28_identity_raw_v18":29},
        "pair_hash_preserved_v10_v18":sha256(FINAL/PAIR_NAME)==PAIR_HASH,
        "pair_shape_276480_720_384":len(pair)==276480 and pair.ligand_inchikey.nunique()==720 and pair.target_chembl_id.nunique()==384,
        "every_drug_384_pairs":pair.groupby("ligand_inchikey").size().eq(384).all(),
        "target18_384_unique_ids_uniprots":len(target18)==target18.target_chembl_id.nunique()==target18.uniprot_accession.nunique()==384,
        "all_v17_target_rows_preserved":target17.equals(target18[target17.columns]),
        "universe_888_480_24_384":(
            summary["universe"]["official_targets_before_gates"]==888
            and summary["universe"]["hard_gate_excluded"]==480
            and summary["universe"]["later_pocket_quality_excluded"]==24
            and summary["universe"]["active_targets"]==384
        ),
        "pocket_partition338_46":target18.active_target_branch.value_counts().to_dict()=={"STRICT_EXPERIMENTAL_POCKET_MAINLINE_338":338,"RECOVERED_NO_EXPERIMENTAL_POCKET_46":46},
        "deployment_partition185_199":target18.old_drug_target_deployment_branch_v10.value_counts().to_dict()=={"UNSEEDED_TARGET_DTA_199":199,"SEEDED_KNOWN_GRAPH_185":185},
        "target_aggregates_30_pairs_16_targets":target18.v18_candidate_pair_count.sum()==30 and target18.v18_candidate_pair_count.gt(0).sum()==16,
        "all_v17_master_rows_preserved":master17.equals(master18[master17.columns]),
        "master63_layer_partition":len(master18)==63 and master18.validation_layer.value_counts().to_dict()=={"L2_PROSPECTIVE_COMPUTATIONAL_HYPOTHESIS":30,"L0_EXTERNAL_SOURCE_REDISCOVERY_CONTROL":17,"L1_W1_TARGET_MATCHED_POSITIVE_CONTROL":8,"L1_W1_TARGET_MATCHED_NUMERIC_WEAK_CONTROL":8},
        "exact30_candidate_rows_annotated_no_controls":candidate_rows.v18_full_candidate_portfolio_scope.eq("FULL30_IDENTITY_READINESS_AND_INTERNAL_ROBUSTNESS_AUDITED").all() and noncandidate.v18_full_candidate_portfolio_scope.eq("NOT_IN_FROZEN_TOP30_PAIR_SCOPE").all(),
        "review30_exact_order_names":len(review)==30 and review.candidate_rank.tolist()==list(range(1,31)) and review.drug_name.str.lower().tolist()==expected_drugs,
        "review28_entities16targets":review.project_full_inchikey.nunique()==28 and review.target_chembl_id.nunique()==16,
        "wave_partition8_21_1":review.execution_wave.value_counts().to_dict()=={"W2_CONTINGENT_ONLY":21,"W1_BLINDED_CANDIDATE_PILOT":8,"VETO_NOT_AUTHORIZED":1},
        "deployment19_11":review.deployment_branch.value_counts().to_dict()=={"SEEDED_KNOWN_GRAPH_185":19,"UNSEEDED_TARGET_DTA_199":11},
        "entities28_pubchem_exact_and_chembl27plus1":len(entities)==28 and entities.pubchem_identity_status.eq("EXACT_FULL_INCHIKEY_MATCH").all() and entities.project_vs_chembl_full_inchikey_status.value_counts().to_dict()=={"EXACT_FULL_INCHIKEY_MATCH":27,"FULL_INCHIKEY_REPRESENTATION_MISMATCH":1},
        "rank21_only_name_hold_levoketoconazole_key":len(hold)==1 and hold.iloc[0].candidate_rank==21 and hold.iloc[0].modeled_chembl_preferred_entity=="LEVOKETOCONAZOLE" and hold.iloc[0].project_full_inchikey=="XMAYWYJOQHXEEK-ZEQKJWHPSA-N",
        "prodrugs_exact3_species_routes_never_merge":set(entities.loc[entities.prodrug.eq(1),"pref_name"])=={"NITAZOXANIDE","ROMIDEPSIN","SERDEXMETHYLPHENIDATE"} and len(species)==3 and species.entity_merge_policy.str.startswith("NEVER_MERGE").all(),
        "risk_partition8_17_5":review.experimental_handling_risk_tier.value_counts().to_dict()=={"R2_MODERATE_COUNTERASSAY":17,"R3_HIGH_SPECIAL_HANDLING":8,"R1_STANDARD":5},
        "routes30_and_authorization8_21_1":len(routes)==30 and routes.execution_authorization.str.startswith("W1_").sum()==8 and routes.execution_authorization.str.startswith("W2_").sum()==21 and routes.execution_authorization.str.startswith("VETO_").sum()==1,
        "w2_veto_no_procurement_assay_authorization":review.loc[review.execution_wave.ne("W1_BLINDED_CANDIDATE_PILOT"),"portfolio_decision_state"].str.contains("NOT_PROCUREMENT_OR_ASSAY_AUTHORIZED|NO_PROCUREMENT_OR_ASSAY_AUTHORIZED").all(),
        "ranks240_and_historical_ties3":len(ranks)==240 and set(ranks.loc[ranks.stored_rank_recalculation_status.str.contains("HISTORICAL_MIN_TIE"),"candidate_rank"])=={12,19,23},
        "concordance180":len(concord)==180 and concord.groupby("candidate_rank").size().eq(6).all(),
        "high_discordance16_priority_route":len(high)==16 and high.orthogonal_priority_reason.str.contains("HIGH_MODEL_PERCENTILE_RANGE").all(),
        "physical29_boltz10_gnina_rank21_missing":physical.boltz_primary_completed.sum()==29 and physical.gnina_primary_completed.sum()==10 and physical.loc[physical.candidate_rank.eq(21),"physical_metadata_consistency_flag"].tolist()==["MULTISEED_SUPPORT_METADATA_WITHOUT_MAIN_RESULT_DO_NOT_IMPUTE"],
        "external_validation_zero30":len(independent)==30 and ~independent.exact_pair_external_evidence_present.any() and review.independent_exact_pair_validation_status.eq("NONE_IN_FROZEN_CHEMBL_BINDINGDB_AND_LITERATURE_AUDITS").all(),
        "w1_real_results_release_corroboration_zero":summary["v17_result_ingestion"]["real_result_rows_received"]==0 and summary["v17_result_ingestion"]["assay_released_candidates"]==0 and summary["v17_result_ingestion"]["technically_corroborated_pairs"]==0 and ~v17status.v17_assay_released.astype(bool).any(),
        "claims_preserve_hardgate_only46_and_no_binding_authorization":any("No hard-gate target was recovered" in x for x in summary["claim_boundaries"]) and any("Only the 46 targets" in x for x in summary["claim_boundaries"]) and any("do not authorize procurement" in x for x in summary["claim_boundaries"]) and any("establishes binding" in x for x in summary["claim_boundaries"]),
    }
    checks={k:bool(v) for k,v in checks.items()}
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"checks_passed":sum(checks.values()),"checks_total":len(checks),"key_hashes":{"pair_core_sha256":sha256(FINAL/PAIR_NAME),"summary_sha256":sha256(SUMMARY),"manifest_sha256":sha256(MANIFEST),"review_sha256":sha256(FINAL/"FULL30_CANDIDATE_PORTFOLIO_REVIEW_V18.csv")},"independent_interpretation":"V18 preserves the 384-target universe and only 46 no-pocket recoveries, extends audited readiness and sensitivity to all 30 frozen pairs, holds the ketoconazole-labeled levoketoconazole entity, leaves missing physical evidence missing, and authorizes neither W2 nor veto execution."}
    AUDIT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks_passed":report["checks_passed"],"checks_total":report["checks_total"],"failed_checks":[k for k,v in checks.items() if not v],"audit_sha256":sha256(AUDIT)},ensure_ascii=False,indent=2))
    if report["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
