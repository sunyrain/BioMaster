"""Build a 64-target review draft, not a selected 512-pair wet-lab package."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/retargetmap_spr64_target_review_20260909'
ADDITIONS = {
    'DPP4': '代谢及肽类调节', 'HSD11B1': '代谢及激素转换', 'HMGCR': '脂质代谢',
    'ACHE': '胆碱能通路', 'BCHE': '胆碱酯酶选择性', 'MAOA': '单胺代谢',
    'MAOB': '单胺代谢', 'COMT': '儿茶酚胺代谢', 'FAAH': '脂质递质',
    'PTGS2': '炎症脂质介质', 'PTGS1': '炎症脂质介质选择性', 'ALOX5': '炎症脂质介质',
    'EPHX2': '脂质介质代谢', 'PDE4D': '环核苷酸信号', 'PDE9A': '环核苷酸信号',
    'PDE10A': '环核苷酸信号', 'PARP1': 'DNA损伤修复', 'PARP2': 'DNA修复选择性',
    'EZH2': '表观遗传', 'DHODH': '嘧啶合成', 'CTSK': '蛋白水解', 'CTSS': '蛋白水解',
    'PPARD': '核受体及代谢', 'VDR': '核受体', 'MMP13': '基质蛋白水解',
    'ESR2': '雌激素受体选择性', 'F2': '凝血蛋白酶', 'F10': '凝血蛋白酶',
    'JAK1': '细胞因子信号', 'BTK': '免疫信号', 'MAPK14': '应激炎症信号', 'ESR1': '激素受体',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        'registry': ROOT/'outputs/target_discovery_scope_ch37_v3/TARGET_PRIMARY_BIOCHEMICAL_367_V3.csv.gz',
        'old': ROOT/'outputs/retargetmap_spr512_ours_frozen_20260904/RETARGETMAP_SPR512_OURS_FROZEN_TARGET_ROSTER_V2.csv',
        'ot': ROOT/'outputs/current_production_package_v2/full_untruncated_universe_v4/opentargets_top3000_target_completion_v4/opentargets_final1000_target_disease_long.csv',
        'ranks': ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz',
        'routed': ROOT/'outputs/retrain_20260901/bidirectional_720x745_full_fit_rows4/BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SCORES_V1.csv.gz',
    }
    registry = pd.read_csv(paths['registry'])
    old = pd.read_csv(paths['old'])
    assert len(ADDITIONS) == 32 and not set(ADDITIONS).intersection(old.gene_symbol)
    ordered = old.gene_symbol.tolist() + list(ADDITIONS)
    draft = registry.set_index('gene_symbol').loc[ordered].reset_index()
    assert len(draft) == 64 and draft.gene_symbol.is_unique
    assert draft.calibration_8x8.all() and draft.in_supervised_training_target_vocabulary.all()
    rank = pd.read_csv(paths['ranks'], usecols=['ligand_inchikey','gene_symbol','independent_validation_rank_score'])
    rank['select_rank'] = rank.groupby('ligand_inchikey', sort=False).independent_validation_rank_score.rank(method='first', ascending=False)
    routed = pd.read_csv(paths['routed'], usecols=['gene_symbol','routed_rank_within_drug','routed_candidate_target_count'])
    ot = pd.read_csv(paths['ot'])
    ot64 = ot[ot.target_gene.isin(ordered)].copy()
    ot64['association_is_treatment_direction'] = False
    ot64['snapshot'] = 'OpenTargets_26.06_existing_215_target_cache'
    ot64.to_csv(OUT/'OPENTARGETS_64_AVAILABLE_ASSOCIATIONS.csv.gz',index=False)
    oldmap = old.set_index('gene_symbol')
    records=[]
    for row in draft.to_dict('records'):
        g=row['gene_symbol'];original=g in oldmap.index
        core=oldmap.loc[g,'experiment_arm']=='FROZEN_384_CORE_DISCOVERY' if original else bool(row['in_historical_scored_384'])
        rr=rank.loc[rank.gene_symbol.eq(g),'select_rank'] if core else routed.loc[routed.gene_symbol.eq(g),'routed_rank_within_drug']
        ds=ot64[ot64.target_gene.eq(g)].sort_values('overall_score',ascending=False).head(3)
        records.append({
            'gene_symbol':g,'uniprot':row['uniprot_accession'],'origin':'ORIGINAL_32_REVIEW_NOT_AUTOMATIC_RETAIN' if original else 'ADDITIONAL_32_REVIEW',
            'mechanism_review_group':ADDITIONS.get(g,'原靶点：参照原清单及疾病审查'),
            'rank_source':'frozen_validation_rank_384' if core else 'fullfit_biochemical_rank_367',
            'model_target_warm':row['in_supervised_training_target_vocabulary'],
            'positive_reference_compounds':row['positive_compounds'],'negative_reference_compounds':row['negative_compounds'],
            'strict_pocket_ready':row['structure_ready_strict'],
            'model_high_rows_BEFORE_candidate_exclusions':int((rr.le(20 if core else 30)).sum()),
            'model_mid_rows_BEFORE_candidate_exclusions':int(rr.between(80,180).sum()),
            'model_low_rows_BEFORE_candidate_exclusions':int(rr.gt(250).sum()),
            'historical_control':oldmap.loc[g,'positive_control_drug'] if original else '',
            'control_status':'SCOUT_AND_CONSTRUCT_CONFIRMATION_PENDING' if original else 'NOT_YET_SELECTED',
            'ot_cached_association_count':len(ot64[ot64.target_gene.eq(g)]),
            'ot_top3_UNCURATED_NOT_THERAPEUTIC_RECOMMENDATION':' | '.join(ds.disease_name.astype(str)),
            'proposed_high_slots':5,'proposed_mid_slots':1,'proposed_low_slots':1,'proposed_control_slots':1,
            'release_status':'TARGET_REVIEW_ONLY_NOT_LAB_READY',
        })
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'TARGET64_REVIEW_DRAFT.csv',index=False)
    frame[frame.origin.eq('ADDITIONAL_32_REVIEW')].to_csv(OUT/'ADDITIONAL32_REVIEW.csv',index=False)
    registry.assign(already_in_original_32=registry.gene_symbol.isin(old.gene_symbol)).to_csv(OUT/'BIOCHEMICAL367_REVIEW_POOL.csv.gz',index=False)
    summary={
        'status':'TARGET_REVIEW_DRAFT_NOT_FINAL_SELECTION','targets':64,'original_targets':32,'additional_targets':32,
        'requested_pair_budget':512,'proposed_roles':{'high':320,'intermediate':64,'low_background':64,'positive_control':64},
        'candidate_pairs':448,'unique_pairs_not_injections_or_total_cost':True,
        'registry_367_targets_with_8x8_references':int(registry.calibration_8x8.sum()),
        'all64_warm_with_8x8_references':True,
        'new_targets_with_strict_pocket':int(frame[frame.origin.eq('ADDITIONAL_32_REVIEW')].strict_pocket_ready.sum()),
        'targets_with_ot_in_existing_cache':int(frame.ot_cached_association_count.gt(0).sum()),
        'raw_rank_band_shortages':frame[(frame.model_high_rows_BEFORE_candidate_exclusions<5)|(frame.model_mid_rows_BEFORE_candidate_exclusions<1)|(frame.model_low_rows_BEFORE_candidate_exclusions<1)].gene_symbol.tolist(),
        'screened_out_for_raw_high_candidate_shortage':{'SIRT1':0,'PTPN11':2},
        'deprioritized_examples':{'BACE1':'Alzheimer clinical inhibition failures; no auto-priority from disease association','IDO1':'ECHO-301 combination failed; require indication-specific hypothesis'},
        'selection_method':'Manually proposed mechanism-diverse review list, registry and raw rank coverage checked; not validated optimization',
        'unfinished':['additional target controls and constructs','candidate drug-level exclusion and novelty audit','pair-level evidence and disease direction review','drug reuse and chemistry balance optimization','actual 512 pair assignment','budget for repetitions and confirmation'],
        'sources_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
    }
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
