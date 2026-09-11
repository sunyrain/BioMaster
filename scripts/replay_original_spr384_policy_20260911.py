#!/usr/bin/env python3
"""Replay frozen original gates, review evidence, utility and resource policy."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.portable_ranker_v2 import digest
from build_joint384_comprehensive_20260909 import solve

OUT = ROOT / 'outputs/biomaster_original_policy_replay_20260911'
REVIEW = ROOT / 'outputs/joint384_comprehensive_20260909'
AB = ROOT / 'outputs/biomaster_ab_ranking_selection_20260911'
FINAL = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
BUDGETS = [(64,10), (80,10), (96,10), (96,12), (112,10), (112,12)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def original_quality(p):
    return (4*p.usable_chemical_support_ge04.astype(int)
            +1.5*(p.positive_max_tanimoto.ge(.3)&p.positive_max_tanimoto.lt(.4)&~p.chemical_reference_transfer_hold).astype(int)
            +3*p.any_exact_genetic.astype(int)+.5*p.any_exact_disease.astype(int)
            +np.select([p.binding_rank_384.le(5),p.binding_rank_384.le(10)],[2.,1.5],default=1.)
            -.75*p.chemical_risk_points-1.5*p.opposing_near_reference.astype(int)
            +2*p.has_label_based_cross_area.astype(int)+p.cross_area_exact_genetic.astype(int))


def original_review_pass(r):
    return (r.decision.isin(['KEEP_REVIEW','LOW_PRIORITY']) & ~r.close_negative_review & ~r.new_identity_hold
            & ~r.disease_investment_hold & ~r.raw_prior_activity_hold & ~r.deep_review_investment_hold)


def replay_optimizer(p):
    if len(p)<384 or p.ligand_inchikey.nunique()<128:
        return None, dict(status='INSUFFICIENT_REVIEWED_ELIGIBLE_PAIRS', reviewed_eligible=len(p),
                          reviewed_drugs=int(p.ligand_inchikey.nunique()), required_pairs=384, required_min_drugs=128,
                          reason='Original fixed-size portfolio constraints are analytically infeasible; no padding or unreviewed substitutions.'), []
    solutions, stats = {}, []
    for count,cap in BUDGETS:
        selected,meta = solve(p,count,targetcap=cap)
        stats.append(meta)
        if selected is not None:
            solutions[(count,cap)] = selected
    valid = [s for s in stats if 'candidates' in s]
    if not valid:
        return None, dict(status='ORIGINAL_RESOURCE_CONSTRAINTS_INFEASIBLE'), stats
    best_priority=max(s['integrated_priority'] for s in valid)
    best_quality=max(s['evidence_utility'] for s in valid if s['integrated_priority']==best_priority)
    best_chemical=max(s['chemical_ge04'] for s in valid if s['integrated_priority']==best_priority)
    acceptable=[s for s in valid if s['integrated_priority']==best_priority and s['chemical_ge04']==best_chemical
                and s['evidence_utility']>=.95*best_quality]
    chosen=min(acceptable,key=lambda s:(s['target_count'],s['target_cap']))
    return solutions[(chosen['target_count'],chosen['target_cap'])], dict(status='SELECTED384',**chosen), stats


def main():
    OUT.mkdir(exist_ok=True)
    paths=[FINAL, REVIEW/'FULL574_COMPREHENSIVE_AUDIT.csv', REVIEW/'RECOMMENDED_CANDIDATES_384.csv',
           REVIEW/'TARGET_BUDGET_COMPARISON.csv', AB/'CORE276480_SELECTION_AUDIT.parquet',
           AB/'old_production_DIRECTIONAL_LOGITS.npy', ROOT/'scripts/build_joint384_comprehensive_20260909.py',
           ROOT/'scripts/screen_joint_720x384_20260909.py']
    frozen={str(p.relative_to(ROOT)):digest(p) for p in paths}
    write_json(OUT/'PROTOCOL.json',dict(input_sha256=frozen,
        operation='Restore original selection pipeline; replace entire initial scoring source only, not retrain or reconstruct a hybrid score.',
        full_scope='Same 720 drugs x 384 targets, ranking before all exclusions, pandas rank(method=first) in original row order',
        original_screen='Frozen base_eligible; new within-drug rank<=20; TxGNN top50 / OT>=0.3 same disease; three biochemical lanes',
        original_review='Recompute frozen574 review-pass formula; missing review remains pending, never assumed passed',
        inherited_identity_hold='Previously established molecule identity hold applies to other pairs for that same exact molecule',
        original_utility='Unchanged original_quality + 20*integrated_priority; original solve() also rewards distinct drugs by 0.001',
        resource_budgets=BUDGETS, drug_and_connectivity_cap=4, minimum_drugs=128, candidate_count=384,
        budget_policy='Same max integrated priority, max usable chemical support, >=95% best evidence utility, then lowest target count/cap',
        not_done='No transfer of review approval to new pairs; no lowering thresholds; no fixed per-target slots; no replacement of wet-lab list'))
    full=pd.read_parquet(AB/'CORE276480_SELECTION_AUDIT.parquet').reset_index(drop=True)
    r=pd.read_csv(REVIEW/'FULL574_COMPREHENSIVE_AUDIT.csv')
    final=pd.read_csv(FINAL)
    assert len(full)==720*384 and full.pair_id.is_unique and len(r)==574 and r.pair_id.is_unique
    reviewpass=original_review_pass(r)
    assert reviewpass.equals(r.eligible_after_comprehensive_review) and reviewpass.sum()==449
    finalids=set(final.pair_id)
    gate=(full.base_eligible & full.joint_r50_ot03.eq(True)
          & full.assay_lane.isin(['ENZYME_BIOCHEMICAL','KINASE_BIOCHEMICAL','NUCLEAR_EPIGENETIC_DOMAIN']))
    scorecols={'original_legacy':'independent_validation_rank_score'}
    tensor=np.load(AB/'old_production_DIRECTIONAL_LOGITS.npy')
    full['old_neural_exact_head0']=tensor[full.di.to_numpy(),full.ti.to_numpy(),0]
    scorecols['old_neural']='old_neural_exact_head0'
    for arm,label in [('kdki_inactive','A'),('all_inactive','B')]:
        for seed in [20260921,20260922,20260923]:
            scorecols[f'{label}_seed_{seed}']=f'{arm}_{seed}_score'
        scorecols[label+'_consensus']=arm+'_consensus_score'
    scorecols['DTIAM']='dtiam_probability'
    oldapproved=set(r.loc[reviewpass,'pair_id'])
    identityholds=set(r.loc[r.new_identity_hold,'ligand_inchikey'])
    records, replaytables, pendingtables, budgets = [], [], [], []
    detail=final[['排序','原候选编号','小分子药物名称','新靶点名称','当前实验建议','pair_id']].copy()
    for model,column in scorecols.items():
        rank=full.groupby('ligand_inchikey',sort=False)[column].rank(method='first',ascending=False).astype(int)
        if model=='original_legacy':
            assert rank.eq(full.binding_rank_384).all()
            assert set(full.loc[gate & rank.le(20),'pair_id'])==set(r.pair_id)
        eligible_gate=gate & rank.le(20)
        passed=full.loc[eligible_gate,['pair_id','ligand_inchikey','drug_names','gene_symbol','target_chembl_id','assay_lane']].copy()
        passed['replacement_rank384']=rank.loc[passed.index]
        passed['model']=model
        passed['review_status']=np.select([passed.ligand_inchikey.isin(identityholds),passed.pair_id.isin(oldapproved),passed.pair_id.isin(r.pair_id)],
                                          ['KNOWN_IDENTITY_HOLD','FROZEN_REVIEW_PASSED','FROZEN_REVIEW_HOLD_OR_EXCLUSION'],default='NEW_PAIR_REVIEW_PENDING')
        passed['in_frozen384']=passed.pair_id.isin(finalids)
        replaytables.append(passed)
        pendingtables.append(passed[passed.review_status.eq('NEW_PAIR_REVIEW_PENDING')])
        reviewed=r.merge(passed[['pair_id','replacement_rank384','review_status']],on='pair_id',validate='one_to_one')
        reviewed=reviewed[reviewed.review_status.eq('FROZEN_REVIEW_PASSED')].copy()
        reviewed['historical_rank384']=reviewed.binding_rank_384
        reviewed['binding_rank_384']=reviewed.replacement_rank384
        reviewed['primary_joint_selected']=True
        reviewed['integrated_priority']=(reviewed.eligible_after_comprehensive_review & reviewed.has_label_based_cross_area
                                        & reviewed.cross_area_exact_genetic & reviewed.usable_chemical_support_ge04 & ~reviewed.opposing_near_reference)
        reviewed['evidence_utility_unvalidated']=original_quality(reviewed)
        reviewed['optimization_utility_unvalidated']=reviewed.evidence_utility_unvalidated+20*reviewed.integrated_priority.astype(int)
        reviewed['replay_status']='PASSES_ORIGINAL_REVIEW_AND_REPLACEMENT_TOP20_NOT_A_FULL384_PORTFOLIO'
        selected,meta,attempts=replay_optimizer(reviewed)
        budgets.extend(dict(model=model,**s) for s in attempts)
        if selected is not None:
            selected=selected.sort_values(['integrated_priority','evidence_utility_unvalidated','binding_rank_384','gene_symbol','drug_names'],
                                         ascending=[False,False,True,True,True])
            selected.to_csv(OUT/f'{model}_ORIGINAL_POLICY_SELECTED384.csv',index=False,encoding='utf-8-sig')
            if model=='original_legacy':
                assert set(selected.pair_id)==finalids
                frozen_selected=pd.read_csv(REVIEW/'RECOMMENDED_CANDIDATES_384.csv').set_index('pair_id')
                assert selected.pair_id.tolist()==frozen_selected.index.tolist()
                np.testing.assert_array_equal(selected.evidence_utility_unvalidated, frozen_selected.loc[selected.pair_id,'evidence_utility_unvalidated'])
        reviewed.sort_values(['integrated_priority','evidence_utility_unvalidated','binding_rank_384'],ascending=[False,False,True]).to_csv(
            OUT/f'{model}_REVIEWED_ELIGIBLE_POOL.csv',index=False,encoding='utf-8-sig')
        rankmap=pd.Series(rank.to_numpy(),index=full.pair_id)
        detail[model+'_rank384']=detail.pair_id.map(rankmap)
        detail[model+'_passes_original_gate_and_frozen_review']=detail.pair_id.isin(reviewed.pair_id)
        row=dict(model=model,screen_pass_before_review=len(passed),reviewed_eligible=len(reviewed),
                 retained_original384_in_eligible_pool=len(set(reviewed.pair_id)&finalids),
                 pending_new_pair_reviews=int(passed.review_status.eq('NEW_PAIR_REVIEW_PENDING').sum()),
                 known_holds=int(passed.review_status.isin(['KNOWN_IDENTITY_HOLD','FROZEN_REVIEW_HOLD_OR_EXCLUSION']).sum()),
                 complete384_selected=selected is not None,selected384_overlap=None if selected is None else len(set(selected.pair_id)&finalids),
                 reviewed_drugs=int(reviewed.ligand_inchikey.nunique()),reviewed_targets=int(reviewed.target_chembl_id.nunique()),
                 solver_status=meta['status'])
        records.append(row)
        write_json(OUT/f'{model}_OPTIMIZER_STATUS.json',meta)
        print(row,flush=True)
    pd.DataFrame(records).to_csv(OUT/'ORIGINAL_POLICY_COMPARISON.csv',index=False)
    pd.DataFrame(budgets).to_csv(OUT/'ORIGINAL_POLICY_BUDGET_ATTEMPTS.csv',index=False)
    pd.concat(replaytables,ignore_index=True).to_csv(OUT/'SCREEN_AND_REVIEW_DISPOSITION.csv.gz',index=False)
    pd.concat(pendingtables,ignore_index=True).to_csv(OUT/'NEW_PAIR_REVIEW_QUEUE.csv.gz',index=False)
    detail.to_csv(OUT/'SPR384_ORIGINAL_POLICY_AB_CHECK.csv',index=False,encoding='utf-8-sig')
    assert all(digest(ROOT/p)==h for p,h in frozen.items())
    write_json(OUT/'SUMMARY.json',dict(status='COMPLETED_ORIGINAL_POLICY_REPLAY',original384_exactly_reproduced=True,
        original_selection_order_and_utility_exactly_reproduced=True,frozen_inputs_unchanged=True,
        original_reviewed_eligible_pairs=449,no_unreviewed_pairs_promoted=True,no_padding=True,
        experiments_and_website_unchanged=True,new_model_training=False,results=records,
        source_sha256=digest(Path(__file__)),artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='SUMMARY.json'}))


if __name__=='__main__':
    main()
