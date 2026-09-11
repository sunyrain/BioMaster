#!/usr/bin/env python3
"""Separate rank-gate loss of frozen candidates from review gaps in new pools."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.portable_ranker_v2 import digest

OUT=ROOT/'outputs/biomaster_spr384_ab_rank_explanation_20260911'
REPLAY=ROOT/'outputs/biomaster_original_policy_replay_20260911'
REVIEW=ROOT/'outputs/joint384_comprehensive_20260909'
FINAL=ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'


def main():
    OUT.mkdir(exist_ok=True)
    paths=[REPLAY/'SPR384_ORIGINAL_POLICY_AB_CHECK.csv', REPLAY/'SCREEN_AND_REVIEW_DISPOSITION.csv.gz',
           REPLAY/'ORIGINAL_POLICY_COMPARISON.csv',REVIEW/'FULL574_COMPREHENSIVE_AUDIT.csv',FINAL,
           ROOT/'outputs/biomaster_ab_ranking_selection_20260911/SPR384_MODEL_RANK_AND_RETENTION.csv']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}
    f=pd.read_csv(paths[0]);disposition=pd.read_csv(paths[1]);review=pd.read_csv(paths[3]);final=pd.read_csv(FINAL)
    assert len(f)==f.pair_id.nunique()==384 and set(f.pair_id)==set(final.pair_id)
    out=f[['排序','原候选编号','小分子药物名称','新靶点名称','当前实验建议','pair_id','original_legacy_rank384']].copy()
    out=out.rename(columns={'original_legacy_rank384':'原药内排名_384'})
    bands=['1–10','11–20','21–50','51–100','101–200','201–384']
    bins=[0,10,20,50,100,200,384]
    dist=[];stats=[]
    reverse=pd.read_csv(paths[5]).set_index('pair_id')
    for label,arm in [('A','kdki_inactive'),('B','all_inactive')]:
        ranks=f[label+'_consensus_rank384']
        passed=f[label+'_consensus_passes_original_gate_and_frozen_review']
        # On the original384, no new non-model rejection exists: rank alone reproduces disposition.
        assert passed.equals(ranks.le(20))
        out[label+'药内排名_384']=ranks
        out[label+'排名变化_正数为后移']=ranks-out['原药内排名_384']
        band=pd.cut(ranks,bins,labels=bands)
        out[label+'排名区间']=band
        out[label+'原规则去向']=np.where(passed,'通过原Top20及既有审核','仅因药内排名超过20而未通过')
        votes=sum(f[f'{label}_seed_{s}_rank384'].le(20).astype(int) for s in [20260921,20260922,20260923])
        out[label+'三个种子Top20次数']=votes
        for s in [20260921,20260922,20260923]:out[f'{label}_种子{s}_药内排名']=f[f'{label}_seed_{s}_rank384']
        out[label+'靶点内药物排名_720']=f.pair_id.map(reverse[arm+'_consensus_target_rank720'])
        for name,n in band.value_counts(sort=False).items():dist.append(dict(model=label,rank_band=name,pairs=int(n),percent=100*n/384))
        stats.append(dict(model=label,pairs=384,top20=int(passed.sum()),rank_only_gate_loss=int((~passed).sum()),
                          additional_nonmodel_rejections=0,top50=int(ranks.le(50).sum()),top100=int(ranks.le(100).sum()),
                          mean_rank=float(ranks.mean()),median_rank=float(ranks.median()),
                          q25=float(ranks.quantile(.25)),q75=float(ranks.quantile(.75)),minimum=int(ranks.min()),maximum=int(ranks.max()),
                          all_three_seeds_top20=int(votes.eq(3).sum()),any_seed_top20=int(votes.ge(1).sum())))
    out['共识Top20组合']=np.select([f.A_consensus_rank384.le(20)&f.B_consensus_rank384.le(20),f.A_consensus_rank384.le(20),f.B_consensus_rank384.le(20)],
                                 ['A和B均在Top20','仅A在Top20','仅B在Top20'],default='A和B均未进Top20')
    out.to_csv(OUT/'SPR384_AB_RANKING_EXPLANATION.csv',index=False,encoding='utf-8-sig')
    out[out['共识Top20组合'].eq('A和B均在Top20')].to_csv(OUT/'SPR384_BOTH_TOP20_17.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(dist).to_csv(OUT/'RANK_DISTRIBUTION.csv',index=False)
    pd.DataFrame(stats).to_csv(OUT/'RANK_STATISTICS.csv',index=False)
    rows=[]
    columns=['pair_id','decision','close_negative_review','new_identity_hold','disease_investment_hold','raw_prior_activity_hold','deep_review_investment_hold','reason','deep_reason']
    for model in ['A_consensus','B_consensus']:
        g=disposition[disposition.model.eq(model)].merge(review[columns],on='pair_id',how='left',validate='one_to_one')
        # One reported primary reason per row; retain all source flags for overlapping causes.
        g['primary_reason']=np.select([
            g.review_status.eq('FROZEN_REVIEW_PASSED'),g.review_status.eq('NEW_PAIR_REVIEW_PENDING'),
            g.review_status.eq('KNOWN_IDENTITY_HOLD'),g.decision.isin(['HOLD','EXCLUDE']),
            g.close_negative_review.eq(True),g.disease_investment_hold.eq(True),
            g.raw_prior_activity_hold.eq(True),g.deep_review_investment_hold.eq(True)],
            ['REVIEW_PASSED','NEW_PAIR_REVIEW_PENDING','MOLECULE_IDENTITY_HOLD','PRIOR_AGENT_HOLD_OR_EXCLUSION',
             'CLOSE_NEGATIVE_REFERENCE','DISEASE_CONTEXT_HOLD','PRIOR_ACTIVITY_RECORD','DEEP_REVIEW_HOLD'],default='UNEXPLAINED')
        assert not g.primary_reason.eq('UNEXPLAINED').any()
        rows.append(g)
    reasons=pd.concat(rows,ignore_index=True)
    reasons.to_csv(OUT/'NEW_POOL_REASONS_ALL_ROWS.csv.gz',index=False)
    reasons[~reasons.primary_reason.isin(['REVIEW_PASSED','NEW_PAIR_REVIEW_PENDING'])].to_csv(OUT/'KNOWN_HOLD_DETAILS.csv',index=False,encoding='utf-8-sig')
    reasons.groupby(['model','primary_reason']).size().rename('pairs').reset_index().to_csv(OUT/'NEW_POOL_REASON_COUNTS.csv',index=False)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    summary=dict(status='COMPLETE',input_sha256=hashes,frozen_inputs_unchanged=True,original384_loss_is_rank_gate_only=True,
                 original384_rank_stats=stats,original384_joint_top20=out['共识Top20组合'].value_counts().to_dict(),
                 primary_reason_priority='passed, pending, identity, prior HOLD/EXCLUDE, close-negative, disease, prior-activity, deep-review; all overlapping flags retained',
                 caveats=['Ranks are within each drug among 384 targets, not among the 384 experimental candidates.',
                          'Review-pending is not a failed biological hypothesis; prior activity is not necessarily positive binding.',
                          'This explains deterministic filtering, not neural causal mechanisms or SPR success probability.'],
                 source_sha256=digest(Path(__file__)),artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='SUMMARY.json'})
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(stats).to_string(index=False))
    print(reasons.groupby(['model','primary_reason']).size().to_string())


if __name__=='__main__':main()
