#!/usr/bin/env python3
"""Explore frozen catalog disagreements and existing labels; no fitted or deployed changes."""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/model_interpretation_20260920'
MODELS = ['biomaster', 'drugclip', 'dtiam', 'conplex', 'nesso', 'probematch', 'dtbind']
KEYS = ['drug_id', 'target_id']


def top_weights(scores, k):
    cutoff = scores.nlargest(k).iloc[-1]
    above, tied = scores.gt(cutoff), scores.eq(cutoff)
    weights = above.astype(float)
    weights.loc[tied] = (k - int(above.sum())) / int(tied.sum())
    assert abs(weights.sum() - k) < 1e-10
    return weights.to_numpy()


def catalog_diagnostics(shared):
    overlap, landing, zeros, frequencies = [], [], [], []
    for target_id, original in shared.groupby('target_id'):
        g = original.sort_values('drug_id', kind='stable').copy()
        g['biomaster'] = g.biomaster_reverse
        ranks = g[MODELS].rank(ascending=False, method='average')
        weights = {k: {m: top_weights(g[m], k) for m in MODELS} for k in [10, 20, 50, 100]}
        for k, by_model in weights.items():
            for a, b in combinations(MODELS, 2):
                hit = float(by_model[a] @ by_model[b])
                overlap.append(dict(target_id=target_id, model_a=a, model_b=b, k=k,
                                    expected_overlap=hit, overlap_fraction=hit/k, chance_fraction=k/len(g)))
        for a in MODELS:
            wa = weights[10][a]
            for row, w in zip(g.itertuples(), wa):
                if w > 0:
                    frequencies.append(dict(model=a, drug_id=row.drug_id, drug_name=row.drug_name,
                                            target_id=target_id, top10_weight=float(w)))
            for b in MODELS:
                if a == b:
                    continue
                # Receiver membership weights handle ties at its Top100 / half boundary.
                landing.append(dict(target_id=target_id, nominator=a, receiver=b,
                                    mean_receiver_rank=float(wa @ ranks[b].to_numpy()/10),
                                    receiver_top100_fraction=float(wa @ weights[100][b]/10),
                                    receiver_bottom_half_fraction=float(wa @ (1-top_weights(g[b],len(g)//2))/10)))
        nonzero = g[g.conplex.gt(0)]
        for m in [x for x in MODELS if x != 'conplex']:
            zeros.append(dict(target_id=target_id, gene=g.gene.iloc[0], comparator=m,
                              total_pairs=len(g), zero_pairs=int(g.conplex.eq(0).sum()),
                              conplex_top10_cutoff=float(g.conplex.nlargest(10).iloc[-1]),
                              conplex_top10_zero_weight=float(weights[10]['conplex'] @ g.conplex.eq(0).to_numpy()),
                              full_spearman=float(spearmanr(g.conplex,g[m]).statistic),
                              nonzero_pairs=len(nonzero),
                              nonzero_spearman=float(spearmanr(nonzero.conplex,nonzero[m]).statistic)))
    ov = pd.DataFrame(overlap); land = pd.DataFrame(landing); zero = pd.DataFrame(zeros)
    ov.to_csv(OUT/'PER_TARGET_TOPK_OVERLAP.csv',index=False)
    ov.groupby(['model_a','model_b','k'],as_index=False).agg(
        targets=('target_id','size'), mean_overlap=('expected_overlap','mean'),
        mean_overlap_fraction=('overlap_fraction','mean'),chance_fraction=('chance_fraction','mean')
    ).to_csv(OUT/'TOPK_OVERLAP_SUMMARY.csv',index=False)
    land.to_csv(OUT/'PER_TARGET_TOP10_LANDING.csv',index=False)
    land.groupby(['nominator','receiver'],as_index=False).agg(
        targets=('target_id','size'),mean_receiver_rank=('mean_receiver_rank','mean'),
        receiver_top100_fraction=('receiver_top100_fraction','mean'),
        receiver_bottom_half_fraction=('receiver_bottom_half_fraction','mean')
    ).to_csv(OUT/'TOP10_LANDING_SUMMARY.csv',index=False)
    zero.to_csv(OUT/'CONPLEX_ZERO_AND_TOP10.csv',index=False)
    zero.groupby('comparator',as_index=False).agg(targets=('target_id','size'),
        full_spearman=('full_spearman','mean'),nonzero_spearman=('nonzero_spearman','mean'),
        mean_nonzero_pairs=('nonzero_pairs','mean')
    ).to_csv(OUT/'CONPLEX_NONZERO_SENSITIVITY.csv',index=False)
    freq = pd.DataFrame(frequencies).groupby(['model','drug_id','drug_name'],as_index=False).agg(
        top10_weight=('top10_weight','sum'), targets_with_top10_weight=('target_id','size'))
    freq.sort_values(['model','top10_weight'],ascending=[True,False]).to_csv(OUT/'DRUG_TOP10_FREQUENCY.csv',index=False)
    return dict(zero_pairs=int(shared.conplex.eq(0).sum()),
                zero_fraction=float(shared.conplex.eq(0).mean()),
                targets_with_zero_in_top10=int(zero.drop_duplicates('target_id').conplex_top10_zero_weight.gt(0).sum()),
                minimum_top10_cutoff=float(zero.conplex_top10_cutoff.min()),
                constant_drug_queries=int(shared.groupby('drug_id').conplex.nunique().eq(1).sum()))


def nesso_heads(shared):
    old = pd.read_csv(ROOT/'outputs/model_architecture_audit_20260919/NESSO_SAME_BACKBONE_HEADS.csv.gz',float_precision='round_trip')
    lookup = old.set_index(KEYS).to_dict('index')
    rows = []
    for r in shared.itertuples():
        value = lookup.get((r.drug_id,r.target_id))
        if value is None:
            name = f'{r.drug_id}__{r.target_id}'
            paths = [ROOT/f'outputs/catalog_seven_models_20260916/nesso/targets/{r.target_id}/predictions/{name}/affinity.json',
                     ROOT/f'outputs/frontier_dti_20260916/nesso/predictions/{name}/affinity.json']
            raw = json.loads(next(p for p in paths if p.is_file()).read_text())
            value = dict(nesso=raw['affinity_probability_binary'], nesso_pic50=6-raw['affinity_pred_value'],
                         entropy_crop_pl=raw.get('entropy_crop_pl'))
        assert abs(float(value['nesso'])-r.nesso) < 1e-12
        rows.append(dict(drug_id=r.drug_id,target_id=r.target_id,gene=r.gene,nesso=r.nesso,
                         nesso_pic50=value['nesso_pic50'],entropy_crop_pl=value['entropy_crop_pl']))
    heads = pd.DataFrame(rows)
    heads.to_csv(OUT/'NESSO_CATALOG_HEADS.csv.gz',index=False)
    result = []
    for target_id,g in heads.groupby('target_id'):
        for name,part in [('all',g), ('nonzero_crop_entropy',g[g.entropy_crop_pl.gt(0)]),
                          ('binder_lt_0.5',g[g.nesso.lt(.5)]),('binder_ge_0.5',g[g.nesso.ge(.5)]),
                          ('binder_ge_0.8',g[g.nesso.ge(.8)]),
                          ('binder_top50',g.nlargest(50,'nesso'))]:
            result.append(dict(target_id=target_id,gene=g.gene.iloc[0],stratum=name,n=len(part),
                               spearman=float(spearmanr(part.nesso,part.nesso_pic50).statistic) if len(part)>=10 else np.nan))
    by = pd.DataFrame(result)
    by['correlation_pairs'] = by.n.where(by.spearman.notna(),0)
    by.to_csv(OUT/'NESSO_HEAD_STRATA.csv',index=False)
    summary = by.groupby('stratum',as_index=False).agg(targets=('target_id','size'),
        valid_targets=('spearman','count'),pairs=('n','sum'),correlation_pairs=('correlation_pairs','sum'),
        mean_spearman=('spearman','mean'),
        negative_queries=('spearman',lambda s:int(s.lt(0).sum())))
    summary.to_csv(OUT/'NESSO_HEAD_STRATA_SUMMARY.csv',index=False)
    return summary.to_dict('records')


def label_diagnostics(full):
    bench = pd.read_csv(ROOT/'outputs/model_agreement_20260917/BENCHMARK_INPUT_SNAPSHOT.csv',low_memory=False)
    bench = bench[bench.cohort.eq('BINDINGDB479')].dropna(subset=MODELS)
    bench = bench.rename(columns={'dtiam':'dtiam_historical'}).merge(
        full[KEYS+['dtiam','biomaster_reverse']],on=KEYS,validate='one_to_one')
    assert len(bench)==378 and int(bench.label.sum())==103
    bench.to_csv(OUT/'FIXED_378_LABELS_AND_SCORES.csv',index=False)
    metric, joint, single, headcorr = [], [], [], []
    models = MODELS + ['nesso_pic50']
    for scope,part in [('COMMON_378',bench),('AB_UNSEEN_141',bench[bench.ab_unseen.eq(True)])]:
        if scope=='AB_UNSEEN_141':
            assert len(part)==141 and int(part.label.sum())==33
        for m in models:
            metric.append(dict(scope=scope,model=m,n=len(part),positive=int(part.label.sum()),
                               ap=average_precision_score(part.label,part[m]),auroc=roc_auc_score(part.label,part[m])))
        for category,sub in [('all_labels',part),('positive',part[part.label.eq(1)]),('negative',part[part.label.eq(0)])]:
            headcorr.append(dict(scope=scope,label_stratum=category,n=len(sub),
                                 spearman=spearmanr(sub.nesso,sub.nesso_pic50).statistic))
        for key in KEYS:
            for entity,g in part.groupby(key):
                if g.label.nunique()!=2:
                    continue
                scores = g[models].copy()
                if key=='target_id':
                    scores['biomaster']=g.biomaster_reverse
                pos = scores[g.label.eq(1)].to_numpy()
                neg = scores[g.label.eq(0)].to_numpy()
                # Binary labels constrain positive-over-negative order, not order within each class.
                signs = np.sign(pos[:,None,:]-neg[None,:,:]).reshape(-1,len(models))
                for i,m in enumerate(models):
                    auc = float((signs[:,i]>0).mean()+.5*(signs[:,i]==0).mean())
                    assert abs(auc-roc_auc_score(g.label,scores[m]))<1e-12
                    single.append(dict(scope=scope,query=key,entity=entity,model=m,n=len(g),
                                       ordered_comparisons=len(signs),correct_fraction=float((signs[:,i]>0).mean()),
                                       tie_fraction=float((signs[:,i]==0).mean()),auc=auc,
                                       ap=average_precision_score(g.label,scores[m])))
                for i,j in combinations(range(len(models)),2):
                    a,b = signs[:,i],signs[:,j]
                    fractions = dict(both_correct=float(((a>0)&(b>0)).mean()),
                                     only_a_correct=float(((a>0)&(b<0)).mean()),
                                     only_b_correct=float(((a<0)&(b>0)).mean()),
                                     both_wrong=float(((a<0)&(b<0)).mean()),
                                     any_tie=float(((a==0)|(b==0)).mean()))
                    assert abs(sum(fractions.values())-1)<1e-12
                    joint.append(dict(scope=scope,query=key,entity=entity,model_a=models[i],model_b=models[j],
                                      n=len(g),ordered_comparisons=len(signs),**fractions))
    j=pd.DataFrame(joint);s=pd.DataFrame(single)
    j.to_csv(OUT/'PER_QUERY_LABELED_ORDER_AGREEMENT.csv',index=False)
    js=j.groupby(['scope','query','model_a','model_b'],as_index=False).agg(
        queries=('entity','size'),labeled_pairs=('n','sum'),ordered_comparisons=('ordered_comparisons','sum'),
        both_correct=('both_correct','mean'),only_a_correct=('only_a_correct','mean'),
        only_b_correct=('only_b_correct','mean'),both_wrong=('both_wrong','mean'),any_tie=('any_tie','mean'))
    js.to_csv(OUT/'LABELED_ORDER_AGREEMENT_SUMMARY.csv',index=False)
    matched = []
    for row in j[j.model_a.eq('biomaster') & j.model_b.eq('dtiam')].itertuples():
        catalog = full[full[row.query].eq(row.entity)]
        score = 'biomaster_reverse' if row.query=='target_id' else 'biomaster'
        assert len(catalog)==(720 if row.query=='target_id' else 384)
        matched.append(dict(scope=row.scope,query=row.query,entity=row.entity,
                            catalog_n=len(catalog),labeled_n=row.n,
                            catalog_spearman=spearmanr(catalog[score],catalog.dtiam).statistic,
                            both_correct=row.both_correct,only_retargetmap_correct=row.only_a_correct,
                            only_dtiam_correct=row.only_b_correct,both_wrong=row.both_wrong))
    matched = pd.DataFrame(matched)
    matched.to_csv(OUT/'MATCHED_QUERY_CATALOG_VS_LABELS.csv',index=False)
    matched.groupby(['scope','query'],as_index=False).agg(queries=('entity','size'),
        catalog_n_per_query=('catalog_n','mean'),labeled_pairs=('labeled_n','sum'),
        mean_catalog_spearman=('catalog_spearman','mean'),both_correct=('both_correct','mean'),
        only_retargetmap_correct=('only_retargetmap_correct','mean'),
        only_dtiam_correct=('only_dtiam_correct','mean'),both_wrong=('both_wrong','mean')
    ).to_csv(OUT/'MATCHED_QUERY_SUMMARY.csv',index=False)
    s.to_csv(OUT/'PER_QUERY_LABEL_METRICS.csv',index=False)
    s.groupby(['scope','query','model'],as_index=False).agg(queries=('entity','size'),
        labeled_pairs=('n','sum'),ordered_comparisons=('ordered_comparisons','sum'),
        macro_auc=('auc','mean'),macro_ap=('ap','mean'),correct_fraction=('correct_fraction','mean'),
        tie_fraction=('tie_fraction','mean')).to_csv(OUT/'QUERY_LABEL_METRICS.csv',index=False)
    pd.DataFrame(metric).to_csv(OUT/'FIXED_LABEL_METRICS.csv',index=False)
    pd.DataFrame(headcorr).to_csv(OUT/'NESSO_LABELED_HEAD_CORRELATION.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/model_disagreement_literature_20260918/SAME_LABEL_METRICS.csv')
    now=pd.DataFrame(metric);now['scope']=now.scope.replace({'AB_UNSEEN_141':'AB_UNSEEN_COMMON'})
    paired=now.merge(old,on=['scope','model'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==16
    np.testing.assert_allclose(paired.ap_new,paired.ap_old,atol=1e-12,rtol=0)
    return dict(fixed_labels=378,positive=103,ab_unseen_labels=141,ab_unseen_positive=33,
                prior_ap_reproduced=True,scope='Repeatedly examined historical panel; unknown public-model training overlap. Binary label ordering is not affinity ordering.')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.read_csv(OUT/'TOPK_OVERLAP_SUMMARY.csv')
    names=dict(drugclip='DrugCLIP',dtiam='DTIAM A',conplex='ConPLex',nesso='Nesso binder',probematch='ProbeMatchDTI',dtbind='DTBind')
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for m,name in names.items():
        v=s[s.model_a.eq('biomaster') & s.model_b.eq(m)].sort_values('k')
        ax.plot(v.k,100*v.mean_overlap_fraction,marker='o',label=name)
    ax.plot([10,20,50,100],100*np.array([10,20,50,100])/719,'k--',label='Independent uniform reference')
    ax.set(xlabel='K: drugs selected per target',ylabel='Expected TopK overlap / K (%)',
           title='Overlap with ReTargetMap: 50 targets × 719 drugs\nTarget-balanced; ties averaged; agreement is not accuracy')
    ax.set_xticks([10,20,50,100]);ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'TOPK_OVERLAP.png',dpi=180);fig.savefig(OUT/'TOPK_OVERLAP.svg');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=OUT/'latest_results/SEVEN_SHARED_COMPLETE_TARGETS.csv.gz'
    shared=pd.read_csv(source,float_precision='round_trip')
    full=pd.read_csv(OUT/'latest_results/SCORE_SNAPSHOT.csv.gz',float_precision='round_trip')
    assert len(shared)==35950 and shared.groupby('target_id').size().eq(719).all()
    zero=catalog_diagnostics(shared)
    print('catalog diagnostics complete',flush=True)
    heads=nesso_heads(shared)
    print('Nesso head diagnostics complete',flush=True)
    labels=label_diagnostics(full)
    plot()
    result=dict(checks='PASS',shared_targets=50,shared_drugs=719,shared_pairs=len(shared),
                conplex=zero,nesso_head_strata=heads,label_checks=labels,
                limitations=['Exploratory post-hoc diagnostics; no additional training, no model selection or production changes.',
                             'Fifty early-completed targets are not a representative random sample.',
                             'Conditioning on a model output changes the distribution and cannot establish causality.',
                             'Unknown labels remain unknown; candidate overlap is not precision.',
                             'Positive-negative ordering contrasts share entities and observations; they are not independent experiments.'],
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
