#!/usr/bin/env python3
"""Release historical tests only after the complete validation selection freezes."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from biomaster.odti_v3_incremental import IncrementalConfig,IncrementalV3
from biomaster.old_drug_ranking import known_ranking,observed_ranking
from train_biomaster_v3_incremental_ablation import Runtime,two_class_groups
from prepare_biomaster_v3_incremental_ablation import OUT,OLD
from build_biomaster_odti_v4_features import sha256,write_json
from summarize_old_drug_bidirectional_20260906 import annotate
from train_biomaster_v3_old_relation_ablation import STAGE, source_identity as stage2_source_identity


def paired_interval(first,second,confidence=.95):
    aligned=first.set_index('query_id').ap.to_frame('first').join(second.set_index('query_id').ap.rename('second'),how='outer')
    assert aligned.notna().all().all()
    delta=(aligned['first']-aligned['second']).to_numpy()
    rng=np.random.default_rng(20260906)
    samples=np.mean(delta[rng.integers(len(delta),size=(10000,len(delta)))],axis=1)
    alpha=(1-confidence)/2
    lo,hi=np.quantile(samples,[alpha,1-alpha])
    return {'queries':len(delta),'delta_ap':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),
            'confidence':confidence,'bootstrap':'10000 paired query resamples; seed variability reported separately'}


def main():
    torch.set_num_threads(4)
    selection=json.loads((OUT/'SELECTION.json').read_text())
    assert selection['status']=='FROZEN_BEFORE_TEST' and not selection['selection_used_test']
    assert selection['protocol_sha256']==sha256(OUT/'PROTOCOL.json')
    assert selection['training_manifest_sha256']==sha256(OUT/'TRAINING_MANIFEST.json')
    rt=Runtime()
    assert all(row['identity']==rt.identity for row in selection['results'])
    stage2=json.loads((STAGE/'SELECTION.json').read_text())
    assert stage2['status']=='FROZEN_BEFORE_TEST' and not stage2['test_used_for_selection']
    assert stage2['stage1_selection_sha256']==sha256(OUT/'SELECTION.json')
    assert stage2['identity']['backbone_and_cache']==rt.identity
    assert stage2['identity']['protocol_sha256']==sha256(STAGE/'PROTOCOL.json')
    assert stage2['identity']['source_sha256']==stage2_source_identity()
    stage2_protocol=json.loads((STAGE/'PROTOCOL.json').read_text())
    variants={**rt.protocol['variants'],**stage2_protocol['variants']}
    all_runs=[(OUT,r) for r in selection['results']]+[(STAGE,r) for r in stage2['results']]
    winner=stage2['winner']
    release={'selection_sha256':sha256(OUT/'SELECTION.json'),'identity':rt.identity,
             'stage2_selection_sha256':sha256(STAGE/'SELECTION.json'),'winner':winner,
             'claim':'historical development regression; test reviewed in previous rounds, not untouched external confirmation'}
    if (OUT/'TEST_RELEASE.json').exists():
        assert json.loads((OUT/'TEST_RELEASE.json').read_text())==release
    else:
        write_json(OUT/'TEST_RELEASE.json',release)
    score_dir=OUT/'scores';score_dir.mkdir(exist_ok=True)
    ev=OUT/'evaluation';ev.mkdir(exist_ok=True)
    test=rt.positions['test']
    test_y=rt.labels[test]
    score_hashes={}

    def save(name,old,observed):
        assert old.shape==(720,384,2) and observed.shape==(len(test),2)
        assert np.isfinite(old).all() and np.isfinite(observed).all()
        path=score_dir/f'{name}.npz'
        np.savez_compressed(path,old=old,observed_test=observed)
        score_hashes[name]=sha256(path)

    save('frozen_v3',rt.predict(None,'OLD',np.arange(720*384)).reshape(720,384,2),rt.predict(None,'RELATION',test))
    for run_root,row in all_runs:
        variant,seed=row['variant'],row['seed']
        name=f'{variant}__{seed}'
        checkpoint_path=run_root/'runs'/variant/f'seed_{seed}'/'BEST.pt'
        assert sha256(checkpoint_path)==row['checkpoint_sha256']
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert checkpoint['identity']==row['identity'] and checkpoint['epoch']==row['selected_epoch']
        model=IncrementalV3(IncrementalConfig(**checkpoint['config'])).cuda()
        model.load_state_dict(checkpoint['model'])
        if model.config.pretrained:
            rt.load_pretrained()
        old=rt.predict(model,'OLD',np.arange(720*384)).reshape(720,384,2)
        observed=rt.predict(model,'RELATION',test)
        if checkpoint['epoch']==0:
            with np.load(score_dir/'frozen_v3.npz') as baseline:
                assert np.array_equal(old,baseline['old']) and np.array_equal(observed,baseline['observed_test'])
        save(name,old,observed)
        print(json.dumps({'scored':name,'selected_epoch':checkpoint['epoch']}),flush=True)
    baseline=np.load(score_dir/'frozen_v3.npz')
    for variant in variants:
        members=[np.load(score_dir/f'{variant}__{seed}.npz') for seed in rt.protocol['seeds']]
        # Mean residuals preserve an exactly unchanged backbone when all seeds
        # select epoch zero, avoiding roundoff from summing the same base thrice.
        save(variant,baseline['old']+np.mean([m['old']-baseline['old'] for m in members],axis=0),
             baseline['observed_test']+np.mean([m['observed_test']-baseline['observed_test'] for m in members],axis=0))
        for m in members:m.close()
    support=rt.banks['RELATION']['SUPPORT'][torch.as_tensor(test,device='cuda')].cpu().numpy()
    nearest=support[:,1]
    pn=json.loads((ROOT/'outputs/biomaster_odti_v4_20260905/evaluation/PN_BASELINE_V4.json').read_text())
    pn_score=((support-np.asarray(pn['scaler_mean']))/np.asarray(pn['scaler_scale']))@np.asarray(pn['coefficient'])[0]+pn['intercept'][0]
    for name,values in [('positive_nearest',nearest),('pn_logistic',pn_score)]:
        with np.load(OLD/'scores'/f'{name}.npz') as legacy:
            save(name,np.stack([legacy['drug_to_target'],legacy['target_to_drug']],axis=-1),np.repeat(values[:,None],2,axis=1))
    write_json(score_dir/'MANIFEST.json',{'scores_sha256':score_hashes,'selection_sha256':sha256(OUT/'SELECTION.json'),
                                       'stage2_selection_sha256':sha256(STAGE/'SELECTION.json')})

    with np.load(OLD/'LABELS_AND_SCOPES.npz') as f:
        labels={key:f[key] for key in f.files}
    scopes={'project_720x384':np.arange(720),'old_validation_52':rt.old_val,
            'historical_test_54':np.flatnonzero(labels['historical_test_drugs']),
            'absent_old_drugs_223':np.flatnonzero(labels['absent_drugs'])}
    summary,queries,measured_rows=[],[],[]
    general_groups=[two_class_groups(test_y,ids[test]) for ids in [rt.drugs,rt.targets]]
    for name in score_hashes:
        prediction=np.load(score_dir/f'{name}.npz')
        values=prediction['old']
        for scope,di in scopes.items():
            y=labels['known_relationship'][di]
            measured_y=labels['observed_binary'][di]
            ds=rt.old_drugs.ligand_inchikey.to_numpy()[di]
            ts=rt.old_targets.target_chembl_id.to_numpy()
            for direction,known,scores,q,c,obs in [('d2t',y,values[di,:,0],ds,ts,measured_y),
                                                  ('t2d',y.T,values[di,:,1].T,ts,ds,measured_y.T)]:
                result,query,_=known_ranking(known,scores,q,c)
                measured,_=observed_ranking(obs,scores,q,c)
                query.insert(0,'direction',direction);query.insert(0,'model',name);query.insert(0,'scope',scope)
                queries.append(query)
                summary.append({'scope':scope,'model':name,'direction':direction,
                    'ap':result['macro_ap'],'r5':result['macro_recall_at_5'],'r10':result['macro_recall_at_10'],
                    'r20':result['macro_recall_at_20'],'mrr':result['macro_mrr'],'ndcg20':result['macro_ndcg_at_20'],
                    'queries':result['evaluated_queries'],'candidates':result['candidates_per_query'],
                    'known_pairs':result['known_relationships'],'observed_ap':measured['macro_ap'],
                    'observed_auroc':measured['macro_auroc'],'observed_queries':measured['two_class_queries']})
        for i,direction in enumerate(['d2t','t2d']):
            s=prediction['observed_test'][:,i]
            measured_rows.append({'model':name,'direction':direction,'rows':len(test),'queries':len(general_groups[i]),
                'macro_ap':float(np.mean([average_precision_score(test_y[g],s[g]) for g in general_groups[i]])),
                'macro_auroc':float(np.mean([roc_auc_score(test_y[g],s[g]) for g in general_groups[i]]))})
        prediction.close()
        print(json.dumps({'evaluated':name}),flush=True)
    frame=pd.DataFrame(summary);frame.to_csv(ev/'SUMMARY.csv',index=False)
    query_frame=pd.concat(queries,ignore_index=True);query_frame.to_csv(ev/'QUERY_METRICS.csv.gz',index=False)
    pd.DataFrame(measured_rows).to_csv(ev/'OBSERVED_TEST.csv',index=False)

    def qselect(scope,name,direction):
        return query_frame[query_frame.scope.eq(scope)&query_frame.model.eq(name)&query_frame.direction.eq(direction)]

    contrasts=[]
    defined_contrasts={**rt.protocol['factorial_contrasts'],**{f'oldtask_{k}':v for k,v in stage2_protocol['contrasts'].items()}}
    for contrast,(added,control) in defined_contrasts.items():
        for scope in ['old_validation_52','historical_test_54','absent_old_drugs_223']:
            for direction in ['d2t','t2d']:
                paired=[]
                per_seed=[]
                for seed in rt.protocol['seeds']:
                    a=qselect(scope,f'{added}__{seed}',direction).set_index('query_id').ap
                    b=qselect(scope,f'{control}__{seed}',direction).set_index('query_id').ap
                    delta=a-b
                    assert delta.notna().all()
                    paired.append(delta)
                    per_seed.append(float(delta.mean()))
                mean=pd.concat(paired,axis=1).mean(axis=1)
                first=mean.rename('ap').reset_index()
                zero=first.copy();zero['ap']=0.
                interval=paired_interval(first,zero)
                contrasts.append({'contrast':contrast,'added':added,'control':control,'scope':scope,'direction':direction,
                    **interval,'positive_seeds':int((np.asarray(per_seed)>1e-8).sum()),'seed_deltas':per_seed,
                    'interpretation':'mean per-query AP across 3 individually validation-selected seeds; exploratory contrasts, unadjusted intervals'})
    write_json(ev/'PAIRED_CONTRASTS.json',contrasts)
    pd.DataFrame(contrasts).to_csv(ev/'PAIRED_CONTRASTS.csv',index=False)
    primary=[]
    for scope in scopes:
        for direction in ['d2t','t2d']:
            primary.append({'scope':scope,'direction':direction,'winner':winner,
                **paired_interval(qselect(scope,winner,direction),qselect(scope,'frozen_v3',direction),.975),
                'interpretation':'ensemble logits ranked; 97.5% marginal interval for the two direction AP comparisons within scope'})
    write_json(ev/'WINNER_VS_FROZEN_V3.json',primary)

    # Export the chosen upgrade, old V3 and the strongest simple comparator.
    top=[]
    for name in ['frozen_v3',winner,'positive_nearest']:
        values=np.load(score_dir/f'{name}.npz')['old']
        ds=rt.old_drugs.ligand_inchikey.to_numpy();ts=rt.old_targets.target_chembl_id.to_numpy()
        for direction,scores,q,c in [('drug_to_target',values[:,:,0],ds,ts),('target_to_drug',values[:,:,1].T,ts,ds)]:
            for query,row in zip(q,scores):
                for rank,j in enumerate(np.lexsort((c,-row))[:20],1):
                    top.append({'model':name,'direction':direction,'query_id':query,'candidate_id':c[j],
                                'rank':rank,'candidate_count':len(c),'score':float(row[j])})
    named=annotate(pd.DataFrame(top),rt.old_drugs,rt.old_targets,labels)
    for direction in ['drug_to_target','target_to_drug']:
        named[named.direction.eq(direction)].to_csv(ev/f'TOP20_{direction.upper()}.csv.gz',index=False)
    write_json(OUT/'STATUS.json',{'status':'COMPLETE','training_runs':len(all_runs),
        'validation_selected_winner':winner,'test_used_for_selection':False,'production_promotion':False,
        'evaluation_code_sha256':sha256(Path(__file__))})
    print(json.dumps({'status':'COMPLETE','winner':winner}),flush=True)


if __name__=='__main__':
    main()
