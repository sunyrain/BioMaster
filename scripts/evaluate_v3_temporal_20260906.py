#!/usr/bin/env python3
"""Release temporal tests only after all training/selection/predictions are frozen."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.ranking_audit import measured_ranking,risk_set_ranking,query_bootstrap
from build_biomaster_odti_v4_features import sha256
from train_biomaster_selectivity_v3 import write_json,now

OUT=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'


def random_expectation(candidates, positives):
    """Exact expected AP under a uniformly random strict ordering (not tied scores)."""
    n,m=int(candidates),int(positives)
    if not 1<=m<=n:raise ValueError('at least one positive and valid candidate count required')
    harmonic=float((1/np.arange(1,n+1)).sum())
    return 1. if n==1 else (harmonic+(m-1)/(n-1)*(n-harmonic))/n


def main():
    release=json.loads((OUT/'TEST_RELEASE.json').read_text())
    assert release['train_max_year']==2022 and release['fresh_base_and_adapter_refit'] and not release['test_labels_read']
    assert release['selection_sha256']==sha256(OUT/'SELECTION.json')
    assert release['scores_manifest_sha256']==sha256(OUT/'final/scores/MANIFEST.json')
    for path,digest in release['checkpoints'].items():assert sha256(ROOT/path)==digest
    manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    scoremanifest=json.loads((OUT/'final/scores/MANIFEST.json').read_text())
    for file,digest in scoremanifest['files'].items():assert sha256(OUT/'final/scores'/file)==digest
    pool=pd.read_csv(OUT/'QUERY_POOL.csv.gz')
    index=pd.MultiIndex.from_frame(pool[['drug_feature_index','target_feature_index']])
    old=pd.read_csv(OUT/'OLD_DRUG_INDEX.csv');targets=pd.read_csv(OUT/'TARGET_INDEX.csv.gz')
    approval_protocol=json.loads((OUT/'REGISTRY_APPROVAL_PROTOCOL.json').read_text())
    for path,digest in approval_protocol['files'].items():assert sha256(OUT/path)==digest
    approval=pd.read_csv(OUT/'REGISTRY_APPROVAL_AUDIT.csv')
    assert np.array_equal(old.old_drug_index,approval.old_drug_index)
    approved=approval.confirmed_approved_by_2022.to_numpy(bool)
    approved_ids=set(old.loc[approved,'drug_feature_index'])
    scores={k:v for k,v in np.load(OUT/'final/scores/POOL_SCORES.npz').items()}
    old_scores={k:v.reshape(720,384,2) for k,v in np.load(OUT/'final/scores/OLD_SCORES.npz').items()}
    risk=np.load(OUT/'RISK_SETS.npz')['final']
    first=pd.read_csv(OUT/'FIRST_OBSERVATIONS.csv.gz')
    historical=first[first.first_document_year.le(2022)|first.has_undated]
    table_names=['test_2023_2025_new','test_2023_new','test_2024_new','test_2025_new','test_2023_2025_all']
    summaries=[];query_frames=[];pair_frames=[];comparisons=[];frames={}

    def collect(scope,model,direction,result,queries):
        summaries.append({'scope':scope,'model':model,'direction':direction,**result})
        if len(queries):
            queries=queries.copy();queries['scope']=scope;queries['model']=model;queries['direction']=direction
            query_frames.append(queries);frames[(scope,model,direction)]=queries

    mapping=old.set_index('drug_feature_index').old_drug_index
    drug_ids=old.ligand_inchikey.to_numpy();target_ids=targets.target_chembl_id.to_numpy()
    prediction_frames=[];label_matrices={}
    for table_name in table_names:
        path=OUT/f'{table_name.upper()}.csv.gz'
        assert sha256(path)==manifest['files'][str(path.relative_to(ROOT))]
        table=pd.read_csv(path)
        assert table.min_document_year.min()>=2023 and table.max_document_year.max()<=2025
        if table_name.endswith('_new'):
            assert table.merge(historical,on=['drug_feature_index','target_feature_index']).empty
        rows=index.get_indexer(pd.MultiIndex.from_frame(table[['drug_feature_index','target_feature_index']]))
        assert (rows>=0).all()
        aligned=table.rename(columns={'drug_feature_index':'drug_id','target_feature_index':'target_id','binary_label':'label'})
        aligned['scope']=table_name
        for model,values in scores.items():
            for i,direction in enumerate(['d2t','t2d']):
                aligned[f'{model}_{direction}']=values[rows,i]
                for population in ['all_compounds','old720','approved_pre2023']:
                    if population=='all_compounds':mask=np.ones(len(table),bool)
                    elif population=='old720':mask=table.is_project_old_drug.to_numpy(bool)
                    else:mask=table.drug_feature_index.isin(approved_ids).to_numpy()
                    summary,queries=measured_ranking(aligned.loc[mask],values[rows[mask],i],direction)
                    scope=f'{table_name}_measured_{population}'
                    collect(scope,model,direction,summary,queries)
        prediction_frames.append(aligned)
        if table_name.endswith('_all'):continue
        labels=np.zeros((720,384),bool)
        positive=table[table.is_project_old_drug&table.binary_label.eq(1)]
        labels[positive.drug_feature_index.map(mapping).to_numpy(int),positive.target_feature_index.to_numpy(int)]=True
        assert not (labels&~risk).any();label_matrices[table_name]=labels
        for model,values in old_scores.items():
            for i,direction in enumerate(['d2t','t2d']):
                for population,mask in [('old720',np.ones(720,bool)),('approved_pre2023',approved)]:
                    y,s,r,q,c=labels[mask],values[mask,:,i],risk[mask],drug_ids[mask],target_ids
                    if i:y,s,r,q,c=y.T,s.T,r.T,c,q
                    summary,queries,pairs=risk_set_ranking(y,s,r,q,c)
                    scope=table_name+'_dense_'+population
                    collect(scope,model,direction,summary,queries)
                    if len(pairs):
                        pairs['scope']=scope;pairs['model']=model;pairs['direction']=direction;pair_frames.append(pairs)
    # Sparse observed AP can be high even under random ordering. Give the exact
    # expected random AP/Recall for the same query candidate and positive counts.
    for row in list(summaries):
        if row['model']!='temporal_v3':continue
        queries=frames.get((row['scope'],'temporal_v3',row['direction']))
        if queries is None:continue
        random_queries=queries[['query_id','candidates','positives']].copy()
        random_queries['ap']=[random_expectation(n,m) for n,m in zip(queries.candidates,queries.positives)]
        for k in [5,10,20]:random_queries[f'recall_{k}']=np.minimum(k,queries.candidates)/queries.candidates
        result={k:v for k,v in row.items() if k not in ['scope','model','direction'] and not k.startswith('macro_')}
        result.update({f'macro_{key}':float(random_queries[key].mean()) for key in ['ap','recall_5','recall_10','recall_20']})
        collect(row['scope'],'random_ranking_expected',row['direction'],result,random_queries)
    for scope in sorted({s['scope'] for s in summaries}):
        for direction in ['d2t','t2d']:
            candidate=frames.get((scope,'temporal_J',direction))
            if candidate is None:continue
            for control in ['temporal_v3','positive_nearest','pn_logistic']:
                comparison=query_bootstrap(candidate,frames[(scope,control,direction)])
                comparisons.append({'scope':scope,'direction':direction,'candidate':'temporal_J','control':control,**comparison})
    pd.DataFrame(summaries).to_csv(OUT/'TEST_METRICS.csv',index=False)
    pd.concat(query_frames,ignore_index=True).to_csv(OUT/'TEST_QUERY_METRICS.csv.gz',index=False)
    pd.concat(pair_frames,ignore_index=True).to_csv(OUT/'TEST_POSITIVE_RANKS.csv.gz',index=False)
    pd.concat(prediction_frames,ignore_index=True).to_csv(OUT/'TEST_MEASURED_PREDICTIONS.csv.gz',index=False)
    np.savez_compressed(OUT/'TEST_DENSE_LABELS.npz',**label_matrices,risk=risk)
    write_json(OUT/'TEST_PAIRED_COMPARISONS.json',comparisons)
    result={'status':'COMPLETE','utc':now(),'test_release_sha256':sha256(OUT/'TEST_RELEASE.json'),
            'evaluation_source_sha256':sha256(Path(__file__)),'test_used_for_selection':False,
            'labels':'publication-year-qualified experimental binary activities; positive-only dense retrieval',
            'counts':manifest['counts'],'metrics_sha256':sha256(OUT/'TEST_METRICS.csv'),
            'approval_sensitivity':approval_protocol,
            'limitation':'current project 720/384 registry and current public pretrained feature banks; not all entities proven approved before 2023',
            'not_a_prospective_experiment':True}
    write_json(OUT/'TEST_RESULT.json',result)
    report=pd.DataFrame(summaries)
    selected=report.scope.isin(['test_2023_2025_new_dense_old720','test_2023_2025_new_measured_all_compounds','test_2023_2025_new_measured_old720'])
    print(report.loc[selected,['scope','model','direction','macro_ap','macro_recall_20']].to_string(index=False),flush=True)


if __name__=='__main__':main()
