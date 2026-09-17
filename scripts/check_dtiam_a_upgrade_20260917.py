#!/usr/bin/env python3
"""Verify release identity, exact score replacement and unchanged other channels."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_data import ExplorerData,MODELS
from biomaster.catalog_models import connect
from biomaster.dtiam_release import load_scores,RELEASE_ID,DIRECTORY
OUT=ROOT/DIRECTORY

def main():
    data=ExplorerData(ROOT,infer=False);data.ensure_loaded()
    release=load_scores(ROOT).set_index(['ligand_inchikey','target_chembl_id']).dtiam_probability
    live=data.pairs.set_index(['ligand_inchikey','target_chembl_id'])
    assert len(live)==len(release)==276480 and np.array_equal(live.dtiam,release.loc[live.index])
    assert data.pairs.groupby('ligand_inchikey').drug_dtiam_denominator.first().eq(384).all()
    assert data.pairs.groupby('target_chembl_id').target_dtiam_denominator.first().eq(720).all()
    previous=pd.read_csv(ROOT/'outputs/model_agreement_20260917/SCORE_SNAPSHOT.csv.gz').set_index(['drug_id','target_id'])
    stable={}
    for model in ['biomaster','biomaster_reverse','drugclip','conplex']:
        a=live.loc[previous.index,model].to_numpy();b=previous[model].to_numpy()
        # The old CSV serialized these two FP32 arrays with shortest FP32 decimals;
        # parsing as FP64 changes the numeric expansion, not the underlying scores.
        if model in ('biomaster','biomaster_reverse'):
            assert a.dtype==np.float32
            b=b.astype(np.float32)
            assert np.array_equal(a,b,equal_nan=True),model
        else:
            assert np.allclose(a,b,rtol=0,atol=1e-12,equal_nan=True),model
        stable[model]=dict(pairs=len(b),comparison_dtype=str(a.dtype),maximum_absolute_difference=float(np.nanmax(np.abs(a-b))))
    db=connect(ROOT)
    scores=pd.read_sql_query("SELECT model,drug_id,target_id,score FROM predictions WHERE status='completed'",db);db.close()
    for model in ['nesso','probematch','dtbind']:
        old=previous[model].dropna();new=scores[scores.model.eq(model)].set_index(['drug_id','target_id']).score.loc[old.index]
        assert np.allclose(old,new,rtol=0,atol=1e-12),model
        stable[model]=dict(preexisting_pairs=len(old),maximum_absolute_difference=float(np.max(np.abs(old-new))))
    frontier=ROOT/'outputs/frontier_dti_20260916'
    before=pd.read_csv(OUT/'previous_review_snapshot/ALL_PREDICTIONS.csv',low_memory=False).set_index('pair_id')
    after=pd.read_csv(frontier/'ALL_PREDICTIONS.csv',low_memory=False).set_index('pair_id').loc[before.index]
    assert after.dtiam_model_version.eq(RELEASE_ID).all()
    for model in MODELS:
        if model!='dtiam':assert np.allclose(before[model],after[model],rtol=0,atol=1e-12,equal_nan=True),model
    expected=pd.Series(release.to_numpy(),index=[d+'__'+t for d,t in release.index])
    assert np.array_equal(after.dtiam,expected.loc[after.index])
    frozen=ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
    protocol=json.loads((frontier/'PROTOCOL.json').read_text())
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==protocol['original_spr_sha256']
    snapshot=json.loads((frontier/'WEBSITE_SNAPSHOT.json').read_text())
    assert snapshot['dtiam_model_version']==RELEASE_ID
    assert snapshot['model_names']['dtiam']=='DTIAM A（九月加强版）'
    for item in snapshot['items']:
        assert item['scores']['dtiam']==expected.loc[item['pair_id']]
    review=pd.read_csv(frontier/'SPR384_MODEL_REVIEW.csv')
    assert len(review)==384 and review.dtiam_model_version.eq(RELEASE_ID).all()
    # Report the upgrade on the old, frozen common labeled set, without reselecting.
    b=before[before.cohort.eq('BINDINGDB479')].dropna(subset=list(MODELS))
    comparisons=[]
    for name,series in [('historical',b.dtiam),('september_A',after.loc[b.index,'dtiam'])]:
        comparisons.append(dict(version=name,pairs=len(b),positive=int(b.label.sum()),negative=int((b.label==0).sum()),
            ap=average_precision_score(b.label,series),auroc=roc_auc_score(b.label,series)))
    pd.DataFrame(comparisons).to_csv(OUT/'SAME_378_LABEL_COMPARISON.csv',index=False)
    targets=['CHEMBL1871','CHEMBL203'];queries=[]
    for target in targets:
        if target not in data.targets or not data.targets[target]['scored']:continue
        result=data.rankings('target',target,'dtiam',page_size=10)
        rows=live.xs(target,level=1).sort_values('dtiam',ascending=False,kind='stable').head(10)
        assert [r['id'] for r in result['items']]==list(rows.index)
        assert [r['rank'] for r in result['items']]==list(range(1,11))
        assert result['source']['version']==RELEASE_ID
        queries.append(dict(target_id=target,top10=[dict(drug=r['name'],score=r['score']) for r in result['items']]))
    checks=dict(status='PASS',updated_utc=datetime.now(timezone.utc).isoformat(),release_id=RELEASE_ID,
        scored_pairs=276480,drug_queries=720,target_queries=384,review_pairs=384,review_and_benchmark_pairs=863,
        unchanged_other_scores=stable,frozen_spr_unchanged=True,common_label_comparison=comparisons,query_checks=queries)
    (OUT/'UPGRADE_CHECK.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False))

if __name__=='__main__':main()
