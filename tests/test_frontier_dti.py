import json
import math
import numpy as np
import pandas as pd
import pytest
from biomaster.frontier_dti import common_ranks,metrics,finite_json,load_snapshot,DIRECTORY

def test_common_pool_excludes_missing_and_does_not_change_scores():
    frame=pd.DataFrame({'a':[0.9,0.8,0.8,0.1],'b':[None,0.1,0.3,0.3]})
    ranks,percent,n=common_ranks(frame,('a','b'))
    assert n==3
    assert ranks.loc[0].isna().all()
    assert ranks.loc[1,'a']==1.5  # exact ties retain equal ranks
    assert ranks.loc[2,'a']==1.5
    assert ranks.loc[1,'b']==3
    assert percent.loc[1,'b']==1
    assert frame.loc[0,'a']==0.9

def test_no_false_metrics_without_two_label_classes():
    frame=pd.DataFrame({'label':[1,1,0],'nesso':[0.4,0.7,np.nan]})
    row=metrics(frame,'nesso','test')
    assert row['pairs']==2 and row['positive']==2 and row['negative']==0
    assert row['ap'] is None and row['auroc'] is None
    assert row['prevalence']==1
    frame.loc[2,'nesso']=0.1
    row=metrics(frame,'nesso','test')
    assert row['ap']==1 and row['auroc']==1
    assert row['prevalence']==pytest.approx(2/3)

def test_filter_keeps_original_common_scope_and_live_refresh(tmp_path):
    path=tmp_path/DIRECTORY/'WEBSITE_SNAPSHOT.json';path.parent.mkdir(parents=True)
    payload={'common_spr_pairs':300,'items':[{'drug_id':'d1','target_id':'t1','common_rank':9},{'drug_id':'d2','target_id':'t1','common_rank':1}]}
    path.write_text(json.dumps(payload))
    filtered=load_snapshot(tmp_path,'drug','d1')
    assert filtered['common_spr_pairs']==300 and filtered['items'][0]['common_rank']==9
    assert filtered['visible_count']==1
    payload['common_spr_pairs']=301;path.write_text(json.dumps(payload))
    assert load_snapshot(tmp_path)['common_spr_pairs']==301
    with pytest.raises(ValueError):load_snapshot(tmp_path,'../../private')

def test_export_has_json_null_for_nonfinite_not_negative():
    value=finite_json({'score':np.float32(np.nan),'rank':float('inf'),'valid':np.int64(2)})
    assert value=={'score':None,'rank':None,'valid':2}
    json.dumps(value,allow_nan=False)
