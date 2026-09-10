import numpy as np
import pandas as pd
import pytest

from biomaster.ranking_audit import measured_ranking, risk_set_ranking, query_bootstrap


def test_measured_axes_use_only_explicit_pairs_and_skip_single_class_queries():
    frame=pd.DataFrame({'drug_id':['a','a','b','b','c'],'target_id':['x','y','x','y','z'],'label':[1,0,0,1,1]})
    scores=np.array([.9,.8,.1,.2,.5])
    d,q=measured_ranking(frame,scores,'d2t')
    t,_=measured_ranking(frame,scores,'t2d')
    assert d['two_class_queries']==2 and d['all_queries']==3 and d['macro_ap']==1
    assert t['macro_ap']==.75 and q.candidates.tolist()==[2,2]
    with pytest.raises(ValueError):measured_ranking(pd.concat([frame,frame.iloc[:1]]),np.r_[scores,.9],'d2t')


def test_risk_mask_changes_candidates_without_removing_new_positive():
    labels=np.array([[0,1,0],[0,0,1]])
    scores=np.array([[9.,2.,1.],[3.,2.,1.]])
    risk=np.array([[0,1,1],[1,0,1]],bool)
    result,queries,pairs=risk_set_ranking(labels,scores,risk,['a','b'],['x','y','z'])
    assert result['macro_ap']==.75
    assert queries.candidates.tolist()==[2,2] and pairs['rank'].tolist()==[1,2]
    risk[0,1]=False
    with pytest.raises(ValueError,match='positives'):risk_set_ranking(labels,scores,risk,['a','b'],['x','y','z'])


def test_risk_ties_are_deterministic_by_candidate_id_and_empty_queries_skip():
    result,q,_=risk_set_ranking([[1,0],[0,0]],np.ones((2,2)),np.ones((2,2)),['a','b'],['z','x'])
    assert result['positive_queries']==1 and q.ap.iloc[0]==.5
    with pytest.raises(ValueError):risk_set_ranking([[1,0]],[[1,np.nan]],[[1,1]],['a'],['x','y'])


def test_paired_bootstrap_aligns_queries_and_requires_identical_query_sets():
    first=pd.DataFrame({'query_id':['a','b'],'ap':[.5,.8]})
    second=pd.DataFrame({'query_id':['b','a'],'ap':[.7,.4]})
    result=query_bootstrap(first,second,iterations=100)
    assert np.isclose(result['delta_ap'],.1) and np.isclose(result['ci_low'],.1)
    with pytest.raises(ValueError):query_bootstrap(first,second.iloc[:1],iterations=100)
