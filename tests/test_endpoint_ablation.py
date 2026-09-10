import numpy as np
import pandas as pd
from biomaster.endpoint_ablation import ARMS,CyclingRows,combine


def labels(rows):
    f=pd.DataFrame(rows,columns=['pair_id','task','has_positive','has_negative','has_conflict'])
    f['drug_feature_index']=f.pair_id.map({'a':0,'b':1,'c':2,'d':3})
    f['target_feature_index']=0;f['molecule_id']=f.pair_id;f['target_id']='T'
    f['split']='train';f['split_group']=f.pair_id;f['seen_by_frozen_parent_connectivity_pair']=False
    return f


def test_explicit_inactivity_is_real_shared_negative_in_both_arms():
    f=labels([('a','AFFINITY_KD_KI',True,False,False),('b','ANNOTATED_INACTIVITY',False,True,False),
              ('c','ACTIVITY_IC50',False,True,False)])
    for tasks in ARMS.values():
        r=combine(f,{'b','c'},tasks).set_index('pair_id')
        assert r.loc[['b','c'],'eligible'].all()
        assert r.loc[['b','c'],'binary_label'].eq(0).all()
        assert r.loc[['b','c'],'explicit_inactive'].all()


def test_unselected_numeric_negatives_do_not_enter_kdki_arm():
    f=labels([('a','AFFINITY_KD_KI',True,False,False),('b','ACTIVITY_IC50',False,True,False)])
    r=combine(f,set(),ARMS['kdki_inactive'])
    assert r.pair_id.tolist()==['a']


def test_inactivity_positive_contradiction_is_quarantined_in_both_arms():
    f=labels([('a','ACTIVITY_IC50',True,False,False),('a','ANNOTATED_INACTIVITY',False,True,False)])
    for tasks in ARMS.values():
        r=combine(f,{'a'},tasks)
        assert r.conflict.all() and not r.eligible.any()


def test_cross_endpoint_disagreement_is_not_voted_away():
    f=labels([('a','AFFINITY_KD_KI',True,False,False)]+[('a','ACTIVITY_IC50',False,True,False)]*10)
    a=combine(f,set(),ARMS['kdki_inactive']);b=combine(f,set(),ARMS['all_inactive'])
    assert len(a)==len(b)==1 and a.eligible.all() and not b.eligible.any()


def test_within_task_conflict_is_not_recovered_by_other_positive():
    f=labels([('a','AFFINITY_KD_KI',True,True,True),('a','ACTIVITY_IC50',True,False,False)])
    assert not combine(f,set(),ARMS['all_inactive']).eligible.any()


def test_shuffle_cycles_cover_every_row_before_repeating():
    stream=CyclingRows(7,384)
    drawn=np.concatenate([stream.take(5),stream.take(5),stream.take(4)])
    assert set(drawn[:7])==set(range(7)) and set(drawn[7:])==set(range(7))
    assert stream.passes==2 and stream.offset==0


def test_negative_only_panel_reports_false_positives_without_auc_or_ap():
    from scripts.run_endpoint_ablation_20260911 import basic
    r=basic([0,0,0,0],[1,-1,2,-2],[.8,.2,.9,.1],.5)
    assert r['auroc'] is None and r['ap'] is None and r['recall'] is None
    assert r['false_positive_rate']==.5 and r['top1pct_precision']==0
