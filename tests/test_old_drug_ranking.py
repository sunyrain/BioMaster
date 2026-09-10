import numpy as np
import pytest
from biomaster.old_drug_ranking import known_ranking, observed_ranking, directional_borda


def test_directions_have_different_candidates_and_skip_only_empty_queries():
    y=np.array([[1,0,0],[0,0,0]],bool)
    s=np.array([[2,1,0],[3,0,-1]])
    forward,_,_=known_ranking(y,s,['drugA','drugB'],['t1','t2','t3'])
    reverse,_,_=known_ranking(y.T,s.T,['t1','t2','t3'],['drugA','drugB'])
    assert forward['evaluated_queries']==1 and forward['candidates_per_query']==3
    assert reverse['evaluated_queries']==1 and reverse['candidates_per_query']==2
    assert forward['macro_ap']==1 and reverse['macro_ap']==.5


def test_unknowns_never_enter_observed_pn_metrics():
    y=np.array([[1,0,np.nan],[1,np.nan,np.nan]])
    s=np.array([[2,1,100],[0,20,50]])
    out,_=observed_ranking(y,s,['a','b'],['x','y','z'])
    assert out['two_class_queries']==1 and out['observed_pairs']==3
    assert out['macro_ap']==out['macro_auroc']==1


def test_known_rank_ties_are_independent_of_column_order():
    y=np.array([[1,0,1]])
    s=np.zeros((1,3))
    first,_,_=known_ranking(y,s,['q'],['c','a','b'])
    ix=np.array([2,0,1])
    second,_,_=known_ranking(y[:,ix],s[:,ix],['q'],np.array(['c','a','b'])[ix])
    assert first==second


def test_reverse_borda_uses_raw_scores_within_target():
    s=np.array([[10,9],[2,1]])
    forward=directional_borda(s,s,axis=1)
    reverse=directional_borda(s,s,axis=0)
    assert np.array_equal(forward[0],forward[1])
    assert np.all(reverse[0]>reverse[1])


def test_missing_scores_cannot_silently_shrink_candidate_scope():
    with pytest.raises(ValueError):
        known_ranking(np.array([[1,0]]),np.array([[1,np.nan]]),['q'],['a','b'])
