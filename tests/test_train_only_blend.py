import numpy as np
import pandas as pd
import pytest

from biomaster.train_only_blend import TrainOnlyEvidence, validation_rank_selection


def training():
    return pd.DataFrame(dict(pair_id=['a','a_stereo','b','c','n'],
        molecule_id=['AAAA-X','AAAA-Y','BBBB-X','CCCC-X','NNNN-X'],
        target_id=['T1','T1','T1','T2','T1'],drug_feature_index=[0,1,2,3,4],
        binary_label=[1,1,1,1,0],split=['train']*5))


def test_neighbors_use_only_positive_same_target_and_exclude_self_connectivity():
    fp=np.array([[1,1,0,0],[1,1,0,0],[1,0,1,0],[1,1,0,0],[1,1,0,0]],np.uint8)
    evidence=TrainOnlyEvidence(training(),fp,device='cpu')
    # A perfect negative or a perfect ligand for another target must not be retrieved.
    out=evidence.transform(['AAAA-Z','UNKNOWN-X','UNKNOWN-X'],['T1','T3','T2'],
                           np.array([[1,1,0,0]]*3,np.uint8))
    assert out.positive_neighbor.to_list()==pytest.approx([1/3,0,1])
    assert out.nearest_training_positive.to_list()==['BBBB-X','','CCCC-X']
    assert out.same_connectivity_references_excluded.iloc[0]==2
    assert out.eligible_positive_neighbors.iloc[0]==1


def test_prior_is_smoothed_training_frequency_and_unseen_target_uses_training_mean():
    train=training();fp=np.eye(5,dtype=np.uint8)
    evidence=TrainOnlyEvidence(train,fp,device='cpu',prior_strength=10)
    out=evidence.transform(['Q','Q'],['T1','UNSEEN'],np.zeros((2,5),np.uint8))
    assert out.target_prior.to_list()==pytest.approx([(3+10*.8)/(4+10),.8])


def test_validation_cannot_enter_reference_pool_or_test_enter_weight_selection():
    train=training();train.loc[0,'split']='validation'
    with pytest.raises(ValueError,match='Only training'):
        TrainOnlyEvidence(train,np.eye(5,dtype=np.uint8),device='cpu')
    with pytest.raises(ValueError,match='exclusively on validation'):
        validation_rank_selection(pd.DataFrame({'split':['test']}),np.zeros((1,3)),[(0,0)])


@pytest.mark.skipif(not __import__('torch').cuda.is_available(),reason='CUDA unavailable')
def test_gpu_neighbor_result_matches_binary_set_arithmetic():
    import torch
    rng=np.random.default_rng(14)
    fp=rng.integers(0,2,size=(5,2048),dtype=np.uint8)
    queries=rng.integers(0,2,size=(7,2048),dtype=np.uint8)
    out=TrainOnlyEvidence(training(),fp,device='cuda').transform(['Q']*7,['T1']*7,queries)
    exact=[]
    for q in queries:
        exact.append(max(np.count_nonzero(q&r)/np.count_nonzero(q|r) for r in fp[:3]))
    assert out.positive_neighbor.to_numpy()==pytest.approx(exact,abs=1e-7)
