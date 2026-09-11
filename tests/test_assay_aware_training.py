import numpy as np
import pandas as pd
import pytest
import torch

from biomaster.assay_aware_training import (balanced_target_weights, RotatingTargetClassRows,
    row_stream_state, restore_row_stream, AssayPairs, assay_difference_huber)


def test_target_balance_preserves_class_mass_and_bounds_rare_target_amplification():
    f=pd.DataFrame({'target_id':['popular']*1200+['rare']*4,
                    'binary_label':[1]*1000+[0]*200+[1,1,0,0]})
    w=balanced_target_weights(f)
    for y in [0,1]:
        mask=f.binary_label.eq(y).to_numpy()
        assert w[mask].sum()==pytest.approx(len(f)/2,rel=1e-6)
        assert np.max(w[mask]/(.5/mask.mean()))<=5.000001
    assert w[-1]>w[1000] and w[-4]>w[0]


def test_rotating_cap_reaches_all_members_and_restarts_reproducibly():
    f=pd.DataFrame({'target_id':['large']*370+['small']*4,'binary_label':[1]*300+[0]*70+[1,1,0,0]})
    sampler=RotatingTargetClassRows(f,seed=5,cap=40)
    first=sampler.take(sampler.round_size)
    assert len(first)==84 and len(np.unique(first))==len(first)
    assert (f.iloc[first].groupby(['target_id','binary_label']).size()<=40).all()
    state=row_stream_state(sampler)
    future=sampler.take(sampler.round_size*8+7)
    assert set(np.concatenate([first,future]))==set(range(len(f)))
    other=RotatingTargetClassRows(f,seed=99,cap=40);restore_row_stream(other,state)
    np.testing.assert_array_equal(future,other.take(len(future)))
    assert sampler.positive_fraction==pytest.approx(.5)
    repeated=RotatingTargetClassRows(f,seed=8,cap=40)
    for _ in range(12):
        ids=repeated.take(repeated.round_size)
        assert len(ids)==len(np.unique(ids))


def test_assay_sampler_never_pairs_across_source_assay_or_endpoint():
    f=pd.DataFrame(dict(assay_group=['a']*5+['b']*5,endpoint=['Kd']*5+['Ki']*5,
        pair_id=[str(x) for x in range(10)],drug_feature_index=np.arange(10),
        target_feature_index=[2]*5+[8]*5,p_activity=[4,5,6,7,8]*2))
    pool=AssayPairs(f);d,t,y=pool.sample(np.random.default_rng(17),100)
    assert (t[:,0]==t[:,1]).all() and (d[:,0]!=d[:,1]).all()
    assert ((d[:,0]<5)==(d[:,1]<5)).all()
    assert (np.abs(y[:,0]-y[:,1])>=.3).all()
    f.loc[0,'target_feature_index']=9
    with pytest.raises(ValueError,match='Mixed target'):AssayPairs(f)


def test_assay_difference_is_offset_invariant_and_gradient_improves_order():
    p=torch.tensor([[0.,0.]],requires_grad=True);truth=torch.tensor([[8.,6.]])
    loss=assay_difference_huber(p,truth,2.)
    assert loss.item()==pytest.approx(1.5)
    assert assay_difference_huber(p+100,truth+20,2.).item()==pytest.approx(loss.item())
    loss.backward();assert p.grad[0,0]<0 and p.grad[0,1]>0
    assert assay_difference_huber(torch.tensor([[1.,0.]]),truth,2.).item()==0


def test_transfer_preserves_physical_regression_values_with_new_scaling(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA transfer integration')
    from scripts.run_assay_aware_20260911 import create_model
    from biomaster.endpoint_multitask import AuxiliaryInteraction
    from types import SimpleNamespace
    import json
    from pathlib import Path
    config=json.loads(Path('outputs/biomaster_endpoint_multitask_20260911/MODEL_CONFIG.json').read_text())
    source=AuxiliaryInteraction(config,['Kd','Ki','IC50','EC50'])
    state=dict(endpoints=['Kd','Ki','IC50','EC50'],model=source.state_dict(),
               regression_means=[5.,6.,4.,3.],regression_scales=[2.,3.,4.,5.])
    path=tmp_path/'parent.pt';torch.save(state,path)
    target=SimpleNamespace(endpoints=['Kd','Ki'],means=np.array([7.,8.]),scales=np.array([.5,1.5]))
    model,_=create_model(target,4,config,path)
    h=torch.randn(10,source.regression.in_features)
    expected=source.regression(h).detach()[:,:2]*torch.tensor([2.,3.])+torch.tensor([5.,6.])
    actual=model.regression(h.cuda()).detach().cpu()*torch.tensor([.5,1.5])+torch.tensor([7.,8.])
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)


def test_strict_assay_holdout_rejects_shared_id_across_targets_and_partial_metadata():
    from scripts.report_assay_aware_20260911 import strict_assay_flags
    observations=pd.DataFrame(dict(pair_id=['train','shared','unseen','partial','partial'],
        resolved_assay_id=['assay1','assay1','assay2','assay3',''],target=['t1','t2','t3','t4','t4']))
    flags=strict_assay_flags(observations,['train'],pd.Series(['shared','unseen','partial','missing']))
    assert flags.tolist()==[False,True,False,False]
