import copy
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from biomaster.best_model_training import ObservedQueries,measured_pair_loss,WeightAverage
from biomaster.refined_interaction import RefinedInteraction,RefinedConfig,matched_refinement_config
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig

# Reuse the existing physically unaligned synthetic input fixture.
spec=importlib.util.spec_from_file_location('unified_fixture',Path(__file__).with_name('test_unified_interaction.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_queries_only_contain_observations_and_preserve_query_identity():
    frame=pd.DataFrame(dict(drug_feature_index=[0,0,1,1,2,3],target_feature_index=[0,1,0,1,2,2],binary_label=[1,0,0,1,1,1]))
    seen=set(zip(frame.drug_feature_index,frame.target_feature_index,frame.binary_label))
    for head,key in enumerate(['drug_feature_index','target_feature_index']):
        stream=ObservedQueries(frame,key,head)
        for seed in range(10):
            for d,t,y,h in stream.sample(np.random.default_rng(seed),queries=99):
                assert h==head and y.any() and not y.all()
                assert len(np.unique(d if head==0 else t))==1
                assert set(zip(d,t,y))<=seen
    with pytest.raises(ValueError):ObservedQueries(frame.assign(binary_label=np.nan),'drug_feature_index',0)


def test_measured_ranking_loss_order_and_query_shift_invariance():
    labels=torch.tensor([True,False,True,False])
    good=torch.tensor([3.,-2.,2.,-1.],requires_grad=True)
    loss=measured_pair_loss(good,labels)
    assert loss < measured_pair_loss(-good,labels)
    torch.testing.assert_close(loss,measured_pair_loss(good+100,labels))
    loss.backward()
    assert (good.grad[labels]<0).all() and (good.grad[~labels]>0).all()
    with pytest.raises(ValueError):measured_pair_loss(good,torch.ones(4,dtype=torch.bool))


def test_ema_is_independent_serializable_and_does_not_change_online_model():
    model=torch.nn.Linear(3,2);ema=WeightAverage(model)
    before=copy.deepcopy(model.state_dict())
    with torch.no_grad():model.weight.add_(2)
    changed=copy.deepcopy(model.state_dict());ema.update(model)
    assert torch.equal(model.weight,changed['weight'])
    assert (ema.model.weight>before['weight']).all() and (ema.model.weight<model.weight).all()
    restored=torch.nn.Linear(3,2);restored.load_state_dict(ema.model.state_dict())
    x=torch.randn(4,3)
    torch.testing.assert_close(restored(x),ema.model(x),rtol=0,atol=0)
    assert not any(p.requires_grad for p in ema.model.parameters())


def test_refinements_start_exactly_at_global_and_all_parameters_train():
    torch.set_num_threads(2)
    batch=fixture.example()
    # Exercise an actual stereo category; category zero is the padding entry.
    batch['stereo'][batch['neighbors']>=0]=1
    base=UnifiedInteraction(UnifiedConfig(variant='global',width=24,dropout=0)).eval()
    for variant in ['global','capacity','site','geometry']:
        cfg=RefinedConfig(variant=variant,width=24,pair_width=8,dropout=0,capacity_hidden=67)
        model=RefinedInteraction(cfg).eval();model.warm_start(base.state_dict())
        torch.testing.assert_close(base(batch),model(batch),rtol=0,atol=0)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        # The first update opens the zero-initialized residual projection.
        model(batch).square().mean().backward();optimizer.step();optimizer.zero_grad()
        model(batch).square().mean().backward()
        unused=[name for name,p in model.named_parameters() if p.grad is None or p.grad.abs().sum()==0]
        assert not unused,unused


def test_refinement_missing_structure_and_missing_atoms_are_exact_fallbacks():
    batch=fixture.example();cfg=RefinedConfig(variant='geometry',width=24,pair_width=8,dropout=0)
    model=RefinedInteraction(cfg).eval()
    with torch.no_grad():model.refine[-1].weight.normal_(0,.05)
    site=RefinedInteraction(RefinedConfig(variant='site',width=24,pair_width=8,dropout=0)).eval()
    site.load_state_dict({k:v for k,v in model.state_dict().items() if '.geometry.' not in k},strict=True)
    batch['geometry_mask'].zero_()
    torch.testing.assert_close(model(batch),site(batch),rtol=0,atol=0)
    no_atoms=copy.deepcopy(batch);no_atoms['atom_mask'].zero_()
    global_model=UnifiedInteraction(UnifiedConfig(variant='global',width=24,dropout=0)).eval()
    global_model.load_state_dict({k:model.state_dict()[k] for k in global_model.state_dict()},strict=True)
    torch.testing.assert_close(model(no_atoms),global_model(no_atoms),rtol=0,atol=0)


def test_refinement_capacity_control_is_matched():
    cfg=RefinedConfig(variant='geometry',width=24,pair_width=8)
    control=matched_refinement_config(cfg)
    actual=sum(p.numel() for p in RefinedInteraction(cfg).parameters())
    matched=sum(p.numel() for p in RefinedInteraction(control).parameters())
    assert abs(actual-matched)<=cfg.width+1
