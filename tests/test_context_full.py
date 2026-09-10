"""Formal data flow, held-out query coverage and durable replay contracts."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from biomaster.context_pocket_full import FullContextPocketRanker, source_features, observed_structure_losses
from biomaster.context_full_training import (PermutationStream, query_cycle, validation_pairs,
    evaluate_matrices, capture_rng, restore_rng)
from biomaster.molecular_controls import MolecularControlConfig, MolecularControlInteraction
from biomaster.pocket_precision import PocketPrecisionConfig, PocketPrecision
from test_context_pocket import case, thread_limit


def full_case():
    _,g,b,_=case()
    pc=MolecularControlConfig(variant='global',drug_representation='drugclip_morgan',width=32,dropout=0)
    parent=MolecularControlInteraction(pc)
    cfg=PocketPrecisionConfig(width=32,pair_width=16,heads=4,blocks=2,outer_width=4,parent_width=32,dropout=.1)
    trunk=PocketPrecision(cfg)
    m=FullContextPocketRanker(dict(config=pc.to_dict(),model=parent.state_dict()),
        dict(config=cfg.to_dict(),local_model=trunk.state_dict()),cfg)
    return m,g,b


def test_real_context_is_required_and_receives_structural_gradients():
    m,g,b=full_case();m.train()
    with pytest.raises(ValueError,match='real global'):m.structural_forward(b,None,source='experimental_p2rank')
    aux=m.structural_forward(b,g,source='experimental_p2rank')
    aux['contact_logits'].square().mean().backward()
    for name in ['drug_global.1.weight','target_global.1.weight','context.0.1.weight','blocks.0.atom_q.weight']:
        assert dict(m.named_parameters())[name].grad.abs().sum()>0,name


def test_receptor_and_pocket_sources_have_separate_quality_semantics():
    _,_,b=full_case();h=source_features(b,'experimental_p2rank');a=source_features(b,'alphafold_p2rank')
    assert (h['quality_fields'][:,1]==0).all() and (a['quality_fields'][:,0]==0).all()
    torch.testing.assert_close(h['quality_fields'][:,2],a['quality_fields'][:,2])
    assert (h['quality_fields'][:,3]==1).all() and (h['quality_fields'][:,4]==0).all()
    assert (h['quality_fields'][:,5]==1).all() and (a['quality_fields'][:,5]==1).all()


def test_predicted_site_supervision_accepts_multiple_or_missing_positive_sites():
    m,g,b=full_case();aux=m.structural_forward(b,g,source='experimental_p2rank')
    truth=torch.full_like(aux['contact_logits'],20.);truth[0,0,0]=3.;truth[1,0,0]=3.
    loss=observed_structure_losses(aux,b,dict(distance=truth),m.cfg)
    assert torch.isfinite(loss).all()
    # Both candidates for owner 0 are positive, owner 1 has no positive site.
    assert abs(float(loss[2].detach()))<1e-6
    loss.sum().backward()


def test_full_query_union_includes_all_candidates_and_both_directions():
    labels=np.zeros((7,5),bool);labels[1,2]=1;labels[3,4]=1
    stage=SimpleNamespace(val_known=labels,risk=np.ones_like(labels))
    dq,tq,d,t=validation_pairs(stage)
    assert len(d)==2*5+2*7-2*2
    scores=np.full((7,5,2),np.nan,np.float32);scores[d,t]=0
    result,frames,_=evaluate_matrices(stage,scores)
    assert len(frames['d2t'])==len(frames['t2d'])==2
    scores[1,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):evaluate_matrices(stage,scores)


def test_rows_are_exhausted_before_resampling_and_resume_keeps_cycle():
    rng=np.random.default_rng(2);s=PermutationStream(np.arange(11),rng)
    first=s.take(7,rng);state=copy.deepcopy(s.state_dict());rstate=copy.deepcopy(rng.bit_generator.state)
    following=s.take(9,rng)
    assert len(np.unique(np.r_[first,following[:4]]))==11
    s.load_state_dict(state);rng.bit_generator.state=rstate
    np.testing.assert_array_equal(s.take(9,rng),following)
    cycle=query_cycle([np.arange(4),np.arange(6)],rng)
    assert len(set(map(tuple,cycle)))==10


def test_checkpoint_resume_reproduces_dropout_optimizer_and_sampling(tmp_path):
    torch.manual_seed(4);rng=np.random.default_rng(4)
    m=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Dropout(.4),torch.nn.Linear(7,1)).train()
    opt=torch.optim.AdamW(m.parameters(),lr=.01);scheduler=torch.optim.lr_scheduler.StepLR(opt,2,.8)
    stream=PermutationStream(np.arange(13),rng)
    def step(model,optimizer,schedule,sampler,random):
        indices=sampler.take(4,random);x=torch.randn(4,3);y=torch.tensor(indices[:,None],dtype=torch.float32)/13
        optimizer.zero_grad();loss=(model(x)-y).square().mean();loss.backward();optimizer.step();schedule.step()
        return indices,float(loss)
    step(m,opt,scheduler,stream,rng)
    path=tmp_path/'resume.pt'
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
               rng=capture_rng(rng),stream=stream.state_dict()),path)
    expected=step(m,opt,scheduler,stream,rng)
    saved=torch.load(path,weights_only=False)
    restored=copy.deepcopy(m);restored.load_state_dict(saved['model']);o=torch.optim.AdamW(restored.parameters(),lr=.01)
    sch=torch.optim.lr_scheduler.StepLR(o,2,.8);o.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler'])
    stream.load_state_dict(saved['stream']);restore_rng(saved['rng'],rng)
    actual=step(restored,o,sch,stream,rng)
    np.testing.assert_array_equal(expected[0],actual[0]);assert expected[1]==actual[1]
    for x,y in zip(m.parameters(),restored.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)
