from dataclasses import asdict
import copy
import importlib.util
from pathlib import Path
import torch
from biomaster.anchored_interaction import (AnchoredConfig,AnchoredInteraction,parent_model,
    chemical_width,matched_anchored_config)

spec=importlib.util.spec_from_file_location('anchored_fixture',Path(__file__).with_name('test_unified_interaction.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def test_each_supported_parent_is_preserved_at_initialization():
    torch.set_num_threads(2)
    for representation in ['drugclip','bermol','bermol_morgan','morgan','drugclip_morgan']:
        for parent_variant in ['global','capacity']:
            batch=fixture.example();batch['drug_global']=torch.randn(2,chemical_width(representation))
            cfg=AnchoredConfig(width=24,pair_width=8,dropout=0,parent_variant=parent_variant,
                parent_capacity_hidden=47,drug_representation=representation)
            parent=parent_model(cfg).eval()
            for variant in ['global','capacity','site','geometry']:
                model=AnchoredInteraction(AnchoredConfig(**{**asdict(cfg),'variant':variant})).eval()
                model.warm_start(parent.state_dict())
                torch.testing.assert_close(model(batch),parent(batch),rtol=0,atol=0)


def test_anchored_local_is_trainable_without_moving_the_global_readout():
    batch=fixture.example();batch['drug_global']=torch.randn(2,2816)
    batch['stereo'][batch['neighbors']>=0]=1
    model=AnchoredInteraction(AnchoredConfig(variant='geometry',width=24,pair_width=8,
        dropout=0,drug_representation='bermol_morgan')).eval()
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    model(batch).square().mean().backward();optimizer.step();optimizer.zero_grad()
    model(batch).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    counts=[]
    cfg=model.cfg;matched=matched_anchored_config(cfg)
    for c in [cfg,matched]:counts.append(sum(p.numel() for p in AnchoredInteraction(c).parameters()))
    assert abs(counts[0]-counts[1])<=cfg.width+1


def test_anchored_geometry_invariance_and_exact_structure_fallback():
    batch=fixture.example();cfg=AnchoredConfig(variant='geometry',width=24,pair_width=8,dropout=0)
    model=AnchoredInteraction(cfg).eval()
    with torch.no_grad():model.refine[-1].weight.normal_(0,.1)
    moved=copy.deepcopy(batch);q,_=torch.linalg.qr(torch.randn(3,3));moved['ca']=batch['ca']@q+20
    torch.testing.assert_close(model(batch),model(moved),rtol=2e-5,atol=2e-6)
    site=AnchoredInteraction(AnchoredConfig(**{**asdict(cfg),'variant':'site'})).eval()
    site.load_state_dict({k:v for k,v in model.state_dict().items() if '.geometry.' not in k},strict=True)
    batch['geometry_mask'].zero_()
    torch.testing.assert_close(model(batch),site(batch),rtol=0,atol=0)
