import copy
import numpy as np
import torch

from biomaster.unified_interaction import (UnifiedInteraction,UnifiedConfig,PocketGeometry,
                                         BondMessage,capacity_matched_config,masked_softmax)


def example():
    torch.manual_seed(29)
    b,a,r=2,5,7
    neighbors=torch.full((b,a,8),-1,dtype=torch.long)
    for j in range(a):neighbors[:,j,0]=(j+1)%a;neighbors[:,j,1]=(j-1)%a
    return dict(drug_global=torch.randn(b,512),drug_graph_mean=torch.randn(b,40),
                pretrained_available=torch.ones(b),target_global=torch.randn(b,1280),
                atom_tokens=torch.randn(b,a,512),atom_chemistry=torch.randn(b,a,40),
                atom_mask=torch.ones(b,a,dtype=torch.bool),neighbors=neighbors,
                bond=(neighbors>=0).long(),stereo=torch.zeros_like(neighbors),
                residue_tokens=torch.randn(b,r,1280),residue_position=torch.rand(b,r),
                residue_mask=torch.ones(b,r,dtype=torch.bool),ca=torch.randn(b,r,3)*3,
                quality=torch.ones(b,r)*.8,geometry_mask=torch.ones(b,r,dtype=torch.bool))


def test_all_information_paths_receive_gradient():
    batch=example()
    for key in ['drug_global','target_global','atom_tokens','residue_tokens','ca']:
        batch[key].requires_grad_(True)
    model=UnifiedInteraction(UnifiedConfig(width=24,pair_width=8,dropout=0))
    model(batch).square().mean().backward()
    for key in ['drug_global','target_global','atom_tokens','residue_tokens','ca']:
        assert torch.isfinite(batch[key].grad).all()
        assert batch[key].grad.abs().sum()>0,key
    assert model.blocks[0].geometry.kernel[0].weight.grad.abs().sum()>0


def test_geometry_is_rigid_motion_invariant_and_missing_is_exact_fallback():
    batch=example();model=UnifiedInteraction(UnifiedConfig(width=24,pair_width=8,dropout=0)).eval()
    rotated=copy.deepcopy(batch)
    q,_=torch.linalg.qr(torch.randn(3,3))
    rotated['ca']=batch['ca']@q+torch.tensor([35.,-21.,83.])
    torch.testing.assert_close(model(batch),model(rotated),rtol=2e-5,atol=2e-6)
    geometry=PocketGeometry(24)
    residues=torch.randn(2,7,24)
    no_structure=torch.zeros(2,7,dtype=torch.bool)
    assert torch.equal(geometry(residues,batch['ca'],batch['quality'],no_structure),residues)
    site=UnifiedInteraction(UnifiedConfig(variant='site',width=24,pair_width=8,dropout=0)).eval()
    site.load_state_dict({k:v for k,v in model.state_dict().items() if '.geometry.' not in k},strict=True)
    batch['geometry_mask']=no_structure
    torch.testing.assert_close(model(batch),site(batch),rtol=0,atol=0)


def test_padding_and_residue_permutation_do_not_change_scores():
    batch=example();model=UnifiedInteraction(UnifiedConfig(width=24,pair_width=8,dropout=0)).eval()
    permuted=copy.deepcopy(batch);order=torch.randperm(7)
    for key in ['residue_tokens','residue_position','residue_mask','ca','quality','geometry_mask']:
        permuted[key]=permuted[key][:,order]
    torch.testing.assert_close(model(batch),model(permuted),rtol=2e-5,atol=2e-6)
    padded=copy.deepcopy(batch)
    for key in ['residue_tokens','residue_position','residue_mask','ca','quality','geometry_mask']:
        x=padded[key];shape=(2,3,*x.shape[2:]);extra=torch.zeros(shape,dtype=x.dtype)
        if key in ['residue_tokens','ca']:extra=torch.randn(shape)*1e3
        padded[key]=torch.cat([x,extra],1)
    torch.testing.assert_close(model(batch),model(padded),rtol=2e-5,atol=2e-6)
    # Padded atoms may contain arbitrary cached values from neighboring molecules.
    batch['atom_mask'][:,4]=False
    padded=copy.deepcopy(batch)
    padded['atom_tokens'][:,4]=1e5;padded['atom_chemistry'][:,4]=-1e5
    torch.testing.assert_close(model(batch),model(padded),rtol=0,atol=0)


def test_bond_neighbor_association_is_preserved():
    torch.manual_seed(91)
    block=BondMessage(8)
    atoms=torch.randn(1,3,8);mask=torch.ones(1,3,dtype=torch.bool)
    neighbors=torch.tensor([[[1,2],[0,2],[0,1]]]);bonds=torch.tensor([[[1,2],[1,1],[2,1]]])
    changed=bonds.clone();changed[0,0]=torch.tensor([2,1])
    stereo=torch.zeros_like(bonds)
    assert not torch.allclose(block(atoms,mask,neighbors,bonds,stereo)[:,0],
                              block(atoms,mask,neighbors,changed,stereo)[:,0])


def test_atom_permutation_with_remapped_bonds_is_invariant():
    batch=example();model=UnifiedInteraction(UnifiedConfig(width=24,pair_width=8,dropout=0)).eval()
    changed=copy.deepcopy(batch);order=torch.tensor([4,2,0,3,1]);inverse=torch.argsort(order)
    for key in ['atom_tokens','atom_chemistry','atom_mask','neighbors','bond','stereo']:
        changed[key]=changed[key][:,order]
    neighbors=changed['neighbors']
    changed['neighbors']=torch.where(neighbors>=0,inverse[neighbors.clamp_min(0)],-1)
    torch.testing.assert_close(model(batch),model(changed),rtol=2e-5,atol=2e-6)


def test_empty_local_inputs_and_capacity_control():
    batch=example();batch['atom_mask'][:]=False;batch['residue_mask'][:]=False
    model=UnifiedInteraction(UnifiedConfig(width=24,pair_width=8,dropout=0))
    assert torch.isfinite(model(batch)).all()
    for dtype in [torch.float16,torch.float32]:
        x=masked_softmax(torch.zeros(2,5,dtype=dtype),torch.zeros(2,5,dtype=torch.bool),1)
        assert torch.equal(x,torch.zeros_like(x))
    reference=UnifiedConfig()
    matched=capacity_matched_config(reference)
    a=sum(p.numel() for p in UnifiedInteraction(reference).parameters())
    b=sum(p.numel() for p in UnifiedInteraction(matched).parameters())
    assert abs(a-b)/a<.001


def test_temporal_feature_sources_and_real_residue_subsampling():
    from scripts.prepare_biomaster_unified_interaction import actual_indices,conformer
    assert np.array_equal(actual_indices(4,96),np.arange(4))
    selected=actual_indices(1000,96)
    assert len(selected)==96 and selected[0]==0 and selected[-1]==999
    _,record=conformer((5,'N[C@@H](C)C(=O)O'))
    assert record['coordinates'].shape==(len(record['atoms']),3)
    assert record['atom_features'][:,30:34].sum()>0
    assert 'label' not in record


def test_isotope_hydrogen_removal_preserves_heavy_atom_graph_mapping():
    from scripts.prepare_biomaster_unified_interaction import conformer,heavy_record
    _,original=conformer((17,'[2H]C([2H])([2H])OC'))
    result=heavy_record(original)
    assert len(result['atoms'])==3 and 'H' not in result['atoms']
    assert result['coordinates'].shape==(3,3)
    assert result['edges'].min()>=0 and result['edges'].max()<3
    assert result['atom_features'].shape==(3,40)
    assert np.array_equal(result['graph_mean'],original['graph_mean'])
