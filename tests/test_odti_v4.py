import pytest
import torch

from biomaster.odti_v4 import GlobalConfigV4, GlobalPairV4, SignedEvidenceV4


def fixture():
    torch.manual_seed(9)
    model = SignedEvidenceV4(GlobalPairV4(GlobalConfigV4(drug_dim=12, target_dim=10, hidden=16, dropout=0.1)))
    model.eval()
    return model, torch.randn(3,12), torch.randn(3,10), torch.randn(3,2,4,12), torch.rand(3,2,4), torch.ones(3,2,4,dtype=torch.bool)


def test_negative_evidence_cannot_raise_score_or_change_positive_branch():
    model,d,t,r,s,m = fixture()
    full = model(d,t,r,s,m)
    m[:,0] = False
    positive_only = model(d,t,r,s,m)
    assert torch.equal(full['base_logit'], positive_only['base_logit'])
    assert torch.equal(full['positive_bonus'], positive_only['positive_bonus'])
    assert (full['logit'] <= positive_only['logit']).all()
    assert (full['negative_penalty'] >= 0).all()
    assert (full['negative_penalty'] <= 2).all()


def test_empty_branches_are_exact_zero_with_nan_padding():
    model,d,t,r,s,m = fixture()
    m[:] = False
    r[:] = float('nan')
    s[:] = float('nan')
    result = model(d,t,r,s,m)
    assert torch.equal(result['logit'], result['base_logit'])
    assert torch.equal(result['positive_bonus'], torch.zeros(3))
    assert torch.equal(result['negative_penalty'], torch.zeros(3))
    result['logit'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_reference_delta_is_shared_potential_difference_and_antisymmetric():
    model,d,t,r,s,m = fixture()
    model.train()  # Evidence module forces deterministic shared potentials.
    result = model(d,t,r,s,m)
    q = model.base.drug(d)
    rs = model.base.drug(r)
    ts = model.base.target(t)
    f_q = model.base.potential(q,ts)
    f_r = model.base.potential(rs,ts[:,None,None].expand_as(rs))
    assert torch.allclose(result['reference_delta'], f_q[:,None,None]-f_r, atol=1e-6)
    a,b,c = f_r[:,0,0],f_r[:,0,1],f_r[:,0,2]
    assert torch.allclose((a-b)+(b-c),a-c,atol=1e-6)


def test_fresh_state_cache_matches_direct_forward():
    model,d,t,r,s,m = fixture()
    direct = model(d,t,r,s,m)
    q,ts,rs = model.base.drug(d), model.base.target(t), model.base.drug(r)
    pot = model.base.potential(rs,ts[:,None,None].expand_as(rs))
    cached = model.forward_states(q,ts,rs,s,m,pot)
    for key in direct:
        assert torch.allclose(direct[key],cached[key],atol=1e-6)


def test_initial_evidence_heads_receive_gradients():
    model,d,t,r,s,m = fixture()
    model.train()
    model.ramp.fill_(0.5)
    model(d,t,r,s,m)['logit'].sum().backward()
    for branch in model.branches:
        assert branch[-1].weight.grad.abs().sum() > 0


def test_negative_input_gradients_cannot_flow_into_positive_bonus():
    model,d,t,r,s,m = fixture()
    r.requires_grad_()
    result = model(d,t,r,s,m)
    grad = torch.autograd.grad(result['positive_bonus'].sum(),r)[0]
    assert torch.equal(grad[:,0],torch.zeros_like(grad[:,0]))


def test_wrong_label_axis_is_rejected():
    model,d,t,r,s,m = fixture()
    with pytest.raises(ValueError):
        model(d,t,r[:,:1],s[:,:1],m[:,:1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA inference regression')
def test_cuda_reference_cache_dtype_and_direct_equivalence():
    import sys
    from pathlib import Path
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
    from train_biomaster_odti_v4 import Runtime
    model,d,t,r,s,m = fixture()
    model.cuda().eval()
    runtime = Runtime.__new__(Runtime)
    runtime.molecules = {'bermol': torch.randn(10,12,device='cuda')}
    runtime.proteins = torch.randn(3,10,device='cuda')
    runtime.drugs = np.repeat(np.arange(3,10),3)
    runtime.targets = np.tile(np.arange(3),7)
    runtime.positions = {'train': np.arange(21)}
    ds,ts = np.array([0,1]),np.array([0,2])
    evidence = {'indices':np.array([[[3,4],[5,6]],[[4,5],[6,7]]]),
                'similarities': np.full((2,2,2),.5,np.float32), 'mask':np.ones((2,2,2),bool)}
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        direct = runtime.batch(model,'bermol',ds,ts,evidence,np.arange(2))['logit'].float().cpu().numpy()
    cached = runtime.predict(model,'bermol',ds,ts,evidence)['logit']
    np.testing.assert_allclose(direct,cached,atol=.02,rtol=.02)
