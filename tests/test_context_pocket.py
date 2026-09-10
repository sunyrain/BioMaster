"""Information flow, transfer, masking and invariance of the new architecture."""
import pytest
import torch
import numpy as np

from biomaster.context_pocket import ContextPocketRanker, sourced_pockets, structural_consistency
from biomaster.molecular_controls import MolecularControlConfig, MolecularControlInteraction
from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig


@pytest.fixture(autouse=True)
def thread_limit():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case():
    torch.manual_seed(8)
    parent_cfg = MolecularControlConfig(variant='global', drug_representation='drugclip_morgan', width=32, dropout=0)
    parent = MolecularControlInteraction(parent_cfg)
    cfg = PocketPrecisionConfig(width=32, pair_width=16, heads=4, blocks=2, outer_width=4,
                                parent_width=32, dropout=0, checkpoint_blocks=True)
    trunk = PocketPrecision(cfg)
    model = ContextPocketRanker(dict(config=parent_cfg.to_dict(), model=parent.state_dict()),
                               dict(config=cfg.to_dict(), local_model=trunk.state_dict()), cfg).eval()
    g = dict(drug_global=torch.randn(2, 2560), drug_graph_mean=torch.randn(2, 40),
             pretrained_available=torch.ones(2), target_global=torch.randn(2, 1280))
    n, a, r = 3, 4, 5
    b = dict(atom_tokens=torch.randn(n, a, 512), atom_chemistry=torch.randn(n, a, 40),
             atom_mask=torch.ones(n, a, dtype=torch.bool), atom_distance=torch.rand(n, a, a)*8,
             atom_edges=torch.randn(n, a, a, 13), residue_tokens=torch.randn(n, r, 1280),
             pocket_residue_tokens=torch.randn(n, r, 512), residue_mask=torch.ones(n, r, dtype=torch.bool),
             residue_distance=torch.rand(n, r, r)*12, residue_geometry=torch.randn(n, r, r, 17),
             drug_aligned=torch.randn(n, 128), pocket_aligned=torch.randn(n, 128),
             pocket_metadata=torch.tensor([[.8, .9], [.7, .8], [.6, .95]]), owner=torch.tensor([0, 0, 1]))
    return model, g, b, trunk


def test_old_readout_and_residual_bridge_are_absent_and_structural_weights_transfer():
    m, _, _, trunk = case()
    assert not hasattr(m, 'parent') and not hasattr(m, 'refine')
    for name in ['blocks.0.atom_q.weight', 'contact_head.1.weight', 'distance_head.1.weight']:
        torch.testing.assert_close(m.state_dict()[name], trunk.state_dict()[name], atol=0, rtol=0)


def test_global_context_changes_local_contacts_before_final_fusion():
    m, g, b, _ = case()
    _, before = m(g, b, True)
    changed = {k: v.clone() for k, v in g.items()}
    changed['target_global'][:, :40] += 3
    _, after = m(changed, b, True)
    assert (before['contact_logits']-after['contact_logits']).abs().max() > 1e-5


def test_adaptation_learns_global_context_and_scoring_without_updating_structural_trunk():
    m, g, b, _ = case()
    m.set_phase('adaptation'); m.train()
    m(g, b).square().mean().backward()
    for name in ['drug_global.1.weight', 'target_global.1.weight', 'context.0.1.weight',
                 'pocket_gate.1.weight', 'ranking_head.1.weight', 'ranking_head.6.weight']:
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in m.blocks.parameters())
    m.zero_grad(set_to_none=True); m.set_phase('joint')
    m(g, b).square().mean().backward()
    assert m.blocks[0].atom_q.weight.grad.abs().sum() > 0


def test_pocket_permutation_and_missing_input_are_safe():
    m, g, b, _ = case()
    expected = m(g, b)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(m(g, {k: v[perm] for k, v in b.items()}), expected, atol=2e-6, rtol=2e-6)
    b['atom_mask'].zero_(); b['residue_mask'].zero_()
    scores, aux = m(g, b, True)
    assert torch.isfinite(scores).all()
    assert not aux['relation_available'].any()
    assert torch.count_nonzero(aux['evidence'][:, m.cfg.parent_width:]) == 0


def test_quality_fields_are_source_specific_and_cached_inputs_unchanged():
    _, _, b, _ = case()
    geometry = b['residue_geometry'].clone()
    predicted = sourced_pockets(b, 'predicted')
    observed = sourced_pockets(b, 'experimental')
    assert torch.count_nonzero(predicted['quality_fields'][:, 0]) == 0
    assert torch.count_nonzero(observed['quality_fields'][:, 1:]) == 0
    torch.testing.assert_close(observed['quality_fields'][:, 0], b['pocket_metadata'][:, 1])
    torch.testing.assert_close(predicted['quality_fields'][:, 1], b['pocket_metadata'][:, 1])
    torch.testing.assert_close(predicted['residue_geometry'][..., -1], torch.ones_like(geometry[..., -1]))
    torch.testing.assert_close(b['residue_geometry'], geometry, atol=0, rtol=0)
    with pytest.raises(ValueError):
        sourced_pockets(b, 'unknown')


def test_single_pocket_keeps_explicit_matching_score_and_context():
    m, g, b, _ = case()
    _, aux = m(g, b, True)
    cosine_column = m.cfg.parent_width+m.cfg.width+128
    expected_cosine = (b['drug_aligned'][2]*b['pocket_aligned'][2]).sum()
    torch.testing.assert_close(aux['evidence'][1, cosine_column], expected_cosine)
    assert aux['pocket_weights'][2] == 1


def test_consistency_does_not_train_teacher_and_zero_when_equal():
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    c = torch.randn(2, 3, 4, requires_grad=True)
    d = torch.randn(2, 3, 4, 8, requires_grad=True)
    teacher = dict(pair_mask=mask, contact_logits=c, distance_logits=d)
    student = dict(pair_mask=mask, contact_logits=c.detach().clone().requires_grad_(),
                   distance_logits=d.detach().clone().requires_grad_())
    assert abs(float(structural_consistency(student, teacher).detach())) < 1e-6
    student['contact_logits'] = student['contact_logits']+1
    loss = structural_consistency(student, teacher); loss.backward()
    assert loss > 0 and c.grad is None and d.grad is None


def test_cached_full_query_gradient_matches_direct_new_model():
    from biomaster.pocket_precision_training import cached_query_backward, multi_positive_ranking_loss
    m,g,b,_=case()
    positive=torch.tensor([True,False])
    direct=multi_positive_ranking_loss(m(g,b)[:,0],positive)
    direct.backward()
    expected={name:p.grad.clone() for name,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    def forward(drug_ids,target_ids):
        ids=torch.as_tensor(drug_ids)
        take=torch.isin(b['owner'],ids)
        pb={key:value[take] for key,value in b.items()}
        mapping={int(original):new for new,original in enumerate(ids)}
        pb['owner']=torch.tensor([mapping[int(owner)] for owner in pb['owner']])
        return m({key:value[ids] for key,value in g.items()},pb)
    cached=cached_query_backward(forward,np.arange(2),np.zeros(2,int),positive,0,m,batch_size=1)
    assert cached==pytest.approx(float(direct.detach()),abs=2e-6)
    for name,p in m.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],atol=4e-6,rtol=5e-4,msg=name)
