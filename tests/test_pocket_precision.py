"""Scientific invariants and supervision semantics for the new interaction trunk."""
import numpy as np
import pytest
import torch

from biomaster.pocket_precision import PocketPrecision, PocketPrecisionConfig, structural_loss


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(n=3, na=5, nr=6):
    torch.manual_seed(72)
    b = dict(atom_tokens=torch.randn(n, na, 512), atom_chemistry=torch.randn(n, na, 40),
             atom_mask=torch.ones(n, na, dtype=torch.bool), atom_distance=torch.rand(n, na, na) * 10,
             atom_edges=torch.randn(n, na, na, 13), residue_tokens=torch.randn(n, nr, 1280),
             pocket_residue_tokens=torch.randn(n, nr, 512), residue_mask=torch.ones(n, nr, dtype=torch.bool),
             residue_distance=torch.rand(n, nr, nr) * 20, residue_geometry=torch.randn(n, nr, nr, 17),
             drug_aligned=torch.nn.functional.normalize(torch.randn(n, 128), dim=-1),
             pocket_aligned=torch.nn.functional.normalize(torch.randn(n, 128), dim=-1),
             pocket_metadata=torch.rand(n, 2), owner=torch.tensor([0, 0, 1])[:n])
    return b


def model():
    # Small dimensions ONLY for invariance tests; production defaults are tested
    # separately on real pockets by the full-width GPU validation script.
    return PocketPrecision(PocketPrecisionConfig(width=32, pair_width=16, heads=4, blocks=2,
                                                 outer_width=4, dropout=0, checkpoint_blocks=True)).eval()


def test_pocket_permutation_and_missing_pocket_identity():
    m, b, parent = model(), inputs(), torch.randn(2, 192)
    expected = m(b, parent)
    perm = torch.tensor([2, 0, 1])
    actual = m({k: v[perm] for k, v in b.items()}, parent)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    b['atom_mask'].zero_(); b['residue_mask'].zero_()
    actual, aux = m(b, parent, return_aux=True)
    torch.testing.assert_close(actual, parent, atol=0, rtol=0)
    assert torch.equal(aux['pocket_weights'], torch.zeros(3))


def test_atom_and_residue_permutation_equivariance():
    m, b, parent = model(), inputs(), torch.randn(2, 192)
    y, aux = m(b, parent, return_aux=True)
    ap, rp = torch.tensor([3, 1, 4, 0, 2]), torch.tensor([4, 0, 2, 5, 1, 3])
    c = {k: v.clone() for k, v in b.items()}
    for key in ['atom_tokens', 'atom_chemistry', 'atom_mask']:
        c[key] = c[key][:, ap]
    for key in ['atom_distance', 'atom_edges']:
        c[key] = c[key][:, ap][:, :, ap]
    for key in ['residue_tokens', 'pocket_residue_tokens', 'residue_mask']:
        c[key] = c[key][:, rp]
    for key in ['residue_distance', 'residue_geometry']:
        c[key] = c[key][:, rp][:, :, rp]
    actual, aaux = m(c, parent, return_aux=True)
    torch.testing.assert_close(actual, y, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(aaux['contact_logits'], aux['contact_logits'][:, ap][:, :, rp], atol=2e-5, rtol=2e-5)


def test_padding_cannot_change_scores_or_valid_contacts():
    m, b, parent = model(), inputs(), torch.randn(2, 192)
    b['atom_mask'][:, -2:] = False; b['residue_mask'][:, -2:] = False
    expected, aux = m(b, parent, return_aux=True)
    c = {k: v.clone() for k, v in b.items()}
    for key in ['atom_tokens', 'atom_chemistry']:
        c[key][:, -2:] = 1e4
    for key in ['residue_tokens', 'pocket_residue_tokens']:
        c[key][:, -2:] = -1e4
    actual, aaux = m(c, parent, return_aux=True)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(aaux['contact_logits'][aux['pair_mask']], aux['contact_logits'][aux['pair_mask']])


def test_first_step_gradients_and_unknown_contact_mask():
    m, b, parent = model().train(), inputs(), torch.randn(2, 192)
    out, aux = m(b, parent, return_aux=True)
    absent = torch.zeros_like(aux['pair_mask'])
    loss0 = structural_loss(aux, torch.full_like(aux['contact_logits'], float('nan')), absent, m.cfg)
    assert loss0.item() == 0
    distance = torch.rand_like(aux['contact_logits']) * 20
    loss = out.square().mean() + structural_loss(aux, distance, aux['pair_mask'], m.cfg)
    loss.backward()
    for name in ['atom_input.1.weight', 'residue_input.1.weight', 'atom_bias.net.0.weight',
                 'residue_bias.net.0.weight', 'blocks.0.outer_out.weight', 'distance_head.1.weight']:
        gradient = dict(m.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name


def test_receptor_rigid_motion_invariance_including_frames():
    from scripts.prepare_biomaster_pocket_precision import geometry_features
    rng = np.random.default_rng(13)
    ca = rng.normal(size=(3, 3)).astype(np.float32)
    record = dict(coordinates=np.repeat(ca, 2, axis=0) + rng.normal(size=(6, 3)).astype(np.float32),
                  atom_residue=np.repeat(np.arange(3), 2), ca=ca,
                  frames=np.tile(np.eye(3, dtype=np.float32), (3, 1, 1)),
                  frame_mask=np.ones(3, bool), atom_quality=np.ones(6, np.float32))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotation = rotation.astype(np.float32)
    moved = {**record, 'coordinates': record['coordinates'] @ rotation.T + 7,
             'ca': ca @ rotation.T + 7, 'frames': np.einsum('ij,rjk->rik', rotation, record['frames'])}
    before, after = geometry_features(record), geometry_features(moved)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, atol=3e-6)


def test_full_candidate_gradient_cache_matches_direct_gradient():
    from biomaster.pocket_precision_training import cached_query_backward, multi_positive_ranking_loss
    torch.manual_seed(81)
    model = torch.nn.Linear(3, 2)
    features = torch.randn(11, 3)
    labels = torch.tensor([True, False, False, True, False, False, False, False, True, False, False])
    loss = multi_positive_ranking_loss(model(features)[:, 1], labels) * 0.7
    loss.backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    def forward(d, t):
        return model(features[d])
    value = cached_query_backward(forward, np.arange(11), np.zeros(11, int), labels, 1,
                                  model, batch_size=3, weight=0.7)
    assert value == pytest.approx(float(loss.detach()), abs=1e-6)
    for parameter, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-6, rtol=1e-6)


def test_development_evaluation_keeps_full_risk_set():
    from types import SimpleNamespace
    import pandas as pd
    from scripts.train_biomaster_pocket_precision import evaluate
    stage = SimpleNamespace(val_known=np.array([[True, False, False], [False, False, True]]),
                            risk=np.array([[True, True, False], [True, True, True]]),
                            old=pd.DataFrame({'drug_feature_index': [0, 1]}))
    table = torch.tensor([[[3., 3.], [1., 1.], [100., 100.]], [[1., 1.], [2., 2.], [3., 3.]]])
    result = evaluate(lambda d, t: table[d, t], torch.nn.Linear(1, 1), stage, 2)
    assert result['d2t']['macro_ap'] == 1
    assert result['d2t']['positive_pairs'] == 2
    assert result['t2d']['positive_pairs'] == 2
