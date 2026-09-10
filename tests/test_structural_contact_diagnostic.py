import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_biomaster_structural_contact_learning import contact_metrics, paired_interval


def test_native_diagnostic_distinguishes_residue_shortcut_from_atom_matching():
    truth = np.array([[3., 9.], [9., 3.]])
    matched = np.array([[4., 0.], [0., 4.]])
    result = contact_metrics(truth, matched, seed=71)
    assert result['native_contact_ap'] == 1.
    assert result['native_residue_only_score_ap'] == 0.5
    shortcut = np.broadcast_to(np.array([4., 0.]), (2, 2))
    flat = contact_metrics(truth, shortcut, seed=71)
    assert flat['native_contact_ap'] == flat['native_atom_permuted_score_ap']
    assert flat['native_contact_ap'] == flat['native_residue_only_score_ap']


def test_cluster_bootstrap_preserves_constant_paired_difference():
    frame = pd.DataFrame({'cluster': ['a', 'a', 'b'], 'first': [.6, .7, .4], 'second': [.4, .5, .2]})
    result = paired_interval(frame, 'first', 'second')
    assert result['clusters'] == 2
    assert result['mean_difference'] == pytest.approx(.2)
    assert result['ci95'] == pytest.approx([.2, .2])
