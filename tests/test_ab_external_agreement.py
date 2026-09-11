import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('ab_external_agreement', Path(__file__).resolve().parents[1] / 'scripts/audit_ab_external_agreement_20260911.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_identical_rankings_have_twenty_shared_hits():
    score = np.arange(100)[None, :]
    result, rho, hits = module.agreement(score, score * 5 + 3)
    assert np.isclose(rho[0], 1)
    assert hits[0] == 20 and result['mean_top20_overlap_fraction'] == 1
    assert result['random_independent_top20_overlap_fraction'] == .2


def test_reversed_rankings_do_not_share_top_twenty():
    score = np.arange(100)[None, :]
    _, rho, hits = module.agreement(score, -score)
    assert np.isclose(rho[0], -1) and hits[0] == 0


def test_missing_scores_require_an_explicit_common_scope():
    score = np.arange(100, dtype=float)[None, :]
    missing = score.copy(); missing[0, 0] = np.nan
    with pytest.raises(AssertionError):
        module.agreement(score, missing)
