import importlib.util
import itertools
from pathlib import Path

import numpy as np

spec=importlib.util.spec_from_file_location('temporal_evaluation',Path(__file__).parents[1]/'scripts/evaluate_v3_temporal_20260906.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_exact_random_ap_matches_exhaustive_orderings():
    for n in range(1,7):
        for m in range(1,n+1):
            empirical=[]
            for positions in itertools.combinations(range(1,n+1),m):
                empirical.append(np.mean(np.arange(1,m+1)/np.array(positions)))
            assert np.isclose(module.random_expectation(n,m),np.mean(empirical))
