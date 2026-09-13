"""Cover AutoGluon's dependency predictions when selecting an ensemble subset."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
try:
    from train_dtiam_ab_20260912 import prediction
except ModuleNotFoundError as error:
    if error.name and error.name.startswith('autogluon'):
        prediction = None
    else:
        raise


class DependencyPredictor:
    def predict_proba_multi(self, data, models, **kwargs):
        # AutoGluon may return ancestors alongside a requested ensemble. A fresh
        # request list also protects against backends that append dependencies.
        if 'component' in models:
            raise AssertionError('Caller reused a mutated model request')
        models.append('component')
        return {'ensemble': data.value.to_numpy(), 'component': np.zeros(len(data))}


@unittest.skipIf(prediction is None, 'Run with .venv_dtiam_compat/bin/python')
class PredictionTests(unittest.TestCase):
    def test_dependency_outputs_and_duplicate_panels_across_batches(self):
        n = 16385
        unique = pd.DataFrame({'pair_id': [str(i) for i in range(n)],
                               'value': np.linspace(0, 1, n, dtype=np.float32)})
        frame = pd.concat([unique, unique.iloc[[4, 0, n-1, 4]]], ignore_index=True)
        requested = ['ensemble']
        with patch('train_dtiam_ab_20260912.table', lambda f, bank: f):
            result = prediction(DependencyPredictor(), frame, None, requested)
        self.assertEqual(requested, ['ensemble'])
        self.assertEqual(set(result), {'ensemble'})
        np.testing.assert_array_equal(result['ensemble'], frame.value.to_numpy())

    def test_missing_requested_model_fails_instead_of_leaving_uninitialized_values(self):
        class MissingPredictor:
            def predict_proba_multi(self, data, **kwargs):
                return {'component': np.zeros(len(data))}
        frame = pd.DataFrame({'pair_id': ['p'], 'value': [.2]})
        with patch('train_dtiam_ab_20260912.table', lambda f, bank: f):
            with self.assertRaisesRegex(KeyError, 'ensemble'):
                prediction(MissingPredictor(), frame, None, ['ensemble'])


if __name__ == '__main__':
    unittest.main()
