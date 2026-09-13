"""Dense transport must preserve CSR's numeric-zero missingness after save/load."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from dtiam_numeric_xgboost_20260913 import NumericDenseXGBoostModel,XGBoostModel
    from autogluon.tabular import TabularPredictor
except ModuleNotFoundError as error:
    if error.name and error.name.startswith('autogluon'):
        NumericDenseXGBoostModel=None
    else:raise


@unittest.skipIf(NumericDenseXGBoostModel is None,'Run with .venv_dtiam_compat/bin/python')
class DenseTransportTests(unittest.TestCase):
    def test_native_probability_and_save_load_parity_with_zeros_and_nan(self):
        r=np.random.default_rng(19)
        x=pd.DataFrame(r.normal(size=(600,16)).astype('float32'),columns=[f'f{i}' for i in range(16)])
        x[r.random(x.shape)<.2]=0;x[r.random(x.shape)<.03]=np.nan
        original=x.copy(deep=True)
        y=pd.Series((x.f0.fillna(0)+x.f1.fillna(0)>.1).astype('uint8'))
        with tempfile.TemporaryDirectory() as folder:
            predictions=[]
            for cls,name in [(XGBoostModel,'sparse'),(NumericDenseXGBoostModel,'dense')]:
                m=cls(path=folder,name=name,problem_type='binary',eval_metric='roc_auc',hyperparameters={'n_estimators':30})
                m.fit(X=x.iloc[:480],y=y.iloc[:480],X_val=x.iloc[480:],y_val=y.iloc[480:],num_cpus=2,num_gpus=0,verbosity=0)
                pred=m.predict_proba(x);predictions.append(pred)
                m.save();loaded=cls.load(m.path)
                np.testing.assert_allclose(pred,loaded.predict_proba(x),rtol=0,atol=2e-7)
            np.testing.assert_allclose(*predictions,rtol=0,atol=2e-7)
        pd.testing.assert_frame_equal(original,x)

    def test_reject_non_embedding_types(self):
        m=NumericDenseXGBoostModel(name='guard',problem_type='binary')
        m._is_features_in_same_as_ex=True
        for f in [pd.DataFrame({'c':pd.Categorical(['a','b'])}),pd.DataFrame({'c':[1,2]})]:
            with self.assertRaisesRegex(ValueError,'float32'):m._preprocess(f)

    def test_predictor_preserves_native_name_and_loads_transport(self):
        x=pd.DataFrame(np.random.default_rng(7).normal(size=(240,4)).astype('float32'))
        x.columns=[f'f{i}' for i in range(4)];x.loc[::3,'f0']=0
        x['y']=np.tile([0,1],120)
        with tempfile.TemporaryDirectory() as folder:
            p=TabularPredictor(label='y',problem_type='binary',eval_metric='roc_auc',path=folder,verbosity=0)
            p.fit(x,hyperparameters={NumericDenseXGBoostModel:{'n_estimators':4}},num_cpus=2,num_gpus=0,
                memory_limit=4,fit_weighted_ensemble=False,calibrate_decision_threshold=False)
            self.assertEqual(p.model_names(),['XGBoost'])
            pred=p.predict_proba(x.drop(columns='y'),as_pandas=False)
            p=TabularPredictor.load(folder,verbosity=0)
            np.testing.assert_allclose(pred,p.predict_proba(x.drop(columns='y'),as_pandas=False),rtol=0,atol=2e-7)


if __name__=='__main__':unittest.main()
