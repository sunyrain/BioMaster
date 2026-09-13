"""Memory-bounded transport for the native XGBoost learner's numeric embeddings.

AutoGluon's default CSR encoding omits numeric zeros, which XGBoost treats as
missing. Encode those same entries as NaN in a dense array on both fit and
inference. All native tree parameters, callbacks, stopping and persistence stay
in XGBoostModel; categorical and non-float32 inputs are deliberately rejected.
"""
import numpy as np
from autogluon.core.models import AbstractModel
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel


class NumericDenseXGBoostModel(XGBoostModel):
    ag_name='XGBoost'

    def _preprocess(self,X,**kwargs):
        X=AbstractModel._preprocess(self,X,**kwargs)
        if not all(dtype==np.dtype('float32') for dtype in X.dtypes):
            raise ValueError('DTIAM dense XGBoost transport requires only float32 numeric embeddings')
        values=X.to_numpy(copy=True)
        for start in range(0,len(values),32768):
            part=values[start:start+32768]
            part[part==0]=np.nan
        return values
