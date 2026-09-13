"""Ensure isolated continuation retains native parameters, members and learned weights."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_dtiam_isolated_core_20260913 import (
    core_plan,member_roles,inventory,verify_preserved,TabularPredictor,get_hyperparameter_config)


class IsolatedCoreTests(unittest.TestCase):
    def test_plan_keeps_every_binary_default_and_no_learning_overrides(self):
        defaults=get_hyperparameter_config('default');before=copy.deepcopy(defaults)
        plan=core_plan(defaults)
        self.assertEqual(len(plan),10)
        self.assertEqual(plan['LightGBM'],{'GBM':[{}]})
        self.assertEqual([plan[n]['GBM'][0] for n in ['LightGBMXT','LightGBM','LightGBMLarge']],defaults['GBM'])
        for family,names in [('RF',['RandomForestGini','RandomForestEntr']),('XT',['ExtraTreesGini','ExtraTreesEntr'])]:
            self.assertEqual([plan[n][family][0] for n in names],defaults[family][:2])
        plan['LightGBMXT']['GBM'][0]['num_boost_round']=2
        self.assertEqual(defaults,before)

    def test_native_continuation_preserves_inner_members_and_prior_model(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            rng=np.random.default_rng(41)
            x=pd.DataFrame(rng.normal(size=(240,4)),columns=list('abcd'))
            x['y']=np.tile([0,1],120)
            members=pd.DataFrame({'pair_id':[f'p{i}' for i in range(len(x))],'binary_label':x.y})
            p=TabularPredictor(label='y',problem_type='binary',eval_metric='roc_auc',path=str(run/'predictor'),verbosity=0)
            p.fit(x,hyperparameters={'GBM':[{'extra_trees':True,'ag_args':{'name_suffix':'XT'},'num_boost_round':2}]},
                num_cpus=2,num_gpus=0,memory_limit=4,fit_weighted_ensemble=False,calibrate_decision_threshold=False)
            old_roles,nt,nv=member_roles(p,members)
            self.assertEqual(nt+nv,len(x))
            old=inventory(p,run)
            old_probability=p.predict_proba(x.drop(columns='y'),model='LightGBMXT',as_pandas=False)
            p=TabularPredictor.load(str(run/'predictor'),verbosity=0)
            p.fit_extra(hyperparameters={'GBM':{'num_boost_round':2}},num_cpus=2,num_gpus=0,
                memory_limit=4,fit_weighted_ensemble=False)
            self.assertEqual(set(verify_preserved(p,run,old,['LightGBM'])),{'LightGBMXT','LightGBM'})
            new_roles,_,_=member_roles(p,members)
            pd.testing.assert_frame_equal(old_roles,new_roles)
            np.testing.assert_array_equal(old_probability,p.predict_proba(x.drop(columns='y'),model='LightGBMXT',as_pandas=False))
            wrong=members.copy();wrong.loc[0,'binary_label']=1-wrong.loc[0,'binary_label']
            with self.assertRaises(AssertionError):member_roles(p,wrong)
            with self.assertRaises(AssertionError):verify_preserved(p,run,old,['UnexpectedModel'])


if __name__=='__main__':unittest.main()
