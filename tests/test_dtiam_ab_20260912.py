"""Fair-comparison guards for the September native DTIAM retraining."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dtiam_ab_common_20260912 import basic,calibrate,query_summary,probability_score,query_rows


class ComparisonTests(unittest.TestCase):
    def test_classification_matches_existing_endpoint_evaluator(self):
        from run_endpoint_ablation_20260911 import basic as existing
        rng=np.random.default_rng(919)
        for y in [rng.integers(0,2,201),np.zeros(80,dtype=int),np.ones(80,dtype=int)]:
            p=rng.choice(np.linspace(0,1,31),size=len(y))
            old=existing(y,p,p,.47);new=basic(y,p,p,.47)
            for k,value in old.items():
                if value is None:self.assertIsNone(new[k],k)
                else:self.assertAlmostEqual(new[k],value,msg=k)

    def test_tied_bidirectional_rank_metrics_match_existing(self):
        from run_endpoint_multitask_20260911 import QueryMetrics
        rng=np.random.default_rng(12)
        f=pd.DataFrame({'binary_label':rng.integers(0,2,600),
                        'target_id':np.repeat(np.arange(20),30),
                        'molecule_id':np.tile(np.arange(30),20)})
        # Include small and all-inactive queries that must not inflate macro AP.
        f.loc[f.target_id.eq(0),'binary_label']=0
        f.loc[:4,'target_id']=30
        p=rng.choice([.05,.2,.7,.9],len(f))
        result=query_summary(f,p)
        for direction,col in [('target','target_id'),('drug','molecule_id')]:
            expected=QueryMetrics(f,col).evaluate(p)
            for name in ['queries','macro_ap','p5','p10','p20']:
                self.assertAlmostEqual(result[direction][name],expected[name],msg=direction+name)

    def test_calibration_excludes_nonbinding_panels(self):
        from run_endpoint_ablation_20260911 import calibrate as existing
        f=pd.DataFrame({'panel':['AFFINITY_KD_KI']*8+['EXPLICIT_INACTIVE']*4,
                        'binary_label':[0,0,1,0,1,1,0,1]+[0]*4})
        raw=np.array([.01,.06,.1,.2,.3,.7,.4,.95,.9,.9,.9,.9])
        score=probability_score(raw)
        new=calibrate(f,score);old=existing(f,score)
        for key in ['slope','intercept','threshold','validation_f1']:
            self.assertAlmostEqual(new[key],old[key],places=10)
        changed=score.copy();changed[8:]=-30
        self.assertEqual(new,calibrate(f,changed))

    def test_ranking_retains_extreme_native_probability_order(self):
        p=np.array([1e-10,1e-9,1e-8,1e-7,.3,.4,.5,.6,.8,.9])
        f=pd.DataFrame({'target_id':['t']*10,'binary_label':[0,1,0,1,0,1,0,1,0,1]})
        q=query_rows(f,p,'target_id')[0]
        # Raw probabilities determine ordering; calibration clipping cannot introduce ties here.
        from sklearn.metrics import average_precision_score
        self.assertAlmostEqual(q['ap'],average_precision_score(f.binary_label,p))
        self.assertFalse(np.isinf(probability_score([0,1])).any())


if __name__=='__main__':unittest.main()
