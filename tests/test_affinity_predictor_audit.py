import importlib.util
from pathlib import Path
import pandas as pd
import pytest
p=Path(__file__).resolve().parents[1]/'scripts/evaluate_affinity_refresh_predictor_20260910.py'
spec=importlib.util.spec_from_file_location('affinity_eval',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.mark.parametrize('v,r,label',[(1000,'=','POSITIVE'),(1000,'<','POSITIVE'),(1000,'>','UNRESOLVED'),(10000,'=','WEAK_NEGATIVE'),(10000,'>','WEAK_NEGATIVE'),(10000,'<','UNRESOLVED'),(2000,'=','GREY'),(1,'~','UNRESOLVED'),(0,'=','UNRESOLVED')])
def test_threshold_bounds(v,r,label):assert m.classify(v,r)==label

def frame(labels):
 return pd.DataFrame([dict(id=i,pair_id='a__t',ligand_key='a',target_chembl_id='t',gene_symbol='gene',source='same',endpoint='Kd',relation='=',value_nM=100 if s=='POSITIVE' else 10000,observation_label=s) for i,s in enumerate(labels)])
def test_conflict_not_strongest_value():
 assert m.aggregate(frame(['POSITIVE']*4+['WEAK_NEGATIVE'])).label.iloc[0]=='CONFLICT'
def test_duplicate_invariance():
 assert m.aggregate(frame(['WEAK_NEGATIVE'])).label.iloc[0]==m.aggregate(frame(['WEAK_NEGATIVE']*10)).label.iloc[0]
def test_unknown_not_negative():
 a=m.aggregate(frame(['UNRESOLVED']));assert a.label.iloc[0]=='UNRESOLVED' and pd.isna(a.binary_label.iloc[0])
