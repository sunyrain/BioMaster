#!/usr/bin/env python3
"""Run the five publisher complex examples on CPU; these are not SPR384 results."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
import types
from pathlib import Path
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'.external/DTBind/script/affinity'
OUT=ROOT/'outputs/frontier_dti_20260916/dtbind'
torch.set_num_threads(2)
sys.path.insert(0,str(REPO));os.chdir(REPO)
# The released helper unconditionally sends tensors to CUDA even in CPU mode.
# Only tensor placement is adapted; graph construction and model remain intact.
module=types.ModuleType('aff_utils');module.__file__=str(REPO/'aff_utils.py')
source=(REPO/'aff_utils.py').read_text().replace('.cuda()', '.to("cpu")')
exec(compile(source,str(REPO/'aff_utils.py'),'exec'),module.__dict__);sys.modules['aff_utils']=module
import aff_test
rows=[]
for path in sorted((ROOT/'.external/DTBind/sample_test/affinity').glob('*.pkl')):
    pdbid,truth,pred,error=aff_test.predict_single_sample(str(path))
    rows.append(dict(pdbid=pdbid,true_value=float(truth),pred_value=float(pred),error=float(error),scope='publisher_example_only_not_SPR384'))
assert len(rows)==5
pd.DataFrame(rows).to_csv(OUT/'AFFINITY_SMOKE.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
