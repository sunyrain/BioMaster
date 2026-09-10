#!/usr/bin/env python3
"""Compare exported-code CPU scores to native-axis CPU scores outside the repo."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.model_registry import infer_family,build_model
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json
OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main(bundle):
    bundle=Path(bundle).resolve();torch.set_num_threads(4)
    meta=json.loads((bundle/'metadata.json').read_text());checkpoint=Path(meta['source_checkpoint']['path'])
    if file_identity(checkpoint)!=meta['source_checkpoint']:raise ValueError('source weights drift')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    native=build_model(infer_family(state),state['config']).eval();native.load_state_dict(state['model'])
    bank=AnchoredFeatureBank(OUTPUT,OUT/'data/supplemental_features',SOURCE,
        meta['drug_representation'],device='cpu',local=native.is_local)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    n=max(len(old),len(targets));d=np.arange(n)%len(old);t=np.arange(n)%len(targets)
    drugs=old.ligand_inchikey.to_numpy()[d];proteins=targets.uniprot_accession.to_numpy()[t]
    expected=np.empty((n,2),np.float32)
    with torch.inference_mode():
        for lo in range(0,n,128):
            sl=slice(lo,min(lo+128,n));expected[sl]=native(bank.batch(old.drug_feature_index.to_numpy()[d[sl]],t[sl],native.cfg.variant)).numpy()
    with tempfile.TemporaryDirectory(prefix='retargetmap_cpu_parity_') as temp:
        root=Path(temp);copy=root/'bundle';shutil.copytree(bundle,copy)
        (root/'queries.json').write_text(json.dumps(dict(drugs=drugs.tolist(),targets=proteins.tolist())))
        code='''import json,sys,torch,numpy as np
from pathlib import Path
torch.set_num_threads(4)
sys.path.insert(0,sys.argv[1])
from retargetmap import CatalogRanker
m=CatalogRanker(sys.argv[1],device="cpu")
q=json.loads(Path(sys.argv[2]).read_text())
y=m.score_pairs(q["drugs"],q["targets"],batch_size=128)
assert not any(k=="biomaster" or k.startswith("biomaster.") for k in sys.modules)
np.save(sys.argv[3],y[["drug_to_target_score","target_to_drug_score"]].to_numpy())
'''
        result=subprocess.run([sys.executable,'-I','-c',code,str(copy),str(root/'queries.json'),str(root/'observed.npy')],
            cwd=root,env={**os.environ,'PYTHONPATH':'','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},
            capture_output=True,text=True,timeout=180)
        if result.returncode:raise RuntimeError(result.stderr)
        actual=np.load(root/'observed.npy');np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    report=dict(status='PASS',bundle=file_identity(bundle/'MANIFEST.json'),source_checkpoint=file_identity(checkpoint),
        pairs=n,all_drugs_covered=True,all_targets_covered=True,all_cross_pairs_tested=False,
        cpu_max_abs_error=float(np.abs(actual-expected).max()),exported_code_imported_without_research_repository=True,
        verifier=file_identity(Path(__file__)))
    write_json(OUT/'EXPORTED_CPU_PARITY.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);main(p.parse_args().bundle)
