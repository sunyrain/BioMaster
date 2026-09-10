#!/usr/bin/env python3
"""Check catalog export against native feature axes and isolated inference."""
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
from biomaster.portable_ranker import CatalogRanker
from biomaster.unified_features import UnifiedFeatureBank
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig
from biomaster.refined_interaction import RefinedInteraction,RefinedConfig
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import OUTPUT,SOURCE,write_json


def verify(directory):
    directory=Path(directory).resolve();torch.set_num_threads(2)
    packaged=CatalogRanker(directory,device='cuda')
    state=torch.load(packaged.metadata['source_checkpoint']['path'],map_location='cpu',weights_only=False)
    if state.get('refinement'):original=RefinedInteraction(RefinedConfig(**state['config']))
    else:original=UnifiedInteraction(UnifiedConfig(**state['config']))
    original.load_state_dict(state['model'],strict=True);original.cuda().eval()
    bank=UnifiedFeatureBank(OUTPUT,local=original.is_local)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    pd.testing.assert_series_equal(packaged.drugs.drug_id,old.ligand_inchikey,check_names=False)
    pd.testing.assert_series_equal(packaged.targets.target_id,targets.uniprot_accession,check_names=False)
    rng=np.random.default_rng(20260906);di=rng.integers(len(old),size=19);ti=rng.integers(len(targets),size=19)
    # Include the one old-drug pretrained fallback in the verification batch.
    fallback=np.flatnonzero(~np.load(OUTPUT/'PRETRAINED_AVAILABLE.npy')[old.drug_feature_index.to_numpy(int)])
    if len(fallback):di[0]=fallback[0]
    drugs=packaged.drugs.drug_id.iloc[di].tolist();protein=packaged.targets.target_id.iloc[ti].tolist()
    native_d=old.drug_feature_index.to_numpy(int)[di]
    with torch.inference_mode():
        reference=original(bank.batch(native_d,ti,original.cfg.variant)).float().cpu().numpy()
    observed=packaged.score_pairs(drugs,protein,batch_size=19)[['drug_to_target_score','target_to_drug_score']].to_numpy()
    np.testing.assert_allclose(reference,observed,rtol=2e-5,atol=2e-6)
    first=packaged.rank_targets(drugs[0],candidates=protein[:4],top_k=3) if len(set(protein[:4]))==4 else None
    if first is not None:assert first.candidate_count.eq(4).all() and first['rank'].tolist()==[1,2,3]
    for operation in [lambda:packaged.score_pairs(['INVALID'],[protein[0]]),
                      lambda:packaged.score_pairs([drugs[0]],[None]),
                      lambda:packaged.score_pairs([drugs[0]],protein[:2]),
                      lambda:packaged.score_pairs([drugs[0]],[protein[0]],batch_size=-1),
                      lambda:packaged.rank_drugs(protein[0],candidates=[drugs[0],drugs[0]]),
                      lambda:packaged.rank_targets(drugs[0],top_k=0)]:
        try:operation()
        except ValueError:pass
        else:raise AssertionError('invalid input accepted')
    reloaded=CatalogRanker(directory,device='cuda')
    np.testing.assert_array_equal(observed,reloaded.score_pairs(drugs,protein,batch_size=19)[['drug_to_target_score','target_to_drug_score']].to_numpy())
    geometry_error=None
    # Mutate only a temporary copy, never the delivered bundle or source cache.
    with tempfile.TemporaryDirectory(prefix='biomaster_portable_') as temp:
        copied=Path(temp)/'bundle';shutil.copytree(directory,copied)
        if original.is_local:
            off=packaged.score_pairs(drugs,protein,batch_size=19,use_geometry=False)[['drug_to_target_score','target_to_drug_score']].to_numpy()
            for name in ['ca','quality','geometry_mask']:(copied/'features'/(name+'.npy')).unlink()
            missing=CatalogRanker(copied,device='cuda')
            assert not missing.geometry_available
            no_geometry=missing.score_pairs(drugs,protein,batch_size=19)[['drug_to_target_score','target_to_drug_score']].to_numpy()
            np.testing.assert_array_equal(off,no_geometry);geometry_error=float(np.max(np.abs(off-no_geometry)))
            del missing
        # PYTHONPATH is empty and cwd lies outside the project. Only exported
        # package files are available through the explicit copied bundle path.
        code='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import retargetmap
from retargetmap import CatalogRanker
m=CatalogRanker(sys.argv[1],device="cpu")
d=m.drugs.drug_id.iloc[0];t=m.targets.target_id.iloc[0]
a=m.rank_targets(d,candidates=m.targets.target_id.iloc[:7].tolist(),top_k=3)
b=m.rank_drugs(t,candidates=m.drugs.drug_id.iloc[:5].tolist(),top_k=3)
assert len(a)==3 and a.candidate_count.eq(7).all()
assert len(b)==3 and b.candidate_count.eq(5).all()
assert Path(retargetmap.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert not any(name=="biomaster" or name.startswith("biomaster.") for name in sys.modules)
print(json.dumps({"independent_import":True,"cpu_bidirectional_inference":True}))
'''
        env={**os.environ,'PYTHONPATH':'','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        proc=subprocess.run([sys.executable,'-I','-c',code,str(copied)],cwd=temp,env=env,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120)
        if proc.returncode:raise RuntimeError(proc.stderr)
        independent=json.loads(proc.stdout.strip())
    result=dict(status='PASS',bundle=file_identity(directory/'MANIFEST.json'),
        checkpoint=file_identity(Path(packaged.metadata['source_checkpoint']['path'])),
        tested_pairs=len(di),fp32_max_abs_error=float(np.max(np.abs(reference-observed))),
        complete_registry_axes_match=True,reload_bitwise_equal=True,invalid_inputs_rejected=6,
        missing_geometry_max_abs_error=geometry_error,**independent,verifier=file_identity(Path(__file__)))
    write_json(directory.parent/(directory.name+'_VALIDATION.json'),result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    verify(parser.parse_args().directory)
