#!/usr/bin/env python3
"""Audit every catalog input and weight, plus exhaustive global score parity."""
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.model_registry import infer_family,build_model
from biomaster.portable_ranker_v2 import CatalogRanker
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import OUTPUT,SOURCE,write_json
from verify_biomaster_catalog_model_v2 import verify
OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main(directory,smoke=False):
    directory=Path(directory).resolve();torch.set_num_threads(4)
    metadata=json.loads((directory/'metadata.json').read_text())
    if not smoke and metadata['status']!='SELECTED_CATALOG_MODEL':raise ValueError('not selected release weights')
    source=Path(metadata['source_checkpoint']['path'])
    if file_identity(source)!=metadata['source_checkpoint']:raise ValueError('source checkpoint changed')
    if not smoke:
        selection=json.loads((OUT/'DEPLOYMENT_SELECTION.json').read_text())
        if selection['checkpoint']!=file_identity(source):raise ValueError('not the frozen deployment checkpoint')
    # Independently tests reload, erroneous inputs, optional structure fallback,
    # and exported CPU inference from a directory outside the research project.
    verify(directory);gc.collect();torch.cuda.empty_cache()
    packaged=CatalogRanker(directory,device='cuda')
    native_state=torch.load(source,map_location='cpu',weights_only=False)
    safe_state=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    if set(safe_state)!=set(['architecture','config','model']):raise ValueError('unexpected training payload in export')
    if safe_state['architecture']!=infer_family(native_state) or safe_state['config']!=native_state['config']:
        raise ValueError('architecture changed during export')
    if set(safe_state['model'])!=set(native_state['model']):raise ValueError('weight keys differ')
    for key,value in native_state['model'].items():
        if not torch.equal(value,safe_state['model'][key]):raise ValueError('exported weight differs: '+key)
    original=build_model(infer_family(native_state),native_state['config']).cuda().eval()
    original.load_state_dict(native_state['model'],strict=True)
    bank=AnchoredFeatureBank(OUTPUT,OUT/'data/supplemental_features',SOURCE,
        metadata['drug_representation'],local=original.is_local)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    nd,nt=len(old),len(targets);drug_axis=old.drug_feature_index.to_numpy(int)
    np.testing.assert_array_equal(packaged.drugs.native_feature_index,drug_axis)
    np.testing.assert_array_equal(packaged.targets.native_feature_index,targets.target_feature_index)
    # This covers every drug and every target at least once. Compare actual
    # model inputs from independent native and compact batching implementations.
    count=max(nd,nt);fields=set();max_error=0.;total_pairs=0
    with torch.inference_mode():
        for lo in range(0,count,64):
            ids=np.arange(lo,min(lo+64,count));d=ids%nd;t=ids%nt
            a=bank.batch(drug_axis[d],t,original.cfg.variant);b=packaged._batch(d,t)
            if set(a)!=set(b):raise ValueError('batch fields differ')
            for key in a:
                x,y=a[key],b[key]
                if key in ['bond','stereo']:
                    # Values beyond true molecule length are ignored padding;
                    # native and compact stores have different following rows.
                    x=x[a['atom_mask']];y=y[b['atom_mask']]
                if not torch.equal(x,y):raise ValueError('catalog input mapping differs: '+key)
                fields.add(key)
            reference=original(a).float().cpu().numpy();actual=packaged.model(b).float().cpu().numpy()
            np.testing.assert_allclose(reference,actual,rtol=2e-5,atol=2e-6)
            max_error=max(max_error,float(np.abs(reference-actual).max()));total_pairs+=len(ids)
        if not original.is_local:
            # Exhaustive global-catalog prediction equivalence, not just a
            # small random forward sample. Identical FP32 batch size both sides.
            for lo in range(0,nd*nt,256):
                ids=np.arange(lo,min(lo+256,nd*nt));d=ids//nt;t=ids%nt
                reference=original(bank.batch(drug_axis[d],t,original.cfg.variant)).float().cpu().numpy()
                actual=packaged.model(packaged._batch(d,t)).float().cpu().numpy()
                np.testing.assert_allclose(reference,actual,rtol=2e-5,atol=2e-6)
                max_error=max(max_error,float(np.abs(reference-actual).max()))
            total_pairs=nd*nt
    result=dict(status='PASS',smoke_only=smoke,bundle=file_identity(directory/'MANIFEST.json'),
        source_checkpoint=file_identity(source),all_weight_tensors_exact=len(native_state['model']),
        all_catalog_input_values_exact=True,drug_axis_count=nd,target_axis_count=nt,compared_input_fields=sorted(fields),
        atom_padding_comparison='bond/stereo beyond true atom length excluded; masks/tokens/neighbors fully compared',
        fp32_forward_pairs=total_pairs,all_catalog_forward_pairs_compared=not original.is_local,
        fp32_max_abs_error=max_error,independent_and_input_checks=file_identity(directory.parent/(directory.name+'_VALIDATION.json')),
        verifier=file_identity(Path(__file__)))
    write_json(directory.parent/(directory.name+'_RELEASE_VALIDATION.json'),result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    main(a.directory,a.smoke)
