#!/usr/bin/env python3
"""Verify exact preservation of the selected real global parent before training."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.anchored_interaction import AnchoredInteraction,AnchoredFeatureBank,parent_model,anchored_config,matched_anchored_config
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json,SOURCE,OUTPUT

OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main():
    torch.set_num_threads(4)
    selected=json.loads((OUT/'GLOBAL_PARENT_SELECTION.json').read_text())
    parent=selected['parents'][0];state=torch.load(parent['checkpoint']['path'],map_location='cpu',weights_only=False)
    if file_identity(Path(parent['checkpoint']['path']))!=parent['checkpoint']:raise ValueError('parent drift')
    representation=selected['selected']['representation']
    bank=AnchoredFeatureBank(OUTPUT,OUT/'data/supplemental_features',SOURCE,representation,local=True)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    di=np.array([0,1,17,33,121,309,400,519,600,719]);ti=np.array([0,1,13,55,111,211,300,350,382,383])
    unavailable=np.flatnonzero(~np.load(OUTPUT/'PRETRAINED_AVAILABLE.npy')[old.drug_feature_index.to_numpy(int)])
    if len(unavailable):di[0]=unavailable[0]
    missing=np.flatnonzero(~np.load(OUTPUT/'TARGET_GEOMETRY_MASK.npy').any(1))
    if len(missing):ti[0]=missing[0]
    d=old.drug_feature_index.to_numpy(int)[di]
    reference_config=anchored_config(state['config'],'geometry')
    baseline=parent_model(reference_config).cuda().eval();baseline.load_state_dict(state['model'],strict=True)
    rows=[]
    with torch.inference_mode():
        reference=baseline(bank.batch(d,ti,'global'))
        for variant in ['global','capacity','site','geometry']:
            cfg=anchored_config(state['config'],variant)
            if variant=='capacity':cfg=matched_anchored_config(reference_config)
            model=AnchoredInteraction(cfg).cuda().eval();model.warm_start(state['model'])
            value=model(bank.batch(d,ti,variant))
            torch.testing.assert_close(reference,value,rtol=0,atol=0)
            rows.append(dict(variant=variant,parameters=sum(x.numel() for x in model.parameters()),
                exact_initial_parent_prediction=True,max_abs_error=float((reference-value).abs().max())))
            del model
    counts={r['variant']:r['parameters'] for r in rows}
    assert abs(counts['geometry']-counts['capacity'])<=reference_config.width+1
    write_json(OUT/'ANCHORED_PREFLIGHT.json',dict(status='PASS',global_selection=file_identity(OUT/'GLOBAL_PARENT_SELECTION.json'),
        parent=parent['checkpoint'],pairs=len(d),variants=rows,includes_missing_molecule_3d=bool(len(unavailable)),
        includes_missing_receptor_geometry=bool(len(missing)),producer=file_identity(Path(__file__))))
    print(json.dumps(dict(status='PASS',variants=rows)))


if __name__=='__main__':main()
