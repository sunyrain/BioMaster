#!/usr/bin/env python3
"""Cache frozen V3 hidden states and logits without changing model weights."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.odti_support_data_v3 import SupportBatch
from biomaster.selectivity_training_v3 import support_summary_features
from train_biomaster_selectivity_v3 import Runtime
from build_biomaster_odti_v4_features import sha256, write_json

OUT = ROOT / 'outputs/biomaster_v3_incremental_20260906'
OLD = ROOT / 'outputs/biomaster_old_drug_bidirectional_20260906'
DATA = ROOT / 'outputs/biomaster_v3_20260905/data'
CFG = ROOT / 'configs/biomaster_v3_incremental_ablation_20260906.json'


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    if (OUT/'FEATURES.json').exists():
        raise FileExistsError('Frozen cache already exists; use training entrypoint')
    config = json.loads(CFG.read_text())
    protocol = {**config, 'config_sha256': sha256(CFG), 'status': 'FROZEN'}
    if (OUT/'PROTOCOL.json').exists():
        assert json.loads((OUT/'PROTOCOL.json').read_text()) == protocol
    else:
        write_json(OUT/'PROTOCOL.json', protocol)
    audit = json.loads((DATA/'DATA_MANIFEST_V3.json').read_text())
    assert sha256(DATA/'RELATIONS_V3.csv.gz') == audit['prepared_relations_sha256']
    data = pd.read_csv(DATA/'RELATIONS_V3.csv.gz')
    old_protocol = json.loads((OLD/'PROTOCOL.json').read_text())
    drugs = pd.read_csv(OLD/'OLD_DRUGS_720.csv')
    targets = pd.read_csv(OLD/'TARGETS_384.csv.gz')
    sources = [CFG, DATA/'DATA_MANIFEST_V3.json', DATA/'RELATIONS_V3.csv.gz', OLD/'PROTOCOL.json',
               OLD/'FEATURE_MANIFEST.json', OLD/'LABELS_AND_SCOPES.npz']
    for path,digest in json.loads((OLD/'FEATURE_MANIFEST.json').read_text())['files'].items():
        assert sha256(ROOT/path) == digest, path
    support = SupportBatch(**{n:np.load(audit[f'support_{n}_path'],mmap_mode='r') for n in ['indices','similarities','mask']})
    old_support = SupportBatch(**{n:np.load(p,mmap_mode='r') for n,p in old_protocol['support'].items()})
    np.save(OUT/'RELATION_SUPPORT.npy', support_summary_features(support.similarities,support.mask))
    np.save(OUT/'OLD_SUPPORT.npy', support_summary_features(old_support.similarities,old_support.mask))
    shapes = {'RELATION':len(data), 'OLD':720*384}
    hidden = {name:np.lib.format.open_memmap(OUT/f'{name}_HIDDEN.npy', mode='w+',dtype=np.float16,shape=(n,384)) for name,n in shapes.items()}
    logits = {name:np.zeros((n,2),np.float32) for name,n in shapes.items()}
    families = {name:i for i,name in enumerate(audit['audit']['families'])}
    old_ds = np.repeat(drugs.global_drug_index.to_numpy(np.int64),384)
    old_ts = np.tile(targets.global_target_index.to_numpy(np.int64),720)
    old_fs = np.tile(targets.target_assay_family.map(families).to_numpy(np.int64),720)
    missing = targets.global_target_index.ge(843).to_numpy()
    deploy = ROOT/'outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1'
    extra_prot_path = deploy/'PROJECT384_PROTBERT1024_FLOAT32_V1.npy'
    extra_aux_path = ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy'
    sources += [extra_prot_path, extra_aux_path]
    for j,seed in enumerate([20260905,20260906]):
        path = ROOT/f'outputs/biomaster_v3_20260905/experiments/B_support_pair/seed_{seed}/BEST_MODEL_V3.pt'
        sources.append(path)
        checkpoint = torch.load(path,map_location='cpu',weights_only=False)
        args = argparse.Namespace(**checkpoint['arguments'])
        rt = Runtime(args,data,audit['audit'],None,support)
        rt.restore(checkpoint)
        rt.drug_bank = torch.cat([rt.drug_bank,torch.from_numpy(np.load(OLD/'features/OLD720_MORGAN2048.npy')).cuda()])
        rt.target_bank = torch.cat([rt.target_bank,torch.from_numpy(np.load(extra_prot_path)[missing]).cuda()])
        rt.aux_bank = torch.cat([rt.aux_bank,torch.from_numpy(np.load(extra_aux_path)[missing]).cuda()])
        rt.mode(False)
        with torch.inference_mode():
            for name,ds,ts,fs,refs in [('RELATION',rt.drugs,rt.targets,rt.families,support),('OLD',old_ds,old_ts,old_fs,old_support)]:
                for start in range(0,len(ds),1024):
                    ix = np.arange(start,min(start+1024,len(ds)))
                    result = rt.model(**rt.inputs(ds[ix],ts[ix],fs[ix],refs,ix))
                    logits[name][ix,j] = result['final_logit'].cpu().numpy()
                    hidden[name][ix,j*192:(j+1)*192] = result['interaction_hidden'].cpu().numpy()
                hidden[name].flush()
                print(json.dumps({'cache':name,'seed':seed,'rows':len(ds)}),flush=True)
        del rt,checkpoint
        torch.cuda.empty_cache()
    for name,values in logits.items():
        np.save(OUT/f'{name}_BASE.npy',values.mean(1))
    reference = np.load(OLD/'scores/v3_support_ensemble.npz')['drug_to_target'].reshape(-1)
    error = float(np.max(np.abs(logits['OLD'].mean(1)-reference)))
    assert np.allclose(logits['OLD'].mean(1),reference,atol=5e-5),error
    np.savez_compressed(OUT/'ROW_AXES.npz',relation_drugs=data.drug_feature_index.to_numpy(np.int64),
        relation_targets=data.target_feature_index.to_numpy(np.int64),old_drugs=old_ds,old_targets=old_ts)
    paths = list(OUT.glob('*.npy'))+[OUT/'ROW_AXES.npz']
    write_json(OUT/'FEATURES.json', {'status':'COMPLETE','old_score_max_abs_error':error,
        'no_label_dependent_feature_extension':True,'hidden':'concatenated independent 192-wide frozen V3 query states; never averaged across feature bases',
        'sources':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'files':{str(p.relative_to(ROOT)):sha256(p) for p in paths}})
    print(json.dumps({'status':'COMPLETE','old_score_max_abs_error':error}),flush=True)


if __name__ == '__main__':
    main()
