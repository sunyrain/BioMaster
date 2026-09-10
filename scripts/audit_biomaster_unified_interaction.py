#!/usr/bin/env python3
"""Feature alignment, temporal exposure and frozen-baseline checks before fitting."""
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import lmdb
import numpy as np
import pandas as pd
from biomaster.odti_pockets_v3 import file_identity
from biomaster.unified_interaction import UnifiedConfig,UnifiedInteraction,capacity_matched_config
from prepare_biomaster_unified_interaction import OUTPUT,SOURCE,write_json,heavy_record
from train_biomaster_unified_interaction import TemporalStage,identity,CONFIG


def main():
    checks={};features={}
    manifest=json.loads((SOURCE/'DATA_MANIFEST.json').read_text())
    for name in ['DEVELOPMENT_TRAIN.csv.gz','DEVELOPMENT_VALIDATION_NEW.csv.gz','FINAL_TRAIN.csv.gz',
                 'OLD_DRUG_INDEX.csv','TARGET_INDEX.csv.gz','RISK_SETS.npz']:
        path=SOURCE/name
        assert file_identity(path)['sha256']==manifest['files'][str(path.relative_to(ROOT))]
    checks['actual_temporal_training_files_match_frozen_manifest']=True
    for stem in ['ATOM','TARGET']:
        manifest=json.loads((OUTPUT/(stem+'_MANIFEST.json')).read_text())
        assert manifest['status']=='COMPLETE' and not manifest['labels_used']
        for item in list(manifest.get('files',{}).values())+list(manifest.get('globals',{}).values()):
            assert file_identity(item['path'])==item
        features[stem]=manifest
    checks['feature_manifest_hashes']=True
    index=np.load(OUTPUT/'ATOM_INDEX.npz');length=index['lengths'];offset=index['offsets']
    ids=np.load(OUTPUT/'REQUIRED_MOLECULE_IDS.npy')
    assert np.array_equal(np.diff(offset),length) and (length<=128).all()
    assert np.load(OUTPUT/'ATOM_DONE.npy')[ids].all()
    token=np.load(OUTPUT/'ATOM_TOKENS.npy',mmap_mode='r')
    chemistry=np.load(OUTPUT/'ATOM_CHEMISTRY.npy',mmap_mode='r')
    available=np.load(OUTPUT/'PRETRAINED_AVAILABLE.npy')
    for start in range(0,len(token),50000):assert np.isfinite(token[start:start+50000]).all()
    global_bank=np.load(OUTPUT/'MOLECULE_GLOBAL.npy',mmap_mode='r')
    assert np.isfinite(global_bank).all()
    assert (np.linalg.norm(global_bank[available],axis=1)>0).all()
    assert not (global_bank[~available]!=0).any()
    checks['complete_finite_atom_banks']=True
    source=pd.read_csv(SOURCE/'MOLECULES.csv.gz')
    isotope=source.index[source.model_ligand_smiles.str.contains(r'\[[23]H\]')].to_numpy()
    chosen=np.unique(np.r_[np.random.default_rng(17).choice(ids,128,replace=False),np.intersect1d(ids,isotope)])
    env=lmdb.open(str(OUTPUT/'MOLECULES.lmdb'),subdir=False,readonly=True,lock=False)
    with env.begin() as txn:
        for i in chosen:
            record=heavy_record(pickle.loads(txn.get(str(i).encode())))
            assert record['smiles']==source.model_ligand_smiles.iloc[i]
            assert 'H' not in record['atoms']
            if length[i]:
                assert len(record['atoms'])==length[i]
                assert np.array_equal(record['atom_features'],chemistry[offset[i]:offset[i+1]])
    env.close();checks['atom_coordinate_chemistry_order_samples']=len(chosen)
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz')
    coverage=pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv')
    assert np.array_equal(targets.sequence_sha256,coverage.sequence_sha256)
    assert np.array_equal(targets.sequence.str.len(),coverage.full_mean_coverage)
    for mode in ['SEQUENCE','SITE']:
        indices=np.load(OUTPUT/f'TARGET_{mode}_INDICES.npy')
        states=np.load(OUTPUT/f'TARGET_{mode}_TOKENS.npy')
        for i,row in enumerate(indices):
            valid=row[row>=0]
            assert len(np.unique(valid))==len(valid) and (valid<len(targets.sequence.iloc[i])).all()
        assert np.isfinite(states).all()
    checks['target_sequence_and_actual_residue_indices']=True
    protocol=json.loads(CONFIG.read_text());stages={}
    for name in ['development','final']:
        stage=TemporalStage(name);rng=np.random.default_rng(97)
        d,t,known,groups=stage.retrieval(rng,protocol)
        mapping=stage.old.set_index('drug_feature_index').old_drug_index
        actual=stage.known[pd.Series(d).map(mapping).to_numpy(int),t]
        assert np.array_equal(actual,known)
        for lo,hi,head in groups:
            if head==0:
                row=stage.known[mapping[d[lo]]];assert int(known[lo:hi].sum())==int(row.sum())
            else:
                row=stage.known[:,t[lo]];assert int(known[lo:hi].sum())==int(row.sum())
        stages[name]=dict(train_rows=len(stage.train),cutoff=stage.cutoff,old_positive_pairs=int(stage.known.sum()),
                          retrieval_all_known_positives_retained=True)
    checks['temporal_training_and_retrieval_contracts']=True
    old_identity=json.loads((SOURCE/'TRAINING_IDENTITY.json').read_text())
    for name,digest in old_identity['source_sha256'].items():assert file_identity(ROOT/name)['sha256']==digest
    checks['frozen_V3_J_training_source_unchanged']=True
    parameters={}
    for name in protocol['variants']:
        cfg=capacity_matched_config() if name=='capacity' else UnifiedConfig(variant=name)
        parameters[name]=sum(p.numel() for p in UnifiedInteraction(cfg).parameters())
    assert abs(parameters['capacity']/parameters['geometry']-1)<.001
    checks['global_capacity_control']=parameters
    compatibility=json.loads((OUTPUT/'PRETRAINED_COMPATIBILITY.json').read_text())
    assert compatibility['status']=='PASS'
    checks['upstream_pretrained_compatibility']=compatibility
    report=dict(status='PASS',checks=checks,stages=stages,identity=identity(),
                molecule_required=len(ids),molecule_pretrained_available=int(available[ids].sum()),
                residue_targets=len(targets),pocket_targets=int((coverage.pocket_union_residues>0).sum()),
                prior_test_labels_read=False)
    write_json(OUTPUT.parent/'VALIDATION.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
