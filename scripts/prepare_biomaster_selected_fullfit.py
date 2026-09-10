#!/usr/bin/env python3
"""Prepare dated <=2025 deployment labels after selection and score freeze."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.temporal_relations import aggregate_window
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json
OUT=ROOT/'outputs/biomaster_best_model_20260906'


def main():
    selection=OUT/'FINAL_ARCHITECTURE_SELECTION.json';s=json.loads(selection.read_text())
    if s['status']!='FROZEN_FINAL_ARCHITECTURE' or s['test_labels_used_for_selection']:
        raise ValueError('full fit requires frozen development selection')
    release=OUT/'regression/PREDICTION_RELEASE.json';r=json.loads(release.read_text())
    if r['selection']!=file_identity(selection):raise ValueError('temporal predictions not frozen for this selection')
    folder=OUT/'data/fullfit_2025';folder.mkdir(exist_ok=True)
    identity=dict(selection=file_identity(selection),prediction_release=file_identity(release),
        annual=file_identity(SOURCE/'ANNUAL_STANDARDIZED.csv.gz'),producer=file_identity(Path(__file__)))
    if (folder/'MANIFEST.json').exists():
        prior=json.loads((folder/'MANIFEST.json').read_text())
        if prior['identity']!=identity or prior['train']!=file_identity(folder/'TRAIN.csv.gz'):
            raise ValueError('full fit data identity drift')
        print(json.dumps(prior));return
    annual=pd.read_csv(SOURCE/'ANNUAL_STANDARDIZED.csv.gz');frame=aggregate_window(annual,end=2025)
    frame=frame[frame.binary_label.notna()].copy();frame.binary_label=frame.binary_label.astype(int)
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv');frame['is_project_old_drug']=frame.drug_feature_index.isin(old.drug_feature_index)
    if frame.max_document_year.max()>2025 or frame.duplicated(['drug_feature_index','target_feature_index']).any():
        raise ValueError('invalid full fit date/identity')
    base=np.load(OUTPUT/'REQUIRED_MOLECULE_IDS.npy');extra=np.load(OUT/'data/supplemental_features/REQUIRED_MOLECULE_IDS.npy')
    available=np.union1d(base,extra);required=np.unique(frame.drug_feature_index);missing=np.setdiff1d(required,available)
    write_json(folder/'FEATURE_COVERAGE.json',dict(required_molecules=len(required),available_molecules=len(available),
        missing_molecules=missing.tolist(),no_training_rows_dropped=True,identity=identity))
    if len(missing):raise ValueError(f'{len(missing)} full fit molecules need label-independent feature completion; no rows dropped')
    for directory,ids in [(OUTPUT,np.intersect1d(required,base)),(OUT/'data/supplemental_features',np.intersect1d(required,extra))]:
        if not np.load(directory/'ATOM_DONE.npy')[ids].all():raise ValueError('unfinished drug features')
        if not np.isfinite(np.load(directory/'MOLECULE_GLOBAL.npy',mmap_mode='r')[ids]).all():raise ValueError('invalid drug features')
    frame.to_csv(folder/'TRAIN.csv.gz',index=False);o=frame[frame.is_project_old_drug]
    result=dict(status='COMPLETE',identity=identity,train=file_identity(folder/'TRAIN.csv.gz'),
        rows=len(frame),positives=int(frame.binary_label.sum()),molecules=len(required),old_rows=len(o),
        old_positives=int(o.binary_label.sum()),train_max_year=int(frame.max_document_year.max()),
        date_before_label=True,undated_measurements_excluded=True,no_rows_dropped_for_features=True,
        role='deployment full fit includes 2023-2025; these weights have no independent test score',architecture_or_seed_reselected=False)
    write_json(folder/'MANIFEST.json',result);print(json.dumps(result))


if __name__=='__main__':main()
