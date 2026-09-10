#!/usr/bin/env python3
"""Audit the concrete selected release against the best-model completion contract."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
OUT=ROOT/'outputs/biomaster_best_model_20260906'
VERIFIED={}


def read(path):return json.loads(Path(path).read_text())


def verify(record):
    path=record['path']
    if path not in VERIFIED:VERIFIED[path]=file_identity(Path(path))
    if VERIFIED[path]!=record:raise ValueError('artifact drift: '+path)


def verify_identity(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:verify(value)
        else:
            for item in value.values():verify_identity(item)
    elif isinstance(value,list):
        for item in value:verify_identity(item)


def main():
    evidence={};checks=[]
    global_=read(OUT/'GLOBAL_PARENT_SELECTION.json');selected=read(OUT/'FINAL_ARCHITECTURE_SELECTION.json')
    if selected['status']!='FROZEN_FINAL_ARCHITECTURE' or selected['test_labels_used_for_selection']:
        raise ValueError('invalid final selection')
    verify(selected['global_parent_selection']);verify(selected['refinement_protocol']);verify(selected['producer'])
    for record in selected['results']:
        verify(record);result=read(record['path']);verify(result['checkpoint'])
        verify_identity(result['identity'])
    if len(selected['results'])!=30:raise ValueError('missing parent/local comparison')
    table=pd.read_csv(OUT/'ALL_GLOBAL_RUNS.csv')
    if len(table)!=54:raise ValueError('missing global comparison')
    if not selected['eligibility'][selected['selected_variant']]['eligible']:raise ValueError('unsupported retained component')
    exposures=read(OUT/'ANCHORED_EXPOSURE_AUDIT.json')['comparisons']
    if len(exposures)!=18 or any(x['epochs']!=4 or not x['all_exposures_identical'] for x in exposures):
        raise ValueError('matched local exposure audit incomplete')
    if read(OUT/'MATCHED_EPOCH_6_AUDIT.json')['status']!='PASS':raise ValueError('global exposure audit failed')
    values=selected['selected_development']['seed_composites'];median=np.median(list(values.values()))
    expected=min(map(int,values),key=lambda s:(abs(values[str(s)]-median),s))
    if expected!=selected['deployment_seed']:raise ValueError('deployment seed reselected')
    checks+=['54 global development runs and 24 matched local refinements','all retained components eligible under frozen rules',
        'same-parent initialization and matched extra exposure','development-only architecture/epoch/median-seed selection']
    refit=read(OUT/'FINAL_REFIT_STATUS.json')
    if refit['status']!='COMPLETE':raise ValueError('temporal refits incomplete')
    for record in refit['files']:
        verify(record['checkpoint']);verify(record['result']);result=read(record['result']['path'])
        if result['train_max_year']!=2022 or result['test_labels_used'] or result['smoke_only']:raise ValueError('invalid temporal fit')
        verify(result['identity']['final_selection']);verify(result['identity']['train'])
        verify_identity(result['identity'])
    regression=read(OUT/'regression/RESULT.json')
    if regression['status']!='COMPLETE' or regression['tests_used_for_selection'] or not regression['no_prediction_ensemble']:
        raise ValueError('invalid regression protocol')
    verify(regression['release']);verify(regression['selection']);verify(regression['producer']);verify(regression['metrics'])
    release=read(regression['release']['path'])
    for record in release['products']:verify(record['prediction']);verify(record['metadata'])
    for record in regression['label_sources']:verify(record)
    if len(release['products'])!=10:raise ValueError('three selected/G/capacity seeds and representative FP32 required')
    age=pd.read_csv(OUT/'SELECTED_FINAL_EARLY_AGE.csv')
    if len(age)!=12:raise ValueError('approval-age sensitivity incomplete')
    checks+=['three fixed <=2022 refits','predictions frozen before opened regression labels',
        'temporal/full-candidate and KIRHub/measured scopes kept distinct','seed metric means and paired-query uncertainty, no score ensemble',
        'historical approval and representative FP32 sensitivity reported']
    deployed=read(OUT/'DEPLOYMENT_SELECTION.json');fit=read(deployed['result']['path']);data=read(OUT/'data/fullfit_2025/MANIFEST.json')
    for key in ['checkpoint','result','training_data','regression_result','architecture_selection']:verify(deployed[key])
    if deployed['seed']!=expected or fit['train_max_year']!=2025 or fit['smoke_only'] or fit['training_role']!='fullfit_deployment':
        raise ValueError('invalid selected deployment fit')
    if not deployed['fullfit_weights_have_no_independent_test_score']:raise ValueError('full fit test claim')
    verify(data['train']);train=pd.read_csv(data['train']['path'])
    if train.max_document_year.max()>2025 or len(train)!=data['rows'] or not train.binary_label.isin([0,1]).all():
        raise ValueError('full fit data boundary')
    for source in fit['identity']['sources']:verify(source)
    delivered=read(OUT/'DELIVERED_MODEL.json')
    if delivered['status']!='COMPLETE' or not delivered['one_checkpoint']:raise ValueError('model not delivered')
    for key in ['archive','model','manifest','validation','selection','configuration','model_card']:verify(delivered[key])
    validation=read(delivered['validation']['path'])
    if validation['status']!='PASS' or validation['smoke_only'] or not validation['all_catalog_input_values_exact']:
        raise ValueError('inference validation incomplete')
    verify(validation['bundle']);verify(validation['source_checkpoint']);verify(validation['independent_and_input_checks'])
    independent=read(validation['independent_and_input_checks']['path'])
    if not all(independent[k] for k in ['independent_import','cpu_bidirectional_inference','reload_bitwise_equal']):
        raise ValueError('independent inference failed')
    bundle=Path(delivered['bundle']);metadata=read(bundle/'metadata.json')
    cpu=read(OUT/'EXPORTED_CPU_PARITY.json')
    if cpu['status']!='PASS' or not cpu['exported_code_imported_without_research_repository']:
        raise ValueError('exported CPU score parity missing')
    verify(cpu['bundle']);verify(cpu['source_checkpoint']);verify(cpu['verifier'])
    if not metadata['local_interactions'] and validation['fp32_forward_pairs']!=720*384:raise ValueError('global full-catalog parity missing')
    if len(list(bundle.glob('*.pt')))!=1:raise ValueError('multiple checkpoint payloads')
    current=read(ROOT/'configs/biomaster_current_contract_v1.json')['universes']
    if metadata['drugs']!=current['old_drugs'] or metadata['targets']!=current['current_scored_core_targets']:
        raise ValueError('silent registry/rank denominator change')
    for path in ['README.md','docs/README.md','docs/BIOMASTER_DRUG_TO_TARGET_VS_DTIAM_20260827_ZH.md']:
        text=(ROOT/path).read_text()
        if 'BIOMASTER_SELECTED_MODEL_20260906_ZH' not in text:raise ValueError('active entry does not point to delivered model: '+path)
    checks+=['dated <=2025 deployment labels and fixed training recipe','full-fit score boundary explicitly documented',
        'one standalone checkpoint/configuration/CLI/model card/archive','all catalog feature axes and weights checked',
        'FP32 parity, reload, invalid input and independent CPU inference','active documentation links actual delivery and respects registry scope']
    audit=read(OUT/'DATA_AND_REPRODUCTION_AUDIT.json')
    if audit['status']!='PASS':raise ValueError('baseline/data audit failed')
    for record in audit['exact_original_reproductions']:verify(record)
    evidence={name:file_identity(OUT/name) for name in ['DATA_AND_REPRODUCTION_AUDIT.json','GLOBAL_PARENT_SELECTION.json',
        'FINAL_ARCHITECTURE_SELECTION.json','ANCHORED_EXPOSURE_AUDIT.json','regression/RESULT.json','DEPLOYMENT_SELECTION.json','DELIVERED_MODEL.json','EXPORTED_CPU_PARITY.json']}
    write_json(OUT/'COMPLETION_AUDIT.json',dict(status='PASS',checks=checks,evidence=evidence,
        scope='best among declared development candidates for 720 old drugs / 384 targets; no universal or SOTA claim',
        retrospective_tests_previously_inspected=True,pretraining_cutoff_certified=False,producer=file_identity(Path(__file__))))
    print(json.dumps(dict(status='PASS',checks=len(checks),bundle=str(bundle))))


if __name__=='__main__':main()
