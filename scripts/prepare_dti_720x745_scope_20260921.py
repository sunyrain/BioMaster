#!/usr/bin/env python3
"""Audit the888->745 source rule, freeze both directions, and reuse exact scores."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from target_universe_ch37_common import load_official_targets, load_classification

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dti_ranking_720x745_20260921'
STORE=ROOT/'data/research/dti_ranking_scope_20260921_v4'
UNIVERSE=ROOT/'outputs/target_universe_ch37_v2'
CAT=ROOT/'outputs/catalog_seven_models_20260916'
OLD=ROOT/'outputs/dti_rank_agreement_20260921'
DB=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
PROTOCOL=ROOT/'configs/dti_ranking_scope_20260921/PROTOCOL_v4.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def csv(name, frame):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def main():
    RDLogger.DisableLog('rdApp.warning')
    OUT.mkdir(parents=True,exist_ok=True);STORE.mkdir(parents=True,exist_ok=True)
    columns=['target_chembl_id','gene_symbol','uniprot_accession','target_name','organism','target_type',
        'component_id','sequence','sequence_length','sequence_sha256','chembl_gpcr','ot_class_contains_gpcr','is_gpcr',
        'small_molecule_moa','assay_lane','evidence_class','af_exact_sequence_model','af_pdb_path','structure_ready_strict']
    upstream=UNIVERSE/'TARGET_UNIVERSE_OFFICIAL_888_V2.csv'
    master=pd.read_csv(upstream,usecols=columns)
    with sqlite3.connect(f'file:{DB}?mode=ro',uri=True) as db:
        official=load_official_targets(db)
        classes=load_classification(db,set(official.component_id))
        version=pd.read_sql_query('SELECT * FROM version',db).to_dict('records')
    assert len(official)==len(master)==888
    match=master.merge(official[['target_chembl_id','sequence','uniprot_accession']],on='target_chembl_id',validate='one_to_one',suffixes=('','_sql'))
    assert len(match)==888 and match.sequence.eq(match.sequence_sql).all()
    assert match.uniprot_accession.eq(match.uniprot_accession_sql).all()
    c=master.merge(classes[['component_id','chembl_gpcr']],on='component_id',validate='one_to_one',suffixes=('','_sql'))
    assert c.chembl_gpcr.eq(c.chembl_gpcr_sql).all()
    assert master.chembl_gpcr.eq(master.ot_class_contains_gpcr).all()
    assert master.is_gpcr.eq(master.chembl_gpcr | master.ot_class_contains_gpcr).all()
    chosen=master.loc[~master.is_gpcr].sort_values('target_chembl_id').copy()
    existing=pd.read_csv(UNIVERSE/'TARGET_SET_NON_GPCR_ALL_V2.csv',usecols=['target_chembl_id','sequence_sha256'])
    assert len(chosen)==len(existing)==745
    assert set(zip(chosen.target_chembl_id,chosen.sequence_sha256))==set(zip(existing.target_chembl_id,existing.sequence_sha256))
    assert chosen.sequence.map(lambda s:hashlib.sha256(s.encode()).hexdigest()).eq(chosen.sequence_sha256).all()
    assert chosen.sequence.notna().all() and chosen.sequence.str.len().gt(0).all()
    assert chosen.target_chembl_id.nunique()==chosen.uniprot_accession.nunique()==chosen.sequence_sha256.nunique()==745
    # Independently reconstruct the older384 target list from its historical experimental gates.
    exclusion_path=ROOT/'outputs/final_target_package_ch37/FINAL_TARGET_EXCLUSION_AUDIT_888.csv'
    exclusion=pd.read_csv(exclusion_path,usecols=['target_chembl_id','first_exclusion_reason'])
    old_targets=pd.read_csv(CAT/'TARGETS.csv')
    old_expected=set(exclusion.loc[exclusion.first_exclusion_reason.isin(['INCLUDED','EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET']),'target_chembl_id'])
    assert len(old_expected)==384 and old_expected==set(old_targets.target_id)
    exact=old_targets.merge(chosen,left_on='target_id',right_on='target_chembl_id',validate='one_to_one',suffixes=('_old','_new'))
    assert len(exact)==384 and exact.sequence_old.eq(exact.sequence_new).all()
    master['primary_included']=~master.is_gpcr
    master['primary_decision']=np.where(master.is_gpcr,'GPCR_SEPARATE_SCOPE','INCLUDE_NO_FURTHER_GATES')
    master['legacy_384_member']=master.target_chembl_id.isin(old_expected)
    master=master.merge(exclusion,on='target_chembl_id',validate='one_to_one')
    csv('SOURCE_TO_PRIMARY_AUDIT_888.csv',master.drop(columns=['sequence','af_pdb_path']))
    chosen['legacy_384_member']=chosen.target_chembl_id.isin(old_expected)
    chosen['local_exact_af_file_exists']=chosen.af_exact_sequence_model & chosen.af_pdb_path.fillna('').map(lambda p:bool(p) and Path(p).is_file())
    target_input=chosen.rename(columns={'target_chembl_id':'target_id','gene_symbol':'gene','uniprot_accession':'uniprot_id',
        'sequence_length':'protein_length','sequence_sha256':'protein_sha256'})
    target_input.to_parquet(STORE/'TARGET_INPUTS_745.parquet',index=False)
    csv('TARGETS_745.csv',target_input.drop(columns=['sequence','af_pdb_path']))
    drugs=pd.read_csv(CAT/'DRUGS.csv').sort_values('drug_id')
    parent_drugs=pd.read_csv(ROOT/'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv')
    d=drugs.merge(parent_drugs[['drug_id','smiles']],on='drug_id',validate='one_to_one',suffixes=('','_parent'))
    assert len(d)==len(drugs)==drugs.drug_id.nunique()==720 and d.smiles.eq(d.smiles_parent).all()
    for r in drugs.itertuples():assert Chem.MolToInchiKey(Chem.MolFromSmiles(r.smiles))==r.drug_id
    csv('DRUGS_720.csv',drugs)
    pairs=drugs[['drug_id']].merge(target_input[['target_id']],how='cross')
    assert len(pairs)==536400 and not pairs.duplicated().any()
    pairs.to_parquet(STORE/'PRIMARY_720X745_PAIRS.parquet',index=False)
    family=chosen.groupby('assay_lane',as_index=False).agg(primary_targets=('target_chembl_id','size'),
        legacy_384_targets=('legacy_384_member','sum'),small_molecule_moa_targets=('small_molecule_moa','sum'),
        local_exact_structure_files=('local_exact_af_file_exists','sum'))
    csv('TARGET_FAMILY_COUNTS.csv',family)
    funnel=[(0,'ChEMBL37_HUMAN_SINGLE_PROTEIN_WITH_ANY_DRUG_MOA',888,0),
        (1,'EXCLUDE_GPCR_FROM_PRIMARY_KEEP_SEPARATE_AUDIT',745,143),
        (2,'NO_OTHER_TARGET_EXCLUSIONS',745,0)]
    csv('PRIMARY_FUNNEL.csv',pd.DataFrame(funnel,columns=['step','rule','retained','excluded_in_step']))
    snapshot_manifest=json.loads((OLD/'MANIFEST.json').read_text())
    score_path=OLD/'SIX_CURRENT_SCORE_SNAPSHOT.csv.gz'
    assert sha(score_path)==snapshot_manifest['artifacts'][score_path.name]
    old_scores=pd.read_csv(score_path)
    current=snapshot_manifest['current_models']
    assert set(old_scores.target_id)==old_expected and set(old_scores.drug_id)==set(drugs.drug_id)
    scores=pairs.merge(old_scores[['drug_id','target_id']+current],on=['drug_id','target_id'],how='left',validate='one_to_one')
    pending=['MAMMAL_pKd','BALM','EviDTI','GraphBAN']
    for model in pending:scores[model]=np.nan
    scores.to_parquet(STORE/'PARTIAL_720X745_SCORE_SNAPSHOT.parquet',index=False)
    coverage=[];queue=[]
    native_actions={'DrugCLIP':'QUALIFY_POCKETS_AND_NATIVE_INPUTS_THEN_SCORE',
        'DTBind_occurrence':'QUALIFY_EXACT_NATIVE_PROTEIN_GRAPHS_THEN_SCORE',
        'Nesso-1':'QUALIFY_NATIVE_INPUTS_LENGTH_AND_COST_THEN_SCORE'}
    for model in current+pending:
        n=int(scores[model].notna().sum())
        coverage.append(dict(model=model,requested_pairs=536400,reused_scored_pairs=n,missing_scores=536400-n,
            coverage_percent=100*n/536400,source='EXACT_FROZEN_384_SCORE_REUSE' if model in current else 'NO_QUALIFIED_SCORE_YET',
            scoring_complete=n==536400))
        counts=scores.groupby('target_id')[model].count()
        for target_id,n_scored in counts.items():
            queue.append(dict(model=model,target_id=target_id,requested_drugs=720,reused_scored_drugs=int(n_scored),
                missing_scores=720-int(n_scored),next_action='SCORES_PRESENT_FROM_FROZEN_VERSION' if n_scored==720 else
                'NATIVE_ADAPTER_QUALIFICATION_PENDING' if model in pending else native_actions.get(model,'QUALIFY_SEQUENCE_FEATURES_THEN_SCORE')))
    csv('MODEL_COVERAGE.csv',pd.DataFrame(coverage))
    csv('TARGET_MODEL_QUEUE.csv',pd.DataFrame(queue))
    source_paths=[upstream,UNIVERSE/'TARGET_SET_NON_GPCR_ALL_V2.csv',ROOT/'configs/target_universe_ch37_v2.yaml',
        ROOT/'scripts/target_universe_ch37_common.py',ROOT/'scripts/build_target_universe_ch37_v2.py',
        ROOT/'outputs/target_catalog_quality_audit_v1/opentargets_26_06_project_standard_target_flags.csv',
        exclusion_path,CAT/'DRUGS.csv',CAT/'TARGETS.csv',ROOT/'scripts/prepare_catalog_dti_20260916.py',
        ROOT/'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv',score_path]
    summary=dict(status='INPUTS_READY_PARTIAL_SCORE_REUSE',created_utc=datetime.now(timezone.utc).isoformat(),
        primary_drugs=720,primary_targets=745,primary_pairs=536400,source_targets=888,gpcr_separate=143,
        primary_small_molecule_moa=450,primary_without_small_molecule_moa=295,
        legacy_targets=384,new_targets_relative_to_legacy=361,new_pairs_relative_to_legacy=259920,
        sequence_records=745,local_exact_structure_files=int(chosen.local_exact_af_file_exists.sum()),
        raw_database_version=version,database_path=str(DB.relative_to(ROOT)),database_opened_read_only=True,
        checks={'sql888_exact_sequences':True,'sql_gpcr_classification':True,'gpcr_sources_agree':True,'745_frozen_members_identical':True,
            'legacy384_funnel_reproduced':True,'drug_inchikeys_match':True,'no_scores_or_binding_labels_used_for_target_selection':True,
            'legacy_prediction_identities_exact':True,'unique_complete_request_matrix':True},
        original_score_snapshot_utc=snapshot_manifest['snapshot_utc'],new_inference_started=False,new_training_started=False,
        no_complete_745_rank_claim=True,protocol=str(PROTOCOL.relative_to(ROOT)),protocol_sha256=sha(PROTOCOL),
        sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths},
        artifacts={str(p.relative_to(ROOT)):sha(p) for p in [*sorted(OUT.glob('*.csv')),*sorted(STORE.glob('*.parquet'))]},
        producer_sha256=sha(Path(__file__)))
    (OUT/'MANIFEST.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','primary_drugs','primary_targets','primary_pairs','gpcr_separate','new_targets_relative_to_legacy','checks']},indent=2))
    print(pd.DataFrame(coverage).to_string(index=False))


if __name__=='__main__':main()
