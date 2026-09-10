import json
import pandas as pd
import pytest
from biomaster.explorer_disease_reviews import load_disease_reviews


def package(root):
    path=root/'outputs/spr512_integrated_disease_20260909'
    path.mkdir(parents=True)
    (path/'VALIDATION.json').write_text('{"all_pass":true}')
    key={'ligand_inchikey':'CONTROL_OUTSIDE_CATALOG','target_chembl_id':'T1','selection_role':'POSITIVE_CONTROL','drug_names':'Control','gene_symbol':'Gene'}
    pd.DataFrame([{**key,'txgnn_available':True,'ot_status':'COMPLETE','treatment_direction_established':False,'top50_ot_overlap':2,'disease_hypotheses_json':json.dumps([{'txgnn_logit':1.2,'ot_score':.2}]),'ot_top3_json':'[]'}]).to_csv(path/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv',index=False)
    pd.DataFrame([{**key,'ec_relation_rows':0,'non_text_unambiguous_assertion_rows':0,'predicates':'','primary_sources':'','review_status':'NO_ECKG_RELATION_MATCH'}]).to_csv(path/'SPR512_ECKG_RELATION_REVIEW.csv',index=False)
    return path


def test_validated_package_attaches_catalog_external_control_to_target(tmp_path):
    package(tmp_path)
    result=load_disease_reviews(tmp_path,{}, {'T1':{'name':'Gene'}})
    row=result['targets']['T1']['disease_review']['spr_records'][0]
    assert row['drug_in_catalog'] is False
    assert row['txgnn_available'] is True
    assert row['selection_role']=='POSITIVE_CONTROL'
    assert 'spr512_integrated' in row['source_paths'][0]


def test_missing_companion_or_failed_validation_is_not_silent_fallback(tmp_path):
    path=package(tmp_path)
    (path/'SPR512_ECKG_RELATION_REVIEW.csv').unlink()
    with pytest.raises(ValueError,match='requires both'):
        load_disease_reviews(tmp_path,{}, {})
    (path/'VALIDATION.json').write_text('{"all_pass":false}')
    with pytest.raises(ValueError,match='not passed validation'):
        load_disease_reviews(tmp_path,{}, {})


def test_full_snapshot_replaces_cancer_cache_without_name_fallback(tmp_path):
    from biomaster.explorer_annotations import _load_restored_diseases
    path=tmp_path/'outputs/biomaster_disease_evidence_720x888_20260909'
    path.mkdir(parents=True)
    (path/'VALIDATION.json').write_text('{"all_pass":true}')
    pd.DataFrame([{'ligand_inchikey':'D1','txgnn_disease_id':'123_456','disease_name':'Disease group','logit':2.0,'directions':'immune'}]).to_csv(path/'DRUG_ALL_DISEASE_TOP30.csv.gz',index=False)
    pd.DataFrame([{'target_chembl_id':'T1','disease_id':'MONDO_0000123','disease_name':'Disease','overall_score':.4,'datatype_scores_json':'{"clinical":0.4}'}]).to_parquet(path/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet')
    result={'drugs':{'D1':{'txgnn_diseases':[]},'HELD':{'txgnn_diseases':[{'name':'old cancer'}]}},'targets':{'T1':{'target_diseases':[]},'UNMAPPED':{'target_diseases':[{'id':'old'}]}}}
    _load_restored_diseases(tmp_path,result,lambda *args,**kwargs:None)
    assert result['drugs']['HELD']['txgnn_diseases']==[]
    assert result['targets']['UNMAPPED']['target_diseases']==[]
    assert result['drugs']['D1']['txgnn_diseases'][0]['id']=='123_456'
    assert result['targets']['T1']['target_diseases'][0]['evidence_scores']['clinical']==.4
