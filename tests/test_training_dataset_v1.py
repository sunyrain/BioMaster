import pandas as pd
from biomaster.training_dataset_v1 import concentration,observation_label,molecule_identity,classify_construct,split_for_group,summarize_observations


def test_bounds_are_not_exact_values_or_automatic_negatives():
    assert concentration('>10','uM')==('>',10000.)
    assert observation_label(10000,'>')==0
    assert observation_label(10000,'<') is None
    assert observation_label(1000,'<')==1
    assert concentration('~10','nM')==(None,None)
    assert concentration('<10','nM','>')==(None,None)
    assert observation_label(None,'') is None
    assert observation_label(10,'=',True)==-1


def test_identity_preserves_stereochemistry_and_quarantines_mixtures():
    left,_=molecule_identity('C[C@H](O)C(=O)O')
    right,_=molecule_identity('C[C@@H](O)C(=O)O')
    assert left['molecule_id']!=right['molecule_id']
    assert left['connectivity_key']==right['connectivity_key']
    assert split_for_group(left['scaffold'])==split_for_group(right['scaffold'])
    assert molecule_identity('CCO.CCCO')[1]=='multiple_or_no_organic_components'
    assert molecule_identity('CCO','INCORRECT')[1]=='reported_inchikey_smiles_mismatch'
    ion,_=molecule_identity('C[NH3+].[Cl-]')
    assert ion['formal_charge']==1 and ion['counterions_removed']==1


def test_sequence_fragments_may_map_but_mutants_do_not():
    canonical='M'+'ACDEFGHIKLMNPQRSTVWY'*3
    assert classify_construct(canonical[2:40],canonical)=='exact_wildtype_fragment'
    assert classify_construct(canonical[:-1]+'A',canonical)=='sequence_mismatch_or_mutant'


def test_duplicates_cannot_outvote_negative_or_create_regression_value():
    g=pd.DataFrame([dict(endpoint='Kd',relation='=',value_nM=10.,observation_label=1)]*20+
                   [dict(endpoint='Ki',relation='>',value_nM=10000.,observation_label=0)])
    r=summarize_observations(g)
    assert r['label_status']=='CONFLICT' and r['binary_label'] is None
    assert r['exact_unique_values']==1 and r['unique_numeric_evidence']==2
    r=summarize_observations(g.iloc[-1:])
    assert r['binary_label']==0 and r['median_p_activity_unique'] is None


def test_conflicting_source_variants_and_rejected_identities_are_quarantined():
    from biomaster.training_dataset_v1 import record_quality
    frame=pd.DataFrame([
        dict(source='BindingDB_202609',source_record_id='1:Kd',variant_hash='a'),
        dict(source='BindingDB_202609',source_record_id='1:Kd',variant_hash='b'),
        dict(source='BindingDB_202609',source_record_id='2:Ki',variant_hash='c'),
        dict(source='ChEMBL37',source_record_id='2:Ki',variant_hash='d')])
    assert record_quality(frame,{'2'}).tolist()==['SOURCE_RECORD_VARIANT_CONFLICT','SOURCE_RECORD_VARIANT_CONFLICT',
                                                 'SOURCE_IDENTITY_REJECTED_IN_ANOTHER_VARIANT','ACCEPTED']


def test_connectivity_links_unite_different_scaffold_encodings():
    from scripts.finalize_training_dataset_20260910 import groups
    frame=pd.DataFrame(dict(molecule_id=['a','b','c'],connectivity_key=['X','X','Y'],scaffold=['s1','s2','s2']))
    result=groups(frame)
    assert result.split_group.nunique()==1 and result.scaffold_split.nunique()==1
