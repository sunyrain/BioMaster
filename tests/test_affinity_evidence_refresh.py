"""Numerical and identity boundaries that could otherwise mislead SPR decisions."""
import importlib.util
from pathlib import Path
import sqlite3
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/build_affinity_evidence_refresh_20260910.py'
spec=importlib.util.spec_from_file_location('affinity_build',p)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

@pytest.mark.parametrize('value,unit,relation,nm',[
    ('>10','uM','>',10000),('≤0.5','nM','<=',.5),('1.2e-6','M','=',1200),
    ('0','nM','=',None),('not determined','nM','',None),('5','percent','=',None)])
def test_censored_concentrations(value,unit,relation,nm):
    rel,_,got=b.concentration(value,unit)
    assert rel==relation
    assert got==pytest.approx(nm) if nm is not None else got is None

def test_protein_scope():
    row={'Target Name':'ABL1','Number of Protein Chains in Target (>1 implies a multichain complex)':'1','UniProt (SwissProt) Primary ID of Target Chain 1':'P00519'}
    assert b.bdb_identity(row)==('P00519','single_accession_construct_unverified')
    row['Target Name']='ABL1 [T315I]'
    assert b.bdb_identity(row)[1]=='mutant'
    row['Number of Protein Chains in Target (>1 implies a multichain complex)']='2'
    assert b.bdb_identity(row)[1]=='complex_or_unresolved'

def test_endpoint_normalization_preserves_apparent_kd():
    c=sqlite3.connect(':memory:')
    c.execute('create table measurements('+','.join(x+' text' for x in b.COLUMNS)+')')
    w=b.Writer(c)
    w.add('test','1',endpoint='kd',value='1',unit='nM')
    w.add('test','2',endpoint='Kd_app',value='1',unit='nM')
    w.flush()
    assert c.execute('select endpoint from measurements').fetchall()==[('Kd',),('Kd_app',)]
