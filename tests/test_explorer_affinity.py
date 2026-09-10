import sqlite3
from types import SimpleNamespace
import pytest
from biomaster.explorer_affinity import AffinityAtlas, DB, classify

@pytest.mark.parametrize('relation,value,expected',[('=',1000,'lower'),('=',1001,'higher'),('<',1000,'lower'),('<=',1000,'lower'),('>',1000,'higher'),('>=',1000,'uncertain'),('>=',1001,'higher'),('<',2000,'uncertain'),('>',500,'uncertain'),('~',10,'uncertain'),('=',None,'uncertain')])
def test_threshold_respects_censoring(relation,value,expected):
    assert classify(relation,value,1000)==expected

def test_matrix_preserves_pending_conflict_and_qc(tmp_path):
    path=tmp_path/DB;path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE project_drugs(ligand_inchikey,drug_names);
        INSERT INTO project_drugs VALUES ('D1','Drug 1'),('D2','Drug 2');
        CREATE TABLE project_targets(target_chembl_id,gene_symbol);
        INSERT INTO project_targets VALUES ('T1','Target 1'),('T2','Target 2');
        CREATE TABLE measurements(id,source,record_id,ligand_key,target_chembl_id,endpoint,relation,value_nM);
        INSERT INTO measurements VALUES (1,'S','1','D1','T1','Kd','=',10),(2,'S','2','D1','T1','Kd','>',2000),(3,'S','3','D2','T2','IC50','=',10),(4,'S','4','D2','T1','Kd','=',10);
        CREATE VIEW comparable_numeric_evidence AS SELECT * FROM measurements;
        CREATE TABLE auxiliary_observations(source);
        CREATE TABLE evidence_quality_reviews(source,record_id,decision,reason,source_url);
        INSERT INTO evidence_quality_reviews VALUES ('S','4','HOLD','identity','');
        ''')
    data=SimpleNamespace(ensure_loaded=lambda:None,targets={'T1':{'experiments':[{'pair_id':'D2__T1','drug_id':'D2','target_id':'T1'},{'pair_id':'OUT__T1','drug_id':'OUT','target_id':'T1'}]}})
    spr=SimpleNamespace(results=lambda **kw: {'items':[{'drug_id':'D2','target_id':'T2','result':'detected','KD':1,'qc':'pass','review_status':'pending'}]})
    atlas=AffinityAtlas(tmp_path);m=atlas.matrix(data,spr)
    rows={(r[0],r[1]):r for r in m['cells']}
    assert rows[0,0][3:5]==[1,1]
    assert rows[1,0][2:]==[0,0,0,0,1,0]  # excluded source kept out; design still pending
    assert rows[1,1][3:6]==[0,0,0]  # IC50 or unreviewed uploaded KD is not binding verdict
    assert rows[1,1][7]==1
    assert m['outside_planned_pairs']==1
    assert m['inventory']['exact_pairs']==2
    assert m['counts']['conflicting']==1
    with pytest.raises(ValueError):atlas.matrix(data,spr,threshold=float('nan'))
    with pytest.raises(ValueError):atlas.matrix(data,spr,endpoint='IC50')
