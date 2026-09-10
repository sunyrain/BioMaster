"""Independent file and exclusion checks; no trained predictions are generated."""
import json,hashlib,importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/txgnn_biopathnet_comparison_20260909'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((OUT/'MANIFEST.json').read_text())
checks={'inputs_unchanged':all(sha(ROOT/p)==h for p,h in manifest['inputs'].items()),
 'outputs_unchanged':all(sha(OUT/p)==h for p,h in manifest['files'].items())}
c=pd.read_csv(OUT/'DRUG_COVERAGE_120.csv');q=pd.read_csv(OUT/'COMMON_QUERY_AND_EXCLUSION_MASK.csv.gz',dtype={'id':str});d=pd.read_csv(OUT/'COMMON_DISEASE_UNIVERSE.csv',dtype={'id':str})
checks['full_cartesian_query_panel']=len(q)==int(c.txgnn_graph_covered.sum())*len(d) and not q.duplicated(['graph_drug_id','id']).any()
labels=pd.read_csv(OUT/'ALL_DRUG_DISEASE_LABELS_WITH_SPLIT.csv.gz',dtype={'x_id':str,'y_id':str})
known=set(zip(labels.x_id,labels.y_id))
expected=np.array([(x,y) not in known for x,y in zip(q.graph_drug_id,q.id)])
checks['exact_snapshot_exclusion_mask']=np.array_equal(expected,q.eligible_graph_novel.to_numpy())
folds={}
for s,f in [('train','train2.txt'),('valid','valid.txt'),('test','test.txt')]:
 a=pd.read_csv(OUT/'shared_split'/f,sep='\t',header=None,names=['head','relation','tail'],dtype=str)
 folds[s]=set(zip(a['head'],a['tail']))
checks['exported_endpoint_groups_disjoint']=all(not folds[a]&folds[b] for a,b in [('train','valid'),('train','test'),('valid','test')])
e=pd.read_csv(OUT/'SHARED_TEST_SAMPLED_LABELS.csv',dtype={'disease_id':str})
checks['unlabelled_controls_not_known_graph_pairs']=all((r.graph_drug_id,r.disease_id) not in known for r in e[e.label.eq(0)].itertuples())
checks['all_sampled_positives_are_test_indications']=all(((r.graph_drug_id,r.disease_id) in set(zip(labels[labels.split.eq('test')&labels.relation.eq('indication')].x_id,labels[labels.split.eq('test')&labels.relation.eq('indication')].y_id))) for r in e[e.label.eq(1)].itertuples())
checks['shared_100_background_per_query']=bool(e.groupby('query_key').size().eq(101).all())
spec=importlib.util.spec_from_file_location('evaluation',ROOT/'scripts/evaluate_txgnn_biopathnet_comparison_20260909.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks['metric_perfect_and_reverse']=m.metrics(np.array([2.,1.]),np.array([1,0]))['ap_graph_label_vs_unlabelled']==1 and m.metrics(np.array([1.,2.]),np.array([1,0]))['ap_graph_label_vs_unlabelled']==.5
checks['metric_ties_neutral']=m.metrics(np.ones(2),np.array([1,0]))['auroc_graph_label_vs_unlabelled']==.5
try:m.metrics(np.array([np.nan,1.]),np.array([1,0]));checks['nonfinite_rejected']=False
except ValueError:checks['nonfinite_rejected']=True
(OUT/'INDEPENDENT_VALIDATION.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2));assert all(checks.values())
