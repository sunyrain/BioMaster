"""Compare real predictions on the frozen common panel; never invent missing scores.

Each model CSV: graph_drug_id,disease_id,score. Exactly the shared full panel.
This is a retrospective indication retrieval comparison, not prospective accuracy.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

BASE=Path(__file__).resolve().parents[1]/'outputs/txgnn_biopathnet_comparison_20260909'

def metrics(scores,labels):
 if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
 if not (0<labels.sum()<len(labels)):raise ValueError('Need positives and background')
 # Average tie ranks, avoiding favorable arbitrary ordering.
 ranks=pd.Series(scores).rank(ascending=False,method='average').to_numpy()
 positive_ranks=ranks[labels==1]
 return {'ap_graph_label_vs_unlabelled':float(average_precision_score(labels,scores)),
  'auroc_graph_label_vs_unlabelled':float(roc_auc_score(labels,scores)),
  'mean_reciprocal_positive_rank':float(np.mean(1/positive_ranks)),
  'recall_at_20':float(np.mean(positive_ranks<=20)),'recall_at_50':float(np.mean(positive_ranks<=50))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--txgnn',type=Path,required=True);p.add_argument('--biopathnet',type=Path,required=True)
 p.add_argument('--training-provenance',type=Path,required=True,help='JSON declaring independently audited same-split checkpoint provenance')
 p.add_argument('--out',type=Path,default=BASE/'MODEL_COMPARISON.json');a=p.parse_args()
 import hashlib
 digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
 provenance=json.loads(a.training_provenance.read_text())
 for model in ['txgnn','biopathnet']:
  entry=provenance[model]
  if entry.get('evaluation_protocol')!='shared_pair_grouped_seed42_from_scratch' or not entry.get('no_test_edges_in_pretraining'):
   raise ValueError(f'{model}: checkpoint provenance not eligible; Explorer or unrelated weights cannot be scored as held-out performance')
  if entry['shared_split_hashes']!={f.name:digest(f) for f in (BASE/'shared_split').glob('*.txt')}:raise ValueError('Split mismatch')
  checkpoint=Path(entry['checkpoint_path'])
  if digest(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('Checkpoint changed')
 mask=pd.read_csv(BASE/'COMMON_QUERY_AND_EXCLUSION_MASK.csv.gz',dtype={'id':str,'graph_drug_id':str}).rename(columns={'id':'disease_id'})
 keys=['graph_drug_id','disease_id'];outputs={};drugwise=[]
 labels=pd.read_csv(BASE/'PANEL_TEST_POSITIVES.csv',dtype={'x_id':str,'y_id':str})
 labels=labels[~labels.conflicting_therapeutic_contra]
 positive=set(zip(labels.x_id,labels.y_id));testdrugs=set(labels.x_id)
 for name,path in [('txgnn',a.txgnn),('biopathnet',a.biopathnet)]:
  s=pd.read_csv(path,dtype={'graph_drug_id':str,'disease_id':str})
  if s.duplicated(keys).any():raise ValueError('Duplicate scores')
  merged=mask.merge(s[keys+['score']],on=keys,how='outer',indicator=True,validate='one_to_one')
  if not merged._merge.eq('both').all() or not np.isfinite(merged.score).all():raise ValueError(f'{name}: common universe incomplete or changed')
  for drug,g in merged[merged.graph_drug_id.isin(testdrugs)].groupby('graph_drug_id'):
   y=np.array([(drug,d) in positive for d in g.disease_id],dtype=int)
   # Restore held-out indications for evaluation, exclude OTHER known relations.
   use=g.eligible_graph_novel.to_numpy()|(y==1)
   m=metrics(g.score.to_numpy()[use],y[use]);m.update(model=name,graph_drug_id=drug,positive_count=int(y.sum()),ranking_universe=int(use.sum()))
   drugwise.append(m)
  d=pd.DataFrame([r for r in drugwise if r['model']==name])
  outputs[name]=d.select_dtypes('number').drop(columns=['positive_count','ranking_universe']).mean().to_dict()
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps({'status':'MATCHED_RETROSPECTIVE_SINGLE_SPLIT','macro_over_drugs_with_test_indications':outputs,
  'evaluated_drugs':len(testdrugs),'test_indications':len(positive),'note':'Unknowns are not verified negatives; mapping coverage is reported separately. Does not reproduce paper zero-shot result.'},indent=2)+'\n')
 pd.DataFrame(drugwise).to_csv(a.out.with_suffix('.per_drug.csv'),index=False)

if __name__=='__main__':main()
