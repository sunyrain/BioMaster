"""Recompute all SPR drug-disease x target-disease joins and descriptive controls.
No retraining; evaluates evidence concordance, not measured binding performance.
"""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909';RUN=ROOT/'outputs/spr512_integrated_disease_20260909';OUT=ROOT/'outputs/spr512_txgnn_ot_effect_20260909'

def main():
 OUT.mkdir(exist_ok=True)
 master=pd.read_csv(RUN/'INTEGRATED_MASTER_512.csv');ot=pd.read_parquet(RUN/'OPENTARGETS_64_FULL_ASSOCIATIONS.parquet');ds=pd.read_csv(BASE/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str});order=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');a=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');mask=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy');extra=pd.read_csv(RUN/'TXGNN_SCORE_DRUG_ORDER.csv');b=np.load(RUN/'TXGNN_INDICATION_LOGITS.npy')
 values={k:a[i] for i,k in enumerate(order.ligand_inchikey)};values.update({k:b[i] for i,k in enumerate(extra.ligand_inchikey)});masks={k:mask[i]>0 for i,k in enumerate(order.ligand_inchikey)}
 candidates=master[master.selection_role.ne('POSITIVE_CONTROL')];drugs=sorted(candidates[candidates.txgnn_available].ligand_inchikey.unique());targets=sorted(master.target_chembl_id.unique());drugidx={k:i for i,k in enumerate(drugs)};targetidx={k:i for i,k in enumerate(targets)}
 ranks={}
 for r in master[master.txgnn_available].drop_duplicates(['ligand_inchikey','selection_role']).itertuples():
  key=(r.ligand_inchikey,r.selection_role=='POSITIVE_CONTROL')
  if key in ranks:continue
  excluded=np.zeros(len(ds),bool) if key[1] else masks[key[0]];ids=np.where(~excluded)[0];rank=np.full(len(ds),-1,int);rank[ids[np.argsort(-values[key[0]][ids],kind='stable')]]=np.arange(1,len(ids)+1);ranks[key]=rank
 conditions={'rank50_ot01':(50,.1),'rank50_ot03':(50,.3),'rank50_ot05':(50,.5),'rank100_ot03':(100,.3)}
 matrices={k:np.zeros((len(drugs),len(targets)),bool) for k in [*conditions,'rank50_ot03_exact','rank50_ot03_genetic']}
 mapped=ot[ot.txgnn_score_column>=0].copy();mapped['ci']=mapped.txgnn_score_column.astype(int);bytarget={t:g for t,g in mapped.groupby('target_chembl_id')};rows=[];detail=[]
 for r in master.itertuples():
  row={'experiment_id':r.experiment_id,'drug_names':r.drug_names,'gene_symbol':r.gene_symbol,'ligand_inchikey':r.ligand_inchikey,'target_chembl_id':r.target_chembl_id,'role':r.selection_role,'txgnn_available':bool(r.txgnn_available),'positive_max_tanimoto':r.positive_max_tanimoto,'prior_source_assertion_rows':r.non_text_unambiguous_assertion_rows}
  for k in matrices:row[k]=None
  if r.txgnn_available:
   rank=ranks[(r.ligand_inchikey,r.selection_role=='POSITIVE_CONTROL')];g=bytarget[r.target_chembl_id].copy();g['rank']=rank[g.ci];g=g[g['rank']>0]
   for k,(n,threshold) in conditions.items():row[k]=bool(((g['rank']<=n)&(g.overall_score>=threshold)).any())
   strong=g[(g['rank']<=50)&(g.overall_score>=.3)].sort_values(['rank','overall_score'],ascending=[True,False]);exact=strong[strong.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE')];gen=strong[(strong.genetic_association_score>=.1)|(strong.somatic_mutation_score>=.1)];row['rank50_ot03_exact']=len(exact)>0;row['rank50_ot03_genetic']=len(gen)>0
   for x in strong.to_dict('records'):
    detail.append({'experiment_id':r.experiment_id,'drug_names':r.drug_names,'gene_symbol':r.gene_symbol,'role':r.selection_role,'txgnn_disease_id':ds.iloc[x['ci']].id,'txgnn_disease_name':ds.iloc[x['ci']].node_name,'txgnn_rank':x['rank'],'txgnn_logit':float(values[r.ligand_inchikey][x['ci']]),'ot_disease_id':x['disease_id'],'ot_disease_name':x['disease_name'],'ot_score':x['overall_score'],'mapping_scope':x['txgnn_mapping_scope'],'genetic_association_score':x['genetic_association_score'],'somatic_mutation_score':x['somatic_mutation_score'],'clinical_score':x['clinical_score'],'datatype_scores_json':x['datatype_scores_json']})
   if len(strong):x=strong.iloc[0];row.update(best_joint_disease=x.disease_name,best_joint_rank=int(x['rank']),best_joint_ot=float(x.overall_score))
   # Find best-ranked intersecting disease at OT >= 0.3, including weak TxGNN ranks.
   eligible=g[g.overall_score>=.3].sort_values(['rank','overall_score'],ascending=[True,False])
   row['best_rank_with_ot03']=int(eligible.iloc[0]['rank']) if len(eligible) else None
  rows.append(row)
 # All mapped candidate drugs x all 64 targets for an exploratory reassignment baseline.
 for d,di in drugidx.items():
  rank=ranks[(d,False)]
  for t,ti in targetidx.items():
   g=bytarget[t];rr=rank[g.ci]
   for k,(n,threshold) in conditions.items():matrices[k][di,ti]=bool(((rr>0)&(rr<=n)&(g.overall_score>=threshold)).any())
   common=(rr>0)&(rr<=50)&(g.overall_score>=.3)
   matrices['rank50_ot03_exact'][di,ti]=bool((common&g.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE')).any());matrices['rank50_ot03_genetic'][di,ti]=bool((common&((g.genetic_association_score>=.1)|(g.somatic_mutation_score>=.1))).any())
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'PAIR_RESULTS_512.csv',index=False);details=pd.DataFrame(detail);details.to_csv(OUT/'JOINT_DISEASE_DETAILS.csv.gz',index=False)
 summary=[]
 for role,g in frame.groupby('role'):
  for name in matrices:
   count=int(g[name].fillna(False).sum());n=int(g.txgnn_available.sum());summary.append({'role':role,'criterion':name,'all_pairs':len(g),'mapped_pairs':n,'passing_pairs':count,'fraction_all_pairs':count/len(g),'fraction_mapped_pairs':count/n})
 summ=pd.DataFrame(summary);summ.to_csv(OUT/'GROUP_COMPARISON.csv',index=False)
 rng=np.random.default_rng(20260909);observed=frame[frame.txgnn_available & frame.role.ne('POSITIVE_CONTROL')];ii=observed.ligand_inchikey.map(drugidx).values;jj=observed.target_chembl_id.map(targetidx).values;roles=observed.role.values;null=[]
 for criterion in ['rank50_ot03','rank50_ot03_exact','rank50_ot03_genetic']:
  matrix=matrices[criterion];actual=matrix[ii,jj];perm=np.empty((2000,len(observed)),bool)
  for k in range(2000):perm[k]=matrix[rng.permutation(len(drugs))[ii],jj]
  for role in sorted(set(roles)):
   idx=roles==role;rate=perm[:,idx].mean(axis=1);value=float(actual[idx].mean());null.append({'criterion':criterion,'role':role,'observed_rate_mapped':value,'reassigned_drug_mean':float(rate.mean()),'null_2_5_percentile':float(np.quantile(rate,.025)),'null_97_5_percentile':float(np.quantile(rate,.975)),'exploratory_upper_tail_fraction':float((1+(rate>=value).sum())/2001)})
 pd.DataFrame(null).to_csv(OUT/'DRUG_REASSIGNMENT_BASELINE.csv',index=False)
 with pd.ExcelWriter(OUT/'SPR512_TXGNN_OT_EFFECT_REVIEW.xlsx') as w:
  frame.to_excel(w,sheet_name='512配对联合结果',index=False);summ.to_excel(w,sheet_name='高中低组对比',index=False);pd.DataFrame(null).to_excel(w,sheet_name='重配对探索基线',index=False);details.to_excel(w,sheet_name='联合疾病证据',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 assert len(frame)==512 and frame.experiment_id.is_unique
 assert frame[~frame.txgnn_available][list(matrices)].isna().all().all()
 assert all((g[g.criterion.eq('rank50_ot05')].passing_pairs.iloc[0]<=g[g.criterion.eq('rank50_ot03')].passing_pairs.iloc[0]<=g[g.criterion.eq('rank50_ot01')].passing_pairs.iloc[0]) for _,g in summ.groupby('role'))
 sources=[RUN/'INTEGRATED_MASTER_512.csv',RUN/'OPENTARGETS_64_FULL_ASSOCIATIONS.parquet',BASE/'TXGNN_INDICATION_LOGITS.npy',BASE/'TXGNN_EXCLUSION_MASK.npy',RUN/'TXGNN_INDICATION_LOGITS.npy']
 (OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':True,'rows':len(frame),'mapped_candidates':len(observed),'no_training':True,'binding_performance_measured':False,'permutations':2000,'permutation_unit':'drug profile; repeated drug rows move together; targets and role assignments fixed','baseline_limitation':'Reassignment does not reapply DTI novelty/chemical/assay selection; not a matched biological negative control','sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
 print(summ.to_string(index=False));print(pd.DataFrame(null).to_string(index=False))
if __name__=='__main__':main()
