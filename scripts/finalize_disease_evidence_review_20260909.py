"""Apply identity holds to interpretation, create review workbook, verify actual outputs."""
import json,hashlib,gzip
from pathlib import Path
import pandas as pd,numpy as np,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,j):(OUT/n).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
def main():
 c=pd.read_csv(OUT/'DRUG_COVERAGE_720.csv');m=pd.read_csv(OUT/'ECKG_PROJECT_NODE_MATCHES.csv.gz');rows=[]
 for r in c.itertuples():
  x=m[m.project_key.eq('drug:'+r.ligand_inchikey)];a=set(x[x.mapping_rule.eq('TXGNN_NAME_IDENTITY')].ec_id);b=set(x[~x.mapping_rule.eq('TXGNN_NAME_IDENTITY')].ec_id)
  status='STRUCTURE_AND_GRAPH_IDENTIFIER_CONFLICT_HOLD' if a and b and not a&b else ('SAME_NORMALIZED_EC_NODE_NOT_STEREO_PROOF' if a&b else 'NOT_ESTABLISHED')
  rows.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'txgnn_inference_eligible':r.inference_eligible,'ec_graph_id_nodes':';'.join(sorted(a)),'ec_structure_id_nodes':';'.join(sorted(b)),'identity_concordance':status})
 identity=pd.DataFrame(rows);identity.to_csv(OUT/'TXGNN_EC_IDENTITY_CONCORDANCE.csv',index=False)
 holds=set(identity[identity.identity_concordance.eq('STRUCTURE_AND_GRAPH_IDENTIFIER_CONFLICT_HOLD')].ligand_inchikey);order=pd.read_csv(OUT/'TXGNN_SCORE_DRUG_ORDER.csv');mask=np.load(OUT/'TXGNN_EXCLUSION_MASK.npy');blocked=order.ligand_inchikey.isin(holds).values;mask[blocked]|=4;np.save(OUT/'TXGNN_EXCLUSION_MASK.npy',mask)
 for name in ['DRUG_ALL_DISEASE_TOP30.csv.gz','DRUG_EVERY_DIRECTION_TOP3.csv.gz']:
  d=pd.read_csv(OUT/name);d=d[~d.ligand_inchikey.isin(holds)]
  if 'novelty_scope' in d:d['novelty_scope']='EXCLUDES_FROZEN_GRAPH_LABELS_AND_CONSERVATIVE_ECKG_TREAT_TRIAL_CONTRA_FLAGS_NOT_COMPLETE_CURRENT_INDICATION_REVIEW'
  d.to_csv(OUT/name,index=False)
 pairs=pd.read_parquet(OUT/'PAIR_COVERAGE_720X888.parquet');bad=pairs.ligand_inchikey.isin(holds);pairs['identity_review_status']=pairs.ligand_inchikey.map(identity.set_index('ligand_inchikey').identity_concordance);pairs.loc[bad,'top50_graph_novel_diseases_with_ot_evidence']=-1;pairs.loc[bad,'txgnn_status']='RAW_SCORED_IDENTITY_CONFLICT_HOLD';pairs.to_parquet(OUT/'PAIR_COVERAGE_720X888.parquet',index=False)
 spr=pd.read_csv(OUT/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv');assert not spr.ligand_inchikey.isin(holds).any(),'New holds affect SPR: review before finalization'
 c=c.merge(identity[['ligand_inchikey','identity_concordance']],on='ligand_inchikey',validate='one_to_one');c['interpretation_eligible']=c.inference_eligible & ~c.ligand_inchikey.isin(holds)
 mcounts=m.groupby('project_key').ec_id.nunique();c['ec_node_count']=c.ligand_inchikey.map(lambda x:int(mcounts.get('drug:'+x,0)));c.to_csv(OUT/'DRUG_COVERAGE_FINAL_720.csv',index=False)
 summary=json.loads((OUT/'ASSEMBLY_SUMMARY.json').read_text());summary.update({'raw_scored_drugs':len(order),'identity_conflict_hold_drugs':len(holds),'interpretation_eligible_drugs':int(c.interpretation_eligible.sum()),'identity_hold_query_cells':int(((mask&4)>0).sum()),'total_masked_query_cells':int((mask>0).sum()),'ec_unique_node_mapped_drugs':int(c.ec_node_count.eq(1).sum()),'ec_multi_node_mapped_drugs':int(c.ec_node_count.gt(1).sum()),'ec_only_drugs_without_txgnn_scores':int((c.ec_node_count.gt(0)&~c.inference_eligible).sum()),'ec_unique_only_drugs_without_txgnn_scores':int((c.ec_node_count.eq(1)&~c.inference_eligible).sum())});dump('FINAL_SUMMARY.json',summary)
 edges=pd.read_parquet(OUT/'ECKG_RELEVANT_RELATION_AUDIT.parquet')
 source_text=edges.primary_sources.fillna('').str.contains('semmeddb|text-mining|text_mining',case=False,regex=True)
 edges.loc[source_text,'evidence_class']='TEXT_MINING_OR_PREDICTION'
 uncertain=edges.evidence_class.eq('OTHER_SOURCE_ASSERTION_REVIEW') & (~edges.knowledge_level.eq('knowledge_assertion') | edges.primary_sources.fillna('').eq(''))
 edges.loc[uncertain,'evidence_class']='INFERRED_OR_UNSPECIFIED_HOLD'
 edges.to_parquet(OUT/'ECKG_RELEVANT_RELATION_AUDIT.parquet',index=False)
 groups=edges.groupby(['relation_class','increment_class','evidence_class']).size().reset_index(name='unique_evidence_rows');groups.to_csv(OUT/'ECKG_INCREMENT_COUNTS.csv',index=False)
 effective=edges[edges.increment_class.eq('NEW_ENDPOINT_PAIR_ON_COMPARABLE_NODES') & edges.evidence_class.eq('OTHER_SOURCE_ASSERTION_REVIEW') & ~edges.mapping_ambiguous & ~edges.project_key.isin({'drug:'+x for x in holds})]
 effective[['project_key','other_id','relation_class']].drop_duplicates().to_csv(OUT/'ECKG_NEW_COMPARABLE_ASSERTION_ENDPOINT_PAIRS.csv.gz',index=False)
 inc=json.loads((OUT/'ECKG_INCREMENT_SUMMARY.json').read_text());inc['new_non_text_comparable_pairs']=len(effective[['project_key','other_id','relation_class']].drop_duplicates());inc['scope']='Non-text source assertions, unique project node mapping, no observed identity conflict; not independently verified biological novelty or efficacy.';dump('ECKG_INCREMENT_SUMMARY.json',inc)
 tc=pd.read_csv(OUT/'TARGET_COVERAGE_888.csv');otcov=pd.read_csv(OUT/'OT_TARGET_COVERAGE_888.csv');ds=pd.read_csv(OUT/'TXGNN_ALL_DISEASE_DIRECTIONS.csv')
 dirs=pd.DataFrame([{'direction':k.removeprefix('speciality_'),'disease_nodes':v} for k,v in summary['direction_disease_node_counts'].items()])
 st=spr.groupby('gene_symbol').agg(pair_rows=('gene_symbol','size'),txgnn_mapped_rows=('txgnn_available','sum'),pairs_with_top50_ot_overlap=('top50_ot_overlap',lambda x:int(x.fillna(0).gt(0).sum()))).reset_index();st.to_csv(OUT/'SPR64_DISEASE_COVERAGE_SUMMARY.csv',index=False)
 with pd.ExcelWriter(OUT/'DISEASE_EVIDENCE_REVIEW_720_888.xlsx',engine='openpyxl') as w:
  pd.DataFrame([{'metric':k,'value':str(v)} for k,v in summary.items() if not isinstance(v,dict)]).to_excel(w,sheet_name='Summary',index=False)
  c.to_excel(w,sheet_name='Drug720',index=False);tc.to_excel(w,sheet_name='Target888',index=False);otcov.to_excel(w,sheet_name='OTPagination',index=False);dirs.to_excel(w,sheet_name='Directions',index=False);st.to_excel(w,sheet_name='SPR64',index=False);spr.to_excel(w,sheet_name='SPR512',index=False)
  pd.read_csv(OUT/'ECKG_INCREMENT_COUNTS.csv').to_excel(w,sheet_name='ECIncrement',index=False)
  pd.read_csv(OUT/'TARGET_DISEASE_SUMMARY.csv').to_excel(w,sheet_name='TargetDiseaseTop5',index=False)
  pd.read_csv(OUT/'DRUG_EVERY_DIRECTION_TOP3.csv.gz').to_excel(w,sheet_name='EveryDirectionTop3',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 historical=pd.read_csv(ROOT/'data/processed/txgnn_drug_disease_scores.csv').drop_duplicates('txgnn_drugbank_id')
 disease_order=pd.read_csv(OUT/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str});ci=disease_order.index[disease_order.id.eq('4992')][0]
 reproduced=order.copy();reproduced['current_logit']=np.load(OUT/'TXGNN_INDICATION_LOGITS.npy')[:,ci];reproduced=reproduced.merge(historical,left_on='graph_drug_id',right_on='txgnn_drugbank_id')
 reproduced[['ligand_inchikey','graph_drug_id','current_logit','txgnn_indication_logit']].to_csv(OUT/'HISTORICAL_CANCER_REPRODUCTION.csv',index=False)
 dump('HISTORICAL_CANCER_REPRODUCTION.json',{'shared_drug_nodes':len(reproduced),'max_abs_logit_difference':float((reproduced.current_logit-reproduced.txgnn_indication_logit).abs().max()),'pearson_correlation':reproduced.current_logit.corr(reproduced.txgnn_indication_logit)})
 checks={}
 checks['all720_all888_exact_cartesian_bookkeeping']=len(pairs)==720*888 and not pairs.duplicated(['ligand_inchikey','target_chembl_id']).any()
 checks['full_score_matrix_finite']=np.isfinite(np.load(OUT/'TXGNN_INDICATION_LOGITS.npy')).all().item()
 checks['score_shape_matches_orders']=np.load(OUT/'TXGNN_INDICATION_LOGITS.npy').shape==(len(order),len(ds))
 checks['native_decoder_agreement']=json.loads((OUT/'TXGNN_DECODER_VERIFICATION.json').read_text())['max_abs_error']<1e-5
 checks['historical_cancer_reproduced']=json.loads((OUT/'HISTORICAL_CANCER_REPRODUCTION.json').read_text())['max_abs_logit_difference']<1e-5
 meta0=json.loads((OUT/'OT_META_START.json').read_text());meta1=json.loads((OUT/'OT_META_END.json').read_text());checks['ot_version_stable']=meta0==meta1
 good=otcov[otcov.status.str.startswith('COMPLETE')];ok=True
 for r in good.itertuples():
  with gzip.open(OUT/'ot_cache'/f'{r.ot_id}.json.gz','rt') as f:j=json.load(f)
  ok &=len(j['rows'])==j['count']==len({x['disease']['id'] for x in j['rows']})
 checks['ot_complete_counts_and_unique_diseases']=bool(ok)
 checks['identity_holds_absent_from_top_tables']=all(not pd.read_csv(OUT/n).ligand_inchikey.isin(holds).any() for n in ['DRUG_ALL_DISEASE_TOP30.csv.gz','DRUG_EVERY_DIRECTION_TOP3.csv.gz'])
 ref=json.loads((ROOT/'outputs/txgnn_biopathnet_comparison_20260909/MANIFEST.json').read_text())['inputs'];checks['model_graph_and_spr_inputs_unchanged']=all(sha(ROOT/k)==v for k,v in ref.items() if k.startswith('data/raw/txgnn/') or k.endswith('INTERNAL_MASTER_512.csv'))
 man=json.loads((OUT/'eckg/SOURCE_MANIFEST.json').read_text());checks['all_ec_shards_checksum_verified']=all(sha(OUT/'eckg'/n/Path(f['path']).name)==f['lfs']['oid'] for n,info in man.items() for f in info['files'])
 scan=json.loads((OUT/'ECKG_SCAN_SUMMARY.json').read_text());checks['all_ec_graph_shards_scanned']=scan['edge_shards_scanned']==49 and scan['node_shards_scanned']==8
 oldcols={str(x):i for i,x in enumerate(pd.read_csv(OUT/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str}).id)};drugidx={x:i for i,x in enumerate(order.ligand_inchikey)};top=pd.read_csv(OUT/'DRUG_ALL_DISEASE_TOP30.csv.gz',dtype={'txgnn_disease_id':str})
 checks['shortlists_exclude_known_and_identity_holds']=all(mask[drugidx[r.ligand_inchikey],oldcols[r.txgnn_disease_id]]==0 for r in top.itertuples())
 dump('VALIDATION.json',{'checks':checks,'all_pass':all(checks.values()),'performance_validated':False,'no_training':True});assert all(checks.values()),checks
 print(json.dumps(summary,ensure_ascii=False,indent=2));print('ALL CHECKS PASS')
if __name__=='__main__':main()
