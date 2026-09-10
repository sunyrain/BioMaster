"""Expose evidence gaps for graph-outside drugs and prior-relation flags for SPR."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def main():
 top=pd.read_csv(OUT/'DRUG_ALL_DISEASE_TOP30.csv.gz');top['novelty_scope']='EXCLUDES_FROZEN_GRAPH_LABELS_AND_CONSERVATIVE_ECKG_TREAT_TRIAL_CONTRA_FLAGS_NOT_COMPLETE_CURRENT_INDICATION_REVIEW';top.to_csv(OUT/'DRUG_ALL_DISEASE_TOP30.csv.gz',index=False)
 e=pd.read_parquet(OUT/'ECKG_RELEVANT_RELATION_AUDIT.parquet');c=pd.read_csv(OUT/'DRUG_COVERAGE_FINAL_720.csv');d=e[e.relation_class.eq('DRUG_DISEASE')];counts=d.groupby('project_key').other_id.nunique();a=d[d.evidence_class.eq('OTHER_SOURCE_ASSERTION_REVIEW')&~d.mapping_ambiguous];ac=a.groupby('project_key').other_id.nunique()
 c['ec_disease_endpoint_count']=c.ligand_inchikey.map(lambda x:int(counts.get('drug:'+x,0)));c['ec_non_text_asserted_disease_endpoint_count']=c.ligand_inchikey.map(lambda x:int(ac.get('drug:'+x,0)))
 c['disease_evidence_status']=['TXGNN_QUERY_AND_SOURCE_EVIDENCE_REVIEW' if r.interpretation_eligible else ('SOURCE_EVIDENCE_ONLY_NO_USABLE_TXGNN_SCORE' if r.ec_disease_endpoint_count else 'IDENTIFIED_ENTITY_NO_DISEASE_EVIDENCE_IN_THIS_RUN') for r in c.itertuples()];c.to_csv(OUT/'DRUG_COVERAGE_FINAL_720.csv',index=False)
 c[~c.interpretation_eligible].to_csv(OUT/'DRUGS_WITHOUT_USABLE_TXGNN_SCORES.csv',index=False)
 spr=pd.read_csv(OUT/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv');dt=e[e.relation_class.eq('DRUG_TARGET')];rows=[]
 for r in spr.itertuples():
  x=dt[dt.project_key.eq('drug:'+r.ligand_inchikey)&dt.other_id.eq('target:'+r.target_chembl_id)];a=x[x.evidence_class.eq('OTHER_SOURCE_ASSERTION_REVIEW')&~x.mapping_ambiguous]
  rows.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'selection_role':r.selection_role,'ec_relation_rows':len(x),'non_text_unambiguous_assertion_rows':len(a),'predicates':';'.join(sorted(set(a.predicate))),'primary_sources':';'.join(sorted(set(a.primary_sources))),'review_status':'REVIEW_EXISTING_SOURCE_ASSERTIONS_NOT_AUTOMATIC_BINDING_CONFIRMATION' if len(a) else ('TEXT_OR_MAPPING_REVIEW_ONLY' if len(x) else 'NO_ECKG_RELATION_MATCH')})
 r=pd.DataFrame(rows);r.to_csv(OUT/'SPR512_ECKG_RELATION_REVIEW.csv',index=False)
 with pd.ExcelWriter(OUT/'DISEASE_EVIDENCE_REVIEW_720_888.xlsx',engine='openpyxl',mode='a',if_sheet_exists='replace') as w:
  c.to_excel(w,sheet_name='Drug720',index=False);r.to_excel(w,sheet_name='SPRRelationReview',index=False)
  for name in ['Drug720','SPRRelationReview']:w.book[name].freeze_panes='A2';w.book[name].auto_filter.ref=w.book[name].dimensions
 s=json.loads((OUT/'FINAL_SUMMARY.json').read_text());s.update({'ec_drugs_with_disease_relations':int(c.ec_disease_endpoint_count.gt(0).sum()),'txgnn_unmapped_drugs_with_ec_disease_relations':int((~c.inference_eligible & c.ec_disease_endpoint_count.gt(0)).sum()),'drug_disease_evidence_status_counts':c.disease_evidence_status.value_counts().to_dict(),'spr_candidate_prior_non_text_assertion_pairs':int((r.selection_role.ne('POSITIVE_CONTROL') & r.non_text_unambiguous_assertion_rows.gt(0)).sum())});(OUT/'FINAL_SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
 print(s['drug_disease_evidence_status_counts'])
if __name__=='__main__':main()
