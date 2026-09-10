"""Broad-disease-area audit, explicitly separating label-reviewed drugs from KG proxies."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909';JOINT=ROOT/'outputs/joint_screen_720x384_20260909';PREV=ROOT/'outputs/joint384_design_20260909';OUT=ROOT/'outputs/joint_cross_indication_20260909'

def split(x):return set(str(x).split(';')) if pd.notna(x) and str(x) else set()
def macro(flags):
 f=set(flags)
 # Cancer identity takes priority over organ site: lung/breast/prostate remain oncology.
 for key,out in [('neoplasm','ONCOLOGY'),('infection','INFECTION'),('immune','IMMUNE_INFLAMMATORY')]:
  if key in f:return {out}
 groups={'ENDOCRINE_METABOLIC_REPRODUCTIVE':{'endocrine','metabolic','reproductive','obstetric','breast'},'NEURO_PSYCHIATRIC':{'neurological','psychiatric'},'MUSCULOSKELETAL_CONNECTIVE':{'musculoskeletal','connective_tissue'},'CARDIOVASCULAR':{'cardiovascular'},'RESPIRATORY':{'respiratory'},'GASTROINTESTINAL':{'gastrointestinal'},'DERMATOLOGIC':{'dermatologic'},'RENAL_URINARY':{'renal_and_urinary'},'HEMATOLOGIC':{'hematologic'},'EYE_ENT':{'eye_and_adnexa','ear_nose_throat'},'TOXICITY':{'poisoning_and_toxicity'}}
 return {k for k,v in groups.items() if f&v}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 reviewed=pd.read_csv(PREV/'ALL_574_AGENT_REVIEW.csv');selected=pd.read_csv(PREV/'CANDIDATES_384.csv')
 detail=pd.read_csv(JOINT/'JOINT_DISEASE_EVIDENCE.csv.gz',dtype={'txgnn_disease_id':str});detail=detail[detail.pair_id.isin(reviewed.pair_id)].copy()
 ds=pd.read_csv(BASE/'TXGNN_ALL_DISEASE_DIRECTIONS.csv',dtype={'id':str});dm={r.id:macro(split(r.direction_labels)) for r in ds.itertuples()}
 ecpath=next((BASE/'eckg/disease-list').glob('*.parquet'));ec=pd.read_parquet(ecpath);spec=[c for c in ec if c.startswith('speciality_')]
 edm={r['id'].replace('MONDO:','MONDO_'):macro([c.removeprefix('speciality_') for c in spec if r[c]]) for r in ec.to_dict('records')}
 labels_path=ROOT/'outputs/txgnn_biopathnet_comparison_20260909/ALL_DRUG_DISEASE_LABELS_WITH_SPLIT.csv.gz'
 labels=pd.read_csv(labels_path,dtype={'x_id':str,'y_id':str});labels=labels[labels.relation.isin(['indication','off-label use'])].drop_duplicates(['x_id','relation','y_id'])
 labels['macro_areas']=labels.y_id.map(lambda x:';'.join(sorted(dm.get(x,set()))));labels['disease_name']=labels.y_id.map(ds.set_index('id').node_name)
 labels=labels[labels.x_id.isin(reviewed.graph_drug_id)];labels.to_csv(OUT/'KG_PRIOR_USE_LABELS_NOT_APPROVALS.csv.gz',index=False)
 baselines={d:set().union(*(split(x) for x in g.macro_areas)) for d,g in labels.groupby('x_id')}
 # Only these five drugs have a focused official-indication review this run.
 verified={'tecovirimat':{'INFECTION'},'mebendazole':{'INFECTION'},'vismodegib':{'ONCOLOGY'},'raloxifene':{'ONCOLOGY','ENDOCRINE_METABOLIC_REPRODUCTIVE','MUSCULOSKELETAL_CONNECTIVE'},'bazedoxifene':{'ENDOCRINE_METABOLIC_REPRODUCTIVE','MUSCULOSKELETAL_CONNECTIVE'}}
 proof=pd.read_csv(OUT/'PRIORITY_DRUG_INDICATION_REVIEW.csv').set_index('drug_names');vis=json.loads((OUT/'VISMODEGIB_CROSS_INDICATION_REVIEW.json').read_text())
 registry=[]
 for r in reviewed.drop_duplicates('ligand_inchikey').itertuples():
  areas=verified.get(r.drug_names,baselines.get(r.graph_drug_id,set()))
  registry.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'graph_drug_id':r.graph_drug_id,'baseline_areas':';'.join(sorted(areas)),'baseline_evidence_scope':'FOCUSED_OFFICIAL_INDICATION_REVIEW' if r.drug_names in verified else 'KG_PRIOR_USE_PROXY_NOT_APPROVED_INDICATION_REGISTRY','baseline_officially_reviewed':r.drug_names in verified,'kg_prior_use_areas':';'.join(sorted(baselines.get(r.graph_drug_id,set()))),'official_indication_sources':';'.join(vis['sources']) if r.drug_names=='vismodegib' else proof.loc[r.drug_names,'sources'] if r.drug_names in proof.index else ''})
 registry=pd.DataFrame(registry);registry.to_csv(OUT/'DRUG_ORIGINAL_AREA_AUDIT.csv',index=False)
 detail=detail.merge(reviewed[['pair_id','ligand_inchikey','decision','priority_after_agent_review','positive_chemical_support_ge04']],on='pair_id',validate='many_to_one').merge(registry.drop(columns=['drug_names','graph_drug_id']),on='ligand_inchikey',validate='many_to_one')
 rows=[]
 for r in detail.to_dict('records'):
  before=split(r['baseline_areas']);after=edm.get(r['disease_id'],set());r['candidate_macro_areas']=';'.join(sorted(after));r['same_or_overlapping_area']=bool(before&after)
  if not before or not after:state='ORIGIN_OR_CANDIDATE_AREA_UNKNOWN'
  elif before&after:state='SAME_OR_ADJACENT_MAJOR_AREA'
  elif r['txgnn_mapping_scope']!='EXACT_SINGLE_NODE':state='CROSS_AREA_BUT_MERGED_DISEASE_HOLD'
  elif r['baseline_officially_reviewed']:state='CROSS_MAJOR_AREA_OFFICIAL_ORIGIN_REVIEWED'
  else:state='CROSS_AREA_PROXY_OFFICIAL_INDICATION_REVIEW_PENDING'
  r['cross_indication_status']=state
  # These clinical novelty/direction holds are not claims about molecular binding.
  r['disease_investment_hold']=r['drug_names'] in ['raloxifene','bazedoxifene'] and ('endometriosis' in r['disease_name'].lower())
  r['cross_priority_eligible']=state=='CROSS_MAJOR_AREA_OFFICIAL_ORIGIN_REVIEWED' and r['decision']=='KEEP_REVIEW' and r['priority_after_agent_review'] and r['same_disease_exact_genetic'] and r['positive_chemical_support_ge04'] and not r['disease_investment_hold']
  rows.append(r)
 h=pd.DataFrame(rows);h.to_csv(OUT/'ALL_574_COMMON_DISEASE_CROSS_AREA_AUDIT.csv.gz',index=False)
 pairrows=[]
 for r in reviewed.itertuples():
  g=h[h.pair_id.eq(r.pair_id)];cross=g[g.cross_indication_status.isin(['CROSS_MAJOR_AREA_OFFICIAL_ORIGIN_REVIEWED','CROSS_AREA_PROXY_OFFICIAL_INDICATION_REVIEW_PENDING'])&~g.disease_investment_hold]
  pairrows.append({'pair_id':r.pair_id,'drug_names':r.drug_names,'gene_symbol':r.gene_symbol,'agent_decision':r.decision,'in_previous384':r.pair_id in set(selected.pair_id),'original_priority':bool(r.priority_after_agent_review),'common_disease_count':len(g),'cross_area_exact_hypotheses':len(cross),'approved_origin_cross_hypotheses':int(cross.baseline_officially_reviewed.sum()),'cross_priority_eligible':bool(g.cross_priority_eligible.any()),'statuses':';'.join(sorted(set(g.cross_indication_status))),'has_same_area':bool(g.same_or_overlapping_area.any()),'has_disease_investment_hold':bool(g.disease_investment_hold.any())})
 pairs=pd.DataFrame(pairrows);pairs.to_csv(OUT/'PAIR_REVIEW_574.csv',index=False)
 provisional=pairs[pairs.agent_decision.eq('KEEP_REVIEW')&pairs.cross_area_exact_hypotheses.gt(0)];provisional.to_csv(OUT/'CROSS_AREA_LABEL_REVIEW_QUEUE.csv',index=False)
 best=h[h.cross_priority_eligible].sort_values(['pair_id','txgnn_rank','overall_score'],ascending=[True,True,False]).drop_duplicates('pair_id');best.to_csv(OUT/'PRIORITY_CROSS_MAJOR_AREA_REVIEW.csv',index=False)
 five=pairs[pairs.original_priority];five.to_csv(OUT/'PREVIOUS5_REAUDIT.csv',index=False)
 summary={'reviewed_pairs':len(pairs),'reviewed_drugs':len(registry),'official_origin_reviewed_drugs':int(registry.baseline_officially_reviewed.sum()),'non_low_agent_pairs':int(pairs.agent_decision.eq('KEEP_REVIEW').sum()),'non_low_with_exact_cross_area_proxy_or_verified':len(provisional),'previous384_non_low_with_exact_cross_area_proxy_or_verified':int(provisional.in_previous384.sum()),'priority_cross_area_pairs':len(best),'priority_pair_names':(best.drug_names+'–'+best.gene_symbol).tolist(),'prior5_same_or_adjacent_or_clinically_preexplored':five.loc[~five.cross_priority_eligible,['drug_names','gene_symbol']].to_dict('records'),'complete_approved_indication_registry':False,'new384_created':False}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
 with pd.ExcelWriter(OUT/'CROSS_INDICATION_REVIEW.xlsx') as w:
  best[['drug_names','gene_symbol','baseline_areas','disease_name','candidate_macro_areas','txgnn_rank','overall_score','genetic_association_score','somatic_mutation_score','official_indication_sources']].to_excel(w,sheet_name='跨大类优先复核',index=False);five.to_excel(w,sheet_name='原5项复审',index=False);provisional.to_excel(w,sheet_name='原适应症待核队列',index=False);pairs.to_excel(w,sheet_name='574配对复审',index=False);registry.to_excel(w,sheet_name='药物原用途证据范围',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 checks={'all574_accounted':len(pairs)==574 and pairs.pair_id.is_unique,'no_low_hold_exclude_in_priority':best.decision.eq('KEEP_REVIEW').all(),'priority_official_origin_reviewed':best.baseline_officially_reviewed.all(),'priority_exact_disease':best.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE').all(),'priority_no_shared_major_area':all(not(split(x.baseline_areas)&split(x.candidate_macro_areas)) for x in best.itertuples()),'vismodegib_NTRK1_not_cross':not pairs.loc[pairs.drug_names.eq('vismodegib')&pairs.gene_symbol.eq('NTRK1'),'cross_priority_eligible'].any(),'no_cancer_to_cancer_cross':not h.loc[h.baseline_areas.str.contains('ONCOLOGY')&h.candidate_macro_areas.str.contains('ONCOLOGY'),'cross_priority_eligible'].any()}
 checks={k:bool(v) for k,v in checks.items()};(OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks},indent=2));assert all(checks.values())
 files=[PREV/'ALL_574_AGENT_REVIEW.csv',PREV/'CANDIDATES_384.csv',JOINT/'JOINT_DISEASE_EVIDENCE.csv.gz',BASE/'TXGNN_ALL_DISEASE_DIRECTIONS.csv',labels_path,ecpath,OUT/'PRIORITY_DRUG_INDICATION_REVIEW.csv',OUT/'VISMODEGIB_CROSS_INDICATION_REVIEW.json',Path(__file__)]
 (OUT/'MANIFEST.json').write_text(json.dumps({'input_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files},'output_sha256':{f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='MANIFEST.json'}},indent=2))
 print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
