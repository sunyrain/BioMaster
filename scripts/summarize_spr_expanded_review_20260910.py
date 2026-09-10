"""Summarize baseline reason reassessment and additional evidence review, without selecting experiments."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_spr_expansion import DIR, baseline_reasons, expanded_items
from biomaster.explorer_spr_reviews import load_spr_reviews, BASELINE_PATH
p=ROOT/DIR;baseline=ROOT/BASELINE_PATH;old=list(csv.DictReader(baseline.open()));reviews=load_spr_reviews(ROOT,baseline,old);reasons=baseline_reasons(ROOT);new=expanded_items(ROOT)
deferred=[reasons[r['pair_id']] for r in old if reviews[r['pair_id']]['verdict']=='DEFER' and r['pair_id'] in reasons]
summary={'baseline_candidates':384,'baseline_reason_reviewed':len(reasons),'baseline_action_counts':dict(Counter(r['action_class'] for r in reasons.values())),
 'old_defer_count':275,'old_defer_reclassified':dict(Counter(r['action_class'] for r in deferred)),
 'expanded_queue':len(new),'expanded_reviewed':sum('llm_review' in r for r in new),'expanded_pending':sum('llm_review' not in r for r in new),
 'expanded_action_counts':dict(Counter(r['resource_review']['action_class'] for r in new if 'resource_review' in r)),
 'combined_pairs':384+len(new),'remaining_outside_review_queue':json.loads((p/'SCOPE_SUMMARY.json').read_text())['remaining_outside_review_queue'],
 'experimental_release':False,'all_complete':len(reasons)==384 and len(new)==384 and all('llm_review' in r for r in new)}
rows=[]
for r in old:
 review=reviews[r['pair_id']];action=reasons.get(r['pair_id'],{})
 rows.append({'scope':'BASELINE_384','candidate_id':r['candidate_id'],'pair_id':r['pair_id'],'drug_name':r['modeled_entity_name'],'gene_symbol':r['gene_symbol'],
 'original_indications':r['origin_summary'],'proposed_indication':r['recommended_disease'],'route':'ORIGINAL_BASELINE','review_status':'REVIEWED',**{'llm_'+k:review.get(k,'') for k in ['verdict','summary','support','concern','next_step','source_urls','search_log','source_notes']},**{k:action.get(k,'') for k in ['action_class','action_label','reason_tags','reason_detail','can_test','critical_condition']}})
for r in new:
 review=r.get('llm_review',{});action=r.get('resource_review',{})
 rows.append({'scope':'EXPANDED_EVIDENCE_NOT_EXPERIMENT_DESIGN',**{k:r.get(k,'') for k in ['pair_id','drug_name','gene_symbol','original_indications','proposed_indication']},'candidate_id':r['id'],'route':r['role_label'],'review_status':r['llm_review_status'],**{'llm_'+k:review.get(k,'') for k in ['verdict','summary','support','concern','next_step','source_urls','search_log','source_notes']},**{k:action.get(k,'') for k in ['action_class','action_label','reason_tags','reason_detail','can_test','critical_condition']}})
for name,value in [('REVIEW_SUMMARY.json',summary),('COMBINED_REVIEWS.json',rows)]:
 temp=p/(name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(p/name)
with (p/'COMBINED_REVIEWS.csv.tmp').open('w',newline='',encoding='utf-8-sig') as h:
 fields=list(rows[0]);writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
 for r in rows:writer.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k,'') for k in fields})
(p/'COMBINED_REVIEWS.csv.tmp').replace(p/'COMBINED_REVIEWS.csv')
print(json.dumps(summary,ensure_ascii=False))
