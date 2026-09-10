"""Validate the completed evidence pool without releasing or editing an experiment design."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from biomaster.explorer_spr_expansion import expanded_items,baseline_reasons
p=root/'outputs/spr_expanded_review_20260910';q=list(csv.DictReader((p/'NEW_CANDIDATES_FOR_LLM.csv').open()))
b=root/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'
old=list(csv.DictReader(b.open()));items=expanded_items(root);reasons=baseline_reasons(root)
checks={
'baseline_sha_unchanged':hashlib.sha256(b.read_bytes()).hexdigest()=='bf1ac323287c3b54c4ca92a5326e9da3e39f355fad60f2ac3edf672d0c4ef042',
'new384_unique_disjoint':len(q)==len({r['pair_id'] for r in q})==384 and not {r['pair_id'] for r in q}&{r['pair_id'] for r in old},
'baseline384_reclassified':len(reasons)==384,
'new384_reviewed':len(items)==384 and all('llm_review' in x for x in items),
'no_local_raw_activity':all(float(r['raw_activity_rows'])==0 for r in q),
'identity_chemistry_novelty_gates':all(r[k].lower()=='true' for r in q for k in ['identity_scope_pass','chemistry_policy_pass','novelty_pass']),
'route_192_192':Counter(r['route'] for r in q)=={'BINDING_CHEMISTRY_WITHOUT_GRAPH_GATE':192,'RELAXED_JOINT':166,'BOTH':26},
'drug_cap4_target_cap8':max(Counter(r['ligand_inchikey'] for r in q).values())<=4 and max(Counter(r['target_chembl_id'] for r in q).values())<=8,
'joint_thresholds':all(float(r['binding_rank_384'])<=30 and float(r['recommended_txgnn_rank'])<=100 and float(r['recommended_ot_score'])>=.3 for r in q if r['route']!='BINDING_CHEMISTRY_WITHOUT_GRAPH_GATE'),
'graph_independent_thresholds':all(float(r['binding_rank_384'])<=10 and float(r['positive_max_tanimoto'])>=.35 for r in q if r['route']=='BINDING_CHEMISTRY_WITHOUT_GRAPH_GATE'),
'new_pool_not_experimental_release':all(not r['is_control'] and r['design_id']=='EXPANDED_EVIDENCE_POOL_20260910' for r in items)
}
for i in (1,2,3):
 rows=json.loads((p/f'EXPANDED_REVIEW_{i}.json').read_text());batch=json.loads((p/f'expanded_batch_{i}.json').read_text())
 checks[f'batch{i}_complete_identity_sources']=len(rows)==128 and {(r['candidate_id'],r['pair_id']) for r in rows}=={(r['candidate_id'],r['pair_id']) for r in batch} and all(r['search_log'] and isinstance(r['source_urls'],list) for r in rows)
result={'checks':checks,'all_passed':all(checks.values()),'experimental_release':False};print(json.dumps(result,ensure_ascii=False,indent=2))
if all(checks.values()):(p/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
