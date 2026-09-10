"""Audit candidate coverage and create a shared, pair-grouped retrospective benchmark.

No trained-model performance is inferred from mapping or from literature benchmarks.
"""
import hashlib,json,re,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/txgnn_biopathnet_comparison_20260909'
KG=ROOT/'data/raw/txgnn'
DD={'indication','off-label use','contraindication'}
SEED=42

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def norm(s):return re.sub('[^a-z0-9]','',str(s).lower())
def ident(s):return re.sub(r'\.0$','',str(s))
def js(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
 OUT.mkdir(exist_ok=True,parents=True);(OUT/'shared_split').mkdir(exist_ok=True)
 node=pd.read_csv(KG/'node.csv',dtype=str)
 node['node_id']=node.node_id.map(ident)
 drugs=node[node.node_type.eq('drug')].copy()
 byname={k:g for k,g in drugs.groupby(drugs.node_name.map(norm))}
 master=pd.read_csv(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv')
 candidates=master[master.selection_role.ne('POSITIVE_CONTROL')]
 panel=candidates.drop_duplicates('ligand_inchikey')
 assert len(panel)==120
 kg=pd.read_csv(KG/'kg_directed.csv',dtype={'x_id':str,'y_id':str})
 for col in ['x_id','y_id']:kg[col]=kg[col].map(ident)
 for col in ['x_idx','y_idx']:kg[col]=kg[col].astype(int)
 assert not kg.relation.str.startswith('rev_').any()
 graphnodes=pd.concat([kg[['x_type','x_id','x_idx']].set_axis(['type','id','idx'],axis=1),kg[['y_type','y_id','y_idx']].set_axis(['type','id','idx'],axis=1)]).drop_duplicates()
 assert not graphnodes.duplicated(['type','id']).any()
 graph_drugs=set(graphnodes[graphnodes.type.eq('drug')].id)
 disease=graphnodes[graphnodes.type.eq('disease')].merge(node[node.node_type.eq('disease')],left_on='id',right_on='node_id',how='left',validate='one_to_one').sort_values('idx')
 disease['comparison_scope']='SAME_GRAPH_INPUT_FOR_BOTH_MODELS_NOT_CHECKPOINT_COVERAGE'
 disease.to_csv(OUT/'COMMON_DISEASE_UNIVERSE.csv',index=False)
 rows=[]
 for r in panel.itertuples():
  matches=byname.get(norm(r.drug_names),drugs.iloc[:0]);rule='EXACT_PUNCTUATION_CASE_NORMALIZED_NAME';evidence='local node.csv'
  lookup_status='NOT_NEEDED_FOR_EXACT_NAME'
  if matches.empty:
   f=OUT/'raw'/f'{r.ligand_inchikey}.json'
   rule='FULL_INCHIKEY_PUBCHEM_SYNONYM_TO_GRAPH_NODE'
   if f.exists():
    j=json.loads(f.read_text());syn=[];lookup_status=str(j.get('status','UNKNOWN'))
    for a in j.get('body',{}).get('InformationList',{}).get('Information',[]):syn.extend(a.get('Synonym',[]))
    matches=drugs[drugs.node_name.map(norm).isin(set(map(norm,syn)))|drugs.node_id.isin(syn)]
    evidence=str(f.relative_to(ROOT))
  matches=matches.drop_duplicates('node_id')
  accepted=len(matches)==1 and matches.iloc[0].node_id in graph_drugs
  status='MAPPED_NAME_IDENTITY_STRUCTURE_NOT_IN_GRAPH' if accepted else ('AMBIGUOUS_HOLD' if len(matches)>1 else 'UNMAPPED_IN_CURRENT_GRAPH')
  rows.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'mapping_status':status,'mapping_rule':rule if accepted else '',
   'graph_drug_id':matches.iloc[0].node_id if accepted else '', 'graph_drug_name':matches.iloc[0].node_name if accepted else '',
   'candidate_node_ids':';'.join(matches.node_id),'mapping_evidence':evidence,'synonym_http_status':lookup_status,'txgnn_graph_covered':accepted,
   'biopathnet_shared_input_covered':accepted,'biopathnet_drug_disease_checkpoint_covered':'NOT_ESTABLISHED',
   'candidate_pair_count':int(candidates.ligand_inchikey.eq(r.ligand_inchikey).sum()),
   'high_pair_count':int((candidates.ligand_inchikey.eq(r.ligand_inchikey)&candidates.selection_role.eq('OUR_FROZEN_MODEL_HIGH')).sum())})
 coverage=pd.DataFrame(rows)
 covered=coverage[coverage.txgnn_graph_covered]
 assert not covered.graph_drug_id.duplicated().any(),'Two molecular entities map to same graph drug; explicit review required'
 coverage.to_csv(OUT/'DRUG_COVERAGE_120.csv',index=False)
 coverage[~coverage.txgnn_graph_covered].to_csv(OUT/'UNMAPPED_DRUGS.csv',index=False)
 # Group ALL drug-disease labels by endpoint pair, preventing indication/offlabel/contra crossover.
 dd=kg[kg.relation.isin(DD)].copy()
 assert dd.x_type.eq('drug').all() and dd.y_type.eq('disease').all()
 dd=dd.drop_duplicates(['x_id','relation','y_id'])
 dd['pair_key']=dd.x_id+'::'+dd.y_id
 def fold(key):
  bucket=int(hashlib.sha256(f'{SEED}|{key}'.encode()).hexdigest()[:16],16)%10
  return 'test' if bucket==0 else 'valid' if bucket==1 else 'train'
 dd['split']=dd.pair_key.map(fold)
 dd['in_candidate_panel']=dd.x_id.isin(covered.graph_drug_id)
 dd.to_csv(OUT/'ALL_DRUG_DISEASE_LABELS_WITH_SPLIT.csv.gz',index=False)
 panel_dd=dd[dd.in_candidate_panel].merge(covered[['graph_drug_id','ligand_inchikey','drug_names']],left_on='x_id',right_on='graph_drug_id')
 panel_dd.to_csv(OUT/'PANEL_KNOWN_RELATIONS.csv',index=False)
 # Explicitly differentiate novel exploration from retrospective recovery of held-out known indications.
 drug_counts=[];matrix=[]
 known=dd.groupby(['x_id','y_id']).relation.agg(lambda x:';'.join(sorted(set(x))))
 for r in covered.itertuples():
  diseases=disease[['id','idx','node_name']].copy()
  diseases['ligand_inchikey']=r.ligand_inchikey;diseases['graph_drug_id']=r.graph_drug_id
  diseases['known_graph_relations']=[known.get((r.graph_drug_id,x),'') for x in diseases.id]
  diseases['exclude_known_therapeutic']=diseases.known_graph_relations.str.contains(r'(?:^|;)(?:indication|off-label use)(?:;|$)',regex=True)
  diseases['exclude_contraindication']=diseases.known_graph_relations.str.contains('contraindication')
  diseases['eligible_graph_novel']=~(diseases.exclude_known_therapeutic|diseases.exclude_contraindication)
  drug_counts.append({'drug_names':r.drug_names,'graph_drug_id':r.graph_drug_id,'disease_count':len(diseases),
    'known_therapeutic_exclusions':int(diseases.exclude_known_therapeutic.sum()),'contraindication_exclusions':int(diseases.exclude_contraindication.sum()),
    'eligible_graph_novel_pairs':int(diseases.eligible_graph_novel.sum())})
  matrix.append(diseases)
 pd.concat(matrix).to_csv(OUT/'COMMON_QUERY_AND_EXCLUSION_MASK.csv.gz',index=False)
 pd.DataFrame(drug_counts).to_csv(OUT/'DRUG_QUERY_COUNTS.csv',index=False)
 # Typed entity tokens preserve source ids independently of each framework's private indices.
 kg['head']=kg.x_type+':'+kg.x_id;kg['tail']=kg.y_type+':'+kg.y_id
 dd['head']=dd.x_type+':'+dd.x_id;dd['tail']=dd.y_type+':'+dd.y_id
 splitdir=OUT/'shared_split'
 background=kg[~kg.relation.isin(DD)].drop_duplicates(['head','relation','tail'])
 background[['head','relation','tail']].to_csv(splitdir/'train1.txt',sep='\t',header=False,index=False)
 for split,file in [('train','train2.txt'),('valid','valid.txt'),('test','test.txt')]:
  dd[dd.split.eq(split)][['head','relation','tail']].to_csv(splitdir/file,sep='\t',header=False,index=False)
 graphnodes['token']=graphnodes.type+':'+graphnodes.id
 types={t:i for i,t in enumerate(sorted(graphnodes.type.unique()))};graphnodes['type_id']=graphnodes.type.map(types)
 graphnodes[['token','type_id']].to_csv(splitdir/'entity_types.txt',sep='\t',header=False,index=False)
 graphnodes.to_csv(splitdir/'ENTITY_ID_CROSSWALK.csv',index=False)
 # Evaluate indication recovery only; conflicting therapeutic/contra labels are explicitly held out from metrics.
 conflicts=dd.groupby('pair_key').relation.agg(lambda x:'contraindication' in set(x) and bool({'indication','off-label use'}&set(x)))
 evalpos=dd[dd.split.eq('test')&dd.in_candidate_panel&dd.relation.eq('indication')].copy()
 evalpos['conflicting_therapeutic_contra']=evalpos.pair_key.map(conflicts)
 evalpos.to_csv(OUT/'PANEL_TEST_POSITIVES.csv',index=False)
 validpos=dd[dd.split.eq('valid')&dd.in_candidate_panel&dd.relation.eq('indication')].copy()
 validpos.to_csv(OUT/'PANEL_VALID_POSITIVES.csv',index=False)
 # Shared 100 graph-unlabelled controls per test-positive. These are NOT proven biological negatives.
 rng=np.random.default_rng(SEED);diseaseids=disease.id.tolist();evalrows=[]
 for r in evalpos[~evalpos.conflicting_therapeutic_contra].itertuples():
  evalrows.append({'query_key':r.pair_key,'graph_drug_id':r.x_id,'disease_id':r.y_id,'label':1,'label_source':'heldout_graph_indication'})
  possible=[d for d in diseaseids if (r.x_id,d) not in known.index]
  for d in rng.choice(possible,size=min(100,len(possible)),replace=False):
   evalrows.append({'query_key':r.pair_key,'graph_drug_id':r.x_id,'disease_id':d,'label':0,'label_source':'graph_unlabelled_NOT_verified_inactive'})
 pd.DataFrame(evalrows).to_csv(OUT/'SHARED_TEST_SAMPLED_LABELS.csv',index=False)
 # The historical full_graph validation/test may overlap; check directly, never reuse silently.
 old={}
 for split in ['train','valid','test']:
  a=pd.read_csv(KG/'full_graph_42'/f'{split}.csv',usecols=['x_id','relation','y_id'],dtype=str)
  a=a[a.relation.isin(DD)]
  old[split]=set(zip(a.x_id.map(ident),a.relation,a.y_id.map(ident)))
 oldoverlap={a+'_'+b:len(old[a]&old[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
 pairs={s:set(dd[dd.split.eq(s)].pair_key) for s in ['train','valid','test']}
 checks={'panel_120':len(coverage)==120,'unique_graph_drug_mapping':covered.graph_drug_id.is_unique,
  'no_pair_overlap':not any(pairs[a]&pairs[b] for a,b in [('train','valid'),('train','test'),('valid','test')]),
  'no_dd_in_background':not background.relation.isin(DD).any(),'no_reverse_edges_in_canonical_inputs':not kg.relation.str.startswith('rev_').any(),
  'test_labels_present':len(evalrows)>0,'all_graph_entities_exported':graphnodes.token.is_unique,
  'source_snapshot_only_exclusions':True}
 assert all(checks.values());js(OUT/'VALIDATION.json',checks)
 counts=dd.groupby(['split','relation']).size().unstack(fill_value=0).to_dict(orient='index')
 summary={'status':'COVERAGE_AND_SHARED_BENCHMARK_READY_PERFORMANCE_NOT_RUN','candidate_drugs':120,
  'mapped_graph_drugs':len(covered),'unmapped_graph_drugs':int((~coverage.txgnn_graph_covered).sum()),
  'mapping_rules':covered.mapping_rule.value_counts().to_dict(),'shared_diseases':len(disease),
  'covered_candidate_pairs':int(covered.candidate_pair_count.sum()),'covered_high_pairs':int(covered.high_pair_count.sum()),
  'candidate_graph_known_relations':panel_dd.relation.value_counts().to_dict(),'graph_novel_query_pairs':sum(x['eligible_graph_novel_pairs'] for x in drug_counts),
  'shared_split_counts':counts,'background_edges':len(background),
  'panel_test_indications':len(evalpos),'panel_test_drugs':evalpos.x_id.nunique(),'panel_test_diseases':evalpos.y_id.nunique(),
  'panel_test_conflicting_labels':int(evalpos.conflicting_therapeutic_contra.sum()),'sampled_eval_rows':len(evalrows),
  'historical_full_graph_dd_overlap':oldoverlap,
  'trained_model_metrics':None,'reason':'No audited BioPathNet drug-disease checkpoint; historical TxGNN Explorer must not be evaluated on a newly hidden subset it may have seen.',
  'limitations':['Mapping is node identity via names/synonyms; graph lacks structure verification','Exclusions are graph snapshot only, not complete current approved-indication audit','Pair-grouped random split is retrospective/transductive, not disease zero-shot or temporal','Input graph coverage is not checkpoint coverage','Shared unknown negatives do not establish biological inactivity']}
 js(OUT/'SUMMARY.json',summary)
 inputs=[ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv',KG/'node.csv',KG/'kg_directed.csv',KG/'TxGNNExplorer/model.pt',KG/'TxGNNExplorer/config.pkl']
 inputs.extend(sorted((ROOT/'third_party/TxGNN/txgnn').rglob('*.py')))
 js(OUT/'MANIFEST.json',{'seed':SEED,'builder_sha256':sha(Path(__file__)),
  'inputs':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
  'git_commits':{'.external/BioPathNet':subprocess.check_output(['git','-C',str(ROOT/'.external/BioPathNet'),'rev-parse','HEAD'],text=True).strip()},
  'txgnn_version_note':'Local vendored source identified by individual file hashes, not parent BioMaster git commit.',
  'files':{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and p.suffix!='.log'}})
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
