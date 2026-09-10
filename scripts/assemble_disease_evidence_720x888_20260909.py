"""Assemble source-traceable disease evidence, new-edge audit, and unchanged SPR annotations."""
import json,gzip,collections,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def mondo(x):return 'MONDO:'+str(x).zfill(7)
def main():
 c=pd.read_csv(OUT/'DRUG_COVERAGE_720.csv').fillna('');t=pd.read_csv(OUT/'TARGET_REGISTRY_888.csv').fillna('');ds=pd.read_csv(OUT/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str});order=pd.read_csv(OUT/'TXGNN_SCORE_DRUG_ORDER.csv');scores=np.load(OUT/'TXGNN_INDICATION_LOGITS.npy')
 kg=pd.read_csv(ROOT/'data/raw/txgnn/kg_directed.csv',dtype={'x_id':str,'y_id':str});kg.x_id=kg.x_id.str.replace(r'\.0$','',regex=True);kg.y_id=kg.y_id.str.replace(r'\.0$','',regex=True)
 didx={};disease_map=[]
 for i,r in enumerate(ds.itertuples()):
  members=str(r.id).split('_')
  for member in members:
   curie=mondo(member);didx[curie]=i;disease_map.append({'disease_id':curie,'txgnn_disease_id':r.id,'score_column':i,'txgnn_disease_name':r.node_name,'mapping_scope':'EXACT_SINGLE_NODE' if len(members)==1 else 'MEMBER_OF_MERGED_NODE','merged_member_count':len(members)})
 pd.DataFrame(disease_map).to_csv(OUT/'DISEASE_ID_CROSSWALK.csv',index=False)
 ecdis=pd.read_parquet(next((OUT/'eckg/disease-list').glob('*.parquet')));special=[s for s in ecdis.columns if s.startswith('speciality_')]
 dirmat=np.zeros((len(ds),len(special)),bool)
 for r in ecdis.to_dict('records'):
  if r['id'] in didx:
   for j,s in enumerate(special):dirmat[didx[r['id']],j]|=bool(r[s])
 ds['direction_labels']=[';'.join(special[j].removeprefix('speciality_') for j in np.where(x)[0]) or 'UNCLASSIFIED_BY_EC_ONTOLOGY' for x in dirmat]
 ds.to_csv(OUT/'TXGNN_ALL_DISEASE_DIRECTIONS.csv',index=False)
 # OT full long table; source disease IDs preserved; merged-node mappings remain explicitly labelled.
 target_idx={r.target_chembl_id:i for i,r in enumerate(t.itertuples())};otmatrix=np.zeros((len(t),len(ds)),np.float32);otrows=[];otbest={};target_summary=[]
 for ti,r in enumerate(t.itertuples()):
  p=OUT/'ot_cache'/f'{r.ot_id}.json.gz'
  if not p.exists():continue
  with gzip.open(p,'rt') as f:j=json.load(f)
  for a in j['rows']:
   dis=a['disease'];curie=dis['id'].replace('MONDO_','MONDO:');mi=didx.get(curie);parts={x['id']:x['score'] for x in a['datatypeScores']}
   row={'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'ot_id':r.ot_id,'disease_id':dis['id'],'disease_name':dis['name'],'overall_score':a['score'],'therapeutic_areas':';'.join(x['name'] for x in dis.get('therapeuticAreas',[])),'therapeutic_area_ids':';'.join(x['id'] for x in dis.get('therapeuticAreas',[])),'txgnn_score_column':mi if mi is not None else -1,'txgnn_mapping_scope':('MEMBER_OF_MERGED_NODE' if '_' in ds.iloc[mi].id else 'EXACT_SINGLE_NODE') if mi is not None else 'NO_EXACT_MONDO_MAPPING','datatype_scores_json':json.dumps(parts),'source_ensembl_id':a.get('source_ensembl_id',r.ot_id),'source':'OpenTargets_26.06','treatment_direction_established':False}
   for dt in ['genetic_association','somatic_mutation','known_drug','clinical','affected_pathway','literature','animal_model','rna_expression']:row[dt+'_score']=parts.get(dt,0)
   otrows.append(row)
   if mi is not None and a['score']>otmatrix[ti,mi]:otmatrix[ti,mi]=a['score'];otbest[(ti,mi)]=row
  top=sorted(j['rows'],key=lambda x:-x['score'])[:5]
  target_summary.append({'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'fetched_count':len(j['rows']),'api_count':j['count'],'top5_diseases':'; '.join(x['disease']['name'] for x in top),'mapped_txgnn_disease_nodes':int((otmatrix[ti]>0).sum())})
 ot=pd.DataFrame(otrows);ot.to_parquet(OUT/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet',index=False);pd.DataFrame(target_summary).to_csv(OUT/'TARGET_DISEASE_SUMMARY.csv',index=False)
 del otrows
 print('OT assembled',len(ot),flush=True)
 # Map old disease/gene/drug endpoints through pinned EC identifiers.
 matches=pd.read_csv(OUT/'ECKG_PROJECT_NODE_MATCHES.csv.gz').fillna('');ecnodes=pq.read_table(OUT/'ECKG_PROJECT_AND_DISEASE_NODES.parquet').to_pylist()
 ec_to_disease={};ec_to_gene={};ec_to_drug={}
 for r in ecnodes:
  ids=[r['id']]+(r['equivalent_identifiers'] or [])
  diseaseids={x for x in ids if x.startswith('MONDO:')};geneids={x.split(':',1)[1] for x in ids if x.lower().startswith('ncbigene:')};drugids={x.split(':',1)[1] for x in ids if x.lower().startswith('drugbank:')}
  if diseaseids:ec_to_disease[r['id']]=diseaseids
  if geneids:ec_to_gene[r['id']]=geneids
  if drugids:ec_to_drug[r['id']]=drugids
 node_counts=matches.groupby('project_key').ec_id.nunique().to_dict()
 ecprojects=matches.groupby('ec_id').project_key.agg(lambda v:set(v)).to_dict()
 old_node=pd.read_csv(ROOT/'data/raw/txgnn/node.csv',dtype=str);oldgenes=old_node[old_node.node_type.eq('gene/protein')]
 gene_byname=oldgenes.groupby('node_name').node_id.agg(lambda v:set(v.str.replace(r'\.0$','',regex=True))).to_dict()
 target_old={r.target_chembl_id:gene_byname.get(r.gene_symbol,set()) for r in t.itertuples()}
 for r in matches[matches.entity_type.eq('target')].itertuples():target_old[r.project_key.split(':',1)[1]].update(ec_to_gene.get(r.ec_id,set()))
 old_dd=set(map(tuple,kg[kg.relation.isin(['indication','off-label use','contraindication'])][['x_id','y_id']].values));old_dp=set(map(tuple,kg[kg.relation.eq('drug_protein')][['x_id','y_id']].values));old_td=set(map(tuple,kg[kg.relation.eq('disease_protein')][['x_id','y_id']].values))
 olddrug=c.set_index('ligand_inchikey').graph_drug_id.to_dict();old_did={m:ds.iloc[i].id for m,i in didx.items()}
 records=[];new_treat=collections.defaultdict(set);new_dt=set();seen_edges=set();raw_relevant=0
 for batch in pq.ParquetFile(OUT/'ECKG_PROJECT_INCIDENT_EDGES.parquet').iter_batches(batch_size=50000):
  for e in batch.to_pylist():
   a,b=e['subject'],e['object'];srcs=set(e['primary_knowledge_sources'] or [])|{e['primary_knowledge_source'] or ''};srcs.discard('')
   support='TEXT_MINING_OR_PREDICTION' if e['agent_type']=='text_mining_agent' or e['knowledge_level']=='prediction' else ('PRIMEKG_ONLY_REUSE' if srcs=={'infores:primekg'} else 'OTHER_SOURCE_ASSERTION_REVIEW')
   for pe,other in [(a,b),(b,a)]:
    for project in ecprojects.get(pe,set()):
     typ,key=project.split(':',1);pairs=[]
     if other in ec_to_disease:
      for mid in ec_to_disease[other]:
       olddis=old_did.get(mid);baseline=bool(olddis) and ((olddrug.get(key),olddis) in old_dd if typ=='drug' else any((g,olddis) in old_td for g in target_old[key]))
       pairs.append((('DRUG_DISEASE' if typ=='drug' else 'TARGET_DISEASE'),mid,baseline))
     if typ=='drug':
      for otherproject in ecprojects.get(other,set()):
       if otherproject.startswith('target:'):
        tk=otherproject.split(':',1)[1];baseline=any((olddrug.get(key),g) in old_dp for g in target_old[tk]);pairs.append(('DRUG_TARGET',otherproject,baseline))
     for relation,otherid,baseline in pairs:
      raw_relevant+=1;finger=(project,otherid,e['predicate'],e['qualified_predicate'],e['object_direction_qualifier'],';'.join(sorted(srcs)),a,b)
      if finger in seen_edges:continue
      seen_edges.add(finger)
      old_comparable=(bool(olddrug.get(key)) if typ=='drug' else bool(target_old[key])) and (otherid in old_did if relation.endswith('DISEASE') else bool(target_old.get(otherid.split(':',1)[1])))
      increment='EXISTING_ENDPOINT_PAIR' if baseline else ('NEW_ENDPOINT_PAIR_ON_COMPARABLE_NODES' if old_comparable else 'NEW_OR_UNMAPPED_ENDPOINT_NOT_COMPARABLE')
      records.append({'project_key':project,'other_id':otherid,'relation_class':relation,'ec_subject':a,'ec_object':b,'predicate':e['predicate'],'qualified_predicate':e['qualified_predicate'],'object_direction_qualifier':e['object_direction_qualifier'],'primary_sources':';'.join(sorted(srcs)),'knowledge_level':e['knowledge_level'],'agent_type':e['agent_type'],'evidence_class':support,'mapping_ambiguous':node_counts.get(project,0)>1 or (relation=='DRUG_TARGET' and node_counts.get(otherid,0)>1),'increment_class':increment,'publications':';'.join(e['publications'] or [])})
      if relation=='DRUG_DISEASE' and pe==a and e['predicate'] in ['biolink:treats','biolink:clinical_trials_for','biolink:contraindicated_in'] and support!='TEXT_MINING_OR_PREDICTION' and node_counts.get(project)==1 and key!='LPAUOXUZGSBGDU-STDDISTJSA-N':new_treat[key].add(otherid)
      if relation=='DRUG_TARGET':new_dt.add((key,otherid.split(':',1)[1]))
 edges=pd.DataFrame(records);edges.to_parquet(OUT/'ECKG_RELEVANT_RELATION_AUDIT.parquet',index=False)
 groups=edges.groupby(['relation_class','increment_class','evidence_class']).size().reset_index(name='unique_evidence_rows');groups.to_csv(OUT/'ECKG_INCREMENT_COUNTS.csv',index=False)
 dump('ECKG_INCREMENT_SUMMARY.json',{'relevant_raw_expanded_rows':raw_relevant,'deduplicated_evidence_rows':len(edges),'note':'Endpoint novelty relative to frozen graph, not biological novelty. Evidence dedup preserves selected predicate/direction/source fields; raw complete qualified edges retained separately.','new_non_text_comparable_pairs':int(edges[(edges.increment_class=='NEW_ENDPOINT_PAIR_ON_COMPARABLE_NODES')&(edges.evidence_class=='OTHER_SOURCE_ASSERTION_REVIEW')&(~edges.mapping_ambiguous)][['project_key','other_id','relation_class']].drop_duplicates().shape[0])})
 print('EC audited',len(edges),flush=True)
 # Full score matrix exclusions, then all-direction top tables. Keep raw matrix unchanged.
 known=kg[kg.relation.isin(['indication','off-label use','contraindication'])].groupby('x_id').y_id.agg(set).to_dict();oldcol={x:i for i,x in enumerate(ds.id)};mask=np.zeros(scores.shape,np.uint8);novel=scores.copy();toprows=[];dirrows=[];top50=np.zeros(scores.shape,np.int16)
 for di,r in enumerate(order.itertuples()):
  for x in known.get(r.graph_drug_id,set()):
   if x in oldcol:mask[di,oldcol[x]]|=1
  for x in new_treat.get(r.ligand_inchikey,set()):
   if x in didx:mask[di,didx[x]]|=2
  novel[di,mask[di]>0]=-np.inf
  ids=np.argsort(-novel[di],kind='stable');top50[di,ids[:50]]=1
  for rank,ci in enumerate(ids[:30],1):toprows.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'rank':rank,'txgnn_disease_id':ds.iloc[ci].id,'disease_name':ds.iloc[ci].node_name,'logit':float(scores[di,ci]),'directions':ds.iloc[ci].direction_labels,'novelty_scope':'EXCLUDES_FROZEN_KG_LABELS_AND_NON_TEXT_ECKG_TREAT_TRIAL_CONTRA;NOT_COMPLETE_CURRENT_INDICATION_REVIEW'})
  for si,s in enumerate(special):
   selected=[ci for ci in ids if dirmat[ci,si] and np.isfinite(novel[di,ci])][:3]
   for rank,ci in enumerate(selected,1):dirrows.append({'ligand_inchikey':r.ligand_inchikey,'drug_names':r.drug_names,'direction':s.removeprefix('speciality_'),'rank':rank,'txgnn_disease_id':ds.iloc[ci].id,'disease_name':ds.iloc[ci].node_name,'logit':float(scores[di,ci])})
 np.save(OUT/'TXGNN_EXCLUSION_MASK.npy',mask);pd.DataFrame(toprows).to_csv(OUT/'DRUG_ALL_DISEASE_TOP30.csv.gz',index=False);pd.DataFrame(dirrows).to_csv(OUT/'DRUG_EVERY_DIRECTION_TOP3.csv.gz',index=False)
 # All 720 x 888 bookkeeping; overlap is descriptive, not a binding or efficacy model.
 overlap=top50 @ (otmatrix>0).astype(np.int16).T;drugi={r.ligand_inchikey:i for i,r in enumerate(order.itertuples())};coverage=[]
 oldtargets=set(pd.read_csv(ROOT/'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz',usecols=['target_chembl_id']).target_chembl_id)
 routedtargets=set(pd.read_csv(ROOT/'outputs/target_discovery_scope_ch37_v3/TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz',usecols=['target_chembl_id']).target_chembl_id)
 ots=pd.read_csv(OUT/'OT_TARGET_COVERAGE_888.csv').set_index('target_chembl_id').status.to_dict()
 for r in c.itertuples():
  di=drugi.get(r.ligand_inchikey)
  for ti,tr in enumerate(t.itertuples()):coverage.append({'ligand_inchikey':r.ligand_inchikey,'target_chembl_id':tr.target_chembl_id,'txgnn_status':'SCORED_ALL_17080' if di is not None else r.mapping_status,'ot_status':ots[tr.target_chembl_id],'top50_graph_novel_diseases_with_ot_evidence':int(overlap[di,ti]) if di is not None else -1,'eckg_drug_target_relation_present':(r.ligand_inchikey,tr.target_chembl_id) in new_dt,'historical_dti_scope':'384_COMPARISON_CORE' if tr.target_chembl_id in oldtargets else ('745_REGISTRY_EXTENSION_NOT_SAME_RANK_DENOMINATOR' if tr.target_chembl_id in routedtargets else 'OUTSIDE_745_NO_DTI_INFERENCE_THIS_RUN')})
 pd.DataFrame(coverage).to_parquet(OUT/'PAIR_COVERAGE_720X888.parquet',index=False)
 master=pd.read_csv(ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv');spr=[]
 for r in master.to_dict('records'):
  di=drugi.get(r['ligand_inchikey']);ti=target_idx.get(r['target_chembl_id']);row={k:r[k] for k in ['ligand_inchikey','drug_names','target_chembl_id','gene_symbol','selection_role']};row['txgnn_available']=di is not None;row['ot_status']=ots.get(r['target_chembl_id'],'UNKNOWN');row['treatment_direction_established']=False
  if di is not None and ti is not None:
   ids=np.where((otmatrix[ti]>0)&(mask[di]==0))[0];ids=ids[np.argsort(-scores[di,ids],kind='stable')][:3];row['top50_ot_overlap']=int(overlap[di,ti]);row['disease_hypotheses_json']=json.dumps([{'txgnn_disease_id':ds.iloc[ci].id,'txgnn_disease_name':ds.iloc[ci].node_name,'txgnn_logit':float(scores[di,ci]),'ot_disease_id':otbest[(ti,ci)]['disease_id'],'ot_disease_name':otbest[(ti,ci)]['disease_name'],'ot_score':float(otmatrix[ti,ci]),'mapping_scope':otbest[(ti,ci)]['txgnn_mapping_scope']} for ci in ids],ensure_ascii=False)
  spr.append(row)
 pd.DataFrame(spr).to_csv(OUT/'SPR512_DISEASE_EVIDENCE_APPEND_ONLY.csv',index=False)
 tc=[]
 for r in t.itertuples():tc.append({'target_chembl_id':r.target_chembl_id,'gene_symbol':r.gene_symbol,'old_gene_ids':';'.join(sorted(target_old[r.target_chembl_id])),'old_gene_mapping_status':'MAPPED' if target_old[r.target_chembl_id] else 'UNMAPPED','ec_node_count':matches[matches.project_key.eq('target:'+r.target_chembl_id)].ec_id.nunique(),'ot_status':ots[r.target_chembl_id]})
 pd.DataFrame(tc).to_csv(OUT/'TARGET_COVERAGE_888.csv',index=False)
 dump('ASSEMBLY_SUMMARY.json',{'project_drugs':len(c),'project_targets':len(t),'scored_drugs':len(order),'scored_diseases':len(ds),'score_count':int(scores.size),'ot_evidence_rows':len(ot),'ot_status_counts':dict(collections.Counter(ots.values())),'old_graph_exclusions':int(((mask&1)>0).sum()),'new_graph_exclusions':int(((mask&2)>0).sum()),'total_excluded_query_pairs':int((mask>0).sum()),'directions':len(special),'direction_disease_node_counts':{s:int(dirmat[:,i].sum()) for i,s in enumerate(special)},'unclassified_old_disease_nodes':int((~dirmat.any(axis=1)).sum()),'all_pair_bookkeeping_rows':len(coverage),'spr_rows':len(spr),'model_training':False})
 print('ASSEMBLY COMPLETE',flush=True)
if __name__=='__main__':main()
