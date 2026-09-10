"""Frozen-score joint screen; no training, no edits to original SPR candidates."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
import build_retargetmap_spr512_ours_frozen_v2 as old
from design_spr64_512_20260909 import annotate_support
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
OUT=ROOT/'outputs/joint_screen_720x384_20260909'
MASTER=ROOT/'outputs/retargetmap_spr64_design_20260909/INTERNAL_MASTER_512.csv'
COMPONENTS=ROOT/'outputs/strict_affinity_main_queue_2005_2026_v2/STRICT_DRUG_KNOWN_MECHANISM_COMPONENT_AUDIT.csv.gz'
EXPECTED_MASTER='e6501353312b62717c766156fdea4b145bbf9b135eef3efff332d491d1f02487'
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 assert sha(MASTER)==EXPECTED_MASTER
 p=pd.read_csv(old.RANK_PATH).rename(columns={'pairId':'pair_id'})
 p['binding_rank_384']=p.groupby('ligand_inchikey',sort=False).independent_validation_rank_score.rank(method='first',ascending=False).astype(int)
 assert len(p)==276480 and p.pair_id.is_unique and p.ligand_inchikey.nunique()==720 and p.target_chembl_id.nunique()==384
 registry=pd.read_csv(BASE/'DRUG_REGISTRY_720.csv',low_memory=False)
 cols=['ligand_inchikey','ligand_smiles','exact_structure_in_fda_registry','final_drug_scope_status','known_target_chembl_ids','active_species_status','chembl_any_prodrug','pubchem_evidence_interpretation']
 p=p.merge(registry[cols],on='ligand_inchikey',validate='many_to_one')
 cov=pd.read_csv(BASE/'DRUG_COVERAGE_FINAL_720.csv')
 p=p.merge(cov.drop(columns=['drug_names']),on='ligand_inchikey',validate='many_to_one')
 targets=pd.read_csv(ROOT/'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv',low_memory=False)
 tc=[c for c in ['target_chembl_id','uniprot_accession','assay_lane','assay_confirmation_status','structure_ready_strict','positive_compounds','negative_compounds'] if c in targets]
 p=p.merge(targets[tc],on='target_chembl_id',validate='many_to_one')
 assert p.uniprot_accession.notna().all()
 assert not p[['ligand_inchikey','uniprot_accession','pair_id']].drop_duplicates().duplicated(['ligand_inchikey','uniprot_accession']).any(), 'External scan requires unique accession per target'
 feat=old.molecule_features(registry[['ligand_inchikey','ligand_smiles']])
 checks=[]
 for r in registry.itertuples():
  mol=Chem.MolFromSmiles(r.ligand_smiles)
  checks.append({'ligand_inchikey':r.ligand_inchikey,'smiles_full_key_matches':Chem.MolToInchiKey(mol)==r.ligand_inchikey,'actual_fragment_count':len(Chem.GetMolFrags(mol)),'actual_formal_charge':Chem.GetFormalCharge(mol)})
 feat=feat.merge(pd.DataFrame(checks),on='ligand_inchikey',validate='one_to_one')
 p=p.merge(feat.drop(columns='ligand_smiles'),on='ligand_inchikey',validate='many_to_one')
 p['chemical_risk_points']=p.rdkit_mw.gt(550).astype(int)+p.rdkit_clogp.gt(5).astype(int)+p.rdkit_tpsa.gt(140).astype(int)+p.rdkit_rotatable_bonds.gt(12).astype(int)
 p['active_species_review_hold']=p.active_species_status.eq('DUAL_SPECIES_REQUIRED')|p.chembl_any_prodrug.eq(True)
 p['identity_scope_pass']=p.exact_structure_in_fda_registry & p.smiles_full_key_matches & p.final_drug_scope_status.eq('INCLUDE_CORE_DISCOVERY') & ~p.identity_hold & ~p.active_species_review_hold & ~p.drug_names.isin(['enalapril','serdexmethylphenidate'])
 p['chemistry_policy_pass']=p.rdkit_mw.between(120,700)&p.actual_fragment_count.eq(1)&p.actual_formal_charge.abs().le(2)&p.chemical_risk_points.le(1)
 log('Full space loaded; screening exact known/training relationships')
 strict=pd.read_csv(old.STRICT_PATH,low_memory=False)
 ledger=[]
 for name,path,key in [('chembl37',old.STRICT_PATH,'parent_standard_inchi_key'),('training',old.TRAINING_PATH,'parent_standard_inchi_key'),('kirhub',old.KIRHUB_PATH,'ligand_inchikey')]:
  d=pd.read_csv(path,usecols=[key,'target_chembl_id']);ids=d[key].astype(str)+'__'+d.target_chembl_id.astype(str)
  p['known_'+name]=p.pair_id.isin(set(ids));ledger.extend({'pair_id':x,'source':name} for x in sorted(set(ids)&set(p.pair_id)))
 p['known_moa']=[t in old.parse_id_set(x) for t,x in zip(p.target_chembl_id,p.known_target_chembl_ids)]
 ledger.extend({'pair_id':x,'source':'FDA_ChEMBL_MOA'} for x in p.loc[p.known_moa,'pair_id'])
 # Known family/complex mechanisms must not reappear as novel subunit hypotheses.
 component=pd.read_csv(COMPONENTS).drop(columns=['component_sequence'])
 fda=pd.read_csv(old.FDA_PATH,usecols=['project_entity_id','model_inchikey']).drop_duplicates()
 component=component.merge(fda,on='project_entity_id',validate='many_to_many')
 component=component[component.component_organism.eq('Homo sapiens')]
 component_hits=p[['pair_id','ligand_inchikey','uniprot_accession']].merge(component,left_on=['ligand_inchikey','uniprot_accession'],right_on=['model_inchikey','component_accession'])
 component_hits.to_csv(OUT/'KNOWN_MOA_COMPONENT_RECORDS.csv.gz',index=False)
 p['known_moa_component']=p.pair_id.isin(component_hits.pair_id)
 ledger.extend({'pair_id':x,'source':'FDA_ChEMBL_MOA_COMPONENT'} for x in sorted(set(component_hits.pair_id)))
 index=p[['pair_id','ligand_inchikey','uniprot_accession']]
 for tag,path in [('bindingdb_articles',old.BINDINGDB_ARTICLES),('bindingdb_pubchem',old.BINDINGDB_PUBCHEM),('gtopdb',None)]:
  cache=OUT/(tag+'_EXACT_RECORDS.csv.gz');meta=OUT/(tag+'_CACHE_IDENTITY.json')
  sources=[path] if path else [old.GTOPDB_DIR/'ligands.csv',old.GTOPDB_DIR/'interactions.csv']
  ident={'rank_sha256':sha(old.RANK_PATH),'sources':{str(s):sha(s) for s in sources}}
  if cache.exists() and meta.exists() and json.loads(meta.read_text())==ident:d=pd.read_csv(cache,low_memory=False)
  else:
   log('Scanning '+tag)
   d=old.scan_gtopdb(index) if path is None else old.scan_bindingdb(path,tag+'_2026-07',index)
   if 'pair_id' not in d:d=pd.DataFrame(columns=['pair_id'])
   d.to_csv(cache,index=False);meta.write_text(json.dumps(ident,indent=2))
  p['known_'+tag]=p.pair_id.isin(set(d.pair_id));log(tag+': '+str(p['known_'+tag].sum())+' pairs')
  ledger.extend({'pair_id':x,'source':tag} for x in sorted(set(d.pair_id)))
 pd.DataFrame(ledger).to_csv(OUT/'KNOWN_RELATION_SOURCE_LEDGER.csv.gz',index=False)
 ec=pd.read_parquet(BASE/'ECKG_RELEVANT_RELATION_AUDIT.parquet')
 ec=ec[ec.relation_class.eq('DRUG_TARGET') & ec.project_key.str.startswith('drug:') & ec.other_id.str.startswith('target:')].copy()
 ec['pair_id']=ec.project_key.str.removeprefix('drug:')+'__'+ec.other_id.str.removeprefix('target:')
 ec=ec[ec.pair_id.isin(p.pair_id)]
 ec.to_parquet(OUT/'ECKG_ALL_MATCHING_RELATION_ROWS.parquet',index=False)
 nontext=~ec.evidence_class.isin(['TEXT_MINING_OR_PREDICTION','INFERRED_OR_UNSPECIFIED_HOLD'])&~ec.mapping_ambiguous
 physical=ec.predicate.isin(['biolink:directly_physically_interacts_with','biolink:physically_interacts_with'])
 p['known_ec_physical']=p.pair_id.isin(set(ec.loc[nontext&physical,'pair_id']))
 p['ec_other_assertion_hold']=p.pair_id.isin(set(ec.loc[nontext&~physical,'pair_id']))
 p['ec_ambiguous_relation_hold']=p.pair_id.isin(set(ec.loc[ec.mapping_ambiguous&~ec.evidence_class.eq('TEXT_MINING_OR_PREDICTION'),'pair_id']))
 knowncols=[c for c in p if c.startswith('known_') and c!='known_target_chembl_ids']
 p['known_relation_excluded']=p[knowncols].any(axis=1)
 p['prior_relation_review_hold']=p.ec_other_assertion_hold|p.ec_ambiguous_relation_hold
 p['novelty_pass']=~p.known_relation_excluded&~p.prior_relation_review_hold
 log('Computing all-disease joint criteria from existing 608 x 17080 logits')
 order=pd.read_csv(BASE/'TXGNN_SCORE_DRUG_ORDER.csv');ds=pd.read_csv(BASE/'TXGNN_SCORE_DISEASE_ORDER.csv',dtype={'id':str})
 values=np.load(BASE/'TXGNN_INDICATION_LOGITS.npy');mask=np.load(BASE/'TXGNN_EXCLUSION_MASK.npy')
 ot=pd.read_parquet(BASE/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet')
 ot=ot[ot.target_chembl_id.isin(p.target_chembl_id)].copy();ot.to_parquet(OUT/'OPENTARGETS_384_FULL_ASSOCIATIONS.parquet',index=False)
 tids=sorted(p.target_chembl_id.unique());ti={t:i for i,t in enumerate(tids)}
 mapped=ot[ot.txgnn_score_column.ge(0)].copy();mapped['ti']=mapped.target_chembl_id.map(ti);mapped['ci']=mapped.txgnn_score_column.astype(int)
 criteria={'joint_r50_ot01':(.1,50),'joint_r50_ot03':(.3,50),'joint_r50_ot05':(.5,50),'joint_r100_ot03':(.3,100)}
 matrices={k:np.zeros((384,len(ds)),bool) for k in [*criteria,'joint_r50_ot03_exact','joint_r50_ot03_genetic','joint_r50_ot03_exact_genetic']}
 for name in matrices:
  threshold=criteria.get(name,(.3,50))[0];sel=mapped.overall_score.ge(threshold)
  if 'exact' in name:sel &= mapped.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE')
  if 'genetic' in name:sel &= mapped.genetic_association_score.ge(.1)|mapped.somatic_mutation_score.ge(.1)
  sub=mapped[sel];matrices[name][sub.ti,sub.ci]=True
 mappedkeys=set(cov.loc[cov.interpretation_eligible,'ligand_inchikey']);rows=[];rankcache={};valueidx={k:i for i,k in enumerate(order.ligand_inchikey)}
 for d in sorted(mappedkeys):
  ix=valueidx[d];ids=np.where(mask[ix]==0)[0];rank=np.full(len(ds),-1,np.int32);rank[ids[np.argsort(-values[ix,ids],kind='stable')]]=np.arange(1,len(ids)+1);rankcache[d]=rank
  results={k:m[:,np.where((rank>0)&(rank<=criteria.get(k,(.3,50))[1]))[0]].any(axis=1) for k,m in matrices.items()}
  for t,j in ti.items():rows.append({'pair_id':d+'__'+t,**{k:bool(v[j]) for k,v in results.items()}})
 p=p.merge(pd.DataFrame(rows),on='pair_id',how='left',validate='one_to_one')
 p['base_eligible']=p.identity_scope_pass&p.chemistry_policy_pass&p.novelty_pass&p.interpretation_eligible
 p['primary_joint_selected']=p.base_eligible&p.binding_rank_384.le(20)&p.joint_r50_ot03.eq(True)
 p['expanded_rank50_selected']=p.base_eligible&p.binding_rank_384.le(50)&p.joint_r50_ot03.eq(True)
 p['screen_status']=np.select([~p.identity_scope_pass,~p.chemistry_policy_pass,p.known_relation_excluded,p.prior_relation_review_hold,~p.interpretation_eligible,p.primary_joint_selected,p.expanded_rank50_selected],['IDENTITY_OR_SCOPE_HOLD','CHEMISTRY_POLICY_HOLD','KNOWN_RELATION_OR_TRAINING_EXCLUDED','PRIOR_RELATION_REVIEW_HOLD','TXGNN_UNAVAILABLE','PRIMARY_JOINT_REVIEW','RANK21_50_SENSITIVITY_REVIEW'],default='DOES_NOT_MEET_JOINT_PRIORITY')
 flags=['identity_scope_pass','chemistry_policy_pass','novelty_pass','interpretation_eligible']
 p['all_exclusion_reasons']=[';'.join([f for f in flags if not r[f]]+[c for c in knowncols if r[c]]+(['ec_other_assertion_hold'] if r['ec_other_assertion_hold'] else [])+(['ec_ambiguous_relation_hold'] if r['ec_ambiguous_relation_hold'] else [])) for r in p[flags+knowncols+['ec_other_assertion_hold','ec_ambiguous_relation_hold']].to_dict('records')]
 oldmaster=pd.read_csv(MASTER);oldids=set(oldmaster.pair_id);oldcand=set(oldmaster.loc[oldmaster.selection_role.ne('POSITIVE_CONTROL'),'pair_id'])
 p['in_original_512']=p.pair_id.isin(oldids);p['in_original_448_candidates']=p.pair_id.isin(oldcand)
 p.to_parquet(OUT/'FULL_276480_PAIR_AUDIT.parquet',index=False)
 review=p[p.expanded_rank50_selected].copy();log('Joint rank<=20: '+str(p.primary_joint_selected.sum())+'; rank<=50: '+str(len(review))+'; computing chemistry references')
 # Missing reference structures cannot support a similarity claim.
 review=annotate_support(review,strict[strict.parent_canonical_smiles.notna()])
 review['positive_chemical_support_ge04']=review.positive_max_tanimoto.ge(.4)
 review['close_negative_review']=review.negative_max_tanimoto.ge(.7)&review.negative_max_tanimoto.gt(review.positive_max_tanimoto)
 review['priority_multi_evidence']=review.primary_joint_selected&review.joint_r50_ot03_exact_genetic.eq(True)&review.positive_chemical_support_ge04&~review.close_negative_review
 review['release_status']='REVIEW_ONLY_NOT_RELEASED_FOR_EXPERIMENT'
 details=[]
 bytarget={t:g for t,g in mapped.groupby('target_chembl_id')}
 for r in review.itertuples():
  g=bytarget[r.target_chembl_id].copy();g['txgnn_rank']=rankcache[r.ligand_inchikey][g.ci];g=g[g.txgnn_rank.between(1,50)&g.overall_score.ge(.3)]
  for x in g.to_dict('records'):
   ci=x['ci'];details.append({'pair_id':r.pair_id,'drug_names':r.drug_names,'gene_symbol':r.gene_symbol,'txgnn_disease_id':ds.iloc[ci].id,'txgnn_disease_name':ds.iloc[ci].node_name,'txgnn_logit':float(values[valueidx[r.ligand_inchikey],ci]),'same_disease_exact_genetic':x['txgnn_mapping_scope']=='EXACT_SINGLE_NODE' and max(x['genetic_association_score'],x['somatic_mutation_score'])>=.1,**{k:x[k] for k in ['disease_id','disease_name','overall_score','txgnn_rank','txgnn_mapping_scope','genetic_association_score','somatic_mutation_score','known_drug_score','literature_score','datatype_scores_json','therapeutic_areas','source']}})
 detail=pd.DataFrame(details).sort_values(['pair_id','same_disease_exact_genetic','txgnn_rank','overall_score'],ascending=[True,False,True,False]);detail.to_csv(OUT/'JOINT_DISEASE_EVIDENCE.csv.gz',index=False)
 best=detail.drop_duplicates('pair_id')[['pair_id','disease_id','disease_name','overall_score','txgnn_rank','txgnn_mapping_scope','therapeutic_areas']].rename(columns={'disease_name':'review_disease','overall_score':'review_ot_score','txgnn_rank':'review_txgnn_rank'})
 review=review.merge(best,on='pair_id',validate='one_to_one').sort_values(['priority_multi_evidence','primary_joint_selected','joint_r50_ot03_exact_genetic','positive_chemical_support_ge04','binding_rank_384','positive_max_tanimoto'],ascending=[False,False,False,False,True,False])
 review.to_csv(OUT/'JOINT_RANK50_REVIEW.csv',index=False)
 primary=review[review.primary_joint_selected];primary.to_csv(OUT/'PRIMARY_JOINT_REVIEW.csv',index=False)
 priority=review[review.priority_multi_evidence];priority.to_csv(OUT/'PRIORITY_MULTI_EVIDENCE_REVIEW.csv',index=False)
 funnel=[];keep=pd.Series(True,index=p.index)
 for name,condition in [('full_space',keep.copy()),('identity_and_scope',p.identity_scope_pass),('chemistry_policy',p.chemistry_policy_pass),('known_relation_exclusion',~p.known_relation_excluded),('prior_relation_hold_exclusion',~p.prior_relation_review_hold),('TxGNN_interpretable',p.interpretation_eligible),('binding_rank_top20',p.binding_rank_384.le(20)),('TxGNN_top50_OT03_same_disease',p.joint_r50_ot03.eq(True))]:
  keep &=condition;g=p[keep];funnel.append({'stage':name,'pairs':len(g),'drugs':g.ligand_inchikey.nunique(),'targets':g.target_chembl_id.nunique()})
 funnel=pd.DataFrame(funnel);funnel.to_csv(OUT/'SEQUENTIAL_FUNNEL.csv',index=False)
 sensitivity=[]
 for br in [20,50,384]:
  for c in matrices:
   g=p[p.base_eligible&p.binding_rank_384.le(br)&p[c].eq(True)];sensitivity.append({'binding_rank_max':br,'criterion':c,'pairs':len(g),'drugs':g.ligand_inchikey.nunique(),'targets':g.target_chembl_id.nunique()})
 sensitivity=pd.DataFrame(sensitivity);sensitivity.to_csv(OUT/'THRESHOLD_SENSITIVITY.csv',index=False)
 overlap=pd.DataFrame([{'reason':c,'pairs':int(p[c].sum())} for c in knowncols+['ec_other_assertion_hold','ec_ambiguous_relation_hold']]);overlap.to_csv(OUT/'EXCLUSION_COUNTS_OVERLAPPING.csv',index=False)
 original=oldmaster.copy();original['in_scored384']=original.target_chembl_id.isin(tids)
 comparison=original[['experiment_id','pair_id','selection_role','in_scored384']].merge(p[['pair_id','screen_status','base_eligible','binding_rank_384','primary_joint_selected','joint_r50_ot03','all_exclusion_reasons']],on='pair_id',how='left',validate='one_to_one');comparison.to_csv(OUT/'ORIGINAL_512_REAUDIT.csv',index=False)
 targetsummary=primary.groupby(['target_chembl_id','gene_symbol','assay_lane']).agg(pairs=('pair_id','size'),drugs=('ligand_inchikey','nunique'),chem_support=('positive_chemical_support_ge04','sum'),priority=('priority_multi_evidence','sum')).reset_index().sort_values('pairs',ascending=False);targetsummary.to_csv(OUT/'PRIMARY_TARGET_COUNTS.csv',index=False)
 summary={'full_pairs':len(p),'txgnn_interpretable_drugs':len(mappedkeys),'ot_targets':ot.target_chembl_id.nunique(),'primary_pairs':len(primary),'primary_drugs':primary.ligand_inchikey.nunique(),'primary_targets':primary.target_chembl_id.nunique(),'primary_new_vs_old_candidates':int((~primary.in_original_448_candidates).sum()),'primary_positive_chem_ge04':int(primary.positive_chemical_support_ge04.sum()),'priority_pairs':len(priority),'priority_drugs':priority.ligand_inchikey.nunique(),'priority_targets':priority.target_chembl_id.nunique(),'rank50_pairs':len(review),'rank50_targets':review.target_chembl_id.nunique(),'old_candidate_pairs_in384':int((original.in_scored384&original.selection_role.ne('POSITIVE_CONTROL')).sum()),'old_targets_outside384':original.loc[~original.in_scored384,'target_chembl_id'].nunique(),'status_counts':p.screen_status.value_counts().to_dict()}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
 with pd.ExcelWriter(OUT/'JOINT_720X384_REVIEW.xlsx') as w:
  show=['drug_names','gene_symbol','binding_rank_384','review_disease','review_txgnn_rank','review_ot_score','txgnn_mapping_scope','positive_max_tanimoto','negative_max_tanimoto','priority_multi_evidence','in_original_448_candidates','ligand_inchikey','target_chembl_id','release_status']
  priority[show].to_excel(w,sheet_name='优先人工核验',index=False);primary[show].to_excel(w,sheet_name='主筛选Top20',index=False);review[show].to_excel(w,sheet_name='放宽Top50',index=False);funnel.to_excel(w,sheet_name='筛选漏斗',index=False);sensitivity.to_excel(w,sheet_name='阈值敏感性',index=False);targetsummary.to_excel(w,sheet_name='靶点分布',index=False);comparison.to_excel(w,sheet_name='原512复审',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 checks={'full_cartesian_unique':len(p)==720*384 and p.pair_id.is_unique,'each_drug384':bool(p.groupby('ligand_inchikey').size().eq(384).all()),'all_rank_denominators384':bool(p.groupby('ligand_inchikey').binding_rank_384.max().eq(384).all()),'unmapped_joint_null':bool(p.loc[~p.interpretation_eligible,list(matrices)].isna().all().all()),'selected_all_gates':bool(primary.base_eligible.all() and primary.binding_rank_384.le(20).all() and primary.joint_r50_ot03.eq(True).all()),'selected_no_known':bool(~primary[knowncols].any().any()),'all_review_pairs_have_disease_trace':set(review.pair_id)==set(detail.pair_id),'priority_same_disease_exact_genetic':set(priority.pair_id)<=set(detail.loc[detail.same_disease_exact_genetic,'pair_id']),'disease_exclusions_rechecked':all(mask[valueidx[r.ligand_inchikey],int(x)]==0 for r in review.itertuples() for x in detail.loc[detail.pair_id.eq(r.pair_id),'txgnn_disease_id'].map({v:i for i,v in enumerate(ds.id)})),'original_master_unchanged':sha(MASTER)==EXPECTED_MASTER}
 (OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks,'no_training':True,'no_new_neural_inference':True,'experimental_validation':False},indent=2));assert all(checks.values()),checks
 sources=[old.RANK_PATH,old.STRICT_PATH,old.TRAINING_PATH,old.KIRHUB_PATH,old.FDA_PATH,ROOT/'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv',BASE/'DRUG_REGISTRY_720.csv',BASE/'DRUG_COVERAGE_FINAL_720.csv',BASE/'TXGNN_SCORE_DRUG_ORDER.csv',BASE/'TXGNN_SCORE_DISEASE_ORDER.csv',BASE/'TXGNN_INDICATION_LOGITS.npy',BASE/'TXGNN_EXCLUSION_MASK.npy',BASE/'OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet',BASE/'ECKG_RELEVANT_RELATION_AUDIT.parquet',MASTER,Path(__file__),Path(old.__file__),ROOT/'scripts/design_spr64_512_20260909.py']
 sources.append(COMPONENTS)
 (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'sources_sha256':{str(f.relative_to(ROOT)):sha(f) for f in sources},'outputs_sha256':{str(f.relative_to(OUT)):sha(f) for f in OUT.iterdir() if f.is_file() and f.name!='SOURCE_MANIFEST.json'}},indent=2))
 log(json.dumps(summary));print(funnel.to_string(index=False),flush=True)
if __name__=='__main__':main()
