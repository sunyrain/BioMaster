"""Unified, source-audited portfolio selection; no empirical success-probability claim."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from design_joint384_agent_review_20260909 import control_pool,STRICT,ORIGINAL
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint384_comprehensive_20260909';OLD=ROOT/'outputs/joint384_design_20260909';CROSS=ROOT/'outputs/joint_cross_indication_20260909';BASE=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def areas(x):return {s.strip() for s in str(x).split(';') if s.strip() and s!='nan'}
def solve(p,ntarget,drugcap=4,targetcap=10,cross_weight=2.,chem_weight=4.):
 p=p.reset_index(drop=True);ts=sorted(p.target_chembl_id.unique());ds=sorted(p.ligand_inchikey.unique());n=len(p);ti={x:n+i for i,x in enumerate(ts)};di={x:n+len(ts)+i for i,x in enumerate(ds)};N=n+len(ts)+len(ds);rr=[];cc=[];vv=[];lo=[];hi=[]
 def add(co,a,b):
  k=len(lo);lo.append(a);hi.append(b)
  for j,v in co.items():rr.append(k);cc.append(j);vv.append(v)
 add({i:1 for i in range(n)},384,384);add({i:1 for i in ti.values()},ntarget,ntarget);add({i:1 for i in di.values()},128,len(ds))
 for t,g in p.groupby('target_chembl_id'):
  co={int(i):1 for i in g.index};add({**co,ti[t]:-targetcap},-np.inf,0);add({**co,ti[t]:-1},0,np.inf)
 for d,g in p.groupby('ligand_inchikey'):
  co={int(i):1 for i in g.index};add({**co,di[d]:-drugcap},-np.inf,0);add({**co,di[d]:-1},0,np.inf)
 for _,g in p.groupby('connectivity_key'):add({int(i):1 for i in g.index},0,drugcap)
 # Unified evidence utility. Agent KEEP/LOW and old priority labels receive no points.
 quality=(chem_weight*p.usable_chemical_support_ge04.astype(int)+1.5*(p.positive_max_tanimoto.ge(.3)&p.positive_max_tanimoto.lt(.4)&~p.chemical_reference_transfer_hold).astype(int)+3*p.any_exact_genetic.astype(int)+.5*p.any_exact_disease.astype(int)+np.select([p.binding_rank_384.le(5),p.binding_rank_384.le(10)],[2.,1.5],default=1.)-.75*p.chemical_risk_points-1.5*p.opposing_near_reference.astype(int)+cross_weight*p.has_label_based_cross_area.astype(int)+1.*p.cross_area_exact_genetic.astype(int))
 # Strong cross-area mechanistic hypotheses take priority over weaker combinations.
 objective=quality+20*p.integrated_priority.astype(int)
 c=np.zeros(N);c[:n]=-objective.to_numpy();c[n+len(ts):]=-.001
 A=coo_matrix((vv,(rr,cc)),shape=(len(lo),N)).tocsc();res=milp(c,integrality=np.ones(N),bounds=Bounds(np.zeros(N),np.ones(N)),constraints=LinearConstraint(A,lo,hi),options={'time_limit':50,'mip_rel_gap':.001})
 gap=getattr(res,'mip_gap',None);meta={'target_count':ntarget,'drug_cap':drugcap,'target_cap':targetcap,'cross_weight':cross_weight,'chem_weight':chem_weight,'solver_status':int(res.status),'solver_message':res.message,'mip_gap':None if gap is None else float(gap)}
 if res.x is None:return None,meta
 x=np.rint(res.x);v=A@x;assert np.all(v>=np.array(lo)-1e-6) and np.all(v<=np.array(hi)+1e-6)
 chosen=p.loc[np.flatnonzero(x[:n])].copy();chosen['evidence_utility_unvalidated']=quality.loc[chosen.index];chosen['optimization_utility_unvalidated']=objective.loc[chosen.index]
 meta.update(candidates=len(chosen),drugs=int(chosen.ligand_inchikey.nunique()),integrated_priority=int(chosen.integrated_priority.sum()),chemical_ge04=int(chosen.usable_chemical_support_ge04.sum()),cross_area_pairs=int(chosen.has_label_based_cross_area.sum()),exact_genetic_pairs=int(chosen.any_exact_genetic.sum()),evidence_utility=float(chosen.evidence_utility_unvalidated.sum()),objective_utility=float(chosen.optimization_utility_unvalidated.sum()),legacy_low_priority=int(chosen.decision.eq('LOW_PRIORITY').sum()))
 return chosen,meta

def main():
 OUT.mkdir(exist_ok=True);oldhash=sha(OLD/'CANDIDATES_384.csv');originalhash=sha(ORIGINAL)
 (OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':False,'status':'BUILDING'}))
 r=pd.read_csv(OLD/'ALL_574_AGENT_REVIEW.csv');old=pd.read_csv(OLD/'CANDIDATES_384.csv');origin=pd.concat([pd.read_csv(OUT/f'ORIGIN_REVIEWED_PART_{i}.csv') for i in range(1,4)],ignore_index=True)
 deep_path=ROOT/'outputs/best_candidate_synthesis_20260909/TARGETED24_DEEP_REVIEW.csv'
 deep=pd.read_csv(deep_path)[['pair_id','recommendation','reason','sources','critical_gap']].rename(columns={'recommendation':'deep_recommendation','reason':'deep_reason','sources':'deep_sources','critical_gap':'deep_critical_gap'})
 r=r.merge(deep,on='pair_id',how='left',validate='one_to_one')
 raw=pd.read_csv(OUT/'ALL574_RAW_ACTIVITY_NOVELTY_REAUDIT.csv')[['pair_id','raw_activity_rows','raw_high_confidence_direct_target_annotation','novelty_reaudit_status']]
 r=r.merge(raw,on='pair_id',validate='one_to_one')
 assert len(origin)==266 and origin.drug_names.is_unique
 origin['origin_scope_note']='Reviewed retrieved official-label/regulatory material; not a complete worldwide approval registry; formulation/combination/historical limits retained.'
 origin.to_csv(OUT/'ORIGIN_REVIEWED_266.csv',index=False)
 r=r.merge(origin,on='drug_names',how='left',validate='many_to_one')
 registry=pd.read_csv(BASE/'DRUG_REGISTRY_720.csv',low_memory=False)
 r=r.merge(registry[['ligand_inchikey','normalized_active_moiety_name','ingredient_name_normalized','gsrs_display_name']].rename(columns={'normalized_active_moiety_name':'modeled_entity_name','ingredient_name_normalized':'fda_precise_ingredient_name','gsrs_display_name':'fda_gsrs_substance_name'}),on='ligand_inchikey',validate='many_to_one')
 r['merged_source_aliases']=r.drug_names.str.contains(';',regex=False)
 h=pd.read_csv(CROSS/'ALL_574_COMMON_DISEASE_CROSS_AREA_AUDIT.csv.gz').drop(columns=['baseline_areas','baseline_evidence_scope','baseline_officially_reviewed','official_indication_sources','cross_indication_status','cross_priority_eligible'],errors='ignore')
 h=h.merge(origin[['drug_names','origin_areas','origin_status','origin_summary','source_urls','scope_limits']],on='drug_names',how='left',validate='many_to_one')
 h['new_cross_area']=[bool(areas(a) and areas(b) and not(areas(a)&areas(b))) and state in ['LABEL_REVIEWED','EXTERNAL_REVIEWED'] and scope=='EXACT_SINGLE_NODE' for a,b,state,scope in zip(h.origin_areas,h.candidate_macro_areas,h.origin_status,h.txgnn_mapping_scope)]
 h['same_or_overlapping_area']=[bool(areas(a)&areas(b)) for a,b in zip(h.origin_areas,h.candidate_macro_areas)]
 h['oncology_to_oncology']=['ONCOLOGY' in areas(a) and 'ONCOLOGY' in areas(b) for a,b in zip(h.origin_areas,h.candidate_macro_areas)]
 h['disease_policy_hold']=h.drug_names.isin(['raloxifene','bazedoxifene'])&h.disease_name.str.contains('endometriosis',case=False,na=False)
 h['exact_genetic_or_somatic']=h.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE')&(h.genetic_association_score.ge(.1)|h.somatic_mutation_score.ge(.1))
 h['hypothesis_interpretation']='Drug-target binding and treatment direction unproven; cross-area classification uses reviewed label scope, not exhaustive approvals.'
 h.to_csv(OUT/'ALL_COMMON_DISEASE_REAUDIT.csv.gz',index=False)
 summaries=[];bestrows=[]
 for pid,g in h.groupby('pair_id'):
  g=g[~g.disease_policy_hold].copy();cross=g[g.new_cross_area];exact=g[g.txgnn_mapping_scope.eq('EXACT_SINGLE_NODE')]
  summaries.append({'pair_id':pid,'allowed_disease_count':len(g),'has_label_based_cross_area':len(cross)>0,'cross_area_exact_genetic':bool((cross.exact_genetic_or_somatic).any()),'cross_area_germline':bool(cross.genetic_association_score.ge(.1).any()),'cross_area_somatic':bool(cross.somatic_mutation_score.ge(.1).any()),'any_exact_genetic':bool(g.exact_genetic_or_somatic.any()),'any_exact_disease':len(exact)>0,'has_oncology_to_oncology':bool(g.oncology_to_oncology.any()),'any_same_area':bool(g.same_or_overlapping_area.any())})
  if len(g):
   g=g.sort_values(['new_cross_area','exact_genetic_or_somatic','txgnn_rank','overall_score'],ascending=[False,False,True,False]);x=g.iloc[0]
   bestrows.append({'pair_id':pid,'recommended_disease_id':x.disease_id,'recommended_disease':x.disease_name,'recommended_disease_areas':x.candidate_macro_areas,'recommended_disease_is_cross_area':bool(x.new_cross_area),'recommended_txgnn_rank':int(x.txgnn_rank),'recommended_ot_score':float(x.overall_score),'recommended_germline_score':float(x.genetic_association_score),'recommended_somatic_score':float(x.somatic_mutation_score),'recommended_mapping_scope':x.txgnn_mapping_scope})
 r=r.merge(pd.DataFrame(summaries),on='pair_id',validate='one_to_one').merge(pd.DataFrame(bestrows),on='pair_id',how='left',validate='one_to_one')
 # Entity-level findings apply to every target, not only the deeply reviewed pair.
 # Unlabelled iobenguane must not inherit an I-131 label; deacetylated ulipristal
 # must not inherit ulipristal acetate's approved-product identity.
 r['new_identity_hold']=r.drug_names.eq('3-iodobenzylguanidine')|r.ligand_inchikey.eq('HKDLNTKNLJPAIY-WKWWZUSTSA-N')
 r['disease_investment_hold']=r.allowed_disease_count.eq(0)
 r['opposing_near_reference']=r.negative_max_tanimoto.ge(.4)&r.negative_max_tanimoto.gt(r.positive_max_tanimoto)
 r['raw_prior_activity_hold']=r.raw_activity_rows.gt(0)
 r['deep_review_investment_hold']=r.deep_recommendation.eq('DO_NOT_PRIORITIZE')
 reference_cases={('pomalidomide','MIF'),('fentanyl','CA1'),('fentanyl','CA7'),('baricitinib','HDAC1'),('baricitinib','HDAC6'),('baricitinib','HDAC11'),('mirdametinib','PIK3CD'),('vismodegib','TYK2')}
 r['chemical_reference_transfer_hold']=[(d,g) in reference_cases for d,g in zip(r.drug_names,r.gene_symbol)]
 r['usable_chemical_support_ge04']=r.positive_chemical_support_ge04&~r.chemical_reference_transfer_hold
 r['eligible_after_comprehensive_review']=r.decision.isin(['KEEP_REVIEW','LOW_PRIORITY'])&~r.close_negative_review&~r.new_identity_hold&~r.disease_investment_hold&~r.raw_prior_activity_hold&~r.deep_review_investment_hold
 r['integrated_priority']=r.eligible_after_comprehensive_review&r.has_label_based_cross_area&r.cross_area_exact_genetic&r.usable_chemical_support_ge04&~r.opposing_near_reference
 # These independent dimensions are easier to interpret than agent-specific adjectives.
 r['binding_reference_tier']=np.select([r.positive_chemical_support_ge04,r.positive_max_tanimoto.ge(.3)],['POSITIVE_NEIGHBOR_GE04','POSITIVE_NEIGHBOR_03_TO04'],default='LIMITED_POSITIVE_CHEMICAL_SUPPORT')
 r['disease_evidence_tier']=np.select([r.cross_area_exact_genetic,r.has_label_based_cross_area,r.any_exact_genetic],['CROSS_AREA_EXACT_GENETIC_OR_SOMATIC','CROSS_AREA_NO_STRONG_GENETIC_SOURCE','SAME_AREA_EXACT_GENETIC_OR_SOMATIC'],default='WEAKER_OR_MERGED_DISEASE_CONTEXT')
 r['in_previous384']=r.pair_id.isin(old.pair_id)
 r.to_csv(OUT/'FULL574_COMPREHENSIVE_AUDIT.csv',index=False)
 p=r[r.eligible_after_comprehensive_review].copy();assert len(p)>=384
 solutions={};stats=[]
 for count,cap in [(64,10),(80,10),(96,10),(96,12),(112,10),(112,12)]:
  solution,stat=solve(p,count,targetcap=cap);stats.append(stat)
  if solution is not None:solutions[(count,cap)]=solution
 pd.DataFrame(stats).to_csv(OUT/'TARGET_BUDGET_COMPARISON.csv',index=False)
 valid=[x for x in stats if 'candidates' in x];assert valid
 best_priority=max(x['integrated_priority'] for x in valid);best_quality=max(x['evidence_utility'] for x in valid if x['integrated_priority']==best_priority);best_chemical=max(x['chemical_ge04'] for x in valid if x['integrated_priority']==best_priority)
 # Prefer fewer target preparations if all best-supported hypotheses and >=95% of
 # the best defined evidence utility can be retained. 95% is a resource policy, not CI.
 acceptable=[x for x in valid if x['integrated_priority']==best_priority and x['chemical_ge04']==best_chemical and x['evidence_utility']>=.95*best_quality]
 chosenstat=min(acceptable,key=lambda x:(x['target_count'],x['target_cap']));target_count=chosenstat['target_count'];target_cap=chosenstat['target_cap'];selected=solutions[(target_count,target_cap)].copy();initial_ids=set(selected.pair_id)
 sensitivity=[];sets=[]
 for name,cw,chw in [('binding_emphasis',1.,6.),('cross_area_emphasis',3.,4.)]:
  alt,st=solve(p,target_count,targetcap=target_cap,cross_weight=cw,chem_weight=chw);st['scenario']=name
  if alt is not None:st['overlap_with_recommended']=len(initial_ids&set(alt.pair_id));sets.append(set(alt.pair_id));alt[['pair_id','drug_names','gene_symbol']].to_csv(OUT/f'SENSITIVITY_{name}.csv',index=False)
  sensitivity.append(st)
 pd.DataFrame(sensitivity).to_csv(OUT/'WEIGHT_SENSITIVITY.csv',index=False)
 selected['weight_sensitivity_selected_count']=selected.pair_id.map(lambda pid:1+sum(pid in ids for ids in sets))
 selected['recommendation_band']=np.select([selected.integrated_priority,selected.usable_chemical_support_ge04|selected.any_exact_genetic],['PRIORITY_CROSS_AREA_MECHANISM_REVIEW','ADDITIONAL_SOURCE_OR_CHEMICAL_SUPPORT'],default='EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT')
 selected=selected.sort_values(['integrated_priority','evidence_utility_unvalidated','binding_rank_384','gene_symbol','drug_names'],ascending=[False,False,True,True,True]).reset_index(drop=True)
 selected['candidate_id']=[f'C384-{i:03d}' for i in range(1,385)];selected['release_status']='COMPREHENSIVE_REVIEW_NOT_LAB_RELEASE';selected['selection_role']='DISCOVERY_CANDIDATE_384_CONTROLS_EXTRA'
 selected.to_csv(OUT/'RECOMMENDED_CANDIDATES_384.csv',index=False)
 refs=control_pool(pd.read_csv(STRICT),set(selected.target_chembl_id));refs.to_csv(OUT/'REFERENCE_CONTROLS_EXTRA.csv',index=False)
 roster=selected.groupby(['target_chembl_id','gene_symbol','assay_lane']).agg(candidates=('pair_id','size'),drugs=('ligand_inchikey','nunique'),cross_area=('has_label_based_cross_area','sum'),chemical_ge04=('usable_chemical_support_ge04','sum'),priority=('integrated_priority','sum')).reset_index().merge(refs[['target_chembl_id','drug_names','reference_endpoint_category']].rename(columns={'drug_names':'proposed_reference'}),on='target_chembl_id',validate='one_to_one');roster['construct_status']='LAB_BOUNDARIES_ISOFORM_ACTIVITY_AND_PROCUREMENT_CONFIRMATION_PENDING';roster.to_csv(OUT/'TARGET_ROSTER.csv',index=False)
 selectedids=set(selected.pair_id);r['selected_in_recommended384']=r.pair_id.isin(selectedids)
 r['final_disposition']=np.select([r.selected_in_recommended384,r.decision.eq('EXCLUDE'),r.decision.eq('HOLD'),r.new_identity_hold,r.disease_investment_hold,r.raw_prior_activity_hold,r.deep_review_investment_hold,r.close_negative_review],['SELECTED','PRIOR_EXCLUSION','PRIOR_HOLD','MODEL_APPROVED_ENTITY_IDENTITY_HOLD','DISEASE_INVESTMENT_HOLD','RAW_PRIOR_ACTIVITY_NOT_NOVEL','DEEP_REVIEW_INVESTMENT_HOLD','NEGATIVE_NEIGHBOR_REVIEW_HOLD'],default='RESERVE_NOT_SELECTED_BY_COMPREHENSIVE_POLICY')
 r.to_csv(OUT/'ALL574_FINAL_DISPOSITION.csv',index=False)
 change=r[r.in_previous384|r.selected_in_recommended384].copy();change['change']=np.select([change.in_previous384&change.selected_in_recommended384,change.in_previous384],['RETAINED','REMOVED_FROM_NEW_VERSION'],default='ADDED_TO_NEW_VERSION');change.to_csv(OUT/'PREVIOUS384_CHANGE_LEDGER.csv',index=False)
 pd.concat([selected,refs],ignore_index=True,sort=False).to_csv(OUT/'CANDIDATES_AND_EXTRA_CONTROLS.csv',index=False)
 with pd.ExcelWriter(OUT/'RECOMMENDED384_COMPREHENSIVE_REVIEW.xlsx') as w:
  display=['candidate_id','modeled_entity_name','drug_names','gene_symbol','recommendation_band','origin_summary','origin_areas','recommended_disease','recommended_disease_is_cross_area','binding_rank_384','positive_max_tanimoto','negative_max_tanimoto','recommended_txgnn_rank','recommended_ot_score','recommended_germline_score','recommended_somatic_score','decision','reason','deep_recommendation','deep_reason','deep_sources','chemical_reference_transfer_hold','assay_notes','scope_limits','source_urls','ligand_inchikey','target_chembl_id','weight_sensitivity_selected_count','release_status']
  selected[display].to_excel(w,sheet_name='综合推荐384',index=False);refs.to_excel(w,sheet_name='另计参考对照',index=False);roster.to_excel(w,sheet_name='靶点与配额',index=False);change[['drug_names','gene_symbol','change','final_disposition','has_label_based_cross_area','binding_reference_tier','disease_evidence_tier','reason']].to_excel(w,sheet_name='与旧384比较',index=False);origin.to_excel(w,sheet_name='266药原用途核验',index=False);pd.DataFrame(stats).to_excel(w,sheet_name='靶点预算比较',index=False);pd.DataFrame(sensitivity).to_excel(w,sheet_name='权重敏感性',index=False)
  for ws in w.book.worksheets:ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
 summary={'candidate_pairs':len(selected),'drugs':selected.ligand_inchikey.nunique(),'targets':selected.target_chembl_id.nunique(),'extra_controls':len(refs),'total_pairs_including_controls':384+len(refs),'origin_review_drugs':len(origin),'origin_status_counts':origin.origin_status.value_counts().to_dict(),'eligible_pairs':len(p),'cross_area_pairs':int(selected.has_label_based_cross_area.sum()),'cross_area_germline_pairs':int(selected.cross_area_germline.sum()),'cross_area_somatic_pairs':int(selected.cross_area_somatic.sum()),'chemical_support_ge04':int(selected.usable_chemical_support_ge04.sum()),'raw_chemical_similarity_ge04':int(selected.positive_chemical_support_ge04.sum()),'exact_genetic_or_somatic_pairs':int(selected.any_exact_genetic.sum()),'priority_pairs':int(selected.integrated_priority.sum()),'priority_names':(selected.loc[selected.integrated_priority,'modeled_entity_name']+'–'+selected.loc[selected.integrated_priority,'gene_symbol']).tolist(),'bands':selected.recommendation_band.value_counts().to_dict(),'legacy_agent_decisions':selected.decision.value_counts().to_dict(),'same_old_pairs':len(selectedids&set(old.pair_id)),'replaced_pairs':384-len(selectedids&set(old.pair_id)),'stable_in_three_weight_settings':int(selected.weight_sensitivity_selected_count.eq(3).sum()),'reference_categories':refs.reference_endpoint_category.value_counts().to_dict(),'dispositions_574':r.final_disposition.value_counts().to_dict(),'chosen_resource_policy':'Smallest feasible target-budget scenario retaining max integrated-priority and usable-chemical-support counts and >=95% of best declared utility, then lower per-target cap; not measured performance.'}
 (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 checks={'exactly384_unique':len(selected)==384 and selected.pair_id.is_unique,'original384_unchanged':sha(OLD/'CANDIDATES_384.csv')==oldhash,'original512_unchanged':sha(ORIGINAL)==originalhash,'all574_accounted':len(r)==574 and r.pair_id.is_unique,'all_old_gates_retained':selected.base_eligible.all() and selected.primary_joint_selected.all(),'no_prior_hold_or_exclusion':selected.decision.isin(['KEEP_REVIEW','LOW_PRIORITY']).all(),'no_new_identity_or_disease_hold':not(selected.new_identity_hold|selected.disease_investment_hold).any(),'no_raw_prior_activity':not selected.raw_prior_activity_hold.any(),'no_deep_review_investment_hold':not selected.deep_review_investment_hold.any(),'cross_area_has_reviewed_origin':selected.loc[selected.has_label_based_cross_area,'origin_status'].isin(['LABEL_REVIEWED','EXTERNAL_REVIEWED']).all(),'cross_area_best_disease_exact':selected.loc[selected.has_label_based_cross_area,'recommended_mapping_scope'].eq('EXACT_SINGLE_NODE').all(),'no_cancer_to_cancer_as_cross':all(not('ONCOLOGY' in areas(x.origin_areas) and 'ONCOLOGY' in areas(x.recommended_disease_areas)) for x in selected[selected.has_label_based_cross_area].itertuples()),'one_reference_per_target':len(refs)==target_count and refs.target_chembl_id.is_unique,'no_reference_candidate_overlap':not(set(refs.pair_id)&selectedids),'drug_and_target_caps':selected.groupby('ligand_inchikey').size().max()<=4 and selected.groupby('target_chembl_id').size().max()<=target_cap,'priority_counts_match':selected.integrated_priority.sum()==chosenstat['integrated_priority']}
 checks={k:bool(v) for k,v in checks.items()};(OUT/'VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks,'experimental_validation':False},indent=2));assert all(checks.values()),checks
 inputs=[OLD/'ALL_574_AGENT_REVIEW.csv',OLD/'CANDIDATES_384.csv',CROSS/'ALL_574_COMMON_DISEASE_CROSS_AREA_AUDIT.csv.gz',BASE/'DRUG_REGISTRY_720.csv',STRICT,ORIGINAL,deep_path,OUT/'ALL574_RAW_ACTIVITY_NOVELTY_REAUDIT.csv',OUT/'ALL574_RAW_CHEMBL_ACTIVITY_RECORDS.csv.gz',Path(__file__),ROOT/'scripts/design_joint384_agent_review_20260909.py',ROOT/'scripts/audit_comprehensive_raw_activities_20260909.py',*[OUT/f'ORIGIN_REVIEWED_PART_{i}.csv' for i in range(1,4)]]
 report_artifacts={'MANIFEST.json','REPORT_MANIFEST.json','FAIR_OLD_NEW_COMPARISON.csv','REPLACEMENT_REASON_COUNTS.csv','STAGED_EXECUTION_REVIEW.csv','INDEPENDENT_VALIDATION.json','LABEL_CACHE_MANIFEST.json','AGENT384_BEFORE_COMPREHENSIVE.md','DESIGN_BEFORE_COMPREHENSIVE.md'}
 (OUT/'MANIFEST.json').write_text(json.dumps({'input_sha256':{str(f.relative_to(ROOT)):sha(f) for f in inputs},'output_sha256':{f.name:sha(f) for f in OUT.iterdir() if f.is_file() and f.name not in report_artifacts}},indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
