"""Independent computational design; never writes the frozen SPR32 package.

Produces review-ready candidates and separate blinded worksheets, not lab release.
"""
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import build_retargetmap_spr512_ours_frozen_v2 as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/retargetmap_spr64_design_20260909'
REPLACEMENTS={'JAK1':'JAK2','PTGS1':'CA2','VDR':'RXRA'}
QUOTAS={'OUR_FROZEN_MODEL_HIGH':5,'OUR_FROZEN_MODEL_INTERMEDIATE':1,'OUR_FROZEN_MODEL_LOW_BACKGROUND':1}
CONTROLS={
 'DPP4':'LINAGLIPTIN','HSD11B1':'BMS-823778 FREE BASE','HMGCR':'ATORVASTATIN',
 'ACHE':'HUPERZINE A','BCHE':'ETHOPROPAZINE','MAOA':'HARMINE','MAOB':'LAZABEMIDE',
 'COMT':'OPICAPONE','FAAH':'MK-3168','PTGS2':'CELECOXIB','CA2':'DORZOLAMIDE',
 'ALOX5':'ZILEUTON','EPHX2':'GSK2256294','PDE4D':'ROLIPRAM','PDE9A':'TOVINONTRINE',
 'PDE10A':'MK-8189','PARP1':'VELIPARIB','PARP2':'TALAZOPARIB','EZH2':'GSK2816126',
 'DHODH':'TERIFLUNOMIDE','CTSK':'ODANACATIB','CTSS':'RELACATIB','PPARD':'GW501516',
 'RXRA':'ALITRETINOIN','MMP13':'PRINOMASTAT','ESR2':'ERTEBEREL','F2':'ARGATROBAN',
 'F10':'OTAMIXABAN','JAK2':'RUXOLITINIB','BTK':'FENEBRUTINIB','MAPK14':'DORAMAPIMOD','ESR1':'LASOFOXIFENE'}
CONSTRUCTS={
 'DPP4':'可溶性胞外催化域；去跨膜段；确认二聚、糖基化与活性',
 'HSD11B1':'可溶性催化域；明确NADP(H)与构建状态',
 'HMGCR':'人源可溶性催化域；不能将全长膜蛋白特征等同于该构建；明确辅因子',
 'ACHE':'人源成熟催化域；明确糖基化、寡聚状态与固定化后活性',
 'BCHE':'人源成熟催化域；明确糖基化与寡聚状态',
 'MAOA':'人源FAD结合活性酶；膜结合体系与固定化后活性需专门确认',
 'MAOB':'人源FAD结合活性酶；膜结合体系与固定化后活性需专门确认',
 'COMT':'明确使用可溶COMT或膜型对应催化域；记录SAM/Mg2+状态',
 'FAAH':'膜结合酶或已验证去膜锚活性构建；明确去污剂环境与固定化后活性',
 'PTGS2':'人源活性COX-2；血红素与膜相关蛋白环境需确认，不能默认普通可溶酶',
 'CA2':'可溶性全长胞内酶；保留Zn2+催化位点',
 'ALOX5':'人源活性5-LOX；确认铁、Ca2+及脂质环境；区分ALOX5与FLAP抑制',
 'EPHX2':'人源可溶性环氧化物水解酶；明确催化域或全长构建',
 'PDE4D':'指定异构体与催化域；记录调控域是否保留和金属离子状态',
 'PDE9A':'人源催化域；确认金属离子状态',
 'PDE10A':'人源催化域；明确调控域及金属离子状态',
 'PARP1':'注明催化域或全长；全长时明确DNA与活化状态',
 'PARP2':'注明催化域或全长；明确DNA与活化状态',
 'EZH2':'优先经功能验证的PRC2复合物；EZH2单独构建不能默认具备正常催化状态',
 'DHODH':'人源FMN结合催化构建；明确膜锚截除、电子受体与蛋白环境',
 'CTSK':'成熟活性催化域；记录激活、pH及还原状态',
 'CTSS':'成熟活性催化域；记录激活、pH及还原状态',
 'PPARD':'PPAR-delta配体结合域；注明是否有共调节因子',
 'RXRA':'RXR-alpha配体结合域；注明二聚与共调节因子状态',
 'MMP13':'成熟催化域；确认Zn2+与激活状态',
 'ESR2':'ER-beta配体结合域；注明激素去除与共调节因子状态',
 'F2':'成熟活性人凝血酶，不用未激活凝血酶原替代',
 'F10':'成熟活化人凝血因子Xa，不用未活化因子X替代',
 'JAK2':'明确JH1催化域；记录磷酸化与ATP/Mg2+状态',
 'BTK':'人源激酶催化域；明确磷酸化与ATP/Mg2+状态',
 'MAPK14':'人源p38-alpha催化构建；明确活化/磷酸化状态',
 'ESR1':'ER-alpha配体结合域；注明激素去除与共调节因子状态'}
SPECIAL={'MAOA','MAOB','FAAH','PTGS2','ALOX5','EZH2','DHODH','F2','F10','COMT'}
EXCEPTIONS={
 'ALOX5':('ZILEUTON','https://www.guidetopharmacology.org/GRAC/FamilyDisplayForward?familyId=271&objId=1390'),
 'PPARD':('GW501516','https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=2687&tab=biology')}


def write_json(name,value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def configure():
 draft=pd.read_csv(ROOT/'outputs/retargetmap_spr64_target_review_20260909/TARGET64_REVIEW_DRAFT.csv')
 genes=[REPLACEMENTS.get(g,g) for g in draft.gene_symbol]
 core=[REPLACEMENTS.get(r.gene_symbol,r.gene_symbol) for r in draft.itertuples() if r.rank_source=='frozen_validation_rank_384' or r.gene_symbol in REPLACEMENTS]
 old.BENCHMARK_TARGETS=core;old.DISCOVERY_TARGETS=[g for g in genes if g not in core]
 old.OURS_ONLY_QUOTAS=QUOTAS;old.CONTROL_NAMES.update(CONTROLS);old.CONSTRUCTS.update(CONSTRUCTS)
 assert len(set(genes))==64
 return genes


def prepare():
 p,t,strict,_=old.build_candidate_pool()
 p.to_csv(OUT/'FINAL64_LOCAL_POOL.csv.gz',index=False)
 p,b,g=old.add_external_evidence(p)
 p.to_csv(OUT/'FINAL64_EXTERNAL_POOL.csv.gz',index=False)
 t.to_csv(OUT/'FINAL64_TARGETS.csv',index=False)
 b.to_csv(OUT/'FINAL64_BINDINGDB_EXCLUDED.csv',index=False)
 g.to_csv(OUT/'FINAL64_GTOPDB_EXCLUDED.csv',index=False)
 write_json('PREPARATION_IDENTITY.json',{'source_script_sha256':old.sha256(Path(old.__file__)),'genes':old.BENCHMARK_TARGETS+old.DISCOVERY_TARGETS,
     'sources':{str(p.relative_to(ROOT)):old.sha256(p) for p in [old.RANK_PATH,old.SCORE_PATH,old.REGISTRY_PATH,old.STRICT_PATH,old.TRAINING_PATH,old.FDA_PATH,old.KIRHUB_PATH,old.BINDINGDB_ARTICLES,old.BINDINGDB_PUBCHEM,old.GTOPDB_DIR/'ligands.csv',old.GTOPDB_DIR/'interactions.csv']},
     'cache_sha256':old.sha256(OUT/'FINAL64_EXTERNAL_POOL.csv.gz')})


def optimize(pool,genes):
 p=pool.reset_index(drop=True);drugs=sorted(p.ligand_inchikey.unique());ys={d:len(p)+i for i,d in enumerate(drugs)}
 n=len(p)+len(drugs);rr=[];cc=[];vv=[];lo=[];hi=[]
 def add(coef,lower,upper):
  k=len(lo);lo.append(lower);hi.append(upper)
  for j,v in coef.items():rr.append(k);cc.append(j);vv.append(v)
 for g in genes:
  for role,count in QUOTAS.items():
   ids=p.index[p.gene_symbol.eq(g)&p.selection_role.eq(role)]
   if len(ids)<count:raise ValueError(f'Cannot fill {g}/{role}: {len(ids)} < {count}')
   add(dict.fromkeys(ids,1),count,count)
 for d,ids in p.groupby('ligand_inchikey').groups.items():
  add({**dict.fromkeys(ids,1),ys[d]:-2},0,np.inf);add({**dict.fromkeys(ids,1),ys[d]:-6},-np.inf,0)
 add(dict.fromkeys(ys.values(),1),120,120)
 dm=p.drop_duplicates('ligand_inchikey')
 for _,grp in dm.groupby('connectivity_key'):add({ys[d]:1 for d in grp.ligand_inchikey},-np.inf,1)
 for scaffold,grp in dm.groupby('murcko_scaffold'):
  if pd.notna(scaffold) and scaffold:add({ys[d]:1 for d in grp.ligand_inchikey},-np.inf,5)
 add({ys[d]:1 for d in dm.loc[dm.known_kinase_drug,'ligand_inchikey']},-np.inf,30)
 # Avoid five same-scaffold high picks on a target; unlike the earlier design.
 for (_,scaffold),grp in p[p.selection_role.eq('OUR_FROZEN_MODEL_HIGH')].groupby(['gene_symbol','murcko_scaffold']):
  if pd.notna(scaffold) and scaffold:add(dict.fromkeys(grp.index,1),-np.inf,2)
 # Descriptive chemical balance across arms: pair-weighted mean gaps, not causal matching.
 for role in ['OUR_FROZEN_MODEL_INTERMEDIATE','OUR_FROZEN_MODEL_LOW_BACKGROUND']:
  for field,tolerance in [('rdkit_mw',40.0),('rdkit_clogp',0.5)]:
   coeff={int(i):float(p.loc[i,field])/64 for i in p.index[p.selection_role.eq(role)]}
   coeff.update({int(i):-float(p.loc[i,field])/320 for i in p.index[p.selection_role.eq('OUR_FROZEN_MODEL_HIGH')]})
   add(coeff,-tolerance,tolerance)
 c=np.zeros(n);c[:len(p)]=p.apply(old.edge_cost,axis=1);c[len(p):]=1e-7
 result=milp(c,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
   constraints=LinearConstraint(coo_matrix((vv,(rr,cc)),shape=(len(lo),n)).tocsr(),lo,hi),
   options={'time_limit':300,'mip_rel_gap':0.001,'presolve':True})
 write_json('SOLVER.json',{'status':result.status,'message':result.message,'fun':result.fun,'mip_gap':getattr(result,'mip_gap',None)})
 if result.x is None:raise RuntimeError(result.message)
 rounded=np.rint(result.x)
 residual=coo_matrix((vv,(rr,cc)),shape=(len(lo),n)).tocsr()@rounded
 if not (np.all(residual>=np.array(lo)-1e-6) and np.all(residual<=np.array(hi)+1e-6)):raise RuntimeError('Integer solution violates design constraints')
 chosen=p.iloc[np.flatnonzero(rounded[:len(p)])].copy()
 chosen['optimization_objective_contribution']=chosen.apply(old.edge_cost,axis=1)
 return chosen


def controls(strict,targets,genes):
 rows=[]
 for g in genes:
  name=old.CONTROL_NAMES[g]
  a=strict[strict.gene_symbol.eq(g)&strict.parent_molecule_name.fillna('').str.upper().eq(name)].copy()
  if a.empty:raise ValueError(f'No control identity {g}/{name}')
  positive=a[a.calibration_label.eq('positive')]
  if len(positive):a=positive
  elif g not in EXCEPTIONS:raise ValueError(f'Unresolved reference not explicitly reviewed: {g}/{name}')
  x=a.sort_values('max_pchembl',ascending=False).iloc[0];t=targets.set_index('gene_symbol').loc[g]
  types=set(str(x.standard_types).split(','))
  rows.append(dict(gene_symbol=g,target_chembl_id=t.target_chembl_id,uniprot_accession=t.uniprot_accession,
   target_name=t.target_name,assay_lane=t.assay_lane,drug_names=name.title(),ligand_inchikey=x.parent_standard_inchi_key,
   ligand_smiles=x.parent_canonical_smiles,pair_id=f'CONTROL::{x.parent_standard_inchi_key}__{t.target_chembl_id}',
   selection_role='POSITIVE_CONTROL',experiment_arm='TARGET_QC_POSITIVE_CONTROL',
   control_chembl_id=x.parent_molecule_chembl_id,control_standard_types=x.standard_types,
   control_mean_pchembl_mixed_endpoints=x.mean_pchembl,control_min_pchembl=x.min_pchembl,control_max_pchembl=x.max_pchembl,
   control_local_label=x.calibration_label,control_assay_ids=x.assay_ids,control_doc_ids=x.doc_ids,
   control_evidence_category='Kd_record_present' if 'Kd' in types else 'Ki_without_Kd' if 'Ki' in types else 'IC50_only',
   control_external_reference=EXCEPTIONS[g][1] if g in EXCEPTIONS else '',
   control_status='LOCAL_CONFLICT_EXTERNAL_REFERENCE_REQUIRES_SCOUT' if g in EXCEPTIONS else 'REFERENCE_ASSIGNED_SPR_SCOUT_REQUIRED',
   reference_note='Mixed-endpoint pChEMBL is not a SPR Kd; retain original assay/document IDs.'))
 return pd.DataFrame(rows)


def annotate_support(selected,strict):
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
 qcache={}
 def fp(s):
  if s not in qcache:
   mol=Chem.MolFromSmiles(str(s));qcache[s]=gen.GetFingerprint(mol) if mol else None
  return qcache[s]
 rows=[]
 for g,group in selected.groupby('gene_symbol'):
  refs=strict[strict.gene_symbol.eq(g)&strict.calibration_label.isin(['positive','negative_or_inactive'])].drop_duplicates('parent_standard_inchi_key')
  banks={}
  for label in ['positive','negative_or_inactive']:
   bank=[]
   for r in refs[refs.calibration_label.eq(label)].itertuples():
    f=fp(r.parent_canonical_smiles)
    if f is not None:bank.append((r,f))
   banks[label]=bank
  for r in group.itertuples():
   entry={'pair_id':r.pair_id};q=fp(r.ligand_smiles)
   for label,prefix in [('positive','positive'),('negative_or_inactive','negative')]:
    bank=[(t,f) for t,f in banks[label] if t.parent_standard_inchi_key!=r.ligand_inchikey]
    sims=DataStructs.BulkTanimotoSimilarity(q,[f for _,f in bank]) if q is not None and bank else []
    if sims:
     idx=int(np.argmax(sims));t=bank[idx][0]
     entry.update({f'{prefix}_max_tanimoto':float(sims[idx]),f'{prefix}_reference_chembl':t.parent_molecule_chembl_id,
      f'{prefix}_reference_name':t.parent_molecule_name,f'{prefix}_reference_inchikey':t.parent_standard_inchi_key,
      f'{prefix}_reference_endpoints':t.standard_types,f'{prefix}_reference_assays':t.assay_ids})
   rows.append(entry)
 return selected.merge(pd.DataFrame(rows),on='pair_id',validate='one_to_one')


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);genes=configure()
 if args.prepare:prepare()
 ident=json.loads((OUT/'PREPARATION_IDENTITY.json').read_text())
 if ident['genes']!=old.BENCHMARK_TARGETS+old.DISCOVERY_TARGETS or ident['cache_sha256']!=old.sha256(OUT/'FINAL64_EXTERNAL_POOL.csv.gz'):raise ValueError('Preparation identity mismatch')
 if any(old.sha256(ROOT/path)!=digest for path,digest in ident['sources'].items()):raise ValueError('Source changed; regenerate preparation with --prepare')
 pool=pd.read_csv(OUT/'FINAL64_EXTERNAL_POOL.csv.gz',low_memory=False);targets=pd.read_csv(OUT/'FINAL64_TARGETS.csv')
 # Conservative design exclusions from prior review, not claims about biological inactivity.
 mask=pool.drug_names.isin(['enalapril','serdexmethylphenidate'])|pool.chemical_risk_points.gt(1)
 excluded=pool[mask].copy();excluded['reason']='confirmed_example_prodrug_unresolved_OR_original_chemical_risk_gt1'
 excluded.to_csv(OUT/'DESIGN_POLICY_EXCLUDED.csv',index=False);pool=pool[~mask].copy()
 pool.groupby(['gene_symbol','selection_role']).size().unstack(fill_value=0).to_csv(OUT/'ELIGIBLE_ROLE_COUNTS.csv')
 selected=optimize(pool,genes)
 strict=pd.read_csv(old.STRICT_PATH)
 selected=annotate_support(selected,strict)
 selected['support_note']='Descriptive full ChEMBL37 reference audit; NOT a temporally held-out test or binding probability; not used in optimizer.'
 selected['novelty_status']='Snapshot exact-pair exclusions passed; full-text/patent review pending'
 selected['pair_review_status']=np.where(selected.positive_max_tanimoto.lt(.4),'LOW_POSITIVE_CHEMICAL_SUPPORT_REVIEW',
   np.where(selected.negative_max_tanimoto.ge(.7)&selected.negative_max_tanimoto.gt(selected.positive_max_tanimoto),'CLOSE_NEGATIVE_REFERENCE_REVIEW','PAIR_LITERATURE_AND_APPLICABILITY_REVIEW'))
 control=controls(strict,targets,genes)
 roster=targets.set_index('gene_symbol').loc[genes].reset_index()
 roster=roster.merge(control[['gene_symbol','drug_names','control_status','control_evidence_category','control_assay_ids','control_doc_ids']].rename(columns={'drug_names':'reference_control'}),on='gene_symbol')
 roster['construct_recommendation']=roster.gene_symbol.map(old.CONSTRUCTS)
 roster['construct_status']='EXPERIMENT_TEAM_CONFIRM_BOUNDARIES_ISOFORM_ACTIVITY_REQUIRED'
 roster['special_system_review']=roster.gene_symbol.isin(SPECIAL)
 roster['release_status']='NOT_RELEASED_FOR_EXPERIMENT'
 roster['logistics_batch']=np.arange(64)//16+1
 roster['batch_note']='Four whole-target blocks, 16 targets/128 pairs each; logistics draft, not biological-priority ranking'
 allrows=pd.concat([selected,control],ignore_index=True,sort=False)
 blocks=[]
 for i,g in enumerate(genes):
  block=allrows[allrows.gene_symbol.eq(g)].sample(frac=1,random_state=20260909+i).copy();block['logistics_batch']=i//16+1;blocks.append(block)
 allrows=pd.concat(blocks,ignore_index=True);allrows['experiment_id']=[f'SPR64D-{i:03d}' for i in range(1,513)]
 allrows['row_type']=np.where(allrows.selection_role.eq('POSITIVE_CONTROL'),'REFERENCE_CONTROL','CANDIDATE_BLINDED_ROLE')
 allrows['construct_recommendation']=allrows.gene_symbol.map(old.CONSTRUCTS)
 allrows['release_status']='DESIGN_REVIEW_NOT_RELEASED_FOR_EXPERIMENT'
 allrows['endpoint_plan']='QC-qualified direct binding; assess kinetic model suitability; no-signal only interpretable at qualified tested concentration'
 degree=selected.groupby('ligand_inchikey').size()
 means=selected.groupby('selection_role')[['rdkit_mw','rdkit_clogp']].mean()
 checks={
  'rows512':len(allrows)==512,'targets64':allrows.gene_symbol.nunique()==64,
  'unique_entity_pairs':not allrows.duplicated(['ligand_inchikey','target_chembl_id']).any(),
  'candidates448_controls64':len(selected)==448 and len(control)==64,
  'roles_per_target':all(selected[selected.gene_symbol.eq(g)].selection_role.value_counts().to_dict()==QUOTAS for g in genes),
  'drugs120_degree2to6':selected.ligand_inchikey.nunique()==120 and degree.between(2,6).all(),
  'connectivity_unique':selected.drop_duplicates('ligand_inchikey').connectivity_key.is_unique,
  'no_known_or_train_exact_pairs':not selected[['local_chembl_pair_found','kirhub_pair_found','known_moa_target_collision','exact_pair_in_full_fit_training','bindingdb_exact_pair_found','gtopdb_exact_pair_found']].any().any(),
  'max_chemical_risk1':selected.chemical_risk_points.le(1).all(),
  'two_explicit_prodrug_examples_absent':not selected.drug_names.isin(['enalapril','serdexmethylphenidate']).any(),
  'all_control_identities_and_constructs':control.ligand_inchikey.notna().all() and roster.construct_recommendation.notna().all(),
  'batches128':allrows.groupby('logistics_batch').size().eq(128).all(),
  'mean_chemistry_balance':all(abs(means.loc[role,field]-means.loc['OUR_FROZEN_MODEL_HIGH',field])<=tol+1e-6 for role in ['OUR_FROZEN_MODEL_INTERMEDIATE','OUR_FROZEN_MODEL_LOW_BACKGROUND'] for field,tol in [('rdkit_mw',40),('rdkit_clogp',.5)]),
 }
 write_json('VALIDATION.json',{k:bool(v) for k,v in checks.items()})
 if not all(checks.values()):raise RuntimeError('Validation failed; do not use outputs')
 allrows.to_csv(OUT/'INTERNAL_MASTER_512.csv',index=False)
 roster.to_csv(OUT/'TARGET_ROSTER_64.csv',index=False)
 control.to_csv(OUT/'CONTROL_EVIDENCE_64.csv',index=False)
 # No ranks, selection roles, nearest-neighbour evidence or internal sheets in this workbook.
 blind=allrows[['experiment_id','logistics_batch','row_type','drug_names','ligand_inchikey','ligand_smiles','gene_symbol','target_chembl_id','uniprot_accession','construct_recommendation','release_status','endpoint_plan']]
 blind.to_csv(OUT/'BLINDED_DESIGN_512.csv',index=False)
 procurement=allrows.groupby(['ligand_inchikey','drug_names','ligand_smiles'],dropna=False).agg(pair_count=('pair_id','size'),targets=('gene_symbol',lambda v:';'.join(sorted(set(v)))),roles=('row_type',lambda v:';'.join(sorted(set(v))))).reset_index()
 procurement['status']='IDENTITY_ROSTER_NOT_PURCHASE_AUTHORIZATION';procurement.to_csv(OUT/'COMPOUND_IDENTITY_ROSTER.csv',index=False)
 with pd.ExcelWriter(OUT/'INTERNAL_DESIGN_REVIEW.xlsx',engine='openpyxl') as w:
  allrows.to_excel(w,sheet_name='内部候选与角色',index=False);roster.to_excel(w,sheet_name='靶点与构建审查',index=False);control.to_excel(w,sheet_name='对照来源',index=False);procurement.to_excel(w,sheet_name='分子实体清单',index=False)
 with pd.ExcelWriter(OUT/'BLINDED_DESIGN_REVIEW.xlsx',engine='openpyxl') as w:
  blind.to_excel(w,sheet_name='盲化设计待确认',index=False)
 support_counts=selected[selected.selection_role.eq('OUR_FROZEN_MODEL_HIGH')].pair_review_status.value_counts().to_dict()
 write_json('SUMMARY.json',{
  'status':'COMPUTATIONAL_DESIGN_COMPLETE_LAB_RELEASE_PENDING','utc':datetime.now(timezone.utc).isoformat(),
  'targets':64,'candidate_pairs':448,'controls':64,'total_pairs':512,'candidate_drugs':120,
  'all_compound_entities_including_controls':allrows.ligand_inchikey.nunique(),
  'core_targets':len(old.BENCHMARK_TARGETS),'routed_targets':len(old.DISCOVERY_TARGETS),
  'roles':selected.selection_role.value_counts().to_dict(),'replacements':REPLACEMENTS,
  'control_evidence':control.control_evidence_category.value_counts().to_dict(),
  'control_local_conflict_exceptions':list(EXCEPTIONS),'high_pair_support_flags':support_counts,
  'high_candidates_are_not_certified_binders':True,'means_by_role':means.to_dict(orient='index'),
  'special_system_targets':sorted(SPECIAL),'training_started':False,'original_lab_package_modified':False,
  'remaining':['full-text/patent and pair-level applicability review','complete active-species audit beyond two excluded examples','control conflict resolution and SPR scout','construct and special-system confirmation','disease intervention direction for final pairs','repeat/confirmation budget'],
  'interpretation':'Computational feasibility and selection, not expected hit-rate certification. Similarity audit post-selection; holds do not silently change rank bands.',
 })
 write_json('ARTIFACT_MANIFEST.json',{'inputs':ident,'builder_sha256':old.sha256(Path(__file__)),
  'files':{p.name:old.sha256(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ['.csv','.xlsx','.json'] and p.name!='ARTIFACT_MANIFEST.json'}})
 print((OUT/'SUMMARY.json').read_text(),flush=True)


if __name__=='__main__':main()
