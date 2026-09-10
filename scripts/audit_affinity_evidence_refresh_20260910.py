#!/usr/bin/env python3
"""Finish provenance joins and report exact-pair coverage without changing selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_affinity_evidence_refresh_20260910 import *

def assay_index(c):
 ids={r[0].split(':')[0] for r in c.execute("select record_id from measurements where source='BindingDB_202609'")}
 links=[];needed=set()
 for _,h,row in ziprows(RAW/'BindingDB_rsid_eaids_202609_tsv.zip'):
  if row[0] in ids:
   for k in re.split(r'[,; ]+',row[1]):
    if k:links.append((row[0],k));needed.add(k)
 desc=[]
 for _,h,row in ziprows(RAW/'BindingDB_Assays_202609_tsv.zip'):
  k=row[0]+'_'+row[1]
  if k in needed:desc.append((k,row[2],row[3] if len(row)>3 else ''))
 c.execute('drop table if exists bindingdb_assay_links');c.execute('create table bindingdb_assay_links(reactant_set_id text,entry_assay_id text)')
 c.executemany('insert into bindingdb_assay_links values (?,?)',links)
 c.execute('drop table if exists bindingdb_assays');c.execute('create table bindingdb_assays(entry_assay_id text primary key,assay_name text,description text)')
 c.executemany('insert or replace into bindingdb_assays values (?,?,?)',desc)
 c.execute('create index if not exists bdb_assay_link_rsid on bindingdb_assay_links(reactant_set_id)');c.commit()
 return {'indexed_links':len(links),'indexed_assay_descriptions':len(desc),'unique_reactant_ids_with_links':len(set(x[0] for x in links))}

def delta_audit():
 results=[];details=[]
 for part in ['BindingDB_Articles','PubChem']:
  snapshots=[]
  for version in ['202607','202609']:
   p=(OLD if version=='202607' else RAW)/f'BindingDB_{part}_{version}_tsv.zip';m={}
   for _,h,row in ziprows(p):
    d={k:v.strip() for k,v in zip(h,row) if v.strip()};rid=row[0]
    if rid not in m:m[rid]=d;m[rid]['_core_fingerprints']=set()
    m[rid]['_core_fingerprints'].add(bdb_fingerprint(d))
   snapshots.append(m)
  old,new=snapshots;ok=set(old);nk=set(new);shared=ok&nk
  changed=[rid for rid in shared if old[rid]['_core_fingerprints']!=new[rid]['_core_fingerprints']]
  results.append(dict(subset=part,old_ids=len(ok),new_ids=len(nk),added_ids=len(nk-ok),removed_ids=len(ok-nk),shared_changed_identity_target_or_endpoint=len(changed),unchanged_core=len(shared)-len(changed),shared_ligand_key_changed=sum(old[r].get('Ligand InChI Key')!=new[r].get('Ligand InChI Key') for r in shared),shared_numeric_endpoint_changed=sum(any(old[r].get(e+' (nM)')!=new[r].get(e+' (nM)') for e in ['Kd','Ki','IC50','EC50']) for r in shared)))
  for rid in changed+sorted(nk-ok):
   d=new[rid];k=d.get('Ligand InChI Key','');a,s=bdb_identity(d)
   details.append(dict(subset=part,reactant_set_id=rid,change='ADDED_ID' if rid not in old else 'REVISED_CORE',ligand_inchikey=k,target_accessions=a,target_scope=s,drug_name=d.get('BindingDB Ligand Name'),target_name=d.get('Target Name'),old_core=dump({x:old.get(rid,{}).get(x) for x in ['Ligand InChI Key','Ki (nM)','Kd (nM)','IC50 (nM)','EC50 (nM)']}),new_core=dump({x:d.get(x) for x in ['Ligand InChI Key','Ki (nM)','Kd (nM)','IC50 (nM)','EC50 (nM)']}),date_in_bindingdb=d.get('Date in BindingDB'),publication_date=d.get('Date of publication')))
 pd.DataFrame(results).to_csv(OUT/'BINDINGDB_JULY_SEPT_DELTA.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(details).to_csv(OUT/'BINDINGDB_CHANGED_RECORDS.csv',index=False,encoding='utf-8-sig')
 return results

def auxiliary(c):
 c.execute('drop table if exists auxiliary_observations');c.execute('create table auxiliary_observations(source text,sheet text,row_number integer,metadata_json text)')
 for p in list(RAW.glob('RAF*.xlsx'))+list(RAW.glob('CACHE*CACHE3*.xlsx')):
  x=pd.ExcelFile(p)
  for s in x.sheet_names:
   if 'SPR data' in s:continue
   d=pd.read_excel(x,s).fillna('')
   c.executemany('insert into auxiliary_observations values (?,?,?,?)',[(p.name,s,int(i+2),dump(r.to_dict())) for i,r in d.iterrows()])
 # Kinobead displacement is contextual target-engagement evidence, never KD.
 c.execute("delete from measurements where source='RAF_MEK_2026'");w=Writer(c)
 d=pd.read_excel(RAW/'RAF_MEK_MOESM4.xlsx').fillna('')
 ch=sqlite3.connect(f'file:{CHEMBL}?mode=ro',uri=True)
 names={'Tovorafenib':'TOVORAFENIB','LXH254':'NAPORAFENIB','LF268':'LF-268','AZ628':'AZ-628'}
 for prefix,name in names.items():
  hit=ch.execute('select cs.standard_inchi_key,cs.canonical_smiles from molecule_dictionary md join compound_structures cs using(molregno) where md.pref_name=?',(name,)).fetchone();k,sm=hit or ('','')
  for i,r in d.iterrows():
   f=prefix+' log2FC from DMSO';flag=next((col for col in d if col.upper().startswith(prefix.upper()+' TARGET?')),'')
   w.add('RAF_MEK_2026',prefix+':'+str(i+2),k,name,sm,clean(r.UNIPROT),clean(r['GENE NAME']),'lysate_protein_or_group','Homo sapiens','kinobead_log2FC',r[f],'log2_fold_change',method='kinobead_competition',quality='author_target_flag:'+str(r.get(flag,'')),doi='10.1038/s41589-026-02212-2',url='https://www.nature.com/articles/s41589-026-02212-2',group='RAF_MEK:'+prefix+':'+str(i+2),meta=r.to_dict())
 w.flush();ch.close();c.commit()
 return dict(c.execute('select source,count(*) from auxiliary_observations group by source').fetchall())

def coverage(c):
 base=pd.read_csv(BASE).fillna('');full=pd.read_parquet(ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet');targets=pd.read_csv(TARGET);drugs=pd.read_csv(DRUG)
 d=pd.read_sql_query('select * from comparable_numeric_evidence',c)
 d['pair_id']=d.ligand_key+'__'+d.target_chembl_id
 # conservative duplicate fingerprint is a review aid, not an independent-experiment count
 def fingerprint(r):
  meta=json.loads(r.metadata_json);citation=clean(r.doi).lower() or clean(r.pmid)
  if not citation:return 'ungrouped:'+r.source+':'+r.provenance_group
  return hashlib.sha256(dump([r.ligand_key,r.target_accessions,r.endpoint,r.relation,r.value_nM,citation]).encode()).hexdigest()
 d['possible_shared_experiment_group']=d.apply(fingerprint,axis=1)
 d['existing_pair_in_original_scored_audit']=d.pair_id.isin(set(full.loc[full.known_relation_excluded,'pair_id']))
 d.to_csv(OUT/'PROJECT_EXACT_NUMERIC_EVIDENCE.csv.gz',index=False,compression='gzip')
 c.execute('drop table if exists project_evidence_dedup_groups');d[['id','possible_shared_experiment_group']].to_sql('project_evidence_dedup_groups',c,index=False)
 base_set=set(base.pair_id);full_set=set(full.pair_id);old=set(d.loc[d.source=='ChEMBL37_existing','pair_id']);recent=d[d.source!='ChEMBL37_existing']
 summary=[]
 for source,x in [(src,d[d.source==src]) for src, in c.execute('select distinct source from measurements')]+[('REFRESH_UNION',recent),('WITH_EXISTING_CHEMBL37',d)]:
  for scope,den,space in [('SPR384',384,base_set),('720x384',276480,full_set),('720x888',639360,None)]:
   z=x if space is None else x[x.pair_id.isin(space)]
   a=set(z.pair_id);kd=set(z.loc[z.endpoint=='Kd','pair_id']);ki=set(z.loc[z.endpoint=='Ki','pair_id']);eq=set(z.loc[(z.endpoint=='Kd')&(z.relation=='='),'pair_id'])
   summary.append(dict(source=source,scope=scope,denominator=den,pairs_any_numeric_endpoint=len(a),pairs_Kd=len(kd),pairs_Ki=len(ki),pairs_Kd_or_Ki=len(kd|ki),pairs_exact_value_Kd=len(eq),coverage_pct=100*len(a)/den,source_rows=len(z),pairs_not_in_existing_ChEMBL37_numeric=len(a-old)))
 pd.DataFrame(summary).to_csv(OUT/'COVERAGE_SUMMARY.csv',index=False,encoding='utf-8-sig')
 cols=['candidate_id','pair_id','drug_names','ligand_inchikey','target_chembl_id','gene_symbol','binding_rank_384']
 b=base[cols].copy()
 for name,sub in [('all',d),('refresh',recent),('Kd',d[d.endpoint=='Kd']),('Ki',d[d.endpoint=='Ki']),('IC50_EC50',d[d.endpoint.isin(['IC50','EC50'])])]:
  counts=sub.groupby('pair_id').size();b[name+'_evidence_rows']=b.pair_id.map(counts).fillna(0).astype(int)
 sources=d.groupby('pair_id').source.agg(lambda x:';'.join(sorted(set(x))))
 b['sources']=b.pair_id.map(sources).fillna('');b['interpretation']=b.all_evidence_rows.map(lambda n:'有历史数值；需复核端点、上下界、构建和数值，不能直接视作阳性' if n else '本轮未找到精确配对数值证据；不等于阴性或不可结合')
 b.to_csv(OUT/'SPR384_AFFINITY_COVERAGE.csv',index=False,encoding='utf-8-sig');d[d.pair_id.isin(base_set)].to_csv(OUT/'SPR384_MATCHED_EVIDENCE.csv',index=False,encoding='utf-8-sig')
 # Keep excluded identity/complex/domain/context matches visible, outside exact coverage.
 allm=pd.read_sql_query('select id,source,record_id,ligand_key,target_accessions,target_scope,endpoint,relation,value_nM,quality from measurements',c)
 tmap=dict(zip(targets.uniprot_accession,targets.target_chembl_id));blocks=defaultdict(list)
 for k in drugs.ligand_inchikey:blocks[k.split('-')[0]].append(k)
 review=[]
 for r in allm.itertuples():
  if r.ligand_key.split('-')[0] not in blocks:continue
  for a in r.target_accessions.split(';'):
   t=tmap.get(a)
   if not t:continue
   for k in blocks[r.ligand_key.split('-')[0]]:
    pair=k+'__'+t
    if k==r.ligand_key and r.target_scope=='single_accession_construct_unverified':continue
    review.append(dict(pair_id=pair,measurement_id=r.id,source=r.source,endpoint=r.endpoint,source_ligand_key=r.ligand_key,project_ligand_key=k,scope=r.target_scope,identity='EXACT_FULL_KEY' if k==r.ligand_key else 'CONNECTIVITY_ONLY',in_SPR384=pair in base_set,value_nM=r.value_nM,relation=r.relation,quality=r.quality))
 pd.DataFrame(review).to_csv(OUT/'IDENTITY_CONSTRUCT_CONTEXT_REVIEW.csv',index=False,encoding='utf-8-sig')
 # Pair-wise incremental relation audit (not a claim of new publications).
 p=recent.groupby('pair_id').agg(sources=('source',lambda x:';'.join(sorted(set(x)))),endpoints=('endpoint',lambda x:';'.join(sorted(set(x)))),rows=('id','size')).reset_index()
 p['absent_from_existing_ChEMBL37_numeric']=~p.pair_id.isin(old)
 p=p.merge(full[['pair_id','drug_names','gene_symbol','known_relation_excluded','known_chembl37','known_training','known_bindingdb_articles','known_bindingdb_pubchem','novelty_pass']],on='pair_id',how='left')
 p['in_SPR384']=p.pair_id.isin(base_set)
 p.to_csv(OUT/'PAIR_INCREMENT_AUDIT.csv',index=False,encoding='utf-8-sig')
 return summary,{'matched_baseline_pairs':int((b.all_evidence_rows>0).sum()),'refresh_matched_baseline_pairs':int((b.refresh_evidence_rows>0).sum()),'baseline_review_only_pairs':len(set(x['pair_id'] for x in review if x['in_SPR384'])),'refresh_pairs_absent_from_original_scored_known_flags':int((p.known_relation_excluded==False).sum()),'possible_duplicate_groups_multi_source':int((d.groupby('possible_shared_experiment_group').source.nunique()>1).sum())}

def main():
 c=sqlite3.connect(DB)
 print('assays',flush=True);assays=assay_index(c)
 print('auxiliary',flush=True);aux=auxiliary(c)
 print('coverage',flush=True);summary,notes=coverage(c)
 print('delta',flush=True);delta=delta_audit()
 catalog=pd.read_csv(SURVEY/'RECENT_AFFINITY_SOURCE_CATALOG.csv').fillna('');catalog['本轮状态']=['全库归档；项目药物证据索引及assay描述入库','原始2733测量行入库；非人靶点','两轮SPR入库；HTRF和DLS独立辅助表','31572小分子结构附带标签入库；保留重复来源','Data6全部拟合行入库；仅Kd_app','文件目录已归档；元数据下载HTTP401；未入库预测结构或重复标签','全表归档；SPR及QC入库；限定WDR/PWWP结构域','补充表1–4及Data1归档；kinobead信号独立入库']
 catalog.to_csv(OUT/'SOURCE_IMPORT_STATUS.csv',index=False,encoding='utf-8-sig');catalog.to_sql('source_catalog',c,if_exists='replace',index=False)
 if (RAW/'DOWNLOAD_MANIFEST.json').exists():pd.DataFrame(json.loads((RAW/'DOWNLOAD_MANIFEST.json').read_text())).to_sql('source_downloads',c,if_exists='replace',index=False)
 c.commit();integrity=c.execute('pragma integrity_check').fetchone()[0];counts=dict(c.execute('select source,count(*) from measurements group by source').fetchall());c.close()
 manifest=json.loads((OUT/'BUILD_MANIFEST.json').read_text());assert sha(BASE)==manifest['baseline_sha256']
 result=dict(database=str(DB.relative_to(ROOT)),source_rows=counts,assay_join=assays,auxiliary_rows=aux,coverage=summary,notes=notes,bindingdb_delta=delta,integrity_check=integrity,baseline_unchanged=True)
 (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
