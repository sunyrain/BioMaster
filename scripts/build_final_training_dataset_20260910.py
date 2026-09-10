#!/usr/bin/env python3
"""Full-source, resumable data curation. Does not fit or deploy a model."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,io,json,math,re,sqlite3,sys,time,zipfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import rdkit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.training_dataset_v1 import ENDPOINT_TASK,MUTANT,sha_text,concentration,observation_label,molecule_identity,classify_construct,split_for_group
from biomaster.portable_ranker_v2 import digest

OUT=ROOT/'data/processed/biomaster_training_full_20260910_v1'
REPORT=ROOT/'outputs/training_dataset_full_20260910_v1'
CHEMBL=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
RAW=ROOT/'data/external/affinity_refresh_20260910'
EXTRA=ROOT/'data/processed/biomaster_affinity_evidence_20260910.sqlite'
QC=RAW/'SOURCE_RECORD_QC_OVERRIDES_20260910.csv'
FINAL=ROOT/'outputs/spr384_final_experiment_table_20260910'
PREVIOUS=ROOT/'outputs/biomaster_bindingdb_incremental_20260910/FROZEN_SPLIT.csv'
ARCHIVES=['All','BindingDB_Articles','PubChem','ChEMBL','Patents','PDSPKi']
FIELDS=['measurement_id','source','source_record_id','variant_hash','molecule_id','target_id','endpoint','task','relation','value_nM',
        'observation_label','explicit_inactive','doi','pmid','patent','document_year','assay_id','construct_scope','evidence_tier','target_name','source_url']


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def log(**row):
    row['utc']=datetime.now(timezone.utc).isoformat();write_json(REPORT/'STATUS.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)


def zip_rows(path):
    with zipfile.ZipFile(path) as z:
        member=next(n for n in z.namelist() if n.endswith(('.tsv','.txt')))
        with io.TextIOWrapper(z.open(member),encoding='utf-8-sig',errors='strict',newline='') as f:
            reader=csv.reader(f,delimiter='\t');header=next(reader)
            for line,row in enumerate(reader,2):yield line,header,row


def text(x):
    return '' if x is None or isinstance(x,float) and math.isnan(x) else str(x).strip()


def normalize_doi(value):
    return re.sub(r'^(?:https?://(?:dx\.)?doi.org/|doi:\s*)','',text(value).lower())


def normalize_args(args):
    return molecule_identity(*args)


def chembl_precheck(r,resolved):
    allowed=['B','F'] if r['standard_type']=='EC50' else ['B']
    if r['assay_type'] not in allowed:return 'assay_type_outside_task'
    if r['confidence_score'] is None or r['confidence_score']<9 or r['relationship_type']!='D':return 'low_confidence_or_indirect'
    if r['variant_id'] is not None:return 'annotated_variant'
    if r['potential_duplicate']==1:return 'potential_duplicate_flag'
    if r['data_validity_comment'] not in [None,'','Manually validated']:return 'source_validity_flag'
    if r['year'] is None or r['year']>2026:return 'missing_or_future_document_year'
    if not resolved:return 'unresolved_canonical_target_sequence'
    if MUTANT.search(text(r['description'])):return 'assay_mutation_or_fusion_review'
    return ''


class Builder:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
        self.c=sqlite3.connect(OUT/'EVIDENCE.sqlite');self.c.execute('pragma journal_mode=WAL');self.c.execute('pragma synchronous=NORMAL')
        self.c.execute('pragma cache_size=-262144')
        self.c.executescript('''
        CREATE TABLE IF NOT EXISTS observations (measurement_id TEXT PRIMARY KEY,source TEXT,source_record_id TEXT,variant_hash TEXT,
          molecule_id TEXT,target_id TEXT,endpoint TEXT,task TEXT,relation TEXT,value_nM REAL,observation_label INTEGER,explicit_inactive INTEGER,
          doi TEXT,pmid TEXT,patent TEXT,document_year INTEGER,assay_id TEXT,construct_scope TEXT,evidence_tier TEXT,target_name TEXT,source_url TEXT);
        CREATE TABLE IF NOT EXISTS origins (measurement_id TEXT,source_file TEXT,UNIQUE(measurement_id,source_file));
        CREATE TABLE IF NOT EXISTS molecules (molecule_id TEXT PRIMARY KEY,connectivity_key TEXT,smiles TEXT,scaffold TEXT,heavy_atoms INTEGER,formal_charge INTEGER);
        CREATE TABLE IF NOT EXISTS target_sequences (target_id TEXT PRIMARY KEY,sequence TEXT,accessions TEXT,chembl_ids TEXT,names TEXT,identity_tier TEXT);
        CREATE TABLE IF NOT EXISTS rejected_source_identity (source TEXT,source_record_id TEXT,reason TEXT,UNIQUE(source,source_record_id,reason));
        ''')
        self.pending=[];self.origins=[];self.mols={};self.targets={};self.acc={};self.chembl={};self.seqmap={}
        self.qc={(r.source,r.record_id) for r in pd.read_csv(QC).itertuples()}
        self.pool=ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'))
        self.load_targets()

    def prepared_chembl(self,c,query):
        cursor=c.execute(query)
        while True:
            batch=[dict(r) for r in cursor.fetchmany(12000)]
            if not batch:break
            reasons=[chembl_precheck(r,r['target_chembl_id'] in self.chembl) for r in batch]
            ids=[i for i,r in enumerate(reasons) if not r]
            normalized=dict(zip(ids,self.pool.map(normalize_args,[(text(batch[i]['canonical_smiles']),text(batch[i]['standard_inchi_key'])) for i in ids],chunksize=64)))
            for i,r in enumerate(batch):
                mol,reason=normalized.get(i,(None,reasons[i]))
                yield r,mol,reason

    def prepared_bindingdb(self,path):
        batch=[]
        def emit(items):
            ids=[i for i,x in enumerate(items) if not x[-1]]
            normalized=dict(zip(ids,self.pool.map(normalize_args,[(items[i][1].get('Ligand SMILES',''),items[i][1].get('Ligand InChI Key','')) for i in ids],chunksize=64)))
            for i,(line,d,target,scope,reason) in enumerate(items):
                mol,reason=normalized.get(i,(None,reason))
                yield line,d,target,scope,mol,reason
        for line,header,row in zip_rows(path):
            if len(row)<50 or not row[0].isdigit():
                batch.append((line,{'BindingDB Reactant_set_id':row[0][:40] if row else ''},None,None,'malformed_rows'))
            else:
                d={k:v.strip() for k,v in zip(header,row) if v.strip()}
                target,scope,reason=self.resolve_bdb_target(d);batch.append((line,d,target,scope,reason))
            if len(batch)>=12000:
                yield from emit(batch);batch=[]
        if batch:yield from emit(batch)

    def load_targets(self):
        c=sqlite3.connect(f'file:{CHEMBL}?mode=ro',uri=True)
        rows=c.execute("SELECT t.tid,t.chembl_id,t.pref_name,cs.accession,cs.sequence FROM target_dictionary t JOIN target_components tc ON tc.tid=t.tid JOIN component_sequences cs ON cs.component_id=tc.component_id WHERE t.organism='Homo sapiens' AND t.target_type='SINGLE PROTEIN'").fetchall();c.close()
        bytid=defaultdict(list)
        for row in rows:bytid[row[0]].append(row)
        for tid,group in bytid.items():
            seqs={r[4] for r in group if r[4] and re.fullmatch('[ACDEFGHIKLMNPQRSTVWYOU]+',r[4])}
            if len(seqs)!=1:continue
            seq=next(iter(seqs));target='SEQ:'+sha_text(seq)
            item=self.targets.setdefault(target,dict(target_id=target,sequence=seq,accessions=set(),chembl_ids=set(),names=set(),identity_tier='ChEMBL37_human_single_protein'))
            for _,cid,name,acc,_ in group:
                item['chembl_ids'].add(cid);item['names'].add(name or cid)
                if acc:item['accessions'].add(acc)
                self.chembl[cid]=target
            self.seqmap[seq]=target
        for target,item in self.targets.items():
            for acc in item['accessions']:
                self.acc.setdefault(acc,set()).add(target)
        for row in self.c.execute('select * from target_sequences'):
            target,seq,acc,cids,names,tier=row
            if target not in self.targets:
                self.targets[target]=dict(target_id=target,sequence=seq,accessions=set(filter(None,acc.split(';'))),chembl_ids=set(filter(None,cids.split(';'))),names=set(filter(None,names.split(';'))),identity_tier=tier)
                self.seqmap[seq]=target
                for a in filter(None,acc.split(';')):self.acc.setdefault(a,set()).add(target)
        self.save_targets()

    def save_targets(self):
        self.c.executemany('insert or replace into target_sequences values (?,?,?,?,?,?)',[
            (k,v['sequence'],';'.join(sorted(v['accessions'])),';'.join(sorted(v['chembl_ids'])),';'.join(sorted(v['names'])),v['identity_tier']) for k,v in self.targets.items()]);self.c.commit()

    def resolve_bdb_target(self,d):
        species=text(d.get('Target Source Organism According to Curator or DataSource'))
        if species and species.lower() not in ['homo sapiens','human']:return None,None,'nonhuman_target'
        count=text(d.get('Number of Protein Chains in Target (>1 implies a multichain complex)'))
        if count!='1':return None,None,'multichain_or_unknown_chain_count'
        name=text(d.get('Target Name'))
        if MUTANT.search(name):return None,None,'mutant_or_fusion_target'
        accessions={text(v) for k,v in d.items() if re.fullmatch(r'UniProt \((?:SwissProt|TrEMBL)\) Primary ID of Target Chain 1',k) and text(v)}
        if len(accessions)>1:return None,None,'ambiguous_accession'
        sequence=text(d.get('BindingDB Target Chain Sequence 1')).replace(' ','').replace('\n','').upper()
        if sequence and not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYOU]+',sequence):return None,None,'ambiguous_or_invalid_amino_acids'
        accession=next(iter(accessions),'');mapped=self.acc.get(accession,set())
        if len(mapped)>1:return None,None,'accession_multiple_sequences'
        if len(mapped)==1:
            target=next(iter(mapped));scope=classify_construct(sequence,self.targets[target]['sequence'])
            if scope=='sequence_mismatch_or_mutant':return None,None,scope
            return target,scope,''
        if sequence in self.seqmap:return self.seqmap[sequence],'exact_human_reference_sequence',''
        if not sequence or not accession or species.lower() not in ['human','homo sapiens']:
            return None,None,'unresolved_human_sequence_identity'
        # Source-provided sequences outside ChEMBL remain distinct; never mapped by gene name.
        target='SEQ:'+sha_text(sequence)
        self.targets.setdefault(target,dict(target_id=target,sequence=sequence,accessions={accession},chembl_ids=set(),names={name},identity_tier='BindingDB_source_reported_human_sequence'))
        self.acc.setdefault(accession,set()).add(target);self.seqmap[sequence]=target
        return target,'source_reported_human_sequence',''

    def add(self,source,rid,file,mol,target,endpoint,relation,value,explicit=False,**meta):
        if (source,rid) in self.qc:return 'manual_source_qc_exclusion'
        label=observation_label(value,relation,explicit)
        identity=[mol['molecule_id'],target,endpoint,relation,format(value,'.12g') if value is not None else '',int(explicit)]
        variant=sha_text(json.dumps(identity,separators=(',',':')));mid=sha_text(source+'|'+rid+'|'+variant)
        row=dict(measurement_id=mid,source=source,source_record_id=rid,variant_hash=variant,molecule_id=mol['molecule_id'],target_id=target,
                 endpoint=endpoint,task=ENDPOINT_TASK[endpoint],relation=relation,value_nM=value,observation_label=label,explicit_inactive=int(explicit),
                 doi=normalize_doi(meta.get('doi')),pmid=text(meta.get('pmid')).removesuffix('.0'),patent=text(meta.get('patent')).upper(),
                 document_year=meta.get('document_year'),assay_id=text(meta.get('assay_id')),construct_scope=text(meta.get('construct_scope')),
                 evidence_tier=text(meta.get('evidence_tier')),target_name=text(meta.get('target_name')),source_url=text(meta.get('source_url')))
        self.pending.append(tuple(row[k] for k in FIELDS));self.origins.append((mid,file));self.mols[mol['molecule_id']]=mol
        if len(self.pending)>=10000:self.flush()
        return ''

    def flush(self):
        if self.pending:
            self.c.executemany('insert or ignore into observations values ('+','.join('?' for _ in FIELDS)+')',self.pending)
            self.c.executemany('insert or ignore into origins values (?,?)',self.origins)
            self.c.executemany('insert or ignore into molecules values (?,?,?,?,?,?)',[(k,m['connectivity_key'],m['smiles'],m['scaffold'],m['heavy_atoms'],m['formal_charge']) for k,m in self.mols.items()])
            self.c.commit();self.pending=[];self.origins=[];self.mols={}

    def source_done(self,tag):return (REPORT/(tag+'_IMPORT.json')).exists()

    def chembl_import(self):
        tag='ChEMBL37'
        if self.source_done(tag):return
        c=sqlite3.connect(f'file:{CHEMBL}?mode=ro',uri=True);c.row_factory=sqlite3.Row
        query="""SELECT a.activity_id,a.standard_type,a.standard_relation,a.standard_value,a.standard_units,a.activity_comment,a.standard_text_value,a.text_value,a.potential_duplicate,a.data_validity_comment,
        ass.assay_type,ass.confidence_score,ass.relationship_type,ass.variant_id,ass.chembl_id assay_chembl_id,ass.description,
        t.chembl_id target_chembl_id,t.pref_name target_name,cs.standard_inchi_key,cs.canonical_smiles,
        d.year,d.doi,d.pubmed_id,d.patent_id
        FROM target_dictionary t JOIN assays ass ON ass.tid=t.tid JOIN activities a ON a.assay_id=ass.assay_id
        LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
        LEFT JOIN compound_structures cs ON cs.molregno=COALESCE(mh.parent_molregno,a.molregno)
        LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
        WHERE t.organism='Homo sapiens' AND t.target_type='SINGLE PROTEIN'
        AND (a.standard_type IN ('Kd','Ki','IC50','EC50') OR LOWER(COALESCE(a.activity_comment,'')) IN ('inactive','not active','no activity'))"""
        (REPORT/'CHEMBL_EXTRACTION.sql').write_text(query);counts=Counter();started=time.monotonic()
        with gzip.open(OUT/'ChEMBL37_EXCLUSIONS.csv.gz','wt',newline='') as f:
            writer=csv.writer(f);writer.writerow(['source_record_id','reason','target_chembl_id'])
            for r,mol,reason in self.prepared_chembl(c,query):
                counts['scanned']+=1
                endpoint=r['standard_type'];context='functional_assay' if endpoint=='EC50' else 'direct_binding_assay'
                if not reason:
                    explicit=any(text(r[k]).lower() in ['inactive','not active','no activity'] for k in ['activity_comment','standard_text_value','text_value'])
                    rel,value=concentration(r['standard_value'],r['standard_units'],r['standard_relation'])
                    if value is None and not explicit:reason='unusable_numeric_value_or_unit'
                    else:
                        if endpoint not in ENDPOINT_TASK:endpoint='INACTIVE'
                        reason=self.add(tag,str(r['activity_id']),tag,mol,self.chembl[r['target_chembl_id']],endpoint,rel or '',value,explicit,
                            doi=r['doi'],pmid=r['pubmed_id'],patent=r['patent_id'],document_year=r['year'],assay_id=r['assay_chembl_id'],
                            construct_scope='ChEMBL_variant_unannotated',evidence_tier=context+'_confidence9',target_name=r['target_name'],
                            source_url='https://www.ebi.ac.uk/chembl/explore/assay/'+r['assay_chembl_id'])
                if reason:counts[reason]+=1;writer.writerow([r['activity_id'],reason,r['target_chembl_id']])
                else:counts['accepted_observation_attempts']+=1
                if counts['scanned']%250000==0:log(stage='chembl',counts=dict(counts))
        self.flush();c.close();write_json(REPORT/(tag+'_IMPORT.json'),dict(counts=counts,seconds=time.monotonic()-started));log(stage='chembl_complete',counts=dict(counts))

    def bdb_import(self,part):
        tag='BindingDB_'+part;source='BindingDB_202609';path=RAW/f'BindingDB_{part}_202609_tsv.zip'
        if self.source_done(tag):return
        counts=Counter();started=time.monotonic();rejects=[]
        with gzip.open(OUT/(tag+'_EXCLUSIONS.csv.gz'),'wt',newline='') as f:
            writer=csv.writer(f);writer.writerow(['line','reactant_set_id','reason'])
            for line,d,target,scope,mol,reason in self.prepared_bindingdb(path):
                counts['scanned_rows']+=1
                rid=d['BindingDB Reactant_set_id']
                if reason=='malformed_rows':counts['malformed_rows']+=1;writer.writerow([line,rid,'malformed_row']);continue
                if reason:
                    counts[reason]+=1;writer.writerow([line,rid,reason])
                    rejects.append((source,rid,reason))
                else:
                    values=0
                    for endpoint in ['Kd','Ki','IC50','EC50']:
                        raw=d.get(endpoint+' (nM)','')
                        if not raw:continue
                        rel,value=concentration(raw)
                        if value is None:counts['unusable_endpoint']+=1;writer.writerow([line,rid+':'+endpoint,'unusable_endpoint']);continue
                        date=d.get('Date of publication','');year_match=re.search(r'(?:19|20)\d{2}',date)
                        year=int(year_match[0]) if year_match else None
                        if year is not None and year>2026:counts['future_document_year']+=1;continue
                        reason2=self.add(source,rid+':'+endpoint,path.name,mol,target,endpoint,rel,value,
                            doi=d.get('Article DOI'),pmid=d.get('PMID'),patent=d.get('Patent Number'),document_year=year,
                            assay_id=d.get('BindingDB Entry DOI',''),construct_scope=scope,evidence_tier='source_reported_identifier_resolved',
                            target_name=d.get('Target Name'),source_url=d.get('Link to Ligand-Target Pair in BindingDB'))
                        if reason2:counts[reason2]+=1;writer.writerow([line,rid+':'+endpoint,reason2])
                        else:counts['accepted_observation_attempts']+=1;values+=1
                    if values:counts['rows_with_accepted_endpoint']+=1
                if len(rejects)>=10000:self.c.executemany('insert or ignore into rejected_source_identity values (?,?,?)',rejects);self.c.commit();rejects=[]
                if counts['scanned_rows']%500000==0:log(stage=tag,counts=dict(counts))
        self.flush();self.c.executemany('insert or ignore into rejected_source_identity values (?,?,?)',rejects);self.save_targets()
        write_json(REPORT/(tag+'_IMPORT.json'),dict(counts=counts,seconds=time.monotonic()-started));log(stage=tag+'_complete',counts=dict(counts))

    def extra_import(self):
        tag='Auxiliary_collected_sources'
        if self.source_done(tag):return
        c=sqlite3.connect(f'file:{EXTRA}?mode=ro',uri=True);c.row_factory=sqlite3.Row;counts=Counter()
        exclusions=[]
        for r in c.execute("select * from measurements where source not in ('BindingDB_202609','ChEMBL37_existing')"):
            counts['scanned']+=1;reason=''
            if (r['source'],r['record_id']) in self.qc:reason='manual_source_qc_exclusion'
            elif r['endpoint'] not in ['Kd','Ki','IC50','EC50']:reason='noncomparable_endpoint'
            elif r['target_scope']!='single_accession_construct_unverified':reason='auxiliary_context_or_construct_unverified'
            elif r['organism'] not in ['','Human','Homo sapiens']:reason='nonhuman_target'
            elif len(self.acc.get(r['target_accessions'],set()))!=1:reason='unresolved_human_accession'
            elif r['source']=='HiQBind_v3':reason='structure_attached_label_requires_construct_and_primary_source_review'
            if not reason:
                mol,reason=molecule_identity(r['smiles'],r['ligand_key'])
            if not reason:
                rel,value=concentration(r['value_raw'],r['unit_raw'],r['relation'])
                if value is None:reason='unusable_numeric_value_or_unit'
                else:reason=self.add(r['source'],r['record_id'],EXTRA.name,mol,next(iter(self.acc[r['target_accessions']])),r['endpoint'],rel,value,
                                      doi=r['doi'],pmid=r['pmid'],construct_scope=r['target_scope'],evidence_tier='additional_source_reviewed',source_url=r['source_url'])
            if reason:counts[reason]+=1;exclusions.append(dict(source=r['source'],record_id=r['record_id'],reason=reason,source_url=r['source_url']))
            else:counts['accepted_observation_attempts']+=1
        self.flush();c.close();pd.DataFrame(exclusions).to_csv(OUT/'AUXILIARY_EXCLUSIONS.csv.gz',index=False)
        write_json(REPORT/(tag+'_IMPORT.json'),dict(counts=counts))


def initialize():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    final_manifest=REPORT/'MANIFEST.json'
    if final_manifest.exists():
        manifest=json.loads(final_manifest.read_text())
        for name,expected in manifest['code_sha256'].items():
            if digest(ROOT/name)!=expected:raise ValueError('published dataset code changed; use a new dataset version')
    inputs=[CHEMBL,EXTRA,QC,PREVIOUS,FINAL/'SPR384_FINAL_DETAILED.csv',FINAL/'SPR112_REFERENCE_CONTROLS.csv']+[RAW/f'BindingDB_{p}_202609_tsv.zip' for p in ARCHIVES]
    protocol=dict(version='20260910_v1',scope='human single protein; no project drug/target allowlist',
        endpoints=ENDPOINT_TASK,positive_max_nM=1000,negative_min_nM=10000,
        conflict='any decisive positive and negative within pair/task is quarantined; no majority vote',
        units='nM; exact and censored measurements separate; no conversion of censored values to regression targets',
        chemistry='RDKit sanitized single organic component; preserve stereo, isotope and formal charge; no additional neutralization or tautomer canonicalization beyond standard InChI identity; verify reported full InChIKey; fixed source-order representative SMILES; no metabolite/prodrug replacement',
        target_identity='canonical human amino-acid sequence SHA256; exact accession or exact sequence; wildtype exact fragments may map to canonical; mutant/mismatched constructs quarantined',
        dedup='source record ID plus normalized value identity; conflicting record-ID variants and rejected identity variants quarantined; one label per molecule/sequence/task',
        split='80/10/10 deterministic scaffold groups; connectivity links unite scaffold groups; reserve prior holdout scaffolds and wetlab exact connectivity/sequence pairs',
        documents='shared DOI/PMID/patent reported; additional strict source-purged training subset',
        temporal='source release <=2026-09; missing publication year retained with flag for BindingDB, not an independent temporal benchmark',
        source_limits='HiQBind structure-attached labels and auxiliary domain/lysate/nonhuman measurements remain review-only until primary construct identity is established',
        rdkit_version=rdkit.__version__,unknowns_are_negative=False,training_started=False)
    path=REPORT/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('frozen protocol differs; use a new version')
    write_json(path,protocol)
    identity_path=REPORT/'INPUT_MANIFEST.json'
    if identity_path.exists():
        identities=json.loads(identity_path.read_text())
        for p in inputs:
            r=identities[str(p.relative_to(ROOT))]
            if p.stat().st_size!=r['bytes'] or p.stat().st_mtime_ns!=r['mtime_ns']:raise ValueError(f'input changed {p}')
    else:
        log(stage='hashing_input_sources')
        identities={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for p in inputs}
        write_json(identity_path,identities)
    return identities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['import','finalize','all'],default='all');args=parser.parse_args()
    initialize()
    if (REPORT/'MANIFEST.json').exists():
        manifest=json.loads((REPORT/'MANIFEST.json').read_text())
        for name,expected in manifest['files'].items():
            if digest(OUT/name)!=expected['sha256']:raise ValueError('published artifact changed: '+name)
        log(stage='already_complete_verified',dataset=str(OUT));return
    builder=Builder()
    if args.stage in ['import','all']:
        builder.chembl_import()
        for part in ARCHIVES:builder.bdb_import(part)
        builder.extra_import();builder.save_targets()
    builder.pool.shutdown();builder.c.execute('pragma wal_checkpoint(TRUNCATE)');builder.c.close()
    if args.stage in ['finalize','all']:
        from finalize_training_dataset_20260910 import finalize
        finalize()


if __name__=='__main__':main()
