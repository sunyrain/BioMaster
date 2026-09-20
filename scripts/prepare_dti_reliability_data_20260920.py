#!/usr/bin/env python3
"""Prepare auditable retrospective training views; never fit or deploy a model."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/processed/biomaster_training_full_20260910_v1'
LEGACY = ROOT/'outputs/biomaster_endpoint_ablation_20260911'
OUT = ROOT/'data/research/dti_reliability_20260920_v1'
REPORT = ROOT/'outputs/dti_research_preparation_20260920'
SPLIT_SEED = 20260920
ARMS = {'A':'kdki_inactive','B':'all_inactive'}
VIEWS = ['scaffold_replay','source_purged','cold_target','double_cold','temporal_conservative']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class UnionFind:
    def __init__(self,items=()):
        self.parents={x:x for x in items}
    def find(self,x):
        self.parents.setdefault(x,x)
        if self.parents[x]!=x:
            self.parents[x]=self.find(self.parents[x])
        return self.parents[x]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:
            lo,hi=sorted([a,b]);self.parents[hi]=lo


def hashed_split(identifier):
    value=int(hashlib.sha256(f'{SPLIT_SEED}|{identifier}'.encode()).hexdigest()[:16],16)/2**64
    return 'train' if value<.8 else ('validation' if value<.9 else 'test')


def ensure_homology_search():
    targets=pd.read_parquet(SOURCE/'TARGETS.parquet')
    fasta=OUT/'TARGETS.fasta'
    sequence_column='sequence' if 'sequence' in targets else 'target_sequence'
    content=''.join(f'>t{r.target_feature_index}\n{getattr(r,sequence_column)}\n' for r in targets.itertuples())
    if fasta.exists():
        assert fasta.read_text()==content,'Existing homology FASTA differs from the target registry'
    else:fasta.write_text(content)
    command=['mmseqs','easy-search',str(fasta),str(fasta),str(OUT/'PROTEIN_HOMOLOGY_HITS.tsv'),str(OUT/'mmseqs_tmp'),
        '--threads','4','--min-seq-id','0.3','-c','0.8','--cov-mode','0','-s','7.5','--max-seqs','10000',
        '-e','0.001','--format-output','query,target,pident,qcov,tcov,evalue','--remove-tmp-files','1']
    if not (OUT/'PROTEIN_HOMOLOGY_HITS.tsv').exists():
        with (OUT/'MMSEQS.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(REPORT/'HOMOLOGY_SEARCH.json',dict(command=command,fasta_sha256=sha(fasta),
        hits_sha256=sha(OUT/'PROTEIN_HOMOLOGY_HITS.tsv'),target_registry_sha256=sha(SOURCE/'TARGETS.parquet'),
        version=subprocess.check_output(['mmseqs','version'],text=True).strip()))


def protein_clusters(targets):
    hits=pd.read_csv(OUT/'PROTEIN_HOMOLOGY_HITS.tsv',sep='\t',names=['query','target','pident','qcov','tcov','evalue'])
    ids={'t'+str(r.target_feature_index):r.target_id for r in targets.itertuples()}
    uf=UnionFind(ids.values())
    # Recheck rounded output conservatively. MMseqs already applies these cutoffs internally.
    assert hits.pident.ge(29.999).all() and hits.qcov.ge(.799).all() and hits.tcov.ge(.799).all()
    for r in hits.itertuples():uf.union(ids[r.query],ids[r.target])
    clusters=targets[['target_feature_index','target_id','sequence_length']].copy()
    clusters['homology_cluster']=clusters.target_id.map(uf.find)
    clusters['target_split']=clusters.homology_cluster.map(hashed_split)
    lookup=clusters.set_index('target_id').target_split
    assert all(lookup[ids[r.query]]==lookup[ids[r.target]] for r in hits.itertuples())
    clusters.to_parquet(OUT/'TARGET_CLUSTERS.parquet',index=False)
    clusters.groupby('homology_cluster',as_index=False).agg(
        sequences=('target_id','size'),split=('target_split','first')
    ).to_csv(REPORT/'PROTEIN_CLUSTER_COUNTS.csv',index=False)
    return clusters,dict(sequences=len(targets),clusters=int(clusters.homology_cluster.nunique()),
                         hit_rows=len(hits),max_cluster_sequences=int(clusters.groupby('homology_cluster').size().max()),
                         mmseqs_version=subprocess.check_output(['mmseqs','version'],text=True).strip(),
                         identity=.30,bidirectional_coverage=.80,evalue=.001,sensitivity=7.5,
                         method='Connected components of detected MMseqs all-against-all hits; heuristic homology screen, not exhaustive biological homology certification.')


def document_partitions(canonical):
    # Alias links come from co-reported DOI/PMID/patent on an observation, never merely a shared drug-target pair.
    cols=['molecule_id','target_id','doi','pmid','patent','record_qc','document_year']
    obs=pd.read_parquet(SOURCE/'OBSERVATIONS_WITH_QC.parquet',columns=cols)
    obs=obs[obs.record_qc.eq('ACCEPTED')].copy()
    for col,prefix in [('doi','doi:'),('pmid','pmid:'),('patent','patent:')]:
        text=obs[col].fillna('').astype(str).str.strip()
        if col=='doi':
            text=text.str.lower().str.replace(r'^https?://(?:dx\.)?doi\.org/','',regex=True)
        if col=='pmid':text=text.str.replace(r'\.0$','',regex=True)
        if col=='patent':text=text.str.upper()
        obs[col]=text.where(text.eq(''),prefix+text)
    uf=UnionFind()
    for r in obs[['doi','pmid','patent']].drop_duplicates().itertuples(index=False,name=None):
        tokens=[x for x in r if x]
        for token in tokens:uf.union(tokens[0],token)
    # A source can be cited under different identifiers in two databases.
    primary=obs.doi.where(obs.doi.ne(''),obs.pmid.where(obs.pmid.ne(''),obs.patent))
    mapping={key:uf.find(key) for key in primary.unique() if key}
    obs['document_group']=primary.map(mapping)
    obs['pair_id']=obs.molecule_id+'__'+obs.target_id
    obs['missing_document']=primary.eq('')
    obs['missing_year']=obs.document_year.isna()
    observed=obs[obs.pair_id.isin(canonical.pair_id)]
    pair_qc=observed.groupby('pair_id',as_index=False).agg(
        missing_any_document=('missing_document','max'),missing_any_year=('missing_year','max'),
        first_year=('document_year','min'),last_year=('document_year','max'))
    citations=observed.loc[~observed.missing_document,['pair_id','document_group']].drop_duplicates()
    citations=citations.merge(canonical[['pair_id','pair_index','original_split']],on='pair_id',validate='many_to_one')
    # Strict source split keeps original scaffold partitions and purges overlaps asymmetrically.
    held=set(citations.loc[citations.original_split.isin(['validation','test']),'document_group'])
    test_docs=set(citations.loc[citations.original_split.eq('test'),'document_group'])
    train_bad=set(citations.loc[citations.document_group.isin(held),'pair_id'])
    validation_bad=set(citations.loc[citations.document_group.isin(test_docs),'pair_id'])
    c=canonical.merge(pair_qc,on='pair_id',how='left',validate='one_to_one')
    c['source_purged']=c.original_split
    unknown=c.missing_any_document.fillna(True)
    conflict=(c.original_split.eq('train') & c.pair_id.isin(train_bad)) | (c.original_split.eq('validation') & c.pair_id.isin(validation_bad))
    c.loc[unknown|conflict,'source_purged']='excluded'
    # No missing-date or straddling observations may contribute future measurements to training labels.
    c['temporal_conservative']='excluded'
    dated=~c.missing_any_year.fillna(True)
    c.loc[dated & c.last_year.le(2022),'temporal_conservative']='train'
    c.loc[dated & c.first_year.ge(2023) & c.last_year.le(2023),'temporal_conservative']='validation'
    c.loc[dated & c.first_year.ge(2024),'temporal_conservative']='test'
    part= c.set_index('pair_id').source_purged
    docs={split:set(citations.loc[citations.pair_id.map(part).eq(split),'document_group']) for split in ['train','validation','test']}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert docs[a].isdisjoint(docs[b])
    citations[['pair_index','document_group']].to_parquet(OUT/'PAIR_DOCUMENT_GROUPS.parquet',index=False)
    c[['pair_index','missing_any_document','missing_any_year','first_year','last_year']].to_parquet(OUT/'PAIR_SOURCE_QC.parquet',index=False)
    aliases=pd.DataFrame([dict(identifier=x,document_group=uf.find(x)) for x in sorted(uf.parents)])
    aliases.to_csv(OUT/'DOCUMENT_ALIASES.csv.gz',index=False)
    counts=dict(alias_identifiers=len(aliases),document_groups=int(aliases.document_group.nunique()),
                pairs_missing_any_source=int(unknown.sum()),pairs_with_known_source=int((~unknown).sum()),
                source_purged_document_overlap=0,
                temporal_policy='All accepted observations for the pair dated <=2022 / entirely2023 / first>=2024; missing or boundary-straddling dates excluded. Retrospective, not newly collected.')
    return c,counts


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    ensure_homology_search()
    inputs=[SOURCE/n for n in ['ALL_PAIR_TASK_LABELS.parquet','TARGETS.parquet','MOLECULES.parquet',
        'OBSERVATIONS_WITH_QC.parquet','ENDPOINT_REGRESSION_LABELS_WITH_QC.parquet']]
    inputs += [LEGACY/f'{name}_PAIR_AUDIT.parquet' for name in ARMS.values()]
    inputs += [OUT/'TARGETS.fasta',OUT/'PROTEIN_HOMOLOGY_HITS.tsv']
    arms={}
    for arm,name in ARMS.items():
        f=pd.read_parquet(LEGACY/f'{name}_PAIR_AUDIT.parquet')
        f=f[f.eligible].copy()
        assert f.pair_id.is_unique and f.binary_label.isin([0,1]).all()
        assert not f.conflict.any() and not f.inactive_positive_conflict.any()
        assert f.loc[f.explicit_inactive,'binary_label'].eq(0).all()
        arms[arm]=f
    cols=['pair_id','molecule_id','target_id','drug_feature_index','target_feature_index','split_group','split']
    canonical=pd.concat([f[cols] for f in arms.values()]).drop_duplicates().sort_values('pair_id').reset_index(drop=True)
    assert canonical.pair_id.is_unique
    canonical.insert(0,'pair_index',np.arange(len(canonical),dtype=np.int64))
    canonical=canonical.rename(columns={'split':'original_split'})
    protected=pd.read_parquet(SOURCE/'ALL_PAIR_TASK_LABELS.parquet',columns=['pair_id','wetlab_candidate','wetlab_reference_control','previous_benchmark_scaffold'])
    protected=set(protected.loc[protected.wetlab_candidate | protected.wetlab_reference_control | protected.previous_benchmark_scaffold,'pair_id'])
    assert set(canonical.pair_id).isdisjoint(protected)
    canonical.to_parquet(OUT/'CANONICAL_PAIRS.parquet',index=False,compression='zstd')
    print('Canonical pairs',len(canonical),flush=True)
    clusters,cluster_summary=protein_clusters(pd.read_parquet(SOURCE/'TARGETS.parquet'))
    c,source_summary=document_partitions(canonical)
    c=c.merge(clusters[['target_id','homology_cluster','target_split']],on='target_id',validate='many_to_one')
    c['scaffold_replay']=c.original_split
    c['cold_target']=c.target_split
    c['double_cold']=c.original_split.where(c.original_split.eq(c.target_split),'excluded')
    c[['pair_index','original_split','homology_cluster']+VIEWS].to_parquet(OUT/'PARTITIONS.parquet',index=False,compression='zstd')
    print('Homology and document partitions prepared',flush=True)
    counts=[];query_counts=[];materialized=[]
    for arm,old in arms.items():
        f=old.merge(c[['pair_id','pair_index','homology_cluster']+VIEWS],on='pair_id',validate='one_to_one')
        keep=['pair_index','drug_feature_index','target_feature_index','binary_label','explicit_inactive']
        for view in VIEWS:
            splits={}
            for split in ['train','validation','test','excluded']:
                part=f[f[view].eq(split)]
                counts.append(dict(arm=arm,view=view,split=split,pairs=len(part),positive=int(part.binary_label.sum()),
                    negative=int(part.binary_label.eq(0).sum()),explicit_inactive=int(part.explicit_inactive.sum()),
                    molecules=part.molecule_id.nunique(),targets=part.target_id.nunique(),
                    homology_clusters=part.homology_cluster.nunique(),scaffold_groups=part.split_group.nunique()))
                if split=='excluded':continue
                path=OUT/f'{arm}_{view}_{split}.parquet';part[keep].to_parquet(path,index=False,compression='zstd');materialized.append(path)
                splits[split]=part
                for direction,key in [('target_to_drug','target_id'),('drug_to_target','molecule_id')]:
                    q=part.groupby(key).binary_label.agg(['size','sum'])
                    mixed=q['sum'].gt(0)&q['sum'].lt(q['size'])
                    query_counts.append(dict(arm=arm,view=view,split=split,direction=direction,total_queries=len(q),
                        mixed_queries=int(mixed.sum()),mixed_queries_ge10=int((mixed&q['size'].ge(10)).sum()),
                        all_negative_queries=int(q['sum'].eq(0).sum()),all_positive_queries=int(q['sum'].eq(q['size']).sum())))
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                assert set(splits[a].pair_index).isdisjoint(splits[b].pair_index)
                if view in ['scaffold_replay','source_purged','double_cold']:
                    assert set(splits[a].split_group).isdisjoint(splits[b].split_group)
                if view in ['cold_target','double_cold']:
                    assert set(splits[a].homology_cluster).isdisjoint(splits[b].homology_cluster)
        # Retain a compact full member table for joins with endpoint/assay supervision.
        f[keep].to_parquet(OUT/f'{arm}_ALL_MEMBERS.parquet',index=False,compression='zstd')
    counts=pd.DataFrame(counts);counts.to_csv(REPORT/'DATASET_COUNTS.csv',index=False)
    pd.DataFrame(query_counts).to_csv(REPORT/'QUERY_SUPPORT_COUNTS.csv',index=False)
    for arm,expected in [('A',337570),('B',1116270)]:
        row=counts[(counts.arm==arm)&(counts.view=='scaffold_replay')&(counts.split=='train')].iloc[0]
        assert row.pairs==expected and row.explicit_inactive==105368
    # Matched and nested members isolate specific data-composition changes; not all confounders.
    a=arms['A'].query('split == "train"').copy();b=arms['B'].query('split == "train"').copy()
    common=a.merge(b[['pair_id','binary_label']],on='pair_id',suffixes=('','_b'),validate='one_to_one')
    conflict=common.binary_label.ne(common.binary_label_b)
    common=common[~conflict].copy()
    extra=b[~b.pair_id.isin(a.pair_id)&b.target_id.isin(a.target_id)].copy()
    strata=['target_feature_index','binary_label','explicit_inactive']
    quota=a.groupby(strata).size().to_frame('a').join(b.groupby(strata).size().to_frame('b'),how='inner')
    quota['quota']=quota[['a','b']].min(axis=1).astype(int)
    matched={}
    for name,frame in [('A',a),('B',b)]:
        frame=frame.copy();frame['sample_hash']=frame.pair_id.map(lambda x:hashlib.sha256(f'{SPLIT_SEED}|match|{x}'.encode()).hexdigest())
        frame=frame.join(quota['quota'],on=strata,how='inner').sort_values('sample_hash')
        frame['within_stratum']=frame.groupby(strata).cumcount()
        matched[name]=frame[frame.within_stratum.lt(frame.quota)]
    pd.testing.assert_series_equal(matched['A'].groupby(strata).size(),matched['B'].groupby(strata).size())
    special={'A_B_COMMON_TRAIN':common,'B_EXTRA_SHARED_TARGETS_TRAIN':extra,
             'A_MATCHED_TRAIN':matched['A'],'B_MATCHED_TRAIN':matched['B']}
    data_contrasts=[]
    for name,f in special.items():
        f=f.merge(canonical[['pair_id','pair_index']],on='pair_id',validate='one_to_one')
        f[['pair_index','drug_feature_index','target_feature_index','binary_label','explicit_inactive']].to_parquet(OUT/f'{name}.parquet',index=False,compression='zstd')
        data_contrasts.append(dict(name=name,pairs=len(f),positive=int(f.binary_label.sum()),explicit_inactive=int(f.explicit_inactive.sum()),targets=f.target_id.nunique()))
    pd.DataFrame(data_contrasts).to_csv(REPORT/'DATA_CONTRAST_COUNTS.csv',index=False)
    # Fixed A/B-common evaluation endpoints for the data-composition experiment.
    for split in ['validation','test']:
        left=arms['A'].query('split == @split')
        right=arms['B'].query('split == @split')
        common_eval=left.merge(right[['pair_id','binary_label']],on='pair_id',suffixes=('','_b'),validate='one_to_one')
        common_eval=common_eval[common_eval.binary_label.eq(common_eval.binary_label_b)]
        common_eval=common_eval.merge(canonical[['pair_id','pair_index']],on='pair_id',validate='one_to_one')
        common_eval[keep].to_parquet(OUT/f'A_B_COMMON_{split.upper()}.parquet',index=False,compression='zstd')
    # Exact endpoint regression remains separate from binary inactivity and mixed-endpoint labels.
    reg=pd.read_parquet(SOURCE/'ENDPOINT_REGRESSION_LABELS_WITH_QC.parquet')
    reg=reg[reg.regression_eligible].merge(canonical[['pair_id','pair_index']],on='pair_id',validate='many_to_one')
    reg_cols=['pair_index','endpoint','median_p_activity_unique','min_p_activity','max_p_activity','exact_unique_values']
    assert not reg.duplicated(['pair_index','endpoint']).any()
    assert np.isfinite(reg.median_p_activity_unique).all()
    reg[reg_cols].to_parquet(OUT/'REGRESSION_EXACT.parquet',index=False,compression='zstd')
    # Reuse previously audited genuine-assay auxiliary observations, with provenance and explicit scope.
    assay_path=ROOT/'outputs/biomaster_assay_aware_20260911/all_inactive_ASSAY_TRAIN.parquet'
    assay=pd.read_parquet(assay_path).merge(canonical[['pair_id','pair_index']],on='pair_id',validate='many_to_one')
    assay.to_parquet(OUT/'ASSAY_AUXILIARY.parquet',index=False,compression='zstd');inputs.append(assay_path)
    aux=[]
    for arm in ARMS:
        for view in VIEWS:
            for split in ['train','validation','test']:
                members=pd.read_parquet(OUT/f'{arm}_{view}_{split}.parquet',columns=['pair_index'])
                r=reg[reg.pair_index.isin(members.pair_index)]
                s=assay[assay.pair_index.isin(members.pair_index)]
                if arm=='A':r=r[r.endpoint.isin(['Kd','Ki'])];s=s[s.endpoint.isin(['Kd','Ki'])]
                group_n=s.groupby('assay_group').pair_index.nunique()
                good=set(group_n[group_n.ge(5)].index)
                for endpoint in ['Kd','Ki','IC50','EC50']:
                    aux.append(dict(arm=arm,view=view,split=split,endpoint=endpoint,
                        exact_regression_rows=int(r.endpoint.eq(endpoint).sum()),
                        assay_rows_min5=int((s.endpoint.eq(endpoint)&s.assay_group.isin(good)).sum())))
    pd.DataFrame(aux).to_csv(REPORT/'AUXILIARY_SUPPORT_COUNTS.csv',index=False)
    # Feature files are referenced in place, retaining their original dense indices.
    feature_specs=[('bermol','outputs/biomaster_dtiam_ab_20260912/features/BERMOL.npy','outputs/biomaster_dtiam_ab_20260912/features/BERMOL_DONE.npy','drug_feature_index'),
        ('esm2_dtiam','outputs/biomaster_dtiam_ab_20260912/features/ESM2.npy','outputs/biomaster_dtiam_ab_20260912/features/ESM2_DONE.npy','target_feature_index'),
        ('drugclip','outputs/biomaster_endpoint_ablation_20260911/features/DRUG_CLIP.npy','outputs/biomaster_endpoint_ablation_20260911/features/DRUG_DONE.npy','drug_feature_index'),
        ('morgan','outputs/biomaster_endpoint_ablation_20260911/features/MORGAN.npy','outputs/biomaster_endpoint_ablation_20260911/features/CHEM_DONE.npy','drug_feature_index'),
        ('esm2_full','outputs/biomaster_endpoint_ablation_20260911/features/TARGET.npy','outputs/biomaster_endpoint_ablation_20260911/features/TARGET_DONE.npy','target_feature_index')]
    features=[]
    for name,path,done,key in feature_specs:
        arr=np.load(ROOT/path,mmap_mode='r');flags=np.load(ROOT/done,mmap_mode='r');ids=np.sort(canonical[key].unique())
        assert flags[ids].all(),name
        availability_path='outputs/biomaster_endpoint_ablation_20260911/features/AVAILABLE.npy' if name=='drugclip' else None
        availability=np.load(ROOT/availability_path,mmap_mode='r') if availability_path else None
        zero=0
        for start in range(0,len(ids),8192):
            values=arr[ids[start:start+8192]]
            assert np.isfinite(values).all(),name
            zero_mask=np.all(values==0,axis=1)
            zero+=int(zero_mask.sum())
            if availability is not None:
                assert np.array_equal(zero_mask,~availability[ids[start:start+8192]]),name
            else:
                assert not zero_mask.any(),name
        features.append(dict(name=name,path=path,done_path=done,shape=list(arr.shape),dtype=str(arr.dtype),
            required_rows=len(ids),nonfinite_rows=0,zero_rows=zero,sha256=sha(ROOT/path),bytes=(ROOT/path).stat().st_size,
            availability_path=availability_path,availability_sha256=sha(ROOT/availability_path) if availability_path else None,
            missing_policy='Explicit availability mask; zero is an unavailable placeholder, never a valid embedding.' if availability_path else 'All required rows verified finite and nonzero.'))
        print('Verified feature bank',name,flush=True)
    write_json(REPORT/'FEATURE_REGISTRY.json',features)
    summary=dict(status='PREPARED_AND_VALIDATED',created_utc=datetime.now(timezone.utc).isoformat(),
        canonical_pairs=len(canonical),molecules=int(canonical.molecule_id.nunique()),targets=int(canonical.target_id.nunique()),
        views=VIEWS,materialized_member_files=len(materialized),protein_clusters=cluster_summary,source_audit=source_summary,
        protected_pairs_in_training_universe=0,matched_train_pairs=len(matched['A']),overlapping_AB_label_conflicts=int(conflict.sum()),
        exact_regression_rows=len(reg),assay_auxiliary_rows=len(assay),assay_scope='Reused audited original-training assay records only; every new split must filter by its member IDs and recalculate group>=5.',
        no_unmeasured_negative_pairs=True,new_training_started=False,wetlab_results_access_status='UNKNOWN_PENDING_USER',
        interpretation='Retrospective repartition for freshly initialized supervised models. Previously trained checkpoints are ineligible for independent cold/temporal claims.',
        limitations=['Homology screening is heuristic at30% identity/80% bidirectional coverage; it is not an exhaustive domain-homology exclusion.',
            'Source groups conservatively union co-reported DOI/PMID/patent; undisclosed or incorrect source metadata cannot be certified.',
            'Conservative temporal view uses wholly dated intervals, not a complete historical database reconstruction.',
            'Kd/Ki classifier pooling, IC50/EC50, exact regression and inactivity remain distinct declared views; no numeric affinity fabricated for inactivity.',
            'No new external-confirmatory or prospectively unseen labels were created by repartitioning.'])
    write_json(REPORT/'DATA_PREPARATION_SUMMARY.json',summary)
    write_json(REPORT/'DATA_MANIFEST.json',dict(inputs={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in inputs},
        prepared={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.glob('*.parquet'))},
        producer_sha256=sha(Path(__file__)),split_seed=SPLIT_SEED))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':prepare()
