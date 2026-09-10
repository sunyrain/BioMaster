#!/usr/bin/env python3
"""Publish deduplicated multitask labels, frozen splits, and auditable exclusions."""
from __future__ import annotations
import json,sqlite3,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.training_dataset_v1 import sha_text,split_for_group,record_quality
from biomaster.portable_ranker_v2 import digest
from build_final_training_dataset_20260910 import OUT,REPORT,FINAL,PREVIOUS,ARCHIVES,write_json,log,initialize


def groups(molecules):
    """Unite different scaffold encodings of the same connectivity identity."""
    parents={}
    def root(k):
        parents.setdefault(k,k)
        if parents[k]!=k:parents[k]=root(parents[k])
        return parents[k]
    repeated=molecules[molecules.duplicated('connectivity_key',keep=False)]
    for _,g in repeated.groupby('connectivity_key',sort=False):
        scaffolds=g.scaffold.unique()
        if len(scaffolds)>1:
            roots=[root(s) for s in scaffolds];small=min(roots)
            for r in roots:parents[r]=small
    molecules['split_group']=molecules.scaffold.map(lambda x:'SG:'+sha_text(root(x)))
    molecules['scaffold_split']=molecules.split_group.map(split_for_group)
    return molecules


def citation_rows(observations):
    frames=[]
    for col in ['doi','pmid','patent']:
        q=observations.loc[observations[col].fillna('').ne(''),['pair_id',col]].drop_duplicates().rename(columns={col:'citation'})
        q['citation']=col+':'+q.citation.astype(str);frames.append(q)
    return pd.concat(frames,ignore_index=True).drop_duplicates()


def readme(summary,counts):
    lines=['| 划分 | 任务 | 标签行数 | 阳性 | 阴性 |','|---|---|---:|---:|---:|']
    for r in counts:
        lines.append(f"| {r['split']} | {r['task']} | {r['rows']:,} | {r['positive']:,} | {r['negative']:,} |")
    table='\n'.join(lines)
    content=f'''# BioMaster 全靶点最终训练数据集

数据版本：`20260910_v1`。范围为人源单蛋白；不限制项目720药、384/888靶点或MoA目录。数据整理完成不表示已经训练新模型。

## 交付数量

- 可判定二分类标签：**{summary['binary_pair_task_rows']:,}行**，对应**{summary['binary_unique_drug_target_pairs']:,}个不同药物—蛋白配对**。
- 涉及**{summary['supervised_molecules']:,}个标准化分子身份**、**{summary['supervised_exact_sequences']:,}条不同蛋白序列**。
- 主训练集：**{summary['train_task_rows']:,}行**，其中阳性{summary['train_positive_task_rows']:,}、阴性{summary['train_negative_task_rows']:,}；涉及{summary['train_unique_pairs']:,}个不同配对。
- 严格去除与验证/测试共享文献后的训练备选集：{summary['document_purged_train_rows']:,}行。

同一药物—蛋白可以有Kd/Ki、IC50、EC50等不同任务，故标签行数不等于独立配对数，更不等于独立实验数。所有灰区、冲突与来源重复均有独立审计文件。

{table}

## 直接使用哪些文件

主入口为本目录的 `TRAIN.parquet`、`VALIDATION.parquet`、`TEST.parquet`，并提供同名 `.csv.gz`。这三个表是多任务长表，**必须保留task列**。

若先训练直接亲和力分类器，使用 `TRAIN_AFFINITY_KD_KI.parquet` 及对应验证/测试文件。`ACTIVITY_IC50`和`ACTIVITY_EC50`为不同辅助任务，不能自动当成Kd或直接结合概率。`ANNOTATED_INACTIVITY`仅表示来源明确失活注释，不是未知关系生成的负例。

`MOLECULES.parquet`含标准化SMILES、完整InChIKey、连接身份、骨架、重原子数和形式电荷。`TARGETS.parquet`含实际氨基酸序列、序列SHA256、UniProt/ChEMBL别名和来源等级。**drug_feature_index / target_feature_index只是本次数据版本的内部稠密编号，不可套用旧模型的特征数组。** 全量特征编码和模型训练尚未执行。

```python
from pathlib import Path
import pandas as pd

root = Path('data/processed/biomaster_training_full_20260910_v1')
train = pd.read_parquet(root / 'TRAIN_AFFINITY_KD_KI.parquet')
drugs = pd.read_parquet(root / 'MOLECULES.parquet')
targets = pd.read_parquet(root / 'TARGETS.parquet')
train = train.merge(drugs[['drug_feature_index', 'smiles']], on='drug_feature_index', validate='many_to_one')
train = train.merge(targets[['target_feature_index', 'sequence']], on='target_feature_index', validate='many_to_one')
assert train.binary_label.isin([0, 1]).all()
```

## 标签与身份规则

数值统一为nM。Kd/Ki、IC50、EC50分别保存；≤1000nM为阳性，≥10000nM为弱活性/弱结合阴性；1–10µM为灰区。严格解析`< <= > >=`，不能把上界当成精确浓度，也不能把模糊的`>100nM`当成阴性。明确阳性和阴性同时出现时，整条配对/任务隔离，重复来源不能投票消除矛盾。

ChEMBL要求直接、高可信、无标注变体的靶点归属。Kd/Ki/IC50要求结合类assay，EC50可来自结合或功能assay并独立建任务；保留出处和年份。BindingDB按人源单链、确切序列或可解析accession映射，突变/融合、序列不一致、多链和不明身份单列排除。已知错误记录，包括nevirapine误配NVP-BHG712，按人工QC表隔离。

分子以标准完整InChIKey归并；验证源SMILES与登记key一致。保留立体化学、同位素、形式电荷，不额外执行中和或互变异构体规范化；标准InChI本身仍有自身归一化规则。多有机组分混合物隔离；单有机组分可去除无机对离子。不会将前药直接替换成活性代谢物。模型SMILES取固定来源扫描顺序的首个合格标准化表示。

蛋白以完整氨基酸序列SHA256归并；相同序列的不同数据库ID作为别名。可精确匹配的野生型片段映射到参考序列并保留构建范围；没有外部参考序列的BindingDB条目须有明确人源声明、accession和序列，来源等级单列。序列标准化不是逐篇确认实验构建或治疗机制。

本版逐一对比BindingDB全量包及Articles、PubChem、ChEMBL、Patents、PDSPKi分包。同一source record ID若出现不同有效身份/数值版本，或另一个分包明确拒绝其身份，整组隔离。ChEMBL与BindingDB之间相同数值证据记录保留出处，按配对/任务只生成一个标签，不按重复次数加权。独立实验也可能给出相同数值，所以重复数值组只是保守的审计单位。

## 验证集与实验保护

主划分按化学骨架固定80/10/10哈希；同一连接身份涉及不同骨架表示时合并分组，保证分子连接身份和骨架不跨train/validation/test。实际行数比例不要求恰好80/10/10。无环分子按连接身份分组，不把全部无环分子放成一组。

当前384实验候选和112个参考对照的历史证据分别放入独立holdout文件。此前增量实验的留出药物按整个骨架组保留，不能再次混入主训练。湿实验保留是配对层面的，不宣称对应药物/靶点从未出现于历史训练。

主骨架划分可以共享论文/专利；`shares_document_with_*`明确标记。要求来源分离时，使用 `TRAIN_DOCUMENT_PURGED.parquet`、`VALIDATION_DOCUMENT_PURGED.parquet`与`TEST.parquet`。这仍是回顾性数据，不是前瞻湿实验，也不是靶点同源簇冷启动验证。`seen_by_frozen_parent_connectivity_pair`用于识别旧模型已见关系，不能把其表现冒充独立评估。

回归数据在 `*_REGRESSION.parquet`：按Kd、Ki、IC50、EC50分别取去重精确数值的p尺度中位数；范围超过1 log、分类冲突或与删失界限相抵触的值不进入推荐回归表。删失观测完整保留在证据表，可供以后使用区间损失。

## 追溯与复现

`OBSERVATIONS_WITH_QC.parquet`、`SOURCE_OCCURRENCES.parquet`、`PAIR_CITATIONS.parquet`、`EVIDENCE.sqlite`提供来源记录和引用；`QUARANTINED_SOURCE_RECORDS.parquet`、`GREY_AND_CONFLICT_PAIR_TASKS.parquet`及各来源`*_EXCLUSIONS.csv.gz`保留排除原因。HiQBind附带结构标签以及其他构建/端点/物种不符的数据保留在辅助审查表，本版不硬塞入人源直接亲和力训练。

协议、源文件SHA256、分阶段计数、完整文件清单和验证结果位于 `outputs/training_dataset_full_20260910_v1/`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_final_training_dataset_20260910.py
```

已经完成的导入阶段会复用；源文件或协议改变时拒绝静默继续。大数据文件留在本地，代码、说明和紧凑审计快照进入Git。没有修改生产模型、网站排名或冻结实验表。
'''
    (OUT/'README.md').write_text(content)
    (ROOT/'docs/BIOMASTER_FULL_TRAINING_DATASET_20260911_ZH.md').write_text(content)


def finalize():
    started=time.monotonic();initialize()
    if (REPORT/'MANIFEST.json').exists():
        manifest=json.loads((REPORT/'MANIFEST.json').read_text())
        for name,expected in manifest['files'].items():assert digest(OUT/name)==expected['sha256'],name
        log(stage='already_complete_verified');return
    for source in ['ChEMBL37','Auxiliary_collected_sources']+['BindingDB_'+p for p in ARCHIVES]:
        if not (REPORT/(source+'_IMPORT.json')).exists():raise ValueError('incomplete source '+source)
    log(stage='finalize_loading_evidence')
    c=sqlite3.connect(OUT/'EVIDENCE.sqlite')
    integrity=c.execute('pragma integrity_check').fetchone()[0];assert integrity=='ok'
    obs=pd.read_sql_query('select * from observations',c)
    molecules=pd.read_sql_query('select * from molecules order by molecule_id',c)
    targets=pd.read_sql_query('select * from target_sequences order by target_id',c)
    rejected=pd.read_sql_query("select distinct source_record_id from rejected_source_identity where source='BindingDB_202609'",c)
    origins=pd.read_sql_query('select * from origins',c)
    c.close()
    assert obs.measurement_id.is_unique and molecules.molecule_id.is_unique and targets.target_id.is_unique
    obs['record_qc']=record_quality(obs,set(rejected.source_record_id))
    obs.loc[obs.record_qc.ne('ACCEPTED')].to_parquet(OUT/'QUARANTINED_SOURCE_RECORDS.parquet',index=False)
    obs.to_parquet(OUT/'OBSERVATIONS_WITH_QC.parquet',index=False,compression='zstd')
    origins.to_parquet(OUT/'SOURCE_OCCURRENCES.parquet',index=False,compression='zstd')
    total_observations=len(obs);record_qc=obs.record_qc.value_counts().to_dict()
    obs=obs[obs.record_qc.eq('ACCEPTED')].copy()
    assert not obs.empty
    # Stable source-independent pair identity; target_id encodes an exact amino-acid sequence.
    obs['pair_id']=obs.molecule_id+'__'+obs.target_id
    keys=['pair_id','molecule_id','target_id','task']
    obs['has_positive']=obs.observation_label.eq(1);obs['has_negative']=obs.observation_label.eq(0);obs['has_conflict']=obs.observation_label.eq(-1)
    obs['missing_year']=obs.document_year.isna()
    log(stage='aggregating_pair_task_labels',accepted_observations=len(obs))
    labels=obs.groupby(keys,sort=True).agg(source_records=('measurement_id','size'),
        has_positive=('has_positive','max'),has_negative=('has_negative','max'),has_conflict=('has_conflict','max'),
        source_count=('source','nunique'),earliest_year=('document_year','min'),latest_year=('document_year','max'),
        missing_year_records=('missing_year','sum')).reset_index()
    labels['has_conflict'] |= labels.has_positive & labels.has_negative
    labels['label_status']=np.select([labels.has_conflict,labels.has_positive,labels.has_negative],['CONFLICT','POSITIVE','NEGATIVE'],default='GREY_OR_UNRESOLVED')
    labels['binary_label']=labels.label_status.map({'POSITIVE':1,'NEGATIVE':0}).astype('Int8')
    numeric=obs.drop_duplicates(['pair_id','task','endpoint','relation','value_nM'])
    ncounts=numeric.groupby(['pair_id','task']).size().rename('unique_numeric_evidence').reset_index()
    labels=labels.merge(ncounts,on=['pair_id','task'],validate='one_to_one')
    sources=obs[['pair_id','task','source']].drop_duplicates().groupby(['pair_id','task']).source.agg(lambda v:';'.join(sorted(v))).rename('sources').reset_index()
    labels=labels.merge(sources,on=['pair_id','task'],validate='one_to_one')
    duplicates=obs.groupby(['molecule_id','target_id','endpoint','relation','value_nM'],dropna=False).agg(source_records=('measurement_id','size'),source_count=('source','nunique')).reset_index()
    duplicates[duplicates.source_records.gt(1)].to_parquet(OUT/'REPEATED_NUMERIC_EVIDENCE_GROUPS.parquet',index=False)
    duplicate_cross_sources=int(duplicates.source_count.gt(1).sum())
    del duplicates,numeric
    used_molecules=set(labels.molecule_id);used_targets=set(labels.target_id)
    molecules=groups(molecules[molecules.molecule_id.isin(used_molecules)].copy()).reset_index(drop=True)
    targets=targets[targets.target_id.isin(used_targets)].copy().reset_index(drop=True)
    molecules.insert(0,'drug_feature_index',np.arange(len(molecules),dtype=np.int32))
    targets.insert(0,'target_feature_index',np.arange(len(targets),dtype=np.int32))
    targets['sequence_sha256']=targets.target_id.str.removeprefix('SEQ:')
    targets['sequence_length']=targets.sequence.str.len()
    assert all(sha_text(s)==h for s,h in zip(targets.sequence,targets.sequence_sha256))
    labels=labels.merge(molecules[['molecule_id','drug_feature_index','connectivity_key','split_group','scaffold_split']],on='molecule_id',validate='many_to_one')
    labels=labels.merge(targets[['target_id','target_feature_index']],on='target_id',validate='many_to_one')
    chembl_to_seq={cid:r.target_id for r in targets.itertuples() for cid in r.chembl_ids.split(';') if cid}
    wetlab=pd.read_csv(FINAL/'SPR384_FINAL_DETAILED.csv')
    controls=pd.read_csv(FINAL/'SPR112_REFERENCE_CONTROLS.csv')
    wetpairs={k[:14]+'__'+chembl_to_seq[t] for k,t in zip(wetlab['药物完整InChIKey'],wetlab['新靶点ChEMBL编号']) if t in chembl_to_seq}
    controlpairs={k[:14]+'__'+chembl_to_seq[t] for k,t in zip(controls['对照完整InChIKey'],controls['新靶点ChEMBL编号']) if t in chembl_to_seq}
    labels['connectivity_pair']=labels.connectivity_key+'__'+labels.target_id
    labels['wetlab_candidate']=labels.connectivity_pair.isin(wetpairs)
    labels['wetlab_reference_control']=labels.connectivity_pair.isin(controlpairs)
    exactwet={k+'__'+chembl_to_seq[t] for k,t in zip(wetlab['药物完整InChIKey'],wetlab['新靶点ChEMBL编号']) if t in chembl_to_seq}
    labels['wetlab_candidate_exact_identity']=labels.pair_id.isin(exactwet)
    coverage=wetlab[['原候选编号','药物完整InChIKey','新靶点ChEMBL编号','小分子药物名称','新靶点基因']].copy()
    coverage['canonical_target_id']=coverage['新靶点ChEMBL编号'].map(chembl_to_seq)
    coverage['exact_pair_id']=coverage['药物完整InChIKey']+'__'+coverage.canonical_target_id.fillna('')
    coverage['connectivity_pair']=coverage['药物完整InChIKey'].str[:14]+'__'+coverage.canonical_target_id.fillna('')
    exact_counts=obs.groupby('pair_id').size()
    connected_counts=obs.groupby(obs.molecule_id.str[:14]+'__'+obs.target_id).size()
    coverage['accepted_exact_source_records']=coverage.exact_pair_id.map(exact_counts).fillna(0).astype(int)
    coverage['accepted_connectivity_related_source_records']=coverage.connectivity_pair.map(connected_counts).fillna(0).astype(int)
    coverage['note']='All related historical records reserved; connectivity-only matches are not exact experimental evidence for the frozen candidate.'
    coverage.to_csv(REPORT/'WETLAB384_HISTORICAL_COVERAGE.csv',index=False)
    previous=pd.read_csv(PREVIOUS);oldhold=set(previous.loc[previous.split.eq('holdout'),'ligand_inchikey'].str[:14])
    reserve_groups=set(molecules.loc[molecules.connectivity_key.isin(oldhold),'split_group'])
    labels['previous_benchmark_scaffold']=labels.split_group.isin(reserve_groups)
    labels['split']=np.select([labels.wetlab_candidate,labels.wetlab_reference_control,labels.previous_benchmark_scaffold],
                              ['wetlab_candidate_holdout','reference_control_holdout','previous_benchmark_scaffold_holdout'],default=labels.scaffold_split)
    labels['eligible_binary']=labels.binary_label.notna()
    # Prior model overlap must remain visible when comparing a newly trained model to that model.
    prior_root=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
    prior=pd.read_csv(ROOT/'outputs/biomaster_best_model_20260906/data/fullfit_2025/TRAIN.csv.gz')
    pk=pd.read_csv(ROOT/'outputs/biomaster_bindingdb_incremental_20260910/PARENT_MOLECULE_KEYS.csv.gz')
    pt=pd.read_csv(prior_root/'TARGET_INDEX.csv.gz')
    prior=prior.merge(pk,on='drug_feature_index',validate='many_to_one').merge(pt[['target_feature_index','sequence_sha256']],on='target_feature_index',validate='many_to_one')
    prior_keys=set(prior.ligand_inchikey.str[:14]+'__SEQ:'+prior.sequence_sha256)
    labels['seen_by_frozen_parent_connectivity_pair']=labels.connectivity_pair.isin(prior_keys)
    # Literature separation is an explicit stricter option, not silently assumed for scaffold splits.
    citations=citation_rows(obs)
    pairsplits=labels[['pair_id','split']].drop_duplicates();assert pairsplits.pair_id.is_unique
    citations=citations.merge(pairsplits,on='pair_id',validate='many_to_one')
    testdocs=set(citations.loc[citations.split.eq('test'),'citation'])
    valdocs=set(citations.loc[citations.split.eq('validation'),'citation'])
    sharedtest=set(citations.loc[citations.citation.isin(testdocs),'pair_id'])
    sharedval=set(citations.loc[citations.citation.isin(valdocs),'pair_id'])
    labels['shares_document_with_test']=labels.pair_id.isin(sharedtest)
    labels['shares_document_with_validation']=labels.pair_id.isin(sharedval)
    labels['has_citation']=labels.pair_id.isin(citations.pair_id)
    labels['strict_document_purged_training']=labels.split.eq('train') & ~labels.shares_document_with_test & ~labels.shares_document_with_validation & labels.has_citation & labels.eligible_binary
    citations.to_parquet(OUT/'PAIR_CITATIONS.parquet',index=False,compression='zstd')
    # Affinity and cellular/activity labels can disagree without being the same task's contradiction.
    binary=labels[labels.eligible_binary].copy()
    disagreement=binary.groupby('pair_id').binary_label.nunique().gt(1)
    labels['cross_task_label_disagreement']=labels.pair_id.map(disagreement).fillna(False).astype(bool)
    labels['label_id']=labels.pair_id+'__'+labels.task
    labels=labels.sort_values(['drug_feature_index','target_feature_index','task']).reset_index(drop=True)
    assert labels.label_id.is_unique
    labels.to_parquet(OUT/'ALL_PAIR_TASK_LABELS.parquet',index=False,compression='zstd')
    molecules.to_parquet(OUT/'MOLECULES.parquet',index=False,compression='zstd')
    targets.to_parquet(OUT/'TARGETS.parquet',index=False,compression='zstd')
    molecules.to_csv(OUT/'MOLECULES.csv.gz',index=False)
    targets.to_csv(OUT/'TARGETS.csv.gz',index=False)
    publish_cols=['label_id','pair_id','drug_feature_index','target_feature_index','molecule_id','target_id','task','binary_label','split','split_group',
                  'source_records','unique_numeric_evidence','sources','source_count','earliest_year','latest_year','missing_year_records',
                  'has_citation','shares_document_with_validation','shares_document_with_test','seen_by_frozen_parent_connectivity_pair','cross_task_label_disagreement']
    counts=[]
    for split in ['train','validation','test','wetlab_candidate_holdout','reference_control_holdout','previous_benchmark_scaffold_holdout']:
        selected=labels[labels.eligible_binary & labels.split.eq(split)].copy()
        path=OUT/(split.upper()+'.parquet');selected[publish_cols].to_parquet(path,index=False,compression='zstd')
        selected[publish_cols].to_csv(OUT/(split.upper()+'.csv.gz'),index=False)
        for task,g in selected.groupby('task'):
            counts.append(dict(split=split,task=task,rows=len(g),unique_pairs=g.pair_id.nunique(),positive=int(g.binary_label.sum()),negative=int(g.binary_label.eq(0).sum()),molecules=g.molecule_id.nunique(),targets=g.target_id.nunique()))
            if split in ['train','validation','test']:
                g[publish_cols].to_parquet(OUT/(split.upper()+'_'+task+'.parquet'),index=False,compression='zstd')
    labels[labels.strict_document_purged_training][publish_cols].to_parquet(OUT/'TRAIN_DOCUMENT_PURGED.parquet',index=False,compression='zstd')
    labels[labels.eligible_binary & labels.split.eq('validation') & ~labels.shares_document_with_test & labels.has_citation][publish_cols].to_parquet(OUT/'VALIDATION_DOCUMENT_PURGED.parquet',index=False,compression='zstd')
    labels[labels.eligible_binary & labels.task.eq('AFFINITY_KD_KI')][publish_cols].to_parquet(OUT/'AFFINITY_KD_KI_ALL_SPLITS.parquet',index=False,compression='zstd')
    labels[~labels.eligible_binary].to_parquet(OUT/'GREY_AND_CONFLICT_PAIR_TASKS.parquet',index=False,compression='zstd')
    pd.DataFrame(counts).to_csv(REPORT/'SPLIT_TASK_COUNTS.csv',index=False)
    # Exact regression labels are endpoint-specific; do not pool Ki with Kd or treat bounds as exact.
    rk=['pair_id','molecule_id','target_id','endpoint']
    exact=obs[obs.relation.eq('=') & obs.value_nM.notna()].drop_duplicates(rk+['value_nM']).copy()
    exact['p_activity']=9-np.log10(exact.value_nM)
    regress=exact.groupby(rk).agg(median_p_activity_unique=('p_activity','median'),min_p_activity=('p_activity','min'),max_p_activity=('p_activity','max'),exact_unique_values=('p_activity','size')).reset_index()
    regress['task']=regress.endpoint.map({'Kd':'AFFINITY_KD_KI','Ki':'AFFINITY_KD_KI','IC50':'ACTIVITY_IC50','EC50':'ACTIVITY_EC50'})
    regress=regress.merge(labels[['pair_id','task','drug_feature_index','target_feature_index','split','split_group','label_status','binary_label']],on=['pair_id','task'],validate='many_to_one')
    lower=obs[obs.relation.isin(['>','>='])].groupby(['pair_id','endpoint']).value_nM.max().rename('reported_lower_nM').reset_index()
    upper=obs[obs.relation.isin(['<','<='])].groupby(['pair_id','endpoint']).value_nM.min().rename('reported_upper_nM').reset_index()
    regress=regress.merge(lower,on=['pair_id','endpoint'],how='left',validate='one_to_one').merge(upper,on=['pair_id','endpoint'],how='left',validate='one_to_one')
    minimum_nM=10**(9-regress.max_p_activity);maximum_nM=10**(9-regress.min_p_activity)
    # Boundary touching is conservatively held, including strict > / < reports.
    regress['interval_review']=minimum_nM.le(regress.reported_lower_nM*(1+1e-9)) | maximum_nM.ge(regress.reported_upper_nM*(1-1e-9))
    regress['regression_eligible']=regress.label_status.ne('CONFLICT') & (regress.max_p_activity-regress.min_p_activity).le(1.) & ~regress.interval_review
    regress.to_parquet(OUT/'ENDPOINT_REGRESSION_LABELS_WITH_QC.parquet',index=False,compression='zstd')
    for split in ['train','validation','test']:
        regress[regress.regression_eligible & regress.split.eq(split)].to_parquet(OUT/(split.upper()+'_REGRESSION.parquet'),index=False,compression='zstd')
    # Mandatory release checks cover identifier joins, split leakage, censoring and frozen experimental data.
    main=labels[labels.eligible_binary & labels.split.isin(['train','validation','test'])]
    checks=dict(unique_pair_task_labels=labels.label_id.is_unique,
        binary_labels_only=main.binary_label.isin([0,1]).all(),
        no_scaffold_split_overlap=main.groupby('split_group').split.nunique().max()==1,
        no_connectivity_split_overlap=main.groupby('connectivity_key').split.nunique().max()==1,
        no_pair_task_split_overlap=main.groupby('pair_id').split.nunique().max()==1,
        no_wetlab_candidates_in_main=not main.wetlab_candidate.any(),no_controls_in_main=not main.wetlab_reference_control.any(),
        no_previous_holdout_scaffolds_in_main=not main.previous_benchmark_scaffold.any(),
        no_conflicts_in_main=not main.label_status.eq('CONFLICT').any(),
        molecule_index_complete=labels.drug_feature_index.notna().all(),target_index_complete=labels.target_feature_index.notna().all(),
        strict_train_no_holdout_documents=not labels.loc[labels.strict_document_purged_training,['shares_document_with_test','shares_document_with_validation']].any().any(),
        database_integrity=integrity=='ok')
    identities=json.loads((REPORT/'INPUT_MANIFEST.json').read_text())
    for path,item in identities.items():
        p=ROOT/path;assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns==item['mtime_ns'],path
        if path.startswith(str(FINAL.relative_to(ROOT))):assert digest(p)==item['sha256']
    checks['frozen_experiment_files_unchanged']=True
    checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
    write_json(REPORT/'VALIDATION.json',dict(all_pass=True,checks=checks))
    pd.DataFrame([dict(task=t,label_status=s,rows=n) for (t,s),n in labels.groupby(['task','label_status']).size().items()]).to_csv(REPORT/'LABEL_COUNTS.csv',index=False)
    target_coverage=labels[labels.eligible_binary].groupby('target_id').agg(labeled_pair_tasks=('label_id','size'),unique_molecules=('molecule_id','nunique'),positive=('binary_label','sum')).reset_index()
    target_coverage=target_coverage.merge(targets[['target_id','accessions','chembl_ids','names','identity_tier']],on='target_id')
    target_coverage.to_csv(REPORT/'TARGET_SUPERVISION_COVERAGE.csv',index=False)
    source_stats=obs.groupby('source').agg(records=('measurement_id','size'),molecules=('molecule_id','nunique'),targets=('target_id','nunique'),unique_pairs=('pair_id','nunique')).reset_index()
    source_stats.to_csv(REPORT/'SOURCE_COUNTS.csv',index=False)
    trained=labels[labels.eligible_binary & labels.split.eq('train')]
    valid=labels[labels.eligible_binary]
    summary=dict(status='FINAL_DATASET_READY_NOT_TRAINED',data_directory=str(OUT.relative_to(ROOT)),
        imported_unique_source_observations=total_observations,record_qc_counts=record_qc,
        accepted_source_observations=len(obs),source_occurrences=len(origins),
        cross_source_repeated_numeric_groups=duplicate_cross_sources,
        all_pair_task_rows=len(labels),binary_pair_task_rows=len(valid),binary_unique_drug_target_pairs=valid.pair_id.nunique(),
        binary_positive_task_rows=int(valid.binary_label.sum()),binary_negative_task_rows=int(valid.binary_label.eq(0).sum()),
        supervised_molecules=valid.molecule_id.nunique(),supervised_exact_sequences=valid.target_id.nunique(),
        molecules_exceeding_128_heavy_atoms=int(molecules.heavy_atoms.gt(128).sum()),
        targets_with_U_or_O=int(targets.sequence.str.contains('[UO]').sum()),
        train_task_rows=len(trained),train_unique_pairs=trained.pair_id.nunique(),
        train_positive_task_rows=int(trained.binary_label.sum()),train_negative_task_rows=int(trained.binary_label.eq(0).sum()),
        document_purged_train_rows=int(labels.strict_document_purged_training.sum()),
        source_purged_validation_rows=int((labels.eligible_binary & labels.split.eq('validation') & ~labels.shares_document_with_test & labels.has_citation).sum()),
        regression_eligible_rows=int(regress.regression_eligible.sum()),
        wetlab_candidate_connectivity_related_pairs_reserved=labels.loc[labels.wetlab_candidate,'pair_id'].nunique(),
        wetlab_candidate_exact_pairs_with_historical_observations=labels.loc[labels.wetlab_candidate_exact_identity,'pair_id'].nunique(),
        wetlab_candidate_binary_pairs_reserved=valid.loc[valid.wetlab_candidate,'pair_id'].nunique(),
        split_counts=valid.groupby('split').size().to_dict(),source_counts=source_stats.to_dict('records'),
        tests_passed=all(checks.values()),features_generated=False,training_performed=False,website_modified=False,
        limitations=['Counts are separate tasks, not all pure affinity. Never discard task column when pooling rows.',
          'Sequence-resolved identifiers do not certify every experimental construct or therapeutic mechanism.',
          'BindingDB outside ChEMBL human coverage can use source-reported human sequence identity; tier is retained.',
          'Primary scaffold split is not document-independent. TRAIN_DOCUMENT_PURGED provides a stricter option.',
          'Held-out data are retrospective; old-model seen flags and prior benchmark reserves are provided.',
          'Full molecular/protein feature preparation and model fitting are separate future stages.'],seconds=round(time.monotonic()-started,1))
    write_json(REPORT/'SUMMARY.json',summary)
    readme(summary,counts)
    log(stage='hashing_final_artifacts')
    files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(OUT.iterdir()) if p.is_file() and not p.name.endswith(('-wal','-shm'))}
    manifest=dict(status='FROZEN_DATASET',files=files,protocol_sha256=digest(REPORT/'PROTOCOL.json'),input_manifest_sha256=digest(REPORT/'INPUT_MANIFEST.json'),
                  code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'scripts/build_final_training_dataset_20260910.py',ROOT/'biomaster/training_dataset_v1.py']})
    write_json(REPORT/'MANIFEST.json',manifest)
    log(stage='complete',train_rows=len(trained),binary_pair_task_rows=len(valid),supervised_sequences=valid.target_id.nunique())


if __name__=='__main__':finalize()
