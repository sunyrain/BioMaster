#!/usr/bin/env python3
"""Publish completed predictions atomically without changing the SPR baseline."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.frontier_dti import ALL_MODELS,BASE_MODELS,NEW_MODELS,NAMES,metrics,common_ranks,finite_json,DIRECTORY
OUT=ROOT/DIRECTORY

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(finite_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)

def running(path):
    try:
        pid=int(path.read_text());stat=Path(f'/proc/{pid}/stat').read_text().split();return stat[2]!='Z'
    except (OSError,ValueError):return False

def baseline(manifest):
    path=OUT/'BASELINE_SCORES.csv'
    if not path.exists():
        from biomaster.explorer_data import ExplorerData
        data=ExplorerData(ROOT,infer=False);data.ensure_loaded()
        pairs=data.pairs.copy();pairs['pair_id']=pairs.ligand_inchikey+'__'+pairs.target_chembl_id
        subset=pairs[pairs.pair_id.isin(manifest.pair_id)][['pair_id',*BASE_MODELS]]
        assert len(subset)==len(manifest)
        subset.to_csv(path,index=False)
    return pd.read_csv(path)

def collect():
    manifest=pd.read_csv(OUT/'INPUT_MANIFEST.csv',keep_default_na=False)
    manifest['label']=pd.to_numeric(manifest.label,errors='coerce')
    data=manifest.merge(baseline(manifest),on='pair_id',validate='one_to_one')
    protocol=json.loads((OUT/'PROTOCOL.json').read_text())
    frozen=ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
    if hashlib.sha256(frozen.read_bytes()).hexdigest()!=protocol['original_spr_sha256']:raise ValueError('Frozen SPR384 changed')
    statuses={};per_pair={}
    for model in NEW_MODELS:
        directory=OUT/model;alive=running(directory/'PID')
        if model=='nesso':
            rows=[]
            for path in (directory/'predictions').glob('*/affinity.json'):
                try:
                    j=json.loads(path.read_text());score=float(j['affinity_probability_binary']);pic50=6-float(j['affinity_pred_value'])
                    if np.isfinite(score) and np.isfinite(pic50):rows.append(dict(pair_id=path.parent.name,score=score,pic50=pic50,status='completed',reason=''))
                except (ValueError,KeyError,OSError):continue
            successful={r['pair_id'] for r in rows}
            try:history=json.loads((directory/'QUEUE_HISTORY.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):history=[]
            for stage in history:
                if stage['stage']=='SPR384':attempted=data[data.cohort.eq('SPR384')]
                elif stage['stage']=='BINDINGDB_regular':attempted=data[data.cohort.eq('BINDINGDB479') & data.protein_length.le(2000)]
                else:attempted=data[data.pair_id.eq(stage['stage'].removeprefix('long_'))]
                for key in attempted.pair_id:
                    if key not in successful:rows.append(dict(pair_id=key,score=None,pic50=None,status='failed',reason='long_protein_timeout_900s' if stage['result']=='timeout' else 'inference_missing_after_stage'))
            pred=pd.DataFrame(rows,columns=['pair_id','score','pic50','status','reason'])
            state='running' if alive else ('completed' if len(successful)==len(data) else 'stopped_with_missing')
            status=dict(state=state,completed=len(successful),total=len(data))
        else:
            try:pred=pd.read_csv(directory/'PREDICTIONS.csv')
            except (FileNotFoundError,pd.errors.EmptyDataError):pred=pd.DataFrame(columns=['pair_id','score','status','reason'])
            try:status=json.loads((directory/'STATUS.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):status=dict(state='preparing' if alive else 'not_running')
            if not alive and status['state'] not in ('completed','completed_with_missing'):status['state']='stopped_before_completion'
        if pred.pair_id.duplicated().any():raise ValueError('Duplicate predictions: '+model)
        pred['score']=pd.to_numeric(pred.score,errors='coerce');pred.loc[~pred.status.eq('completed'),'score']=np.nan
        pred.loc[~np.isfinite(pred.score),'score']=np.nan
        scores=pred.set_index('pair_id').score;data[model]=data.pair_id.map(scores)
        if model=='nesso':data['nesso_pic50']=data.pair_id.map(pred.set_index('pair_id').pic50)
        per_pair[model]=pred.set_index('pair_id').to_dict('index')
        spr=data.cohort.eq('SPR384');bench=~spr
        pair_status=data.pair_id.map(pred.set_index('pair_id').status)
        statuses[model]=dict(id=model,name=NAMES[model],**status,running=alive,spr_scored=int(data.loc[spr,model].notna().sum()),spr_total=384,
            benchmark_scored=int(data.loc[bench,model].notna().sum()),benchmark_total=479,
            spr_unavailable=int(pair_status.loc[spr].eq('unavailable').sum()),spr_failed=int(pair_status.loc[spr].eq('failed').sum()),
            failed=int(pred.status.eq('failed').sum()),unavailable=int(pred.status.eq('unavailable').sum()))
    spr=data[data.cohort.eq('SPR384')].copy();bench=data[data.cohort.eq('BINDINGDB479')].copy()
    common=bench[list(ALL_MODELS)].notna().all(axis=1)
    metric_rows=[]
    for name,frame in [('BINDINGDB479_COMMON',bench[common]),('BINDINGDB175_COMMON',bench[common & bench.ab_unseen.eq(True)])]:
        for model in ALL_MODELS+('nesso_pic50',):metric_rows.append(metrics(frame,model,name))
    # Individual new-model coverage and baseline comparators on exactly that subset.
    for new in NEW_MODELS:
        for suffix,mask in [('479',pd.Series(True,index=bench.index)),('175',bench.ab_unseen.eq(True))]:
            subset=bench[mask & bench[new].notna()]
            if new=='nesso':metric_rows.append(metrics(subset,'nesso_pic50',f'BINDINGDB{suffix}_{new}_AVAILABLE'))
            for model in BASE_MODELS+(new,):
                matched=subset[subset[model].notna()]
                metric_rows.append(metrics(matched,model,f'BINDINGDB{suffix}_{new}_AVAILABLE'))
                if model in BASE_MODELS and len(matched)!=len(subset):metric_rows.append(metrics(matched,new,f'BINDINGDB{suffix}_{new}_MATCHED_{model}'))
    ranks,percent,common_n=common_ranks(spr)
    items=[]
    for i,row in spr.iterrows():
        ps=percent.loc[i];span=float(ps.max()-ps.min()) if ps.notna().all() else None
        item={k:row[k] for k in ('pair_id','drug_id','target_id','drug_name','gene','candidate_id','priority')}
        item.update(scores={m:row[m] for m in ALL_MODELS},nesso_pic50=row['nesso_pic50'],
            common_ranks=ranks.loc[i].to_dict(),common_percentiles=ps.to_dict(),common_denominator=common_n,
            disagreement_span=span,in_common_pool=ps.notna().all(),
            status={m:per_pair[m].get(row.pair_id,{}).get('status','pending' if statuses[m]['running'] else 'missing') for m in NEW_MODELS},
            reasons={m:per_pair[m].get(row.pair_id,{}).get('reason','') for m in NEW_MODELS})
        items.append(item)
    agreements=[]
    for i,a in enumerate(ALL_MODELS):
        for b in ALL_MODELS[i+1:]:
            valid=spr[[a,b]].dropna();n=len(valid)
            rho=float(valid[a].corr(valid[b],method='spearman')) if n>2 and valid[a].nunique()>1 and valid[b].nunique()>1 else None
            agreements.append(dict(a=a,b=b,pairs=n,spearman=rho))
    finished=all(not s['running'] for s in statuses.values())
    snapshot=dict(available=True,updated_utc=datetime.now(timezone.utc).isoformat(),finished=finished,
        provisional=not finished,scope='冻结SPR384候选；对照另计',models=list(statuses.values()),model_names=NAMES,
        model_order=list(ALL_MODELS),items=items,common_spr_pairs=common_n,common_benchmark_pairs=int(common.sum()),
        metrics=metric_rows,agreements=agreements,
        interpretation='共同覆盖的同一批候选内比较排名。分歧只是模型排序差异，不代表实测结合或实验成败；未覆盖不等于阴性。',
        benchmark_note='BindingDB回顾性标签；175子集仅排除了我们A/B的训练及验证重叠。公开新模型训练重叠未排除，不能宣称独立泛化性能。',
        ranking_note='排名分母是七模型共同覆盖的SPR候选数；运行中会变化，不是完整384靶点或720药物矩阵排名。分位跨度越大，排序分歧越大。',
        source_urls={'nesso':'https://github.com/recursionpharma/nesso','probematch':'https://github.com/developer-hq/ProbeMatchDTI','dtbind':'https://github.com/liqy09/DTBind'},
        nesso_note='pIC50 = 6 − 官方回归值；这是模型预测，不能当作实测Kd。',
        frozen_sha256=protocol['original_spr_sha256'])
    atomic_json(OUT/'WEBSITE_SNAPSHOT.json',snapshot)
    pd.DataFrame(metric_rows).to_csv(OUT/'METRICS.csv',index=False)
    pd.DataFrame(agreements).to_csv(OUT/'SPR384_AGREEMENT.csv',index=False)
    temp=OUT/'ALL_PREDICTIONS.tmp';data.to_csv(temp,index=False);temp.replace(OUT/'ALL_PREDICTIONS.csv')
    export=spr[['candidate_id','priority','drug_name','gene','drug_id','target_id','pair_id',*ALL_MODELS,'nesso_pic50']].copy()
    for model in ALL_MODELS:export[model+'_common_rank']=ranks[model]
    export['common_denominator']=common_n
    for model in NEW_MODELS:
        export[model+'_status']=export.pair_id.map(lambda key:per_pair[model].get(key,{}).get('status','pending' if statuses[model]['running'] else 'missing'))
        export[model+'_reason']=export.pair_id.map(lambda key:per_pair[model].get(key,{}).get('reason',''))
    temp=OUT/'SPR384_MODEL_REVIEW.tmp';export.to_csv(temp,index=False,encoding='utf-8-sig');temp.replace(OUT/'SPR384_MODEL_REVIEW.csv')
    report=['# SPR384 新模型复核 · 自动更新', '', f"更新：{snapshot['updated_utc']}", '',
        '本文件与网站每30秒同步。运行未完成时指标只代表当前已完成样本。', '',
        '| 模型 | SPR384已评分 | BindingDB479已评分 | 状态 |', '|---|---:|---:|---|']
    report += [f"| {s['name']} | {s['spr_scored']}/384 | {s['benchmark_scored']}/479 | {s['state']} |" for s in statuses.values()]
    report += ['',f'七模型共同覆盖：SPR {common_n}/384；BindingDB {int(common.sum())}/479。', '',
        '原384及交付实验CSV未修改；逐对CSV见 SPR384_MODEL_REVIEW.csv。', '',
        '## ProbeMatchDTI 完整测试', '', '| 集合 | 模型 | 样本 | 阳性 | 阴性 | AP | AUROC |', '|---|---|---:|---:|---:|---:|---:|']
    for r in metric_rows:
        if r['scope'] in ('BINDINGDB479_probematch_AVAILABLE','BINDINGDB175_probematch_AVAILABLE') and r['ap'] is not None:
            report.append(f"| {r['scope'].split('_')[0]} | {r['name']} | {r['pairs']} | {r['positive']} | {r['negative']} | {r['ap']:.4f} | {r['auroc']:.4f} |")
    report += ['',snapshot['benchmark_note'], '',snapshot['ranking_note'], '',
        'DTBind采用模型自身Sigmoid输出；官方预测脚本重复Sigmoid仅用于示例复现，已从正式评分路径移除。',
        'DTBind只在官方蛋白序列完全匹配、图特征齐全时预测，不以零特征补缺；没有复合物坐标的候选不输出其亲和模型结果。',
        'ProbeMatchDTI保留官方1200蛋白位置、100分子词元窗口；长输入截断记录于其逐对预测文件。',
        'Nesso长蛋白（>2000aa）在SPR384之后单独尝试，每对限时15分钟；超时列为缺失，不计阴性。', '']
    temp=OUT/'RUN_REPORT.tmp';temp.write_text('\n'.join(report));temp.replace(OUT/'RUN_REPORT.md')
    print(snapshot['updated_utc'],{k:(v['spr_scored'],v['benchmark_scored'],v['state']) for k,v in statuses.items()},flush=True)
    return finished

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        done=collect()
        if not args.watch or done:break
        time.sleep(30)
