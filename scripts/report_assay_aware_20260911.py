#!/usr/bin/env python3
"""Write a validation-selected report only after the training/test gate completes."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'outputs/biomaster_assay_aware_20260911'
PARENT=ROOT/'outputs/biomaster_endpoint_multitask_20260911'


def wait_for_completion():
    while not (OUT/'SUMMARY.json').exists():
        if (OUT/'ERROR.json').exists():
            raise RuntimeError('Training failed; see ERROR.json. No completed result is claimed.')
        process=json.loads((OUT/'PROCESS.json').read_text())
        try:os.kill(process['pid'],0)
        except ProcessLookupError:raise RuntimeError('Training exited without completion summary')
        time.sleep(20)


def strict_assay_flags(observations,train_pairs,test_pairs):
    """Require unseen raw source assay ID, regardless of endpoint or target."""
    known=observations.resolved_assay_id.ne('')
    train_assays=set(observations.loc[known & observations.pair_id.isin(train_pairs),'resolved_assay_id'])
    shared=set(observations.loc[known & observations.resolved_assay_id.isin(train_assays),'pair_id'])
    missing=set(observations.loc[~known,'pair_id'])
    available=set(observations.loc[known,'pair_id'])
    return test_pairs.isin(available) & ~test_pairs.isin(shared | missing)


def strict_source_metrics():
    from audit_assay_context_frontier_20260911 import zipped
    from prepare_assay_aware_20260911 import SOURCE,DATA
    from run_endpoint_ablation_20260911 import basic
    from scipy.special import expit
    obs=pd.read_parquet(DATA/'OBSERVATIONS_WITH_QC.parquet',columns=[
        'source','source_record_id','assay_id','molecule_id','target_id','record_qc'])
    obs=obs[obs.record_qc.eq('ACCEPTED')].copy();obs['pair_id']=obs.molecule_id+'__'+obs.target_id
    obs['resolved_assay_id']=''
    meta=pd.read_parquet(OUT/'CHEMBL_ASSAY_METADATA.parquet')
    available=set(meta.loc[meta.description.fillna('').str.strip().ne(''),'chembl_id'])
    cm=obs.source.eq('ChEMBL37') & obs.assay_id.isin(available)
    obs.loc[cm,'resolved_assay_id']='ChEMBL:'+obs.loc[cm,'assay_id']
    mapping=zipped('rsid_eaids').drop_duplicates()
    assert mapping.groupby('REACTANT_SET_ID').ENTRYID_ASSAYID.nunique().max()==1
    mapping=mapping.drop_duplicates('REACTANT_SET_ID').set_index('REACTANT_SET_ID').ENTRYID_ASSAYID
    desc=zipped('Assays');desc['key']=desc.ENTRYID+'_'+desc.ASSAYID
    description_keys=set(desc.loc[desc.DESCRIPTION.str.strip().ne(''),'key'])
    bm=obs.source.eq('BindingDB_202609');resolved=obs.loc[bm,'source_record_id'].str.split(':').str[0].map(mapping)
    good=resolved.notna() & resolved.isin(description_keys)
    obs.loc[resolved[good].index,'resolved_assay_id']='BindingDB:'+resolved[good]
    predictions=pd.read_parquet(OUT/'TEST_PREDICTIONS.parquet')
    previous=pd.read_parquet(PARENT/'TEST_PREDICTIONS.parquet')
    assert predictions[['pair_id','panel']].equals(previous[['pair_id','panel']])
    flags=predictions[['pair_id','panel','document_disjoint']].copy();rows=[]
    for arm in ['kdki_inactive','all_inactive']:
        train=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet',columns=['pair_id'])
        flags[arm+'_source_assay_disjoint']=strict_assay_flags(obs,train.pair_id,flags.pair_id)
        configs=pd.read_csv(OUT/'OBJECTIVE_COMPARISON_PER_SEED.csv')
        baseline=pd.DataFrame([dict(arm=arm,variant='binary',seed=seed) for seed in json.loads((OUT/'PROTOCOL.json').read_text())['seeds']])
        configs=pd.concat([configs[['arm','variant','seed']],baseline],ignore_index=True)
        for _,r in configs[configs.arm.eq(arm)].iterrows():
            data,model_root=(previous,PARENT) if r.variant=='binary' else (predictions,OUT)
            score=data[f'{arm}__{r.variant}__{r.seed}_score'].to_numpy()
            cal=json.loads((model_root/f'{arm}__{r.variant}__seed_{r.seed}'/'CALIBRATION.json').read_text())
            prob=expit(cal['slope']*score+cal['intercept'])
            for scope in ['raw_source_assay_disjoint','raw_source_assay_and_document_disjoint']:
                mask=flags[arm+'_source_assay_disjoint'].to_numpy()
                if scope.endswith('and_document_disjoint'):mask=mask & flags.document_disjoint.to_numpy()
                for panel in flags.panel.unique():
                    use=mask & flags.panel.eq(panel).to_numpy()
                    if use.any():rows.append(dict(arm=arm,variant=r.variant,seed=r.seed,panel=panel,scope=scope,
                        **basic(predictions.loc[use,'binary_label'],score[use],prob[use],cal['threshold'])))
    flags.to_parquet(OUT/'STRICT_SOURCE_ASSAY_FLAGS.parquet',index=False)
    pd.DataFrame(rows).to_csv(OUT/'STRICT_SOURCE_ASSAY_METRICS.csv',index=False)


def paired_query_intervals():
    pred=pd.read_parquet(OUT/'TEST_PREDICTIONS.parquet')
    old=pd.read_parquet(PARENT/'TEST_PREDICTIONS.parquet')
    assert pred[['pair_id','panel']].equals(old[['pair_id','panel']])
    p=json.loads((OUT/'PROTOCOL.json').read_text())
    meta=pd.read_csv(OUT/'OBJECTIVE_COMPARISON_PER_SEED.csv')
    kd=pred.panel.eq('AFFINITY_KD_KI').to_numpy()
    f=pred[kd].reset_index(drop=True);baseline=old[kd].reset_index(drop=True)
    y=f.binary_label.to_numpy(int);rows=[];cache={};rng=np.random.default_rng(20260911)
    for direction,column in [('target','target_id'),('drug','molecule_id')]:
        groups=[ix for ix in f.groupby(column,sort=True).indices.values() if len(ix)>=10 and 0<y[ix].sum()<len(ix)]
        draws=rng.integers(len(groups),size=(2000,len(groups)))
        def metrics(frame,arm,variant):
            key=(direction,arm,variant)
            if key in cache:return cache[key]
            values=[]
            for seed in p['seeds']:
                scores=frame[f'{arm}__{variant}__{seed}_score'].to_numpy()
                values.append([[average_precision_score(y[ix],scores[ix]),
                    y[ix[np.argsort(-scores[ix],kind='stable')[:5]]].mean()] for ix in groups])
            result=np.mean(values,axis=0);cache[key]=result;return result
        for (arm,variant),_ in meta.groupby(['arm','variant']):
            current=metrics(f,arm,variant);previous=metrics(baseline,arm,'binary')
            delta=current-previous
            for j,metric in enumerate(['macro_AP','P5']):
                boot=delta[draws,j].mean(axis=1)
                rows.append(dict(arm=arm,variant=variant,direction=direction,metric=metric,queries=len(groups),
                    seeds=len(p['seeds']),baseline_value=previous[:,j].mean(),optimized_value=current[:,j].mean(),
                    delta=delta[:,j].mean(),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975),
                    inference_scope='paired query bootstrap conditional on three seeds; exploratory, no multiple-comparison correction'))
    pd.DataFrame(rows).to_csv(OUT/'PAIRED_QUERY_INTERVALS.csv',index=False)


def report():
    summary=json.loads((OUT/'SUMMARY.json').read_text())
    assert summary['status']=='COMPLETE_30_VALIDATION_CONVERGED_FITS'
    f=pd.read_csv(OUT/'OBJECTIVE_COMPARISON.csv')
    old=pd.read_csv(OUT/'EXISTING_BINARY_BASELINES.csv')
    complete=pd.read_csv(OUT/'CONVERGENCE_SUMMARY.csv')
    assert len(complete)==30 and complete.unique_binary_fraction.eq(1).all()
    strict_source_metrics()
    paired_query_intervals()
    chosen=f.sort_values(['validation_selection_score_mean','arm','variant'],ascending=[False,True,True]).iloc[0]
    decision=dict(status='RESEARCH_VALIDATION_SELECTION_NOT_DEPLOYED',arm=chosen.arm,variant=chosen.variant,
        validation_selection_score=float(chosen.validation_selection_score_mean),
        selection_rule='Highest mean validation balanced query AP across the three preregistered seeds.',
        test_used_to_choose_variant=False,production_replaced=False)
    (OUT/'VALIDATION_SELECTION.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2)+'\n')
    labels={'balanced':'温和平衡','rotate150':'150轮换','regression':'温和平衡+回归','assay':'温和平衡+回归+实验内差值',
        'continue_A':'A→A继续训练','adapt_from_B':'B→A亲和适应','binary':'原二分类基线'}
    lines=['# A/B实验上下文优化重训结果','',
        f"30次拟合已全部按验证平台期停止，完成时间：{summary['completed_utc']}。所有训练成员均被实际访问。",'',
        f"按预设验证指标，候选方案为 **{chosen.arm} / {labels[chosen.variant]}**。它是研究模型选择结果，未自动替换生产模型。",'',
        '| 模型 | 方案 | Kd/Ki AP | 平衡查询AP | 靶点P5 | 药物P5 | 明确失活FPR |',
        '|---|---|---:|---:|---:|---:|---:|']
    for _,r in pd.concat([old,f],ignore_index=True).iterrows():
        name='A' if r.arm=='kdki_inactive' else 'B'
        lines.append(f"| {name} | {labels[r.variant]} | {r.ap_mean:.4f} | {r.balanced_query_ap_mean:.4f} | {r.target_p5_mean:.4f} | {r.drug_p5_mean:.4f} | {r.explicit_inactive_fpr_mean:.4f} |")
    lines+=['','以上为3种子均值。平衡查询AP是Kd/Ki靶点宏平均AP与药物宏平均AP的均值；不同指标不保证同时改善。',
        '失活FPR使用每个模型在验证集选定的最大F1阈值；召回率可能不同，不能当作固定召回率误报比较或真实SPR失败率。','',
        '当前测试是已经查看过的诊断面板。STRICT_SOURCE_ASSAY_METRICS.csv按原始来源assay编号隔离，跨端点或跨靶点共享同编号也排除；同时报告文献隔离子集。TEST_METRICS.csv中原source_assay命名仅表示实验—靶点—端点组合未见，以STRICT文件作为严格来源指标。两者都不是新采集的独立测试。',
        'PAIRED_QUERY_INTERVALS.csv提供相对于同组原二分类模型的查询配对bootstrap区间：以3种子查询指标均值为条件，未校正多方案比较，不能替代独立确认。','',
        'A→A与B→A使用相同下游A数据、辅助损失和学习率；上游数据量及训练计算仍不同。150轮换采用靶点/类别预算，未复刻历史静态骨架精选规则。','',
        '全部损失只使用训练成员的已测标签；测试评分发生在30个验证选定模型全部冻结之后。研究模型、实验记录和亲和辅助值含义分别保留。','',
        f"停止步数范围：{complete.optimizer_steps.min():,}—{complete.optimizer_steps.max():,}；新拟合累计运行约{complete.seconds.sum()/60:.1f}分钟，不含最终评估。",'',
        '复现入口：scripts/prepare_assay_aware_20260911.py、scripts/run_assay_aware_20260911.py、scripts/report_assay_aware_20260911.py。']
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    (OUT/'REPORT_STATUS.json').write_text(json.dumps({'status':'COMPLETE','report':'RESULTS_ZH.md'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    try:
        if args.wait:wait_for_completion()
        report()
    except Exception as error:
        (OUT/'REPORT_STATUS.json').write_text(json.dumps({'status':'FAILED','error':str(error)},ensure_ascii=False,indent=2)+'\n')
        raise
