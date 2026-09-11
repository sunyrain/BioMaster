#!/usr/bin/env python3
"""Compare completed DTIAM suites with BioMaster using exact shared TEST members."""
import json
import numpy as np
import pandas as pd
from dtiam_ab_common_20260912 import (ROOT,OUT,SOURCE,ARMS,SEEDS,now,digest,write_json,
                                     basic,query_summary,query_rows)

PARENT=ROOT/'outputs/biomaster_endpoint_multitask_20260911'
CONSOLIDATION=ROOT/'outputs/biomaster_model_consolidation_20260911'
KEYS=['pair_id','panel','binary_label']


def summarize():
    frame=pd.read_parquet(SOURCE/'COMMON_TEST.parquet')
    all_predictions=frame[['pair_id','panel','binary_label','molecule_id','target_id','split_group','document_disjoint']].copy()
    strict=pd.read_parquet(ROOT/'outputs/biomaster_assay_aware_20260911/STRICT_SOURCE_ASSAY_FLAGS.parquet')
    assert frame[['pair_id','panel']].equals(strict[['pair_id','panel']])
    coverage=pd.read_parquet(ROOT/'outputs/biomaster_old_production_comparison_20260911/TEST_PREDICTIONS.parquet',
        columns=KEYS+['target_training_coverage','old_train_pair_seen'])
    assert frame[KEYS].equals(coverage[KEYS])
    wetlab=pd.read_csv(ROOT/'outputs/biomaster_model_decision_audit_20260911/WETLAB112_TARGET_BENCHMARK.csv')
    wetlab_targets=set(wetlab.sequence_target_id.dropna())
    metric_rows=[];query_details=[];ledger=[];fit_rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            run=OUT/f'{arm}__seed_{seed}'
            complete=json.loads((run/'TEST_COMPLETE.json').read_text())
            assert complete['status']=='COMPLETE'
            assert digest(run/'TEST_PREDICTIONS.parquet')==complete['predictions_sha256']
            pred=pd.read_parquet(run/'TEST_PREDICTIONS.parquet')
            assert frame[KEYS].equals(pred[KEYS])
            selection=json.loads((run/'SELECTION.json').read_text())
            fit=json.loads((run/'FIT_RETURNED.json').read_text())
            calibration=json.loads((run/'CALIBRATION.json').read_text())
            model_info=json.loads((run/'AUTOGLUON_INFO.json').read_text())['model_info']
            for name,info in model_info.items():
                fit_rows.append(dict(arm=arm,seed=seed,model=name,model_type=info.get('model_type'),
                    fit_seconds=info.get('fit_time'),internal_validation_score=info.get('val_score'),
                    native_parameters=json.dumps(info.get('hyperparameters',{})),
                    fitted_parameters=json.dumps(info.get('hyperparameters_fit',{})),
                    stopping_claim='native budget/early stopping; see fitted parameters and RUN.log'))
            for view,model in selection['choices'].items():
                prefix=f'dtiam__{arm}__{view}__{seed}'
                for col in ['score','prob','native_probability']:
                    all_predictions[prefix+'_'+col]=pred[view+'_'+col]
                score=pred[view+'_score'].to_numpy(float);prob=pred[view+'_prob'].to_numpy(float)
                threshold=calibration[view]['threshold']
                scopes={'all':np.ones(len(frame),bool),
                    'document_disjoint':strict.document_disjoint.to_numpy(bool),
                    'raw_source_assay_disjoint':strict[arm+'_source_assay_disjoint'].to_numpy(bool),
                    'raw_source_assay_and_document_disjoint':strict.document_disjoint.to_numpy(bool)&strict[arm+'_source_assay_disjoint'].to_numpy(bool),
                    'spr112_targets':frame.target_id.isin(wetlab_targets).to_numpy(),
                    'old_parent_pair_unseen':~coverage.old_train_pair_seen.to_numpy(bool)}
                scopes.update({'targets_'+str(c):coverage.target_training_coverage.eq(c).to_numpy()
                    for c in sorted(coverage.target_training_coverage.unique())})
                for panel in frame.panel.unique():
                    for scope,mask in scopes.items():
                        use=mask & frame.panel.eq(panel).to_numpy()
                        if use.any():
                            metric_rows.append(dict(model_id=prefix,arm=arm,view=view,seed=seed,model=model,
                                panel=panel,scope=scope,**basic(frame.loc[use,'binary_label'],score[use],prob[use],threshold)))
                    use=frame.panel.eq(panel).to_numpy();part=frame[use].reset_index(drop=True)
                    for direction,col in [('target','target_id'),('drug','molecule_id')]:
                        for q in query_rows(part,score[use],col):
                            query_details.append(dict(model_id=prefix,arm=arm,view=view,seed=seed,
                                panel=panel,direction=direction,**q))
                kd=frame.panel.eq('AFFINITY_KD_KI').to_numpy()
                q=query_summary(frame[kd].reset_index(drop=True),score[kd])
                validation=next(r for r in selection['validation_scores'] if r['model']==model)
                ledger.append(dict(model_id=prefix,stage='dtiam',arm=arm,variant=view,seed=seed,
                    status='COMPLETE_COMPATIBLE_NATIVE_FIT',checkpoint=str((run/'predictor').relative_to(ROOT)),
                    selected_native_model=model,train_pairs=fit['training_member_pool'],
                    base_train_rows=fit['base_train_rows'],internal_holdout_rows=fit['internal_holdout_rows'],
                    fit_seconds=fit['fit_seconds'],validation_binding_ap=validation['ap'],
                    validation_balanced_query_ap=validation['selection_score'],
                    **{k:v for k,v in basic(frame.loc[kd,'binary_label'],score[kd],prob[kd],threshold).items()
                       if k in ['pairs','positive','negative','ap','auroc']},
                    target_queries=q['target']['queries'],target_macro_ap=q['target']['macro_ap'],
                    target_p5=q['target']['p5'],target_p20=q['target']['p20'],
                    drug_queries=q['drug']['queries'],drug_macro_ap=q['drug']['macro_ap'],
                    drug_p5=q['drug']['p5'],drug_p20=q['drug']['p20'],balanced_query_ap=q['selection_score']))
    all_predictions.to_parquet(OUT/'TEST_PREDICTIONS.parquet',index=False)
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(OUT/'ENDPOINT_METRICS_PER_MODEL.csv',index=False)
    queries=pd.DataFrame(query_details);queries.to_parquet(OUT/'QUERY_METRICS_PER_MODEL.parquet',index=False)
    pd.DataFrame(fit_rows).to_csv(OUT/'NATIVE_FIT_BUDGET_AUDIT.csv',index=False)
    current=pd.DataFrame(ledger);current.to_csv(OUT/'MODEL_LEDGER.csv',index=False)
    previous=pd.read_csv(CONSOLIDATION/'MODEL_LEDGER.csv')
    pd.concat([previous,current],ignore_index=True).to_csv(OUT/'ALL_MODEL_COMPARISON.csv',index=False)
    metriccols=['ap','auroc','target_macro_ap','drug_macro_ap','target_p5','drug_p5','balanced_query_ap',
                'validation_binding_ap','validation_balanced_query_ap','fit_seconds']
    family=current.groupby(['arm','variant'])[metriccols].agg(['mean','std'])
    family.columns=['_'.join(c) for c in family.columns]
    family.reset_index().to_csv(OUT/'FAMILY_COMPARISON.csv',index=False)
    grouped=metrics.groupby(['arm','view','panel','scope'])[
        ['pairs','positive','negative','ap','auroc','false_positive_rate','top1pct_precision','top5pct_precision']].agg(['mean','std'])
    grouped.columns=['_'.join(c) for c in grouped.columns]
    grouped.reset_index().to_csv(OUT/'ENDPOINT_FAMILY_COMPARISON.csv',index=False)
    paired(queries,frame)
    return current,family.reset_index(),metrics


def paired(queries,frame):
    old=pd.read_parquet(PARENT/'TEST_PREDICTIONS.parquet')
    assert old[KEYS].equals(frame[KEYS])
    use=frame.panel.eq('AFFINITY_KD_KI').to_numpy();kd=frame[use].reset_index(drop=True)
    records=[];rng=np.random.default_rng(20260912)
    for direction,col in [('target','target_id'),('drug','molecule_id')]:
        for arm in ARMS:
            for variant in ['binary','joint']:
                base=[]
                for seed in SEEDS:
                    scores=old.loc[use,f'{arm}__{variant}__{seed}_score'].to_numpy()
                    q=pd.DataFrame(query_rows(kd,scores,col)).set_index('query').sort_index()
                    base.append(q[['ap','p5','p20']])
                base_mean=sum(base)/len(base)
                draws=rng.integers(len(base_mean),size=(2000,len(base_mean)))
                for view in ['native','query']:
                    new=[]
                    for seed in SEEDS:
                        part=queries[(queries.arm==arm)&(queries.view==view)&(queries.seed==seed)&
                            (queries.panel=='AFFINITY_KD_KI')&(queries.direction==direction)].set_index('query').sort_index()
                        assert part.index.equals(base_mean.index)
                        new.append(part[['ap','p5','p20']])
                    current=sum(new)/len(new);delta=current-base_mean
                    for metric in ['ap','p5','p20']:
                        valid=delta[metric].notna().to_numpy();d=delta.loc[valid,metric].to_numpy()
                        if len(d)==len(delta):sample=draws
                        else:sample=rng.integers(len(d),size=(2000,len(d)))
                        boot=d[sample].mean(axis=1)
                        records.append(dict(arm=arm,view=view,baseline_variant=variant,direction=direction,metric=metric,
                            queries=len(d),seeds=3,baseline=base_mean.loc[valid,metric].mean(),dtiam=current.loc[valid,metric].mean(),
                            delta=d.mean(),ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975),
                            scope='paired query bootstrap conditional on three seeds; no multiple-comparison correction'))
    pd.DataFrame(records).to_csv(OUT/'PAIRED_QUERY_INTERVALS.csv',index=False)


def main():
    current,family,metrics=summarize()
    manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for name,sha in manifest['frozen_inputs'].items():
        assert digest(ROOT/name)==sha
    # Hash inference artifacts, including every dependency of every retained native learner.
    model_files={}
    for arm in ARMS:
        for seed in SEEDS:
            for f in sorted((OUT/f'{arm}__seed_{seed}'/'predictor').rglob('*')):
                if f.is_file():model_files[str(f.relative_to(ROOT))]=digest(f)
    write_json(OUT/'TRAINED_ARTIFACTS.json',dict(created_utc=now(),files=model_files))
    summary=dict(status='COMPLETE_6_NATIVE_DTIAM_SUITES',completed_utc=now(),fit_suites=6,
        default_base_learners_per_suite=11,weighted_ensembles_per_suite=1,reported_selection_views=12,
        note='12 views are two selections from each of six fits, not 12 independent training runs',
        family_comparison=family.to_dict('records'),production_and_spr_frozen=True,
        pretrained_encoders='frozen; no BerMol/ESM2 pretraining or finetuning',
        test_selection='none; all native and query selections frozen on internal/common validation',
        artifacts_sha256=digest(OUT/'TRAINED_ARTIFACTS.json'))
    write_json(OUT/'SUMMARY.json',summary)
    lines=['# DTIAM 九月 A/B 同数据对照结果','',f'完成时间：{now()}。',
        '', '已完成 A、B 各三个种子的 DTIAM 兼容重训。每套包含 11 个原生基础学习器和一个加权集成；报告原生选择和公共验证集排序选择两个视图。',
        '', 'A 使用 337,570 对，B 使用 1,116,270 对；每组均包含 105,368 对明确失活。测试 Kd/Ki 为 34,327 对（阳性 23,190，阴性 11,137），随机 AP 0.675656、随机 AUROC 0.5。',
        '', '| 数据组 / 选择 | KdKi AP（均值±标准差） | AUROC | 靶点内 AP | 药物内 AP | 双向平均 AP |',
        '|---|---:|---:|---:|---:|---:|']
    for r in family.itertuples():
        lines.append(f'| {r.arm} / {r.variant} | {r.ap_mean:.4f}±{r.ap_std:.4f} | {r.auroc_mean:.4f} | {r.target_macro_ap_mean:.4f} | {r.drug_macro_ap_mean:.4f} | {r.balanced_query_ap_mean:.4f} |')
    lines+=['','现有 BioMaster 及旧生产模型与 DTIAM 的逐模型对照见 `ALL_MODEL_COMPARISON.csv`。排序差异及条件置信区间见 `PAIRED_QUERY_INTERVALS.csv`；失活误报、IC50/EC50 与严格来源子集见 `ENDPOINT_FAMILY_COMPARISON.csv`。',
        '', 'DTIAM 使用原生 AutoGluon 内部留出进行早停及集成，内部成员表已保留；公共验证集没有交给 fit。原生固定迭代预算与验证早停逐模型记录于 `NATIVE_FIT_BUDGET_AUDIT.csv`，不能统一称为已证明收敛。',
        '', 'DTIAM 和 BioMaster 的编码器、学习器及样本权重均不同，结果比较检验完整方法表现，不能单独归因于训练数据量。当前测试仍是回顾性骨架留出，不能当作 SPR 实际命中率。',
        '', '生产模型、网站及 SPR384 实验交付表保持原冻结版本。']
    (ROOT/'docs/BIOMASTER_DTIAM_AB_RESULTS_20260912_ZH.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
