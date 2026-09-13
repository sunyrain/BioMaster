#!/usr/bin/env python3
"""Report completed DTIAM validation only; leave the six-suite TEST gate closed."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from dtiam_ab_common_20260912 import OUT,SOURCE,now,digest,write_json,basic,query_summary
from run_endpoint_multitask_20260911 import Bank,predict,AuxiliaryInteraction

PARENT=ROOT/'outputs/biomaster_endpoint_multitask_20260911'
INTERIM=OUT/'interim_20260913'
KEYS=['pair_id','panel','binary_label']


def main():
    INTERIM.mkdir(exist_ok=True)
    frame=pd.read_parquet(SOURCE/'COMMON_VALIDATION.parquet')
    kd=frame.panel.eq('AFFINITY_KD_KI').to_numpy()
    inactive=frame.panel.eq('EXPLICIT_INACTIVE').to_numpy()
    part=frame[kd].reset_index(drop=True)
    run=OUT/'kdki_inactive__seed_20260921'
    selection=json.loads((run/'SELECTION.json').read_text())
    assert selection['status']=='COMPLETE_FIT_AND_VALIDATION_FROZEN'
    dtiam=pd.read_parquet(run/'VALIDATION_PREDICTIONS.parquet')
    assert frame[KEYS].equals(dtiam[KEYS])
    cals=json.loads((run/'CALIBRATION.json').read_text())
    rows=[];endpoints=[];sources={};predictions=frame[KEYS+['target_id','molecule_id']].copy()

    def collect(model_id,label,scores,prob,threshold,**meta):
        q=query_summary(part,scores[kd])
        summary=basic(frame.loc[kd,'binary_label'],scores[kd],prob[kd],threshold)
        row=dict(model_id=model_id,label=label,**meta,validation_rows=int(kd.sum()),
            positive=int(frame.loc[kd,'binary_label'].sum()),
            kdki_ap=summary['ap'],kdki_auroc=summary['auroc'],kdki_recall=summary['recall'],
            target_macro_ap=q['target']['macro_ap'],drug_macro_ap=q['drug']['macro_ap'],
            balanced_query_ap=q['selection_score'],target_p5=q['target']['p5'],drug_p5=q['drug']['p5'],
            target_queries=q['target']['queries'],drug_queries=q['drug']['queries'],
            explicit_inactive_rows=int(inactive.sum()),explicit_inactive_fpr=float((prob[inactive]>=threshold).mean()))
        for panel,ids in frame.groupby('panel').indices.items():
            m=basic(frame.binary_label.to_numpy(int)[ids],scores[ids],prob[ids],threshold)
            endpoints.append(dict(model_id=model_id,label=label,panel=panel,**m))
            if panel=='ACTIVITY_IC50':row['ic50_ap']=m['ap']
            if panel=='ACTIVITY_EC50':row['ec50_ap']=m['ap']
        rows.append(row);predictions[model_id+'_score']=scores;predictions[model_id+'_prob']=prob

    for view,label in [('native','DTIAM A：原生集成'),('query','DTIAM A：排序优选（随机森林）')]:
        collect('dtiam_'+view,label,dtiam[view+'_score'].to_numpy(float),dtiam[view+'_prob'].to_numpy(float),
            cals[view]['threshold'],family='DTIAM',arm='kdki_inactive',seed=20260921,
            selected_model=selection['choices'][view])
    # Re-score only the existing validation members with the frozen recommended
    # BioMaster checkpoints, to compare false positives at their own frozen thresholds.
    roles=json.loads((ROOT/'outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json').read_text())['roles']
    bank=Bank();torch.set_num_threads(4)
    for role,label in [('binding','BioMaster A：当前亲和模型'),('ranking','BioMaster A：当前联合排序模型'),
                       ('activity','BioMaster B：当前活性模型')]:
        chosen=roles[role];path=ROOT/chosen['checkpoint'];assert digest(path)==chosen['checkpoint_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=AuxiliaryInteraction(state['config'],state['endpoints']).cuda()
        model.load_state_dict(state['model']);scores,_=predict(model,bank,frame)
        cal=json.loads((path.parent/'CALIBRATION.json').read_text())
        probability=expit(cal['slope']*scores+cal['intercept'])
        model_id='biomaster_'+role
        collect(model_id,label,scores,probability,cal['threshold'],family='BioMaster',arm=chosen['arm'],
            seed=chosen['seed'],selected_model=chosen['model_id'])
        expected=json.loads((path.parent/'RESULT.json').read_text())['best_validation']
        # Confirm we reproduced the actual saved checkpoint's validation behavior.
        assert abs(rows[-1]['kdki_ap']-expected['panels']['AFFINITY_KD_KI']['ap'])<1e-4
        assert abs(rows[-1]['balanced_query_ap']-expected['selection_score'])<1e-4
        sources[str(path.relative_to(ROOT))]=digest(path)
        del model;torch.cuda.empty_cache()
    table=pd.DataFrame(rows);table.to_csv(INTERIM/'COMMON_VALIDATION_COMPARISON.csv',index=False)
    pd.DataFrame(endpoints).to_csv(INTERIM/'ENDPOINT_VALIDATION_METRICS.csv',index=False)
    predictions.to_parquet(INTERIM/'COMMON_VALIDATION_PREDICTIONS.parquet',index=False)
    d=table.set_index('model_id');deltas=[]
    for candidate in ['dtiam_native','dtiam_query']:
        for baseline in ['biomaster_binding','biomaster_ranking']:
            for metric in ['kdki_ap','kdki_auroc','balanced_query_ap','target_macro_ap','drug_macro_ap',
                           'target_p5','drug_p5','explicit_inactive_fpr','kdki_recall']:
                deltas.append(dict(candidate=candidate,baseline=baseline,metric=metric,
                    candidate_value=d.loc[candidate,metric],baseline_value=d.loc[baseline,metric],
                    delta=d.loc[candidate,metric]-d.loc[baseline,metric]))
    pd.DataFrame(deltas).to_csv(INTERIM/'VALIDATION_DELTAS.csv',index=False)
    # No supervised work was repeated during recovery; public validation learner
    # results from before the interruption must retain the same row identities.
    old=OUT/'validation_output_repair_20260913/VALIDATION_ALL_LEARNERS.parquet'
    before=pd.read_parquet(old);after=pd.read_parquet(run/'VALIDATION_ALL_LEARNERS.parquet')
    assert before[KEYS].equals(after[KEYS])
    names=selection['validation_scores'];names=[n['model'] for n in names]
    maximum=max(float(np.max(np.abs(before[n].to_numpy()-after[n].to_numpy()))) for n in names)
    assert maximum<=2e-7
    assert digest(run/'FIT_RETURNED.json')==digest(OUT/'validation_output_repair_20260913/FIT_RETURNED.json')
    m=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for name,sha in m['frozen_inputs'].items():assert digest(ROOT/name)==sha
    status=json.loads((OUT/'STATUS.json').read_text())
    fit=json.loads((run/'FIT_RETURNED.json').read_text())
    hours=fit['fit_seconds']/3600
    eta=dict(updated_utc=now(),basis='One measured A suite; B runtime remains unmeasured',
        completed_A_fit_hours=hours,remaining_A_suites=2,remaining_B_suites=3,
        row_scaled_remaining_fit_hours=hours*(2+3*1116270/337570),
        initial_remaining_planning_hours=[30,60],previous_wallclock_ETA_invalidated_by_failure=True)
    write_json(INTERIM/'RECOVERY_ETA.json',eta)
    summary=dict(status='INTERIM_COMMON_VALIDATION_ONLY',created_utc=now(),
        completed_fit_suites_at_report=len(list(OUT.glob('*__seed_*/FIT_RETURNED.json'))),
        completed_validation_suites_at_report=len(list(OUT.glob('*__seed_*/SELECTION.json'))),
        planned_suites=6,runner=status,validation_rows=int(kd.sum()),
        validation_positive=int(frame.loc[kd,'binary_label'].sum()),validation_random_ap=float(part.binary_label.mean()),
        validation_only=True,test_gate_open=(OUT/'TEST_GATE.json').exists(),
        pre_repair_max_probability_difference=maximum,first_A_fit_artifact_unchanged=True,
        model_comparison=rows,eta=eta,decision='Promising classification challenger; ranking gains mixed. Keep current recommendation until gated TEST and repeated fits complete.',
        sources={**sources,str(SOURCE.relative_to(ROOT)/'COMMON_VALIDATION.parquet'):digest(SOURCE/'COMMON_VALIDATION.parquet'),
            str((run/'SELECTION.json').relative_to(ROOT)):digest(run/'SELECTION.json')})
    write_json(INTERIM/'SUMMARY.json',summary)
    lines=['# DTIAM 阶段结果与评价（2026-09-13）','',
        '全套最终测试尚未完成。本报告仅使用公共验证集，不能替代六套拟合完成后的测试对照。',
        '', '第一套 A 的 11 个基础学习器和加权集成已经训练完成，耗时 2.996 小时。队列在 2026-09-11 20:49 UTC 整理验证预测时因程序错误停止，B 当时没有启动。2026-09-13 已修复并恢复，沿用原 A 权重，数据、标签、模型学习参数、选择规则和测试门控保持原定义。',
        '', '错误来自 AutoGluon 返回了所请求集成模型依赖的基础模型预测；结果收集器只为请求模型分配了数组，却遍历了额外依赖，触发 KeyError。修复只收集请求结果，并以单模型独立推理核对，回归测试覆盖额外依赖、缺失输出、重复配对和跨批次恢复。',
        '', '恢复 B 时发现核心进程内存预检查跳过两种 LightGBM：预估另需 46.78 GiB，但旧软预算扣除已驻留数据后只剩 38.44 GiB。已保留该未完成尝试，将 B 核心阶段软总预算从 68 改为 84 GiB（容器实际 90 GiB），A 与独立 FastAI 仍为 68 GiB，实际子进程 RSS 保护线仍为 78 GiB。资源修改及前后哈希已记录，核心模型发生任何失败立即停止；没有缩小 B 或将缺失模型算作完成。原生内存自适应预算可能受可用资源影响，需结合最终拟合参数审计。',
        '', f'报告时队列：{status["stage"]}，已完成验证 {summary["completed_validation_suites_at_report"]}/6；当前 {status.get("current_suite","")}。实时状态请查看上级 STATUS.json。',
        '', '**相同公共 Kd/Ki 验证集：34,685 对，阳性 24,419、阴性 10,266；随机 AP 0.7040。** 靶点查询 519 个、药物查询 38 个，均要求至少 10 对且包含两类。',
        '', '| 模型 | AP | AUROC | 靶点内 AP | 药物内 AP | 双向平均 AP | 靶点内 P@5 | 药物内 P@5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["label"]} | {r["kdki_ap"]:.4f} | {r["kdki_auroc"]:.4f} | {r["target_macro_ap"]:.4f} | {r["drug_macro_ap"]:.4f} | {r["balanced_query_ap"]:.4f} | {r["target_p5"]:.4f} | {r["drug_p5"]:.4f} |')
    lines+=['', 'DTIAM 的排序优选为 RandomForestEntr；这是 DTIAM 的原生 AutoGluon 子模型。原生选择则是 WeightedEnsemble_L2。它们属于同一套拟合的两个预先约定视图，并非两个独立种子。BioMaster 行为已有验证集选定的代表权重，亲和模型种子 20260921，联合排序模型种子 20260923。',
        '', '**明确失活验证面板（11,719 对）**：各模型使用自己的冻结验证最大 F1 阈值，需同时阅读 Kd/Ki 召回率。',
        '', '| 模型 | 失活误报率 | Kd/Ki 阳性召回率 | IC50 AP | EC50 AP |','|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["label"]} | {r["explicit_inactive_fpr"]:.2%} | {r["kdki_recall"]:.2%} | {r["ic50_ap"]:.4f} | {r["ec50_ap"]:.4f} |')
    lines+=['', '评价：DTIAM A 的公共验证分类表现较强，值得作为挑战模型继续完成对照；排序要按方向和 Top-K 分别判断，不能用全局 AP 提升代替实验候选排序提升。原生集成在内部随机留出上的 AUROC 为 0.9918，到了公共骨架验证集为 0.9433，二者划分不同，不能把内部高分当作外部测试成绩。',
        '', '最有价值的初步改善是失活误报：排序优选模型从现有 A 二分类模型的 35.40% 降到 12.49%，同时 Kd/Ki 召回由 91.51% 升到 94.97%。但双向排序 AP 对现有联合排序模型仅提高约 0.0038，且药物内 P@5 为 0.6474，低于其 0.6737。DTIAM A 的 IC50/EC50 AP 也明显低于现有 B 活性模型，因此不是所有任务的统一替代者。这里的 AP、P@5 和失活误报都是对应已测验证配对的指标，不是本轮 384 个未知配对的预期命中率。',
        '', '当前只有一个 DTIAM 种子；用于选择子模型的公共验证集也不能再充当独立性能证明。尚不能断言整体优于现有模型、估算 SPR 命中率，或据此重排已经交付的实验。等待其余拟合及门控测试，之后综合双向排序、明确失活误报、严格来源子集决定是否升级。',
        '', '停止规则审计：NeuralNetTorch 实际运行到 372 轮，保留第 235 轮；FastAI 运行 30 轮，最佳第 13 轮；随机森林和 ExtraTrees 各 300 棵树。LightGBMXT 与 CatBoost 的最佳迭代接近 10,000 轮上限，不能声称每个子模型都已证明收敛。此轮先保留冻结的原生预算，不根据已见验证结果临时扩大个别模型预算。',
        '', f'首套 A 实测 {hours:.3f} 小时；按 B/A 行数比例粗估，剩余拟合约 {eta["row_scaled_remaining_fit_hours"]:.1f} 小时，暂安排恢复后 30—60 小时，仍需以首套 B 实测修正。之前的完成时间估算因中断失效。',
        '', '文件：`COMMON_VALIDATION_COMPARISON.csv`、`ENDPOINT_VALIDATION_METRICS.csv`、`VALIDATION_DELTAS.csv`、`SUMMARY.json`。模型与配对预测留在本地。']
    (INTERIM/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'docs/BIOMASTER_DTIAM_INTERIM_REVIEW_20260913_ZH.md').write_text('\n'.join(lines)+'\n')
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':main()
