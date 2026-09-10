#!/usr/bin/env python3
"""Audit both frozen ablation stages and generate the Chinese results report."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/biomaster_v3_incremental_20260906'
STAGE=OUT/'old_relation_stage'
EV=OUT/'evaluation'
REPORT=ROOT/'docs/BIOMASTER_V3_INCREMENTAL_ABLATION_20260906_ZH.md'
NAMES={'frozen_v3':'冻结旧 V3',
    'A_residual_bce':'A 小型残差＋实测 BCE', 'B_residual_rank':'B A＋双向实测排序',
    'C_evidence_bce':'C A＋正负证据', 'D_evidence_rank':'D C＋双向实测排序',
    'E_pretrained_rank':'E D＋预训练交互', 'F_teacher_rank':'F D＋近邻蒸馏',
    'G_matched_control':'G 第二组匹配训练对照', 'H_old_relation':'H G＋已知关系检索监督',
    'I_old_relation_evidence':'I H＋正负证据', 'J_old_relation_pretrained':'J I＋预训练交互',
    'K_old_relation_teacher':'K I＋近邻蒸馏',
    'positive_nearest':'训练阳性近邻（排除自身）','pn_logistic':'冻结正负证据逻辑回归'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    status=json.loads((OUT/'STATUS.json').read_text());assert status['status']=='COMPLETE'
    first=json.loads((OUT/'SELECTION.json').read_text())
    second=json.loads((STAGE/'SELECTION.json').read_text())
    release=json.loads((OUT/'TEST_RELEASE.json').read_text())
    assert release['selection_sha256']==sha(OUT/'SELECTION.json')
    assert release['stage2_selection_sha256']==sha(STAGE/'SELECTION.json')
    assert second['stage1_selection_sha256']==sha(OUT/'SELECTION.json')
    winner=second['winner'];assert winner==status['validation_selected_winner']
    config=json.loads((OUT/'PROTOCOL.json').read_text())
    config2=json.loads((STAGE/'PROTOCOL.json').read_text())
    baseline=first['results'][0]['baseline_validation']
    rows=[]
    checkpoints=[]
    for stage,s in [(OUT,first),(STAGE,second)]:
        matched={}
        for r in s['results']:
            p=stage/'runs'/r['variant']/f"seed_{r['seed']}"/'BEST.pt'
            assert sha(p)==r['checkpoint_sha256'];checkpoints.append(p)
            epochs=r['observed_exposure_sha256'] if stage==OUT else r['matched_exposure_sha256']
            if r['seed'] in matched:assert matched[r['seed']]==epochs
            else:matched[r['seed']]=epochs
            v=r['validation']
            for direction in ['d2t','t2d']:
                assert v[direction]['ap']>=baseline[direction]['ap']-.01
                assert v[direction]['measured_ap']>=baseline[direction]['measured_ap']-.01
            rows.append({'stage':1 if stage==OUT else 2,'variant':r['variant'],'seed':r['seed'],
                'selected_epoch':r['selected_epoch'],'selection':v['selection'],
                'd2t_ap':v['d2t']['ap'],'t2d_ap':v['t2d']['ap'],'d2t_r20':v['d2t']['r20'],'t2d_r20':v['t2d']['r20'],
                'measured_d2t_ap':v['d2t']['measured_ap'],'measured_t2d_ap':v['t2d']['measured_ap'],
                'seconds':r['seconds'],'trainable_parameters':r['trainable_parameters']})
    validation=pd.DataFrame(rows);validation.to_csv(EV/'VALIDATION_ABLATION.csv',index=False)
    means=validation.groupby('variant',sort=False).mean(numeric_only=True)
    for v,value in second['all_validation_scores'].items():assert np.isclose(means.loc[v,'selection'],value,atol=1e-14)
    assert winner==max(second['all_validation_scores'],key=second['all_validation_scores'].get)
    summary=pd.read_csv(EV/'SUMMARY.csv')
    observed=pd.read_csv(EV/'OBSERVED_TEST.csv')
    contrasts=pd.read_csv(EV/'PAIRED_CONTRASTS.csv')
    previous=pd.read_csv(ROOT/'outputs/biomaster_old_drug_bidirectional_20260906/evaluation/SUMMARY.csv')
    for direction,new_direction in [('drug_to_target','d2t'),('target_to_drug','t2d')]:
        old=previous[(previous.scope=='project_720x384')&(previous.model=='v3_support_ensemble')&(previous.direction==direction)].iloc[0]
        new=summary[(summary.scope=='project_720x384')&(summary.model=='frozen_v3')&(summary.direction==new_direction)].iloc[0]
        assert np.isclose(old.known_ap,new.ap,atol=1e-14)
    score_manifest=json.loads((OUT/'scores/MANIFEST.json').read_text())
    for name,digest in score_manifest['scores_sha256'].items():
        assert sha(OUT/'scores'/f'{name}.npz')==digest
    # Actual teacher rows must be training old drugs in both directions.
    old_drugs=pd.read_csv(ROOT/'outputs/biomaster_old_drug_bidirectional_20260906/OLD_DRUGS_720.csv')
    teacher_count=0
    for seed in config['seeds']:
        with np.load(OUT/f'TEACHER_{seed}.npz') as f:
            for pairs in f.values():
                ids=pairs[:,:2].astype(int)//384
                assert old_drugs.training_role.iloc[ids.reshape(-1)].eq('train').all()
                teacher_count+=len(pairs)
    primary=json.loads((EV/'WINNER_VS_FROZEN_V3.json').read_text())

    def val_table(stage):
        lines=['| 方法 | 正向 AP | 逆向 AP | 正向 R@20 | 逆向 R@20 | 验证综合分 | 选中 epoch（三种子） |',
               '|---|---:|---:|---:|---:|---:|---|',
               f"| 冻结旧 V3 | {baseline['d2t']['ap']:.4f} | {baseline['t2d']['ap']:.4f} | {baseline['d2t']['r20']:.4f} | {baseline['t2d']['r20']:.4f} | {baseline['selection']:.4f} | 0 |"]
        for variant in validation[validation.stage.eq(stage)].variant.unique():
            row=means.loc[variant]
            selected=validation[validation.variant.eq(variant)].selected_epoch.astype(str).str.cat(sep='/')
            vals=[row[x] for x in ['d2t_ap','t2d_ap','d2t_r20','t2d_r20','selection']]
            lines.append('| '+NAMES[variant]+' | '+' | '.join(f'{x:.4f}' for x in vals)+' | '+selected+' |')
        return '\n'.join(lines)

    def test_table(scope,models):
        frame=summary[summary.scope.eq(scope)].set_index(['model','direction'])
        lines=['| 方法 | 老药→靶点 AP | R@20 | 靶点→老药 AP | R@20 |','|---|---:|---:|---:|---:|']
        for name in dict.fromkeys(models):
            vals=[frame.loc[(name,d),m] for d in ['d2t','t2d'] for m in ['ap','r20']]
            lines.append('| '+NAMES[name]+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
        return '\n'.join(lines)

    def observed_table():
        frame=observed.set_index(['model','direction'])
        lines=['| 方法 | 正向实测 AP | 逆向实测 AP |','|---|---:|---:|']
        for name in ['frozen_v3',winner,'positive_nearest','pn_logistic']:
            lines.append(f"| {NAMES[name]} | {frame.loc[(name,'d2t'),'macro_ap']:.4f} | {frame.loc[(name,'t2d'),'macro_ap']:.4f} |")
        return '\n'.join(lines)

    contrast_table=['| 增量对照 | 正向 ΔAP | 95% 区间 | 正向增益种子 | 逆向 ΔAP | 95% 区间 | 逆向增益种子 |',
                    '|---|---:|---|---:|---:|---|---:|']
    for name,part in contrasts[contrasts.scope.eq('historical_test_54')].groupby('contrast',sort=False):
        d,t=part.set_index('direction').loc['d2t'],part.set_index('direction').loc['t2d']
        contrast_table.append(f"| {NAMES[d['added']].split()[0]} − {NAMES[d['control']].split()[0]} | {d.delta_ap:+.4f} | [{d.ci_low:+.4f}, {d.ci_high:+.4f}] | {int(d.positive_seeds)}/3 | {t.delta_ap:+.4f} | [{t.ci_low:+.4f}, {t.ci_high:+.4f}] | {int(t.positive_seeds)}/3 |")
    intervals=[]
    for row in primary:
        if row['scope']=='historical_test_54':
            intervals.append(f"{'正向' if row['direction']=='d2t' else '逆向'} ΔAP={row['delta_ap']:+.4f}，97.5% 配对区间 [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
    meanwinner=means.loc[winner]
    milliseconds=int(validation.seconds.sum())
    models=['frozen_v3',*config2['variants'], 'positive_nearest','pn_logistic']
    report=f'''# 旧 V3 增量升级与严谨消融：训练和测试结果

2026-09-06，状态：**33 组训练及冻结后的双向测试已完成**。保留旧 V3，首轮 6 配置×3 种子，第二轮 5 配置×3 种子。验证集选中的候选为 **{NAMES[winner]}**。候选分数和 Top 20 已生成，生产默认模型尚未替换。

**本轮找到的有效部分**：训练老药的已知关系检索监督、直接正负支持证据，以及在这套训练方案下接入的预训练交互。最终候选在历史未见老药上，正向 AP 从 0.4193 提高到 0.6274，逆向从 0.4854 提高到 0.6201；二元关系表缺席老药上分别从 0.3101/0.3178 提高到 0.5219/0.4609。近邻蒸馏没有稳定双向增益，不纳入最终候选。

仍未全面超过近邻：54 老药测试的正向 AP，近邻为 0.6617，候选为 0.6274；候选正向 R@20 为 0.8011，高于近邻的 0.7549。逆向 AP 候选较高，但 R@20 仍低于近邻。不能只挑有利指标宣布全面胜出。

两轮合计适配器训练与逐轮验证耗时约 {milliseconds} 秒，未计入首次缓存准备与最终评分。这里训练的是固定旧 V3 上的小型增量头，不能写成 33 次预训练大模型全量微调。

## 1. 固定什么，改变什么

底座固定为旧 B_support_pair 的两个 checkpoint（20260905、20260906），原始 logit 取均值，两个 192 维查询隐状态按列拼接。缓存重新计算的 720×384 底座分数与上一轮逐项一致，最大绝对差为 0。

新输出为“旧分数＋有界双向修正”。残差头 384 维隐状态输入、96 维中间层、两个方向输出；全零输出层使 epoch 0 精确保留旧模型。正证据加分、负证据减分，缺失支持贡献为零；独立预训练分支接入冻结 BerMol 与完整 ESM2 均值，再学习分子/蛋白投影及乘积交互。公共编码器和旧 V3 权重均保持冻结。本轮没有残基/原子局部交互、三维接触建模或公共编码器微调。

近邻蒸馏只使用训练老药和原训练支持库，将高相似度且有相似度差距的候选对作为软排序先验；同一化学实体从自身支持中排除，实测正负顺序与教师相反的配对被过滤。它不是新增实验标签。

## 2. 训练、验证、测试边界

- 实测训练：原 V3 的 **342,031 条**训练关系，每轮全覆盖。每组采用相同种子、同阶段同批次/查询安排；曝光哈希已核对。
- 老药验证：原实体划分中的 **52 老药**，33 个有已知靶点的正向 query、54 个有已知老药的逆向靶点 query；正向候选 **384 靶点**，逆向候选 **52 老药**。
- 实测验证守门：43,331 条原验证关系，807 个正向、379 个逆向双类别 query。两个方向的实测 AP 和老药已知关系 AP，各自都不得比旧 V3 低超过 0.01。
- 选模分数：`0.35 × (正向AP + 逆向AP) + 0.15 × (正向R@20 + 逆向R@20)`。每个种子可以选择未更新的 epoch 0；配置由三种子验证分数均值决定，不挑种子。
- 第一轮全部完成后，仅依据验证收益很小这一结果，另立第二轮协议；第一轮协议与结果保留。第二轮全部完成、两个阶段的候选合并选定后，才写入测试解封标记。
- 54 个历史 test 老药和 223 个二元关系表缺席老药用于冻结后的回归检查。这些对象在前几轮已经被查看，**仍是回顾性开发证据，不是全新的外部盲测**。

## 3. 第一轮：只用生化实测监督，逐项消融

同样的 8 个 epoch、学习率 0.0003、三个适配器种子（20260908/09/10）。A–D 构成“直接证据×双向实测排序损失”的 2×2 消融；E、F 分别对 D 增加预训练交互或近邻蒸馏。

下表为**三个独立选定 checkpoint 的指标均值**，不是三模型融合后计算的 AP。

{val_table(1)}

这一轮多数种子回退到原模型，增益幅度很小。保留零增量候选防止为了交付“新模型”而强行采用退化 checkpoint；这并不等于模块已被证明有效。

## 4. 第二轮：直接学习训练老药的已知关系排序

训练老药 **391 个**中的 **471 条**冻结已知关系进入额外检索监督，覆盖 **240 个药物 query、{config2['known_training_target_queries']} 个靶点 query**。验证和测试药物的已知关系没有进入训练。

已知药理关系与实测活性标签分开计算损失。多正例 softmax 同时鼓励该 query 的所有已知关系靠前；未知关系参与检索分母和相对排名约束，**没有被新增为生化 BCE 的实验阴性**。两套标签的潜在冲突保留，联合训练的取舍由独立验证守门约束。

第二轮固定 24 epoch、学习率 0.001；G 是相同预算、相同候选前向与优化器调用安排的“关闭已知关系损失”对照，H–K 逐项加入已知关系监督、证据、预训练和蒸馏。因此应在第二轮内部比较 H−G、I−H、J−I、K−I，不能将跨阶段差异全部归因于某个模块。

{val_table(2)}

最终候选 **{NAMES[winner]}** 的验证综合分均值为 **{meanwinner['selection']:.4f}**，底座为 **{baseline['selection']:.4f}**。这个选择在下面测试结果计算之前已冻结；测试优劣不会更换本轮胜出配置。

## 5. 冻结后测试：历史未见老药 54×384

正向 34 个有已知关系的老药 query、71 条关系；逆向 57 个靶点 query，候选仅为 54 老药。以下为三种子**平均残差加回同一底座之后重新排序**的结果，等价于平均原始 logit，并保持全零残差时底座数值不变。

{test_table('historical_test_54',models)}

验证选中候选相对底座：{'；'.join(intervals)}。区间使用 10,000 次 query 配对重采样，每个方向用 97.5% 边际区间处理该范围内两个方向的比较；不代表已消除多轮开发和多模块探索的全部选择偏差。

## 6. 关系表缺席老药 223×384

正向 91 个已知 query、188 条关系；逆向 72 个已知靶点 query，候选 223 老药。这些药物不在 V3 二元训练/验证/测试关系表中，但不保证公共预训练完全未见。

{test_table('absent_old_drugs_223',models)}

## 7. 完整项目 720×384：已知关系恢复

807 条已知关系，398 个正向、185 个逆向有标签 query。第二轮明确训练了其中训练老药的 471 条已知关系，因此本表主要用于部署恢复与回归，不能作为未见关系泛化成绩。

{test_table('project_720x384',['frozen_v3',winner,'positive_nearest','pn_logistic'])}

## 8. 实测活性没有被已知关系检索取代

在原 V3 历史 test 的 42,032 条实测关系上，按每个 query 实际测量候选计算：772 个正向和 351 个逆向双类别 query。这里包含一般化合物，不是完整老药面板，也不是全候选密集活性真值。

{observed_table()}

## 9. 哪些模块有独立增益

以下比较同阶段的匹配配置，展示历史 54 老药测试上每个 query 的 AP 差值，先在三种子间平均，再做 query 配对重采样；另列逐种子增益次数。全部对照均为验证选定 checkpoint，包含 epoch 0 回退。这些区间属于探索性分析，未对所有模块和范围做多重比较校正。

{chr(10).join(contrast_table)}

保留模块应依据它相对匹配对照的增量、跨种子一致性、未见老药表现及实测 AP 代价。验证领先而测试不稳定的模块仅作为候选；不能因为总模型得分提高，就把其中每个模块都称为有效。若近邻仍领先，必须继续把它作为固定对照。

当前取舍：**保留 H 的已知关系监督与 I 的正负证据；J 的预训练交互保留为最终候选增量；K 的蒸馏不加入。** H−G 和 I−H 在历史老药测试的两个方向都为三种子正增益，探索性 95% 区间高于零。J−I 也是两个方向三种子正增益，但正向区间为约 [−0.0041, +0.0980]，仍跨零，不能单独宣称其正向增益已获统计确认。K−I 的正向平均增量为负且仅 1/3 种子为正。

下一步优先将 J 固定为新候选，与旧 V3、近邻和 PN 一起接受新的冻结外部或同条件活性面板检验；重点分析近邻正向 AP 仍占优的老药及候选的逆向召回损失，再决定局部交互和公共编码器微调的独立实验。已经查看的 54/223 子集继续作回归检查，不继续用于调参后再称为盲测。

## 10. 产物与复现

- [逐种子验证与选中 epoch](../outputs/biomaster_v3_incremental_20260906/evaluation/VALIDATION_ABLATION.csv)
- [全部方法、范围、方向指标](../outputs/biomaster_v3_incremental_20260906/evaluation/SUMMARY.csv)
- [匹配消融差值与区间](../outputs/biomaster_v3_incremental_20260906/evaluation/PAIRED_CONTRASTS.csv)
- [候选相对底座的配对区间](../outputs/biomaster_v3_incremental_20260906/evaluation/WINNER_VS_FROZEN_V3.json)
- [老药→靶点 Top 20](../outputs/biomaster_v3_incremental_20260906/evaluation/TOP20_DRUG_TO_TARGET.csv.gz)
- [靶点→老药 Top 20](../outputs/biomaster_v3_incremental_20260906/evaluation/TOP20_TARGET_TO_DRUG.csv.gz)
- [第一轮协议](../outputs/biomaster_v3_incremental_20260906/PROTOCOL.json)、[第二轮协议](../outputs/biomaster_v3_incremental_20260906/old_relation_stage/PROTOCOL.json)、[冻结选择](../outputs/biomaster_v3_incremental_20260906/old_relation_stage/SELECTION.json)、[验证记录](../outputs/biomaster_v3_incremental_20260906/VALIDATION.json)
- [候选模型与推理产物清单](../outputs/biomaster_v3_incremental_20260906/CANDIDATE.json)、[32 项相关测试记录](../outputs/biomaster_v3_incremental_20260906/VALIDATION_TESTS.json)

```bash
python scripts/prepare_biomaster_v3_incremental_ablation.py
python scripts/train_biomaster_v3_incremental_ablation.py
python scripts/train_biomaster_v3_old_relation_ablation.py
python scripts/evaluate_biomaster_v3_incremental_ablation.py
python scripts/summarize_biomaster_v3_incremental_ablation.py
```

准备与训练阶段对已有冻结产物设有防覆盖/身份核对；当前已完成时只需复核评估与汇总。实际文件哈希记录在 FEATURES、SELECTION、TEST_RELEASE 和 RESULT_MANIFEST 中。适配器位于各 `runs/<variant>/seed_<seed>/BEST.pt`；推理还依赖固定 V3 底座和对应特征，不能将适配器单独当成完整模型。
'''
    REPORT.write_text(report)
    validation_record={'status':'PASS','completed_training_runs':len(rows),'score_matrices_verified':len(score_manifest['scores_sha256']),
        'checkpoint_hashes_verified':len(checkpoints),'all_matched_exposures_identical_within_stage_and_seed':True,
        'selected_checkpoints_pass_all_four_validation_AP_guards':True,'old_v3_matches_previous_720x384_metrics':True,
        'teacher_pairs_audited':teacher_count,'teacher_drugs_all_original_train':True,
        'winner_frozen_before_test':winner,'test_used_for_selection':False,'production_promotion':False,
        'selected_epoch_zero_runs':int(validation.selected_epoch.eq(0).sum()),'summary_rows':len(summary),
        'report_code_sha256':sha(Path(__file__))}
    write(OUT/'VALIDATION.json',validation_record)
    chosen=[r for r in second['results']+first['results'] if r['variant']==winner]
    chosen_root=STAGE if winner in config2['variants'] else OUT
    feature_sources=json.loads((OUT/'FEATURES.json').read_text())['sources']
    candidate={'status':'VALIDATION_SELECTED_AND_RETROSPECTIVELY_EVALUATED','name':winner,
        'base_checkpoints':{p:h for p,h in feature_sources.items() if p.endswith('BEST_MODEL_V3.pt')},
        'adapter_checkpoints':[{'path':str((chosen_root/'runs'/winner/f"seed_{r['seed']}"/'BEST.pt').relative_to(ROOT)),
            'sha256':r['checkpoint_sha256'],'seed':r['seed'],'epoch':r['selected_epoch']} for r in chosen],
        'combination':'mean frozen V3 logits + mean of 3 selected adapter residuals',
        'current_scores':str((OUT/'scores'/f'{winner}.npz').relative_to(ROOT)),
        'scores_sha256':score_manifest['scores_sha256'][winner],
        'old_score_array':'old','old_score_shape':[720,384,2],
        'direction_axis':['drug_to_target','target_to_drug'],
        'drug_axis':'outputs/biomaster_old_drug_bidirectional_20260906/OLD_DRUGS_720.csv',
        'target_axis':'outputs/biomaster_old_drug_bidirectional_20260906/TARGETS_384.csv.gz',
        'score_semantics':'ranking logits, not calibrated probabilities',
        'known_training_pairs':471,'test_claim':'retrospective entity holdout, not new external confirmation',
        'selection_sha256':sha(STAGE/'SELECTION.json'),'production_default_replaced':False}
    write(OUT/'CANDIDATE.json',candidate)
    files=[REPORT,OUT/'VALIDATION.json',OUT/'PROTOCOL.json',OUT/'FEATURES.json',OUT/'SELECTION.json',
        OUT/'CANDIDATE.json',
        STAGE/'PROTOCOL.json',STAGE/'SELECTION.json',OUT/'TEST_RELEASE.json',OUT/'STATUS.json',
        *EV.glob('*.csv'),*EV.glob('*.json'),OUT/'scores/MANIFEST.json',
        ROOT/'biomaster/odti_v3_incremental.py',ROOT/'biomaster/old_relation_retrieval.py',
        *[ROOT/'scripts'/name for name in ['prepare_biomaster_v3_incremental_ablation.py',
            'train_biomaster_v3_incremental_ablation.py','train_biomaster_v3_old_relation_ablation.py',
            'evaluate_biomaster_v3_incremental_ablation.py','summarize_biomaster_v3_incremental_ablation.py']]]
    if (OUT/'VALIDATION_TESTS.json').exists():files.append(OUT/'VALIDATION_TESTS.json')
    write(OUT/'RESULT_MANIFEST.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
    print(json.dumps(validation_record,ensure_ascii=False));print(REPORT)


if __name__=='__main__':
    main()
