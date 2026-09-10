#!/usr/bin/env python3
"""Verify frozen artifacts and render the chronological/external audit report."""
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_biomaster_odti_v4_features import sha256
from train_biomaster_selectivity_v3 import write_json

OUT=ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906'
T=OUT/'temporal';K=OUT/'kirhub'
DOC=ROOT/'docs/BIOMASTER_V3_KIRHUB_TEMPORAL_20260906_ZH.md'
NAMES={'temporal_v3':'时间重训 V3','temporal_J':'时间重训 J','positive_nearest':'阳性近邻',
       'pn_logistic':'正负近邻 Logistic','random_ranking_expected':'随机排序期望',
       'frozen_v3':'原 V3','J_old_relation_pretrained':'当前升级 J',
       'production_independent':'现生产独立头','v6_directional_full_fit':'V6 full-fit',
       'dtiam':'本地重训 DTIAM','conplex':'本地 ConPLex'}


def number(v):
    return '—' if v is None or pd.isna(v) else f'{float(v):.4f}'


def markdown(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def metric_table(frame,scope,models):
    rows=[]
    subset=frame[frame.scope.eq(scope)]
    for model in models:
        select=subset[subset.model.eq(model)].set_index('direction')
        rows.append([NAMES[model]]+[number(select.loc[d,k]) for d,k in [('d2t','macro_ap'),('t2d','macro_ap'),
                                                                               ('d2t','macro_recall_20'),('t2d','macro_recall_20')]])
    return markdown(['模型','药→靶 AP','靶→药 AP','药→靶 R@20','靶→药 R@20'],rows)


def main():
    if not (OUT/'VALIDATION_TESTS.json').exists():
        names=['tests/test_temporal_evaluation.py','tests/test_temporal_relations.py','tests/test_ranking_audit.py',
               'tests/test_odti_v3_incremental.py','tests/test_old_relation_retrieval.py','tests/test_old_drug_ranking.py']
        command=[sys.executable,'-m','pytest','-q',*names]
        execution=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        output=execution.stdout+execution.stderr
        if execution.returncode:raise RuntimeError(output)
        match=re.search(r'(\d+) passed',output)
        assert match,output
        write_json(OUT/'VALIDATION_TESTS.json',{'status':'PASS','command':command,'passed':int(match.group(1)),
            'output':output,'test_source_sha256':{name:sha256(ROOT/name) for name in names}})
    tests=json.loads((OUT/'VALIDATION_TESTS.json').read_text());assert tests['status']=='PASS'
    for path,digest in tests['test_source_sha256'].items():assert sha256(ROOT/path)==digest
    result=json.loads((T/'TEST_RESULT.json').read_text());assert result['status']=='COMPLETE'
    selection=json.loads((T/'SELECTION.json').read_text());release=json.loads((T/'TEST_RELEASE.json').read_text())
    manifest=json.loads((T/'DATA_MANIFEST.json').read_text());kir_protocol=json.loads((K/'PROTOCOL.json').read_text())
    assert not selection['test_labels_read'] and not result['test_used_for_selection']
    assert result['metrics_sha256']==sha256(T/'TEST_METRICS.csv')
    for path,digest in release['identity']['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    for path,digest in release['checkpoints'].items():assert sha256(ROOT/path)==digest,path
    for path,digest in kir_protocol['source_sha256'].items():assert sha256(ROOT/path)==digest,path
    for stage,cutoff in [('development',2020),('final',2022)]:
        for r in (T/stage).glob('*/seed_*/RESULT.json'):
            data=json.loads(r.read_text());assert data['train_max_year']==cutoff and not data['test_used']
            if stage=='final':
                key='base_epochs' if r.parent.parent.name=='base' else 'adapter_epochs'
                assert data['selected_epoch']==selection[key][str(data['seed'])]
    metrics=pd.read_csv(T/'TEST_METRICS.csv');kir=pd.read_csv(K/'METRICS.csv')
    comparisons=json.loads((T/'TEST_PAIRED_COMPARISONS.json').read_text())
    kir_ci=json.loads((K/'PAIRED_COMPARISONS.json').read_text())
    approval=json.loads((T/'REGISTRY_APPROVAL_PROTOCOL.json').read_text())
    old=pd.read_csv(T/'OLD_DRUG_INDEX.csv');targets=pd.read_csv(T/'TARGET_INDEX.csv.gz')
    positive_ranks=pd.read_csv(T/'TEST_POSITIVE_RANKS.csv.gz')
    positive_ranks['ligand_inchikey']=np.where(positive_ranks.direction.eq('d2t'),positive_ranks.query_id,positive_ranks.candidate_id)
    positive_ranks['target_chembl_id']=np.where(positive_ranks.direction.eq('d2t'),positive_ranks.candidate_id,positive_ranks.query_id)
    positive_ranks['drug_names']=positive_ranks.ligand_inchikey.map(old.set_index('ligand_inchikey').drug_names)
    positive_ranks['gene_symbol']=positive_ranks.target_chembl_id.map(targets.set_index('target_chembl_id').gene_symbol)
    positive_ranks.to_csv(T/'TEST_POSITIVE_RANKS_NAMED.csv.gz',index=False)
    counts=manifest['counts'];new=counts['test_2023_2025_new']
    models=['temporal_v3','temporal_J','positive_nearest','pn_logistic','random_ranking_expected']
    main_scope='test_2023_2025_new_dense_old720';approved_scope='test_2023_2025_new_dense_approved_pre2023'
    main=metrics[metrics.scope.eq(main_scope)&metrics.model.eq('temporal_J')].set_index('direction')
    conservative=metrics[metrics.scope.eq(approved_scope)&metrics.model.eq('temporal_J')].set_index('direction')
    def delta(scope,control,direction):
        row=next(r for r in comparisons if r['scope']==scope and r['control']==control and r['direction']==direction)
        return f"{row['delta_ap']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
    ci_table=markdown(['集合','比较','药→靶 ΔAP [95% CI]','靶→药 ΔAP [95% CI]'],[
        [label,f'J − {NAMES[control]}',delta(scope,control,'d2t'),delta(scope,control,'t2d')]
        for scope,label in [(main_scope,'720 目录首次关系'),(approved_scope,'594 早期获批药首次关系')]
        for control in ['temporal_v3','positive_nearest','pn_logistic']])
    year_rows=[]
    for year in [2023,2024,2025]:
        s=metrics[metrics.scope.eq(f'test_{year}_new_dense_old720')]
        for model in ['temporal_v3','temporal_J','positive_nearest']:
            r=s[s.model.eq(model)].set_index('direction')
            year_rows.append([year,NAMES[model],int(r.loc['d2t','positive_pairs']),
                              int(r.loc['d2t','positive_queries']),int(r.loc['t2d','positive_queries']),
                              number(r.loc['d2t','macro_ap']),number(r.loc['t2d','macro_ap'])])
    years=markdown(['首次年份','模型','阳性关系','药查询','靶查询','药→靶 AP','靶→药 AP'],year_rows)
    scope_rows=[]
    for scope in ['test_2023_2025_new_measured_all_compounds','test_2023_2025_new_measured_old720','test_2023_2025_new_measured_approved_pre2023']:
        s=metrics[metrics.scope.eq(scope)&metrics.model.eq('temporal_J')].set_index('direction')
        scope_rows.append([scope.replace('test_2023_2025_new_measured_',''),int(s.loc['d2t','rows']),int(s.loc['d2t','positive_rows']),
                           int(s.loc['d2t','two_class_queries']),int(s.loc['t2d','two_class_queries']),
                           number(s.loc['d2t','median_candidates']),number(s.loc['t2d','median_candidates'])])
    coverage=markdown(['实测集合','关系','阳性','双类药查询','双类靶查询','药方向候选中位数','靶方向候选中位数'],scope_rows)
    directions=[]
    for d in ['d2t','t2d']:
        r=next(r for r in comparisons if r['scope']==main_scope and r['control']=='temporal_v3' and r['direction']==d)
        word='点估计提高' if r['delta_ap']>0 else '点估计下降'
        certainty='区间不跨 0' if r['ci_low']>0 or r['ci_high']<0 else '区间跨 0'
        directions.append(f"{'药→靶' if d=='d2t' else '靶→药'} {word} {abs(r['delta_ap']):.4f}，{certainty}")
    body=f'''# V3 升级版：KIRHub 与 2023–2025 时间测试（2026-09-06）

两项实测已完成。当前全年份 J 在严格 KIRHub 上为 **0.3816 / 0.4021 AP**（药→靶 / 靶→药），原 V3 为 **0.3380 / 0.3407**，本地重训 DTIAM 为 **0.3867 / 0.3041**。这支持逆向排序的提升，不能声称正向已经超过 DTIAM。

时间测试另行从头训练，监督和支持库仅使用 **≤2022 年**测量。2023–2025 首次关系中，项目老药全候选检索的 J AP 为 **{main.loc['d2t','macro_ap']:.4f} / {main.loc['t2d','macro_ap']:.4f}**；相对相同时间训练的 V3：{'；'.join(directions)}。这组分数和 KIRHub、历史807关系恢复的 AP 分母不同。

## 1. KIRHub：冻结当前模型的功能抑制测试

保持上一轮选定的 `J_old_relation_pretrained`、两底座和三 adapter 分数不变；没有拟合 KIRHub 功能适配器。使用本地冻结的 KIRHub WT 配对表，**1 μM 下残余活性 ≤30%**算阳性，即功能抑制 ≥70%，并非 Ki/Kd 标签。

历史严格切片有 **2,823 配对、202 阳性、78 药、103 靶点**。药方向只有 **33 个双类查询**，靶方向 **63 个双类查询**。逐对核验：当前 V3 二元训练与 J 已知关系训练接触均为 **0**。候选只包含对应查询实际测过的条目，R@20 不是完整 384 靶点召回。

{metric_table(kir,'historical_strict_2823',['frozen_v3','J_old_relation_pretrained','production_independent','dtiam','positive_nearest','pn_logistic'])}

逐查询配对 bootstrap（10,000 次）中，J−V3 的药→靶 ΔAP 为 **+0.0435，95% CI [−0.0017, +0.1103]**；靶→药为 **+0.0614 [+0.0153, +0.1126]**。J−DTIAM 为 **−0.0052 [−0.1212, +0.1088] / +0.0980 [+0.0328, +0.1671]**。区间未做多重比较校正；靶点之间存在家族相关性。项目曾查看过 KIRHub，因此是回顾性外部审计，不能称全新盲测或公开基准 SOTA。

补充切片：**单一 WT construct 且当前 J/V3 未训练过的配对**有 **4,844 条、501 阳性、79 药、96 靶点**，药/靶双类查询数为 **63/73**。这不是对所有比较方法共同排除训练接触的集合，DTIAM和历史生产的对照仅为相同候选评分。

{metric_table(kir,'single_construct_unseen_by_J',['frozen_v3','J_old_relation_pretrained','dtiam','positive_nearest'])}

完整映射 8,058 对中有 2,886 个 V3 二元训练配对、166 个 J 已知训练配对；单 construct 7,505 对也有训练接触。它们的高 AP 仅列入原始指标表，不能用来证明新关系泛化。

## 2. 时间测试到底怎样划分

数据来自本地 ChEMBL 37 SQLite，限定当前384核心靶点的人源 SINGLE PROTEIN、binding assay、confidence ≥9、direct、无变体；保留等号 Ki/Kd/IC50 pChEMBL 和明确 inactive 测量，排除重复和不合格记录。共提取 **549,356** 条原始活动记录；其中 **11,753** 条缺文献年份，另有 **{manifest['unresolved_structure_activity_rows']}** 条无法可靠标准化结构。缺年份不进入训练/测试；同一关系有缺年份记录时，也不计入首次出现关系。

关键顺序是：化学结构标准化并合并别名 → **按原始文献年份筛选** → 同一期间聚合均值与冲突 → 生成标签。pChEMBL 均值 ≥6 为正，≤5 或明确 inactive 为负；灰区和正负冲突排除。未来高活性测量不会改变早期阴性标签。首次关系要求此前连合格的灰区或阴性测量都没有；仅排除“早期阳性”不够严格。

这里的“首次”限定于**本地ChEMBL37中满足上述规则的测量记录**，不是该药靶关系在人类知识中的首次发现。更早的功能、突变体、定性机制或其他数据库记录可能已记载该关系。例如已知药理靶点在2023年后才出现本次规则接受的WT测量，也会进入这一集合；本报告不把它称为全新生物学发现。

{markdown(['阶段','测量年份','明确关系','阳性关系','项目老药关系 / 阳性'],[
    ['开发训练','≤2020',counts['development_train']['rows'],counts['development_train']['positives'],f"{counts['development_train']['old_drug_rows']} / {counts['development_train']['old_drug_positives']}"],
    ['开发验证（首次）','2021–2022',counts['development_validation_new']['rows'],counts['development_validation_new']['positives'],f"{counts['development_validation_new']['old_drug_rows']} / {counts['development_validation_new']['old_drug_positives']}"],
    ['最终从头训练','≤2022',counts['final_train']['rows'],counts['final_train']['positives'],f"{counts['final_train']['old_drug_rows']} / {counts['final_train']['old_drug_positives']}"],
    ['主测试（首次）','2023–2025',new['rows'],new['positives'],f"{new['old_drug_rows']} / {new['old_drug_positives']}"],
    ['补充测试（含早期已见）','2023–2025',counts['test_2023_2025_all']['rows'],counts['test_2023_2025_all']['positives'],f"{counts['test_2023_2025_all']['old_drug_rows']} / {counts['test_2023_2025_all']['old_drug_positives']}"],
])}

“2023年前”明确为截至2022年。年份使用文献发表年，不是数据库入库日期或实验执行日期；ChEMBL 37 对2025年的收录也可能不完整。候选靶点和公共预训练特征使用当前资产，所以这是**监督标签/支持证据的时间隔离**，不是对2022年可获得全部资源的历史重演。

## 3. 真正重新训练了什么

底座为同拓扑 V3 B：Morgan、ProtBERT/legacy ESM2 输入、低秩 FiLM 交互、按靶点检索的正负支持编码；训练支持自身化学实体排除。开发与最终训练各建独立的 k=16 支持库，并对所有缓存条目核对来源、标签、靶点和自身实体排除。

J 保留两个底座的冻结均值 logit 与384维拼接隐状态，加初始为零的有界双向残差、有符号正负相似证据，以及冻结 BerMol/完整序列 ESM2 的交互分支。底座两种子、adapter三种子；**没有加载此前全年份 V3/J 权重**。时间 J 的检索监督用截止期内项目老药的实验阳性（最终815对），没有使用缺少时间归属的807条药理关系表。因此它是同拓扑的时间安全重训版，不能与原J当作同一个checkpoint。

开发训练每个底座最多6轮，只用2021–2022首次关系的双向实测AP均值选轮数；最终选择 **{list(selection['base_epochs'].values())}**。固定J拓扑每种子最多24轮，用早期验证的双向全候选AP和R@20选择，并要求两方向实测AP各不低于底座0.01；epoch0可回退。最终选择 **{list(selection['adapter_epochs'].values())}**。随后所有权重重新初始化，在≤2022数据训练固定轮数；学习率调度仍用原6/24轮分母，未按最终短轮数压缩。

2023–2025 标签只在模型、epoch选择、最终分数和清单冻结后，由单独测试脚本读取。阳性近邻和正负Logistic也只使用对应≤2022支持库，Logistic只拟合≤2022标签。未用全年份DTIAM冒充同时间训练对照。

## 4. 老药首次关系：完整候选双向检索

固定当前 **720 药×384 靶点**，从每个查询的候选中移除截至2022年已测关系及年份不明关系；其余未知配对作为检索背景，不是生化阴性。只评价有未来阳性的查询。71个新阳性分布在 **{int(main.loc['d2t','positive_queries'])} 药查询 / {int(main.loc['t2d','positive_queries'])} 靶查询**；可选候选数中位数为 **{int(main.loc['d2t','median_candidates'])} 靶 / {int(main.loc['t2d','median_candidates'])} 药**。

{metric_table(metrics,main_scope,models)}

AP越高表示已确认的新阳性越集中在前列。未知配对中仍可能存在未发表的真阳性，因此这是不完整标注的未来关系检索；不能据此把低分未知对判定无活性。随机排序期望使用各查询真实候选/阳性数量的解析值，不用全相同分数的AP代替随机排名。

项目720清单并非全部在2022年前已上市：精确InChIKey核对ChEMBL首次获批年，**594个≤2022、48个>2022、78个缺年**。预先冻结的保守子集只留下594个可确认早期获批药，药方向查询与逆向候选均相应限制。该子集有 **{int(conservative.loc['d2t','positive_pairs'])}** 个新阳性，分布在 **{int(conservative.loc['d2t','positive_queries'])}/{int(conservative.loc['t2d','positive_queries'])}** 个药/靶查询。

{metric_table(metrics,approved_scope,models)}

这个子集更接近用户要求的老药新用，但first_approval是ChEMBL的历史记录，不等于核验每个国家当年的适应证和上市状态。缺年药被保守排除。

{ci_table}

配对bootstrap以药或靶query为抽样单位，10,000次，不把同一查询下的配对当作相互独立样本。小查询数及靶点家族相关性限制精度；查看多个年份/切片产生多重比较问题，显著性不作独立确认。

## 5. 分年检查

以下都使用同一批≤2022权重，不滚动更新。每年只取该年首次出现关系；跨年合并时会重新判断期间冲突，所以分年数不必完全加总到合并期。

{years}

2025年的项目老药仅9个新阳性，不能把单年波动解释成稳定优劣或真实部署趋势。

## 6. 实测正负排序：另一个分母

{coverage}

一般化合物的明确测量候选：

{metric_table(metrics,'test_2023_2025_new_measured_all_compounds',models)}

项目720药的明确测量候选：

{metric_table(metrics,'test_2023_2025_new_measured_old720',models)}

只保留早期获批药的明确测量候选：

{metric_table(metrics,'test_2023_2025_new_measured_approved_pre2023',models)}

实测AP只在同时含正负的查询上计算。少量候选和高阳性占比会抬高随机AP、让R@20轻易达到1；不能把这组高AP换算成720×384全空间筛选能力。完整逐查询记录包含候选数，可复核这一差异。

## 7. 这轮能支持的下一步

KIRHub显示双向增益不对称：当前J的逆向功能排序提升较明确，正向还没有可靠超过DTIAM或近邻。时间测试中J相对V3的双向AP提升在720目录和594早期获批药集合均有正的配对区间，说明增量路线有实际价值。但相对阳性近邻的AP差异区间均跨0；720目录的药→靶R@20为0.5444，低于近邻0.6833，不能宣称架构已经全面胜出。

失败个案显示J仍会压低近邻排在前面的未来阳性：在594药子集的药→靶排序中，entrectinib–ACVRL1为近邻第16/J第279，donepezil–PDE5A为第11/第199；另一方面，abemaciclib–CLK2从近邻第118提升到J第2。这些只是本次规则下的实验记录，不是临床用药判断。前两项底座logit分别约−4.71、−4.26，J进一步降到−7.61、−6.97，表明“加入预训练交互”本身不能保证纠正错误底座和证据融合。由这些个案推断，下一轮应重点消融可学习底座权重、正负证据冲突处理和对Top-K漏检的训练约束，而不是只扩大残差网络。

优先保留在两类测试中有增益的方向，并检查`TEST_POSITIVE_RANKS_NAMED.csv.gz`中的跌落关系：区分缺近邻覆盖、功能与结合标签不一致、靶点家族泛化和直接记忆训练老药关系的问题。后续架构与损失改动应在更早的多个滚动时间窗口做消融，2023–2025集合现已打开，不再作为未触碰的调参验证集。局部交互或更大预训练模型是否值得加入，需要同时改善未来老药全候选排序和功能集合；不以稀疏实测AP单独选模型。

## 8. 产物、复现与验证

- [KIRHub完整指标](../outputs/biomaster_v3_kirhub_temporal_20260906/kirhub/METRICS.csv)、[协议与训练接触](../outputs/biomaster_v3_kirhub_temporal_20260906/kirhub/PROTOCOL.json)、[配对区间](../outputs/biomaster_v3_kirhub_temporal_20260906/kirhub/PAIRED_COMPARISONS.json)。
- [时间数据清单](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/DATA_MANIFEST.json)、[冻结选择](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/SELECTION.json)、[最终权重与测试解封清单](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/TEST_RELEASE.json)。
- [时间全部指标](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/TEST_METRICS.csv)、[配对区间](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/TEST_PAIRED_COMPARISONS.json)、[获批年份审计](../outputs/biomaster_v3_kirhub_temporal_20260906/temporal/REGISTRY_APPROVAL_PROTOCOL.json)。
- 本地大文件：`temporal/final/base/seed_*/BEST.pt`、`temporal/final/adapter/seed_*/BEST.pt`、`temporal/final/scores/*_SCORES.npz`；`TEST_POSITIVE_RANKS_NAMED.csv.gz`是带药名/基因名的每个新阳性名次，`TEST_MEASURED_PREDICTIONS.csv.gz`包含全部实测测试预测。大权重/特征/逐对分数留本地，不纳入Git。

从空输出目录复现依次执行：

```bash
python scripts/evaluate_v3_upgrade_kirhub_20260906.py
python scripts/extract_v3_temporal_raw_20260906.py
python scripts/prepare_v3_temporal_20260906.py
python scripts/audit_v3_temporal_registry_age_20260906.py
python scripts/train_v3_temporal_20260906.py
python scripts/evaluate_v3_temporal_20260906.py
python scripts/summarize_v3_kirhub_temporal_20260906.py
```

已完成的训练/数据目录有防覆盖检查；重评估只需后三个入口中的评估、汇总两个脚本，不要覆盖或清空现有证据。自动验证覆盖未来测量污染、灰区/缺年份/别名的首次关系排除、双向候选分母、历史风险掩码、随机AP解析值和既有残差/排序损失。训练与评分文件均绑定数据、配置、源码与checkpoint哈希。生产默认未修改。
'''
    DOC.write_text(body)
    validation={'status':'PASS','frozen_training_source_hashes_verified':True,'checkpoint_hashes_verified':True,
                'kirhub_source_hashes_verified':True,'final_epochs_match_early_selection':True,
                'all_cached_supports_train_only_and_self_excluded':True,'period_first_seen_disjointness_asserted':True,
                'training_runs':10,'tests_passed':tests['passed'],'test_evidence_sha256':sha256(OUT/'VALIDATION_TESTS.json'),
                'report_sha256':sha256(DOC)}
    write_json(OUT/'VALIDATION.json',validation)
    files=[K/'PROTOCOL.json',K/'METRICS.csv',K/'PAIRED_COMPARISONS.json',T/'DATA_MANIFEST.json',T/'SELECTION.json',
           T/'TEST_RELEASE.json',T/'TEST_RESULT.json',T/'TEST_METRICS.csv',T/'TEST_PAIRED_COMPARISONS.json',
           T/'TEST_POSITIVE_RANKS_NAMED.csv.gz',OUT/'VALIDATION.json',OUT/'VALIDATION_TESTS.json',DOC]
    write_json(OUT/'RESULT_MANIFEST.json',{'status':'COMPLETE','files':{str(f.relative_to(ROOT)):sha256(f) for f in files},
               'report_source_sha256':sha256(Path(__file__)),'production_changed':False})
    print(str(DOC),flush=True)


if __name__=='__main__':main()
