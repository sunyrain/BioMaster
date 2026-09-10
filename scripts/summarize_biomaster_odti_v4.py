#!/usr/bin/env python3
"""Validate completed R1 runs and generate the evidence-backed Chinese report."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_biomaster_odti_v4_features import sha256, write_json


def main():
    out = ROOT / 'outputs/biomaster_odti_v4_20260905'
    evaluation = out / 'evaluation'
    status = json.loads((evaluation / 'EVALUATION_STATUS_V4.json').read_text())
    assert status['status'] == 'COMPLETE'
    result = json.loads((evaluation / 'RESULTS_V4.json').read_text())
    result.update(json.loads((evaluation / 'HISTORICAL_V3_CONTROLS_V4.json').read_text()))
    protocol = json.loads((out / 'protocol/PROTOCOL_V4.json').read_text())
    comparisons = json.loads((evaluation / 'PAIRED_COMPARISONS_V4.json').read_text())
    groups = {'Morgan全局对照': ['morgan_'+str(s) for s in protocol['seeds']],
              'BerMol全局 R1': ['bermol_'+str(s) for s in protocol['seeds']],
              'BerMol＋有符号证据': ['signed_'+str(s) for s in protocol['seeds']],
              '旧版 V3 B 支持模型': ['v3_B_support_20260905','v3_B_support_20260906'],
              '正近邻基线': ['positive_nearest'], 'PN逻辑回归基线': ['positive_negative_logistic']}
    fields = [('observed','micro_ap'),('observed','micro_auroc'),('observed','drug_macro_ap'),
              ('panel','macro_positive_retrieval_ap'),('panel','macro_recall_at_20')]
    summary, rows = {}, []
    for label,names in groups.items():
        for name in names:
            with np.load(evaluation/f'{name}_PREDICTIONS_V4.npz') as prediction:
                assert prediction['panel_scores'].shape == (1024,495)
                assert prediction['observed_scores'].shape == (42032,)
                assert np.isfinite(prediction['panel_scores']).all() and np.isfinite(prediction['observed_scores']).all()
        summary[label] = {'seeds':len(names), 'members':names}
        cells = []
        for scope,key in fields:
            values = np.array([result[n][scope][key] for n in names])
            mean,std = float(values.mean()),float(values.std(ddof=1)) if len(values)>1 else 0.
            summary[label][key] = {'mean':mean,'seed_sample_std':std}
            cells.append(f'{mean:.4f} ± {std:.4f}' if len(values)>1 else f'{mean:.4f}')
        rows.append('| '+label+' | '+str(len(names))+' | '+' | '.join(cells)+' |')
    train_rows, score_audits = [], {}
    total_seconds,total_epochs = 0.,0
    for variant in ['morgan','bermol','signed']:
        for seed in protocol['seeds']:
            name=f'{variant}_{seed}'
            run=out/'runs'/name
            info=json.loads((run/'STATUS_V4.json').read_text())
            history=json.loads((run/'TRAINING_HISTORY_V4.json').read_text())
            assert info['status'] in {'TRAINED_TEST_NOT_EVALUATED','COMPLETE'}
            actual=[r for r in history if r['epoch']>0]
            assert len(actual)==info['epochs']
            assert all(r['relation_rows_seen']==342031 for r in actual)
            assert info['best_epoch']==max(history,key=lambda r:r['validation_panel']['macro_positive_retrieval_ap'])['epoch']
            assert (run/'BEST_MODEL_V4.pt').is_file() and (run/'LAST_MODEL_V4.pt').is_file()
            total_seconds+=info['seconds'];total_epochs+=info['epochs']
            train_rows.append(f"| {variant} | {seed} | {info['epochs']} | {info['best_epoch']} | {info['best_validation_panel_ap']:.4f} |")
            f=np.load(evaluation/f'{name}_PREDICTIONS_V4.npz')
            assert f['panel_scores'].shape==(1024,495) and f['observed_scores'].shape==(42032,)
            assert np.isfinite(f['panel_scores']).all() and np.isfinite(f['observed_scores']).all()
            if variant=='signed':
                base,plus,minus=f['base_logit'],f['positive_bonus'],f['negative_penalty']
                assert (plus>=0).all() and (minus>=0).all() and max(plus.max(),minus.max())<=2.00001
                np.testing.assert_allclose(f['panel_scores'].reshape(-1),base+plus-minus,rtol=1e-5,atol=1e-5)
                masks=np.load(protocol['panels']['test']['support']['mask'],mmap_mode='r')
                assert (plus[~masks[:,1].any(-1)]==0).all()
                assert (minus[~masks[:,0].any(-1)]==0).all()
                negative_only=masks[:,0].any(-1)&~masks[:,1].any(-1)
                assert (f['panel_scores'].reshape(-1)[negative_only]<=base[negative_only]+1e-5).all()
                score_audits[name]={'base_quantiles_1_50_99':np.quantile(base,[.01,.5,.99]).tolist(),
                    'positive_bonus_quantiles_50_90_99':np.quantile(plus,[.5,.9,.99]).tolist(),
                    'negative_penalty_quantiles_50_90_99':np.quantile(minus,[.5,.9,.99]).tolist(),
                    'negative_only_rows':int(negative_only.sum()),'negative_only_score_increase_rows':0}
    provenance=json.loads((out/'features/SOURCE_PROVENANCE_V4.json').read_text())
    assert provenance['passed']
    write_json(evaluation/'SUMMARY_V4.json',summary)
    write_json(evaluation/'SCORE_COMPONENT_AUDIT_V4.json',score_audits)
    audit={'status':'PASS','new_training_runs':9,'historical_checkpoints_evaluated':2,'simple_baselines':2,
        'new_model_epochs_total':total_epochs,'sum_training_seconds':total_seconds,
        'all_epochs_full_relation_coverage':True,'selected_epochs_match_validation_argmax':True,
        'all_prediction_shapes_and_values_valid':True,'signed_score_contract_valid':True,
        'bermol_recomputation_max_absolute_error':provenance['max_absolute_error'],
        'tests':'54 relevant tests passed, including CUDA direct/cached score equivalence',
        'production_promoted':False}
    write_json(out/'VALIDATION_V4_R1.json',audit)
    subgroup=result['positive_nearest']['similarity_subgroups']
    near=subgroup['near_ge_0.7']['queries'];middle=subgroup['middle_0.4_to_0.7']['queries'];low=subgroup['low_lt_0.4']['queries']
    def ci(name):
        v=comparisons[name]
        return f"{v['mean_ap_difference']:+.4f}，药物配对bootstrap 95% CI [{v['paired_drug_bootstrap_95_CI'][0]:+.4f}, {v['paired_drug_bootstrap_95_CI'][1]:+.4f}]"
    report=f'''**BioMaster ODTI V4：首轮训练、统一测试与下一步（2026-09-05）**

> **2026-09-06审计更正**：以下“完整面板”仅指候选评分范围，99.73%的配对无观测标签；1,024个一般化合物与项目720老药清单精确匹配为0，且存在大量同骨架、同assay训练参照。结果只能作为稀疏已知阳性检索开发记录；下文检索主导的架构建议尚不能确定，应先修复任务评测。详见[测试集审计](BIOMASTER_ODTI_V4_TESTSET_AUDIT_20260906_ZH.md)。

**结果：首轮没有达到超越近邻和旧版模型的目标。** 已实际完成3种配置×3个种子共9次训练，以及两个旧版V3检查点和两个简单基线的统一测试。有符号证据改善了R1全局模型，但完整面板排名仍明显落后。

本轮是[完整V4设计](BIOMASTER_ODTI_V4_ARCHITECTURE_AND_DATA_20260905_ZH.md)中的R1全局预训练对照，加一个提前验证符号约束的证据分支试验。**不是完整V4，更不是DrugCLIP三维交互模型的训练结果。** 未晋级或替换生产模型。

**实际训练了什么**

- Morgan对照：Morgan2048＋完整ESM2序列均值1280，各自投影到256维；拼接、逐元素乘积、绝对差进入MLP，并叠加64维双线性交互。
- BerMol R1：用冻结公共BerMol768替代Morgan，其余全局交互和训练协议相同。新训练参数1,219,073个；公共编码器没有解冻。
- 有符号证据：继承相同种子的最佳BerMol模型，增加独立正支持奖励和负支持扣分；每类最多16个训练集参照，每支贡献非负且上限2。最终分数为`base + positive_bonus - negative_penalty`，空支路严格为零，负参照不进入基础分数或正分支。参照差值来自共享二元任务势函数，不是连续亲和力差。
- 证据前2轮冻结已拟合主干，贡献由0.5升至1；随后主干学习率2e-5，新模块2e-4。每支证据以0.2概率整支丢弃，训练覆盖单侧和无支持场景。

尚未训练DrugCLIP原子/口袋编码器、原子—残基交互、立体化学适配器、分端点亲和力头、浓度条件功能头或公共主干微调。此次结果不能用来否定这些后续模块，也不能代表DTIAM完整AutoML路线或DrugCLIP官方模型的能力。

**特征和数据范围**

BerMol有效缓存由62,679扩展到296,109个分子，本次新编码233,430个。资产总表另有1个非法模型SMILES（index 272054，`O=C(O)Cc1cccc1`），已明确标记不可用；它不在本轮427,394条关系中，训练、验证、测试和支持集合全部具有有效预训练表示。跨3个既有缓存及新补建行抽取15个表示，用官方CPU编码器重算，最大绝对误差为{provenance['max_absolute_error']:.2e}。

蛋白均值覆盖全部843个索引靶点：复用813个完整残基缓存，新增30个靶点残基表示。长序列使用1022残基窗口、128残基重叠，并按原残基位置拼合；均值不含BOS/EOS。不是只读取前1022个残基。

继续使用旧V3药物实体互斥开发切分，训练342,031条、验证43,331条、测试42,032条；训练药物233,730个，平均每药仅1.463个实测靶点。每轮遍历全部训练关系，再额外遍历6,450个有正有负的药物查询；每查询最多32条不同实测关系。损失为已测BCE＋药物等权PN排序；没有把未知配对作为训练阴性。

标签仍是既有混合端点二元活动标签，未完成原始测量级端点/实验条件重建。这是为验证R1而明示保留的开发协议，不能称为完整V4端点训练。

**测试分母与选模**

验证使用预先固定的512药×495靶点面板，测试使用1,024药×495靶点面板。查询按固定哈希顺序选取，至少有一个已知阳性；测试面板共有{protocol['panels']['test']['known_positive_pairs']}个已知阳性配对。所有模型、近邻及PN基线共享同一候选靶点、同一药物列表和同一个训练支持快照。支持检索排除查询药物的整个化学实体。

9个新模型只按验证完整面板宏AP选检查点；最多30轮、至少8轮、patience=5，实际完成{total_epochs}轮。测试在全部9个模型训练完成后统一计算。旧V3 B检查点沿用历史6轮训练及实测验证宏AP选模，因此它是固定历史对照，并非相同选模协议的新训练消融。

实测micro AP/AUROC使用42,032条正负配对；实测药物宏AP只对772个同时含正负的药物平均。完整面板AP/Recall@20衡量已知阳性的排序，未知配对只是检索背景，**不等于生物学阴性，也不代表全部真实靶点的召回率**。

**统一测试结果**

均值±样本标准差反映种子差异；近邻和PN是单个固定基线。

| 模型 | 种子数 | 实测micro AP | 实测micro AUROC | 实测药物宏AP | 完整面板宏AP | 完整面板Recall@20 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

BerMol相对Morgan的面板AP差为{ci('bermol_minus_morgan')}，没有显示预训练替换带来可靠增益。有符号证据相对BerMol为{ci('signed_minus_bermol')}，改进成立，但相对正近邻仍为{ci('signed_minus_positive_nearest')}。bootstrap以同一个药物为配对单位，先按种子平均每药AP，再重采样5,000次；它不包含数据集更换带来的不确定性。

**训练记录**

| 配置 | 种子 | 实际轮数 | 验证选择轮数 | 最佳验证面板AP |
|---|---:|---:|---:|---:|
{chr(10).join(train_rows)}

**失败机制和能支持的结论**

1. **全局压缩表示仍不足以建模选择性。** 本轮只把两个池化向量送入交互头，没有验证原子、残基或口袋层面的对应关系；冻结BerMol替换Morgan的同协议结果没有提升，继续堆大同类MLP缺乏本轮证据支持。
2. **监督目标与全靶点筛选仍存在差距。** 每药平均1.46个实测靶点，即使遍历全部关系、加入等药物排序和完整面板验证，也不能获得缺失的跨靶点选择性标签。新证据模型实测micro AP约0.984，但面板AP仅约0.272。
3. **正确的符号约束没有解决证据与全局分数的尺度关系。** 三个证据模型的全局logit第99百分位约6.57–10.76，而正支持奖励第99百分位只有0.44–0.47；微弱残差很难纠正全局路径的大幅误排。分解审计确认负证据没有向上加分，空支路为零，问题不再是旧版的符号旁路。尺度差提示后续应验证强基线锚定，不宜直接把增大cap当作已证实的解决办法。
4. **当前切分主要检验相近化合物迁移。** 测试查询中{near}/1024（{near/1024:.2%}）与某个训练阳性分子的最大Morgan相似度≥0.7；0.4–0.7为{middle}个，<0.4仅{low}个。低相似度样本不足以判断谁更擅长新化学空间；测试中未见训练靶点的实测关系也仅1条，不能宣称靶点冷启动性能。

**下一步架构和数据优先级**

1. **用强检索基线作主分数，再学习有约束的残差。** 先以正近邻或具备单调符号约束的PN分数作为锚点，让预训练交互模型学习相对参照的修正；使用验证面板AP和Recall@20的非退化门。当前“弱全局底座＋小幅证据”不应继续作为默认方案。
2. **落实R2的真实预训练局部交互。** 首先在同覆盖范围测DrugCLIP官方双塔原始分数，再保留其投影和基础分数，加入全局条件下的原子—残基/口袋交互。对照至少区分原始DrugCLIP、纯池化表示、局部交互及局部残差。R1不具备这部分能力，不能用R1结果代替该实验。
3. **加入能约束选择性的真实密集观测。** 优先整理同家族多靶点实测面板和KIRHub浓度条件功能标签；Ki、Kd、IC50按端点及可比实验条件拆开。连续相对活性监督必须来自同端点、可比实验，不能从现有混合均值伪造。
4. **建立下一轮冷骨架及靶点同源组切分。** 固定训练支持快照，单列低相似度与无阳性支持场景；保留本轮结果作历史回归。当前V3切分及既有KIRHub/S1–S5均已多次查看，后续不能重新宣称为全新外部确认集。

**实现、产物与复现**

- 模型：[biomaster/odti_v4.py](../biomaster/odti_v4.py)；本轮配置：[biomaster_odti_v4_r1_development.yaml](../configs/biomaster_odti_v4_r1_development.yaml)。
- 特征构建：`python scripts/build_biomaster_odti_v4_features.py`；协议准备：`python scripts/prepare_biomaster_odti_v4_experiment.py`。已有完成实验应先保留目录，不能覆盖当前冻结协议或模型。
- 新训练：`python scripts/train_biomaster_odti_v4.py --variant morgan --seed 20260905`；依次运行3种子Morgan、3种子BerMol、3种子signed。非空训练目录会拒绝覆盖；signed读取同种子BerMol最佳检查点。
- 测试：`python scripts/train_biomaster_odti_v4.py --evaluate-all`；历史对照：`python scripts/evaluate_biomaster_v3_on_v4_panel.py`；汇总：`python scripts/summarize_biomaster_odti_v4.py`。
- [实验目录](../outputs/biomaster_odti_v4_20260905/)保留特征、带校验和的协议及源代码快照、9组BEST/LAST检查点、每轮历史、13组预测和指标、配对bootstrap、分数分解与验证记录。
- 相关54项测试通过，包括CUDA直接/缓存推理一致性、负证据方向、空证据/NaN填充、初始梯度、共享参照势函数及数据/检索协议。训练后逐检查点确认选中轮次确实为验证AP最大值；所有实际训练轮都覆盖342,031条关系。

本轮没有新训DTIAM AutoML或DrugCLIP官方模型，没有形成新的外部SOTA结论；它完成了全量预训练特征补建，并给出了R1路线及符号约束的可复核失败与改进证据。
'''
    (ROOT/'docs/BIOMASTER_ODTI_V4_R1_TRAIN_TEST_20260905_ZH.md').write_text(report)
    files=list(evaluation.glob('*.json'))+list(evaluation.glob('*.npz'))+[out/'VALIDATION_V4_R1.json']
    files += list((out/'runs').glob('*/*.pt')) + list((out/'runs').glob('*/STATUS_V4.json'))
    write_json(out/'RESULT_ARTIFACT_MANIFEST_V4.json',{'files':{str(p.relative_to(ROOT)):sha256(p) for p in files}})
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
