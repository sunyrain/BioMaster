#!/usr/bin/env python3
"""Write a factual portable model card and compact evaluation evidence."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from biomaster.portable_ranker_v2 import digest
from prepare_biomaster_unified_interaction import write_json
OUT=ROOT/'outputs/biomaster_best_model_20260906'


def number(value):
    return '—' if pd.isna(value) else f'{value:.4f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',
        *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def main(bundle):
    bundle=Path(bundle).resolve();meta=json.loads((bundle/'metadata.json').read_text())
    selected=json.loads((OUT/'FINAL_ARCHITECTURE_SELECTION.json').read_text())
    deployed=json.loads((OUT/'DEPLOYMENT_SELECTION.json').read_text())
    fitted=json.loads(Path(deployed['result']['path']).read_text())
    train=json.loads((OUT/'data/fullfit_2025/MANIFEST.json').read_text())
    if meta['source_checkpoint']!=deployed['checkpoint']:raise ValueError('model card must describe selected weights')
    if meta['status']!='SELECTED_CATALOG_MODEL':raise ValueError('cannot document a prototype as selected')
    globals_=pd.read_csv(OUT/'ALL_GLOBAL_CANDIDATES.csv');local=pd.read_csv(OUT/'ANCHORED_CANDIDATES.csv')
    metrics=pd.read_csv(OUT/'regression/METRICS_SEED_SUMMARY.csv');representative=pd.read_csv(OUT/'regression/REPRESENTATIVE_PRECISION_METRICS.csv')
    artifacts={'development.csv':OUT/'ALL_GLOBAL_CANDIDATES.csv','refinement.csv':OUT/'ANCHORED_CANDIDATES.csv',
        'evaluation.csv':OUT/'regression/METRICS_SEED_SUMMARY.csv','representative_precision.csv':OUT/'regression/REPRESENTATIVE_PRECISION_METRICS.csv',
        'architecture_selection.json':OUT/'FINAL_ARCHITECTURE_SELECTION.json','training_data.json':OUT/'data/fullfit_2025/MANIFEST.json',
        'paired_regression.json':OUT/'regression/PAIRED_COMPARISONS.json'}
    for name,source in artifacts.items():shutil.copy2(source,bundle/name)
    gp=table(['全局候选','综合分','药物→靶点 AP','靶点→药物 AP'],[
        [r['name'],number(r['selection_mean']),number(r['d2t_ap_mean']),number(r['t2d_ap_mean'])] for r in globals_.to_dict('records')])
    lp=table(['局部阶段','参数','综合分','药物→靶点 AP','靶点→药物 AP'],[
        [r['variant'],int(r['parameters']),number(r['selection_mean']),number(r['d2t_ap_mean']),number(r['t2d_ap_mean'])] for r in local.to_dict('records')])
    scope_names={'2023_2025_dense_old720':'2023–2025 新关系 / 当前720药',
        '2023_2025_dense_approved_pre2023':'2023–2025 新关系 / 截点前获批594药',
        'kirhub_historical_strict_2823':'KIRHub严格2823对',
        'kirhub_single_construct_unseen_through2022':'KIRHub单构建体且历史未见3914对',
        '2023_2025_measured_all_compounds':'2023–2025 一般化合物实测候选'}
    tests=[]
    for scope,title in scope_names.items():
        rows=[]
        for name in ['selected','global','capacity','temporal_J','positive_nearest','temporal_v3']:
            part=metrics[(metrics.scope==scope)&(metrics.model==name)].set_index('direction')
            def cell(direction,key):
                row=part.loc[direction];value=number(row[key+'_mean']);sd=row[key+'_std']
                return value+(f' ± {sd:.4f}' if pd.notna(sd) else '')
            rows.append([name,cell('d2t','macro_ap'),cell('t2d','macro_ap'),cell('d2t','macro_recall_20'),cell('t2d','macro_recall_20')])
        tests.append('### '+title+'\n\n'+table(['模型','药物→靶点 AP','靶点→药物 AP','药物→靶点 R@20','靶点→药物 R@20'],rows))
    rep_rows=[]
    for scope,title in scope_names.items():
        part=representative[representative.scope==scope]
        for precision in ['bf16','fp32']:
            q=part[part.precision==precision].set_index('direction')
            rep_rows.append([title,precision,number(q.loc['d2t','macro_ap']),number(q.loc['t2d','macro_ap'])])
    rep=table(['范围','精度','药物→靶点 AP','靶点→药物 AP'],rep_rows)
    params=int(selected['selected_development']['parameters']);seed=selected['deployment_seed']
    local_note=('本次保留全局模型。局部原子—残基与口袋几何已实际训练并经过同父权重、同额外曝光、近似等容量对照，但未达到预定保留规则或未超过父模型。'
        if selected['selected_variant']=='parent' else f"局部阶段选择 `{selected['selected_variant']}`，保留依据见 architecture_selection.json 中逐窗口、逐种子的 eligibility。")
    diagram='''```mermaid
flowchart LR
    D[DrugCLIP 全局分子 512维] --> DP[药物投影 2601→192]
    FP[Morgan 2048维] --> DP
    G[图均值40维及可用标志1维] --> DP
    T[完整蛋白 ESM2 均值1280维] --> TP[蛋白投影1280→192]
    DP --> X[拼接 d、t、d×t、绝对差]
    TP --> X
    X --> H[共享全局交互与残差 MLP / 192维]
    H --> S[一个共享输出层 / 两个方向 logit]
    S --> DT[固定药物 排靶点]
    S --> TD[固定靶点 排老药]
```'''
    if meta['local_interactions']:
        diagram=diagram.replace('    H --> S[', '    A[原子与化学键] --> L[全局条件化原子—残基修正]\n    R[位点残基与可选受体CA几何] --> L\n    H --> L\n    L --> S[')
    card=f'''# ReTargetMap 2026-09-06 目录模型

交付一个可独立运行的神经检查点，用于720个目录老药与384个匹配靶点的双向排序。选型范围是本轮明确比较的候选；不作通用最优、SOTA或前瞻验证声明。

架构为 `{meta['family']}` / `{meta['variant']}`，分子输入 `{meta['drug_representation']}`，**{params:,}** 个可训练参数。EMA只在训练时平滑同一网络权重，推理使用一个检查点。两个方向共用交互表示和输出层；没有V3/J/近邻模型分数集成。

## 实际架构

{diagram}

DrugCLIP和ESM2提供缓存的冻结表征，本轮未端到端微调这些公共编码器。Morgan、图均值与分子CLS一次拼接后投影；蛋白输入来自完整序列均值。两侧192维表示的拼接、逐元素乘积和绝对差进入共享MLP。输出为两个方向logit，不能当作亲和力、结合概率或跨不同查询已校准的数值。

{local_note} 这不证明局部相互作用不重要，只限定当前的残基选择、训练监督与优化方案。局部对照使用独立分子构象与受体内部CA距离，没有配体—受体共同坐标系或真实原子接触监督。

## 开发选择和消融

54次全局训练覆盖9候选、两个时间窗口和三个种子；随后24次同父模型局部消融。窗口为≤2018训练/2019–2020验证和≤2020训练/2021–2022验证。三个种子为20260921、20260922、20260923。以下均先对每个种子平均两个窗口，再求种子均值；种子波动见随包CSV。

预设选择分为 `0.35×正向AP + 0.35×逆向AP + 0.15×正向R@20 + 0.15×逆向R@20`。第一组分子结果出现后追加Morgan单独与DrugCLIP＋Morgan对照，该开发扩展已记录；不将整个比较伪称一次预注册。2023–2025与KIRHub不参与本轮架构、轮次或部署种子选择。

{gp}

统一取所有全局候选第6轮的曝光诊断，DrugCLIP＋Morgan＋EMA综合分0.4264，原G为0.3796，G＋EMA为0.3928。同种子/窗口的实测BCE与既有关系检索曝光一致，输入增益不只来自不同早停轮次。该诊断未重新选型。

{lp}

局部四组均从同种子/同窗口的实际父模型启动，初始化时FP32预测完全相同；额外训练4轮、取最终EMA，不能在这4轮之间挑最好结果。等容量对照与geometry相差185个参数。局部保留要求同时超过继续训练与容量对照，并满足逐窗口、至少两个种子的重复性条件；geometry还须稳定超过site。

双向存在权衡：全局实测查询排序方案的逆向AP均值更高。DrugCLIP＋Morgan相对G＋EMA在≤2020窗口的逆向AP差为−0.0740，未校正配对查询95%区间为[−0.1544, −0.0046]；它并非每个窗口、每个方向都提升。开发查询少、多个比较未作多重校正，不能把选择分优势等同于统计上全面优胜。

## 固定时间与KIRHub回归

下表神经候选使用**≤2022标签重新训练**的三个检查点，按种子分别计算后报告均值±样本标准差；不是先平均模型分数再评估。G/capacity沿用各自早期验证选择的轮次，selected固定全局{selected['fixed_refit_global_epochs']}轮、局部{selected['refinement_epochs']}轮。V3/J/近邻为冻结的历史单次分数，没有伪造种子标准差。

主结果沿用开发的bf16 autocast（保存float32分数）；部署推理默认FP32，代表种子的精度敏感性另列。所有本轮时间预测先冻结权重和分数，再读取回归标签。这些测试在以前研究轮次中已经看过，属于回顾性回归，不能称为新的外部确认。

{chr(10).join(tests)}

密集新关系检索排除≤2022及无日期的既有测量；其余候选是未报告背景，不等于实测阴性。2023–2025当前720药有71条新合格阳性、45个药物和49个靶点查询；594个可确认早期获批药子集有54条阳性、34个药物和40个靶点查询。年份来自文献年份与本地合格测量，并不等于关系的生物学首次发现。720目录还包含48个较晚获批药与78个获批日期缺失药。

KIRHub严格集合为2823对/202阳性，单构建体且≤2022未见集合为3914对/322阳性；两者历史测量重叠均为0。阳性定义为1μM残余活性≤30%，这是功能抑制迁移评估，不是Kd/Ki亲和力排序。KIRHub和一般实测表只在实际测量的候选内计算宏AP；不能与全目录新关系AP直接比较。配对查询区间在 paired_regression.json，各种子先在同一查询内平均AP再做查询重采样，不把重复种子当成独立样本。

### 预先选定代表种子 {seed}

代表种子取开发综合分最接近三种子中位数者。以下为其**≤2022重训版本**，与本包部署权重属于相同训练方案、不同训练数据。

{rep}

## 部署拟合数据与范围

本包 `model.pt` 使用≤2025合格测量拟合：{train['rows']:,}条唯一实测二分类关系，{train['positives']:,}条阳性，{train['molecules']:,}个分子；目录老药{train['old_rows']:,}条关系、{train['old_positives']:,}条阳性。先截断日期再聚合标签，均值pChEMBL≥6为阳性，≤5或合格明确inactive为阴性；跨阈值冲突排除，未知不进入实测阴性BCE。既有关系检索将未报告项作为排序背景，不宣称其生物学无活性。无日期测量不进入此次日期拟合。

固定训练方案为 `{selected['recipe']}`，全局{selected['fixed_refit_global_epochs']}轮，选中后局部{selected['refinement_epochs']}轮，种子{seed}。所选部署阶段训练日志耗时{fitted['seconds']:.1f}秒，峰值已分配GPU显存{fitted['peak_gpu_bytes']/1024**3:.2f}GiB；这些数字使用已缓存的公共表征，不包含离线编码/数据准备。局部开发由两个GPU进程并行运行，其墙钟时间不作为隔离的算法速度比较。**全量部署权重使用了2023–2025关系，没有独立测试分数。** 上述时间回归评价的是≤2022版本，不能转贴为这份全量权重的独立成绩。

包中保存一个网络与所需目录缓存，默认目标候选数384、老药候选数720。745是完整登记目录，450主路由还需覆盖和跨体系校准；不扩大本包排名分母。新SMILES或新蛋白不属于当前API，需先完成一致的特征扩展与验证。公共编码器的预训练年代、关系重叠未认证；DrugCLIP表征取自所记录的2026权重来源，不冒称严格的2022年前预训练体系。上游来源和使用条款见 provenance.json。

## 使用与验证

安装 requirements.txt 后运行 README.md 的 infer.py 命令。Python接口为 `CatalogRanker`，提供 `score_pairs`、`rank_targets`、`rank_drugs`；返回目录名称、分数、排名及实际候选数。默认CPU FP32，支持CUDA。

发布验证程序逐项比较全部720药/384靶点的模型输入和全部权重；全局架构进一步检查276480配对的导出前后FP32输出。另测目录外独立CPU导入、双向查询、重载一致性和错误输入拒绝。局部架构才适用的可选结构缺失检查会显式关闭几何。最终验证记录保存于模型目录旁的 `_RELEASE_VALIDATION.json`，不以少量随机配对替代全目录映射核验。

原始检查点SHA256：`{meta['source_checkpoint']['sha256']}`。本包不包含优化器状态、不在推理时调用研究训练脚本或下载编码器。范围内最佳仅指本次开发协议；模型预测仍需后续实验检验。
'''
    (bundle/'MODEL_CARD.md').write_text(card)
    files={}
    for path in sorted(bundle.rglob('*')):
        if not path.is_file() or path.name=='MANIFEST.json' or '__pycache__' in path.parts:continue
        name=str(path.relative_to(bundle));files[name]=dict(sha256=digest(path),size_bytes=path.stat().st_size)
        if name in ['features/ca.npy','features/quality.npy','features/geometry_mask.npy']:files[name]['optional_geometry']=True
    write_json(bundle/'MANIFEST.json',dict(format_version=2,files=files))
    write_json(OUT/'MODEL_CARD_BUILD.json',dict(status='COMPLETE',model_card=file_identity(bundle/'MODEL_CARD.md'),
        manifest=file_identity(bundle/'MANIFEST.json'),producer=file_identity(Path(__file__))))
    print(json.dumps(dict(model_card=str(bundle/'MODEL_CARD.md'),files=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);main(p.parse_args().bundle)
