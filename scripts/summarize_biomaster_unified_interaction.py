#!/usr/bin/env python3
"""Summarize actual first-round results without promoting a single-seed model."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import pandas as pd
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import OUTPUT,write_json


def markdown_table(frame):
    columns=list(frame.columns)
    def cell(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(cell(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def training_plot(histories,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(global_='#4b5563',capacity='#927329',sequence='#2878b5',site='#e78532',geometry='#8059b5')
    fig,axes=plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
    panels=[('Observed training BCE',lambda r:r['mean_bce']),
            ('Old-drug validation: drug to target AP',lambda r:r['validation']['d2t']['ap']),
            ('Old-drug validation: target to drug AP',lambda r:r['validation']['t2d']['ap']),
            ('Predefined validation selection score',lambda r:r['validation']['selection'])]
    for ax,(title,value) in zip(axes.flat,panels):
        for name,history in histories.items():
            color=colors['global_' if name=='global' else name]
            x=[r['epoch'] for r in history];y=[value(r) for r in history]
            ax.plot(x,y,'o-',color=color,label=name,linewidth=1.5,markersize=3)
            if title.startswith('Predefined'):
                best=history[-1]['selected_epoch'];ax.scatter(best,y[best-1],marker='*',s=115,color=color,zorder=5)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.suptitle('Unified interaction ablation — one seed; <=2020 training, 2021–2022 validation',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'TRAINING_CURVES.{suffix}',dpi=200)
    plt.close(fig)


def main():
    out=OUTPUT.parent;p=json.loads((ROOT/'configs/biomaster_unified_interaction_20260906.json').read_text())
    selection=json.loads((out/'SELECTION.json').read_text());metrics=pd.read_csv(out/'TEST_METRICS.csv')
    validation=json.loads((out/'VALIDATION.json').read_text());geometry=json.loads((out/'GEOMETRY_PROBE.json').read_text())
    precision=json.loads((out/'PRECISION_PROBE.json').read_text())
    atom=json.loads((OUTPUT/'ATOM_MANIFEST.json').read_text());conformer=json.loads((OUTPUT/'CONFORMER_AUDIT.json').read_text())
    histories={};rows=[]
    for variant in p['variants']:
        for seed in p['seeds']:
            folder=out/'development'/variant/f'seed_{seed}'
            result=json.loads((folder/'RESULT.json').read_text());histories[variant]=json.loads((folder/'HISTORY.json').read_text())
            v=result['validation']
            rows.append(dict(variant=variant,seed=seed,parameters=result['trainable_parameters'],
                             trained_epochs=result['trained_epochs'],selected_epoch=result['selected_epoch'],
                             d2t_ap=v['d2t']['ap'],t2d_ap=v['t2d']['ap'],d2t_r20=v['d2t']['r20'],t2d_r20=v['t2d']['r20'],
                             measured_d2t_ap=v['d2t']['measured_ap'],measured_t2d_ap=v['t2d']['measured_ap'],
                             seconds=result['seconds'],max_epoch_reached=result['max_epoch_reached']))
    # Identical sample ordering AND additional retrieval exposures for every common epoch.
    for variant,history in histories.items():
        for other,second in histories.items():
            for a,b in zip(history,second):
                assert a['exposure_sha256']==b['exposure_sha256'],(variant,other,a['epoch'])
                assert a['coverage_rows']==b['coverage_rows']==292408
    ablation=pd.DataFrame(rows);ablation.to_csv(out/'DEVELOPMENT_ABLATION.csv',index=False)
    matched_rows=[]
    for variant,history in histories.items():
        r=history[p['min_epochs']-1];v=r['validation']
        matched_rows.append(dict(variant=variant,epoch=r['epoch'],d2t_ap=v['d2t']['ap'],t2d_ap=v['t2d']['ap'],
                                 d2t_r20=v['d2t']['r20'],t2d_r20=v['t2d']['r20'],selection=v['selection'],
                                 exposure_sha256=r['exposure_sha256']))
    matched=pd.DataFrame(matched_rows);matched.to_csv(out/'DEVELOPMENT_MATCHED_EPOCH.csv',index=False)
    training_plot(histories,out)
    cols=['model','direction','macro_ap','macro_recall_20']
    temporal=metrics[metrics.scope.eq('2023_2025_dense_old720')][cols]
    approval=metrics[metrics.scope.eq('2023_2025_dense_approved_pre2023')][cols]
    kir=metrics[metrics.scope.eq('kirhub_historical_strict_2823')][cols]
    local=next(v for v in selection['final_variants'] if v not in ['global','capacity'])
    overall=max(selection['development_results'],key=lambda r:r['selection'])['variant']
    primary=temporal.set_index(['model','direction'])
    findings=[]
    for candidate in dict.fromkeys([overall,local]):
        d=float(primary.loc[(candidate,'d2t'),'macro_ap']);t=float(primary.loc[(candidate,'t2d'),'macro_ap'])
        differences=[]
        for baseline in ['global','temporal_J','positive_nearest']:
            if baseline==candidate:continue
            dd=d-float(primary.loc[(baseline,'d2t'),'macro_ap']);dt=t-float(primary.loc[(baseline,'t2d'),'macro_ap'])
            differences.append(f'相对{baseline}为{dd:+.4f}/{dt:+.4f}')
        findings.append(f'{candidate}的2023–2025双向AP为{d:.4f}/{t:.4f}；'+'，'.join(differences)+'。')
    comparisons=json.loads((out/'TEST_PAIRED_COMPARISONS.json').read_text())
    ci=[r for r in comparisons if r['scope']=='2023_2025_dense_old720' and r['candidate']==local and r['control'] in ['global','capacity','temporal_J','positive_nearest']]
    citable=pd.DataFrame(ci)[['control','direction','delta_ap','ci_low','ci_high']]
    global_ci=pd.DataFrame([r for r in comparisons if r['scope']=='2023_2025_dense_old720' and r['candidate']=='global' and r['control'] in ['temporal_J','positive_nearest']])[['control','direction','delta_ap','ci_low','ci_high']]
    kir_ci=pd.DataFrame([r for r in comparisons if r['scope']=='kirhub_historical_strict_2823' and r['candidate']=='geometry' and r['control'] in ['capacity','temporal_J']])[['control','direction','delta_ap','ci_low','ci_high']]
    counts=conformer['counts'];not_converged=sum(n for key,n in counts.items() if key.endswith('_1'))
    probe=[]
    for name in ['original','shuffled']:
        r=geometry[name];probe.append(dict(input=name,d2t_ap=r['d2t']['ap'],t2d_ap=r['t2d']['ap'],d2t_r20=r['d2t']['r20'],t2d_r20=r['t2d']['r20']))
    lines=[
      '# 统一交互主干：实现与首轮受控训练测试',
      '',
      '2026-09-06。状态：**五组单种子开发对照与三个早期选定模型的时间重训回归已完成；未替换生产/V3＋J。**',
      '',
      '这轮已实现共享交互表示，并完成训练与测试。各表为本次实际输出。单种子、已观察过的测试集合及公共预训练来源限制了结论：不能据此宣称超过DTIAM/DrugCLIP或获得新的前瞻验证。',
      '',
      f'早期验证的总体优胜者是 **{overall}**，局部方案中优胜者是 **{local}**；总体选择来自2021–2022验证分数，未在后期测试中重选。',
      '',
      '\n\n'.join(findings),
      '',
      '**本轮主要发现：预先纳入的简单G对照在时间回归取得0.3722/0.1828双向AP，原J为0.2806/0.1565；完整geometry为0.2550/0.1789，没有超过G。KIRHub的逆向提升也能由等容量全局模型达到，不能归因于口袋几何。几何对应被打乱后早期老药检索AP没有下降，尚不能证明几何匹配对目标任务有稳定益处。** G在后期测试表现较好属于结果观察，不回改已冻结的早期选择。',
      '',
      '## 1. 实际架构',
      '',
      '下图为已实现并训练的geometry完整候选；global/capacity仅使用同源全局输入。公共分子与蛋白编码器保持冻结，训练的是共享交互模块和读出。',
      '',
      '```mermaid',
      'flowchart LR',
      ' D[完整分子图＋独立分子构象] --> A["分子表示<br/>DrugCLIP原子/CLS＋键与手性"]',
      ' S[完整蛋白序列] --> R["冻结ESM2 650M<br/>1280维真实残基＋完整均值"]',
      ' P[P2Rank口袋＋精确序列映射] --> C["蛋白CA内部距离<br/>结构质量与有效性mask"]',
      ' P -->|局部残基选取| R',
      ' A --> U["统一交互主干<br/>192维token／32维持续pair／2层更新"]',
      ' R --> U',
      ' C --> U',
      ' U --> H[一个共享交互表示]',
      ' H --> O["两个线性方向读出<br/>药→靶／靶→药"]',
      '```',
      '',
      '全局摘要每层参与原子—残基pair更新，pair状态再更新两侧token与全局状态；分子消息绑定具体邻居的键类型和立体信息。口袋模块只用蛋白内部CA距离，未使用分子—蛋白跨坐标系距离。缺失口袋时几何增量严格为零。近邻和旧J只作为独立对照，不是该网络的必加分数。',
      '',
      'G使用同源CLS、全分子化学摘要和完整蛋白均值；局部版增加逐原子化学特征和同源预训练token。原子数超过128的分子保留完整图摘要并回退，不截断后冒充完整分子。所有标签行继续参与训练。',
      '',
      '五组都使用由分子构象产生的DrugCLIP全局表示，G也包含分子内部三维预训练信息。本轮比较的是增加显式局部交互和蛋白口袋几何的增量，不是“三维对纯二维”的实验。',
      '',
      '## 2. 数据与表示覆盖',
      '',
      f'- 必需化合物：{atom["required_molecules"]:,}；成功DrugCLIP预训练表示：{atom["pretrained_available"]:,}；保存真实重原子状态：{atom["atom_count"]:,}。',
      '- 384/384个靶点有完整逐残基ESM2，补齐了原先缺失的24个；352个靶点有严格映射的759个预测口袋，32个靶点走序列回退。',
      '- 720个目录老药全部保留局部完整重原子图，其中719个具备DrugCLIP三维预训练表示，1个使用化学图回退。',
      '- 局部上限96个真实残基，完整蛋白摘要始终保留；352个有口袋靶点中334个保留整个口袋残基并集，其余18个显式子采样。覆盖率记录在TARGET_COVERAGE.csv，不能把局部96个token宣称为整条蛋白逐残基交互。',
      f'- 构象采用ETKDGv3＋最多50步MMFF/UFF；{not_converged:,}个优化返回未收敛标志，因此是可复现的初始构象，不能称为已充分优化的低能构象。{counts.get("embedding_failed_graph_fallback",0):,}个构象失败与{counts.get("oversized_global_graph_fallback",0):,}个超限分子使用明确的图/全局回退。',
      '- 开发训练292,408条≤2020年实测二元关系，其中720目录有770条阳性；验证使用2021–2022首次合格测量，老药53条阳性、34个药物查询和39个靶点查询。最终重训351,620条≤2022关系，老药815条阳性。',
      '',
      '## 3. 首轮消融：早期验证集',
      '',
      'global＝G；capacity＝等参数量全局对照；sequence＝G＋真实序列残基局部交互；site＝改用预测口袋残基，无三维几何；geometry＝site＋内部CA几何和质量。参数量对照与几何版仅相差78个参数。site到geometry仍额外增加约15万参数，因此几何特异性还需后续局部容量对照及复现，不只看一个AP差值。',
      '',
      markdown_table(ablation[['variant','parameters','trained_epochs','selected_epoch','d2t_ap','t2d_ap','d2t_r20','t2d_r20','seconds']]),
      '',
      '各公共轮次的逐样本曝光SHA256完全一致；每轮完整覆盖292,408条观测，并使用相同种子抽取老药检索查询。BCE只接收实测0/1；额外检索分母包含未知关系，保留查询的全部已知阳性。此检索目标仍会受到未标注真阳性的影响，不能解释成把未知关系证实为无活性。',
      '',
      '补充列出固定第6轮（协议预设的最小训练轮数），观察相同累计标签曝光下的表现。此表不假定各架构已同等收敛；主选择仍按各自早期验证最佳轮次。',
      '',markdown_table(matched[['variant','epoch','d2t_ap','t2d_ap','d2t_r20','t2d_r20']]),
      '',
      '训练上限12轮，至少6轮、验证4轮无改善早停；最终从头重训到开发选定轮数。达到上限不等于充分收敛，训练时间和验证轨迹一并保存。这里没有同条件连续亲和力或复合物接触标签监督。',
      '',
      '![训练与早期验证曲线](../outputs/biomaster_unified_interaction_20260906/TRAINING_CURVES.png)',
      '',
      f'早期验证预先规定的AP/R@20复合目标选择局部候选 **{local}**；最终仅重训global、capacity和该局部候选，未用后面的测试成绩回选。',
      '',
      '## 4. ≤2022重训，2023–2025首次合格测量回归',
      '',
      '同一720药×384靶点目录，移除截止2022已测或无日期的关系；未知候选仅为检索背景。此处71条新阳性，45个药物查询、49个靶点查询。所有模型均按相同下游年份比较。',
      '',markdown_table(temporal),'',
      'G对照相对J及近邻的AP差与查询bootstrap 95%区间：药→靶下界略高于0，靶→药区间跨0。它们是单种子、已观察过回归集上的未校正区间，不等于独立确认；G的药→靶R@20仍低于近邻。',
      '',markdown_table(global_ci),'',
      f'局部候选{local}相对对照的AP差与查询bootstrap 95%区间（不是跨种子区间，也未作多重比较校正）：',
      '',markdown_table(citable),'',
      '594个可确认2023年前获批药物的敏感性分析（54条阳性、34个药物查询、40个靶点查询）：',
      '',markdown_table(approval),'',
      '分年、实测候选AP及有结构共同范围的结果见TEST_METRICS.csv。各范围分母不同，不能直接横向比较其AP高低。',
      '',
      '## 5. KIRHub回归与几何敏感性',
      '',
      '历史严格2823条关系，标签为1μM下残余活性≤30%。本表的V3、J、NN也使用≤2022时间版，与此前全时段J的KIRHub数字不是同一组权重。逐项记录了与早期已测关系的重叠，另列单构建体且截止2022未测的范围；KIRHub是功能抑制测量，不是统一Ki/Kd亲和力。',
      '',markdown_table(kir),'',
      'geometry相对同时间J与等容量全局对照的AP差与查询bootstrap 95%区间：逆向相对J为正，但相对capacity接近0且区间跨0。容量和交互输入的贡献不能混为一谈。',
      '',markdown_table(kir_ci),'',
      'geometry开发权重的早期验证坐标打乱检查（保持口袋选择、残基状态、质量与mask，仅打乱几何对应）：',
      '',markdown_table(pd.DataFrame(probe)),'',
      '扰动只用于观察模型是否利用几何；敏感性不等于已学到真实接触，不能替代有复合物真值的验证。',
      '',
      '主实验采用BF16 autocast、保存FP32分数，固定候选ID处理并列。早期G权重的额外全FP32推理诊断不回改选模：'+
      f'BF16双向AP={precision["results"]["bf16_autocast"]["d2t"]["ap"]:.4f}/{precision["results"]["bf16_autocast"]["t2d"]["ap"]:.4f}，'+
      f'FP32={precision["results"]["fp32"]["d2t"]["ap"]:.4f}/{precision["results"]["fp32"]["t2d"]["ap"]:.4f}。微小差异需结合数值精度和跨种子复现解释。',
      '',
      '## 6. 结论边界与下一轮',
      '',
      f'本轮总体验证优胜者{overall}与局部候选{local}的成绩均保留；整套方案的变化不能全部归因于三维交互。G的时间排序改善、geometry的KIRHub逆向改善、以及geometry未超过等容量对照，是不同层面的结果。',
      '',
      '早期验证只含53条老药阳性，选择capacity而后期G表现更好，提示需要更早的滚动时间验证和跨种子检查来评估选择稳定性。尚不能据此把G事后改写成预先选定的优胜模型。',
      '',
      '本轮回答“统一交互能否训练、哪些信息在首轮有增量”。生产候选不自动晋级。下一轮先复现20260922/20260923种子并加入局部等容量/几何打乱训练对照；只对可重复增益部分继续投入。若局部版未超过简单G，应先定位优化目标、原子/残基对齐和预训练失配，不能靠继续叠评分分支解释。',
      '',
      '当前训练是全体实测BCE＋老药已知关系的均匀候选检索，没有同assay的实测难阴性/相近靶点选择性排序，也没有局部接触真值。现有结果只约束这套训练方案，不能否定原子—残基或三维信息本身。优先验证两项算法改动：由本轮G初始化共享全局表示，再逐步开放局部层并联合优化；加入同条件实测选择性排序，检验局部特征是否终于被用于区分相近候选。前者须配套“G继续训练相同步数”的对照，不固定添加旧模型分数。',
      '',
      '当前交互仅受关系标签监督；冻结公共编码器＋CA口袋几何不等于训练过真实复合物接触。值得进一步检验的是复合物接触/结构对比预训练、口袋侧配套预训练、可靠的构象与结构质量、部分编码器微调，以及对未标注阳性更稳健的双向检索目标。每项先做来源重叠审计，再进行独立消融。',
      '',
      '**时间声明只针对下游标签。** 当前DrugCLIP发布权重和ESM2语料尚未证明早于2020/2022，DrugCLIP还是分子—口袋对比预训练，可能存在训练复合物重叠；完整预训练年代与重叠审计未完成。当前目录本身也不是历史冻结目录。本轮与先前已打开的2023–2025/KIRHub都只能作回顾性开发/回归，不作严格历史前瞻或新SOTA结论。',
      '',
      '## 7. 实现与复现',
      '',
      '- 模型：`biomaster/unified_interaction.py`；特征接口：`biomaster/unified_features.py`。',
      '- 特征：`scripts/prepare_biomaster_unified_interaction.py`；官方接口核验：`scripts/verify_biomaster_unified_pretrained.py`。',
      '- 训练：`scripts/train_biomaster_unified_interaction.py`；顺序运行：`scripts/run_biomaster_unified_round.py`。',
      '- 协议：`configs/biomaster_unified_interaction_20260906.json`；产物：`outputs/biomaster_unified_interaction_20260906/`。',
      '- 20项模型/时间/排序测试通过；完整特征、残基索引、相同曝光、预训练接口及旧V3/J源码哈希检查通过。大特征和权重只存本地，Git保留紧凑审计文件。',
      '',
      '```bash',
      'python scripts/prepare_biomaster_unified_interaction.py all --workers 48',
      'python scripts/verify_biomaster_unified_pretrained.py',
      'python scripts/run_biomaster_unified_round.py',
      '```',
      '']
    report=ROOT/'docs/BIOMASTER_UNIFIED_INTERACTION_FIRST_ROUND_20260906_ZH.md'
    report.write_text('\n'.join(lines))
    write_json(out/'RESULT_MANIFEST.json',dict(status='COMPLETE',single_seed=True,promoted=False,early_validation_winner=overall,best_local=local,
               files={name:file_identity(out/name) for name in ['VALIDATION.json','VALIDATION_TESTS.json','DEVELOPMENT_ABLATION.csv','DEVELOPMENT_MATCHED_EPOCH.csv','SELECTION.json','TEST_RELEASE.json','TEST_RESULT.json','TEST_METRICS.csv','TEST_PAIRED_COMPARISONS.json','GEOMETRY_PROBE.json','PRECISION_PROBE.json','BASELINE_REPRODUCTION.json','FEATURE_QUALITY.json','ATOM_GRAPH_COORDINATE_CHECK.json','ESM2_ALIGNMENT_CHECK.json','ENCODER_PROVENANCE.json','ENVIRONMENT.json']},
               report=file_identity(report),matched_exposure_all_common_epochs=True))
    print(str(report))


if __name__=='__main__':main()
