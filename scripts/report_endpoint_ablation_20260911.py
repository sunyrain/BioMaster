#!/usr/bin/env python3
"""Publish a concise evidence-led report after the six-run comparison finishes."""
import json,sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_endpoint_ablation_20260911 import OUT,FEATURES,write_json
from biomaster.portable_ranker_v2 import digest

def main():
    summary=json.loads((OUT/'SUMMARY.json').read_text());assert summary['status']=='COMPLETE_RESEARCH_COMPARISON'
    metrics=pd.read_csv(OUT/'TEST_METRICS.csv');data=pd.read_csv(OUT/'ARM_DATA_COUNTS.csv')
    bootstrap=json.loads((OUT/'PAIRED_SCAFFOLD_BOOTSTRAP.json').read_text())
    models=json.loads((OUT/'ALL_FITS_FROZEN.json').read_text())['runs']
    rows=[]
    for (arm,panel,scope),g in metrics.groupby(['arm','panel','scope']):
        row=dict(arm=arm,panel=panel,scope=scope,pairs=int(g.pairs.iloc[0]),positive=int(g.positive.iloc[0]),negative=int(g.negative.iloc[0]))
        for key in ['auroc','ap','brier','top1pct_precision','top5pct_precision','false_positive_rate','precision','recall']:
            row[key+'_mean']=float(g[key].mean()) if g[key].notna().any() else None
            row[key+'_sd']=float(g[key].std()) if g[key].notna().sum()>1 else None
        rows.append(row)
    grouped=pd.DataFrame(rows);grouped.to_csv(OUT/'RESULT_COMPARISON.csv',index=False)
    q=pd.read_csv(OUT/'TEST_QUERY_METRICS.csv.gz')
    qm=q.groupby(['arm','seed','panel','direction']).agg(queries=('query','size'),macro_ap=('ap','mean'),macro_auroc=('auroc','mean'),macro_top1pct_precision=('top1pct_precision','mean')).reset_index()
    qm.to_csv(OUT/'QUERY_MACRO_SUMMARY.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    for ax,(panel,metric,title) in zip(axes.flat,[('AFFINITY_KD_KI','auroc','Kd/Ki AUROC'),('AFFINITY_KD_KI','ap','Kd/Ki AP'),
                                               ('ACTIVITY_IC50','ap','IC50 AP'),('EXPLICIT_INACTIVE','false_positive_rate','Explicit inactive false-positive rate')]):
        f=metrics[metrics.panel.eq(panel)&metrics.scope.eq('all')]
        pivot=f.pivot(index='seed',columns='arm',values=metric)
        for _,r in pivot.iterrows():ax.plot([0,1],[r.kdki_inactive,r.all_inactive],color='#aab4be',alpha=.6,linewidth=1)
        for x,arm in enumerate(['kdki_inactive','all_inactive']):
            v=pivot[arm].to_numpy();ax.scatter(np.full(len(v),x),v,color=['#3266a8','#d97732'][x],s=35)
            ax.errorbar(x,v.mean(),yerr=v.std(ddof=1),fmt='s',color='black',capsize=5,markersize=5)
        ax.set_xticks([0,1],['Kd/Ki + inactive','All + inactive']);ax.set_title(title);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched scaffold test: 3 seeds, mean +/- SD\nMeasured-pair evaluation; not prospective SPR hit rates',fontsize=12)
    fig.savefig(OUT/'ENDPOINT_COMPARISON.png',dpi=160);plt.close(fig)
    names={'kdki_inactive':'A：Kd/Ki＋明确失活','all_inactive':'B：全部端点＋明确失活'}
    panels={'AFFINITY_KD_KI':'Kd/Ki','ACTIVITY_IC50':'IC50','ACTIVITY_EC50':'EC50','ALL_ENDPOINT_UNION':'全部端点合并','EXPLICIT_INACTIVE':'明确失活'}
    def number(x):return '不适用' if pd.isna(x) else f'{x:.4f}'
    lines=['# Kd/Ki 与全端点训练对照结果','', '状态：全部6次训练与共同测试已完成。生产模型、网站分数和冻结湿实验清单保持原样。', '',
        '![共同测试结果](../outputs/biomaster_endpoint_ablation_20260911/ENDPOINT_COMPARISON.png)','',
        '## 训练数据和实际曝光','', '| 组别 | 训练配对 | 阳性 | 阴性 | 明确失活配对 |','|---|---:|---:|---:|---:|']
    for r in data[data.split.eq('train')].itertuples():lines.append(f'| {names[r.arm]} | {r.pairs:,} | {r.positive:,} | {r.negative:,} | {r.explicit_inactive_pairs:,} |')
    lines+=['','明确失活包含带数值端点的失活注释，两组共享同一批105,368对。来源明确失活与任一明确阳性矛盾时，两组一致隔离；全端点组另外隔离跨端点数值正负矛盾。每对只生成一个标签，不把重复来源作为更多票数。','',
        '两组使用相同DrugCLIP＋Morgan／ESM2全局交互架构，3个种子逐一匹配随机初始化。监督网络从头训练，公共编码器冻结。每次6000步、每步1024条，完整遍历后再打乱重复；因此是相同优化预算，A组遍历次数更多。使用明确标签的类别平衡BCE，没有未知负例或旧模型监督权重；本轮未使用生产训练中的检索损失，不能把它当作完全复刻生产训练方案。','',
        '## 共同测试结果','', '以下为3个种子均值；逐种子结果、标准差和分层结果在CSV中。AP应结合阳性比例解读。','',
        '| 测试任务 | 测试配对 | 阳性比例 | 组别 | AUROC | AP | Top 1%精确率 | Brier |','|---|---:|---:|---|---:|---:|---:|---:|']
    for panel in panels:
        for r in grouped[grouped.panel.eq(panel)&grouped.scope.eq('all')].itertuples():
            lines.append(f'| {panels[panel]} | {r.pairs:,} | {r.positive/r.pairs:.1%} | {names[r.arm]} | {number(r.auroc_mean)} | {number(r.ap_mean)} | {number(r.top1pct_precision_mean)} | {number(r.brier_mean)} |')
    ci=summary['primary_paired_bootstrap'];primary=grouped[grouped.panel.eq('AFFINITY_KD_KI')&grouped.scope.eq('all')].set_index('arm')
    delta=primary.loc['all_inactive','ap_mean']-primary.loc['kdki_inactive','ap_mean']
    if ci['ap_delta_ci95'][0]>0 and delta>0:verdict='本次共同Kd/Ki测试支持全端点训练提高AP；仍需结合靶点内排序和前瞻实验验证。'
    elif ci['ap_delta_ci95'][1]<0 and delta<0:verdict='本次共同Kd/Ki测试支持Kd/Ki＋明确失活获得更高AP；全端点混合在直接亲和力任务上出现负迁移。'
    else:verdict='本次共同Kd/Ki测试尚不足以证明任一数据方案的AP稳定更优，不据此自动替换生产模型。'
    lines+=['','## 主要结论与不确定性','',verdict,'',
        f"全端点组减Kd/Ki组：逐种子AP均值差为{delta:+.4f}。先对3个种子的校准预测取均值，再按化学骨架配对bootstrap，AP差为{ci['ap_delta']:+.4f}，95%区间[{ci['ap_delta_ci95'][0]:+.4f}, {ci['ap_delta_ci95'][1]:+.4f}]；AUROC差为{ci['auroc_delta']:+.4f}，95%区间[{ci['auroc_delta_ci95'][0]:+.4f}, {ci['auroc_delta_ci95'][1]:+.4f}]。这是对回顾性测试骨架的抽样不确定性，不包含所有训练/数据选择不确定性，也不是已部署的集成模型。",'',
        '## 明确失活的误报','', '概率校准与F1阈值仅由共同Kd/Ki验证集拟合。下面的误报率使用该阈值，反映阈值迁移到明确失活测试的表现。明确失活测试只有负类，不计算AUROC/AP。','',
        '| 组别 | 明确失活测试对 | 误报率 |','|---|---:|---:|']
    for r in grouped[grouped.panel.eq('EXPLICIT_INACTIVE')&grouped.scope.eq('all')].itertuples():lines.append(f'| {names[r.arm]} | {r.pairs:,} | {r.false_positive_rate_mean:.2%} |')
    lines+=['','两组的Kd/Ki召回率约为93%，因此该F1阈值偏向保留阳性；低于另一组的误报率不表示已达到实验采购所需的高精确率。加入明确失活证据也没有消除误报。本实验两组都加入了失活，不能用它证明“加入失活”相对“不加入”的因果增益。','',
        '## 更严格的Kd/Ki分层','', '| 分层 | 测试配对 | 组别 | AUROC | AP |','|---|---:|---|---:|---:|']
    scopes={'document_disjoint':'与训练/验证无共享文献','targets_both_arms':'两组均有该靶点监督','targets_only_all':'仅全端点组有该靶点监督','targets_neither':'两组均无该靶点监督'}
    for scope,label in scopes.items():
        for r in grouped[grouped.panel.eq('AFFINITY_KD_KI')&grouped.scope.eq(scope)].itertuples():
            lines.append(f'| {label} | {r.pairs:,} | {names[r.arm]} | {number(r.auroc_mean)} | {number(r.ap_mean)} |')
    for c in bootstrap:
        if c['panel']=='AFFINITY_KD_KI' and c['scope']=='document_disjoint':
            lines+=['',f"文献隔离子集仅{c['pairs']:,}对，全端点减Kd/Ki的AP差为{c['ap_delta']:+.4f}，95%区间[{c['ap_delta_ci95'][0]:+.4f}, {c['ap_delta_ci95'][1]:+.4f}]。若区间跨零，不能把主测试上的优势升级为已证实的文献外泛化优势。"]
    novel=grouped[grouped.panel.eq('AFFINITY_KD_KI')&grouped.scope.eq('targets_neither')]
    if not novel.empty and novel.auroc_mean.max()<.6:
        lines+=['','两组均未监督过的靶点子集区分能力接近随机。给全新靶点生成分数，不代表已证明这些分数可用于高可靠实验筛选。']
    lines+=['','## 查询内排序','', '以下仅纳入同一查询至少10个已测配对、且同时有正负标签的查询；候选池仍不是全未知空间。','',
        '| Kd/Ki排序方向 | 组别 | 查询数 | 宏平均AP | 宏平均AUROC |','|---|---|---:|---:|---:|']
    for (arm,direction),g in qm[qm.panel.eq('AFFINITY_KD_KI')].groupby(['arm','direction']):
        lines.append(f"| {'同靶点内排分子' if direction=='within_target' else '同分子内排靶点'} | {names[arm]} | {int(g.queries.iloc[0])} | {g.macro_ap.mean():.4f} | {g.macro_auroc.mean():.4f} |")
    lines+=['','## 评价边界','',
        '- 数据按既有化学骨架/连接身份划分，冻结实验配对、对照和此前留出骨架不进入训练。',
        '- 主测试可能共享论文/专利；`document_disjoint`另报与训练/验证均不共享引用的子集，不能省略其样本量。',
        '- `targets_both_arms`与`targets_only_all`等分层说明两组靶点监督覆盖差异；这不是蛋白同源簇冷启动验证。',
        '- Top百分位精确率来自有测量的测试候选池，不是完整未知药物×靶点空间的SPR命中率。',
        '- 公共DrugCLIP/ESM2预训练成员重叠未经认证。IC50、EC50与失活注释不是Kd的物理等价量。',
        '- 全端点混合是本次被检验的数据方案；不代表已经完成更复杂的多任务解耦模型优化。',
        '- 本轮评估两种数据方案之间的差异，没有在同一严格未见测试集上直接比较旧生产模型，不能宣称已超过旧模型。','',
        '## 文件','', '[逐种子测试指标](../outputs/biomaster_endpoint_ablation_20260911/TEST_METRICS.csv) · [均值与标准差](../outputs/biomaster_endpoint_ablation_20260911/RESULT_COMPARISON.csv) · [查询内排序](../outputs/biomaster_endpoint_ablation_20260911/QUERY_MACRO_SUMMARY.csv) · [配对置信区间](../outputs/biomaster_endpoint_ablation_20260911/PAIRED_SCAFFOLD_BOOTSTRAP.json) · [冻结协议](../outputs/biomaster_endpoint_ablation_20260911/PROTOCOL.json)','',
        '逐对预测在本地 `outputs/biomaster_endpoint_ablation_20260911/TEST_PREDICTIONS.parquet`，每次研究权重在相应组别/种子目录的 `model.pt`。']
    (ROOT/'docs/BIOMASTER_ENDPOINT_ABLATION_20260911_ZH.md').write_text('\n'.join(lines)+'\n')
    write_json(OUT/'REPORT_DECISION.json',dict(primary_metric='AP on shared Kd/Ki test',verdict=verdict,per_seed_mean_ap_delta=float(delta),primary_bootstrap=ci,production_promoted=False))
    print(verdict,flush=True)

if __name__=='__main__':main()
