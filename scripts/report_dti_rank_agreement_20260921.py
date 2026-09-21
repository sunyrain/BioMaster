#!/usr/bin/env python3
"""Publish descriptive ranks with matched coverage; never infer correctness."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_rank_agreement_20260921'
MODELS = ['ConPLex', 'DrugCLIP', 'DTIAM_A', 'Nesso-1', 'ProbeMatchDTI', 'DTBind_occurrence']
LABELS = ['ConPLex', 'DrugCLIP', 'DTIAM A*', 'Nesso-1', 'ProbeMatch', 'DTBind occ.']


def main():
    # Narrative numbers refer to this frozen snapshot, not a later live database.
    assert hashlib.sha256((OUT / 'MANIFEST.json').read_bytes()).hexdigest() == '35e12506c6dc679c27ed67e644331672c367f79e94ee2b034d8d933ab8b9a8b9'
    stats = pd.read_csv(OUT / 'RANK_AGREEMENT_SUMMARY.csv')
    coverage = pd.read_csv(OUT / 'COVERAGE.csv')
    detail = pd.read_csv(OUT / 'PER_QUERY_RANK_AGREEMENT.csv.gz')
    for k in [5, 10, 20]:
        null = detail[f'top{k}_random_expected_overlap']
        # N=K makes all candidates selected, leaving no head-ranking information.
        detail[f'top{k}_chance_adjusted'] = ((detail[f'top{k}_tie_expected_overlap'] - null) / (k-null)).where(detail.candidates.gt(k))
    group_keys = ['matrix', 'view', 'direction', 'model_a', 'model_b']
    adjusted = detail.groupby(group_keys, as_index=False).agg(
        queries=('query_id', 'size'), valid_correlations=('spearman', 'count'),
        spearman_q25=('spearman', lambda x: x.quantile(.25)),
        spearman_q75=('spearman', lambda x: x.quantile(.75)),
        mean_top5_chance_adjusted=('top5_chance_adjusted', 'mean'),
        mean_top10_chance_adjusted=('top10_chance_adjusted', 'mean'),
        mean_top20_chance_adjusted=('top20_chance_adjusted', 'mean'),
        informative_top20_queries=('top20_chance_adjusted', 'count'))
    adjusted.to_csv(OUT / 'CHANCE_ADJUSTED_TOPK_SUMMARY.csv', index=False, encoding='utf-8-sig')
    primary = stats.loc[stats.matrix.eq('PRIMARY_720X96')]
    common = primary.loc[primary.view.eq('ALL_SIX_COMMON')]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    for col, direction in enumerate(['target_to_drug', 'drug_to_target']):
        rows = common.loc[common.direction.eq(direction)]
        for r, (metric, title, low, high, cmap) in enumerate([
            ('mean_spearman', 'Mean Spearman', -1, 1, 'RdBu_r'),
            ('top10_overlap_ratio_to_random', 'Top10 overlap / random expectation', 0, 6, 'viridis'),
        ]):
            matrix = np.full((6, 6), np.nan)
            for row in rows.itertuples():
                i, j = MODELS.index(row.model_a), MODELS.index(row.model_b)
                matrix[i, j] = matrix[j, i] = getattr(row, metric)
            ax = axes[r, col]
            im = ax.imshow(matrix, vmin=low, vmax=high, cmap=cmap)
            ax.set_xticks(range(6), LABELS, rotation=35, ha='right')
            ax.set_yticks(range(6), LABELS)
            label = 'Rank drugs per target' if col == 0 else 'Rank targets per drug'
            denominator = '23 queries; 719 candidates each' if col == 0 else '719 queries; 23-29 candidates/query'
            ax.set_title(f'{label}\n{title}\n{denominator}', fontsize=11)
            for i in range(6):
                for j in range(6):
                    if np.isfinite(matrix[i, j]):
                        value = matrix[i, j]
                        white = abs(value) > .55 if r == 0 else value < 3
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if white else 'black', fontsize=9)
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle('Six current channels: common-coverage rank agreement\n720 x 96 requested matrix; *DTIAM uses local A, not official task weights\nDescriptive snapshot; neither agreement nor disagreement establishes correctness', fontsize=12)
    fig.savefig(OUT / 'COMMON_COVERAGE_RANK_AGREEMENT.png', dpi=180)
    fig.savefig(OUT / 'COMMON_COVERAGE_RANK_AGREEMENT.pdf')
    plt.close(fig)
    lines = ['# 六个现有通道：96靶点补充切片的排序分歧结果', '',
        '**当前主范围已更新为720×745；本页保留96靶点切片的已有统计，不代表745范围完成。** 见[当前范围和来源](BIOMASTER_DTI_RANKING_SCOPE_20260921_ZH.md)。', '',
        '工作日期：2026-09-21。此报告只描述排序，不使用Davis或实测标签，不判断哪个模型正确。本页统计720药物×96靶点的补充切片；旧720×384统计另表提供，两者均保留原分母。DTIAM为已授权本地A，其余五个为现有官方权重通道；新增四个主模型尚未纳入，不能称为最终十模型结果。', '',
        '## 覆盖先于相关性', '', '| 通道 | 本页切片已评分/69,120 | 覆盖率 |', '|---|---:|---:|']
    for row in coverage.loc[coverage.matrix.eq('PRIMARY_720X96')].itertuples():
        lines.append(f'| {row.model} | {row.scored_pairs:,} | {row.coverage_percent:.2f}% |')
    lines += ['', '六通道共同覆盖视图中，靶点→药物方向只有23个查询达到至少20个共同候选，每个查询719个药物；药物→靶点方向有719个查询，每个23–29个靶点。这个交集不是完整96靶点矩阵，不具有相同的家族代表性。两两交集可以利用更多分数，但分母随模型对变化，不适合直接当成统一模型排行榜。', '',
        '## 三个可进一步研究的现象', '',
        '**整体相关性和头部偏好可以不同。** 两两覆盖中，ConPLex与DTIAM A在96个靶点、每靶点720个药物上，平均Spearman为0.061，平均Top10重合1.063个；独立随机Top10期望重合0.139个。整体秩相关较低，但头部仍存在高于随机参照的重合。这是描述性效应量，不是显著性检验，也不说明共同推荐就正确。', '',
        '**模型关系随推荐方向改变。** 六通道共同候选视图中，ProbeMatchDTI与DTBind occurrence在药物→靶点方向平均Spearman为0.395，在靶点→药物方向为0.125。Nesso-1与DTIAM A分别为0.190和−0.037。两个方向的查询及候选空间不同，这些数字提示方向性规律，尚不能直接归因于网络架构。', '',
        '**覆盖会影响表面结论。** ConPLex与DTIAM A在六通道交集的靶点→药物平均Top10重合0.565个，在其自身两两交集为1.063个；这是所含靶点范围改变后的统计差异，不是两个模型重新推理后发生了变化。需要同时给出共同交集和两两交集。', '',
        '## 共同候选下的图示', '',
        '![六通道共同覆盖的双向秩相关与Top10随机参照比](../outputs/dti_rank_agreement_20260921/COMMON_COVERAGE_RANK_AGREEMENT.png)', '',
        '图中上排为逐查询Spearman的均值；下排为平均Top10重合数/相同分母的随机期望。TopK边界同分按独立随机打破并列的期望交集处理。相关性遇常数输出记为未定义，分母在CSV中列出；药物→靶点719个查询中，涉及ConPLex的相关性有718个有效值。图中自身比较留空。', '',
        '随机倍数的上限N/K仍随候选数量改变，因此不能仅凭两个方向的倍数大小宣布哪个方向更一致。补充CSV提供逐查询机会校正重合度：(实际交集−K²/N)/(K−K²/N)，再对查询求均值；0是随机参照，1是无并列且TopK完全相同。N=K时该指标未定义，单列有效查询数。并列边界含义与主表一致。', '',
        '这张图展示描述性关系，不把模型分簇当作学习机制相同的证据。下一步补齐新增模型的原生推理核验与同矩阵评分，再完成家族、输入长度、并列分数及候选池的分层分析。若做置信区间或显著性判断，需要考虑查询间同源/化学相关性；本报告没有给出未经校正的显著性结论。', '',
        '## 可复查文件', '',
        '- [覆盖表](../outputs/dti_rank_agreement_20260921/COVERAGE.csv)',
        '- [全部双向比较汇总](../outputs/dti_rank_agreement_20260921/RANK_AGREEMENT_SUMMARY.csv)',
        '- [按家族汇总](../outputs/dti_rank_agreement_20260921/FAMILY_RANK_AGREEMENT_SUMMARY.csv)',
        '- [机会校正TopK与相关性四分位数](../outputs/dti_rank_agreement_20260921/CHANCE_ADJUSTED_TOPK_SUMMARY.csv)',
        '- [逐查询结果](../outputs/dti_rank_agreement_20260921/PER_QUERY_RANK_AGREEMENT.csv.gz)',
        '- [固定分数快照与校验信息](../outputs/dti_rank_agreement_20260921/MANIFEST.json)',
        '- [可导出PDF](../outputs/dti_rank_agreement_20260921/COMMON_COVERAGE_RANK_AGREEMENT.pdf)',
        '- [当前研究范围](BIOMASTER_DTI_RANKING_SCOPE_20260921_ZH.md)',
    ]
    (ROOT / 'docs/BIOMASTER_DTI_RANK_AGREEMENT_PRELIMINARY_20260921_ZH.md').write_text('\n'.join(lines) + '\n')
    # This is a separate presentation layer; never rewrite the analysis snapshot manifest.
    artifacts = [OUT / name for name in ['COMMON_COVERAGE_RANK_AGREEMENT.png', 'COMMON_COVERAGE_RANK_AGREEMENT.pdf', 'CHANCE_ADJUSTED_TOPK_SUMMARY.csv']]
    artifacts.append(ROOT / 'docs/BIOMASTER_DTI_RANK_AGREEMENT_PRELIMINARY_20260921_ZH.md')
    record = {'source_manifest_sha256': hashlib.sha256((OUT / 'MANIFEST.json').read_bytes()).hexdigest(),
        'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
    (OUT / 'PRESENTATION_MANIFEST.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'status': 'PASS', 'analysis_rows': len(stats), 'figure': str(OUT / 'COMMON_COVERAGE_RANK_AGREEMENT.png')}, indent=2))


if __name__ == '__main__':
    main()
