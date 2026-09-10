#!/usr/bin/env python3
"""Summarize sampled BCE and combined-ranking logs without changing training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
DEST = OUT / 'loss_diagnostics/20260907'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source = OUT / 'program_logs/train_biomaster_pocket_precision_--cutoff.log'
    raw = source.read_bytes()
    rows = []
    for line in raw.decode().splitlines():
        try:
            value = json.loads(line)
            if all(k in value for k in ['epoch', 'updates', 'loss', 'seconds']):
                rows.append(value)
        except json.JSONDecodeError:
            pass
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.loss).all():
        raise ValueError('nonempty finite training loss observations required')
    config = json.loads((ROOT / 'configs/biomaster_pocket_precision_20260906.json').read_text())['training']
    every = config['retrieval_every_updates']
    frame['kind'] = np.where(frame.updates.mod(every).eq(0), 'bce_plus_weighted_ranking', 'bce_only')
    groups = []
    windows = [(0,1000), (1000,3000), (3000,6000), (6000,9138), (8138,9138), (9138,int(frame.updates.max()))]
    for low, high in windows:
        for kind in ['bce_only', 'bce_plus_weighted_ranking']:
            part = frame[frame.updates.gt(low) & frame.updates.le(high) & frame.kind.eq(kind)]
            if len(part):
                groups.append(dict(lower_exclusive=low, upper_inclusive=high, kind=kind, observations=len(part),
                                   mean=float(part.loss.mean()), median=float(part.loss.median()),
                                   q10=float(part.loss.quantile(.1)), q90=float(part.loss.quantile(.9))))
    (DEST / 'TRAINING_LOG_SNAPSHOT.jsonl').write_bytes(raw)
    frame.to_csv(DEST / 'SAMPLED_LOSSES.csv', index=False)
    now = datetime.now(timezone.utc).isoformat()
    result = dict(status='COMPLETE_READ_ONLY_LOG_SUMMARY', utc=now, last_logged_update=int(frame.updates.max()),
                  source=str(source), snapshot_sha256=hashlib.sha256(raw).hexdigest(), windows=groups,
                  interpretation='One minibatch observation every 20 updates, not full-epoch or fixed-panel loss. Ranking steps contain combined BCE and weighted query loss; pure query loss cannot be recovered.',
                  training_modified=False, used_for_selection=False)
    (DEST / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), constrained_layout=True)
    for ax, kind, count, title, color in [
        (axes[0], 'bce_only', 40, 'Classification-only steps', '#1769a4'),
        (axes[1], 'bce_plus_weighted_ranking', 10, 'BCE + weighted bidirectional ranking', '#b8541a')]:
        part = frame[frame.kind.eq(kind)]
        ax.scatter(part.updates, part.loss, s=10, alpha=.25, color=color, label='Logged minibatch loss')
        ax.plot(part.updates, part.loss.rolling(count, min_periods=count).mean(), color=color,
                linewidth=2, label=f'Rolling mean ({count} logged observations)')
        ax.axvline(9138, color='gray', linestyle='--', linewidth=1, label='First downstream epoch ends')
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Loss (different objectives across panels)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
    fig.suptitle('Sampled training losses; different minibatches/queries, not validation losses', fontsize=11)
    for suffix in ['png','svg']:
        fig.savefig(DEST / f'LOSS_CURVES.{suffix}', dpi=160)
    plt.close(fig)
    first = next(x for x in groups if x['lower_exclusive']==0 and x['kind']=='bce_only')
    last = next(x for x in groups if x['lower_exclusive']==8138 and x['kind']=='bce_only')
    lines = ['# 下游训练的“第一轮”与loss变化', '', f'日志快照时间：{now}；记录至第{int(frame.updates.max()):,}次更新。', '',
             '“第一轮”指结构预训练完成后，下游关系训练的第1个epoch：遍历292,408条关系训练记录一次。有效batch为32，因此每轮9,138次优化器更新；另在每100次更新加入双向全候选排序监督。日志epoch从0开始，所以epoch=0是第一轮，epoch=1是第二轮。', '',
             '全局父模型已有训练好的权重，局部模型也已完成6轮真实结构预训练；这里的“第一轮”只描述接入后的关系微调阶段。', '',
             '## 分类loss的变化', '',
             '| 更新区间 | 纯分类loss日志数 | 均值 | 中位数 |', '|---|---:|---:|---:|']
    for row in groups:
        if row['kind']=='bce_only':
            lines.append(f"| {row['lower_exclusive']+1}–{row['upper_inclusive']} | {row['observations']} | {row['mean']:.6f} | {row['median']:.6f} |")
    lines += ['', f"第一轮最初1,000步与最后1,000步各有40条纯分类日志，均值由{first['mean']:.6f}降到{last['mean']:.6f}，下降{1-last['mean']/first['mean']:.1%}；中位数由{first['median']:.6f}降到{last['median']:.6f}。", '',
              '这些是每20步保存一次的当前训练batch损失，不是全部训练batch的精确epoch均值，也不是在固定验证样本上计算的loss。第一轮中后段有反复，不能概括为单调下降；第二轮刚开始，不能拿少量日志直接证明进一步收敛。', '',
              '## 周期尖峰的含义', '',
              '普通更新的日志仅为分类BCE；每100次更新的日志还包括两方向排序项，当前权重为：', '',
              '`loss = BCE + 0.25 × 老药→靶点排序loss + 0.25 × 靶点→老药排序loss`', '',
              '因此普通步骤的0.02与排序步骤的0.7或1.0不能直接比较。每次排序监督随机选取不同查询，难度及已知阳性数也不同。当前训练器没有分别保存排序步的BCE及两个排序分量，无法从混合总值恢复独立的排序loss曲线。', '',
              '![分类步骤与包含排序监督的步骤分别显示](../outputs/biomaster_pocket_precision_20260906/loss_diagnostics/20260907/LOSS_CURVES.png)', '',
              '**当前证据：分类训练loss下降，但排序loss是否持续改善还不能确认。**开发集双向AP仅有小幅变化，说明不能把分类拟合改善直接当作检索能力提升。', '',
              '[完整日志统计](../outputs/biomaster_pocket_precision_20260906/loss_diagnostics/20260907/RESULT.json) · [逐条采样loss](../outputs/biomaster_pocket_precision_20260906/loss_diagnostics/20260907/SAMPLED_LOSSES.csv) · [排序验证结果](BIOMASTER_POCKET_FIRST_RANKING_20260907_ZH.md)', '']
    (ROOT / 'docs/BIOMASTER_POCKET_LOSS_PROGRESS_20260907_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
