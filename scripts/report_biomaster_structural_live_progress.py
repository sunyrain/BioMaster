#!/usr/bin/env python3
"""Publish read-only training progress; never load or change model checkpoints."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906'
TRAIN = OUT / 'structural_pretraining/cutoff_2020/seed_20260921'
DOWNSTREAM = OUT / 'training/structurally_pretrained/cutoff_2020/seed_20260921'
LIVE = OUT / 'live_progress'
REPORT = ROOT / 'docs/BIOMASTER_STRUCTURAL_LIVE_PROGRESS_20260906_ZH.md'
SCRIPTS = {'run_biomaster_structural_followup.py', 'pretrain_biomaster_structural_interaction.py',
           'train_biomaster_pocket_precision.py', 'follow_biomaster_structural_contact_results.py',
           'diagnose_biomaster_structural_contact_learning.py', 'audit_biomaster_pocket_ranking_fp32.py'}
STAGES = {'STRUCTURAL_PRETRAINING_RUNNING': '结构预训练进行中',
          'DOWNSTREAM_2020_DEVELOPMENT_RUNNING': '≤2020关系训练与2021–2022开发验证进行中',
          'ONE_DEVELOPMENT_FIT_COMPLETE_NOT_SELECTED': '单种子开发训练完成，尚未选型',
          'FAILED_REQUIRES_REVIEW': '任务失败，需检查原因'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {} if default is None else default


def write(path, content):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(content)
    temp.replace(path)


def training_events(path, stage=None):
    events = []
    if path.exists():
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - 32768))
            for line in handle.read().decode(errors='replace').splitlines():
                try:
                    value = json.loads(line)
                    if 'updates' in value and 'seconds' in value and (stage is None or value.get('stage') == stage):
                        events.append(value)
                except json.JSONDecodeError:
                    pass
    return events


def collect():
    program = read(OUT / 'PROGRAM_STATUS.json')
    structural = read(TRAIN / 'STATUS.json')
    downstream = read(DOWNSTREAM / 'STATUS.json')
    downstream_config = read(ROOT / 'configs/biomaster_pocket_precision_20260906.json')['training']
    protocol = read(ROOT / 'configs/biomaster_structural_pretraining_20260906.json')
    processes = []
    for proc in psutil.process_iter(['pid', 'cmdline', 'status', 'create_time']):
        args = proc.info['cmdline'] or []
        if len(args) > 1 and Path(args[1]).name in SCRIPTS and proc.info['status'] != psutil.STATUS_ZOMBIE:
            processes.append(dict(pid=proc.pid, script=Path(args[1]).name,
                                  started_utc=datetime.fromtimestamp(proc.info['create_time'], timezone.utc).isoformat()))
    log = OUT / 'program_logs/pretrain_biomaster_structural_interaction_run.log'
    events = training_events(log, 'training')
    downstream_events = training_events(OUT / 'program_logs/train_biomaster_pocket_precision_--cutoff.log')
    downstream_seconds_per_update = None
    if len(downstream_events) > 1:
        first, last = downstream_events[max(0, len(downstream_events)-30)], downstream_events[-1]
        du = last['updates'] - first['updates']
        if du > 0 and last['seconds'] > first['seconds']:
            downstream_seconds_per_update = (last['seconds'] - first['seconds']) / du
    seconds_per_update = None
    if len(events) > 1:
        first, last = events[max(0, len(events)-30)], events[-1]
        du = last['updates'] - first['updates']
        if du > 0 and last['seconds'] > first['seconds']:
            seconds_per_update = (last['seconds'] - first['seconds']) / du
    max_updates = math.ceil(structural.get('train_complexes', 0) / protocol['effective_batch_size']) * protocol['max_epochs']
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        gpu_sample = gpu.stdout.strip() if gpu.returncode == 0 else '读取失败'
    except (OSError, subprocess.TimeoutExpired):
        gpu_sample = '不可用'
    now = datetime.now(timezone.utc)
    return dict(utc=now.isoformat(), beijing=now.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S'),
                reporter_pid=os.getpid(), program={k:v for k,v in program.items() if k != 'source_identities'},
                structural={k:v for k,v in structural.items() if k != 'identity'},
                downstream={k:v for k,v in downstream.items() if k != 'source_identity'},
                downstream_log_latest=downstream_events[-1] if downstream_events else {},
                downstream_seconds_per_update=downstream_seconds_per_update,
                downstream_updates_per_epoch=math.ceil(downstream.get('training_rows', 0) / downstream_config['effective_batch_size']),
                structural_history=read(TRAIN / 'HISTORY.json', []),
                downstream_history=read(DOWNSTREAM / 'HISTORY.json', []),
                parent_precision_audit=read(OUT / 'downstream_diagnostics/parent_precision_20260907/RESULT.json'),
                fp32_audit_status=read(OUT / 'downstream_diagnostics/epoch1_fp32_20260907/STATUS.json'),
                fp32_audit_result=read(OUT / 'downstream_diagnostics/epoch1_fp32_20260907/RESULT.json'),
                system_audit=read(OUT / 'system_audit_20260907/RESULT.json'),
                final_contact_diagnostic=read(OUT / 'contact_diagnostics/completed_pretraining/RESULT.json'),
                diagnostics={k:v for k,v in read(OUT / 'contact_diagnostics/STATUS.json').items() if k != 'diagnostic_script'},
                processes=processes, max_epochs=protocol['max_epochs'], max_updates=max_updates,
                recent_seconds_per_update=seconds_per_update, gpu_utilization_used_total=gpu_sample,
                structural_status_age_seconds=time.time()-(TRAIN / 'STATUS.json').stat().st_mtime)


def plot(history):
    if not history:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [r['updates'] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot(x, [r['metrics']['macro_contact_ap'] for r in history], 'o-', markersize=3)
    axes[0].axhline(history[0]['metrics']['macro_contact_prevalence'], color='gray', linestyle='--', label='Constant-score AP')
    axes[0].set(title='Structural contact AP (higher is better)', ylabel='Complex-macro AP', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [r['metrics']['macro_distance_mae_capped32_A'] for r in history], 'o-', markersize=3, color='#bb5522')
    axes[1].set(title='Distance MAE (lower is better)', ylabel='Angstrom; distances capped at 32 A')
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
        ax.grid(alpha=.2)
    fig.suptitle('1,223 cluster-held-out complexes; internal validation, not DTI ranking', fontsize=10)
    for suffix in ['png', 'svg']:
        dest = LIVE / f'STRUCTURAL_CURVE.{suffix}'
        temp = dest.with_name(f'.curve.{os.getpid()}.{suffix}')
        fig.savefig(temp, dpi=160)
        temp.replace(dest)
    plt.close(fig)


def render(s):
    train, history = s['structural'], s['structural_history']
    phase = s['program'].get('status', 'UNKNOWN')
    lines = ['# 结构交互训练实时进展', '',
             f"采样时间：**{s['beijing']} 北京时间**（{s['utc']}）。监控每30秒刷新本文件；验证指标只在训练器完成验证后更新。", '',
             f"当前阶段：**{STAGES.get(phase, phase)}**。", '',
             (f"结构预训练已完成：{train.get('epoch', 0)}轮、{train.get('updates', 0):,}次更新，按既定规则早停。"
              if train.get('status') == 'STRUCTURAL_PRETRAINING_COMPLETE'
              else f"结构训练：{train.get('updates', 0):,}/{s['max_updates']:,}次计划上限更新；最多{s['max_epochs']}轮，可能提前早停。"), '',
             f"数据：{train.get('train_complexes', 0):,}个训练复合物、{train.get('validation_complexes', 0):,}个按簇留出的内部验证复合物。", '']
    audit = s['system_audit']
    if audit:
        retention = audit['structural_retention']
        ap,mae = retention['native_contact_ap'],retention['native_distance_mae_capped32_A']
        lines += ['**已完成首轮训练衔接审查：结构预训练能力在关系微调后明显退化，统一FP32排序AP增益的两个95%区间均跨零。**', '',
                  '| 同一结构验证集的能力保持 | 结构预训练后 | 关系微调第1轮后 |', '|---|---:|---:|',
                  f"| 真实口袋内接触AP | {ap['before']:.6f} | {ap['after']:.6f} |",
                  f"| 真实口袋内距离MAE | {mae['before']:.4f}Å | {mae['after']:.4f}Å |", '',
                  '这项诊断固定在第1轮检查点；不能把下方结构预训练曲线的成绩直接当作之后关系模型仍保留的能力。', '',
                  '[完整数据、训练及实际架构审查](BIOMASTER_DATA_TRAINING_ARCHITECTURE_AUDIT_20260907_ZH.md)', '']
    if train.get('status') == 'STRUCTURAL_PRETRAINING_RUNNING':
        lines += [f"当前为第{train.get('epoch', 0)+1}轮；状态文件距采样{round(s['structural_status_age_seconds'])}秒。", '']
        if s['recent_seconds_per_update']:
            remaining = max(0, s['max_updates'] - train.get('updates', 0)) * s['recent_seconds_per_update'] / 60
            lines += [f"近段平均{s['recent_seconds_per_update']:.2f}秒/更新；若跑满计划，结构阶段约还需{remaining:.0f}分钟。此估计包含近段验证开销，可能早停，且不含下游训练时间。", '']
    if history:
        best = max(history, key=lambda r:r['metrics']['macro_contact_ap'])
        latest = history[-1]
        lines += ['| 内部结构验证 | 更新次数 | 接触AP ↑ | 距离MAE ↓ |', '|---|---:|---:|---:|']
        for label, item in [('训练前', history[0]), ('接触AP最佳检查点', best), ('最近检查点', latest)]:
            m = item['metrics']
            lines.append(f"| {label} | {item['updates']} | {m['macro_contact_ap']:.6f} | {m['macro_distance_mae_capped32_A']:.4f}Å |")
        lines += ['', '距离为原子到残基的最小重原子距离，采用32Å截断；不是结合姿态RMSD。', '',
                  '![内部结构验证曲线](../outputs/biomaster_pocket_precision_20260906/live_progress/STRUCTURAL_CURVE.png)', '']
    lines += ['**以上接触AP不是老药→靶点或靶点→老药的排序AP。**', '']
    diagnostic = s['final_contact_diagnostic']
    if diagnostic.get('status') == 'COMPLETE_READ_ONLY_DIAGNOSTIC':
        m = diagnostic['means']
        lines += ['结构预训练最终检查点的独立诊断（同一1,223个体系，仅已知真实口袋内部）：', '',
                  '| 接触评分方式 | AP ↑ |', '|---|---:|',
                  f"| 完整原子—残基预测 | {m['native_contact_ap']:.6f} |",
                  f"| 打乱预测与原子的对应关系 | {m['native_atom_permuted_score_ap']:.6f} |",
                  f"| 同一残基对所有原子使用平均分 | {m['native_residue_only_score_ap']:.6f} |",
                  f"| 常数评分基线 | {m['native_contact_prevalence']:.6f} |", '',
                  f"真实口袋内距离MAE为{m['native_distance_mae_capped32_A']:.4f}Å（32Å截断），与上表包含远端区域的MAE口径不同。", '',
                  '两项控制的AP差异按162个簇重采样，95%区间均高于零。这支持原子级区分信息，尚不能替代完整几何消融或证明老药排序收益。', '',
                  '[最终结构诊断及区间](../outputs/biomaster_pocket_precision_20260906/contact_diagnostics/completed_pretraining/RESULT.json)', '']
    event = s['downstream_log_latest']
    logged_updates = max(s['downstream'].get('updates', 0), event.get('updates', 0))
    per_epoch = s['downstream_updates_per_epoch']
    if per_epoch and s['downstream'].get('status') == 'TRAINING':
        epoch = max(s['downstream'].get('epoch', 0), event.get('epoch', 0))
        epoch_position = min(per_epoch, max(0, logged_updates - epoch * per_epoch))
        lines += [f"下游关系训练第{epoch+1}轮：日志记录到{logged_updates:,}次更新，本轮约{epoch_position/per_epoch:.1%}；状态检查点已保存到{s['downstream'].get('updates', 0):,}次。", '']
        rate = s['downstream_seconds_per_update']
        if rate:
            hours = max(0, (per_epoch-epoch_position)*rate/3600)
            lines += [f"每轮{per_epoch:,}次更新，目前平均{rate:.2f}秒/更新。按当前速度，本轮训练约还需{hours:.1f}小时，之后才进行完整开发验证；不含验证本身耗时。", '']
    if s['downstream_history']:
        lines += ['| 下游开发验证轮次 | 老药→靶点AP | 靶点→老药AP | 综合选型分 |', '|---|---:|---:|---:|']
        for item in s['downstream_history'][-8:]:
            m = item['metrics']
            lines.append(f"| {item['epoch']} | {m['d2t']['macro_ap']:.6f} | {m['t2d']['macro_ap']:.6f} | {item['composite']:.6f} |")
        lines += ['', '上述为≤2020训练、2021–2022开发验证的单种子结果；完整多种子消融和后期回归尚未完成。', '']
        audit = s['parent_precision_audit']
        if audit:
            parent = audit['parent_metrics']; new = audit['downstream_first_epoch']['metrics']
            lines += ['同设置BF16核查：旧模型使用新训练器相同父模型路径、batch=1、TF32关闭；以下固定比较第一轮。', '',
                      '| 指标 | 旧模型重算 | 新模型第1轮 | 差值 |', '|---|---:|---:|---:|']
            for d, label in [('d2t', '老药→靶点'), ('t2d', '靶点→老药')]:
                for key, metric in [('macro_ap', 'AP'), ('macro_recall_20', 'Recall@20')]:
                    a,b = parent[d][key],new[d][key]
                    lines.append(f'| {label} {metric} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
            lines += ['', '**首轮AP仅小幅上升，逆向Recall@5和Recall@10下降。旧权重改变评测设置后，Recall@20也明显变化，因此尚不能宣布可靠升级。**', '',
                      '[首轮结果、归档对照和数值问题](BIOMASTER_POCKET_FIRST_RANKING_20260907_ZH.md)', '']
    elif s['downstream'] or phase == 'DOWNSTREAM_2020_DEVELOPMENT_RUNNING':
        lines += ['下游关系训练已经启动；尚无完整一轮开发验证结果。', '']
    else:
        lines += ['下游老药双向排序：尚未启动。主程序将在结构预训练完成后自动衔接。', '']
    fp32 = s['fp32_audit_status']
    if fp32:
        lines += [f"冻结首轮检查点的统一FP32复核：`{fp32.get('status')}`，已处理{fp32.get('completed_pairs', 0):,}/{fp32.get('total_pairs', 41136):,}个方向内候选配对。", '']
        if fp32.get('error'):
            lines += [f"复核失败原因：`{fp32['error']}`。", '']
        result = s['fp32_audit_result']
        if result:
            lines += ['| 统一FP32 AP | 旧模型 | 新模型第1轮 | 差值95%查询重采样区间 |', '|---|---:|---:|---|']
            for d,label in [('d2t','老药→靶点'),('t2d','靶点→老药')]:
                m,c = result['metrics'][d],result['paired_query_ap_intervals'][d]
                lines.append(f"| {label} | {m['parent']['macro_ap']:.6f} | {m['new']['macro_ap']:.6f} | [{c['ci_low']:+.6f}, {c['ci_high']:+.6f}] |")
            lines += ['', '这是单开发窗口、单种子的固定首轮诊断，未参与选型。', '']
        lines += ['[FP32复核状态](../outputs/biomaster_pocket_precision_20260906/downstream_diagnostics/epoch1_fp32_20260907/STATUS.json)', '']
    lines += [f"独立接触诊断：`{s['diagnostics'].get('status', 'UNKNOWN')}`；当前阶段`{s['diagnostics'].get('phase', '—')}`。", '',
              f"GPU瞬时采样（利用率%、已用MiB、总MiB）：`{s['gpu_utilization_used_total']}`。", '',
              '| 运行进程 | PID |', '|---|---:|']
    for proc in s['processes']:
        lines.append(f"| {proc['script']} | {proc['pid']} |")
    if s['program'].get('error'):
        lines += ['', f"失败原因：`{s['program']['error']}`。"]
    if not s['processes'] and phase not in {'ONE_DEVELOPMENT_FIT_COMPLETE_NOT_SELECTED', 'FAILED_REQUIRES_REVIEW'}:
        lines += ['', '**未检测到训练/主管进程；上述状态可能已过期，不能据此判断仍在训练。**']
    lines += ['', '本页由只读监控自动生成，不修改训练协议、检查点或模型选型。聊天消息不会由此脚本发送。', '',
              '[实验协议与原子级诊断](BIOMASTER_STRUCTURAL_PRETRAINING_FOLLOWUP_20260906_ZH.md) · '
              '[完整实时快照](../outputs/biomaster_pocket_precision_20260906/live_progress/STATUS.json)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    LIVE.mkdir(exist_ok=True)
    lock = (LIVE / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    while True:
        try:
            s = collect()
            signature = json.dumps([s['program'].get('status'), s['structural_history'], s['downstream_history'], s['diagnostics'].get('status'), s['final_contact_diagnostic'].get('checkpoint_sha256')], sort_keys=True)
            if signature != previous:
                plot(s['structural_history'])
                with (LIVE / 'EVENTS.jsonl').open('a') as log:
                    log.write(json.dumps(s, ensure_ascii=False) + '\n')
                previous = signature
            write(LIVE / 'STATUS.json', json.dumps(s, ensure_ascii=False, indent=2) + '\n')
            write(REPORT, render(s))
            program_done = s['program'].get('status') in {'ONE_DEVELOPMENT_FIT_COMPLETE_NOT_SELECTED', 'FAILED_REQUIRES_REVIEW'}
            diagnostics_done = s['diagnostics'].get('status') in {'ALL_STRUCTURAL_CONTACT_DIAGNOSTICS_COMPLETE', 'FAILED_REQUIRES_REVIEW'}
            if args.once or (program_done and diagnostics_done):
                return
        except Exception as error:
            print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), error=repr(error))), flush=True)
            if args.once:
                raise
        time.sleep(30)


if __name__ == '__main__':
    main()
