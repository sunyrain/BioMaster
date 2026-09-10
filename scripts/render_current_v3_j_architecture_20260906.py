#!/usr/bin/env python3
"""Render an editable architecture figure from the implemented V3/J contract."""
from pathlib import Path
from html import escape
import hashlib
import json

import cairosvg
from PIL import ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/presentations/current_v3_j_architecture_20260906'
W,H=1920,1370
C={'ink':'#182D43','muted':'#576C80','line':'#A2B3C5','blue':'#326AB4',
   'bluebg':'#EDF4FD','teal':'#087E78','tealbg':'#EAF7F3','purple':'#7250AE',
   'purplebg':'#F3EEFA','amber':'#AE6F17','amberbg':'#FCF5E7','white':'#FFFFFF'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
parts=[]


def rect(x,y,w,h,fill,stroke='none',radius=16,sw=1.5):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x,y,text,size=22,color=None,bold=False,anchor='start',width=None):
    if width:
        while ImageFont.truetype(BOLD if bold else FONT,size).getlength(text)>width:
            size-=1
            if size<13:raise ValueError(f'text too long: {text}')
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}" '
                 f'fill="{color or C["ink"]}" text-anchor="{anchor}">{escape(text)}</text>')


def lines(x,y,values,size=21,step=31,color=None,width=None):
    for i,value in enumerate(values):txt(x,y+i*step,value,size,color,width=width)


def arrow(points,color=None,dash=False):
    color=color or C['blue']
    d='M '+' L '.join(f'{x} {y}' for x,y in points)
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4" '
                 f'stroke-linejoin="round" stroke-linecap="round" marker-end="url(#{color[1:]})" '
                 +(f'stroke-dasharray="7 5" ' if dash else '')+'/>' )


def port(x,y,name,color):
    parts.append(f'<circle cx="{x}" cy="{y}" r="15" fill="{C["white"]}" stroke="{color}" stroke-width="2"/>')
    txt(x,y+6,name,17,color,True,'middle')


def card(x,y,w,h,title,content,color='blue',size=21,step=30):
    rect(x,y,w,h,C[color+'bg'],C[color],12,1.2)
    txt(x+20,y+35,title,25,C[color],True,width=w-40)
    lines(x+20,y+70,content,size,step,width=w-40)


def main():
    candidate=json.loads((ROOT/'outputs/biomaster_v3_incremental_20260906/CANDIDATE.json').read_text())
    assert candidate['name']=='J_old_relation_pretrained'
    assert len(candidate['base_checkpoints'])==2 and len(candidate['adapter_checkpoints'])==3
    OUT.mkdir(parents=True,exist_ok=True)
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                 'font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">')
    parts.append('<title>BioMaster 当前已实现架构：V3 冻结底座与 J 双向增量模块</title>')
    parts.append('<desc>两底座冻结集成，三个独立训练的增量模块。输入为Morgan、蛋白预训练特征、训练期正负支持库和BerMol/完整ESM2全局向量。输出药找靶和靶找药的不同排序分数。</desc>')
    parts.append('<defs>')
    for color in [C['blue'],C['teal'],C['purple'],C['amber']]:
        parts.append(f'<marker id="{color[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>')
    parts.append('</defs>')
    rect(0,0,W,H,'#F7F9FC',radius=0)
    rect(0,0,W,10,C['teal'],radius=0)
    txt(56,68,'BioMaster 当前架构',43,bold=True)
    txt(56,108,'V3 底座 + J 双向增量模块  ·  J_old_relation_pretrained  ·  2026-09-06',23,C['muted'])
    legend=[(1150,'冻结底座（训练 J 时）','blue'),(1450,'可训练 J 参数','teal'),(1685,'固定输入 / 缓存','purple')]
    for x,label,color in legend:
        rect(x,52,16,16,C[color],radius=4)
        txt(x+25,68,label,19,C['muted'])

    for x,w,num,title,subtitle,color in [
        (40,402,'01','输入与训练期证据','分子、蛋白和可追溯实测关系','purple'),
        (470,576,'02','V3 底座 × 2','两个独立种子；此处展开单个底座','blue'),
        (1090,790,'03','J 增量模块 × 3','三个独立种子；此处展开单个模块','teal')]:
        rect(x,150,w,875,C['white'],'#DCE4EC',20)
        txt(x+22,191,num,22,C[color],True)
        txt(x+70,191,title,27,C[color],True,width=w-95)
        txt(x+22,222,subtitle,19,C['muted'],width=w-44)

    card(60,252,362,130,'药物输入',[
        'SMILES → 标准化分子',
        'Morgan 2048维  /  BerMol 768维'], 'purple',20,30)
    port(405,365,'P',C['purple'])
    card(60,414,362,165,'靶点输入',[
        '蛋白序列 + 靶点家族',
        'ProtBERT 1024 + legacy ESM2 1280',
        '完整序列 ESM2 均值 1280维'], 'purple',20,31)
    port(405,559,'P',C['purple'])
    card(60,620,362,262,'正负支持库',[
        '仅使用训练期明确实测标签',
        '按当前靶点检索 Morgan 近邻',
        '正样本 / 负样本分别 Top 16',
        'Tanimoto 相似度 + 有效掩码',
        '排除查询分子自身及其别名',
        '另提取 8 项相似度 / 可用性统计'], 'amber',20,31)
    port(405,862,'E',C['amber'])
    txt(78,925,'P：公共预训练全局特征缓存',19,C['purple'])
    txt(78,958,'E：同一支持库的直接证据统计',19,C['amber'])
    txt(78,991,'同名圆点表示相同输入来源',18,C['muted'])

    card(496,252,524,122,'模态投影与蛋白融合',[
        'Morgan → 药物向量 d：192维',
        'ProtBERT + ESM2 门控融合 → t：192维'], 'blue',21,31)
    arrow([(422,296),(496,296)])
    arrow([(422,462),(457,462),(457,340),(496,340)])
    arrow([(755,374),(755,401)])
    card(496,406,524,180,'全局药物–靶点交互',[
        '双向 FiLM 调制 + 低秩双线性（rank 48）',
        '拼接 d、t、逐元素积 / 差、双线性、家族',
        'MLP 256 → 共享头 + 6 专家路由 + 余弦项',
        '并行投影 → 交互隐状态 h_i：192维'], 'blue',20,31)
    arrow([(755,586),(755,615)])
    card(496,620,524,142,'按查询聚合正负支持',[
        '共享药物编码 → 正 / 负分离的注意力池化',
        '学习支持门控 g 和分数修正 Δsupport',
        'z_i = z_global + g × Δsupport'], 'blue',21,30)
    arrow([(422,708),(496,708)],C['amber'])
    arrow([(755,762),(755,795)])
    card(496,800,524,128,'两个底座汇总',[
        '基础分数 z_base = (z1 + z2) / 2',
        'H = concat(h1, h2)：384维'], 'blue',23,33)
    txt(515,967,'H 来自支持融合前的全局交互投影',19,C['muted'])
    txt(515,996,'训练 J 时底座与特征缓存均不更新',19,C['muted'])

    # Three independently understandable branches within one adapter.
    card(1114,273,225,321,'全局残差 MLP',[
        '输入：H 与 z_base / 5',
        '385 → 96 → 2',
        'LayerNorm / GELU',
        'Dropout 0.1',
        '',
        '输出：rMLP（双向）'], 'teal',20,34)
    card(1354,273,249,321,'预训练向量交互',[
        'BerMol：768 → 48',
        '完整 ESM2：1280 → 48',
        '拼接 d′、t′、',
        'd′⊙t′、|d′−t′|',
        '192 → 64 → 2',
        '输出：rpre（双向）'], 'teal',19,34)
    port(1477,254,'P',C['purple'])
    card(1618,273,237,321,'有符号直接证据',[
        '从 E 取有效正 / 负',
        '最大相似度 s_pos / s_neg',
        '使用 s 与 s² 两项',
        '非负权重分别学习',
        '正证据 − 负证据',
        '输出：rE（双向）'], 'teal',20,34)
    port(1736,254,'E',C['amber'])
    arrow([(1020,835),(1070,835),(1070,467),(1114,467)])
    for x in [1226,1478,1736]:arrow([(x,594),(x,636)],C['teal'])
    rect(1114,642,741,122,C['tealbg'],C['teal'],14)
    txt(1136,678,'零初始化、有界的双向修正',25,C['teal'],True)
    txt(1484,720,'Δ = 4 tanh((rMLP + rpre) / 4) + 6 tanh(rE / 6)',24,C['ink'],anchor='middle',width=700)
    txt(1136,747,'两个输出分量分别服务药→靶、靶→药',19,C['muted'])
    arrow([(1484,764),(1484,800)],C['teal'])
    rect(1114,807,741,102,'#E0F1EE',C['teal'],14,2)
    txt(1136,843,'跨 3 个增量模块求均值，再加回冻结基础分数',24,C['teal'],True,width=695)
    txt(1484,883,'z_dir = z_base + mean(Δ1_dir, Δ2_dir, Δ3_dir)',25,anchor='middle',width=690)
    arrow([(1020,889),(1076,889),(1076,865),(1114,865)])
    arrow([(1285,909),(1285,935)],C['teal'])
    arrow([(1685,909),(1685,935)],C['teal'])
    rect(1114,940,356,65,C['teal'],radius=11)
    rect(1490,940,365,65,C['teal'],radius=11)
    txt(1292,968,'药 → 靶点排序',23,C['white'],True,'middle')
    txt(1292,993,'固定药物，按目标集合排序',18,C['white'],anchor='middle')
    txt(1672,968,'靶点 → 老药排序',23,C['white'],True,'middle')
    txt(1672,993,'固定靶点，按老药集合排序',18,C['white'],anchor='middle')

    rect(40,1050,1840,149,C['ink'],radius=18)
    txt(63,1084,'训练方式',25,C['white'],True)
    columns=[(65,'① 先训练 V3 底座',[
        '明确正负标签 BCE + 按药物均衡采样',
        '得到两个独立种子的底座权重']),
        (665,'② 冻结底座，训练 J',[
        '0.25 × 实测 BCE + 0.5 × 双向关系检索损失',
        '未知配对进入检索分母，不伪装成生化阴性']),
        (1265,'③ 验证集选择，再冻结测试',[
        '按双向 AP / Recall 选轮数，保留实测 AP 约束',
        '最终输出是排序 logit，尚非校准概率'])]
    for x,title,content in columns:
        txt(x,1120,title,23,'#BCDFDA',True,width=565)
        lines(x,1152,content,19,28,'#E3EAF2',width=565)
    txt(57,1238,'当前 J',22,C['blue'],True)
    txt(157,1238,'实体划分；471 条训练老药药理关系；用于本轮 KIRHub 评分。',21,width=1700)
    txt(57,1275,'时间 J',22,C['teal'],True)
    txt(157,1275,'同拓扑、独立权重；≤2022 年实验关系重训，815 条老药阳性用于检索监督；测试 2023–2025。',21,width=1680)
    txt(57,1323,'当前边界：全局向量交互；原子–残基交叉注意力、三维口袋 / DrugCLIP 分支未在该候选中启用。',22,C['muted'],width=1770)
    txt(57,1354,'评分覆盖：720 老药 × 384 核心靶点；评估时按具体协议筛选候选。图示为已实现升级候选，生产默认尚未替换。',18,C['muted'],width=1780)
    parts.append('</svg>')
    svg=OUT/'BIOMASTER_CURRENT_V3_J_ARCHITECTURE_20260906_ZH.svg'
    svg.write_text('\n'.join(parts))
    cairosvg.svg2png(url=str(svg),write_to=str(svg.with_suffix('.png')),output_width=W*2,output_height=H*2)
    cairosvg.svg2pdf(url=str(svg),write_to=str(svg.with_suffix('.pdf')))
    inputs=['biomaster/odti_support_v3.py','biomaster/odti_v2.py','biomaster/odti_v3_incremental.py',
            'outputs/biomaster_v3_incremental_20260906/CANDIDATE.json','configs/biomaster_v3_temporal_20260906.json']
    manifest={'figure':'current implemented V3 + J candidate','dimensions':[W,H],
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [svg,svg.with_suffix('.png'),svg.with_suffix('.pdf')]}}
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(svg.with_suffix('.png'))


if __name__=='__main__':main()
