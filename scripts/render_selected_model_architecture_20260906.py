#!/usr/bin/env python3
"""Render an editable schematic of the actual selected catalog network."""
from html import escape
import hashlib
import json
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
OUT = ROOT / 'docs/presentations/selected_model_20260906'
STEM = 'RETARGETMAP_SELECTED_ARCHITECTURE_20260906_ZH'


def main():
    meta = json.loads((BUNDLE / 'metadata.json').read_text())
    selected = json.loads((BUNDLE / 'architecture_selection.json').read_text())
    assert meta['variant'] == 'global' and meta['drug_representation'] == 'drugclip_morgan'
    assert not meta['local_interactions']
    assert selected['selected_development']['parameters'] == 1235604
    OUT.mkdir(parents=True, exist_ok=True)
    items = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1080" viewBox="0 0 1800 1080">
<title>ReTargetMap 当前交付模型架构</title>
<desc>冻结DrugCLIP分子表征、Morgan指纹、化学图均值及完整ESM2蛋白均值，经投影和全局配对交互，进入一个共享MLP和双向输出层。原子残基与口袋几何没有纳入最终模型。</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker>
</defs>
<rect width="1800" height="1080" fill="#f8fafc"/>
<g font-family="Noto Sans CJK SC, sans-serif">
''']
    def rect(x, y, w, h, fill='white', stroke='#d7e0e9', radius=18):
        items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    def text(x, y, value, size=23, color='#15263c', weight=400, anchor='start'):
        items.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(value)}</text>')
    def line(path, color='#64748b', arrow=True, width=2.5):
        mark=' marker-end="url(#arrow)"' if arrow else ''
        items.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{mark}/>')
    def pill(x, y, w, label, fill, color):
        rect(x, y, w, 34, fill, fill, 17);text(x+w/2, y+24, label, 17, color, 500, 'middle')

    text(60, 72, 'ReTargetMap', 39, '#14243b', 700)
    text(352, 72, '当前交付模型架构', 36, '#14243b', 700)
    text(61, 118, 'DrugCLIP + Morgan + ESM2  ·  全局交互  ·  单网络双向排序', 24, '#516177')
    pill(1322, 48, 416, '1,235,604 个可训练参数', '#e7eefa', '#24446c')
    text(1734, 116, 'retargetmap_selected_v1  /  2026-09-06', 17, '#738196', anchor='end')
    line('M 60 149 H 1740', '#d7e0e9', False, 1)

    text(60, 196, '01  冻结表征与固定特征', 24, '#18756c', 700)
    text(530, 196, '02  可训练的全局交互网络', 24, '#6651a4', 700)
    text(1519, 196, '03  双向排序', 24, '#2d6793', 700)

    # Drug features: 512 + 2048 + 40 + 1 = 2601.
    rect(60, 234, 410, 290, '#ffffff', '#a9d4ce')
    text(85, 275, '药物', 27, '#155c55', 700)
    text(157, 275, 'SMILES / 分子构象', 21, '#5e7573')
    pill(85, 298, 358, '离线提取，推理时读取缓存', '#e9f5f2', '#28766d')
    for y, label, dim in [(371, 'DrugCLIP 全局分子表征', '512'), (411, 'Morgan 指纹', '2048'),
                           (451, '化学图均值', '40'), (491, '分子预训练特征可用标志', '1')]:
        text(85, y, label, 21);text(442, y, dim, 22, '#18756c', 600, 'end')

    rect(60, 603, 410, 212, '#ffffff', '#a9d4ce')
    text(85, 646, '靶点', 27, '#155c55', 700)
    text(157, 646, '完整蛋白序列', 21, '#5e7573')
    text(85, 697, 'ESM2 冻结编码器', 23)
    text(85, 740, '完整序列残基表征取均值', 21)
    text(85, 782, '全局蛋白表征', 21);text(442, 782, '1280', 22, '#18756c', 600, 'end')

    # Trainable projections. Encoders are not fine-tuned in this release.
    rect(530, 292, 238, 148, '#f3f0fb', '#c8bde8')
    text(649, 332, '药物投影', 25, '#5d4699', 700, 'middle')
    text(649, 371, '2601 → 192', 26, '#342550', 600, 'middle')
    text(649, 408, 'LayerNorm · Linear · GELU', 15.5, '#685f80', anchor='middle')
    text(649, 479, '药物向量 d', 22, '#675096', 500, 'middle')
    line('M 470 366 H 530')

    rect(530, 634, 238, 148, '#f3f0fb', '#c8bde8')
    text(649, 674, '蛋白投影', 25, '#5d4699', 700, 'middle')
    text(649, 713, '1280 → 192', 26, '#342550', 600, 'middle')
    text(649, 750, 'LayerNorm · Linear · GELU', 15.5, '#685f80', anchor='middle')
    text(649, 821, '蛋白向量 t', 22, '#675096', 500, 'middle')
    line('M 470 708 H 530')

    rect(845, 414, 276, 246, '#eeeaf9', '#c1b4e3')
    text(983, 459, '全局配对交互', 26, '#5d4699', 700, 'middle')
    text(983, 503, '拼接两个向量及其关系', 20, '#685f80', anchor='middle')
    text(983, 554, '[ d, t, d ⊙ t, |d − t| ]', 23, '#35264f', 600, 'middle')
    pill(923, 581, 120, '768 维', '#ddd5f0', '#59458b')
    text(983, 639, '乘积 / 绝对差均为逐元素运算', 15, '#736588', anchor='middle')
    line('M 768 366 H 807 V 490 H 845')
    line('M 768 708 H 807 V 586 H 845')

    rect(1170, 354, 300, 365, '#ffffff', '#c1b4e3')
    text(1320, 397, '共享交互主干', 26, '#5d4699', 700, 'middle')
    text(1320, 442, 'Linear 768 → 192 + GELU', 20, '#44345f', anchor='middle')
    line('M 1194 465 H 1446', '#e5dff0', False, 1)
    text(1320, 501, '残差 MLP × 2', 24, '#35264f', 600, 'middle')
    text(1320, 536, '每个：192 → 384 → 192', 20, '#736588', anchor='middle')
    line('M 1194 558 H 1446', '#e5dff0', False, 1)
    text(1320, 594, '共享变换 192 → 192', 21, '#44345f', anchor='middle')
    text(1320, 624, 'LayerNorm · Linear · GELU · Dropout', 14.5, '#736588', anchor='middle')
    rect(1193, 649, 254, 49, '#6651a4', '#6651a4', 12)
    text(1320, 681, '统一输出 Linear 192 → 2', 19, '#ffffff', 500, 'middle')
    line('M 1121 537 H 1170')

    # These are the two coordinates of the same readout, not two models.
    rect(1520, 272, 230, 198, '#edf5fc', '#b6d1e8')
    text(1635, 315, '药物 → 靶点', 25, '#285d87', 700, 'middle')
    text(1635, 358, '固定一种老药', 22, anchor='middle')
    text(1635, 398, '按正向分数排序', 21, '#597184', anchor='middle')
    text(1635, 440, '384 个候选靶点', 22, '#285d87', 600, 'middle')
    rect(1520, 604, 230, 198, '#edf5fc', '#b6d1e8')
    text(1635, 647, '靶点 → 药物', 25, '#285d87', 700, 'middle')
    text(1635, 690, '固定一个靶点', 22, anchor='middle')
    text(1635, 730, '按逆向分数排序', 21, '#597184', anchor='middle')
    text(1635, 772, '720 个候选老药', 22, '#285d87', 600, 'middle')
    line('M 1447 674 H 1494 V 371 H 1520')
    line('M 1494 674 V 703 H 1520')
    text(1635, 525, '同一输出层的两个 logit', 17, '#687d91', anchor='middle')
    text(1635, 556, '分数用于查询内排序', 17, '#687d91', anchor='middle')

    rect(60, 880, 1690, 74, '#eef1f7', '#e0e5ee', 15)
    text(85, 926, '训练', 22, '#495b76', 700)
    text(165, 926, '实测二分类监督 + 已知关系检索排序', 22, '#495b76')
    text(782, 926, 'EMA：训练时平滑权重，推理使用一个检查点', 22, '#495b76')
    text(65, 998, '当前架构边界', 22, '#526278', 700)
    text(252, 998, '保留向量级全局交互；原子—残基交互与口袋几何经消融后未纳入本版。', 22, '#526278')
    text(65, 1040, '图中预训练编码器表示特征来源；交付包直接读取必要缓存。排序分数不是结合概率或亲和力。', 18, '#7b899a')
    items.append('</g></svg>')
    svg=OUT/(STEM+'.svg');svg.write_text('\n'.join(items))
    cairosvg.svg2png(url=str(svg),write_to=str(OUT/(STEM+'.png')),output_width=3600,output_height=2160)
    cairosvg.svg2pdf(url=str(svg),write_to=str(OUT/(STEM+'.pdf')))
    def identity(path):
        return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest=dict(selected_bundle=identity(BUNDLE/'MANIFEST.json'),renderer=identity(Path(__file__)),
        architecture=dict(representation='drugclip_morgan',variant='global',parameters=1235604,
            drug_projection=[2601,192],target_projection=[1280,192],pair_width=768,residual_mlps=2,readout=[192,2]),
        figures=[identity(OUT/(STEM+ext)) for ext in ['.svg','.png','.pdf']])
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(svg)


if __name__=='__main__':main()
