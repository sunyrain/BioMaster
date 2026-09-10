#!/usr/bin/env python3
"""Validate frozen old-drug ranking artifacts and export a readable report.

No training, score transformation or metric recomputation happens here.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_old_drug_bidirectional_20260906'
EV = OUT / 'evaluation'
REPORT = ROOT / 'docs/BIOMASTER_OLD_DRUG_BIDIRECTIONAL_TEST_20260906_ZH.md'
NAMES = {
    'production_independent': '现行独立 Borda 系统¹',
    'v6_directional_full_fit': 'V6 双向头 FULL_FIT²',
    'v3_support_ensemble': 'V3 支持集模型（2 种子）',
    'morgan_ensemble': 'V4 R1 Morgan（3 种子）',
    'bermol_ensemble': 'V4 R1 BerMol（3 种子）',
    'signed_ensemble': 'V4 R1 有符号证据（3 种子）',
    'dtiam': 'DTIAM 兼容重训版³',
    'conplex': 'ConPLex 既有评分',
    'positive_nearest': '训练阳性近邻（排除自身实体）',
    'pn_logistic': '冻结正负证据逻辑回归',
    'drugclip': 'DrugCLIP 既有 6-fold 均值',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def annotate(frame, drugs, targets, labels):
    frame = frame.copy()
    forward = frame.direction.eq('drug_to_target')
    frame['ligand_inchikey'] = frame.query_id.where(forward, frame.candidate_id)
    frame['target_chembl_id'] = frame.candidate_id.where(forward, frame.query_id)
    dm = drugs.set_index('ligand_inchikey')
    tm = targets.set_index('target_chembl_id')
    frame['drug_names'] = frame.ligand_inchikey.map(dm.drug_names)
    frame['gene_symbol'] = frame.target_chembl_id.map(tm.gene_symbol)
    frame['drug_training_role_v3_r1'] = frame.ligand_inchikey.map(dm.training_role)
    frame['query_name'] = frame.drug_names.where(forward, frame.gene_symbol)
    frame['candidate_name'] = frame.gene_symbol.where(forward, frame.drug_names)
    di = frame.ligand_inchikey.map(dm.drug_feature_index).to_numpy(int)
    ti = frame.target_chembl_id.map(tm.target_feature_index).to_numpy(int)
    frame['is_frozen_known_relationship'] = labels['known_relationship'][di, ti]
    frame['observed_binary_label'] = labels['observed_binary'][di, ti]
    frame['seen_binary_training_pair_v3_r1'] = labels['seen_training_pair'][di, ti]
    assert frame[['drug_names', 'gene_symbol', 'drug_training_role_v3_r1']].notna().all().all()
    return frame


def main():
    protocol = json.loads((OUT / 'PROTOCOL.json').read_text())
    status = json.loads((OUT / 'STATUS.json').read_text())
    assert status['status'] == 'COMPLETE' and not status['new_training']
    assert status['original_protocol_sha256'] == digest(OUT / 'PROTOCOL.json')
    checked = {}

    def check(path, expected):
        path = str(path)
        if path not in checked:
            checked[path] = digest(ROOT / path)
        assert checked[path] == expected, path

    for path, sha in protocol['input_sha256'].items():
        check(path, sha)
    for path, sha in json.loads((OUT / 'FEATURE_MANIFEST.json').read_text())['files'].items():
        check(path, sha)
    drugs = pd.read_csv(OUT / 'OLD_DRUGS_720.csv')
    targets = pd.read_csv(OUT / 'TARGETS_384.csv.gz')
    with np.load(OUT / 'LABELS_AND_SCOPES.npz') as arrays:
        labels = {key: arrays[key] for key in arrays.files}
    known = labels['known_relationship']
    assert known.shape == (720, 384) and known.sum() == 807
    assert known.any(1).sum() == 398 and known.any(0).sum() == 185
    assert np.array_equal(drugs.drug_feature_index, np.arange(720))
    assert np.array_equal(targets.target_feature_index, np.arange(384))
    assert drugs.ligand_inchikey.is_unique and targets.target_chembl_id.is_unique
    assert np.array_equal(targets.sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest()), targets.sequence_sha256)
    score_count = 0
    for path in sorted((OUT / 'scores').glob('*.npz')):
        metadata = json.loads(path.with_suffix('.json').read_text())
        check(str(path.relative_to(ROOT)), metadata['score_sha256'])
        for key in ['checkpoint', 'source', 'target_identity_source']:
            if key in metadata:
                check(metadata[key], metadata[key + '_sha256'])
        with np.load(path) as scores:
            for direction in ['drug_to_target', 'target_to_drug']:
                values = scores[direction]
                assert values.shape == (720, 384)
                if path.stem == 'drugclip':
                    with np.load(OUT / 'DRUGCLIP_COMMON_SCOPE.npz') as common:
                        assert np.isfinite(values[np.ix_(common['drug_indices'], common['target_indices'])]).all()
                        assert np.isfinite(values).sum() == 548 * 269
                else:
                    assert np.isfinite(values).all()
        score_count += 1
    # Audit the actual cached references, beyond the preparation-time assertion.
    with np.load(OUT / 'support_store/support_index.npz') as store:
        with np.load(ROOT / 'outputs/biomaster_v3_20260905/data/support_store/support_index.npz') as original:
            for key in ['drug_indices', 'target_indices', 'labels']:
                assert np.array_equal(store[key], original[key])
        entity_codes = store['entity_codes']
        indices = np.load(protocol['support']['indices'], mmap_mode='r')
        mask = np.load(protocol['support']['mask'], mmap_mode='r')
        queries = np.repeat(drugs.global_drug_index.to_numpy(), 384)
        for start in range(0, len(queries), 4096):
            ix, valid = indices[start:start+4096], mask[start:start+4096]
            assert ((ix[valid] >= 0) & (ix[valid] < 296110)).all()
            same = entity_codes[ix.clip(min=0)] == entity_codes[queries[start:start+4096], None, None]
            assert not (same & valid).any()
        support_references = int(mask.sum())
    ranks = annotate(pd.read_csv(EV / 'KNOWN_RELATION_RANKS.csv.gz'), drugs, targets, labels)
    assert ranks.is_frozen_known_relationship.all()
    assert ranks[ranks.scope.eq('project_720x384')].groupby(['model', 'direction']).size().eq(807).all()
    ranks.to_csv(EV / 'KNOWN_RELATION_RANKS.csv.gz', index=False)
    top = annotate(pd.read_csv(EV / 'TOP20_BOTH_DIRECTIONS.csv.gz'), drugs, targets, labels)
    assert len(top) == 10 * (720 + 384) * 20
    assert top.groupby(['model', 'direction', 'query_id']).size().eq(20).all()
    top['candidate_count'] = np.where(top.direction.eq('drug_to_target'), 384, 720)
    top.to_csv(EV / 'TOP20_BOTH_DIRECTIONS.csv.gz', index=False)
    top[top.direction.eq('drug_to_target')].to_csv(EV / 'TOP20_DRUG_TO_TARGET.csv.gz', index=False)
    top[top.direction.eq('target_to_drug')].to_csv(EV / 'TOP20_TARGET_TO_OLD_DRUG.csv.gz', index=False)
    d, t = np.where(known & (labels['observed_binary'] == 0))
    pd.DataFrame({'drug_names': drugs.drug_names.to_numpy()[d],
                  'ligand_inchikey': drugs.ligand_inchikey.to_numpy()[d],
                  'gene_symbol': targets.gene_symbol.to_numpy()[t],
                  'target_chembl_id': targets.target_chembl_id.to_numpy()[t]}).to_csv(
                      OUT / 'KNOWN_RELATION_BIOCHEMICAL_NEGATIVE_OVERLAP.csv', index=False)
    summary = pd.read_csv(EV / 'SUMMARY.csv')
    assert len(summary) == 66 and summary.known_ap.notna().all()

    def table(scope, measured=False):
        frame = summary[summary.scope.eq(scope)].set_index(['model', 'direction'])
        title = ('| 方法 | 老药→靶点 AP | 老药→靶点 AUROC | 靶点→老药 AP | 靶点→老药 AUROC |'
                 if measured else '| 方法 | 老药→靶点 AP | 老药→靶点 R@20 | 靶点→老药 AP | 靶点→老药 R@20 |')
        lines = [title, '|---|---:|---:|---:|---:|']
        columns = ['observed_ap', 'observed_auroc'] if measured else ['known_ap', 'known_recall_20']
        for model, name in NAMES.items():
            if model not in frame.index.get_level_values(0):
                continue
            vals = [frame.loc[(model, direction), key] for direction in ['drug_to_target', 'target_to_drug'] for key in columns]
            lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
        return '\n'.join(lines)

    exposure = []
    for role in ['train', 'validation', 'test', 'absent_from_binary_relations']:
        selected = drugs.training_role.eq(role).to_numpy()
        exposure.append(f'| {role} | {int(selected.sum())} | {int(known[selected].any(1).sum())} | {int(known[selected].sum())} |')
    report = f'''# 项目老药双向排序实测：720 老药 × 384 靶点

2026-09-06，状态：**已完成评分与测试**。本轮按用户要求，使用项目实际 720 个老药，测试“已知老药找靶点”和“已知靶点找老药”。冻结既有模型，未重新训练、选择 checkpoint 或调节融合权重。

核心发现：在项目已知关系恢复上，V3 优于新 V4 R1，训练阳性近邻强于两者；实测正负活性的排序呈现不同结果。不能把上一轮一般化合物面板的 AP 当成本轮成绩，也不能把本轮全量已知关系恢复当作独立新靶点发现验证。

## 1. 实际测试对象、标签与指标

- 正向：每个老药在完整 **384 靶点**中排序；720 个老药均输出 Top 20，其中 **398** 个有冻结已知靶点，用于宏平均指标。
- 逆向：每个靶点在完整 **720 老药**中排序；384 个靶点均输出 Top 20，其中 **185** 个有冻结已知老药，用于宏平均指标。
- 主标签：**807 条**冻结关系，含项目已知关系 791 条和 ChEMBL37 MoA 补充 16 条。它们表示已知药理/机制关系，并非全部经过统一条件下的直接结合实验确认。
- 未知关系仍作为候选背景参与检索，未被称为生化阴性。没有已知正例的 query 仍输出排名，但不计入已知恢复指标。
- AP 为每个 query 的已知正例位置计算的 Average Precision，再做宏平均；R@20 为前 20 位找回的已知正例比例，再做宏平均。并列分数按候选 ID 升序打破。完整指标还包括 MRR、Hit、NDCG、R@1/5/10/20 和每条已知关系的排名。
- 384 是当前完整已评分核心，不能将此结果的分母写成 450 或 745；此任务也不覆盖全部药品和全部人类靶点。

## 2. 完整老药空间的双向已知关系恢复

下面所有行使用同一个 720×384 候选矩阵、807 条已知关系、398/185 个有标签的正向/逆向 query。

{table('project_720x384')}

¹ 正向复现现行独立 Borda 分数并逐项核对；逆向是本次用同一组原始分数组件，**在每个靶点内重新对 720 老药做 Borda 融合**，属于新增的确定性逆向基线，不能称作此前已验证的生产逆向头。

² V6 使用自己分别训练的正向和逆向 logit。此处是 FULL_FIT 评分，已有生化标签参与过训练，不能将其高分视为独立测试泛化。

³ DTIAM 使用项目既有的官方表示兼容 AutoML S5 重训版，不是原论文逐位复现。它和现行系统、ConPLex 的训练边界与 V3/R1 不完全相同，整表用于部署空间恢复审计，不支持无偏 SOTA 宣称。

V3 为两种子原始 logit 均值；R1 各变体为三种子原始 logit 均值，然后计算排名指标。没有按本次结果挑种子。V3/R1 的逆向使用相同 pair logit 在靶点内排序，R1 尚未训练专用逆向头。

## 3. 训练接触审计与未见老药子集

| 老药相对 V3/R1 二元关系表的角色 | 老药数 | 有已知关系的老药数 | 已知关系数 |
|---|---:|---:|---:|
{chr(10).join(exposure)}

807 条已知关系中，**291 条本身出现在 V3/R1 二元训练对中**。支持库保留原始 342,031 条训练关系，逐条检索排除同一化学实体；训练过的模型权重仍可能包含该老药的信息。分子通过标准化 SMILES 与 InChI 实体映射，靶点通过完整序列 SHA256 对齐，未直接混用不同特征库的整数索引。

### 历史实体划分的 test 老药：54×384

54 个老药在原 V3/R1 划分中属于 test，正向有 34 个已知 query、71 条关系。逆向有 57 个已知靶点 query，**候选药物只有这 54 个**。它是未见老药子集的逆向排序，不能与完整 720 候选的逆向 AP/R@20 直接对比。该划分不是 scaffold/assay/time 隔离。

{table('original_test_old_drugs')}

### 二元关系表中完全不存在的老药：223×384

223 个药不在 V3/R1 train/validation/test 二元关系表中，正向 91 个已知 query、188 条关系；逆向 72 个已知靶点 query，候选为 **223 老药**。这不保证公共预训练、其他数据库或历史系统从未见过它们。此子集没有可用实测二元标签，不能补报生化 AP/AUROC。

{table('old_drugs_absent_from_binary_relations')}

以上两张表仅列训练边界一致的 V3/R1、正近邻和冻结 PN 基线。历史 test 老药的正向 AP：V3 0.4193、R1 有符号证据 0.3365、正近邻 0.6617；缺席老药分别为 0.3101、0.1698、0.4525。这支持“R1 在本轮真实老药任务上仍未超过 V3”的判断，但不支持单凭该结果否定交互架构或预训练路线。

## 4. 实测生化正负排序，单独报告

720×384 空间中另有 **5,996 条**二元观测：894 阳性、5,102 阴性。只在同一 query 下有实测正负两类的子集中算 AP/AUROC：正向 150 个老药 query，逆向 144 个靶点 query；有效候选数量因 query 而异，中位数分别为 8.5 和 26，均不是完整 384/720 候选的密集实测结果。

{table('project_720x384', measured=True)}

已知药理关系中有 **40 条**同时被现有二元生化表标为阴性。这可能涉及机制定义、前药/代谢物、测量条件或阈值，尚未逐条裁定原因；本轮保留两套原始标签、独立评估，并输出冲突交集清单。不能悄悄将这 40 条改为阳性，也不能把所有未知对补为阴性。

因此，近邻更擅长恢复项目已知关系，并不等于它在实测活性辨别上全面最好；V6 FULL_FIT 的 0.9578/0.9300 也包含训练接触，不能作独立泛化成绩。

## 5. DrugCLIP：只在相同覆盖范围内比较

既有 DrugCLIP 原生输出，经**精确模型 SMILES + 完整蛋白序列**映射，覆盖 **548 老药 × 269 靶点**。下面所有模型均限制在同一个矩形候选集合：424 条已知关系，正向 234 个已知 query，逆向 119 个已知 query。未覆盖部分没有补零；本表不能与完整空间表跨分母比较。

{table('drugclip_common')}

这是项目既有预测口袋和原生 6-fold cosine 均值的应用质量审计。DrugCLIP 在这套老药已知关系恢复上表现较低；它不等同于 DrugCLIP 原论文基准，不证明所有结构预训练都弱。本轮没有训练 V4 设计中的三维局部交互模块，也未排除 DrugCLIP 公共训练与这些已知关系的接触。

## 6. 对下一轮架构与训练的直接要求

1. **把双向排序设为独立验收目标。** 现行独立系统正向 AP 0.5126、逆向 0.2132，V3 为 0.4495/0.3564。查询方向、候选归一化和损失需要分别处理；R1 只有共享 pair logit，缺少专用逆向头。
2. **优先解释并修复相对 V3 的回退。** 先在冻结未见老药划分上拆开全局表示、支持证据和方向损失的贡献，再引入残基/原子或口袋局部交互。当前 R1 只是全局表示及有符号证据首轮实现，不能视作完整新架构的测试结果。
3. **把结构相似性作为必须超越的固定基线。** 保留训练阳性近邻与正负证据基线，分析模型损失发生在哪些老药、靶点家族及支持覆盖层；是否保留近邻残差须用验证集选择，不能在本次结果上直接调权重。
4. **分开优化已知机制恢复与实测选择性。** 需要不同标签语义和实测候选分母；优先核对 40 条冲突、补同条件活性面板，在具备相应标签的 query 内进行双向 listwise/pairwise 训练。
5. **下一轮建立新的冻结验收。** 本次全量榜可作回归测试；确认性比较需重新固定训练接触边界、scaffold/assay/time 隔离和同覆盖外部对照。老药新靶点发现最终还需未知关系的独立测量，而不能由已知关系检索 AP 替代。

## 7. 可查看与复现的产物

- [双向指标总表](../outputs/biomaster_old_drug_bidirectional_20260906/evaluation/SUMMARY.csv)：4 个范围、66 行方法/方向组合；含 R@5/10/20、MRR、NDCG 和实测 AP/AUROC。
- [老药→靶点 Top 20](../outputs/biomaster_old_drug_bidirectional_20260906/evaluation/TOP20_DRUG_TO_TARGET.csv.gz)：10 种完整覆盖方法，包含药名、基因名、已知关系标志、实测标签和 V3/R1 训练角色。
- [靶点→老药 Top 20](../outputs/biomaster_old_drug_bidirectional_20260906/evaluation/TOP20_TARGET_TO_OLD_DRUG.csv.gz)：同样包含可读名称与标签，完整分母为 720。
- [所有已知关系的实际名次](../outputs/biomaster_old_drug_bidirectional_20260906/evaluation/KNOWN_RELATION_RANKS.csv.gz)：包含 DrugCLIP 同覆盖表和未见老药子集，`scope` 和 `candidate_count` 必须同时读取。
- [标签交集待核查清单](../outputs/biomaster_old_drug_bidirectional_20260906/KNOWN_RELATION_BIOCHEMICAL_NEGATIVE_OVERLAP.csv)、[冻结协议](../outputs/biomaster_old_drug_bidirectional_20260906/PROTOCOL.json)、[产物验证](../outputs/biomaster_old_drug_bidirectional_20260906/VALIDATION.json)。

产物核查通过：22 份评分矩阵、48 个来源/产物哈希、7,479,350 条实际缓存支持参照；支持池与原训练池一致，同一查询实体出现在自身支持集中的次数为 0。相关测试结果见 [VALIDATION_TESTS.json](../outputs/biomaster_old_drug_bidirectional_20260906/VALIDATION_TESTS.json)。

首次特征准备入口为 `scripts/prepare_old_drug_bidirectional_20260906.py`，冻结协议存在时会拒绝覆盖。当前特征已准备完毕，包含 720 个 BerMol 输入与补齐的 24 个完整 ESM2 序列均值。

```bash
python scripts/evaluate_old_drug_bidirectional_20260906.py --evaluate-only
python scripts/summarize_old_drug_bidirectional_20260906.py
python -m pytest -q tests/test_old_drug_ranking.py tests/test_odti_v3_panel_metrics.py tests/test_odti_v4.py
```

去掉 `--evaluate-only` 会从冻结 checkpoint 生成缺失评分并读取既有对照分数，不进行拟合。模型、来源、评分和输入哈希随产物保存。大特征、checkpoint 和逐对排名保留本地，仓库白名单仅纳入紧凑指标与审计记录。
'''
    REPORT.write_text(report)
    validation = {'status': 'PASS', 'score_artifacts_checked': score_count,
                  'source_and_artifact_hashes_checked': len(checked),
                  'cached_support_references_checked': support_references,
                  'support_pool_identical_to_original_train': True,
                  'query_entity_present_in_own_support': False,
                  'known_pairs': 807, 'summary_rows': len(summary),
                  'top20_rows': len(top), 'known_relation_rank_rows': len(ranks),
                  'no_new_fitting': True, 'code_sha256': digest(Path(__file__))}
    dump(OUT / 'VALIDATION.json', validation)
    artifacts = [OUT / 'PROTOCOL.json', OUT / 'STATUS.json', OUT / 'FEATURE_MANIFEST.json',
                 OUT / 'VALIDATION.json', REPORT, *EV.glob('*METRICS.json'), EV / 'SUMMARY.csv',
                 EV / 'KNOWN_RELATION_RANKS.csv.gz', EV / 'TOP20_DRUG_TO_TARGET.csv.gz',
                 EV / 'TOP20_TARGET_TO_OLD_DRUG.csv.gz',
                 ROOT / 'scripts/prepare_old_drug_bidirectional_20260906.py',
                 ROOT / 'scripts/evaluate_old_drug_bidirectional_20260906.py',
                 Path(__file__), ROOT / 'biomaster/old_drug_ranking.py']
    if (OUT / 'VALIDATION_TESTS.json').exists():
        artifacts.append(OUT / 'VALIDATION_TESTS.json')
    dump(OUT / 'RESULT_MANIFEST.json', {str(p.relative_to(ROOT)): digest(p) for p in artifacts})
    print(json.dumps(validation, ensure_ascii=False))
    print(REPORT)


if __name__ == '__main__':
    main()
