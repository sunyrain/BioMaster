#!/usr/bin/env python3
"""Read-only diagnostic of model disagreement; no refitting or reranking."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from scipy.stats import hypergeom

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict_biomaster_target_comparators_20260908 import OUT as SOURCE, GENES, drugs, sha
OUT=SOURCE/'disagreement_audit'


def main():
    OUT.mkdir(exist_ok=True)
    library=drugs();mols=[Chem.MolFromSmiles(s) for s in library.smiles]
    props=pd.DataFrame(dict(drug_id=library.drug_id,
        molecular_weight=[Descriptors.MolWt(m) for m in mols],
        logp=[Crippen.MolLogP(m) for m in mols],tpsa=[rdMolDescriptors.CalcTPSA(m) for m in mols],
        heavy_atoms=[m.GetNumHeavyAtoms() for m in mols]))
    props.to_csv(OUT/'MOLECULAR_DESCRIPTORS_720.csv',index=False)
    old=ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_720x384_deployment_v1/DTIAM_720X384_SCORES_V1.csv.gz'
    grid=pd.read_csv(old,usecols=['ligand_inchikey','target_chembl_id','dtiam_probability'])
    assert len(grid)==720*384 and not grid.duplicated(['ligand_inchikey','target_chembl_id']).any()
    background=grid.groupby('ligand_inchikey').dtiam_probability.mean()
    background.rename('mean_probability_across_384_targets').to_csv(OUT/'DTIAM_BACKGROUND_PROPENSITY_720.csv')
    rows=[];partial=[];propensity=[];normalization=[];inputs=[old]
    for gene in GENES:
        p=SOURCE/f'{gene}_THREE_MODEL_COMPARISON_720.csv';inputs.append(p)
        d=pd.read_csv(p).merge(props,on='drug_id',validate='one_to_one')
        assert len(d)==720 and d.drug_id.is_unique
        cols=['target_to_drug_logit','dtiam_probability','drugclip_score']
        models=['ours','dtiam','drugclip']
        for model,col in zip(models,cols):
            top=d.nlargest(20,col)
            rows.append(dict(gene=gene,model=model,
                score_mw_spearman=float(d[col].corr(d.molecular_weight,method='spearman')),
                score_logp_spearman=float(d[col].corr(d.logp,method='spearman')),
                top20_median_molecular_weight=float(top.molecular_weight.median()),
                top20_median_logp=float(top.logp.median())))
        X=np.c_[np.ones(720),d[['molecular_weight','logp','tpsa']].rank().to_numpy()]
        y=d[cols].rank().to_numpy();residual=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
        corr=np.corrcoef(residual.T)
        for a,b in itertools.combinations(range(3),2):
            partial.append(dict(gene=gene,model_a=models[a],model_b=models[b],
                raw_spearman=float(d[cols[a]].corr(d[cols[b]],method='spearman')),
                partial_rank_correlation=float(corr[a,b])))
        propensity.append(dict(gene=gene,query_vs_mean_384_score_spearman=float(d.set_index('drug_id').dtiam_probability.corr(background,method='spearman'))))
        normalization.append(dict(gene=gene,raw_cosine_vs_robust_z_rank_spearman=float(d.drugclip_rank.corr(d.drugclip_cosine_rank,method='spearman')),
            top20_overlap=int(((d.drugclip_rank<=20)&(d.drugclip_cosine_rank<=20)).sum()),
            mean_fold_top20_count_for_ensemble_top20=float(d.nsmallest(20,'drugclip_rank').fold_top20_count.mean())))
    pd.DataFrame(rows).to_csv(OUT/'MODEL_CHEMICAL_ASSOCIATIONS.csv',index=False)
    pd.DataFrame(partial).to_csv(OUT/'PARTIAL_RANK_CORRELATIONS.csv',index=False)
    pd.DataFrame(propensity).to_csv(OUT/'DTIAM_QUERY_VS_BACKGROUND.csv',index=False)
    pd.DataFrame(normalization).to_csv(OUT/'DRUGCLIP_NORMALIZATION_SENSITIVITY.csv',index=False)
    k=np.arange(21)
    null=dict(candidate_entries=720,top_k=20,assumption='Independent uniformly random size-20 subsets; illustrative reference, not a fitted null or significance test',
        pair_expected_overlap=20**2/720,pair_probability_zero=float(hypergeom.pmf(0,720,20,20)),
        triple_expected_overlap=20**3/720**2,
        triple_probability_zero=float(np.sum(hypergeom.pmf(k,720,20,20)*hypergeom.pmf(0,720,k,20))))
    result=dict(
        status='PASS',chemical_associations=rows,partial_correlations=partial,dtiam_background=propensity,
        normalization_sensitivity=normalization,random_overlap_reference=null,
        interpretation='Descriptive associations in 3 targets. Neither causal feature attribution nor prospective accuracy.',
        model_predictions_modified=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},producer_sha256=sha(__file__))
    (OUT/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 三靶点模型分歧：原因诊断与结论修正','',
        '2026-09-08。本次只读取已经生成的预测，计算描述性诊断；没有重新训练、修改候选排名或用一致性选择模型。','',
        '## 结论','',
        '目前有证据支持三个问题同时存在：外部靶点泛化尚未验证，模型表现出不同的通用药物偏好，结构路线对口袋输入敏感。不同训练目标也使三者的分数不代表完全相同的物理量。现有证据还不足以把各原因分摊为因果贡献。','',
        '上一轮强调“三模型 Top20 没有共同候选”过重。独立均匀随机抽取三份 Top20 时，期望共同条目只有 0.0154，约 98.47% 的情况下交集为空；两份名单也有约 56.49% 概率没有交集。这只是解释指标尺度的随机参照，不能视为本数据的正式显著性检验。需要结合全榜相关性、已知阳性富集及输入敏感性。','',
        '## 量化证据','',
        '| 靶点 | DTIAM 分数与分子量相关性 | DrugCLIP 分数与分子量相关性 | DTIAM Top20 分子量中位数 | DrugCLIP Top20 分子量中位数 |',
        '|---|---:|---:|---:|---:|']
    for gene in GENES:
        a=next(r for r in rows if r['gene']==gene and r['model']=='dtiam');b=next(r for r in rows if r['gene']==gene and r['model']=='drugclip')
        lines.append(f'| {gene} | {a["score_mw_spearman"]:.3f} | {b["score_mw_spearman"]:.3f} | {a["top20_median_molecular_weight"]:.1f} | {b["top20_median_molecular_weight"]:.1f} |')
    lines += ['',f'整个库的分子量中位数为 {props.molecular_weight.median():.1f}。DTIAM 在本次查询里偏向较大分子，DrugCLIP 对当前口袋偏向较小分子。分子量、脂溶性与药物类别相关，这些结果不能单独证明模型的因果机制或认定偏好全部错误。','',
        '对 MW、logP、TPSA 的秩作线性调整后，DTIAM–DrugCLIP 的偏相关分别约 −0.128、−0.206、−0.048，比原始负相关弱，但没有出现强正相关。化学性质偏好解释了部分统计分歧，不能解释全部，也不能把该调整直接用于改排名。','',
        'DTIAM 新靶点分数与其原有 384 个靶点的每药平均分的相关性：'+ '；'.join(f'{r["gene"]}={r["query_vs_mean_384_score_spearman"]:.3f}' for r in propensity)+'。这支持通用药物倾向较强的解释；平均背景只是诊断参照，不能减去后就宣称提升。','',
        '我们模型的分子量相关性接近零，不能据此认定更正确。其三个外部靶点之间仍有 0.56–0.61 的排名相关性，且在这些靶点没有独立性能验证。它与 DTIAM 的全榜相关性接近零，说明二者尚未产生一致且经过验证的目标特异信号。','',
        'HGF 的 DrugCLIP 更换口袋来源后，全榜相关性只有 0.337、Top20 重合 1 个。改用未标准化的余弦也仅为 0.424 的跨口袋来源相关性，说明分歧不只是标准化造成。原始余弦与 robust-z 的主路线 Top20 分别重合 19/20、12/20、13/20；标准化也会影响多口袋的相对竞争。','',
        '35 项既有验证通过：ID/顺序、官方权重哈希、ESM 特征定义、历史预测控制与打分重算没有发现错误。但这些检查不能证明预测口袋就是药物的结合位点，也未排除所有输入或实现风险。','',
        '## 实际需要补齐的输入与验证','',
        'BioMaster 当前交付版和 DTIAM 本次兼容重训版用全蛋白表示预测关系；DrugCLIP 用局部三维口袋检索配体。后者不直接回归 Kd，见 [DrugCLIP 原论文](https://arxiv.org/abs/2310.06367)。本次 DTIAM 是项目数据重训的分类器，不能把结果归结为原论文全部模型质量；其不同任务由不同预测器训练，见 [官方代码](https://github.com/CSUBioGroup/DTIAM/blob/main/code/training_validation.py)。','',
        '- **SLC8A1：输入选择有明确改进空间。** 已有 [人源 NCX1–SEA0400 实验复合物 8SGI](https://www.rcsb.org/structure/8SGI)，2024-04-24 发布；人源 NCX1 是链 A，伴随鼠源 Fab 链。上一轮泛 AF/P2Rank 输入未充分利用此证据。下一步应审计构建体/亚型与残基映射，提取实验位点，用独立构象 SEA0400 做结构检索参照，再重排 720 库。实验结构和已知配体只能作为当前预测的辅助验证，不能宣称是排除预训练重叠的独立基准。',
        '- **LYVE1：位点证据薄弱。** P2Rank 最高概率仅 0.013，不能证明没有可结合位点，但不能把官方缓存的位点当作已经验证的小分子结合口袋。LYVE1 的透明质酸结合及糖基化依赖有 [原始实验依据](https://pubmed.ncbi.nlm.nih.gov/19033446/)；无糖 AF 输入不足以表达这些条件。',
        '- **HGF：先明确测 HGF 本体。** 它是 [MET 的配体](https://www.uniprot.org/uniprotkb/P14210-1/entry)。HGF 本体结合、HGF–MET 相互作用阻断、MET 激酶抑制不是可直接混用的标签。需要直接 HGF 结合参照及合理构建体，不能用 MET 抑制剂名单反向验证本次 HGF 排名。','',
        '## 对模型升级的含义','',
        '目标应当是正确靶点、正确位点上有可测的富集和选择性。强行让三模型输出趋同，只会提高一致性表面指标。优先增加同药跨靶点、同靶点性质匹配药物的对比；统一直接结合与功能标签的分层；用验证过的局部位点加强交互监督。各改进都要用留出的已知关系与阴性参照验收。','',
        '先用诊断控制定位问题：药物平均倾向基线、靶点置换、药物性质匹配、口袋来源和构象扰动。靶点置换后若指标不明显下降，说明模型利用靶点信息不足；更换不相关口袋仍保留高分，则要检查结构分支的位点特异性。这些控制尚未在本次诊断中训练或全量测试，不能将其预期结果写成既成结论。','',
        '新 ContextPocket 未用于这些查询；其预训练编码器和更细的交互结构也不能自动保证改善，仍需通过上述验证。当前名单应作为低置信度探索性假设，不能给出命中率或亲和力承诺。','',
        '[诊断数值和输入哈希](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/disagreement_audit/RESULT.json)','']
    report=ROOT/'docs/BIOMASTER_QUERY_DISAGREEMENT_ANALYSIS_20260908_ZH.md';report.write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS',report=str(report),random_overlap=null,dtiam_background=propensity),ensure_ascii=False))


if __name__=='__main__':main()
