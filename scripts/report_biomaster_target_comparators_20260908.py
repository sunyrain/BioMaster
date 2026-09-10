#!/usr/bin/env python3
"""Compare the three frozen query rankings; verify inputs and inference outputs."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import sys
import lmdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from predict_biomaster_target_comparators_20260908 import QUERY, OUT, BUNDLE, GENES, sha, save, drugs


def markdown(frame):
    escape=lambda x:str(x).replace('|','\\|').replace('\n',' ')
    rows=['| '+' | '.join(map(escape,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(map(escape,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def main():
    library=drugs();frames={};summary=[];agreement=[];primary=[];tops=[]
    for gene in GENES:
        o=pd.read_csv(QUERY/f'{gene}_RANKED_720.csv').rename(columns={'rank_720':'ours_rank'})
        d=pd.read_csv(OUT/f'{gene}_DTIAM_720.csv')
        route='DTWG' if gene=='LYVE1' else 'P2RANK'
        c=pd.read_csv(OUT/f'{gene}_DRUGCLIP_{route}_720.csv')
        table=o.merge(d[['drug_id','dtiam_probability','dtiam_rank']],on='drug_id',validate='one_to_one')
        table=table.merge(c.drop(columns=['name','smiles','native_feature_index','gene_symbol']),on='drug_id',validate='one_to_one')
        table['drugclip_pocket_route']=route
        table['models_in_top20']=(table[['ours_rank','dtiam_rank','drugclip_rank']]<=20).sum(axis=1)
        table.to_csv(OUT/f'{gene}_THREE_MODEL_COMPARISON_720.csv',index=False)
        primary.append(table);frames[gene]=table.set_index('drug_id')
        records={}
        for model,col in [('BioMaster','ours_rank'),('DTIAM','dtiam_rank'),('DrugCLIP','drugclip_rank')]:
            top=table.sort_values(col).head(20).copy();top['model']=model;tops.append(top)
            records[model]=top[['drug_id','name',col]].head(10).to_dict('records')
        for a,b in itertools.combinations(['ours','dtiam','drugclip'],2):
            agreement.append(dict(gene_symbol=gene,model_a=a,model_b=b,
                spearman=float(table[a+'_rank'].corr(table[b+'_rank'],method='spearman')),
                top20_overlap=int(((table[a+'_rank']<=20)&(table[b+'_rank']<=20)).sum())))
        summary.append(dict(gene_symbol=gene,primary_pocket_route=route,top10=records,
            triple_top20=int(table.models_in_top20.eq(3).sum()),
            amiodarone=table[table.name.eq('amiodarone')][['ours_rank','dtiam_rank','drugclip_rank']].to_dict('records')))
    all_table=pd.concat(primary,ignore_index=True);all_table.to_csv(OUT/'ALL_2160_THREE_MODEL_COMPARISONS.csv',index=False)
    pd.concat(tops,ignore_index=True).to_csv(OUT/'ALL_MODELS_TOP20_WITH_CROSS_RANKS.csv',index=False)
    pd.DataFrame(agreement).to_csv(OUT/'MODEL_AGREEMENT.csv',index=False)
    cross=[]
    for model in ['ours','dtiam','drugclip']:
        for a,b in itertools.combinations(GENES,2):
            x,y=frames[a],frames[b];col=model+'_rank'
            cross.append(dict(model=model,target_a=a,target_b=b,
                spearman=float(x[col].corr(y[col],method='spearman')),
                top20_overlap=len(set(x.index[x[col]<=20])&set(y.index[y[col]<=20]))))
    pd.DataFrame(cross).to_csv(OUT/'CROSS_TARGET_SIMILARITY.csv',index=False)
    hgf_dtwg=pd.read_csv(OUT/'HGF_DRUGCLIP_DTWG_720.csv').set_index('drug_id')
    hgf=frames['HGF'].join(hgf_dtwg[['drugclip_rank']].rename(columns={'drugclip_rank':'drugclip_dtwg_rank'}))
    hgf.reset_index().to_csv(OUT/'HGF_POCKET_SOURCE_SENSITIVITY.csv',index=False)
    sensitivity=dict(spearman=float(hgf.drugclip_rank.corr(hgf.drugclip_dtwg_rank,method='spearman')),
        top20_overlap=int(((hgf.drugclip_rank<=20)&(hgf.drugclip_dtwg_rank<=20)).sum()))

    checks={}
    def check(name,value):checks[name]=bool(value)
    for gene,t in frames.items():
        check(gene+'_same_exact_720_drug_ids',len(t)==720 and t.index.is_unique and set(t.index)==set(library.drug_id))
        for col in ['ours_rank','dtiam_rank','drugclip_rank']:
            check(gene+'_'+col+'_permutation',sorted(t[col].tolist())==list(range(1,721)))
        check(gene+'_finite_all_scores',np.isfinite(t[['target_to_drug_logit','dtiam_probability','drugclip_score']].to_numpy()).all())
    manifest=json.loads((BUNDLE/'MANIFEST.json').read_text())
    check('released_bundle_manifest_unchanged',sha(BUNDLE/'MANIFEST.json')=='c653906a64aa4f58803a74630287deabb352ced8f254e0d2e39e2b015012e44e')
    check('released_bundle_all_file_hashes',all(sha(BUNDLE/p)==v['sha256'] for p,v in manifest['files'].items()))
    for stage in ['ESM','PREPARE','DTIAM','DRUGCLIP']:
        complete=json.loads((OUT/(stage+'_COMPLETE.json')).read_text())
        check(stage+'_completed_source_snapshot_matches',complete['status']=='PASS' and sha(OUT/(stage.lower()+'_source.py'))==complete['producer_sha256'])
    for name in ['DTIAM_ESM_AUDIT.json','DTIAM_INFERENCE_CONTROL.json','PREPARATION_AUDIT.json','DTIAM_MANIFEST.json','DRUGCLIP_MANIFEST.json']:
        check(name+'_pass',json.loads((OUT/name).read_text())['status']=='PASS')
    for name,expected in [('molecules.lmdb',library.drug_id.tolist()),('pockets.lmdb',pd.read_csv(OUT/'P2RANK_POCKET_MANIFEST.csv').pocket.tolist())]:
        e=lmdb.open(str(OUT/name),subdir=False,readonly=True,lock=False)
        with e.begin() as tx:
            rows=[pickle.loads(v) for k,v in tx.cursor()]
        e.close(); key='name' if name=='molecules.lmdb' else 'pocket'
        check(name+'_exact_lexical_order',[r[key] for r in rows]==expected)
    m=np.load(OUT/'DRUGCLIP_MOLECULE_EMBEDDINGS.npy');p=np.load(OUT/'DRUGCLIP_P2RANK_EMBEDDINGS.npy')
    check('new_drugclip_all_embeddings_unit_norm',np.allclose(np.linalg.norm(m,axis=-1),1,atol=2e-6) and np.allclose(np.linalg.norm(p,axis=-1),1,atol=2e-6))
    # Independent dot products and the upstream retrieval formula, covering all
    # six folds/all 720 candidates, instead of trusting the exported rank column.
    base=ROOT/'third_party/sota_dti_2026/Drug-The-Whole-Genome/data_downloads/benchmark_throughput'
    dtmeta=pd.read_csv(OUT/'OFFICIAL_DTWG_POCKETS.csv')
    dt=np.asarray(np.load(base/'dtwg_af_embeddings.npy',mmap_mode='r')[dtmeta.official_row.to_numpy(int)]).reshape(-1,6,128)
    for route,emb,meta in [('P2RANK',p,pd.read_csv(OUT/'P2RANK_POCKET_MANIFEST.csv')),('DTWG',dt,dtmeta)]:
        folded=np.stack([emb[:,f]@m[:,f].T for f in range(6)],axis=2)
        raw=np.load(OUT/f'DRUGCLIP_{route}_POCKET_SCORES.npz')['fold_cosines']
        check(route+'_all_cosines_independently_reproduced',np.allclose(raw,folded,atol=2e-6))
        mean=folded.mean(axis=2);median=np.median(mean,axis=1,keepdims=True)
        z=.6745*(mean-median)/(np.median(np.abs(mean-median),axis=1,keepdims=True)+1e-6)
        for gene in meta.gene_symbol.unique():
            t=pd.read_csv(OUT/f'{gene}_DRUGCLIP_{route}_720.csv').set_index('drug_id').loc[library.drug_id]
            computed=z[meta.gene_symbol.eq(gene)].max(axis=0)
            check(route+'_'+gene+'_official_score_formula',np.allclose(computed,t.drugclip_score,atol=2e-5))
    save('VERIFICATION.json',dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks))
    if not all(checks.values()):raise ValueError({k:v for k,v in checks.items() if not v})
    save('COMPARISON_SUMMARY.json',dict(status='EXPLORATORY_PREDICTIONS_COMPLETE',targets=summary,
        model_agreement=agreement,cross_target_similarity=cross,hgf_pocket_sensitivity=sensitivity,
        ground_truth_metrics=None,reason='No complete direct-binding labels for these 3 x 720 pairs; cannot estimate AP or experimental hit rate',
        formal_context_model_used=False,verification_checks=len(checks)))

    report=ROOT/'docs/BIOMASTER_LYVE1_SLC8A1_HGF_COMPARATORS_20260908_ZH.md'
    lines=['# LYVE1、SLC8A1、HGF：BioMaster、DTIAM、DrugCLIP 实际预测对比','',
        '2026-09-08；已经完成推理。本次是同一候选库上的外部靶点排序，不是带完整标签的准确率测试。',
        '', '**结论：DTIAM 的跨靶点名单高度相似；DrugCLIP 对 HGF 口袋来源敏感。目前不能把任何一张名单当作已经确认的高亲和力药物，也不能据此宣布某模型胜出。**','',
        '解读修正：三模型 Top20 无交集本身不是充分证据。独立均匀随机抽取三份 20/720 名单时，约 98.47% 的情况下也无共同条目。应结合全榜相关性和实验参照判断；见[分歧原因补充诊断](BIOMASTER_QUERY_DISAGREEMENT_ANALYSIS_20260908_ZH.md)。','',
        '## 模型与输入','',
        '| 模型 | 本次实际版本 | 运行范围 |','|---|---|---|',
        '| BioMaster | 已交付 selected_global_fullfit_2025；沿用上一轮真实推理结果 | 3 × 720；新 ContextPocket 未用于本次 |',
        '| DTIAM | 官方 BerMol768 + ESM2-650M1280；项目 S5 兼容重训的 WeightedEnsemble_L2 | 3 × 720；65,276 训练关系；AutoGluon 1.4.0 |',
        '| DrugCLIP | 官方 Drug-The-Whole-Genome 六折权重，全部实际编码推理 | 3 × 720 主结果；另加 HGF 720 条官方口袋敏感性结果 |','',
        'DTIAM 是此前项目部署使用的兼容重训版本，不是论文原始下游预测器。三靶点在其原始 86,674 行数据及靶点序列目录中都没有精确匹配；公共预训练编码器是否见过相关序列/结构未确定。BioMaster 与 DTIAM 的训练数据和目标并不相同，候选库相同不等于公平的性能基准。官方定义见 [DTIAM 代码](https://github.com/CSUBioGroup/DTIAM) 与 [DrugCLIP 代码和权重](https://github.com/THU-ATOM/Drug-The-Whole-Genome)。','',
        'DTIAM 使用官方的 FP32 ESM2 末层均值：去除 BOS，保留 EOS；三个查询均短于 1022 aa，无截断。现有特征控制向量最大误差为 0，32 个历史预测控制的最大误差为 2.76×10⁻⁸。','',
        'DrugCLIP 使用官方六折余弦均值，再按每个口袋在固定 720 条目上的 median/MAD 标准化，最后取口袋最大值。分数依赖候选库，不是 Kd、IC50 或结合概率。新编码为 FP32、TF32 关闭；完整九个 P2Rank 口袋均小于 256 原子，没有裁剪。','',
        '719 个配体复用已有独立 ETKDG 构象。奎尼丁原构象缺失且重试失败，改用 [PubChem 3D](https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/LOUPRKONTZGTKE-LHHVKLHASA-N/SDF?record_type=3d)，完整 InChIKey 和重原子数验证通过。这里的 720 是冻结库中 720 个唯一结构 ID，部分通用名对应不同结构条目；未按名称合并。','',
        '| 靶点 | 人源序列 | DrugCLIP 主输入 | 输入限制 |','|---|---|---|---|',
        '| LYVE1 | Q9Y5Y7，322 aa | 官方 DTWG AFv4 同一位点的原始/精修两个表示 | 新 AFv6 的 P2Rank 最高概率仅 0.013，未达预设 0.2；没有强行降低阈值 |',
        '| SLC8A1 | P32418，973 aa | AFv6 + P2Rank 六个口袋 | 官方 DTWG 表中无此 accession；全长单体 AF 输入，未显式建模膜环境或运输构象循环 |',
        '| HGF | P14210，728 aa | AFv6 + P2Rank 三个口袋 | 另有官方 DTWG 六个位点各原始/精修，共十二个表示；HGF 本身不是 MET |','',
        'P2Rank 概率 ≥ 0.2 是事先设定的输入阈值。保留全部达标口袋，不根据药物分数挑选口袋。LYVE1 的官方缓存无法仅凭 embedding 文件检查残基范围或 pLDDT；原始/精修表示也不是两份独立实验。AF 结构的序列已与本次 UniProt 查询逐位匹配。','',
        '## 各模型的候选与交叉排名','',
        '下表的排名均在同一 720 个结构条目中，越小越靠前；不同模型的原始分数不作数值大小比较。所有条目仍是未验证假设。','']
    for gene in GENES:
        t=frames[gene]
        lines += [f'### {gene}','']
        for model,col in [('DTIAM','dtiam_rank'),('DrugCLIP','drugclip_rank')]:
            x=t.sort_values(col).head(10)[['name','ours_rank','dtiam_rank','drugclip_rank']].rename(columns={'name':'药物（库中原名）','ours_rank':'BioMaster 排名','dtiam_rank':'DTIAM 排名','drugclip_rank':'DrugCLIP 排名'})
            lines += [f'{model} Top10：','',markdown(x),'']
        lines += [f'[完整 {gene} 三模型对照](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/{gene}_THREE_MODEL_COMPARISON_720.csv)','']
    lines += ['## 一致性与不稳定性','',markdown(pd.DataFrame(agreement).round(4)),'',
        '三个靶点的三模型 Top20 交集均为 0；DTIAM 与 DrugCLIP 的 Top20 两两交集也均为 0。模型不一致提示候选选择依赖模型与输入，但不能直接证明其中任何一方“错误”。BioMaster 和 DrugCLIP 还共享 DrugCLIP 预训练成分，因此即使一致，也不应当当作完全独立证据。','',
        'DTIAM 跨靶点名单相似性：','',markdown(pd.DataFrame(cross).query("model == 'dtiam'").round(4)),'',
        '0.87–0.91 的跨靶点相关性提示该兼容模型可能较多依赖药物自身特征，对这些外部靶点的区分有限；这是针对本次查询的诊断线索，不是论文模型整体质量结论。其约 0.9 的分类输出在这些查询上没有概率校准证据。','',
        f'HGF 的 P2Rank 与官方 DTWG DrugCLIP 排名相关性为 **{sensitivity["spearman"]:.4f}**，Top20 仅重合 **{sensitivity["top20_overlap"]}** 个条目。',
        'P2Rank 路线前三名为 famotidine、esmolol、metoprolol；官方 DTWG 路线前三名为 valproate、ibuprofen、belinostat。',
        '我们之前排首位的 berotralstat：BioMaster 第 1、DTIAM 第 20、DrugCLIP P2Rank 第 477、DrugCLIP DTWG 第 31。现阶段不能把它称为稳定的多模型高亲和力候选。','',
        'SLC8A1 的 amiodarone：BioMaster 第 87、DTIAM 第 172、DrugCLIP 第 429。已有的是 [豚鼠 NCX 电流急性抑制证据](https://pmc.ncbi.nlm.nih.gov/articles/PMC1572287/)，不是人源 SLC8A1 的直接结合 Kd；因此只能作为功能文献参照，不能据单个排名计算模型准确率。','',
        '六折 Top20 次数已保存到 CSV，用于观察各折名单稳定性；六折来自同一模型家族，不能把次数解释为实验命中概率。','',
        '## 如何使用这些结果','',
        '当前没有这 2,160 对关系的完整直接结合标签，因此不报告 AP、Recall、命中率或 nM 亲和力。上一轮本地 ChEMBL 核验也没有与该 720 库精确匹配的直接证据；这不代表文献中完全没有关系。','',
        'LYVE1 应先确认构建体、糖基化状态和可测试结合位点；SLC8A1 应把膜蛋白功能实验与直接结合实验分开；HGF 应以 HGF 本体结合证据筛选，不能混入 MET 激酶抑制结果。三者都需要先解决具体实验机制和输入结构匹配，再决定实验优先级。','',
        '本次名单适合形成待核验的候选池。没有依据直接将三模型分数相加，或因模型给出高分就扩大实验采购。HGF 的口袋来源敏感性尤其需要先处理。','',
        '## 可复现产物','',
        '- [全部 2,160 行三模型比较](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/ALL_2160_THREE_MODEL_COMPARISONS.csv)',
        '- [各模型 Top20 与交叉排名](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/ALL_MODELS_TOP20_WITH_CROSS_RANKS.csv)',
        '- [HGF 口袋来源敏感性](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/HGF_POCKET_SOURCE_SENSITIVITY.csv)',
        '- [验证清单](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/VERIFICATION.json)',
        '- [机器可读摘要](../outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators/COMPARISON_SUMMARY.json)',
        '- [推理脚本](../scripts/predict_biomaster_target_comparators_20260908.py)；各执行阶段保留源码快照和哈希。','',
        f'验证结果：{len(checks)} 项全部通过，包含 720 条目身份和顺序、三组完整排名、DTIAM 历史控制、DrugCLIP 六折权重哈希、独立打分公式重算、已交付模型全部文件哈希。训练、冻结实验候选及已交付模型均未修改。','']
    report.write_text('\n'.join(lines))
    save('ARTIFACT_MANIFEST.json',dict(report_sha256=sha(report),report_producer_sha256=sha(__file__),
        files={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.npy','.npz','.py'} and p.name!='ARTIFACT_MANIFEST.json'}))
    print(json.dumps(dict(status='PASS',checks=len(checks),report=str(report),hgf_sensitivity=sensitivity),ensure_ascii=False))


if __name__=='__main__':main()
