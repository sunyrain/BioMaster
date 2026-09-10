import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/spr512_txgnn_ot_effect_20260909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 pairs=pd.read_csv(OUT/'PAIR_RESULTS_512.csv');details=pd.read_csv(OUT/'JOINT_DISEASE_DETAILS.csv.gz');high=pairs[pairs.role.eq('OUR_FROZEN_MODEL_HIGH')];strong=high[high.rank50_ot03.eq(True)]
 candidates=pairs[pairs.role.ne('POSITIVE_CONTROL')].copy();candidates['review_bucket']=['TXGNN_MISSING_NOT_NEGATIVE' if not r.txgnn_available else ('JOINT_SIGNAL_REVIEW_NOT_BINDING_VALIDATION' if r.rank50_ot03 else 'NO_JOINT_SIGNAL_AT_THIS_THRESHOLD_NOT_BINDING_NEGATIVE') for r in candidates.itertuples()];candidates.to_csv(OUT/'CANDIDATE_REVIEW_BUCKETS_448.csv',index=False)
 shortlist=strong[strong.rank50_ot03_genetic.eq(True)&strong.positive_max_tanimoto.ge(.4)&strong.prior_source_assertion_rows.eq(0)];shortlist.to_csv(OUT/'HIGH_MULTI_EVIDENCE_REVIEW_NOT_LAB_RELEASE.csv',index=False)
 examples=details.merge(pairs[['experiment_id','positive_max_tanimoto','prior_source_assertion_rows']],on='experiment_id');examples=examples[examples.experiment_id.isin(shortlist.experiment_id)&((examples.genetic_association_score>=.1)|(examples.somatic_mutation_score>=.1))].sort_values(['experiment_id','txgnn_rank','ot_score'],ascending=[True,True,False]).drop_duplicates('experiment_id')
 labels={'OUR_FROZEN_MODEL_HIGH':'高排','OUR_FROZEN_MODEL_INTERMEDIATE':'中排','OUR_FROZEN_MODEL_LOW_BACKGROUND':'低排','POSITIVE_CONTROL':'阳性对照'}
 lines=[]
 for role,name in labels.items():
  g=pairs[pairs.role.eq(role)];n=int(g.txgnn_available.sum());lines.append(f"| {name} | {len(g)} | {n} | {int(g.rank50_ot01.fillna(False).sum())} | {int(g.rank50_ot03.fillna(False).sum())} | {int(g.rank50_ot05.fillna(False).sum())} |")
 examples_text='\n'.join(f"| {r.drug_names}—{r.gene_symbol} | {r.ot_disease_name} | {int(r.txgnn_rank)} | {r.ot_score:.3f} | {r.positive_max_tanimoto:.3f} |" for r in examples.itertuples())
 doc=ROOT/'docs/BIOMASTER_SPR512_TXGNN_OT_JOINT_EFFECT_20260909_ZH.md'
 doc.write_text('''# 512对TxGNN × Open Targets联合证据效果检查

## 结论

**能筛出一小批疾病线索更集中的配对，但本轮没有证明融合提高了结合预测准确率或实验命中率。** 448个候选中54对满足本轮主要联合条件，其中高排44对、中排5对、低排5对。不是402对阳性，也不是54对已验证结合。

本次重新读取完整TxGNN分数矩阵、候选排除遮罩及64靶点全部71,755条OT关联，计算全部可映射的疾病交集，未只取上一轮展示用的前三条，未重新训练。矩阵来自已经执行并通过原生接口核对的前向推理；不重复计算相同权重和输入的神经网络分数。

## 1. “结合得好”按什么算

主要探索条件：同一可对齐疾病进入该药TxGNN候选疾病前50，且靶点OT整体关联分数≥0.3。以0.1、0.5阈值和Top100做敏感性检查，不是经过验证或预注册的临床/结合阈值。TxGNN排名只用于同一药物内；不把不同药的logit当成可直接比较的校准概率。

候选排除旧图已知适应证/off-label/禁忌及既有保守新图标记，身份暂扣项不进入候选线索。对照允许显示已有适应证，因此不能把对照组比例与候选组做同条件性能对比。

| 分组 | 原配对数 | 有可用药物推理 | Top50 + OT≥0.1 | Top50 + OT≥0.3 | Top50 + OT≥0.5 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(lines)+f'''

主要条件下，高排组是44/320；只计算有推理的配对，则为44/287=15.3%。中排5/59=8.5%，低排5/56=8.9%。分母应同时报告；缺失药物不等于阴性。阈值从0.1升到0.5，高排结果从92降至14，说明单一“有交集”对阈值非常敏感。

44个高排联合线索中，限制为精确单MONDO节点而不是合并节点成员后剩36对；另要求存在遗传关联或体细胞突变分数≥0.1时为20对。这两个条件分别测试，不能将36和20误作逐层嵌套筛选。遗传证据仍未确定抑制/激动方向，也不自动独立于训练资料。

## 2. 是否比随机配对更有意义

做2,000次探索性药物重配对：在108个可映射候选药中置换整条药物疾病画像，同一药的多行一起移动，保留靶点和高中低角色配置。然后重新计算联合条件。

高排组观察率15.3%，重配对均值17.2%，其中心95%区间约13.6%–21.3%；原配对没有显示超出这一粗基线的增强。遗传/体细胞支持条件下，观察率7.0%，基线约6.5%，也没有明显超出。

**这是提醒疾病交集可能由常见疾病、靶点关联密度、药物类别和共享来源产生，不是证明这些候选无效。** 重配对未重新执行原药靶已知结合排除、化学/实验设计限制，不是严格匹配的生物学阴性对照；此外做了多阈值探索，不能将输出的尾部比例当作确认性显著性结论。多行共用药物/靶点也不能当成独立实验样本。

## 3. 哪些具体配对值得优先核查

44个高排联合线索里，只有{int(strong.positive_max_tanimoto.ge(.4).sum())}对同时有最近阳性参考Tanimoto≥0.4；{int(strong.prior_source_assertion_rows.gt(0).sum())}对已有需核查的来源断言。化学相似度也是描述性参考，不是留出测试。

再同时要求遗传/体细胞支持、Tanimoto≥0.4、无本轮检出的既有非文本来源断言，得到{len(shortlist)}对“多证据优先核查”，**并非自动送实验名单**：

| 分子—靶点 | 一条符合条件的疾病线索 | 药内疾病排名 | OT分数 | 最近阳性Tanimoto |
|---|---|---:|---:|---:|
{examples_text}

这些例子可能仍只反映已有药理领域，而不能解释新靶点是否真实结合。例如ribociclib—PARP1的肿瘤疾病线索不能证明PARP1直接结合；F2相关疾病线索也没有自动给出干预方向。逐条来源及精确/合并节点状态见明细。

ulipristal—AR虽同时有较高化学相似度和联合疾病线索，但已有DrugBank物理相互作用标注，需先核查是否属于已有关系，不应按全新发现优先。

反例：belinostat—NAMPT要到药内疾病排名13,146才出现OT≥0.3的交集，zonisamide—EPHX2为13,003。它们不能因为“可以找出某个共同疾病”就被称作高联合支持。

## 4. 对实验安排的实际建议

保留原448候选、64对照及角色，不以本轮后验筛选改写原盲测性能分母。54个联合线索作为疾病合理性复核入口；其中4个多证据配对优先查原始结合/作用方向和测试物种。其余配对缺少该条件下的疾病支持，不意味着不结合；有疾病支持也不意味着更可能结合。

原274/320高排的低阳性化学支持、构建和对照未确认、Ponesimod身份等问题仍需解决。现在更合理的结论是：**TxGNN×OT可做候选说明和复核，尚不足以单独决定把宝贵的实验预算投给谁。**

## 5. 交付

[联合效果工作簿](../outputs/spr512_txgnn_ot_effect_20260909/SPR512_TXGNN_OT_EFFECT_REVIEW.xlsx)包括512逐对结果、高中低组对比、药物重配对基线及全部符合主要条件的疾病证据。

- `PAIR_RESULTS_512.csv`：缺失与条件不满足分别保存。
- `CANDIDATE_REVIEW_BUCKETS_448.csv`：54个主要条件线索、348个有推理但不满足条件、46个无可用推理。
- `HIGH_MULTI_EVIDENCE_REVIEW_NOT_LAB_RELEASE.csv`：多证据优先核查表，不是实验放行。
- `JOINT_DISEASE_DETAILS.csv.gz`：每条的原始分数、排名、OT数据类型和映射范围。
- `DRUG_REASSIGNMENT_BASELINE.csv`：探索性基线，不是结合模型性能。
- `VALIDATION.json`：输入哈希、分母及缺失/阈值一致性检查。

复现：`python scripts/analyze_spr512_txgnn_ot_effect_20260909.py`，再运行 `python scripts/report_spr512_txgnn_ot_effect_20260909.py`。没有训练，没有宣称前瞻有效性。
''')
 design=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md';before=OUT/'DESIGN_BEFORE_JOINT_EFFECT.md'
 if not before.exists():before.write_bytes(design.read_bytes())
 marker='## 9. TxGNN × Open Targets联合效果检查'
 design.write_text(design.read_text().split(marker)[0].rstrip()+f'''\n\n{marker}

已完成[全部疾病交集与探索性基线分析](BIOMASTER_SPR512_TXGNN_OT_JOINT_EFFECT_20260909_ZH.md)。主要条件为药内TxGNN前50、同疾病OT≥0.3：高排44/320、中排5/64、低排5/64。只计算可推理配对时，高排15.3%，中排8.5%，低排8.9%。

高排结果没有明显超过药物重配对粗基线（均值17.2%）；该基线不是严格生物学阴性对照，不能由此判定候选无效。当前未证明联合证据提高结合预测或实验命中率。

44个高排联合线索中10对另有Tanimoto≥0.4；同时要求遗传/体细胞支持且未检出既有非文本来源断言后，{len(shortlist)}对进入多证据优先核查表。不是实验放行名单，原配对和角色不变。新增报告版本哈希关联见 `outputs/spr512_txgnn_ot_effect_20260909/MANIFEST.json`。
''')
 manifest={'no_training':True,'experimental_performance_measured':False,'parent_design_sha256':sha(before),'docs':{str(p.relative_to(ROOT)):sha(p) for p in [doc,design]},'files':{str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'},'scripts':{p.name:sha(p) for p in [ROOT/'scripts/analyze_spr512_txgnn_ot_effect_20260909.py',Path(__file__)]}}
 (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(doc);print(candidates.review_bucket.value_counts().to_dict());print(examples[['drug_names','gene_symbol','ot_disease_name','txgnn_rank','ot_score']].to_string(index=False))
if __name__=='__main__':main()
