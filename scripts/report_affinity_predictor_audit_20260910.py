#!/usr/bin/env python3
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_affinity_refresh_predictor_20260910 import metrics,MODELS
OUT=ROOT/'outputs/affinity_predictor_audit_20260910'
def main():
 c=pd.read_csv(OUT/'LABEL_COUNTS.csv');m=pd.read_csv(OUT/'PREDICTOR_METRICS.csv');a=pd.read_csv(OUT/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz',low_memory=False)
 ci=json.loads((OUT/'CLUSTER_BOOTSTRAP_CI.json').read_text());manifest=json.loads((OUT/'SUMMARY.json').read_text())
 k=a[(a.endpoint_group=='Kd_Ki')&a.refresh_endpoint_covered&~a.local_training_connectivity_pair]
 train=k[k.binary_label.notna()].copy();train['use_status']='CANDIDATE_ONLY_REVIEW_ASSAY_AND_FREEZE_INDEPENDENT_EVALUATION_BEFORE_TRAINING'
 train.to_csv(OUT/'SUPPLEMENTAL_KD_KI_LABEL_CANDIDATES.csv',index=False,encoding='utf-8-sig')
 g=k[k.scored&k.binary_label.notna()].copy()
 sens=[]
 for name,z in [('TRAIN_PAIR_UNSEEN_IDENTITY_CHEMISTRY_PASS',g[g.identity_scope_pass.eq(True)&g.chemistry_policy_pass.eq(True)]),('NO_PRIOR_RELATION_IDENTITY_CHEMISTRY_PASS',g[g.identity_scope_pass.eq(True)&g.chemistry_policy_pass.eq(True)&g.known_relation_excluded.eq(False)&~g.existing_ChEMBL37_numeric_pair])]:
  for model in MODELS:sens.append(dict(slice=name,**metrics(z,model)))
 pd.DataFrame(sens).to_csv(OUT/'ELIGIBILITY_SENSITIVITY_METRICS.csv',index=False,encoding='utf-8-sig')
 mainm=m[(m.endpoint_group=='Kd_Ki')&(m.slice=='REFRESH_NO_LOCAL_TRAIN_PAIR')].set_index('model')
 # ROC and PR on the same locally pair-unseen retrospective cohort.
 fig,axes=plt.subplots(1,2,figsize=(11,4.3))
 names={'SPR_current':'Current SPR ranker','contract_primary':'Contract primary Borda','neural_full_fit':'Full-fit neural head'}
 for model,label in names.items():
  y=g.binary_label.to_numpy();v=g[MODELS[model]].to_numpy();fpr,tpr,_=roc_curve(y,v);prec,rec,_=precision_recall_curve(y,v)
  axes[0].plot(fpr,tpr,label=f'{label} ({mainm.loc[model,"auroc"]:.3f})')
  axes[1].plot(rec,prec,label=f'{label} ({mainm.loc[model,"average_precision"]:.3f})')
 axes[0].plot([0,1],[0,1],'k--',alpha=.4);axes[1].axhline(g.binary_label.mean(),color='k',ls='--',alpha=.4,label=f'Prevalence ({g.binary_label.mean():.3f})')
 axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='ROC (legend: AUROC)');axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall (legend: AP)')
 for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.grid(alpha=.15)
 fig.suptitle(f'Kd/Ki retrospective audit: {len(g)} pairs, {int(g.binary_label.sum())} positive, {int((g.binary_label==0).sum())} weak negative\nLocal training-pair overlap excluded; not a prospective SPR hit-rate estimate',fontsize=11)
 fig.tight_layout();fig.savefig(OUT/'FROZEN_PREDICTOR_ROC_PR.png',dpi=180);plt.close(fig)
 def counts(ep,sl,scope):return c[(c.endpoint_group==ep)&(c.slice==sl)&(c.scope==scope)].iloc[0]
 refresh=counts('Kd_Ki','REFRESH_COVERED','720x888');unseen=counts('Kd_Ki','REFRESH_NO_LOCAL_TRAIN_PAIR','720x888');novel=counts('Kd_Ki','REFRESH_NO_PRIOR_RELATION','720x384')
 current=mainm.loc['SPR_current'];currci=next(x for x in ci if x['endpoint_group']=='Kd_Ki' and x['slice']=='REFRESH_NO_LOCAL_TRAIN_PAIR')
 lines=['# 新增证据标签、训练重叠与预测器回顾性评估（2026-09-10）','',
 '**结论：一部分证据可进入下一轮补充训练候选池，但不能将入库行数当作新训练样本数。现用SPR排序器在本地训练配对未出现的回顾性Kd/Ki样本上有区分能力；对“新药物—新靶点关系”的发现能力，现有数据不足以确认。** 本轮未训练、未调整模型权重、未改变384或实验优先级。', '',
 '## 1. 阳性/阴性如何定义','',
 '- 正类：Kd/Ki ≤1,000 nM；弱结合负类：Kd/Ki ≥10,000 nM。负类不是“任何浓度均不结合”。',
 '- 1–10 µM区间为灰区；无法据上下界判断的记录和约值为未决；同一配对出现明确正、负证据即列冲突，不以最强一条或多数票定标签。',
 '- 仅纳入浓度单位可解析、精确分子/蛋白映射及上一轮注释规则通过的数值；原始源的实验构建并未全部逐文献验证。',
 '- IC50、EC50另行统计；混合端点表仅用于描述总体活性，不等同于纯亲和评估。Kd_app、kinobead响应、未报告关系不作为普通亲和负例。',
 '- 同一对多来源/重复结构不重复计配对。下表标签结合既有ChEMBL记录做冲突复核，而非只挑新来源的阳性。', '',
 '## 2. 可见配对数与补充训练候选','',
 '|范围（Kd/Ki）|合计|正类|弱结合负类|灰区|冲突|未决|','|---|---:|---:|---:|---:|---:|---:|']
 for label,r in [('本轮来源覆盖，720×888',refresh),('剔除本地训练同配对/同连接骨架配对，720×888',unseen),('剔除本地训练同配对/同连接骨架配对，有现成评分',counts('Kd_Ki','REFRESH_NO_LOCAL_TRAIN_PAIR','720x384')),('再剔除既有ChEMBL和原已知关系旗标，有评分',novel)]:
  lines.append('|'+label+'|'+'|'.join(str(int(r[x])) for x in ['total_pairs','POSITIVE','WEAK_NEGATIVE','GREY','CONFLICT','UNRESOLVED'])+'|')
 lines += ['',f"因此，当前可作为**补充标签候选**的是 {len(train)} 对：{int(train.binary_label.sum())}正、{int((train.binary_label==0).sum())}弱负；尚需assay/构建审查及冻结独立评估集后再决定训练用途。其余灰区、冲突和未决不能直接进入普通二分类。训练合同中的affinity-only连续标签也不应因本次评价而强行改成binary训练标签。",'',
 '“模型未见过”在此仅指：未出现在已审计的两版综合训练表、历史capped benchmark、BindingDB affinity补充表中的同配对/同连接骨架配对。不是全新药物、冷靶点、全新骨架或外部模型预训练零泄漏的证明。训练表所有角色都保守计入见过；详见`SUMMARY.json`输入清单与SHA256。', '',
 '不能把上一轮“相对ChEMBL新增490对”与本轮“评估样本数”混为一谈：前者按来源覆盖差集计数，后者按训练重叠、端点、冲突及质量规则计数。两个数字偶然接近不代表同一集合。', '',
 '## 3. 预测器性能：统一样本、冻结分数','',
 f"主分析为 {len(g)} 个可二分、已有评分、本地训练配对未出现的Kd/Ki配对：{int(g.binary_label.sum())}正、{int((g.binary_label==0).sum())}弱负，覆盖{g.ligand_inchikey.nunique()}种药和{g.target_chembl_id.nunique()}个靶点。随机排序AP参考值为正类比例 {g.binary_label.mean():.3f}。模型分数没有做概率校准，不报告0.5阈值accuracy，也不将排序分数计算成Kd回归RMSE。",'',
 '|冻结预测器|AUROC|AP|药物内宏平均AUROC|完整384轴Top20捕获已观察正类|Top20内已观察弱负|','|---|---:|---:|---:|---:|---:|']
 for model in MODELS:
  r=mainm.loc[model];lines.append(f"|{names.get(model,model)}|{r.auroc:.3f}|{r.average_precision:.3f}|{r.within_drug_macro_auroc:.3f}|{int(r.top20_observed_positive)}/{int(r.positive)}|{int(r.top20_observed_weak_negative)}|")
 lines += ['',f"现用SPR分数的药物分组bootstrap（1,000次）：AUROC 95%区间 **{currci['auroc_ci95'][0]:.3f}–{currci['auroc_ci95'][1]:.3f}**；AP区间 {currci['ap_ci95'][0]:.3f}–{currci['ap_ci95'][1]:.3f}。药物内AUROC仅有 {int(current.within_drug_both_class_queries)} 个同时具备正/负类的药物可计算，不能代表全部药物。",'',
 f"Top20的“已观察正类率”为{int(current.top20_observed_positive)}/({int(current.top20_observed_positive)}+{int(current.top20_observed_weak_negative)})，分母只含有合格实测标签的Top20配对。绝不能用它宣称全部Top20或384的实验命中率；未知配对没有计入负类。",'',
 'FULL_FIT神经头在这张回顾性表的全局AUROC/AP更高；现用SPR排序器的药物内宏平均AUROC更高。这说明“全局区分已测试配对”和“每药排靶点”并不是同一任务，不能仅按一个指标直接更换生产模型。外部DTIAM的预训练来源未完全审计，成绩仅列对照。', '',
 '纯Kd敏感性分析、IC50/EC50、全部已知混合集、局部训练药物未见切片见`PREDICTOR_METRICS.csv`。加上既有训练关系后的高分仅是回顾性一致性，不能用作独立外部性能。身份/化学规则进一步过滤的结果见`ELIGIBILITY_SENSITIVITY_METRICS.csv`。', '',
 '## 4. 更接近新关系发现的切片仍不足','',
 f"进一步要求此前没有已知关系旗标、无既有ChEMBL数值且无本地训练同配对后，Kd/Ki仅有{int(novel.total_pairs)}对：{int(novel.POSITIVE)}正、{int(novel.WEAK_NEGATIVE)}弱负、{int(novel.GREY)}灰区。二分类只有{int(novel.POSITIVE+novel.WEAK_NEGATIVE)}对。正类太少，不给确认性的AUROC结论或bootstrap区间。明细表里的点估计只用于调试，尤其不能把神经头的1.0解释成完美泛化。",'',
 '剩余正类为clotrimazole–CYP46A1，当前SPR排名13/384，HiQBind列Kd=4 nM；PDB 3MDV确认该分子–蛋白结构且无标注突变，但数值标签的assay条件仍需要原论文核验。[RCSB 3MDV](https://www.rcsb.org/structure/3MDV)', '',
 '这次实验数据仍未覆盖冻结384的精确配对。因此目前证据支持“排序器有一定回顾性筛选信号”，不支持“384的成功率已得到验证”。', '',
 '## 5. 数据质量复核发现及处理','',
 '本轮完整保留定向复核前后结果。以下两条不再进入当前主评估，原始数据没有删除：', '',
 '1. **BindingDB 118672:Kd，nevirapine–EPHA2，13 nM：分子映射错误。** 原专利Table 2中的NVP指NVP-BHG712；原文实验部分与表格均可对应，不能把其Kd转给奈韦拉平。该行Article DOI还指向另一篇论文。已写入数据库质量审查表并排除训练/评估。[专利原文](https://patents.google.com/patent/US12486270B2/en)',
 '2. **HiQBind 18434，注册显示ciclesonide–NR3C2，Ki=0.18 nM：活性实体与构建待审。** 精确结构实际为desisobutyryl ciclesonide，原PDB MR含3处突变及共调节肽。不能直接当作给药原药—野生型靶点标签；此条先隔离。[RCSB 4UDB](https://www.rcsb.org/structure/4UDB)', '',
 '这不是全量逐论文QC已完成的声明：定向复核聚焦更接近新关系切片的关键阳性，其他记录仍为来源报告标签。旧的入库报告统计的是源注释覆盖，本报告给出质量过滤后的可评价口径。相比上一轮覆盖数字，当前审查视图剔除了2条来源记录/2对；冻结384的0覆盖结论不变。', '',
 '## 6. 现在怎么使用','',
 '1. **先冻结这次评价和来源审查，不立刻训练。** 否则本批数据也会变成已见数据，后续无法再作为同口径独立比较。',
 '2. 把931对数值标签候选（以实际导出计数为准）作为下一轮补充数据准备集，分离Kd与Ki、实验条件、正负冲突，保留原始连续值和上下界。按药物/骨架/文献来源分组设计训练与评估划分，不能随机拆来源重复行。',
 '3. 模型改进可优先比较现用排序器与FULL_FIT神经头的互补性，但不能用这次看过的成绩调参后仍称独立外部验证。先冻结下一批未使用的对照/评估数据再选择融合方案。',
 '4. 现有384继续保持冻结。实验首批用于校准真实阳性率和非特异结合率，并把真实SPR结果作为下一轮最有价值的前瞻数据。', '',
 '## 7. 产物与复现','',
 '- `LABEL_COUNTS.csv`：各端点、来源/训练重叠切片的正负灰冲突数。',
 '- `PREDICTOR_METRICS.csv`：5个冻结预测器的同样本比较；`CLUSTER_BOOTSTRAP_CI.json`：按药物抽样区间。',
 '- `PAIR_LABELS_AND_FROZEN_SCORES.csv.gz`：逐对标签、训练重叠、分数和完整384轴排名。',
 '- `SUPPLEMENTAL_KD_KI_LABEL_CANDIDATES.csv`：未进入本地训练表的正/弱负补充候选；非自动训练发布清单。',
 '- `NOVEL_KD_KI_PAIR_AUDIT.csv`和`NOVEL_PAIR_SOURCE_EVIDENCE.csv.gz`：接近新关系切片及出处。',
 '- `before_targeted_source_qc/`：清洗前审计成绩，不作为当前有效成绩。',
 '- `source_verification/`：原专利及PDB元数据；`data/external/affinity_refresh_20260910/SOURCE_RECORD_QC_OVERRIDES_20260910.csv`：可重建的质量覆盖规则。',
 '- SQLite新增`evidence_quality_reviews`与`reviewed_comparable_numeric_evidence`；查询接口同时返回quality_review。原测量和原覆盖视图保留，以免历史审计口径被悄悄替换。',
 '', '复现：`python scripts/evaluate_affinity_refresh_predictor_20260910.py`，再运行`python scripts/report_affinity_predictor_audit_20260910.py`。20项单位/上下界/冲突/重复不变性等测试通过；模型分数和冻结384未修改。', '']
 (ROOT/'docs/BIOMASTER_AFFINITY_PREDICTOR_VALIDATION_20260910_ZH.md').write_text('\n'.join(lines))
 result=dict(refreshed_Kd_Ki_counts=refresh.to_dict(),new_local_training_candidate_Kd_Ki_counts=unseen.to_dict(),main_cohort=dict(pairs=len(g),positive=int(g.binary_label.sum()),weak_negative=int((g.binary_label==0).sum()),drugs=g.ligand_inchikey.nunique(),targets=g.target_chembl_id.nunique()),main_metrics=mainm.reset_index().to_dict('records'),strict_novel_counts=novel.to_dict())
 (OUT/'REPORT_SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('REPORT_COMPLETE',len(train),len(g))
if __name__=='__main__':main()
