"""Publish review artifacts for the comprehensive 384, preserving earlier designs."""
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint384_comprehensive_20260909';DOC=ROOT/'docs/BIOMASTER_RECOMMENDED384_COMPREHENSIVE_20260909_ZH.md';ACTIVE=ROOT/'docs/BIOMASTER_JOINT384_AGENT_REVIEW_20260909_ZH.md';DESIGN=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def table(d):
 def c(x):return str(x).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(map(c,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(c,r))+' |' for r in d.itertuples(index=False,name=None))
def main():
 val=json.loads((OUT/'VALIDATION.json').read_text());assert val['all_pass'];s=json.loads((OUT/'SUMMARY.json').read_text());p=pd.read_csv(OUT/'RECOMMENDED_CANDIDATES_384.csv');a=pd.read_csv(OUT/'ALL574_FINAL_DISPOSITION.csv');old=a[a.in_previous384];roster=pd.read_csv(OUT/'TARGET_ROSTER.csv');ctrl=pd.read_csv(OUT/'REFERENCE_CONTROLS_EXTRA.csv');budgets=pd.read_csv(OUT/'TARGET_BUDGET_COMPARISON.csv');sens=pd.read_csv(OUT/'WEIGHT_SENSITIVITY.csv')
 compare=pd.DataFrame([{'指标':'原始ChEMBL已有活动记录','旧384':int(old.raw_activity_rows.gt(0).sum()),'新384':int(p.raw_activity_rows.gt(0).sum())},{'指标':'按本轮同一标签口径判定跨大类','旧384':int(old.has_label_based_cross_area.sum()),'新384':int(p.has_label_based_cross_area.sum())},{'指标':'同时存在精确疾病与遗传/体细胞证据','旧384':int(old.any_exact_genetic.sum()),'新384':int(p.any_exact_genetic.sum())},{'指标':'未触发参照转移风险的阳性近邻>=0.4','旧384':int(old.usable_chemical_support_ge04.sum()),'新384':int(p.usable_chemical_support_ge04.sum())}]);compare.to_csv(OUT/'FAIR_OLD_NEW_COMPARISON.csv',index=False)
 changes=pd.read_csv(OUT/'PREVIOUS384_CHANGE_LEDGER.csv');removed=changes[changes.change.eq('REMOVED_FROM_NEW_VERSION')];removal=removed.groupby('final_disposition').size().rename('pairs').reset_index();removal.to_csv(OUT/'REPLACEMENT_REASON_COUNTS.csv',index=False)
 # Keep experimental sequencing as a proposal. It never marks a construct/activity QC as passed.
 stage=p[['candidate_id','pair_id','modeled_entity_name','drug_names','gene_symbol','target_chembl_id','recommendation_band','evidence_utility_unvalidated']].copy()
 stage['within_target_review_order']=stage.groupby('target_chembl_id',sort=False).evidence_utility_unvalidated.rank(method='first',ascending=False).astype(int)
 stage['execution_condition']='Qualified matching reference, construct/activity, compound identity/solubility and buffer stability required before interpretation.'
 stage['proposed_stage']=stage.recommendation_band.map({'PRIORITY_CROSS_AREA_MECHANISM_REVIEW':'FIRST_MECHANISM_SCOUT_AFTER_QC','ADDITIONAL_SOURCE_OR_CHEMICAL_SUPPORT':'FOLLOW_UP_AFTER_TARGET_QC','EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT':'RESERVE_EXPLORATORY_EXPANSION_AFTER_INITIAL_RESULTS'})
 stage.to_csv(OUT/'STAGED_EXECUTION_REVIEW.csv',index=False)
 # Independent recomputation of novelty/identity and disease gate against source rows.
 raw=pd.read_csv(OUT/'ALL574_RAW_CHEMBL_ACTIVITY_RECORDS.csv.gz');deep=pd.read_csv(ROOT/'outputs/best_candidate_synthesis_20260909/TARGETED24_DEEP_REVIEW.csv');h=pd.read_csv(OUT/'ALL_COMMON_DISEASE_REAUDIT.csv.gz');source=pd.read_csv(ROOT/'outputs/joint_screen_720x384_20260909/BIOCHEMICAL_LANE_574_REVIEW.csv')
 checks={'not_in_any_raw_activity_record':not bool(set(p.pair_id)&set(raw.pair_id)),'no_deep_DO_NOT_PRIORITIZE':not bool(set(p.pair_id)&set(deep.loc[deep.recommendation.eq('DO_NOT_PRIORITIZE'),'pair_id'])),'all_selected_source_ranks_and_scores_retained':len(p.merge(source[['pair_id','binding_rank_384','independent_validation_rank_score']],on=['pair_id','binding_rank_384','independent_validation_rank_score']))==384,'one_actual_qualifying_cross_disease_per_cross_pair':set(p.loc[p.has_label_based_cross_area,'pair_id'])<=set(h.loc[h.new_cross_area&~h.disease_policy_hold&h.txgnn_rank.le(50)&h.overall_score.ge(.3),'pair_id']),'isotope_and_covalent_form_holds_absent':not p.ligand_inchikey.isin(['PDWUPXJEEYOOTR-UHFFFAOYSA-N','HKDLNTKNLJPAIY-WKWWZUSTSA-N']).any(),'case_mebendazole_ESR1_absent':not((p.drug_names=='mebendazole')&(p.gene_symbol=='ESR1')).any(),'case_baricitinib_HDAC6_absent':not((p.drug_names=='baricitinib')&(p.gene_symbol=='HDAC6')).any()}
 checks={k:bool(v) for k,v in checks.items()};(OUT/'INDEPENDENT_VALIDATION.json').write_text(json.dumps({'all_pass':all(checks.values()),'checks':checks},indent=2));assert all(checks.values()),checks
 displaybud=budgets[['target_count','target_cap','solver_status','candidates','integrated_priority','chemical_ge04','cross_area_pairs','exact_genetic_pairs','evidence_utility']].fillna('不可行')
 report=f'''# 综合推荐384候选（原用途、原始活性、跨大类与实验约束整合版）

日期：2026-09-09。用户目标为**384个候选，对照另计**。建议使用本版进行后续审查与预算安排；原384和原512保留作版本对照，没有覆盖删除。没有训练模型、没有用预测分数冒充实测结合。

## 1. 当前推荐与核心结论

本版为**384个候选／{s['drugs']}个药物结构实体／{s['targets']}靶点**，另列{s['extra_controls']}参考对照提案，合计{s['total_pairs_including_controls']}个分子—靶点配对。实际实验孔数仍取决于浓度、重复、空白与溶剂对照。

- 保留旧384中的{s['same_old_pairs']}对，替换{s['replaced_pairs']}对；每项去向均有记录。
- 按本轮已阅读官方标签／监管来源的用途大类，{s['cross_area_pairs']}对（{s['cross_area_pairs']/384:.1%}）有至少一个精确疾病节点的跨大类假设。它们不是已经验证的新适应症，来源覆盖也不是完整全球获批史。
- {s['exact_genetic_or_somatic_pairs']}对有达标的精确疾病节点与遗传／体细胞来源支持；遗传和体细胞分别记录，疾病关联不证明药物结合。
- {s['raw_chemical_similarity_ge04']}对阳性参考Tanimoto≥0.4；其中{ s['raw_chemical_similarity_ge04']-s['chemical_support_ge04'] }对因药效团拼接／参照转移问题没有获得化学支持加分，实际计入{s['chemical_support_ge04']}对。
- 当前同时满足较强跨领域、化学支持且未被深核降级的首选仍只有**tecovirimat–AR**，适合先做单靶点侦察；并未证实结合或抗癌效果。pregabalin–MME保留条件备选，其余条件项在表中明确列出。

**这份384是现有数据和明确约束下的综合探索组合，不是384个高置信阳性。** {s['bands']['EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT']}对仍缺较强的额外化学／精确遗传支持，单列为探索扩展层。在保留当前Top20+TxGNN/OT门槛并严格移走已测关系后，只有{s['eligible_pairs']}对可入池；要选384就会纳入大量探索位。不能用优化分数把这个事实掩盖成高把握。

## 2. 本轮补齐了哪些信息

**原用途**：对266个药物名称/模型实体做分片审阅，217条基于抓取标签，48条另查官方来源，1项未解决。数据包括openFDA API标签、FDA/DailyMed及EMA等补充来源；逐药归类不是只做关键词扫描。原用途含多适应症、预防和必要的辅助场景，单药、复方、历史撤回、剂型及相关结构限制记录在`scope_limits`。

openFDA从公开结构化标签提供`indications_and_usage`等字段；API名称或申请号命中不是单药结构／完整适应症核验。实际发现部分generic_name元数据命中仍带复方正文，均交给逐药阅读处理；不能仅凭API字段把复方效果归给单药。[openFDA标签字段](https://open.fda.gov/apis/drug/label/searchable-fields/)。例如amisulpride、glycopyrronium、timolol、varenicline、alpelisib、tadalafil、colchicine等补齐了会影响跨领域判断的用途。265条有来源不等于265个模型结构均与所有标签制剂等价。

**原始端点**：对全部574个生化配对重新只读查询完整ChEMBL37活动表，纳入百分比抑制、检测限、阴性、细胞／功能和缺pChEMBL记录，同时记录完整键及其父结构关联制剂范围。共69对、131条原始活动记录；17对有高置信直接靶点注释。69不是69个阳性，也不是69个直接Kd；本轮出于新关系实验预算政策，已有原始配对记录先不占新颖性候选位。来源查新仍限既定本地快照和补查范围，不等于全球首次。

**深核**：整合项目已有24对重点深入审查，避免旧快审标签覆盖更晚发现。特别是mebendazole–ESR1、baricitinib–HDAC6、articaine–ACHE等已有负面／百分比／检测限记录，不能继续拿图谱交集维持优先。mebendazole–ESR1的AC50>10µM不是10µM阳性；baricitinib–HDAC6的百分比读数也不能当Kd。原始activity_id、条件、来源与判断全部保留在原始记录表和深核字段。

**实体问题**：未标同位素的3-iodobenzylguanidine不能直接继承I-131肿瘤产品用途；模型去乙酰ulipristal不是获批ulipristal acetate的简单去盐形式。这类问题传播到该分子的所有目标配对，而不是只暂停被深入审查的那一个靶点。新输出保留FDA模型实体名、精确成分名及完整InChIKey，合并别名不作为采购身份。

## 3. 门禁与选择方式

所有候选继续满足原完整384靶点内结合Top20、排除后TxGNN疾病Top50、同一疾病OT≥0.3；此前身份、化学、MOA／家族／复合物、BindingDB、GtoPdb、EC来源和训练配对规则保留。没有通过放宽这些门槛凑数。

新增排除包括完整原始活动、深核投资暂停、跨靶点传播的结构身份问题，以及raloxifene/bazedoxifene只剩内异症的疾病投资方向暂停。后者区分不利RCT与已有复方探索，不意味着两个药都不能结合KDR。

对可入池的{s['eligible_pairs']}对，取消旧agent `KEEP_REVIEW`直接加分，也不因其触发旧优先标签额外加50分。统一使用可追溯的化学近邻、同病精确遗传／体细胞、原结合排名、跨领域、化学风险和更近阴性参考。复合/衍生分子的关键药效团不能直接转移给母药，因此相关Tanimoto保留但不作为结合支持加分。

本版效用权重是透明的项目偏好，不是学习得到的成功概率：可用T≥0.4 +4，T0.3–0.4且无参照转移问题+1.5；精确疾病遗传/体细胞+3；精确疾病+0.5；结合排名1–5/6–10/11–20分别+2/+1.5/+1；标签范围跨大类+2，其同一跨类疾病另有遗传/体细胞+1；化学风险点−0.75/点，更近的T≥0.4阴性参考−1.5。综合首选标记额外+20，本轮只有1项且全部可行方案均保留；这不是一般性的词典序最优保证。

## 4. 为什么最终是{s['targets']}个靶点

比较了靶点准备数量与重复分配上限，不用固定旧配额去挤掉有效新颖性过滤：

{table(displaybud)}

每药/连接结构最多4对、至少128个药物结构实体；常规每靶点上限10，另检查12的放宽方案。64、80及96×上限10不可行，**仅指当前候选池与这些联合约束下不可行**，不代表这些靶点数量在生物学上不可能。

96靶点、上限12可以凑到384，但只能保留19对可用化学支持、117对精确遗传/体细胞；112靶点、上限10保留20对和126对，并降低单靶点集中度。资源规则先保留可行方案中最多综合首选及可用化学支持，再在达到最高声明效用95%的方案中优先减少靶点数、随后降低单靶点上限。本轮因此选112／上限10，而非96／上限12。95%是资源取舍阈值，不是置信区间或实际命中率差异。

同时输出偏结合、偏跨领域两套权重敏感性方案，分别与本推荐重叠{int(sens.iloc[0].overlap_with_recommended)}、{int(sens.iloc[1].overlap_with_recommended)}对，三套共同保留{s['stable_in_three_weight_settings']}对。稳定性只描述该候选池内选择对权重的敏感程度；固定384占可入池的大多数，本来就会提高重叠，不能当独立模型一致性或真实药效证据。

## 5. 与旧384的公平比较

{table(compare)}

旧报告99条跨领域线索主要依赖图谱既往用途代理；本轮把**旧384也用同一组新原用途资料重算**后为193条，新384为195条。因此不能把99→195写成筛选性能翻倍。新版本的主要改进是移走已测试关系、实体不符和深核不支持的配对，同时为余下假设补齐来源；不是单纯增加图谱交集率。

83对替换原因如下（互斥分类）：

{table(removal)}

原384文件完全保留。未入选不一律代表不能结合：部分属于预算／分布备选；已有测试排除是新颖性政策；深核暂停则有不同机制、暴露或实验可行性原因。每条详见`PREVIOUS384_CHANGE_LEDGER.csv`和`ALL574_FINAL_DISPOSITION.csv`。

## 6. 如何使用这384

| 层级 | 数量 | 用途 |
| --- | ---: | --- |
| 跨领域机制优先侦察 | {s['priority_pairs']} | tecovirimat–AR，先做适配构建的直接结合侦察 |
| 另有化学或精确遗传/体细胞来源 | {s['bands']['ADDITIONAL_SOURCE_OR_CHEMICAL_SUPPORT']} | 可优先复核，但疾病遗传不能替代药物结合 |
| 额外支持较薄的探索扩展 | {s['bands']['EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT']} | 明确作为探索，不与上两层同等投入 |

旧agent标签在新384中有{s['legacy_agent_decisions'].get('LOW_PRIORITY',0)}对LOW_PRIORITY，仍保留供追溯，但不同agent尺度不再直接决定效用；新的统一分层与旧“175对低优先”不是同一统计口径。

参考对照另计112对：90个有Kd记录、18个Ki无Kd、4个只有其他功能端点。后4个是CYP11B1、UGCG、CYP11B2、CYP26A1，**必须先补直接结合适用性或验证合适参考物**。任何混合端点pChEMBL均不当SPR Kd；所有参考仍是提案，不保证可采购或适配当前构建。

实际顺序建议：先确认靶点构建、活化态、辅因子及匹配参考物；通过QC后先处理该靶点较有依据的候选；再根据首轮结果扩展探索层。化合物形态、溶解性、聚集、工作液稳定性和目标方向均不能由图谱分数替代。没有在本轮标记任何实际QC通过、临床有效或实验放行。

## 7. 交付与核验

- [综合推荐384工作簿](../outputs/joint384_comprehensive_20260909/RECOMMENDED384_COMPREHENSIVE_REVIEW.xlsx)：384逐对理由、原用途、目标疾病、化学/基因证据、原始深核意见、来源及配额。
- [候选384 CSV](../outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv)、[另计参考对照](../outputs/joint384_comprehensive_20260909/REFERENCE_CONTROLS_EXTRA.csv)、[靶点配额](../outputs/joint384_comprehensive_20260909/TARGET_ROSTER.csv)。
- [新旧替换明细](../outputs/joint384_comprehensive_20260909/PREVIOUS384_CHANGE_LEDGER.csv)、[阶段性执行审查](../outputs/joint384_comprehensive_20260909/STAGED_EXECUTION_REVIEW.csv)。
- `ORIGIN_REVIEWED_266.csv`与各分片：原用途及来源边界；原始API响应另有缓存。`ALL574_RAW_CHEMBL_ACTIVITY_RECORDS.csv.gz`保留131条原始活动、activity_id、单位、关系、assay/doc和来源URL。
- 原始生成及复核代码：`fetch_joint384_official_labels_20260909.py`、`audit_comprehensive_raw_activities_20260909.py`、`build_joint384_comprehensive_20260909.py`及本报告脚本。

主检查{len(val['checks'])}项及独立回读检查{len(checks)}项全部通过：恰好384、来源排名不变、原始既有活动不混入、深核暂停不混入、身份问题传播、跨大类同病与精确节点、无癌症→癌症冒充跨大类、对照与候选无重复、原384和512哈希不变。输入与产物哈希见`MANIFEST.json`，报告和新增文件版本链见`REPORT_MANIFEST.json`。验证的是计算、来源及门禁一致性，不是实验成功率。
'''
 DOC.write_text(report)
 for d,name in [(ACTIVE,'AGENT384_BEFORE_COMPREHENSIVE.md'),(DESIGN,'DESIGN_BEFORE_COMPREHENSIVE.md')]:
  f=OUT/name
  if not f.exists():f.write_bytes(d.read_bytes())
 head='> 当前综合推荐：已生成新的384候选（229药、112靶点，对照另计112）。整合原始全端点、24项深核和266药原用途后替换83对；旧384保持不变。请使用[综合推荐版](BIOMASTER_RECOMMENDED384_COMPREHENSIVE_20260909_ZH.md)；下面保留旧版审查与统计历史。\n\n'
 text=ACTIVE.read_text();section='\n## 10. 综合推荐384新版本'
 # The prior document may have later synthesis sections; append a uniquely titled section without truncation.
 if head not in text:text=text.replace('\n\n','\n\n'+head,1)
 if section not in text:text+=section+'''\n\n已形成[综合推荐384](BIOMASTER_RECOMMENDED384_COMPREHENSIVE_20260909_ZH.md)，229药／112靶点，另112参考对照。保留301旧对、替换83对；195条按本轮标签范围判定的跨大类假设。全574原始ChEMBL端点复审发现69个已测配对，不进入新颖性实验位；另整合24项深核与分子实体问题。当前只有tecovirimat–AR保留首选侦察标记，240对仍为额外支持较薄的探索层。原384／512不变，新文件和各项局限见新报告。\n'''
 ACTIVE.write_text(text)
 section2='\n## 14. 当前综合推荐384候选'
 text=DESIGN.read_text()
 if section2 not in text:text+=section2+'''\n\n按用户再次要求完整384候选，已综合原始活性、新靶点、原用途／跨大类和实验分布形成[新推荐](BIOMASTER_RECOMMENDED384_COMPREHENSIVE_20260909_ZH.md)：384对、229药、112靶点；另112参考对照，合计496配对。严格排除已测与待审，不把旧agent保留标签直接当加分。候选分层、旧384的83项替换及资源方案比较均可追溯；未放行实验，原384和512产物不改写。\n'''
 DESIGN.write_text(text)
 cache={f.name:sha(f) for f in (OUT/'fda_label_cache').glob('*.json.gz')};(OUT/'LABEL_CACHE_MANIFEST.json').write_text(json.dumps({'files':cache,'count':len(cache),'source':'openFDA public labeling endpoint; raw metadata is not identity/approval validation'},indent=2))
 files=[DOC,ACTIVE,DESIGN,OUT/'AGENT384_BEFORE_COMPREHENSIVE.md',OUT/'DESIGN_BEFORE_COMPREHENSIVE.md',OUT/'FAIR_OLD_NEW_COMPARISON.csv',OUT/'REPLACEMENT_REASON_COUNTS.csv',OUT/'STAGED_EXECUTION_REVIEW.csv',OUT/'INDEPENDENT_VALIDATION.json',OUT/'LABEL_CACHE_MANIFEST.json',Path(__file__)]
 (OUT/'REPORT_MANIFEST.json').write_text(json.dumps({'run_manifest_sha256':sha(OUT/'MANIFEST.json'),'artifact_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files}},indent=2))
 print(str(DOC));print(json.dumps({'independent_checks':checks,'cache_records':len(cache)},ensure_ascii=False))
if __name__=='__main__':main()
