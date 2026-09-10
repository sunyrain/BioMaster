"""Document the reviewed 384 candidates, with controls outside the candidate budget."""
import json,hashlib,re
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint384_design_20260909';REV=ROOT/'outputs/joint384_agent_review_20260909';BASE=ROOT/'outputs/joint_screen_720x384_20260909';DOC=ROOT/'docs/BIOMASTER_JOINT384_AGENT_REVIEW_20260909_ZH.md';DESIGN=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(d):
 def cell(x):return str(x).replace('|','/').replace('\n',' ')
 return '| '+' | '.join(map(cell,d.columns))+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,r))+' |' for r in d.itertuples(index=False,name=None))
def main():
 assert json.loads((OUT/'VALIDATION.json').read_text())['all_pass']
 s=json.loads((OUT/'SUMMARY.json').read_text());p=pd.read_csv(OUT/'CANDIDATES_384.csv');r=pd.read_csv(OUT/'ALL_574_AGENT_REVIEW.csv');c=pd.read_csv(OUT/'REFERENCE_CONTROLS_EXTRA.csv');left=pd.read_csv(OUT/'NOT_SELECTED_FROM_574.csv');roster=pd.read_csv(OUT/'TARGET_ROSTER.csv')
 counts=pd.crosstab(r.review_agent,r.decision).reset_index();old14=pd.read_csv(BASE/'PRIORITY_MULTI_EVIDENCE_REVIEW.csv')
 old14=old14[['pair_id','drug_names','gene_symbol']].merge(r[['pair_id','decision','reason','sources','priority_after_agent_review']],on='pair_id',how='left',validate='one_to_one');old14['selected_in384']=old14.pair_id.isin(p.pair_id);old14['decision']=old14.decision.fillna('OUTSIDE_CONVENTIONAL_SPR_SCOPE');old14.to_csv(OUT/'ORIGINAL14_AFTER_AGENT_REVIEW.csv',index=False)
 counts.to_csv(OUT/'AGENT_REVIEW_COUNTS.csv',index=False)
 prior=p[p.priority_after_agent_review];low=int(p.decision.eq('LOW_PRIORITY').sum());removed=r[r.decision.eq('EXCLUDE')];held=r[r.decision.eq('HOLD')]
 urls=int(r.sources.fillna('').str.contains(r'https?://').sum());reserve=int(left.selection_disposition.eq('RESERVE_NOT_SELECTED_UNDER_DIVERSITY_AND_BUDGET').sum())
 columns=['drug_names','gene_symbol','decision','selected_in384','priority_after_agent_review']
 reasons=[]
 for x in removed.itertuples():
  links=re.findall(r'https?://[^\s;]+',str(x.sources));citations=' '.join(f'[来源{i+1}]({u})' for i,u in enumerate(links))
  reasons.append(f'- **{x.drug_names}–{x.gene_symbol}**：{x.reason} {citations}')
 report=f'''# 384候选：三组agent逐行快审与独立设计

日期：2026-09-09。用户口径：**384个候选，参考对照另计**。本轮没有训练、没有新增结合评分、没有更改原512冻结清单。结果仍是待实验确认的审查方案。

## 1. 已完成的结果

三组agent按酶335对、激酶128对、核受体／表观遗传111对分工，覆盖全部574个生化路线候选，每行保留判断、理由、来源范围和实验条件。其余472个通道／膜转运体配对仍在专项体系池，不是“绝对不结合”。

- 明确移出本轮候选：**{len(removed)}对**；包括新颖性不成立或已有具体负面功能结果，不统一解释为绝对不能结合。
- 暂缓：**{len(held)}对**；包括构建／膜体系、旧文种属／亚型、近阴性及既有药理待核问题，不为凑数回填。
- 通过可进入选择池的规则：**{s['usable_before_optimization']}对**。按证据审查与药物／靶点分布，选出**384对、{s['drugs']}药、{s['targets']}靶点**；另{reserve}对作为备选。
- 参考对照另列**{len(c)}对**，合计建议配对数**{384+len(c)}**。这是分子－靶点关系数，不包括浓度梯度、重复、空白或溶剂对照的实际孔数。
- 384中有**{low}对LOW_PRIORITY探索项**；并非384对都经过高置信验证。原14个多证据项经本轮审查，当前入选且仍保留优先标记的是**{len(prior)}对**。

**建议先做幸存优先项的原始证据／构建复核与检测体系预实验，再决定扩展到384。** 达到用户指定候选数量是预算设计完成，不是实验放行或命中率保证。

## 2. “逐一快审”的实际范围

{table(counts)}

所有574行都完成了本地评分、化学近邻、图谱解释与实验路线审查；其中{urls}行带在线来源。来源中包含靶点构建层面资料，不能把这个数字称作{urls}对均有直接结合文献。各agent对高优先项、疑似既有药理及特殊构建做了定向原始来源核验；其余明确标记`local_descriptor_review_no_pair_literature`。**没有对574对逐一完成穷尽文献／专利综述。**

`KEEP_REVIEW`表示快审没有发现需暂停的问题，`LOW_PRIORITY`表示支持较薄、机制解释间接或证据相矛盾，可作为明确标识的探索位；这两个标签都不是阳性标签。三个agent的判断也不是三个独立实验重复或投票置信度。

## 3. 有依据的排除及暂缓

{chr(10).join(reasons)}

重要区别：rivaroxaban–F2的反证是所报告实验条件下thrombin功能抑制IC50>20 µM，不能推出任何浓度、任何位点均无结合。已发表配对测试即使没有测得Kd，也会使其不再适合占用“未经测试的新关系”名额。

暂缓项覆盖：膜体系SRD5A2/PORCN/DGAT1/VKORC1/NOX4；ERBB3等域／构建与催化解释；人／动物或受体亚型未澄清的既有核受体药理；显著更近的阴性参考；IDH1野生型／突变背景等。具体每对判断以574逐行表为准。成熟可溶胞外催化域可以设计的靶点没有仅因全长含膜就被一概排除。

olaparib–PIK3CA查得的联用论文不能作为本药直绑PIK3CA证据；lisinopril–F2的高浓度凝血读数也不能提升为高亲和结合依据。本轮降低它们的优先级，仍可能列入明确标记的探索位。没有因为“没找到文献”就断言没有结合。

## 4. 原14个优先项现在怎么样

{table(old14[columns].fillna('不适用'))}

“仍保留优先”只表示原多证据条件存在且本轮agent未降级／暂缓；不提供疾病治疗方向、靶点占有率或有效暴露保证。tecovirimat–AR等没有直接结合验证的假设，不能因反证较少而被称为已证实。

## 5. 384选择规则

所有入选候选都来自原574池，继续满足：完整384靶点中的结合Top20、排除后TxGNN疾病Top50、同一疾病OT≥0.3、身份与既有关系过滤。不引入新模型、不放宽这些主条件、不从HOLD／EXCLUDE回填。

在固定384候选条件下，优化优先保留未被降级的多证据项，随后参考agent意见、阳性化学近邻、精确疾病／遗传支持及原结合排名；这些是公开的探索性选择权重，**不是校准过的总概率**。低优先级项不能被原先的多证据标记自动抬回高优先级。

约束包括64–96靶点、至少128个药物结构实体，并对每药及同连接骨架的重复配对数、每靶点配对数设上限。若第一档不满足384，只尝试事先列明的分布上限调整，不放松证据暂停规则。实际每药最多{s['candidate_drug_max']}对、每靶点最多{s['candidate_target_max']}对；所用档位及求解状态见`SOLVER.json`。这是组合设计偏好，不是生物学可行性的保证。

{table(p.groupby('review_tier').agg(pairs=('pair_id','size'),drugs=('ligand_inchikey','nunique'),targets=('target_chembl_id','nunique')).reset_index())}

仅{s['selected_chemical_ge04']}个入选配对有阳性参考Tanimoto≥0.4；相似度不能独立证明结合，低相似度也没有被当作绝对阴性。选后图谱一致性是选择规则产生的，不能拿来宣布命中率提高。

## 6. 对照与实验就绪边界

每个入选靶点附一个本地ChEMBL37阳性参考提案，排除阳／阴性冲突并核对完整结构键；优先Kd记录，其次Ki，再其他功能端点，兼顾原对照、名称与重复来源。对照按单独预算列出：

{table(c.groupby('reference_endpoint_category').size().rename('targets').reset_index())}

“有Kd记录”仅表示汇总行包含该端点，混合pChEMBL不能当作纯SPR Kd；Ki／IC50也不等于直接结合对照验证。所有参考仍需回看原始assay、核实采购和当前构建适用性。不同靶点即使选同一化合物也分别计一个对照配对。全部靶点的具体边界、异构体、活化态、辅因子与真实活性均待实验室确认；没有自动把这些状态改成就绪。

该384是发现候选集，未额外占用其中名额加入低排名生物学比较组；若要衡量联合筛选是否提高命中率，应另设计匹配比较组，不能仅凭这384的结果反推筛选相对收益。

## 7. 文件与验证

- [384候选与逐行审查工作簿](../outputs/joint384_design_20260909/JOINT384_AGENT_REVIEW.xlsx)。
- [384候选CSV](../outputs/joint384_design_20260909/CANDIDATES_384.csv)。
- [另计参考对照](../outputs/joint384_design_20260909/REFERENCE_CONTROLS_EXTRA.csv)。
- [574逐行总审查](../outputs/joint384_design_20260909/ALL_574_AGENT_REVIEW.csv)、[排除／待审／备选](../outputs/joint384_design_20260909/NOT_SELECTED_FROM_574.csv)、[靶点配额](../outputs/joint384_design_20260909/TARGET_ROSTER.csv)。
- Agent来源与说明：[酶](../outputs/joint384_agent_review_20260909/ENZYME_REVIEW_NOTES.md)、[核受体](../outputs/joint384_agent_review_20260909/nuclear_review.md)，激酶原始意见见同目录CSV与说明。

唯一配对、574全部去向、恰好384候选、无HOLD／EXCLUDE混入、每靶点对照、候选／对照无重复及原512哈希等检查全部通过。结果见`VALIDATION.json`；输入与产物哈希见`SOURCE_MANIFEST.json`。原512及上轮1046筛选结果均保留，当前384是新的独立审查设计。
'''
 DOC.write_text(report)
 before=OUT/'DESIGN_BEFORE_AGENT384.md'
 if not before.exists():before.write_bytes(DESIGN.read_bytes())
 header='\n## 11. Agent逐行快审后的384候选'
 prefix=DESIGN.read_text().split(header)[0]
 DESIGN.write_text(prefix+header+f'''

按用户确认的“384候选、对照另计”，完成[三组agent逐行快审和独立384设计](BIOMASTER_JOINT384_AGENT_REVIEW_20260909_ZH.md)。574对中移出{len(removed)}对、暂缓{len(held)}对；从可进入选择池的{s['usable_before_optimization']}对中选384对（{s['drugs']}药、{s['targets']}靶点），另列{len(c)}参考对照提案。全部原主筛选条件保留，HOLD／EXCLUDE不回填。

384中{low}对仍属低优先级探索；原14项中{len(prior)}对仍以优先标记入选。逐行快审不等于574对穷尽文献核验，没有把证据不足写成绝对不能结合，未放行实验。查看[工作簿](../outputs/joint384_design_20260909/JOINT384_AGENT_REVIEW.xlsx)。本节前快照与版本链见新目录`REPORT_MANIFEST.json`；原512不变。
''')
 artifact=[DOC,DESIGN,before,Path(__file__),OUT/'ORIGINAL14_AFTER_AGENT_REVIEW.csv',OUT/'AGENT_REVIEW_COUNTS.csv']
 (OUT/'REPORT_MANIFEST.json').write_text(json.dumps({'source_manifest_sha256':sha(OUT/'SOURCE_MANIFEST.json'),'design_before_sha256':sha(before),'design_after_sha256':sha(DESIGN),'artifacts_sha256':{str(f.relative_to(ROOT)):sha(f) for f in artifact},'agent_supporting_files_sha256':{str(f.relative_to(ROOT)):sha(f) for f in REV.iterdir() if f.is_file()}},indent=2))
 print(json.dumps({'report':str(DOC),'priority_retained':len(prior),'low_priority':low,'exclude':len(removed),'hold':len(held),'with_online_sources_rows':urls,'reserve':reserve},ensure_ascii=False))
if __name__=='__main__':main()
