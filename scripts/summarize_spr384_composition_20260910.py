"""Describe frozen SPR384 composition; do not select or alter candidates."""
import json,hashlib,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_spr_expansion import baseline_reasons
p=ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'
x=pd.read_csv(p).copy();reasons=baseline_reasons(ROOT)
assert len(x)==384 and x.pair_id.is_unique
x['action_class']=x.pair_id.map(lambda k:reasons[k]['action_class'])
x['rank_band']=pd.cut(x.binding_rank_384,[0,5,10,20,50,384],labels=['1–5','6–10','11–20','21–50','51–384']).astype(str)
N=len(x)
def row(label,mask):
 n=int(mask.sum());return {'label':label,'count':n,'percent':round(n/N*100,1)}
def counts(col,labels):return [row(label,x[col].eq(key)) for key,label in labels.items()]
rank=[]
for label in ['1–5','6–10','11–20','21–50','51–384']:
 m=x.rank_band.eq(label);r=row(label,m)
 for a in ['EXPLORABLE','FIRST_RESOLVE','EVIDENCE_DEPRIORITIZE']:r[a]=int((m&x.action_class.eq(a)).sum())
 r['cross_area']=int((m&x.recommended_disease_is_cross_area).sum());rank.append(r)
chem=x.positive_max_tanimoto
summary={'scope':'冻结384候选，不含112参考对照；新增384证据池不计入','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'total':N,'drugs':int(x.ligand_inchikey.nunique()),'targets':int(x.target_chembl_id.nunique()),'rank_definition':'binding_rank_384：按 independent_validation_rank_score 降序，在每药完整384靶点中排名（并列沿用原表顺序）；不是候选编号、全局排名或命中概率。','rank_bands':rank,
 'sections':[
 {'title':'排名利用与探索（本次描述口径）','rows':[row('高排名利用：1–10',x.binding_rank_384.le(10)),row('相对较低排名探索：11–20',x.binding_rank_384.between(11,20)),row('前20以外探索',x.binding_rank_384.gt(20))]},
 {'title':'最新LLM实验安排','rows':counts('action_class',{'EXPLORABLE':'仍可探索','FIRST_RESOLVE':'先解决具体问题','EVIDENCE_DEPRIORITIZE':'有证据支持降低投入'})},
 {'title':'冻结时额外证据分层（与LLM分类不同）','rows':counts('recommendation_band',{'PRIORITY_CROSS_AREA_MECHANISM_REVIEW':'优先单靶点侦察','ADDITIONAL_SOURCE_OR_CHEMICAL_SUPPORT':'有额外来源或化学支持','EXPLORATORY_LIMITED_INDEPENDENT_SUPPORT':'独立支持有限的探索'})},
 {'title':'推荐疾病是否跨领域','rows':[row('原用途与推荐疾病跨大类',x.recommended_disease_is_cross_area),row('未标为跨大类',~x.recommended_disease_is_cross_area)]},
 {'title':'阳性近邻化学支持（不等于结合证据）','rows':[row('可用阳性近邻T≥0.4',x.usable_chemical_support_ge04),row('T≥0.4但未满足可用条件',chem.ge(.4)&~x.usable_chemical_support_ge04),row('0.3≤T<0.4',chem.ge(.3)&chem.lt(.4)),row('T<0.3或缺失',chem.lt(.3)|chem.isna())]},
 {'title':'实验对象类型','rows':counts('assay_lane',{'ENZYME_BIOCHEMICAL':'酶类','KINASE_BIOCHEMICAL':'激酶','NUCLEAR_EPIGENETIC_DOMAIN':'核受体／表观结构域'})},
 {'title':'推荐共同疾病的图谱映射','rows':counts('recommended_mapping_scope',{'EXACT_SINGLE_NODE':'精确单疾病节点','MEMBER_OF_MERGED_NODE':'合并疾病节点成员'})}
 ],'all_joint_thresholds':bool((x.recommended_txgnn_rank.le(50)&x.recommended_ot_score.ge(.3)).all()),'lab_confirmed':int(x.assay_confirmation_status.ne('NOT_LAB_OR_VENDOR_CONFIRMED').sum()),'target_pair_range':[int(x.groupby('target_chembl_id').size().min()),int(x.groupby('target_chembl_id').size().max())]}
assert sum(r['count'] for r in rank)==384
assert all(sum(r['count'] for r in s['rows'])==384 for s in summary['sections'])
out=ROOT/'outputs/spr384_composition_20260910';out.mkdir(exist_ok=True)
for f in [out/'COMPOSITION.json',ROOT/'web/src/generated/spr384Composition.json']:f.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
x[['candidate_id','pair_id','modeled_entity_name','gene_symbol','binding_rank_384','rank_band','action_class','recommendation_band','recommended_disease_is_cross_area','recommended_disease','positive_max_tanimoto','usable_chemical_support_ge04']].to_csv(out/'PAIR_COMPOSITION_384.csv',index=False,encoding='utf-8-sig')
lines=['# 当前冻结SPR384内部配比','',summary['scope'],'',summary['rank_definition'],'','本报告为已有名单的事后描述，并非按这些比例重新优化选药。所有百分比以384候选为分母。','']
for s in summary['sections']:
 lines+=['## '+s['title'],'','| 类别 | 数量 | 占比 |','|---|---:|---:|']+[f"| {r['label']} | {r['count']} | {r['percent']}% |" for r in s['rows']]+['']
lines+=['## 排名与实验建议交叉分布','','| 排名/384 | 总数 | 占比 | 仍可探索 | 先解决 | 降低投入 | 跨领域 |','|---|---:|---:|---:|---:|---:|---:|']+[f"| {r['label']} | {r['count']} | {r['percent']}% | {r['EXPLORABLE']} | {r['FIRST_RESOLVE']} | {r['EVIDENCE_DEPRIORITIZE']} | {r['cross_area']} |" for r in rank]
lines+=['','## 怎样理解这批资源投入','','当前384全部位于每药前20/384，即排名前5.2%的范围；没有真正的低排名或随机配对探索。前10占46.6%，11–20占53.4%，后一组仅是本候选集内部的相对较低排名。前5只有22.7%，所以也不能称作主要押注每药最前几个靶点。','','“独立支持有限”240条（62.5%）描述额外证据，不意味着其结合排名低。最新“仍可探索”223条（58.1%）表示不能提前否决，也不意味着已验证或优先上机。应分别看排名与实际风险，不能拿LLM标签替代模型校准。','','所有384的推荐共同疾病均满足TxGNN前50且OT≥0.3；195条（50.8%）推荐疾病跨原用途大类。跨领域不能证明新靶点作用方向，未跨大类也不等同于全部是癌症内转换。单疾病节点精确映射不证明药物图谱映射具有立体化学精确性。','','可用阳性化学近邻T≥0.4仅20条（5.2%）。这不构成对其余364条的阴性判断，但说明本批多数配对并无较强的这类化学支撑，不能因为模型排名高就按高把握命中预算。','','本批覆盖229药、112靶点，每靶1–10个候选；广覆盖有探索价值，也意味着蛋白体系准备成本较分散。冻结记录中384条均未获实验室／供应商确认；这是记录状态，不代表蛋白实际不可用。112个参考对照另计，且不能代替随机／低排名候选对照来估计模型富集效果。','','建议保留现有384为基线，用页面排名筛选与“实验安排”联合筛选首批：优先审阅前10中仍可探索者，再逐项解决构建与证据问题，并保留部分11–20的跨领域假设。没有实测校准和实际体系信息，不应声称某个新比例已经最优，也不宜为了标签好看直接重排成新384。','','[逐对配比表](../outputs/spr384_composition_20260910/PAIR_COMPOSITION_384.csv) · [统计JSON](../outputs/spr384_composition_20260910/COMPOSITION.json)']
(ROOT/'docs/BIOMASTER_SPR384_COMPOSITION_20260910_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'rank_bands':rank,'sections':summary['sections'][:2]},ensure_ascii=False))
