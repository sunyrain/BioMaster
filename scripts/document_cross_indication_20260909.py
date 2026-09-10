import json,hashlib
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint_cross_indication_20260909';DOC=ROOT/'docs/BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md';ACTIVE=ROOT/'docs/BIOMASTER_JOINT384_AGENT_REVIEW_20260909_ZH.md';DESIGN=ROOT/'docs/BIOMASTER_SPR64_512_DESIGN_20260909_ZH.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 s=json.loads((OUT/'SUMMARY.json').read_text());assert json.loads((OUT/'VALIDATION.json').read_text())['all_pass']
 text=f'''# 跨疾病大类与新靶点：重新定义候选优先级

2026-09-09。按用户最新目标，**原适应症→新靶点假设→另一疾病大类**成为优先条件。癌症药→另一癌种不进入本轮跨领域优先项；384改为历史预算上限，不再要求填满。本轮没有更改旧384／512清单，而是另存新条件的逐对审查。

## 1. 三个条件必须分别成立

1. **药物实体可用且原用途清楚**：区分给药分子、代谢物、单药与复方；多适应症和预防用途都要保留，不能挑一个原用途制造“跨领域”。
2. **药物→靶点有新颖性及结合依据**：沿用之前身份、直接关系、家族／复合物及agent审查排除；模型Top20只提供相对排序，新靶点仍是假设，需实验确认。
3. **目标疾病跨原用途大类，并有独立理由**：同一疾病TxGNN Top50及OT≥0.3只是起点；药物作用方向、靶点致病／保护关系、可达到暴露和检测体系还要确认。换癌种、换疾病亚型、换本体器官标签不自动构成跨大类。

因此，不能只问“这个药和这个靶点都和某种癌有关吗”，而应记录完整的原用途—新靶点—目标疾病假设。跨大类是本项目的探索偏好，不是药效更好的统计保证，也不能覆盖既有负面临床证据。

## 2. 原5个优先项的变化

| 配对 | 药物原用途→共同疾病线索 | 当前判断 |
| --- | --- | --- |
| tecovirimat–AR | 天花等病毒感染→HER2阳性乳腺癌 | 符合跨大类，保留机制与结合核验 |
| mebendazole–ESR1 | 肠道蠕虫感染→肺鳞癌；另有痤疮线索 | 符合跨大类；肺鳞癌条目还满足原遗传／体细胞支持条件，痤疮较弱 |
| vismodegib–NTRK1 | 基底细胞癌→前列腺癌／泛癌 | 全部仍为癌症，退出跨大类优先项 |
| raloxifene–KDR | 骨质疏松、乳腺癌风险降低等→内异症 | 妇科／内分泌邻近且已有不利RCT，该疾病投资方向暂缓 |
| bazedoxifene–KDR | 绝经／骨质疏松相关用途→内异症 | 邻近既有治疗领域且已有复方案例，不列为新的跨大类优先项 |

tecovirimat和mebendazole的原用途分别依据[FDA TPOXX标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/208627s008%2C214518s003lbl.pdf)及[EMVERM标签](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=a8c46363-f739-4f6e-bca8-1ce5e8d3f78d)。这证明的是原用途分类，不是新靶点或抗癌疗效。不能据此声称此前从未有人研究过该药抗癌；“跨获批用途”和“药物—疾病首次提出”是不同的新颖性问题。

vismodegib按[FDA标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/203388s017lbl.pdf)和[EMA产品信息](https://www.ema.europa.eu/en/documents/product-information/erivedge-epar-product-information_en.pdf)核验；本地所有达标疾病只有前列腺癌、cancer，没有通过当前阈值的非癌备选。前列腺癌行的遗传／体细胞复合标记来自体细胞突变分，不是生殖系遗传分。

raloxifene不能只登记“骨质疏松”而遗漏特定人群乳腺癌风险降低；bazedoxifene需区分欧盟单药与美国复方，不能把复方适应症或疗效全部归单药。[EVISTA标签](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=01143c04-f5d7-481d-95bb-d384f2413585)、[Conbriza监管信息](https://www.ema.europa.eu/en/medicines/human/EPAR/conbriza)、[DUAVEE标签](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=e1b75458-2e5b-46b9-92c6-fa6daba3770f)。raloxifene内异症RCT因疼痛更早复发而停止；bazedoxifene联合雌激素的内异症病例已经发表。前者是疾病投资风险，后者提示已有临床探索，均不能证明或否定KDR直接结合。[原始RCT](https://pmc.ncbi.nlm.nih.gov/articles/PMC2755201/)、[原始病例](https://pubmed.ncbi.nlm.nih.gov/29995747/)。

## 3. 全574对已做什么，还没有做什么

已对574个生化配对的**全部达标共同疾病**重新标注，未只看每对原先第一名疾病。优先大类按肿瘤、感染、免疫炎症等区分，并将内分泌／代谢／生殖相邻领域保守合并；肿瘤分类优先于肺、乳腺、前列腺等器官分类，避免癌症跨器官被误算为跨领域。多标签有重叠不自动算跨类；缺失标签不算新领域；合并疾病节点进入待审。

在全部236个agent未标低优先级的配对中，**{s['non_low_with_exact_cross_area_proxy_or_verified']}对存在精确疾病节点的跨大类线索，进入原用途核验队列**，其中{s['previous384_non_low_with_exact_cross_area_proxy_or_verified']}对在旧384里。额外8对来自之前因预算／分布未选中的候选，避免旧384的配额限制阻碍新标准。

**这65对不是65个已经确认跨获批适应症的实验候选。** 本轮只对5个重点药物做了官方适应症核验；其余药物以旧图`indication`和`off-label use`作为既往用途的保守代理。项目尚无完整、当前、逐药获批适应症登记表；旧图边、FDA上市申请记录和MOA字段都不能替代它。代理覆盖不全，仍可能把已知用途误判成新方向，因此所有代理线索都明确标为官方适应症待核。禁忌边没有混入原治疗用途。

本轮“原用途已核＋跨大类＋原多证据条件＋agent未降级”的优先复核项只有**2对：tecovirimat–AR、mebendazole–ESR1**。这不是对574中其他配对无价值的判断，也不是已经验证的2个阳性；只是当前核验深度支持的候选优先顺序。

## 4. 下一步与文件

先核实65条线索所涉及药物的完整原用途，记录获批地区／产品／单复方、既有药物—疾病试验及原靶点机制；排除同大类与已有明确负面证据的疾病方向，再对保留配对补作用方向、暴露和构建审查。病例或预临床探索不是获批用途，但必须进入新颖性说明。数量由这些证据决定，暂不重新生成384固定数量名单。

- [跨适应症审查工作簿](../outputs/joint_cross_indication_20260909/CROSS_INDICATION_REVIEW.xlsx)：2个优先复核、原5项变化、65条待核队列、574逐对审计及药物用途来源范围。
- [优先跨大类假设](../outputs/joint_cross_indication_20260909/PRIORITY_CROSS_MAJOR_AREA_REVIEW.csv)。
- [原用途待核队列](../outputs/joint_cross_indication_20260909/CROSS_AREA_LABEL_REVIEW_QUEUE.csv)。
- `ALL_574_COMMON_DISEASE_CROSS_AREA_AUDIT.csv.gz`保留每个达标疾病的分类、来源范围与原用途交集；`KG_PRIOR_USE_LABELS_NOT_APPROVALS.csv.gz`保留代理来源。
- 5药官方／原始来源见同目录`PRIORITY_DRUG_INDICATION_REVIEW.csv`及`VISMODEGIB_CROSS_INDICATION_REVIEW.json`。分类规则是项目启发式，非经验证的医学距离量表。

所有574配对去向、优先项的官方原用途核验、精确疾病映射、无大类重叠及无癌症→癌症混入检查均通过，见`VALIDATION.json`。新报告版本链见`REPORT_MANIFEST.json`。原384与512产物保持不变，不能继续把原5个优先标记当作符合最新跨大类目标。
'''
 DOC.write_text(text)
 for p,name in [(ACTIVE,'JOINT384_BEFORE_CROSS_INDICATION.md'),(DESIGN,'DESIGN_BEFORE_CROSS_INDICATION.md')]:
  snap=OUT/name
  if not snap.exists():snap.write_bytes(p.read_bytes())
 header='\n## 8. 最新目标：跨疾病大类'
 original=ACTIVE.read_text().split(header)[0]
 notice='> 最新口径：优先跨疾病大类，不再强求384。原5项中仅2项保留跨大类优先复核，65条线索待核原适应症；下文384为上一版设计，详见[跨适应症复审](BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md)。\n\n'
 if notice not in original:original=original.replace('\n\n','\n\n'+notice,1)
 ACTIVE.write_text(original+header+f'''\n\n已完成[跨适应症复审](BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md)：癌症→另一癌种不符合本轮优先目标。vismodegib–NTRK1退出；raloxifene/bazedoxifene–KDR因邻近用途和既有临床探索降级，raloxifene内异症方向另有不利RCT。tecovirimat–AR与mebendazole–ESR1保留跨大类假设核验。574对中236个非低优先项有65条跨类线索待核原用途；仅5药完成本轮官方原用途审查，不能把65当确认通过。原384清单不变，本轮不再强求数量。\n''')
 header2='\n## 12. 跨疾病大类优先条件'
 DESIGN.write_text(DESIGN.read_text().split(header2)[0]+header2+'''\n\n用户进一步要求药物原用途→新靶点假设→不同疾病大类，癌症→另一癌种不进入本轮跨领域优先项。已完成[跨适应症审查](BIOMASTER_CROSS_INDICATION_REVIEW_20260909_ZH.md)：原5项目前2项保留跨类优先复核，另65条线索待核完整原用途，未生成新的固定384名单。旧图用途标签与正式获批适应症严格区分；原384和512产物均不改写。\n''')
 files=[DOC,ACTIVE,DESIGN,Path(__file__),OUT/'JOINT384_BEFORE_CROSS_INDICATION.md',OUT/'DESIGN_BEFORE_CROSS_INDICATION.md']
 (OUT/'REPORT_MANIFEST.json').write_text(json.dumps({'audit_manifest_sha256':sha(OUT/'MANIFEST.json'),'artifact_sha256':{str(f.relative_to(ROOT)):sha(f) for f in files}},indent=2))
 print(str(DOC))
if __name__=='__main__':main()
