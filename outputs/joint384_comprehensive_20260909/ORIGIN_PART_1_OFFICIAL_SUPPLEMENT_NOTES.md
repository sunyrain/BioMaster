# Part 1 官方用途补核（2026-09-09）

已更新 `ORIGIN_REVIEWED_PART_1.csv`：89 药名行完整保留；72 LABEL_REVIEWED、16 EXTERNAL_REVIEWED、1 UNRESOLVED。原始审查快照保存为 `ORIGIN_REVIEWED_PART_1_BEFORE_OFFICIAL_SUPPLEMENT.csv`。`supplement_origin_review_part_1.py` 记录逐药补核，需在旧生成脚本之后运行；本次未修改主筛选脚本或候选。

13 个原 UNRESOLVED 中，12 个找到可归属的原使用场景。cilastatin 的场景限 imipenem/cilastatin 复方保护组分，**不是 cilastatin 单药抗感染疗效**；其 INFECTION 类仅为用途背景。历史标签可证明既有用途，但不证明当前商业供应/全球现行批准。每行保存官方 URL、制剂、年龄、联合与排除条件；并非声称完整全球登记已穷尽。

合并名按主线程核定的 registry 完整结构键解释：

| 原药名组 | 当前模型实体 | 本次用途处理 |
|---|---|---|
| aripiprazole;aripiprazole lauroxil | aripiprazole | 采用口服 aripiprazole 精神科标签；不把长效前药视作同一制剂 |
| balsalazide;mesalamine | mesalamine | 采用 mesalamine 口服 UC、直肠炎标签；不承接 balsalazide 产品人群限制 |
| dexmethylphenidate;serdexmethylphenidate | dexmethylphenidate | 采用单成分口服 ADHD 标签，不把 AZSTARYS 复方当模型 |
| droxidopa;norepinephrine | norepinephrine | 仅采用成人严重急性低血压静脉升压；未继承 droxidopa 的神经源性体位低血压适应证 |

新增/修正影响跨领域判定的内容：

- amisulpride：除美国静脉术后止吐，还核到法国国家授权 Solian 口服精神分裂症。[ANSM SmPC](https://m.base-donnees-publique.medicaments.gouv.fr/rcp-60019927-5)
- glycopyrronium：除外用腋窝多汗，还核到单成分 Seebri 吸入成人 COPD 维持治疗；仍未穷尽 glycopyrrolate 同义名下其他制剂。[EMA](https://www.ema.europa.eu/en/medicines/human/EPAR/seebri-breezhaler)
- timolol：除眼用，单成分口服标签确有高血压、稳定 MI 后风险降低、偏头痛预防，不能按仅眼科药计算跨领域。[DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=0bc40b2c-65eb-4095-87e6-4752d5b19a3a)
- tafamidis：保留美国 ATTR 心肌病及欧盟 I 期症状性多发性神经病；不能把欧盟神经病适应证说成美国批准。来源逐行。
- selinexor：DLBCL 美国加速批准已于 **2026-04-30 撤回**，仅记历史，不作为现行用途；仍为 ONCOLOGY 背景。[FDA 撤回表](https://www.fda.gov/drugs/resources-information-approved-drugs/withdrawn-cancer-accelerated-approvals)
- dasabuvir：Exviera 欧盟授权于 **2024-09-25 撤回**，保留既有联合慢性 HCV 用途，不称现行单药批准。[EMA](https://www.ema.europa.eu/en/medicines/human/EPAR/exviera)
- conivaptan：低钠血症升血钠适应证；标签明确不批准治疗心衰症状，未把心衰共病当原用途。[FDA 标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/021697s005lbl.pdf)

唯一仍未解决的是 `3-iodobenzylguanidine`。本地 DRUG_REGISTRY_720 的来源为 IOBENGUANE I-131/AZEDRA，而模型 SMILES 为 `N=C(N)NCc1cccc(I)c1`、InChIKey 为 `PDWUPXJEEYOOTR-UHFFFAOYSA-N`，未标明放射同位素。FDA I-131 PPGL 肿瘤治疗来源已确认，不能将该放射药的批准直接转移到当前未标同位素模型；该行宏类留空、UNRESOLVED，且记录来源肿瘤背景。其空类不能用来证明跨领域。[FDA AZEDRA 标签](https://www.accessdata.fda.gov/drugsatfda_docs/label/2023/209607Orig1s002lbl.pdf)
