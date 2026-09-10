# SPR112 靶点 FDA 对照可行性初筛（2026-09-10）

可以把 FDA 获批药设为对照的优先选择，但当前证据不足以把 112 个靶点全部改成已落实的 FDA 药物 SPR 阳性对照。此前对照按活性证据选择，未优先要求 FDA 身份，因此存在可改进空间。

本目录是替代线索审计，不是采购或实验放行表。当前 384 候选及网站指定对照未因此替换。表中每靶点展示的药物是程序选出的代表线索，不代表已完成人工比较或最佳对照认定；全部备选见 FDA_CONTROL_OPTIONS.csv。

| 本次初筛状态 | 靶点数 | 涉及384候选行数 |
| --- | ---: | ---: |
| A：项目 FDA 注册表完整结构匹配，且有达标 Kd/Ki 记录 | 69 | 236 |
| B：FDA 成分/活性母体名称匹配，且有达标 Kd/Ki 记录；结构身份待核对 | 8 | 35 |
| C：本次只找到通过初筛的结合类 IC50 记录 | 14 | 43 |
| D：本次未找到通过筛选的线索 | 21 | 70 |
| 合计 | 112 | 384 |

其中 D 类的 ALOX5 并不是没有 FDA 药物：zileuton 有活性记录，但本次同端点强弱矛盾规则将其留待复核。其他 D 类也不能解释为全世界没有获批药，例如 PPARD 的已获批激动剂可能以功能端点为主要证据，不满足本次检索口径。[FDA 的 seladelpar 批准信息](https://www.fda.gov/drugs/drug-trials-snapshots/drug-trials-snapshots-livdelzi)

## 方法与限制

- FDA 身份依据本地 2026-08-03 官方产品快照中的 NDA/ANDA、正式批准日期和 Prescription/Over-the-counter 状态；排除暂定批准和该快照中仅停产的产品。ChEMBL max_phase=4 不作为 FDA 证明。该快照不能代替全部药物截至今日的状态复查。
- A 类完整 InChIKey 匹配同时要求项目注册表 exact_structure_in_fda_registry=true。B 类仅 FDA 成分或 RxNorm 活性母体名称匹配，不能声称已完成精确结构认证。药品盐型、立体化学和实物仍需落实。
- 活性来源为 ChEMBL37 原始 activity，按精确分子实体查询，不把母体或代谢物的活性直接迁移。限定人源单蛋白、confidence=9、relationship=D、assay_type=B、无已标注突变和有效性警告，标准单位 nM。
- Kd/Ki/IC50 ≤1 µM 为备选线索；同一精确分子—靶点—端点出现 ≥10 µM 记录时暂不入选。不同实验体系可以造成这种差异，矛盾不是药物不能结合的定论。
- Kd 也可能来自细胞裂解液竞争、kinobeads 等间接估计，Ki 也不是 SPR 测量。数据库 assay_type=B 可能包含细胞读数；未逐篇核实的记录只能称为线索。
- 具体结构域必须相符，例如本次 JAK1—deucravacitinib 代表记录描述的是 JH2 域，不能移用为 JH1 构建的阳性对照。DNMT1—decitabine 的 IC50 记录也不能跳过代谢活化和 DNA 依赖机制，直接认定游离蛋白 SPR 对照。[FDA 的 decitabine 机制说明](https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/205582s014lbl.pdf)
- 程序选择顺序为：无强弱矛盾、优先 Kd/Ki、优先已匹配 FDA 完整结构；使用同端点精确值中位数排序。仍需要按本次蛋白构建、作用位点、溶解性和实验方法人工定稿。

## 已可明确推进的例子

- PDE5A：原对照仅研究编号，可优先复核 tadalafil。FDA 说明书明确其 PDE5 抑制作用，本地还检索到 Kd 记录。[FDA 说明书](https://www.accessdata.fda.gov/drugsatfda_docs/label/2022/214522s000lbl.pdf)
- HDAC2：原对照仅研究编号，可优先复核 vorinostat。FDA 说明书列出 HDAC2 等作用靶点，本地还检索到 Kd 记录。[FDA 说明书](https://www.accessdata.fda.gov/drugsatfda_docs/label/2009/021991s004lbl.pdf)
- CTSC：原有 brensocatib 不必仅因 ChEMBL 的阶段标记而排除；FDA 已于 2025-08-12 批准该药。[FDA 批准信息](https://www.fda.gov/drugs/drug-trials-snapshots/drug-trials-snapshots-brinsupri)
- gedatolisib 的 FDA 状态另核对到 NDA219908 官方批准信，不能依照过时记忆将其一概视作在研药。[FDA 批准信](https://www.accessdata.fda.gov/drugsatfda_docs/appletter/2026/219908Orig1s000ltr.pdf)

建议采用“FDA 药优先，匹配构建的可靠结合证据优先”。如要严格执行全 FDA 对照，尚未落实者应明确标为待补齐，而不能把功能药、前药或缺乏适用结合证据的药硬填为阳性对照。

## 文件

- SPR112_FDA_CONTROL_FEASIBILITY.csv：逐靶点原对照、FDA 代表备选、审批申请号、活性来源和覆盖状态。
- FDA_CONTROL_OPTIONS.csv：全部通过活性阈值的 FDA 身份备选及矛盾标志。
- RAW_EXACT_ENTITY_ACTIVITY.csv：本次精确实体的原始活动与实验描述。
- FDA_DRUG_IDENTITY_CANDIDATES.csv：FDA 身份匹配依据与层级。
- AUDIT.json：计数、范围和限制。生成脚本：scripts/audit_spr112_fda_control_options_20260910.py。
