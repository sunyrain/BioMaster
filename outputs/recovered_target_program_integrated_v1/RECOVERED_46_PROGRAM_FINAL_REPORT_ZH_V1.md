# 46 个“仅因无实验口袋”靶点综合推进报告 V1

## 1. 冻结范围

- ChEMBL 37 正式全集：888 个靶点。
- 前序硬门槛淘汰：480 个，包括 GPCR 143、无 ChEMBL 小分子 MoA 295、不支持靶点类别 42。
- 原严格实验口袋主线：338 个。
- 本项目找回：46 个，唯一条件为 `first_exclusion_reason == EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET`。
- 后续实验口袋质量淘汰仍保留：24 个，包括无优先实验口袋 21、无药物样 holo 口袋 3。
- 当前活跃并集：`338 + 46 = 384`，两分支无重叠。
- 151 个“无已知口袋”原始计算缓存不是正式范围。

## 2. 机制级完整分类

原有五类实验车道继续保留用于历史兼容：非激酶酶 24、离子通道 8、蛋白激酶 6、转运体车道 6、核内/表观遗传车道 2。为避免错误套用计算与实验协议，已细化为 12 个机制分支：

| 机制分支 | 数量 | 靶点 |
|---|---:|---|
| 蛋白激酶 | 6 | AKT3、FRK、MAP3K10、MAP3K13、MAPKAPK5、YES1 |
| 脂质/代谢物激酶 | 3 | PIK3CB、PIKFYVE、SPHK2 |
| 辅因子/金属依赖氧化还原酶 | 11 | CYP26A1、DHCR24、DIO1、DPYD、EGLN3、GPD2、HSD3B2、LOXL2、TBXAS1、TPO、TYR |
| 膜 NADPH 氧化酶 | 2 | NOX1、NOX4 |
| 非氧化还原代谢酶 | 7 | ABAT、CPT1B、CPT2、FAAH、PPP2CB、RPE65、UGCG |
| 多亚基 RNA 聚合酶 | 2 | POLR2A、POLR3A |
| 泛素连接/衔接蛋白 | 1 | TRAF3IP2 |
| 成孔离子通道 | 8 | ANO1、KCNA5、KCND3、KCNJ2、KCNJ8、KCNK18、RYR1、SCN11A |
| 离子通道辅助亚基 | 1 | CACNA2D2 |
| 溶质转运体 | 3 | SLC10A2、SLC12A1、SLC22A8 |
| 表观遗传酶 | 1 | HDAC11 |
| 转录因子/蛋白互作 | 1 | NFE2L2 |

关键分类纠正包括：CPT1B 按膜相关酰基转移酶而非转运体处理；PIK3CB、PIKFYVE、SPHK2 使用脂质底物激酶体系；NOX1、NOX4 使用膜电子传递/ROS 体系；CACNA2D2 必须在 CaV 复合物中评估；NFE2L2 使用蛋白互作与报告基因体系。

## 3. 口袋预测

- P2Rank 与 fpocket 支持同一位点：37 个。
- fpocket 几何补救：7 个。
- 低置信片段结构口袋假设：2 个，即 DIO1、RYR1。
- DIO1、RYR1 不进入正式三维候选排序；必须先取得可信全长、同源或复合物结构。

## 4. DTA 与口袋校准计算

- ConPLEx：720 个药物 × 46 个靶点，共 33,120 对，全部完成。
- DrugCLIP：720 个药物 × 44 个精确序列全长结构靶点，共 31,680 对；DIO1、RYR1 排除。
- 两模型靶点内 top 10% 一致：276 对。
- 两模型双向 top 10% 一致：58 对。
- DIO1/RYR1 序列模型探索性 top 5%：72 对，只能用于结构获取优先级。
- 23 个具备同一对接域内至少 12 阳性和 12 阴性历史对照的靶点完成 GNINA 口袋资格评估：552 个对照、2,760 个姿势、无计算失败。

GNINA 资格结论：

- 强回顾性对照分离：3 个——LOXL2、NOX1、SPHK2。
- 边缘、未获资格：9 个——AKT3、CPT2、CYP26A1、FAAH、KCNA5、MAP3K13、PIKFYVE、TBXAS1、YES1。
- 对照分离失败：11 个——ANO1、CPT1B、DHCR24、FRK、HDAC11、MAPKAPK5、NOX4、PIK3CB、SLC10A2、TYR、UGCG。
- 未满足 12×12 正负对照门槛：23 个；不得进行正式发现候选晋级。

## 5. 候选和正交计算结论

共完成 27 个候选对的 GNINA 对接、135 个姿势：

- LOXL2：tecovirimat、tiotropium 通过计算分诊，必须实验验证；其余 4 个候选没有对照校准支持。
- NOX1：treprostinil 没有获得对照校准支持，不晋级候选。
- SPHK2：双 DTA 模型明显分歧；nilotinib、dabigatran etexilate、ponatinib、imatinib、cabozantinib 仅作为诊断面板，不是正式发现候选；其余 15 个无正交支持。

LOXL2 两个候选各完成 5 个 Boltz 构象和 10 组非自身两两姿势比较：

| 候选 | ligand-iPTM 中位数 | complex-ipLDDT 中位数 | 口袋对齐配体 RMSD 中位数 | 接触 Jaccard 中位数 | MD |
|---|---:|---:|---:|---:|---|
| tecovirimat | 0.556 | 0.295 | 8.48 Å | 0.131 | 不授权 |
| tiotropium | 0.471 | 0.292 | 8.93 Å | 0.078 | 不授权 |

两者均缺少 LOXL2 的 Cu/LTQ 成熟辅因子环境，复合物整体置信度和姿势复现性不足。结论为：保留实验分诊，不运行会被误解为稳定性证据的 MD。

## 6. 湿实验执行顺序

1. `W1_CANDIDATE_CONFIRMATION`：LOXL2。先 tecovirimat，后 tiotropium；使用 Cu/LTQ 重构成熟 LOXL2，10 点半对数剂量、技术三复孔、至少 3 次独立实验；以直接结合和细胞外基质/产物读数正交复核。
2. `W2_MODEL_DIAGNOSTIC_PANEL`：SPHK2。5 个救援化合物只用于解决 DTA 模型分歧；使用脂质底物激酶测定和 S1P 定量。
3. `W2_PREDICTED_SITE_VALIDATION`：NOX1。先用 12 阳性和 12 阴性对照验证预测位点；treprostinil 作为诊断阴性；使用 NOX1 复合物依赖 ROS 测定、同工酶和非 NOX ROS 反筛。
4. `W3_MARGIN_REPLICATION`：9 个边缘靶点。冻结对照面板，在替代口袋或构象上独立重复；未通过前不选择发现候选。
5. `W4_STRUCTURE_AND_POCKET_REMODEL`：11 个失败靶点。纠正膜环境、辅因子、复合物或构象后再以原冻结对照复核。
6. `W5_CONTROL_ACQUISITION`：21 个。收集同一测定体系至少 12 阳性和 12 阴性后再做正式口袋校准。
7. `W6_STRUCTURE_ACQUISITION_FIRST`：DIO1、RYR1。结构质量优先，不进行三维候选晋级。

统一候选晋级要求：主测定形成可重复完整剂量反应；至少两次独立实验的 IC50/EC50 相差不超过 3 倍；正交结合或机制读数同方向；通过聚集、反应性和信号干扰排查；关键近缘靶点至少 3 倍选择性窗口，高优先级目标为 10 倍。

## 7. 证据边界

预测口袋、DTA、GNINA 和 Boltz 均为计算分诊证据，不是实测结合或活性。当前正式结合证据仍为 0 对。计算端已推进到现有证据允许的门槛；湿实验方案已完成到逐靶点执行层，实际物理测量需要外部实验室实施并回填结果。

最终自动覆盖性审计为 15/15 通过。
