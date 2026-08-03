# BioMaster 正式流程状态

更新时间：2026-07-11 UTC

## 当前主线

当前唯一正式主线为 `structure-collapsed / non-GPCR / physics-first v4`。旧 106,561、Top300 recall、strict895、DiffDock Top10000 和疾病图谱排序均为历史探索，不再定义正式候选。

| 阶段 | 状态 | 权威证据 |
| --- | --- | --- |
| ChEMBL-MoA / Open Targets 靶点审计 | 完成 | `outputs/current_production_package_v2/universe_scope_audit_v4/` |
| FDA 模型配体结构标准化 | 完成 | 750 个药物记录、723 个唯一去盐模型结构；`data/processed/drug_library_active_moiety_v4.csv` |
| 915 × 891 ConPLEx active-moiety 重算 | 完成 | `outputs/full_conplex_active_moiety_v4/` |
| 750 × 463 项目空间构建 | 完成 | `outputs/current_production_package_v2/full_untruncated_universe_v4/` |
| 无 Top300 门槛的 Top3000 预选 | 完成 | `pre_boltz_top3000_v4_fully_audited.csv` |
| Top3000 靶点 Open Targets 全疾病补全 | 完成 | Open Targets 数据 26.06 / API 26.6.3；`opentargets_top3000_target_completion_v4/`（215/215 靶点，231,148 条关联） |
| Top3000 Boltz-2 模板精修 | 完成 | 3,000/3,000；600/600 batch success；`boltz_full_run_v4_seeded/` |
| 双样本条件姿势审计 | 完成 | A 1,923、B 673、C 404；3,000/3,000 有效 |
| 正式 final1000 / review512 | 完成 | final1000=1,000；review512=512 |
| ChEMBL activity / PubMed 精确 pair 审计 | 完成 | ChEMBL 512/512；PubMed 512/512 查询成功 |
| 子智能体逐条可行性审阅 | 完成 | 512/512 初审；305 条原始 D 已完成统一量表二次裁决 |
| 审阅后 final384 | 完成 | 384 条、169 个药物、116 个靶点、152 个骨架 |
| Boltz 已知阳性校准 | 完成 | 96/96 结构、亲和和姿势输出完整 |
| 最终中文 PDF / Excel / CSV | 完成并审计 | `final_delivery_v4/`；交付审计、文件哈希、PDF 页完整性和 4x96 提名映射均通过 |

## 冻结口径

| 指标 | 数值 |
| --- | ---: |
| FDA 结构条目 | 915 |
| ChEMBL-MoA 基因 | 892 |
| 唯一蛋白序列 | 891 |
| 原始 ConPLEx pair | 815,265 |
| v4 项目药物 | 750 |
| v4 项目靶点序列 | 463 |
| v4 ID 审计 pair | 347,250 |
| v4 唯一结构物理 pair | 334,749 |
| 已知阳性校准 ID-pair | 491 |
| 活性母体折叠后校准 pair | 473 |
| Top3000 活性母体 | 436 |
| Top3000 靶点 | 215 |
| Top3000 active-moiety × target 重复 | 0 |
| Top3000 位于旧 106k 之外 | 20 |
| Top3000 Boltz 完整输出 | 3,000 / 3,000 |
| Top3000 条件姿势 A/B | 2,596 / 3,000 |
| final1000 Boltz A/B | 574 / 1,000 |
| review512 完整人工审阅 | 512 / 512 |
| 二次裁决后硬淘汰 D | 101 / 512 |
| 审阅后 final384 | 384 |
| final384 唯一药物 / 靶点 / 骨架 | 169 / 116 / 152 |

Top3000 的描述性分布：ConPLEx 中位数 0.3021（范围 0.0836–0.8910），药物内 rank 中位数 19，靶点内 rank 中位数 28；口袋共识 A/B 为 2,999/1；Open Targets 可做性为已上市小分子先例 2,511、临床先例 474、高质量配体或口袋 15。这些是连续排序后的结果分布，不是额外硬阈值。

## 审阅后 final384

- assay family（校正后的 `target_assay_family_v2`）：enzyme 245、transporter 78、nuclear/epigenetic 38、kinase 23；不含 GPCR 和 ion channel。
- 统一可行性等级：A 1、B 31、C 352；最终包不含 D。
- 文献类别：具体 pair 直接验证 10、功能证据 13、间接/家族证据 28、未找到精确 pair 报道 333。
- 候选角色：validated control / rediscovery 10、novel hypothesis 374。
- 老药新用定位：新疾病领域 215、同领域新适应症 45、仅新靶点暂不宣称疾病 116、原领域/非老药新用 8。
- 活性物种：盐型归一化充分 186、母体药物适用 168、活性物种仍不确定 30；需要活性代谢物重算的条目已全部退出。
- 条件姿势：A 331、B 53；每药最多 5 条、每靶点最多 8 条、每骨架最多 10 条。

`未找到精确 pair 报道`表示在本轮 ChEMBL、PubMed 和逐条人工复核中未发现直接验证，不等于已经证明不存在，也不等于真实结合。final384 是湿实验优先级假说包，不是 384 条已证实相互作用。

## 召回口径

active-moiety ConPLEx 对 491 个已知 ID-pair 的 463 靶点空间校准：Recall@10 30.14%、Recall@50 49.29%、Recall@100 59.27%、Recall@300 83.10%。折叠为 473 个唯一 active-moiety × target 后分别为 30.02%、49.47%、59.20%、83.30%。rank 使用并列平均名次；这些数字可能包含模型训练知识与 ChEMBL 标签重叠，只表示已知阳性校准，不表示未来未知 pair 的真实召回率。

v4 不以 Top300 或绝对 ConPLEx 分数作硬门槛。结构与直接小分子 tractability 层保留 427/491 个已知对；所有已知对随后从 discovery 中排除。

## 运行合同

- Top3000 全部从零运行，不复用旧 Boltz 结果。
- ConPLEx、Boltz、去重统一使用 `model_ligand_smiles`。
- 每个 Boltz 输入均绑定 YAML、蛋白序列、模板 PDB、口袋约束和模型配体 SHA-256；运行参数、批次输入与结果 provenance 分层签名。
- 固定种子使用 `20260710 + batch_index`，模型版本和两份 checkpoint SHA-256 写入运行计划。
- 正式完成要求 confidence、affinity、model0 CIF、model1 CIF 均存在，关键数值有限且位于合法范围；四类输出逐文件计算 SHA-256。
- 正式 complete 要求 3000/3000 Boltz 与 3000/3000 双样本姿势审计完成。
- partial 只能输出 `checkpoint_not_formal`，不能生成正式 384。
- review512 必须是 final1000 子集；原始 D 先用统一量表复核，校准后 D、contradictory、未解决数据库失败和活性代谢物重算项退出；final384 每条条件姿势稳定性为 A/B。
- 审阅后冻结的多样性上限为每药 5、每靶点 8、每骨架 10、enzyme 245；实际 final384 最大值为每药 5、每靶点 8、每骨架 10，enzyme 为 245。

详细方法见 `docs/PRODUCTION_PIPELINE_V4_ZH.md`。
