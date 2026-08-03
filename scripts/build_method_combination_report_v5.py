#!/usr/bin/env python3
"""Generate the Chinese method-combination and calibrated portfolio report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current_production_package_v2/calibrated_portfolio_v5"
REPORT = ROOT / "docs/FDA_OLD_DRUG_METHOD_COMBINATION_AND_PORTFOLIO_V5_ZH.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percent(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f}%" if total else "NA"


def main() -> None:
    chembl = load_json(
        ROOT
        / "outputs/current_production_package_v2/chembl37_target_calibration_v5/CHEMBL37_CALIBRATION_SUMMARY_V5.json"
    )
    conplex = load_json(
        ROOT
        / "outputs/current_production_package_v2/conplex_target_calibration_v5_official/evaluation/CONPLEX_TARGET_CALIBRATION_SUMMARY_V5.json"
    )
    qsar = load_json(
        ROOT / "outputs/current_production_package_v2/target_qsar_calibration_v5/TARGET_QSAR_CALIBRATION_SUMMARY_V5.json"
    )
    boltz = load_json(
        ROOT
        / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/evaluation/BOLTZ_SMOKE_CALIBRATION_DECISION_V5.json"
    )
    portfolio = load_json(OUT / "CALIBRATED_PORTFOLIO_V5_SUMMARY.json")
    invariant = load_json(OUT / "CALIBRATED_PORTFOLIO_INVARIANT_AUDIT_V5.json")
    governance_summary = load_json(
        ROOT
        / "outputs/current_production_package_v2/calibrated_method_governance_v5/METHOD_COMBINATION_AUDIT_V5_SUMMARY.json"
    )
    governance = pd.read_csv(
        ROOT / "outputs/current_production_package_v2/calibrated_method_governance_v5/METHOD_GOVERNANCE_V5.csv"
    )
    dependency = pd.read_csv(
        ROOT / "outputs/current_production_package_v2/calibrated_method_governance_v5/V4_SCORE_DEPENDENCY_SPEARMAN.csv"
    )
    conplex_metrics = pd.read_csv(
        ROOT
        / "outputs/current_production_package_v2/conplex_target_calibration_v5_official/evaluation/CONPLEX_TARGET_CALIBRATION_METRICS_V5.csv"
    )
    qsar_metrics = pd.read_csv(
        ROOT / "outputs/current_production_package_v2/target_qsar_calibration_v5/TARGET_QSAR_SCAFFOLD_HOLDOUT_METRICS_V5.csv"
    )
    final = pd.read_csv(OUT / "FINAL1000_EVIDENCE_STRATIFIED_V5.csv", low_memory=False)

    conplex_medians = conplex_metrics[
        conplex_metrics["conplex_target_use_status_v5"].eq("T2_signal_without_baseline_superiority")
    ][["conplex_pr_auc", "similarity_pr_auc", "conplex_roc_auc", "similarity_roc_auc"]].median()
    qsar_t1 = qsar_metrics[qsar_metrics["target_ligand_model_status_v5"].eq("T1_qsar_beats_similarity")]
    qsar_medians = qsar_t1[
        ["qsar_oof_pr_auc", "similarity_oof_pr_auc", "qsar_oof_roc_auc", "similarity_oof_roc_auc"]
    ].median()
    temporal_counts = qsar_metrics["temporal_validation_status_v5"].value_counts().to_dict()
    rho_boltz = dependency[
        ((dependency["left"] == "pair_boltz_component_v2") & (dependency["right"] == "priority_score_v4"))
        | ((dependency["right"] == "pair_boltz_component_v2") & (dependency["left"] == "priority_score_v4"))
    ]["spearman_rho"].iloc[0]
    rho_affinity = dependency[
        ((dependency["left"] == "boltz_affinity_probability_refined") & (dependency["right"] == "priority_score_v4"))
        | ((dependency["right"] == "boltz_affinity_probability_refined") & (dependency["left"] == "priority_score_v4"))
    ]["spearman_rho"].iloc[0]
    family = final["target_assay_family_v5"].value_counts().to_dict()
    indications = final["original_indication_v5"].ne("原FDA表未提供").sum()

    lines = [
        "# FDA 老药新靶点：方法组合、校准结果与分层候选包 V5",
        "",
        "## 一、结论",
        "",
        "本项目不再把所有方法换算成一个 100 分总分。方法按其能回答的科学问题分工：",
        "",
        "1. 实体、化学风险和序列一致性用于硬门控。",
        "2. 口袋、结构和 Open Targets tractability 用于靶点可做性与实验路由。",
        "3. 只有经过同靶点阳性–阴性校准、scaffold holdout 和适用域检查的方法，才允许给具体 drug–target pair 排序。",
        "4. ConPLEx 降为远程探索召回；Boltz 降为条件结构/pose 证据；二者均不再解释为结合概率。",
        "5. 疾病图谱、通路、组织表达和文献用于结合假说之后的适应症与机制解释，不参与亲和主排序。",
        "",
        f"最终形成 1000 条分层组合：P1 {portfolio['portfolio_lane_counts'].get('P1_calibrated_target_qsar_in_domain', 0)} 条、P2 {portfolio['portfolio_lane_counts'].get('P2_validated_ligand_similarity_in_domain', 0)} 条、P3 {portfolio['portfolio_lane_counts'].get('P3_remote_uncalibrated_physics_exploration', 0)} 条。三层不是同一置信度，不能混称为 1000 条高置信结合候选。",
        "",
        "## 二、为什么旧的综合加权口径不能继续",
        "",
        f"旧流程共使用 {governance_summary['method_count']} 类方法，但其中很多输出回答的是同一个问题，或只是靶点级先验。V4 最终分与 Boltz 分量 Spearman 相关系数为 {rho_boltz:.3f}，与 Boltz affinity 原始输出为 {rho_affinity:.3f}；这说明旧总分在实际排序中主要由 Boltz 驱动。",
        "",
        f"旧 Boltz 校准集只有 {governance_summary['known_positive_rows']} 条阳性、{governance_summary['known_negative_rows']} 条阴性，因此无法估计 specificity、precision 或 FDR。口袋共识、tractability、药物物性和 assay readiness 即使全部通过，也不能证明某个具体药物结合该靶点。",
        "",
        "## 三、ChEMBL 37 正负校准基准",
        "",
        f"- 项目靶点：{chembl['project_targets']} 个；canonical UniProt 修正后映射 {chembl['mapped_targets']}/{chembl['project_targets']}。",
        f"- 严格定量或明确 inactive 的靶点–化合物记录：{chembl['strict_pair_rows']:,} 条。",
        f"- 强活性：{chembl['positive_pair_rows']:,}；弱活性/不活跃：{chembl['negative_pair_rows']:,}；灰区：{chembl['grey_pair_rows']:,}；冲突排除：{chembl['conflicting_pair_rows']:,}。",
        f"- 靶点数据层级：T1 50+50 为 {chembl['target_tiers'].get('T1_target_calibrated', 0)}；T2 20+20 为 {chembl['target_tiers'].get('T2_target_calibrated_limited', 0)}；正类为主 T3 为 {chembl['target_tiers'].get('T3_positive_only', 0)}；稀疏 T4 为 {chembl['target_tiers'].get('T4_sparse', 0)}。",
        "",
        "标签口径为人源 SINGLE PROTEIN、binding assay、confidence 9、Ki/Kd/IC50、等号关系；pChEMBL >= 6 为强活性，<= 5 或明确 inactive 为负类，5–6 为灰区。它是项目校准定义，不等价于所有实验语境下的绝对真值。",
        "",
        "## 四、模型实测与取舍",
        "",
        "| 方法 | 校准结果 | 正式角色 |",
        "| --- | --- | --- |",
        f"| ConPLEx | {conplex['targets_with_20_each']} 个靶点具备 20+20；无靶点显著优于配体相似性。T2 中位 PR-AUC {conplex_medians['conplex_pr_auc']:.3f}，相似性 {conplex_medians['similarity_pr_auc']:.3f} | 仅作远程探索召回，不作独立物理证据 |",
        f"| 靶点专属 Morgan-QSAR | 402 个靶点评估；时间审计后 {qsar['target_status'].get('T1_qsar_beats_similarity', 0)} 个 T1、{qsar['target_status'].get('T2_similarity_supported', 0)} 个 T2 | T1 且处于适用域时可作 P1 排序 |",
        f"| 配体相似性 | 在多数有数据靶点上强于 ConPLEx；但依赖已知配体化学空间 | P2 exploitation，不包装成远程新发现 |",
        f"| Boltz-2 | 20 条、10 个同靶点正负对；affinity ROC-AUC {boltz['affinity_roc_auc']:.2f}、阳性胜 {boltz['affinity_positive_wins']}/10；综合分 ROC-AUC {boltz['composite_roc_auc']:.2f}、阳性胜 {boltz['composite_positive_wins']}/10 | 停止扩展至 1000；仅作 pose 可生成性/条件质量 |",
        "| AlphaFold2、P2Rank、PUResNet、holo pocket | 靶点级结构与口袋信息 | 路由、模板和硬门控，不给 pair 加结合分 |",
        "| docking、pose consistency、PoseBusters、ProLIF | 可检查几何和接触合理性 | 质量否决或解释；未做同靶点正负校准前不排名 |",
        "| MD | 可检查给定 pose 的短程保持 | 只用于少量入围 pose 压力测试，不用于全量召回 |",
        "| Open Targets、TxGNN、STRING、通路、表达和组织 | 疾病/机制语境 | 结合之后作 disease-mechanism 收敛 |",
        "",
        f"P1 靶点在 scaffold holdout 下的中位 PR-AUC 为 {qsar_medians['qsar_oof_pr_auc']:.3f}，相似性基线为 {qsar_medians['similarity_oof_pr_auc']:.3f}；中位 ROC-AUC 为 {qsar_medians['qsar_oof_roc_auc']:.3f}。时间切分中，{temporal_counts.get('supports_or_noninferior', 0)} 个靶点支持或不劣于相似性，{temporal_counts.get('contradicts_scaffold_result', 0)} 个出现矛盾并被降级，{temporal_counts.get('not_evaluable', 0)} 个因新时期正负样本不足无法评估。",
        "",
        "## 五、新的组合流程",
        "",
        "### 1. 实体与范围硬门控",
        "",
        "FDA active moiety、盐型/前药归一化、RDKit 解析、非治疗性实体排除、项目 463 个直接 target-engagement 靶点、序列与结构 provenance。通过只表示可以计算和实验，不构成结合支持。",
        "",
        "### 2. 靶点路由",
        "",
        "根据 assay family、实验结构/AlphaFold、口袋共识和 Open Targets tractability 决定该靶点是否进入 enzyme、kinase、transporter、ion-channel 或 nuclear/epigenetic 实验通道。",
        "",
        "### 3. 已知化学空间 exploitation",
        "",
        "每个靶点单独训练 Morgan-QSAR，以 Murcko scaffold 五折留出评估，并与同折配体相似性比较；再做 2022 年及以前训练、2023–2025 年首次记录测试。只有显著优于相似性、无时间矛盾且 FDA 药物位于已知配体适用域的 pair 进入 P1。相似性有效但 QSAR 无显著增益的进入 P2。",
        "",
        "### 4. 远程探索",
        "",
        "与靶点已知活性配体最大 Tanimoto < 0.40 或无可用已知配体映射的候选进入 P3。ConPLEx 只负责召回；Boltz 完成、序列匹配、pose A/B 只作为结构可生成门控。P3 没有经校准的结合概率。",
        "",
        "### 5. 机制与疾病后处理",
        "",
        "候选通过结合证据分层后，再用 Open Targets target–disease、通路网络、组织表达、转录扰动和文献判断作用方向、疾病细分和实验 readout。疾病证据不能反向补强一个物理证据不足的 pair。",
        "",
        "## 六、最终 1000 条的构成",
        "",
        f"- P1：{portfolio['portfolio_lane_counts'].get('P1_calibrated_target_qsar_in_domain', 0)} 条。靶点专属 QSAR 经 scaffold 校准、无时间矛盾、处于适用域。",
        f"- P2：{portfolio['portfolio_lane_counts'].get('P2_validated_ligand_similarity_in_domain', 0)} 条。已知配体相似性有效，但新颖性较低。",
        f"- P3：{portfolio['portfolio_lane_counts'].get('P3_remote_uncalibrated_physics_exploration', 0)} 条。远程骨架、结构可生成、适合探索，但亲和未校准。",
        f"- 覆盖 {portfolio['portfolio_unique_drugs']} 个药物、{portfolio['portfolio_unique_targets']} 个靶点、{portfolio['portfolio_unique_scaffolds']} 个 Murcko scaffold；与旧 Final1000 重叠 {portfolio['previous_final1000_overlap']} 条。",
        f"- 靶点类型：enzyme {family.get('enzyme', 0)}、kinase {family.get('kinase', 0)}、nuclear/epigenetic {family.get('nuclear_epigenetic', 0)}、transporter {family.get('transporter', 0)}、ion channel {family.get('ion_channel', 0)}。",
        f"- 原 FDA 适应症覆盖 {indications}/1000；该字段只描述药物原用途，不是推荐新病种。",
        "- 已知 FDA pair、同家族扩展风险和 exact known-active structure 均不进入 discovery 1000；另有 family-extension review 和 positive-control/rediscovery 队列单列。",
        "",
        "## 七、可以与不可以声称的内容",
        "",
        "可以声称：项目建立了同靶点正负校准；识别了哪些模型在什么适用域内有增益；形成了 1000 条证据分层、可审计、非已知 FDA pair 的组合。",
        "",
        "不可以声称：P1 是真实结合概率；P2 是全新骨架发现；P3 已被 Boltz 证明结合；1000 条具有相同质量；疾病图谱能证明直接亲和。",
        "",
        "## 八、下一步",
        "",
        "1. 计算上不继续全量扩展 Boltz 1000 校准，除非更换 pocket protocol 后先通过新的正负配对门槛。",
        "2. P1/P2 优先做 exact-pair 文献与 ChEMBL 排重、active species、暴露和 assay readiness 深审。",
        "3. P3 若要推进，应作为探索/主动学习队列，实验同时配置同靶点阳性和阴性；其首轮价值是产生项目自己的校准数据。",
        "4. 只在 pair 通过上述层级后再选择疾病、作用方向和细胞 readout。",
        "",
        "## 九、主要交付文件",
        "",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/FINAL1000_EVIDENCE_STRATIFIED_V5.csv`",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/FINAL1000_EVIDENCE_STRATIFIED_TEACHER_ZH_V5.csv`",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/P1_CALIBRATED_TARGET_QSAR_IN_DOMAIN_V5.csv`",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/P2_VALIDATED_LIGAND_SIMILARITY_IN_DOMAIN_V5.csv`",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/P3_REMOTE_UNCALIBRATED_PHYSICS_EXPLORATION_V5.csv`",
        "- `outputs/current_production_package_v2/calibrated_portfolio_v5/CALIBRATED_PORTFOLIO_INVARIANT_AUDIT_V5.json`",
        "",
        "## 附录：58 类方法的正式角色",
        "",
        "| 方法 | 证据家族 | V5 决策 | 满足校准条件后可进入 pair 排序 |",
        "| --- | --- | --- | --- |",
    ]
    for row in governance.itertuples(index=False):
        lines.append(
            f"| {row.method_name} | {row.evidence_family_v5} | {row.decision_v5} | {'是' if row.enters_pair_ranking_v5 else '否'} |"
        )
    lines.extend(
        [
            "",
            f"完整性审计：`{invariant['status']}`；失败项 {len(invariant['failures'])}；Final1000 SHA256 `{invariant['full_sha256']}`。",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    (OUT / REPORT.name).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(REPORT), "lines": len(lines)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
