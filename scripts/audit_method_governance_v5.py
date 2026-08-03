#!/usr/bin/env python3
"""Audit evidence dependence and define the calibrated v5 method contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/calibrated_method_governance_v5"
METHOD_INDEX = (
    ROOT
    / "outputs/current_production_package_v2/methodology_all_methods_zh"
    / "FDA_OLD_DRUG_NEW_TARGET_ALL_METHODS_INDEX_ZH.json"
)
TOP3000 = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4"
    / "pre_boltz_top3000_v4_fully_audited.csv"
)
FINAL1000 = (
    ROOT
    / "outputs/current_production_package_v2/final_delivery_v4"
    / "FINAL1000_RESERVE_FULL_V4.csv"
)
KNOWN96 = (
    ROOT
    / "outputs/current_production_package_v2/full_untruncated_universe_v4"
    / "known_control_boltz96_refined_pose_audited_v4.csv"
)
CONFIG = ROOT / "configs/calibrated_pipeline_v5.yaml"


METHOD_POLICY: dict[str, tuple[str, str, bool, bool, str]] = {}


def policy(
    names: list[str],
    family: str,
    decision: str,
    enters_pair_ranking: bool,
    requires_calibration: bool,
    rationale: str,
) -> None:
    for name in names:
        METHOD_POLICY[name] = (
            family,
            decision,
            enters_pair_ranking,
            requires_calibration,
            rationale,
        )


policy(
    [
        "FDA 小分子结构库整合",
        "ChEMBL-MoA 人源成药锚点构建",
        "UniProt 映射与唯一蛋白序列折叠",
        "Active moiety、盐型、前药与代谢物归一化",
        "RDKit 结构标准化与有效性检查",
        "非 target-engagement 药物实体排除",
    ],
    "entity_integrity",
    "retain_hard_gate",
    False,
    False,
    "Defines the computable entity and scope; failure invalidates the pair but success is not binding evidence.",
)
policy(
    ["靶点实验模态与 target-engagement 可做性分层"],
    "target_eligibility",
    "retain_route_or_stratum",
    False,
    False,
    "Routes targets to assay-specific lanes; it answers how to test, not whether the pair binds.",
)
policy(
    ["结构–序列–配体输入一致性与 provenance 审计", "计算产物 manifest、哈希与可复现性 inventory"],
    "engineering_qc",
    "retain_engineering_qc",
    False,
    False,
    "Required for traceability and mismatch prevention; it contributes no biological support.",
)
policy(
    ["FDA 已知标签靶点与序列等价映射"],
    "calibration_controls",
    "retain_control_mapping",
    False,
    False,
    "Defines positives and leakage exclusions; known labels must never boost discovery ranking.",
)
policy(
    ["Ligand-similarity target fishing"],
    "target_specific_ligand_baseline",
    "retain_calibrated_in_domain_exploitation",
    True,
    True,
    "May rank only within a target whose scaffold-holdout similarity baseline is validated, and only for compounds inside the known-ligand applicability domain; it is exploitation, not remote target fishing.",
)
policy(
    ["本地监督式独立 DTI（ExtraTrees）"],
    "retrieval_baseline",
    "baseline_or_validation_only",
    False,
    True,
    "Useful as a benchmark that a discovery model must beat; highly dependent on known chemical space.",
)
policy(
    ["靶点专属 Morgan-QSAR（scaffold/time holdout）"],
    "target_specific_ligand_model",
    "retain_calibrated_in_domain_exploitation",
    True,
    True,
    "May rank only within targets where scaffold-holdout discrimination is supported and only for compounds inside the known-ligand applicability domain.",
)
policy(
    ["ConPLEx 序列–配体 DTI 预测", "药物内、靶点内与全局相对秩校准"],
    "fast_pair_retrieval",
    "retain_retrieval_only_until_calibrated",
    False,
    True,
    "Fast high-recall retrieval; score and ranks are not affinity or probability without target calibration.",
)
policy(
    ["EviDTI 不确定性 DTI 重排", "DrugCLIP / Drug-The-Whole-Genome 口袋–配体对比学习"],
    "experimental_pair_models",
    "defer_not_production",
    False,
    True,
    "Potential future orthogonal models, but no project-wide leakage-aware calibration currently supports production use.",
)
policy(
    ["其他现代 DTI 模型工程评估"],
    "model_engineering",
    "retain_engineering_assessment",
    False,
    False,
    "Documents feasibility only; unexecuted or incomplete models cannot contribute evidence.",
)
policy(
    ["已知阳性召回与富集审计", "消融、分层随机基线与排名稳定性审计"],
    "validation",
    "retain_validation_only",
    False,
    False,
    "Measures retrospective sensitivity and dependence; without negatives it cannot estimate precision or FDR.",
)
policy(
    [
        "AlphaFold2 / AlphaFold DB 受体结构",
        "实验 holo 结构与已知配体结构映射",
        "fpocket 几何口袋识别",
        "P2Rank 机器学习口袋预测",
        "PUResNet 三维深度学习口袋分割",
        "多口袋模型空间共识",
    ],
    "target_structure_prior",
    "retain_target_prior_no_pair_score",
    False,
    False,
    "Determines receptor and pocket readiness at target level; cannot discriminate drugs on the same target.",
)
policy(
    [
        "DiffDock 扩散式盲对接",
        "AutoDock Vina 经典 docking 与重打分",
        "smina 可定制结构重打分",
        "GNINA 三维卷积神经网络重打分",
        "Boltz-2 蛋白–配体共折叠与 affinity prediction",
    ],
    "pair_structure_model",
    "conditional_pair_evidence_after_target_calibration",
    True,
    True,
    "May rank pairs only on targets where the identical pipeline separates held-out actives from inactives.",
)
policy(
    ["PoseBusters 与本地几何 pose 质控", "ProLIF 蛋白–配体相互作用指纹"],
    "pose_quality",
    "retain_quality_veto_or_interpretation",
    False,
    False,
    "Rejects impossible geometry or describes contacts; passing is not positive affinity evidence.",
)
policy(
    ["重复采样与条件姿势稳定性", "短程分子动力学（MD）pose-retention 审计"],
    "pose_stress_test",
    "retain_conditional_quality_axis",
    False,
    True,
    "Tests conditional local stability; it can veto rapid failures but cannot prove binding free energy.",
)
policy(
    ["基础药物样物性、QED 与规则审计", "PAINS、Brenk 与 NIH 化学干扰警报", "TDC 端点的本地机器学习 ADMET/QSAR"],
    "chemistry_risk",
    "retain_veto_or_risk_only",
    False,
    False,
    "Chemistry risk and developability cannot rescue weak pair physics; severe liabilities may veto a candidate.",
)
policy(
    ["药物暴露、给药途径与实验浓度可行性", "Assay family 与 readout 映射", "阳性对照、阴性对照与 counterscreen 设计"],
    "experimental_readiness",
    "retain_gate_or_readiness",
    False,
    False,
    "Defines whether and how the hypothesis can be falsified; it is not affinity evidence.",
)
policy(
    ["Murcko scaffold、Morgan 指纹与 Butina 聚类"],
    "portfolio_diversity",
    "retain_diversity_constraint",
    False,
    False,
    "Controls portfolio redundancy after evidence stratification; it must not alter scientific evidence values.",
)
policy(
    ["序列同源、靶点家族与 rediscovery 风险审计"],
    "novelty_leakage",
    "retain_queue_label_or_exclusion",
    False,
    False,
    "Separates rediscovery/family transfer from novel target hypotheses; it is not positive binding evidence.",
)
policy(
    ["Open Targets 小分子 tractability 注释"],
    "target_tractability",
    "retain_target_prior_no_pair_score",
    False,
    False,
    "Routes target modality and feasibility; much of its evidence overlaps ChEMBL and PDB.",
)
policy(
    [
        "Open Targets 靶点–疾病证据",
        "TxGNN 药物–疾病知识图谱推断",
        "STRING/HuRI 网络近邻与网络医学",
        "Reactome、GO 与 KEGG 通路/过程注释",
        "LINCS/CMap 与 CREEDS 表达签名反转",
        "GTEx 与 Human Protein Atlas 组织表达语境",
        "DepMap 肿瘤依赖性",
        "知识图谱可解释路径、机制桶与作用方向",
    ],
    "disease_context",
    "retain_post_binding_annotation",
    False,
    False,
    "Explains possible disease context after a binding hypothesis exists; it cannot verify direct binding.",
)
policy(
    [
        "ChEMBL exact-pair 定量活性审计",
        "PubMed/E-utilities 药物–靶点文献检索",
        "FDA 标签机制与 action type 审计",
        "时间分层与上市后证据审计",
        "新颖性与已知机制泄漏审计",
    ],
    "known_novelty_audit",
    "retain_control_or_queue_label",
    False,
    False,
    "Classifies known, rediscovered, post-approval, or unreported hypotheses; literature absence is not a negative.",
)
policy(
    ["结构化 AI agent 机制与可行性审阅"],
    "structured_review",
    "retain_review_no_new_evidence",
    False,
    False,
    "Integrates cited fields and identifies contradictions; it must not create facts or raise physical scores.",
)


COMPONENTS = {
    "pair_conplex_component_v2": (25.0, "fast_pair_retrieval", "replace_with_calibrated_axis"),
    "pair_boltz_component_v2": (30.0, "pair_structure_model", "replace_with_target_calibrated_axis"),
    "target_pocket_prior_component_v2": (15.0, "target_structure_prior", "move_to_target_gate"),
    "target_tractability_component_v2": (10.0, "target_tractability", "move_to_target_stratum"),
    "drug_feasibility_component_v2": (10.0, "chemistry_risk", "move_to_veto_or_readiness"),
    "experimental_feasibility_component_v2": (5.0, "experimental_readiness", "move_to_readiness_axis"),
    "novelty_component_v2": (5.0, "novelty_leakage", "move_to_queue_label"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def method_governance() -> pd.DataFrame:
    source = json.loads(METHOD_INDEX.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for chapter in source["chapters"]:
        for method in chapter["methods"]:
            name = method["name"]
            if name not in METHOD_POLICY:
                raise ValueError(f"Method lacks v5 policy: {name}")
            family, decision, enters, requires, rationale = METHOD_POLICY[name]
            rows.append(
                {
                    "chapter": chapter["title"],
                    "method_name": name,
                    "previous_tag": method["tag"],
                    "evidence_family_v5": family,
                    "decision_v5": decision,
                    "enters_pair_ranking_v5": enters,
                    "requires_positive_negative_calibration": requires,
                    "scientific_question": method["question"],
                    "correct_interpretation": method["interpretation"],
                    "v5_rationale": rationale,
                }
            )
    family, decision, enters, requires, rationale = METHOD_POLICY[
        "靶点专属 Morgan-QSAR（scaffold/time holdout）"
    ]
    rows.append(
        {
            "chapter": "新增：校准后靶点专属模型",
            "method_name": "靶点专属 Morgan-QSAR（scaffold/time holdout）",
            "previous_tag": "NEW_V5",
            "evidence_family_v5": family,
            "decision_v5": decision,
            "enters_pair_ranking_v5": enters,
            "requires_positive_negative_calibration": requires,
            "scientific_question": "已知配体化学空间内，目标药物能否与同靶点活性/弱活性化合物区分？",
            "correct_interpretation": "靶点内、适用域内的监督式结合优先级；不是跨靶点亲和概率，也不是远程骨架发现。",
            "v5_rationale": rationale,
        }
    )
    if len(rows) != source["method_count"] + 1:
        raise ValueError("Method index count mismatch")
    return pd.DataFrame(rows)


def saturation(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column, (max_points, family, action) in COMPONENTS.items():
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        counts = values.value_counts(dropna=False)
        rows.append(
            {
                "scope": scope,
                "component": column,
                "evidence_family_v5": family,
                "n": int(len(values)),
                "unique_values": int(values.nunique()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "max_points": max_points,
                "normalized_std": float(values.std(ddof=0) / max_points),
                "fraction_at_max": float((values == max_points).mean()),
                "dominant_value": float(counts.index[0]),
                "dominant_fraction": float(counts.iloc[0] / len(values)),
                "v5_action": action,
            }
        )
    return pd.DataFrame(rows)


def correlation_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *COMPONENTS,
        "conplex_score",
        "conplex_reference_percentile_v2",
        "boltz_affinity_probability_refined",
        "boltz_confidence_score_refined",
        "boltz_iptm_refined",
        "boltz_ligand_iptm_refined",
        "boltz_complex_iplddt_refined",
        "pose_ligand_rmsd",
        "pose_ligand_centroid_distance",
        "pose_interface_residue_jaccard",
        "conditional_pose_stability_component_v3",
        "priority_score_v4",
        "formal_selection_score_v2",
    ]
    columns = [column for column in columns if column in frame.columns]
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman")
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value):
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "spearman_rho": float(value),
                        "abs_rho": abs(float(value)),
                        "high_dependency_flag": abs(float(value)) >= 0.70,
                    }
                )
    return pd.DataFrame(rows).sort_values(["abs_rho", "left", "right"], ascending=[False, True, True])


def known_positive_comparison(known: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "conplex_score",
        "boltz_affinity_probability_refined",
        "boltz_confidence_score_refined",
        "boltz_iptm_refined",
        "boltz_complex_iplddt_refined",
        "pose_ligand_rmsd",
        "pose_interface_residue_jaccard",
    ]
    rows = []
    for column in columns:
        if column not in known or column not in candidates:
            continue
        positive = pd.to_numeric(known[column], errors="coerce").dropna()
        selected = pd.to_numeric(candidates[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column,
                "known_positive_n": int(len(positive)),
                "known_positive_median": float(positive.median()),
                "final1000_n": int(len(selected)),
                "final1000_median": float(selected.median()),
                "negative_control_n": 0,
                "valid_specificity_inference": False,
            }
        )
    return pd.DataFrame(rows)


def report_text(
    methods: pd.DataFrame,
    saturation_table: pd.DataFrame,
    correlations: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    final = saturation_table[saturation_table["scope"].eq("final1000")].set_index("component")
    high = correlations[correlations["high_dependency_flag"]].head(12)
    decisions = methods["decision_v5"].value_counts()
    decision_lines = "\n".join(f"- `{key}`: {value}" for key, value in decisions.items())
    high_lines = "\n".join(
        f"- `{row.left}` vs `{row.right}`: rho={row.spearman_rho:.3f}"
        for row in high.itertuples()
    )
    return f"""# FDA老药新靶点方法取舍与证据组合审计 v5

生成时间：{now()}

## 结论

当前方法库包含 {len(methods)} 种方法，但它们不能作为平权投票项。v5 将方法拆分为实体硬门、靶点路由、快速召回、经正负校准的 pair 证据、化学/实验否决、疾病后置注释和工程质控。

现有 v4 的 0-100 分是透明的启发式优先级，不是结合概率。Top3000 经过准入后，多数 target-level 和 novelty 分量已饱和；final1000 的总分主要由 Boltz 分量决定。因此 v5 停止让口袋、tractability、新颖性和实验便利重复进入 pair 物理总分。

## 方法决策分布

{decision_lines}

## 关键饱和结果

- final1000 口袋分满分比例：{final.loc['target_pocket_prior_component_v2', 'fraction_at_max']:.3%}
- final1000 新颖性分满分比例：{final.loc['novelty_component_v2', 'fraction_at_max']:.3%}
- final1000 tractability 分满分比例：{final.loc['target_tractability_component_v2', 'fraction_at_max']:.3%}
- final1000 药物可行性分满分比例：{final.loc['drug_feasibility_component_v2', 'fraction_at_max']:.3%}
- final1000 ConPLEx 分量标准差：{final.loc['pair_conplex_component_v2', 'std']:.3f}/25
- final1000 Boltz 分量标准差：{final.loc['pair_boltz_component_v2', 'std']:.3f}/30

## 高相关字段

{high_lines}

这些相关项来自相同模型、相同受体或派生变换，不能被解释为独立支持。特别是 Boltz affinity、Boltz复合分和总分之间的高相关，说明固定总分没有实现真正的多模型校准共识。

## 已知阳性校准边界

当前 Boltz known96 只有阳性，没有配套实验 inactive/negative。它可以检查已知阳性是否能产生完整结构和合理数值，但不能计算 specificity、precision 或 FDR，也不能据此学习 Boltz 与 ConPLEx 的最优权重。

## v5执行合同

1. 实体、结构错配和严重化学风险采用硬门，不加正分。
2. 口袋和 Open Targets tractability 只决定靶点是否可进入某条 assay/model 路线。
3. ConPLEx保留为全空间快速召回器；在完成同靶点校准前不解释为概率。
4. Boltz、Vina/GNINA等只有在同靶点 held-out 阳性与 inactive 上证明区分能力后，才获得该靶点的排序权。
5. 同一模型的 affinity、confidence、ipTM和pose稳定性归入一个证据族；pose质控主要承担否决和不确定性描述。
6. 疾病图谱、通路、表达和组织数据后置，不提高亲和分。
7. 最终输出校准层级、pair物理轴、模型冲突、适用域、化学否决、实验可行性、新颖性和疾病语境，不再只输出一个总分。

完整机器可读规则见 `configs/calibrated_pipeline_v5.yaml`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for path in [METHOD_INDEX, TOP3000, FINAL1000, KNOWN96, CONFIG]:
        if not path.is_file():
            raise FileNotFoundError(path)
    yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    methods = method_governance()
    top3000 = pd.read_csv(TOP3000, low_memory=False)
    final1000 = pd.read_csv(FINAL1000, low_memory=False)
    known96 = pd.read_csv(KNOWN96, low_memory=False)
    saturation_table = pd.concat(
        [saturation(top3000, "top3000"), saturation(final1000, "final1000")],
        ignore_index=True,
    )
    correlations = correlation_audit(final1000)
    comparison = known_positive_comparison(known96, final1000)

    methods.to_csv(out / "METHOD_GOVERNANCE_V5.csv", index=False)
    saturation_table.to_csv(out / "V4_SCORE_SATURATION_AUDIT.csv", index=False)
    correlations.to_csv(out / "V4_SCORE_DEPENDENCY_SPEARMAN.csv", index=False)
    comparison.to_csv(out / "KNOWN96_POSITIVE_ONLY_LIMIT_AUDIT.csv", index=False)
    (out / "METHOD_COMBINATION_AUDIT_V5_ZH.md").write_text(
        report_text(methods, saturation_table, correlations, comparison), encoding="utf-8"
    )

    summary = {
        "status": "passed",
        "created_utc": now(),
        "method_count": int(len(methods)),
        "method_policy_complete": True,
        "top3000_rows": int(len(top3000)),
        "final1000_rows": int(len(final1000)),
        "known_positive_rows": int(len(known96)),
        "known_negative_rows": 0,
        "can_estimate_specificity": False,
        "high_dependency_pairs_abs_rho_ge_0_70": int(correlations["high_dependency_flag"].sum()),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [METHOD_INDEX, TOP3000, FINAL1000, KNOWN96, CONFIG]
        },
        "outputs": {},
    }
    for path in sorted(out.iterdir()):
        if path.name == "METHOD_COMBINATION_AUDIT_V5_SUMMARY.json":
            continue
        summary["outputs"][path.name] = sha256(path)
    (out / "METHOD_COMBINATION_AUDIT_V5_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
