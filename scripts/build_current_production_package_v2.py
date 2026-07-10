#!/usr/bin/env python3
"""Audit and rebuild the current formal candidate package.

This stage does not rerun ConPLEx, pocket prediction, or Boltz-2.  It corrects
classification and calibration issues in the completed refined Top3000, then
produces a reproducible final1000 and a 384-hypothesis nomination queue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import (
    add_evidence_tier_v2,
    add_priority_score_v2,
    annotate_candidate_risk,
    assert_unique_pairs,
    bool_series,
    diverse_select,
    file_sha256,
    require_columns,
    select_formal_packages,
    shared_reference_percentile,
)


DEFAULT_CONFIG = ROOT / "configs" / "current_pipeline_v2.yaml"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def resolve_inputs(config: dict[str, Any]) -> dict[str, Path]:
    return {name: ROOT / value for name, value in config["inputs"].items()}


def validate_inputs(inputs: dict[str, Path], contracts: dict[str, int]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing production input {name}: {path}")
        frames[name] = pd.read_csv(path, low_memory=False).fillna("")
    contract_names = {
        "all_direct_rows": "all_direct_pairs",
        "discovery_rows": "discovery_pairs",
        "known_control_rows": "known_controls",
        "refined_top3000_rows": "refined_top3000",
        "current_final1000_rows": "current_final1000",
        "current_final384_rows": "current_final384",
    }
    failures = []
    for contract, frame_name in contract_names.items():
        expected = int(contracts[contract])
        observed = len(frames[frame_name])
        if observed != expected:
            failures.append(f"{frame_name}: expected {expected}, observed {observed}")
    if failures:
        raise ValueError("Input row-count contracts failed: " + "; ".join(failures))
    for name in ["all_direct_pairs", "discovery_pairs", "known_controls", "refined_top3000", "current_final1000", "current_final384"]:
        assert_unique_pairs(frames[name], name)
    return frames


def known_calibration(all_pairs: pd.DataFrame, known: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(all_pairs, ["conplex_score", "rank_within_drug", "is_known_fda_target_pair"], "all_direct_pairs")
    calibration = known.copy()
    calibration["conplex_reference_percentile_v2"] = shared_reference_percentile(
        calibration["conplex_score"], all_pairs["conplex_score"]
    )
    score = pd.to_numeric(all_pairs["conplex_score"], errors="coerce").fillna(0)
    known_mask = bool_series(all_pairs, "is_known_fda_target_pair")
    baseline = float(known_mask.mean())
    enrichments = []
    for fraction in [0.01, 0.05, 0.10, 0.25]:
        n = max(1, round(len(all_pairs) * fraction))
        top_index = score.nlargest(n, keep="first").index
        retained = int(known_mask.loc[top_index].sum())
        prevalence = retained / n
        enrichments.append(
            {
                "top_fraction": fraction,
                "rows": n,
                "known_retained": retained,
                "known_total": int(known_mask.sum()),
                "known_recall": retained / max(int(known_mask.sum()), 1),
                "known_prevalence": prevalence,
                "enrichment_over_all_direct": prevalence / baseline if baseline else None,
            }
        )
    ranks = pd.to_numeric(calibration["rank_within_drug"], errors="coerce").fillna(9999)
    summary = {
        "scope": "known-vs-unlabelled calibration within the 106803 direct-action target-engagement space",
        "known_total": int(len(calibration)),
        "baseline_known_prevalence": baseline,
        "known_conplex_median": float(pd.to_numeric(calibration["conplex_score"], errors="coerce").median()),
        "known_reference_percentile_median": float(calibration["conplex_reference_percentile_v2"].median()),
        "recall_by_drug_rank": {
            f"top{cutoff}": float((ranks <= cutoff).mean()) for cutoff in [10, 50, 100, 300]
        },
        "enrichment": enrichments,
        "warning": "Unlabelled pairs are not confirmed negatives; these figures are calibration/enrichment, not accuracy.",
    }
    return calibration, summary


def attach_membership(df: pd.DataFrame, v1_1000: pd.DataFrame, v1_384: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key1000 = set(zip(v1_1000["drug_chembl_id"], v1_1000["sequence_key"]))
    key384 = set(zip(v1_384["drug_chembl_id"], v1_384["sequence_key"]))
    keys = list(zip(out["drug_chembl_id"], out["sequence_key"]))
    out["in_v1_final1000"] = [key in key1000 for key in keys]
    out["in_v1_final384"] = [key in key384 for key in keys]
    return out


def build_teacher_table(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "final384_rank": "384推荐顺序",
        "final1000_rank": "1000推荐顺序",
        "drug_names": "药物",
        "fda_therapeutic_area": "原治疗领域",
        "fda_indication": "原适应症",
        "fda_moa": "原FDA_MoA",
        "fda_action_type": "原作用类型",
        "fda_target_names": "原FDA靶点",
        "primary_gene": "候选靶点",
        "protein_names": "候选靶点蛋白",
        "target_assay_family_v2": "校正后实验类型",
        "evidence_tier_v2": "物理证据等级",
        "priority_score_v2": "优先级分数",
        "pair_specific_evidence_score_v2": "pair特异物理证据分",
        "conplex_score": "ConPLEx分数",
        "rank_within_drug": "药物内ConPLEx排名",
        "target_rank": "靶点内ConPLEx排名",
        "boltz_support_tier_refined": "Boltz精修等级",
        "boltz_review_class_v3": "Boltz多指标复核类别",
        "boltz_affinity_probability_refined": "Boltz_affinity_probability",
        "boltz_ligand_iptm_refined": "Boltz_ligand_iPTM",
        "pose_stability_tier": "Boltz双样本条件姿势稳定性",
        "pose_ligand_rmsd": "双样本配体RMSD_A",
        "pose_interface_residue_jaccard": "双样本界面残基Jaccard",
        "structure_bin": "靶点口袋等级",
        "max_known_target_local_identity": "与原已知靶点最大局部序列一致性",
        "max_known_target_local_coverage": "同源比对短序列覆盖度",
        "nearest_known_target_gene": "最近原已知靶点",
        "sequence_homology_extension_risk": "同源家族扩展风险",
        "compound_liability_notes": "化合物实验风险警报",
        "assay_interference_review": "需实验干扰复核",
        "brenk_developability_review": "Brenk药化复核",
        "severe_compound_liability": "严重化合物风险",
        "multi_product_label_review": "多产品标签复核",
        "composite_drug_id_review": "复合映射ID复核",
        "anchor_project_standard_direct_sm": "OT小分子可做性",
        "candidate_role_v2": "候选角色",
        "candidate_action_status": "候选作用方向状态",
        "pair_evidence_limit": "证据边界",
        "default_assay_strategy": "默认实验策略",
        "risk_notes_v2": "风险说明",
        "fda_route": "给药途径",
        "model_ligand_smiles": "统一模型配体SMILES",
        "active_moiety_smiles": "活性母体SMILES",
        "canonical_smiles": "FDA原结构SMILES",
    }
    present = [column for column in mapping if column in df.columns]
    table = df[present].rename(columns=mapping).copy()
    for column in [
        "优先级分数",
        "pair特异物理证据分",
        "ConPLEx分数",
        "Boltz_affinity_probability",
        "Boltz_ligand_iPTM",
        "与原已知靶点最大局部序列一致性",
        "同源比对短序列覆盖度",
        "双样本配体RMSD_A",
        "双样本界面残基Jaccard",
    ]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce").round(4)
    return table


def assay_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["target_assay_family_v2", "primary_gene"], dropna=False)
        .agg(
            pair_count=("pair_id", "count"),
            unique_drugs=("drug_chembl_id", "nunique"),
            boltz_A=("boltz_support_tier_refined", lambda s: int(s.astype(str).str.startswith("A_").sum())),
            median_priority=("priority_score_v2", "median"),
        )
        .reset_index()
        .sort_values(["target_assay_family_v2", "pair_count", "median_priority"], ascending=[True, False, False])
    )


def comparison_table(top3000: pd.DataFrame, final1000: pd.DataFrame, final384: pd.DataFrame) -> pd.DataFrame:
    out = top3000[
        [
            "pair_id",
            "drug_chembl_id",
            "drug_names",
            "primary_gene",
            "target_assay_family_legacy",
            "target_assay_family_v2",
            "assay_family_corrected",
            "candidate_role_v2",
            "risk_notes_v2",
            "evidence_tier_v2",
            "priority_score_v2",
            "in_v1_final1000",
            "in_v1_final384",
        ]
    ].copy()
    final1000_keys = set(final1000["pair_id"])
    final384_keys = set(final384["pair_id"])
    out["in_v2_final1000"] = out["pair_id"].isin(final1000_keys)
    out["in_v2_final384"] = out["pair_id"].isin(final384_keys)
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def write_report(path: Path, summary: dict[str, Any]) -> None:
    mismatch = summary["classification_audit"]
    v2 = summary["v2_outputs"]
    calibration = summary["known_calibration"]
    report = f"""# FDA 老药新用项目实现审计与 v2 流水线报告

生成时间：{summary['created_utc']}

## 审计结论

当前项目已经完成一条可用的物理优先候选链：106,561 条 discovery pair 经 ConPLEx、靶点口袋和 Boltz-2 refined Top3000 后形成候选包。原始计算结果可以保留，但旧版正式排序存在三项实现偏差，不能原样视为最终湿实验板：

1. 旧靶点实验类型有 {mismatch['assay_family_corrected_rows_3000']} / 3,000 条被 Open Targets 标准分类纠正；其中旧版漏标 kinase {mismatch['legacy_nonkinase_to_ot_kinase_rows_3000']} 条。
2. 旧 `same_family_or_label_risk` 把 assay 大类当作蛋白家族；v2 已拆为精确已知、具体家族扩展、kinase-to-kinase、核受体扩展、CA 再发现和 ion-channel 可行性六类。
3. 已知对照和 discovery 的 ConPLEx 百分位曾分别计算。v2 统一使用 106,803 条 direct-action 空间作为参考分布。

## v2 输出

- refined Top3000 重审：{v2['top3000_rows']} 条。
- v2 final1000：{v2['final1000_rows']} 条，{v2['final1000_unique_drugs']} 个药物，{v2['final1000_unique_targets']} 个靶点；Boltz A/B {v2['final1000_boltz_ab']} 条。
- v2 384 提名队列：{v2['final384_rows']} 条，{v2['final384_unique_drugs']} 个药物，{v2['final384_unique_targets']} 个靶点，全部为 Boltz A/B、非已知 pair、非家族/再发现风险且非 ion channel。
- v1 final1000 与 v2 final1000 重叠：{v2['v1_v2_final1000_overlap']} 条。
- v1 384 与 v2 384 重叠：{v2['v1_v2_final384_overlap']} 条。

## 阳性校准

- 校准空间：106,803 条 direct-action target-engagement pair，其中 242 条已知 FDA direct-action pair。
- 已知 pair 的共享参考 ConPLEx 百分位中位数：{calibration['known_reference_percentile_median']:.3f}。
- 药物内 Recall@50：{calibration['recall_by_drug_rank']['top50']:.2%}；Recall@100：{calibration['recall_by_drug_rank']['top100']:.2%}；Recall@300：{calibration['recall_by_drug_rank']['top300']:.2%}。
- 这些数字是已知-vs-未标注的校准与富集，不是未知互作准确率；未标注 pair 不能当作真实阴性。

## 证据解释

- ConPLEx 是序列-SMILES 的 pair 排序信号，不能单独证明亲和。
- AlphaFold/P2Rank/PUResNet 是靶点级口袋先验；同一靶点的所有药物共享该信息，不能当作药物特异证据。
- 当前 Boltz-2 使用 P2Rank pocket contacts 作为非强制约束、empty MSA 和有限采样。A/B 表示在指定口袋条件下的二阶段结构兼容信号，不是盲对接成功率，也不是实验 Kd/Ki。
- Open Targets tractability 说明靶点适合小分子研究，不说明当前药物会结合，也不说明适应症有效。

## 湿实验边界

`final384_nomination_v2` 是 384 条 drug-target 假说，不是可直接移液的 4x96 板图。它覆盖多个靶点和 assay family，且尚未为每个 assay 定义剂量、重复、阳性对照、阴性对照和 counterscreen。正式板图应在实验组确定 assay 后按靶点/读数分组生成。

当前仍缺少的 go/no-go 证据：pair 级文献新颖性审计、具体 assay/reagent/readout、可达暴露与建议浓度、Boltz pose 的 clash/contact/关键残基审计，以及独立于现有选择规则的盲法验证集。
"""
    path.write_text(report, encoding="utf-8")


def build(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = resolve_inputs(config)
    frames = validate_inputs(inputs, config["contracts"])
    out_dir = ROOT / config["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = frames["all_direct_pairs"]
    known = frames["known_controls"]
    current1000 = frames["current_final1000"]
    current384 = frames["current_final384"]
    top3000 = annotate_candidate_risk(frames["refined_top3000"], known, frames["anchor_table"])
    top3000 = add_priority_score_v2(top3000, all_pairs["conplex_score"])
    top3000 = add_evidence_tier_v2(top3000)
    top3000 = attach_membership(top3000, current1000, current384)

    completed = bool_series(top3000, "boltz_completed_refined")
    final1000, final384 = select_formal_packages(top3000, config["selection"])
    f1000_cfg = config["selection"]["final1000"]
    f384_cfg = config["selection"]["final384"]

    risk_pool = top3000[
        completed & ~top3000["exact_known_target_v2"] & top3000["family_or_rediscovery_risk_v2"]
    ].copy()
    positive_controls = diverse_select(
        risk_pool,
        min(64, len(risk_pool)),
        drug_cap=4,
        target_cap=8,
        scaffold_cap=10,
        family_caps={"kinase": 24, "enzyme": 20, "nuclear_epigenetic": 16, "transporter": 4},
    )

    if len(final1000) != int(f1000_cfg["size"]):
        raise RuntimeError(f"v2 final1000 has {len(final1000)} rows")
    if len(final384) != int(f384_cfg["size"]):
        raise RuntimeError(f"v2 final384 has {len(final384)} rows")
    assert_unique_pairs(final1000, "v2 final1000")
    assert_unique_pairs(final384, "v2 final384")
    if final384["family_or_rediscovery_risk_v2"].any() or final384["exact_known_target_v2"].any():
        raise RuntimeError("v2 final384 contains a blocked novelty/control risk")
    if not set(final384["pair_id"]).issubset(set(final1000["pair_id"])):
        raise RuntimeError("v2 final384 is not a strict subset of v2 final1000")

    calibration, calibration_summary = known_calibration(all_pairs, known)
    comparison = comparison_table(top3000, final1000, final384)
    assay = assay_summary(final384)
    top3000.to_csv(out_dir / "refined_top3000_reaudited_v2.csv", index=False)
    final1000.to_csv(out_dir / "final1000_candidates_v2.csv", index=False)
    final384.to_csv(out_dir / "final384_nomination_v2.csv", index=False)
    build_teacher_table(final1000).to_csv(out_dir / "final1000_teacher_readable_zh_v2.csv", index=False)
    build_teacher_table(final384).to_csv(out_dir / "final384_teacher_readable_zh_v2.csv", index=False)
    positive_controls.to_csv(out_dir / "positive_control_family_extension_queue_v2.csv", index=False)
    calibration.to_csv(out_dir / "known_control_calibration_shared_scale_v2.csv", index=False)
    comparison.to_csv(out_dir / "v1_v2_pair_membership_and_risk_comparison.csv", index=False)
    assay.to_csv(out_dir / "final384_assay_group_summary_v2.csv", index=False)
    with pd.ExcelWriter(out_dir / "CURRENT_PRODUCTION_PACKAGE_V2.xlsx", engine="openpyxl") as writer:
        build_teacher_table(final384).to_excel(writer, index=False, sheet_name="384_nomination")
        build_teacher_table(final1000).to_excel(writer, index=False, sheet_name="final1000")
        build_teacher_table(positive_controls).to_excel(writer, index=False, sheet_name="control_queue")
        assay.to_excel(writer, index=False, sheet_name="assay_groups")

    current1000_keys = set(zip(current1000["drug_chembl_id"], current1000["sequence_key"]))
    current384_keys = set(zip(current384["drug_chembl_id"], current384["sequence_key"]))
    v2_1000_keys = set(zip(final1000["drug_chembl_id"], final1000["sequence_key"]))
    v2_384_keys = set(zip(final384["drug_chembl_id"], final384["sequence_key"]))
    summary = {
        "created_utc": now_utc(),
        "pipeline_name": config["name"],
        "pipeline_version": config["version"],
        "git_commit": git_commit(),
        "classification_audit": {
            "assay_family_corrected_rows_3000": int(top3000["assay_family_corrected"].sum()),
            "legacy_nonkinase_to_ot_kinase_rows_3000": int(
                ((top3000["target_assay_family_legacy"] != "kinase") & (top3000["target_assay_family_v2"] == "kinase")).sum()
            ),
            "family_or_rediscovery_risk_rows_3000": int(top3000["family_or_rediscovery_risk_v2"].sum()),
            "kinase_to_kinase_rows_3000": int(top3000["kinase_to_kinase_risk"].sum()),
            "nuclear_receptor_extension_rows_3000": int(top3000["nuclear_receptor_extension_risk"].sum()),
            "carbonic_anhydrase_rows_3000": int(top3000["carbonic_anhydrase_rediscovery_risk"].sum()),
            "specific_family_extension_rows_3000": int(top3000["specific_target_family_extension_risk"].sum()),
            "ion_channel_feasibility_rows_3000": int(top3000["ion_channel_feasibility_flag"].sum()),
        },
        "known_calibration": calibration_summary,
        "v2_outputs": {
            "top3000_rows": int(len(top3000)),
            "final1000_rows": int(len(final1000)),
            "final1000_unique_drugs": int(final1000["drug_chembl_id"].nunique()),
            "final1000_unique_targets": int(final1000["primary_gene"].nunique()),
            "final1000_boltz_ab": int(final1000["boltz_support_tier_refined"].astype(str).str.startswith(("A_", "B_")).sum()),
            "final1000_tiers": final1000["evidence_tier_v2"].value_counts().to_dict(),
            "final1000_assay_families": final1000["target_assay_family_v2"].value_counts().to_dict(),
            "final384_rows": int(len(final384)),
            "final384_unique_drugs": int(final384["drug_chembl_id"].nunique()),
            "final384_unique_targets": int(final384["primary_gene"].nunique()),
            "final384_unique_scaffolds": int(final384["murcko_scaffold"].replace("", "NO_SCAFFOLD").nunique()),
            "final384_tiers": final384["evidence_tier_v2"].value_counts().to_dict(),
            "final384_assay_families": final384["target_assay_family_v2"].value_counts().to_dict(),
            "v1_v2_final1000_overlap": int(len(current1000_keys & v2_1000_keys)),
            "v1_v2_final384_overlap": int(len(current384_keys & v2_384_keys)),
            "positive_control_queue_rows": int(len(positive_controls)),
            "plate_ready": False,
            "plate_ready_reason": "Multiple target-specific assays; dose, replicates, controls and counterscreens are not yet defined.",
            "pair_level_literature_audit_coverage": 0,
        },
    }
    write_json(out_dir / "production_audit_summary_v2.json", summary)
    write_report(out_dir / "IMPLEMENTATION_AUDIT_AND_PIPELINE_V2_ZH.md", summary)
    manifest = {
        "created_utc": summary["created_utc"],
        "config": str(config_path.relative_to(ROOT)),
        "git_commit": summary["git_commit"],
        "inputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "rows": int(len(frames[name])),
            }
            for name, path in inputs.items()
        },
        "outputs": sorted(str(path.relative_to(ROOT)) for path in out_dir.iterdir() if path.is_file()),
    }
    write_json(out_dir / "artifact_manifest_v2.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and build the current production package v2.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    summary = build(config_path)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
