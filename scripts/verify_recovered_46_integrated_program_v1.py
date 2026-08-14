#!/usr/bin/env python3
"""Requirement-level completion audit for the 46-target recovery program."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTEGRATED = ROOT / "outputs/recovered_target_program_integrated_v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main() -> None:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, requirement: str, observed: Any, expected: Any, passed: bool, evidence: Path) -> None:
        checks.append({
            "check_id": check_id,
            "requirement": requirement,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "evidence_file": str(evidence),
        })

    recovered_path = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1/RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
    exclusion_path = ROOT / "outputs/final_target_package_ch37/FINAL_TARGET_EXCLUSION_AUDIT_888.csv"
    recovered = pd.read_csv(recovered_path, low_memory=False)
    exclusion = pd.read_csv(exclusion_path, low_memory=False)
    exact_ids = set(exclusion.loc[
        exclusion["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"),
        "target_chembl_id",
    ])
    recovery_ids = set(recovered["target_chembl_id"])
    check("SCOPE_01", "正式找回范围恰为首次因无合格实验口袋淘汰的靶点", len(recovery_ids), 46, recovery_ids == exact_ids and len(recovered) == 46, recovered_path)
    pocket_counts = recovered["computed_pocket_evidence"].value_counts().to_dict()
    expected_pockets = {
        "P1_P2RANK_FPOCKET_SAME_SITE": 37,
        "P2_FPOCKET_GEOMETRIC_RESCUE": 7,
        "P3_LOW_CONFIDENCE_FRAGMENT_HYPOTHESIS": 2,
    }
    check("POCKET_01", "46个靶点均有明确的预测口袋证据等级", pocket_counts, expected_pockets, pocket_counts == expected_pockets, recovered_path)

    dta_path = ROOT / "outputs/recovered_dta_720x46_v1/RECOVERED_DTA_720_X_46_EVIDENCE_MATRIX_V1.csv.gz"
    dta = pd.read_csv(dta_path, low_memory=False)
    dta_shape = {"rows": len(dta), "targets": dta["target_chembl_id"].nunique(), "ligands": dta["ligand_inchikey"].nunique()}
    check("DTA_01", "ConPLEx覆盖720药物×46靶点", dta_shape, {"rows": 33120, "targets": 46, "ligands": 720}, dta_shape == {"rows": 33120, "targets": 46, "ligands": 720} and dta["conplex_score"].notna().all(), dta_path)
    eligible = dta["drugclip_formal_eligible"].fillna(False).astype(bool)
    drugclip_observed = {"eligible_rows": int(eligible.sum()), "eligible_targets": dta.loc[eligible, "target_chembl_id"].nunique()}
    check("DTA_02", "DrugCLIP仅覆盖44个精确序列全长结构靶点", drugclip_observed, {"eligible_rows": 31680, "eligible_targets": 44}, drugclip_observed == {"eligible_rows": 31680, "eligible_targets": 44} and dta.loc[eligible, "drugclip_predicted_pocket_cosine_mean"].notna().all(), dta_path)
    shortlist_observed = {
        "two_model_target_top10": int(dta["dta_target_top10pct_concordant"].fillna(False).astype(bool).sum()),
        "bidirectional_top10": int(dta["dta_bidirectional_top10pct_concordant"].fillna(False).astype(bool).sum()),
        "sequence_fragment_exploratory": int(dta["sequence_only_fragment_exploratory_top5pct"].fillna(False).astype(bool).sum()),
    }
    check("DTA_03", "DTA短名单三类计数与冻结结果一致", shortlist_observed, {"two_model_target_top10": 276, "bidirectional_top10": 58, "sequence_fragment_exploratory": 72}, shortlist_observed == {"two_model_target_top10": 276, "bidirectional_top10": 58, "sequence_fragment_exploratory": 72}, dta_path)

    control_path = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_CONTROL_PANEL_23_X_24_V1.csv.gz"
    control_pose_path = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_CONTROL_POSES_V1.csv.gz"
    metrics_path = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_PREDICTED_POCKET_METRICS_23_V1.csv"
    controls = pd.read_csv(control_path, low_memory=False)
    control_poses = pd.read_csv(control_pose_path, low_memory=False)
    metrics = pd.read_csv(metrics_path, low_memory=False)
    per_target_total = controls.groupby("target_chembl_id").size()
    per_target_labels = controls.groupby(["target_chembl_id", "binary_label"]).size().unstack(fill_value=0)
    labels_ok = set(per_target_labels.columns) == {0, 1} and (per_target_labels == 12).all().all()
    control_observed = {"targets": controls["target_chembl_id"].nunique(), "controls": len(controls), "poses": len(control_poses)}
    check("GNINA_01", "23个靶点各有12阳性+12阴性且每配体5个GNINA姿势", control_observed, {"targets": 23, "controls": 552, "poses": 2760}, control_observed == {"targets": 23, "controls": 552, "poses": 2760} and (per_target_total == 24).all() and labels_ok and (control_poses.groupby("control_pair_id").size() == 5).all(), control_path)
    qualification_counts = metrics["predicted_pocket_qualification"].value_counts().to_dict()
    expected_qualifications = {
        "FAILED_CONTROL_SEPARATION": 11,
        "MARGINAL_NOT_QUALIFIED": 9,
        "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION": 3,
    }
    check("GNINA_02", "GNINA口袋资格结论为3强、9边缘、11失败", qualification_counts, expected_qualifications, qualification_counts == expected_qualifications and len(metrics) == 23, metrics_path)

    candidate_path = ROOT / "outputs/recovered_gnina_candidate_docking_v1/RECOVERED_GNINA_CANDIDATE_EVIDENCE_V1.csv"
    candidate_pose_path = ROOT / "outputs/recovered_gnina_candidate_docking_v1/RECOVERED_GNINA_CANDIDATE_POSES_V1.csv.gz"
    candidates = pd.read_csv(candidate_path, low_memory=False)
    candidate_poses = pd.read_csv(candidate_pose_path, low_memory=False)
    candidate_status = candidates["candidate_triage_status"].value_counts().to_dict()
    expected_candidate_status = {
        "MODEL_DISAGREEMENT_NO_ORTHOGONAL_SUPPORT": 15,
        "ORTHOGONAL_RESCUE_SIGNAL_MODEL_DISAGREEMENT_REVIEW": 5,
        "NO_GNINA_CONTROL_CALIBRATED_SUPPORT": 5,
        "COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT": 2,
    }
    candidate_observed = {"pairs": len(candidates), "poses": len(candidate_poses), "statuses": candidate_status}
    candidate_expected = {"pairs": 27, "poses": 135, "statuses": expected_candidate_status}
    check("CANDIDATE_01", "27个候选完成5姿势对接并按证据门控分层", candidate_observed, candidate_expected, candidate_observed == candidate_expected and (candidate_poses.groupby("candidate_pair_id").size() == 5).all(), candidate_path)

    boltz_summary_path = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1/RECOVERED_BOLTZ2_LOXL2_PAIR_SUMMARY_V1.csv"
    boltz_sample_path = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1/RECOVERED_BOLTZ2_LOXL2_SAMPLE_METRICS_V1.csv"
    boltz_pairwise_path = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1/RECOVERED_BOLTZ2_LOXL2_PAIRWISE_POSE_CONSISTENCY_V1.csv"
    boltz = pd.read_csv(boltz_summary_path)
    boltz_samples = pd.read_csv(boltz_sample_path)
    boltz_pairwise = pd.read_csv(boltz_pairwise_path)
    boltz_observed = {"pairs": len(boltz), "models": len(boltz_samples), "pairwise_comparisons": len(boltz_pairwise), "md_authorized": int(boltz["md_authorized"].sum())}
    check("BOLTZ_01", "LOXL2两个候选各完成5模型和10组两两一致性比较并执行MD门控", boltz_observed, {"pairs": 2, "models": 10, "pairwise_comparisons": 20, "md_authorized": 0}, boltz_observed == {"pairs": 2, "models": 10, "pairwise_comparisons": 20, "md_authorized": 0}, boltz_summary_path)

    class_path = INTEGRATED / "RECOVERED_46_MECHANISTIC_CLASSIFICATION_V1.csv"
    outcomes_path = INTEGRATED / "RECOVERED_46_INTEGRATED_TARGET_OUTCOMES_V1.csv"
    wet_path = INTEGRATED / "RECOVERED_46_WETLAB_EXECUTION_PANEL_V1.csv"
    branch_path = INTEGRATED / "RECOVERED_46_CLASS_BRANCH_SUMMARY_V1.csv"
    classification = pd.read_csv(class_path, low_memory=False)
    outcomes = pd.read_csv(outcomes_path, low_memory=False)
    wet = pd.read_csv(wet_path, low_memory=False)
    branches = pd.read_csv(branch_path, low_memory=False)
    required_class_columns = ["mechanistic_branch", "mechanistic_subclass", "required_structural_biochemical_context_zh", "class_specific_compute_protocol_zh", "target_specific_primary_assay_zh", "orthogonal_assay_and_counterscreen_zh", "required_controls_zh"]
    class_complete = classification[required_class_columns].notna().all().all() and not (classification[required_class_columns].astype(str).apply(lambda s: s.str.strip().eq("")).any().any())
    class_observed = {"targets": classification["target_chembl_id"].nunique(), "branches": classification["mechanistic_branch"].nunique(), "branch_total": int(branches["target_count"].sum())}
    check("CLASS_01", "46个靶点完成12个机制分支、结构条件、计算和实验分类", class_observed, {"targets": 46, "branches": 12, "branch_total": 46}, class_observed == {"targets": 46, "branches": 12, "branch_total": 46} and class_complete, class_path)
    outcome_required = ["current_program_status", "discovery_authorization", "next_action_zh", "conplex_execution", "drugclip_execution", "gnina_control_execution", "scientific_boundary"]
    outcome_complete = len(outcomes) == 46 and outcomes["target_chembl_id"].nunique() == 46 and outcomes[outcome_required].notna().all().all()
    check("OUTCOME_01", "46个靶点均有计算执行状态、发现授权和下一步", int(outcome_complete), 1, outcome_complete, outcomes_path)
    wet_required = ["target_specific_primary_assay_zh", "orthogonal_assay_and_counterscreen_zh", "required_controls_zh", "wetlab_stage", "test_articles", "dose_and_replication_design_zh", "global_activity_gate_zh", "global_selectivity_gate_zh", "promotion_rule"]
    wet_complete = len(wet) == 46 and wet["target_chembl_id"].nunique() == 46 and wet[wet_required].notna().all().all()
    check("WETLAB_01", "46个靶点均有主测定、正交测定、对照、剂量设计和晋级门槛", int(wet_complete), 1, wet_complete, wet_path)

    active_path = INTEGRATED / "ACTIVE_TARGET_BRANCHES_384_V1.csv"
    full_audit_path = INTEGRATED / "FULL_TARGET_SCOPE_AUDIT_888_V1.csv"
    active = pd.read_csv(active_path, low_memory=False)
    full_audit = pd.read_csv(full_audit_path, low_memory=False)
    active_counts = active["active_target_branch"].value_counts().to_dict()
    expected_active = {"STRICT_EXPERIMENTAL_POCKET_MAINLINE_338": 338, "RECOVERED_NO_EXPERIMENTAL_POCKET_46": 46}
    check("SCOPE_02", "活跃靶点严格为338实验口袋主线与46预测口袋找回的互斥并集", {"rows": len(active), "unique": active["target_chembl_id"].nunique(), "branches": active_counts}, {"rows": 384, "unique": 384, "branches": expected_active}, len(active) == 384 and active["target_chembl_id"].nunique() == 384 and active_counts == expected_active, active_path)
    macro_counts = full_audit["final_scope_branch"].value_counts().to_dict()
    expected_macro = {"HARD_GATE_EXCLUDED_480": 480, "ACTIVE_STRICT_MAINLINE_338": 338, "ACTIVE_RECOVERED_NO_EXPERIMENTAL_POCKET_46": 46, "LATER_POCKET_QUALITY_EXCLUDED_24": 24}
    check("SCOPE_03", "888全集完成互斥分区且未混入151原始缓存", {"rows": len(full_audit), "unique": full_audit["target_chembl_id"].nunique(), "branches": macro_counts}, {"rows": 888, "unique": 888, "branches": expected_macro}, len(full_audit) == 888 and full_audit["target_chembl_id"].nunique() == 888 and macro_counts == expected_macro, full_audit_path)

    workbook = INTEGRATED / "RECOVERED_46_INTEGRATED_PROGRAM_V1.xlsx"
    check("PACKAGE_01", "综合工作簿已生成", workbook.stat().st_size if workbook.exists() else 0, ">0", workbook.exists() and workbook.stat().st_size > 0, workbook)
    final_report = INTEGRATED / "RECOVERED_46_PROGRAM_FINAL_REPORT_ZH_V1.md"
    check("PACKAGE_02", "中文综合报告已生成", final_report.stat().st_size if final_report.exists() else 0, ">0", final_report.exists() and final_report.stat().st_size > 0, final_report)

    frame = pd.DataFrame(checks)
    passed = bool(frame["passed"].all())
    csv_path = INTEGRATED / "RECOVERED_46_COMPLETION_AUDIT_CHECKS_V1.csv"
    json_path = INTEGRATED / "RECOVERED_46_COMPLETION_AUDIT_V1.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        "audit_status": "PASS" if passed else "FAIL",
        "checks_total": len(frame),
        "checks_passed": int(frame["passed"].sum()),
        "checks_failed": int((~frame["passed"]).sum()),
        "scope_boundary": "Exact 46 first-exclusion no-pocket targets; active union 338+46=384",
        "physical_wetlab_boundary": "Protocols are execution-ready plans; physical measurements require an external laboratory",
        "failed_checks": frame.loc[~frame["passed"], "check_id"].tolist(),
        "checks_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
