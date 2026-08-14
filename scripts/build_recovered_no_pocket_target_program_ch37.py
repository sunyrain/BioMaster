#!/usr/bin/env python3
"""Build the formal program for the 46 targets excluded only for no pocket."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXCLUSION = ROOT / "outputs/final_target_package_ch37/FINAL_TARGET_EXCLUSION_AUDIT_888.csv"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
PREDICTION = (
    ROOT
    / "outputs/no_experimental_pocket_prediction_ch37_v1/"
    "NO_EXPERIMENTAL_POCKET_CONSENSUS_TARGETS_151_V1.csv"
)
STRUCTURE_FIRST = (
    ROOT
    / "outputs/no_experimental_pocket_prediction_ch37_v1/structure_first_models_v1/"
    "STRUCTURE_FIRST_TARGET_POCKET_SUMMARY_2_V1.csv"
)
DEFAULT_OUTDIR = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1"


PROTOCOLS = {
    "KINASE_BIOCHEMICAL": {
        "class_zh": "激酶",
        "compute": "预测ATP/变构口袋ensemble；历史正负对照校准GNINA；Boltz-2多seed；通过门控后短程MD",
        "wetlab": "激酶生化活性/结合剂量反应 + NanoBRET或细胞磷酸化正交复核",
        "controls": "参考抑制剂、无活性类似物、载体、近缘激酶选择性面板",
    },
    "ENZYME_BIOCHEMICAL": {
        "class_zh": "非激酶酶",
        "compute": "辅因子/金属感知预测口袋ensemble；历史正负对照GNINA；Boltz-2多seed；标准或专项MD",
        "wetlab": "纯化酶活性剂量反应 + SPR/BLI/MST或DSF直接结合复核",
        "controls": "已知底物/抑制剂、无酶空白、载体、结构相近阴性化合物",
    },
    "ION_CHANNEL_FUNCTIONAL": {
        "class_zh": "离子通道",
        "compute": "开放/关闭/失活状态和孔道口袋ensemble；Boltz-2；膜内离子通道MD",
        "wetlab": "自动膜片钳多电压/多状态剂量反应 + 离子通量或位点突变复核",
        "controls": "参考阻断剂/激动剂、未表达细胞、载体、电生理质量控制",
    },
    "TRANSPORTER_MEMBRANE_FUNCTIONAL": {
        "class_zh": "转运体",
        "compute": "内向/外向状态和底物通道口袋ensemble；历史对照校准；Boltz-2；膜环境MD",
        "wetlab": "细胞/膜泡摄取或外排剂量反应 + 靶点占有率/竞争结合复核",
        "controls": "已知底物/抑制剂、敲除细胞、载体、非相关转运体",
    },
    "NUCLEAR_EPIGENETIC_DOMAIN": {
        "class_zh": "核内/表观遗传蛋白",
        "compute": "结构域和辅因子状态预测口袋；历史对照GNINA；Boltz-2多seed；局部MD",
        "wetlab": "TR-FRET/AlphaScreen/酶活或共调节因子招募 + NanoBRET/SPR/BLI复核",
        "controls": "参考配体、突变结构域、载体、非相关核内蛋白",
    },
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na", "n/a"} else text


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value).lower() in {"true", "1", "yes", "y"}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calibration_evidence(value: str) -> tuple[str, str]:
    if value.startswith("T1_"):
        return "H1_RICH_POSITIVE_NEGATIVE", "历史正负对照丰富"
    if value.startswith("T2_"):
        return "H2_ADEQUATE_POSITIVE_NEGATIVE", "历史正负对照充分"
    if value.startswith("T3_"):
        return "H3_MINIMUM_POSITIVE_NEGATIVE", "历史正负对照达到最低门槛"
    if value.startswith("T4_"):
        return "H4_POSITIVE_ONLY", "仅有历史阳性；必须补阴性对照"
    return "H5_SPARSE_OR_NONE", "历史对照稀疏或缺失"


def pocket_evidence(row: pd.Series) -> tuple[str, str]:
    consensus = clean(row.get("pocket_consensus_class"))
    if consensus == "C1_DUAL_METHOD_SAME_SITE":
        return "P1_P2RANK_FPOCKET_SAME_SITE", "P2Rank与fpocket支持同一预测位点"
    if consensus == "C4_FPOCKET_RESCUE":
        return "P2_FPOCKET_GEOMETRIC_RESCUE", "P2Rank较弱，由fpocket提供几何口袋补救"
    fragment = clean(row.get("fragment_pocket_evidence_tier"))
    if fragment.startswith("PF3_"):
        return "P3_LOW_CONFIDENCE_FRAGMENT_HYPOTHESIS", "低置信片段模型上双方法检出口袋，仅作假设"
    if fragment:
        return "P3_FRAGMENT_MODEL_HYPOTHESIS", "片段模型上的计算口袋假设"
    return "P4_POCKET_PREDICTION_INCOMPLETE", "口袋预测尚未完成"


def paired_route(history: str, pocket: str) -> tuple[str, str, str]:
    calibrated = history.startswith(("H1_", "H2_", "H3_"))
    if pocket.startswith("P1_") and calibrated:
        return (
            "R1_DUAL_POCKET_WITH_MEASURED_CONTROLS",
            "先用历史正负化合物在预测位点建立靶点内分布；通过后运行类别专属GNINA/Boltz",
            "LOCAL_AND_PREDECLARED_REMOTE_ONLY_AFTER_PREDICTED_POCKET_GATE",
        )
    if pocket.startswith("P1_"):
        return (
            "R2_DUAL_POCKET_CONTROL_ACQUISITION",
            "双方法口袋可用，但先补足实测阴性/阳性对照；当前仅做探索性局部计算",
            "DIAGNOSTIC_LOCAL_ONLY",
        )
    if pocket.startswith("P2_") and calibrated:
        return (
            "R3_GEOMETRIC_RESCUE_WITH_MEASURED_CONTROLS",
            "用历史正负对照筛选构象与口袋；至少两构象复现后才进入pair计算",
            "POCKET_ENSEMBLE_DIAGNOSTIC_UNTIL_REPRODUCED",
        )
    if pocket.startswith("P2_"):
        return (
            "R4_GEOMETRIC_RESCUE_CONTROL_ACQUISITION",
            "同时补对照和构象证据；不得用单一fpocket位点做发现排序",
            "NO_DISCOVERY_RANKING_YET",
        )
    return (
        "R5_LOW_CONFIDENCE_STRUCTURE_FIRST",
        "先用实验/同源结构或带MSA模型提高局部结构置信度，再复核口袋",
        "NO_PAIR_PROMOTION_FROM_LOW_CONFIDENCE_FRAGMENT_MODEL",
    )


def build(outdir: Path) -> None:
    exclusion = pd.read_csv(EXCLUSION, low_memory=False)
    universe = pd.read_csv(UNIVERSE, low_memory=False)
    prediction = pd.read_csv(PREDICTION, low_memory=False)
    fragments = pd.read_csv(STRUCTURE_FIRST, low_memory=False)
    recovered = exclusion[
        exclusion["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET")
    ].copy()
    if len(recovered) != 46 or recovered["target_chembl_id"].nunique() != 46:
        raise RuntimeError("Recovered scope is not the authoritative 46-target set")
    if not (
        recovered["passes_non_gpcr"].map(as_bool).all()
        and recovered["passes_chembl_small_molecule_moa"].map(as_bool).all()
        and recovered["passes_supported_target_class"].map(as_bool).all()
        and ~recovered["passes_any_experimental_pocket"].map(as_bool).any()
    ):
        raise RuntimeError("Recovered targets do not satisfy the intended funnel boundary")

    universe_columns = [
        "target_chembl_id",
        "target_class_l1",
        "target_class_l2",
        "target_class_l3",
        "target_class_leaf",
        "sequence_length",
        "sequence_sha256",
        "evidence_class",
        "small_molecule_moa_record_count",
        "small_molecule_count",
        "approved_small_molecule_moa",
        "calibration_status",
        "positive_compounds",
        "negative_compounds",
        "strict_document_count",
        "af_exact_sequence_model",
        "af_pdb_path",
        "af_mean_plddt",
        "p2rank_status",
        "p2rank_tier",
        "p2rank_top_score",
        "p2rank_top_probability",
        "pocket_mean_plddt",
    ]
    matrix = recovered.merge(
        universe[universe_columns], on="target_chembl_id", how="left", validate="one_to_one"
    )
    prediction_columns = [
        "target_chembl_id",
        "fpocket_status",
        "fpocket_pocket_count",
        "fpocket_tier",
        "fpocket_top_score",
        "fpocket_top_druggability_score",
        "fpocket_top_volume_A3",
        "p2rank_top3_matches_any_fpocket",
        "best_method_match_p2rank_rank",
        "best_method_match_fpocket_rank",
        "best_method_match_residue_jaccard",
        "best_method_match_center_distance_A",
        "pocket_consensus_class",
        "primary_prediction_policy",
        "predicted_pocket_identified",
    ]
    matrix = matrix.merge(
        prediction[prediction_columns], on="target_chembl_id", how="left", validate="one_to_one"
    )
    fragment_columns = [
        "target_chembl_id",
        "fragments_expected",
        "fragments_completed",
        "full_sequence_covered",
        "p2rank_positive_fragments",
        "fpocket_positive_fragments",
        "dual_method_positive_fragments",
        "mean_fragment_plddt",
        "structure_strategy",
        "pocket_evidence_tier",
    ]
    fragment_subset = fragments[fragment_columns].rename(
        columns={"pocket_evidence_tier": "fragment_pocket_evidence_tier"}
    )
    matrix = matrix.merge(fragment_subset, on="target_chembl_id", how="left", validate="one_to_one")
    history = matrix["calibration_status"].fillna("").map(calibration_evidence)
    matrix["historical_experimental_evidence"] = history.map(lambda value: value[0])
    matrix["historical_experimental_evidence_zh"] = history.map(lambda value: value[1])
    pockets = matrix.apply(pocket_evidence, axis=1)
    matrix["computed_pocket_evidence"] = pockets.map(lambda value: value[0])
    matrix["computed_pocket_evidence_zh"] = pockets.map(lambda value: value[1])
    routes = matrix.apply(
        lambda row: paired_route(
            row["historical_experimental_evidence"], row["computed_pocket_evidence"]
        ),
        axis=1,
    )
    matrix["evidence_compute_route"] = routes.map(lambda value: value[0])
    matrix["next_compute_action_zh"] = routes.map(lambda value: value[1])
    matrix["authorization_scope"] = routes.map(lambda value: value[2])
    for key, output_column in [
        ("class_zh", "target_compute_class_zh"),
        ("compute", "class_specific_compute_bundle_zh"),
        ("wetlab", "wetlab_primary_orthogonal_assay_zh"),
        ("controls", "wetlab_control_design_zh"),
    ]:
        matrix[output_column] = matrix["assay_lane"].map(
            lambda lane: PROTOCOLS[clean(lane)][key]
        )
    matrix["wetlab_priority"] = matrix.apply(
        lambda row: (
            "W1_IMMEDIATE_CONTROL_AND_BINDING_PANEL"
            if row["historical_experimental_evidence"].startswith(("H1_", "H2_"))
            and row["computed_pocket_evidence"].startswith("P1_")
            else (
                "W2_MINIMUM_CONTROL_AND_POCKET_VALIDATION"
                if row["historical_experimental_evidence"].startswith("H3_")
                or row["computed_pocket_evidence"].startswith("P2_")
                else "W3_CONTROL_OR_STRUCTURE_ACQUISITION"
            )
        ),
        axis=1,
    )
    matrix["experimental_pocket_status"] = "ABSENT_BY_FROZEN_ATLAS"
    matrix["recovery_scope_reason"] = "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"
    matrix = matrix.sort_values(
        ["wetlab_priority", "evidence_compute_route", "assay_lane", "gene_symbol"]
    )
    outdir.mkdir(parents=True, exist_ok=True)
    target_path = outdir / "RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
    matrix.to_csv(target_path, index=False)

    protocol_rows = [
        {"assay_lane": lane, **payload} for lane, payload in PROTOCOLS.items()
    ]
    protocols = pd.DataFrame(protocol_rows)
    protocols.to_csv(outdir / "TARGET_CLASS_COMPUTE_AND_EXPERIMENT_PROTOCOLS_5_V1.csv", index=False)
    cross = (
        matrix.groupby(
            [
                "assay_lane",
                "historical_experimental_evidence",
                "computed_pocket_evidence",
                "evidence_compute_route",
            ]
        )
        .size()
        .rename("targets")
        .reset_index()
    )
    cross.to_csv(outdir / "EXPERIMENTAL_BY_COMPUTED_EVIDENCE_MATRIX_V1.csv", index=False)
    wetlab_columns = [
        "target_chembl_id",
        "gene_symbol",
        "target_compute_class_zh",
        "assay_lane",
        "historical_experimental_evidence",
        "computed_pocket_evidence",
        "evidence_compute_route",
        "wetlab_priority",
        "wetlab_primary_orthogonal_assay_zh",
        "wetlab_control_design_zh",
        "authorization_scope",
    ]
    matrix[wetlab_columns].to_csv(outdir / "WETLAB_ASSAY_ROUTE_46_V1.csv", index=False)
    for route, group in matrix.groupby("evidence_compute_route"):
        safe = route.split("_", 1)[0]
        group.to_csv(outdir / f"TASK_{safe}_{route}_V1.csv", index=False)

    class_summary = (
        matrix.groupby(["assay_lane", "target_compute_class_zh"])
        .agg(
            targets=("target_chembl_id", "count"),
            dual_same_site=("computed_pocket_evidence", lambda values: int(values.str.startswith("P1_").sum())),
            geometric_rescue=("computed_pocket_evidence", lambda values: int(values.str.startswith("P2_").sum())),
            low_confidence_fragment=("computed_pocket_evidence", lambda values: int(values.str.startswith("P3_").sum())),
            rich_or_adequate_controls=("historical_experimental_evidence", lambda values: int(values.str.startswith(("H1_", "H2_")).sum())),
            sparse_controls=("historical_experimental_evidence", lambda values: int(values.str.startswith(("H4_", "H5_")).sum())),
        )
        .reset_index()
    )
    class_summary.to_csv(outdir / "RECOVERED_TARGET_CLASS_SUMMARY_V1.csv", index=False)
    with pd.ExcelWriter(outdir / "RECOVERED_NO_POCKET_TARGET_PROGRAM_46_V1.xlsx") as writer:
        matrix.to_excel(writer, sheet_name="恢复靶点46", index=False)
        class_summary.to_excel(writer, sheet_name="分类汇总", index=False)
        cross.to_excel(writer, sheet_name="实验计算证据矩阵", index=False)
        protocols.to_excel(writer, sheet_name="类别计算实验方案", index=False)
        matrix[wetlab_columns].to_excel(writer, sheet_name="湿实验路线", index=False)

    invariants = {
        "exactly_46_targets": len(matrix) == 46 and matrix["target_chembl_id"].nunique() == 46,
        "all_pass_non_gpcr_gate": matrix["passes_non_gpcr"].map(as_bool).all(),
        "all_pass_small_molecule_moa_gate": matrix["passes_chembl_small_molecule_moa"].map(as_bool).all(),
        "all_pass_supported_class_gate": matrix["passes_supported_target_class"].map(as_bool).all(),
        "all_fail_only_experimental_pocket_gate": (
            matrix["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET").all()
            and ~matrix["passes_any_experimental_pocket"].map(as_bool).any()
        ),
        "all_have_computed_pocket_route": matrix["computed_pocket_evidence"].notna().all(),
        "all_have_class_specific_compute": matrix["class_specific_compute_bundle_zh"].notna().all(),
        "all_have_wetlab_route": matrix["wetlab_primary_orthogonal_assay_zh"].notna().all(),
        "44_exact_structures_plus_2_fragment_models": (
            matrix["af_exact_sequence_model"].map(as_bool).sum() == 44
            and matrix["fragment_pocket_evidence_tier"].notna().sum() == 2
        ),
        "class_summary_sums_to_46": class_summary["targets"].sum() == 46,
    }
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(invariants.values()) else "FAIL",
        "scope_definition": "targets whose first exclusion reason is only EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET",
        "targets": int(len(matrix)),
        "assay_lane_counts": matrix["assay_lane"].value_counts().to_dict(),
        "historical_experimental_evidence_counts": matrix["historical_experimental_evidence"].value_counts().to_dict(),
        "computed_pocket_evidence_counts": matrix["computed_pocket_evidence"].value_counts().to_dict(),
        "evidence_compute_route_counts": matrix["evidence_compute_route"].value_counts().to_dict(),
        "wetlab_priority_counts": matrix["wetlab_priority"].value_counts().to_dict(),
        "invariants": invariants,
        "boundaries": [
            "These 46 targets passed all earlier hard gates and are recovered only for computational pocket follow-up.",
            "Predicted pockets do not become experimental pocket evidence.",
            "Historical target activity controls calibrate target-level protocols; they do not validate a new drug-target pair.",
            "Only prospective binding and functional assays can promote recovered targets into the experimental target set.",
        ],
        "outputs": {
            "target_matrix": str(target_path.resolve()),
            "workbook": str((outdir / "RECOVERED_NO_POCKET_TARGET_PROGRAM_46_V1.xlsx").resolve()),
        },
    }
    write_json(outdir / "RECOVERED_NO_POCKET_TARGET_PROGRAM_SUMMARY_V1.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("Recovered target program audit failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    build(args.outdir.resolve())


if __name__ == "__main__":
    main()
