#!/usr/bin/env python3
"""Build an evidence-stratified 1000-pair portfolio without a universal score."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QSAR = ROOT / "outputs/current_production_package_v2/target_qsar_calibration_v5/PROJECT_DRUG_TARGET_QSAR_PREDICTIONS_V5.csv.gz"
QSAR_METRICS = ROOT / "outputs/current_production_package_v2/target_qsar_calibration_v5/TARGET_QSAR_SCAFFOLD_HOLDOUT_METRICS_V5.csv"
TOP3000 = ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/refined_top3000_v4_complete.csv"
UNIVERSE_V3 = ROOT / "outputs/current_production_package_v2/full_untruncated_universe_v3/full_project_universe_344190_scored_v3.csv"
CONPLEX_METRICS = ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5_official/evaluation/CONPLEX_TARGET_CALIBRATION_METRICS_V5.csv"
BOLTZ_DECISION = ROOT / "outputs/current_production_package_v2/boltz_target_calibration_smoke_v5/evaluation/BOLTZ_SMOKE_CALIBRATION_DECISION_V5.json"
CURRENT1000 = ROOT / "outputs/current_production_package_v2/final_delivery_v4/FINAL1000_RESERVE_FULL_V4.csv"
FDA_XLSX = ROOT / "FDA_approved_small_molecules_2005_2026_with_structures.xlsx"
OUT = ROOT / "outputs/current_production_package_v2/calibrated_portfolio_v5"


FAMILY_CAPS = {
    "enzyme": 550,
    "kinase": 180,
    "nuclear_epigenetic": 160,
    "transporter": 100,
    "ion_channel": 60,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bools(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def fda_annotations() -> pd.DataFrame:
    source = pd.read_excel(FDA_XLSX, sheet_name="FDA Small Molecules 2005-2026")
    mapping = {
        "ChEMBL ID": "drug_chembl_id",
        "Generic Name (INN)": "fda_generic_name",
        "Brand Name": "fda_brand_name",
        "Indication": "fda_indication",
        "Therapeutic Area": "fda_therapeutic_area",
        "Mechanism of Action": "fda_moa",
        "Target Name": "fda_target_names",
        "Action Type": "fda_action_type",
        "Approval Year": "fda_approval_year",
        "Route": "fda_route",
    }
    source = source[list(mapping)].rename(columns=mapping)

    def combine(values: pd.Series) -> str:
        unique = []
        for value in values:
            text = clean(value)
            if text and text not in unique:
                unique.append(text)
        return "; ".join(unique)

    return source.groupby("drug_chembl_id", as_index=False, dropna=False).agg(
        {column: combine for column in mapping.values() if column != "drug_chembl_id"}
    )


class DiverseSelector:
    def __init__(self) -> None:
        self.selected_keys: set[tuple[str, str]] = set()
        self.counts = {name: Counter() for name in ["drug", "target", "scaffold", "family"]}

    def take(
        self,
        frame: pd.DataFrame,
        count: int,
        drug_cap: int,
        target_cap: int,
        scaffold_cap: int,
        family_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        indices = []
        for index, row in frame.iterrows():
            if len(indices) >= count:
                break
            drug = clean(row.get("drug_chembl_id"))
            target = clean(row.get("sequence_key"))
            key = (drug, target)
            if key in self.selected_keys:
                continue
            scaffold = clean(row.get("murcko_scaffold")) or clean(row.get("model_ligand_smiles")) or drug
            family = clean(row.get("target_assay_family_v2")) or clean(row.get("target_assay_family"))
            family_cap = int(FAMILY_CAPS.get(family, 50) * family_multiplier)
            if (
                self.counts["drug"][drug] >= drug_cap
                or self.counts["target"][target] >= target_cap
                or self.counts["scaffold"][scaffold] >= scaffold_cap
                or self.counts["family"][family] >= family_cap
            ):
                continue
            indices.append(index)
            self.selected_keys.add(key)
            for name, value in [("drug", drug), ("target", target), ("scaffold", scaffold), ("family", family)]:
                self.counts[name][value] += 1
        return frame.loc[indices].copy()


def enrich_qsar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qsar = pd.read_csv(QSAR, low_memory=False)
    metrics = pd.read_csv(QSAR_METRICS, low_memory=False)
    metric_columns = [
        "sequence_key",
        "qsar_oof_pr_auc",
        "similarity_oof_pr_auc",
        "qsar_oof_roc_auc",
        "similarity_oof_roc_auc",
        "qsar_minus_similarity_ap_ci95_low",
        "temporal_qsar_pr_auc",
        "temporal_similarity_pr_auc",
        "temporal_qsar_roc_auc",
        "temporal_similarity_roc_auc",
    ]
    qsar = qsar.merge(metrics[metric_columns], on="sequence_key", how="left", validate="many_to_one")
    universe = pd.read_csv(
        UNIVERSE_V3,
        usecols=[
            "drug_chembl_id",
            "sequence_key",
            "is_known_fda_target_pair",
            "family_or_rediscovery_risk_v2",
            "risk_notes_v2",
            "conplex_score",
            "rank_within_drug_full891",
        ],
        low_memory=False,
    )
    qsar = qsar.merge(universe, on=["drug_chembl_id", "sequence_key"], how="left", validate="one_to_one")
    qsar = qsar.merge(fda_annotations(), on="drug_chembl_id", how="left", validate="many_to_one")
    qsar["is_known_fda_target_pair"] = bools(qsar["is_known_fda_target_pair"])
    qsar["family_or_rediscovery_risk_v2"] = bools(qsar["family_or_rediscovery_risk_v2"])
    qsar["target_qsar_percentile_v5"] = qsar.groupby("sequence_key")[
        "target_qsar_probability_v5"
    ].rank(pct=True, method="average")
    qsar["in_known_ligand_applicability_domain_v5"] = (
        pd.to_numeric(qsar["max_known_active_similarity_v5"], errors="coerce").ge(0.40)
        | bools(qsar["same_known_active_scaffold_v5"])
    )
    nonknown = ~qsar["is_known_fda_target_pair"] & ~bools(qsar["exact_known_active_smiles_v5"])
    nonfamily = ~qsar["family_or_rediscovery_risk_v2"]
    percentile = qsar["target_qsar_percentile_v5"].ge(0.80)
    domain = qsar["in_known_ligand_applicability_domain_v5"]
    primary = qsar[
        qsar["target_ligand_model_status_v5"].eq("T1_qsar_beats_similarity")
        & nonknown
        & nonfamily
        & percentile
        & domain
    ].copy()
    primary["portfolio_lane_v5"] = "P1_calibrated_target_qsar_in_domain"
    primary["binding_evidence_interpretation_v5"] = (
        "Target-specific Morgan-QSAR beat ligand similarity under scaffold holdout; pair is in ligand applicability domain."
    )
    primary = primary.sort_values(
        ["target_qsar_percentile_v5", "max_known_active_similarity_v5", "qsar_oof_pr_auc", "drug_chembl_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )

    secondary = qsar[
        qsar["target_ligand_model_status_v5"].eq("T2_similarity_supported")
        & nonknown
        & nonfamily
        & percentile
        & domain
    ].copy()
    secondary["portfolio_lane_v5"] = "P2_validated_ligand_similarity_in_domain"
    secondary["binding_evidence_interpretation_v5"] = (
        "Ligand similarity was predictive under scaffold holdout; target-specific QSAR did not add significant value."
    )
    secondary = secondary.sort_values(
        ["max_known_active_similarity_v5", "target_qsar_percentile_v5", "similarity_oof_pr_auc", "drug_chembl_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    family_extension = qsar[nonknown & qsar["family_or_rediscovery_risk_v2"] & percentile & domain].copy()
    family_extension["portfolio_lane_v5"] = "R_family_extension_review"
    controls = qsar[qsar["is_known_fda_target_pair"] | bools(qsar["exact_known_active_smiles_v5"])].copy()
    controls["portfolio_lane_v5"] = "C_positive_control_or_rediscovery"
    return primary, secondary, family_extension, controls


def exploration_queue() -> pd.DataFrame:
    top = pd.read_csv(TOP3000, low_memory=False)
    qsar_columns = [
        "drug_chembl_id",
        "sequence_key",
        "max_known_active_similarity_v5",
        "same_known_active_scaffold_v5",
        "exact_known_active_smiles_v5",
        "target_ligand_model_status_v5",
        "target_qsar_probability_v5",
    ]
    qsar = pd.read_csv(QSAR, usecols=qsar_columns, low_memory=False)
    conplex = pd.read_csv(
        CONPLEX_METRICS,
        usecols=["sequence_key", "conplex_target_use_status_v5", "conplex_pr_auc", "similarity_pr_auc"],
        low_memory=False,
    )
    out = top.merge(qsar, on=["drug_chembl_id", "sequence_key"], how="left", validate="one_to_one")
    out = out.merge(conplex, on="sequence_key", how="left", validate="many_to_one")
    remote = pd.to_numeric(out["max_known_active_similarity_v5"], errors="coerce").lt(0.40) | out[
        "max_known_active_similarity_v5"
    ].isna()
    pose_ready = out["pose_stability_tier"].fillna("").astype(str).str.startswith(("A_", "B_"))
    completed = bools(out["boltz_completed_refined"])
    sequence_ok = ~bools(out["structure_sequence_mismatch_v4"])
    severe = bools(out["severe_compound_liability"])
    known = bools(out["is_known_fda_target_pair"])
    family_risk = bools(out["family_or_rediscovery_risk_v2"])
    out = out[remote & pose_ready & completed & sequence_ok & ~severe & ~known & ~family_risk].copy()
    out["portfolio_lane_v5"] = "P3_remote_uncalibrated_physics_exploration"
    out["binding_evidence_interpretation_v5"] = (
        "Remote from known target ligands; ConPLEx and Boltz are uncalibrated retrieval/pose evidence only."
    )
    out["conplex_role_v5"] = "uncalibrated_retrieval_only"
    out["boltz_role_v5"] = "conditional_pose_generation_not_binding_discrimination"
    out["exploration_pose_order_v5"] = out["pose_stability_tier"].map(
        {"A_stable_conditional_pose": 0, "B_moderate_conditional_pose": 1}
    ).fillna(2)
    return out.sort_values(
        [
            "exploration_pose_order_v5",
            "experimental_feasibility_component_v2",
            "drug_feasibility_component_v2",
            "rank_within_drug_full891",
            "pair_id",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )


def compact_chinese(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "portfolio_rank_v5": "组合顺序",
        "portfolio_lane_zh_v5": "证据分层",
        "display_drug_name_v5": "药物名称",
        "drug_chembl_id": "药物ChEMBL",
        "primary_gene": "候选靶点",
        "sequence_key": "项目序列",
        "target_assay_family_v5": "靶点实验类型",
        "target_ligand_model_status_v5": "靶点配体模型状态",
        "target_qsar_percentile_v5": "靶点内QSAR百分位",
        "target_qsar_probability_v5": "QSAR原始输出_不可跨靶点比较",
        "max_known_active_similarity_v5": "与已知活性配体最大相似度",
        "same_known_active_scaffold_v5": "同已知活性骨架",
        "conplex_score": "ConPLEx原始分_仅探索召回",
        "rank_within_drug_full891": "ConPLEx药物内序位",
        "boltz_affinity_probability_refined": "Boltz原始affinity_未校准",
        "pose_stability_tier": "条件姿势稳定性",
        "binding_evidence_interpretation_v5": "证据解释",
        "original_indication_v5": "原适应症_不是推荐新病种",
        "pair_id": "Pair_ID",
    }
    columns = [column for column in mapping if column in frame]
    return frame[columns].rename(columns={column: mapping[column] for column in columns})


def main() -> None:
    inputs = [QSAR, QSAR_METRICS, TOP3000, UNIVERSE_V3, CONPLEX_METRICS, BOLTZ_DECISION, CURRENT1000, FDA_XLSX]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    boltz_decision = json.loads(BOLTZ_DECISION.read_text(encoding="utf-8"))
    if boltz_decision["expansion_gate"] != "stop_do_not_expand":
        raise RuntimeError("Portfolio contract expects failed Boltz expansion gate")

    primary, secondary, family_extension, controls = enrich_qsar()
    exploration = exploration_queue()
    selector = DiverseSelector()
    parts = []
    # Evidence-qualified P1 rows are retained preferentially; the looser
    # generic-scaffold cap avoids treating every phenyl-containing drug as one
    # redundant chemical series.
    primary_selected = selector.take(primary, min(350, len(primary)), 10, 20, 100)
    parts.append(primary_selected)
    secondary_selected = selector.take(secondary, 300, 8, 15, 60)
    parts.append(secondary_selected)
    remaining = 1000 - sum(len(part) for part in parts)
    exploration_selected = selector.take(exploration, remaining, 6, 12, 30)
    parts.append(exploration_selected)
    portfolio = pd.concat(parts, ignore_index=True, sort=False)
    if len(portfolio) < 1000:
        # Relax only diversity caps, never evidence or exclusion criteria.
        remainder_sources = pd.concat([primary, secondary, exploration], ignore_index=True, sort=False)
        remainder_sources = remainder_sources[~remainder_sources.set_index(["drug_chembl_id", "sequence_key"]).index.isin(selector.selected_keys)]
        fill = selector.take(remainder_sources, 1000 - len(portfolio), 10, 20, 35, family_multiplier=1.25)
        portfolio = pd.concat([portfolio, fill], ignore_index=True, sort=False)
    if len(portfolio) != 1000:
        raise RuntimeError(f"Could not build 1000-pair portfolio; got {len(portfolio)}")
    if portfolio.duplicated(["drug_chembl_id", "sequence_key"]).any():
        raise RuntimeError("Duplicate pair in portfolio")
    if bools(portfolio["is_known_fda_target_pair"]).any():
        raise RuntimeError("Known FDA target pair leaked into portfolio")
    portfolio.insert(0, "portfolio_rank_v5", range(1, 1001))
    portfolio["selection_contract_v5"] = "evidence_stratified_no_universal_score"
    portfolio["portfolio_lane_zh_v5"] = portfolio["portfolio_lane_v5"].map(
        {
            "P1_calibrated_target_qsar_in_domain": "P1 靶点专属QSAR经骨架外推验证且处于适用域",
            "P2_validated_ligand_similarity_in_domain": "P2 配体相似性经验证且处于适用域",
            "P3_remote_uncalibrated_physics_exploration": "P3 远程物理探索，亲和未校准",
        }
    )
    portfolio["binding_evidence_interpretation_v5"] = portfolio["portfolio_lane_v5"].map(
        {
            "P1_calibrated_target_qsar_in_domain": "该靶点QSAR在Murcko骨架留出评估中显著优于配体相似性；该pair位于已知配体适用域。",
            "P2_validated_ligand_similarity_in_domain": "配体相似性在Murcko骨架留出评估中有效；QSAR没有提供显著增益。",
            "P3_remote_uncalibrated_physics_exploration": "远离该靶点已知配体；ConPLEx仅用于召回，Boltz仅说明条件结构可生成，不代表结合概率。",
        }
    )
    name_columns = [column for column in ["fda_generic_name", "drug_name", "drug_names", "drug_chembl_id"] if column in portfolio]
    display_name = pd.Series("", index=portfolio.index, dtype=object)
    for column in name_columns:
        values = portfolio[column].fillna("").astype(str).str.strip()
        display_name = display_name.where(display_name.ne(""), values)
    portfolio["display_drug_name_v5"] = display_name
    portfolio["target_assay_family_v5"] = portfolio.get(
        "target_assay_family_v2", pd.Series("", index=portfolio.index)
    ).fillna("").astype(str)
    fallback_family = portfolio.get("target_assay_family", pd.Series("", index=portfolio.index)).fillna("").astype(str)
    portfolio["target_assay_family_v5"] = portfolio["target_assay_family_v5"].where(
        portfolio["target_assay_family_v5"].ne(""), fallback_family
    )
    portfolio["original_indication_v5"] = portfolio.get(
        "fda_indication", pd.Series("", index=portfolio.index)
    ).fillna("").replace("", "原FDA表未提供")
    current = pd.read_csv(CURRENT1000, usecols=["drug_chembl_id", "sequence_key"])
    current_keys = set(map(tuple, current[["drug_chembl_id", "sequence_key"]].itertuples(index=False, name=None)))
    portfolio["in_previous_final1000_v4"] = [
        (drug, target) in current_keys for drug, target in zip(portfolio["drug_chembl_id"], portfolio["sequence_key"])
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    primary.to_csv(OUT / "P1_CALIBRATED_TARGET_QSAR_IN_DOMAIN_V5.csv", index=False)
    secondary.to_csv(OUT / "P2_VALIDATED_LIGAND_SIMILARITY_IN_DOMAIN_V5.csv", index=False)
    exploration.to_csv(OUT / "P3_REMOTE_UNCALIBRATED_PHYSICS_EXPLORATION_V5.csv", index=False)
    family_extension.to_csv(OUT / "R_FAMILY_EXTENSION_REVIEW_V5.csv", index=False)
    controls.to_csv(OUT / "C_POSITIVE_CONTROL_OR_REDISCOVERY_V5.csv", index=False)
    portfolio.to_csv(OUT / "FINAL1000_EVIDENCE_STRATIFIED_V5.csv", index=False)
    compact_chinese(portfolio).to_csv(OUT / "FINAL1000_EVIDENCE_STRATIFIED_TEACHER_ZH_V5.csv", index=False)

    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "portfolio_rows": int(len(portfolio)),
        "portfolio_lane_counts": {
            str(key): int(value) for key, value in portfolio["portfolio_lane_v5"].value_counts().items()
        },
        "portfolio_unique_drugs": int(portfolio["drug_chembl_id"].nunique()),
        "portfolio_unique_targets": int(portfolio["sequence_key"].nunique()),
        "portfolio_unique_scaffolds": int(portfolio["murcko_scaffold"].nunique()),
        "previous_final1000_overlap": int(portfolio["in_previous_final1000_v4"].sum()),
        "source_queue_rows": {
            "P1": int(len(primary)),
            "P2": int(len(secondary)),
            "P3": int(len(exploration)),
            "family_extension_review": int(len(family_extension)),
            "positive_control_or_rediscovery": int(len(controls)),
        },
        "method_contract": {
            "P1": "pair ranking allowed only for target-specific QSAR that beat similarity under scaffold holdout and within ligand applicability domain",
            "P2": "ligand similarity exploitation; not remote-scaffold discovery",
            "P3": "testable remote exploration; no calibrated binding probability",
            "ConPLEx": "retrieval only",
            "Boltz": "conditional pose-generation only after failed paired smoke discrimination",
            "pocket_and_tractability": "target routing/readiness only",
            "disease_methods": "post-binding annotation only",
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
    }
    (OUT / "CALIBRATED_PORTFOLIO_V5_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
