#!/usr/bin/env python3
"""Audit the v5 evidence-stratified portfolio contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/current_production_package_v2/calibrated_portfolio_v5"
FULL = OUT / "FINAL1000_EVIDENCE_STRATIFIED_V5.csv"
ZH = OUT / "FINAL1000_EVIDENCE_STRATIFIED_TEACHER_ZH_V5.csv"


def bools(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = pd.read_csv(FULL, low_memory=False)
    teacher = pd.read_csv(ZH, low_memory=False)
    checks: dict[str, bool] = {
        "full_rows_1000": len(data) == 1000,
        "teacher_rows_1000": len(teacher) == 1000,
        "pair_unique": not data.duplicated(["drug_chembl_id", "sequence_key"]).any(),
        "rank_exact_1_to_1000": data["portfolio_rank_v5"].tolist() == list(range(1, 1001)),
        "known_fda_pair_absent": not bools(data["is_known_fda_target_pair"]).any(),
        "family_extension_absent": not bools(data["family_or_rediscovery_risk_v2"]).any(),
        "selection_contract_exact": data["selection_contract_v5"].eq(
            "evidence_stratified_no_universal_score"
        ).all(),
        "teacher_columns_unique": not teacher.columns.duplicated().any(),
    }
    p1 = data[data["portfolio_lane_v5"].eq("P1_calibrated_target_qsar_in_domain")]
    checks.update(
        {
            "p1_status_exact": p1["target_ligand_model_status_v5"].eq("T1_qsar_beats_similarity").all(),
            "p1_in_domain": bools(p1["in_known_ligand_applicability_domain_v5"]).all(),
            "p1_percentile_ge_0_8": pd.to_numeric(p1["target_qsar_percentile_v5"], errors="coerce").ge(0.80).all(),
            "p1_qsar_bootstrap_superiority": pd.to_numeric(
                p1["qsar_minus_similarity_ap_ci95_low"], errors="coerce"
            ).gt(0).all(),
            "p1_no_temporal_contradiction": ~p1["temporal_validation_status_v5"].eq(
                "contradicts_scaffold_result"
            ).any(),
            "p1_exact_active_absent": not bools(p1["exact_known_active_smiles_v5"]).any(),
        }
    )
    p2 = data[data["portfolio_lane_v5"].eq("P2_validated_ligand_similarity_in_domain")]
    checks.update(
        {
            "p2_status_exact": p2["target_ligand_model_status_v5"].eq("T2_similarity_supported").all(),
            "p2_in_domain": bools(p2["in_known_ligand_applicability_domain_v5"]).all(),
            "p2_percentile_ge_0_8": pd.to_numeric(p2["target_qsar_percentile_v5"], errors="coerce").ge(0.80).all(),
            "p2_exact_active_absent": not bools(p2["exact_known_active_smiles_v5"]).any(),
        }
    )
    p3 = data[data["portfolio_lane_v5"].eq("P3_remote_uncalibrated_physics_exploration")]
    remote = pd.to_numeric(p3["max_known_active_similarity_v5"], errors="coerce").lt(0.40) | p3[
        "max_known_active_similarity_v5"
    ].isna()
    checks.update(
        {
            "p3_remote_or_unmapped": remote.all(),
            "p3_pose_ready": p3["pose_stability_tier"].fillna("").astype(str).str.startswith(("A_", "B_")).all(),
            "p3_boltz_completed": bools(p3["boltz_completed_refined"]).all(),
            "p3_sequence_match": not bools(p3["structure_sequence_mismatch_v4"]).any(),
            "p3_conplex_role_retrieval_only": p3["conplex_role_v5"].eq("uncalibrated_retrieval_only").all(),
            "p3_boltz_role_pose_only": p3["boltz_role_v5"].eq(
                "conditional_pose_generation_not_binding_discrimination"
            ).all(),
        }
    )
    checks = {key: bool(value) for key, value in checks.items()}
    failures = sorted(key for key, value in checks.items() if not value)
    summary = {
        "status": "passed" if not failures else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failures": failures,
        "lane_counts": {str(key): int(value) for key, value in data["portfolio_lane_v5"].value_counts().items()},
        "full_sha256": sha256(FULL),
        "teacher_sha256": sha256(ZH),
    }
    (OUT / "CALIBRATED_PORTFOLIO_INVARIANT_AUDIT_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
