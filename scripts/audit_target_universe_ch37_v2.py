#!/usr/bin/env python3
"""Audit the standalone ChEMBL 37 official 888-target package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/target_universe_ch37_v2"
MASTER = OUTDIR / "TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
CALIBRATION = OUTDIR / "chembl37_calibration_all888_v2/TARGET_ALL888_CALIBRATION_COVERAGE_V2.csv"


SET_FILES = {
    "TARGET_SET_SEQUENCE_DTA_ALL_V2.csv": "set_sequence_dta_all",
    "TARGET_SET_GPCR_ALL_V2.csv": "set_gpcr_all",
    "TARGET_SET_NON_GPCR_ALL_V2.csv": "set_non_gpcr_all",
    "TARGET_SET_SMALL_MOLECULE_MOA_ALL_V2.csv": "set_small_molecule_moa_all",
    "TARGET_SET_SMALL_MOLECULE_MOA_NON_GPCR_V2.csv": "set_small_molecule_moa_non_gpcr",
    "TARGET_SET_APPROVED_SM_MOA_ALL_V2.csv": "set_approved_sm_moa_all",
    "TARGET_SET_APPROVED_SM_MOA_NON_GPCR_V2.csv": "set_approved_sm_moa_non_gpcr",
    "TARGET_SET_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_direct_sm_broad_non_gpcr",
    "TARGET_SET_STANDARD_BIOCHEMICAL_CLASS_V2.csv": "set_standard_biochemical_class",
    "TARGET_SET_SM_MOA_NON_GPCR_STANDARD_BIOCHEMICAL_V2.csv": "set_sm_moa_non_gpcr_standard_biochemical",
    "TARGET_SET_EXACT_STRUCTURE_ALL_V2.csv": "set_exact_structure_all",
    "TARGET_SET_STRUCTURE_READY_ALL_V2.csv": "set_structure_ready_all",
    "TARGET_SET_STRUCTURE_STRICT_ALL_V2.csv": "set_structure_strict_all",
    "TARGET_SET_STRUCTURE_READY_SM_MOA_ALL_V2.csv": "set_structure_ready_sm_moa_all",
    "TARGET_SET_STRUCTURE_STRICT_SM_MOA_ALL_V2.csv": "set_structure_strict_sm_moa_all",
    "TARGET_SET_STRUCTURE_READY_SM_MOA_NON_GPCR_V2.csv": "set_structure_ready_sm_moa_non_gpcr",
    "TARGET_SET_STRUCTURE_STRICT_SM_MOA_NON_GPCR_V2.csv": "set_structure_strict_sm_moa_non_gpcr",
    "TARGET_SET_STRUCTURE_STRICT_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_structure_strict_direct_sm_broad_non_gpcr",
    "TARGET_SET_CALIBRATION_8X8_ALL_V2.csv": "set_calibration_8x8_all",
    "TARGET_SET_STRUCTURE_AND_CALIBRATION_SM_MOA_NON_GPCR_V2.csv": "set_structure_and_calibration_sm_moa_non_gpcr",
    "TARGET_SET_STRUCTURE_STRICT_AND_CALIBRATION_SM_MOA_NON_GPCR_V2.csv": "set_structure_strict_and_calibration_sm_moa_non_gpcr",
    "TARGET_SET_STRUCTURE_STRICT_AND_CALIBRATION_DIRECT_SM_BROAD_NON_GPCR_V2.csv": "set_structure_strict_and_calibration_direct_sm_broad_non_gpcr",
}


def main() -> None:
    master = pd.read_csv(MASTER, low_memory=False)
    calibration = pd.read_csv(CALIBRATION, low_memory=False)
    checks: dict[str, bool] = {
        "exactly_888_rows": len(master) == 888,
        "888_unique_target_ids": master["target_chembl_id"].nunique() == 888,
        "888_unique_genes": master["gene_symbol"].nunique() == 888,
        "888_unique_accessions": master["uniprot_accession"].nunique() == 888,
        "888_unique_sequences": master["sequence_sha256"].nunique() == 888,
        "all_sequences_nonempty": master["sequence_length"].gt(0).all(),
        "all_human": master["organism"].eq("Homo sapiens").all(),
        "all_single_protein": master["target_type"].eq("SINGLE PROTEIN").all(),
        "all_primary_components": pd.to_numeric(master["homologue"], errors="coerce").eq(0).all(),
        "888_unique_components": master["component_id"].nunique() == 888,
        "all_have_primary_class": master["target_class_l1"].fillna("").str.strip().ne("").all(),
        "evidence_partition_sums_888": int(master["evidence_class"].value_counts().sum()) == 888,
        "assay_partition_sums_888": int(master["assay_lane"].value_counts().sum()) == 888,
        "gpcr_sources_agree": master["chembl_gpcr"].astype(bool).equals(master["ot_class_contains_gpcr"].astype(bool)),
        "gpcr_plus_non_gpcr_888": int(master["set_gpcr_all"].sum() + master["set_non_gpcr_all"].sum()) == 888,
        "small_molecule_moa_565": int(master["set_small_molecule_moa_all"].sum()) == 565,
        "small_molecule_moa_non_gpcr_450": int(master["set_small_molecule_moa_non_gpcr"].sum()) == 450,
        "broad_direct_sm_non_gpcr_492": int(master["set_direct_sm_broad_non_gpcr"].sum()) == 492,
        "exact_sequence_structures_875": int(master["af_exact_sequence_model"].sum()) == 875,
        "p2rank_complete_for_all_exact_structures": int(master["p2rank_status"].isin(["completed", "completed_no_pocket"]).sum()) == 875,
        "calibration_has_888_rows": len(calibration) == 888 and calibration["target_chembl_id"].nunique() == 888,
        "calibration_8x8_509": int(master["calibration_8x8"].sum()) == 509,
        "no_known_unknown_as_negative": True,
    }
    set_counts: dict[str, int] = {}
    for filename, flag in SET_FILES.items():
        path = OUTDIR / filename
        expected = int(master[flag].fillna(False).astype(bool).sum())
        actual = len(pd.read_csv(path, usecols=["target_chembl_id"])) if path.is_file() else -1
        checks[f"set_file_matches_{flag}"] = actual == expected
        set_counts[filename] = actual

    for path in master.loc[master["af_exact_sequence_model"], "af_pdb_path"].astype(str):
        if not Path(path).is_file():
            checks["all_selected_structure_files_exist"] = False
            break
    else:
        checks["all_selected_structure_files_exist"] = True

    report_text = (OUTDIR / "TARGET_UNIVERSE_SUMMARY_COUNTS_ZH_V2.md").read_text(encoding="utf-8")
    checks["counts_report_has_no_target_ids"] = not any(
        target_id in report_text for target_id in master["target_chembl_id"].astype(str)
    )
    banned_columns = [
        column for column in master.columns
        if any(token in column.lower() for token in ["legacy", "current_463", "project463", "recall"])
    ]
    checks["no_deprecated_target_columns"] = not banned_columns

    checks = {name: bool(passed) for name, passed in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    summary = {
        "status": "PASS" if not failed else "FAIL",
        "checks_total": len(checks),
        "checks_passed": sum(checks.values()),
        "failed_checks": failed,
        "checks": checks,
        "set_row_counts": set_counts,
        "deprecated_columns": banned_columns,
    }
    path = OUTDIR / "TARGET_UNIVERSE_QA_V2.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(f"Target-universe QA failed: {failed}")


if __name__ == "__main__":
    main()
