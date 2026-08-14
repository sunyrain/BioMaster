#!/usr/bin/env python3
"""Audit temporal first-observation DTI feasibility without training a model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "outputs/current_production_package_v2/chembl37_target_calibration_v5/PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
FROZEN = ROOT / "outputs/old_drug_target_sota_v1/benchmark_splits_v1/CHEMBL37_86674_FROZEN_SPLIT_ASSIGNMENTS_V1.csv.gz"
OLD_DRUGS = ROOT / "configs/project_drugs_v4.csv"
OUT = ROOT / "outputs/old_drug_target_sota_v1/first_event_dti_feasibility_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(frame: pd.DataFrame, period: str) -> dict[str, object]:
    labels = frame["calibration_label"].astype(str)
    return {
        "period": period,
        "pairs": int(len(frame)),
        "positive": int(labels.eq("positive").sum()),
        "negative_or_inactive": int(labels.eq("negative_or_inactive").sum()),
        "compounds": int(frame["parent_standard_inchi_key"].nunique()),
        "targets": int(frame["sequence_key"].nunique()),
        "old_drug_pairs": int(frame["is_deployment_old_drug"].sum()),
        "old_drugs": int(frame.loc[frame["is_deployment_old_drug"], "parent_molecule_chembl_id"].nunique()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "sequence_key",
        "target_chembl_id",
        "parent_molecule_chembl_id",
        "parent_standard_inchi_key",
        "parent_canonical_smiles",
        "min_document_year",
        "max_document_year",
        "calibration_label",
        "numeric_positive_negative_conflict",
        "explicit_inactive_positive_conflict",
    ]
    full = pd.read_csv(FULL, usecols=columns, low_memory=False)
    frozen = pd.read_csv(
        FROZEN,
        usecols=["sequence_key", "target_chembl_id", "target_homology_cluster", "temporal_role"],
        low_memory=False,
    ).drop_duplicates("sequence_key")
    old = pd.read_csv(OLD_DRUGS, usecols=["base_chembl_id"]).drop_duplicates()
    old_ids = set(old["base_chembl_id"].astype(str))

    eligible_sequences = set(frozen["sequence_key"].astype(str))
    work = full[full["sequence_key"].astype(str).isin(eligible_sequences)].copy()
    work["is_deployment_old_drug"] = work["parent_molecule_chembl_id"].astype(str).isin(old_ids)
    work["first_year"] = pd.to_numeric(work["min_document_year"], errors="coerce")
    conditions = [
        work["first_year"].le(2014),
        work["first_year"].between(2015, 2018, inclusive="both"),
        work["first_year"].between(2019, 2022, inclusive="both"),
        work["first_year"].between(2023, 2025, inclusive="both"),
        work["first_year"].gt(2025),
    ]
    work["event_period"] = np.select(
        conditions,
        ["THROUGH_2014", "2015_2018", "2019_2022", "2023_2025", "AFTER_2025"],
        default="MISSING_YEAR",
    )

    period_order = ["THROUGH_2014", "2015_2018", "2019_2022", "2023_2025", "AFTER_2025", "MISSING_YEAR"]
    rows = [summarize(work[work["event_period"].eq(period)], period) for period in period_order]
    period_summary = pd.DataFrame(rows)
    period_path = OUT / "FIRST_EVENT_DTI_PERIOD_COUNTS_V1.csv"
    period_summary.to_csv(period_path, index=False)

    year_summary = (
        work.dropna(subset=["first_year"])
        .assign(first_year=lambda x: x["first_year"].astype(int))
        .groupby(["first_year", "calibration_label"], as_index=False)
        .agg(
            pairs=("sequence_key", "size"),
            compounds=("parent_standard_inchi_key", "nunique"),
            targets=("sequence_key", "nunique"),
            old_drug_pairs=("is_deployment_old_drug", "sum"),
        )
        .sort_values(["first_year", "calibration_label"], kind="mergesort")
    )
    year_path = OUT / "FIRST_EVENT_DTI_YEAR_LABEL_COUNTS_V1.csv"
    year_summary.to_csv(year_path, index=False)

    label_ok = work["calibration_label"].isin(["positive", "negative_or_inactive"])
    pair_unique = ~work.duplicated(["sequence_key", "parent_standard_inchi_key"], keep=False)
    no_conflict = ~(
        work["numeric_positive_negative_conflict"].fillna(False).astype(bool)
        | work["explicit_inactive_positive_conflict"].fillna(False).astype(bool)
    )
    valid_year = work["first_year"].between(1900, 2025, inclusive="both")
    eligible = label_ok & pair_unique & no_conflict & valid_year

    future_2019_2022 = work[eligible & work["first_year"].between(2019, 2022, inclusive="both")]
    future_2023_2025 = work[eligible & work["first_year"].between(2023, 2025, inclusive="both")]
    summary = {
        "schema_version": "FIRST_EVENT_DTI_FEASIBILITY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "Full uncapped ChEMBL37 strict-binding pair table restricted to the frozen 428 benchmark sequences",
        "counts": {
            "full_input_pairs": int(len(full)),
            "restricted_pairs": int(len(work)),
            "restricted_compounds": int(work["parent_standard_inchi_key"].nunique()),
            "restricted_targets": int(work["sequence_key"].nunique()),
            "deployment_old_drugs_with_any_event": int(
                work.loc[work["is_deployment_old_drug"], "parent_molecule_chembl_id"].nunique()
            ),
            "eligible_event_pairs": int(eligible.sum()),
        },
        "development_windows": {
            "train_through_2014_develop_2015_2018": summarize(
                work[eligible & work["first_year"].between(2015, 2018, inclusive="both")], "2015_2018"
            ),
            "train_through_2018_develop_2019_2022": summarize(future_2019_2022, "2019_2022"),
        },
        "reused_nonindependent_pressure_test": summarize(future_2023_2025, "2023_2025"),
        "checks": {
            "exactly_428_sequences": int(work["sequence_key"].nunique()) == 428,
            "both_event_types_in_2015_2018": work.loc[
                eligible & work["first_year"].between(2015, 2018, inclusive="both"), "calibration_label"
            ].nunique()
            == 2,
            "both_event_types_in_2019_2022": future_2019_2022["calibration_label"].nunique() == 2,
            "both_event_types_in_2023_2025": future_2023_2025["calibration_label"].nunique() == 2,
            "old_drug_future_events_exist": int(future_2019_2022["is_deployment_old_drug"].sum()) > 0,
        },
        "claim_boundary": {
            "unobserved_pairs_are_inactive_labels": False,
            "2023_2025_is_independent_confirmatory_test": False,
            "reason": "S4 labels and baseline results have already been inspected during earlier model development",
        },
        "inputs": {
            str(FULL.relative_to(ROOT)): sha256(FULL),
            str(FROZEN.relative_to(ROOT)): sha256(FROZEN),
            str(OLD_DRUGS.relative_to(ROOT)): sha256(OLD_DRUGS),
        },
        "artifacts": {
            str(period_path.relative_to(ROOT)): sha256(period_path),
            str(year_path.relative_to(ROOT)): sha256(year_path),
        },
    }
    summary_path = OUT / "FIRST_EVENT_DTI_FEASIBILITY_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
