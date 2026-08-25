#!/usr/bin/env python3
"""Cross-audit local-unreported physical candidates against the 2026 KiRHub dataset."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1"
    / "drug_centric_cross_target_v1"
)
CANDIDATES = BASE / "DRUG_CENTRIC_LOCAL_UNREPORTED_PHYSICAL_CANDIDATES_V1.csv"
WORKBOOK = BASE / "external_benchmark/41587_2026_3090_MOESM4_ESM.xlsx"
SOURCE_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41587-026-03090-8/MediaObjects/41587_2026_3090_MOESM4_ESM.xlsx"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_drug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def target_columns(columns: list[object], gene: str) -> list[str]:
    gene = gene.upper()
    output = []
    for value in columns:
        column = str(value)
        tokens = set(re.findall(r"[A-Z0-9]+", column.upper()))
        if column.upper() == gene or gene in tokens:
            output.append(column)
    return output


def main() -> None:
    for path in [CANDIDATES, WORKBOOK]:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    wild_type = pd.read_excel(WORKBOOK, sheet_name="Table S4", header=8)
    mutants = pd.read_excel(WORKBOOK, sheet_name="Table S13", header=7)
    wild_type["normalized_drug"] = wild_type["Compound"].map(normalize_drug)
    mutants["normalized_drug"] = mutants["Compound"].map(normalize_drug)
    wild_index = wild_type.set_index("normalized_drug", drop=False)
    mutant_drugs = set(mutants["normalized_drug"])
    wt_columns = [column for column in wild_type.columns if column not in {"Compound", "normalized_drug"}]
    mutant_columns = [
        column for column in mutants.columns if column not in {"Compound", "normalized_drug"}
    ]
    target_map = {
        gene: target_columns(wt_columns, gene)
        for gene in sorted(set(candidates["gene_symbol"].astype(str)))
    }
    mutant_target_map = {
        gene: target_columns(mutant_columns, gene)
        for gene in sorted(set(candidates["gene_symbol"].astype(str)))
    }
    rows = []
    for row in candidates.itertuples(index=False):
        drug_key = normalize_drug(row.drug_names)
        gene = str(row.gene_symbol).upper()
        matched_columns = target_map.get(gene, [])
        values: list[float] = []
        if drug_key in wild_index.index and matched_columns:
            selected = wild_index.loc[[drug_key], matched_columns]
            values = [
                float(value)
                for value in selected.to_numpy().ravel()
                if pd.notna(value) and str(value).strip() != ""
            ]
        if drug_key not in wild_index.index:
            status = "E0_DRUG_NOT_IN_KIRHUB_92_PANEL"
        elif not matched_columns:
            status = "E1_TARGET_NOT_IN_KIRHUB_409_WT_PANEL"
        elif not values:
            status = "E2_KIRHUB_WT_MEASUREMENT_MISSING"
        elif min(values) <= 30.0:
            status = "E3_EXTERNAL_SUPPORTED_GE70PCT_INHIBITION_AT_1UM"
        else:
            status = "E4_EXTERNAL_NOT_SUPPORTED_LT70PCT_INHIBITION_AT_1UM"
        variant_columns = mutant_target_map.get(gene, [])
        rows.append(
            {
                "pairId": row.pairId,
                "kirhub_wt_benchmark_status": status,
                "kirhub_drug_in_92_panel": drug_key in wild_index.index,
                "kirhub_target_wt_columns": ";".join(matched_columns),
                "kirhub_wt_min_residual_activity_pct_1uM": min(values) if values else None,
                "kirhub_wt_max_inhibition_pct_1uM": 100.0 - min(values) if values else None,
                "kirhub_target_mutant_columns": ";".join(variant_columns),
                "kirhub_target_mutant_variant_count": len(variant_columns),
                "kirhub_drug_in_mutant_panel": drug_key in mutant_drugs,
                "external_rediscovery_control": status
                == "E3_EXTERNAL_SUPPORTED_GE70PCT_INHIBITION_AT_1UM",
                "retained_after_kirhub_novelty_gate": status
                in {
                    "E0_DRUG_NOT_IN_KIRHUB_92_PANEL",
                    "E1_TARGET_NOT_IN_KIRHUB_409_WT_PANEL",
                    "E2_KIRHUB_WT_MEASUREMENT_MISSING",
                },
            }
        )
    data = candidates.merge(pd.DataFrame(rows), on="pairId", how="left", validate="one_to_one")
    data["kirhub_novelty_priority"] = data["kirhub_wt_benchmark_status"].map(
        {
            "E0_DRUG_NOT_IN_KIRHUB_92_PANEL": 0,
            "E1_TARGET_NOT_IN_KIRHUB_409_WT_PANEL": 1,
            "E2_KIRHUB_WT_MEASUREMENT_MISSING": 2,
            "E3_EXTERNAL_SUPPORTED_GE70PCT_INHIBITION_AT_1UM": 3,
            "E4_EXTERNAL_NOT_SUPPORTED_LT70PCT_INHIBITION_AT_1UM": 4,
        }
    )
    data = data.sort_values(
        [
            "kirhub_novelty_priority", "drug_centric_stage_priority",
            "dta_bidirectional_top10pct_concordant_384", "dta_cross_target_consensus_score",
        ],
        ascending=[True, True, False, False], kind="mergesort",
    ).reset_index(drop=True)
    data["kirhub_audited_candidate_rank"] = np.arange(1, len(data) + 1)
    retained = data[
        data["retained_after_kirhub_novelty_gate"]
        & data["drug_centric_evidence_stage"].eq("D1_STATE_OR_SEED_STABLE")
    ].copy()
    rediscovery = data[data["external_rediscovery_control"]].copy()
    contradicted = data[
        data["kirhub_wt_benchmark_status"].eq(
            "E4_EXTERNAL_NOT_SUPPORTED_LT70PCT_INHIBITION_AT_1UM"
        )
    ].copy()
    all_path = BASE / "DRUG_CENTRIC_LOCAL_UNREPORTED_KIRHUB_AUDIT_V1.csv"
    retained_path = BASE / "DRUG_CENTRIC_KIRHUB_UNCOVERED_STABLE_CANDIDATES_V1.csv"
    rediscovery_path = BASE / "DRUG_CENTRIC_KIRHUB_EXTERNAL_REDISCOVERY_CONTROLS_V1.csv"
    contradicted_path = BASE / "DRUG_CENTRIC_KIRHUB_EXTERNAL_NOT_SUPPORTED_V1.csv"
    data.to_csv(all_path, index=False)
    retained.to_csv(retained_path, index=False)
    rediscovery.to_csv(rediscovery_path, index=False)
    contradicted.to_csv(contradicted_path, index=False)
    checks = {
        "all_local_physical_candidates_audited": len(data) == len(candidates),
        "workbook_has_92_drugs": len(wild_type) == 92 and len(mutants) == 92,
        "workbook_has_409_wild_type_kinases": len(wt_columns) == 409,
        "workbook_has_349_mutant_or_fusion_constructs": len(mutant_columns) == 349,
        "benchmark_status_complete": data["kirhub_wt_benchmark_status"].notna().all(),
        "rediscovery_and_retained_are_disjoint": not set(rediscovery["pairId"]) & set(
            retained["pairId"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "input_candidate_pairs": int(len(data)),
        "benchmark_status_counts": {
            str(key): int(value) for key, value in data["kirhub_wt_benchmark_status"].value_counts().items()
        },
        "stable_candidates_outside_kirhub_coverage": int(len(retained)),
        "external_rediscovery_controls": int(len(rediscovery)),
        "external_not_supported_pairs": int(len(contradicted)),
        "benchmark_scope": (
            "KiRHub Table S4 is a duplicate 1 uM biochemical activity screen of 92 clinical "
            "kinase inhibitors against 409 wild-type kinases. <=30% residual activity is used "
            "only as an external rediscovery-control gate. >30% is external non-support under "
            "that assay, not proof of universal inactivity. Drugs outside the 92-compound panel "
            "remain uncovered, not validated."
        ),
        "source": {
            "article_doi": "10.1038/s41587-026-03090-8",
            "supplementary_workbook_url": SOURCE_URL,
            "workbook_sha256": sha256(WORKBOOK),
        },
        "inputs": {str(CANDIDATES.relative_to(ROOT)): sha256(CANDIDATES)},
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [all_path, retained_path, rediscovery_path, contradicted_path]
        },
    }
    summary_path = BASE / "DRUG_CENTRIC_KIRHUB_AUDIT_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
