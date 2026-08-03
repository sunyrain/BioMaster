#!/usr/bin/env python3
"""Audit official lightweight SCOPE-DTI target coverage against the V8 universe."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / ".external/scope_dti_lightweight"
PROJECT = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/"
    "PHYSICAL_PAIR_UNIVERSE_334749_HOMOLOGY_AUDITED_V1.csv.gz"
)
OUT = ROOT / "outputs/affinity_experiment_package_v8/report"
SCOPE_TABLES = {
    "Total": SCOPE / "protein_targets/Total_predict.parquet",
    "GPCR": SCOPE / "protein_targets/GPCR_predict.parquet",
    "IC": SCOPE / "protein_targets/IC_predict.parquet",
    "Kinase": SCOPE / "protein_targets/Kinase_predict.parquet",
    "NHR": SCOPE / "protein_targets/NHR_predict.parquet",
}


def read_scope_ids(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["target_uniprot_id"])
    return {str(value) for value in table.column(0).to_pylist() if value}


def read_project_targets() -> dict[str, dict[str, str]]:
    targets: dict[str, dict[str, str]] = {}
    fields = [
        "sequence_key",
        "primary_gene",
        "anchor_canonical_uniprot",
        "representative_protein_id",
        "target_assay_family",
    ]
    with gzip.open(PROJECT, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["sequence_key"]
            if key not in targets:
                targets[key] = {field: row.get(field, "") for field in fields}
    return targets


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scope_sets = {name: read_scope_ids(path) for name, path in SCOPE_TABLES.items()}
    project = read_project_targets()
    rows: list[dict[str, object]] = []
    for key, target in sorted(project.items()):
        canonical = target["anchor_canonical_uniprot"].strip()
        representative = target["representative_protein_id"].split("-")[0].strip()
        lookup = canonical or representative
        families = [
            name
            for name, identifiers in scope_sets.items()
            if name != "Total" and lookup in identifiers
        ]
        rows.append(
            {
                **target,
                "scope_lookup_uniprot_v8": lookup,
                "scope_total_available_v8": lookup in scope_sets["Total"],
                "scope_family_models_v8": ";".join(families),
            }
        )

    csv_path = OUT / "SCOPE_DTI_TARGET_OVERLAP_AUDIT_V8.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(bool(row["scope_total_available_v8"]) for row in rows)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(SCOPE), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        commit = ""
    checkpoints = list((SCOPE / "models_path").glob("**/*.pth"))
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "official_repository": (
            "https://github.com/Yigang-Chen/"
            "Lightweight-SCOPE-DTI-for-Inference"
        ),
        "repository_commit": commit,
        "scope_total_targets": len(scope_sets["Total"]),
        "scope_family_target_counts": {
            name: len(values)
            for name, values in scope_sets.items()
            if name != "Total"
        },
        "official_model_checkpoints": len(checkpoints),
        "project_targets": len(rows),
        "project_targets_matched": matched,
        "project_coverage_fraction": matched / len(rows),
        "project_targets_missing": [
            {
                "sequence_key": row["sequence_key"],
                "gene": row["primary_gene"],
                "uniprot": row["scope_lookup_uniprot_v8"],
            }
            for row in rows
            if not row["scope_total_available_v8"]
        ],
        "interpretation": (
            "Target vocabulary coverage only. SCOPE-DTI is semi-inductive and "
            "uses public DTI supervision, so coverage or a high score must not "
            "be interpreted as remote physical-binding evidence."
        ),
        "csv": str(csv_path.relative_to(ROOT)),
    }
    json_path = OUT / "SCOPE_DTI_TARGET_OVERLAP_AUDIT_V8_SUMMARY.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
