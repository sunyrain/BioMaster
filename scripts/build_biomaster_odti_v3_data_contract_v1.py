#!/usr/bin/env python3
"""Audit the frozen ODTI pair store and publish the V3/W1 label contract.

This is a data-readiness artifact, not a model trainer.  It deliberately
does not infer unknown pairs as negatives and does not manufacture censoring
or replicate metadata that is absent from the frozen ChEMBL37 feature store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / (
    "outputs/old_drug_target_sota_v1/feature_store_v1/"
    "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_v3_data_contract_v1"

REQUIRED_COLUMNS = {
    "calibration_pair_id",
    "parent_molecule_chembl_id",
    "target_chembl_id",
    "calibration_label",
    "binary_label",
    "any_explicit_inactive",
    "explicit_inactive_positive_conflict",
    "numeric_positive_negative_conflict",
    "min_pchembl",
    "max_pchembl",
    "mean_pchembl",
    "numeric_rows",
    "activity_rows",
    "assay_count",
    "document_count",
    "standard_types",
    "relationship_types",
    "assay_ids",
    "doc_ids",
    "min_document_year",
    "max_document_year",
    "target_assay_family",
    "scaffold_group",
    "target_homology_cluster",
}

W1_REQUIRED_FIELDS = [
    "w1_candidate_id",
    "prospective_pair_id",
    "calibration_pair_id",
    "training_store_overlap_status",
    "drug_entity_key",
    "target_entity_key",
    "assay_id",
    "assay_lane",
    "plate_id",
    "blinded_sample_code",
    "independent_run",
    "replicate_id",
    "readout_value",
    "readout_unit",
    "activity_class",
    "censor_lower",
    "censor_upper",
    "assay_status",
    "replicate_variance",
    "assay_metadata_json",
    "raw_data_filename",
    "raw_data_file_sha256",
    "unblinding_status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def finite_stats(frame: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[values.notna()]
    if values.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def build_contract(pairs_path: Path, out_dir: Path) -> dict[str, Any]:
    if not pairs_path.is_file():
        raise FileNotFoundError(pairs_path)
    frame = pd.read_csv(pairs_path, low_memory=False)
    missing_required = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_required:
        raise RuntimeError(f"Frozen pair store missing required columns: {missing_required}")

    checks = {
        "required_columns_present": not missing_required,
        "pair_ids_unique": not frame["calibration_pair_id"].duplicated().any(),
        "binary_label_matches_calibration_label": bool(
            frame["binary_label"].eq(frame["calibration_label"].eq("positive").astype(int)).all()
        ),
        "no_positive_explicit_inactive_conflict": bool(
            (~frame["explicit_inactive_positive_conflict"].astype(bool)).all()
        ),
        "no_numeric_positive_negative_conflict": bool(
            (~frame["numeric_positive_negative_conflict"].astype(bool)).all()
        ),
        "all_smiles_and_inchikeys_present": bool(
            frame["model_ligand_smiles"].notna().all()
            and frame["parent_standard_inchi_key"].notna().all()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))

    explicit_inactive = frame["any_explicit_inactive"].astype(bool)
    numeric_present = pd.to_numeric(frame["mean_pchembl"], errors="coerce").notna()
    missing_numeric = ~numeric_present
    likely_affinity_fields = {
        "exact_or_aggregated_pchembl": ["min_pchembl", "max_pchembl", "mean_pchembl"],
        "numeric_rows": ["numeric_rows"],
        "standard_types": ["standard_types"],
        "relationship_types": ["relationship_types"],
    }
    absent_semantic_fields = [
        "observation_indicator",
        "replicate_id",
        "replicate_variance",
        "assay_condition_json",
        "censor_lower",
        "censor_upper",
        "raw_readout_value",
        "raw_readout_unit",
        "assay_failure_reason",
    ]

    contract: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "contract_name": "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1",
        "claim_status": "DATA_READINESS_ONLY; NO NEW MODEL OR PROSPECTIVE CLAIM",
        "source": {
            "path": str(pairs_path),
            "relative_path": relative_or_absolute(pairs_path),
            "sha256": sha256(pairs_path),
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
        },
        "current_training_pool": {
            "pairs": int(len(frame)),
            "unique_drugs": int(frame["parent_molecule_chembl_id"].nunique()),
            "unique_targets": int(frame["target_chembl_id"].nunique()),
            "unique_scaffolds": int(frame["scaffold_group"].nunique(dropna=True)),
            "positive_rows": int(frame["calibration_label"].eq("positive").sum()),
            "negative_or_inactive_rows": int(frame["calibration_label"].eq("negative_or_inactive").sum()),
            "explicit_inactive_rows": int(explicit_inactive.sum()),
            "numeric_affinity_rows": int(numeric_present.sum()),
            "missing_numeric_affinity_rows": int(missing_numeric.sum()),
            "feature_unavailable_rows": int((~frame["drug_feature_available"].astype(bool)).sum()),
            "missing_murcko_scaffold_rows": int(frame["murcko_scaffold"].isna().sum()),
            "target_assay_family_counts": count_values(frame, "target_assay_family"),
            "calibration_label_counts": count_values(frame, "calibration_label"),
            "explicit_inactive_counts": count_values(frame, "any_explicit_inactive"),
            "temporal_role_counts": count_values(frame, "temporal_role"),
        },
        "affinity_and_assay_coverage": {
            "pchembl_stats": {
                "min": finite_stats(frame, "min_pchembl"),
                "mean": finite_stats(frame, "mean_pchembl"),
                "max": finite_stats(frame, "max_pchembl"),
            },
            "activity_rows": finite_stats(frame, "activity_rows"),
            "assay_count": finite_stats(frame, "assay_count"),
            "document_count": finite_stats(frame, "document_count"),
            "metadata_fields_available": likely_affinity_fields,
            "semantic_fields_absent_from_frozen_store": absent_semantic_fields,
            "censoring_contract_available": False,
            "replicate_variance_available": False,
            "observation_indicator_available": False,
        },
        "label_contract": {
            "positive_rule_current": "calibration_label == positive (binary_label == 1)",
            "negative_rule_current": "calibration_label == negative_or_inactive (binary_label == 0)",
            "explicit_inactive_is_preserved": True,
            "unknown_pair_is_negative": False,
            "grey_zone_policy": "already resolved in frozen calibration store; do not reconstruct from unknown pairs",
            "conflict_rows": {
                "explicit_inactive_positive_conflict": int(frame["explicit_inactive_positive_conflict"].astype(bool).sum()),
                "numeric_positive_negative_conflict": int(frame["numeric_positive_negative_conflict"].astype(bool).sum()),
            },
        },
        "w1_required_fields": W1_REQUIRED_FIELDS,
        "w1_allowed_activity_classes": ["active", "inactive", "borderline", "failed", "not_interpretable"],
        "w1_rules": {
            "freeze_predictions_before_assay": True,
            "keep_blinding_key_separate_from_operator_file": True,
            "calibration_pair_id_nullable_for_unseen_prospective_pairs": True,
            "prospective_pair_id_required_when_calibration_pair_id_is_null": True,
            "retain_raw_data_hashes": True,
            "keep_failed_and_borderline_results": True,
            "do_not_use_w1_for_tuning_before_prospective_evaluation": True,
            "do_not_convert_failed_or_unobserved_pairs_to_negative": True,
        },
        "v3_training_requirements": [
            "Add explicit assay-level inactive and censored bounds without replacing unknown-pair semantics.",
            "Add replicate variance or a documented surrogate only when real replicate data exist.",
            "Add observation indicator/propensity only when observation mechanism is explicitly recorded.",
            "Keep untouched external and W1 evaluation sets outside model selection.",
            "Re-run S1-S5, cluster bootstrap, calibration and external claims after data update.",
        ],
        "checks": checks,
        "outputs": {
            "json": str(out_dir / "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json"),
            "markdown": str(out_dir / "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.md"),
        },
    }
    return contract


def render_markdown(contract: dict[str, Any]) -> str:
    pool = contract["current_training_pool"]
    coverage = contract["affinity_and_assay_coverage"]
    lines = [
        "# BioMaster ODTI：V3/W1 标签与数据契约",
        "",
        f"生成时间：{contract['created_utc']}",
        "",
        f"状态：`{contract['status']}`；本文件只描述数据 readiness，不代表新模型或 prospective validation。",
        "",
        "## 当前冻结训练池",
        "",
        f"- pair：{pool['pairs']}；drug：{pool['unique_drugs']}；target：{pool['unique_targets']}；scaffold：{pool['unique_scaffolds']}",
        f"- positive：{pool['positive_rows']}；negative_or_inactive：{pool['negative_or_inactive_rows']}",
        f"- explicit inactive：{pool['explicit_inactive_rows']}",
        f"- numeric pChEMBL：{pool['numeric_affinity_rows']}；缺少 numeric pChEMBL：{pool['missing_numeric_affinity_rows']}",
        "",
        "## 当前缺口",
        "",
        "冻结 feature store 没有以下字段：",
        "",
    ]
    lines.extend(f"- `{field}`" for field in coverage["semantic_fields_absent_from_frozen_store"])
    lines += [
        "",
        "因此当前模型可以使用显式 inactive 和聚合 pChEMBL，但不能把模型输出解释为 assay-level calibrated probability，也不能从未知 pair 推断真实阴性。",
        "",
        "## W1 回流字段",
        "",
        "W1 每个候选至少应保留：",
        "",
    ]
    lines.extend(f"- `{field}`" for field in contract["w1_required_fields"])
    lines += [
        "",
        "## V3 更新规则",
        "",
    ]
    lines.extend(f"{index}. {rule}" for index, rule in enumerate(contract["v3_training_requirements"], 1))
    lines += [
        "",
        "## 关键不变量",
        "",
        "```text",
        "unknown pair != negative",
        "W1 frozen prediction -> prospective evaluation -> V3 update -> W2/external revalidation",
        "failed/borderline/replicate metadata must be retained",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = ROOT / pairs_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract(pairs_path, out_dir)
    json_path = out_dir / "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.json"
    md_path = out_dir / "BIOMASTER_ODTI_V3_W1_DATA_CONTRACT_V1.md"
    json_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    md_path.write_text(render_markdown(contract))
    print(json.dumps({
        "status": contract["status"],
        "rows": contract["source"]["rows"],
        "positive_rows": contract["current_training_pool"]["positive_rows"],
        "explicit_inactive_rows": contract["current_training_pool"]["explicit_inactive_rows"],
        "missing_numeric_affinity_rows": contract["current_training_pool"]["missing_numeric_affinity_rows"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
