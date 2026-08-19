#!/usr/bin/env python3
"""Fail-closed validator for an external BioMaster E2 feature package.

The validator checks the package contract before any model screen is allowed.
It does not train, normalize, impute or modify the candidate features.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = ROOT / "outputs/biomaster_odti_e2_feature_package_v1"
FROZEN_TARGET_INDEX = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
DEFAULT_EXPECTED_DIM = 1280


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def seq_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def validate(
    package_dir: Path,
    feature_matrix_path: Path,
    target_index_path: Path,
    provenance_path: Path,
    normalization_path: Path,
    expected_dim: int,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    required = {
        "feature_matrix": feature_matrix_path,
        "target_index": target_index_path,
        "provenance": provenance_path,
        "normalization": normalization_path,
        "frozen_target_index": FROZEN_TARGET_INDEX,
    }
    for name, path in required.items():
        if not path.is_file():
            errors.append({"code": "MISSING_ARTIFACT", "field": name, "path": str(path)})

    provenance = read_json(provenance_path)
    normalization = read_json(normalization_path)
    candidate_rows: list[dict[str, str]] = []
    frozen_rows: list[dict[str, str]] = []
    matrix: np.ndarray | None = None
    if target_index_path.is_file():
        try:
            candidate_rows = read_rows(target_index_path)
        except Exception as exc:
            errors.append({"code": "TARGET_INDEX_READ_ERROR", "detail": f"{type(exc).__name__}: {exc}"})
    if FROZEN_TARGET_INDEX.is_file():
        try:
            frozen_rows = read_rows(FROZEN_TARGET_INDEX)
        except Exception as exc:
            errors.append({"code": "FROZEN_TARGET_INDEX_READ_ERROR", "detail": f"{type(exc).__name__}: {exc}"})
    if feature_matrix_path.is_file():
        try:
            matrix = np.load(feature_matrix_path, allow_pickle=False)
            if not isinstance(matrix, np.ndarray):
                errors.append({"code": "FEATURE_MATRIX_NOT_NDARRAY"})
        except Exception as exc:
            errors.append({"code": "FEATURE_MATRIX_READ_ERROR", "detail": f"{type(exc).__name__}: {exc}"})

    required_index_columns = {"target_feature_index", "sequence_key", "protein_sequence", "sequence_sha256"}
    if candidate_rows:
        missing_columns = sorted(required_index_columns - set(candidate_rows[0]))
        if missing_columns:
            errors.append({"code": "TARGET_INDEX_MISSING_COLUMNS", "columns": ",".join(missing_columns)})
        else:
            keys = [row["sequence_key"] for row in candidate_rows]
            sequences = [row["protein_sequence"] for row in candidate_rows]
            if len(keys) != len(set(keys)):
                errors.append({"code": "DUPLICATE_SEQUENCE_KEY"})
            if len(sequences) != len(set(sequences)):
                errors.append({"code": "DUPLICATE_PROTEIN_SEQUENCE"})
            for row in candidate_rows:
                if row["sequence_sha256"] != seq_sha256(row["protein_sequence"]):
                    errors.append({"code": "SEQUENCE_HASH_MISMATCH", "sequence_key": row["sequence_key"]})
                    break

    if candidate_rows and frozen_rows:
        frozen_by_key = {row.get("sequence_key", ""): row for row in frozen_rows}
        candidate_by_key = {row.get("sequence_key", ""): row for row in candidate_rows}
        if set(candidate_by_key) != set(frozen_by_key):
            errors.append({"code": "FROZEN_TARGET_KEY_SET_MISMATCH"})
        else:
            for key, frozen in frozen_by_key.items():
                if candidate_by_key[key].get("protein_sequence") != frozen.get("protein_sequence"):
                    errors.append({"code": "FROZEN_TARGET_SEQUENCE_MISMATCH", "sequence_key": key})
                    break

    if matrix is not None:
        if matrix.ndim != 2:
            errors.append({"code": "FEATURE_MATRIX_NOT_2D", "ndim": str(matrix.ndim)})
        else:
            if matrix.shape[0] != len(candidate_rows):
                errors.append({"code": "FEATURE_ROW_COUNT_MISMATCH", "matrix_rows": str(matrix.shape[0]), "index_rows": str(len(candidate_rows))})
            if matrix.shape[1] != expected_dim:
                errors.append({"code": "FEATURE_DIMENSION_MISMATCH", "observed": str(matrix.shape[1]), "expected": str(expected_dim)})
            if matrix.dtype != np.float32:
                errors.append({"code": "FEATURE_DTYPE_MISMATCH", "observed": str(matrix.dtype), "expected": "float32"})
            if not np.isfinite(matrix).all():
                errors.append({"code": "FEATURE_NONFINITE"})

    required_provenance = {
        "model_id",
        "revision",
        "source_repository",
        "weight_sha256",
        "license",
        "input_modalities",
        "structure_dependency",
        "training_label_dependency",
    }
    missing_provenance = sorted(key for key in required_provenance if key not in provenance)
    if missing_provenance:
        errors.append({"code": "PROVENANCE_MISSING_FIELDS", "fields": ",".join(missing_provenance)})
    if provenance.get("training_label_dependency") is not False:
        errors.append({"code": "LABEL_DEPENDENCY_NOT_FALSE"})
    if provenance.get("structure_dependency") is True:
        warnings.append({"code": "STRUCTURE_DEPENDENCY_DECLARED", "detail": "candidate is not sequence-only; confirm complete structure coverage before screen"})

    required_normalization = {"method", "fit_scope", "fit_entity_indices", "normalization_sha256"}
    missing_normalization = sorted(key for key in required_normalization if key not in normalization)
    if missing_normalization:
        errors.append({"code": "NORMALIZATION_MISSING_FIELDS", "fields": ",".join(missing_normalization)})
    if normalization.get("fit_scope") != "train_fold_only":
        errors.append({"code": "NORMALIZATION_SCOPE_NOT_TRAIN_ONLY"})

    package_manifest = {
        "feature_matrix_path": str(feature_matrix_path.relative_to(ROOT)) if feature_matrix_path.is_relative_to(ROOT) else str(feature_matrix_path),
        "feature_matrix_sha256": sha256(feature_matrix_path),
        "target_index_path": str(target_index_path.relative_to(ROOT)) if target_index_path.is_relative_to(ROOT) else str(target_index_path),
        "target_index_sha256": sha256(target_index_path),
        "provenance_path": str(provenance_path.relative_to(ROOT)) if provenance_path.is_relative_to(ROOT) else str(provenance_path),
        "normalization_path": str(normalization_path.relative_to(ROOT)) if normalization_path.is_relative_to(ROOT) else str(normalization_path),
        "expected_dim": expected_dim,
    }
    return {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ODTI_E2_FEATURE_PACKAGE_VALIDATION_V1",
        "claim_status": "FEATURE_PACKAGE_READINESS_ONLY; NO_MODEL_RESULT",
        "package_dir": str(package_dir),
        "package_manifest": package_manifest,
        "observed": {
            "candidate_target_rows": len(candidate_rows),
            "frozen_target_rows": len(frozen_rows),
            "feature_matrix_shape": list(matrix.shape) if matrix is not None else None,
            "feature_matrix_dtype": str(matrix.dtype) if matrix is not None else None,
        },
        "errors": errors,
        "warnings": warnings,
        "decision": {
            "accepted": not errors,
            "training_ready": not errors,
            "start_formal_screen": False,
            "fail_closed": True,
            "reason": "accepted package may proceed to E2 stage-1 screen only after paired baseline alignment" if not errors else "package rejected; do not train or promote",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--feature-matrix", default=None)
    parser.add_argument("--target-index", default=None)
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--normalization", default=None)
    parser.add_argument("--expected-dim", type=int, default=DEFAULT_EXPECTED_DIM)
    parser.add_argument("--out-dir", default="outputs/biomaster_odti_e2_feature_package_validation_v1")
    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()
    feature_matrix = Path(args.feature_matrix).resolve() if args.feature_matrix else package_dir / "CANDIDATE_PROTEIN_FEATURES_FLOAT32.npy"
    target_index = Path(args.target_index).resolve() if args.target_index else package_dir / "CANDIDATE_TARGET_FEATURE_INDEX.csv.gz"
    provenance = Path(args.provenance).resolve() if args.provenance else package_dir / "MODEL_PROVENANCE.json"
    normalization = Path(args.normalization).resolve() if args.normalization else package_dir / "TRAIN_FOLD_NORMALIZATION.json"
    report = validate(package_dir, feature_matrix, target_index, provenance, normalization, args.expected_dim)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "E2_FEATURE_PACKAGE_VALIDATION_V1.json"
    md_path = out_dir / "E2_FEATURE_PACKAGE_VALIDATION_V1.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# BioMaster ODTI：E2 外部 feature package 验证",
        "",
        f"生成时间：{report['created_utc']}",
        "",
        f"结论：`{'ACCEPTED_FOR_STAGE1_SCREEN' if report['decision']['accepted'] else 'REJECTED_NOT_TRAINING_READY'}`。",
        "",
        f"候选 target rows：`{report['observed']['candidate_target_rows']}`；冻结 target rows：`{report['observed']['frozen_target_rows']}`；feature shape：`{report['observed']['feature_matrix_shape']}`。",
        "",
        "## 错误",
        "",
    ]
    lines.extend([f"- `{item['code']}`：{item.get('detail', item.get('fields', ''))}" for item in report["errors"]] or ["- 无"])
    lines += ["", "## 决策", "", "- 不通过时 fail-closed：不生成训练输入、不启动 E2 formal screen。", "- 通过时仍需 paired E0 baseline alignment，然后只启动冻结的 S2/S3/S5 stage-1 两 seed screen。"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "accepted": report["decision"]["accepted"],
        "training_ready": report["decision"]["training_ready"],
        "errors": len(report["errors"]),
        "json": str(json_path),
    }, ensure_ascii=False, indent=2))
    return 0 if report["decision"]["accepted"] else 2


if __name__ == "__main__":
    sys.exit(main())
