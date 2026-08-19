from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_biomaster_odti_e2_feature_package_v1 import (
    FROZEN_TARGET_INDEX,
    read_rows,
    seq_sha256,
    validate,
)


def test_e2_feature_package_validator_rejects_missing_package(tmp_path: Path) -> None:
    report = validate(
        tmp_path,
        tmp_path / "features.npy",
        tmp_path / "targets.csv.gz",
        tmp_path / "provenance.json",
        tmp_path / "normalization.json",
        1280,
    )
    assert report["decision"]["accepted"] is False
    assert report["decision"]["training_ready"] is False
    assert report["decision"]["fail_closed"] is True
    assert any(item["code"] == "MISSING_ARTIFACT" for item in report["errors"])


def test_e2_feature_package_validator_accepts_contract_complete_package(tmp_path: Path) -> None:
    if not FROZEN_TARGET_INDEX.is_file():
        pytest.skip("local frozen 428-target feature index is not distributed with the source tree")
    frozen = read_rows(FROZEN_TARGET_INDEX)
    target_index = tmp_path / "targets.csv.gz"
    rows = []
    for index, row in enumerate(frozen):
        rows.append(
            {
                "target_feature_index": str(index),
                "sequence_key": row["sequence_key"],
                "protein_sequence": row["protein_sequence"],
                "sequence_sha256": seq_sha256(row["protein_sequence"]),
            }
        )
    with gzip.open(target_index, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    features = tmp_path / "features.npy"
    np.save(features, np.zeros((len(rows), 1280), dtype=np.float32), allow_pickle=False)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "model_id": "independent-test-plm",
                "revision": "test-revision",
                "source_repository": "test://model",
                "weight_sha256": "a" * 64,
                "license": "research-only",
                "input_modalities": ["protein_sequence"],
                "structure_dependency": False,
                "training_label_dependency": False,
            }
        )
    )
    normalization = tmp_path / "normalization.json"
    normalization.write_text(
        json.dumps(
            {
                "method": "train-fold-only mean/std",
                "fit_scope": "train_fold_only",
                "fit_entity_indices": [0, 1],
                "normalization_sha256": "b" * 64,
            }
        )
    )

    report = validate(tmp_path, features, target_index, provenance, normalization, 1280)
    assert report["decision"]["accepted"] is True
    assert report["decision"]["training_ready"] is True
    assert report["decision"]["start_formal_screen"] is False
    assert report["errors"] == []
