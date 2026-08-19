#!/usr/bin/env python3
"""Materialize and audit the deployment-aligned S6 retrieval protocol.

S6 asks the exact application question that S4/S5 do not jointly answer:
for a homology-cold target, rank old drugs whose entities and observed
scaffolds were excluded from fitting.  It extends, but does not mutate, the
frozen S1--S5 assignment artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_biomaster_odti_baselines_v1 import split_masks  # noqa: E402


SOURCE = (
    ROOT
    / "outputs/old_drug_target_sota_v1/benchmark_splits_v1"
    / "CHEMBL37_86674_FROZEN_SPLIT_ASSIGNMENTS_V1.csv.gz"
)
OUT = ROOT / "outputs/biomaster_recent_target_strengthening_v1"
MANIFEST = OUT / "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD_MANIFEST_V1.csv"
AUDIT = OUT / "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD_AUDIT_V1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role_row(data: pd.DataFrame, mask: np.ndarray, fold: int, role: str) -> dict[str, object]:
    frame = data.loc[mask]
    return {
        "protocol": "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD",
        "fold": fold,
        "role": role.upper(),
        "rows": int(len(frame)),
        "positives": int(frame["binary_label"].sum()),
        "negatives": int((frame["binary_label"] == 0).sum()),
        "prevalence": float(frame["binary_label"].mean()),
        "compounds": int(frame["parent_standard_inchi_key"].nunique()),
        "scaffolds": int(frame["scaffold_group"].nunique()),
        "targets": int(frame["target_chembl_id"].nunique()),
        "target_homology_clusters": int(frame["target_homology_cluster"].nunique()),
    }


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    data = pd.read_csv(SOURCE, low_memory=False)
    rows: list[dict[str, object]] = []
    checks: dict[str, bool] = {
        "source_population_is_frozen_86674": len(data) == 86_674,
        "source_pair_ids_unique": bool(data["calibration_pair_id"].is_unique),
    }
    for fold in range(5):
        masks = split_masks(data, "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD", fold)
        for role, mask in masks.items():
            rows.append(role_row(data, mask, fold, role))
        train = data.loc[masks["train"]]
        valid = data.loc[masks["valid"]]
        test = data.loc[masks["test"]]
        prefix = f"fold_{fold}"
        checks[f"{prefix}_roles_nonempty"] = all(masks[role].any() for role in masks)
        checks[f"{prefix}_valid_test_have_both_classes"] = (
            valid["binary_label"].nunique() == 2 and test["binary_label"].nunique() == 2
        )
        checks[f"{prefix}_train_has_no_old_drug_entity"] = not bool(
            train["is_deployment_old_drug"].astype(bool).any()
        )
        checks[f"{prefix}_train_old_drug_scaffold_disjoint"] = set(
            train["scaffold_group"]
        ).isdisjoint(set(valid["scaffold_group"]) | set(test["scaffold_group"]))
        checks[f"{prefix}_train_target_cluster_disjoint"] = set(
            train["target_homology_cluster"]
        ).isdisjoint(set(valid["target_homology_cluster"]) | set(test["target_homology_cluster"]))
        checks[f"{prefix}_valid_test_target_cluster_disjoint"] = set(
            valid["target_homology_cluster"]
        ).isdisjoint(test["target_homology_cluster"])
        checks[f"{prefix}_valid_test_only_old_drugs"] = bool(
            valid["is_deployment_old_drug"].astype(bool).all()
            and test["is_deployment_old_drug"].astype(bool).all()
        )
    manifest = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST, index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": "S6_NEW_TARGET_OLD_DRUG_DOUBLE_COLD",
        "question": "For a homology-cold target, retrieve active old drugs excluded from fitting by entity and observed scaffold.",
        "claim_boundary": (
            "Observed ChEMBL pairs only; this validates ranking under double-cold identity constraints, "
            "not open-world unknown-pair truth or prospective efficacy."
        ),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "checks": checks,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST),
        "folds": manifest.to_dict("records"),
    }
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "manifest": report["manifest"],
    }, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
