#!/usr/bin/env python3
"""Audit a frozen ODTI V2 multi-seed suite before downstream promotion.

The audit is intentionally independent of the training runner.  It checks
artifacts and provenance at run level, and reports INCOMPLETE rather than
silently treating a partial suite as a formal aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ESM2 = Path(
    "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_folds(protocol: str) -> list[int]:
    return [-1] if protocol in {
        "S4_FIRST_SEEN_TEMPORAL_2023_2025",
        "S5_OLD_DRUG_ENTITY_COLD",
    } else list(range(5))


def expected_tasks(protocol: str, seeds: list[int]) -> list[tuple[str, int, int]]:
    return [
        (protocol, fold, seed)
        for fold in task_folds(protocol)
        for seed in seeds
    ]


def audit_run(
    out_dir: Path,
    protocol: str,
    fold: int,
    seed: int,
    expected_structure_dim: int | None,
    expected_target_aux: Path | None,
    expected_drug_aux: Path | None,
) -> dict[str, object]:
    run_dir = out_dir / f"{protocol}__fold_{fold}__seed_{seed}"
    summary_path = run_dir / "RUN_SUMMARY_V2.json"
    prediction_path = run_dir / "TEST_PREDICTIONS_V2.csv.gz"
    checkpoint_path = run_dir / "BEST_MODEL_V2.pt"
    checks: dict[str, bool] = {
        "summary_exists": summary_path.is_file(),
        "prediction_exists": prediction_path.is_file(),
        "checkpoint_exists": checkpoint_path.is_file(),
    }
    if not checks["summary_exists"]:
        return {
            "protocol": protocol,
            "fold": fold,
            "seed": seed,
            "status": "MISSING",
            "checks": checks,
        }
    summary = json.loads(summary_path.read_text())
    checks.update({
        "summary_pass": summary.get("status") == "PASS",
        "summary_identity": (
            summary.get("protocol") == protocol
            and int(summary.get("fold", -999)) == fold
            and int(summary.get("seed", -999)) == seed
        ),
        "all_recorded_checks_pass": all(bool(value) for value in summary.get("checks", {}).values()),
    })
    structure = summary.get("structure", {})
    if expected_structure_dim is not None:
        checks["structure_dim_match"] = int(structure.get("feature_dim", -1)) == expected_structure_dim
    target_aux = summary.get("target_auxiliary", {})
    if expected_target_aux is not None:
        checks["target_aux_enabled"] = bool(target_aux.get("enabled"))
        checks["target_aux_hash_match"] = target_aux.get("sha256") == sha256(expected_target_aux)
    drug_aux = summary.get("drug_auxiliary", {})
    if expected_drug_aux is not None:
        if expected_drug_aux == Path("."):
            checks["drug_aux_disabled_for_baseline"] = not bool(drug_aux.get("enabled"))
        else:
            checks["drug_aux_enabled"] = bool(drug_aux.get("enabled"))
            checks["drug_aux_hash_match"] = drug_aux.get("sha256") == sha256(expected_drug_aux)
    prediction_rows = None
    prediction_finite = False
    prediction_bounded = False
    if prediction_path.is_file():
        try:
            frame = pd.read_csv(
                prediction_path,
                usecols=["calibration_pair_id", "v2_probability_calibrated"],
                low_memory=False,
            )
            prediction_rows = int(len(frame))
            values = frame["v2_probability_calibrated"].to_numpy(dtype=np.float64)
            prediction_finite = bool(np.isfinite(values).all())
            prediction_bounded = bool(((values >= 0.0) & (values <= 1.0)).all())
            checks["prediction_pair_ids_unique"] = not frame["calibration_pair_id"].duplicated().any()
        except Exception as exc:  # pragma: no cover - audit report should preserve the reason
            checks["prediction_readable"] = False
            prediction_finite = False
            prediction_bounded = False
            prediction_rows = f"ERROR: {exc}"
        else:
            checks["prediction_readable"] = True
    checks["prediction_finite"] = prediction_finite
    checks["prediction_bounded"] = prediction_bounded
    passed = all(checks.values())
    return {
        "protocol": protocol,
        "fold": fold,
        "seed": seed,
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "prediction_rows": prediction_rows,
        "summary_path": str(summary_path),
        "prediction_path": str(prediction_path),
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seeds", default="20260816,20260817,20260818,20260819,20260820")
    parser.add_argument("--structure-dim", type=int, default=19)
    parser.add_argument("--target-aux-features", default=str(DEFAULT_ESM2))
    parser.add_argument("--expect-drug-aux-disabled", action="store_true")
    parser.add_argument("--drug-aux-features", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    expected_target_aux = Path(args.target_aux_features) if args.target_aux_features else None
    if args.expect_drug_aux_disabled and args.drug_aux_features:
        raise ValueError("--expect-drug-aux-disabled and --drug-aux-features are mutually exclusive")
    expected_drug_aux = (
        Path(".") if args.expect_drug_aux_disabled else
        (Path(args.drug_aux_features) if args.drug_aux_features else None)
    )
    tasks = expected_tasks(args.protocol, seeds)
    records = [
        audit_run(
            out_dir,
            protocol,
            fold,
            seed,
            args.structure_dim,
            expected_target_aux,
            expected_drug_aux,
        )
        for protocol, fold, seed in tasks
    ]
    completed = [item for item in records if item["status"] == "PASS"]
    missing = [item for item in records if item["status"] == "MISSING"]
    failed = [item for item in records if item["status"] == "FAIL"]
    aggregate_path = out_dir / "V2_MULTI_SEED_SUMMARY.json"
    aggregate_present = aggregate_path.is_file()
    status = "PASS" if len(completed) == len(tasks) and not failed and aggregate_present else "INCOMPLETE"
    progress_path = out_dir / "RUN_PROGRESS_V2.json"
    progress_report: dict[str, object] = {}
    if progress_path.is_file():
        progress_report = json.loads(progress_path.read_text())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": args.protocol,
        "expected_tasks": len(tasks),
        "completed_pass": len(completed),
        "missing": len(missing),
        "failed": len(failed),
        "aggregate_present": aggregate_present,
        "progress_manifest": {
            "status": progress_report.get("status"),
            "completed_count": len(progress_report.get("completed", [])),
            "remaining_count": len(progress_report.get("remaining", [])),
        },
        "feature_expectations": {
            "structure_dim": args.structure_dim,
            "target_aux_source": str(expected_target_aux) if expected_target_aux else None,
            "target_aux_sha256": sha256(expected_target_aux) if expected_target_aux and expected_target_aux.is_file() else None,
            "drug_aux_expected_disabled": bool(args.expect_drug_aux_disabled),
            "drug_aux_expected_source": str(expected_drug_aux) if expected_drug_aux and expected_drug_aux != Path(".") else None,
            "drug_aux_expected_sha256": (
                sha256(expected_drug_aux)
                if expected_drug_aux and expected_drug_aux != Path(".") and expected_drug_aux.is_file()
                else None
            ),
        },
        "runs": records,
        "claim_status": "RUN_LEVEL_ARTIFACT_AUDIT_ONLY; NO_EXTERNAL_OR_PROSPECTIVE_CLAIM",
    }
    output = out_dir / f"{args.protocol}_SUITE_AUDIT_V1.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
