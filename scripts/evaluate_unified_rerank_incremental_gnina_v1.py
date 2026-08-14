#!/usr/bin/env python3
"""Evaluate the unified-rerank GNINA increment against frozen target controls."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_gnina_calibration_338_v1 import parse_target


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/unified_pair_compute_increment_384_v1"
QUEUE = BASE / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENTAL_PAIR_QUEUE_V1.csv"
JOBS = BASE / "execution_v1/gnina_inputs/GNINA_UNIFIED_REMOTE_INCREMENT_JOBS_V1.csv"
CONTROLS = (
    ROOT
    / "outputs/gnina_calibration_338_v1/evaluation"
    / "GNINA_CALIBRATION_LIGAND_EVIDENCE_FULL_V1.csv.gz"
)
TARGETS = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v2"
    / "TARGET_EVIDENCE_LAYER_ROUTING_338_V2.csv"
)
OUT = BASE / "execution_v1/gnina_evaluation"


def quantile(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    return float(np.quantile(numeric, q)) if len(numeric) else math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queue = pd.read_csv(QUEUE, low_memory=False)
    queue = queue[queue["gnina_increment_required"]].copy()
    jobs = pd.read_csv(JOBS, low_memory=False)
    controls = pd.read_csv(CONTROLS, low_memory=False)
    targets = pd.read_csv(TARGETS, low_memory=False).set_index("target_chembl_id")

    frames = []
    integrity_rows = []
    for job in jobs.itertuples(index=False):
        target = str(job.target_chembl_id)
        output = Path(str(job.output_sdf))
        status_path = output.parent / "run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        poses, ligands = parse_target(target, output) if output.exists() else (pd.DataFrame(), pd.DataFrame())
        expected = int(job.prepared_controls)
        complete = bool(status.get("completed")) and len(ligands) == expected
        integrity_rows.append(
            {
                "target_chembl_id": target,
                "expected_candidates": expected,
                "observed_candidates": len(ligands),
                "pose_rows": len(poses),
                "integrity_status": "PASS" if complete else "FAIL",
            }
        )
        if not complete:
            continue
        if target not in targets.index or str(
            targets.loc[target, "gnina_target_qualification"]
        ) != "REMOTE_STRONG":
            raise RuntimeError(f"{target}: frozen GNINA remote qualification is absent")

        candidate = ligands.rename(columns={"control_pair_id": "pairId"})
        task = queue[queue["target_chembl_id"].astype(str).eq(target)]
        candidate = task.merge(
            candidate, on=["target_chembl_id", "pairId"], how="inner", validate="one_to_one"
        )
        if len(candidate) != expected:
            raise RuntimeError(f"{target}: parsed {len(ligands)} but mapped {len(candidate)}")

        development = controls[
            controls["target_chembl_id"].astype(str).eq(target)
            & controls["calibration_split"].eq("PROTOCOL_DEVELOPMENT")
        ]
        positive = development[development["control_class"].eq("positive")]
        negative = development[development["control_class"].eq("negative")]
        if len(positive) < 8 or len(negative) < 8:
            raise RuntimeError(f"{target}: inadequate frozen development controls")
        thresholds = {
            "cnn_negative_q90": quantile(negative["primary_cnn_affinity"], 0.90),
            "cnn_negative_q95": quantile(negative["primary_cnn_affinity"], 0.95),
            "cnn_positive_q25": quantile(positive["primary_cnn_affinity"], 0.25),
            "cnn_positive_q50": quantile(positive["primary_cnn_affinity"], 0.50),
            "vina_negative_q05": quantile(negative["vina_directional"], 0.05),
            "vina_negative_q10": quantile(negative["vina_directional"], 0.10),
            "vina_negative_q20": quantile(negative["vina_directional"], 0.20),
        }
        for key, value in thresholds.items():
            candidate[key] = value
        candidate["cnn_percentile_vs_development_positive"] = candidate[
            "primary_cnn_affinity"
        ].map(lambda score: float((positive["primary_cnn_affinity"] <= score).mean()))
        candidate["cnn_percentile_vs_development_negative"] = candidate[
            "primary_cnn_affinity"
        ].map(lambda score: float((negative["primary_cnn_affinity"] <= score).mean()))
        candidate["vina_percentile_vs_development_negative"] = candidate[
            "vina_directional"
        ].map(lambda score: float((negative["vina_directional"] <= score).mean()))
        tier_a = (
            candidate["primary_cnn_affinity"].ge(thresholds["cnn_negative_q95"])
            & candidate["primary_cnn_affinity"].ge(thresholds["cnn_positive_q50"])
            & candidate["vina_directional"].ge(thresholds["vina_negative_q20"])
        )
        tier_b = (
            candidate["primary_cnn_affinity"].ge(thresholds["cnn_negative_q95"])
            & candidate["primary_cnn_affinity"].ge(thresholds["cnn_positive_q25"])
            & candidate["vina_directional"].ge(thresholds["vina_negative_q10"])
        )
        tier_c = (
            candidate["primary_cnn_affinity"].ge(thresholds["cnn_negative_q90"])
            & candidate["vina_directional"].ge(thresholds["vina_negative_q05"])
        )
        candidate["gnina_remote_evidence_tier"] = "NO_CALIBRATED_SUPPORT"
        candidate.loc[tier_c, "gnina_remote_evidence_tier"] = "GNINA_REMOTE_C"
        candidate.loc[tier_b, "gnina_remote_evidence_tier"] = "GNINA_REMOTE_B"
        candidate.loc[tier_a, "gnina_remote_evidence_tier"] = "GNINA_REMOTE_A"
        candidate["gnina_remote_support"] = candidate["gnina_remote_evidence_tier"].ne(
            "NO_CALIBRATED_SUPPORT"
        )
        candidate["evidence_interpretation"] = (
            "Same-target frozen development-control tier; not a cross-target score or binder probability."
        )
        frames.append(candidate)

    integrity = pd.DataFrame(integrity_rows)
    if len(integrity) != len(jobs) or not integrity["integrity_status"].eq("PASS").all():
        raise RuntimeError("One or more GNINA increment jobs failed integrity")
    evidence = pd.concat(frames, ignore_index=True).sort_values(
        ["incremental_queue_rank", "pairId"], kind="mergesort"
    )
    if len(evidence) != len(queue):
        raise RuntimeError("GNINA increment evidence does not cover the frozen task")
    integrity_path = OUT / "GNINA_REMOTE_INCREMENT_RUN_INTEGRITY_V1.csv"
    evidence_path = OUT / "GNINA_REMOTE_INCREMENT_PAIR_EVIDENCE_V1.csv"
    integrity.to_csv(integrity_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "targets": int(evidence["target_chembl_id"].nunique()),
        "candidate_pairs": len(evidence),
        "supported_pairs": int(evidence["gnina_remote_support"].sum()),
        "tier_counts": evidence["gnina_remote_evidence_tier"].value_counts().to_dict(),
        "outputs": {"integrity": str(integrity_path), "pair_evidence": str(evidence_path)},
    }
    (OUT / "GNINA_REMOTE_INCREMENT_EVALUATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
