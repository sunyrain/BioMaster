#!/usr/bin/env python3
"""Build a compact, provenance-aware V2 evidence matrix for the briefing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
ESM2_HASH = "879a351e2574e3a159697496a2a2245c2eedabc30acd18d7816838157db6cd52"
DIRS = {
    "S1": ROOT / "outputs/odti_unified_champion_s1_20260817",
    "S2": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s2_esm2_full",
    "S3": ROOT / "outputs/odti_unified_champion_s3_20260817",
    "S4": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s4_esm2_formal",
    "S5": ROOT / "outputs/old_drug_target_sota_v1/biomaster_odti_v2_s5_esm2_formal",
}


def row_for(name: str, path: Path) -> dict[str, object]:
    summaries = sorted(path.glob("*/RUN_SUMMARY_V2.json")) if path.is_dir() else []
    checkpoints = sorted(path.glob("*/BEST_MODEL_V2.pt")) if path.is_dir() else []
    aggregate_path = path / "V2_MULTI_SEED_SUMMARY.json"
    if not summaries:
        return {
            "role": name,
            "path": str(path),
            "status": "NOT_STARTED",
            "runs": 0,
            "aggregate_present": False,
            "unified_compatible": False,
            "claim_status": "NOT_STARTED",
        }
    first = json.loads(summaries[0].read_text())
    target_aux = first.get("target_auxiliary", {})
    structure = first.get("structure", {})
    drug_aux = first.get("drug_auxiliary", {})
    config = {}
    if checkpoints:
        config = torch.load(checkpoints[0], map_location="cpu", weights_only=False).get("config", {})
    metrics = {}
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text())
        if aggregate.get("aggregates"):
            metrics = aggregate["aggregates"][0].get("metric", {})
    else:
        values = []
        for summary_path in summaries:
            values.append(json.loads(summary_path.read_text()).get("test_metrics", {}))
        for key in ["micro_auprc", "target_macro_auprc", "drug_macro_auprc", "ece_15"]:
            numeric = [float(item[key]) for item in values if item.get(key) is not None]
            if numeric:
                metrics[key] = sum(numeric) / len(numeric)
    compatible = (
        int(structure.get("feature_dim", -1)) == 19
        and target_aux.get("sha256") in {ESM2_HASH, None}
        and not bool(drug_aux.get("enabled", False))
        and int(config.get("expert_count", 6)) == 6
        and int(config.get("structure_input_dim", 19)) == 19
        and int(config.get("target_aux_input_dim", 1280)) == 1280
    )
    return {
        "role": name,
        "path": str(path),
        "protocol": first.get("protocol"),
        "status": "COMPLETE" if aggregate_path.is_file() else "PARTIAL",
        "runs": len(summaries),
        "aggregate_present": aggregate_path.is_file(),
        "structure_dim": structure.get("feature_dim"),
        "structure_source": structure.get("source"),
        "target_aux_enabled": target_aux.get("enabled"),
        "target_aux_sha256": target_aux.get("sha256"),
        "drug_aux_enabled": drug_aux.get("enabled", False),
        "target_token_enabled": first.get("target_token_auxiliary", {}).get("enabled", False),
        "expert_count": config.get("expert_count"),
        "micro_auprc": metrics.get("micro_auprc"),
        "target_macro_auprc": metrics.get("target_macro_auprc"),
        "drug_macro_auprc": metrics.get("drug_macro_auprc"),
        "ece_15": metrics.get("ece_15"),
        "unified_compatible": compatible,
        "claim_status": "INTERNAL_ARTIFACT_EVIDENCE; S1/S3 aggregate or external/W1 gates may remain",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/biomaster_odti_v2_evidence_matrix_v1")
    args = parser.parse_args()
    rows = [row_for(name, path) for name, path in DIRS.items()]
    frame = pd.DataFrame(rows)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "BIOMASTER_ODTI_V2_EVIDENCE_MATRIX_V1.csv", index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "rows": rows,
        "claim_status": "EVIDENCE_MATRIX; NOT_A_SOTA_OR_PROSPECTIVE_CLAIM",
    }
    (output / "BIOMASTER_ODTI_V2_EVIDENCE_MATRIX_V1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
