from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


STAGE6_CSV = Path("outputs/report_scale/stage6_top1000_consensus_candidates.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def latest_audit_path(root: Path) -> Path:
    dated = sorted((root / "outputs/report_scale").glob("biomaster_status_audit_20*.json"))
    if dated:
        return dated[-1]
    fallback = root / "outputs/report_scale/biomaster_status_audit_latest.json"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No BioMaster status audit JSON found.")


def number(value: str | None) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def build_candidate(row: dict[str, str]) -> dict[str, Any]:
    return {
        "rank": int(row["stage6_rank"]),
        "pairId": row["pair_id"],
        "drugId": row["drug_id"],
        "drug": row["drug_name"],
        "target": row["gene_name"],
        "protein": row["protein_id"],
        "proteinName": row["protein_name"],
        "aiScore": number(row["combined_ai_score"]),
        "diseasePriority": number(row["stage5_final_priority_score"]),
        "directDisease": number(row["direct_disease_score"]),
        "networkDisease": number(row["network_disease_score"]),
        "txgnn": number(row["txgnn_indication_score"]),
        "diffdock": number(row["diffdock_confidence"]),
        "consensus": number(row["stage6_consensus_score"]),
        "status": row["structural_status"],
        "receptorStatus": row["diffdock_receptor_status"],
        "evidenceStatus": row["disease_evidence_status"],
        "txgnnStatus": row["txgnn_component_status"],
    }


def build_payload(root: Path) -> dict[str, Any]:
    stage6_rows = read_csv(root / STAGE6_CSV)
    audit = load_json(latest_audit_path(root))

    target_counts = Counter(row["gene_name"] for row in stage6_rows)
    status_counts = Counter(row["structural_status"] for row in stage6_rows)
    receptor_counts = Counter(row["diffdock_receptor_status"] for row in stage6_rows)
    txgnn_counts = Counter(row["txgnn_component_status"] for row in stage6_rows)

    return {
        "updated": audit["created_utc"],
        "metrics": {
            "drugs": 915,
            "targets": 1000,
            "pairs": 915000,
            "structureCandidates": 1000,
            "top1000Completed": status_counts.get("completed", 0),
            "top1000Missing": status_counts.get("missing_output", 0),
            "fullScoreFiles": audit["full_diffdock"]["score_files"],
            "fullJobsTotal": audit["full_diffdock"]["jobs_total"],
            "fullRowsScored": audit["full_diffdock"]["rows_scored"],
            "fullRowsTotal": audit["full_diffdock"]["rows_total_diffdock_ready"],
            "fullRowProgressPct": audit["full_diffdock"]["row_progress_pct"],
            "fullCompletedOutputs": audit["full_diffdock"]["completed_outputs"],
            "fullMissingOutputs": audit["full_diffdock"]["missing_outputs"],
            "fullOutputRatePct": audit["full_diffdock"]["success_rate_among_scored_pct"],
            "zeroCompletedChunks": audit["full_diffdock"]["zero_completed_chunks"],
        },
        "charts": {
            "evidenceCoverage": [
                {"label": "direct + network", "value": 666120},
                {"label": "direct only", "value": 234240},
                {"label": "network only", "value": 915},
                {"label": "none", "value": 13725},
            ],
            "topTargets": [{"label": label, "value": value} for label, value in target_counts.most_common(10)],
            "structuralStatus": [{"label": label, "value": value} for label, value in status_counts.items()],
            "receptorStatus": [{"label": label, "value": value} for label, value in receptor_counts.items()],
            "txgnnStatus": [{"label": label, "value": value} for label, value in txgnn_counts.items()],
        },
        "candidates": [build_candidate(row) for row in stage6_rows[:30]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build static dashboard assets for GitHub Pages.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="docs/assets/dashboard-data.js")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload(root)
    out_path.write_text(
        "window.BIOMASTER_DATA = "
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out_path), "candidates": len(payload["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
