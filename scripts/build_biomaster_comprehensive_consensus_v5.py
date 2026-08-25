#!/usr/bin/env python3
"""Build the frozen six-checkpoint comprehensive deployment consensus V5."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomaster.comprehensive_balanced import fuse_percentile_scores  # noqa: E402


SEEDS = (20260816, 20260817, 20260820)
CONSERVATIVE = ROOT / (
    "outputs/biomaster_comprehensive_balanced_full_fit_720x384_v3/"
    "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
)
CONSERVATIVE_SUMMARY = CONSERVATIVE.parent / "AUGMENTED_FULL_FIT_720X384_SUMMARY_V1.json"
RECALL = ROOT / (
    "outputs/biomaster_comprehensive_binary_selected_720x384_v4/"
    "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
)
RECALL_CASES = RECALL.parent / "CURRENT_DOUBLE_WARM_CASE_RANKS_V1.csv"
RECALL_SUMMARY = RECALL.parent / "AUGMENTED_FULL_FIT_720X384_SUMMARY_V1.json"
CALIBRATION = ROOT / (
    "outputs/biomaster_comprehensive_consensus_fusion_calibration_v5/"
    "FUSION_CALIBRATION_SUMMARY_V2.json"
)
OUT = ROOT / "outputs/biomaster_comprehensive_consensus_720x384_v5"

CASES = (
    ("PLK4", "lorlatinib", "IIXWYSCJSQVBQM-LLVKDONJSA-N"),
    ("IDO1", "pitavastatin", "VGYFMXBACGZSIL-MCBHFWOFSA-N"),
    ("IRAK1", "tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N"),
    ("IRAK4", "tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [
        str(path) for path in (
            CONSERVATIVE,
            CONSERVATIVE_SUMMARY,
            RECALL,
            RECALL_CASES,
            RECALL_SUMMARY,
            CALIBRATION,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(missing)
    calibration = json.loads(CALIBRATION.read_text())
    family_summaries = {
        "conservative_v3": json.loads(CONSERVATIVE_SUMMARY.read_text()),
        "recall_v4": json.loads(RECALL_SUMMARY.read_text()),
    }
    strategy = str(calibration["selected_fusion_strategy"])
    affinity_weight = float(calibration["selected_affinity_weight"])
    if strategy != "top20_refine" or affinity_weight != 0.45:
        raise RuntimeError("frozen V5 fusion contract changed")

    conservative = pd.read_csv(CONSERVATIVE, low_memory=False)
    recall = pd.read_csv(RECALL, low_memory=False)
    recall_cases = pd.read_csv(RECALL_CASES, low_memory=False).set_index(
        ["target_chembl_id", "drug_inchikey"]
    )
    if len(conservative) != 720 * 384 or len(recall) != len(conservative):
        raise RuntimeError("source score matrices are not 720x384")
    if not np.array_equal(
        conservative["pairId"].astype(str).to_numpy(),
        recall["pairId"].astype(str).to_numpy(),
    ):
        raise RuntimeError("source score matrices are not pair-aligned")

    frame = recall.copy()
    binary_percentiles = []
    affinity_percentiles = []
    for family, source in (("conservative", conservative), ("recall", recall)):
        family_binary = []
        family_affinity = []
        for seed in SEEDS:
            binary = source.groupby("target_chembl_id", sort=False)[f"binary_{seed}"].rank(
                method="average", pct=True
            ).to_numpy(dtype=np.float64)
            affinity = source.groupby("target_chembl_id", sort=False)[f"affinity_{seed}"].rank(
                method="average", pct=True
            ).to_numpy(dtype=np.float64)
            binary_percentiles.append(binary)
            affinity_percentiles.append(affinity)
            family_binary.append(binary)
            family_affinity.append(affinity)
        family_score = np.mean(np.stack(family_binary), axis=0)
        family_rank = pd.Series(family_score).groupby(
            frame["target_chembl_id"], sort=False
        ).rank(method="min", ascending=False).astype(int)
        frame[f"{family}_binary_score"] = family_score
        frame[f"{family}_binary_rank"] = family_rank.to_numpy()

    frame["consensus_binary_percentile"] = np.mean(
        np.stack(binary_percentiles), axis=0
    )
    frame["consensus_affinity_percentile"] = np.mean(
        np.stack(affinity_percentiles), axis=0
    )
    frame["ensemble_fusion_score"] = fuse_percentile_scores(
        frame["consensus_binary_percentile"].to_numpy(),
        frame["consensus_affinity_percentile"].to_numpy(),
        strategy,
        affinity_weight,
    )
    frame["ensemble_rank_within_target_720"] = frame.groupby(
        "target_chembl_id", sort=False
    )["ensemble_fusion_score"].rank(method="min", ascending=False).astype(int)
    frame["ensemble_percentile_top"] = (
        1.0 - (frame["ensemble_rank_within_target_720"] - 1.0) / 720.0
    )

    cases = []
    for gene, drug_name, inchikey in CASES:
        matched = frame.loc[
            frame["gene_symbol"].eq(gene) & frame["ligand_inchikey"].eq(inchikey)
        ]
        if len(matched) != 1:
            raise RuntimeError(f"case does not map exactly once: {gene}, {drug_name}")
        row = matched.iloc[0]
        audit = recall_cases.loc[(str(row["target_chembl_id"]), inchikey)]
        cases.append({
            "gene_symbol": gene,
            "target_chembl_id": str(row["target_chembl_id"]),
            "drug_name": drug_name,
            "drug_inchikey": inchikey,
            "rank_within_target_720": int(row["ensemble_rank_within_target_720"]),
            "percentile_top": float(row["ensemble_percentile_top"]),
            "conservative_binary_rank": int(row["conservative_binary_rank"]),
            "recall_binary_rank": int(row["recall_binary_rank"]),
            "structure_mask": int(row["structure_mask"]),
            "cold_start_class": str(audit["cold_start_class"]),
            "exact_relation_in_full_fit": bool(audit["exact_relation_in_full_fit"]),
            "exact_positive_relation_in_full_fit": bool(
                audit["exact_positive_relation_in_full_fit"]
            ),
        })
    cases_frame = pd.DataFrame(cases).sort_values("rank_within_target_720")

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_families = {}
    for family, family_summary in family_summaries.items():
        checkpoint_rows = []
        for value in family_summary["checkpoints"]:
            checkpoint_path = ROOT / value
            run_summary_path = checkpoint_path.parent / "FULL_FIT_RUN_SUMMARY_V2.json"
            run_summary = json.loads(run_summary_path.read_text())
            checkpoint_rows.append({
                "seed": int(run_summary["seed"]),
                "selected_epoch": int(run_summary["selection"]["best_epoch"]),
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint_path),
                "run_summary": str(run_summary_path.relative_to(ROOT)),
                "run_summary_sha256": sha256(run_summary_path),
            })
        checkpoint_families[family] = {
            "family_weight": 0.5,
            "checkpoints": checkpoint_rows,
        }
    checkpoint_manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "COMPREHENSIVE_FULL_FIT_SIX_CHECKPOINT_MANIFEST_V5",
        "aggregation": "equal mean of target-wise rank percentiles across both families and all seeds",
        "families": checkpoint_families,
    }
    checkpoint_manifest_path = OUT / "FULL_FIT_CHECKPOINT_MANIFEST_V5.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2) + "\n"
    )
    scores_path = OUT / "CONSENSUS_FULL_FIT_720X384_SCORES_V5.csv.gz"
    cases_path = OUT / "CURRENT_DOUBLE_WARM_CASE_RANKS_V5.csv"
    frame.to_csv(scores_path, index=False, compression="gzip")
    cases_frame.to_csv(cases_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "COMPREHENSIVE_FULL_FIT_SIX_CHECKPOINT_CONSENSUS_V5",
        "ranking_universe": "720 frozen old drugs independently within each of 384 targets",
        "model_family_weights": {"conservative_v3": 0.5, "recall_v4": 0.5},
        "checkpoint_seeds_per_family": list(SEEDS),
        "fusion": {
            "strategy": strategy,
            "binary_shortlist_top_fraction": 0.20,
            "within_shortlist_binary_weight": 1.0 - affinity_weight,
            "within_shortlist_affinity_weight": affinity_weight,
        },
        "selection_current_external_case_labels_used": False,
        "validation": {
            "development_selection": calibration["selected_development_metrics"],
            "test_after_fusion_freeze": calibration["test_metrics_after_weight_freeze"],
            "selection_reason": calibration["selection_reason"],
            "temporal_test_labels_used_for_selection": calibration[
                "selection_2024_2025_temporal_test_labels_used"
            ],
        },
        "sources": {
            str(CONSERVATIVE.relative_to(ROOT)): sha256(CONSERVATIVE),
            str(CONSERVATIVE_SUMMARY.relative_to(ROOT)): sha256(CONSERVATIVE_SUMMARY),
            str(RECALL.relative_to(ROOT)): sha256(RECALL),
            str(RECALL_CASES.relative_to(ROOT)): sha256(RECALL_CASES),
            str(RECALL_SUMMARY.relative_to(ROOT)): sha256(RECALL_SUMMARY),
            str(CALIBRATION.relative_to(ROOT)): sha256(CALIBRATION),
        },
        "cases": cases,
        "artifacts": {
            "scores": str(scores_path.relative_to(ROOT)),
            "scores_sha256": sha256(scores_path),
            "case_ranks": str(cases_path.relative_to(ROOT)),
            "case_ranks_sha256": sha256(cases_path),
            "checkpoint_manifest": str(checkpoint_manifest_path.relative_to(ROOT)),
            "checkpoint_manifest_sha256": sha256(checkpoint_manifest_path),
        },
    }
    summary_path = OUT / "CONSENSUS_FULL_FIT_720X384_SUMMARY_V5.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(cases_frame.to_string(index=False))
    print(json.dumps({"status": "PASS", "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
