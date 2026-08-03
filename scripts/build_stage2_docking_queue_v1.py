#!/usr/bin/env python3
"""Build a 30k multi-lane docking queue from remote ConPLEx and DrugCLIP evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "affinity_first_remote_discovery_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(BASE / "drugclip_inference_v1/REMOTE_STRICT_DTA_DRUGCLIP_SCORED_V1.csv.gz"),
    )
    parser.add_argument("--master", default=str(BASE / "PHYSICAL_PAIR_UNIVERSE_334749_HOMOLOGY_AUDITED_V1.csv.gz"))
    parser.add_argument("--output-dir", default=str(BASE / "stage2_docking_queue_v1"))
    parser.add_argument("--size", type=int, default=30000)
    parser.add_argument("--target-cap", type=int, default=130)
    parser.add_argument("--ligand-cap", type=int, default=70)
    parser.add_argument("--scaffold-cap", type=int, default=300)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    master = pd.read_csv(
        args.master,
        usecols=[
            "physical_pair_id",
            "murcko_scaffold",
            "pdb_path",
            "top_pocket_center_x",
            "top_pocket_center_y",
            "top_pocket_center_z",
            "top_pocket_residue_ids",
            "source_indications",
            "source_therapeutic_areas",
        ],
        low_memory=False,
    )
    data = data.merge(master, on="physical_pair_id", how="left", validate="one_to_one")
    data = data[data["drugclip_completed_v1"].fillna(False).astype(bool)].copy()

    cp_ligand = (1.0 - (pd.to_numeric(data["rank_within_active_moiety"], errors="coerce") - 1.0) / 462.0).clip(0, 1)
    cp_target = (1.0 - (pd.to_numeric(data["rank_within_target_active_collapsed"], errors="coerce") - 1.0) / 722.0).clip(0, 1)
    dc_ligand = (1.0 - (pd.to_numeric(data["drugclip_rank_within_ligand_v1"], errors="coerce") - 1.0) / 306.0).clip(0, 1)
    dc_target = (1.0 - (pd.to_numeric(data["drugclip_rank_within_target_v1"], errors="coerce") - 1.0) / 721.0).clip(0, 1)
    data["conplex_bidirectional_percentile_v1"] = np.sqrt(cp_ligand * cp_target)
    data["drugclip_bidirectional_percentile_v1"] = np.sqrt(dc_ligand * dc_target)
    data["orthogonal_consensus_score_v1"] = 0.5 * (
        data["conplex_bidirectional_percentile_v1"] + data["drugclip_bidirectional_percentile_v1"]
    )

    cp50 = data["rank_within_active_moiety"].le(50) & data["rank_within_target_active_collapsed"].le(50)
    cp100 = data["rank_within_active_moiety"].le(100) & data["rank_within_target_active_collapsed"].le(100)
    dc50 = data["drugclip_rank_within_ligand_v1"].le(50) & data["drugclip_rank_within_target_v1"].le(50)
    dc100 = data["drugclip_rank_within_ligand_v1"].le(100) & data["drugclip_rank_within_target_v1"].le(100)
    data["stage2_evidence_lane_v1"] = "single_model_or_exploration"
    data.loc[cp100 & dc100, "stage2_evidence_lane_v1"] = "broad_two_model_consensus"
    data.loc[cp50 & ~dc50, "stage2_evidence_lane_v1"] = "conplex_bidirectional_top50"
    data.loc[dc50 & ~cp50, "stage2_evidence_lane_v1"] = "drugclip_bidirectional_top50"
    data.loc[cp50 & dc50, "stage2_evidence_lane_v1"] = "strong_two_model_consensus"
    lane_order = {
        "strong_two_model_consensus": 0,
        "broad_two_model_consensus": 1,
        "drugclip_bidirectional_top50": 2,
        "conplex_bidirectional_top50": 3,
        "single_model_or_exploration": 4,
    }
    data["stage2_lane_order_v1"] = data["stage2_evidence_lane_v1"].map(lane_order)
    data = data.sort_values(
        ["stage2_lane_order_v1", "orthogonal_consensus_score_v1", "drugclip_cosine_std_v1"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    selected_indices: list[int] = []
    selected_set: set[int] = set()
    target_counts: Counter[str] = Counter()
    ligand_counts: Counter[str] = Counter()
    scaffold_counts: Counter[str] = Counter()

    def accept(index: int, coverage_pass: bool = False) -> bool:
        if index in selected_set:
            return False
        row = data.loc[index]
        target = str(row["sequence_key"])
        ligand = str(row["model_ligand_smiles"])
        scaffold = str(row.get("murcko_scaffold") or "NO_SCAFFOLD")
        if not coverage_pass:
            if target_counts[target] >= args.target_cap:
                return False
            if ligand_counts[ligand] >= args.ligand_cap:
                return False
            if scaffold_counts[scaffold] >= args.scaffold_cap:
                return False
        selected_indices.append(index)
        selected_set.add(index)
        target_counts[target] += 1
        ligand_counts[ligand] += 1
        scaffold_counts[scaffold] += 1
        return True

    # Preserve broad target coverage with each target's ten strongest
    # orthogonal hypotheses before filling by evidence lanes.
    coverage = data.sort_values(
        ["sequence_key", "orthogonal_consensus_score_v1"], ascending=[True, False], kind="mergesort"
    ).groupby("sequence_key", sort=False).head(10)
    for index in coverage.index:
        accept(index, coverage_pass=True)
    for index in data.index:
        if len(selected_indices) >= args.size:
            break
        accept(index)
    if len(selected_indices) != args.size:
        raise RuntimeError(f"Could select only {len(selected_indices)} of {args.size} rows")

    selected = data.loc[selected_indices].copy()
    selected = selected.sort_values(
        ["stage2_lane_order_v1", "orthogonal_consensus_score_v1"],
        ascending=[True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    selected["stage2_queue_rank_v1"] = np.arange(1, len(selected) + 1)
    selected["requires_target_calibration_before_interpretation_v1"] = True
    selected["docking_result_is_not_affinity_fact_v1"] = True

    output = output_dir / "GNINA_DISCOVERY_QUEUE_30000_V1.csv.gz"
    selected.to_csv(output, index=False, compression={"method": "gzip", "compresslevel": 5})
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_remote_scored_rows": int(len(data)),
        "selected_rows": int(len(selected)),
        "unique_ligands": int(selected["model_ligand_smiles"].nunique()),
        "unique_targets": int(selected["sequence_key"].nunique()),
        "unique_scaffolds": int(selected["murcko_scaffold"].nunique()),
        "assay_family_counts": selected["target_assay_family_v2"].value_counts().to_dict(),
        "evidence_lane_counts": selected["stage2_evidence_lane_v1"].value_counts().to_dict(),
        "max_rows_per_target": int(selected["sequence_key"].value_counts().max()),
        "max_rows_per_ligand": int(selected["model_ligand_smiles"].value_counts().max()),
        "policy": (
            "The queue is a compute allocation, not a final ranking. It preserves target coverage, "
            "uses ConPLEx and DrugCLIP as orthogonal retrieval channels, and requires target-wise "
            "positive/negative docking calibration before discovery scores are interpreted."
        ),
        "output": str(output.relative_to(ROOT)),
    }
    (output_dir / "STAGE2_DOCKING_QUEUE_V1_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
