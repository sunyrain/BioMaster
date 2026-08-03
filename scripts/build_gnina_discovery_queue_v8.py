#!/usr/bin/env python3
"""Build the frozen GNINA discovery queue from target-admitted remote pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/affinity_experiment_package_v8.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {key: ROOT / value for key, value in config["inputs"].items()}
    base = ROOT / config["outputs"]["directory"]
    calibration_path = base / "target_calibration/GNINA_TARGET_CHANNEL_CALIBRATION_V8.csv"
    output_dir = base / "discovery_queue"
    output_dir.mkdir(parents=True, exist_ok=True)

    queue = pd.read_csv(inputs["stage2_queue"], low_memory=False)
    calibration = pd.read_csv(calibration_path, low_memory=False)
    receptors = pd.read_csv(inputs["receptors"], low_memory=False)
    admitted_tiers = set(config["discovery_docking"]["include_admission_tiers"])
    target_columns = [
        "sequence_key",
        "target_admission_tier_v8",
        "target_admitted_v8",
        "cnn_affinity_pass_v8",
        "cnn_affinity_strong_v8",
        "vina_affinity_pass_v8",
        "vina_affinity_strong_v8",
        "auroc_cnn_affinity_v8",
        "average_precision_cnn_affinity_v8",
        "auroc_vina_affinity_v8",
        "average_precision_vina_affinity_v8",
    ]
    receptor_columns = [
        "sequence_key",
        "docking_receptor_source",
        "docking_receptor_path",
        "selected_pdb_id",
        "box_center_x",
        "box_center_y",
        "box_center_z",
        "box_size_x",
        "box_size_y",
        "box_size_z",
        "selection_status",
    ]
    queue = queue.merge(
        calibration[target_columns], on="sequence_key", how="left", validate="many_to_one"
    ).merge(receptors[receptor_columns], on="sequence_key", how="left", validate="many_to_one")
    queue = queue[queue["target_admission_tier_v8"].isin(admitted_tiers)].copy()
    tier_order = {
        "T1_dual_strong": 0,
        "T2_dual_pass": 1,
        "T3_single_strong": 2,
        "T4_single_pass": 3,
    }
    queue["target_admission_order_v8"] = queue["target_admission_tier_v8"].map(tier_order)
    queue["experimental_holo_order_v8"] = ~queue["docking_receptor_source"].eq(
        "experimental_holo"
    )
    queue = queue.sort_values(
        [
            "target_admission_order_v8",
            "experimental_holo_order_v8",
            "stage2_lane_order_v1",
            "stage2_queue_rank_v1",
            "physical_pair_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    queue.insert(0, "gnina_discovery_queue_rank_v8", range(1, len(queue) + 1))
    queue["gnina_discovery_interpretation_v8"] = (
        "Raw scores must be converted to within-target control percentiles before selection."
    )
    output = output_dir / "GNINA_REMOTE_DISCOVERY_QUEUE_V8.csv.gz"
    queue.to_csv(output, index=False, compression={"method": "gzip", "compresslevel": 5})
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_queue_rows": 30000,
        "selected_rows": int(len(queue)),
        "unique_ligands": int(queue["model_ligand_smiles"].nunique()),
        "unique_targets": int(queue["sequence_key"].nunique()),
        "unique_scaffolds": int(queue["murcko_scaffold"].nunique()),
        "target_admission_tier_counts": queue["target_admission_tier_v8"].value_counts().to_dict(),
        "receptor_source_counts": queue["docking_receptor_source"].value_counts().to_dict(),
        "assay_family_counts": queue["target_assay_family_v2"].value_counts().to_dict(),
        "policy": {
            "pair_universe": "strict remote chemistry and homology-audited pairs only",
            "target_admission": "CNNaffinity or Vina channel passes same-target positive/negative calibration",
            "ranking": "compute allocation only; no cross-model weighted affinity score",
        },
        "output": str(output.relative_to(ROOT)),
    }
    summary_path = output_dir / "GNINA_REMOTE_DISCOVERY_QUEUE_V8_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
