#!/usr/bin/env python3
"""Formal paired warm-start audit for the local graph residual.

For each frozen protocol/seed, a global-upgrade baseline is trained first.
The local-pair model is then initialized from that baseline checkpoint, with
the base trunk frozen for a short warm-up before joint fine-tuning.  This
isolates the local branch from a second random reinitialization of the 10M
parameter global trunk while preserving train/validation-only selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_biomaster_odti_v2.py"
TARGET_AUX = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
STRUCTURE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz"
LOCAL_GRAPH = ROOT / "outputs/biomaster_odti_local_graph_features_v1"
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_local_graph_formal_v1"


def command_base(protocol: str, fold: int, seed: int, epochs: int, patience: int, out_dir: Path) -> list[str]:
    return [
        sys.executable, str(TRAINER), "--protocol", protocol, "--fold", str(fold),
        "--seed", str(seed), "--epochs", str(epochs), "--patience", str(patience),
        "--batch-size", "512", "--inference-batch-size", "512",
        "--target-aux-features", str(TARGET_AUX), "--target-aux-dim", "1280",
        "--structure-features", str(STRUCTURE), "--structure-dim", "47",
        "--structure-encoder", "grouped", "--enhanced-structure-interaction",
        "--interaction-mode", "low_rank_film", "--interaction-rank", "48",
        "--film-scale", "0.10", "--structure-gate-init-bias", "-4.0",
        "--cache-dense-features", "--out-dir", str(out_dir),
    ]


def execute(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wt") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"training command failed; inspect {log}")


def load_summary(root: Path, protocol: str, fold: int, seed: int) -> dict[str, object]:
    path = root / f"{protocol}__fold_{fold}__seed_{seed}" / "RUN_SUMMARY_V2.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"run is not PASS: {path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", default="S2_HOMOLOGY_COLD_TARGET,S3_STRICT_DOUBLE_COLD,S5_OLD_DRUG_ENTITY_COLD")
    parser.add_argument("--seeds", default="20260819,20260820")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--freeze-base-epochs", type=int, default=4)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for path in [TRAINER, TARGET_AUX, STRUCTURE, LOCAL_GRAPH / "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json"]:
        if not path.exists(): raise FileNotFoundError(path)
    protocols = [x.strip() for x in args.protocols.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    frozen_folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        protocol_folds = [-1] if protocol in {
            "S4_FIRST_SEEN_TEMPORAL_2023_2025", "S5_OLD_DRUG_ENTITY_COLD"
        } else frozen_folds
        for fold in protocol_folds:
         for seed in seeds:
          baseline_dir = out_dir / "GLOBAL_UPGRADE"
          candidate_dir = out_dir / "FULL_LOCAL_WARMSTART"
          baseline_summary_path = baseline_dir / f"{protocol}__fold_{fold}__seed_{seed}" / "RUN_SUMMARY_V2.json"
          if not baseline_summary_path.is_file():
                execute(
                    command_base(protocol, fold, seed, args.epochs, args.patience, baseline_dir),
                    out_dir / f"GLOBAL_UPGRADE__{protocol}__{seed}.log",
                )
          baseline = load_summary(baseline_dir, protocol, fold, seed)
          baseline_checkpoint = baseline_dir / f"{protocol}__fold_{fold}__seed_{seed}" / "BEST_MODEL_V2.pt"
          candidate_summary_path = candidate_dir / f"{protocol}__fold_{fold}__seed_{seed}" / "RUN_SUMMARY_V2.json"
          if not candidate_summary_path.is_file():
                command = command_base(protocol, fold, seed, args.epochs, args.patience, candidate_dir)
                command.extend([
                    "--local-graph-features", str(LOCAL_GRAPH),
                    "--local-pair-hidden-dim", "96", "--local-pair-layers", "2",
                    "--local-pair-heads", "4", "--local-pair-gate-init-bias", "-4.0",
                    "--init-checkpoint", str(baseline_checkpoint),
                    "--freeze-base-epochs", str(args.freeze_base_epochs),
                ])
                execute(
                    command,
                    out_dir / f"FULL_LOCAL_WARMSTART__{protocol}__{seed}.log",
                )
          candidate = load_summary(candidate_dir, protocol, fold, seed)
          b = baseline["test_metrics"]; c = candidate["test_metrics"]
          row = {
                "protocol": protocol, "fold": fold, "seed": seed,
                "baseline_micro_auprc": b["micro_auprc"],
                "candidate_micro_auprc": c["micro_auprc"],
                "delta_micro_auprc": c["micro_auprc"] - b["micro_auprc"],
                "baseline_target_macro_auprc": b["target_macro_auprc"],
                "candidate_target_macro_auprc": c["target_macro_auprc"],
                "delta_target_macro_auprc": c["target_macro_auprc"] - b["target_macro_auprc"],
                "baseline_drug_macro_auprc": b["drug_macro_auprc"],
                "candidate_drug_macro_auprc": c["drug_macro_auprc"],
                "delta_drug_macro_auprc": c["drug_macro_auprc"] - b["drug_macro_auprc"],
                "delta_brier": c["brier"] - b["brier"],
                "delta_ece_15": c["ece_15"] - b["ece_15"],
                "baseline_parameters": baseline["interaction"]["trainable_parameters"],
                "candidate_parameters": candidate["interaction"]["trainable_parameters"],
                "candidate_local_only_parameter_count": candidate["training"]["local_only_parameter_count"],
                "candidate_init_loaded": candidate["training"]["init_checkpoint"]["loaded"],
                "candidate_freeze_base_epochs": candidate["training"]["freeze_base_epochs"],
            }
          rows.append(row)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "claim_status": "FORMAL_PAIRED_WARMSTART_CANDIDATE_NOT_PROMOTION_UNTIL_CLUSTER_BOOTSTRAP_EXTERNAL_GATES",
        "epochs": args.epochs, "patience": args.patience,
        "freeze_base_epochs": args.freeze_base_epochs,
        "protocols": protocols, "seeds": seeds, "folds": frozen_folds,
        "results": rows,
    }
    path = out_dir / "LOCAL_GRAPH_FORMAL_SUMMARY_V1.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
