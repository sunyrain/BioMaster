#!/usr/bin/env python3
"""Paired S3/S5 screen for the local atom--pocket graph residual."""

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
OLD_STRUCTURE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
NEW_STRUCTURE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz"
LOCAL_GRAPH = ROOT / "outputs/biomaster_odti_local_graph_features_v1"
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_local_graph_screen_v1"

VARIANTS = {
    "E0_GLOBAL": {
        "structure": OLD_STRUCTURE, "dim": 19, "encoder": "flat",
        "interaction": "legacy_full", "enhanced": False, "local": False,
    },
    "GLOBAL_UPGRADE": {
        "structure": NEW_STRUCTURE, "dim": 47, "encoder": "grouped",
        "interaction": "low_rank_film", "enhanced": True, "local": False,
    },
    "LOCAL_PAIR_ONLY": {
        "structure": OLD_STRUCTURE, "dim": 19, "encoder": "flat",
        "interaction": "legacy_full", "enhanced": False, "local": True,
    },
    "FULL_LOCAL_UPGRADE": {
        "structure": NEW_STRUCTURE, "dim": 47, "encoder": "grouped",
        "interaction": "low_rank_film", "enhanced": True, "local": True,
    },
}


def run_one(name, settings, protocol, fold, seed, epochs, batch_size, out_dir):
    variant_dir = out_dir / name
    run_name = f"{protocol}__fold_{fold}__seed_{seed}"
    summary_path = variant_dir / run_name / "RUN_SUMMARY_V2.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "PASS":
            return summary
    command = [
        sys.executable, str(TRAINER), "--protocol", protocol,
        "--fold", str(fold), "--seed", str(seed), "--epochs", str(epochs),
        "--patience", str(epochs), "--batch-size", str(batch_size),
        "--inference-batch-size", str(batch_size),
        "--target-aux-features", str(TARGET_AUX), "--target-aux-dim", "1280",
        "--structure-features", str(settings["structure"]),
        "--structure-dim", str(settings["dim"]),
        "--structure-encoder", str(settings["encoder"]),
        "--interaction-mode", str(settings["interaction"]),
        "--interaction-rank", "48", "--film-scale", "0.10",
        "--structure-gate-init-bias", "-4.0", "--cache-dense-features",
        "--out-dir", str(variant_dir),
    ]
    if settings["enhanced"]:
        command.append("--enhanced-structure-interaction")
    if settings["local"]:
        command.extend([
            "--local-graph-features", str(LOCAL_GRAPH),
            "--local-pair-hidden-dim", "96", "--local-pair-layers", "2",
            "--local-pair-heads", "4", "--local-pair-gate-init-bias", "-4.0",
        ])
    log = out_dir / f"{name}__{protocol}__fold_{fold}__seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wt") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"run failed; inspect {log}")
    return json.loads(summary_path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocols", default="S3_STRICT_DOUBLE_COLD,S5_OLD_DRUG_ENTITY_COLD")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--s3-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for path in [TRAINER, TARGET_AUX, OLD_STRUCTURE, NEW_STRUCTURE, LOCAL_GRAPH / "LOCAL_GRAPH_FEATURE_MANIFEST_V1.json"]:
        if not path.exists(): raise FileNotFoundError(path)
    rows=[]
    for protocol in [x.strip() for x in args.protocols.split(",") if x.strip()]:
        fold = -1 if protocol == "S5_OLD_DRUG_ENTITY_COLD" else args.s3_fold
        for name, settings in VARIANTS.items():
            summary = run_one(name, settings, protocol, fold, args.seed, args.epochs, args.batch_size, out_dir)
            metric = summary["test_metrics"]
            rows.append({"protocol":protocol,"fold":fold,"seed":args.seed,"variant":name,
                         "micro_auprc":metric["micro_auprc"],"micro_auroc":metric["micro_auroc"],
                         "target_macro_auprc":metric["target_macro_auprc"],"drug_macro_auprc":metric["drug_macro_auprc"],
                         "brier":metric["brier"],"ece_15":metric["ece_15"],
                         "parameters":summary["interaction"]["trainable_parameters"],
                         "local_enabled":summary["interaction"]["local_pair_enabled"]})
    bases={r["protocol"]:r for r in rows if r["variant"]=="E0_GLOBAL"}
    for row in rows:
        base=bases[row["protocol"]]
        for metric in ["micro_auprc","target_macro_auprc","drug_macro_auprc","brier","ece_15"]:
            row[f"delta_{metric}"]=row[metric]-base[metric]
    payload={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS",
             "claim_status":"EXPLORATORY_PAIRED_SHORT_SCREEN_NOT_PROMOTION_EVIDENCE",
             "epochs":args.epochs,"seed":args.seed,"results":rows}
    path=out_dir/"LOCAL_GRAPH_SCREEN_SUMMARY_V1.json"
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
