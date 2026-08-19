#!/usr/bin/env python3
"""Run a paired four-cell screen for structure and interaction upgrades.

This is an exploratory, same-budget screen.  It deliberately separates the
47-D grouped pocket representation from the low-rank FiLM/tri-linear pair
interaction so a joint result remains attributable.
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
STRUCTURE_V1 = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
STRUCTURE_V2 = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz"
TARGET_AUX = ROOT / "outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_structure_interaction_screen_v2"


VARIANTS = {
    "E0_LEGACY": {
        "structure": STRUCTURE_V1,
        "dim": 19,
        "encoder": "flat",
        "interaction": "legacy_full",
        "enhanced_structure": False,
        "structure_gate_init_bias": None,
    },
    "STRUCTURE_V2_ONLY": {
        "structure": STRUCTURE_V2,
        "dim": 47,
        "encoder": "grouped",
        "interaction": "legacy_full",
        "enhanced_structure": False,
        "structure_gate_init_bias": -4.0,
    },
    "INTERACTION_V2_ONLY": {
        "structure": STRUCTURE_V1,
        "dim": 19,
        "encoder": "flat",
        "interaction": "low_rank_film",
        "enhanced_structure": True,
        "structure_gate_init_bias": None,
    },
    "STRUCTURE_INTERACTION_V2": {
        "structure": STRUCTURE_V2,
        "dim": 47,
        "encoder": "grouped",
        "interaction": "low_rank_film",
        "enhanced_structure": True,
        "structure_gate_init_bias": -4.0,
    },
}


def run_one(
    variant: str,
    settings: dict[str, object],
    protocol: str,
    fold: int,
    seed: int,
    epochs: int,
    batch_size: int,
    out_dir: Path,
) -> dict[str, object]:
    variant_dir = out_dir / variant
    run_name = f"{protocol}__fold_{fold}__seed_{seed}"
    summary_path = variant_dir / run_name / "RUN_SUMMARY_V2.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "PASS":
            return summary
    command = [
        sys.executable, str(TRAINER),
        "--protocol", protocol,
        "--fold", str(fold),
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--patience", str(epochs),
        "--batch-size", str(batch_size),
        "--inference-batch-size", str(max(batch_size * 2, 1024)),
        "--target-aux-features", str(TARGET_AUX),
        "--target-aux-dim", "1280",
        "--structure-features", str(settings["structure"]),
        "--structure-dim", str(settings["dim"]),
        "--structure-encoder", str(settings["encoder"]),
        "--interaction-mode", str(settings["interaction"]),
        "--interaction-rank", "48",
        "--film-scale", "0.10",
        "--cache-dense-features",
        "--out-dir", str(variant_dir),
    ]
    if settings["structure_gate_init_bias"] is not None:
        command.extend(
            ["--structure-gate-init-bias", str(settings["structure_gate_init_bias"])]
        )
    if bool(settings["enhanced_structure"]):
        command.append("--enhanced-structure-interaction")
    log_path = out_dir / f"{variant}__{protocol}__fold_{fold}__seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wt") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not summary_path.is_file():
        raise RuntimeError(f"screen run failed; inspect {log_path}")
    return json.loads(summary_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocols",
        default="S3_STRICT_DOUBLE_COLD,S5_OLD_DRUG_ENTITY_COLD",
    )
    parser.add_argument("--s3-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [TRAINER, STRUCTURE_V1, STRUCTURE_V2, TARGET_AUX]:
        if not path.is_file():
            raise FileNotFoundError(path)

    protocols = [value.strip() for value in args.protocols.split(",") if value.strip()]
    rows: list[dict[str, object]] = []
    for protocol in protocols:
        fold = -1 if protocol == "S5_OLD_DRUG_ENTITY_COLD" else args.s3_fold
        for variant, settings in VARIANTS.items():
            summary = run_one(
                variant, settings, protocol, fold, args.seed,
                args.epochs, args.batch_size, out_dir,
            )
            metric = summary["test_metrics"]
            rows.append({
                "protocol": protocol,
                "fold": fold,
                "seed": args.seed,
                "variant": variant,
                "micro_auprc": metric["micro_auprc"],
                "micro_auroc": metric["micro_auroc"],
                "target_macro_auprc": metric["target_macro_auprc"],
                "drug_macro_auprc": metric["drug_macro_auprc"],
                "brier": metric["brier"],
                "ece_15": metric["ece_15"],
                "best_epoch": summary["training"]["best_epoch"],
                "parameters": summary["interaction"]["trainable_parameters"],
            })

    baselines = {
        row["protocol"]: row for row in rows if row["variant"] == "E0_LEGACY"
    }
    for row in rows:
        baseline = baselines[row["protocol"]]
        row["delta_micro_auprc"] = row["micro_auprc"] - baseline["micro_auprc"]
        row["delta_target_macro_auprc"] = (
            row["target_macro_auprc"] - baseline["target_macro_auprc"]
        )
        row["delta_drug_macro_auprc"] = (
            row["drug_macro_auprc"] - baseline["drug_macro_auprc"]
        )
        row["delta_brier"] = row["brier"] - baseline["brier"]
        row["delta_ece_15"] = row["ece_15"] - baseline["ece_15"]

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "claim_status": "EXPLORATORY_PAIRED_SHORT_SCREEN_NOT_PROMOTION_EVIDENCE",
        "epochs": args.epochs,
        "seed": args.seed,
        "protocols": protocols,
        "variants": {
            name: {
                key: str(value) if isinstance(value, Path) else value
                for key, value in settings.items()
            }
            for name, settings in VARIANTS.items()
        },
        "results": rows,
    }
    summary_path = out_dir / "STRUCTURE_INTERACTION_SCREEN_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
