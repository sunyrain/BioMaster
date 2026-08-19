#!/usr/bin/env python3
"""Run and summarize a paired MoLFormer/ESM-C residual screen.

The four cells share frozen labels, S3/S5 roles, seeds, optimization budget,
ESM2 champion auxiliary, and structure context:

* E0: current champion inputs;
* MoLFormer: E0 + drug-side MoLFormer residual;
* ESM-C: E0 + a separately gated ESM-C target residual;
* both: both new residuals together.

The default three-epoch, one-seed run is exploratory.  It is designed to reject
weak candidates cheaply, not to promote a new champion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_biomaster_odti_v2.py"
STORE = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1"
FROZEN_PAIRS = STORE / "CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
STRUCTURE = STORE / "ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
ESM2 = (
    ROOT
    / "outputs/old_drug_target_sota_v1/public_retrained_v1/"
    "dtiam_official_feature_store_v1/DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy"
)
MOLFORMER = (
    ROOT
    / "outputs/biomaster_odti_pretrained_features_v1/molformer_xl_both_10pct/"
    "MOLFORMER_XL_768_FLOAT32_V1.npy"
)
MOLFORMER_INDEX = (
    ROOT
    / "outputs/biomaster_odti_pretrained_features_v1/molformer_xl_both_10pct/"
    "MOLFORMER_XL_DRUG_INDEX_V1.csv.gz"
)
ESMC = (
    ROOT
    / "outputs/biomaster_odti_pretrained_features_v1/esmc_600m/"
    "ESMC_600M_1152_FLOAT32_V1.npy"
)
DEFAULT_OUT = ROOT / "outputs/biomaster_odti_pretrained_residual_screen_v1"


CANDIDATES: dict[str, dict[str, bool]] = {
    "e0": {"molformer": False, "esmc": False},
    "molformer": {"molformer": True, "esmc": False},
    "esmc": {"molformer": False, "esmc": True},
    "both": {"molformer": True, "esmc": True},
}
PROTOCOLS = ["S3_STRICT_DOUBLE_COLD", "S5_OLD_DRUG_ENTITY_COLD"]


def run_cell(
    output_root: Path,
    candidate: str,
    protocol: str,
    args: argparse.Namespace,
) -> Path:
    short = "s3" if protocol.startswith("S3_") else "s5"
    out_dir = output_root / f"{candidate}_{short}"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(EVALUATOR),
        "--protocol",
        protocol,
        "--seeds",
        args.seeds,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--structure-features",
        str(STRUCTURE),
        "--structure-dim",
        "19",
        "--target-aux-features",
        str(ESM2),
        "--target-aux-dim",
        "1280",
        "--cache-dense-features",
        "--out-dir",
        str(out_dir),
    ]
    cell = CANDIDATES[candidate]
    if cell["molformer"]:
        command.extend(
            [
                "--drug-aux-features",
                str(MOLFORMER),
                "--drug-aux-index",
                str(MOLFORMER_INDEX),
                "--drug-aux-dim",
                "768",
                "--drug-aux-gate-init-bias",
                str(args.residual_gate_bias),
            ]
        )
    if cell["esmc"]:
        command.extend(
            [
                "--target-extra-features",
                str(ESMC),
                "--target-extra-dim",
                "1152",
                "--target-extra-gate-init-bias",
                str(args.residual_gate_bias),
            ]
        )
    if args.resume_existing:
        command.append("--resume-existing")
    log_path = out_dir / "SCREEN_EXECUTION.log"
    print(f"running {candidate}/{protocol} -> {out_dir}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary_path = out_dir / "V2_MULTI_SEED_SUMMARY.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing suite summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "PASS":
        raise RuntimeError(f"suite did not pass: {summary_path}")
    return summary_path


def prediction_path(summary_path: Path) -> Path:
    summary = json.loads(summary_path.read_text())
    aggregates = summary.get("aggregates", [])
    if len(aggregates) != 1:
        raise RuntimeError(f"expected one aggregate in {summary_path}")
    return Path(aggregates[0]["prediction_path"])


def metric_row(summary_path: Path, candidate: str) -> dict[str, object]:
    summary = json.loads(summary_path.read_text())
    aggregate = summary["aggregates"][0]
    return {"candidate": candidate, **aggregate["metric"]}


def paired_predictions(left: Path, right: Path) -> pd.DataFrame:
    columns = ["calibration_pair_id", "binary_label", "score_mean"]
    left_frame = pd.read_csv(left, usecols=columns).rename(
        columns={"binary_label": "label_left", "score_mean": "score_left"}
    )
    right_frame = pd.read_csv(right, usecols=columns).rename(
        columns={"binary_label": "label_right", "score_mean": "score_right"}
    )
    frame = left_frame.merge(right_frame, on="calibration_pair_id", validate="one_to_one")
    if not frame["label_left"].eq(frame["label_right"]).all():
        raise RuntimeError("paired model outputs disagree on frozen labels")
    frozen = pd.read_csv(
        FROZEN_PAIRS,
        usecols=["calibration_pair_id", "target_homology_cluster", "scaffold_group"],
        low_memory=False,
    )
    return frame.merge(frozen, on="calibration_pair_id", validate="one_to_one")


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_column: str,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    cluster_values = frame[cluster_column].astype(str).to_numpy()
    clusters = pd.unique(cluster_values)
    members = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)
    deltas_ap: list[float] = []
    deltas_roc: list[float] = []
    for _ in range(repeats):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        selected = np.concatenate([members[cluster] for cluster in sampled])
        part = frame.iloc[selected]
        labels = part["label_left"].to_numpy(dtype=np.int8)
        if labels.min() == labels.max():
            continue
        deltas_ap.append(
            float(
                average_precision_score(labels, part["score_left"])
                - average_precision_score(labels, part["score_right"])
            )
        )
        deltas_roc.append(
            float(
                roc_auc_score(labels, part["score_left"])
                - roc_auc_score(labels, part["score_right"])
            )
        )
    rows: list[dict[str, object]] = []
    for metric, values in [("micro_auprc", deltas_ap), ("micro_auroc", deltas_roc)]:
        array = np.asarray(values, dtype=np.float64)
        low, high = np.quantile(array, [0.025, 0.975])
        rows.append(
            {
                "cluster_column": cluster_column,
                "metric": metric,
                "bootstrap_replicates": int(len(array)),
                "mean_delta_candidate_minus_e0": float(array.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return rows


def summarize(
    output_root: Path,
    summaries: dict[tuple[str, str], Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    metric_rows = [
        metric_row(path, candidate)
        for (candidate, _), path in summaries.items()
    ]
    metrics = pd.DataFrame(metric_rows)
    metrics_path = output_root / "PRETRAINED_RESIDUAL_SCREEN_METRICS_V1.csv"
    metrics.to_csv(metrics_path, index=False)
    deltas: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    delta_columns = [
        "micro_auprc",
        "micro_auroc",
        "target_macro_auprc",
        "drug_macro_auprc",
        "ece_15",
    ]
    for protocol in PROTOCOLS:
        baseline = metrics[(metrics["candidate"] == "e0") & (metrics["protocol"] == protocol)].iloc[0]
        base_prediction = prediction_path(summaries[("e0", protocol)])
        for candidate in ("molformer", "esmc", "both"):
            row = metrics[
                (metrics["candidate"] == candidate) & (metrics["protocol"] == protocol)
            ].iloc[0]
            delta = {"protocol": protocol, "candidate": candidate}
            for column in delta_columns:
                delta[f"{column}_delta"] = float(row[column] - baseline[column])
            deltas.append(delta)
            frame = paired_predictions(
                prediction_path(summaries[(candidate, protocol)]), base_prediction
            )
            for offset, cluster in enumerate(
                ["target_homology_cluster", "scaffold_group"]
            ):
                for item in cluster_bootstrap(
                    frame,
                    cluster,
                    args.bootstrap_repeats,
                    args.bootstrap_seed + offset,
                ):
                    bootstrap_rows.append(
                        {"protocol": protocol, "candidate": candidate, **item}
                    )
    delta_path = output_root / "PRETRAINED_RESIDUAL_SCREEN_DELTAS_V1.csv"
    bootstrap_path = output_root / "PRETRAINED_RESIDUAL_SCREEN_BOOTSTRAP_V1.csv"
    pd.DataFrame(deltas).to_csv(delta_path, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "protocol": "PAIRED_MOLFORMER_ESMC_RESIDUAL_SCREEN_V1",
        "screen_level": (
            "EXPLORATORY_SHORT_BUDGET"
            if args.epochs < 40 or len(args.seeds.split(",")) < 2
            else "CONTROLLED_TWO_SEED_SCREEN"
        ),
        "cells": CANDIDATES,
        "training": {
            "protocols": PROTOCOLS,
            "seeds": args.seeds,
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "residual_gate_bias": float(args.residual_gate_bias),
        },
        "deltas_candidate_minus_e0": deltas,
        "artifacts": {
            "metrics": str(metrics_path),
            "deltas": str(delta_path),
            "bootstrap": str(bootstrap_path),
        },
        "claim_status": "EXPLORATORY_PAIRED_INTERNAL_SCREEN; NO_CHAMPION_PROMOTION",
    }
    summary_path = output_root / "PRETRAINED_RESIDUAL_SCREEN_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seeds", default="20260816")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--residual-gate-bias", type=float, default=-4.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    required = [FROZEN_PAIRS, STRUCTURE, ESM2, MOLFORMER, MOLFORMER_INDEX, ESMC]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    summaries: dict[tuple[str, str], Path] = {}
    for candidate in CANDIDATES:
        for protocol in PROTOCOLS:
            short = "s3" if protocol.startswith("S3_") else "s5"
            expected = output_root / f"{candidate}_{short}/V2_MULTI_SEED_SUMMARY.json"
            if args.summarize_only:
                if not expected.is_file():
                    raise FileNotFoundError(expected)
                summaries[(candidate, protocol)] = expected
            else:
                summaries[(candidate, protocol)] = run_cell(
                    output_root, candidate, protocol, args
                )
    summarize(output_root, summaries, args)


if __name__ == "__main__":
    main()
