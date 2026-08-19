#!/usr/bin/env python3
"""Run the frozen OFER-DTI Phase-A variant matrix.

The runner deliberately keeps model selection inside each historical window's
source-event validation split.  Development-window labels are only collected
after a variant has finished training, then compared side-by-side without
retuning.  Full 40-epoch execution is intentionally opt-in; a small
``--max-train-rows`` screen is useful for checking the matrix before opening
the expensive runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from train_biomaster_ofer_dti_phase_a import train


WINDOWS = {
    "DEV_2015_2018": (2015, 2018, 2014),
    "DEV_2019_2022": (2019, 2022, 2018),
}
VARIANTS = [
    "TARGET_PRIOR",
    "STATIC_OBSERVED_ONLY_BCE",
    "STATIC_FNML_STYLE",
    "DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD",
    "TIMESTAMP_SHUFFLED_OFER",
    "OFER_FULL",
]


def make_args(args: argparse.Namespace, window: str, variant: str) -> SimpleNamespace:
    dev_start, dev_end, cutoff = WINDOWS[window]
    return SimpleNamespace(
        window=window,
        variant=variant,
        dev_start=dev_start,
        dev_end=dev_end,
        cutoff_year=cutoff,
        epochs=args.epochs,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        observation_weight=args.observation_weight,
        active_weight=args.active_weight,
        drug_rank_weight=args.drug_rank_weight,
        target_rank_weight=args.target_rank_weight,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        out_dir=str(Path(args.out_dir) / variant),
        cpu=args.cpu,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="DEV_2015_2018,DEV_2019_2022")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--observation-weight", type=float, default=1.0)
    parser.add_argument("--active-weight", type=float, default=1.0)
    parser.add_argument("--drug-rank-weight", type=float, default=0.25)
    parser.add_argument("--target-rank-weight", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--out-dir", default="outputs/old_drug_target_sota_v1/ofer_dti_phase_a_suite_v1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    windows = [value.strip() for value in args.windows.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown_windows = sorted(set(windows) - set(WINDOWS))
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if unknown_windows or unknown_variants:
        raise ValueError(f"unknown windows={unknown_windows}, variants={unknown_variants}")
    tasks = [(window, variant) for window in windows for variant in variants]
    if args.plan_only:
        print(json.dumps({"status": "PLAN_ONLY", "tasks": tasks}, ensure_ascii=False, indent=2))
        return

    records: list[dict[str, object]] = []
    for window, variant in tasks:
        result = train(make_args(args, window, variant))
        row = {
            "window": window,
            "variant": variant,
            **result.get("metrics", {}),
            "status": result.get("status"),
            "claim_status": result.get("claim_status"),
        }
        records.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(out_dir / "PHASE_A_VARIANT_COMPARISON.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "PASS" for row in records) else "FAIL",
        "windows": windows,
        "variants": variants,
        "records": records,
        "selection_policy": "no development-label checkpoint selection; compare only after each window run",
        "claim_status": "PHASE_A_VARIANT_MATRIX; NO_PROSPECTIVE_OR_SOTA_CLAIM",
    }
    (out_dir / "PHASE_A_VARIANT_COMPARISON_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
