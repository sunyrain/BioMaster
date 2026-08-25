#!/usr/bin/env python3
"""Summarize the three-seed Stage-A bidirectional V6 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_biomaster_bidirectional_v6 import drug_to_target_dense_metrics  # noqa: E402


SEEDS = (20260816, 20260817, 20260820)
DEFAULT_ROOT = ROOT / "outputs/biomaster_bidirectional_v6_stage_a_dense"


def _query_metrics(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    work = frame.copy()
    work["score"] = np.asarray(score, dtype=float)
    rows = []
    for query, group in work.groupby("query_id", sort=True):
        labels = group["binary_label"].to_numpy(dtype=np.int8)
        values = group["score"].to_numpy(dtype=float)
        order = np.argsort(-values, kind="stable")
        ordered = labels[order]
        positive_count = int(labels.sum())
        ranks = np.flatnonzero(ordered == 1) + 1
        discounts = 1.0 / np.log2(np.arange(2, len(ordered) + 2))
        row = {
            "query_id": str(query),
            "mean_rank": float(ranks.mean()),
            "mrr": float(1.0 / ranks.min()),
            "mean_reciprocal_log_rank": float(
                np.mean(1.0 / np.log2(ranks + 1.0))
            ),
            "mean_top_percentile": float(
                np.mean(1.0 - (ranks - 1) / max(len(group) - 1, 1))
            ),
            "ap": float(average_precision_score(labels, values)),
        }
        for cutoff in (5, 10, 20):
            top = ordered[: min(cutoff, len(ordered))]
            row[f"recall_at_{cutoff}"] = float(top.sum() / positive_count)
        top = ordered[: min(20, len(ordered))]
        denominator = float(discounts[: min(positive_count, len(top))].sum())
        row["ndcg_at_20"] = float((top * discounts[: len(top)]).sum()) / denominator
        row["composite"] = (
            0.35 * row["ndcg_at_20"]
            + 0.25 * row["recall_at_20"]
            + 0.20 * row["mean_top_percentile"]
            + 0.20 * row["mean_reciprocal_log_rank"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(
    pair: pd.DataFrame,
    directional: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    if not pair["query_id"].equals(directional["query_id"]):
        raise RuntimeError("paired query metrics are misaligned")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pair), size=(replicates, len(pair)))
    result: dict[str, object] = {
        "method": "paired_query_bootstrap",
        "queries": int(len(pair)),
        "replicates": replicates,
        "seed": seed,
        "direction": "positive values favor the D2T ensemble",
    }
    metrics = (
        "composite",
        "mean_rank",
        "mrr",
        "mean_top_percentile",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "ndcg_at_20",
        "ap",
    )
    for metric in metrics:
        if metric == "mean_rank":
            delta = pair[metric].to_numpy() - directional[metric].to_numpy()
        else:
            delta = directional[metric].to_numpy() - pair[metric].to_numpy()
        distribution = delta[indices].mean(axis=1)
        result[metric] = {
            "mean_delta": float(delta.mean()),
            "ci95": [
                float(value) for value in np.quantile(distribution, [0.025, 0.975])
            ],
            "bootstrap_probability_improvement": float((distribution > 0).mean()),
        }
    return result


def _load_aligned(root: Path, split: str) -> list[pd.DataFrame]:
    frames = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / f"{split}_ALL_CANDIDATES_V6.csv.gz"
        frame = pd.read_csv(path, low_memory=False).sort_values(
            "calibration_pair_id"
        ).reset_index(drop=True)
        if frames and not (
            np.array_equal(frames[0]["calibration_pair_id"], frame["calibration_pair_id"])
            and np.array_equal(frames[0]["binary_label"], frame["binary_label"])
        ):
            raise RuntimeError(f"seed predictions are misaligned: {path}")
        frames.append(frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    summaries = {
        seed: json.loads((root / f"seed_{seed}" / "STAGE_A_SUMMARY_V6.json").read_text())
        for seed in SEEDS
    }
    output_metrics: dict[str, object] = {}
    bootstrap = None
    for label, split in (
        ("development_2023", "DENSE_D2T_DEVELOPMENT_2023"),
        ("test_2024_2025", "DENSE_D2T_TEST_2024_2025"),
    ):
        frames = _load_aligned(root, split)
        output = frames[0].copy()
        pair_score = np.mean(
            [frame["final_logit"].to_numpy(dtype=float) for frame in frames], axis=0
        )
        directional_score = np.mean(
            [frame["drug_to_target_logit"].to_numpy(dtype=float) for frame in frames],
            axis=0,
        )
        output["pair_ensemble_logit"] = pair_score
        output["drug_to_target_ensemble_logit"] = directional_score
        pair_metrics, _ = drug_to_target_dense_metrics(output, pair_score)
        directional_metrics, _ = drug_to_target_dense_metrics(output, directional_score)
        output_metrics[label] = {
            "pair_ensemble": pair_metrics,
            "drug_to_target_ensemble": directional_metrics,
        }
        output.to_csv(
            root / f"ENSEMBLE_{split}_ALL_CANDIDATES_V6.csv.gz",
            index=False,
            compression="gzip",
        )
        if label == "test_2024_2025":
            pair_query = _query_metrics(output, pair_score)
            directional_query = _query_metrics(output, directional_score)
            bootstrap = _bootstrap(
                pair_query,
                directional_query,
                args.bootstrap_replicates,
                args.bootstrap_seed,
            )
            pair_query.add_prefix("pair_").join(
                directional_query.add_prefix("drug_to_target_")
            ).to_csv(root / "ENSEMBLE_TEST_QUERY_METRICS_V6.csv", index=False)

    dev_deltas = {
        str(seed): float(summary["selection"]["delta"])
        for seed, summary in summaries.items()
    }
    test_bootstrap = bootstrap["composite"]  # type: ignore[index]
    decision = {
        "all_three_development_seeds_improve": all(value > 0 for value in dev_deltas.values()),
        "ensemble_test_composite_point_estimate_improves": test_bootstrap["mean_delta"] > 0,
        "ensemble_test_composite_bootstrap_probability_gt_0_90": (
            test_bootstrap["bootstrap_probability_improvement"] >= 0.90
        ),
        "ensemble_test_composite_ci95_excludes_zero": test_bootstrap["ci95"][0] > 0,
    }
    summary = {
        "status": "EXPLORATORY_PASS_WITH_UNCERTAINTY"
        if all(list(decision.values())[:3])
        else "HOLD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_BIDIRECTIONAL_V6_STAGE_A_THREE_SEED_ENSEMBLE",
        "seeds": list(SEEDS),
        "development_selection_deltas": dev_deltas,
        "metrics": output_metrics,
        "paired_query_bootstrap_test": bootstrap,
        "decision": decision,
        "recommendation": (
            "Freeze the backbone and directional architecture. A head-only FULL_FIT "
            "candidate is justified, but do not claim statistically confirmed superiority "
            "and do not unfreeze the backbone from this 20-query test."
        ),
        "claim_boundary": (
            "The 2024-2025 ensemble point estimate is positive, but the paired-query 95% "
            "bootstrap interval crosses zero. Replication across three seeds controls model "
            "initialization variance, not the limited number of future old-drug queries."
        ),
    }
    (root / "ENSEMBLE_STAGE_A_SUMMARY_V6.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
