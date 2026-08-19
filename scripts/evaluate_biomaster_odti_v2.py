#!/usr/bin/env python3
"""Run and aggregate the strengthened BioMaster-ODTI V2 suite.

This runner deliberately uses the frozen V1 split definitions.  It does not
change labels, folds, or test-role selection.  For each protocol/fold/seed it
calls the V2 trainer, then averages pair-aligned calibrated predictions across
seeds.  S1--S3 are pooled OOF-style evaluations; S4/S5 are fixed test roles.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_biomaster_odti_v2 import train
from run_biomaster_odti_baselines_v1 import metrics


PROTOCOLS = [
    "S1_SCAFFOLD_COLD_DRUG",
    "S2_HOMOLOGY_COLD_TARGET",
    "S3_STRICT_DOUBLE_COLD",
    "S4_FIRST_SEEN_TEMPORAL_2023_2025",
    "S5_OLD_DRUG_ENTITY_COLD",
]
SEEDS = [20260816, 20260817, 20260818, 20260819, 20260820]


def task_folds(protocol: str) -> list[int]:
    return [-1] if protocol in {
        "S4_FIRST_SEEN_TEMPORAL_2023_2025",
        "S5_OLD_DRUG_ENTITY_COLD",
    } else list(range(5))


def parse_seeds(value: str) -> list[int]:
    if value.strip().lower() == "default":
        return list(SEEDS)
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def progress_state(
    tasks: list[tuple[str, int, int]],
    summaries_by_protocol: dict[str, list[dict[str, object]]],
    resumed_runs: int,
    status: str = "RUNNING",
) -> dict[str, object]:
    """Build a progress manifest with stable tuple keys.

    The old implementation compared tuple tasks to list-shaped completed
    entries, so ``remaining`` appeared unchanged throughout a run.  Keep the
    on-disk JSON list-shaped for readability, but compare normalized tuples.
    """

    completed_tuples = {
        (str(item["protocol"]), int(item["fold"]), int(item["seed"]))
        for values in summaries_by_protocol.values()
        for item in values
    }
    completed = [list(task) for task in tasks if task in completed_tuples]
    remaining = [list(task) for task in tasks if task not in completed_tuples]
    if status == "PASS":
        completed = [list(task) for task in tasks]
        remaining = []
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "tasks": [list(task) for task in tasks],
        "completed": completed,
        "remaining": remaining,
        "resumed_runs": resumed_runs,
    }


def trainer_args(args: argparse.Namespace, protocol: str, fold: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        protocol=protocol,
        fold=fold,
        seed=seed,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=1e-4,
        batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        experts=args.experts,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        drug_rank_weight=args.drug_rank_weight,
        expert_balance_weight=args.expert_balance_weight,
        listwise_weight=args.listwise_weight,
        rank_max_pairs=args.rank_max_pairs,
        affinity_weight=args.affinity_weight,
        observation_weight=args.observation_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        gradient_clip=args.gradient_clip,
        structure_features=args.structure_features,
        structure_dim=args.structure_dim,
        structure_encoder=args.structure_encoder,
        enhanced_structure_interaction=args.enhanced_structure_interaction,
        structure_gate_init_bias=args.structure_gate_init_bias,
        interaction_mode=args.interaction_mode,
        interaction_rank=args.interaction_rank,
        film_scale=args.film_scale,
        local_graph_features=args.local_graph_features,
        local_pair_hidden_dim=args.local_pair_hidden_dim,
        local_pair_layers=args.local_pair_layers,
        local_pair_heads=args.local_pair_heads,
        local_pair_gate_init_bias=args.local_pair_gate_init_bias,
        init_checkpoint=args.init_checkpoint,
        freeze_base_epochs=args.freeze_base_epochs,
        drug_aux_features=args.drug_aux_features,
        drug_aux_dim=args.drug_aux_dim,
        drug_aux_index=args.drug_aux_index,
        drug_aux_gate_init_bias=args.drug_aux_gate_init_bias,
        target_aux_features=args.target_aux_features,
        target_aux_dim=args.target_aux_dim,
        target_aux_gate_init_bias=args.target_aux_gate_init_bias,
        target_extra_features=args.target_extra_features,
        target_extra_dim=args.target_extra_dim,
        target_extra_gate_init_bias=args.target_extra_gate_init_bias,
        target_token_features=args.target_token_features,
        target_token_index=args.target_token_index,
        target_token_pocket_mask=args.target_token_pocket_mask,
        target_token_dim=args.target_token_dim,
        target_token_heads=args.target_token_heads,
        target_token_max_len=args.target_token_max_len,
        use_conplex=args.use_conplex,
        max_rows=args.max_rows,
        selection_metric=args.selection_metric,
        random_batches=args.random_batches,
        batch_sampler=args.batch_sampler,
        max_rows_per_target=args.max_rows_per_target,
        max_rows_per_drug=args.max_rows_per_drug,
        observation_column=args.observation_column,
        out_dir=str(args.out_dir),
        cpu=args.cpu,
        cache_dense_features=args.cache_dense_features,
    )


def aggregate_protocol(
    protocol: str,
    run_summaries: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, object]:
    prediction_frames: list[pd.DataFrame] = []
    for summary in run_summaries:
        prediction_path = Path(str(summary["prediction_path"]))
        frame = pd.read_csv(prediction_path, low_memory=False)
        frame["seed"] = int(summary["seed"])
        frame["fold"] = int(summary["fold"])
        prediction_frames.append(frame)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    key = "calibration_pair_id"
    if all_predictions.duplicated([key, "seed"]).any():
        raise RuntimeError(f"duplicate pair/seed predictions for {protocol}")
    score = "v2_probability_calibrated"
    aggregate = (
        all_predictions.groupby(key, as_index=False)
        .agg(
            binary_label=("binary_label", "first"),
            target_chembl_id=("target_chembl_id", "first"),
            parent_standard_inchi_key=("parent_standard_inchi_key", "first"),
            score_mean=(score, "mean"),
            score_std=(score, "std"),
            seed_count=("seed", "nunique"),
        )
    )
    aggregate["score_std"] = aggregate["score_std"].fillna(0.0)
    metric_frame = aggregate.rename(columns={"score_mean": "score"})
    metric_row = metrics(metric_frame, metric_frame["score"].to_numpy(dtype=np.float64))
    metric_row.update({
        "protocol": protocol,
        "evaluation": "V2_SEED_MEAN",
        "runs": len(run_summaries),
        "seeds": sorted({int(summary["seed"]) for summary in run_summaries}),
        "folds": sorted({int(summary["fold"]) for summary in run_summaries}),
    })
    prediction_path = out_dir / f"{protocol}_V2_SEED_MEAN_PREDICTIONS.csv.gz"
    aggregate.to_csv(prediction_path, index=False, compression="gzip")
    return {
        "protocol": protocol,
        "metric": metric_row,
        "prediction_path": str(prediction_path),
        "pair_count": int(len(aggregate)),
        "seed_count_min": int(aggregate["seed_count"].min()),
        "seed_count_max": int(aggregate["seed_count"].max()),
    }


def reusable_run(
    out_dir: Path,
    protocol: str,
    fold: int,
    seed: int,
) -> dict[str, object] | None:
    """Return a validated completed-run record for resume mode."""

    run_name = f"{protocol}__fold_{fold}__seed_{seed}"
    run_dir = out_dir / run_name
    summary_path = run_dir / "RUN_SUMMARY_V2.json"
    prediction_path = run_dir / "TEST_PREDICTIONS_V2.csv.gz"
    if not summary_path.is_file() or not prediction_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("status") == "PASS"
        and summary.get("protocol") == protocol
        and int(summary.get("fold", -999)) == fold
        and int(summary.get("seed", -999)) == seed
    ):
        return None
    return {
        "protocol": protocol,
        "fold": fold,
        "seed": seed,
        "status": "PASS",
        "prediction_path": str(prediction_path),
        "summary_path": str(summary_path),
        "resumed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=PROTOCOLS + ["all"], default="all")
    parser.add_argument("--seeds", default="default")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.12)
    parser.add_argument(
        "--drug-rank-weight",
        type=float,
        default=0.0,
        help="optional within-drug pairwise ranking weight for old-drug retrieval",
    )
    parser.add_argument(
        "--expert-balance-weight",
        type=float,
        default=0.0,
        help="optional anti-collapse regularizer for routed expert usage",
    )
    parser.add_argument("--listwise-weight", type=float, default=0.0)
    parser.add_argument("--rank-max-pairs", type=int, default=4096)
    parser.add_argument("--affinity-weight", type=float, default=0.06)
    parser.add_argument("--observation-weight", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--structure-features")
    parser.add_argument("--structure-dim", type=int)
    parser.add_argument(
        "--structure-encoder", choices=["flat", "grouped"], default="flat"
    )
    parser.add_argument("--enhanced-structure-interaction", action="store_true")
    parser.add_argument("--structure-gate-init-bias", type=float, default=None)
    parser.add_argument(
        "--interaction-mode",
        choices=["legacy_full", "low_rank_film"],
        default="legacy_full",
    )
    parser.add_argument("--interaction-rank", type=int, default=48)
    parser.add_argument("--film-scale", type=float, default=0.10)
    parser.add_argument("--local-graph-features", default=None)
    parser.add_argument("--local-pair-hidden-dim", type=int, default=96)
    parser.add_argument("--local-pair-layers", type=int, default=2)
    parser.add_argument("--local-pair-heads", type=int, default=4)
    parser.add_argument("--local-pair-gate-init-bias", type=float, default=-4.0)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--freeze-base-epochs", type=int, default=0)
    parser.add_argument(
        "--drug-aux-features",
        default=None,
        help="optional drug-aligned molecular pretraining feature matrix (.npy)",
    )
    parser.add_argument("--drug-aux-dim", type=int, default=None)
    parser.add_argument(
        "--drug-aux-index",
        default=None,
        help="optional dense availability/provenance index for --drug-aux-features",
    )
    parser.add_argument(
        "--drug-aux-gate-init-bias",
        type=float,
        default=None,
        help="optional molecular residual gate bias; unset preserves zero-input behavior",
    )
    parser.add_argument("--target-aux-features")
    parser.add_argument("--target-aux-dim", type=int, default=None)
    parser.add_argument(
        "--target-aux-gate-init-bias",
        type=float,
        default=None,
        help="optional pooled target-aux residual gate bias; unset preserves audited V2 initialization",
    )
    parser.add_argument("--target-extra-features")
    parser.add_argument("--target-extra-dim", type=int, default=None)
    parser.add_argument("--target-extra-gate-init-bias", type=float, default=-4.0)
    parser.add_argument("--target-token-features")
    parser.add_argument("--target-token-index")
    parser.add_argument("--target-token-pocket-mask")
    parser.add_argument("--target-token-dim", type=int, default=None)
    parser.add_argument("--target-token-heads", type=int, default=4)
    parser.add_argument("--target-token-max-len", type=int, default=1022)
    parser.add_argument("--use-conplex", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        choices=["composite", "micro_auprc", "target_macro_auprc", "drug_macro_auprc"],
        default="composite",
    )
    parser.add_argument(
        "--random-batches",
        action="store_true",
        help="disable the default target-aware minibatch sampler",
    )
    parser.add_argument(
        "--batch-sampler",
        choices=["target", "dual_query"],
        default="target",
        help="target-aware default or dual target+drug query neighborhoods",
    )
    parser.add_argument("--max-rows-per-target", type=int, default=16)
    parser.add_argument("--max-rows-per-drug", type=int, default=16)
    parser.add_argument("--observation-column")
    parser.add_argument("--out-dir", default="outputs/old_drug_target_sota_v1/biomaster_odti_v2")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cache-dense-features",
        action="store_true",
        help="cache dense Morgan/target features on device to reduce formal-run I/O overhead",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="reuse only matching PASS runs with an existing prediction artifact",
    )
    args = parser.parse_args()

    protocols = PROTOCOLS if args.protocol == "all" else [args.protocol]
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(protocol, fold, seed) for protocol in protocols for fold in task_folds(protocol) for seed in seeds]
    if args.plan_only:
        print(json.dumps({"status": "PLAN_ONLY", "tasks": tasks}, ensure_ascii=False, indent=2))
        return

    summaries_by_protocol: dict[str, list[dict[str, object]]] = {protocol: [] for protocol in protocols}
    resumed_runs = 0
    progress_path = out_dir / "RUN_PROGRESS_V2.json"
    progress_path.write_text(
        json.dumps(progress_state(tasks, summaries_by_protocol, resumed_runs), ensure_ascii=False, indent=2)
        + "\n"
    )
    for protocol, fold, seed in tasks:
        reused = reusable_run(out_dir, protocol, fold, seed) if args.resume_existing else None
        if reused is not None:
            summaries_by_protocol[protocol].append(reused)
            resumed_runs += 1
            progress_path.write_text(
                json.dumps(progress_state(tasks, summaries_by_protocol, resumed_runs), ensure_ascii=False, indent=2)
                + "\n"
            )
            continue
        result = train(trainer_args(args, protocol, fold, seed))
        run_name = f"{protocol}__fold_{fold}__seed_{seed}"
        summaries_by_protocol[protocol].append({
            "protocol": protocol,
            "fold": fold,
            "seed": seed,
            "status": result["status"],
            "prediction_path": str(out_dir / run_name / "TEST_PREDICTIONS_V2.csv.gz"),
            "summary_path": str(out_dir / run_name / "RUN_SUMMARY_V2.json"),
            "resumed": False,
        })
        progress_path.write_text(
            json.dumps(progress_state(tasks, summaries_by_protocol, resumed_runs), ensure_ascii=False, indent=2)
            + "\n"
        )

    aggregates = [
        aggregate_protocol(protocol, summaries, out_dir)
        for protocol, summaries in summaries_by_protocol.items()
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["metric"]["rows"] > 0 for item in aggregates) else "FAIL",
        "model": "BIOMASTER_ODTI_ROUTED_INTERACTION_RANKER_V2",
        "protocols": protocols,
        "seeds": seeds,
        "task_count": len(tasks),
        "resumed_runs": resumed_runs,
        "aggregates": aggregates,
        "claim_status": "V2_MULTI_SEED_INTERNAL_EVIDENCE; EXTERNAL_AND_PROSPECTIVE_GATES_REQUIRED",
    }
    progress_path.write_text(
        json.dumps(progress_state(tasks, summaries_by_protocol, resumed_runs, status="PASS"), ensure_ascii=False, indent=2)
        + "\n"
    )
    (out_dir / "V2_MULTI_SEED_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
