#!/usr/bin/env python3
"""Train and smoke-test the observation-aware OFER-DTI Phase-A model.

This entry point intentionally starts with the frozen ``OFER_FULL`` objective
only.  It consumes the precomputed, leakage-audited risk sets and never turns
pre-event survival rows into biochemical inactive labels.  Additional frozen
variants can be added after this implementation passes the smoke and
development gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.ofer_dti import OFERDTIConfig, OFERDTIModel, ofer_discovery_score, ofer_phase_a_loss  # noqa: E402


BASE = ROOT / "outputs/old_drug_target_sota_v1"
FEATURE = BASE / "first_event_dti_feature_store_v1"
RISK = BASE / "ofer_dti_phase_a_risk_sets_v1"
MORGAN = FEATURE / "OFER_DTI_MORGAN2048_PACKED_UINT8_V1.npy"
TARGET = BASE / "feature_store_v1/PROTBERT1024_FLOAT32_V1.npy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack_morgan(packed: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(packed[indices], dtype=np.uint8)
    return np.unpackbits(selected, axis=1, bitorder="big").astype(np.float32)


def grouped_metrics(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float | int | None]:
    score = np.asarray(score, dtype=np.float64)
    labels = frame["active_event_by_horizon"].to_numpy(dtype=np.int8)

    def grouped_ap(column: str) -> float | None:
        values: list[float] = []
        work = frame.assign(__score=score)
        for _, part in work.groupby(column, sort=False):
            y = part["active_event_by_horizon"].to_numpy(dtype=np.int8)
            if y.min() == y.max():
                continue
            values.append(float(average_precision_score(y, part["__score"])))
        return float(np.mean(values)) if values else None

    drug_ap = grouped_ap("drug_feature_index")
    target_ap = grouped_ap("target_feature_index")
    recall_values: list[float] = []
    for _, part in frame.assign(__score=score).groupby("drug_feature_index", sort=False):
        y = part["active_event_by_horizon"].to_numpy(dtype=np.int8)
        if y.sum() == 0:
            continue
        ordered = part.sort_values("__score", ascending=False)
        recall_values.append(
            float(ordered.head(20)["active_event_by_horizon"].sum() / y.sum())
        )
    return {
        "rows": int(len(frame)),
        "positives": int(labels.sum()),
        "micro_auprc": float(average_precision_score(labels, score)) if labels.any() else None,
        "drug_macro_auprc": drug_ap,
        "target_macro_auprc": target_ap,
        "drug_macro_recall_at_20": float(np.mean(recall_values)) if recall_values else None,
    }


def target_prior_predictions(
    risk: np.lib.npyio.NpzFile,
    evaluation: pd.DataFrame,
) -> np.ndarray:
    """Historical exact-event target prior with Laplace smoothing."""

    role = risk["sampling_role"]
    observed = risk["observation_event"]
    active = risk["active_given_observed"]
    mask = (role == 0) & (observed == 1) & (active >= 0)
    target = risk["target_feature_index"][mask].astype(np.int64)
    labels = active[mask].astype(np.float64)
    positives = np.bincount(target, weights=labels, minlength=428)
    counts = np.bincount(target, minlength=428)
    prior = (positives + 1.0) / (counts + 2.0)
    return prior[evaluation["target_feature_index"].to_numpy(dtype=np.int64)]


def predict_horizon(
    model: OFERDTIModel,
    frame: pd.DataFrame,
    morgan: np.ndarray,
    target_features: np.ndarray,
    years: list[int],
    device: torch.device,
    batch_size: int,
    variant: str = "OFER_FULL",
    cutoff_year: int | None = None,
) -> np.ndarray:
    model.eval()
    active_logits: list[np.ndarray] = []
    hazards: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        part = frame.iloc[start : start + batch_size]
        compound = part["event_compound_feature_index"].to_numpy(dtype=np.int64)
        target = part["target_feature_index"].to_numpy(dtype=np.int64)
        drug = torch.from_numpy(unpack_morgan(morgan, compound)).to(device)
        target_tensor = torch.from_numpy(np.asarray(target_features[target], dtype=np.float32)).to(device)
        with torch.no_grad():
            base = model.encode_pair(drug, target_tensor)
            year_hazards = []
            scoring_years = years
            if variant in {"STATIC_FNML_STYLE", "STATIC_OBSERVED_ONLY_BCE"}:
                scoring_years = [int(cutoff_year if cutoff_year is not None else years[0])]
            for year in scoring_years:
                output = model(
                    drug,
                    target_tensor,
                    torch.full((len(part),), year, device=device, dtype=torch.float32),
                )
                year_hazards.append(output["observation_logit"])
            active_logits.append(base["active_logit"].cpu().numpy())
            hazards.append(torch.stack(year_hazards, dim=1).cpu().numpy())
    active = torch.from_numpy(np.concatenate(active_logits)).to(device)
    hazard = torch.from_numpy(np.concatenate(hazards, axis=0)).to(device)
    if variant == "STATIC_OBSERVED_ONLY_BCE":
        return active.sigmoid().cpu().numpy()
    if variant == "STATIC_FNML_STYLE":
        return (active.sigmoid() * hazard[:, -1].sigmoid()).cpu().numpy()
    if variant == "DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD":
        direct_logits: list[np.ndarray] = []
        for start in range(0, len(frame), batch_size):
            part = frame.iloc[start : start + batch_size]
            compound = part["event_compound_feature_index"].to_numpy(dtype=np.int64)
            target = part["target_feature_index"].to_numpy(dtype=np.int64)
            drug = torch.from_numpy(unpack_morgan(morgan, compound)).to(device)
            target_tensor = torch.from_numpy(np.asarray(target_features[target], dtype=np.float32)).to(device)
            with torch.no_grad():
                values = []
                for year in years:
                    values.append(
                        model(
                            drug,
                            target_tensor,
                            torch.full((len(part),), year, device=device, dtype=torch.float32),
                        )["direct_active_logit"]
                    )
                direct_logits.append(torch.stack(values, dim=1).cpu().numpy())
        direct = torch.from_numpy(np.concatenate(direct_logits, axis=0)).to(device)
        return (1.0 - torch.cumprod(1.0 - direct.sigmoid(), dim=1)[:, -1]).cpu().numpy()
    return ofer_discovery_score(active, hazard).cpu().numpy()


def validation_risk_loss(
    model: OFERDTIModel,
    positions: np.ndarray,
    risk: np.lib.npyio.NpzFile,
    packed: np.ndarray,
    target_features: np.ndarray,
    config: OFERDTIConfig,
    device: torch.device,
    batch_size: int,
    year_values: np.ndarray,
    variant: str,
) -> float:
    model.eval()
    totals: list[float] = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            index = positions[start : start + batch_size]
            compound = risk["compound_feature_index"][index].astype(np.int64)
            target = risk["target_feature_index"][index].astype(np.int64)
            output = model(
                torch.from_numpy(unpack_morgan(packed, compound)).to(device),
                torch.from_numpy(np.asarray(target_features[target], dtype=np.float32)).to(device),
                torch.from_numpy(year_values[index].astype(np.float32)).to(device),
            )
            losses = ofer_phase_a_loss(
                output,
                torch.from_numpy(risk["observation_event"][index]).to(device),
                torch.from_numpy(risk["active_given_observed"][index]).to(device),
                drug_group=torch.from_numpy(compound).to(device),
                target_group=torch.from_numpy(target).to(device),
                sample_weight=torch.from_numpy(
                    risk["inverse_sampling_probability_weight"][index].astype(np.float32)
                ).to(device),
                sampling_role=torch.from_numpy(risk["sampling_role"][index]).to(device),
                drug_query_group=torch.from_numpy(
                    compound.astype(np.int64) * 4096 + year_values[index].astype(np.int64)
                ).to(device),
                target_query_group=torch.from_numpy(
                    target.astype(np.int64) * 4096 + year_values[index].astype(np.int64)
                ).to(device),
                variant=variant,
                config=config,
            )
            totals.append(float(losses["total"]))
    return float(np.mean(totals))


def train(args: argparse.Namespace) -> dict[str, object]:
    window_dir = RISK / args.window
    train_path = window_dir / "OFER_DTI_TRAIN_RISK_SAMPLES_V1.npz"
    eval_path = window_dir / "OFER_DTI_OLD_DRUG_TARGET_DEVELOPMENT_EVALUATION_V1.csv.gz"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(f"missing frozen risk-set artifacts for {args.window}")
    cutoff_year = int(
        args.cutoff_year
        if args.cutoff_year is not None
        else (2014 if args.window == "DEV_2015_2018" else 2018)
    )
    packed = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(TARGET, mmap_mode="r")
    risk = np.load(train_path)
    rows = len(risk["calendar_year"])
    if args.variant == "TARGET_PRIOR":
        evaluation = pd.read_csv(eval_path, low_memory=False)
        if args.max_eval_rows > 0:
            evaluation = evaluation.iloc[: args.max_eval_rows].copy()
        final_score = target_prior_predictions(risk, evaluation)
        final_metrics = grouped_metrics(evaluation, final_score)
        run_dir = Path(args.out_dir) / args.window
        run_dir.mkdir(parents=True, exist_ok=True)
        scored = evaluation.copy()
        scored["target_prior_score"] = final_score
        scored.to_csv(
            run_dir / "DEVELOPMENT_SCORES_TARGET_PRIOR.csv.gz",
            index=False,
            compression="gzip",
        )
        summary = {
            "status": "PASS",
            "model": "TARGET_PRIOR",
            "window": args.window,
            "device": "none",
            "split_counts": {"risk_rows": int(rows), "development": int(len(evaluation))},
            "training": {
                "epochs": 0,
                "development_labels_used_for_selection": False,
                "historical_exact_event_rows_only": True,
            },
            "metrics": final_metrics,
            "checks": {
                "scores_finite": bool(np.isfinite(final_score).all()),
                "scores_bounded": bool(((final_score >= 0) & (final_score <= 1)).all()),
            },
            "claim_status": "HISTORICAL_TARGET_PRIOR_CONTROL; NO_PROSPECTIVE_OR_SOTA_CLAIM",
        }
        (run_dir / "RUN_SUMMARY_OFER_DTI.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    positions = np.arange(rows, dtype=np.int64)
    if args.max_train_rows > 0:
        positions = positions[: min(rows, args.max_train_rows)]
    source = risk["source_event_index"][positions]
    valid_mask = (source % 10) == 0
    train_positions = positions[~valid_mask]
    valid_positions = positions[valid_mask]
    if len(train_positions) == 0 or len(valid_positions) == 0:
        raise ValueError("risk-set train/valid split is empty")

    config = OFERDTIConfig(
        dropout=args.dropout,
        observation_weight=args.observation_weight,
        active_weight=args.active_weight,
        drug_rank_weight=args.drug_rank_weight,
        target_rank_weight=args.target_rank_weight,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = OFERDTIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    if args.variant == "TIMESTAMP_SHUFFLED_OFER":
        time_rng = np.random.default_rng(args.seed + 100003)
        year_values = risk["calendar_year"].copy()
        year_values[:] = year_values[time_rng.permutation(rows)]
    elif args.variant in {"STATIC_FNML_STYLE", "STATIC_OBSERVED_ONLY_BCE"}:
        year_values = np.full(rows, cutoff_year, dtype=np.int16)
    else:
        year_values = risk["calendar_year"]
    best = np.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_positions.copy()
        rng.shuffle(order)
        total = 0.0
        steps = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            compound = risk["compound_feature_index"][index].astype(np.int64)
            target = risk["target_feature_index"][index].astype(np.int64)
            drug = torch.from_numpy(unpack_morgan(packed, compound)).to(device)
            target_tensor = torch.from_numpy(np.asarray(target_features[target], dtype=np.float32)).to(device)
            year = torch.from_numpy(year_values[index].astype(np.float32)).to(device)
            output = model(drug, target_tensor, year)
            losses = ofer_phase_a_loss(
                output,
                torch.from_numpy(risk["observation_event"][index]).to(device),
                torch.from_numpy(risk["active_given_observed"][index]).to(device),
                drug_group=torch.from_numpy(compound).to(device),
                target_group=torch.from_numpy(target).to(device),
                sample_weight=torch.from_numpy(
                    risk["inverse_sampling_probability_weight"][index].astype(np.float32)
                ).to(device),
                sampling_role=torch.from_numpy(risk["sampling_role"][index]).to(device),
                drug_query_group=torch.from_numpy(
                    compound.astype(np.int64) * 4096 + year_values[index].astype(np.int64)
                ).to(device),
                target_query_group=torch.from_numpy(
                    target.astype(np.int64) * 4096 + year_values[index].astype(np.int64)
                ).to(device),
                variant=args.variant,
                config=config,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total += float(losses["total"].detach())
            steps += 1
        valid_loss = validation_risk_loss(
            model,
            valid_positions,
            risk,
            packed,
            target_features,
            config,
            device,
            args.batch_size,
            year_values,
            args.variant,
        )
        history.append({
            "epoch": epoch,
            "loss": total / max(steps, 1),
            "historical_internal_valid_loss": valid_loss,
        })
        if valid_loss < best:
            best = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation checkpoint")
    model.load_state_dict(best_state)
    evaluation = pd.read_csv(eval_path, low_memory=False)
    if args.max_eval_rows > 0:
        evaluation = evaluation.iloc[: args.max_eval_rows].copy()
    final_score = predict_horizon(
        model,
        evaluation,
        packed,
        target_features,
        list(range(args.dev_start, args.dev_end + 1)),
        device,
        args.inference_batch_size,
        variant=args.variant,
        cutoff_year=cutoff_year,
    )
    final_metrics = grouped_metrics(evaluation, final_score)
    out_dir = Path(args.out_dir)
    run_dir = out_dir / args.window
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "BEST_MODEL_OFER_DTI.pt"
    torch.save({"model_state_dict": best_state, "config": config.__dict__, "window": args.window}, checkpoint_path)
    pd.DataFrame(history).to_csv(run_dir / "TRAINING_HISTORY_OFER_DTI.csv", index=False)
    scored = evaluation.copy()
    scored[f"{args.variant.lower()}_score"] = final_score
    scored.to_csv(run_dir / "DEVELOPMENT_SCORES_OFER_FULL.csv.gz", index=False, compression="gzip")
    summary = {
        "status": "PASS",
        "model": args.variant,
        "window": args.window,
        "device": str(device),
        "split_counts": {"risk_rows": int(rows), "train": int(len(train_positions)), "valid": int(len(valid_positions))},
        "training": {
            "epochs": int(args.epochs),
            "variant": args.variant,
            "cutoff_year": cutoff_year,
            "best_historical_internal_valid_loss": float(best),
            "development_labels_used_for_selection": False,
        },
        "metrics": final_metrics,
        "checks": {
            "risk_set_exists": train_path.is_file(),
            "feature_store_hashable": bool(sha256(MORGAN) and sha256(TARGET)),
            "scores_finite": bool(np.isfinite(final_score).all()),
            "scores_bounded": bool(((final_score >= 0) & (final_score <= 1)).all()),
        },
        "claim_status": "PHASE_A_COMPONENT_SMOKE_OR_DEVELOPMENT; NO_PROSPECTIVE_OR_SOTA_CLAIM",
    }
    (run_dir / "RUN_SUMMARY_OFER_DTI.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=["DEV_2015_2018", "DEV_2019_2022"], default="DEV_2015_2018")
    parser.add_argument(
        "--variant",
        choices=[
            "TARGET_PRIOR",
            "OFER_FULL",
            "STATIC_OBSERVED_ONLY_BCE",
            "STATIC_FNML_STYLE",
            "DIRECT_ACTIVE_SURVIVAL_NO_OBSERVATION_HEAD",
            "TIMESTAMP_SHUFFLED_OFER",
        ],
        default="OFER_FULL",
    )
    parser.add_argument("--dev-start", type=int, default=2015)
    parser.add_argument("--dev-end", type=int, default=2018)
    parser.add_argument("--cutoff-year", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--out-dir", default="outputs/old_drug_target_sota_v1/ofer_dti_phase_a_training_v1")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
