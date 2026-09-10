#!/usr/bin/env python3
"""Score all 720 old drugs against the routed 745-target registry.

The complete-registry rank is emitted only as a diagnostic.  Production ranks
are computed separately for the 450 primary targets, 42 special-system targets,
8 high-novelty assayable targets, and 245 registry-only special targets.
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from score_biomaster_full_fit_current_new_relations_v1 import inference_arrays  # noqa: E402
from train_biomaster_odti_v2 import predict  # noqa: E402


SEEDS = (20260816, 20260817, 20260820)
PAIRS = ROOT / "outputs/target_discovery_scope_ch37_v3/OLD_DRUG_TARGET_REGISTRY_720X745_V3.csv.gz"
DRUG = ROOT / (
    "outputs/old_drug_target_sota_v1/deployment_720x384_feature_store_v1/"
    "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
)
DEFAULT_FEATURE_ROOT = ROOT / "outputs/retrain_20260901/target_registry_745_feature_store_v1"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/retrain_20260901/bidirectional_full_fit"
DEFAULT_OUT = ROOT / "outputs/retrain_20260901/bidirectional_720x745_full_fit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--checkpoint-filename", default="FULL_FIT_BIDIRECTIONAL_V6.pt")
    parser.add_argument("--run-label", default="full_fit_candidate")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    feature_root = Path(args.feature_root).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    out = Path(args.out_dir).resolve()
    target_index_path = feature_root / "TARGET_REGISTRY_745_FEATURE_INDEX_V1.csv.gz"
    target_path = feature_root / "TARGET_REGISTRY_745_PROTBERT1024_FLOAT32_V1.npy"
    target_aux_path = feature_root / "TARGET_REGISTRY_745_ESM2_650M_1280_FLOAT32_V1.npy"
    structure_path = feature_root / "TARGET_REGISTRY_745_STRUCTURE_CONTEXT_19D_V1.csv.gz"
    checkpoints = [
        checkpoint_root / f"seed_{seed}" / args.checkpoint_filename for seed in SEEDS
    ]
    required = [
        PAIRS, DRUG, target_index_path, target_path, target_aux_path, structure_path,
        *checkpoints,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    frame = pd.read_csv(PAIRS, low_memory=False)
    target_index = pd.read_csv(target_index_path, low_memory=False).sort_values(
        "target_feature_index"
    )
    if len(frame) != 720 * 745 or len(target_index) != 745:
        raise RuntimeError("720x745 routed registry contract failed")
    frame["target_feature_index"] = frame["target_registry_index"].astype(np.int64)
    family = target_index.set_index("target_feature_index")["target_assay_family"]
    frame["target_assay_family"] = frame["target_feature_index"].map(family)
    frame["parent_standard_inchi_key"] = frame["ligand_inchikey"].astype(str)
    frame["calibration_pair_id"] = frame["candidate_pair_id"].astype(str)
    frame["binary_label"] = 0
    frame["binary_observed"] = 0
    frame["mean_pchembl"] = np.nan
    frame["min_pchembl"] = np.nan
    frame["max_pchembl"] = np.nan
    frame["conplex_score"] = 0.0

    drug = np.load(DRUG, mmap_mode="r")
    target = np.load(target_path, mmap_mode="r")
    target_aux = np.load(target_aux_path, mmap_mode="r")
    structure_index = pd.read_csv(structure_path, low_memory=False).sort_values(
        "target_feature_index"
    )
    structure_columns = [
        column for column in structure_index.columns
        if column not in {"target_feature_index", "target_chembl_id", "structure_mask", "structure_route"}
    ]
    if drug.shape != (720, 2048) or target.shape != (745, 1024) or target_aux.shape != (745, 1280):
        raise RuntimeError("registry feature shape contract failed")
    if len(structure_columns) != 19 or len(structure_index) != 745:
        raise RuntimeError("registry structure contract failed")
    target_positions = frame["target_feature_index"].to_numpy(dtype=np.int64)
    structure = structure_index[structure_columns].to_numpy(dtype=np.float32)[target_positions]
    structure_mask = structure_index["structure_mask"].to_numpy(dtype=np.float32)[target_positions]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    positions = np.arange(len(frame), dtype=np.int64)
    columns: dict[str, list[np.ndarray]] = {
        "pair": [], "drug_to_target": [], "target_to_drug": [],
        "drug_to_target_residual": [], "target_to_drug_residual": [],
    }
    checkpoint_audit = []
    for seed, checkpoint_path in zip(SEEDS, checkpoints, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ODTIV2Config(**checkpoint["config"])
        if not config.directional_heads_enabled:
            raise RuntimeError(f"directional heads disabled: {checkpoint_path}")
        model = RoutedInteractionRankerV2(len(checkpoint["families"]), config, use_conplex=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        arrays = inference_arrays(frame, checkpoint)
        drug_cache = torch.from_numpy(np.asarray(drug, dtype=np.float32)).to(device)
        target_cache = torch.from_numpy(
            (np.asarray(target, dtype=np.float32) - arrays["target_mean"]) / arrays["target_std"]
        ).to(device)
        target_aux_cache = torch.from_numpy(
            (np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"])
            / arrays["target_aux_std"]
        ).to(device)
        result = predict(
            model, positions, frame, drug, None, target, target_aux,
            None, None, None, None, config.target_token_max_len, None, None,
            structure, structure_mask, arrays, device, args.inference_batch_size,
            drug_feature_cache=drug_cache,
            target_feature_cache=target_cache,
            target_aux_feature_cache=target_aux_cache,
        )
        mapping = {
            "pair": "final_logit",
            "drug_to_target": "drug_to_target_logit",
            "target_to_drug": "target_to_drug_logit",
            "drug_to_target_residual": "drug_to_target_residual",
            "target_to_drug_residual": "target_to_drug_residual",
        }
        for name, key in mapping.items():
            columns[name].append(np.asarray(result[key], dtype=np.float32))
        contract = checkpoint.get("full_fit_contract") or checkpoint.get("stage_a_contract") or {}
        checkpoint_audit.append(
            {
                "seed": seed,
                "checkpoint": portable(checkpoint_path),
                "sha256": sha256(checkpoint_path),
                "contract": contract,
            }
        )
        del model, drug_cache, target_cache, target_aux_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, values in columns.items():
        frame[f"ensemble_{name}_logit"] = np.mean(np.stack(values), axis=0)
    score = "ensemble_drug_to_target_logit"
    frame["diagnostic_rank_within_drug_745"] = frame.groupby(
        "ligand_inchikey", sort=False
    )[score].rank(method="min", ascending=False).astype(np.int16)
    frame["diagnostic_percentile_within_drug_745"] = (
        1.0 - (frame["diagnostic_rank_within_drug_745"] - 1.0) / 744.0
    )
    frame["routed_rank_within_drug"] = frame.groupby(
        ["ligand_inchikey", "production_route"], sort=False
    )[score].rank(method="min", ascending=False).astype(np.int16)
    route_sizes = frame.groupby("production_route")["target_chembl_id"].nunique()
    frame["routed_candidate_target_count"] = frame["production_route"].map(route_sizes).astype(np.int16)
    frame["routed_percentile_within_drug"] = np.where(
        frame["routed_candidate_target_count"].gt(1),
        1.0 - (frame["routed_rank_within_drug"] - 1.0)
        / (frame["routed_candidate_target_count"] - 1.0),
        1.0,
    )
    frame["aux_rank_drug_within_target_720"] = frame.groupby(
        "target_chembl_id", sort=False
    )["ensemble_target_to_drug_logit"].rank(method="min", ascending=False).astype(np.int16)
    frame["structure_mask"] = structure_mask.astype(np.int8)
    frame["scoring_run_label"] = args.run_label

    checks = {
        "exact_720x745": len(frame) == 720 * 745,
        "exact_720_drugs": frame["ligand_inchikey"].nunique() == 720,
        "exact_745_targets": frame["target_chembl_id"].nunique() == 745,
        "route_sizes": route_sizes.to_dict() == {
            "EXPLORATORY_ASSAYABLE_NO_DIRECT_SM": 8,
            "PRIMARY_BIOCHEMICAL_DIRECT_SM": 367,
            "PRIMARY_FUNCTIONAL_DIRECT_SM": 83,
            "REGISTRY_ONLY_SPECIAL_NO_DIRECT_SM": 245,
            "SPECIAL_SYSTEM_DIRECT_SM": 42,
        },
        "all_scores_finite": bool(
            np.isfinite(frame[[
                "ensemble_pair_logit", "ensemble_drug_to_target_logit",
                "ensemble_target_to_drug_logit",
            ]]).all().all()
        ),
        "all_route_ranks_in_range": bool(
            frame["routed_rank_within_drug"].between(
                1, frame["routed_candidate_target_count"]
            ).all()
        ),
        "unknown_pairs_not_labelled_negative": frame["pair_label_status"].eq(
            "UNOBSERVED_OR_HISTORICAL_LOOKUP_PENDING_NOT_NEGATIVE"
        ).all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SCORES_V1.csv.gz"
    top20 = frame.loc[
        ~frame["production_route"].eq("REGISTRY_ONLY_SPECIAL_NO_DIRECT_SM")
        & frame["routed_rank_within_drug"].le(20)
    ].sort_values(["ligand_inchikey", "production_route", "routed_rank_within_drug"])
    top20_path = out / "BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_TOP20_V1.csv"
    frame.to_csv(scores_path, index=False, compression={"method": "gzip", "mtime": 0})
    top20.to_csv(top20_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_BIDIRECTIONAL_ROUTED_720X745_V1",
        "run_label": args.run_label,
        "counts": {
            "rows": len(frame), "drugs": 720, "targets": 745,
            "route_targets": route_sizes.sort_index().to_dict(),
            "target_warm": int(target_index["in_supervised_training_target_vocabulary"].sum()),
            "target_cold": int((~target_index["in_supervised_training_target_vocabulary"].astype(bool)).sum()),
            "rows_with_structure": int((structure_mask > 0).sum()),
        },
        "score": score,
        "rank_contract": {
            "primary": "rank biochemical 367 and functional 83 within their routed assay families; release rank/450 only after cross-route calibration",
            "special": "separate rank/42",
            "exploratory": "separate rank/8 with high-novelty uncertainty",
            "registry_only": "diagnostic rank/245, no current production claim",
            "complete_registry_745": "diagnostic only, not a global production denominator",
        },
        "checkpoints": checkpoint_audit,
        "checks": checks,
        "artifacts": {
            portable(scores_path): sha256(scores_path),
            portable(top20_path): sha256(top20_path),
        },
        "claim_boundary": (
            "All 745 targets are scored for coverage and route auditing.  Scores across "
            "assay routes are not assumed calibrated; unobserved pairs remain unknown."
        ),
    }
    summary_path = out / "BIOMASTER_BIDIRECTIONAL_720X745_ROUTED_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
