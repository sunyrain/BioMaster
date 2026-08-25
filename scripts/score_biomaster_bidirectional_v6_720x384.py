#!/usr/bin/env python3
"""Score the 720x384 deployment matrix with the V6 FULL_FIT ensemble."""

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
from score_biomaster_deployment_augmented_720x384_v1 import (  # noqa: E402
    DRUG,
    PAIRS,
    TARGET,
    TARGET_AUX,
    augmented_structure,
)
from score_biomaster_full_fit_current_new_relations_v1 import inference_arrays  # noqa: E402
from train_biomaster_odti_v2 import predict  # noqa: E402


SEEDS = (20260816, 20260817, 20260820)
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/biomaster_bidirectional_v6_full_fit"
DEFAULT_OUT = ROOT / "outputs/biomaster_bidirectional_v6_720x384"
CASES = (
    ("lazertinib", "RRMJMHOQSALEJJ-UHFFFAOYSA-N", "ERBB4"),
    ("lazertinib", "RRMJMHOQSALEJJ-UHFFFAOYSA-N", "AXL"),
    ("repotrectinib", "FIKPXCOQUIZNHB-WDEREUQCSA-N", "AXL"),
    ("repotrectinib", "FIKPXCOQUIZNHB-WDEREUQCSA-N", "PLK4"),
    ("lorlatinib", "IIXWYSCJSQVBQM-LLVKDONJSA-N", "PLK4"),
    ("pitavastatin", "VGYFMXBACGZSIL-MCBHFWOFSA-N", "IDO1"),
    ("tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N", "IRAK1"),
    ("tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N", "IRAK4"),
    ("deucravacitinib", "BZZKEPGENYLQSC-FIBGUPNXSA-N", "FGFR4"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoints = [
        checkpoint_root / f"seed_{seed}" / "FULL_FIT_BIDIRECTIONAL_V6.pt"
        for seed in SEEDS
    ]
    required = [PAIRS, DRUG, TARGET, TARGET_AUX, *checkpoints]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frame = pd.read_csv(PAIRS, low_memory=False)
    if len(frame) != 720 * 384:
        raise RuntimeError("deployment pair count is not 720x384")
    frame["parent_standard_inchi_key"] = frame["ligand_inchikey"].astype(str)
    frame["calibration_pair_id"] = frame["pairId"].astype(str)
    frame["binary_label"] = 0
    frame["binary_observed"] = 0
    frame["mean_pchembl"] = np.nan
    frame["min_pchembl"] = np.nan
    frame["max_pchembl"] = np.nan
    structure, structure_mask, structure_audit = augmented_structure(frame)
    drug = np.load(DRUG, mmap_mode="r")
    target = np.load(TARGET, mmap_mode="r")
    target_aux = np.load(TARGET_AUX, mmap_mode="r")
    if drug.shape != (720, 2048) or target.shape != (384, 1024) or target_aux.shape != (384, 1280):
        raise RuntimeError("deployment feature shape contract failed")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    positions = np.arange(len(frame), dtype=np.int64)
    output_columns = {
        "pair": [],
        "drug_to_target": [],
        "target_to_drug": [],
        "drug_to_target_residual": [],
        "target_to_drug_residual": [],
    }
    checkpoint_audit = []
    for seed, checkpoint_path in zip(SEEDS, checkpoints, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        contract = checkpoint.get("full_fit_contract", {})
        if contract.get("protocol") != "BIOMASTER_BIDIRECTIONAL_V6_HEAD_ONLY_FULL_FIT":
            raise RuntimeError(f"checkpoint contract failed: {checkpoint_path}")
        config = ODTIV2Config(**checkpoint["config"])
        if not config.directional_heads_enabled:
            raise RuntimeError("directional heads are disabled")
        model = RoutedInteractionRankerV2(
            len(checkpoint["families"]), config, use_conplex=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        arrays = inference_arrays(frame, checkpoint)
        drug_cache = torch.from_numpy(np.asarray(drug, dtype=np.float32)).to(device)
        target_cache = torch.from_numpy(
            (np.asarray(target, dtype=np.float32) - arrays["target_mean"])
            / arrays["target_std"]
        ).to(device)
        target_aux_cache = torch.from_numpy(
            (np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"])
            / arrays["target_aux_std"]
        ).to(device)
        result = predict(
            model,
            positions,
            frame,
            drug,
            None,
            target,
            target_aux,
            None,
            None,
            None,
            None,
            config.target_token_max_len,
            None,
            None,
            structure,
            structure_mask,
            arrays,
            device,
            args.inference_batch_size,
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
            frame[f"{name}_logit_{seed}"] = result[key]
            output_columns[name].append(result[key])
        checkpoint_audit.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "selected_epochs": int(contract["selected_epochs_from_stage_a"]),
            }
        )
        del model, drug_cache, target_cache, target_aux_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, values in output_columns.items():
        frame[f"ensemble_{name}_logit"] = np.mean(np.stack(values), axis=0)
    frame["v6_rank_target_within_drug_384"] = frame.groupby(
        "ligand_inchikey", sort=False
    )["ensemble_drug_to_target_logit"].rank(method="min", ascending=False).astype(int)
    frame["v6_percentile_target_within_drug"] = (
        1.0 - (frame["v6_rank_target_within_drug_384"] - 1.0) / 383.0
    )
    frame["pair_rank_target_within_drug_384"] = frame.groupby(
        "ligand_inchikey", sort=False
    )["ensemble_pair_logit"].rank(method="min", ascending=False).astype(int)
    frame["v6_aux_rank_drug_within_target_720"] = frame.groupby(
        "target_chembl_id", sort=False
    )["ensemble_target_to_drug_logit"].rank(method="min", ascending=False).astype(int)
    frame["structure_mask"] = structure_mask
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / "BIDIRECTIONAL_V6_FULL_FIT_720X384_SCORES.csv.gz"
    frame.to_csv(scores_path, index=False, compression="gzip")
    cases = []
    for drug_name, inchikey, gene in CASES:
        matched = frame.loc[
            frame["ligand_inchikey"].astype(str).eq(inchikey)
            & frame["gene_symbol"].astype(str).eq(gene)
        ]
        if len(matched) != 1:
            raise RuntimeError(f"case does not map exactly once: {drug_name}, {gene}")
        row = matched.iloc[0]
        cases.append(
            {
                "drug_name": drug_name,
                "drug_inchikey": inchikey,
                "gene_symbol": gene,
                "target_chembl_id": str(row["target_chembl_id"]),
                "v6_rank_target_within_drug_384": int(
                    row["v6_rank_target_within_drug_384"]
                ),
                "pair_rank_target_within_drug_384": int(
                    row["pair_rank_target_within_drug_384"]
                ),
                "rank_change_positive_is_better": int(
                    row["pair_rank_target_within_drug_384"]
                    - row["v6_rank_target_within_drug_384"]
                ),
                "v6_aux_rank_drug_within_target_720": int(
                    row["v6_aux_rank_drug_within_target_720"]
                ),
                "ensemble_drug_to_target_logit": float(
                    row["ensemble_drug_to_target_logit"]
                ),
                "ensemble_drug_to_target_residual": float(
                    row["ensemble_drug_to_target_residual_logit"]
                ),
                "structure_mask": int(row["structure_mask"]),
            }
        )
    cases_frame = pd.DataFrame(cases).sort_values(
        ["v6_rank_target_within_drug_384", "drug_name"]
    )
    cases_path = output / "BIDIRECTIONAL_V6_SHOWCASE_CASE_RANKS.csv"
    cases_frame.to_csv(cases_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_BIDIRECTIONAL_V6_FULL_FIT_720X384",
        "rows": int(len(frame)),
        "drug_queries": int(frame["ligand_inchikey"].nunique()),
        "candidate_targets": int(frame["target_chembl_id"].nunique()),
        "primary_score": "equal-mean ensemble_drug_to_target_logit across three seeds",
        "primary_rank": "v6_rank_target_within_drug_384",
        "auxiliary_rank": "v6_aux_rank_drug_within_target_720",
        "checkpoints": checkpoint_audit,
        "structure": structure_audit,
        "showcase_cases": cases,
        "artifacts": {
            "scores": str(scores_path.relative_to(ROOT)),
            "scores_sha256": sha256(scores_path),
            "showcase_case_ranks": str(cases_path.relative_to(ROOT)),
            "showcase_case_ranks_sha256": sha256(cases_path),
        },
        "claim_boundary": (
            "FULL_FIT scores are for production candidate prioritization and have no "
            "unbiased post-refit performance estimate. Unreported pairs are not negatives. "
            "Use rank/384 within each drug; do not interpret the directional logit as a "
            "binding probability or physical affinity."
        ),
    }
    summary_path = output / "BIDIRECTIONAL_V6_FULL_FIT_720X384_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
