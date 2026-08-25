#!/usr/bin/env python3
"""Score the frozen 720x384 deployment matrix with augmented FULL_FIT models."""

from __future__ import annotations

import argparse
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

from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2
from biomaster.comprehensive_balanced import fuse_percentile_scores
from score_biomaster_full_fit_current_new_relations_v1 import inference_arrays
from score_biomaster_odti_720x384_v2 import _structure_for_deployment
from train_biomaster_odti_v2 import predict


SEEDS = (20260816, 20260817, 20260820)
DEFAULT_AFFINITY_WEIGHT = 0.45
BASE = ROOT / "outputs/old_drug_target_sota_v1"
DEPLOY = BASE / "deployment_720x384_feature_store_v1"
PAIRS = DEPLOY / "OLD_DRUG_TARGET_INDEXED_PAIRS_276480_V1.csv.gz"
DRUG = DEPLOY / "OLD_DRUG_MORGAN2048_UINT8_V1.npy"
TARGET = DEPLOY / "PROJECT384_PROTBERT1024_FLOAT32_V1.npy"
TARGET_AUX = BASE / "public_retrained_v1/dtiam_deployment_feature_store_v1/DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy"
CAL_PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BASE_STRUCTURE = BASE / "feature_store_v1/ODTI_STRUCTURE_CONTEXT_V1.csv.gz"
AUGMENT = ROOT / "outputs/biomaster_deployment_augmentation_v1"
RECOVERED = AUGMENT / "RECOVERED_CHEMBL37_RELATIONS_V1.csv.gz"
AUGMENT_TARGET_STRUCTURE = AUGMENT / "TARGET_STRUCTURE_CONTEXT_DEPLOYMENT_AUGMENTED_V1.csv.gz"
BDB_PAIRS = ROOT / "outputs/biomaster_bindingdb_affinity_feature_package_v1/BINDINGDB_DIRECT_KI_KD_AFFINITY_PAIRS_V1.csv.gz"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs/biomaster_deployment_augmented_full_fit_v1"
DEFAULT_OUT = ROOT / "outputs/biomaster_deployment_augmented_720x384_v1"

CASES = (
    ("PLK4", "lorlatinib", "IIXWYSCJSQVBQM-LLVKDONJSA-N"),
    ("IDO1", "pitavastatin", "VGYFMXBACGZSIL-MCBHFWOFSA-N"),
    ("IRAK1", "tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N"),
    ("IRAK4", "tepotinib", "AHYMHWXQRWRBKT-UHFFFAOYSA-N"),
)


def augmented_structure(deploy: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    calibration = pd.read_csv(CAL_PAIRS, low_memory=False)
    base_structure = pd.read_csv(BASE_STRUCTURE, low_memory=False)
    values, mask, _ = _structure_for_deployment(deploy, calibration, base_structure)
    target_structure = pd.read_csv(AUGMENT_TARGET_STRUCTURE, low_memory=False)
    feature_columns = [
        column for column in base_structure.columns
        if column not in {"calibration_pair_id", "structure_mask"}
    ]
    target_structure = target_structure.sort_values("target_feature_index").drop_duplicates("target_chembl_id")
    indexed = target_structure.set_index("target_chembl_id")
    target_ids = deploy["target_chembl_id"].astype(str)
    available = target_ids.isin(indexed.index).to_numpy()
    replacement = np.zeros(len(deploy), dtype=bool)
    if available.any():
        rows = indexed.loc[target_ids[available], ["structure_mask", *feature_columns]]
        safe = rows["structure_mask"].to_numpy(dtype=np.float32) > 0
        positions = np.flatnonzero(available)
        replacement_positions = positions[safe & (mask[positions] <= 0)]
        if len(replacement_positions):
            replacement_rows = indexed.loc[target_ids.iloc[replacement_positions], feature_columns]
            values[replacement_positions] = replacement_rows.to_numpy(dtype=np.float32)
            mask[replacement_positions] = 1.0
            replacement[replacement_positions] = True
    return values, mask, {
        "rows_mask_one": int((mask > 0).sum()),
        "rows_mask_zero": int((mask <= 0).sum()),
        "rows_recovered_from_augmentation": int(replacement.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="comma-separated checkpoint seeds",
    )
    parser.add_argument("--checkpoint-dir-template", default="FULL_FIT_2026__seed_{seed}")
    parser.add_argument("--checkpoint-filename", default="FULL_FIT_MODEL_BINDINGDB_AFFINITY_V1.pt")
    parser.add_argument("--known-relations", default=str(CAL_PAIRS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--fusion-strategy",
        choices=("linear", "product", "geometric", "top20_refine", "top10_refine"),
        default="linear",
    )
    parser.add_argument("--affinity-weight", type=float, default=DEFAULT_AFFINITY_WEIGHT)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.affinity_weight <= 1.0:
        raise ValueError("affinity-weight must be in [0, 1]")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique integers")
    checkpoint_root = Path(args.checkpoint_root).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        checkpoint_root / args.checkpoint_dir_template.format(seed=seed) / args.checkpoint_filename
        for seed in seeds
    ]
    known_relations_path = Path(args.known_relations).resolve()
    required = [
        PAIRS, DRUG, TARGET, TARGET_AUX, CAL_PAIRS, BASE_STRUCTURE,
        RECOVERED, AUGMENT_TARGET_STRUCTURE, BDB_PAIRS, known_relations_path, *checkpoints,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    frame = pd.read_csv(PAIRS, low_memory=False)
    if len(frame) != 720 * 384:
        raise RuntimeError("deployment pair count is not 720x384")
    frame["parent_standard_inchi_key"] = frame["ligand_inchikey"].astype(str)
    frame["calibration_pair_id"] = frame["pairId"].astype(str)
    frame["binary_label"] = 0
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
    seed_fusion_columns = []
    for seed, checkpoint_path in zip(seeds, checkpoints, strict=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ODTIV2Config(**checkpoint["config"])
        model = RoutedInteractionRankerV2(len(checkpoint["families"]), config, use_conplex=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        arrays = inference_arrays(frame, checkpoint)
        drug_cache = torch.from_numpy(np.asarray(drug, dtype=np.float32)).to(device)
        target_cache = torch.from_numpy(
            (np.asarray(target, dtype=np.float32) - arrays["target_mean"]) / arrays["target_std"]
        ).to(device)
        target_aux_cache = torch.from_numpy(
            (np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"]) / arrays["target_aux_std"]
        ).to(device)
        output = predict(
            model, positions, frame, drug, None, target, target_aux,
            None, None, None, None, config.target_token_max_len, None, None,
            structure, structure_mask, arrays, device, args.inference_batch_size,
            drug_feature_cache=drug_cache, target_feature_cache=target_cache,
            target_aux_feature_cache=target_aux_cache,
        )
        binary_column = f"binary_{seed}"
        affinity_column = f"affinity_{seed}"
        fusion_column = f"fusion_{seed}"
        frame[binary_column] = output["final_logit"]
        frame[affinity_column] = output["affinity"]
        binary_rank = frame.groupby("target_chembl_id", sort=False)[binary_column].rank(method="average", pct=True)
        affinity_rank = frame.groupby("target_chembl_id", sort=False)[affinity_column].rank(method="average", pct=True)
        frame[fusion_column] = fuse_percentile_scores(
            binary_rank.to_numpy(dtype=np.float64),
            affinity_rank.to_numpy(dtype=np.float64),
            args.fusion_strategy,
            args.affinity_weight,
        )
        seed_fusion_columns.append(fusion_column)
        del model, drug_cache, target_cache, target_aux_cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame["ensemble_fusion_score"] = frame[seed_fusion_columns].mean(axis=1)
    frame["ensemble_rank_within_target_720"] = frame.groupby("target_chembl_id", sort=False)[
        "ensemble_fusion_score"
    ].rank(method="min", ascending=False).astype(int)
    frame["ensemble_percentile_top"] = 1.0 - (frame["ensemble_rank_within_target_720"] - 1) / 720.0
    frame["structure_mask"] = structure_mask

    known = pd.read_csv(known_relations_path, low_memory=False)
    required_known = {"parent_standard_inchi_key", "target_chembl_id"}
    if required_known - set(known.columns):
        raise RuntimeError(f"known relation table is missing {sorted(required_known - set(known.columns))}")
    known["parent_standard_inchi_key"] = known["parent_standard_inchi_key"].fillna("").astype(str)
    known["target_chembl_id"] = known["target_chembl_id"].fillna("").astype(str)
    warm_drugs = set(known["parent_standard_inchi_key"]) - {""}
    warm_targets = set(known["target_chembl_id"]) - {""}
    if {"binary_label", "binary_observed"}.issubset(known.columns):
        positive_known = known.loc[
            pd.to_numeric(known["binary_observed"], errors="coerce").fillna(0).eq(1)
            & pd.to_numeric(known["binary_label"], errors="coerce").fillna(0).eq(1)
        ].copy()
    elif "binary_label" in known.columns:
        positive_known = known.loc[
            pd.to_numeric(known["binary_label"], errors="coerce").fillna(0).eq(1)
        ].copy()
    else:
        positive_known = known.iloc[0:0].copy()
    positive_warm_drugs = set(positive_known["parent_standard_inchi_key"]) - {""}
    positive_warm_targets = set(positive_known["target_chembl_id"]) - {""}
    cases = []
    for gene, drug_name, inchikey in CASES:
        matched = frame.loc[frame["gene_symbol"].eq(gene) & frame["ligand_inchikey"].eq(inchikey)]
        if len(matched) != 1:
            raise RuntimeError(f"case does not map exactly once: {gene}, {drug_name}")
        row = matched.iloc[0]
        drug_seen = inchikey in warm_drugs
        target_seen = str(row["target_chembl_id"]) in warm_targets
        drug_positive_seen = inchikey in positive_warm_drugs
        target_positive_seen = str(row["target_chembl_id"]) in positive_warm_targets
        exact_any = (
            known["parent_standard_inchi_key"].eq(inchikey)
            & known["target_chembl_id"].eq(str(row["target_chembl_id"]))
        )
        exact_positive = (
            positive_known["parent_standard_inchi_key"].eq(inchikey)
            & positive_known["target_chembl_id"].eq(str(row["target_chembl_id"]))
        )
        cold_class = (
            "BOTH_SEEN_RELATION_HELD_OUT" if drug_seen and target_seen else
            "TARGET_COLD_SINGLE" if drug_seen else
            "DRUG_COLD_SINGLE" if target_seen else "DOUBLE_COLD"
        )
        cases.append({
            "gene_symbol": gene,
            "target_chembl_id": str(row["target_chembl_id"]),
            "drug_name": drug_name,
            "drug_inchikey": inchikey,
            "cold_start_class": cold_class,
            "drug_any_relation_warm": drug_seen,
            "target_any_relation_warm": target_seen,
            "drug_positive_relation_warm": drug_positive_seen,
            "target_positive_relation_warm": target_positive_seen,
            "exact_relation_in_full_fit": bool(exact_any.any()),
            "exact_positive_relation_in_full_fit": bool(exact_positive.any()),
            "rank_within_target_720": int(row["ensemble_rank_within_target_720"]),
            "percentile_top": float(row["ensemble_percentile_top"]),
            "structure_mask": int(row["structure_mask"]),
        })
    # Seed ranks require ranking over the entire target group, not the one-row
    # case slice used above.
    for case in cases:
        subset = frame.loc[frame["target_chembl_id"].eq(case["target_chembl_id"])]
        row_index = subset.index[subset["ligand_inchikey"].eq(case["drug_inchikey"])][0]
        for seed in seeds:
            case[f"rank_seed_{seed}"] = int(subset[f"fusion_{seed}"].rank(method="min", ascending=False).loc[row_index])
    cases_frame = pd.DataFrame(cases).sort_values("rank_within_target_720")
    scores_path = out / "AUGMENTED_FULL_FIT_720X384_SCORES_V1.csv.gz"
    cases_path = out / "CURRENT_DOUBLE_WARM_CASE_RANKS_V1.csv"
    frame.to_csv(scores_path, index=False, compression="gzip")
    cases_frame.to_csv(cases_path, index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "AUGMENTED_FULL_FIT_720X384_TARGET_TO_DRUG_RANKING_V1",
        "ranking_universe": "720 frozen old drugs independently within each of 384 targets",
        "checkpoints": [str(path.relative_to(ROOT)) for path in checkpoints],
        "fusion": {
            "strategy": args.fusion_strategy,
            "binary_weight": 1.0 - args.affinity_weight,
            "affinity_weight": args.affinity_weight,
        },
        "structure": structure_audit,
        "warm_frozen_drugs": len(set(frame["ligand_inchikey"].astype(str)) & warm_drugs),
        "positive_warm_frozen_drugs": len(set(frame["ligand_inchikey"].astype(str)) & positive_warm_drugs),
        "known_relations": str(known_relations_path.relative_to(ROOT)),
        "cases": cases,
        "artifacts": {
            "scores": str(scores_path.relative_to(ROOT)),
            "case_ranks": str(cases_path.relative_to(ROOT)),
        },
    }
    summary_path = out / "AUGMENTED_FULL_FIT_720X384_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(cases_frame.to_string(index=False))
    print(json.dumps({"status": "PASS", "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
